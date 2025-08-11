#!/usr/bin/env python3
# dwa_node.py

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan

from nav_msgs.msg import Odometry, Path, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer  
from dwa_convert_global_path import GlobalPathConverter
from dwa_obstacle_cost import ObstacleCost
from dwa_distance_cost import DistanceCosts
from dwa_data_provider import DataProvider
from dwa_trj_generator import TrajectoryGenerator
from dwa_cost_evaluator import CostEvaluator
from builtin_interfaces.msg import Duration

class DWANode(Node):
    """
    DWANode with integrated costmap update inside control loop.
    """
    def __init__(self):
        super().__init__('dwa_node')
        # ROS2 parameters
        self.declare_parameter('global_path_resolution', 0.1)
        self.declare_parameter('sim_period', 15.)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('sim_time', 5.0)
        self.declare_parameter('map_width', 4.)
        self.declare_parameter('map_height', 4.)
        self.declare_parameter('cell_resolution', 0.1)
        self.declare_parameter('min_z_threshold', -0.15)
        self.declare_parameter('max_z_threshold', 0.4)
        self.declare_parameter('obstacle_cost', 100)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('robot_radius', '0.5')
        self.declare_parameter('inflation_radius', '0.55')
        self.declare_parameter('cost_scaling_factor', '10.0')
        self.declare_parameter('acc_lim_v', '4.5')
        self.declare_parameter('acc_lim_w', '3.0')
        self.declare_parameter('v_samples', '19')
        self.declare_parameter('w_samples', '19')
        self.declare_parameter('wc_obstacle', '0.2')
        self.declare_parameter('wc_path', '0.2')
        self.declare_parameter('wc_align, 0.2')
        self.declare_parameter('wc_goal', '0.2')
        self.declare_parameter('wc_goal_center', '0.2')



        # Collect parameters
        self.cfg = {
            'global_path_resolution':   self.get_parameter('global_path_resolution').value,
            'dt':                       self.get_parameter('dt').value,
            'map_width':                self.get_parameter('map_width').value,
            'map_height':               self.get_parameter('map_height').value,
            'cell_resolution':          self.get_parameter('cell_resolution').value,
            'min_z_threshold':          self.get_parameter('min_z_threshold').value,
            'max_z_threshold':          self.get_parameter('max_z_threshold').value,
            'obstacle_cost':            self.get_parameter('obstacle_cost').value,
            'robot_radius':             self.get_parameter('robot_radius').value,
            'sim_time':                 self.get_parameter('sim_time').value,
            'sim_period':               self.get_parameter('sim_period').value,
            'robot_base_frame':         self.get_parameter('robot_base_frame').value,
            'inflation_radius':         self.get_parameter('inflation_radius').value,
            'cost_scaling_factor':      self.get_parameter('cost_scaling_factor').value,
            'acc_lim_v':                self.get_parameter('acc_lim_v').value,
            'acc_lim_w':                self.get_parameter('acc_lim_w').value,
            'v_samples':                self.get_parameter('v_samples').value,
            'w_samples':                self.get_parameter('w_samples').value,
            'wc_obstacle':              self.get_parameter('wc_obstacle').value,
            'wc_path':                  self.get_parameter('wc_path').value,
            'wc_align':                 self.get_parameter('wc_align').value,
            'wc_goal':                  self.get_parameter('wc_goal').value,
            'wc_goal_center':           self.get_parameter('wc_goal_center').value,

        }
        params_str = "\n".join(f"{k}: {v}" for k, v in self.cfg.items())
        self.get_logger().info(f"Loaded parameters:\n{params_str}")

        # Internal state
        self.current_pc = None
        self.current_odom = None
        self.current_path = None
        self.waypoint = None
        
        self.global_path_params = None

        self.sim_time   = self.cfg['sim_time']
        self.dt         = self.cfg['dt']
        self.prev_pair  = (0.0, 0.0)
        self.sim_period = float(self.cfg['sim_period'])

        # ---- message_filters sync: /voxel_cloud + /scan ----
        
        # ZED(PointCloud2) 기본 QoS: RELIABLE / KEEP_LAST(10) / VOLATILE
        zed_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        scan_qos        = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pc_sub     = Subscriber(self, PointCloud2, '/voxel_cloud', qos_profile=zed_qos)
        self.scan_sub   = Subscriber(self, LaserScan,   '/scan',        qos_profile=scan_qos)
        self.sync       = ApproximateTimeSynchronizer(
            [self.pc_sub, self.scan_sub],
            queue_size=10,
            slop=0.6,
            allow_headerless=False
        )

        self.sync.registerCallback(self.synced_cb)
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            lambda msg: setattr(self, 'current_odom', msg),
            10
        )
        
        self.create_subscription(
            PointStamped,
            '/target_point',
            lambda msg: setattr(self, 'waypoint', msg),
            10
        )
        
        self.create_subscription(
            Vector3Stamped, 
            '/global_path_params', 
            self.global_path_params_cb,
            10
        )

        # Publishers
        self.path_pub       = self.create_publisher(Path, '/global_path', 10)
        self.costmap_pub    = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.cmd_pub        = self.create_publisher(Twist, '/cmd_vel', 10)
        self.viz_pub        = self.create_publisher(Marker, 'trajectory_viz', 10)
        self.all_viz_pub    = self.create_publisher(Marker, 'all_trajectory_viz', 10)


        # Modules
        dp                  = DataProvider(self)
        self.obs_cost       = ObstacleCost(self.cfg, dp)
        self.traj_gen       = TrajectoryGenerator(self.cfg, dp)
        self.distance_costs = DistanceCosts(dp)
        self.evaluator      = CostEvaluator(self.cfg, dp)
        # Control loop timer only
        self.create_timer(1/self.sim_period, self._control_loop)
        self.get_logger().info('DWANode initialized with integrated costmap update.')
        self.twist = Twist()

    def synced_cb(self, pc_msg: PointCloud2, scan_msg: LaserScan):
        self.current_pc = pc_msg
        self.laserscan  = scan_msg

    def global_path_params_cb(self, msg: Vector3Stamped):
        self.global_path_params = msg

    def publish_trajectory_markers(self, samples, color=(1.,0.,0.),all_pub=False):
        for idx, (v, w) in enumerate(samples):
            marker = Marker()
            marker.lifetime = Duration(sec=0, nanosec=1000_000_000)
            marker.header.frame_id = self.cfg['robot_base_frame']
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'trajectories'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02            # 선 굵기
            # 색상: 첫 번째는 빨강, 두 번째는 파랑
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0

            x = y = theta = 0.0
            steps = int(self.sim_time / self.dt)
            for _ in range(steps):
                x     += v * math.cos(theta) * self.dt
                y     += v * math.sin(theta) * self.dt
                theta += w * self.dt
                p = Point(x=x, y=y, z=0.0)
                marker.points.append(p)
            if not all_pub:
                self.viz_pub.publish(marker)
            else:
                self.all_viz_pub.publish(marker)

    def _control_loop(self):
        # Ensure all inputs available
        if not (self.current_pc and self.current_odom and self.waypoint and self.laserscan):            
            return
        # 1) Generate trajectories
        vel_pairs, trj_samples = self.traj_gen.generate_trajectories()
        if vel_pairs is None:
            return
        
        # 2) Update local costmap
        self.obs_cost.update_costmap()
        grid = self.obs_cost.get_costmap_msg(
            frame_id=self.cfg['robot_base_frame'],
            stamp=self.get_clock().now().to_msg()
        )
        self.costmap_pub.publish(grid) # for debugging

        # samples = [(3.5, 1.), (3.5, -1.)]
        # self.publish_trajectory_markers(trj_samples)

        # 2) 한 번만 전체 평가 => [cost_of_sample0, cost_of_sample1]
        path_costs, alignment_costs, goal_costs, goal_front_costs = self.distance_costs.evaluate(trj_samples)
        obs_costs = self.obs_cost.evaluate_velocity_samples(vel_pairs)
        best_pair, best_cost = self.evaluator.evaluate(vel_pairs, obs_costs, path_costs, alignment_costs, goal_costs, goal_front_costs)

        if best_pair is None: 
            best_pair = self.prev_pair
        self.publish_trajectory_markers([(best_pair[0],best_pair[1])])
        self.publish_trajectory_markers(vel_pairs,color=(0.,1.,0.),all_pub=True)
        self.prev_pair          = best_pair
        self.twist.linear.x     = best_pair[0]
        self.twist.angular.z    = best_pair[1]
        self.cmd_pub.publish(self.twist)
        # # 4) Select best and publish cmd_vel
        # best = trajectories[scores.index(min(scores))]
        # cmd = self.executor.to_cmd(best)
        # self.cmd_pub.publish(cmd)
        # self.get_logger().debug(f'Published cmd_vel: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}')


def main(args=None):

    rclpy.init(args=args)
    node = DWANode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
