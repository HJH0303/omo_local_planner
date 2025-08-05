#!/usr/bin/env python3
# dwa_node.py

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, Path, OccupancyGrid

from dwa_convert_global_path import GlobalPathConverter
from dwa_obstacle_cost import ObstacleCost
from dwa_data_provider import DataProvider
# from dwa_trajectory_generator import TrajectoryGenerator
# from dwa_cost_evaluator import CostEvaluator
# from dwa_executor import Executor

class DWANode(Node):
    """
    DWANode with integrated costmap update inside control loop.
    """
    def __init__(self):
        super().__init__('dwa_node')
        # ROS2 parameters
        self.declare_parameter('global_path_resolution', 0.1)
        self.declare_parameter('sim_period', 10)
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('map_width', 3.5)
        self.declare_parameter('map_height', 3.5)
        self.declare_parameter('cell_resolution', 0.1)
        self.declare_parameter('min_z_threshold', -0.2)
        self.declare_parameter('max_z_threshold', 2.0)
        self.declare_parameter('obstacle_cost', 100)
        self.declare_parameter('sim_time', 1.0)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('robot_radius', '0.6')


        # Collect parameters
        self.cfg = {
            'global_path_resolution': self.get_parameter('global_path_resolution').value,
            'dt':                     self.get_parameter('dt').value,
            'map_width':              self.get_parameter('map_width').value,
            'map_height':             self.get_parameter('map_height').value,
            'cell_resolution':        self.get_parameter('cell_resolution').value,
            'min_z_threshold':        self.get_parameter('min_z_threshold').value,
            'max_z_threshold':        self.get_parameter('max_z_threshold').value,
            'obstacle_cost':          self.get_parameter('obstacle_cost').value,
            'robot_radius':           self.get_parameter('robot_radius').value,
            'sim_time':               self.get_parameter('sim_time').value,
            'sim_period':             self.get_parameter('sim_period').value,
            'robot_base_frame':       self.get_parameter('robot_base_frame').value,
        }

        # Internal state
        self.current_pc = None
        self.current_odom = None
        self.current_path = None

        # Subscriptions
        self.create_subscription(
            PointCloud2,
            '/voxel_cloud',
            lambda msg: setattr(self, 'current_pc', msg),
            10
        )
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            lambda msg: setattr(self, 'current_odom', msg),
            10
        )
        self.path_converter = GlobalPathConverter(self.cfg)
        self.create_subscription(
            PointStamped,
            '/target_point',
            self._waypoint_cb,
            10
        )

        # Publishers
        self.path_pub    = self.create_publisher(Path, '/global_path', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel', 10)

        # Modules
        self.obs_cost  = ObstacleCost(self.cfg, self)
        dp             = DataProvider(self)
        # self.traj_gen  = TrajectoryGenerator(self.cfg, dp)
        # self.evaluator = CostEvaluator(self.cfg, dp)
        # self.executor  = Executor(self.cfg, dp)

        # Control loop timer only
        self.create_timer(self.cfg['dt'], self._control_loop)

        self.get_logger().info('DWANode initialized with integrated costmap update.')

    def _waypoint_cb(self, msg: PointStamped):
        # Generate and publish global path
        path = self.path_converter.generate(msg)
        self.get_logger().info(f'Generated global path with {len(path.poses)} poses.')
        self.current_path = path
        self.path_pub.publish(path)

    def _control_loop(self):
        # Ensure all inputs available
        if not (self.current_pc and self.current_odom and self.current_path):
            self.get_logger().info("hi")
            
            return
        # 1) Generate trajectories
        # trajectories = self.traj_gen.generate()
        # if not trajectories:
        #     return
        
        # 2) Update local costmap


        self.obs_cost.update_costmap(self.current_pc)
        grid = self.obs_cost.get_costmap_msg(
            frame_id=self.cfg['robot_base_frame'],
            stamp=self.get_clock().now().to_msg()
        )
        # self.costmap_pub.publish(grid) # for debugging

        
        samples = [(1.0, 0.0)]  # for dubgging
        # 3) Evaluate each trajectory with obstacle cost
        scores = []
        

        # 5) Compute obstacle cost for each (v, w) over sim_time

        for traj in samples:
            obstacle_costs = self.obs_cost.evaluate_velocity_samples(vel_samples)
            # score = self.evaluator.evaluate(traj, costmap_cost)
            scores.append(obstacle_costs)

        print(score)
        # for traj in trajectories:
        #     costmap_cost = self.obs_cost.compute(traj)
        #     score = self.evaluator.evaluate(traj, costmap_cost)
        #     scores.append(score)

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
