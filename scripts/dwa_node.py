#!/usr/bin/env python3
# dwa_node.py

import os
import json
import math
import numpy as np
from datetime import datetime

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
from builtin_interfaces.msg import Duration

from dwa_convert_global_path import GlobalPathConverter
from dwa_obstacle_cost import ObstacleCost
from dwa_distance_cost import DistanceCosts
from dwa_data_provider import DataProvider
from dwa_trj_generator import TrajectoryGenerator
from dwa_cost_evaluator import CostEvaluator


class DWANode(Node):
    """
    DWANode with integrated costmap update inside control loop.
    """
    def __init__(self):
        super().__init__('dwa_node')
        # ROS2 parameters
        self.declare_parameter('global_path_resolution', 0.1)
        self.declare_parameter('sim_period', 20.)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('sim_time', 6.0)
        self.declare_parameter('map_width', 4.)
        self.declare_parameter('map_height', 4.)
        self.declare_parameter('cell_resolution', 0.1)
        self.declare_parameter('min_z_threshold', -0.15)
        self.declare_parameter('max_z_threshold', 0.4)
        self.declare_parameter('obstacle_cost', 100)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('robot_radius', '0.4')
        self.declare_parameter('inflation_radius', '0.3')
        self.declare_parameter('cost_scaling_factor', '10.0')
        self.declare_parameter('acc_lim_v', '5.0')
        self.declare_parameter('acc_lim_w', '8.0')
        self.declare_parameter('v_samples', '15')
        self.declare_parameter('w_samples', '15')
        self.declare_parameter('wc_obstacle', '0.1')
        self.declare_parameter('wc_path', '0.3')
        self.declare_parameter('wc_align', 0.3)
        self.declare_parameter('wc_goal', '0.5')
        self.declare_parameter('wc_goal_center', '0.2')

        self.declare_parameter('log_base_dir', '/root/omo_ws/src/omo_local_planner/logs')

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
            'wc_goal':                  self.get_parameter('wc_goal').value,
            'wc_align':                 self.get_parameter('wc_align').value,
            'wc_goal_center':           self.get_parameter('wc_goal_center').value
        }
        params_str = "\n".join(f"{k}: {v}" for k, v in self.cfg.items())
        self.get_logger().info(f"Loaded parameters:\n{params_str}")

        # ----- logging setup -----
        self.run_start = datetime.now()
        ts = self.run_start.strftime("%Y%m%d_%H%M%S")
        base_log_dir = self.get_parameter('log_base_dir').value  # <- /root/omo_ws/src/omo_local_planner/logs
        self.log_dir = os.path.join(base_log_dir, ts)
        os.makedirs(self.log_dir, exist_ok=True)
        # save params immediately
        with open(os.path.join(self.log_dir, "params.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        # in-memory buffers
        self.logs = {
            'timestamps': [],
            'v_min': [],
            'v_max': [],
            'vel_pairs': [],
            'path_costs': [],
            'align_costs': [],
            'goal_costs': [],
            'goal_front_costs': [],
            'obs_costs': [],
            'norm_costs': [],
            'best_pairs': [],
            'best_costs': [],
        }

        # Internal state
        self.current_pc = None
        self.current_odom = None
        self.current_path = None
        self.waypoint = None

        self.sim_time = self.cfg['sim_time']
        self.dt = self.cfg['dt']
        self.prev_pair = (0.0, 0.0)
        self.sim_period = float(self.cfg['sim_period'])

        # ---- message_filters sync: /voxel_cloud + /scan ----
        # ZED(PointCloud2) 기본 QoS: RELIABLE / KEEP_LAST(10) / VOLATILE
        zed_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        scan_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pc_sub   = Subscriber(self, PointCloud2, '/voxel_cloud', qos_profile=zed_qos)
        self.scan_sub = Subscriber(self, LaserScan,   '/scan',        qos_profile=scan_qos)
        self.sync = ApproximateTimeSynchronizer(
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
        self.path_pub    = self.create_publisher(Path, '/global_path', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel_dwa', 10)
        self.viz_pub     = self.create_publisher(Marker, 'trajectory_viz', 10)
        self.all_viz_pub = self.create_publisher(Marker, 'all_trajectory_viz', 10)

        # Modules
        dp                   = DataProvider(self)
        self.obs_cost        = ObstacleCost(self.cfg, dp)
        self.traj_gen        = TrajectoryGenerator(self.cfg, dp)
        self.distance_costs  = DistanceCosts(dp)
        self.evaluator       = CostEvaluator(self.cfg, dp)

        # Control loop timer only
        self.create_timer(1 / self.sim_period, self._control_loop)
        self.get_logger().info(f'DWANode initialized. Logs -> {self.log_dir}')
        self.twist = Twist()

    def synced_cb(self, pc_msg: PointCloud2, scan_msg: LaserScan):
        self.current_pc = pc_msg
        self.laserscan  = scan_msg

    def global_path_params_cb(self, msg: Vector3Stamped):
        self.global_path_params = msg

    def publish_trajectory_markers(self, samples, color=(1., 0., 0.), all_pub=False):
        for idx, (v, w) in enumerate(samples):
            marker = Marker()
            marker.lifetime = Duration(sec=0, nanosec=2_000_000_000)
            marker.header.frame_id = self.cfg['robot_base_frame']
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'trajectories'
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0

            x = y = theta = 0.0
            steps = int(self.sim_time / self.dt)
            for _ in range(steps):
                x     += v * math.cos(theta) * self.dt
                y     += v * math.sin(theta) * self.dt
                theta += w * self.dt
                marker.points.append(Point(x=x, y=y, z=0.0))
            (self.viz_pub if not all_pub else self.all_viz_pub).publish(marker)

    def _control_loop(self):
        # Ensure all inputs available
        if not (self.current_pc and self.current_odom and self.waypoint and self.laserscan):
            return

        # 1) Generate trajectories
        vel_pairs, trj_samples, v_min, v_max = self.traj_gen.generate_trajectories()
        if vel_pairs is None:
            return

        # 2) Update local costmap
        self.obs_cost.update_costmap()
        grid = self.obs_cost.get_costmap_msg(
            frame_id=self.cfg['robot_base_frame'],
            stamp=self.get_clock().now().to_msg()
        )
        self.costmap_pub.publish(grid)  # for debugging

        # 3) Evaluate once
        path_costs, alignment_costs, goal_costs, goal_front_costs = self.distance_costs.evaluate(trj_samples)
        obs_costs = self.obs_cost.evaluate_velocity_samples(vel_pairs)
        best_pair, best_cost, norm_cost = self.evaluator.evaluate(vel_pairs, obs_costs, path_costs, 
                                                                  alignment_costs, goal_costs, goal_front_costs)
        # 4) Visualize + Command
        if best_pair is None:
            best_pair = self.prev_pair
        self.publish_trajectory_markers([(best_pair[0], best_pair[1])])
        self.publish_trajectory_markers(vel_pairs, color=(0., 1., 0.), all_pub=True)
        self.prev_pair = best_pair

        self.twist.linear.x = best_pair[0]
        self.twist.angular.z = best_pair[1]
        self.cmd_pub.publish(self.twist)

        # 5) ---- logging per tick ----
        now_msg = self.get_clock().now().to_msg()
        now_sec = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9
        # store as arrays so that later stacking is simple; allow ragged via dtype=object
        self.logs['timestamps'].append(now_sec)
        self.logs['v_min'].append(float(v_min))
        self.logs['v_max'].append(float(v_max))
        self.logs['vel_pairs'].append(np.asarray(vel_pairs, dtype=float))
        self.logs['path_costs'].append(np.asarray(path_costs, dtype=float))
        self.logs['goal_costs'].append(np.asarray(goal_costs, dtype=float))
        self.logs['goal_front_costs'].append(np.asarray(goal_front_costs, dtype=float))
        self.logs['align_costs'].append(np.asarray(alignment_costs, dtype=float))
        self.logs['obs_costs'].append(np.asarray(obs_costs, dtype=float))
        self.logs['norm_costs'].append(np.asarray(norm_cost, dtype=float))
        self.logs['best_pairs'].append(np.asarray(best_pair, dtype=float))
        self.logs['best_costs'].append(float(best_cost))

    # ---- save all logs on shutdown ----
    def save_logs(self):
        try:
            meta = {
                "start_time": self.run_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "samples_per_tick_hint": {
                    "v_samples": self.cfg['v_samples'],
                    "w_samples": self.cfg['w_samples']
                }
            }
            with open(os.path.join(self.log_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # pack arrays; ragged series stored as dtype=object
            out_path = os.path.join(self.log_dir, "run_data.npz")
            np.savez(
                out_path,
                timestamps=np.asarray(self.logs['timestamps'], dtype=float),
                v_min=np.asarray(self.logs['v_min'], dtype=float),
                v_max=np.asarray(self.logs['v_max'], dtype=float),
                best_pairs=np.asarray(self.logs['best_pairs'], dtype=float),
                best_costs=np.asarray(self.logs['best_costs'], dtype=float),
                norm_costs=(np.vstack(self.logs['norm_costs'])
                            if len(self.logs['norm_costs']) > 0
                            else np.empty((0, 4), dtype=float)),
                vel_pairs=np.asarray(self.logs['vel_pairs'], dtype=object),
                path_costs=np.asarray(self.logs['path_costs'], dtype=object),
                align_costs=np.asarray(self.logs['align_costs'], dtype=object),
                goal_costs=np.asarray(self.logs['goal_costs'], dtype=object),
                goal_front_costs=np.asarray(self.logs['goal_front_costs'], dtype=object),
                obs_costs=np.asarray(self.logs['obs_costs'], dtype=object),
            )
            self.get_logger().info(f"Saved logs: {out_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save logs: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_logs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
