#!/usr/bin/env python3
# dwa_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, Path, OccupancyGrid

from dwa_convert_global_path import GlobalPathConverter
from dwa_obstacle_cost import ObstacleCost
# from dwa_data_provider import DataProvider
# from dwa_trajectory_generator import TrajectoryGenerator
# from dwa_cost_evaluator import CostEvaluator
# from dwa_executor import Executor

class DWANode(Node):
    """
    Debug mode: only global path conversion from waypoint.
    Other DWA modules are commented out for path conversion testing.
    """
    def __init__(self):
        super().__init__('dwa_node')
        # ROS2 파라미터 로드
        self.declare_parameter('global_path_resolution', 0.1)
        self.declare_parameter('sim_period', 10)
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('map_width', 3.5)       # meters
        self.declare_parameter('map_height', 3.5)      # meters
        self.declare_parameter('cell_resolution', 0.1)     # m/cell
        self.declare_parameter('min_z_threshold', -0.2)     # meters
        self.declare_parameter('max_z_threshold', 2.0)     # m/cell
        self.declare_parameter('obstacle_cost', 100)    # int8 value
        self.declare_parameter('robot_base_frame', 'base_link')

        self.cfg = {
            'global_path_resolution':       self.get_parameter('global_path_resolution').value,
            'sim_period':                   self.get_parameter('sim_period').value,
            'dt':                           self.get_parameter('dt').value,
            'map_width':                    self.get_parameter('map_width').value,
            'map_height':                   self.get_parameter('map_height').value,
            'cell_resolution':              self.get_parameter('cell_resolution').value,
            'min_z_threshold':              self.get_parameter('min_z_threshold').value,
            'max_z_threshold':              self.get_parameter('max_z_threshold').value,
            'obstacle_cost':                self.get_parameter('obstacle_cost').value,
            'robot_base_frame':             self.get_parameter('robot_base_frame').value,
        }
        period = self.cfg['sim_period']
        sim_period = 1.0/period
        # 내부 메시지 저장용 변수
        self.current_odom = None
        self.current_path = None
        self.base_frame = self.cfg['robot_base_frame']

        # 1) 토픽 구독: 데이터 저장
        self.create_subscription(
            PointCloud2,
            '/voxel_cloud',
            lambda msg: setattr(self, 'current_pc', msg),
            10
        )
        # self.create_subscription(
        #     Odometry,
        #     '/odometry/filtered',
        #     lambda msg: setattr(self, 'current_odom', msg),
        #     10
        # )

        # 2) waypoint_point -> path 생성
        self.path_converter = GlobalPathConverter(self.cfg)
        self.create_subscription(
            PointStamped,
            '/target_point',
            self._waypoint_cb,
            10
        )

        # Path 퍼블리셔
        self.path_pub = self.create_publisher(Path, '/global_path', 10)

        # OccupancyGrid 퍼블리셔
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap', 10
        )
        # DWA 모듈 인스턴스화 (주석 처리)
        # dp = DataProvider(self)
        # self.traj_gen  = TrajectoryGenerator(self.cfg, dp)
        self.obs_cost = ObstacleCost(self.cfg, self)
        # self.evaluator = CostEvaluator(self.cfg, dp)
        # self.executor  = Executor(self.cfg, dp)

        # 제어 루프 타이머 (주석 처리)
        # self.create_timer(self.cfg['dt'], self._control_loop)

        
        # —– 주기 타이머 —– #
        self.create_timer(sim_period, self._on_timer)

        self.get_logger().info('DWANode initialized (publishing local costmap).')
    
    def _waypoint_cb(self, msg: PointStamped):
        # Global path 생성 및 퍼블리시
    
        path = self.path_converter.generate(msg)
        self.get_logger().info(f'Generated global path with {len(path.poses)} poses.')
        self.current_path = path
        self.path_pub.publish(path)

    def _on_timer(self):
        if not hasattr(self, 'current_pc') or self.current_pc is None:

            return

        # 1) CostMap2D 업데이트
        self.obs_cost.update_costmap(self.current_pc)
        # 2) OccupancyGrid 메시지 생성 및 퍼블리시
        grid = self.obs_cost.get_costmap_msg(
            frame_id=self.base_frame,
            stamp=self.get_clock().now().to_msg()
        )
        self.costmap_pub.publish(grid)
        self.get_logger().debug('Published /local_costmap')

    # DWA 전체 제어 루프 (주석 처리)
    # def _control_loop(self):
    #     if not (self.current_pc and self.current_odom and self.current_path):
    #         return
    #     trajectories = self.traj_gen.generate()
    #     scores = [self.evaluator.evaluate(t, self.obs_cost.compute(t)) for t in trajectories]
    #     best = trajectories[scores.index(min(scores))]
    #     cmd = self.executor.to_cmd(best)
    #     self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
