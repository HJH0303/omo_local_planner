#!/usr/bin/env python3
# rectangle_test_node.py

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

class RectTest(Node):
    def __init__(self):
        super().__init__('rect_test_node')
        self.declare_parameter('corners', [6.0, 0.0])
        self.declare_parameter('goal_tolerance', 0.1)

        points = self.get_parameter('corners').value
        self.corners = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        self.tolerance = float(self.get_parameter('goal_tolerance').value)
        self.current_index = 0
        self.current_pose = None

        # Subscriber
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)

        # Publisher
        self.goal_pub = self.create_publisher(PointStamped, '/target_point', 10)

        # Timer 핸들 저장(중요!)
        self.timer = self.create_timer(0.1, self.timer_cb)

    def odom_cb(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def publish_goal(self, dx, dy):
        goal = PointStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.point.x = float(dx)
        goal.point.y = float(dy)
        goal.point.z = 0.0
        self.goal_pub.publish(goal)

    def timer_cb(self):
        if self.current_pose is None or not self.corners:
            return

        gx, gy = self.corners[self.current_index]
        dx = gx - self.current_pose.x
        dy = gy - self.current_pose.y
        dist = math.hypot(dx, dy)

        # # 목표 도달하면 타이머 중지
        # if dist <= self.tolerance:
        #     self.get_logger().info(
        #         f"Reached goal ({gx:.2f}, {gy:.2f}) within tol {self.tolerance:.2f}. Stopping timer."
        #     )
        #     # 필요하면 마지막으로 0 오프셋(정지 명령 대용) 한번 더 보냄
        #     self.publish_goal(0.0, 0.0)
        #     self.timer.cancel()  # ← 핵심: create_timer의 반환 핸들로 cancel 호출
        #     return

        # 아직 못 도달: 현재 목표까지의 상대 오프셋 퍼블리시
        self.publish_goal(dx, dy)


def main(args=None):
    rclpy.init(args=args)
    node = RectTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
