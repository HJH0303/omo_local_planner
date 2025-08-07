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

        self.declare_parameter('corners', [3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 0.0, 0.0])
        self.declare_parameter('goal_tolerance', 0.3)
        
        points = self.get_parameter('corners').value

        self.corners = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        self.tolerance = self.get_parameter('goal_tolerance').value
        self.current_index = 0
        self.current_pose = None
        self.lap_count = 0

        # Subscriber 
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)

        #Publisher
        self.goal_pub = self.create_publisher(PointStamped, '/target_point', 10)

        #Timer
        self.create_timer(0.1, self.timer_cb)

        #Publish first waypoint
        self.publish_goal()

    def odom_cb(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def publish_goal(self):
        x, y = self.corners[self.current_index]
        goal = PointStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.point.x = x
        goal.point.y = y
        goal.point.z = 0.0
        self.goal_pub.publish(goal)

    def timer_cb(self):
        if self.current_pose is None:
            return 
        gx, gy = self.corners[self.current_index]     
        dx = gx - self.current_pose.x
        dy = gy - self.current_pose.y
        dist = math.hypot(dx, dy)

        # if dist <= self.tolerance:
        #     self.current_index = (self.current_index +1) % len(self.corners)
        #     self.publish_goal()
        if dist <= self.tolerance:
            next_index = (self.current_index +1) % len(self.corners)
            if lap_count == 0:
                lap_count +=1
                if lap_count >=2:
                    self.timer.cancle()
                    return
            self.current_index = next_index
            self.publish_goal()



def main(args=None):
    rclpy.init(args=args)
    node = RectTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()