#!/usr/bin/env python3

import rclpy as rp
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import threading
import time

class RobotTest(Node):
    def __init__(self):
        super().__init__('move')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        # For initial pose storage
        self.pose = type('', (), {})()
        self.pose.x = None
        self.pose.y = None
        self.pose.theta = None

        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

    def odom_callback(self, msg: Odometry):
        self.pose.x = msg.pose.pose.position.x
        self.pose.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose.theta = math.atan2(siny, cosy)

    def move(self, target_dist, kp_x=0.8, kp_theta=4, ki_x=0.2, ki_theta=0.2,
             max_spd=0.2, max_omg=0.8):
        # Flush any previous commands
        zero = Twist()
        for _ in range(20):
            self.cmd_pub.publish(zero)
            time.sleep(0.01)

        # Wait until we have a valid pose
        while self.pose.x is None or self.pose.y is None or self.pose.theta is None:
            time.sleep(0.01)

        start_x = self.pose.x
        start_y = self.pose.y
        goal_x = start_x + target_dist
        goal_y = start_y

        integral_x = 0.0
        integral_theta = 0.0
        prev_time = self.get_clock().now()

        theta_thresh = 0.1
        leak_rate = 0.5

        while True:
            now = self.get_clock().now()
            dt = (now - prev_time).nanoseconds * 1e-9
            prev_time = now

            moved = self.pose.x - start_x
            error_x = target_dist - moved
            if error_x <= 0:
                break

            # Integral for distance
            integral_x += error_x * dt
            I_max_x = max_spd / ki_x
            integral_x = max(min(integral_x, I_max_x), -I_max_x)

            p_gain_x = kp_x * error_x
            i_gain_x = ki_x * integral_x
            raw_speed = p_gain_x + i_gain_x
            speed = max(min(raw_speed, max_spd), 0.0)

            # Heading control
            dx = goal_x - self.pose.x
            dy = goal_y - self.pose.y
            desired_theta = math.atan2(dy, dx)
            error_theta = math.atan2(math.sin(desired_theta - self.pose.theta),
                                     math.cos(desired_theta - self.pose.theta))

            # Conditional integration to avoid wind-up
            if abs(error_theta) < theta_thresh:
                integral_theta += error_theta * dt
            else:
                integral_theta -= leak_rate * integral_theta * dt

            I_max_theta = max_omg / ki_theta
            integral_theta = max(min(integral_theta, I_max_theta), -I_max_theta)

            p_gain_theta = kp_theta * error_theta
            i_gain_theta = ki_theta * integral_theta
            raw_omega = p_gain_theta + i_gain_theta
            omega = max(min(raw_omega, max_omg), -max_omg)

            self.get_logger().info(f"[speed, omega] {speed:.3f}, {omega:.3f}")

            # Publish motion command
            twist = Twist()
            twist.linear.x = speed
            twist.angular.z = omega
            self.cmd_pub.publish(twist)

            time.sleep(0.001)

        # Stop the robot at the end
        stop = Twist()
        self.cmd_pub.publish(stop)


def main():
    rp.init()
    node = RobotTest()

    # Run spin in a separate thread to handle callbacks continuously
    spin_thread = threading.Thread(target=rp.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Execute movement
    node.move(2.0)

    # Ensure robot is stopped
    stop = Twist()
    node.cmd_pub.publish(stop)

    # Shutdown and cleanup
    rp.shutdown()
    spin_thread.join()


if __name__ == "__main__":
    main()
