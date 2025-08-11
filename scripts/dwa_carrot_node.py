#!/usr/bin/env python3
# dwa_carrot_node.py

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from types import SimpleNamespace



def yaw_from_quat(q) -> float:
    # REP-103 yaw
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)

class CarrotNode(Node):
    def __init__(self):
        super().__init__('carrot_node')

        # ---- Parameters ----
        self.declare_parameter('corners', [7.0, 0.0])
        self.declare_parameter('goal_tolerance', 0.1)    # [m]
        self.declare_parameter('lookahead_max', 6.0)      # [m]
        self.declare_parameter('lookahead_min', 0.12)      # [m] set 0.0 to disable
        self.declare_parameter('timer_period', 0.1)       # [s]
        self.declare_parameter('advance_when_reached', True)
        self.declare_parameter('loop_corners', False)

        pts = self.get_parameter('corners').value
        if len(pts) < 2 or len(pts) % 2 != 0:
            raise ValueError("corners must be a flat list of [x0, y0, x1, y1, ...]")

        self.goals = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
        self.idx = 0

        self.tol    = float(self.get_parameter('goal_tolerance').value)
        self.Lmax   = float(self.get_parameter('lookahead_max').value)
        self.Lmin   = float(self.get_parameter('lookahead_min').value)
        period      = float(self.get_parameter('timer_period').value)
        self.advance= bool(self.get_parameter('advance_when_reached').value)
        self.loop   = bool(self.get_parameter('loop_corners').value)

        self.state: SimpleNamespace | None = None  # x, y, yaw

        # ---- ROS I/O ----
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)
        self.tp_pub = self.create_publisher(PointStamped, '/target_point', 10)
        
        self.gpath_seg_pub = self.create_publisher(Path, '/global_path_segment', 10)
        self.gpath_param_pub = self.create_publisher(Vector3Stamped, '/global_path_params', 10)

        # ---- Timer ----
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(
            f"CarrotNode: tol={self.tol:.2f}m, Lmax={self.Lmax:.2f}m, Lmin={self.Lmin:.2f}m, goals={len(self.goals)}, "
            f"advance={self.advance}, loop={self.loop}"
        )

        # ------global_path_convert--------
    def publish_path_seg(self, x0, y0, xg, yg):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'

        def mk_pose(x, y):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            return ps
        
        path.poses = [mk_pose(x0, y0), mk_pose(xg, yg)]
        self.gpath_seg_pub.publish(path)

    def publish_path_params(self, x0, y0, xg, yg):
        dx = xg -x0
        dy = yg -y0
        L = math.hypot(dx, dy)
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        if L < 1e-9:
            msg.vector.x = 0.0
            msg.vector.y = 0.0
            msg.vector.z = 0.0
        else:
            A = dy/L
            B = -dx/L
            C = (dx * y0 - dy * x0)/L

            msg.vector.x = A
            msg.vector.y = B
            msg.vector.z = C
        self.gpath_param_pub.publish(msg)

    

    # ---------------- Callbacks ----------------
    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.state = SimpleNamespace(x=p.x, y=p.y, yaw=yaw_from_quat(q))

    def timer_cb(self):
        if self.state is None or not self.goals:
            return

        gx, gy = self.goals[self.idx]

        # 1) World offset & distance to current goal
        dx_w = gx - self.state.x
        dy_w = gy - self.state.y
        dist = math.hypot(dx_w, dy_w)

        # 2) Reached goal?
        if dist <= self.tol:
            if self.advance:
                if self.idx + 1 < len(self.goals):
                    self.idx += 1
                    self.get_logger().info(f"Reached goal {self.idx-1}, advancing to {self.idx}")
                elif self.loop and len(self.goals) > 1:
                    self.idx = 0
                    self.get_logger().info("Reached last goal, looping to index 0")
                else:
                    # Stop at last goal: publish (0,0) and keep idling
                    self.publish_target(0.0, 0.0)
                    return
            else:
                self.publish_target(0.0, 0.0)
                return

            # update current goal after advancing
            gx, gy = self.goals[self.idx]
            dx_w = gx - self.state.x
            dy_w = gy - self.state.y
            dist = math.hypot(dx_w, dy_w)

        # 3) Rotate into base_link (ex, ey) = R(-yaw) * [dx_w, dy_w]
        cy, sy = math.cos(self.state.yaw), math.sin(self.state.yaw)
        ex =  cy * dx_w + sy * dy_w
        ey = -sy * dx_w + cy * dy_w

        # 4) Carrot length L = clamp(dist, Lmin, Lmax)
        r = math.hypot(ex, ey)
        if r < 1e-6:
            self.publish_target(0.0, 0.0)
            return

        L = min(dist, self.Lmax)
        if self.Lmin > 0.0:
            # keep a small forward carrot to reduce jitter near zero, but don't exceed the actual remaining distance
            L = max(self.Lmin, L)
            L = min(L, dist)

        k = L / r
        tx = ex * k
        ty = ey * k

        # 5) Publish single carrot target for DWA
        self.publish_target(tx, ty)
        self.publish_path_seg(self.state.x, self.state.y, gx, gy)
        self.publish_path_params(self.state.x, self.state.y, gx, gy)

        # ---------------- Utils ----------------
    def publish_target(self, x: float, y: float):
        msg = PointStamped()
        msg.header.frame_id = 'base_link'  # IMPORTANT: relative to robot
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        self.tp_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = CarrotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()