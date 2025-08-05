#!/usr/bin/env python3

# dwa_convert_global_path.py

import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped


class GlobalPathConverter:
    """
    Converts a single 3D waypoint into a straight-line 2D Path from the robot origin.
    """
    def __init__(self, cfg):
        """
        Initialize with configuration dict (expects 'global_path_resolution').
        """
        self.resolution = cfg.get('global_path_resolution', 0.1)

    def generate(self, waypoint_msg: PointStamped) -> Path:
        """
        Generate Path message from PointStamped waypoint.

        Args:
            waypoint_msg (PointStamped): input waypoint

        Returns:
            Path: straight-line path from (0,0) to waypoint in 2D
        """
        # 1) 시작점 (로봇 원점)
        start_x, start_y = 0.0, 0.0

        # 2) 목표점 2D 좌표 추출
        goal_x = waypoint_msg.point.x
        goal_y = waypoint_msg.point.y

        # 3) 직선 거리 및 샘플 포인트 개수 계산
        dist = math.hypot(goal_x - start_x, goal_y - start_y)
        num_points = max(2, int(dist / self.resolution) + 1)

        # 4) 방향 계산 및 쿼터니언 생성
        yaw = math.atan2(goal_y - start_y, goal_x - start_x)
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        # 5) Path 메시지 구성
        path = Path()
        path.header = waypoint_msg.header
        for i in range(num_points):
            ratio = i / (num_points - 1)
            x = start_x + (goal_x - start_x) * ratio
            y = start_y + (goal_y - start_y) * ratio

            pose = PoseStamped()
            pose.header = waypoint_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            # orientation: 직선 방향 고정
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path.poses.append(pose)

        return path
