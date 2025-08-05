#!/usr/bin/env python3
# dwa_obstacle_cost.py

import math
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class CostMap2D:
    """
    2D grid-based local costmap builder:
      - z > z_threshold 인 점군을 obstacle로 표시
      - obstacle_cost 값을 셀에 기록
    """
    def __init__(self, dt, width_m, height_m, resolution, min_z_threshold, max_z_threshold, obstacle_cost):
        self.resolution = resolution
        self.dt = dt
        self.size_x = int(np.ceil(width_m / resolution))
        self.size_y = int(np.ceil(height_m / resolution))
        # origin: 로봇이 맵 중앙에 오도록 설정
        self.origin_x = 0.0
        self.origin_y = -height_m / 2.0

        self.min_z_thresh = min_z_threshold
        self.max_z_thresh = max_z_threshold

        self.obstacle_cost = np.int8(obstacle_cost)
        self.map = np.zeros((self.size_y, self.size_x), dtype=np.int8)

    def clear(self):
        self.map.fill(0)

    def world_to_map(self, x, y):
        i = int((x - self.origin_x) / self.resolution)
        j = int((y - self.origin_y) / self.resolution)
        if 0 <= i < self.size_x and 0 <= j < self.size_y:
            return i, j
        return None

    def update_from_pointcloud(self, cloud: PointCloud2):
        # 1) PointCloud2 → (x,y,z) 튜플 리스트로 변환
        pts_list = []
        for px, py, pz in pc2.read_points(
                cloud, field_names=('x','y','z'), skip_nans=True):
            # NaN/Inf 필터링
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                continue
            # 높이 임계치 필터
            if pz <= self.min_z_thresh:
                continue
            if pz >= self.max_z_thresh:
                continue
            pts_list.append((px, py, pz))

        if not pts_list:
            # 처리할 포인트가 없으면 costmap만 초기화 후 종료
            self.clear()
            return

        # 2) 튜플 리스트 → NumPy 2D float32 array (N×3)
        pts = np.array(pts_list, dtype=np.float32)

        # 3) 인덱스 벡터 연산
        inv_res = 1.0 / self.resolution
        xis = np.floor((pts[:,0] - self.origin_x) * inv_res).astype(np.int32)
        yis = np.floor((pts[:,1] - self.origin_y) * inv_res).astype(np.int32)

        # 4) 범위 클리핑
        valid = (
            (xis >= 0) & (xis < self.size_x) &
            (yis >= 0) & (yis < self.size_y)
        )
        xis = xis[valid]
        yis = yis[valid]

        # 5) 한 번에 맵 clear + 업데이트
        self.clear()
        self.map[yis, xis] = self.obstacle_cost

    def to_occupancy_grid(self, frame_id, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = frame_id

        info = grid.info
        info.resolution = self.resolution
        info.width = self.size_x
        info.height = self.size_y

        # Pose 필드 직접 할당
        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.position.z = 0.0
        origin.orientation.x = 0.0
        origin.orientation.y = 0.0
        origin.orientation.z = 0.0
        origin.orientation.w = 1.0
        info.origin = origin

        flat = self.map.flatten(order='C')
        grid.data = [int(v) for v in flat]

        return grid


class ObstacleCost:
    """
    Obstacle cost computation & local costmap builder for DWA.
    """
    def __init__(self, cfg, data_provider):
        self.cfg = cfg
        self.dp = data_provider

        # CostMap2D 초기화
        self.costmap = CostMap2D(
            cfg['dt'],
            cfg['map_width'],
            cfg['map_height'],
            cfg['cell_resolution'],
            cfg['min_z_threshold'],
            cfg['max_z_threshold'],
            cfg['obstacle_cost']
        )
        self.radius     = self.cfg['robot_radius']
        self.dt         = self.cfg['dt']
        self.sim_time   = self.cfg['sim_time']
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_theta = 0.0

    def update_costmap(self, cloud_msg: PointCloud2):
        self.costmap.update_from_pointcloud(cloud_msg)

    def get_costmap_msg(self, frame_id, stamp):
        return self.costmap.to_occupancy_grid(frame_id, stamp)
    def evaluate_velocity_samples(self, samples):
        """
        Given a list of (v, w) velocity samples, simulate each over the
        planner's sim_time in steps of dt, sample multiple concentric rings
        (outer, mid, inner) plus center, and return a list of average
        obstacle costs—one per sample.
        """
        num_steps = int(self.sim_time / self.dt)
        angle_samples = [24, 16, 8]  # points per ring
        # three concentric rings: outer, mid, inner
        ring_radii = [self.radius, self.radius * 0.66, self.radius * 0.33]

        costs = []
        for v, w in samples:
            x, y, theta = self.init_x, self.init_y, self.init_theta
            path_cost = 0.0
            valid_poses = 0

            for _ in range(num_steps):
                # 1) forward-integrate pose
                x += v * math.cos(theta) * self.dt
                y += v * math.sin(theta) * self.dt
                theta += w * self.dt

                # 2) sample each ring + center
                footprint_vals = []
                for (r_idx, r) in enumerate(ring_radii):
                    for a in np.linspace(0, 2*math.pi, angle_samples[r_idx], endpoint=False):
                        fx = x + r * math.cos(a)
                        fy = y + r * math.sin(a)
                        idx = self.costmap.world_to_map(fx, fy)
                        if idx is not None:
                            i, j = idx
                            footprint_vals.append(self.costmap.map[j, i])

                # center point
                center_idx = self.costmap.world_to_map(x, y)
                if center_idx is not None:
                    ci, cj = center_idx
                    footprint_vals.append(self.costmap.map[cj, ci])

                if footprint_vals:
                    path_cost += sum(footprint_vals) / len(footprint_vals)
                    valid_poses += 1
                else:
                    path_cost += float('inf')
                    valid_poses += 1
                    break

            if valid_poses > 0 and path_cost != float('inf'):
                costs.append(path_cost / valid_poses)
            else:
                costs.append(float('inf'))

        return costs