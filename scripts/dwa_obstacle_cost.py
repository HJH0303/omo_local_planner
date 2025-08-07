#!/usr/bin/env python3
# dwa_obstacle_cost.py

import math
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class CostMap2D:
    def __init__(self, dt, width_m, height_m, resolution,
                 min_z_threshold, max_z_threshold,
                 obstacle_cost, inflation_radius,
                 cost_scaling_factor, inscribed_radius):
        self.resolution = float(resolution)
        self.dt = dt
        self.size_x = int(np.ceil(width_m / resolution))
        self.size_y = int(np.ceil(height_m / resolution))
        self.origin_x = 0.0
        self.origin_y = -height_m / 2.0
        self.min_z_thresh=min_z_threshold
        self.max_z_thresh=max_z_threshold

        self.obstacle_cost = int(obstacle_cost)
        self.inflation_radius = float(inflation_radius)
        self.cost_scaling_factor = float(cost_scaling_factor)
        self.inscribed_radius = float(inscribed_radius)

        self.cell_inflation_radius = int(math.ceil(self.inflation_radius / self.resolution))
        self.map = np.zeros((self.size_y, self.size_x), dtype=np.int32)

        self._compute_caches()

    def clear(self):
        self.map.fill(0)

    def world_to_map(self, x, y):
        i = int((x - self.origin_x) / self.resolution)
        j = int((y - self.origin_y) / self.resolution)
        if 0 <= i < self.size_x and 0 <= j < self.size_y:
            return i, j
        return None


    def _compute_caches(self):
        r = self.cell_inflation_radius
        size = 2 * r + 1
        self.distance_matrix = [[0] * size for _ in range(size)]
        self.cost_matrix = [[0] * size for _ in range(size)]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = math.hypot(dx * self.resolution, dy * self.resolution)
                self.distance_matrix[dy + r][dx + r] = dist
                if dist <= self.inflation_radius:
                    if dist <= self.inscribed_radius:
                        cost = self.obstacle_cost
                    else:
                        cost = self.obstacle_cost * math.exp(
                            -self.cost_scaling_factor * (dist - self.inscribed_radius)
                        )
                    self.cost_matrix[dy + r][dx + r] = int(min(max(cost, 0), self.obstacle_cost))
                else:
                    self.cost_matrix[dy + r][dx + r] = 0

    def inflate_obstacles(self):
        radius_cells = int(self.inflation_radius / self.resolution)
        new_map = self.map.copy()
        obst_coords = np.column_stack(np.where(self.map == self.obstacle_cost))

        for j, i in obst_coords:
            for dj in range(-radius_cells, radius_cells + 1):
                for di in range(-radius_cells, radius_cells + 1):
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.size_y and 0 <= ni < self.size_x:
                        dist = math.hypot(dj * self.resolution,
                                          di * self.resolution)
                        if dist <= self.inflation_radius:
                            cost = int(self.obstacle_cost *
                                       (1 - (dist / self.inflation_radius)))
                            new_map[nj, ni] = max(new_map[nj, ni], cost)
        self.map = new_map

    def update_from_pointcloud(self, cloud: PointCloud2):
        pts_list = []
        for px, py, pz in pc2.read_points(
                cloud, field_names=('x','y','z'), skip_nans=True):
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                continue
            if pz <= self.min_z_thresh or pz >= self.max_z_thresh:
                continue
            pts_list.append((px, py, pz))

        if not pts_list:
            self.clear()
            return

        pts = np.array(pts_list, dtype=np.float32)
        inv_res = 1.0 / self.resolution
        xis = np.floor((pts[:,0] - self.origin_x) * inv_res).astype(np.int32)
        yis = np.floor((pts[:,1] - self.origin_y) * inv_res).astype(np.int32)

        valid = (
            (xis >= 0) & (xis < self.size_x) &
            (yis >= 0) & (yis < self.size_y)
        )
        xis, yis = xis[valid], yis[valid]

        self.clear()
        self.map[yis, xis] = self.obstacle_cost
        # inflate after marking obstacles
        self.inflate_obstacles()

    def to_occupancy_grid(self, frame_id, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = frame_id

        info = grid.info
        info.resolution = self.resolution
        info.width = self.size_x
        info.height = self.size_y

        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.orientation.w = 1.0
        info.origin = origin

        flat = self.map.flatten(order='C')
        grid.data = [int(v) for v in flat]
        return grid


class ObstacleCost:
    """
    Obstacle cost computation & local costmap builder for DWA with Nav2-style inflation.
    """
    def __init__(self, cfg, data_provider):
        self.cfg = cfg
        self.dp = data_provider
        self.costmap = CostMap2D(
            cfg['dt'], cfg['map_width'], cfg['map_height'],
            cfg['cell_resolution'],
            cfg['min_z_threshold'], cfg['max_z_threshold'],
            cfg['obstacle_cost'], cfg['inflation_radius'],
            cfg['cost_scaling_factor'], float(cfg['robot_radius'])
        )
        self.dt = float(cfg['dt'])
        self.sim_time = float(cfg['sim_time'])
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_theta = 0.0

    def update_costmap(self, cloud_msg: PointCloud2):
        self.costmap.update_from_pointcloud(cloud_msg)

    def get_costmap_msg(self, frame_id, stamp):
        return self.costmap.to_occupancy_grid(frame_id, stamp)

    def evaluate_velocity_samples(self, samples):
        num_steps = int(self.sim_time / self.dt)
        angle_samples = [24, 8]
        ring_radii = [float(self.cfg['robot_radius']), float(self.cfg['robot_radius']) * 0.5]

        costs = []
        for v, w in samples:
            x, y, theta = self.init_x, self.init_y, self.init_theta
            path_cost = 0.0
            valid_poses = 0
            early_exit = False

            for _ in range(num_steps):
                x += v * math.cos(theta) * self.dt
                y += v * math.sin(theta) * self.dt
                theta += w * self.dt

                footprint_vals = []
                for r_idx, r in enumerate(ring_radii):
                    for a in np.linspace(0, 2*math.pi, angle_samples[r_idx], endpoint=False):
                        fx = x + r * math.cos(a)
                        fy = y + r * math.sin(a)
                        idx = self.costmap.world_to_map(fx, fy)
                        if idx:
                            i, j = idx
                            footprint_vals.append(int(self.costmap.map[j, i]))
                center_idx = self.costmap.world_to_map(x, y)
                if center_idx:
                    ci, cj = center_idx
                    footprint_vals.append(int(self.costmap.map[cj, ci]))

                if any(val == self.costmap.obstacle_cost for val in footprint_vals):
                    costs.append(-1000.0)
                    early_exit = True
                    break

                if footprint_vals:
                    path_cost += sum(footprint_vals) / len(footprint_vals)
                    valid_poses += 1
                else:
                    path_cost = float('inf')
                    valid_poses += 1
                    break

            if early_exit:
                continue
            if valid_poses > 0 and path_cost != float('inf'):
                costs.append(path_cost / valid_poses)
            else:
                costs.append(float('inf'))

        return costs
