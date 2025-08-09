#!/usr/bin/env python3
# dwa_trj_generator.py

import numpy as np
import math

class TrajectoryGenerator:
    """
    Trajectory sampling module for DWA.
    """
    def __init__(self, cfg, data_provider):
        """
        Initialize with configuration and data provider.
        :param cfg: dict with keys sim_period, acc_lim_v, acc_lim_w,
                    sim_time, dt, v_samples, w_samples.
        :param data_provider: DataProvider instance to fetch odometry and logger.
        """
        self.cfg = cfg
        self.dp = data_provider

        # Dynamic window parameters
        self.sim_period = cfg['sim_period']
        self.sim_period = 1/self.sim_period
        self.acc_lim_v = float(cfg['acc_lim_v'])
        self.acc_lim_w = float(cfg['acc_lim_w'])

        # Simulation parameters
        self.sim_time = cfg['sim_time']
        self.dt = cfg['dt']
        self.v_samples = int(cfg['v_samples'])
        self.w_samples = int(cfg['w_samples'])
        self.prev_min = 0.0
        self.prev_max = 0.0
        
    def sample_velocities(self):
        """
        Compute dynamic window of (v, w) samples based on current robot state.
        :returns: np.ndarray of shape (N, 2) containing [v, w] pairs.
        """
        odom = self.dp.get_odometry()

        # Current velocities
        v_cur = odom.twist.twist.linear.x
        w_cur = odom.twist.twist.angular.z

        # Compute min/max velocities
        # v_cur = max(0.05, v_cur)
        v_min = v_cur - self.acc_lim_v * self.sim_period
        v_max = v_cur + self.acc_lim_v * self.sim_period
        # v_min = 0.
        # v_max = 0.3
        w_min =  - self.acc_lim_w * self.sim_period
        w_max =  + self.acc_lim_w * self.sim_period
        # w_min =  - self.acc_lim_w * self.sim_period
        # w_max =  + self.acc_lim_w * self.sim_period
        
        if v_min <= 0.0:
            v_min = 0.0
        
        if v_max < 0.0:
            v_max = self.prev_max

        if v_max > 0.4: v_max = 0.4

        self.prev_min= v_min 
        self.prev_max= v_max 

        # self.dp._node.get_logger().info(
        #         f'linear_min={v_min}, linear_max={v_max}'
        #     )
        # Discretize into samples

        vs = np.linspace(v_min, v_max, self.v_samples)
        ws = np.linspace(w_min, w_max, self.w_samples)
        V, W = np.meshgrid(vs, ws, indexing='ij')
        return np.stack((V.ravel(), W.ravel()), axis=1)

    def generate_trajectories(self):
        """
        Generate candidate trajectories based on current odometry.
        :param vel_pairs: optional np.ndarray of [v, w] samples; if None, uses sample_velocities().
        :returns: list of trajectories, each a list of (x, y, theta) tuples.
        """
        odom = self.dp.get_odometry()

        # Extract position and compute yaw from quaternion directly
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        # Determine velocity samples
        vel_pairs = self.sample_velocities()
        # Simulate each (v, w) pair
        num_steps = int(self.sim_time / self.dt)
        trajectories = []
        for v, w in vel_pairs:
            x = y = th = 0.0
            traj = []
            for _ in range(num_steps):
                traj.append((x, y, th))
                x += v * math.cos(th) * self.dt
                y += v * math.sin(th) * self.dt
                th += w * self.dt
            trajectories.append(traj)

        return vel_pairs, trajectories