#!/usr/bin/env python3
# dwa_distance_cost.py

import math

class DistanceCosts:
    """
    Compute path cost and goal cost for DWA trajectories.

    Path cost: sum of perpendicular distances from trajectory points
               to the straight-line path between current position and a given waypoint.
    Goal cost: Euclidean distance between the trajectory's final point and a given waypoint.
    """
    def __init__(self, data_provider):
        """
        :param data_provider: DataProvider instance providing odometry.
        """
        self.dp = data_provider

    def path_cost(self, trajectory):
        """
        Perpendicular-distance path cost using precomputed line params from CarrotNode.

        Uses normalized line coefficients (A, B, C) for the line Ax + By + C = 0,
        where (A^2 + B^2) == 1. If not normalized, this function will normalize them.

        :param trajectory: list of (x, y, theta) tuples in the SAME FRAME as the line params (e.g., 'odom')
        :return: average perpendicular distance from trajectory points to the line
        """
        if not trajectory:
            return float('inf')

        path = self.dp.get_gpath_params()  # geometry_msgs/Vector3Stamped or None
        if path is None:
            return float('inf')

        A = float(path.vector.x)
        B = float(path.vector.y)
        C = float(path.vector.z)

        # Ensure (A, B, C) are normalized so distance = |A*x + B*y + C|
        norm = math.hypot(A, B)
        if norm < 1e-9:
            # Degenerate line (goal == start or invalid) -> no lateral error
            return 0.0
        if abs(norm - 1.0) > 1e-6:
            A /= norm
            B /= norm
            C /= norm

        acc = 0.0
        for x, y, _ in trajectory:
            acc += abs(A * x + B * y + C)

        return acc / len(trajectory)
    

    def alignment_cost(self, trajectory, xshift: float = -0.3, yshift: float = 0.0):
        if not trajectory:
            return float('inf')
        
        path = self.dp.get_gpath_params()
        if path is None:
            return float('inf')
        
        A = float(path.vector.x)
        B = float(path.vector.y)
        C = float(path.vector.z)

        norm = math.hypot(A, B)
        if norm < 1e-9:
            # Degenerate line (goal == start or invalid) -> no lateral error
            return 0.0
        if abs(norm - 1.0) > 1e-6:
            A /= norm
            B /= norm
            C /= norm

        acc = 0.0
        for x, y, theta in trajectory:
            px = x + xshift * math.cos(theta) + yshift * math.cos(theta + math.pi / 2.0)
            py = y + xshift * math.sin(theta) + yshift * math.sin(theta + math.pi / 2.0)
            acc += abs(A * px + B * py + C)

        return acc / len(trajectory)


    def goal_cost(self, trajectory):
        """
        Calculate goal cost for a single trajectory.
        :param trajectory: list of (x, y, theta) tuples
        :param waypoint_msg: geometry_msgs/PointStamped of the local goal
        :return: Euclidean distance from last trajectory point to local goal
        """
        waypoint = self.dp.get_waypoint()
        # Last trajectory point
        x_end, y_end, _ = trajectory[-1]

        # Waypoint coordinates
        x_goal = waypoint.point.x
        y_goal = waypoint.point.y
        
        # Euclidean distance
        return math.hypot(x_goal - x_end, y_goal - y_end)
        
    def goal_center_cost(self, trajectory, xshift: float = -0.3, yshift: float = 0.0) -> float:
        """
        Distance from shifted end-of-trajectory point to local goal.

        We shift the last trajectory point by `xshift` meters along its heading (theta)
        and by `yshift` meters sideways (left = positive), then compute Euclidean distance
        to the current waypoint.
        """
        if not trajectory:
            return float("inf")

        waypoint = self.dp.get_waypoint()
        if waypoint is None:
            return float("inf")

        # Goal (local waypoint)
        x_goal = waypoint.point.x -0.3
        y_goal = waypoint.point.y

        # Last trajectory pose
        x_end, y_end, theta_end = trajectory[-1]

        # Apply forward/backward and lateral shifts in the local heading frame
        px = x_end + xshift * math.cos(theta_end) + yshift * math.cos(theta_end + math.pi / 2.0)
        py = y_end + xshift * math.sin(theta_end) + yshift * math.sin(theta_end + math.pi / 2.0)

        # Cost = distance from shifted point to goal
        return math.hypot(x_goal - px, y_goal - py)

    def evaluate(self, trajectories):
        """
        Evaluate costs for a list of trajectories against a given waypoint.
        :param trajectories: list of trajectories (each a list of (x, y, theta) tuples)
        :param waypoint_msg: geometry_msgs/PointStamped of the local goal
        :return: two lists: path_costs, goal_costs
        """
        path_costs          = [self.path_cost(traj) for traj in trajectories]
        alignment_costs     = [self.alignment_cost(traj) for traj in trajectories]
        goal_costs          = [self.goal_cost(traj) for traj in trajectories]
        goal_center_costs   = [self.goal_center_cost(traj) for traj in trajectories]
        return path_costs, alignment_costs, goal_costs, goal_center_costs