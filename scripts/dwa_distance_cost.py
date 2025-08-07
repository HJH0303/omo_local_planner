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
        Calculate path cost for a single trajectory using perpendicular distance.
        :param trajectory: list of (x, y, theta) tuples
        :param waypoint_msg: geometry_msgs/PointStamped of the local goal
        :return: sum of perpendicular distances to the line from current position to local goal
        """
        # 1) Current position (start)
        odom = self.dp.get_odometry()
        # waypoint = self.dp.get_waypoint()
        start = odom.pose.pose.position
        x0, y0 = start.x, start.y

        # # 2) Waypoint coordinates
        # x_goal = waypoint.point.x
        # y_goal = waypoint.point.y
        x_goal = 6.0
        y_goal = 0.0
        # 3) Line parameters
        dx = x_goal - x0
        dy = y_goal - y0
        L = math.hypot(dx, dy)
        if L == 0.0:
            return 0.0

        # 4) Sum perpendicular distances
        cost = 0.0
        for x, y, _ in trajectory:
            dist = abs(dy * (x - x0) - dx * (y - y0)) / L
            cost += dist
        cost_avg = cost/len(trajectory)
        return cost_avg

    def goal_cost(self, trajectory):
        """
        Calculate goal cost for a single trajectory.
        :param trajectory: list of (x, y, theta) tuples
        :param waypoint_msg: geometry_msgs/PointStamped of the local goal
        :return: Euclidean distance from last trajectory point to local goal
        """
        # waypoint = self.dp.get_waypoint()
        x_goal = 6.0
        y_goal = 0.0
        # Last trajectory point
        x_end, y_end, _ = trajectory[-1]

        # Waypoint coordinates
        # x_goal = waypoint.point.x
        # y_goal = waypoint.point.y
        x_goal = 6.0
        y_goal = 0.0
        # Euclidean distance
        return math.hypot(x_goal - x_end*0.01, y_goal - y_end*0.01)
    

    def evaluate(self, trajectories):
        """
        Evaluate costs for a list of trajectories against a given waypoint.
        :param trajectories: list of trajectories (each a list of (x, y, theta) tuples)
        :param waypoint_msg: geometry_msgs/PointStamped of the local goal
        :return: two lists: path_costs, goal_costs
        """
        path_costs = [self.path_cost(traj) for traj in trajectories]
        goal_costs = [self.goal_cost(traj) for traj in trajectories]
        return path_costs, goal_costs