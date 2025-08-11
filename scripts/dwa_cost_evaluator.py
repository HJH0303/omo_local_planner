#!/usr/bin/env python3
# dwa_cost_evaluator.py

class CostEvaluator:
    """
    Total cost evaluation combining multiple criteria.
    """
    def __init__(self, cfg, data_provider):
        """Initialize with configuration and data provider"""
        self.wc_obstacle = float(cfg['wc_obstacle'])
        self.wc_path     = float(cfg['wc_path'])
        self.wc_goal     = float(cfg['wc_goal'])
    def normalize(self, obstacle_cost, path_cost, goal_cost):
        pos_obs = [c for c in obstacle_cost if c > 0]
        if pos_obs:
            min_obs = min(pos_obs)
            max_obs = max(pos_obs)
            range_obs = max_obs - min_obs
        else:
            min_obs = 0.0
            range_obs = 1.0
        obs_norm = []
        if range_obs == 0.0: range_obs = 1.0 
        for c in obstacle_cost:
            if c > 0:
                obs_norm.append((c - min_obs) / range_obs)
            else:
                obs_norm.append(c)

        # Path cost: min-max normalization to [0,1]
        min_path    = min(path_cost) if path_cost else 0.0
        max_path    = max(path_cost) if path_cost else 1.0
        range_path  = max_path - min_path 
        if range_path == 0.0: range_path = 1.0 

        path_norm = [(c - min_path) / range_path for c in path_cost]

        # Goal cost: min-max normalization to [0,1]
        min_goal    = min(goal_cost) if goal_cost else 0.0
        max_goal    = max(goal_cost) if goal_cost else 1.0
        range_goal  = max_goal - min_goal
        if range_goal == 0.0: range_goal = 1.0 

        goal_norm = [(c - min_goal) / range_goal for c in goal_cost]

        return obs_norm, path_norm, goal_norm

    def evaluate(self, vel_pairs, obstacle_cost, path_cost, goal_cost):
        """
        Evaluate each trajectory's total cost and return the best velocity sample.

        :param vel_pairs: list of (v, w) velocity samples
        :param obstacle_costs: list of obstacle costs for each trajectory
        :param path_costs: list of path costs for each trajectory
        :param goal_costs: list of goal costs for each trajectory
        :return: tuple(best_pair, best_cost)
        """
        best_cost = float('inf')
        best_pair = None

        obs_norm, path_norm, goal_norm = self.normalize(obstacle_cost, path_cost, goal_cost)
        # print("obs_costs:      ", obs_norm)
        # print("path_costs:     ", path_norm)
        # print("goal_costs:     ", goal_norm)
        for (v, w), c_obs, c_path, c_goal in zip(
                vel_pairs, obs_norm, path_norm, goal_norm):
            # Weighted sum critic
            if c_obs < 0:
                continue

            # Weighted sum critic
            critic = (
                self.wc_obstacle * c_obs +
                self.wc_path     * c_path +
                self.wc_goal     * c_goal
            )

            if critic < best_cost:
                best_cost = critic
                best_pair = (v, w)

        return best_pair, best_cost
            