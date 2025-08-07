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

        for (v, w), c_obs, c_path, c_goal in zip(
                vel_pairs, obstacle_cost, path_cost, goal_cost):
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
            