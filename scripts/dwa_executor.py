#!/usr/bin/env python3
# dwa_executor.py

from geometry_msgs.msg import Twist

class Executor:
    """
    Execute chosen trajectory by converting to Twist command.
    """
    def __init__(self, cfg, data_provider):
        """Initialize with configuration and data provider"""
        # TODO: store cfg and data_provider
        pass

    def to_cmd(self, trajectory):
        """
        Convert a trajectory into a ROS Twist command.
        Args:
            trajectory (list): selected trajectory
        Returns:
            Twist: velocity command
        """
        # TODO: compute linear and angular velocity from trajectory
        raise NotImplementedError