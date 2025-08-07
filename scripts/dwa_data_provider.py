#!/usr/bin/env python3
# dwa_data_provider.py

class DataProvider():
    """
    Save the latest topics and provide datas to the modules
    """
    def __init__(self, node):
        self._node = node

    def get_pointcloud(self):
        return self._node.current_pc

    def get_odometry(self):
        return self._node.current_odom

    def get_waypoint(self):
        return self._node.waypoint
