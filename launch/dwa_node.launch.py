import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    pkg_share = get_package_share_directory('omo_local_planner')
    params_file = os.path.join(pkg_share, 'config', 'config.yaml')

    return LaunchDescription([
        Node(
            package='omo_local_planner',
            executable='dwa_node.py',
            name='dwa_planner',
            output='screen',
            parameters=[params_file],
        )
    ])

