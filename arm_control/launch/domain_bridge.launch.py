# my_vector_bridge_launch.py

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # adjust this to your package name
    pkg_share = get_package_share_directory('arm_control')

    # path to the YAML we defined earlier
    config_file = os.path.join(pkg_share, 'config', 'vector_bridge.yaml')

    return LaunchDescription([
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            name='vector_bridge',
            output='screen',
            # override domains here if you like:
            arguments=[
                '--from', '221',           # original DDS domain
                '--to',   '222',           # target DDS domain
                config_file             # path to your YAML
            ],
        ),
    ])
