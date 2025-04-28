# launch/all_nodes_launch.py

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include the existing mic_init.launch.py from wheeltec_mic_ros2
    wheeltec_mic = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('wheeltec_mic_ros2'),
                'launch',
                'mic_init.launch.py'
            )
        )
    )

    # voice_recognition_node
    voice_recog = Node(
        package='voice_recognition',
        executable='voice_recognition_node',
        name='voice_recognition_node',
        output='screen'
    )

    # classifier_node
    classifier = Node(
        package='open_task',
        executable='classifier_node',
        name='classifier_node',
        output='screen',
    )

    return LaunchDescription([
        wheeltec_mic,
        voice_recog,
        classifier,
    ])
