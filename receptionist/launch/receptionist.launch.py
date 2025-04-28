import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Include the microphone initialization launch from wheeltec_mic_ros2
    mic_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('wheeltec_mic_ros2'),
                'launch',
                'mic_init.launch.py'
            )
        )
    )

    # Nodes to run
    name_and_drink = Node(
        package='receptionist',
        executable='name_and_drink_node',
        name='name_and_drink_node',
        output='screen'
    )

    voice_recognition = Node(
        package='voice_recognition',
        executable='voice_recognition_node',
        name='voice_recognition_node',
        output='screen'
    )

    detect_chair = Node(
        package='yolo_detection',
        executable='detect_chair',
        name='detect_chair',
        output='screen'
    )

    detect_human = Node(
        package='yolo_detection',
        executable='detect_human',
        name='detect_human',
        output='screen'
    )

    guest_description = Node(
        package='receptionist',
        executable='guest_description_node',
        name='guest_description_node',
        output='screen'
    )

    guest_sentence = Node(
        package='receptionist',
        executable='guest_description_sentence_node',
        name='guest_description_sentence_node',
        output='screen'
    )

    return LaunchDescription([
        mic_launch,
        name_and_drink,
        voice_recognition,
        detect_chair,
        detect_human,
        guest_description,
        guest_sentence,
    ])
