#!/usr/bin/env python3
"""
voice_follower_bringup.launch.py

Ensures `task_state` starts only after the mic-driver, voice-
recognition, Re-ID detector and goal-publisher nodes are already running.
"""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ------------------------------------------------------------------
    # 1.  Microphone initialisation (comes from wheeltec_mic_ros2)
    # ------------------------------------------------------------------
    mic_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('wheeltec_mic_ros2'),
                'launch',
                'mic_init.launch.py'
            )
        )
    )

    # ------------------------------------------------------------------
    # 2.  Core nodes that must be up *before* we launch task_state
    # ------------------------------------------------------------------
    voice_recognition = Node(
        package='voice_recognition',
        executable='voice_recognition_node',
        name='voice_recognition_node',
        output='screen'
    )

    human_reid = Node(
        package='human_follower',
        executable='yolo_detector',
        name='human_reid',
        output='screen'
    )

    human_following = Node(
        package='human_follower',
        executable='goal_publisher',
        name='human_following',
        output='screen'
    )

    # ------------------------------------------------------------------
    # 3.  task_state – *delay-started* so everyone else is ready
    # ------------------------------------------------------------------
    task_state = Node(
        package='human_follower',
        executable='task_state',
        name='task_state',
        output='screen'
    )

    delayed_task_state = TimerAction(
        period=10.0,        # seconds – adjust if you need longer start-up time
        actions=[task_state]
    )

    # ------------------------------------------------------------------
    # 4.  Assemble LaunchDescription
    # ------------------------------------------------------------------
    return LaunchDescription([
        mic_launch,
        voice_recognition,
        human_reid,
        human_following,
        delayed_task_state,   # <-- launches only *after* the delay
    ])
