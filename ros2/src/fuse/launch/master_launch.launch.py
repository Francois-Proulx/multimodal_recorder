# ros2/src/fuse/launch/master_launch.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # --- 1. Get Subsystem Paths ---
    video_launch_file = os.path.join(
        get_package_share_directory("video"), "launch", "video_system.launch.py"
    )

    audio_launch_file = os.path.join(
        get_package_share_directory("audio"), "launch", "audio_system.launch.py"
    )

    imu_launch_file = os.path.join(
        get_package_share_directory("imu"), "launch", "imu_system.launch.py"
    )

    # --- 2. Arguments ---
    record_arg = DeclareLaunchArgument("record_bag", default_value="false")

    # --- 3. Include Subsystems ---

    # Video System (Pass arguments down!)
    video_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(video_launch_file),
        launch_arguments={"save_video": "False"}.items(),
    )

    # Audio System
    audio_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(audio_launch_file),
        launch_arguments={"save_audio": "False"}.items(),
    )

    # IMU System
    imu_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(imu_launch_file),
        launch_arguments={"save_imu": "False"}.items(),
    )

    # --- 4. Fuse Node ---
    pkg_share = get_package_share_directory("fuse")
    fuse_config_path = os.path.join(pkg_share, "config", "fuse_config.yaml")
    fuse_node = Node(
        package="fuse",
        executable="fuse",
        name="fuse",
        parameters=[fuse_config_path],
    )

    # --- 5. Recording Logic ---
    recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",  # ros2 bag record command
            "-s",
            "sqlite3",  # Specify storage format
            # --- TOPIC LIST ---
            "/audio_raw",
            "/audio_loc",
            "/imu_raw",
            "/imu_quat",
            "/video_raw/compressed",
            "/video_quat",
            "/camera_info",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_bag")),
    )

    return LaunchDescription(
        [record_arg, video_system, audio_system, imu_system, fuse_node, recorder]
    )
