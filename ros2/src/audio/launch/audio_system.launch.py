import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # --- 1. Dynamic Paths ---
    pkg_share = get_package_share_directory("audio")

    # Config paths
    audio_params_path = os.path.join(pkg_share, "config", "audio_params.yaml")

    # --- 2. Launch Arguments ---
    # Example: ros2 launch audio audio_system.launch.py save_audio:=True
    save_audio_arg = DeclareLaunchArgument(
        "save_audio",
        default_value="False",
        description="Record processed audio data to disk",
    )
    save_audio_bool = PythonExpression(
        ['"', LaunchConfiguration("save_audio"), '".lower() == "true"']
    )

    # --- 3. NODES ---

    # Audio Acquisition Node
    audio_adq_node = Node(
        package="audio",
        executable="audio_adq",
        name="audio_adq",
        output="screen",
        parameters=[audio_params_path],
    )

    # Audio Processing Node
    audio_loc_node = Node(
        package="audio",  # Your package name
        executable="audio_loc",  # Your python script entry point
        name="audio_loc",
        output="screen",
        parameters=[
            audio_params_path,
            {"save_audio": save_audio_bool},
            # Add other params here directly or via yaml file
        ],
    )

    return LaunchDescription([save_audio_arg, audio_adq_node, audio_loc_node])
