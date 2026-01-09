import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # --- 1. Dynamic Paths ---
    pkg_share = get_package_share_directory("video")

    # Config paths
    usb_cam_config_path = os.path.join(pkg_share, "config", "usb_cam_params.yaml")
    quat_config_path = os.path.join(pkg_share, "config", "video_quat_config.yaml")

    # --- 2. Launch Arguments ---
    # Example: ros2 launch video video_system.launch.py save_video:=True
    save_video_arg = DeclareLaunchArgument(
        "save_video",
        default_value="False",
        description="Record processed video to disk",
    )
    save_video_bool = PythonExpression(
        ['"', LaunchConfiguration("save_video"), '".lower() == "true"']
    )

    # --- 3. NODES ---

    # C++ Camera Driver
    usb_cam_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="camera_driver",
        output="screen",
        parameters=[usb_cam_config_path],
        remappings=[
            # Remap the standard usb_cam topic to match what your system expects
            ("image_raw/compressed", "/video_raw/compressed")
        ],
    )

    # Python Processing Node
    processor_node = Node(
        package="video",  # Your package name
        executable="video_quat",  # Your python script entry point
        name="video_quat",
        output="screen",
        parameters=[
            quat_config_path,
            {"save_video": save_video_bool},
            # Add other params here directly or via yaml file
        ],
    )

    return LaunchDescription([save_video_arg, usb_cam_node, processor_node])
