import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # --- 1. Dynamic Paths ---
    pkg_share = get_package_share_directory("imu")

    # Config paths
    imu_adq_config_path = os.path.join(pkg_share, "config", "imu_adq_config.yaml")
    imu_quat_config_path = os.path.join(pkg_share, "config", "imu_quat_config.yaml")

    # --- 2. Launch Arguments ---
    # Example: ros2 launch imu imu_system.launch.py save_imu:=True
    save_imu_arg = DeclareLaunchArgument(
        "save_imu",
        default_value="False",
        description="Record processed imu data to disk",
    )
    save_imu_bool = PythonExpression(
        ['"', LaunchConfiguration("save_imu"), '".lower() == "true"']
    )

    # --- 3. NODES ---

    # IMU Acquisition Node
    imu_adq_node = Node(
        package="imu",
        executable="imu_adq",
        name="imu_adq",
        output="screen",
        parameters=[imu_adq_config_path],
    )

    # IMU Processing Node
    imu_quat_node = Node(
        package="imu",  # Your package name
        executable="imu_quat",  # Your python script entry point
        name="imu_quat",
        output="screen",
        parameters=[
            imu_quat_config_path,
            {"save_imu": save_imu_bool},
            # Add other params here directly or via yaml file
        ],
    )

    return LaunchDescription([save_imu_arg, imu_adq_node, imu_quat_node])
