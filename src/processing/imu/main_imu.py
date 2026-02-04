# src/processing/imu/main_imu.py

# IO imports
from src.io.utils import project_file
from src.io.rosbag import read_bag
from src.io.dataset import load_imu_data

# IMU imports
from src.processing.imu.fusion.vqf_filter import run_vqf
from src.processing.imu.visualize_imu import plot_raw_imu

# Utils imports
from src.utils.referential_change import (
    quaternion_to_euler,
    rebase_quaternions_to_initial,
)
from src.utils.visualize import plot_euler, plot_quat

# Other imports
import numpy as np


def estimate_quat_from_imu_data(timestamps, gyro, acc, mag, offline=False, params=None):
    """
    Estimate IMU orientation (rotation matrices, euler angles, quaternions).
    Inputs:
        timestamps: np.ndarray [n] in seconds
        gyro: np.ndarray [n, 3] gyroscope data in rad/s
        acc: np.ndarray [n, 3] accelerometer data in m/s^2
        mag: np.ndarray [n, 3] magnetometer data in uT
        offline: bool, whether to run in offline mode
        params: dict, parameters for the VQF filter
    Returns:
        quat: np.ndarray [n, 4] (x, y, z, w)
        roll, pitch, yaw: np.ndarray [n] in degrees
        timestamps: np.ndarray [n] in seconds
    """
    # Ensure inputs are contiguous arrays
    gyro = np.ascontiguousarray(gyro)
    acc = np.ascontiguousarray(acc)
    if mag is not None:
        mag = np.ascontiguousarray(mag)

    # Estimate average sampling period in seconds
    Ts = np.mean(timestamps[1:] - timestamps[:-1])

    # quat: np.ndarray [n, 4] (w, x, y, z)
    quat = run_vqf(Ts, gyro, acc, mag, offline=offline, params=params)

    return quat[:, [1, 2, 3, 0]]


def mag_transform(mag_data, mag_calib_file):
    calib = np.load(mag_calib_file)
    MagBias = calib["MagBias"]
    MagTransform = calib["MagTransform"]
    mag_data_calib = (
        mag_data - MagBias
    ) @ MagTransform.T  # Apply hard-iron and soft-iron correction

    return mag_data_calib


if __name__ == "__main__":
    data_source = "bag"  # "bag" or "csv"
    plot = True
    use_mag = False

    # Paths
    bag_dir_path = project_file("data", "raw", "ros_bag", "drone_indoor")
    bag_name = "rosbag2_2026_01_22-11_22_36"
    imu_csv_path = project_file(
        "data", "raw", "multimodal", "session_20251003_152453", "imu", "imu.csv"
    )

    if data_source == "bag":
        ####### LOAD DATA FROM ROSBAG #######
        # Bag path and output path
        bag_path = project_file(bag_dir_path, bag_name)

        # Read bag
        bag_data = read_bag(bag_path, layers=["imu_raw"])
        imu_data = bag_data["imu_raw"]
        timestamps = imu_data["t"]
        acc = imu_data["data"][:, 0:3]
        gyro = imu_data["data"][:, 3:6]
        mag = imu_data["data"][:, 7:9]

    elif data_source == "csv":
        ####### LOAD DATA FROM CSV DATASET #######
        acc, gyro, mag, timestamps = load_imu_data(imu_csv_path)

    ####### ESTIMATION QUATERNIONS FROM RAW IMU #######
    # Params
    params = None
    # params = dict(
    #     motionBiasEstEnabled=False,
    #     restBiasEstEnabled=False,
    #     magDistRejectionEnabled=False,
    # )

    if use_mag is None:
        mag = None

    # Estimate quaterion
    imu_quat = estimate_quat_from_imu_data(
        timestamps, gyro, acc, mag, offline=True, params=params
    )

    # Rebase quaternions to start at identity
    quat = rebase_quaternions_to_initial(imu_quat)

    # Quaternions to euler angles
    roll, pitch, yaw = quaternion_to_euler(
        imu_quat[:, 0], imu_quat[:, 1], imu_quat[:, 2], imu_quat[:, 3]
    )

    # Visualization
    if plot:
        if mag is None:
            mag = np.zeros_like(acc)
        plot_raw_imu(timestamps, acc, gyro, mag)
        plot_quat(quat, title="Rebased Quaternions")
        plot_euler(roll, pitch, yaw)
