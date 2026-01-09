# For debugging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from src.utils.io import project_file
from src.utils.visualize import plot_euler, plot_quat
from src.processing.imu.utils_imu import (
    load_imu_data,
    quats_to_euler_imu,
    euler_to_rotation_matrices,
    rebase_quaternions_to_initial_imu,
)
from src.processing.imu.fusion.vqf_filter import run_vqf
from src.processing.imu.visualize_imu import plot_raw_imu


def estimate_orientation_from_imu_file(
    imu_file, params=None, offline=False, plot=False, MAG_CALIB_FILE=None
):
    """
    Estimate IMU orientation (rotation matrices, euler angles, quaternions).
    Returns:
        quat: np.ndarray [n, 4] (w, x, y, z)
        roll, pitch, yaw: np.ndarray [n] in degrees
        timestamps: np.ndarray [n] in seconds
    """

    acc, gyro, mag, timestamps, Ts = load_imu_data(imu_file)
    if plot:
        time = timestamps - timestamps[0]
        plot_raw_imu(time, acc, gyro, mag)

        # # check initial gyro drift
        # idx_stationary = np.where(time <= time[0] + 20)[0][-1]
        # print(idx_stationary)
        # bias_gyro = gyro[0,:]
        # drift_gyro = gyro[idx_stationary,:]
        # gyro = gyro - bias_gyro
        # print(drift_gyro)
        # # temporary for debug, should delete!!!!
        # # gyro[:,2] -= 0.12
        # plot_raw_imu(time, acc, gyro, mag)

    # Check if mag data is available
    if np.all(mag == 0):
        mag = None
        print("No magnetometer data found, using 6D mode.")

    if MAG_CALIB_FILE:
        calib = np.load(MAG_CALIB_FILE)
        MagBias = calib["MagBias"]
        MagTransform = calib["MagTransform"]
        mag = (
            mag - MagBias
        ) @ MagTransform.T  # Apply hard-iron and soft-iron correction

    mag = None  # *********** here mag=none because mag not well calibrated yet..

    quat, roll, pitch, yaw, timestamps = estimate_orientation_from_imu_data(
        Ts, timestamps, gyro, acc, mag, offline=offline, params=params, plot=plot
    )

    return quat, roll, pitch, yaw, timestamps


def estimate_orientation_from_imu_data(
    Ts, timestamps, gyro, acc, mag, offline=False, params=None, plot=False
):
    """
    Estimate IMU orientation (rotation matrices, euler angles, quaternions).
    Inputs:
        Ts: float, sampling period in seconds
        timestamps: np.ndarray [n] in seconds
        gyro: np.ndarray [n, 3] gyroscope data in rad/s
        acc: np.ndarray [n, 3] accelerometer data in m/s^2
        mag: np.ndarray [n, 3] magnetometer data in uT
        offline: bool, whether to run in offline mode
        params: dict, parameters for the VQF filter
        plot: bool, whether to plot results
    Returns:
        quat: np.ndarray [n, 4] (w, x, y, z)
        roll, pitch, yaw: np.ndarray [n] in degrees
        timestamps: np.ndarray [n] in seconds
    """
    # --- Sensor fusion ---
    quat = run_vqf(Ts, gyro, acc, mag, offline=offline, params=params)
    if offline and plot:
        plot_quat(quat, title="Quaternions from VQF (offline mode)")
    elif not offline and plot:
        plot_quat(quat, title="Quaternions from VQF (online mode)")

    # --- Rebase quaternions to start at identity ---
    quat = rebase_quaternions_to_initial_imu(quat)
    if plot:
        plot_quat(quat, title="Rebased Quaternions")

    # --- Convert to Euler ---
    roll, pitch, yaw = quats_to_euler_imu(quat)
    if plot:
        plot_euler(roll, pitch, yaw)

    return quat, roll, pitch, yaw, timestamps


def visualize_raw_imu(imu_file, MAG_CALIB_FILE=None):
    acc, gyro, mag, timestamps, Ts = load_imu_data(imu_file)
    time = timestamps - timestamps[0]
    if MAG_CALIB_FILE is not None:
        calib = np.load(MAG_CALIB_FILE)
        MagBias = calib["MagBias"]
        MagTransform = calib["MagTransform"]
        mag = (
            mag - MagBias
        ) @ MagTransform.T  # Apply hard-iron and soft-iron correction
    plot_raw_imu(time, acc, gyro, mag)


if __name__ == "__main__":
    # Basic visualization
    # SIG_FILE = project_file("data", "raw","imu","imu_data_20251013_142045.csv")
    SIG_FILE = project_file(
        "data", "raw", "multimodal", "session_20251013_131149", "imu", "imu.csv"
    )
    MAG_CALIB_FILE = project_file("configs", "imu_mag_calibration_offline.npz")
    visualize_raw_imu(SIG_FILE, MAG_CALIB_FILE)
    plt.show()
    exit()

    # Simple test run
    # # Basic version
    # params = dict(
    #     motionBiasEstEnabled=False,
    #     restBiasEstEnabled=False,
    #     magDistRejectionEnabled=False,
    # )

    # Load raw data from IMU9250
    SIG_FILE = project_file(
        "data", "raw", "multimodal", "session_20251003_152453", "imu", "imu.csv"
    )
    quat, roll, pitch, yaw = estimate_orientation_from_imu_file(
        imu_file=SIG_FILE, params=None, offline=False, plot=True
    )

    # --- Rotation matrices ---
    R = euler_to_rotation_matrices(roll, pitch, yaw)
