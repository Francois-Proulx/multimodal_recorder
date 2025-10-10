# For debugging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom imports
from src.utils.io import project_file
from src.processing.imu.utils_imu import load_imu_data, quats_to_euler_imu, euler_to_rotation_matrices, rebase_quaternions_to_initial_imu
from src.processing.imu.fusion.vqf_filter import run_vqf
from src.processing.imu.visualize_imu import plot_raw, plot_euler, plot_mic_trajectory, plot_quat

def estimate_orientation(imu_file, params = None, offline = False, plot=False):
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
        plot_raw(time, acc, gyro, mag)
        
    # Check if mag data is available
    if np.all(mag == 0):
        mag = None
        print("No magnetometer data found, using 6D mode.")
    
    
    # --- Sensor fusion ---
    if offline: # Offline (batch mode, estimates bias over the recording)
        quat = run_vqf(Ts, gyro, acc, mag, offline=True, params=params)
        if plot:
            plot_quat(quat, title="Quaternions from VQF (offline mode)")
            
    else:  # Online (real-time, no batch bias estimation)
        quat = run_vqf(Ts, gyro, acc, mag, offline=False)
        if plot:
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



if __name__ == "__main__":
    # Simple test run
    # # Basic version
    # params = dict(
    #     motionBiasEstEnabled=False,
    #     restBiasEstEnabled=False,
    #     magDistRejectionEnabled=False,
    # )

    # Load raw data from IMU9250
    SIG_FILE = project_file("data", "raw","multimodal","session_20251003_152453", "imu", "imu.csv")
    quat, roll, pitch, yaw = estimate_orientation(imu_file=SIG_FILE, params = None, offline = False, plot=True)
    
    # --- Rotation matrices ---
    R = euler_to_rotation_matrices(roll, pitch, yaw)
