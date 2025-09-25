# For debugging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from vqf import VQF, BasicVQF, PyVQF, offlineVQF
import pandas as pd

# Custom imports
from src.processing.imu.imu_utils import load_imu_data, run_vqf, quats_to_euler, rotation_matrices
from src.utils.io import project_file
from src.processing.imu.visualize import plot_euler, plot_mic_trajectory, plot_quat


# Load raw data from IMU9250
SIG_FILE = project_file("data", "raw", "imu", "imu_data_20241023-170624.csv")
Ts, acc, gyro, mag, _ = load_imu_data(SIG_FILE)

# --- Sensor fusion ---

# # Basic version
# params = dict(
#     motionBiasEstEnabled=False,
#     restBiasEstEnabled=False,
#     magDistRejectionEnabled=False,
# )

# # Online (real-time, no batch bias estimation)
# quat = run_vqf(Ts, gyro, acc, mag, offline=False)

# Offline (batch mode, estimates bias over the recording)
quat = run_vqf(Ts, gyro, acc, mag, offline=True)
plot_quat(quat, title="Quaternions from VQF (offline mode)")

# --- Convert to Euler ---
roll, pitch, yaw = quats_to_euler(quat)
plot_euler(roll, pitch, yaw)

# --- Rotation matrices ---
R = rotation_matrices(roll, pitch, yaw)

# --- Mic example ---
pB = np.array([[1],[0],[0]])
pA = np.einsum('nij,jk->nik', R, pB)
plot_mic_trajectory(pA)
