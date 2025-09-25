# src/processing/imu/imu_utils.py
import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
from src.utils.referential_change import quaternion_to_euler_angle_vectorized, rotation_matrix_from_euler

def load_imu_data(filename:str):
    """Load IMU CSV file and return numpy arrays ready for fusion."""
    df = pd.read_csv(filename)
    temps = df['timestamp'].to_numpy()
    acc = np.ascontiguousarray(df[['acc_x','acc_y','acc_z']].to_numpy())
    gyro = np.ascontiguousarray(df[['gyro_x','gyro_y','gyro_z']].to_numpy())
    mag = np.ascontiguousarray(df[['mag_x','mag_y','mag_z']].to_numpy())
    
    # normalize time
    temps -= temps[0]
    Ts = np.mean(temps[1:] - temps[:-1])
    return Ts, acc, gyro, mag, temps


def run_vqf(Ts, gyro, acc, mag=None, offline=False, params=None):
    """Run VQF sensor fusion and return quaternion array."""
    if offline:
        out = offlineVQF(gyro, acc, mag, Ts, params)
        quat = out['quat9D'] if mag is not None else out['quat6D']
    else:
        vqf = VQF(Ts, **(params or {}))
        if mag is not None:
            out = vqf.updateBatch(gyro, acc, mag)
            quat = out['quat9D']
        else:
            out = vqf.updateBatch(gyro, acc)
            quat = out['quat6D']
    return quat

def quats_to_euler(quat):
    """Convert quaternion array to roll, pitch, yaw and unwrap."""
    q1, q2, q3, q4 = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    roll, pitch, yaw = quaternion_to_euler_angle_vectorized(q1,q2,q3,q4)
    # unwrap and calibrate to start at 0
    roll_cal = np.unwrap(np.radians(roll)) - np.radians(roll[0])
    pitch_cal = np.unwrap(np.radians(pitch)) - np.radians(pitch[0])
    yaw_cal = np.unwrap(np.radians(yaw)) - np.radians(yaw[0])
    return np.degrees(roll_cal), np.degrees(pitch_cal), np.degrees(yaw_cal)

def rotation_matrices(roll, pitch, yaw):
    """Return rotation matrices from Euler angles."""
    return rotation_matrix_from_euler(roll, pitch, yaw)
