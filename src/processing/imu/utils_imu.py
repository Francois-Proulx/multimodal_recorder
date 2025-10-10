# src/processing/imu/imu_utils.py
import numpy as np
import pandas as pd


from src.utils.referential_change import quaternion_to_euler, rotation_matrix_from_euler, rebase_quaternions_to_initial

def load_imu_data(filename:str):
    """Load IMU CSV file and return numpy arrays ready for fusion."""
    df = pd.read_csv(filename)
    timestamps = df['timestamp'].to_numpy()
    timestamps = timestamps / 1e9  # convert ns to seconds
    
    acc = np.ascontiguousarray(df[['acc_x','acc_y','acc_z']].to_numpy())
    gyro = np.ascontiguousarray(df[['gyro_x','gyro_y','gyro_z']].to_numpy())
    mag = np.ascontiguousarray(df[['mag_x','mag_y','mag_z']].to_numpy())
    
    # normalize time
    print(np.max(timestamps[1:] - timestamps[:-1]), np.min(timestamps[1:] - timestamps[:-1]), np.mean(timestamps[1:] - timestamps[:-1]))
    Ts = np.mean(timestamps[1:] - timestamps[:-1])
    return acc, gyro, mag, timestamps, Ts


def quats_to_euler_imu(quat):
    """Convert quaternion array to roll, pitch, yaw and unwrap."""
    q1, q2, q3, q4 = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    roll, pitch, yaw = quaternion_to_euler(q1,q2,q3,q4)
    return roll, pitch, yaw


def euler_to_rotation_matrices(roll, pitch, yaw):
    """Return rotation matrices from Euler angles."""
    return rotation_matrix_from_euler(roll, pitch, yaw)


def rebase_quaternions_to_initial_imu(quat):
    """Rebase quaternion sequence so that the first quaternion becomes identity."""
    return rebase_quaternions_to_initial(quat)


