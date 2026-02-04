# src/io/dataset.py

import pandas as pd


def load_imu_data(filename: str):
    """Load IMU CSV file and return numpy arrays ready for fusion."""
    df = pd.read_csv(filename)

    # Timestamps
    timestamps = df["timestamp"].to_numpy()

    # Acc, gyro and mag data
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    mag = df[["mag_x", "mag_y", "mag_z"]].to_numpy()

    return acc, gyro, mag, timestamps
