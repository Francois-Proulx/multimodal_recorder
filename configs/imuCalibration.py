import os
import time
import smbus
import numpy as np
import pandas as pd
from imusensor.MPU9250 import MPU9250

def online_calibration(bus_id=1, address=0x68, save_path=None):
    """
    Perform online accelerometer and magnetometer calibration and save results to JSON.
    
    Returns:
        imu: calibrated MPU9250 object
        calib_file: path to saved JSON file
    """
    bus = smbus.SMBus(bus_id)
    imu = MPU9250.MPU9250(bus, address)
    imu.begin()

    print("Accel calibration starting...")
    imu.caliberateAccelerometer()
    print("Accel calibration finished")
    print("Accel bias:", imu.AccelBias)
    print("Accel scales:", imu.Accels)

    print("Mag calibration starting...")
    time.sleep(2)
    imu.caliberateMagPrecise()
    print("Mag calibration finished")
    print("MagBias:", imu.MagBias)
    print("MagTransform:", imu.Magtransform)
    print("Mag scales:", imu.Mags)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "calibMPU9250.json")
    imu.saveCalibDataToFile(save_path)

    return imu, save_path


def offline_mag_calibration(csv_path):
    """
    Compute hard-iron and soft-iron corrections from recorded magnetometer CSV.
    
    Args:
        csv_path (str): Path to CSV file with columns ['mx','my','mz']
        
    Returns:
        MagBias (3,): hard-iron offset
        MagTransform (3,3): soft-iron correction matrix
    """
    mag = pd.read_csv(csv_path).values  # Nx3 array
    # -------------------
    # Hard-iron offset
    MagBias = mag.mean(axis=0)
    mag_corr = mag - MagBias
    
    # -------------------
    # Soft-iron correction via ellipsoid fit
    # Algebraic ellipsoid fit
    def fit_ellipsoid(mag):
        x, y, z = mag[:,0], mag[:,1], mag[:,2]
        D = np.column_stack([x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones_like(x)])
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        v = Vt.T[:, -1]
        Q = np.array([[v[0], v[3], v[4]],
                      [v[3], v[1], v[5]],
                      [v[4], v[5], v[2]]])
        p = np.array([v[6], v[7], v[8]])
        r = v[9]
        center = -np.linalg.solve(Q, p)
        val = center.T @ Q @ center - r
        eigvals, eigvecs = np.linalg.eigh(Q / val)
        A = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return center, A

    center, MagTransform = fit_ellipsoid(mag)
    # Return both bias and transform
    return MagBias, MagTransform


if __name__ == "__main__":
    # Example usage
    imu, calib_file = online_calibration(bus_id=1, address=0x68, save_path="calibMPU9250.json")
    print(f"Calibration saved to {calib_file}")
    
    # For offline calibration, provide path to CSV
    # mag_csv_path = "path_to_mag_data.csv"
    # MagBias, MagTransform = offline_mag_calibration(mag_csv_path)
    # print("Offline MagBias:", MagBias)
    # print("Offline MagTransform:", MagTransform)