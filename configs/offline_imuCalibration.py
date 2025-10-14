import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import project_file

def offline_mag_calibration(csv_path):
    """
    Compute hard-iron and soft-iron corrections from recorded magnetometer CSV.
    
    Args:
        csv_path (str): Path to CSV file with mag values (columns: mag_x, mag_y, mag_z)
        
    Returns:
        MagBias (3,): hard-iron offset
        MagTransform (3,3): soft-iron correction matrix
    """
    df = pd.read_csv(csv_path)
    mag = np.ascontiguousarray(df[['mag_x','mag_y','mag_z']].to_numpy())
    
    # -------------------
    # Hard-iron offset
    MagBias = mag.mean(axis=0)
    print(MagBias.shape)
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
    
    # If the transform contains NaNs (e.g. if mag data is constant), set to identity
    if np.any(np.isnan(MagTransform)):
        print("Warning: MagTransform contains NaNs, setting to identity")
        MagTransform = np.eye(3)
    # Return both bias and transform
    return MagBias, MagTransform


if __name__ == "__main__":
    # For offline calibration, provide path to CSV
    
    mag_csv_path = project_file("data", "raw","imu","imu_data_20251013_142045.csv")
    MagBias, MagTransform = offline_mag_calibration(mag_csv_path)
    print("Offline MagBias:", MagBias)
    print("Offline MagTransform:", MagTransform)
    
    # save to npz
    save_file = project_file("configs", "imu_mag_calibration_offline.npz")
    np.savez(save_file, MagBias=MagBias, MagTransform=MagTransform)
    print("Saved to configs/imu_mag_calibration_offline.npz")