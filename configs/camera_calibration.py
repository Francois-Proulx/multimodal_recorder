import cv2
import numpy as np
import glob

def calibrate_camera(chessboard_size=(8,6), square_size=0.02275, save_path='configs/camera_intrinsics.npz'):
    """
    Calibrate camera using chessboard images.
    
    Args:
        chessboard_size: tuple of inner corners (cols, rows)
        square_size: size of one square in meters
        save_path: file to save camera_matrix and dist_coeffs
    """
    # prepare object points
    objp = np.zeros((chessboard_size[1]*chessboard_size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob('configs/calib_images/*.jpg')  # video_20251010_180006_frames
    print(f"Found {len(images)} images for calibration.")
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"[{idx+1}/{len(images)}] Chessboard detected in {fname}")
        else:
            print(f"[{idx+1}/{len(images)}] No chessboard found in {fname}")
    
    if len(objpoints) == 0:
        raise RuntimeError("No chessboard corners were found in any image. Calibration failed.")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    np.savez(save_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration done. Saved to {save_path}")
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    calibrate_camera()