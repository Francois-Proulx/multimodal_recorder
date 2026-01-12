import cv2
import numpy as np
import glob
from src.utils.io import project_file
import yaml


def save_to_yaml(file_path, camera_matrix, dist_coeffs, resolution):
    """
    Saves calibration data to a ROS-compatible YAML file.
    """
    # 1. Convert NumPy arrays to standard Python lists
    # .flatten().tolist() automatically converts elements to standard floats
    camera_matrix_flat = camera_matrix.flatten().tolist()
    dist_coeffs_flat = dist_coeffs.flatten().tolist()

    # 2. Construct Projection Matrix (P) manually
    # IMPORTANT: We use the values from the python list 'camera_matrix_flat'
    # instead of the numpy array to guarantee they are floats, not numpy objects.
    # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    # P = [fx, 0, cx, 0]
    #     [0, fy, cy, 0]
    #     [0,  0,  1, 0]

    fx = camera_matrix_flat[0]
    cx = camera_matrix_flat[2]
    fy = camera_matrix_flat[4]
    cy = camera_matrix_flat[5]

    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    # 3. Construct the dictionary
    data = {
        "image_width": int(resolution[0]),
        "image_height": int(resolution[1]),
        "camera_name": "default_cam",
        "camera_matrix": {"rows": 3, "cols": 3, "data": camera_matrix_flat},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": dist_coeffs_flat},
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {"rows": 3, "cols": 4, "data": P},
    }

    # 4. Save
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None)
    print(f"ROS Calibration saved to: {file_path}")


def calibrate_camera(
    chessboard_size=(8, 6),
    square_size=0.02275,
    calib_images_path=None,
    calib_save_path=None,
    ros_yaml_path=None,
):
    """
    Calibrate camera using chessboard images.

    Args:
        chessboard_size: tuple of inner corners (cols, rows)
        square_size: size of one square in meters
        calib_images_path: path to images for calibration
        calib_save_path: file to save camera_matrix and dist_coeffs
        ros_yaml_path: path to ROS yaml file for calibration parameters
    """
    # prepare object points
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    objp *= square_size

    objpoints = []
    imgpoints = []

    # Load images
    if calib_images_path is not None:
        images = glob.glob(f"{calib_images_path}/*.jpg")

        if not images:
            print(f"No images found in {calib_images_path}. Please check the path.")
            return
    else:
        print("No calibration images path provided.")
        return

    # images = glob.glob("configs/calib_images/*.jpg")  # video_20251010_180006_frames
    print(f"Found {len(images)} images for calibration.")
    img_shape = None

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue

        # Check resolution consistency
        if img_shape is None:
            img_shape = img.shape[:2][::-1]  # (width, height)
            print(f"Detected Resolution: {img_shape}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"[{idx + 1}/{len(images)}] Chessboard detected in {fname}")
        else:
            print(f"[{idx + 1}/{len(images)}] No chessboard found in {fname}")

    if len(objpoints) == 0:
        raise RuntimeError(
            "No chessboard corners were found in any image. Calibration failed."
        )

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Save to npz file for processing scripts (src/processing folder)
    if calib_save_path is not None:
        np.savez(calib_save_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"Calibration done. Saved to {calib_save_path}")
    else:
        print("Calibration done. No save path provided.")

    # Save to ROS yaml file
    if ros_yaml_path is not None:
        save_to_yaml(ros_yaml_path, camera_matrix, dist_coeffs, gray.shape[::-1])
    else:
        print("No ROS yaml path provided. Skipping ROS yaml save.")


if __name__ == "__main__":
    calib_images_path = project_file("configs", "calib_images_usbcam")
    calib_save_path = project_file("configs", "usb_cam_calib_params.npz")
    ros_yaml_path = project_file(
        "ros2", "src", "video", "config", "usb_cam_calib_params.yaml"
    )
    chessboard_size = (8, 6)  # inner corners
    square_size = 0.02275  # meters
    calibrate_camera(
        chessboard_size, square_size, calib_images_path, calib_save_path, ros_yaml_path
    )
