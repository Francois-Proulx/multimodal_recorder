import numpy as np
import matplotlib.pyplot as plt

from src.utils.io import project_file
from src.utils.visualize import plot_euler
from src.utils.referential_change import (
    interpolate_quaternions,
    quaternion_to_euler,
    quat_error_angle_deg,
    rebase_quaternions_to_initial,
)
from src.processing.multimodal.visualize_multimodal import plot_sensor_drift

# IMU
from src.processing.imu.main_imu import estimate_orientation

# Video
from src.processing.video.main_video import process_apriltag


def trim_video_to_imu(t_video, t_imu, video_quats):
    """
    Trim audio FFT frames to fit within imu and video timestamps.

    Parameters
    ----------
    audio_fft_timestamps : np.ndarray
        Timestamps of the middle of FFT frames (in seconds)
    t_imu : np.ndarray
        IMU timestamps (in seconds)
    t_video : np.ndarray
        Video timestamps (in seconds)
    doas : np.ndarray
        DOAs (theta, phi) for each FFT frame
    doas_coord : np.ndarray
        DOA coordinates (x, y, z) for each FFT frame

    Returns
    -------
    trimmed_audio_frames, trimmed_doas, trimmed_doas_coord
    """
    hop_size = t_video[1] - t_video[0]
    imu_start, imu_end = t_imu[0], t_imu[-1]
    video_start, video_end = t_video[0], t_video[-1]

    # Trim start
    if video_start < imu_start:
        n_trim = min(
            int(np.ceil((imu_start - video_start) / hop_size)), len(video_start)
        )
        print(f"Trimming {n_trim} audio frames at start")
        if n_trim > 0:
            t_video = t_video[n_trim:]
            video_quats = video_quats[n_trim:]

    # Trim end
    if video_end > imu_end:
        n_trim = min(int(np.ceil((video_end - imu_end) / hop_size)), len(t_video))
        print(f"Trimming {n_trim} audio frames at end")
        if n_trim > 0:
            t_video = t_video[:-n_trim]
            video_quats = video_quats[:-n_trim]

    return t_video, video_quats


def fuse_imu_video(imu_file, video_session, tag_family, tag_size, MAG_CALIB_FILE=None):
    # --- 1. Estimate orientation from imu ---
    quats_imu, _, _, _, imu_timestamps = estimate_orientation(
        imu_file, params=None, offline=False, plot=True, MAG_CALIB_FILE=MAG_CALIB_FILE
    )

    # --- 2. Estimate orientation from video ---
    quats_video, _, _, _, video_timestamps = process_apriltag(
        video_session, tag_family, tag_size
    )

    # --- 3 Trim video so imu interpolation works ---
    video_timestamps, quats_video = trim_video_to_imu(
        video_timestamps, imu_timestamps, quats_video
    )
    t = video_timestamps - video_timestamps[0]

    # --- 6 Interpolate quaternions to video timestamps ---
    quat_imu_interp, R_imu = interpolate_quaternions(
        quats_imu, imu_timestamps, video_timestamps
    )
    quat_video_interp, R_video = interpolate_quaternions(
        quats_video, video_timestamps, video_timestamps
    )
    # plot_quat(quats_video, quat_interp, t1=video_timestamps, t2=audio_fft_timestamps,title="Original and Interpolated Quaternions")

    # rebase imu quats
    quat_imu_interp = rebase_quaternions_to_initial(quat_imu_interp)

    # if necessary, unbias..

    # --- 6.1 (VALIDATION) Convert to Euler ---
    roll_imu, pitch_imu, yaw_imu = quaternion_to_euler(
        quat_imu_interp[:, 0],
        quat_imu_interp[:, 1],
        quat_imu_interp[:, 2],
        quat_imu_interp[:, 3],
    )
    plot_euler(roll_imu, pitch_imu, yaw_imu, title="IMU orientation")

    roll_video, pitch_video, yaw_video = quaternion_to_euler(
        quat_video_interp[:, 0],
        quat_video_interp[:, 1],
        quat_video_interp[:, 2],
        quat_video_interp[:, 3],
    )
    plot_euler(roll_video, pitch_video, yaw_video, title="Apriltag orientation")

    plot_sensor_drift(
        t,
        sensors=[yaw_imu, yaw_video],
        sensors_ids=["imu yaw", "apriltag yaw"],
        title="IMU yaw drift vs apriltag",
    )

    angle_error_deg = quat_error_angle_deg(
        quat_tag=quat_video_interp, quat_imu=quat_imu_interp
    )
    plt.figure()
    plt.plot(t, angle_error_deg)
    plt.show()

    return


if __name__ == "__main__":
    # Paths to files
    IMU_FILE = project_file(
        "data", "raw", "multimodal", "session_20251017_170132", "imu", "imu.csv"
    )
    # MAG_CALIB_FILE = project_file("configs","imu_mag_calibration_offline.npz")
    VIDEO_SESSION = session_path = project_file(
        "data", "raw", "multimodal", "session_20251017_170132"
    )

    # Parameters
    tag_family = "tag25h9"
    tag_size = 0.140

    fuse_imu_video(IMU_FILE, VIDEO_SESSION, tag_family, tag_size, MAG_CALIB_FILE=None)
