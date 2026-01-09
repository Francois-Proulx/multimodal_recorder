from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.io import project_file
from src.utils.visualize import plot_quat, plot_euler
from src.processing.video.utils_video import (
    reconstruct_video,
    merge_audio_video,
    process_apriltag_video,
    quats_to_euler_video,
    rebase_quaternions_to_initial_video,
    process_orb_visual_odometry,
    accumulate_quaternions_video,
)


def process_session(session_path):
    # Ensure session_path is a Path object
    session_path = Path(session_path)

    # Define paths based on session_path
    video_dir = session_path / "video"
    audio_dir = session_path / "audio"

    video_frame_path = video_dir / "video_frames"
    video_ts_path = video_dir / "video_timestamps.csv"
    audio_path = audio_dir / "audio.wav"
    audio_ts_path = audio_dir / "audio_timestamps.csv"

    # Replace 'raw' with 'processed' in the session path
    output_path = Path(str(session_path).replace("raw", "processed"))

    # Step 1: reconstruct video
    video_path = reconstruct_video(
        video_frame_path,
        video_ts_path,
        output_dir=video_dir,
        output_filename="video.avi",
    )

    # Step 2: merge audio
    mp4_path = merge_audio_video(
        video_path,
        video_ts_path,
        audio_path,
        audio_ts_path,
        output_dir=output_path,
        output_filename="video_merged.mp4",
    )
    return video_path, mp4_path


def process_apriltag(
    session_path, tag_family, tag_size, save_new_video=False, plot=True
):
    # Ensure session_path is a Path object
    session_path = Path(session_path)

    # Define paths based on session_path
    video_path = session_path / "video" / "video.avi"
    video_ts_path = session_path / "video" / "video_timestamps.csv"

    # Load camera intrinsics
    camera_intrinsics_path = project_file("configs", "camera_intrinsics.npz")
    camera_intrinsics = np.load(camera_intrinsics_path)
    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coeffs = camera_intrinsics["dist_coeffs"]

    # Replace 'raw' with 'processed' in the session path
    output_dir = Path(str(session_path).replace("raw", "processed"))
    output_filename = "video_apriltag.avi"

    # Process apriltag
    quats, timestamps = process_apriltag_video(
        video_path,
        video_ts_path,
        tag_family,
        tag_size,
        camera_matrix,
        dist_coeffs,
        save_new_video,
        output_dir,
        output_filename,
    )

    # --- Rebase quaternions to start at identity ---
    quats = rebase_quaternions_to_initial_video(quats)
    if plot:
        plot_quat(quats, title="Rebased Quaternions")

    # --- Convert to Euler ---
    roll, pitch, yaw = quats_to_euler_video(quats)
    if plot:
        plot_euler(roll, pitch, yaw, title="Euler angles from apriltag")

    return quats, roll, pitch, yaw, timestamps


def process_visual_audometry(
    session_path, max_features=2000, save_new_video=True, plot=True
):
    # Ensure session_path is a Path object
    session_path = Path(session_path)

    # Define paths based on session_path
    video_path = session_path / "video" / "video.avi"
    video_ts_path = session_path / "video" / "video_timestamps.csv"

    # Load camera intrinsics
    camera_intrinsics_path = project_file("configs", "camera_intrinsics.npz")
    camera_intrinsics = np.load(camera_intrinsics_path)
    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coeffs = camera_intrinsics["dist_coeffs"]

    # Replace 'raw' with 'processed' in the session path
    output_dir = Path(str(session_path).replace("raw", "processed"))
    output_filename = "video_orb_vo.avi"

    # Compute delta quaternions with orb visual odometry
    frame_step = 10
    delta_quats, timestamps = process_orb_visual_odometry(
        video_path,
        video_ts_path,
        camera_matrix,
        dist_coeffs,
        save_new_video,
        output_dir,
        output_filename,
        max_features,
        frame_step,
    )

    # Set NaNs delta_quats to the identity
    delta_quats[np.isnan(delta_quats).any(axis=1)] = [0, 0, 0, 1]

    # Remove the big variations > 90deg
    variation_threshold = 90  # deg
    delta_quats_angle = 2 * np.arccos(np.clip(delta_quats[:, 3], -1, 1))

    delta_quats_2 = delta_quats.copy()
    delta_quats_2[delta_quats_angle > np.deg2rad(variation_threshold), :] = [0, 0, 0, 1]

    delta_quats_modifs = delta_quats_2.copy()
    # If quats 1 and 2 > 0, set to average between neighboors
    for i in range(1, len(delta_quats_2) - frame_step):
        if np.abs(delta_quats_2[i, 0]) > 0 or np.abs(delta_quats_2[i, 1]) > 0:
            delta_quats_modifs[i, :] = [
                0,
                0,
                delta_quats[i, 2],
                np.sqrt(1 - delta_quats[i, 2] ** 2),
            ]

    delta_quats_modifs_interp = delta_quats_modifs.copy()
    for i in range(1, len(delta_quats_2) - frame_step):
        if i % frame_step == 0 and np.abs(delta_quats_modifs[i, 2]) == 0:
            new_q3 = (
                delta_quats_modifs[i - frame_step, 2]
                + delta_quats_modifs[i + frame_step, 2]
            ) / 2
            delta_quats_modifs_interp[i, :] = [0, 0, new_q3, np.sqrt(1 - new_q3**2)]

    # # plot angles before and after removing
    # plt.figure()
    # plt.plot(delta_quats_angle, label="before")
    # plt.plot(2 * np.arccos(np.clip(delta_quats[:,3], -1, 1)), label="after")
    # plt.legend(fontsize=8)

    # # Discard variations when not preced or follows by another variation
    # for i in range(1, len(delta_quats)):
    #     if i % frame_step == 0 and i > 1 and i < len(delta_quats):
    #         current_var =

    # Get absolute quaternions from delta quaternions
    quats = accumulate_quaternions_video(delta_quats_modifs_interp)

    if plot:
        plot_quat(delta_quats, title="Delta quaternions from orb vo")
        plot_quat(delta_quats_modifs, title="Delta quaternions modifs from orb vo")
        plot_quat(
            delta_quats_modifs_interp,
            title="Delta quaternions modifs interp from orb vo",
        )
        plot_quat(quats, title="Quaternions from orb vo")

    # --- Convert to Euler ---
    print(quats.shape)
    roll, pitch, yaw = quats_to_euler_video(quats)
    if plot:
        plot_euler(roll, pitch, yaw, title="Euler angles from orb odometry")

    return quats, roll, pitch, yaw, timestamps


if __name__ == "__main__":
    # # Manual video reconstruction
    # video_frame_path = project_file("data", "raw", "multimodal", "session_test_12", "video","video_frames")
    # video_ts_path = project_file("data", "raw", "multimodal", "session_test_12", "video","video_timestamps.csv")
    # video_path = reconstruct_video(video_frame_path, video_ts_path, output_dir=None, output_filename="video.avi")

    # # Process session (video reconstruction + merge audio video)
    # session_path = project_file("data", "raw", "multimodal", "session_test_13")
    # process_session(session_path)

    # # Process vision algos
    # quats_tag, roll_tag, pitch_tag, yaw_tag, timestamps_tag = process_apriltag(session_path, tag_family="tag25h9", tag_size=0.140, save_new_video=False)
    # quats_org, roll_org, pitch_org, yaw_org, timestamps_org = process_visual_audometry(session_path, max_features=2000, save_new_video=False)

    # # Compare algos
    # plt.figure()
    # plt.plot(timestamps_tag, yaw_tag, label="apriltag")
    # plt.plot(timestamps_org, yaw_org, label="org")
    # plt.legend

    # plot_quat(quats_tag, quats_org, timestamps_tag, timestamps_org)
    plt.show()
