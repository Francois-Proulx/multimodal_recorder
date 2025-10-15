from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.io import project_file
from src.utils.visualize import plot_quat, plot_euler
from src.processing.video.utils_video import reconstruct_video, merge_audio_video, process_apriltag_video, quats_to_euler_video, rebase_quaternions_to_initial_video



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
    video_path = reconstruct_video(video_frame_path, video_ts_path, output_dir=video_dir, output_filename="video.avi")
    
    # Step 2: merge audio
    mp4_path = merge_audio_video(video_path, video_ts_path, audio_path, audio_ts_path, output_dir=output_path, output_filename="video_merged.mp4")
    return video_path, mp4_path

    
def process_apriltag(session_path, tag_family, tag_size, save_new_video = False, plot=True):
    # Ensure session_path is a Path object
    session_path = Path(session_path)

    # Define paths based on session_path
    video_path = session_path / "video" / "video.avi"
    video_ts_path = session_path / "video" / "video_timestamps.csv"
    
    # Load camera intrinsics
    camera_intrinsics_path = project_file("configs", "camera_intrinsics.npz")
    camera_intrinsics = np.load(camera_intrinsics_path)
    camera_matrix = camera_intrinsics['camera_matrix']
    dist_coeffs = camera_intrinsics['dist_coeffs']

    # Replace 'raw' with 'processed' in the session path
    output_dir = Path(str(session_path).replace("raw", "processed"))
    output_filename = "video_apriltag.avi"
    
    # Process apriltag
    quats, timestamps = process_apriltag_video(video_path, video_ts_path, tag_family, tag_size, camera_matrix, dist_coeffs, save_new_video, output_dir, output_filename)
    
    # --- Rebase quaternions to start at identity ---
    quats = rebase_quaternions_to_initial_video(quats)
    if plot:
        plot_quat(quats, title="Rebased Quaternions")
    
    # --- Convert to Euler ---
    roll, pitch, yaw = quats_to_euler_video(quats)
    if plot:
        plot_euler(roll, pitch, yaw, title="Euler angles from apriltag")
    
    return quats, roll, pitch, yaw, timestamps


if __name__ == "__main__":
    # Example session folder
    session_path = project_file("data", "raw", "multimodal", "session_20251014_105701")
    
    # process_session(session_path)
    # process_apriltag(session_path, tag_family="tag25h9", tag_size=0.140, save_new_video=True)