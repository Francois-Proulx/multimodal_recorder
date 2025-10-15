from pathlib import Path
import subprocess
import pandas as pd
import cv2
import apriltag
import csv
import numpy as np

from src.utils.referential_change import apply_rotation, quaternion_to_euler, rvec_to_quaternion, rebase_quaternions_to_initial

def reconstruct_video(frames_dir, video_ts_path, output_dir=None, output_filename=None, codec="MJPG"):
    """
    Reconstructs a video from frames and timestamps.

    Args:
        frames_dir (str or Path): Path to the folder containing frame images.
        video_ts_path (str or Path): Path to the CSV containing frame_index and timestamp.
        output_dir (str or Path, optional): Directory to save the reconstructed video. 
                                            Defaults to the folder containing `video_ts_path`.
        output_filename (str, optional): Filename for the output video. Defaults to 'video.avi'.
        codec (str, optional): FourCC codec for the output video. Default is MJPG.

    Returns:
        Path: Path to the saved video file.
    """
    frames_dir = Path(frames_dir)
    video_ts_path = Path(video_ts_path)
    output_dir = Path(output_dir) if output_dir is not None else video_ts_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or "video.avi"
    output_path = output_dir / output_filename

    # Read timestamps
    timestamps = []
    with video_ts_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(float(row["timestamp"]))

    if not timestamps:
        raise ValueError("No timestamps found in CSV.")


    # Determine video size from the first frame
    first_frame_path = frames_dir / "frame_000000.jpg"
    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    first_frame = cv2.imread(str(first_frame_path))
    height, width, channels = first_frame.shape

    # Compute fps based on timestamps (average)
    if len(timestamps) > 1:
        diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        avg_fps = 1.0 / (sum(diffs) / len(diffs))
    else:
        avg_fps = 30  # fallback

    
    # Write video
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, avg_fps, (width, height))

    for idx, ts in enumerate(timestamps):
        frame_path = frames_dir / f"frame_{idx:06d}.jpg"
        if not frame_path.exists():
            print(f"Warning: frame not found: {frame_path}, skipping")
            continue
        frame = cv2.imread(str(frame_path))
        out.write(frame[:, :, :3])  # ensure only RGB/BGR channels

    out.release()
    print(f"Video reconstructed at {output_path} (fps ~{avg_fps:.2f})")
    return output_path


def merge_audio_video(video_path, video_ts_path, audio_path, audio_ts_path, output_dir=None, output_filename=None):
    """
    Merge video (.avi) and audio (.wav), aligning using timestamps.

    Args:
        video_path (str or Path): Path to the input video (.avi) file.
        video_ts_path (str or Path): Path to the CSV containing video timestamps.
        audio_path (str or Path): Path to the input audio (.wav) file.
        audio_ts_path (str or Path): Path to the CSV containing audio timestamps.
        output_dir (str or Path, optional): Directory to save the merged video. Defaults to the folder containing `video_path`.
        output_filename (str, optional): Filename for the output merged video. Defaults to 'video_merged.mp4'.

    Returns:
        Path: Path to the saved merged video file.
    """
    video_path = Path(video_path)
    video_ts_path = Path(video_ts_path)
    audio_path = Path(audio_path)
    audio_ts_path = Path(audio_ts_path)
    output_dir = Path(output_dir) if output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or "video_merged.mp4"
    mp4_path = output_dir / output_filename

    # Load timestamps
    video_ts = pd.read_csv(video_ts_path)
    audio_ts = pd.read_csv(audio_ts_path)

    # Use first timestamp as reference
    video_start = video_ts['timestamp'].iloc[0]
    audio_start = audio_ts['timestamp'].iloc[0]

    offset = audio_start - video_start  # positive = audio starts later

    print(f"Video start: {video_start}, Audio start: {audio_start}, Offset: {offset:.3f}s")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]  # overwrite output

    if offset > 0:
        # Audio starts later: delay audio
        cmd += ["-i", str(video_path), "-itsoffset", f"{offset:.6f}", "-i", str(audio_path)]
    else:
        # Audio starts earlier: delay video
        cmd += ["-itsoffset", f"{-offset:.6f}", "-i", str(video_path), "-i", str(audio_path)]

    cmd += ["-c:v", "copy", "-c:a", "aac", "-strict", "experimental", str(mp4_path)]

    print("Running ffmpeg command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Merged video saved to {mp4_path}")
    return mp4_path


##### AprilTag functions #####
def load_apriltag_detector(family="tag36h11"):
    options = apriltag.DetectorOptions(families=family)
    detector = apriltag.Detector(options)
    return detector


def detect_apriltags_in_frame(detector, frame):
    """Detect AprilTags in a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    return tags


def estimate_apriltag_pose_in_frame(tag, tag_size, camera_matrix, dist_coeffs):
    """
    Estimate the pose of an AprilTag in the camera frame.

    Args:
        tag: Detected AprilTag object from apriltag detector.
        tag_size (float): Size of the tag's side in meters.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Camera distortion coefficients.
    Returns:
        rvec (np.ndarray): Rotation vector (Rodrigues) of the tag in camera frame.
        tvec (np.ndarray): Translation vector of the tag in camera frame.
    """
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    image_points = np.array(tag.corners, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        raise RuntimeError("Could not solve PnP for AprilTag.")
    return success, rvec, tvec


def draw_apritag_pose_axis_in_frame(frame, rvec, tvec, camera_matrix, dist_coeffs, tag_size):
    """
    Draws a coordinate axis on the detected AprilTag for visualization.
    """
    axis_length = tag_size * 0.5
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    return frame


def rvec_to_quat_video(rvec):
    """Convert rotation vector (Rodrigues) to quaternion (x, y, z, w)."""
    rvec = rvec.ravel()
    quat = rvec_to_quaternion(rvec)
    return quat # returns in (x, y, z, w) format


def rebase_quaternions_to_initial_video(quat):
    """Rebase quaternion sequence so that the first quaternion becomes identity."""
    return rebase_quaternions_to_initial(quat)


def quats_to_euler_video(quat):
    """Convert quaternion to roll, pitch, yaw and unwrap."""
    q1, q2, q3, q4 = quat[0], quat[1], quat[2], quat[3]
    roll, pitch, yaw = quaternion_to_euler(q1,q2,q3,q4)
    return roll, pitch, yaw


def process_apriltag_video(video_path, video_ts_path, tag_family, tag_size, camera_matrix, dist_coeffs, save_new_video = True, output_dir=None, output_filename=None, codec="MJPG"):
    """
    Process a video sequence of images to detect AprilTags and estimate their poses.

    Args:
        video_path (str or Path): Path to the .avi video.
        video_ts_path (str or Path): Path to the CSV containing frame_index and timestamp.
        tag_familly (str): AprilTag family to use (e.g., 'tag36h11').
        tag_size (float): Size of the tag's side in meters.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Camera distortion coefficients.
        output_dir (str or Path, optional): Directory to save the processed video. 
                                            Defaults to the folder containing `video_ts_path`.
        output_filename (str, optional): Filename for the output video. Defaults to 'video_apriltag.avi'.
        codec (str, optional): FourCC codec for the output video. Default is MJPG.

    Returns:
    """
    # Paths
    video_path = Path(video_path)
    video_ts_path = Path(video_ts_path)
    output_dir = Path(output_dir) if output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or "video_with_apriltag.avi"
    output_path = output_dir / output_filename
    
    # Load apriltag detector, video and timestamps
    detector = load_apriltag_detector(tag_family)
    cap = cv2.VideoCapture(str(video_path))
    video_ts = pd.read_csv(video_ts_path)
    timestamps = video_ts['timestamp'].to_list()
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames_in_video != len(timestamps):
        print(f"Warning: number of frames in video ({num_frames_in_video}) "
          f"does not match number of timestamps ({len(timestamps)})")
    
    
    # Initialize output vectors and video writer
    quats = np.zeros((len(timestamps), 4), dtype=np.float32)   # x, y, z, w
    writer = None

    # Process each frame
    for idx, ts in enumerate(timestamps):
        ok, frame = cap.read()
        if not ok:
            break
        
        tags = detect_apriltags_in_frame(detector, frame)

        if len(tags) > 0:
            # print(f"Frame {idx}: Detected {len(tags)} tags")
            tag = tags[0]  # assuming one fixed tag
            success, rvec, tvec = estimate_apriltag_pose_in_frame(tag, tag_size, camera_matrix, dist_coeffs)
            if success:
                # print(f"  Tag ID: {tag.tag_id}, tvec: {tvec.ravel()}, rvec: {rvec.ravel()}")
                quats[idx,:] = rvec_to_quat_video(rvec)
            else:
                quats[idx,:] = np.nan
            
            if save_new_video:
                frame = draw_apritag_pose_axis_in_frame(frame, rvec, tvec, camera_matrix, dist_coeffs, tag_size)
        
        else:
            # print(f"Frame {idx}: No tags detected")
            quats[idx,:] = np.nan

        if save_new_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (w,h))
            writer.write(frame)

    cap.release()
    if writer: writer.release()
    if save_new_video: print(f"Video saved at {output_path} (fps ~{fps:.2f})")
    return quats, timestamps