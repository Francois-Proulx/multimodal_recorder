from pathlib import Path
import subprocess
import pandas as pd
import cv2
import apriltag
import csv
import numpy as np

from src.utils.referential_change import (
    quaternion_to_euler,
    rvec_to_quaternion,
    rebase_quaternions_to_initial,
    rotation_matrix_to_quaternion,
    accumulate_quaternions,
)


def reconstruct_video(
    frames_dir, video_ts_path, output_dir=None, output_filename=None, codec="MJPG"
):
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
        print(np.max(diffs), np.min(diffs), np.mean(diffs), np.std(diffs))
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


def merge_audio_video(
    video_path,
    video_ts_path,
    audio_path,
    audio_ts_path,
    output_dir=None,
    output_filename=None,
):
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
    video_start = video_ts["timestamp"].iloc[0]
    audio_start = audio_ts["timestamp"].iloc[0]

    offset = audio_start - video_start  # positive = audio starts later

    print(
        f"Video start: {video_start}, Audio start: {audio_start}, Offset: {offset:.3f}s"
    )

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]  # overwrite output

    if offset > 0:
        # Audio starts later: delay audio
        cmd += [
            "-i",
            str(video_path),
            "-itsoffset",
            f"{offset:.6f}",
            "-i",
            str(audio_path),
        ]
    else:
        # Audio starts earlier: delay video
        cmd += [
            "-itsoffset",
            f"{-offset:.6f}",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
        ]

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
    object_points = np.array(
        [
            [-tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, tag_size / 2, 0],
        ],
        dtype=np.float32,
    )

    image_points = np.array(tag.corners, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        raise RuntimeError("Could not solve PnP for AprilTag.")
    return success, rvec, tvec


def draw_apritag_pose_axis_in_frame(
    frame, rvec, tvec, camera_matrix, dist_coeffs, tag_size
):
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
    return quat  # returns in (x, y, z, w) format


def rebase_quaternions_to_initial_video(quat):
    """Rebase quaternion sequence so that the first quaternion becomes identity."""
    return rebase_quaternions_to_initial(quat)


def accumulate_quaternions_video(delta_quats):
    quats = accumulate_quaternions(delta_quats)
    return quats


def quats_to_euler_video(quat):
    """Convert quaternion to roll, pitch, yaw and unwrap."""
    q1, q2, q3, q4 = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll, pitch, yaw = quaternion_to_euler(q1, q2, q3, q4)
    return roll, pitch, yaw


def rotation_matrix_to_quat_video(R_mat):
    quat = rotation_matrix_to_quaternion(R_mat)
    return quat  # returns in (x, y, z, w) format


def process_apriltag_video(
    video_path,
    video_ts_path,
    tag_family,
    tag_size,
    camera_matrix,
    dist_coeffs,
    save_new_video=True,
    output_dir=None,
    output_filename=None,
    codec="MJPG",
):
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
    timestamps = video_ts["timestamp"].to_numpy()

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames_in_video != len(timestamps):
        print(
            f"Warning: number of frames in video ({num_frames_in_video}) "
            f"does not match number of timestamps ({len(timestamps)})"
        )

    # Initialize output vectors and video writer
    quats = np.zeros((len(timestamps), 4), dtype=np.float32)  # x, y, z, w
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
            success, rvec, tvec = estimate_apriltag_pose_in_frame(
                tag, tag_size, camera_matrix, dist_coeffs
            )
            if success:
                # print(f"  Tag ID: {tag.tag_id}, tvec: {tvec.ravel()}, rvec: {rvec.ravel()}")
                quats[idx, :] = rvec_to_quat_video(rvec)
                # print(np.linalg.norm(tvec))
            else:
                quats[idx, :] = np.nan

            if save_new_video:
                frame = draw_apritag_pose_axis_in_frame(
                    frame, rvec, tvec, camera_matrix, dist_coeffs, tag_size
                )

        else:
            # print(f"Frame {idx}: No tags detected")
            quats[idx, :] = np.nan

        if save_new_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (w, h)
                )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    if save_new_video:
        print(f"Video saved at {output_path} (fps ~{fps:.2f})")
    return quats, timestamps


################# VISUAL ODOMETRY ################
def features_video_writer(frame, writer, output_path, fps, frame_size):
    """
    Handles writing frames to video. Initializes writer if needed, resizes frames, ensures correct type.

    Args:
        frame (np.ndarray): Frame to write (BGR).
        writer (cv2.VideoWriter or None): VideoWriter object or None.
        output_path (str or Path): Path to save video.
        fps (float): Frames per second.
        frame_size (tuple): (width, height) for output video.

    Returns:
        writer (cv2.VideoWriter): Initialized VideoWriter object.
    """
    # Ensure proper type and channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = frame.astype(np.uint8)

    # Resize if necessary
    h, w = frame.shape[:2]
    if (w, h) != frame_size:
        frame = cv2.resize(frame, frame_size)

    # Initialize writer if needed
    if writer is None:
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size
        )

    writer.write(frame)
    return writer


def process_orb_visual_odometry(
    video_path,
    video_ts_path,
    camera_matrix,
    dist_coeffs,
    save_new_video=True,
    output_dir=None,
    output_filename=None,
    max_features=2000,
    frame_step=5,
    matcher_norm=cv2.NORM_HAMMING,
):
    """
    Process a video using ORB-based visual odometry to estimate frame-to-frame rotation (delta quaternions).

    Args:
        video_path (str or Path): Path to the .avi video.
        video_ts_path (str or Path): Path to CSV containing frame_index and timestamp.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Camera distortion coefficients.
        save_new_video (bool): Whether to save annotated video.
        output_dir (str or Path, optional): Directory to save video.
        output_filename (str, optional): Output video filename.
        max_features (int): Max ORB features to detect per frame.
        matcher_norm (int): Norm type for descriptor matching (default Hamming).

    Returns:
        delta_quats (np.ndarray): Nx4 array of relative rotations (x, y, z, w) between consecutive frames.
        timestamps (np.ndarray): Frame timestamps.
    """
    video_path = Path(video_path)
    video_ts_path = Path(video_ts_path)
    output_dir = Path(output_dir) if output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or "video_with_vo.avi"
    output_path = output_dir / output_filename

    # Load timestamps
    video_ts = pd.read_csv(video_ts_path)
    timestamps = video_ts["timestamp"].to_numpy()
    num_frames = len(timestamps)

    # Initialize ORB and BFMatcher
    orb = cv2.ORB_create(nfeatures=max_features)
    bf = cv2.BFMatcher(matcher_norm, crossCheck=False)  # crossCheck=True for bf.match()

    # Initialize video capture and writer
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None

    # Storage for delta quaternions
    delta_quats = np.full((num_frames, 4), np.nan, dtype=np.float32)
    mask_count = np.zeros(num_frames)
    prev_kp, prev_des, prev_gray = None, None, None

    # Process each X frame
    for idx, ts in enumerate(timestamps):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame_size = (w, h)

        # Process only each frame_step frames
        if idx % frame_step == 0:
            # Convert to gray scale and detect features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            if des is None or len(kp) == 0:
                print(f"No features detected at frame {idx}")
                # Draw empty frame if needed
                if save_new_video:
                    writer = features_video_writer(
                        frame.copy(), writer, output_path, fps, frame_size
                    )

                prev_kp, prev_des, prev_gray = (
                    None,
                    None,
                    None,
                )  # Reset previous if invalid
                continue

            # Process motion only if previous frame is valid
            if prev_des is not None and prev_kp is not None:
                # matches = bf.match(prev_des, des)
                # matches = sorted(matches, key=lambda x: x.distance)
                # good_matches = [m for m in matches if m.distance < 30][:100]  # distance filtering
                matches = bf.knnMatch(prev_des, des, k=2)
                good_matches = []
                for m, n in matches:
                    if (
                        m.distance < 0.75 * n.distance and m.distance < 30
                    ):  # typical ratio
                        good_matches.append(m)

                # Need minimum 6 maches to estimate Essential matrix (5 dof, 3rot, 2tran)
                if len(good_matches) >= 6:
                    # Compute the pixel coordinates of the matched features
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

                    # Estimate essential matrix (assume normalized coordinates)
                    E, mask = cv2.findEssentialMat(
                        pts1,
                        pts2,
                        camera_matrix,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=1.0,
                    )
                    mask_count[idx] = mask.sum()

                    if E is not None and E.shape == (3, 3):
                        _, R_mat, _, mask_pose = cv2.recoverPose(
                            E, pts1, pts2, camera_matrix
                        )
                        delta_quats[idx] = rotation_matrix_to_quat_video(R_mat)
                    else:
                        delta_quats[idx] = np.nan

                    # Prepare comparison frame for visualization
                    if mask is not None:
                        inlier_matches = [
                            m for i, m in enumerate(good_matches) if mask[i][0]
                        ]
                    else:
                        inlier_matches = good_matches

                    frame_kp = cv2.drawMatches(
                        prev_gray,
                        prev_kp,
                        gray,
                        kp,
                        inlier_matches,
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )

                else:  # if previous frame valid but not enough maches, draw empty frame
                    delta_quats[idx] = np.nan
                    frame_kp = frame.copy()
            else:  # if previous frame not valid, then no match possible, draw empty frame
                delta_quats[idx] = np.nan
                frame_kp = frame.copy()

            # Write frame
            writer = features_video_writer(
                frame_kp, writer, output_path, fps, frame_size
            )

            # Assign current features to last features
            prev_kp, prev_des, prev_gray = kp, des, gray.copy()

        else:  # draw empty frames, but do not reasign current features to last features
            delta_quats[idx] = np.nan

            # Write frame
            writer = features_video_writer(
                frame.copy(), writer, output_path, fps, frame_size
            )

    cap.release()
    if writer:
        writer.release()
    if save_new_video:
        print(f"VO video saved at {output_path} (fps ~{fps:.2f})")

    # plt.figure()
    # plt.plot(mask_count)
    # plt.show()
    return delta_quats, timestamps


def filter_delta_quats(delta_quats, frame_step=10, max_angle_deg=10):
    """
    Filters quaternion spikes and non-physical jumps.
    Works even if delta_quats contain NaNs.

    Args:
        delta_quats (np.ndarray): Nx4 array of [x,y,z,w].
        frame_step (int): Number of frames skipped between processed frames.
        max_angle_deg (float): Maximum allowed angular difference between consecutive quats.

    Returns:
        np.ndarray: Smoothed quaternion array.
    """
    quats = delta_quats.copy()
    n = len(quats)

    # Compute instantaneous rotation angles
    angles = 2 * np.arccos(np.clip(quats[:, 3], -1, 1))

    # Identify large, non-physical jumps
    jump_mask = np.abs(np.gradient(angles)) > np.deg2rad(max_angle_deg)

    # Replace jumps and NaNs with previous value
    for i in range(1, n):
        if np.any(np.isnan(quats[i])) or jump_mask[i]:
            quats[i] = quats[i - 1]

    # Optionally apply a median filter (over processed frames)
    from scipy.ndimage import median_filter

    quats[:, :3] = median_filter(quats[:, :3], size=(frame_step // 2 + 1, 1))

    return quats
