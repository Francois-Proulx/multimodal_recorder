from pathlib import Path
import subprocess
import pandas as pd
import cv2
import csv

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
