import os
import cv2
import csv
from src.utils.io import project_file

def reconstruct_video(frames_dir, csv_path, output_dir=None, codec="MJPG"):
    """
    Reconstructs a video from frames and timestamps.
    
    Args:
        frames_dir (str): Path to the folder containing frame images.
        csv_path (str): Path to the CSV containing frame_index and timestamp.
        output_dir (str, optional): Directory to save the reconstructed video.
                                    Defaults to 'data/processed/video' in the project.
        codec (str, optional): FourCC codec for the output video. Default is MJPG.
    
    Returns:
        str: Path to the saved video file.
    """
    if output_dir is None:
        output_dir = project_file("data", "processed", "video")
    os.makedirs(output_dir, exist_ok=True)

    # Read timestamps
    timestamps = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(float(row["timestamp"]))

    if len(timestamps) == 0:
        raise ValueError("No timestamps found in CSV.")

    # Determine video size from the first frame
    first_frame_path = os.path.join(frames_dir, f"frame_000000.jpg")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame not found: {first_frame_path}")
    first_frame = cv2.imread(first_frame_path)
    height, width, channels = first_frame.shape

    # Compute fps based on timestamps (average)
    if len(timestamps) > 1:
        diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        avg_fps = 1.0 / (sum(diffs) / len(diffs))
    else:
        avg_fps = 30  # fallback

    # Video filename
    prefix = os.path.basename(csv_path).replace("_timestamps.csv", "")
    output_path = os.path.join(output_dir, f"{prefix}.avi")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, avg_fps, (width, height))

    for idx, ts in enumerate(timestamps):
        frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            print(f"Warning: frame not found: {frame_path}, skipping")
            continue
        frame = cv2.imread(frame_path)
        out.write(frame[:, :, :3])  # ensure only RGB/BGR channels

    out.release()
    print(f"Video reconstructed at {output_path} (fps ~{avg_fps:.2f})")
    return output_path


if __name__ == "__main__":
    # Example usage:
    frames_dir = project_file("data", "raw", "video", "video_20251003_144210_frames")
    csv_path = project_file("data", "raw", "video", "video_20251003_144210_timestamps.csv")
    reconstruct_video(frames_dir, csv_path)
