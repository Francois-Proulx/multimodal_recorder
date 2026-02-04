# IMPORTS MESSAGES
from sensor_msgs.msg import CompressedImage, Image
from droneaudition_msgs.msg import AudioRaw

# OTHER ROS IMPORTS
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

# OTHER IMPORTS
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
import subprocess
from scipy.io import wavfile
import json

# CUSTOM IMPORT
from src.io.utils import project_file


def export_rosbag_vid_to_avi(
    bag_path,
    output_dir,
    video_topic,
    output_file_name="extracted_video.avi",
    TARGET_FPS=30,
    force=False,
):
    """
    Args:
        bag_path (str): Path to bag.
        ...
    """
    video_out_path = os.path.join(output_dir, output_file_name)
    timestamps_output_name = f"{os.path.splitext(output_file_name)[0]}_timestamps.csv"
    timestamps_path = os.path.join(output_dir, timestamps_output_name)

    # 0. Smart check: skip if file exists and not force
    if os.path.exists(video_out_path) and os.path.exists(timestamps_path) and not force:
        print(f"Skipping export: {video_out_path} and {timestamps_path} already exist.")
        # Get start time
        timestamps = np.loadtxt(timestamps_path, delimiter=",")
        if timestamps.size == 0:
            video_start_ns = None
        else:
            video_start_ns = int(timestamps[0] * 1e9)
        return video_out_path, video_start_ns, timestamps_path

    # 1. Setup Bag Reader
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    filter = StorageFilter(topics=[video_topic])
    reader.set_filter(filter)

    # 2. Setup Lists to hold data
    video_writer = None
    video_start_ns = None

    bridge = CvBridge()

    frames_written = 0
    last_frame = None

    timestamps = []

    print(f"Exporting video topic: {video_topic}")

    # 3. Iterate through messages
    while reader.has_next():
        (_, data, t) = reader.read_next()

        # 3.1 Save start time
        if video_start_ns is None:
            video_start_ns = t
            print(f"Video syncing to start time: {t}")

        # 3.2 Deserialize message, first compressed image, then raw image
        frame = None
        try:
            try:
                # Try Compressed image first
                msg = deserialize_message(data, CompressedImage)
                if hasattr(msg, "format") and msg.format:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    raise ValueError("Not compressed")

            except Exception as e:
                print(e)
                # Fallback to Raw Image
                msg = deserialize_message(data, Image)
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        except Exception as e:
            # If both fail, skip this message
            print(f"Failed to decode frame: {e}")
            continue

        if frame is None:
            continue

        # 3.3 Initialize writer on the very first frame
        if video_writer is None:
            height, width, _ = frame.shape

            video_writer = cv2.VideoWriter(
                video_out_path,
                cv2.VideoWriter_fourcc(*"DIVX"),
                TARGET_FPS,
                (width, height),
            )
            print(
                f"Video writer initialized: {width}x{height}. Syncing to start time: {t}"
            )

        # 3.4 Frame timing control
        elapsed_time_s = (t - video_start_ns) / 1e9
        expected_frame_count = int(elapsed_time_s * TARGET_FPS)

        # 3.5 Fill gaps if frame drops
        while frames_written < expected_frame_count:
            if last_frame is not None:
                video_writer.write(last_frame)
                timestamps.append(np.nan)  # Indicate dropped frame with NaN timestamp
            else:
                video_writer.write(frame)  # Only for very first frame
                timestamps.append(t / 1e9)  # Actual timestamp in seconds
            frames_written += 1

        # 3.6. Write current frame
        video_writer.write(frame)
        timestamps.append(t / 1e9)  # Actual timestamp in seconds
        frames_written += 1
        last_frame = frame

    # 4. Save timestamps as csv
    np.savetxt(timestamps_path, np.array(timestamps), delimiter=",")
    print(f"Saved video timestamps to: {timestamps_path}")

    # 5. Cleanup Video and return video path, start time and timestamps path
    if video_writer is not None:
        video_writer.release()
        print(f"Saved to: {video_out_path}")
        return video_out_path, video_start_ns, timestamps_path
    else:
        video_out_path = None
        print(f"No video data found for topic {video_topic}")
        return None, None, None


def export_bag_audio(
    bag_path,
    output_dir,
    audio_topic,
    output_file_name="extracted_audio.wav",
    sync_reference_ns=None,
    SAMPWIDTH=4,
    force=False,
):
    """
    Args:
        bag_path (_type_): _description_
        output_dir (_type_): _description_
        topic_name (_type_): _description_
        sync_reference_ns (_type_, optional): _description_. Defaults to None.
        SAMPWIDTH (int, optional): _description_. Defaults to 4.

    Returns:
        ..
    """
    output_wav = os.path.join(output_dir, output_file_name)
    timestamps_output_name = f"{os.path.splitext(output_file_name)[0]}_metadata.json"
    timestamps_path = os.path.join(output_dir, timestamps_output_name)

    # 0. Smart check: skip if file exists and not force
    if os.path.exists(output_wav) and os.path.exists(timestamps_path) and not force:
        print(f"Skipping export: {output_wav} and {timestamps_path} already exist.")
        try:
            with open(timestamps_path, "r") as f:
                meta = json.load(f)

            target_start_time_ns = meta["start_ns"]

            return output_wav, target_start_time_ns, timestamps_path

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Metadata corrupt or missing keys ({e}). Re-exporting...")

    # 1. Setup Bag Reader
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    filter = StorageFilter(topics=[audio_topic])
    reader.set_filter(filter)

    # 2. Setup Lists to hold data
    audio_samples = []
    SAMPLERATE = None
    CHANNELS = None
    audio_start_ns = None

    # 3. Iterate through messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if audio_start_ns is None:
            audio_start_ns = t
            print(f"Audio syncing to start time: {t}")

        msg = deserialize_message(data, AudioRaw)

        if SAMPLERATE is None:
            SAMPLERATE = msg.fs
            CHANNELS = msg.channels
            print(f"Audio Sample Rate: {msg.fs}, Channels: {msg.channels}")

        # Reshape to (Time, Channels)
        try:
            audio_block = np.array(msg.data, dtype=np.float32)
            audio_block = audio_block.reshape((-1, CHANNELS), order="F")
            audio_samples.append(audio_block)
        except ValueError:
            print("Error reshaping audio block, skipping this block.")

    # 4. Sync and Save Audio to WAV
    if audio_samples and SAMPLERATE is not None:
        full_audio = np.vstack(audio_samples)

        target_start_time_ns = (
            sync_reference_ns if sync_reference_ns is not None else audio_start_ns
        )

        # Calculate Offset (Positive = Audio is Late, Negative = Audio is Early)
        offset_seconds = (audio_start_ns - target_start_time_ns) / 1e9
        offset_samples = int(abs(offset_seconds) * SAMPLERATE)

        print("--- Sync Analysis ---")
        print(f"Video Start: {target_start_time_ns}")
        print(f"Audio Start: {audio_start_ns}")
        print(f"Offset:      {offset_seconds:.4f} seconds")

        if offset_seconds > 0:
            # Audio is late, ad silence at start
            print(
                f"Audio is late by {offset_seconds:.4f} seconds, adding silence at start."
            )
            silence = np.zeros((offset_samples, CHANNELS), dtype=np.float32)
            full_audio = np.vstack((silence, full_audio))
        elif offset_seconds < 0:
            # Audio is early, trim start
            print(f"Audio is early by {offset_seconds:.4f} seconds, trimming start.")
            full_audio = full_audio[offset_samples:, :]

        # Scale data (Float32 -> Int32)
        if SAMPWIDTH == 4:
            print("Converting Float32 -> Int32...")
            # Scale [-1, 1] to [-2147483648, 2147483647]
            full_audio = np.clip(full_audio, -1.0, 1.0) * 2147483647
            full_audio = full_audio.astype(np.int32)
        elif SAMPWIDTH == 2:
            print("Converting Float32 -> Int16...")
            full_audio = np.clip(full_audio, -1.0, 1.0) * 32767
            full_audio = full_audio.astype(np.int16)

        # Save wav
        wavfile.write(output_wav, SAMPLERATE, full_audio)
        print(f"Saved audio to: {output_wav}")
    else:
        print("No audio found!")
        output_wav = None

    # 5. Save timestamps metadata as json
    metadata = {
        "start_ns": target_start_time_ns,
        "topic": audio_topic,
        "original_bag": str(bag_path),
    }

    # Save it
    with open(timestamps_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved audio start timestamp to: {timestamps_path}")

    return output_wav, target_start_time_ns, timestamps_path


def merge_video_audio(video_path, audio_path, output_path):
    """
    Combines AVI and WAV into a single MP4.
    Mixes 16-channel audio down to Stereo for compatibility.
    """
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        print("Missing video or audio file, cannot create MP4.")
        return

    print("Merging Audio and Video into MP4...")

    # FFmpeg command explanation:
    # -y : Overwrite output file
    # -i : Input video
    # -i : Input audio
    # -c:v libx264 : Encode video to H.264 (standard MP4 format)
    # -crf 23 : Video quality (lower is better, 23 is standard)
    # -c:a aac : Encode audio to AAC
    # -ac 2 : Force 2 channels (Stereo) for compatibility
    # -shortest : Finish encoding when the shortest stream ends

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-shortest",
        output_path,
    ]

    try:
        # Run ffmpeg silently (stdout=subprocess.DEVNULL) to avoid clutter
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"SUCCESS! Preview video saved to: {output_path}")
    except subprocess.CalledProcessError:
        print("Error running FFmpeg. Is it installed? (sudo apt install ffmpeg)")
    except FileNotFoundError:
        print("FFmpeg not found. Please install it with 'sudo apt install ffmpeg'")


if __name__ == "__main__":
    # 1. Bag Dir Path and Name
    bag_dir_path = project_file("data", "raw", "ros_bag", "drone_indoor")
    bag_name = "rosbag2_2026_01_22-11_22_36_copy"

    # 2. Bag path and output path
    bag_path = project_file(bag_dir_path, bag_name)
    output_dir = project_file(
        bag_dir_path, f"{os.path.splitext(bag_name)[0]}_extracted"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 3. Export video
    vid_path, vid_start_time_ns, vid_timestamp_path = export_rosbag_vid_to_avi(
        bag_path=bag_path,
        output_dir=output_dir,
        video_topic="/video_raw/compressed",
        output_file_name="extracted_video.avi",
        TARGET_FPS=30,
        force=False,
    )

    # 4. Export Audio (Synced to Video)
    audio_path, audio_start_time_ns, audio_timestamp_path = export_bag_audio(
        bag_path=bag_path,
        output_dir=output_dir,
        audio_topic="/audio_raw",
        output_file_name="extracted_audio.wav",
        sync_reference_ns=vid_start_time_ns,
        SAMPWIDTH=4,
        force=False,
    )

    # # 5. Merge
    # if vid_path and audio_path:
    #     combined_vid_autio_path = project_file(output_dir, "combined_vid_audio.mp4")
    #     merge_video_audio(vid_path, audio_path, combined_vid_autio_path)

    # ##### EXTRACT DEBUG VIDEO ONLY #####
    # # Video loc bag name
    # vid_bag_name = "rosbag2_2026_01_22-11_22_36_audio_loc_video_quat_2"
    # export_rosbag_vid_to_avi(
    #     bag_dir_path, vid_bag_name, topic_name="/video/debug", output_dir=output_dir
    # )
