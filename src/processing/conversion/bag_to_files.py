import os
import cv2
import numpy as np
from scipy.io import wavfile
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import subprocess

from src.utils.io import project_file

# IMPORTS FOR YOUR MESSAGES
# Ensure you source your workspace before running this script!
from sensor_msgs.msg import CompressedImage
from droneaudition_msgs.msg import AudioRaw  # Adjust if your msg name differs


def extract_bag(bag_dir_path, bag_name):
    # 0. Initialize variables
    SAMPLERATE = None
    CHANNELS = None

    # Desired output format
    SAMPWIDTH = 4  # 4 bytes = int32

    # 0. Bag path and output path
    bag_path = project_file(bag_dir_path, bag_name)
    output_dir = project_file(
        bag_dir_path, f"{os.path.splitext(bag_name)[0]}_extracted"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup Bag Reader
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 2. Setup Lists to hold data
    audio_samples = []
    video_writer = None

    # 3. Iterate through messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == "/video_raw/compressed":
            msg = deserialize_message(data, CompressedImage)
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Initialize writer on the very first frame
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_out_path = os.path.join(output_dir, "extracted_video.avi")
                    video_writer = cv2.VideoWriter(
                        video_out_path,
                        cv2.VideoWriter_fourcc(*"DIVX"),
                        30,
                        (width, height),
                    )
                    print(f"Video writer initialized: {width}x{height}")

                # Write immediately (Saves RAM)
                video_writer.write(frame)

        elif topic == "/audio_raw":
            msg = deserialize_message(data, AudioRaw)

            if SAMPLERATE is None:
                SAMPLERATE = msg.fs
                CHANNELS = msg.channels
                print(f"Audio Sample Rate: {msg.fs}, Channels: {msg.channels}")

            # Convert to numpy
            audio_block = np.array(msg.data, dtype=np.float32)

            # Reshape to (Time, Channels)
            try:
                audio_block_reshaped = audio_block.reshape((-1, CHANNELS), order="F")
                audio_samples.append(audio_block_reshaped)
            except ValueError:
                print("Error reshaping audio block, skipping this block.")

    # Cleanup Video
    if video_writer is not None:
        video_writer.release()
        print("Video saved successfully.")
    else:
        print("No video found!")

    # 4. Save Audio to WAV
    if audio_samples:
        print(f"Saving {len(audio_samples)} audio samples...")
        # 1. Stack audio blocks
        full_audio = np.vstack(audio_samples)

        # 2. Scale data (Float32 -> Int32)
        if SAMPWIDTH == 4:
            print("Converting Float32 -> Int32...")
            # Scale [-1, 1] to [-2147483648, 2147483647]
            full_audio = np.clip(full_audio, -1.0, 1.0) * 2147483647
            full_audio = full_audio.astype(np.int32)
        elif SAMPWIDTH == 2:
            print("Converting Float32 -> Int16...")
            full_audio = np.clip(full_audio, -1.0, 1.0) * 32767
            full_audio = full_audio.astype(np.int16)

        # 4. Save wav
        output_wav = os.path.join(output_dir, "extracted_audio.wav")
        wavfile.write(output_wav, SAMPLERATE, full_audio)
        print(f"Saved audio to: {output_wav}")
    else:
        print("No audio found!")

    return video_out_path, output_wav


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
    # Bag Path
    bag_dir_path = project_file("data", "raw", "ros_bag")

    # Bag name
    bag_name = "rosbag2_2026_01_08-15_36_42"

    # Extract bag
    vid_path, audio_path = extract_bag(bag_dir_path, bag_name)

    # Get parent dir
    combined_vid_audio_path = project_file(
        os.path.dirname(vid_path), "combined_vid_audio.mp4"
    )

    merge_video_audio(vid_path, audio_path, combined_vid_audio_path)
