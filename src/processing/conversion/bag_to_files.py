import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import subprocess

from src.utils.io import project_file
from src.processing.imu.visualize_imu import plot_raw_imu
from src.processing.imu.main_imu import estimate_orientation_from_imu_data

# IMPORTS FOR YOUR MESSAGES
# Ensure you source your workspace before running this script!
from sensor_msgs.msg import CompressedImage
from droneaudition_msgs.msg import AudioRaw, IMURaw  # Adjust if your msg name differs


def extract_bag(bag_dir_path, bag_name):
    # 0. Initialize variables
    SAMPLERATE = None
    CHANNELS = None
    TARGET_FPS = 30.0  # Desired output FPS
    VIDEO_TOPIC = "/video_raw/compressed"  # Topic names
    AUDIO_TOPIC = "/audio_raw"
    IMU_TOPIC = "/imu_raw"

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
    imu_data = []
    video_writer = None

    video_start_ns = None
    audio_start_ns = None

    frames_written = 0
    last_frame = None

    # 3. Iterate through messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == VIDEO_TOPIC:
            if video_start_ns is None:
                video_start_ns = t
                print(f"Video syncing to start time: {t}")

            msg = deserialize_message(data, CompressedImage)
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # 1. Initialize writer on the very first frame
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_out_path = os.path.join(output_dir, "extracted_video.avi")
                    video_writer = cv2.VideoWriter(
                        video_out_path,
                        cv2.VideoWriter_fourcc(*"DIVX"),
                        TARGET_FPS,
                        (width, height),
                    )
                    print(
                        f"Video writer initialized: {width}x{height}. Syncing to start time: {t}"
                    )

                # 2. Frame timing control
                elapsed_time_s = (t - video_start_ns) / 1e9
                expected_frame_count = int(elapsed_time_s * TARGET_FPS)

                # 3. Fill gaps if frame drops
                while frames_written < expected_frame_count:
                    if last_frame is not None:
                        video_writer.write(last_frame)
                    else:
                        video_writer.write(frame)  # Only for very first frame
                    frames_written += 1

                # 4. Write current frame
                video_writer.write(frame)
                frames_written += 1
                last_frame = frame

        elif topic == AUDIO_TOPIC:
            if audio_start_ns is None:
                audio_start_ns = t
                print(f"Audio syncing to start time: {t}")

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

        elif topic == IMU_TOPIC:
            msg = deserialize_message(data, IMURaw)

            # Append IMU data
            imu_data.append(
                (
                    msg.time,
                    msg.acc[0],
                    msg.acc[1],
                    msg.acc[2],
                    msg.gyr[0],
                    msg.gyr[1],
                    msg.gyr[2],
                    msg.mag[0],
                    msg.mag[1],
                    msg.mag[2],
                    msg.roll,
                    msg.pitch,
                    msg.yaw,
                )
            )

    # 4. Cleanup Video
    if video_writer is not None:
        video_writer.release()
        print("Video saved successfully.")
    else:
        print("No video found!")

    # 5. Plot IMU data
    if imu_data:
        imu_array = np.array(imu_data)
        time_imu = imu_array[:, 0] - imu_array[0, 0]
        acc = imu_array[:, 1:4]
        gyro = imu_array[:, 4:7]
        mag = imu_array[:, 7:10]
        plot_raw_imu(time_imu, acc, gyro, mag)

        acc = np.ascontiguousarray(acc)
        gyro = np.ascontiguousarray(gyro)
        mag = np.ascontiguousarray(mag)

        Ts = np.mean(time_imu[1:] - time_imu[:-1])
        estimate_orientation_from_imu_data(
            Ts=Ts,
            timestamps=time_imu,
            gyro=gyro,
            acc=acc,
            mag=mag,
            offline=False,
            params=None,
            plot=True,
        )
        plt.show()

    # 6. Save Audio to WAV
    if audio_samples and video_start_ns is not None and audio_start_ns is not None:
        print(f"Saving {len(audio_samples)} audio samples...")
        # Stack audio blocks
        full_audio = np.vstack(audio_samples)

        # Calculate Offset (Positive = Audio is Late, Negative = Audio is Early)
        offset_seconds = (audio_start_ns - video_start_ns) / 1e9
        offset_samples = int(abs(offset_seconds) * SAMPLERATE)

        print("--- Sync Analysis ---")
        print(f"Video Start: {video_start_ns}")
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
    bag_dir_path = project_file("data", "raw", "ros_bag", "drone_indoor")

    # Bag name
    bag_name = "rosbag2_2026_01_09-14_55_23"

    # Extract bag
    vid_path, audio_path = extract_bag(bag_dir_path, bag_name)

    # Get parent dir
    combined_vid_audio_path = project_file(
        os.path.dirname(vid_path), "combined_vid_audio.mp4"
    )

    merge_video_audio(vid_path, audio_path, combined_vid_audio_path)
