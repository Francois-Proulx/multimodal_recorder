import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import os
import numpy as np
import wave
import csv
import time
from multiprocessing import Process, Queue, Event
from queue import Full
from collections import deque
from ament_index_python.packages import get_package_share_directory
from droneaudition_msgs.msg import AudioRaw
from droneaudition_msgs.msg import AudioLoc

from .processing import AudioProcessor

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class FileWriterProcess(Process):
    """
    Writes audio chunks from a queue into a WAV file.
    """

    def __init__(
        self,
        queue,
        stop_event,
        save_path,
        filename="audio.wav",
        samplerate=16000,
        channels=16,
        sampwidth=4,
    ):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        self.samplerate = samplerate
        self.channels = channels
        self.sampwidth = sampwidth  # 4 bytes = int32
        os.makedirs(save_path, exist_ok=True)
        self.wave_path = os.path.join(save_path, filename)
        self.csv_path = self.wave_path.replace(".wav", "_timestamps.csv")

    def run(self):
        print("FileWriterProcess started, saving to", self.wave_path)
        with (
            wave.open(self.wave_path, "wb") as wf,
            open(self.csv_path, "w", newline="") as csvfile,
        ):
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp"])
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sampwidth)
            wf.setframerate(self.samplerate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    wf.writeframes(data["audio"].tobytes())
                    writer.writerow([data["timestamp"]])
                except Exception:
                    continue
        print("FileWriterProcess finished")


class Audio_Loc(Node):
    def __init__(self):
        super().__init__("audio_loc")
        # Shared parameters
        self.declare_parameter("samplerate", 16000)
        self.samplerate = (
            self.get_parameter("samplerate").get_parameter_value().integer_value
        )
        self.declare_parameter("channels", 16)
        self.channels = (
            self.get_parameter("channels").get_parameter_value().integer_value
        )
        self.declare_parameter("blocksize", 1024)
        self.BLOCKSIZE = (
            self.get_parameter("blocksize").get_parameter_value().integer_value
        )
        self.declare_parameter("sampwidth", 4)
        self.sampwidth = (
            self.get_parameter("sampwidth").get_parameter_value().integer_value
        )

        # Specific parameters
        self.declare_parameter("loc_type", "SRP-PHAT")
        self.loc_type = (
            self.get_parameter("loc_type").get_parameter_value().string_value
        )

        self.declare_parameter("grid_type", "fibonacci_half_sphere")
        self.grid_type = (
            self.get_parameter("grid_type").get_parameter_value().string_value
        )

        self.declare_parameter("nb_points", 500)
        self.nb_points = (
            self.get_parameter("nb_points").get_parameter_value().integer_value
        )

        self.declare_parameter(
            "mic_array_file", "half_sphere_array_pos_16mics_clockwise.npy"
        )
        self.mic_array_name = (
            self.get_parameter("mic_array_file").get_parameter_value().string_value
        )

        # Save parameters
        self.declare_parameter("save_audio", False)
        self.save_audio = (
            self.get_parameter("save_audio").get_parameter_value().bool_value
        )

        self.declare_parameter(
            "save_path", "/home/francois/Documents/Git/multimodal_recorder/data"
        )
        self.save_path = (
            self.get_parameter("save_path").get_parameter_value().string_value
        )

        self.declare_parameter("filename", "output.wav")
        self.filename = (
            self.get_parameter("filename").get_parameter_value().string_value
        )

        # Processing parameters
        self.declare_parameter("processing_enabled", True)
        self.processing_enabled = (
            self.get_parameter("processing_enabled").get_parameter_value().bool_value
        )

        self.declare_parameter("frame_size", 512)
        self.FRAME_SIZE = (
            self.get_parameter("frame_size").get_parameter_value().integer_value
        )

        self.declare_parameter("hop_size", 256)
        self.HOP_SIZE = (
            self.get_parameter("hop_size").get_parameter_value().integer_value
        )

        self.declare_parameter("nfft", 512)
        self.nfft = self.get_parameter("nfft").get_parameter_value().integer_value

        # Subscription and Publisher
        self.subscription = self.create_subscription(
            AudioRaw, "/audio_raw", self.audio_callback, 20
        )

        self.publisher = self.create_publisher(AudioLoc, "/audio_loc", 20)

        self.marker_pub = self.create_publisher(Marker, "/audio_marker", 10)

        if self.processing_enabled:
            # Initialize buffer
            self.audio_buffer = deque()

            # Initialize audio frame
            self.audio_frame = np.zeros(
                (self.FRAME_SIZE, self.channels), dtype=np.float32
            )

            # Mic array path
            pkg_path = get_package_share_directory("audio")
            mic_file_path = os.path.join(pkg_path, "config", self.mic_array_name)

            # Initialize AudioProcessor (which will compute the offline parameters)
            self.get_logger().info("Initializing Audio Processor...")
            self.audio_processor = AudioProcessor(
                mic_pos_path=mic_file_path,
                fs=self.samplerate,
                nb_of_channels=self.channels,
                nb_points=self.nb_points,
                loc_type=self.loc_type,
                grid_type=self.grid_type,
                window_size=self.BLOCKSIZE,
                nfft=self.nfft,
                FRAME_SIZE=self.FRAME_SIZE,
            )
            self.get_logger().info("Audio Processor initialized.")

            # Timer callback
            self.timer = self.create_timer(0.005, self.processing_loop)

        if self.save_audio:
            # FileWriter setup
            self.audio_queue = Queue(maxsize=100)
            self.stop_event = Event()
            self.file_writer = FileWriterProcess(
                queue=self.audio_queue,
                stop_event=self.stop_event,
                save_path=self.save_path,
                filename=self.filename,
                samplerate=self.samplerate,
                channels=self.channels,
                sampwidth=self.sampwidth,
            )
            self.file_writer.start()  # starts the separate process

    def audio_callback(self, msg):
        start_time = time.time()
        timestamp = msg.time
        data_orig = np.array(msg.data)
        # channels = msg.channels
        # winlen = msg.winlen
        # fs = msg.fs
        # block_time = winlen / self.samplerate

        audio_data = np.reshape(data_orig, (self.BLOCKSIZE, self.channels), order="F")

        # 1 --- Save audio data to WAV file if enabled ---
        if self.save_audio:
            self.convert_audio_and_save_to_queue(audio_data, msg)

        # 2 --- Put data to audio queue for processing loop ---
        if self.processing_enabled:
            # add to queue...
            self.audio_buffer.append((audio_data, timestamp))

        # 3 --- If no processing, publish dummy localization result ---
        else:
            loc_data = [0.0, 0.0, 0.0]  # Dummy localization data
            self.publish_data(timestamp, loc_data)

        # 4 --- Check if too slow ---
        proc_time = time.time() - start_time
        self.BLOCKTIME = self.BLOCKSIZE / self.samplerate
        if proc_time > self.BLOCKTIME:
            self.get_logger().warning(
                f"Processing time {proc_time:.3f}s in audio callback exceeds block time {self.BLOCKTIME:.3f}s"
            )

    def processing_loop(self):
        start_time = time.time()

        if not self.audio_buffer:
            return

        # New data (ex: [4096, 16])
        new_audio_chunk, chunk_start_timestamp = self.audio_buffer.popleft()
        num_samples_in_chunk = new_audio_chunk.shape[0]

        if num_samples_in_chunk % self.HOP_SIZE != 0:
            self.get_logger().error(
                f"CRITICAL: Chunk size {num_samples_in_chunk} is not divisible by Hop {self.HOP_SIZE}. "
                "Data loss occurring!"
            )

        cursor = 0

        # iterate trough the chunk, and pass frames to audio processing
        while cursor + self.HOP_SIZE <= num_samples_in_chunk:
            # 1. Start time of each frame
            time_offset = cursor / self.samplerate
            frame_timestamp = chunk_start_timestamp + time_offset

            # 2. Update audio frame
            hop_data = new_audio_chunk[cursor : cursor + self.HOP_SIZE, :]
            self.audio_frame = np.roll(self.audio_frame, -self.HOP_SIZE, axis=0)
            self.audio_frame[-self.HOP_SIZE :, :] = hop_data

            # 3. Process frame
            loc_data = self.audio_processor.process_frame(
                self.audio_frame, frame_timestamp
            )

            # 4. Publish message
            self.publish_data(frame_timestamp, loc_data)

            # 5. update cursor
            cursor += self.HOP_SIZE

        # !! DEBUG !! -- Check if too slow ---
        proc_time = time.time() - start_time
        if proc_time > self.BLOCKTIME:
            self.get_logger().warning(
                f"Processing time {proc_time:.3f}s in processing loop exceeds block time {self.BLOCKTIME:.3f}s"
            )

    def convert_audio_and_save_to_queue(self, audio_data, msg):
        # Convert float32 to int32 or int16
        if self.sampwidth == 2:
            if audio_data.dtype == np.float32:
                # scale float32 [-1,1] to int16 [-32768,32767]
                audio_data = np.clip(audio_data, -1.0, 1.0) * 32767
            audio_data = audio_data.astype(np.int16)
        elif self.sampwidth == 4:
            if audio_data.dtype == np.float32:
                # scale float32 [-1,1] to int32 [-2147483648,2147483647]
                audio_data = np.clip(audio_data, -1.0, 1.0) * 2147483647
            audio_data = audio_data.astype(np.int32)
        else:
            raise ValueError("Unsupported sampwidth. Use 2 or 4.")

        # Save audio
        try:
            self.audio_queue.put_nowait({"audio": audio_data, "timestamp": msg.time})
        except Full:
            print("Queue full, dropping frame")

    def publish_data(self, timestamp, loc_data):
        msg_pub = AudioLoc()
        msg_pub.time = timestamp
        msg_pub.pos = loc_data
        msg_pub.header.frame_id = "microphone_base"
        msg_pub.header.stamp = rclpy.time.Time(
            seconds=timestamp
        ).to_msg()  # self.get_clock().now().to_msg()
        self.publisher.publish(msg_pub)

        # 2. Publish the Visual Marker (The Arrow)
        ros_time = rclpy.time.Time(seconds=timestamp).to_msg()
        self.publish_marker(loc_data, ros_time, "microphone_base")

    def publish_marker(self, vector, ros_time, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = ros_time

        marker.ns = "sound_source"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow Start Point (0,0,0 - The center of your mic array)
        start = Point(x=0.0, y=0.0, z=0.0)

        # Arrow End Point (The DoA vector)
        # We multiply by 2.0 just to make the arrow longer/easier to see
        end = Point(x=vector[0] * 2.0, y=vector[1] * 2.0, z=vector[2] * 2.0)

        marker.points = [start, end]

        # Scale: x=shaft diameter, y=head diameter, z=head length
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Color: R, G, B, Alpha (Transparency)
        marker.color.r = 1.0  # Red
        marker.color.g = 1.0  # Yellow
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully visible

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    audioloc = Audio_Loc()

    executor = MultiThreadedExecutor()
    executor.add_node(audioloc)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down Audio_Loc...")
    finally:
        if audioloc.save_audio:
            audioloc.stop_event.set()
            audioloc.file_writer.join()
        audioloc.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
