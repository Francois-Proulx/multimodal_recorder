from email.mime import audio
import rclpy
from rclpy.node import Node
import os
import numpy as np
import wave
import csv
import time
from multiprocessing import Process, Queue, Event
from queue import Full

from droneaudition_msgs.msg import AudioRaw
from droneaudition_msgs.msg import AudioLoc


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
        # Declare parameters with optional default values
        self.declare_parameter("save_audio", True)
        self.declare_parameter("loc_type", "SRP-PHAT")
        self.declare_parameter(
            "save_path", "/home/francois/Documents/Git/multimodal_recorder/data"
        )
        self.declare_parameter("filename", "output.wav")

        # Load parameters from audio_loc_config.yaml
        self.save_audio = (
            self.get_parameter("save_audio").get_parameter_value().bool_value
        )
        self.loc_type = (
            self.get_parameter("loc_type").get_parameter_value().string_value
        )
        self.save_path = (
            self.get_parameter("save_path").get_parameter_value().string_value
        )
        self.filename = (
            self.get_parameter("filename").get_parameter_value().string_value
        )

        # !!!!!!Hardcoded audio params, should place them in config later!!!!
        self.samplerate = 16000
        self.channels = 16
        self.sampwidth = 4
        self.processing_enabled = True

        self.subscription = self.create_subscription(
            AudioRaw, "/audio_raw", self.audio_callback, 20
        )

        self.publisher = self.create_publisher(AudioLoc, "/audio_loc", 20)

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
        channels = msg.channels
        winlen = msg.winlen
        fs = msg.fs
        block_time = winlen / fs

        audio_data = np.reshape(data_orig, (winlen, channels), order="F")

        # 1 --- Save audio data to WAV file if enabled ---
        if self.save_audio:
            self.convert_audio_and_save_to_queue(audio_data, msg)

        # 2 --- Audio Localization ---
        # here do localization with "audio_data",
        # with shape (N,C), N being length, C being channels
        if self.processing_enabled:
            loc_data = self.process_localization(audio_data, winlen, channels, fs)
        else:
            loc_data = [0.0, 0.0, 0.0]  # Dummy localization data

        # 3 --- Publish localization result ---
        self.publish_data(timestamp, loc_data)

        # 4 --- Check if too slow ---
        proc_time = time.time() - start_time
        if proc_time > block_time:
            self.get_logger().warning(
                f"Processing time {proc_time:.3f}s exceeds block time {block_time:.3f}s"
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
        msg_pub.header.stamp = rclpy.time.Time(
            seconds=timestamp
        ).to_msg()  # self.get_clock().now().to_msg()
        self.publisher.publish(msg_pub)

    def process_localization(self, audio_data, winlen, channels, fs):
        # Implement your localization algorithm here
        loc_data = [0.0, 0.0, 0.0]  # Dummy localization data
        return loc_data


def main(args=None):
    rclpy.init(args=args)
    audioloc = Audio_Loc()
    try:
        rclpy.spin(audioloc)
    except KeyboardInterrupt:
        print("Shutting down Audio_Loc...")
    finally:
        if audioloc.save_audio:
            audioloc.stop_event.set()
            audioloc.file_writer.join()
        audioloc.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
