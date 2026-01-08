import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import os
import glob
import pandas as pd


class Video_Adq(Node):
    def __init__(self):
        super().__init__("video_adq")

        self.declare_parameter("csv_path", "test.csv")
        self.csv_path = (
            self.get_parameter("csv_path").get_parameter_value().string_value
        )
        self.declare_parameter("video_folder", "test")
        self.video_folder = (
            self.get_parameter("video_folder").get_parameter_value().string_value
        )
        self.declare_parameter("save_folder", "test")
        self.save_folder = (
            self.get_parameter("save_folder").get_parameter_value().string_value
        )
        self.declare_parameter("width", 1280)
        self.width = self.get_parameter("width").get_parameter_value().integer_value
        self.declare_parameter("height", 720)
        self.height = self.get_parameter("height").get_parameter_value().integer_value
        self.declare_parameter("fps", 30)
        self.fps = self.get_parameter("fps").get_parameter_value().integer_value
        # self.declare_parameter("pixel_format", "MJPG")
        # self.pixel_format = (
        #     self.get_parameter("pixel_format").get_parameter_value().string_value
        # )

        # PUBLISH STANDARD COMPRESSED IMAGE
        self.publisher = self.create_publisher(
            CompressedImage, "/video_raw/compressed", 20
        )

        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.csv_data = pd.read_csv(self.csv_path)
        self.current_index = 0

    def timer_callback(self):
        if self.current_index >= len(self.csv_data):
            self.get_logger().info("Finished publishing CSV and Video.")
            self.timer.cancel()
            return

        row = self.csv_data.iloc[self.current_index]
        ts_float = row["timestamp"]

        if self.video_folder and os.path.exists(self.video_folder):
            image_paths = glob.glob(os.path.join(self.video_folder, "*.jpg"))
            image_paths.sort()
        else:
            self.get_logger().info("Video directory not found: " + self.video_folder)
            exit()

        # 1. Read Bytes directly from image file
        with open(image_paths[self.current_index], "rb") as img_file:
            jpg_bytes = img_file.read()

        # 2. Construct standard message

        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = jpg_bytes

        # 3. Convert float timestamp to ros header
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp.sec = int(ts_float)
        msg.header.stamp.nanosec = int((ts_float - int(ts_float)) * 1e9)

        self.publisher.publish(msg)
        self.current_index += 1


def main(args=None):
    rclpy.init(args=args)
    videoadq = Video_Adq()
    rclpy.spin(videoadq)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    videoadq.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
