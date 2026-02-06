import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped

import os
import numpy as np
from queue import Empty
from multiprocessing import Process, Queue, Event
import csv

from cv_bridge import CvBridge
from .video_processor import VideoProcessor


class VideoFileWriter(Process):
    """
    Saves video frames as individual JPGs and logs timestamps for precise alignment.
    """

    def __init__(self, frame_queue, stop_event, save_path, prefix="video"):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.prefix = prefix

        os.makedirs(save_path, exist_ok=True)
        self.frames_dir = os.path.join(save_path, f"{prefix}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.csv_path = os.path.join(save_path, f"{prefix}_timestamps.csv")

    def run(self):
        print(f"VideoFileWriter started, saving frames to {self.frames_dir}")

        frame_count = 0
        with open(self.csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["frame_index", "timestamp"])

            try:
                while not self.stop_event.is_set() or not self.frame_queue.empty():
                    try:
                        # item is a dict: {"bytes": <binary data>, "ts": <float>}
                        item = self.frame_queue.get(timeout=0.1)
                        jpg_bytes = item["bytes"]
                        ts = item["ts"]

                        # As the message is already JPEG encoded, just write it directly
                        frame_filename = os.path.join(
                            self.frames_dir, f"frame_{frame_count:06d}.jpg"
                        )
                        with open(frame_filename, "wb") as f:
                            f.write(jpg_bytes)

                        # Save timestamp
                        csv_writer.writerow([frame_count, ts])

                        frame_count += 1

                    except Empty:
                        continue
                    except Exception as e:
                        print("VideoFileWriter error:", e)

            finally:
                print("VideoFileWriter finished, total frames saved:", frame_count)


class Video_Quat(Node):
    def __init__(self):
        super().__init__("video_quat")

        # Save parameters
        self.declare_parameter("save_video", False)
        self.save_video = (
            self.get_parameter("save_video").get_parameter_value().bool_value
        )

        self.declare_parameter(
            "save_path", "/home/francois/Documents/Git/multimodal_recorder/data"
        )
        self.save_path = (
            self.get_parameter("save_path").get_parameter_value().string_value
        )

        # Drone pose estimation parameters
        self.declare_parameter("model", "yolo")
        self.loc_type = self.get_parameter("model").get_parameter_value().string_value

        # Processing parameters
        self.declare_parameter("processing_enabled", False)
        self.processing_enabled = (
            self.get_parameter("processing_enabled").get_parameter_value().bool_value
        )

        self.declare_parameter("processing_fps", 1)
        self.processing_fps = (
            self.get_parameter("processing_fps").get_parameter_value().integer_value
        )

        # ROS2 Publisher and Subscriber
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PoseStamped, "/video/orientation", 20)

        self.subscription = self.create_subscription(
            CompressedImage, "/video_raw/compressed", self.video_callback, 10
        )

        self.sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.info_callback, 10
        )

        self.debug_pub = self.create_publisher(Image, "/video/debug", 10)

        # Setup Multiprocessing Writer
        if self.save_video:
            self.frame_queue = Queue(maxsize=100)
            self.stop_event = Event()
            self.file_writer = VideoFileWriter(
                self.frame_queue, self.stop_event, self.save_path, prefix="video"
            )
            self.file_writer.start()

        # Initialize process time
        self.video_processor = VideoProcessor()
        self.last_process_time = 0.0
        self.camera_matrix = None
        self.dist_coeffs = None

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.get_logger().info("Received Camera Calibration")
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.destroy_subscription(self.sub_cam_info)

    def video_callback(self, msg):
        # Dont run until calibration info
        if self.camera_matrix is None:
            self.get_logger().warn(
                "Waiting for camera_info...", throttle_duration_sec=2
            )
            return

        # 1. EXTRACT TIMESTAMP (Standard ROS Header)
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        # --- OPTION A: FOR SAVING (FAST) ---
        if self.save_video:
            # Convert ROS array to python bytes
            try:
                image_bytes = bytes(msg.data)
                self.frame_queue.put_nowait({"bytes": image_bytes, "ts": timestamp})
            except Exception as e:
                self.get_logger().error(f"Video_Quat saving error: {e}")

        # --- OPTION B: FOR PROCESSING (SLOWER) ---
        current_time = self.get_clock().now().nanoseconds * 1e-9

        # Process to different FPS
        if (current_time - self.last_process_time) > (1 / self.processing_fps):
            self.get_logger().info("Time to process video frame.")
            if self.processing_enabled:
                self.get_logger().info("Processing video frame.")
                # Using cv_bridge to decode the jpeg bytes to cv2 image
                img_data = self.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )

                # -------------------------------
                # Process image to compute quaternion
                # CALL OTHER PYTHON FUNCTION CONTAINING LIGHT YOLO MODEL
                quat, tvec, debug_img, roll, pitch, yaw = (
                    self.video_processor.detect_pose(
                        img_data, self.camera_matrix, self.dist_coeffs
                    )
                )
                # -------------------------------
            else:
                quat = None
                tvec = None
                debug_img = None
                roll = None
                pitch = None
                yaw = None

            # Publish data
            if quat is not None and tvec is not None:
                self.get_logger().info("Publishing frame.")
                self.publish_data(quat, tvec, msg.header.stamp)
                self.get_logger().info(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

            if debug_img is not None:
                if len(debug_img.shape) == 2:
                    encoding = "mono8"
                else:
                    encoding = "bgr8"

                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding=encoding)
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

            self.last_process_time = current_time

    def publish_data(self, quat, tvec, msg_timestamp):
        msg_pub = PoseStamped()
        msg_pub.header.stamp = msg_timestamp
        msg_pub.header.frame_id = "video_link"
        msg_pub.pose.orientation.w = float(quat[0])
        msg_pub.pose.orientation.x = float(quat[1])
        msg_pub.pose.orientation.y = float(quat[2])
        msg_pub.pose.orientation.z = float(quat[3])
        msg_pub.pose.position.x = float(tvec[0][0])
        msg_pub.pose.position.y = float(tvec[1][0])
        msg_pub.pose.position.z = float(tvec[2][0])
        self.publisher.publish(msg_pub)


def main(args=None):
    rclpy.init(args=args)
    videoquat = Video_Quat()
    try:
        rclpy.spin(videoquat)
    except KeyboardInterrupt:
        print("Shutting down Video_quat...")
    finally:
        if videoquat.save_video:
            videoquat.stop_event.set()
            videoquat.file_writer.join()
        videoquat.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
