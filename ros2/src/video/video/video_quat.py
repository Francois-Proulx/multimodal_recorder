import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import os
from queue import Empty
from multiprocessing import Process, Queue, Event
import csv

from geometry_msgs.msg import QuaternionStamped
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
        self.publisher = self.create_publisher(
            QuaternionStamped, "/video/orientation", 20
        )

        self.subscription = self.create_subscription(
            CompressedImage, "/video_raw/compressed", self.video_callback, 10
        )

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

    def video_callback(self, msg):
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
                quat = self.video_processor.detect_pose(img_data)
                # -------------------------------
            else:
                quat = [0.0, 1.0, 0.0, 0.0]

            # Publish data
            if quat:
                self.get_logger().info("Publishing frame.")
                self.publish_data(quat, msg.header.stamp)

            self.last_process_time = current_time

    def publish_data(self, quat, msg_timestamp):
        msg_pub = QuaternionStamped()
        msg_pub.header.stamp = msg_timestamp
        msg_pub.header.frame_id = "imu_link"
        msg_pub.quaternion.w = float(quat[0])
        msg_pub.quaternion.x = float(quat[1])
        msg_pub.quaternion.y = float(quat[2])
        msg_pub.quaternion.z = float(quat[3])
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
