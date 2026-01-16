import rclpy
from rclpy.node import Node
import os
import numpy as np
import csv
from vqf import VQF
from queue import Empty
from multiprocessing import Process, Queue, Event
from droneaudition_msgs.msg import IMURaw
from geometry_msgs.msg import QuaternionStamped


class IMUFileWriter(Process):
    def __init__(
        self, queue: Queue, stop_event, save_path: str, filename="imu_data.csv"
    ):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        os.makedirs(save_path, exist_ok=True)
        self.filepath = os.path.join(save_path, filename)

    def run(self):
        print("IMUFileWriter started, saving to", self.filepath)
        fieldnames = [
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "roll",
            "pitch",
            "yaw",
        ]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                    f.flush()
                except Empty:
                    continue
                except Exception as e:
                    print("IMUFileWriter error:", e)
                    continue


class IMU_Quat(Node):
    def __init__(self):
        super().__init__("imu_quat")

        # Declare parameters with optional default values
        self.declare_parameter("quat_type", "inverse")
        self.declare_parameter("Ts", 0.01)
        self.declare_parameter(
            "save_path", "/home/francois/Documents/Git/multimodal_recorder/data"
        )
        self.declare_parameter("filename", "imu.csv")
        self.declare_parameter("save_imu", False)

        # Load parameters from audio_loc_config.yaml
        self.quat_type = (
            self.get_parameter("quat_type").get_parameter_value().string_value
        )
        self.Ts = self.get_parameter("Ts").get_parameter_value().double_value
        self.save_path = (
            self.get_parameter("save_path").get_parameter_value().string_value
        )
        self.filename = (
            self.get_parameter("filename").get_parameter_value().string_value
        )
        self.save_imu = self.get_parameter("save_imu").get_parameter_value().bool_value

        self.vqf = VQF(self.Ts)

        self.subscription = self.create_subscription(
            IMURaw, "imu_raw", self.imu_callback, 20
        )
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(
            QuaternionStamped, "/imu/orientation", 20
        )

        # IMU saving
        if self.save_imu:
            self.imu_queue = Queue(maxsize=100)
            self.stop_event = Event()
            self.file_writer = IMUFileWriter(
                self.imu_queue, self.stop_event, self.save_path, self.filename
            )
            self.file_writer.start()

    def imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        acc_arr = np.array([msg.acc])
        gyr_arr = np.array([msg.gyr])
        mag_arr = np.array([msg.mag])
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw

        # --- OPTION A: SAVING RAW IMU DATA ---
        if self.save_imu:
            try:
                data = {
                    "timestamp": timestamp,
                    "acc_x": acc_arr[0, 0],
                    "acc_y": acc_arr[0, 1],
                    "acc_z": acc_arr[0, 2],
                    "gyro_x": gyr_arr[0, 0],
                    "gyro_y": gyr_arr[0, 1],
                    "gyro_z": gyr_arr[0, 2],
                    "mag_x": mag_arr[0, 0],
                    "mag_y": mag_arr[0, 1],
                    "mag_z": mag_arr[0, 2],
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                }
                self.imu_queue.put_nowait(data)
            except Exception as e:
                self.get_logger().warn(f"Failed to save IMU data: {e}")

        # --- OPTION B: PROCESS RAW DATA TO GET QUAT ---

        self.vqf.update(gyr_arr[0], acc_arr[0])
        quat = self.vqf.getQuat6D()  # [w, x, y, z]

        # self.vqf.update(gyr_arr[0], acc_arr[0], mag_arr[0])
        # quat = self.vqf.getQuat9D()  # [w, x, y, z]

        # Publish data
        if quat is not None:
            self.publish_data(quat, msg.header.stamp)

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
    imuquat = IMU_Quat()
    try:
        rclpy.spin(imuquat)
    except KeyboardInterrupt:
        print("Shutting down Imu_quat...")
    finally:
        if imuquat.save_imu:
            imuquat.stop_event.set()
            imuquat.file_writer.join()
        imuquat.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
