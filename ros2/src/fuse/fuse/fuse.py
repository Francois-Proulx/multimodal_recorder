import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from std_msgs.msg import Float32
from .kalman_filter import VQFVideoFusion
from collections import deque


class Fuse(Node):
    def __init__(self):
        super().__init__("fuse")
        # Parameters
        self.declare_parameter("imu_scale_pitch", 1.0)
        self.imu_scale_pitch = (
            self.get_parameter("imu_scale_pitch").get_parameter_value().double_value
        )
        self.declare_parameter("imu_scale_yaw", 1.0)
        self.imu_scale_yaw = (
            self.get_parameter("imu_scale_yaw").get_parameter_value().double_value
        )
        self.declare_parameter("imu_scale_roll", 1.0)
        self.imu_scale_roll = (
            self.get_parameter("imu_scale_roll").get_parameter_value().double_value
        )

        self.imu_calib_params = {
            "scale_pitch": self.imu_scale_pitch,
            "scale_yaw": self.imu_scale_yaw,
            "scale_roll": self.imu_scale_roll,
        }

        self.get_logger().info(
            f"Scale pitch: {self.imu_scale_pitch}, Scale yaw: {self.imu_scale_yaw}, scale roll: {self.imu_scale_roll}"
        )

        # Subscription and Publisher
        self.subscription_imu = self.create_subscription(
            QuaternionStamped, "/imu/orientation", self.imu_callback, 10
        )
        self.subscription_video = self.create_subscription(
            PoseStamped, "/video/orientation", self.video_callback, 10
        )

        self.pub_fused = self.create_publisher(
            QuaternionStamped, "/fused/orientation", 10
        )

        self.pub_bias = self.create_publisher(Float32, "/fused/yaw_bias", 10)

        # Initialize Kalman filter
        self.kf = VQFVideoFusion()

        # Buffer: store IMU data in case video lags
        self.imu_buffer = deque(maxlen=300)  # 3 sec buffer

    def imu_callback(self, msg):
        """
        High frequency
        1. Predict (same bias, increase uncertainty)
        2. Get and publish current state
        """
        # A. Extract data
        curr_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        q_imu = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]

        # B. Add to imu buffer
        self.imu_buffer.append((curr_time, q_imu))

        # C. Predict
        self.kf.predict()

        # D. Get Data
        q_fused = self.kf.get_corrected_quaternion(q_imu, self.imu_calib_params)

        # E. Publish
        self.publish_quat_data(q_fused, msg.header.stamp)

    def video_callback(self, msg):
        """
        Low frequency
        1. Look in IMU history to match timestamps
        2. Update Bias
        3. Get and Pusblish Bias
        """
        # A. Extract data
        vid_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        q_vid = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]

        # B. Find closest IMU state in buffer
        closest_vqf = self.find_closest_IMU_state(vid_time)

        if closest_vqf is not None:
            # C. Update Filter
            self.kf.update(
                closest_vqf,
                q_vid,
                video_yaw_static_offset=90.0,
                video_yaw_scale=-1.0,
            )

            # D. Get and Publish Bias
            bias = self.kf.get_bias_degrees()
            self.publish_bias_data(bias, msg.header.stamp)

    def find_closest_IMU_state(self, target_time):
        if not self.imu_buffer:
            self.get_logger().warn("MATCH FAIL: IMU Buffer is Empty")
            return None

        best_q = None
        min_diff = 1.0

        # Get buffer boundaries for debugging
        oldest_imu = self.imu_buffer[0][0]
        newest_imu = self.imu_buffer[-1][0]

        for t, q in self.imu_buffer:
            diff = abs(t - target_time)
            if diff < min_diff:
                min_diff = diff
                best_q = q

        # Check tolerance
        if min_diff < 0.05:  # (50 ms tolerance)
            return best_q

        # --- DEBUGGING BLOCK ---
        # If we failed, print WHY we failed
        self.get_logger().warn(
            f"MATCH FAIL: Video Time {target_time:.4f} | "
            f"Closest Diff: {min_diff:.4f}s | "
            f"Buffer Range: [{oldest_imu:.4f} -> {newest_imu:.4f}]"
        )

        # Check if we are looking for data that fell off the buffer (Too Old)
        if target_time < oldest_imu:
            self.get_logger().error(
                f"--> Video is too OLD! Increase deque maxlen. (Lag: {oldest_imu - target_time:.4f}s)"
            )

        # Check if we are looking for data from the future (Clock Sync)
        elif target_time > newest_imu:
            self.get_logger().error(
                f"--> Video is from FUTURE! Check clock sync. (Lead: {target_time - newest_imu:.4f}s)"
            )
        return None

    def publish_quat_data(self, quat, msg_timestamp):
        msg_pub = QuaternionStamped()
        msg_pub.header.stamp = msg_timestamp
        msg_pub.header.frame_id = "fused_link"
        msg_pub.quaternion.x = float(quat[0])
        msg_pub.quaternion.y = float(quat[1])
        msg_pub.quaternion.z = float(quat[2])
        msg_pub.quaternion.w = float(quat[3])
        self.pub_fused.publish(msg_pub)

    def publish_bias_data(self, bias, msg_timestamp):
        msg = Float32()
        msg.data = bias
        # msg.header.stamp = msg_timestamp
        self.pub_bias.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    fuse = Fuse()
    try:
        rclpy.spin(fuse)
    except KeyboardInterrupt:
        print("Shutting down Fuse node...")
    finally:
        fuse.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
