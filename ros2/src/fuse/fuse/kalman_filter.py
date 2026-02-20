import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


class VQFVideoFusion:
    def __init__(self):
        # --- STATE ---
        # x: The Yaw Bias (Yaw_VQF - Yaw_Video)
        self.bias = 0.0

        # P: Uncertainty in the bias (Starts high)
        self.P = 10.0

        # --- TUNING ---
        # Q: Process Noise (How fast does bias change/drift?)
        # Added every IMU step. Small value = Bias is mostly constant.
        self.Q = 0.0001

        # R: Measurement Noise (How much do we trust Video Yaw?)
        # Large value = Trust Video less (smooth updates). Small = Snap to Video.
        self.R = 10.0

        # Store last quat for continuity
        self.last_fused_quat = None

    def predict(self):
        """
        Call this when IMU data arrives (e.g., 100Hz).
        Propagates uncertainty.
        """
        # Random Walk Model: x_k = x_{k-1} (Bias stays same)
        # But uncertainty grows because drift changes randomly over time
        self.P += self.Q

    def get_corrected_quaternion(self, vqf_quat, imu_params=None):
        """
        Call this immediately after predict() to get the stabilized output.
        """
        rot_vqf = R_scipy.from_quat(vqf_quat)

        # 1. Apply Static Calibrations (Parallax/Scaling)
        if imu_params:
            scale_y = imu_params.get("scale_yaw", 1.0)
            scale_p = imu_params.get("scale_pitch", 1.0)

            if scale_y != 1.0 or scale_p != 1.0:
                euler = rot_vqf.as_euler("zyx", degrees=True)

                # Scale yaw and pitch
                y_vqf = euler[0] * scale_y
                p_vqf = euler[1] * scale_p
                r_vqf = euler[2]

                # Compute corrected rotation object
                rot_vqf = R_scipy.from_euler(
                    "zyx", np.column_stack([y_vqf, p_vqf, r_vqf]), degrees=True
                )

        # 2. Apply Bias Correction (Yaw correction = Rotation around z axis)
        rot_correction = R_scipy.from_euler("z", -self.bias, degrees=True)
        rot_fused = rot_correction * rot_vqf
        q_fused = np.squeeze(np.array(rot_fused.as_quat()))

        # 3. Ensure continuity
        if self.last_fused_quat is not None:
            dot_prod = np.dot(q_fused, self.last_fused_quat)
            if dot_prod < 0:
                q_fused = -q_fused

        self.last_fused_quat = q_fused

        return q_fused

    def update(
        self,
        vqf_quat,
        video_quat_raw,
        video_yaw_static_offset=0.0,
        video_yaw_scale=-1.0,
    ):
        """
        Call this when Video data arrives (e.g., 30Hz).
        Updates the bias estimate.
        """
        if video_quat_raw is None:
            return

        # 1. Get Current VQF Yaw (The "Bad" Yaw)
        r_vqf = R_scipy.from_quat(vqf_quat)
        y_vqf = r_vqf.as_euler("zyx", degrees=True)[0]

        # 2. Get Video Yaw (The "Truth")
        r_vid = R_scipy.from_quat(video_quat_raw)
        y_vid_raw = r_vid.as_euler("zyx", degrees=True)[0]

        # 2.1 Video yaw scaling correction
        y_vid_target = y_vid_raw * video_yaw_scale + video_yaw_static_offset

        # 3. Measure the Bias
        # Expected: y_vqf - bias = y_vid_target  =>  bias = y_vqf - y_vid_target
        z_measured_bias = y_vqf - y_vid_target

        # Unwrap (Handle the 359 -> 0 jump)
        z_measured_bias = (z_measured_bias + 180) % 360 - 180

        # 4. Kalman Update Step (Manual 1D equations)
        # Kalman Gain
        K = self.P / (self.P + self.R)

        # Innovation (Measurement - Estimate)
        innovation = z_measured_bias - self.bias
        innovation = (innovation + 180) % 360 - 180  # Unwrap innovation too

        # Update State
        self.bias += K * innovation

        # Update Uncertainty
        self.P = (1 - K) * self.P

    def get_bias_degrees(self):
        return float(np.degrees(self.bias))
