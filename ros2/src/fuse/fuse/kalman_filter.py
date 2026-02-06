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
        self.R = 1000.0

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
        # 1. Decompose VQF
        r_vqf = R_scipy.from_quat(vqf_quat)
        y_vqf, p_vqf, r_vqf = r_vqf.as_euler("zyx", degrees=True).T

        # 2. Apply Static Calibrations (Parallax/Scaling)
        if imu_params:
            y_vqf *= imu_params.get("scale_yaw", 1.0)
            p_vqf *= imu_params.get("scale_pitch", 1.0)

        # 3. Apply Bias Correction (The Main Event)
        y_corrected = y_vqf - self.bias

        # 4. Reconstruct Quaternion
        r_fused = R_scipy.from_euler(
            "zyx", np.column_stack([y_corrected, p_vqf, r_vqf]), degrees=True
        )
        q_fused = np.squeeze(np.array(r_fused.as_quat()))

        # 5. Ensure continuity
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
        video_yaw_static_offset=90.0,
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

        # 3. Video Yaw Correction
        y_vid_target = y_vid_raw * video_yaw_scale + video_yaw_static_offset

        # 4. Measure the Bias
        # Expected: y_vqf - bias = y_vid_target  =>  bias = y_vqf - y_vid_target
        z_measured_bias = y_vqf - y_vid_target

        # Unwrap (Handle the 359 -> 0 jump)
        z_measured_bias = (z_measured_bias + 180) % 360 - 180

        # 5. Kalman Update Step (Manual 1D equations)
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
