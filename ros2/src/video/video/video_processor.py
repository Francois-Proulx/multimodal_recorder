# src/video/video/video_processor.py

import cv2
import numpy as np
import itertools
# import torch
# import torchvision


class VideoProcessor:
    def __init__(self):
        self.OBJECT_POINTS = np.array(
            [
                [0.125, 0.093, 0],
                [0.125, -0.095, 0],
                [-0.125, -0.050, 0],
                [-0.125, 0.054, 0],
            ],
            dtype="double",
        )
        self.no_pts = len(self.OBJECT_POINTS)
        self.max_pt = 8
        self.min_pt = 3
        self.gray_scale_threshold = 180
        self.min_error = 5.0

        # Area size
        self.min_area_pixel = 5
        self.max_area_pixel = 1000

        # Physical contraints
        self.min_dist = 0.2  # m
        self.max_dist = 5.0  # m
        self.max_tilt_deg = 25.0  # deg

        self._precompute_indices()

    def _precompute_indices(self):
        self.permutations = list(itertools.permutations(range(self.no_pts)))
        self.combinations = list(
            itertools.combinations(range(self.max_pt), self.no_pts)
        )
        total_ops = len(self.combinations) * len(self.permutations)
        print(f"Solver Config: Picking {self.no_pts} from {self.max_pt}.")
        print(f"Worst-case complexity: {total_ops} checks per frame.")

    def detect_pose(self, image, camera_matrix, dist_coeffs):
        quat = None
        rvec, tvec, debug_img = self.detect_pose_from_ir_led_img(
            image, camera_matrix, dist_coeffs
        )

        if rvec is not None:
            quat = self.rvec_to_quat(rvec)  # [w, x, y, z]
            roll, pitch, yaw = self.quaternion_to_euler(
                quat[1], quat[2], quat[3], quat[0]
            )
        else:
            roll = None
            pitch = None
            yaw = None

        return quat, tvec, debug_img, roll, pitch, yaw

    def detect_pose_from_ir_led_img(self, image, camera_matrix, dist_coeffs):
        # gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold
        _, mask = cv2.threshold(gray, self.gray_scale_threshold, 255, cv2.THRESH_BINARY)

        # find light spots with findContours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # create copy for debug
        debug_img = image.copy()

        # Loop through the light spots, find center, and draw debug image
        candidates = []
        cv2.drawContours(debug_img, contours, -1, (255, 255, 0), 1)
        for cnt in contours:
            # check if valid shape
            if self.is_valid_shape(cnt):
                area = cv2.contourArea(cnt)
                # Find center of the spot
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    candidates.append({"pt": (cX, cY), "area": area})

                    # VISUALIZATION:
                    # 1. Draw the contour outline in GREEN (Accepted)
                    cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)

                    # Plot a circle at the center of the candidate spots
                    cv2.circle(debug_img, (cX, cY), 2, (0, 0, 255), -1)
            else:
                # VISUALIZATION:
                # Draw REJECTED contours in RED (Thick line)
                # This tells you: "I saw this, but it failed the shape check"
                cv2.drawContours(debug_img, [cnt], -1, (0, 0, 255), 2)

        # Need min no_pts
        if len(candidates) < self.min_pt:
            cv2.putText(
                debug_img,
                f"FAIL: < {self.min_pt} Pts",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return None, None, debug_img

        candidates.sort(key=lambda x: x["area"], reverse=True)

        # Keep best candidates
        best_candidates = candidates[: self.max_pt]

        for c in best_candidates:
            pt = (int(c["pt"][0]), int(c["pt"][1]))
            cv2.circle(debug_img, pt, 8, (255, 0, 0), 2)

        # Call function
        rvec, tvec, error, best_img_pts, debug_img, debug_message = (
            self.solve_pose_generic(
                best_candidates, camera_matrix, dist_coeffs, debug_img
            )
        )

        # PLOT FOR VALIDATION
        if debug_message != "Success":
            if best_img_pts is not None:
                for pt in best_img_pts:
                    cv2.circle(
                        debug_img,
                        (int(pt[0]), int(pt[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                err_text = (
                    f"Err: {error:.2f}, err msg: {debug_message}"
                    if error is not np.inf
                    else "Err: Inf"
                )
                cv2.putText(
                    debug_img,
                    err_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            return None, None, debug_img
        else:
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            for pt in best_img_pts:
                cv2.circle(
                    debug_img,
                    (int(pt[0]), int(pt[1])),
                    5,
                    (0, 255, 0),
                    -1,
                )

            cv2.putText(
                debug_img,
                f"Err: {error:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            return rvec, tvec, debug_img

    def solve_pose_generic(
        self, image_candidates, camera_matrix, dist_coeffs, debug_img
    ):
        candidate_points = np.array(
            [c["pt"] for c in image_candidates], dtype="float32"
        )

        min_error = float(np.inf)
        best_rvec = None
        best_tvec = None
        best_img_pts = None
        for k in range(self.no_pts, self.min_pt - 1, -1):
            # Choose "k" points from candidate points
            # Ex: choose 4 points from 8 candidates
            for subset_img_pts in itertools.combinations(candidate_points, k):
                subset_img_pts = np.array(subset_img_pts, dtype=np.float32)

                # Choos "k" points from the model
                # Ex: choose 4 points from 4 model points
                for subset_model_pts in itertools.combinations(self.OBJECT_POINTS, k):
                    subset_model_pts = np.array(subset_model_pts, dtype=np.float32)

                    # Blind (brute force) solver PnP on all permutations
                    for perm_img_pts in itertools.permutations(subset_img_pts):
                        perm_img_pts = np.array(perm_img_pts, dtype=np.float32)

                        # Run PnP
                        success, rvec, tvec, error, error_msg = self.check_pnp_solution(
                            subset_model_pts,
                            perm_img_pts,
                            camera_matrix,
                            dist_coeffs,
                        )

                        if success:
                            return (
                                rvec,
                                tvec,
                                error,
                                perm_img_pts,
                                debug_img,
                                error_msg,
                                # f"Match found with {k} points",
                            )
                        else:
                            if error is not None:
                                if error < min_error:
                                    min_error = error
                                    best_rvec = rvec
                                    best_tvec = tvec
                                    best_img_pts = perm_img_pts
                                    best_error_msg = error_msg

        return best_rvec, best_tvec, min_error, best_img_pts, debug_img, best_error_msg

    def check_pnp_solution(self, obj_pts, img_pts, K, D):
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            D,
            flags=cv2.SOLVEPNP_SQPNP,  # SQPNP is very robust for small point sets
        )

        if not success:
            return False, None, None, None, "No_fit"

        # Calculate Reprojection Error to see if this match makes sense
        projected_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)

        projected_points = projected_points.reshape(-1, 2).astype(np.float32)

        # Calculate average distance between detected points and projected model points
        total_norm = cv2.norm(img_pts, projected_points, cv2.NORM_L2)
        error = total_norm / len(img_pts)

        # Filter outif error is too large
        if error > self.min_error:
            return False, rvec, tvec, error, "Error"

        # --- PHYSICS FILTER ----
        # Filter out if object is too close
        z_dist = tvec[2][0]
        if z_dist < self.min_dist or z_dist > self.max_dist:
            return False, rvec, tvec, error, "Distance"

        # Filter out if orientation is completly off
        if not self.is_valid_orientation(rvec):
            return False, rvec, tvec, error, f"Orientation{self.tilt_angle_deg}"

        return True, rvec, tvec, error, "Success"

    def rvec_to_quat(self, rvec):
        """
        Converts OpenCV rvec (Rodrigues vector) to a Quaternion (w, x, y, z).
        """
        # 1. Get the angle (magnitude of the vector)
        theta = np.linalg.norm(rvec)

        # 2. Handle the edge case (angle is close to 0)
        if theta < 1e-6:
            # Return Identity Quaternion (0 rotation)
            return 1.0, 0.0, 0.0, 0.0

        # 3. Calculate terms
        # Axis = rvec / theta
        # q = [sin(theta/2) * axis, cos(theta/2)]

        k = np.sin(theta / 2.0) / theta

        w = np.cos(theta / 2.0)
        x = rvec[0, 0] * k
        y = rvec[1, 0] * k
        z = rvec[2, 0] * k

        return [w, x, y, z]

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert quaternions (x, y, z, w) to Euler angles (roll, pitch, yaw) in degrees.

        Parameters
        ----------
        x, y, z, w : array-like
            Quaternion components (can be scalars or numpy arrays of the same shape)

        Returns
        -------
        roll, pitch, yaw : numpy arrays
            Euler angles in degrees
        """
        ysqr = y * y

        # Roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        roll = np.degrees(np.arctan2(t0, t1))

        # Pitch (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # clip to avoid numerical errors
        pitch = np.degrees(np.arcsin(t2))

        # Yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        yaw = np.degrees(np.arctan2(t3, t4))

        return roll, pitch, yaw

    def is_valid_orientation(self, rvec):
        # quat = self.rvec_to_quat(rvec)  # [w, x, y, z]
        # roll, pitch, _ = self.quaternion_to_euler(quat[1], quat[2], quat[3], quat[0])
        # if np.abs(roll) > self.max_tilt_deg or np.abs(pitch) > self.max_tilt_deg:
        #     print(f"DEBUG: Roll={roll:.1f}, Pitch={pitch:.1f}")
        #     return False
        # else:
        #     return True

        R, _ = cv2.Rodrigues(rvec)
        # abs to accept -z and +z for upside down solutions
        self.tilt_angle_deg = np.degrees(np.arccos(np.abs(R[2, 2])))
        if self.tilt_angle_deg > self.max_tilt_deg:
            return False

        return True

    def is_valid_shape(self, cnt):
        # filter out too small or too big
        area = cv2.contourArea(cnt)

        if area < self.min_area_pixel or area > self.max_area_pixel:
            return False

        # Check if rectangle
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w == 0 or h == 0:
            return False

        # Aspect ratio check
        ar = w / h

        if ar < 1.0:  # normalize
            ar = 1.0 / ar

        if ar > 6.0:
            return False

        box_area = w * h
        extent = area / box_area
        if extent < 0.6:
            return False

        return True
