# src/video/video/video_processor.py

import cv2
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R_scipy
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
        self.gray_scale_threshold = 100
        self.min_error = 5.0

        # Area size
        self.min_area_pixel = 5
        self.max_area_pixel = 1000

        # Physical contraints
        self.min_dist = 0.2  # m
        self.max_dist = 5.0  # m
        self.max_tilt_deg = 25.0  # deg

        self._precompute_indices()

        self.last_rvec = None
        self.last_tvec = None
        self.use_temporal_coherence = True
        self.image_id = 0

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
            quat = self.rvec_to_quat(rvec)  # [x, y, z, w]
            yaw, pitch, roll = self.quaternion_to_euler(quat)
        else:
            roll = None
            pitch = None
            yaw = None

        return quat, tvec, debug_img, roll, pitch, yaw

    def detect_pose_from_ir_led_img(self, image, camera_matrix, dist_coeffs):
        self.image_id += 1
        # gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold
        _, mask = cv2.threshold(gray, self.gray_scale_threshold, 255, cv2.THRESH_BINARY)

        # find light spots with findContours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # create copy for debug
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        text = f"ID: {self.image_id}"
        cv2.putText(
            debug_img,
            text,
            (w - 150, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

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
        if debug_message != "Valid":
            if best_img_pts is not None:
                for i, pt in enumerate(best_img_pts):
                    cv2.circle(
                        debug_img,
                        (int(pt[0]), int(pt[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                    # Draw the INDEX number (0, 1, 2, 3)
                    # This tells you exactly how the solver mapped the points.
                    # 0 = First Point in your OBJECT_POINTS list
                    # 1 = Second Point, etc.
                    cv2.putText(
                        debug_img,
                        str(i),
                        (int(pt[0]) + 10, int(pt[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
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
            else:
                print("best img is none")
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
        # print(
        #     f"\n=== [Frame {self.image_id}] Processing {len(image_candidates)} points ==="
        # )

        min_error = float(np.inf)
        best_rvec = None
        best_tvec = None
        best_img_pts = None
        best_error_msg = "No fit"
        has_valid_solution = False
        for k in range(self.no_pts, self.min_pt - 1, -1):
            perm_idx = 0
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
                        perm_idx += 1

                        perm_img_pts = np.array(perm_img_pts, dtype=np.float32)

                        # Run PnP
                        success, is_valid, rvec, tvec, error, error_msg = (
                            self.check_pnp_gen_solution(
                                subset_model_pts,
                                perm_img_pts,
                                camera_matrix,
                                dist_coeffs,
                            )
                        )
                        # # --- DEBUG PRINT ---
                        # status_tag = "VALID" if is_valid else "INVALID (Ghost/Tilt)"
                        # if success:
                        #     print(
                        #         f"[ID {self.image_id}] Perm #{perm_idx} | Pts: {k} | Err: {error:.4f} | {status_tag} | Msg: {error_msg}"
                        #     )
                        # else:
                        #     print(
                        #         f"[ID {self.image_id}] Perm #{perm_idx} | Pts: {k} | NOT SUCCESS | Msg: {error_msg}"
                        #     )
                        # # -------------------

                        if success:
                            # If fit is physicaly valid with small error
                            if is_valid and error < self.min_error:
                                # If best result or no current best
                                if not has_valid_solution or error < min_error:
                                    has_valid_solution = True
                                    min_error = error
                                    best_rvec = rvec
                                    best_tvec = tvec
                                    best_img_pts = perm_img_pts
                                    best_error_msg = "Valid"

                            # If not valid and no current best, update best results for debuging
                            elif not has_valid_solution:
                                if error < min_error:
                                    min_error = error
                                    best_rvec = rvec
                                    best_tvec = tvec
                                    best_img_pts = perm_img_pts
                                    best_error_msg = "Warning_ghost_only"

            # If valid solution after 4 points, no need for 3
            if has_valid_solution:
                return (
                    best_rvec,
                    best_tvec,
                    min_error,
                    best_img_pts,
                    debug_img,
                    best_error_msg,
                )

        return best_rvec, best_tvec, min_error, best_img_pts, debug_img, best_error_msg

    def check_pnp_solution(self, obj_pts, img_pts, K, D):
        # --- CHOSE SOLVER ----
        num_points = len(img_pts)
        if num_points >= 4:
            flags = cv2.SOLVEPNP_IPPE
        else:
            flags = cv2.SOLVEPNP_SQPNP  # SQPNP is very robust for small point sets

        # --- SOLVE ----
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            D,
            flags=flags,
        )

        if not success:
            return False, None, None, None, "No_fit"

        # --- REPROJECTION ERROR FILTER ----
        # Filter outif error is too large
        error = self.get_reprojection_error(obj_pts, img_pts, K, D, rvec, tvec)
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

    def check_pnp_gen_solution(self, obj_pts, img_pts, K, D):
        # 1. CHOOSE MULTI-SOLUTION SOLVER
        # if len(img_pts) >= 4:
        #     # IPPE is best for planar objects and returns both "Flip" solutions
        #     flags = cv2.SOLVEPNP_IPPE
        # else:
        #     # AP3P returns up to 4 solutions for 3 points.
        #     # (SQPNP only returns 1, so we can't use it here)
        #     flags = cv2.SOLVEPNP_P3P

        # obj_pts = np.ascontiguousarray(obj_pts, dtype=np.float64).reshape((-1, 1, 3))
        # img_pts = np.ascontiguousarray(img_pts, dtype=np.float64).reshape((-1, 1, 2))

        # 2. GET ALL CANDIDATES
        try:
            if len(img_pts) == 3:
                n_sols, rvecs, tvecs = cv2.solveP3P(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_AP3P
                )
            else:
                n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE
                )
        except cv2.error as e:
            return False, False, None, None, float("inf"), f"Solver_Error: {e.msg}"

        if n_sols == 0 or not rvecs:
            return False, False, None, None, float("inf"), "No_solution"

        # 3. FILTER CANDIDATES
        # Loop through ALL solutions and pick the one that is:
        # A) Physically valid (Tilt < 30 deg)
        # B) Has the lowest error among the valid ones

        best_valid_rvec = None
        best_valid_tvec = None
        best_valid_error = float("inf")
        found_valid_solution = False

        best_invalid_rvec = None
        best_invalid_tvec = None
        best_invalid_error = float("inf")
        error_msg = None

        # rvecs is a tuple/list of (3,1) arrays
        for i in range(len(rvecs)):
            rvec_candidate = rvecs[i]
            tvec_candidate = tvecs[i]

            # --- Calculate Reprojection Error ---
            error_candidate = self.get_reprojection_error(
                obj_pts, img_pts, K, D, rvec_candidate, tvec_candidate
            )

            # --- Check Physics (Gravity/Tilt) ---
            # - Orientation
            # This is the critical step: Reject the 80deg "Ghost" immediately
            valid_orientation = self.is_valid_orientation(rvec_candidate)

            # - Distance constraints
            z_dist = tvec_candidate[2][0]
            valid_distance = self.min_dist <= z_dist <= self.max_dist

            if valid_orientation and valid_distance:
                if error_candidate < best_valid_error:
                    best_valid_error = error_candidate
                    best_valid_rvec = rvec_candidate
                    best_valid_tvec = tvec_candidate
                    found_valid_solution = True
            else:
                if error_candidate < best_invalid_error:
                    best_invalid_error = error_candidate
                    best_invalid_rvec = rvec_candidate
                    best_invalid_tvec = tvec_candidate
                    error_msg = f"Ori: {valid_orientation}, Dis: {valid_distance}"

        if found_valid_solution:
            return (
                True,
                True,
                best_valid_rvec,
                best_valid_tvec,
                best_valid_error,
                f"Valid, tilt: {self.tilt_angle_deg}",
            )

        # We only found Ghosts. Return the best Ghost (marked as Invalid).
        return (
            True,
            False,
            best_invalid_rvec,
            best_invalid_tvec,
            best_invalid_error,
            f"Invalid_Physics: {error_msg}, tilt: {self.tilt_angle_deg}",
        )

    def rvec_to_quat(self, rvec):
        """
        Converts OpenCV rvec (Rodrigues vector) to a SciPy Quaternion [x, y, z, w].
        """
        # Ensure it's a 1D array (cv2 usually returns shape (3, 1))
        rot = R_scipy.from_rotvec(rvec.flatten())

        # Return standard SciPy format: [x, y, z, w]
        return rot.as_quat()

    def quaternion_to_euler(self, quat):
        """
        Convert quaternions [x, y, z, w] to Euler angles (roll, pitch, yaw) in degrees.

        Parameters
        ----------
        [x, y, z, w] : numpy array
            Quaternion

        Returns
        -------
        roll, pitch, yaw : numpy arrays
            Euler angles in degrees
        """

        rot_obj = R_scipy.from_quat(quat)
        euler = rot_obj.as_euler("zyx", degrees=True)

        if euler.ndim == 1:
            yaw, pitch, roll = euler[0], euler[1], euler[2]
        else:
            yaw = euler[:, 0]
            pitch = euler[:, 1]
            roll = euler[:, 2]

        return yaw, pitch, roll

    def get_reprojection_error(self, obj_pts, img_pts, K, D, rvec, tvec):
        projected_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)

        projected_points = projected_points.reshape(-1, 2).astype(np.float32)

        # Calculate average distance between detected points and projected model points
        total_norm = cv2.norm(img_pts, projected_points, cv2.NORM_L2)
        error = total_norm / len(img_pts)
        return error

    def is_valid_orientation(self, rvec):
        rot_obj = R_scipy.from_rotvec(rvec.flatten())

        # 1: Filter out tilt angle (accept updisde down solutions)
        R_mat = rot_obj.as_matrix()
        self.tilt_angle_deg = np.degrees(np.arccos(np.abs(R_mat[2, 2])))

        if self.tilt_angle_deg > self.max_tilt_deg:
            return False

        # 2: Filter out upside down solutions
        # (not sure here... to test)
        _, _, roll = rot_obj.as_euler("zyx", degrees=True)
        if np.abs(roll) > 90:
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
