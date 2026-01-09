import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


##### Quaternion to Euler transformation #####
def quaternion_to_euler(x, y, z, w):
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


def euler_to_quaternion(roll, pitch, yaw, degrees=True):
    """
    Convert Euler angles (roll, pitch, yaw) in rad or degrees to quaternions (x, y, z, w).

    Parameters
    ----------
    roll, pitch, yaw : numpy arrays
        Euler angles in degrees or radians
    degrees : bool
        If True, input angles are in degrees; if False, in radians.


    Returns
    -------
    x, y, z, w : numpy arrays
        Quaternion components
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    # ZYX intrinsic rotation (same as rotation_matrix_from_euler)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


##### Quaternion interpolation and rebasing #####
def interpolate_quaternions(quat, timestamps, new_timestamps):
    """
    Interpolate quaternion sequence to new timestamps.
    Is NaN-safe as it forward-fill missing quaternions.
    """
    quat_filled = quat.copy()

    # fill first frame if NaN
    if np.isnan(quat_filled[0]).any():
        quat_filled[0] = np.array([0, 0, 0, 1])  # identity quaternion

    # forward-fill remaining NaNs
    for i in range(1, len(quat_filled)):
        if np.isnan(quat_filled[i]).any():
            quat_filled[i] = quat_filled[i - 1]

    # Convert quaternions to Rotation objects
    r = R.from_quat(quat_filled)

    # Create SLERP object
    slerp = Slerp(timestamps, r)

    # Interpolate to new timestamps
    new_r = slerp(new_timestamps)

    return new_r.as_quat(), new_r.as_matrix()


def average_quats(q1, q2):
    # Convert to Rotation objects
    r1, r2 = R.from_quat(q1), R.from_quat(q2)

    # Times for SLERP
    times = [0, 1]
    rotations = R.concatenate([r1, r2])

    # Create SLERP object
    slerp = Slerp(times, rotations)

    # Evaluate at midpoint (0.5)
    r_avg = slerp([0.5])[0]

    # Average using slerp at 0.5 (midpoint)
    return r_avg.as_quat()


def accumulate_quaternions(delta_quats):
    quats = np.zeros_like(delta_quats)
    quats[0] = [0, 0, 0, 1]
    for i in range(1, len(delta_quats)):
        if np.any(np.isnan(delta_quats[i])):
            quats[i] = quats[i - 1]
        else:
            quats[i] = (
                R.from_quat(quats[i - 1]) * R.from_quat(delta_quats[i])
            ).as_quat()
    return quats


def rebase_quaternions_to_initial(quat):
    """
    Rebase quaternion sequence so that the first (non NaN) quaternion becomes identity.
    This makes all rotations relative to the initial orientation.
    Is NaN-safe as it forward-fill missing quaternions.
    """
    quat_filled = quat.copy()

    # Find first valid quaternion
    valid_idx = np.where(~np.isnan(quat_filled).any(axis=1))[0]
    if len(valid_idx) == 0:
        raise ValueError("All quaternions are NaN")

    first_valid_idx = valid_idx[0]
    first_valid_quat = quat_filled[first_valid_idx]

    # Fill all frames before first valid with first valid quaternion
    quat_filled[: first_valid_idx + 1] = first_valid_quat

    # forward-fill remaining NaNs
    for i in range(first_valid_idx + 1, len(quat_filled)):
        if np.isnan(quat_filled[i]).any():
            quat_filled[i] = quat_filled[i - 1]

    r = R.from_quat(quat_filled)
    r0_inv = r[0].inv()
    r_rebased = r0_inv * r  # equivalent to R_rel = R0.T * R
    return r_rebased.as_quat()


def rvec_to_quaternion(rvec):
    """Convert rotation vector (Rodrigues) to quaternion (x, y, z, w)."""
    rvec = np.atleast_1d(rvec).reshape(-1, 3)
    rotation = R.from_rotvec(rvec)
    return rotation.as_quat()  # returns in (x, y, z, w) format


def rotation_matrix_to_quaternion(R_mat):
    rotation = R.from_matrix(R_mat)
    return rotation.as_quat()  # returns in (x, y, z, w) format


def quat_error_angle_deg(quat_tag, quat_imu):
    """
    quat_tag, quat_imu: (N,4) arrays in (x,y,z,w) convention.
    Returns angle_error_deg (N,) the angle of the relative rotation q_tag * inv(q_imu).
    """
    R_tag = R.from_quat(quat_tag)
    R_imu = R.from_quat(quat_imu)
    R_rel = R_tag * R_imu.inv()  # rotation from imu -> tag
    rotvec = R_rel.as_rotvec()  # (N,3) rotation vector
    angle_rad = np.linalg.norm(rotvec, axis=1)
    return np.degrees(angle_rad)


#####  Fixed position on sphere from angles and vice versa #####
def fixed_position_from_angles(theta, phi, rad=1, radian=True):
    if radian:
        pos = (rad) * np.array(
            [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
        )
    else:
        pos = (rad) * np.array(
            [
                np.cos(theta * np.pi / 180) * np.sin(phi * np.pi / 180),
                np.sin(theta * np.pi / 180) * np.sin(phi * np.pi / 180),
                np.cos(phi * np.pi / 180),
            ]
        )

    return pos


def angles_from_fixed_position(pos):
    theta = (np.arctan2(pos[1], pos[0])) * 180 / np.pi
    phi = (np.arccos((pos[2]) / np.linalg.norm(pos, axis=0))) * 180 / np.pi

    return theta, phi


#####  Rotation matrices and point transformation #####
def rotation_matrix_from_euler(roll, pitch, yaw, degrees=True):
    """
    Create 3x3 rotation matrices from Euler angles (roll, pitch, yaw).

    Args:
        roll, pitch, yaw : array-like, shape (N,)
            Euler angles.
        degrees : bool
            Whether input angles are in degrees.

    Returns:
        R : ndarray, shape (N,3,3)
            Rotation matrices for each frame.
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)

    # Preallocate
    N = len(roll)
    R = np.zeros((N, 3, 3))

    # ZYX intrinsic rotation: R = Rz * Ry * Rx
    R[:, 0, 0] = c_y * c_p
    R[:, 0, 1] = c_y * s_p * s_r - s_y * c_r
    R[:, 0, 2] = c_y * s_p * c_r + s_y * s_r

    R[:, 1, 0] = s_y * c_p
    R[:, 1, 1] = s_y * s_p * s_r + c_y * c_r
    R[:, 1, 2] = s_y * s_p * c_r - c_y * s_r

    R[:, 2, 0] = -s_p
    R[:, 2, 1] = c_p * s_r
    R[:, 2, 2] = c_p * c_r

    return R


def apply_rotation(R, pB):
    """
    Apply N rotation matrices to a set of points.

    Args:
        R : ndarray, shape (N_frames,3,3)
            Rotation matrices.
        pB : ndarray, shape (N_frames,3)
            Points in sensor frame.

    Returns:
        pA : ndarray, shape (N_frames,3)
            Points in global frame.
    """
    nb_of_frames = R.shape[0]
    if pB.ndim == 1:
        pB = np.tile(pB, (nb_of_frames, 1))  # Repeat for each frame
    elif pB.shape[0] != nb_of_frames:
        raise ValueError("Number of frames in R and pB must match.")

    # R @ pB for each frame using einsum
    pA = np.einsum("nij,nj->ni", R, pB)
    return pA
