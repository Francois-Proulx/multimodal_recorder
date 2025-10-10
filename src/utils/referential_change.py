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
    cy = np.cos(yaw*0.5)
    sy = np.sin(yaw*0.5)
    cp = np.cos(pitch*0.5)
    sp = np.sin(pitch*0.5)
    cr = np.cos(roll*0.5)
    sr = np.sin(roll*0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return x, y, z, w


##### Quaternion interpolation and rebasing #####
def interpolate_quaternions(quat, timestamps, new_timestamps):
    """Interpolate quaternion sequence to new timestamps."""
    # Convert quaternions to Rotation objects
    r = R.from_quat(quat)
    
    # Create SLERP object
    slerp = Slerp(timestamps, r)
    
    # Interpolate to new timestamps
    new_r = slerp(new_timestamps)
    
    return new_r.as_quat(), new_r.as_matrix()


def rebase_quaternions_to_initial(quat):
    """
    Rebase quaternion sequence so that the first quaternion becomes identity.
    This makes all rotations relative to the initial orientation.
    """
    r = R.from_quat(quat)
    r0_inv = r[0].inv()
    r_rebased = r0_inv * r  # equivalent to R_rel = R0.T * R
    return r_rebased.as_quat()


#####  Fixed position on sphere from angles and vice versa #####
def fixed_position_from_angles(theta, phi, rad=1, radian=True):
    if radian:
        pos = (rad)*np.array([np.cos(theta)*np.sin(phi),
                              np.sin(theta)*np.sin(phi),
                              np.cos(phi)])
    else:
        pos = (rad)*np.array([np.cos(theta*np.pi/180)*np.sin(phi*np.pi/180),
                              np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180),
                              np.cos(phi*np.pi/180)])
    
    return pos


def angles_from_fixed_position(pos):
    theta = (np.arctan2(pos[1], pos[0]))*180/np.pi
    phi = (np.arccos((pos[2])/np.linalg.norm(pos,axis=0)))*180/np.pi

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
    R = np.zeros((N,3,3))
    
    # ZYX intrinsic rotation: R = Rz * Ry * Rx
    R[:,0,0] = c_y * c_p
    R[:,0,1] = c_y * s_p * s_r - s_y * c_r
    R[:,0,2] = c_y * s_p * c_r + s_y * s_r

    R[:,1,0] = s_y * c_p
    R[:,1,1] = s_y * s_p * s_r + c_y * c_r
    R[:,1,2] = s_y * s_p * c_r - c_y * s_r

    R[:,2,0] = -s_p
    R[:,2,1] = c_p * s_r
    R[:,2,2] = c_p * c_r

    return R



def apply_rotation(R, pB):
    """
    Apply N rotation matrices to a set of points.
    
    Args:
        R : ndarray, shape (N_frames,3,3)
            Rotation matrices.
        pB : ndarray, shape (3,N_points)
            Points in sensor frame.
    
    Returns:
        pA : ndarray, shape (N_frames,3,N_points)
            Points in global frame.
    """
    # R @ pB for each frame using einsum
    return np.einsum('nij,jk->nik', R, pB)
