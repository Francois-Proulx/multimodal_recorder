import numpy as np

def quaternion_to_euler_angle_vectorized(w, x, y, z):
    """
    Convert quaternions (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees.
    
    Parameters
    ----------
    w, x, y, z : array-like
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

def Rotation_matrix(alphas, betas, gammas):
    nb_of_samples = alphas.shape[0]
    alphas_rad = alphas/180*np.pi
    betas_rad = betas/180*np.pi
    gammas_rad = gammas/180*np.pi

    # Rz(alpha)
    Rz = np.zeros((nb_of_samples,4,4))
    Rz[:,0,0] = np.cos(alphas_rad)
    Rz[:,0,1] = -np.sin(alphas_rad)
    Rz[:,1,0] = np.sin(alphas_rad)
    Rz[:,1,1] = np.cos(alphas_rad)
    Rz[:,2,2] = np.linspace(1,1,nb_of_samples)
    Rz[:,3,3] = np.linspace(1,1,nb_of_samples)

    # Ry(beta)
    Ry = np.zeros((nb_of_samples,4,4))
    Ry[:,0,0] = np.cos(betas_rad)
    Ry[:,0,2] = np.sin(betas_rad)
    Ry[:,1,1] = np.linspace(1,1,nb_of_samples)
    Ry[:,2,0] = -np.sin(betas_rad)
    Ry[:,2,2] = np.cos(betas_rad)
    Ry[:,3,3] = np.linspace(1,1,nb_of_samples)

    # Rz(gamma)
    Rx = np.zeros((nb_of_samples,4,4))
    Rx[:,0,0] = np.linspace(1,1,nb_of_samples)
    Rx[:,1,1] = np.cos(gammas_rad)
    Rx[:,1,2] = -np.sin(gammas_rad)
    Rx[:,2,1] = np.sin(gammas_rad)
    Rx[:,2,2] = np.cos(gammas_rad)
    Rx[:,3,3] = np.linspace(1,1,nb_of_samples)

    # R
    R = Rz@Ry@Rx

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
