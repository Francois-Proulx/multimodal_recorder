'''
Angle - Position Conversion
Generates trajectory
Array rotation
'''

import numpy as np

######################## Position Generation #######################
def random_position_in_room(room_dims, min_distance_from_wall):
    """
    Generate a random (x, y, z) position inside the room,
    at least min_distance_from_wall from each wall.
    Args:
        room_dims: array-like, [Lx, Ly, Lz]
        min_distance_from_wall: float
    Returns:
        pos: np.ndarray, shape (3,)
    """
    x = np.random.uniform(min_distance_from_wall, room_dims[0] - min_distance_from_wall)
    y = np.random.uniform(min_distance_from_wall, room_dims[1] - min_distance_from_wall)
    z = np.random.uniform(min_distance_from_wall, room_dims[2] - min_distance_from_wall)
    return np.array([x, y, z])


####################### Angle - Position Conversion #######################
def fixed_position_from_angles(theta, phi, room_radius):
    '''
    Get position from angles and room radius. The position is 1m from the side of the room.
    
    Args:
        theta (int):
            Azimuthal angle in degree (on the xy plane)
        phi (int):
            Elevation angle in degree (0 = floor, 180 = drone)
        room_radius (int):
            Radius of the room used for position calculation.

    Returns:
        pos (np.ndarray):
            Position (x,y,z) of the source [3, 1].

    '''
    room_center = np.array([room_radius, room_radius, room_radius])
    # room_center - [x,y,z] 
    pos = room_center + (room_radius-1)*np.array([np.cos(theta*np.pi/180)*np.sin(phi*np.pi/180),
                                                  np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180),
                                                  np.cos(phi*np.pi/180)])
    
    return pos

def angles_from_fixed_position(pos, room_center):
    theta = (np.atan2(pos[1]-room_center[1], pos[0]-room_center[0]))*180/np.pi
    phi = (np.acos((pos[2]-room_center[2])/np.linalg.norm(pos-room_center)))*180/np.pi

    return theta, phi


####################### Trajectory Generation #######################
def linear_trajectory(start_point, end_point, nb_of_points):
    '''
    Calculate linear trajectory poitns from starting and ending poitns as well as number of points in 3D.

    Args:
        start_point (np.ndarray):
            Starting point of linear trajectory [1, 3]
        end_point (np.ndarray):
            Ending point of linear trajectory [1, 3]
        nb_of_points (int):
            Number of points in linear trajectory.

    Returns:
        traj (np.ndarray):
            Linear trajectory [nb_of_points, 3].
    '''

    traj = np.zeros((nb_of_points,3))
    traj[:,0] = np.linspace(start_point[0],end_point[0],nb_of_points)
    traj[:,1] = np.linspace(start_point[1],end_point[1],nb_of_points)
    traj[:,2] = np.linspace(start_point[2],end_point[2],nb_of_points)

    print('Linear trajectory. Starts at ', start_point, ', ends at ', end_point)

    return traj


def circular_trajectory(theta_init, phi, room_radius, nb_of_points, nb_of_rotations=1):
    '''
    Calculate circular trajectory points in 3D (on the xy plane).
    Keel the same elevation angle (phi), but varies theta.

    Args:
        theta_init (int):
            Initial azimuthal angle in degree (Theta)
        phi (int):
            Elevation angle in degree (constant during trajectory)
        room_radius (int):
            Radius of the room used for position calculation.
        nb_of_points (int):
            Number of points in linear trajectory
        nb_of_rotations (int):
            Number of rotations during the trajectory.

    Returns:
        traj (np.ndarray):
            Linear trajectory [nb_of_points, 3].
    '''
    theta = np.linspace(theta_init,theta_init+360*nb_of_rotations,nb_of_points)

    traj = np.zeros((nb_of_points,3))
    for ii in range(nb_of_points):
        traj[ii,:] = fixed_position_from_angles(theta[ii], phi, room_radius)

    return traj


######################## Array Position and Rotation #######################
def get_microphone_position(mic_relative_pos, mic_array_center, mic_array_orientation=None):
    """
    Compute absolute microphone positions given relative positions, array center, and orientation.

    Args:
        mic_relative_pos (np.ndarray): [n_mics, 3] relative positions.
        mic_array_center (np.ndarray): [3] center position in room.
        mic_array_orientation (np.ndarray or None): [3, 3] rotation matrix. If None, no rotation.

    Returns:
        pos_rcv (np.ndarray): [n_mics, 3] absolute positions.
    """
    if mic_array_orientation is not None:
        # Rotate each mic position
        rotated = mic_relative_pos @ mic_array_orientation.T
    else:
        rotated = mic_relative_pos

    pos_rcv = rotated + mic_array_center
    return pos_rcv


def rotation_matrix_from_euler(pitch_deg, yaw_deg, roll_deg):
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    roll = np.deg2rad(roll_deg)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Combined rotation: R = Rz @ Ry @ Rx (yaw, then pitch, then roll)
    return Rz @ Ry @ Rx


def angles_from_array_reference(pos, array_center, array_rot_mat=None):
    """
    Compute DOA (theta, phi) from array reference frame.
    Args:
        pos: (3,) source position in room coordinates
        array_center: (3,) array center in room coordinates
        array_rot_mat: (3,3) array rotation matrix (room to array)
    Returns:
        theta, phi: angles in degrees (array frame)
    """
    if array_rot_mat is None:
        array_rot_mat = np.eye(3)
        
    vec = pos - array_center
    vec_local = array_rot_mat.T @ vec
    theta = np.degrees(np.atan2(vec_local[1], vec_local[0]))
    phi = np.degrees(np.arccos(vec_local[2] / np.linalg.norm(vec_local)))
    return theta, phi


def convert_angle(angle):
    """
    Convert an angle from the range [-180, 180] degrees to [0, 360] degrees.

    Args:
        angle (float): Angle in degrees within the range [-180, 180].

    Returns:
        float: Angle converted to the range [0, 360].
    """
    return angle % 360 if angle >= 0 else (angle + 360) % 360