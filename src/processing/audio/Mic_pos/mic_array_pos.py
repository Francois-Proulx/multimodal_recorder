# Calulate microphone array position and save
import numpy as np

#####################################################################
#######################     CUSTOM ARRAYS     #######################
#####################################################################
##  MIC ARRAY 16 mics V1 
def half_sphere_array_pos_16mics(radius=0.082, save_name="half_sphere_array_pos_16mics"):
    """
    Generate positions for a 16-mic half-sphere array:
    - 4 mics on a ring at 30° elevation
    - 12 mics on a ring at 55° elevation
    
    * The origin of the coordinate system is at the center of the circle, on the plane of the 7 mics.
    * The z axis points downwards, so drone z<0.

    Args:
        radius (float): Sphere radius in meters.
        save_name (str): File name for saving positions.

    Returns:
        np.ndarray: Array of shape (16, 3) with microphone positions.
    """
    # Top ring: 4 mics at 30°
    top_mic_theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
    top_mic_phi = np.deg2rad(30)
    top_mic_pos_x = radius * np.sin(top_mic_phi) * np.cos(top_mic_theta)
    top_mic_pos_y = radius * np.sin(top_mic_phi) * np.sin(top_mic_theta)
    top_mic_pos_z = np.full(4, radius * np.cos(top_mic_phi))

    # Bottom ring: 12 mics at 55°
    bot_mic_theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    bot_mic_phi = np.deg2rad(55)
    bot_mic_pos_x = radius * np.sin(bot_mic_phi) * np.cos(bot_mic_theta)
    bot_mic_pos_y = radius * np.sin(bot_mic_phi) * np.sin(bot_mic_theta)
    bot_mic_pos_z = np.full(12, radius * np.cos(bot_mic_phi))

    # Stack positions
    top_mic_pos = np.column_stack((top_mic_pos_x, top_mic_pos_y, top_mic_pos_z))  # (4, 3)
    bot_mic_pos = np.column_stack((bot_mic_pos_x, bot_mic_pos_y, bot_mic_pos_z))  # (12, 3)
    mic_pos = np.vstack((top_mic_pos, bot_mic_pos))  # (16, 3)

    # Save
    if save_name:
        np.save(f"Mic_pos/{save_name}.npy", mic_pos)
        print(f"Saved to 'Mic_pos/{save_name}.npy'")

    return mic_pos

def half_sphere_array_pos_16mics_clockwise(radius=0.082, save_name="half_sphere_array_pos_16mics"):
    """
    Generate positions for a 16-mic half-sphere array:
    - 4 mics on a ring at 30° elevation
    - 12 mics on a ring at 55° elevation
    
    * The origin of the coordinate system is at the center of the circle, on the plane of the 7 mics.
    * The z axis points downwards, so drone z<0.

    Args:
        radius (float): Sphere radius in meters.
        save_name (str): File name for saving positions.

    Returns:
        np.ndarray: Array of shape (16, 3) with microphone positions.
    """
    # Top ring: 4 mics at 30°
    top_mic_theta = np.linspace(0, -2*np.pi, 4, endpoint=False)
    top_mic_phi = np.deg2rad(30)
    top_mic_pos_x = radius * np.sin(top_mic_phi) * np.cos(top_mic_theta)
    top_mic_pos_y = radius * np.sin(top_mic_phi) * np.sin(top_mic_theta)
    top_mic_pos_z = np.full(4, radius * np.cos(top_mic_phi))

    # Bottom ring: 12 mics at 55°
    bot_mic_theta = np.linspace(0, -2*np.pi, 12, endpoint=False)
    bot_mic_phi = np.deg2rad(55)
    bot_mic_pos_x = radius * np.sin(bot_mic_phi) * np.cos(bot_mic_theta)
    bot_mic_pos_y = radius * np.sin(bot_mic_phi) * np.sin(bot_mic_theta)
    bot_mic_pos_z = np.full(12, radius * np.cos(bot_mic_phi))

    # Stack positions
    top_mic_pos = np.column_stack((top_mic_pos_x, top_mic_pos_y, top_mic_pos_z))  # (4, 3)
    bot_mic_pos = np.column_stack((bot_mic_pos_x, bot_mic_pos_y, bot_mic_pos_z))  # (12, 3)
    mic_pos = np.vstack((top_mic_pos, bot_mic_pos))  # (16, 3)

    # Save
    if save_name:
        np.save(f"{save_name}.npy", mic_pos)
        print(f"Saved to '{save_name}.npy'")

    return mic_pos
half_sphere_array_pos_16mics_clockwise(radius=0.082, save_name="half_sphere_array_pos_16mics_clockwise")

####################### MIC ARRAY V1 ########################
def drone_8mic_array_v1_pos(radius=0.066548, z_center=0.0323, save=True):
    """
    Generate positions for the custom drone 8-mic array:
    - 7 mics on a circle in the XY plane (z=0)
    - 1 mic above the center (z=z_center>0)
    
    * The origin of the coordinate system is at the center of the circle, on the plane of the 7 mics.
    * The z axis points downwards, so drone z<0.
    
    # Mic 1 : [-64.88, 14.81, -32.33]
	# Mic 2 : [-28.87, 59.96, -32.33]
	# Mic 3 : [ 28.87, 59.96, -32.33]
	# Mic 4 : [ 64.88, 14.80, -32.33]
	# Mic 5 : [ 52.02,-41.50, -32.33]
	# Mic 6 : [ 00.00,-65.56, -32.33]
	# Mic 7 : [-52.03,-41.49, -32.33]
	# Mic 8 : [ 00.00, 00.00, -64.67]

    Args:
        radius (float): Radius of the circle in meters.
        z_center (float): Height of the 8th mic above the plane in meters.
        save (bool): If True, saves the positions to a .npy file.

    Returns:
        np.ndarray: Array of shape (8, 3) with microphone positions.
    """
    # 7 mics on the circle
    angles_deg = np.linspace(0, 360, 8)[:-1]  # 7 angles, exclude 360
    angles_rad = np.deg2rad(angles_deg)
    mic_x_pos = np.cos(angles_rad) * radius
    mic_y_pos = np.sin(angles_rad) * radius
    mic_z_pos = np.zeros(7)

    # 8th mic above the center
    mic_x_pos = np.append(mic_x_pos, 0)
    mic_y_pos = np.append(mic_y_pos, 0)
    mic_z_pos = np.append(mic_z_pos, z_center)

    mic_pos = np.column_stack((mic_x_pos, mic_y_pos, mic_z_pos))  # nMic x 3

    if save:
        np.save('Mic_pos/drone_8mic_array_v1_pos.npy', mic_pos)
        print("Microphone positions for drone 8-mic array saved to 'Mic_pos/drone_8mic_array_v1_pos.npy'")

    return mic_pos


#####################################################################
#########################     2D ARRAYS     #########################
#####################################################################

def linear_array_pos(M, d, save=True):
	"""
	Generate positions for a linear microphone array.

	Args:
		M (int): Number of microphones.
		d (float): Distance between microphones in meters.

	Returns:
		np.ndarray: Array of shape (M, 3) with microphone positions.
	"""
	min_val = M//2 * d
	mic_x_pos = np.linspace(-min_val, min_val, M)
	mic_y_pos = np.zeros(M)
	mic_z_pos = np.zeros(M)
	
	mic_pos = np.column_stack((mic_x_pos, mic_y_pos, mic_z_pos))  # nMic x 3
	
	if save:
		mic_array_name = 'linear_array_pos_' + str(M) + 'mics_d' + str(d) + '.npy'
		np.save(mic_array_name, mic_pos)
		print(f"Microphone positions saved to {mic_array_name}")
  
	return mic_pos


def grid_2d_array_pos(M, spacing=0.05, z=0.0, center=True, save_name=None):
    """
    Generate a 2D MxM microphone grid in the XY plane.

    Args:
        M (int): Number of microphones per row/column (M x M total).
        spacing (float): Distance between microphones (in meters).
        z (float): Z-axis value (default 0.0, flat grid).
        center (bool): Center the array around (0, 0), else starts at origin.
        save_name (str): If given, saves to 'Mic_pos/{save_name}.npy'.

    Returns:
        np.ndarray: Array of shape (M*M, 3) with microphone positions.
    """
    x = np.arange(M) * spacing
    y = np.arange(M) * spacing
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=-1)

    if center:
        center_offset = spacing * (M - 1) / 2
        positions[:, 0] -= center_offset
        positions[:, 1] -= center_offset
        
    N = positions.shape[0]

    if save_name:
        file_name = f"{save_name}_MxM_N{N}_S{spacing:.3f}.npy"
        np.save(f"Mic_pos/for_comparison/{file_name}", positions)
        print(f"Saved to 'Mic_pos/{file_name}'")

    return positions

# grid_2d_array_pos(M=4, spacing=0.042, z=0.0, center=True, save_name="grid_2d_array_pos_4x4_minidps")

def respeaker_6mic_array_pos(save=True):
	"""
	Generate positions for the ReSpeaker 6-mic array.

	Returns:
		np.ndarray: Array of shape (6, 3) with microphone positions.
	"""
	Mic_1 = np.array([-0.0232, +0.0401, +0.0000])
	Mic_2 = np.array([-0.0463, +0.0000, +0.0000])
	Mic_3 = np.array([-0.0232, -0.0401, +0.0000])
	Mic_4 = np.array([+0.0232, -0.0401, +0.0000])
	Mic_5 = np.array([+0.0463, +0.0000, +0.0000])
	Mic_6 = np.array([+0.0232, +0.0401, +0.0000])

	mic_pos = np.stack((Mic_1, Mic_2, Mic_3, Mic_4, Mic_5, Mic_6))  # nMic x 3
 
	if save:
		np.save('Mic_pos/respeaker_6mic_array_pos.npy', mic_pos)
		print("Microphone positions for ReSpeaker 6-mic array saved to 'Mic_pos/respeaker_6mic_array_pos.npy'")
  
	return mic_pos


def respeaker_7mic_array_pos(radius=0.0463, save=True):
    """
    Generate positions for the ReSpeaker 7-mic circular array.

    Args:
        radius (float): Radius of the circular mic ring in meters.
        save (bool): If True, save the positions to a .npy file.

    Returns:
        np.ndarray: Array of shape (7, 3) with microphone positions.
    """
    mic_pos = []

    # 6 mics on the circle (XY plane)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        mic_pos.append([x, y, 0.0])

    # Center mic
    mic_pos.append([0.0, 0.0, 0.0])

    mic_pos = np.array(mic_pos)

    if save:
        np.save('Mic_pos/for_comparison/respeaker_7mic_array_pos.npy', mic_pos)
        print("Saved to 'Mic_pos/respeaker_7mic_array_pos.npy'")

    return mic_pos

# respeaker_7mic_array_pos(save=True)


def circular_array_pos(nb_mics, radius=None, spacing=None, add_center=False, save_name=None):
    """
    Generate a 2D circular microphone array in the XY plane.

    Args:
        nb_mics (int): Number of microphones (excluding center if add_center=True).
        radius (float): Radius of the circle (in meters). Optional if spacing is given.
        spacing (float): Desired direct (chord) distance between adjacent microphones (in meters). Overrides radius.
        add_center (bool): Whether to include a microphone at the center.
        save_name (str): If given, saves to 'Mic_pos/{save_name}_N{N}_R{radius:.3f}_S{spacing:.3f}.npy'.

    Returns:
        np.ndarray: Array of shape (N, 3) with microphone positions.
    """
    # If spacing is given, compute radius from chord length
    if spacing is not None:
        angle_between_mics = 2 * np.pi / nb_mics
        radius = spacing / (2 * np.sin(angle_between_mics / 2))

    if radius is None:
        raise ValueError("You must provide either 'radius' or 'spacing'.")

    mic_pos = []
    for i in range(nb_mics):
        angle = 2 * np.pi * i / nb_mics
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        mic_pos.append([x, y, 0.0])  # All in XY plane

    if add_center:
        mic_pos.append([0.0, 0.0, 0.0])

    mic_pos = np.array(mic_pos)
    N = mic_pos.shape[0]

    # Compute actual direct (chord) spacing between adjacent mics
    if nb_mics > 1:
        chord_spacing = np.linalg.norm(mic_pos[0] - mic_pos[1])
    else:
        chord_spacing = 0.0

    if save_name:
        file_name = f"{save_name}_N{N}_R{radius:.3f}_S{chord_spacing:.3f}.npy"
        np.save(f"Mic_pos/for_comparison/{file_name}", mic_pos)
        print(f"Saved to 'Mic_pos/{file_name}'")

    return mic_pos

# circular_array_pos(nb_mics=7, spacing=0.15, add_center=True, save_name="circular_array_with_center")

def co_prime_circular_array_pos(M, N=None, spacing=None, radius=None, add_center=False, save_name=None):
    """
    Generate microphone positions for a 2D co-prime circular array in the XY plane.

    Args:
        M (int): Number of microphones in first subset (first circle).
        N (int, optional): Number of microphones in second subset (second circle). 
                           Defaults to M+1 if None.
        spacing (float, optional): Desired minimum distance between microphones (chord length).
                                   If given, radius is computed from spacing.
        radius (float, optional): Radius of the circle. If None and spacing is None, raises error.
        add_center (bool): Whether to include a microphone at the center (0,0,0).
        save_name (str, optional): If given, saves to 'Mic_pos/{save_name}_M{M}_N{N}_R{radius:.3f}_S{spacing:.3f}.npy'.

    Returns:
        np.ndarray: Array of shape (num_mics, 3) with microphone positions.
    """

    if N is None:
        N = M + 1

    # Compute all unique angles on unit circle
    angles_M = np.array([2 * np.pi * i / M for i in range(M)])
    angles_N = np.array([2 * np.pi * j / N for j in range(N)])
    all_angles = np.unique(np.concatenate((angles_M, angles_N)))
    all_angles = np.sort(all_angles)

    # Compute minimum angular difference on circle
    diffs = np.diff(np.concatenate((all_angles, [all_angles[0] + 2 * np.pi])))
    min_angle = np.min(diffs)

    # If spacing given, compute radius to get desired minimum spacing
    if spacing is not None:
        radius = spacing / (2 * np.sin(min_angle / 2))
    elif radius is None:
        raise ValueError("You must provide either 'spacing' or 'radius'.")

    # Compute positions on circle of computed radius
    mic_pos = []
    for angle in all_angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        mic_pos.append([x, y, 0.0])

    if add_center:
        mic_pos.append([0.0, 0.0, 0.0])

    mic_pos = np.array(mic_pos)
    num_mics = mic_pos.shape[0]

    # Compute actual minimum chord spacing between adjacent mics (optional)
    chord_distances = []
    for i in range(num_mics):
        for j in range(i+1, num_mics):
            d = np.linalg.norm(mic_pos[i] - mic_pos[j])
            chord_distances.append(d)
    actual_min_spacing = np.min(chord_distances)

    if save_name:
        file_name = f"{save_name}_N{num_mics}_R{radius:.3f}_S{actual_min_spacing:.3f}.npy"
        np.save('Mic_pos/for_comparison/' + file_name, mic_pos)
        print(f"Saved to 'Mic_pos/{file_name}'")

    return mic_pos

# co_prime_circular_array_pos(M=8, spacing=0.05, save_name='co_prime_circular_array')

#####################################################################
#########################     3D ARRAYS     #########################
#####################################################################

# def spherical_array_pos(nb_mics, radius=None, spacing=None, add_center=False, save_name=None):
#     """
#     Generate a spherical microphone array using Fibonacci distribution.

#     Args:
#         nb_mics (int): Number of microphones (excluding center mic if add_center=True).
#         radius (float): Radius of the sphere (in meters).
#         spacing (float): Approximate spacing between microphones (in meters). Overrides `radius` if given.
#         add_center (bool): Whether to include a microphone at the center.
#         save_name (str): If provided, saves to 'Mic_pos/{save_name}_N{nb_mics}_R{radius:.3f}_S{spacing:.3f}.npy'.

#     Returns:
#         np.ndarray: Array of shape (N, 3) with microphone positions.
#     """
#     if spacing is not None:
#         # Estimate radius from surface area needed
#         area_per_mic = spacing**2
#         surface_area = nb_mics * area_per_mic
#         radius = np.sqrt(surface_area / (4 * np.pi))

#     if radius is None:
#         raise ValueError("You must provide either a 'radius' or 'spacing'")

#     points = []
#     golden_angle = np.pi * (3 - np.sqrt(5))

#     for i in range(nb_mics):
#         z = 1 - 2 * i / (nb_mics - 1)
#         theta = golden_angle * i
#         r_xy = np.sqrt(1 - z * z)
#         x = r_xy * np.cos(theta)
#         y = r_xy * np.sin(theta)
#         points.append([x * radius, y * radius, z * radius])

#     if add_center:
#         points.append([0.0, 0.0, 0.0])

#     mic_pos = np.array(points)

#     # Compute average spacing between microphones
#     from scipy.spatial.distance import pdist
#     dists = pdist(mic_pos)
#     avg_spacing = np.mean(dists)

#     if save_name:
#         file_name = f"{save_name}_N{mic_pos.shape[0]}_R{radius:.3f}_S{avg_spacing:.3f}.npy"
#         np.save(f'Mic_pos/{file_name}', mic_pos)
#         print(f"Saved to 'Mic_pos/{file_name}'")

#     return mic_pos

# spherical_array_pos(nb_mics=16, radius=0.05, save_name='spherical_array')

def zylia_19mic_array_pos(radius=0.0515, save=True):
    """
    Generate positions for the Zylia 19-mic array.

    Args:
        radius (float): Radius of the spherical array in meters.
        save (bool): If True, saves the positions to a .npy file.

    Returns:
        np.ndarray: Array of shape (19, 3) with microphone positions.
    """
    mic_positions = []

    # Top mic (north pole)
    mic_positions.append([0, 0, radius])

    # Define rings
    ring_thetas_deg = [55, 90, 125]  # approximate elevation angles from +Z
    num_per_ring = 6

    for theta_deg in ring_thetas_deg:
        theta_rad = np.deg2rad(theta_deg)
        for i in range(num_per_ring):
            phi_rad = 2 * np.pi * i / num_per_ring
            x = radius * np.sin(theta_rad) * np.cos(phi_rad)
            y = radius * np.sin(theta_rad) * np.sin(phi_rad)
            z = radius * np.cos(theta_rad)
            mic_positions.append([x, y, z])

    # Bottom mic (south pole)
    mic_positions.append([0, 0, -radius])

    mic_pos = np.array(mic_positions)  # nMic x 3

    if save:
        np.save('Mic_pos/for_comparison/zylia_19mic_array_pos.npy', mic_pos)
        print("Microphone positions for Zylia 19-mic array saved to 'Mic_pos/zylia_19mic_array_pos.npy'")

    return mic_pos

# zylia_19mic_array_pos(save=False)

def cube_array_pos(include_face_centers=False, spacing=0.1, save_name=None):
    """
    Generate a 3D cube microphone array with 8 mics on corners.
    Optionally includes 6 mics at face centers (total 14).

    Args:
        include_face_centers (bool): Add 6 face-centered microphones.
        spacing (float): Side length of the cube (in meters).
        save_name (str): If given, saves to 'Mic_pos/{save_name}.npy'.

    Returns:
        np.ndarray: Array of shape (8 or 14, 3) with mic positions.
    """
    d = spacing / 2.0
    # 8 corners of the cube
    corners = np.array([[x, y, z]
                        for x in [-d, d]
                        for y in [-d, d]
                        for z in [-d, d]])

    positions = [*corners]

    if include_face_centers:
        faces = [
            [0, 0, +d],
            [0, 0, -d],
            [0, +d, 0],
            [0, -d, 0],
            [+d, 0, 0],
            [-d, 0, 0]
        ]
        positions.extend(faces)
        
        spacing = spacing / np.sqrt(2)  # Adjust spacing for face centers

    mic_pos = np.array(positions)

    if save_name:
        N = mic_pos.shape[0]
        file_name = f"{save_name}_cube_N{N}_S{spacing:.3f}.npy"
        np.save(f"Mic_pos/for_comparison/{file_name}", mic_pos)
        print(f"Saved to 'Mic_pos/{file_name}'")

    return mic_pos

# cube_array_pos(spacing=0.05, save_name="cube_array_pos_8mics")

def sphere_with_icosahedrons(levels_count=4):
	"""
	Definition of a spherical space (3D) for localization

	Args:
		levels_count (scalar):
			The number of levels to refine the icosahedron.

	Returns:
		(np.ndarray):
			The sphere (nb_of_points, 3).
	"""

	# Generate points at level 0

	h = np.sqrt(5.0) / 5.0
	r = (2.0/5.0) * np.sqrt(5.0)

	pts = np.zeros((12,3), dtype=float)
	pts[0,:] = [0,0,1]
	pts[11,:] = [0,0,-1]
	pts[np.arange(1,6,dtype=int),0] = r * np.sin(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(1,6,dtype=int),1] = r * np.cos(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(1,6,dtype=int),2] = h
	pts[np.arange(6,11,dtype=int),0] = -1.0 * r * np.sin(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(6,11,dtype=int),1] = -1.0 * r * np.cos(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(6,11,dtype=int),2] = -1.0 * h

	# Generate triangles at level 0

	trs = np.zeros((20,3), dtype=int)

	trs[0,:] = [0,2,1]
	trs[1,:] = [0,3,2]
	trs[2,:] = [0,4,3]
	trs[3,:] = [0,5,4]
	trs[4,:] = [0,1,5]

	trs[5,:] = [9,1,2]
	trs[6,:] = [10,2,3]
	trs[7,:] = [6,3,4]
	trs[8,:] = [7,4,5]
	trs[9,:] = [8,5,1]
	
	trs[10,:] = [4,7,6]
	trs[11,:] = [5,8,7]
	trs[12,:] = [1,9,8]
	trs[13,:] = [2,10,9]
	trs[14,:] = [3,6,10]
	
	trs[15,:] = [11,6,7]
	trs[16,:] = [11,7,8]
	trs[17,:] = [11,8,9]
	trs[18,:] = [11,9,10]
	trs[19,:] = [11,10,6]

	# Generate next levels

	for levels_index in range(0, levels_count):

		#      0
		#     / \
		#    A---B
		#   / \ / \
		#  1---C---2

		trs_count = trs.shape[0]
		subtrs_count = trs_count * 4

		subtrs = np.zeros((subtrs_count,6), dtype=int)

		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,1]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,0]

		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,2]

		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,0]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,2]

		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,1]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,2]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,0]

		subtrs_flatten = np.concatenate((subtrs[:,[0,1]], subtrs[:,[2,3]], subtrs[:,[4,5]]), axis=0)
		subtrs_sorted = np.sort(subtrs_flatten, axis=1)

		unique_values, unique_indices, unique_inverse = np.unique(subtrs_sorted, return_index=True, return_inverse=True, axis=0)

		trs = np.transpose(np.reshape(unique_inverse, (3,-1)))

		pts = pts[unique_values[:,0],:] + pts[unique_values[:,1],:]
		pts /= np.repeat(np.expand_dims(np.sqrt(np.sum(np.power(pts,2.0), axis=1)), axis=1), 3, axis=1)

	return pts

