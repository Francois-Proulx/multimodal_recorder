import numpy as np


#####################################################
##################### 3D GRIDS ######################
#####################################################
def fibonacci_sphere(nb_of_points):
    """
    Generate points on a sphere using the Fibonacci lattice and compute distance statistics.

    Args:
        nb_of_points (int):
                Number of points to generate.

    Returns:
        spherical_grid (np.ndarray):
            Spherical grid of radius = 1 [nb_of_points, 3].
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    ga = 2 * np.pi * (1 - 1 / phi)  # Golden angle

    indices = np.arange(nb_of_points)
    theta = ga * indices
    phi = np.arccos(1 - 2 * (indices + 0.5) / nb_of_points)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.vstack((x, y, z)).T


def fibonacci_half_sphere(nb_of_points):
    """
    Generate points on a hemisphere using the Fibonacci lattice method.

    Args:
        nb_of_points (int):
            Number of points to generate on the hemisphere.

    Returns:
        hemisphere_points (np.ndarray):
            Array of shape [nb_of_points, 3] containing the Cartesian coordinates of the points.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    ga = 2 * np.pi * (1 - 1 / phi)  # Golden angle

    indices = np.arange(
        2 * nb_of_points
    )  # Generate more points to account for filtering
    theta = ga * indices
    phi = np.arccos(1 - 2 * (indices + 0.5) / (2 * nb_of_points))

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    # Stack and filter points to retain only those on the desired hemisphere
    points = np.vstack((x, y, z)).T
    hemisphere_points = points[
        z >= 0
    ]  # Adjust this condition based on the desired hemisphere

    # Select the required number of points
    return hemisphere_points[:nb_of_points]


def spherical_theta_phi_grid(nTheta, nPhi):
    """
    Create spherical grid in rectangular coordinates to scann for source localisation.
    Fallows typical spherical coordinates format :
        theta [0:2*pi] = angle in the xy plane, 0 on x axis, + towards y
        phi   [0:1*pi] = angle between r and z axis, 0 on z axis, + towards xy plane, pi on negative z axis

    In the case of the microphone array used for tests, the xy plane is the microphone plane and the drone is in negative z.

    Args:
        nTheta, nPhi (int):
            number of points for scan
    Returns:
        spherical_grid (np.ndarray):
            Spherical grid of radius = 1 [3, nTheta, nPhi].
    """
    # Meshgrid with theta and phi DOA_scan (direction of arrival) pour scan
    Theta, Phi = np.meshgrid(
        np.linspace(0, 360, nTheta), np.linspace(0, 180, nPhi), indexing="ij"
    )  # [nTheta x nPhi]

    # vecteur unitaire vers DOA_scan
    spherical_grid = np.squeeze(
        np.array(
            [
                [np.cos(Theta * np.pi / 180) * np.sin(Phi * np.pi / 180)],
                [np.sin(Theta * np.pi / 180) * np.sin(Phi * np.pi / 180)],
                [np.cos(Phi * np.pi / 180)],
            ]
        )
    )

    return spherical_grid


def sphere(levels_count=4):
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
    r = (2.0 / 5.0) * np.sqrt(5.0)

    pts = np.zeros((12, 3), dtype=float)
    pts[0, :] = [0, 0, 1]
    pts[11, :] = [0, 0, -1]
    pts[np.arange(1, 6, dtype=int), 0] = r * np.sin(2.0 * np.pi * np.arange(0, 5) / 5.0)
    pts[np.arange(1, 6, dtype=int), 1] = r * np.cos(2.0 * np.pi * np.arange(0, 5) / 5.0)
    pts[np.arange(1, 6, dtype=int), 2] = h
    pts[np.arange(6, 11, dtype=int), 0] = (
        -1.0 * r * np.sin(2.0 * np.pi * np.arange(0, 5) / 5.0)
    )
    pts[np.arange(6, 11, dtype=int), 1] = (
        -1.0 * r * np.cos(2.0 * np.pi * np.arange(0, 5) / 5.0)
    )
    pts[np.arange(6, 11, dtype=int), 2] = -1.0 * h

    # Generate triangles at level 0

    trs = np.zeros((20, 3), dtype=int)

    trs[0, :] = [0, 2, 1]
    trs[1, :] = [0, 3, 2]
    trs[2, :] = [0, 4, 3]
    trs[3, :] = [0, 5, 4]
    trs[4, :] = [0, 1, 5]

    trs[5, :] = [9, 1, 2]
    trs[6, :] = [10, 2, 3]
    trs[7, :] = [6, 3, 4]
    trs[8, :] = [7, 4, 5]
    trs[9, :] = [8, 5, 1]

    trs[10, :] = [4, 7, 6]
    trs[11, :] = [5, 8, 7]
    trs[12, :] = [1, 9, 8]
    trs[13, :] = [2, 10, 9]
    trs[14, :] = [3, 6, 10]

    trs[15, :] = [11, 6, 7]
    trs[16, :] = [11, 7, 8]
    trs[17, :] = [11, 8, 9]
    trs[18, :] = [11, 9, 10]
    trs[19, :] = [11, 10, 6]

    # Generate next levels

    for levels_index in range(0, levels_count):
        #      0
        #     / \
        #    A---B
        #   / \ / \
        #  1---C---2

        trs_count = trs.shape[0]
        subtrs_count = trs_count * 4

        subtrs = np.zeros((subtrs_count, 6), dtype=int)

        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 0] = trs[:, 0]
        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 1] = trs[:, 0]
        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 2] = trs[:, 0]
        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 3] = trs[:, 1]
        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 4] = trs[:, 2]
        subtrs[0 * trs_count + np.arange(0, trs_count, dtype=int), 5] = trs[:, 0]

        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 0] = trs[:, 0]
        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 1] = trs[:, 1]
        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 2] = trs[:, 1]
        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 3] = trs[:, 1]
        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 4] = trs[:, 1]
        subtrs[1 * trs_count + np.arange(0, trs_count, dtype=int), 5] = trs[:, 2]

        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 0] = trs[:, 2]
        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 1] = trs[:, 0]
        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 2] = trs[:, 1]
        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 3] = trs[:, 2]
        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 4] = trs[:, 2]
        subtrs[2 * trs_count + np.arange(0, trs_count, dtype=int), 5] = trs[:, 2]

        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 0] = trs[:, 0]
        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 1] = trs[:, 1]
        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 2] = trs[:, 1]
        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 3] = trs[:, 2]
        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 4] = trs[:, 2]
        subtrs[3 * trs_count + np.arange(0, trs_count, dtype=int), 5] = trs[:, 0]

        subtrs_flatten = np.concatenate(
            (subtrs[:, [0, 1]], subtrs[:, [2, 3]], subtrs[:, [4, 5]]), axis=0
        )
        subtrs_sorted = np.sort(subtrs_flatten, axis=1)

        unique_values, unique_indices, unique_inverse = np.unique(
            subtrs_sorted, return_index=True, return_inverse=True, axis=0
        )

        trs = np.transpose(np.reshape(unique_inverse, (3, -1)))

        pts = pts[unique_values[:, 0], :] + pts[unique_values[:, 1], :]
        pts /= np.repeat(
            np.expand_dims(np.sqrt(np.sum(np.power(pts, 2.0), axis=1)), axis=1),
            3,
            axis=1,
        )

    return pts


#####################################################
###### COMPUTE AVERAGE AREAS AND FILTER POINTS ######
#####################################################
def Voronoi_diagram(points):
    # Compute Voronoi diagram
    from scipy.spatial import SphericalVoronoi

    sv = SphericalVoronoi(points)

    # Calculate areas of Voronoi cells
    areas = sv.calculate_areas()

    # Compute statistics
    mean_area = np.mean(areas)
    max_area = np.max(areas)
    min_area = np.min(areas)

    # Approximation
    approx_area = 4 * np.pi / points.shape[0]
    print("Mean area :", mean_area, ", approximation area :", approx_area)
    print("Max area :", max_area, ", Min area :", min_area)

    lattice_delta_angle = np.arccos(1 - mean_area / 2 / np.pi) * 180 / np.pi
    print("lattice angle sensitivity :", lattice_delta_angle)

    return lattice_delta_angle


def filter_points_by_spherical_angles(points, theta_range, phi_range):
    """
    Filters points within specified spherical coordinate limits.

    Args:
        points (np.ndarray): Array of points with shape [nb_of_points, 3] in Cartesian coordinates (x, y, z).
        theta_range (tuple): Range for azimuthal angle θ in degrees as (theta_min, theta_max).
        phi_range (tuple): Range for polar angle φ in degrees as (phi_min, phi_max).

    Returns:
        np.ndarray: Filtered points within the specified spherical coordinate limits.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arctan2(y, x)) % 360  # Azimuthal angle in [0, 360)
    phi = np.degrees(np.arccos(z / r))  # Polar angle in [0, 180]

    # Define angle ranges
    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    # Create masks for the specified ranges
    theta_mask = (theta >= theta_min) & (theta <= theta_max)
    phi_mask = (phi >= phi_min) & (phi <= phi_max)

    # Combine masks
    mask = theta_mask & phi_mask

    # Apply mask to filter points
    filtered_points = points[mask]

    return filtered_points


def inter_mic_dist(mic_pos):
    nb_of_channels = mic_pos.shape[0]

    interMicDist = np.zeros((nb_of_channels, nb_of_channels, 3))
    for chan_id in range(nb_of_channels):
        interMicDist[chan_id, :, :] = mic_pos - mic_pos[chan_id, :]

    return interMicDist


#####################################################
##################### 2D GRIDS ######################
#####################################################
def circular_2D_grid(nTheta):
    """
    Create circular grid in rectangular coordinates to scann for source localisation.
    Fallows typical spherical coordinates format :
        theta [0:2*pi] = angle in the xy plane, 0 on x axis, + towards y

    In the case of the microphone array used for tests, the xy plane is the microphone plane and the drone is in negative z.

    Args:
        nTheta (int):
            number of points for scan
    Returns:
        2D grid (np.ndarray):
            Circular grid of radius = 1 [3, nTheta].
    """
    theta = np.linspace(0, 360, nTheta)

    # vecteur unitaire vers DOA_scan
    circular_grid = np.squeeze(
        np.array(
            [
                [np.cos(theta * np.pi / 180)],
                [np.sin(theta * np.pi / 180)],
                [np.zeros(theta.shape)],
            ]
        )
    )

    return circular_grid


#####################################################
################## COMPUTE TDOAS ####################
#####################################################
def calculate_tdoa(mic_pos, scan_grid, c=343):
    """
    Calculate tdoas based on microphone position and scanning grid.
    A TDOA of 0 is on the middle of the microphone array's coordinate system.
    A TDOA < 0 means that the sounds arrives BEFORE the center
    A TDOA > 0 means that the sounds arrives AFTER the center

    Args:
        mic_pos (np.ndarray):
            Microphone array position (x,y,z) [nb_of_channels, 3]
        scan_grid (np.ndarray):
            Spherical grid of radius = 1 [nb_of_doas, 3]
    Returns:
        TDOAs_scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas].
    """

    DDOAs_scan = mic_pos @ scan_grid.T  # [nb_of_channels, nb_of_doas]
    TDOAs_scan = -(DDOAs_scan / c)

    return TDOAs_scan
