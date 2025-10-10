import numpy as np



################## Localisation Metrics #########################
def total_angular_error(target, method, target_filter=None):
    """
    Calculate the total angular error between target and measured angles.

    ** Possibility to add a mask if we dont want to visualize some frames

    Args:
        target (np.ndarray): 
            Array of target doas (theta,phi) for each frame [nb_of_frames, 2]
        method (np.ndarray): 
            Array of method doas (theta,phi) for each frame [nb_of_frames, 2]

    Returns:
        error (np.ndarray):
            Total angular errors in degrees for each frame [nb_of_frames,1].
    """
    # Convert degrees to radians
    theta_target, phi_target = np.radians(target[:, 0]), np.radians(target[:, 1])
    theta_method, phi_method = np.radians(method[:, 0]), np.radians(method[:, 1])

    # Convert spherical coordinates to Cartesian coordinates on the unit sphere
    x_target = np.sin(phi_target) * np.cos(theta_target)
    y_target = np.sin(phi_target) * np.sin(theta_target)
    z_target = np.cos(phi_target)

    x_method = np.sin(phi_method) * np.cos(theta_method)
    y_method = np.sin(phi_method) * np.sin(theta_method)
    z_method = np.cos(phi_method)

    # Compute the dot product between target and method vectors
    dot_product = x_target * x_method + y_target * y_method + z_target * z_method

    # Ensure the dot product values are within the valid range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angular error in radians and convert to degrees
    total_error = np.degrees(np.arccos(dot_product))

    if target_filter is not None:
        # Flatten filter if needed
        target_filter = np.asarray(target_filter).squeeze()
        total_error = total_error[target_filter.astype(bool)]
        
    return total_error


def get_validation_threshold(lattice_delta_angle, leakage_parameter=1.5):
    average_angle_between_points = 2 * lattice_delta_angle  # in degrees
    threshold = leakage_parameter * average_angle_between_points
    return threshold


def compute_good_localisation_percentage(angular_errors, angle_thresholds, target_filter=None):
    """
    Compute the percentage of good localisation where the angular error is less than 2 times the average distance between 2 points.

    Args:
        angular_errors (np.ndarray): 
            Array of angular errors in degrees [nb_of_frames].
        angle_thresholds (float or np.ndarray): 
            Angular threshold(s) for good localisation.
        target_filter (np.ndarray or None):
            Optional boolean mask for valid frames.

    Returns:
        percentage (float): 
            Percentage of good hits.
    """
    
    if target_filter is not None:
        valid_mask = target_filter & ~np.isnan(angular_errors)
        errors_to_check = angular_errors[valid_mask]
    else:
        errors_to_check = angular_errors[~np.isnan(angular_errors)]

    # Handle scalar or array thresholds
    angle_thresholds = np.atleast_1d(angle_thresholds)
    percentages = np.empty(angle_thresholds.shape, dtype=np.float32)

    for i, threshold in enumerate(angle_thresholds):
        good_hits = errors_to_check < threshold
        if errors_to_check.size > 0:
            percentages[i] = 100 * np.sum(good_hits) / errors_to_check.size
        else:
            percentages[i] = np.nan
            print('No valid frames (or all NaNs), percentage is nan')

    # Return scalar if input was scalar, else array
    if percentages.size == 1:
        return percentages[0]
    else:
        return percentages


################## Speech Enhancement Metrics ###################
def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf

    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])

    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    # assert reference.dtype == np.float64, reference.dtype
    # assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)
