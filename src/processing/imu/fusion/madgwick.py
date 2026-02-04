import numpy as np
from ahrs.filters import Madgwick


def run_madgwick_filter(timestamps, acc, gyro, mag=None, beta=0.033):
    """
    Runs Madgwick AHRS using the exact frequency calculated from timestamps.

    Args:
        timestamps: (N,) array of time in seconds
        acc:  (N, 3) array in m/s^2
        gyro: (N, 3) array in RAD/s (Must be Radians!)
        mag:  (N, 3) array in uT (Optional)
        beta: Filter gain (0.033 is standard, try 0.01 for smoother/less drift)

    Returns:
        Q: (N, 4) array of Quaternions [w, x, y, z] (Scalar First)
    """
    # 1. Calculate Sampling Frequency
    dt_array = np.diff(timestamps)
    mean_dt = np.mean(dt_array)
    computed_freq = 1.0 / mean_dt
    print(f"Computed Sampling Frequency: {computed_freq:.2f} Hz")

    # Check for Jitter (Variance in time steps)
    jitter = np.std(dt_array)
    if jitter > 0.005:  # If jitter is > 5ms
        print(
            f"WARNING: High Jitter detected (std={jitter:.4f}s). Results may be noisy."
        )

    # 2. Initialize Filter
    # We use the computed frequency for the filter parameters
    madgwick = Madgwick(frequency=computed_freq, gain=beta)

    # 3. Gravity alignment: Initial quaternion from first accelerometer reading
    Q = np.zeros((len(acc), 4))
    a0 = acc[0]
    norm_a0 = np.linalg.norm(a0)
    if norm_a0 == 0:
        # Fallback if sensor dead
        Q[0] = [1.0, 0.0, 0.0, 0.0]  # Initial quaternion [w, x, y, z]
    else:
        # Normalize
        a_norm = a0 / norm_a0

        # World Up vector (Standard convention: Gravity points DOWN, so Sensor sees UP)
        # Note: Check your convention. Usually Accel measures "Normal Force" (UP).
        v_down = np.array([0.0, 0.0, 1.0])

        # Calculate axis-angle to rotate a_norm to v_down
        # Formula: Rotation axis = cross(a, v_down), Angle = arccos(dot(a, v_down))

        c = np.dot(a_norm, v_down)  # Cosine of angle

        # If already aligned (c ~ 1) or opposite (c ~ -1)
        if c > 0.9999:
            Q[0] = [1.0, 0.0, 0.0, 0.0]
        elif c < -0.9999:
            # Upside down - quaternion for 180 deg rotation around X
            Q[0] = [0.0, 1.0, 0.0, 0.0]
        else:
            axis = np.cross(a_norm, v_down)
            axis = axis / np.linalg.norm(axis)
            # Half-angle formula for quaternion
            # q = [cos(theta/2), sin(theta/2)*axis]
            # using trig identity: cos(theta/2) = sqrt((1+c)/2), sin(theta/2) = sqrt((1-c)/2)

            w = np.sqrt((1 + c) * 0.5)
            s = np.sqrt((1 - c) * 0.5)

            # Scipy/Madgwick format is usually Scalar First [w, x, y, z]
            Q[0] = [w, s * axis[0], s * axis[1], s * axis[2]]
    print(f"Initial Quaternion (Gravity Aligned): {Q[0]}")
    # 4. Run Loop
    for t in range(1, len(acc)):
        # Optional: If you have massive jitter, you can force the exact dt
        # by updating madgwick.Dt for every sample, but mean is usually fine.
        # madgwick.Dt = dt_array[t-1]

        if mag is not None:
            Q[t] = madgwick.updateMARG(Q[t - 1], gyr=gyro[t], acc=acc[t], mag=mag[t])
        else:
            Q[t] = madgwick.updateIMU(Q[t - 1], gyr=gyro[t], acc=acc[t])

    return Q
