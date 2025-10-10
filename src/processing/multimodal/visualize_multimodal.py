import matplotlib.pyplot as plt

def plot_doa_alignement_and_yaw(t, doa_measured, yaw, doa_world, title=None, filename=None):
    """
    Compare DOA azimuth/elevation before and after IMU alignment.

    Args:
        t (np.ndarray): time vector [s]
        doa_measured (np.ndarray): original DOA [azimuth, elevation] in deg
        yaw (np.ndarray): IMU yaw angle [deg]
        doa_world (np.ndarray): DOA rotated to world frame [azimuth, elevation] in deg
    """

    # ---- Azimuth ----
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, doa_measured[:, 0], 'g', label='Mic DOA azimuth')
    ax.plot(t, yaw, 'b', label='IMU yaw')
    ax.plot(t, doa_world[:, 0], 'r', label='World DOA azimuth')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Azimuth [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename + "_azimuth.svg", bbox_inches='tight', pad_inches=0.01)
    else:
        # plt.show()
        pass

    # ---- Elevation ----
    fig, ax = plt.subplots(figsize=(6, 3))
    if title:
        plt.title(title)
    ax.plot(t, doa_measured[:, 1], 'g', label='Mic DOA elevation')
    ax.plot(t, yaw, 'b', label='IMU yaw')
    ax.plot(t, doa_world[:, 1], 'r', label='World DOA elevation')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Elevation [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if filename:
        plt.savefig(filename + "_elevation.svg", bbox_inches='tight', pad_inches=0.01)
    else:
        # plt.show()
        pass