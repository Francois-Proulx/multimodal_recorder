import matplotlib.pyplot as plt
import numpy as np

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
    
    
def plot_doa_and_yaw(t, doa_measured, sensors=None, ids=None, title=None, filename=None):
    """
    Plot measured DOAs and yaw(s) from one or multiple sensors.

    Args:
        t (np.ndarray): Time vector [s]
        doa_measured (np.ndarray): Original DOA [azimuth, elevation] in deg
        sensors (list of np.ndarray, optional): List of sensors vectors to plot
        ids (list of str, optional): List of sensor IDs corresponding to yaws
        title (str, optional): Plot title
        filename (str, optional): Base filename to save plots (will add '_azimuth' and '_elevation')
    """
    if sensors is None: sensors = []
    if ids is None: ids = [f"Sensor {i+1}" for i in range(len(sensors))]
    
    # Azimuth
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(t, doa_measured[:,0], 'g', label='Mic DOA azimuth')
    for sensor, sensor_id in zip(sensors, ids):
        ax.plot(t, sensor, label=f"{sensor_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Azimuth [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if title: plt.title(title)
    if filename: plt.savefig(filename + "_azimuth.svg", bbox_inches='tight', pad_inches=0.01)

    # Elevation
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(t, doa_measured[:,1], 'g', label='Mic DOA elevation')
    for sensor, sensor_id in zip(sensors, ids):
        ax.plot(t, sensor, label=f"{sensor_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Elevation [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if title: plt.title(title)
    if filename: plt.savefig(filename + "_elevation.svg", bbox_inches='tight', pad_inches=0.01)
    
    
def plot_doa_world(t, doas_world, doas_ids=None, sensors=None, sensors_ids=None, title=None, filename=None):
    """
    Plot DOAs rotated to world frame, optionally comparing multiple sensors.
    
    Args:
        t (np.ndarray): Time vector [s]
        doa_world (np.ndarray): DOA in world frame [azimuth, elevation] in deg
        doas_world (list or np.ndarray): World DOA(s) [azimuth, elevation] in deg. 
                                         Can be Nx2 for one DOA, 
                                         or list of Nx2 arrays for multiple DOAs.
        sensors (list of np.ndarray, optional): List of DOA arrays to plot for comparison
        ids (list of str, optional): IDs for sensors
        title (str, optional)
        filename (str, optional)
    """
    # Ensure doas_world is a list
    if isinstance(doas_world, np.ndarray) and doas_world.ndim == 2:
        doas_world = [doas_world]
    if doas_ids is None:
        doas_ids = [f"World DOA {i+1}" for i in range(len(doas_world))]
    
    if sensors is None:
        sensors = []
    if sensors_ids is None:
        sensors_ids = [f"Sensor {i+1}" for i in range(len(sensors))]

    # Azimuth
    fig, ax = plt.subplots(figsize=(6,3))
    for doa, label in zip(doas_world, doas_ids):
        ax.plot(t, doa[:,0], label=label)
    for sensor, sensor_id in zip(sensors, sensors_ids):
        ax.plot(t, sensor, label=f"{sensor_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Azimuth [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if title: plt.title(title)
    if filename: plt.savefig(filename + "_world_azimuth.svg", bbox_inches='tight', pad_inches=0.01)

    # Elevation
    fig, ax = plt.subplots(figsize=(6,3))
    for doa, label in zip(doas_world, doas_ids):
        ax.plot(t, doa[:,1], label=label)
    for s, sensor_id in zip(sensors, sensors_ids):
        ax.plot(t, sensor, label=f"{sensor_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Elevation [deg]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':')
    if title: plt.title(title)
    if filename: plt.savefig(filename + "_world_elevation.svg", bbox_inches='tight', pad_inches=0.01)
