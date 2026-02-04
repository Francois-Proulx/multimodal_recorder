import matplotlib.pyplot as plt


def plot_raw_imu(temps, acc, gyro, mag=None, save_file=None):
    """
    Plot raw IMU data (Accelerometer, Gyroscope, and optionally Magnetometer).
    """
    # Determine number of subplots (2 or 3)
    rows = 3 if mag is not None else 2

    # Create figure with shared X axis
    fig, ax = plt.subplots(rows, 1, sharex=True, figsize=(10, 3 * rows))

    # Accelerometer
    ax[0].plot(temps, acc)
    ax[0].set_title("Accelerometer raw [m/s^2]")
    ax[0].grid(True, linestyle=":")
    ax[0].set_ylabel("Amplitude")

    # Gyroscope
    ax[1].plot(temps, gyro)
    ax[1].set_title("Gyroscope raw [rad/s]")
    ax[1].grid(True, linestyle=":")
    ax[1].set_ylabel("Amplitude")

    # Magnetometer (Optional)
    if mag is not None:
        ax[2].plot(temps, mag)
        ax[2].set_title("Magnetometer raw [uT]")
        ax[2].grid(True, linestyle=":")
        ax[2].set_ylabel("Amplitude")
        ax[2].set_xlabel("Time [s]")
    else:
        ax[1].set_xlabel("Time [s]")

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()
