import matplotlib.pyplot as plt
import numpy as np

def plot_euler(roll, pitch, yaw, t=None, title="Euler angles"):
    """
    Plot Euler angles, skipping NaNs without shortening the vectors.

    Args:
        roll, pitch, yaw (array-like): Euler angles, can contain np.nan.
        t (array-like, optional): Time vector. If None, index will be used.
        title (str): Plot title.
    """
    roll = np.asarray(roll)
    pitch = np.asarray(pitch)
    yaw = np.asarray(yaw)

    if t is None:
        t = np.arange(len(roll))  # just indices

    plt.figure(figsize=(8, 6))
    plt.subplot(311)
    plt.plot(t, roll, label="Roll")
    plt.ylabel("Roll")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(t, pitch, label="Pitch")
    plt.ylabel("Pitch")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(t, yaw, label="Yaw")
    plt.ylabel("Yaw")
    plt.xlabel("Frame / Time")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()


def plot_quat(quats, quats2=None, t1=None, t2=None, title="Quaternions"):
    """
    Plot quaternion components over time.

    Args:
        quats (ndarray): Array of shape (N, 4), quaternions at each timestep.
        quats2 (ndarray, optional): Second quaternion array of shape (N, 4) to compare.
        t1 (ndarray, optional): Time vector for quats.
        t2 (ndarray, optional): Time vector for quats2.
        title (str, optional): Figure title. Defaults to "Quaternions".
    """
    labels = ["Q1", "Q2", "Q3", "Q4"]
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(labels):
        plt.subplot(4, 1, i + 1)
        
        if quats2 is not None:
            if t1 is None or t2 is None:
                plt.plot(quats[:, i], label='Original')
                plt.plot(quats2[:, i], '--', label='Rebased')
            else:
                plt.plot(t1, quats[:, i], label='Original')
                plt.plot(t2, quats2[:, i], '--', label='Rebased')
            plt.legend()
        else:
            plt.plot(quats[:, i])
        
        plt.ylabel(label)
        plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()


def plot_mic_trajectory(pA):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pA[:,0,0], pA[:,1,0], pA[:,2,0], label='Mic trajectory')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Mic trajectory in global frame')
    ax.legend()
