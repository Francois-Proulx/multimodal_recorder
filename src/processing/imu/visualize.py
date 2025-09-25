import matplotlib.pyplot as plt

def plot_euler(roll, pitch, yaw, title="Euler angles"):
    plt.figure()
    plt.subplot(311); plt.plot(roll); plt.ylabel("Roll"); plt.grid(True)
    plt.subplot(312); plt.plot(pitch); plt.ylabel("Pitch"); plt.grid(True)
    plt.subplot(313); plt.plot(yaw); plt.ylabel("Yaw"); plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def plot_quat(quats, title="Quaternions"):
    """Plot quaternion components over time.
    
    Args:
        quats (ndarray): Array of shape (N, 4), quaternion at each timestep.
        title (str): Figure title.
    """
    plt.figure(figsize=(8, 6))
    labels = ["Q1", "Q2", "Q3", "Q4"]

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(quats[:, i])
        plt.ylabel(labels[i])
        plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_mic_trajectory(pA):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pA[:,0,0], pA[:,1,0], pA[:,2,0], label='Mic trajectory')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Mic trajectory in global frame')
    ax.legend()
    plt.show()