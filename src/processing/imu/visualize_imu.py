import matplotlib.pyplot as plt

def plot_raw_imu(temps, acc, gyro, mag):
    plt.figure()
    plt.subplot(311); plt.plot(temps, acc); plt.title("Accelerometer raw"); plt.grid(True)
    plt.subplot(312); plt.plot(temps, gyro); plt.title("Gyroscope raw"); plt.grid(True)
    plt.subplot(313); plt.plot(temps, mag); plt.title("Magnetometer raw"); plt.grid(True)
    plt.tight_layout()
    
