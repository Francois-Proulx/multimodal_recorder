import os
import time
import smbus
from imusensor.MPU9250 import MPU9250

def online_calibration(bus_id=1, address=0x68, save_path=None):
    """
    Perform online accelerometer and magnetometer calibration and save results to JSON.
    
    Returns:
        imu: calibrated MPU9250 object
        calib_file: path to saved JSON file
    """
    bus = smbus.SMBus(bus_id)
    imu = MPU9250.MPU9250(bus, address)
    imu.begin()

    print("Accel calibration starting...")
    imu.caliberateAccelerometer()
    print("Accel calibration finished")
    print("Accel bias:", imu.AccelBias)
    print("Accel scales:", imu.Accels)

    print("Mag calibration starting...")
    time.sleep(2)
    imu.caliberateMagPrecise()
    print("Mag calibration finished")
    print("MagBias:", imu.MagBias)
    print("MagTransform:", imu.Magtransform)
    print("Mag scales:", imu.Mags)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "calibMPU9250.json")
    imu.saveCalibDataToFile(save_path)

    return imu, save_path



if __name__ == "__main__":
    # Example usage
    imu, calib_file = online_calibration(bus_id=1, address=0x68, save_path="calibMPU9250.json")
    print(f"Calibration saved to {calib_file}")