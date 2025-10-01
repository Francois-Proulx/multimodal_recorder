import time
import smbus
import os
from multiprocessing import Process, Queue
from imusensor.MPU9250.MPU9250 import MPU9250
import numpy as np

class IMUProcess(Process):
    def __init__(self, imu_queue: Queue, stop_event, calib_file=None, address=0x68, bus_num=1):
        super().__init__()
        self.imu_queue = imu_queue
        self.stop_event = stop_event
        self.address = address
        self.bus_num = bus_num
        self.calib_file = calib_file

        # initialize sensor
        self.bus = smbus.SMBus(bus_num)
        self.imu = MPU9250(self.bus, address)
        self.imu.begin()
        self.imu.setAccelRange("AccelRangeSelect8G")
        self.imu.setGyroRange("GyroRangeSelect1000DPS")
        self.imu.setLowPassFilterFrequency("AccelLowPassFilter184")

        # load calibration file if provided
        if self.calib_file and os.path.exists(self.calib_file):
            self.imu.loadCalibDataFromFile(self.calib_file)
        else:
            print("Calibration file not found:", self.calib_file)

    def run(self):
        print("IMUProcess started")
        while not self.stop_event.is_set():
            self.imu.readSensor()
            self.imu.computeOrientation()
            data = {
                "timestamp": time.time_ns(),
                "acc_x": self.imu.AccelVals[0],
                "acc_y": self.imu.AccelVals[1],
                "acc_z": self.imu.AccelVals[2],
                "gyro_x": self.imu.GyroVals[0],
                "gyro_y": self.imu.GyroVals[1],
                "gyro_z": self.imu.GyroVals[2],
                "mag_x": self.imu.MagVals[0],
                "mag_y": self.imu.MagVals[1],
                "mag_z": self.imu.MagVals[2],
                "roll": self.imu.roll,
                "pitch": self.imu.pitch,
                "yaw": self.imu.yaw
            }
            self.imu_queue.put(data)
            time.sleep(0.01)  # 100 Hz sampling

        print("IMUProcess stopping")
        self.bus.close()
