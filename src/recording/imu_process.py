import time
from multiprocessing import Queue
from imusensor.MPU9250 import MPU9250
from imusensor.filters import kalman
import smbus2 as smbus
import json

class IMUProcess:
    def __init__(self, imu_queue: Queue, stop_event, address=0x68):
        self.imu_queue = imu_queue
        self.stop_event = stop_event

        self.bus = smbus.SMBus(1)
        self.imu = MPU9250.MPU9250(self.bus, address)
        self.imu.begin()
        self.imu.setAccelRange("AccelRangeSelect8G")
        self.imu.setGyroRange("GyroRangeSelect1000DPS")
        self.imu.setLowPassFilterFrequency("AccelLowPassFilter184")
        self.imu.loadCalibDataFromFile("calibMPU9250.json")

        self.kalman_filter = kalman.Kalman()

    def run(self, sample_period=0.01):
        """Read IMU, apply Kalman, push to queue"""
        print("IMU process started")
        while not self.stop_event.is_set():
            self.imu.readSensor()
            self.imu.computeOrientation()

            curr_time = time.time()
            self.kalman_filter.computeAndUpdateRollPitchYaw(
                self.imu.AccelVals[0], self.imu.AccelVals[1], self.imu.AccelVals[2],
                self.imu.GyroVals[0], self.imu.GyroVals[1], self.imu.GyroVals[2],
                self.imu.MagVals[0], self.imu.MagVals[1], self.imu.MagVals[2],
                sample_period
            )

            # push raw + filtered orientation to queue
            self.imu_queue.put({
                "timestamp": curr_time,
                "roll": self.imu.roll,
                "pitch": self.imu.pitch,
                "yaw": self.imu.yaw,
                "kalman_roll": self.kalman_filter.roll,
                "kalman_pitch": self.kalman_filter.pitch,
                "kalman_yaw": self.kalman_filter.yaw
            })

            time.sleep(sample_period)
        print("IMU process stopped")
