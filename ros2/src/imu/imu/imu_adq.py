import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

import threading

from droneaudition_msgs.msg import IMURaw

class IMU_Adq(Node):
  def __init__(self):
    super().__init__('imu_adq')
    
    self.declare_parameter('operation_mode', 'rt')
    self.operation_mode = self.get_parameter('operation_mode').get_parameter_value().string_value
    self.declare_parameter('csv_path', 'test.csv')
    self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
    self.declare_parameter('calib_path', 'test.json')
    self.calib_path = self.get_parameter('calib_path').get_parameter_value().string_value
    
    self.publisher = self.create_publisher(IMURaw, '/imu_raw', 1000)
    
    if self.operation_mode == 'csv':
      self.csv_thread = threading.Thread(target=self.from_csv)
      self.csv_thread.start()
    else:
      self.imu_thread = threading.Thread(target=self.from_imu)
      self.imu_thread.start()
  
  def publish_data(self, timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, roll, pitch, yaw):
    msg = IMURaw()
    msg.time = timestamp
    msg.acc = [acc_x, acc_y, acc_z]
    msg.gyr = [gyr_x, gyr_y, gyr_z]
    msg.mag = [mag_x, mag_y, mag_z]
    msg.roll = roll
    msg.pitch = pitch
    msg.yaw = yaw
    msg.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg)
  
  def from_csv(self):
    import pandas as pd
    
    if self.csv_path and os.path.exists(self.csv_path):
      csv_reader = pd.read_csv(self.csv_path)
    else:
      self.get_logger().info("CSV file not found: "+self.csv_path)
      exit()
    
    self.get_logger().info("Publishing CSV: "+str(self.csv_path))
    
    last_timestamp = None
    for index, row in csv_reader.iterrows():
      #simulating latency through timestamps (all values from IMU data is in nanoseconds)
      if last_timestamp == None:
        last_timestamp = row['timestamp'] / 1e9
        sleep(0.01)
      else:
        this_timestamp =  row['timestamp'] / 1e9
        sleep(this_timestamp - last_timestamp)
        last_timestamp = this_timestamp
      
      self.publish_data(
        row['timestamp'],
        row['acc_x'],
        row['acc_y'],
        row['acc_z'],
        row['gyro_x'],
        row['gyro_y'],
        row['gyro_z'],
        row['mag_x'],
        row['mag_y'],
        row['mag_z'],
        row['roll'],
        row['pitch'],
        row['yaw']
      )
    self.get_logger().info("Finished publishing CSV.")
  
  def from_imu(self):
    import smbus
    from imusensor.MPU9250.MPU9250 import MPU9250
    
    self.address = 0x68
    self.bus_num = 1

    # initialize sensor
    self.bus = smbus.SMBus(self.bus_num)
    self.imu = MPU9250(self.bus, self.address)
    self.imu.begin()
    self.imu.setAccelRange("AccelRangeSelect8G")
    self.imu.setGyroRange("GyroRangeSelect1000DPS")
    self.imu.setLowPassFilterFrequency("AccelLowPassFilter184")

    # load calibration file if provided
    if self.calib_path and os.path.exists(self.calib_path):
      self.imu.loadCalibDataFromFile(self.calib_path)
    else:
      print("Calibration file not found:", self.calib_path)
    
    while True:
      self.imu.readSensor()
      self.imu.computeOrientation()
      self.publish_data(
        float(time.time()),
        self.imu.AccelVals[0],
        self.imu.AccelVals[1],
        self.imu.AccelVals[2],
        self.imu.GyroVals[0],
        self.imu.GyroVals[1],
        self.imu.GyroVals[2],
        self.imu.MagVals[0],
        self.imu.MagVals[1],
        self.imu.MagVals[2],
        self.imu.roll,
        self.imu.pitch,
        self.imu.yaw
      )
      time.sleep(0.01)  # 100 Hz sampling

  

def main(args=None):
  rclpy.init(args=args)
  imuadq = IMU_Adq()
  rclpy.spin(imuadq)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  imuadq.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

