import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

from vqf import VQF

import threading

from droneaudition_msgs.msg import IMURaw
from droneaudition_msgs.msg import Quat

class IMU_Quat(Node):
  def __init__(self):
    super().__init__('imu_quat')
    
    self.declare_parameter('quat_type', 'inverse')
    self.quat_type = self.get_parameter('quat_type').get_parameter_value().string_value
    self.declare_parameter('Ts', 0.01)
    self.Ts = self.get_parameter('Ts').get_parameter_value().double_value
    
    self.vqf = VQF(self.Ts)
    
    self.subscription = self.create_subscription(IMURaw,'imu_raw',self.imu_callback,10)
    self.subscription  # prevent unused variable warning
    
    self.publisher = self.create_publisher(Quat, '/imu_quat', 1000)
  
  def imu_callback(self,msg):
    timestamp = msg.time
    acc_arr = np.array([msg.acc])
    gyr_arr = np.array([msg.gyr])
    mag_arr = np.array([msg.mag])
    roll = msg.roll
    pitch = msg.pitch
    yaw = msg.yaw
    
    # do quaternion magic here
    #quat = (self.vqf.updateBatch(gyr_arr, acc_arr, mag_arr))['quat9D']
    quat = (self.vqf.updateBatch(gyr_arr, acc_arr))['quat6D']
    quat = quat[:, [1, 2, 3, 0]]
    
    msg_pub = Quat()
    msg_pub.time = timestamp
    msg_pub.quat = (quat[0,:]).tolist()
    msg_pub.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg_pub)

def main(args=None):
  rclpy.init(args=args)
  imuquat = IMU_Quat()
  rclpy.spin(imuquat)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  imuquat.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

