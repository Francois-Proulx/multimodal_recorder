import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

import threading

from droneaudition_msgs.msg import AudioLoc
from droneaudition_msgs.msg import Quat

class Fuse(Node):
  def __init__(self):
    super().__init__('fuse')
    
    self.declare_parameter('fuse_method', 'sync')
    self.fuse_method = self.get_parameter('fuse_method').get_parameter_value().string_value
    
    self.subscription_imu = self.create_subscription(Quat,'imu_quat',self.imu_callback,10)
    self.subscription_imu  # prevent unused variable warning
    self.subscription_audio = self.create_subscription(AudioLoc,'audio_loc',self.audio_callback,10)
    self.subscription_audio  # prevent unused variable warning
    self.subscription_video = self.create_subscription(Quat,'video_quat',self.video_callback,10)
    self.subscription_video  # prevent unused variable warning
  
  def imu_callback(self,msg):
    timestamp = msg.time
    imu_quat = msg.quat
    
    self.get_logger().info("IMU: "+str(imu_quat))
    
  def audio_callback(self,msg):
    timestamp = msg.time
    audio_pos = msg.pos
    
    self.get_logger().info("Aud: "+str(audio_pos))
    
  def video_callback(self,msg):
    timestamp = msg.time
    video_quat = msg.quat
    
    self.get_logger().info("Vid: "+str(video_quat))
    

def main(args=None):
  rclpy.init(args=args)
  fuse = Fuse()
  rclpy.spin(fuse)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  fuse.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

