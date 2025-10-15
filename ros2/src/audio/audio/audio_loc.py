import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

import threading

from droneaudition_msgs.msg import AudioRaw
from droneaudition_msgs.msg import AudioLoc

class Audio_Loc(Node):
  def __init__(self):
    super().__init__('audio_loc')
    
    self.declare_parameter('loc_type', 'MUSIC')
    self.loc_type = self.get_parameter('loc_type').get_parameter_value().string_value
    
    self.subscription = self.create_subscription(AudioRaw,'audio_raw',self.audio_callback,10)
    self.subscription  # prevent unused variable warning
    
    self.publisher = self.create_publisher(AudioLoc, '/audio_loc', 1000)
  
  def audio_callback(self,msg):
    timestamp = msg.time
    data_orig = np.array(msg.data)
    channels = msg.channels
    winlen = msg.winlen
    fs = msg.fs
    
    audio_data = np.reshape(data_orig, (winlen,channels), order="F")
    
    # here do DOA localization with "audio_data",
    # with shape (N,C), N being length, C being channels
    
    msg_pub = AudioLoc()
    msg_pub.time = timestamp
    msg_pub.doa = 90.0 # copy DOA result here
    msg_pub.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg_pub)

def main(args=None):
  rclpy.init(args=args)
  audioloc = Audio_Loc()
  rclpy.spin(audioloc)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  audioloc.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

