import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

import threading

import cv2

from sensor_msgs.msg import Image
from droneaudition_msgs.msg import VideoRaw
from cv_bridge import CvBridge
from droneaudition_msgs.msg import Quat

class Video_Quat(Node):
  def __init__(self):
    super().__init__('video_quat')
    
    self.declare_parameter('loc_type', 'apriltag')
    self.loc_type = self.get_parameter('loc_type').get_parameter_value().string_value
    
    self.subscription = self.create_subscription(VideoRaw,'video_raw',self.video_callback,10)
    self.subscription  # prevent unused variable warning
    
    self.bridge = CvBridge()
    self.publisher = self.create_publisher(Quat, '/video_quat', 1000)
  
  def video_callback(self,msg):
    timestamp = msg.time
    img_data = self.bridge.imgmsg_to_cv2(msg.data, desired_encoding='passthrough')
    
    # here do quaternion stuff with "img_data",
    # that is a cv2 image
    
    msg_pub = Quat()
    msg_pub.time = timestamp
    msg_pub.quat = [0.0, 1.0, 2.0, 3.0]
    msg_pub.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg_pub)

def main(args=None):
  rclpy.init(args=args)
  videoquat = Video_Quat()
  rclpy.spin(videoquat)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  videoquat.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

