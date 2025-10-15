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

class Video_Adq(Node):
  def __init__(self):
    super().__init__('video_adq')
    
    self.declare_parameter('operation_mode', 'rt')
    self.operation_mode = self.get_parameter('operation_mode').get_parameter_value().string_value
    self.declare_parameter('csv_path', 'test.csv')
    self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
    self.declare_parameter('video_folder', 'test')
    self.video_folder = self.get_parameter('video_folder').get_parameter_value().string_value
    self.declare_parameter('width', 640)
    self.width = self.get_parameter('width').get_parameter_value().integer_value
    self.declare_parameter('height', 480)
    self.height = self.get_parameter('height').get_parameter_value().integer_value
    self.declare_parameter('fps', 30)
    self.fps = self.get_parameter('fps').get_parameter_value().integer_value
    
    self.bridge = CvBridge()
    self.publisher = self.create_publisher(VideoRaw, '/video_raw', 1000)
    
    if self.operation_mode == 'csv':
      self.csv_thread = threading.Thread(target=self.from_csv)
      self.csv_thread.start()
    else:
      self.mic_thread = threading.Thread(target=self.from_cam)
      self.mic_thread.start()
  
  def publish_data(self, timestamp, img):
    msg = VideoRaw()
    msg.time = timestamp
    msg.data = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
    msg.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg)
  
  def from_csv(self):
    import glob
    import pandas as pd
    
    if self.csv_path and os.path.exists(self.csv_path):
      csv_reader = pd.read_csv(self.csv_path)
    else:
      self.get_logger().info("CSV file not found: "+self.csv_path)
      exit()
    
    if self.video_folder and os.path.exists(self.video_folder):
      image_paths = glob.glob(os.path.join(self.video_folder,"*.jpg"))
      image_paths.sort()
    else:
      self.get_logger().info("Video directory not found: "+self.video_folder)
      exit()
    
    self.get_logger().info("Publishing CSV: "+str(self.csv_path))
    self.get_logger().info("Publishing Images from: "+str(self.video_folder))
    
    last_timestamp = None
    for index, row in csv_reader.iterrows():
      #simulating latency through timestamps (all values from IMU data is in nanoseconds)
      if last_timestamp == None:
        last_timestamp = row['timestamp']
        sleep(1/self.fps)
      else:
        this_timestamp =  row['timestamp']
        sleep(this_timestamp - last_timestamp)
        last_timestamp = this_timestamp
      
      img = cv2.imread(image_paths[index], 0)
      
      self.publish_data(
        row['timestamp'],
        img
      )
    self.get_logger().info("Finished publishing CSV and Audio.")
  
  def from_cam(self):
    from picamera2 import Picamera2
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (self.width, self.height)})
    picam2.configure(config)
    picam2.start()
    
    try:
      while True:
        frame = picam2.capture_array()
        ts = time.time()
        self.publish_data(
          ts,
          frame,
        )
        
        # sleep to ensure desired fps
        sleep(1 / self.fps)
    except Exception as e:
      print("VideoProcess error:", e)
    finally:
      picam2.stop()
      print("VideoProcess stopping")

def main(args=None):
  rclpy.init(args=args)
  videoadq = Video_Adq()
  rclpy.spin(videoadq)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  videoadq.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

