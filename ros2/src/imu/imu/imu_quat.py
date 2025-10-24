import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep
import csv

from vqf import VQF

import threading
from queue import Empty
from multiprocessing import Process, Queue, Event

from droneaudition_msgs.msg import IMURaw
from droneaudition_msgs.msg import Quat

class IMUFileWriter(Process):
    def __init__(self, queue: Queue, stop_event, save_path: str, filename="imu_data.csv"):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        os.makedirs(save_path, exist_ok=True)
        self.filepath = os.path.join(save_path, filename)

    def run(self):
        print('IMUFileWriter started, saving to', self.filepath)
        fieldnames = [
            "timestamp",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "roll", "pitch", "yaw"
        ]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                    f.flush()
                except Empty:
                    continue
                except Exception as e:
                    print("IMUFileWriter error:", e)
                    continue
                  

class IMU_Quat(Node):
  def __init__(self):
    super().__init__('imu_quat')
    
    # Declare parameters with optional default values
    self.declare_parameter('quat_type', 'inverse')
    self.declare_parameter('Ts', 0.01)
    self.declare_parameter('save_path', '/home/francois/Documents/Git/multimodal_recorder/data')
    self.declare_parameter('save_imu', False)
    
    # Load parameters from audio_loc_config.yaml
    self.quat_type = self.get_parameter('quat_type').get_parameter_value().string_value
    self.Ts = self.get_parameter('Ts').get_parameter_value().double_value
    self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
    self.save_imu = self.get_parameter('save_imu').get_parameter_value()
    
    self.vqf = VQF(self.Ts)
    
    self.subscription = self.create_subscription(IMURaw,'imu_raw',self.imu_callback,10)
    self.subscription  # prevent unused variable warning
    
    self.publisher = self.create_publisher(Quat, '/imu_quat', 1000)
    
    # IMU saving
    if self.save_imu:
      self.imu_queue = Queue(maxsize=500)
      self.stop_event = Event()
      self.file_writer = IMUFileWriter(self.imu_queue, self.stop_event, self.save_path, filename="imu_data.csv")
      self.file_writer.start()
  
  def imu_callback(self,msg):
    timestamp = msg.time
    acc_arr = np.array([msg.acc])
    gyr_arr = np.array([msg.gyr])
    mag_arr = np.array([msg.mag])
    roll = msg.roll
    pitch = msg.pitch
    yaw = msg.yaw
    self.get_logger().info("acc arr size"+str(acc_arr[0])+str(acc_arr[0,0]))
    
    # do quaternion magic here
    #quat = (self.vqf.updateBatch(gyr_arr, acc_arr, mag_arr))['quat9D']
    quat = (self.vqf.updateBatch(gyr_arr, acc_arr))['quat6D']
    quat = quat[:, [1, 2, 3, 0]]
    
    msg_pub = Quat()
    msg_pub.time = timestamp
    msg_pub.quat = (quat[0,:]).tolist()
    msg_pub.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg_pub)
    
    # Optionally save imu
    if self.save_imu:
      try:
        data = {
          "timestamp": timestamp,
          "acc_x": acc_arr[0,0],
          "acc_y": acc_arr[0,1],
          "acc_z": acc_arr[0,2],
          "gyro_x": gyr_arr[0,0],
          "gyro_y": gyr_arr[0,1],
          "gyro_z": gyr_arr[0,2],
          "mag_x": mag_arr[0,0],
          "mag_y": mag_arr[0,1],
          "mag_z": mag_arr[0,2],
          "roll": roll,
          "pitch": pitch,
          "yaw": yaw
        }
        self.imu_queue.put_nowait(data)
      except:
        pass  # drop frame if queue full

def main(args=None):
  rclpy.init(args=args)
  imuquat = IMU_Quat()
  try:
    rclpy.spin(imuquat)
  except KeyboardInterrupt:
    print("Shutting down Imu_quat...")
  finally:
    if imuquat.save_imu:
      imuquat.stop_event.set()
      imuquat.file_writer.join()
    imuquat.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()

