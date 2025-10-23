import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
from time import sleep

import threading

from droneaudition_msgs.msg import AudioRaw

class Audio_Adq(Node):
  def __init__(self):
    super().__init__('audio_adq')
    
    self.declare_parameter('operation_mode', 'rt')
    self.operation_mode = self.get_parameter('operation_mode').get_parameter_value().string_value
    self.declare_parameter('csv_path', 'test.csv')
    self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
    self.declare_parameter('audio_path', 'test.wav')
    self.audio_path = self.get_parameter('audio_path').get_parameter_value().string_value
    self.declare_parameter('device', 'hw:0')
    self.device = self.get_parameter('device').get_parameter_value().string_value
    self.declare_parameter('samplerate', 16000)
    self.samplerate = self.get_parameter('samplerate').get_parameter_value().integer_value
    self.declare_parameter('channels', 16)
    self.channels = self.get_parameter('channels').get_parameter_value().integer_value
    self.declare_parameter('blocksize', 1024)
    self.blocksize = self.get_parameter('blocksize').get_parameter_value().integer_value
    self.declare_parameter('sampwidth', 4)
    self.sampwidth = self.get_parameter('sampwidth').get_parameter_value().integer_value
    
    if self.sampwidth == 2:
        self.dtype = 'int16'
    elif self.sampwidth == 4:
        self.dtype = 'int32'
    else:
        raise ValueError("Unsupported sampwidth. Use 2 or 4.")
    
    self.publisher = self.create_publisher(AudioRaw, '/audio_raw', 1000)
    
    if self.operation_mode == 'csv':
      self.csv_thread = threading.Thread(target=self.from_csv)
      self.csv_thread.start()
    else:
      self.mic_thread = threading.Thread(target=self.from_mic)
      self.mic_thread.start()
  
  def publish_data(self, timestamp, audio_data, channels, winlen, fs):
    msg = AudioRaw()
    msg.time = timestamp
    msg.data = audio_data
    msg.fs = fs
    msg.channels = channels
    msg.winlen = winlen
    msg.header.stamp = self.get_clock().now().to_msg()
    self.publisher.publish(msg)
  
  def from_csv(self):
    import pandas as pd
    import soundfile as sf
    
    if self.csv_path and os.path.exists(self.csv_path):
      csv_reader = pd.read_csv(self.csv_path)
    else:
      self.get_logger().info("CSV file not found: "+self.csv_path)
      exit()
    
    if self.audio_path and os.path.exists(self.audio_path):
      data, fs = sf.read(self.audio_path, dtype="float32")
      self.get_logger().info("max data : "+str(np.max(data))+". min data = "+str(np.min(data)))
    else:
      self.get_logger().info("Audio file not found: "+self.audio_path)
      exit()
    
    # data is of shape (N, C), N being length, C being channels
    channels = data.shape[1]
    
    win_num = csv_reader.shape[0]
    data_len = data.shape[0]
    win_len = int(data_len/win_num)
    
    self.get_logger().info("Publishing CSV: "+str(self.csv_path))
    self.get_logger().info("Publishing Audio: "+str(self.audio_path))
    
    self.get_logger().info("  window number: "+str(win_num))
    self.get_logger().info("  window length: "+str(win_len))
    
    last_timestamp = None
    for index, row in csv_reader.iterrows():
      #simulating latency through timestamps (all values from IMU data is in nanoseconds)
      if last_timestamp == None:
        last_timestamp = row['timestamp']
        sleep(win_len/fs)
      else:
        this_timestamp =  row['timestamp']
        sleep(this_timestamp - last_timestamp)
        last_timestamp = this_timestamp
      
      t_ini = index*win_len
      t_fin = t_ini+win_len
      
      # Format data for publishing
      self.get_logger().info("max data : "+str(np.max(data))+". min data = "+str(np.min(data)))
      this_data = np.reshape(data[t_ini:t_fin,:], win_len*channels, order="F").tolist()
      self.get_logger().info("max data : "+str(max(this_data))+". min data = "+str(min(this_data)))
      
      self.publish_data(
        float(np.float32(row['timestamp'])),
        this_data,
        int(np.int32(channels)),
        int(np.int32(win_len)),
        int(np.int32(fs))
      )
    self.get_logger().info("Finished publishing CSV and Audio.")
  
  def audio_callback(self, indata, frames, time_info, status):
      if status:
          print("AudioProcess status:", status)
      
      this_data = np.reshape(indata, frames*self.channels, order="F").tolist()
      
      self.publish_data(
        time.time() - (self.blocksize / self.samplerate),
        this_data,
        self.channels,
        frames,
        self.samplerate
      )
  
  def from_mic(self):
    import sounddevice as sd
    try:
      with sd.InputStream(
        device=self.device,
        channels=self.channels,
        samplerate=self.samplerate,
        blocksize=self.blocksize,
        dtype='float32',
        callback=self.audio_callback
      ) as stream:
        while True:
          sleep(0.05)  # the process will block here
        
        # Force the InputStream to stop immediately
        stream.abort()
    
    except Exception as e:
        print("AudioProcess error:", e)
    finally:
        print("AudioProcess stopping")

def main(args=None):
  rclpy.init(args=args)
  audioadq = Audio_Adq()
  rclpy.spin(audioadq)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  audioadq.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()

