## Dependencies
Install dependencies for nodes:
```
sudo apt install libcap-dev libsndfile1-dev libcamera-dev
```

Install ROS2 following [this guide](https://docs.ros.org/#ros-for-beginners), preferable using the "Tier-1" ROS2-Ubuntu version pairings.

## Install
Assume `ROS2Base` is the base directory of the project, that has both the ROS2 workspace and the multimodal recorder code (they do not need to be in the same directory, though).

```
cd ROS2Base
git clone https://github.com/Francois-Proulx/multimodal_recorder

mkdir -p ROS2Base/droneaudition/
cd ROS2Base/droneaudition/
ln -s ROS2Base/multimodal_recorder/ros2/src .
cd src/
pip install -r requirements.txt
cd ..
colcon build

echo "ROS2Base/droneaudition/install/local_setup.bash" >> ~/.bashrc
```

## Configuration files:

Each node has its configuration file:

**IMU nodes configuration files**:
```
src/imu/config/imu_adq_config.yaml 
src/imu/config/imu_quat_config.yaml 
```

**Audio nodes configuration files**:
```
src/audio/config/audio_adq_config.yaml 
src/audio/config/audio_loc_config.yaml 
```

**Video nodes configuration files**:
```
src/video/config/cam_hardware.yaml
src/video/config/usb_cam_calib_params.yaml 
src/video/config/video_quat_config.yaml 
```

**Fuse node configuration file**:
```
src/fuse/config/fuse_config.yaml 
```

## To run:
Each node has its own launch file.
Each system component (IMU, Audio, Video, Fuse) also have its own launch file.
There is a master launch file to launch everything together, with the option to save to ROS 2 bag files.

**IMU capture**:
```
ros2 launch imu imu_adq.launch
```

**IMU quaternion**:
```
ros2 launch imu imu_quat.launch 
```

**IMU system**:
```
ros2 launch imu imu_system.launch.py
```

**Audio capture**:
```
ros2 launch audio audio_adq.launch 
```

**Audio localization**:
```
ros2 launch audio audio_loc.launch 
```

**Audio system**:
```
ros2 launch audio audio_system.launch.py
```

**Video capture**:
Uses the usb_cam package to capture video from a USB camera

**Video quaternion**:
```
ros2 launch video video_quat.launch 
```

**Video system**:
```
ros2 launch video video_system.launch.py
```

**Fuse node**:
```
ros2 launch fuse fuse.launch 
```

**Master launch for all nodes at once**:
```
ros2 launch fuse master_launch.launch.py 
```

**Master launch for all nodes at once with ROS 2 bag files**:
```
ros2 launch fuse master_launch.launch.py record_bag:=True
```


## Recorded Data Specifications

The `master_launch.launch.py` generates ROS 2 bag files containing the following synchronized streams:

| Topic | Type | Frequency | Description |
| :--- | :--- | :--- | :--- |
| `/audio_raw` | `droneaudition_msgs/AudioRaw` | ~4 Hz | Raw audio. Default parameters are: samplerate: 16000 Hz, channels: 16, blocksize: 4096, sampwidth: 4 |
| `/audio_loc` | `droneaudition_msgs/AudioLoc` | ~4 Hz | Sound Source Localization results (Direction of Arrival). |
| `/imu_raw` | `droneaudition_msgs/IMURaw` | ~100 Hz | Raw Accelerometer, Gyroscope, and Magnetometer data. |
| `/imu_quat` | `droneaudition_msgs/Quat` | ~100 Hz | Estimated array orientation (Quaternion) computed directly from the IMU data stream. |
| `/video_raw/compressed` | `sensor_msgs/CompressedImage` | ~30 Hz | MJPEG stream (640x480). Start time aligned with Audio. |
| `/video_quat` | `droneaudition_msgs/Quat` | ~30 Hz | Drone orientation (Quaternion) computed from video frames. |
| `/camera_info` | `sensor_msgs/CameraInfo` | ~30 Hz | Camera calibration intrinsics (matrix K, D) for the current frame, loaded from the YAML calibration file. |

### Coordinate Systems ** TO VALIDATE!! **
* **Camera:** X-Right, Y-Down, Z-Forward (Standard Optical Frame).
* **IMU:** X-Forward, Y-Left, Z-Up (Standard ROS Body Frame). *[Verify this with your visualize check!]*
* **Audio Array:** X-Forward, Y-Left, Z-Up (Standard ROS Body Frame).
* **Quaternions:** All quaternion topics follow the standard `[w, x, y, z]` convention relative to the fixed starting frame (ENU).
