## Dependencies
sudo apt install libcap-dev libsndfile1-dev libcamera-dev

## Install
Assume `ROS2Base` is the base directory of the project (both ROS2 workspace and multimodal recorder code).

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

## To run:
Each node has its own launch file:

**IMU capture**:
```
ros2 launch imu imu_adq.launch 
```

**Audio capture**:
```
ros2 launch audio audio_adq.launch 
```

**Video capture**:
```
ros2 launch video video_adq.launch 
```

**IMU quaternion**:
```
ros2 launch imu imu_quat.launch 
```

**Audio localization**:
```
ros2 launch audio audio_loc.launch 
```

**Video quaternion**:
```
ros2 launch video video_quat.launch 
```

**Fuse node**:
```
ros2 launch fuse fuse.launch 
```

**All nodes at once**:
```
ros2 launch fuse fuse_full.launch 
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
src/video/config/video_adq_config.yaml 
src/video/config/video_quat_config.yaml 
```

**Fuse node configuration file**:
```
src/fuse/config/fuse_config.yaml 
```

