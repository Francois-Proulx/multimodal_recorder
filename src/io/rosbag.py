# IMPORTS MESSAGES
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from droneaudition_msgs.msg import (
    AudioLoc,
    IMURaw,
)

# OTHER ROS IMPORTS
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter


def read_bag(bag_path, layers=None, custom_map=None):
    """
    Args:
        bag_path (str): Path to bag.
        layers (list[str]): List of keys to read from defaults (e.g. ['imu_raw', 'video_quat']).
                            If None, reads ALL defaults.
        custom_map (dict): If provided, COMPLETELY overrides defaults and layers.
    """
    # 0. Default topics
    DEFAULT_TOPICS = {
        "imu_raw": "/imu_raw",
        "imu_quat": "/imu/orientation",
        "video_quat": "/video/orientation",
        "audio_loc": "/audio_loc",
    }

    # 1. Determine which map to use
    if custom_map:
        # Scenario A: User provides totally custom topic names
        topic_map = custom_map
    else:
        # Scenario B: Use Project Defaults
        topic_map = DEFAULT_TOPICS

        # Scenario C: Filter the Defaults (Optimization)
        if layers is not None:
            # Create a new dict with ONLY the requested keys
            # Checks if key exists to prevent crashes
            topic_map = {k: DEFAULT_TOPICS[k] for k in layers if k in DEFAULT_TOPICS}

    # 2. Invert map for O(1) lookup: {'/ros/topic': 'internal_key'}
    topic_to_key = {v: k for k, v in topic_map.items()}

    # 3. Create Filter List
    # The reader will ONLY look at these topics and skip everything else in the file.
    topics_of_interest = list(topic_map.values())

    # 4. Storage Setup
    data_dic = {key: {"t": [], "data": []} for key in topic_map.keys()}

    # 5. Setup Bag Reader
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    bag_filter = StorageFilter(topics=topics_of_interest)
    reader.set_filter(bag_filter)

    # 2. Iterate through messages
    while reader.has_next():
        (topic, data_bytes, t_ns) = reader.read_next()

        if topic in topic_to_key:
            internal_key = topic_to_key[topic]
            t_sec = t_ns / 1e9

            # Data parsing (bytes to dict.)
            parsed_data = None

            if internal_key == "imu_raw":
                parsed_data = parse_imu_raw(data_bytes)

            elif internal_key == "imu_quat":
                parsed_data = parse_imu_orientation(data_bytes)

            elif internal_key == "video_quat":
                parsed_data = parse_video_pose(data_bytes)

            elif internal_key == "audio_loc":
                parsed_data = parse_audio_loc(data_bytes)

            if parsed_data is not None:
                data_dic[internal_key]["t"].append(t_sec)
                data_dic[internal_key]["data"].append(parsed_data)

    return data_dic


def parse_imu_raw(data_bytes):
    msg = deserialize_message(data_bytes, IMURaw)
    return [
        msg.acc[0],
        msg.acc[1],
        msg.acc[2],
        msg.gyr[0],
        msg.gyr[1],
        msg.gyr[2],
        msg.mag[0],
        msg.mag[1],
        msg.mag[2],
    ]


def parse_imu_orientation(data_bytes):
    msg = deserialize_message(data_bytes, QuaternionStamped)
    return [
        msg.quaternion.x,
        msg.quaternion.y,
        msg.quaternion.z,
        msg.quaternion.w,
    ]


def parse_video_pose(data_bytes):
    msg = deserialize_message(data_bytes, PoseStamped)
    return [
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ]


def parse_audio_loc(data_bytes):
    msg = deserialize_message(data_bytes, AudioLoc)
    return [msg.pos[0], msg.pos[1], msg.pos[2]]
