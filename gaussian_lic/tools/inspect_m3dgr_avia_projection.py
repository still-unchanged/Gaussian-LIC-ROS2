#!/usr/bin/python3
"""Check official M3DGR Avia-to-D435i projection on a converted ROS2 bag.

Source the ROS2 workspace first. Reads the first compressed color image and
nearest of the initially collected Avia scans without modifying the bag.
"""

import argparse

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from cocolic_interfaces.msg import LivoxCustomMsg
from sensor_msgs.msg import CompressedImage


K = np.array([[607.79772949218, 0.0, 328.79772949218],
              [0.0, 607.83526613281, 245.53321838378],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# Official M3DGR calibration, p_camera = R @ p_avia + t.
R_AC = np.array([[0.0210767, -0.9993970, -0.0275965],
                 [-0.0038717, 0.0275209, -0.9996140],
                 [0.9997700, 0.0211754, -0.0032894]], dtype=np.float64)
T_AC = np.array([0.0178448, 0.0894589, 0.1615560], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="path to converted Grass01 ROS2 bag directory")
    args = parser.parse_args()

    image = None
    image_stamp = None
    scans = []
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    expected = {
        "/camera/color/image_raw/compressed": ("sensor_msgs/msg/CompressedImage", CompressedImage),
        "/livox/avia/lidar": ("cocolic_interfaces/msg/LivoxCustomMsg", LivoxCustomMsg),
    }
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    for topic, (type_name, _) in expected.items():
        if types.get(topic) != type_name:
            raise ValueError(f"{topic}: expected {type_name}, got {types.get(topic)}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(expected)))
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        msg = deserialize_message(data, expected[topic][1])
        if topic == "/camera/color/image_raw/compressed" and image is None:
            image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode compressed color image")
            image_stamp = stamp * 1e-9
        elif topic == "/livox/avia/lidar":
            scans.append((stamp * 1e-9, msg))
        if image is not None and len(scans) >= 3:
            break

    if image is None or not scans:
        raise RuntimeError("Could not read D435i image and Avia scan from bag")
    scan_stamp, scan = min(scans, key=lambda item: abs(item[0] - image_stamp))
    points_a = np.asarray([(p.x, p.y, p.z) for p in scan.points], dtype=np.float64)
    points_c = points_a @ R_AC.T + T_AC
    z = points_c[:, 2]
    uvw = points_c @ K.T
    u = uvw[:, 0] / z
    v = uvw[:, 1] / z
    height, width = image.shape[:2]
    front = z > 0.01
    in_image = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    print(f"image={width}x{height}, image_time={image_stamp:.6f}")
    print(f"scan_points={len(points_a)}, scan_time={scan_stamp:.6f}, dt={scan_stamp-image_stamp:+.6f}s")
    print(f"in_front={int(front.sum())}, projected_in_image={int(in_image.sum())}")
    if in_image.any():
        print("depth_median=%.3f, pixel_u=[%.1f, %.1f], pixel_v=[%.1f, %.1f]" % (
            np.median(z[in_image]), u[in_image].min(), u[in_image].max(),
            v[in_image].min(), v[in_image].max()))


if __name__ == "__main__":
    main()
