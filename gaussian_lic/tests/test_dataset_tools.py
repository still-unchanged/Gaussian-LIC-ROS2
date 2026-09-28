#!/usr/bin/env python3
"""Run after sourcing the ROS2 workspace: python3 tests/test_dataset_tools.py."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from cocolic_interfaces.msg import LivoxCustomMsg, LivoxCustomPoint
from rclpy.serialization import serialize_message

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adapter', root/'scripts/m3dgr_adapter.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pose_out, depth_out = [], []
context = SimpleNamespace(
    bridge=CvBridge(), depth_scale=1000.0,
    pose_pub=SimpleNamespace(publish=pose_out.append),
    depth_pub=SimpleNamespace(publish=depth_out.append))
odom = Odometry()
odom.header.frame_id = 'map'
odom.header.stamp.sec = 12
odom.pose.pose.position.x = 3.5
module.M3DGRAdapter.odom_callback(context, odom)
assert pose_out[-1].header == odom.header
assert pose_out[-1].pose == odom.pose.pose
pose = PoseStamped(header=odom.header, pose=odom.pose.pose)
module.M3DGRAdapter.pose_callback(context, pose)
assert pose_out[-1] == pose
for values, encoding, expected in (
        ([[0, 1000, 2500]], '16UC1', [[0., 1., 2.5]]),
        ([[float('nan'), float('inf'), -1., 2.5]], '32FC1', [[0., 0., 0., 2.5]])):
    dtype = np.uint16 if encoding == '16UC1' else np.float32
    msg = context.bridge.cv2_to_imgmsg(np.array(values, dtype=dtype), encoding=encoding)
    msg.header = odom.header
    module.M3DGRAdapter.depth_callback(context, msg)
    out = depth_out[-1]
    assert out.encoding == '32FC1' and out.header == msg.header
    np.testing.assert_allclose(context.bridge.imgmsg_to_cv2(out), expected)
print('PASS: Odometry/PoseStamped, depth units, invalid depth and headers')

with tempfile.TemporaryDirectory(prefix='m3dgr_tools_') as directory:
    bag = Path(directory)/'bag'
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    image_topic = '/camera/color/image_raw/compressed'
    lidar_topic = '/livox/avia/lidar'
    for topic, kind in ((image_topic, 'sensor_msgs/msg/CompressedImage'),
                        (lidar_topic, 'cocolic_interfaces/msg/LivoxCustomMsg')):
        writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=kind, serialization_format='cdr'))
    image = CompressedImage(format='png')
    image.data = cv2.imencode('.png', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    writer.write(image_topic, serialize_message(image), 1_000_000_000)
    for i in range(3):
        scan = LivoxCustomMsg()
        scan.points = [LivoxCustomPoint(x=2.0, y=0.0, z=0.0)]
        scan.point_num = 1
        writer.write(lidar_topic, serialize_message(scan), 1_000_000_000+i*10_000_000)
    del writer
    result = subprocess.run([sys.executable, str(root/'tools/inspect_m3dgr_avia_projection.py'), str(bag)],
                            check=True, text=True, capture_output=True)
    assert 'projected_in_image=1' in result.stdout, result.stdout
    assert 'dt=+0.000000s' in result.stdout, result.stdout
print('PASS: ROS2 bag decoding and calibrated Avia projection')
