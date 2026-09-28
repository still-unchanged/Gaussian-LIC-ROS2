#!/usr/bin/env python3
"""Synthetic ROS1/ROS2 conversion checks; needs rosbags==0.11.5 and numpy."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('converter', root/'fastlivo2_to_ros2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = get_typestore(Stores.ROS1_NOETIC)
for driver in ('livox_ros_driver', 'livox_ros_driver2'):
    source.register(get_types_from_msg('uint32 offset_time\nfloat32 x\nfloat32 y\nfloat32 z\nuint8 reflectivity\nuint8 tag\nuint8 line\n', driver+'/msg/CustomPoint'))
    source.register(get_types_from_msg('std_msgs/Header header\nuint64 timebase\nuint32 point_num\nuint8 lidar_id\nuint8[3] rsvd\n'+driver+'/CustomPoint[] points\n', driver+'/msg/CustomMsg'))
source.register(get_types_from_msg('std_msgs/Header header\nfloat64 value\n', 'dataset/msg/Extra'))
source.register(get_types_from_msg('geometry_msgs/TransformStamped[] transforms\n', 'tf/msg/tfMessage'))
t = source.types
header = t['std_msgs/msg/Header'](37, t['builtin_interfaces/msg/Time'](123, 456), 'camera')
compressed = t['sensor_msgs/msg/CompressedImage'](header, 'jpeg', np.array([1, 2, 3], np.uint8))
cloud = t['sensor_msgs/msg/PointCloud2'](header, 1, 0, [], False, 0, 0, np.array([], np.uint8), True)
extra = t['dataset/msg/Extra'](header, 2.5)

with tempfile.TemporaryDirectory(prefix='fastlivo2_converter_') as d:
    for driver in ('livox_ros_driver', 'livox_ros_driver2', None):
        label = driver or 'pointcloud_only'
        bag, out = Path(d)/(label+'.bag'), Path(d)/(label+'_ros2')
        entries = [('/renamed/image/compressed', compressed), ('/sensor/cloud', cloud),
                   ('/additional/data', extra), ('/tf', t['tf/msg/tfMessage']([]))]
        if driver:
            point = t[driver+'/msg/CustomPoint'](4000000000, 1., 2., 3., 255, 48, 5)
            msg = t[driver+'/msg/CustomMsg'](header, 1700000000000000001, 1, 2,
                                           np.array([1, 2, 255], np.uint8), [point])
            entries.append(('/renamed/lidar', msg))
        with Writer(bag) as writer:
            for i, (topic, message) in enumerate(entries):
                connection = writer.add_connection(topic, message.__msgtype__, typestore=source)
                writer.write(connection, 1000000000+i, source.serialize_ros1(message, message.__msgtype__))
            # A second ROS1 publisher on the same topic/type must not lose messages.
            connection = writer.add_connection(entries[0][0], compressed.__msgtype__, typestore=source,
                                               callerid='/second_camera')
            writer.write(connection, 2000000000, source.serialize_ros1(compressed, compressed.__msgtype__))
        module.convert(bag, out)
        target = get_typestore(Stores.ROS2_HUMBLE)
        with Reader(out) as reader:
            assert reader.message_count == len(entries)+1
            assert {c.topic for c in reader.connections} == {topic for topic, _ in entries}
            for c in reader.connections:
                target.register(get_types_from_msg(c.msgdef.data, c.msgtype))
            observed = []
            for c, stamp, raw in reader.messages():
                observed.append(stamp)
                decoded = target.deserialize_cdr(raw, c.msgtype)
                if hasattr(decoded, 'header'):
                    assert decoded.header.stamp.sec == 123 and decoded.header.stamp.nanosec == 456
                    assert decoded.header.frame_id == 'camera'
                if c.topic == '/renamed/lidar':
                    assert c.msgtype == 'cocolic_interfaces/msg/LivoxCustomMsg'
                    assert decoded.timebase == 1700000000000000001 and decoded.point_num == 1
                    assert decoded.lidar_id == 2
                    np.testing.assert_array_equal(decoded.rsvd, [1, 2, 255])
                    p = decoded.points[0]
                    assert (p.offset_time, p.x, p.y, p.z, p.reflectivity, p.tag, p.line) == (4000000000, 1., 2., 3., 255, 48, 5)
                elif c.topic == '/renamed/image/compressed':
                    assert decoded.format == 'jpeg'
                    np.testing.assert_array_equal(decoded.data, compressed.data)
                elif c.topic == '/additional/data':
                    assert decoded.value == 2.5
            assert observed == [1000000000+i for i in range(len(entries))]+[2000000000]
        try:
            module.convert(bag, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output overwritten')
        print('PASS:', label)
print('PASS: arbitrary topics, compressed images, PointCloud2, TF, custom types, multiple publishers, exact timestamps and overwrite protection')
