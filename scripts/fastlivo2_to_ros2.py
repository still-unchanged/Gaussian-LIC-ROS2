#!/usr/bin/env python3
"""Convert FAST-LIVO2 ROS1 datasets to ROS2 Humble without filtering topics.

Requires: python -m pip install rosbags==0.11.5
Usage: python scripts/fastlivo2_to_ros2.py INPUT.bag [OUTPUT_DIRECTORY]
The default output is INPUT's sibling directory named <stem>_ros2.
Existing output directories are never overwritten. A failed conversion may leave
an incomplete output directory; use a different output path after fixing the error.
"""
import argparse
from collections import Counter
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


TYPE_RENAMES = {
    'livox_ros_driver/msg/CustomMsg': 'cocolic_interfaces/msg/LivoxCustomMsg',
    'livox_ros_driver/msg/CustomPoint': 'cocolic_interfaces/msg/LivoxCustomPoint',
    'livox_ros_driver2/msg/CustomMsg': 'cocolic_interfaces/msg/LivoxCustomMsg',
    'livox_ros_driver2/msg/CustomPoint': 'cocolic_interfaces/msg/LivoxCustomPoint',
    'tf/msg/tfMessage': 'tf2_msgs/msg/TFMessage',
}


def rename(value):
    if isinstance(value, str):
        return TYPE_RENAMES.get(value, value)
    if isinstance(value, tuple):
        return tuple(rename(item) for item in value)
    if isinstance(value, list):
        return [rename(item) for item in value]
    return value


def convert(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
    ros1 = get_typestore(Stores.ROS1_NOETIC)
    cdr_store = get_typestore(Stores.ROS2_HUMBLE)
    ros2 = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(source) as reader:
        source_definitions = {}
        for connection in reader.connections:
            definitions = get_types_from_msg(connection.msgdef.data, connection.msgtype)
            ros1.register(definitions)  # Reject conflicting schemas across connections.
            source_definitions.update(definitions)
        # Conversion uses ROS2 Header (without seq), but original custom type names.
        # All other existing ROS2 message layouts must match the recorded layout.
        for name, definition in source_definitions.items():
            if name == 'std_msgs/msg/Header':
                continue
            if name in cdr_store.fielddefs and definition[1] != cdr_store.fielddefs[name][1]:
                raise ValueError(f'ROS1/ROS2 field layout differs for {name}; explicit conversion needed')
        cdr_store.register({name: definition for name, definition in source_definitions.items()
                            if name not in cdr_store.fielddefs})
        for name, definition in source_definitions.items():
            if name != 'std_msgs/msg/Header':
                ros2.register({TYPE_RENAMES.get(name, name): rename(definition)})
        interfaces = Path(__file__).resolve().parents[1] / 'third_party/cocolic_interfaces/msg'
        for typename in set(TYPE_RENAMES.values()):
            if not typename.startswith('cocolic_interfaces/') or typename not in ros2.fielddefs:
                continue
            name = typename.rsplit('/', 1)[1]
            definitions = get_types_from_msg((interfaces / f'{name}.msg').read_text(), typename)
            if definitions[typename] != ros2.fielddefs[typename]:
                raise ValueError(f'Livox schema does not match project interface: {typename}')
        builtins = get_typestore(Stores.ROS2_HUMBLE).fielddefs
        custom = sorted({TYPE_RENAMES.get(c.msgtype, c.msgtype) for c in reader.connections}
                        - set(builtins) - set(TYPE_RENAMES.values()))
        if custom:
            print('Additional custom types retained; consumers need their ROS2 interfaces: '
                  + ', '.join(custom), flush=True)
        counts = Counter()
        expected_counts = Counter()
        for c in reader.connections:
            expected_counts[c.topic] += c.msgcount
        with Writer(destination, version=8) as writer:
            connections, by_topic_type = {}, {}
            for c in reader.connections:
                key = (c.topic, TYPE_RENAMES.get(c.msgtype, c.msgtype))
                # Multiple ROS1 publishers may share a topic/type.
                if key not in by_topic_type:
                    by_topic_type[key] = writer.add_connection(*key, typestore=ros2)
                connections[c.id] = by_topic_type[key]
            for index, (connection, timestamp, raw) in enumerate(reader.messages(), 1):
                # CDR encodes field values, not package/type names. Renaming the
                # Livox types therefore needs no point-wise copying or truncation.
                cdr = cdr_store.ros1_to_cdr(raw, connection.msgtype)
                writer.write(connections[connection.id], timestamp, cdr)
                counts[connection.topic] += 1
                if index % 5000 == 0:
                    print(f'Converted {index}/{reader.message_count} messages', flush=True)
        if counts != expected_counts:
            raise RuntimeError(f'Message count mismatch: {counts} != {expected_counts}')
    print(f'Converted: {source}\nOutput: {destination}')
    for topic, count in sorted(counts.items()):
        print(f'  {topic}: {count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Any FAST-LIVO2 ROS1 .bag file')
    parser.add_argument('output', type=Path, nargs='?', help='New ROS2 bag directory')
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    destination = (args.output.expanduser().resolve() if args.output else
                   source.with_name(source.stem + '_ros2'))
    convert(source, destination)


if __name__ == '__main__':
    main()
