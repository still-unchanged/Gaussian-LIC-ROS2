"""Original offline LIC/R3LIVE pipeline using ROS2 messages and rosbag2."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    share = FindPackageShare('cocolic_ros2')
    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=PathJoinSubstitution([share, 'config', 'ct_odometry_fastlivo2.yaml'])),
        DeclareLaunchArgument('bag_path', description='Absolute path to the converted ROS2 bag directory'),
        DeclareLaunchArgument('trajectory_output', default_value='/tmp/Retail_Street_ros2'),
        DeclareLaunchArgument('bag_start', default_value='0.0'),
        DeclareLaunchArgument('bag_duration', default_value='-1.0'),
        Node(package='cocolic_ros2', executable='cocolic_original_odometry_node', name='cocolic', output='screen', parameters=[{
            'project_path': share,
            'config_path': LaunchConfiguration('config_path'),
            'bag_path': LaunchConfiguration('bag_path'),
            'trajectory_output': LaunchConfiguration('trajectory_output'),
            'bag_start': ParameterValue(LaunchConfiguration('bag_start'), value_type=float),
            'bag_duration': ParameterValue(LaunchConfiguration('bag_duration'), value_type=float),
        }]),
    ])
