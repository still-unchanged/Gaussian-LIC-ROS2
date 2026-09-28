"""Launch Gaussian-LIC with parameters matching the M3DGR dataset."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the M3DGR Gaussian-LIC launch description."""
    package_share = FindPackageShare('gaussian_lic')

    config_path = LaunchConfiguration('config_path')
    result_path = LaunchConfiguration('result_path')
    lpips_path = LaunchConfiguration('lpips_path')

    return LaunchDescription([
        DeclareLaunchArgument('point_topic', default_value='/m3dgr/lidar/points'),
        DeclareLaunchArgument('image_topic', default_value='/m3dgr/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/m3dgr/camera/depth/image_raw'),
        DeclareLaunchArgument('pose_topic', default_value='/m3dgr/odom'),
        DeclareLaunchArgument('pose_type', default_value='odometry'),
        DeclareLaunchArgument('depth_scale', default_value='1000.0'),
        Node(
            package='gaussian_lic', executable='m3dgr_adapter.py',
            name='m3dgr_adapter', output='screen',
            parameters=[{
                'pose_topic': LaunchConfiguration('pose_topic'),
                'pose_type': LaunchConfiguration('pose_type'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'depth_scale': ParameterValue(LaunchConfiguration('depth_scale'), value_type=float),
            }],
        ),
        DeclareLaunchArgument(
            'config_path',
            default_value=PathJoinSubstitution([package_share, 'config', 'm3dgr.yaml']),
            description='Absolute path to the Gaussian-LIC YAML configuration file.',
        ),
        DeclareLaunchArgument(
            'result_path',
            default_value=PathJoinSubstitution(
                [EnvironmentVariable('HOME'), '.ros', 'gaussian_lic', 'result']
            ),
            description='Directory where the generated map and evaluation outputs are written.',
        ),
        DeclareLaunchArgument(
            'lpips_path',
            default_value=PathJoinSubstitution([package_share, 'lpips']),
            description='Directory containing lpips_alex.pt.',
        ),
        Node(
            package='gaussian_lic',
            executable='gs_mapping',
            name='gs_mapping',
            remappings=[
                ('/points_for_gs', LaunchConfiguration('point_topic')),
                ('/image_for_gs', LaunchConfiguration('image_topic')),
            ],
            output='screen',
            parameters=[{
                'config_path': config_path,
                'result_path': result_path,
                'lpips_path': lpips_path,
            }],
        ),
    ])
