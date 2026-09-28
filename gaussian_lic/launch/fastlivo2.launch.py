"""Launch Gaussian-LIC with parameters matching the FAST-LIVO2 dataset."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the FAST-LIVO2 Gaussian-LIC launch description."""
    package_share = FindPackageShare('gaussian_lic')

    config_path = LaunchConfiguration('config_path')
    result_path = LaunchConfiguration('result_path')
    lpips_path = LaunchConfiguration('lpips_path')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_path',
            default_value=PathJoinSubstitution([package_share, 'config', 'fastlivo2.yaml']),
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
            output='screen',
            parameters=[{
                'config_path': config_path,
                'result_path': result_path,
                'lpips_path': lpips_path,
            }],
        ),
    ])
