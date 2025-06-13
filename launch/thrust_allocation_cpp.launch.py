from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    config = os.path.join(get_package_share_directory('thrust_allocation'), 'config', 'thrust_allocation_config.yaml')
    allocator_mode = DeclareLaunchArgument(
        'allocator',
        default_value='pseudo_inverse',
        description='Allocator mode to use for thrust allocation',
        choices=['pseudo_inverse', 'nlp', 'qp', 'maneuvering'],
    )
    return LaunchDescription([
        allocator_mode,
        Node(
            package='thrust_allocation',
            executable='thrust_allocation_node',
            name='thrust_allocation_node_cpp',
            output='screen',
            parameters=[config, {'allocator': LaunchConfiguration('allocator')}]
        ),
    ])