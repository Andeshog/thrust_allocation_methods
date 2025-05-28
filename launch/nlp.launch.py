from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(get_package_share_directory('thrust_allocation'), 'config', 'thrust_allocation_config.yaml')
    topics = os.path.join(get_package_share_directory('ma_config'), 'config', 'topics.yaml')

    return LaunchDescription([
        Node(
            package='thrust_allocation',
            executable='nlp_node.py',
            name='thrust_allocation_node',
            output='screen',
            parameters=[config, topics]
        ),
    ])