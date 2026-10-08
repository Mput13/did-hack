"""Стенд: симуляция из sim.launch.py, судья с интерфейсом /did/* и показ скрытой правды.

Примеры:
    ros2 launch did_bringup stand.launch.py level:=hard seed:=3
    ros2 launch did_bringup stand.launch.py gui:=true rviz:=true show_truth:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup = get_package_share_directory('did_bringup')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'sim.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items(),
    )

    # Истинные позы подвижных моделей — только судье, под служебным именем: агент их не читает.
    truth_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='did_truth_bridge',
        arguments=['/world/default/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'],
        remappings=[('/world/default/dynamic_pose/info', '/did/gz/dynamic_pose')],
        output='screen',
    )

    judge = Node(
        package='did_ros',
        executable='judge',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'level': ParameterValue(LaunchConfiguration('level'), value_type=str),
            'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int),
            'scenario_file': ParameterValue(LaunchConfiguration('scenario_file'), value_type=str),
            'use_ground_truth': ParameterValue(
                LaunchConfiguration('use_ground_truth'), value_type=bool),
            'show_truth': ParameterValue(LaunchConfiguration('show_truth'), value_type=bool),
            'rules': ParameterValue(LaunchConfiguration('rules'), value_type=str),
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(bringup, 'rviz', 'did.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('level', default_value='easy', description='easy | medium | hard'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('scenario_file', default_value='',
                              description='JSON сценария; если задан, важнее level и seed'),
        DeclareLaunchArgument('gui', default_value='false', description='Открыть окно Gazebo'),
        DeclareLaunchArgument('rules', default_value='base',
                              description='base — прежние правила, science — с несколькими причинами расхода и сбоями'),
        DeclareLaunchArgument('rviz', default_value='false', description='Открыть RViz'),
        DeclareLaunchArgument('show_truth', default_value='true',
                              description='Показать скрытую правду в окне Gazebo'),
        DeclareLaunchArgument('use_ground_truth', default_value='true',
                              description='Поза робота из Gazebo, иначе /odom + база'),
        sim,
        truth_bridge,
        judge,
        rviz,
    ])
