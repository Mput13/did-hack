"""Два TurtleBot3 Burger в мире turtlebot3_world (исследование M1): tb1 на базе, tb2 рядом.

Модель робота берётся из turtlebot3_gazebo без изменений, кроме имён топиков: у штатной модели они
общие (cmd_vel, odom, scan…), и два робота слушали бы одну команду. Здесь каждому подставляется своё
пространство имён, и для каждого поднимается свой мост ros_gz:
    /tb1/cmd_vel, /tb1/odom, /tb1/scan, /tb1/imu, /tb1/joint_states, /tb1/tf — и то же для /tb2.
Основной стенд (sim.launch.py, stand.launch.py) не затрагивается.

    ros2 launch ws/src/did_bringup/launch/team.launch.py gui:=true

Запускается по пути к файлу — пересборка воркспейса не нужна.
"""
import os
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Места на базе — те же, что в did/judge_team.py (SLOTS): одометрия каждого робота отсчитывается от своего.
ROBOTS = (('tb1', -2.0, -0.5), ('tb2', -2.0, -0.1))
TOPICS = (('<topic>imu</topic>', '<topic>{ns}/imu</topic>'),
          ('<topic>scan</topic>', '<topic>{ns}/scan</topic>'),
          ('<topic>cmd_vel</topic>', '<topic>{ns}/cmd_vel</topic>'),
          ('<odom_topic>odom</odom_topic>', '<odom_topic>{ns}/odom</odom_topic>'),
          ('<tf_topic>/tf</tf_topic>', '<tf_topic>{ns}/tf</tf_topic>'),
          ('<topic>joint_states</topic>', '<topic>{ns}/joint_states</topic>'))
BRIDGE = (('odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry', 'GZ_TO_ROS'),
          ('scan', 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan', 'GZ_TO_ROS'),
          ('imu', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU', 'GZ_TO_ROS'),
          ('joint_states', 'sensor_msgs/msg/JointState', 'gz.msgs.Model', 'GZ_TO_ROS'),
          ('cmd_vel', 'geometry_msgs/msg/TwistStamped', 'gz.msgs.Twist', 'ROS_TO_GZ'))


def _robot_files(sdf_text, name, out_dir, with_clock):
    """Модель робота со своими топиками и описание моста для него."""
    for old, new in TOPICS:
        if old not in sdf_text:
            raise RuntimeError(f'в модели turtlebot3_burger нет строки {old}: модель изменилась')
    text = sdf_text
    for old, new in TOPICS:
        text = text.replace(old, new.format(ns=f'/{name}'))
    sdf = os.path.join(out_dir, f'{name}.sdf')
    with open(sdf, 'w') as f:
        f.write(text)
    rows = []
    if with_clock:
        rows.append(('clock', 'clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock', 'GZ_TO_ROS'))
    rows += [(f'/{name}/{t}', f'/{name}/{t}', ros, gz, way) for t, ros, gz, way in BRIDGE]
    bridge = os.path.join(out_dir, f'{name}_bridge.yaml')
    with open(bridge, 'w') as f:
        for ros_topic, gz_topic, ros, gz, way in rows:
            f.write(f'- ros_topic_name: "{ros_topic}"\n  gz_topic_name: "{gz_topic}"\n  ros_type_name: "{ros}"\n'
                    f'  gz_type_name: "{gz}"\n  direction: {way}\n')
    return sdf, bridge


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    gz_sim_launch = os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')
    with open(os.path.join(tb3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')) as f:
        sdf_text = f.read()
    out_dir = tempfile.mkdtemp(prefix='did_team_')

    headless = sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
    server_args = '-r -s -v2 ' + ('--headless-rendering ' if headless else '')
    actions = [
        DeclareLaunchArgument('gui', default_value='false', description='Открыть окно Gazebo'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_sim_launch),
                                 launch_arguments={'gz_args': [server_args, world], 'on_exit_shutdown': 'true'}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_sim_launch),
                                 launch_arguments={'gz_args': '-g -v2 '}.items(),
                                 condition=IfCondition(LaunchConfiguration('gui'))),
    ]
    for i, (name, x, y) in enumerate(ROBOTS):
        sdf, bridge = _robot_files(sdf_text, name, out_dir, with_clock=i == 0)
        actions.append(Node(package='ros_gz_sim', executable='create', output='screen',
                            arguments=['-name', name, '-file', sdf, '-x', str(x), '-y', str(y), '-z', '0.01']))
        actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name=f'{name}_bridge',
                            output='screen', arguments=['--ros-args', '-p', f'config_file:={bridge}']))
    return LaunchDescription(actions)
