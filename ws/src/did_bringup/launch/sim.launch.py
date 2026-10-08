"""Мир turtlebot3_world и TurtleBot3 Burger из turtlebot3_gazebo без изменений.

Отличие от штатного turtlebot3_world.launch.py одно: окно Gazebo включается флагом gui,
чтобы прогоны и тесты шли без него.
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    tb3_launch = os.path.join(tb3_gazebo, 'launch')
    gz_sim_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')

    gui = LaunchConfiguration('gui')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # На Linux без дисплея лидар рендерится через EGL; с дисплеем и на macOS флаг не нужен.
    headless = sys.platform.startswith('linux') and not os.environ.get('DISPLAY')
    server_args = '-r -s -v2 ' + ('--headless-rendering ' if headless else '')

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={'gz_args': [server_args, world], 'on_exit_shutdown': 'true'}.items(),
    )

    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
        condition=IfCondition(gui),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch, 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false', description='Открыть окно Gazebo'),
        # Старт из условия задачи: мировая поза = (-2.0; -0.5) + одометрия.
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models')),
        gz_server,
        gz_gui,
        spawn_turtlebot,
        robot_state_publisher,
    ])
