#!/usr/bin/env python3
"""Проверка уровня 0: симуляция жива, данные идут, робот слушается /cmd_vel.

Запуск при работающей симуляции: pixi run smoke
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

WAIT_DATA_S = 120.0   # первый запуск качает модели пола и света из Fuel
RTF_WINDOW_S = 5.0
DRIVE_SPEED = 0.15    # м/с, у Burger предел 0.22
DRIVE_SIM_S = 4.0
MIN_TRAVEL_M = 0.4


class Smoke(Node):

    def __init__(self):
        super().__init__('did_smoke')
        self.sim_time = None
        self.odom_xy = None
        self.scan_stamps = []
        self.scan_valid = 0
        self.scan_total = 0
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    def on_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_xy = (p.x, p.y)

    def on_scan(self, msg):
        self.scan_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.scan_total = len(msg.ranges)
        self.scan_valid = sum(1 for r in msg.ranges if math.isfinite(r) and r > 0.0)

    def spin_for(self, wall_seconds):
        end = time.monotonic() + wall_seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def drive(self, speed):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = speed
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = Smoke()
    ok = True

    deadline = time.monotonic() + WAIT_DATA_S
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.sim_time and node.odom_xy and node.scan_stamps:
            break
    missing = [name for name, got in (
        ('/clock', node.sim_time), ('/odom', node.odom_xy), ('/scan', node.scan_stamps)) if not got]
    if missing:
        print(f'FAIL: за {WAIT_DATA_S:.0f} с нет данных в {", ".join(missing)}')
        return 1
    print(f'данные идут: /clock, /odom, /scan; одометрия на старте = '
          f'({node.odom_xy[0]:+.3f}; {node.odom_xy[1]:+.3f})')

    sim0, wall0, scans0 = node.sim_time, time.monotonic(), len(node.scan_stamps)
    node.spin_for(RTF_WINDOW_S)
    sim_dt = node.sim_time - sim0
    rtf = sim_dt / (time.monotonic() - wall0)
    scan_hz = (len(node.scan_stamps) - scans0) / sim_dt if sim_dt > 0 else 0.0
    print(f'real-time factor = {rtf:.2f}; /scan = {scan_hz:.1f} Гц (по времени симуляции), '
          f'валидных лучей {node.scan_valid}/{node.scan_total}')
    if rtf < 0.1 or node.scan_valid == 0:
        ok = False

    start_xy, sim_start = node.odom_xy, node.sim_time
    wall_limit = time.monotonic() + 120.0
    while node.sim_time - sim_start < DRIVE_SIM_S and time.monotonic() < wall_limit:
        node.drive(DRIVE_SPEED)
        node.spin_for(0.1)
    for _ in range(5):
        node.drive(0.0)
        node.spin_for(0.1)
    travel = math.dist(start_xy, node.odom_xy)
    print(f'езда {DRIVE_SIM_S:.0f} с на {DRIVE_SPEED} м/с: одометрия '
          f'({start_xy[0]:+.3f}; {start_xy[1]:+.3f}) -> ({node.odom_xy[0]:+.3f}; {node.odom_xy[1]:+.3f}), '
          f'путь {travel:.2f} м')
    if travel < MIN_TRAVEL_M:
        ok = False

    print('OK' if ok else 'FAIL')
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
