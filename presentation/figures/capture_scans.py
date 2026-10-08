#!/usr/bin/env python3
"""Снимает лидар из работающей симуляции в нескольких точках и пишет JSON.

Данные идут в рисунок «лидар на карте»: проверка, что карта совпадает с миром Gazebo,
а мировая поза = старт + одометрия. Запуск: pixi run python presentation/figures/capture_scans.py out.json
"""
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

START = (-2.0, -0.5)      # мировая точка старта из условия
LEGS_M = [0.0, 1.0, 0.9]  # сколько проехать вперёд перед каждым снимком
SPEED = 0.15


class Capture(Node):

    def __init__(self):
        super().__init__('did_capture')
        self.odom = None
        self.scan = None
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (p.x, p.y, yaw)

    def on_scan(self, msg):
        self.scan = msg

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def command(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self.cmd_pub.publish(msg)

    def drive_forward(self, distance):
        x0, y0, _ = self.odom
        limit = time.monotonic() + 120.0
        while math.hypot(self.odom[0] - x0, self.odom[1] - y0) < distance and time.monotonic() < limit:
            self.command(SPEED, -1.5 * self.odom[2])  # держим нулевой курс
            self.spin_for(0.05)
        for _ in range(10):
            self.command(0.0, 0.0)
            self.spin_for(0.1)

    def snapshot(self):
        self.scan = None
        while self.scan is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        s, (ox, oy, yaw) = self.scan, self.odom
        return {
            'odom': [ox, oy, yaw],
            'world': [START[0] + ox, START[1] + oy, yaw],
            'angle_min': s.angle_min,
            'angle_increment': s.angle_increment,
            'range_min': s.range_min,
            'range_max': s.range_max,
            'ranges': [r if math.isfinite(r) else None for r in s.ranges],
        }


def main():
    out_path = sys.argv[1]
    rclpy.init()
    node = Capture()
    deadline = time.monotonic() + 120.0
    while (node.odom is None or node.scan is None) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.odom is None or node.scan is None:
        print('FAIL: нет /odom или /scan')
        return 1
    shots = []
    for leg in LEGS_M:
        if leg > 0.0:
            node.drive_forward(leg)
        node.spin_for(1.0)
        shots.append(node.snapshot())
        print(f'снимок {len(shots)}: мир = ({shots[-1]["world"][0]:+.3f}; {shots[-1]["world"][1]:+.3f}), '
              f'курс {math.degrees(shots[-1]["world"][2]):+.1f}°')
    with open(out_path, 'w') as f:
        json.dump({'start': START, 'shots': shots}, f)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
