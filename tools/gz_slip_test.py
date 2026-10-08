#!/usr/bin/env python3
"""Проскальзывание колёс в Gazebo: насколько одометрия врёт при резких разворотах.

    pixi run stand level:=easy seed:=1          # в другом окне
    pixi run python tools/gz_slip_test.py

Подаёт на /cmd_vel серии разворотов (ступенькой и с ограничением углового ускорения) и после
каждой, когда робот встал, сравнивает поворот по одометрии с истинным (/did/gz/dynamic_pose).
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE = 20.0      # Гц, частота команд

# (название, [(длительность, v, w, предел углового ускорения или None — ступенькой)], ускорение остановки)
A = None
TESTS = [
    ('разворот +1,9 ступенькой туда и обратно в ноль', [(1.5, 0, 1.9, A)], A),
    ('разворот −1,9 ступенькой', [(1.5, 0, -1.9, A)], A),
    ('старт 1,9 ступенькой, плавная остановка', [(1.5, 0, 1.9, A)], 1.0),
    ('плавный старт до 1,9, остановка ступенькой', [(3.0, 0, 1.9, 1.0)], A),
    ('старт 1,0 ступенькой, плавная остановка', [(1.5, 0, 1.0, A)], 1.0),
    ('старт 0,5 ступенькой, плавная остановка', [(1.5, 0, 0.5, A)], 1.0),
    ('реверс −1,9 → +1,9 ступенькой', [(1.0, 0, -1.9, A), (1.0, 0, 1.9, A)], A),
    ('реверс −1,0 → +1,0 ступенькой', [(1.0, 0, -1.0, A), (1.0, 0, 1.0, A)], A),
    ('реверс ±1,9, ускорение не больше 6 рад/с²', [(1.2, 0, -1.9, 6.0), (1.4, 0, 1.9, 6.0)], 6.0),
    ('реверс ±1,9, ускорение не больше 3 рад/с²', [(1.5, 0, -1.9, 3.0), (2.0, 0, 1.9, 3.0)], 3.0),
    ('реверс ±1,9, ускорение не больше 1,5 рад/с²', [(2.0, 0, -1.9, 1.5), (3.5, 0, 1.9, 1.5)], 1.5),
    ('реверс ±1,0, ускорение не больше 3 рад/с²', [(1.2, 0, -1.0, 3.0), (1.6, 0, 1.0, 3.0)], 3.0),
    ('реверс ±1,0, ускорение не больше 1,5 рад/с²', [(1.5, 0, -1.0, 1.5), (2.2, 0, 1.0, 1.5)], 1.5),
    ('как у агента: −1,9 на месте, затем сразу дуга влево',
     [(0.5, 0, -1.9, A), (0.1, 0.13, 1.17, A), (0.1, 0.15, 0.97, A), (0.1, 0.15, 0.89, A), (0.1, 0.16, 0.73, A),
      (0.1, 0.17, 0.61, A), (0.1, 0.18, 0.50, A), (0.1, 0.19, 0.42, A), (0.6, 0.2, 0.0, A)], A),
    ('как у агента: −1,9, дуга, +1,9, −1,9',
     [(0.6, 0, -1.9, A), (0.1, 0.07, -1.9, A), (0.1, 0.11, -1.47, A), (0.1, 0.13, -1.22, A), (0.1, 0.15, -0.91, A),
      (0.5, 0, 1.9, A), (0.2, 0, -1.9, A), (0.5, 0.15, -1.0, A), (1.0, 0.2, 0.2, A)], A),
    ('дуга 0,15 м/с: реверс ±1,0 ступенькой', [(1.0, 0.15, -1.0, A), (1.0, 0.15, 1.0, A)], A),
    ('дуга 0,15 м/с: реверс ±1,0, не больше 3 рад/с²', [(1.2, 0.15, -1.0, 3.0), (1.6, 0.15, 1.0, 3.0)], 3.0),
    ('вперёд 0,22 ступенькой и стоп', [(2.0, 0.22, 0.0, A)], A),
    ('назад 0,10 ступенькой и стоп', [(2.0, -0.10, 0.0, A)], A),
]


class Probe(Node):

    def __init__(self):
        super().__init__('did_slip_test')
        self.clock = None
        self.odom = None            # (x, y, накопленный курс)
        self.truth = None
        self._last = {}
        self._idx = None
        self.slip_max = 0.0
        self.truth_w = 0.0
        self._truth_prev = None
        self.pitch = (0.0, 0.0)
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/did/gz/dynamic_pose', self.on_truth, qos_profile_sensor_data)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    def on_clock(self, msg):
        self.clock = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _turn(self, key, yaw):
        prev, total = self._last.get(key, (yaw, 0.0))
        total += (yaw - prev + math.pi) % (2 * math.pi) - math.pi
        self._last[key] = (yaw, total)
        return total

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (p.x, p.y, self._turn('o', yaw))

    def on_truth(self, msg):
        if self._idx is None:
            self._idx = int(np.argmax([abs(t.transform.translation.x) + abs(t.transform.translation.y)
                                       for t in msg.transforms]))
        tr = msg.transforms[self._idx].transform
        q = tr.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.pitch = (min(self.pitch[0], pitch), max(self.pitch[1], pitch))
        self.truth = (tr.translation.x, tr.translation.y, self._turn('g', yaw))
        if self._truth_prev and self.clock and self.clock - self._truth_prev[0] >= 0.05:
            self.truth_w = (self.truth[2] - self._truth_prev[1]) / (self.clock - self._truth_prev[0])
            self._truth_prev = (self.clock, self.truth[2])
        elif not self._truth_prev and self.clock:
            self._truth_prev = (self.clock, self.truth[2])

    def send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def spin_for(self, sim_s, v=0.0, w=0.0, aw=None, state=None):
        """Держать команду sim_s секунд времени симуляции; state — текущая угловая команда (для плавности)."""
        end = self.clock + sim_s
        cur = state if state is not None else w
        next_cmd = 0.0
        while self.clock < end:
            rclpy.spin_once(self, timeout_sec=0.005)
            now = time.monotonic()
            if now >= next_cmd:
                next_cmd = now + 1.0 / RATE / 0.7       # симуляция идёт примерно в 0,7 реального времени
                if aw is None:
                    cur = w
                else:
                    step = aw / RATE
                    cur += min(max(w - cur, -step), step)
                self.send(v, cur)
                if abs(cur) > 0.05 or abs(self.truth_w) > 0.05:
                    self.slip_max = max(self.slip_max, abs(self.truth_w - cur))
        return cur


def main():
    rclpy.init()
    node = Probe()
    while node.clock is None or node.odom is None or node.truth is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.clock < 12.0:
        node.spin_for(12.0 - node.clock)                # робот оседает после появления в мире
    print(f'{"манёвр":<52}{"поворот одом.":>14}{"истинный":>10}{"уход курса":>12}{"уход, см":>10}{"тангаж":>16}')
    only = sys.argv[1:]
    try:
        for name, segments, stop_aw in TESTS:
            if only and not any(key in name for key in only):
                continue
            node.spin_for(1.0)
            o0, g0 = node.odom, node.truth
            node.slip_max, node.pitch = 0.0, (0.0, 0.0)
            cur = 0.0
            for dur, v, w, aw in segments:
                cur = node.spin_for(dur, v, w, aw, state=cur)
            if stop_aw:
                cur = node.spin_for(abs(cur) / stop_aw + 0.2, 0.0, 0.0, stop_aw, state=cur)
            node.spin_for(2.0)
            o1, g1 = node.odom, node.truth
            do, dg = o1[2] - o0[2], g1[2] - g0[2]
            # Смещение сравниваем в системе начала манёвра, чтобы прежний уход курса не мешал.
            shift_o = math.hypot(o1[0] - o0[0], o1[1] - o0[1])
            shift_g = math.hypot(g1[0] - g0[0], g1[1] - g0[1])
            print(f'{name:<52}{do:>+14.3f}{dg:>+10.3f}{do - dg:>+12.3f}{(shift_o - shift_g) * 100:>+10.1f}'
                  f'{node.pitch[0]:>+9.3f}{node.pitch[1]:>+7.3f}', flush=True)
    finally:
        for _ in range(5):
            node.send(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
