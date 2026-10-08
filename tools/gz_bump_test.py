#!/usr/bin/env python3
"""Что происходит с одометрией в Gazebo, когда робот упирается в столб (исследование G2).

    ROS_DOMAIN_ID=31 GZ_PARTITION=did_g2 ./px python tools/gz_bump_test.py --out runs/g2_bump

Сам поднимает стенд (без окна), подводит робота к столбу по истинной позе (/did/gz/dynamic_pose) и
выполняет манёвры вслепую, как агент после потери положения: наезд в лоб, касание боком, отъезд
задом в столб, разворот на месте в упоре. После каждого манёвра печатает путь и поворот по
одометрии и на самом деле. Рядом пишется журнал поз и сканы (tools/gz_pose_log.py) — по ним
локализация проверяется без Gazebo.
"""
import argparse
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

PILLAR = (-1.1, -1.1)        # ближайший к базе столб, радиус 0,15 м
RATE = 20.0

# (название, точка подхода относительно столба, куда смотреть: точка относительно столба,
#  [(длительность, v, w)]) — после подхода команды подаются вслепую
TESTS = [
    ('наезд в лоб 0,20 м/с, 4 с', (-0.6, 0.0), (0.0, 0.0), [(4.0, 0.20, 0.0)]),
    ('в упоре: разворот +1,9, 1,5 с', None, None, [(1.5, 0.0, 1.9)]),
    ('в упоре: разворот −1,9, 1,5 с', None, None, [(1.5, 0.0, -1.9)]),
    ('в упоре: дуга 0,15 м/с и +1,0, 3 с', None, None, [(3.0, 0.15, 1.0)]),
    ('отъезд назад 0,10 м/с, 1,5 с', None, None, [(1.5, -0.10, 0.0)]),
    ('касание боком (мимо центра на 0,20 м), 4 с', (-0.6, 0.2), (0.6, 0.2), [(4.0, 0.20, 0.0)]),
    ('после касания: разворот −1,9, 1,5 с', None, None, [(1.5, 0.0, -1.9)]),
    ('после касания: вперёд 0,20, 2 с', None, None, [(2.0, 0.20, 0.0)]),
    ('задом в столб 0,10 м/с, 5 с', (-0.5, 0.0), (-1.5, 0.0), [(5.0, -0.10, 0.0)]),
    ('задом в упоре: разворот +1,9, 1,5 с', None, None, [(1.5, 0.0, 1.9)]),
    ('задом в упоре: вперёд 0,10, 1,5 с', None, None, [(1.5, 0.10, 0.0)]),
    ('рядом со столбом (зазор 2 см): разворот +1,9, 3 с', (-0.275, 0.0), (-0.275, 1.0), [(3.0, 0.0, 1.9)]),
]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def drive(out):
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from tf2_msgs.msg import TFMessage

    from did.config import BASE

    class Probe(Node):

        def __init__(self):
            super().__init__('did_bump_test')
            self.clock = self.odom = self.truth = None
            self._idx = None
            self.pitch = [0.0, 0.0]
            self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
            self.create_subscription(TFMessage, '/did/gz/dynamic_pose', self.on_truth, qos_profile_sensor_data)
            self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        def on_clock(self, msg):
            self.clock = msg.clock.sec + msg.clock.nanosec * 1e-9

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.odom = (BASE[0] + p.x, BASE[1] + p.y, yaw)

        def on_truth(self, msg):
            if self._idx is None:
                self._idx = int(np.argmax([abs(t.transform.translation.x) + abs(t.transform.translation.y)
                                           for t in msg.transforms]))
            tr = msg.transforms[self._idx].transform
            q = tr.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
            self.pitch = [min(self.pitch[0], pitch), max(self.pitch[1], pitch)]
            self.truth = (tr.translation.x, tr.translation.y, yaw)

        def send(self, v, w):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
            self.pub.publish(msg)

        def hold(self, sim_s, v=0.0, w=0.0):
            """Держать команду sim_s секунд времени симуляции; возвращает накопленные путь и поворот."""
            end = self.clock + sim_s
            next_cmd = 0.0
            acc = {'o': [0.0, 0.0], 'g': [0.0, 0.0]}
            last = {'o': self.odom, 'g': self.truth}
            while self.clock < end:
                rclpy.spin_once(self, timeout_sec=0.005)
                if time.monotonic() >= next_cmd:
                    next_cmd = time.monotonic() + 1.0 / RATE
                    self.send(v, w)
                for key, cur in (('o', self.odom), ('g', self.truth)):
                    p = last[key]
                    if cur is not p:
                        acc[key][0] += math.cos(p[2]) * (cur[0] - p[0]) + math.sin(p[2]) * (cur[1] - p[1])
                        acc[key][1] += _wrap(cur[2] - p[2])
                        last[key] = cur
            return acc

        def turn_to(self, th):
            for _ in range(400):
                err = _wrap(th - self.truth[2])
                if abs(err) < 0.02:
                    break
                self.hold(0.05, 0.0, max(-1.0, min(1.0, 2.0 * err)))
            self.hold(0.5)

        def go_to(self, x, y):
            for _ in range(2000):
                gx, gy, gth = self.truth
                dist = math.hypot(x - gx, y - gy)
                if dist < 0.03:
                    break
                err = _wrap(math.atan2(y - gy, x - gx) - gth)
                if abs(err) > 0.5:
                    self.hold(0.05, 0.0, max(-1.0, min(1.0, 2.0 * err)))
                else:
                    self.hold(0.05, min(0.15, 0.05 + dist), 1.5 * err)
            self.hold(0.5)

    rclpy.init()
    node = Probe()
    lines = []

    def say(text):
        print(text, flush=True)
        lines.append(text)

    try:
        t0 = time.monotonic()
        while node.clock is None or node.odom is None or node.truth is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - t0 > 120.0:
                raise RuntimeError('нет данных от стенда')
        if node.clock < 40.0:
            node.hold(40.0 - node.clock)                # робот оседает после появления в мире
        say(f'{"манёвр":<52}{"путь одом.":>11}{"истинный":>10}{"поворот одом.":>15}{"истинный":>10}'
            f'{"тангаж":>16}{"до столба, см":>15}')
        for name, start, look, segments in TESTS:
            if start is not None:
                # Отъехать от столба, прежде чем ехать к новой точке подхода.
                gx, gy, _ = node.truth
                if math.hypot(gx - PILLAR[0], gy - PILLAR[1]) < 0.4:
                    away = math.atan2(gy - PILLAR[1], gx - PILLAR[0])
                    node.go_to(PILLAR[0] + 0.6 * math.cos(away), PILLAR[1] + 0.6 * math.sin(away))
                node.go_to(PILLAR[0] + start[0], PILLAR[1] + start[1])
                gx, gy, _ = node.truth
                node.turn_to(math.atan2(PILLAR[1] + look[1] - gy, PILLAR[0] + look[0] - gx))
            node.hold(1.0)
            node.pitch = [0.0, 0.0]
            total = {'o': [0.0, 0.0], 'g': [0.0, 0.0]}
            for dur, v, w in segments:
                acc = node.hold(dur, v, w)
                for key in total:
                    total[key][0] += acc[key][0]
                    total[key][1] += acc[key][1]
            acc = node.hold(1.0)
            for key in total:
                total[key][0] += acc[key][0]
                total[key][1] += acc[key][1]
            gx, gy, _ = node.truth
            gap = math.hypot(gx - PILLAR[0], gy - PILLAR[1]) - 0.15
            say(f'{name:<52}{total["o"][0]:>+11.3f}{total["g"][0]:>+10.3f}{total["o"][1]:>+15.3f}'
                f'{total["g"][1]:>+10.3f}{node.pitch[0]:>+9.3f}{node.pitch[1]:>+7.3f}{gap * 100:>15.1f}')
    finally:
        for _ in range(5):
            node.send(0.0, 0.0)
            time.sleep(0.05)
        (out / 'bump-test.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    from gazebo_run import stop
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='runs/g2_bump')
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'stand.log', 'w')
    launch = ['ros2', 'launch', 'did_bringup', 'stand.launch.py', 'level:=easy', 'seed:=1', 'gui:=false',
              'rviz:=false', 'show_truth:=false', 'rules:=base']
    stand = subprocess.Popen(launch, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pose_log = subprocess.Popen([sys.executable, str(ROOT / 'tools' / 'gz_pose_log.py'), '--out', str(out / 'pose.csv'),
                                 '--scans', str(out / 'pose.npz')], cwd=ROOT)
    try:
        drive(out)
    finally:
        pose_log.send_signal(signal.SIGINT)
        try:
            pose_log.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pose_log.kill()
        stop(stand)
        log.close()


if __name__ == '__main__':
    main()
