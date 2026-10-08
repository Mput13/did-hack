#!/usr/bin/env python3
"""Проверка уровня 0 с судьёй: интерфейс /did/* жив и считает правдоподобно.

Запуск при работающем стенде (pixi run stand): pixi run smoke-judge
Проверка завершает прогон вызовом /did/finish, поэтому для повтора стенд нужно перезапустить.
Скрипт читает /did/truth, чтобы знать ожидаемый расход, — агенту этот топик недоступен.
"""
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from did.config import BASE

WAIT_DATA_S = 120.0   # первый запуск качает модели пола и света из Fuel
RATE_WINDOW_S = 5.0   # секунды симуляции
DRIVE_SPEED = 0.15    # м/с, у Burger предел 0.22
DRIVE_SIM_S = 4.0
EVENT_TYPES = {'collision', 'false_collect', 'hazard_hit', 'sample_collected'}


class Smoke(Node):

    def __init__(self):
        super().__init__('did_smoke_judge')
        self.sim_time = None
        self.odom_xy = None
        self.path = []          # [(t, x, y)] в мировых координатах по одометрии
        self.battery = []       # [(t, значение)]
        self.sensor = []
        self.score = None
        self.score_count = 0
        self.truth = None
        self.events = []
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Float32, '/did/battery', self.on_battery, 50)
        self.create_subscription(Float32, '/did/sample_sensor', self.on_sensor, 50)
        self.create_subscription(String, '/did/score', self.on_score, 10)
        self.create_subscription(String, '/did/events', self.on_event, 50)
        self.create_subscription(String, '/did/truth', self.on_truth, latched)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.collect = self.create_client(Trigger, '/did/collect')
        self.finish = self.create_client(Trigger, '/did/finish')

    def on_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_xy = (BASE[0] + p.x, BASE[1] + p.y)
        self.path.append((self.sim_time, *self.odom_xy))

    def on_battery(self, msg):
        self.battery.append((self.sim_time, msg.data))

    def on_sensor(self, msg):
        self.sensor.append((self.sim_time, msg.data))

    def on_score(self, msg):
        self.score = json.loads(msg.data)
        self.score_count += 1

    def on_event(self, msg):
        self.events.append(json.loads(msg.data))

    def on_truth(self, msg):
        self.truth = json.loads(msg.data)

    def spin_for(self, wall_seconds):
        end = time.monotonic() + wall_seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def spin_sim(self, sim_seconds, each=None, wall_limit=120.0):
        """Крутить узел, пока не пройдёт sim_seconds времени симуляции."""
        start, limit = self.sim_time, time.monotonic() + wall_limit
        while self.sim_time - start < sim_seconds and time.monotonic() < limit:
            if each:
                each()
            self.spin_for(0.05)

    def drive(self, speed):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = speed
        self.cmd_pub.publish(msg)

    def call(self, client, wall_limit=10.0):
        if not client.wait_for_service(timeout_sec=wall_limit):
            return None
        future = client.call_async(Trigger.Request())
        end = time.monotonic() + wall_limit
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
        return future.result() if future.done() else None


def soil_mult(soils, x, y):
    m = 1.0
    for z in soils:
        if z['shape'] == 'circle':
            inside = (x - z['x']) ** 2 + (y - z['y']) ** 2 <= z['r'] ** 2
        else:
            inside = abs(x - z['x']) <= z['w'] / 2 and abs(y - z['y']) <= z['h'] / 2
        if inside:
            m = max(m, z['mult'])
    return m


def main():
    rclpy.init()
    node = Smoke()
    failures = []

    def check(ok, text):
        print(('  ok    ' if ok else '  FAIL  ') + text)
        if not ok:
            failures.append(text)

    # --- данные идут ---------------------------------------------------------------------------
    deadline = time.monotonic() + WAIT_DATA_S
    need = {}
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        need = {'/clock': node.sim_time, '/odom': node.odom_xy, '/did/battery': node.battery,
                '/did/sample_sensor': node.sensor, '/did/score': node.score, '/did/truth': node.truth}
        if all(need.values()):
            break
    missing = [name for name, got in need.items() if not got]
    if missing:
        print(f'FAIL: за {WAIT_DATA_S:.0f} с нет данных в {", ".join(missing)}')
        return 1
    if node.score['finished']:
        print('FAIL: прогон уже завершён (reason = %s) — перезапустите стенд' % node.score['reason'])
        return 1
    rules, level = node.truth['rules'], node.truth['scenario']['name']
    print(f'стенд отвечает: сценарий {level}, поза судьи — {node.truth["pose_source"]}')

    # --- частоты и границы ---------------------------------------------------------------------
    node.battery.clear()
    node.sensor.clear()
    sim0, wall0 = node.sim_time, time.monotonic()
    node.spin_sim(RATE_WINDOW_S)
    sim_dt = node.sim_time - sim0
    rtf = sim_dt / (time.monotonic() - wall0)
    battery_hz = len(node.battery) / sim_dt
    sensor_hz = len(node.sensor) / sim_dt
    b_lo, b_hi = min(v for _, v in node.battery), max(v for _, v in node.battery)
    s_lo, s_hi = min(v for _, v in node.sensor), max(v for _, v in node.sensor)
    print(f'real-time factor = {rtf:.2f}')
    check(abs(battery_hz - 10.0) <= 1.5, f'/did/battery {battery_hz:.1f} Гц по времени симуляции (нужно 10)')
    check(abs(sensor_hz - rules['sensor_hz']) <= 1.0,
          f'/did/sample_sensor {sensor_hz:.1f} Гц (нужно {rules["sensor_hz"]:g})')
    check(0.0 <= b_lo and b_hi <= rules['battery_start'],
          f'батарея в границах 0..{rules["battery_start"]:g}: {b_lo:.3f}..{b_hi:.3f}')
    check(0.0 <= s_lo and s_hi <= 1.0, f'датчик образцов в границах 0..1: {s_lo:.3f}..{s_hi:.3f}')
    idle = (node.battery[0][1] - node.battery[-1][1]) / (node.battery[-1][0] - node.battery[0][0])
    # После появления в мире робот ещё секунд десять оседает и сползает на ~1,5 см: одометрия
    # этого не видит, судья по истинной позе — видит. Отсюда допуск вверх.
    check(-0.005 <= idle - rules['drain_idle_per_s'] <= 0.005 + 0.003 * rules['drain_per_m'],
          f'расход на месте {idle:.4f} ед./с (по правилам {rules["drain_idle_per_s"]:g})')

    # --- езда: батарея убывает на пройденный путь ----------------------------------------------
    node.path.clear()
    node.spin_for(0.3)
    b0 = node.battery[-1]
    node.spin_sim(DRIVE_SIM_S, each=lambda: node.drive(DRIVE_SPEED))
    for _ in range(10):
        node.drive(0.0)
        node.spin_for(0.1)
    b1 = node.battery[-1]
    soils = node.truth['soils']
    travel = expected = 0.0
    for (_, xa, ya), (_, xb, yb) in zip(node.path, node.path[1:]):
        ds = math.dist((xa, ya), (xb, yb))
        travel += ds
        expected += rules['drain_per_m'] * soil_mult(soils, (xa + xb) / 2, (ya + yb) / 2) * ds
    expected += rules['drain_idle_per_s'] * (b1[0] - b0[0])
    spent = b0[1] - b1[1]
    per_m = (spent - rules['drain_idle_per_s'] * (b1[0] - b0[0])) / travel if travel > 0 else float('nan')
    print(f'езда {DRIVE_SIM_S:.0f} с на {DRIVE_SPEED} м/с: путь по одометрии {travel:.3f} м, батарея '
          f'{b0[1]:.3f} -> {b1[1]:.3f}')
    check(travel >= 0.4, f'робот проехал {travel:.2f} м (нужно не меньше 0.4)')
    check(abs(spent - expected) <= 0.05 + 0.1 * expected,
          f'расход {spent:.3f} ед. при ожидаемых {expected:.3f}; за вычетом простоя {per_m:.2f} ед./м')

    # --- сбор вдали от образца -----------------------------------------------------------------
    x, y = node.odom_xy
    near = min((math.dist((x, y), (s['x'], s['y'])) for s in node.truth['remaining']), default=math.inf)
    expect_hit = near <= rules['collect_radius_m']
    expect_type = 'sample_collected' if expect_hit else 'false_collect'
    seen = len(node.events)
    false_before = node.score['false_collects']
    res = node.call(node.collect)
    node.spin_for(1.0)
    fresh = [e['type'] for e in node.events[seen:]]
    print(f'/did/collect в ({x:+.2f}; {y:+.2f}), до ближайшего образца {near:.2f} м: '
          f'{None if res is None else (res.success, res.message)}')
    check(res is not None and res.success == expect_hit, f'ответ /did/collect: success = {expect_hit}')
    check(expect_type in fresh, f'в /did/events пришло {expect_type} (получено: {fresh})')
    if not expect_hit:
        check(node.score['false_collects'] == false_before + 1, 'ложный сбор учтён в /did/score')

    # --- завершение прогона --------------------------------------------------------------------
    at_base = math.dist((x, y), node.truth['scenario']['base']) <= rules['base_radius_m']
    res = node.call(node.finish)
    node.spin_for(1.5)
    print(f'/did/finish: {None if res is None else (res.success, res.message)}')
    check(res is not None and res.success == at_base, f'ответ /did/finish: success = {at_base} (робот '
          f'{"на базе" if at_base else "не на базе"})')
    check(node.score['finished'] and node.score['reason'] == 'finish' and node.score['returned'] == at_base,
          f'/did/score: finished = {node.score["finished"]}, reason = {node.score["reason"]}, '
          f'returned = {node.score["returned"]}')

    # --- после завершения ничего не начисляется ------------------------------------------------
    final = dict(node.score)
    node.battery.clear()
    node.score_count = 0
    res = node.call(node.collect)
    node.spin_sim(2.0, each=lambda: node.drive(-DRIVE_SPEED))
    for _ in range(5):
        node.drive(0.0)
        node.spin_for(0.1)
    frozen = {v for _, v in node.battery}
    check(res is not None and not res.success, f'/did/collect после завершения отклонён: '
          f'{None if res is None else res.message}')
    check(len(frozen) == 1 and node.battery, f'батарея после завершения не меняется: {sorted(frozen)}')
    check(node.score_count >= 2 and node.score == final,
          f'итоговый счёт продолжает публиковаться ({node.score_count} сообщ.) и не меняется: '
          f'{node.score["score"]}')
    alien = sorted({e['type'] for e in node.events} - EVENT_TYPES)
    check(not alien, f'в /did/events только разрешённые виды событий (лишние: {alien})')

    print('OK' if not failures else f'FAIL ({len(failures)})')
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
