"""Два робота в ROS 2 + Gazebo (исследование M1, вторая ступень): судья на двоих, канал /did/team, агент.

У каждого робота свои топики и сервисы из условия задачи, но в своём пространстве имён:
/tb1/cmd_vel, /tb1/scan, /tb1/odom, /tb1/did/battery, /tb1/did/sample_sensor, /tb1/did/collect,
/tb1/did/finish, /tb1/did/score, /tb1/did/events — и то же для /tb2. Общий у роботов один топик
/did/team (std_msgs/String, одна строка JSON на сообщение): по нему они разговаривают.

    python -m did.ros_team judge --level medium --seed 3       # судья на двоих (обычный судья не меняется)
    python -m did.ros_team agent --robot tb1 --level medium --seed 3
    python -m did.ros_team agent --robot tb2 --level medium --seed 3

Всё вместе одной командой — tools/team_gazebo_run.py. Мир с двумя роботами —
ws/src/did_bringup/launch/team.launch.py.
Узлы запускаются как обычные программы Python, сборка воркспейса им не нужна.
"""
import argparse
import json
import math
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from .agent import make_config
from .arena import load_arena
from .config import LEVELS, Rules
from .judge import Judge
from .judge_team import CONTACT_M, CONTACT_MARGIN_M, CONTACT_RELEASE_M, ROBOT_NAMES, SLOTS, team_score
from .planner import HeuristicPlanner
from .recorder import Recorder, save_trace
from .runner import RUNS
from .scenario import generate
from .team import TeamAgent, TeamConfig

TEAM_TOPIC = '/did/team'
TRUTH_TOPIC = '/did/team_truth'
TICK_HZ = 20.0
SETTLE_UNTIL_S = 35.0          # как в did/ros_agent.py: Burger после появления ещё качается
TICK_S = 0.1


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


# ==========================================================================================
# судья на двоих
# ==========================================================================================

class TeamJudgeNode(Node):
    """По судье did.judge.Judge на робота; образцы у них общие, как в did/fastsim_team.py.

    Поза робота — его место на базе плюс одометрия (как в условии задачи). Часы прогона общие и
    начинаются, когда пришла одометрия обоих роботов.
    """

    def __init__(self, level, seed, robots=ROBOT_NAMES):
        super().__init__('did_team_judge', parameter_overrides=[Parameter('use_sim_time', value=True)])
        arena = load_arena()
        self.scenario = generate(level, seed, arena)
        self.rules = Rules()
        self.names = list(robots)
        self.judges, self.odom, self.yaw, self.pub = [], {}, {}, {}
        from dataclasses import replace
        for i, name in enumerate(self.names):
            j = Judge(replace(self.scenario, base=tuple(SLOTS[i])), arena, self.rules, seed=seed)
            if i:
                j.remaining = self.judges[0].remaining
                j.rng = np.random.default_rng([int(seed), 7, i])
                j.rng_battery = np.random.default_rng([int(seed), 8, i])
            self.judges.append(j)
            self.pub[name] = {
                'battery': self.create_publisher(Float32, f'/{name}/did/battery', 10),
                'sensor': self.create_publisher(Float32, f'/{name}/did/sample_sensor', 10),
                'events': self.create_publisher(String, f'/{name}/did/events', 50),
                'score': self.create_publisher(String, f'/{name}/did/score', 10),
            }
            self.create_service(Trigger, f'/{name}/did/collect', lambda rq, rs, i=i: self._act(i, 'collect', rs))
            self.create_service(Trigger, f'/{name}/did/finish', lambda rq, rs, i=i: self._act(i, 'finish', rs))
            self.create_subscription(Odometry, f'/{name}/odom', lambda m, n=name: self._on_odom(n, m),
                                     qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_truth = self.create_publisher(String, TRUTH_TOPIC, latched)
        self.t0 = None
        self.contacts = 0
        self.min_gap = None            # наименьшее расстояние между центрами за прогон, по одометрии обоих
        self._touching = False
        self._n = 0
        self.create_timer(1.0 / TICK_HZ, self.tick)
        self.get_logger().info(f'сценарий {self.scenario.name}: образцов {len(self.scenario.samples)}; '
                               f'жду одометрию роботов {", ".join(self.names)}')

    def _on_odom(self, name, msg):
        p = msg.pose.pose.position
        self.odom[name] = (p.x, p.y)
        self.yaw[name] = _yaw(msg.pose.pose.orientation)

    def _pose(self, i):
        x, y = self.odom[self.names[i]]
        return SLOTS[i][0] + x, SLOTS[i][1] + y

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        now = self._now()
        if self.t0 is None:
            if len(self.odom) < len(self.names):
                return
            self.t0 = now
            self.get_logger().info('прогон начат: одометрия обоих роботов есть')
        t = now - self.t0
        poses = [self._pose(i) for i in range(len(self.names))]
        # Столкновение роботов: центры ближе суммы радиусов с запасом — штраф каждому, кто ещё в прогоне.
        gap = math.dist(poses[0], poses[1]) if len(poses) > 1 else math.inf
        hit = gap < CONTACT_M + CONTACT_MARGIN_M
        self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
        news = hit and not self._touching
        if news:
            self.contacts += 1
        if gap > CONTACT_M + CONTACT_MARGIN_M + CONTACT_RELEASE_M:
            self._touching = False
        elif hit:
            self._touching = True
        self._n += 1
        for i, (name, j) in enumerate(zip(self.names, self.judges)):
            x, y = poses[i]
            j.step(t, x, y, blocked=self._touching, th=self.yaw[name])
            pub = self.pub[name]
            if self._n % 4 == 0:                                   # 5 Гц
                pub['sensor'].publish(Float32(data=float(j.read_sensor(x, y))))
            if self._n % 2 == 0:
                pub['battery'].publish(Float32(data=float(j.read_battery())))
            for ev in j.pop_events():
                pub['events'].publish(String(data=_dumps(ev)))
                self.get_logger().info(f'{name}: {_dumps(ev)}')
            if self._n % 10 == 0:
                pub['score'].publish(String(data=_dumps(j.score())))
        if self._n % 20 == 0 or news:                              # о контакте — сразу: итог идёт в запись прогона
            self.pub_truth.publish(String(data=_dumps(self.truth())))

    def _act(self, i, what, response):
        if self.t0 is None:
            response.success, response.message = False, 'судья ещё не видит роботов'
            return response
        j = self.judges[i]
        x, y = self._pose(i)
        ok, message = getattr(j, what)(x, y)
        response.success, response.message = bool(ok), message
        self.pub[self.names[i]]['score'].publish(String(data=_dumps(j.score())))
        self.pub_truth.publish(String(data=_dumps(self.truth())))
        return response

    def truth(self):
        scores = [j.score() for j in self.judges]
        times = [t for j in self.judges for _, t in j.collected]
        total = team_score(scores, times, len(self.scenario.samples), self.rules.time_limit_s)
        total['robot_contacts'] = self.contacts
        total['min_gap_m'] = None if self.min_gap is None or not math.isfinite(self.min_gap) else round(self.min_gap, 3)
        total['t'] = round(self._now() - self.t0, 2) if self.t0 is not None else 0.0
        return {'scenario': self.scenario.to_dict(), 'rules': self.rules.to_dict(), 'world_log': self.judges[0].world_log,
                'team': total, 'robots': [{'name': n, **s} for n, s in zip(self.names, scores)],
                'done': all(j.done for j in self.judges)}


def judge_main(args):
    rclpy.init()
    node = TeamJudgeNode(args.level, args.seed)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RuntimeError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# ==========================================================================================
# канал и агент
# ==========================================================================================

class RosChannel:
    """Тот же интерфейс, что у did.team.TeamChannel, поверх топика /did/team."""

    def __init__(self, node, name):
        self.name = name
        self.log = []                 # всё, что сказал я и что услышал: идёт в запись прогона
        self._inbox = []
        self._lock = threading.Lock()
        self._pub = node.create_publisher(String, TEAM_TOPIC, 100)
        node.create_subscription(String, TEAM_TOPIC, self._on_msg, 100)

    def _on_msg(self, msg):
        m = json.loads(msg.data)
        if m.get('from') == self.name:
            return
        with self._lock:
            self._inbox.append(m)
            self.log.append(m)

    def post(self, sender, t, kind, /, **body):
        msg = {'t': round(float(t), 2), 'from': sender, 'type': kind, **body}
        with self._lock:
            self.log.append(msg)
        self._pub.publish(String(data=_dumps(msg)))
        return msg

    def read(self, reader):
        with self._lock:
            out, self._inbox = self._inbox, []
        return out


def agent_main(args):
    from .ros_agent import RosIO
    i = ROBOT_NAMES.index(args.robot)
    name, home = ROBOT_NAMES[i], SLOTS[i]
    arena = load_arena()
    rules = Rules()
    rclpy.init()
    io = RosIO(ns=f'/{name}', base=home, name=f'did_agent_{name}')
    truth = {}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    io.create_subscription(String, TRUTH_TOPIC, lambda m: truth.update(json.loads(m.data)), latched)
    channel = RosChannel(io, name) if args.mode == 'team' else None
    executor = SingleThreadedExecutor()
    executor.add_node(io)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 180.0
        while not io.ready():
            if time.monotonic() > deadline:
                raise RuntimeError(f'{name}: нет данных от стенда')
            time.sleep(0.1)
        while io.now() < SETTLE_UNTIL_S:                # ждём, пока робот осядет на колёса; оба стартуют разом
            io.command(0.0, 0.0)
            time.sleep(0.2)
        cfg = make_config(args.agent)
        rec = Recorder()
        from .team_runner import NO_LINK
        team_cfg = TeamConfig() if args.mode == 'team' else TeamConfig(**NO_LINK)
        bot = TeamAgent(arena, cfg, LEVELS[args.level]['samples'], name=name, home=home, channel=channel,
                        team=team_cfg, rules=rules, planner=HeuristicPlanner(), recorder=rec)
        last_tick = -1e9
        while rclpy.ok():
            now = io.now()
            if now - last_tick < TICK_S:
                time.sleep(0.002)
                continue
            last_tick = now
            obs = io.observe()
            bot.tick(obs, io)
            if obs.done or bot.finished:
                break
        for _ in range(5):
            io.command(0.0, 0.0)
            time.sleep(0.05)
        time.sleep(1.5)
        bot.tick(io.observe(), io)
        # Итог судьи на двоих (столкновения роботов, зазор) идёт в запись: ждём сообщение новее конца прогона.
        t_end, wait = io.now(), time.monotonic() + 3.0
        while (truth.get('team') or {}).get('t', -1.0) < t_end and time.monotonic() < wait:
            time.sleep(0.05)
        scenario = truth.get('scenario') or generate(args.level, args.seed, arena).to_dict()
        run_id = f'{args.exp}/{args.mode}/{args.level}-{args.seed}'
        part = rec.build(run_id=run_id, experiment=args.exp, arm=args.mode, backend='gazebo',
                         agent={'name': cfg.name, 'config': cfg.to_dict()}, scenario=scenario, rules=rules.to_dict(),
                         result=io.score, world=truth.get('world_log', []), journal=bot.journal)
        part['robot'] = {'name': name, 'home': list(home)}
        part['messages'] = channel.log if channel else []
        judged = truth.get('team') or {}
        if 'robot_contacts' in judged:
            part['judge'] = {k: judged.get(k) for k in ('t', 'robot_contacts', 'min_gap_m')}
        path = save_trace(part, RUNS / args.exp / '_parts' / f'{args.level}-{args.seed}-{name}.json.gz')
        print(f'{name}: {_dumps(io.score)}\n{name}: запись {path}', flush=True)
    finally:
        try:
            io.command(0.0, 0.0)
        except Exception:      # noqa: BLE001 — узел мог уже закрыться
            pass
        executor.shutdown()
        io.destroy_node()
        rclpy.try_shutdown()


def merge(exp, mode, level, seed):
    """Склеить записи двух агентов в одну запись команды (тот же вид, что у did/team_runner.py)."""
    from .recorder import load_trace
    from .team_runner import merge_parts
    parts = [load_trace(RUNS / exp / '_parts' / f'{level}-{seed}-{n}.json.gz') for n in ROBOT_NAMES]
    trace = merge_parts(parts, mode)
    path = save_trace(trace, RUNS / exp / mode / f'{level}-{seed}.json.gz')
    return path, trace['result']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['judge', 'agent', 'merge'])
    ap.add_argument('--robot', default='tb1', choices=list(ROBOT_NAMES))
    ap.add_argument('--level', default='medium', choices=list(LEVELS))
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--mode', default='team', choices=['team', 'pair_lidar'])
    ap.add_argument('--exp', default='M1gz')
    args = ap.parse_args()
    if args.what == 'judge':
        judge_main(args)
    elif args.what == 'agent':
        agent_main(args)
    else:
        path, total = merge(args.exp, args.mode, args.level, args.seed)
        print(_dumps(total))
        print('запись:', path)


if __name__ == '__main__':
    main()
