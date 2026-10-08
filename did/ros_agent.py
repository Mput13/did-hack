"""Тот же агент в ROS 2 + Gazebo: только интерфейс из условия задачи (/cmd_vel, /scan, /odom, /did/*).

Запуск при работающем стенде (pixi run stand level:=hard seed:=3):

    pixi run agent-ros --level hard --seed 3 --agent adaptive

Узел ничего не знает о сценарии: уровень и seed нужны только для имени записи. Скрытую правду
(/did/truth) он читает лишь в самом конце, чтобы положить её в запись прогона для показа.
"""
import argparse
import json
import math
import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from .agent import Agent, make_config
from .arena import load_arena
from .config import BASE, LEVELS, SCIENCE, Rules
from .metrics import run_metrics, score_inquiries
from .recorder import Recorder, save_trace
from .robot_io import Observation
from .runner import RUNS, make_planner
from .scenario import Scenario, generate
from .waiting import wait_metrics

TICK_S = 0.1
# После появления в мире Burger ещё около 30 с времени симуляции качается на задней опоре: колёса на
# доли миллиметра отрываются от пола, и любая смена их скорости в это время проходит мимо корпуса —
# одометрия засчитывает поворот, которого не было (до нескольких радиан за манёвр). По /odom этого
# не видно, поэтому ехать начинаем не раньше этого времени судьи (он считает от появления робота).
SETTLE_UNTIL_S = 35.0
LIVE = RUNS / '_live' / 'state.json'


class RosIO(Node):
    """RobotIO поверх топиков и сервисов. Колбэки только складывают последние значения."""

    def __init__(self, ns='', base=BASE, name='did_agent'):
        # ns и base — для второго робота в одном мире (did/ros_team.py): свои топики /tb2/... и своя точка старта.
        super().__init__(name)
        self.base = base
        self.lock = threading.Lock()
        self.sim_time = None
        self.odom = None
        self.battery = None
        self.sensor = None
        self.sensor_stamp = None
        self.scan = None
        self.scan_stamp = self.scan_step = None
        self.odom_log = deque(maxlen=150)       # (время, x, y, курс, v, w): поза на момент скана
        self.events = []
        self.score = None
        self.truth = None
        self.clock_offset = None        # время судьи минус время симуляции
        self.create_subscription(Clock, '/clock', self._on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, ns + '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, ns + '/scan', self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Float32, ns + '/did/battery', self._on_battery, 10)
        self.create_subscription(Float32, ns + '/did/sample_sensor', self._on_sensor, 10)
        self.create_subscription(String, ns + '/did/events', self._on_events, 50)
        self.create_subscription(String, ns + '/did/score', self._on_score, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/did/truth', self._on_truth, latched)
        self.cmd_pub = self.create_publisher(TwistStamped, ns + '/cmd_vel', 10)
        self.belief_pub = self.create_publisher(OccupancyGrid, ns + '/did/viz/belief', latched)
        self.path_pub = self.create_publisher(Path, ns + '/did/viz/path', latched)
        self.collect_cli = self.create_client(Trigger, ns + '/did/collect')
        self.finish_cli = self.create_client(Trigger, ns + '/did/finish')

    # --- колбэки -----------------------------------------------------------------------------

    def _on_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_odom(self, msg):
        p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom = (self.base[0] + p.x, self.base[1] + p.y, yaw, tw.linear.x, tw.angular.z)
            self.odom_log.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, *self.odom))

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=float)
        r[~np.isfinite(r) | (r < msg.range_min) | (r > msg.range_max)] = np.inf
        with self.lock:
            self.scan = r
            self.scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.scan_step = msg.angle_increment

    def _on_battery(self, msg):
        self.battery = float(msg.data)

    def _on_sensor(self, msg):
        with self.lock:
            self.sensor = float(msg.data)
            self.sensor_stamp = self.sim_time      # у сообщения нет времени измерения: берём время получения

    def _on_events(self, msg):
        with self.lock:
            self.events.append(json.loads(msg.data))

    def _on_score(self, msg):
        self.score = json.loads(msg.data)
        if self.sim_time is not None and not self.score.get('finished'):
            self.clock_offset = self.score['t'] - self.sim_time

    def _on_truth(self, msg):
        self.truth = json.loads(msg.data)

    # --- RobotIO -----------------------------------------------------------------------------

    def ready(self):
        return None not in (self.sim_time, self.odom, self.battery, self.score, self.clock_offset)

    def now(self):
        return self.sim_time + self.clock_offset

    def observe(self):
        with self.lock:
            x, y, th, v, w = self.odom
            sensor, self.sensor = self.sensor, None
            age = self.sim_time - self.sensor_stamp if sensor is not None and self.sensor_stamp is not None else 0.0
            scan, self.scan = self.scan, None
            events, self.events = self.events, []
            scan_pose = self._odom_at(self.scan_stamp) if scan is not None else None
        return Observation(t=self.now(), x=x, y=y, th=th, v=v, w=w, battery=self.battery, sensor=sensor,
                           scan=scan, events=events, done=bool(self.score.get('finished')),
                           scan_pose=scan_pose, scan_step=self.scan_step, sensor_age=max(0.0, age))

    def _odom_at(self, t):
        """Поза по одометрии на момент t. Скан приходит позже, чем снят, а робот за это время успевает
        повернуться: сравнивать скан с картой нужно из той позы, в которой он сделан."""
        newer = None
        for item in reversed(self.odom_log):
            if item[0] <= t:
                _, x, y, th, v, w = item
                if newer is None:               # одометрия старше скана: продолжаем по её скоростям
                    dt = min(t - item[0], 0.1)
                    return x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt
                k = (t - item[0]) / (newer[0] - item[0])
                turn = (newer[3] - th + math.pi) % (2 * math.pi) - math.pi
                return x + k * (newer[1] - x), y + k * (newer[2] - y), th + k * turn
            newer = item
        return None                             # скан старше всей истории одометрии: берём текущую позу

    def command(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _call(self, client):
        if not client.wait_for_service(timeout_sec=2.0):
            return False, 'сервис судьи недоступен'
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not future.done():
            return False, 'судья не ответил'
        return bool(future.result().success), future.result().message

    def collect(self):
        return self._call(self.collect_cli)

    def restart_clock(self):
        """Попросить судью стенда начать прогон заново: часы сценария пойдут с этой секунды.

        Пока робот оседает на колёса (35 с), часы судьи уже идут, и события трудного уровня (20–70 с) наступают
        раньше, чем в быстром симуляторе, где агент едет с нулевой секунды. Без этого прогон в Gazebo — другой
        опыт, а не тот же сценарий. Сервис /did/reset есть только у нашего стенда; нет сервиса — ничего не делаем.
        """
        client = self.create_client(Trigger, '/did/reset')
        if not client.wait_for_service(timeout_sec=1.0):
            return False
        ok, _ = self._call(client)
        if not ok:
            return False
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not (self.score and self.score.get('t', 99.0) < 2.0):
            time.sleep(0.02)               # ждём сообщение судьи с новым временем: по нему сдвинутся часы агента
        with self.lock:
            self.events = []
            self.sensor = None
        return True

    def finish(self):
        return self._call(self.finish_cli)

    # --- показ в RViz ------------------------------------------------------------------------

    def publish_viz(self, bot):
        stamp = self.get_clock().now().to_msg()
        b = bot.belief
        grid = OccupancyGrid()
        grid.header.stamp, grid.header.frame_id = stamp, 'map'
        grid.info.resolution = b.res
        grid.info.width, grid.info.height = b.w, b.h
        grid.info.origin.position.x, grid.info.origin.position.y = b.x0, b.y0
        grid.info.origin.orientation.w = 1.0
        grid.data = np.clip(np.sqrt(b.grid()) * 100, 0, 100).astype(np.int8).ravel().tolist()
        self.belief_pub.publish(grid)
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, 'map'
        for x, y in bot.follower.pts[bot.follower.i:]:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)


def run(level, seed, agent, experiment, arm=None, llm=None, wait_s=120.0, settle_s=8.0, quiet=False,
        rules=None, knowledge=None, config=None):
    """Провести один прогон в Gazebo. Стенд (симуляция и судья) должен быть уже запущен.

    config — поправки к настройкам агента, например {'llm_act_while_waiting': 'rule'}: пока модель думает
    в своём потоке, робот едет по плану правила (did/waiting.py).
    """
    arena = load_arena()
    rules = Rules(**SCIENCE) if rules == 'science' else Rules()   # должны совпадать с правилами судьи в стенде
    rclpy.init()
    io = RosIO()
    executor = SingleThreadedExecutor()
    executor.add_node(io)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        deadline = time.monotonic() + wait_s
        while not io.ready():
            if time.monotonic() > deadline:
                raise RuntimeError('нет данных от стенда: проверьте, что запущен pixi run stand')
            time.sleep(0.1)
        if io.score.get('finished'):
            raise RuntimeError('судья уже завершил прогон: перезапустите стенд')
        # Сразу после появления в мире робот ещё оседает на колёса. Если поехать в эти секунды,
        # одометрия расходится с истинным положением, поэтому ждём, пока поза перестанет меняться.
        settled_since, last = None, None
        while True:
            pose, now = io.odom[:3], io.sim_time
            if last is not None and max(abs(a - b) for a, b in zip(pose, last)) < 1e-4:
                settled_since = settled_since if settled_since is not None else now
                if now - settled_since >= settle_s and io.now() >= SETTLE_UNTIL_S:
                    break
            else:
                settled_since = None
            last = pose
            io.command(0.0, 0.0)
            time.sleep(0.2)
        io.restart_clock()

        llm_on = make_config(agent).planner == 'llm'
        cfg = make_config(agent, async_planner=llm_on, **(config or {}))
        arm = arm or cfg.name
        rec = Recorder()
        bot_rules = rules
        if getattr(cfg, 'calibrate', False):
            # Калибрующийся агент и в Gazebo стартует с допущений команды о датчике и расходе, а не с правил
            # стенда (как в быстром симуляторе, did/runner.py). Правила стенда остаются для записи и подсчёта.
            from .calibrate import assumed_rules
            bot_rules = assumed_rules(rules)
        bot = Agent(arena, cfg, n_samples=LEVELS[level]['samples'], rules=bot_rules,
                    planner=make_planner(cfg, llm, seed), recorder=rec, knowledge=knowledge)
        scenario = generate(level, seed, arena)     # только для подписи записи, если /did/truth не придёт
        run_id = f'{experiment}/{arm}/{level}-{seed}'

        def snapshot(result=None):
            truth = io.truth or {}
            score = result or io.score
            trace = rec.build(run_id=run_id, experiment=experiment, arm=arm, backend='gazebo',
                              agent={'name': cfg.name, 'config': cfg.to_dict()},
                              scenario=truth.get('scenario') or scenario.to_dict(), rules=rules.to_dict(),
                              result=score, world=truth.get('world_log', []), journal=bot.journal)
            if bot.inv:
                trace.update(bot.inv.export())           # расследования и модель расхода
            return trace

        last_tick = last_live = -1e9
        wall0 = time.monotonic()
        while rclpy.ok():
            now = io.now()
            if now - last_tick < TICK_S:
                time.sleep(0.002)
                continue
            last_tick = now
            obs = io.observe()
            bot.tick(obs, io)
            if now - last_live >= 1.0:
                last_live = now
                io.publish_viz(bot)
                _write_live({'active': True, 'trace': snapshot()})
            if obs.done or bot.finished:
                break
        for _ in range(5):
            io.command(0.0, 0.0)
            time.sleep(0.05)
        time.sleep(1.2)                               # дождаться итогового /did/score и /did/truth
        bot.tick(io.observe(), io)

        score = io.score
        truth = io.truth or {}
        metrics = run_metrics(score, rules, bot.journal, truth.get('world_log', []), rec.plans, rec.llm)
        metrics.update(wait_metrics(rec, bot))
        trace = snapshot({**score, **metrics})
        if bot.inv:
            true_scenario = Scenario.from_dict(truth['scenario']) if truth.get('scenario') else scenario
            trace['result']['inquiries'] = metrics['inquiries'] = score_inquiries(
                trace['inquiries'], true_scenario, truth.get('world_log', []))
        path = save_trace(trace, RUNS / experiment / arm / f'{level}-{seed}.json.gz')
        _write_live({'active': False})
        summary = {'id': run_id, 'experiment': experiment, 'arm': arm, 'agent': cfg.name, 'level': level,
                   'seed': seed, 'backend': 'gazebo', 'metrics': metrics,
                   'wall_s': round(time.monotonic() - wall0, 1), 'file': str(path.relative_to(RUNS))}
        if not quiet:
            print(json.dumps(summary, ensure_ascii=False, indent=1))
        return summary
    finally:
        try:
            io.command(0.0, 0.0)
        except Exception:      # noqa: BLE001 — узел мог уже закрыться
            pass
        executor.shutdown()
        io.destroy_node()
        rclpy.shutdown()


def _write_live(state):
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, separators=(',', ':'), default=_plain), encoding='utf-8')
    os.replace(tmp, LIVE)


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='easy', choices=list(LEVELS))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='gazebo', help='папка внутри runs/')
    ap.add_argument('--arm', default=None)
    ap.add_argument('--llm', default=None, choices=['mock', 'http', 'ollama', 'codex'])
    ap.add_argument('--rules', default=None, choices=['science'], help='те же правила, что у судьи в стенде')
    ap.add_argument('--act-while-waiting', default=None, choices=['rule', 'leash'],
                    help='пока модель думает: rule — ехать по плану правила, leash — без сбора и не дальше привязи от места вопроса (did/waiting.py)')
    args = ap.parse_args()
    run(args.level, args.seed, args.agent, args.exp, arm=args.arm, llm={'kind': args.llm} if args.llm else None,
        rules=args.rules,
        config={'llm_act_while_waiting': args.act_while_waiting} if args.act_while_waiting else None)


if __name__ == '__main__':
    main()
