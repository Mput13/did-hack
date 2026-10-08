"""Судья как узел ROS 2: интерфейс /did/* из условия задачи поверх did.judge.Judge.

Агенту доступны только /did/battery, /did/sample_sensor, /did/collect, /did/finish, /did/score
и /did/events. Всё остальное (/did/truth, /did/viz/truth, /map, подсказки в окне Gazebo) —
скрытая правда для записи прогона и показа зрителям, агент её не читает.

Время — симуляционное (/clock), t = 0 в момент первой полученной позы робота.
"""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from did.arena import load_arena
from did.config import BASE, SCIENCE, Rules
from did.judge import Judge
from did.scenario import Scenario, generate

from . import truth_viz

TICK_HZ = 20.0          # шаг судьи по времени симуляции
BATTERY_HZ = 10.0
SCORE_HZ = 1.0
GT_WAIT_S = 3.0         # сколько ждать истинную позу, прежде чем перейти на /odom
GT_STALE_S = 1.0        # истинная поза молчит дольше (по часам машины) — считаем по одометрии
ROBOT = 'burger'        # имя модели робота в мире Gazebo
GZ_POSES = '/did/gz/dynamic_pose'   # мост /world/default/dynamic_pose/info
SCORE_KEYS = ('samples_collected', 'collisions', 'false_collects', 'hazard_hits',
              'finished', 'reason', 'returned')


class Every:
    """Срабатывает с заданной частотой по переданному времени; пропуски не навёрстывает."""
    slack = 0.5 / TICK_HZ   # проверяют раз в шаг судьи, поэтому полшага — ещё «вовремя»

    def __init__(self, hz):
        self.period = 1.0 / hz
        self.next = None

    def due(self, now):
        if self.next is not None and now < self.next - self.slack:
            return False
        late = self.next is None or now - self.next > self.period
        self.next = (now if late else self.next) + self.period
        return True


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item())   # numpy -> числа


class JudgeNode(Node):

    def __init__(self):
        super().__init__('did_judge')
        level = self.declare_parameter('level', 'easy').value
        seed = self.declare_parameter('seed', 0).value
        scenario_file = self.declare_parameter('scenario_file', '').value
        self.use_gt = self.declare_parameter('use_ground_truth', True).value
        show_truth = self.declare_parameter('show_truth', True).value
        rules_name = self.declare_parameter('rules', 'base').value   # base | science

        arena = load_arena()
        if scenario_file:
            data = json.loads(Path(scenario_file).read_text())
            scenario = Scenario.from_dict(data.get('scenario', data))   # годится и запись /did/truth
        else:
            scenario = generate(level, seed, arena)
        rules = Rules(**SCIENCE) if rules_name == 'science' else Rules()
        self.judge = Judge(scenario, arena, rules, seed=seed)
        self.yaw = None             # курс робота: по нему судья считает расход на повороты

        self.odom_xy = None         # одометрия как есть, от точки старта
        self.gt_xy = None           # истинная поза из Gazebo, мировые координаты
        self.source = None          # 'gazebo' | 'odom'; выбирается один раз, в начале прогона
        self.t0 = None
        self._odom_seen = None
        self._odom_fix = (0.0, 0.0)     # истинная поза минус (база + одометрия), по последнему замеру
        self._gt_wall = 0.0
        self._gt_lost = False
        self._robot_idx = None
        self._robot_n = 0
        self._score_key = None
        self._world_key = None
        self._shape_names = None
        self._done_logged = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_battery = self.create_publisher(Float32, '/did/battery', 10)
        self.pub_sensor = self.create_publisher(Float32, '/did/sample_sensor', 10)
        self.pub_events = self.create_publisher(String, '/did/events', 50)
        self.pub_score = self.create_publisher(String, '/did/score', 10)
        self.pub_truth = self.create_publisher(String, '/did/truth', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/did/viz/truth', latched)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', latched)
        self.create_service(Trigger, '/did/collect', self.on_collect)
        self.create_service(Trigger, '/did/finish', self.on_finish)
        self.create_service(Trigger, '/did/reset', self.on_reset)     # служебный: нужен пульту показа
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        if self.use_gt:
            self.create_subscription(TFMessage, GZ_POSES, self.on_gz_poses, qos_profile_sensor_data)

        self.gz = None
        if show_truth:
            try:
                from .gz_truth import GzTruth
                self.gz = GzTruth(log=self.get_logger().info)
            except Exception as e:
                self.get_logger().warn(f'подсказки в окне Gazebo отключены: {e!r}')

        self._publish_map(arena)
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform(self._map_to_odom())
        self._flush(force_score=True)

        self.battery_due = Every(BATTERY_HZ)
        self.sensor_due = Every(self.judge.rules.sensor_hz)
        self.score_due = Every(SCORE_HZ)
        self.create_timer(1.0 / TICK_HZ, self.tick)     # часы узла: при use_sim_time — /clock
        self.get_logger().info(
            f'сценарий {scenario.name}: образцов {len(scenario.samples)}, грунтов '
            f'{len(scenario.soils)}, опасных зон {len(scenario.hazards)}, событий среды '
            f'{len(scenario.events)}; жду позу робота')

    # --- поза робота -------------------------------------------------------------------------

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_xy = (p.x, p.y)
        if self.gt_xy is None:
            self.yaw = _yaw(msg.pose.pose.orientation)

    def on_gz_poses(self, msg):
        n = len(msg.transforms)
        if n == 0:
            return
        if self._robot_idx is None or n != self._robot_n:
            self._robot_idx, self._robot_n = self._find_robot(msg), n
        p = msg.transforms[self._robot_idx].transform.translation
        self.gt_xy = (p.x, p.y)
        self.yaw = _yaw(msg.transforms[self._robot_idx].transform.rotation)
        self._gt_wall = time.monotonic()
        if self.odom_xy is not None:
            self._odom_fix = (p.x - BASE[0] - self.odom_xy[0], p.y - BASE[1] - self.odom_xy[1])

    def _find_robot(self, msg):
        """Номер робота в списке поз Gazebo.

        Мост ros_gz 1.0 имён сущностей не передаёт (child_frame_id пуст), а кроме позы модели
        в списке идут её звенья с позами относительно модели, то есть около нуля. Поэтому робот —
        запись, ближайшая к одометрии; если её нет, первая: модель в списке стоит раньше звеньев.
        """
        for i, tr in enumerate(msg.transforms):
            if tr.child_frame_id == ROBOT:
                return i
        if self.odom_xy is None:
            return 0
        guess = (BASE[0] + self.odom_xy[0], BASE[1] + self.odom_xy[1])
        dist = [math.dist(guess, (tr.transform.translation.x, tr.transform.translation.y))
                for tr in msg.transforms]
        i = int(np.argmin(dist))
        if dist[i] > 0.5:
            self.get_logger().warn(
                f'истинная поза: ближайшая запись в {dist[i]:.2f} м от одометрии, беру первую')
            return 0
        return i

    def _choose_source(self, now):
        if self.use_gt and self.gt_xy is not None:
            self.source = 'gazebo'
        elif self.odom_xy is not None:
            if self._odom_seen is None:
                self._odom_seen = now
            if not self.use_gt or now - self._odom_seen >= GT_WAIT_S:
                self.source = 'odom'
        if self.source is None:
            return
        self.t0 = now
        x, y = self._pose()
        if self.source == 'gazebo':
            where = f'истинная из Gazebo, запись {self._robot_idx} из {self._robot_n}'
        else:
            where = '/odom + база'
            if self.use_gt:
                self.get_logger().warn(f'нет истинной позы в {GZ_POSES} за {GT_WAIT_S:.0f} с')
        self.get_logger().info(f'прогон начат: поза {where}, робот в ({x:+.3f}; {y:+.3f})')
        self._publish_truth(self.judge.score())     # в записи видно, откуда судья берёт позу

    def _pose(self):
        if self.source == 'gazebo':
            # Мост истинной позы может замолчать посреди прогона. Чтобы робот для судьи не
            # «встал», дальше идём по одометрии с поправкой от последней истинной позы.
            lost = self.odom_xy is not None and time.monotonic() - self._gt_wall > GT_STALE_S
            if lost != self._gt_lost:
                self._gt_lost = lost
                if lost:
                    self.get_logger().warn(f'истинная поза молчит дольше {GT_STALE_S:g} с: '
                                           f'считаю по /odom с поправкой')
                else:
                    self.get_logger().info('истинная поза вернулась')
            if not lost:
                return self.gt_xy
            return (BASE[0] + self.odom_xy[0] + self._odom_fix[0],
                    BASE[1] + self.odom_xy[1] + self._odom_fix[1])
        return BASE[0] + self.odom_xy[0], BASE[1] + self.odom_xy[1]

    # --- ход прогона -------------------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _advance(self, now):
        x, y = self._pose()
        self.judge.step(now - self.t0, x, y, th=self.yaw)
        return x, y

    def tick(self):
        now = self._now()
        if self.source is None:
            self._choose_source(now)
        if self.source is not None:
            x, y = self._advance(now)
            if self.sensor_due.due(now):
                self.pub_sensor.publish(Float32(data=self.judge.read_sensor(x, y)))
        if self.battery_due.due(now):
            self.pub_battery.publish(Float32(data=float(self.judge.read_battery())))
        self._flush(force_score=self.score_due.due(now))

    def on_collect(self, request, response):
        return self._act(self.judge.collect, response)

    def on_finish(self, request, response):
        return self._act(self.judge.finish, response)

    def on_reset(self, request, response):
        """Новый прогон на том же сценарии: время, заряд, образцы и счёт — с начала. Робот остаётся где стоит.

        Агент сервис не вызывает; им пользуется пульт (did/pilot.py), чтобы после ручной езды
        автономная миссия начиналась с полным зарядом.
        """
        old = self.judge
        self.judge = Judge(old.scenario, old.arena, old.rules, seed=old.scenario.seed)
        self.t0 = self._now() if self.source is not None else None
        self._score_key = self._world_key = None
        self._done_logged = False
        self._flush(force_score=True)
        self.get_logger().info('прогон начат заново по запросу /did/reset')
        response.success, response.message = True, 'судья начал прогон заново'
        return response

    def _act(self, action, response):
        if self.source is None:
            response.success, response.message = False, 'судья ещё не видит робота'
            return response
        x, y = self._advance(self._now())
        ok, message = action(x, y)
        response.success, response.message = bool(ok), message
        self._flush()
        return response

    # --- публикация --------------------------------------------------------------------------

    def _flush(self, force_score=False):
        """События — сразу; счёт — по таймеру и при изменении; правда — при изменении мира."""
        for ev in self.judge.pop_events():
            self.pub_events.publish(String(data=_dumps(ev)))
            self.get_logger().info(f'событие: {_dumps(ev)}')
        score = self.judge.score()
        key = tuple(score[k] for k in SCORE_KEYS)
        world = len(self.judge.world_log)
        if force_score or key != self._score_key:
            self.pub_score.publish(String(data=_dumps(score)))
        if key != self._score_key or world != self._world_key:
            for change in self.judge.world_log[self._world_key or 0:]:
                self.get_logger().info(f'среда изменилась (агенту не объявляется): {_dumps(change)}')
            self._publish_truth(score)
        self._score_key, self._world_key = key, world
        if self.judge.done and not self._done_logged:
            self._done_logged = True
            self.get_logger().info(f'прогон завершён: {_dumps(score)}')

    def _publish_truth(self, score):
        j = self.judge
        text = _dumps({
            'scenario': j.scenario.to_dict(),
            'rules': j.rules.to_dict(),
            'world_log': j.world_log,
            'soils': [asdict(z) for z in j.soils],
            'hazards': [asdict(z) for z in j.hazards],
            'remaining': [{'id': i, 'x': p[0], 'y': p[1]} for i, p in sorted(j.remaining.items())],
            'collected': [{'id': i, 't': t} for i, t in j.collected],
            'sensor_sigma': j.sensor_sigma,
            'pose_source': self.source,
            'score': score,
        })
        self.pub_truth.publish(String(data=text))
        items = truth_viz.shapes(json.loads(text))      # после JSON — обычные числа и списки
        names = [s['name'] for s in items]
        if names != self._shape_names:
            self._shape_names = names
            self.pub_markers.publish(truth_viz.markers(items))
            if self.gz:
                self.gz.show(items)

    def _publish_map(self, arena):
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.info.resolution = arena.res
        grid.info.width, grid.info.height = int(arena.w), int(arena.h)
        grid.info.origin.position.x, grid.info.origin.position.y = float(arena.x0), float(arena.y0)
        grid.info.origin.orientation.w = 1.0
        grid.data = np.where(arena.free, 0, 100).astype(np.int8).ravel().tolist()   # строка 0 — нижняя
        self.pub_map.publish(grid)

    def _map_to_odom(self):
        """Кадр map — мировые координаты арены; одометрия отсчитывается от точки старта."""
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'map', 'odom'
        tf.transform.translation.x, tf.transform.translation.y = float(BASE[0]), float(BASE[1])
        tf.transform.rotation.w = 1.0
        return tf

    def close(self):
        if self.gz:
            self.gz.close()


def main():
    rclpy.init()
    node = JudgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Ctrl-C посреди приёма сообщения rclpy отдаёт как RuntimeError; при /clock в 1 кГц это
        # почти каждый раз. Настоящая ошибка — только если контекст ещё жив.
        time.sleep(0.2)
        if rclpy.ok():
            raise
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
