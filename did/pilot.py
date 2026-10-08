"""Пульт: оператор ставит точки на карте — робот едет; карта строится по лидару на глазах.

Один класс Pilot для быстрого симулятора и для Gazebo: наружу он видит только RobotIO
(did/robot_io.py) и то, что в обоих случаях умеет стенд, — счёт судьи и новый прогон.

    pixi run demo                         # стенд Gazebo с окном + пульт + страница #/pilot
    pixi run demo --fast                  # то же без Gazebo, на быстром симуляторе
    python -m did.pilot --ros             # только пульт, при уже запущенном стенде

Что делает пульт:
- знает позу: одометрия с поправкой по лидару и эталонной карте (did/localize.py);
- строит карту занятости по лидару (did/mapping.py) и сравнивает её с эталонной;
- по команде оператора планирует путь через заданные точки (CostGraph) и ведёт по нему (Follower);
- по команде запускает автономную миссию: управление переходит к Agent.tick, пульт только показывает.

Режим карты SLAM (pixi run demo --slam-map, Pilot(slam_nav=True)): готовой карты у робота нет. Стены и
запас вокруг них берутся из сетки SLAM Toolbox (did/slam_map.py::SlamArena), поправка позы — из его
же преобразования «карта → одометрия». Путь строится только по полу, который SLAM уже увидел.
Готовая карта остаётся только для двух чисел на экране (сколько увидено, насколько совпало).

Команды оператора — словари {'cmd': ...}:
  route {points: [[x, y], ...]}  задать маршрут          go      ехать по маршруту
  stop                           остановиться            home    вернуться на базу
  reset                          стереть карту и след, начать прогон судьи заново
  explore                        достроить карту: ехать к границам увиденного (только режим карты SLAM)
  mission {agent}                автономная миссия       finish  завершить миссию
  speed {value}                  скорость времени (только быстрый симулятор)

Пульт Gazebo — отдельный процесс; с сервером интерфейса он обменивается файлами в runs/_live/:
состояние — pilot.json (перезаписывается целиком четыре раза в секунду), команды — по файлу на
команду в pilot_cmd/, ответы — в pilot_ack/. Команда не теряется: файл лежит, пока пульт его не
исполнит, и сервер отвечает оператору только после ответа пульта.
"""
import argparse
import json
import math
import os
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import asdict

import numpy as np

from .agent import PRESETS, Agent, make_config
from .arena import load_arena
from .config import BASE, LEVELS, SCIENCE, V_MAX, Rules
from .fastsim import FastSim
from .judge import Judge
from .mapping import LIDAR_OFFSET, OccupancyMapper
from .metrics import run_metrics
from .mission_agents import MISSION_AGENTS         # агенты, которым не нужна языковая модель
from .nav import ESCAPE_STOP, INFLATE, CostGraph, Follower, corridor_room, escape_plan, moved_since, path_length
from .recorder import Recorder, encode_grid, save_trace
from .runner import RUNS, make_planner
from .scenario import generate
from .slam_map import SlamArena, SlamPose, compare, frontiers, resample

try:                                    # поправка позы по лидару: без неё пульт работает по одометрии
    from .localize import PoseTracker
except ImportError:                     # pragma: no cover
    PoseTracker = None

LIVE_DIR = RUNS / '_live'
STATE = LIVE_DIR / 'pilot.json'
CMD_DIR = LIVE_DIR / 'pilot_cmd'
ACK_DIR = LIVE_DIR / 'pilot_ack'

TICK_S = 0.1
STATE_S = 0.25                 # как часто обновляется состояние для интерфейса, с
LAG_S = 0.35                   # такт пришёл с таким опозданием — управление запаздывает (машина перегружена)
SLOW_V = 0.08                  # м/с: с такой скоростью едем, пока управление запаздывает
MAX_POINTS = 20
EXPLORE_MAX = 40               # столько подъездов к границам увиденного — и объезд заканчивается в любом случае
EXPLORE_NEAR = 0.40            # м: граница «та же», если её центр сдвинулся меньше
# Сроки езды в режиме карты SLAM: карта может обмануть (цель оказалась преградой, пол — столбом), а лидар у
# преграды обнуляет скорость — без срока робот стоял бы с режимом «еду» сколько угодно.
PROGRESS_M, NO_PROGRESS_S = 0.05, 10.0     # робот не сдвинулся на 5 см за 10 с — езда прекращается
LEG_BASE_S, LEG_S_PER_M = 30.0, 15.0       # срок одного подъезда: 30 с и по 15 с на метр пути (вдвое медленнее обычного)
SNAP_MAX = 0.15                # м: дальше цель после смены карты не передвигается — оператор ставит её заново
BASE_NEAR = 0.25               # м: ближе — робот «на базе»
# Новый прогон судьи: сколько ждать подтверждения после вызова сброса. Вызов один: повторяет оператор, а не код.
# Перед миссией ждём долго и не на месте (машина на показе бывает медленной: ложный отказ хуже лишних секунд);
# в ручной езде и по кнопке «Сбросить прогон» ждём на месте и коротко — основной цикл пульта в это время стоит,
# а сервер считает пульт отключённым, если состояние старше PilotHub.FRESH_S.
RESET_WAIT_S, RESET_SYNC_S = 10.0, 2.5
RESET_STUCK_S = 30.0           # с: вызов без ответа старше этого — в причине отказа совет перезапустить показ
RESET_T_MAX = 5.0              # с: время судьи в только что начатом прогоне не больше (см. fresh_score)
RESET_BATTERY_GAP = 2.0        # ед.: заряд «полный», если отличается от начального не больше
# Уход одометрии в быстром симуляторе, чтобы поправка по лидару была видна и на репетиции: около 14 см
# на 10 м пути. В Gazebo на спокойной езде выходит 3–8 см, на резких разворотах — больше.
FAST_DRIFT = dict(odom_turn_slip=0.12, odom_turn_scale=0.006, odom_path_scale=0.006)
EVENT_TEXT = {'collision': 'Столкновение', 'hazard_hit': 'Заезд в опасную зону', 'false_collect': 'Ложный сбор',
              'sample_collected': 'Образец собран'}
REASONS = {'finish': 'миссия завершена', 'battery': 'села батарея', 'timeout': 'вышло время прогона'}


class Refuse(Exception):
    """Команду оператора выполнить нельзя; текст показывается на странице."""


# =================================================================================================
# новый прогон судьи с подтверждением
# =================================================================================================

def fresh_score(score, battery, before, battery_start, answered=False):
    """Счёт судьи — уже нового прогона: сообщение новое, прогон идёт, заряд полный, часы судьи пошли заново.

    before — счёт, который был до вызова сброса; answered — судья уже ответил на вызов «сделано».
    Часы пошли заново, если время судьи стало меньше прежнего (в пределах одного прогона оно только растёт;
    так новый прогон узнаётся, даже когда ответ на вызов потерялся, а первые сообщения пульт пропустил).
    Если прежний прогон сам только начался, меньше прежнего время может и не стать: тогда нужен ответ судьи
    и время не больше RESET_T_MAX.
    """
    if not score or score is before or score.get('finished') or battery is None:
        return False
    t, t_before = score.get('t'), (before or {}).get('t')
    if t is None:
        return False
    restarted = (t_before is not None and t < t_before) or (answered and t < RESET_T_MAX)
    return restarted and battery > battery_start - RESET_BATTERY_GAP


class RunReset:
    """Сброс судьи, который считается удавшимся только когда новый прогон подтверждён его счётом.

    Один объект на всё время работы пульта. start() начинает ожидание, poll() вызывается раз за разом и
    возвращает None, пока ответа нет, затем (True, сообщение) или (False, причина). На подтверждение даётся
    wait_s секунд по часам машины; вызов сброса за одно ожидание отправляется один раз, повторов нет.

    Одновременно у судьи висит не больше одного вызова: если прежний ещё без ответа, start() новый не
    отправляет, а ждёт результата прежнего. Иначе запоздавший вызов обнулил бы часы, образцы и заряд судьи
    уже под работающим агентом. По возрасту вызов не забывается (запоздать он может на сколько угодно):
    если судью перезапустили и ответа не будет никогда, показ перезапускают целиком — об этом сказано
    в причине отказа.

    send() отправляет вызов сброса и возвращает функцию «что ответил судья»: None — ответа ещё нет, иначе
    (успех, сообщение). Если отправить нельзя (сервис ещё не виден), send() возвращает None — попробуем
    на следующем опросе. read() возвращает (последний счёт судьи, заряд).
    """

    def __init__(self, send, read, battery_start, clock=time.monotonic):
        self._send, self._read, self._battery_start, self._clock = send, read, battery_start, clock
        self._t0 = self._sent_at = self._before = self._reply = self._answer = None
        self._wait_s = RESET_WAIT_S
        self.result = (False, 'сброс не вызывался')

    def hanging(self):
        """Прежний вызов отправлен, а ответа судьи на него до сих пор нет."""
        if self._reply is None:
            return False
        if self._answer is None:
            self._answer = self._reply()
        return self._answer is None

    def start(self, wait_s=RESET_WAIT_S):
        if not self.hanging():              # иначе ждём тот же вызов: счёт «до» и функция ответа остаются его
            self._before, self._reply, self._answer = self._read()[0], None, None
        self._t0, self._wait_s, self.result = self._clock(), wait_s, None

    def poll(self):
        if self.result is not None:
            return self.result
        now = self._clock()
        if self._reply is None:
            self._reply, self._sent_at = self._send(), now
        if self._reply is not None and self._answer is None:
            self._answer = self._reply()
        score, battery = self._read()
        answered = self._answer is not None and self._answer[0]
        if self._reply is not None and fresh_score(score, battery, self._before, self._battery_start, answered):
            self.result = (True, (self._answer or (True, ''))[1] or 'судья начал прогон заново')
            return self.result
        refused = self._answer is not None and not self._answer[0]
        if not refused and now - self._t0 < self._wait_s:
            return None
        self.result = (False, self._why(score, battery, refused))
        return self.result

    def _why(self, score, battery, refused):
        if self._reply is None:
            return 'у судьи нет сервиса /did/reset'
        if refused:
            return f'судья отказал: {self._answer[1]}'
        if self._answer is None:
            age = self._clock() - self._sent_at
            if age < RESET_STUCK_S:
                return f'судья не ответил на сброс за {self._wait_s:g} с'
            return (f'судья не отвечает на сброс уже {age:.0f} с; вызов остаётся у судьи, новый не отправляется — '
                    'перезапустите показ (Ctrl+C и та же команда)')
        s = score or {}
        seen = (f"время судьи {s.get('t', '?')} с" + (', прогон закончен' if s.get('finished') else '')
                + ('' if battery is None else f', заряд {_num(battery)}'))
        return f'судья ответил на сброс, но счёт нового прогона за {self._wait_s:g} с не пришёл: {seen}'


# =================================================================================================
# мир: быстрый симулятор
# =================================================================================================

class FastWorld:
    """Быстрый симулятор как мир пульта: RobotIO и то, что в Gazebo делает стенд."""

    backend = 'fastsim'
    dt = TICK_S
    score_delay = 0.0           # итог судьи известен сразу

    def __init__(self, arena, level, seed, rules='base', drift=True):
        self.arena, self.level, self.seed, self.rules_name = arena, level, int(seed), rules
        self.rules = Rules(**SCIENCE) if rules == 'science' else Rules()
        self.scenario = generate(level, self.seed, arena)
        self._sim_args = dict(lidar_hz=5.0, **(FAST_DRIFT if drift else {}))
        self.respawn()

    def respawn(self):
        """Робот снова на базе, судья начинает прогон заново."""
        self.sim = FastSim(self.arena, self.scenario, self.rules, seed=self.seed, **self._sim_args)

    can_respawn = True

    # --- RobotIO -----------------------------------------------------------------------------

    def observe(self):
        return self.sim.observe()

    def command(self, v, w):
        self.sim.command(v, w)

    def collect(self):
        return self.sim.collect()

    def finish(self):
        return self.sim.finish()

    def advance(self):
        self.sim.advance()

    # --- стенд -------------------------------------------------------------------------------

    def reset_begin(self):
        """Новый прогон судьи с того места, где стоит робот: время, заряд и образцы — с начала."""
        s = self.sim
        s.judge = Judge(self.scenario, self.arena, self.rules, seed=self.seed)
        s.t = 0.0
        s._next_sensor = s._next_lidar = 0.0
        s.judge.step(0.0, s.x, s.y)

    def reset_poll(self):
        """Тот же договор, что у стенда: успех — только когда новый прогон подтверждён счётом судьи."""
        j = self.sim.judge
        if j.done or j.t > RESET_T_MAX or j.battery <= self.rules.battery_start - RESET_BATTERY_GAP:
            return False, 'судья быстрого симулятора не начал прогон заново'
        return True, 'судья начал прогон заново'

    def new_run(self):
        self.reset_begin()
        return self.reset_poll()

    def linked(self):
        return True

    def settle_left(self):
        return 0.0

    def score(self):
        return self.sim.judge.score()

    def world_log(self):
        return self.sim.judge.world_log

    def scenario_dict(self):
        return self.scenario.to_dict()

    def truth(self):
        j = self.sim.judge
        return {'soils': [asdict(z) for z in j.soils], 'hazards': [asdict(z) for z in j.hazards],
                'remaining': [[p[0], p[1]] for _, p in sorted(j.remaining.items())],
                'collected': [list(self.scenario.samples[i]) for i, _ in j.collected],
                'robot': [round(self.sim.x, 3), round(self.sim.y, 3), round(self.sim.th, 3)]}


# =================================================================================================
# пульт
# =================================================================================================

class Pilot:

    def __init__(self, arena, world, level, seed, slam_nav=False):
        # arena — готовая карта. В режиме карты SLAM она нужна только mapper'у для чисел на экране;
        # всё, по чему робот едет, — self.nav.
        self.arena, self.world, self.level, self.seed = arena, world, level, int(seed)
        self.rules = world.rules
        self.slam_nav = bool(slam_nav)
        self.follower = Follower()
        self.mapper = OccupancyMapper(arena)
        if self.slam_nav:
            self.nav = SlamArena()              # пусто, пока SLAM не прислал первую сетку
            self.graph = None
            self.tracker = SlamPose(self._slam_reading, clock=lambda: self.t)
        else:
            self.nav = arena
            self.graph = CostGraph(arena)
            self.tracker = PoseTracker(arena) if PoseTracker else None
        self._nav_ver = 0
        self._nav_ms = 0.0
        self._grid_ver = 0                  # последняя сетка SLAM, проверенная во время миссии
        self._grid_bad = False              # она непригодна: в миссии это запрет на движение
        self._slam_fault = None             # что не так со SLAM (текст), пока он молчит
        self._slam_jumps = 0                # сколько скачков позы от SLAM уже учтено
        self._progress = None               # срок текущего подъезда: {'leg', 't0', 'limit', 'x', 'y', 'still'}
        self._explore = None                # объезд границ увиденного: {'visited', 'dead', 'done', 't0'}
        self.mode = 'idle'                  # settle | idle | drive | home | mission
                                            # (mission без self.bot — миссия ждёт нового прогона судьи)
        self.route = []                     # [{'x', 'y', 'done', 'pts'}]: точки оператора и путь до каждой
        self.zones = []                     # ограничения оператора, не скрытые свойства среды
        self._zone_blocked = np.zeros(arena.free.shape, dtype=bool)
        self.trail = []
        self.log = []                       # события для страницы: [{'t', 'kind', 'text'}]
        self.note = {'tone': 'info', 'text': 'Поставьте точку на карте и нажмите «Ехать»'}
        self.pose = (BASE[0], BASE[1], 0.0)
        self.odom = self.pose
        self.t = 0.0
        self.wall0 = time.time()
        self.battery = self.rules.battery_start
        self.sensor = 0.0
        self.distance = 0.0
        self.arrivals = []                  # ошибки приезда в конечные точки, м
        self.mission = None
        self.bot = self.rec = None
        self._after_home = None             # что сделать по приезде на базу
        self._scan = None
        self._front = math.inf
        self._escape = None
        self._stuck = None
        self._last_cmd = (0.0, 0.0)
        self._map_cache = (-1, None)
        self._nav_cache = (-1, None)
        self._belief_cache = (-1e9, None)
        self._finish_at = None
        self._reset = None                  # агент миссии, которая ждёт подтверждения нового прогона судьи
        self._last_odom = None
        self._rerun_at = 0.0
        self._gaps = deque(maxlen=300)      # (время по часам машины, сколько времени симуляции прошло между тактами)
        self._slow_until = 0.0
        self._guard = _Guard(self)
        self._log(0.0, 'info', 'Пульт запущен')

    # ======================================================================================
    # один такт
    # ======================================================================================

    def tick(self, obs, gap=TICK_S):
        """gap — сколько времени симуляции прошло с прошлого такта (в норме 0,1 с)."""
        w = self.world
        self.t = obs.t
        wall = time.monotonic()
        self._gaps.append((wall, gap))
        if gap > LAG_S:
            # Робот какое-то время ехал вслепую по последней команде: дальше — медленно, пока такты не выровняются.
            self._slow_until = wall + 5.0
        self.battery = obs.battery
        if obs.sensor is not None:
            self.sensor = obs.sensor
        for ev in obs.events:
            self._on_event(ev)

        if self.mode == 'mission' and self.bot:
            if self.slam_nav:
                self._check_grid()                  # до команды агента: по непригодной новой карте ехать нельзя
            self.bot.tick(obs, self._guard)         # агент сам поправляет позу (трекер у нас общий)
            x, y, th = self._to_map(obs.x, obs.y, obs.th)
        elif self.tracker:
            x, y, th = self.tracker.update(obs.x, obs.y, obs.th, obs.scan, obs.scan_pose, obs.scan_step)
        else:
            x, y, th = obs.x, obs.y, obs.th
        self.odom, self.pose = (obs.x, obs.y, obs.th), (x, y, th)

        if obs.scan is not None:
            # Карта строится из позы на момент скана и в системе эталонной карты (она сдвинута на доли клетки).
            sx, sy, sth = self._to_map(*(obs.scan_pose or (obs.x, obs.y, obs.th)))
            shift = getattr(self.tracker, 'shift', (0.0, 0.0))
            self.mapper.update(sx + shift[0], sy + shift[1], sth, obs.scan, obs.scan_step)
            self._scan = (sx, sy, sth, obs.scan, obs.scan_step)
            self._front = float(min(obs.scan[:20].min(), obs.scan[-20:].min()))

        if self.slam_nav:
            self._watch_slam()
            if self.mode != 'mission' or self.bot is None:
                self._grid_bad = False      # вне миссии пригодность сетки проверяет _refresh_nav
                self._refresh_nav()         # в миссии агент едет по снимку карты, сделанному на её старте

        if self._last_odom is not None:
            self.distance += math.hypot(obs.x - self._last_odom[0], obs.y - self._last_odom[1])
        self._last_odom = (obs.x, obs.y)
        if not self.trail or math.dist(self.trail[-1], (x, y)) >= 0.03:
            self.trail.append((x, y))
            if len(self.trail) > 6000:
                self.trail = self.trail[::2]

        left = w.settle_left()
        if left > 0.0:
            if self.mode != 'settle':
                self.mode = 'settle'
            return self._command(0.0, 0.0)
        if self.mode == 'settle':
            self.mode = 'idle'
            self._say('info', 'Робот готов: можно ставить точки')

        if self.mode == 'mission':
            return self._mission_tick(obs)
        if self.mode in ('drive', 'home'):
            if obs.done:
                warn = self._fresh_run()
                if warn:
                    self._say('bad', warn)
            return self._drive(obs, x, y, th)
        self._command(0.0, 0.0)

    def _slam_reading(self):
        """Что SLAM сообщает SlamPose: поправка позы и возрасты от стенда плюс пригодность последней сетки."""
        r = self.world.slam_tf()
        if not self._grid_bad or r is None:
            return r
        r = dict(r) if isinstance(r, dict) else {'tf': r, 'tf_age': 0.0, 'grid_age': None}
        r['grid_ok'] = False
        return r

    def _check_grid(self):
        """В миссии агент едет по снимку карты, но каждая свежая сетка проверяется: если на ней роботу негде
        стоять, SlamPose сообщает о потере положения, и защита агента останавливает езду, как при молчании SLAM."""
        ver, values = self.world.slam_grid()
        if values is None or ver == self._grid_ver:
            return
        self._grid_ver = ver
        self._grid_bad = not bool((SlamArena(values, near=self.pose[:2]).clear >= INFLATE).any())

    def _refresh_nav(self):
        """Свежая сетка SLAM → новая арена и граф путей. Путь, который перестал быть проходимым, строится заново.

        Принятая версия, арена, граф и то, что видит оператор, меняются вместе. Если на новой сетке роботу
        негде встать, прежняя карта не остаётся в силе: графа нет, путь стёрт, езда остановлена.
        """
        ver, values = self.world.slam_grid()
        if values is None or ver == self._nav_ver:
            return
        t0 = time.perf_counter()
        nav = SlamArena(values, near=self.pose[:2])
        had = self.graph is not None
        usable = bool((nav.clear >= INFLATE).any())
        self._nav_ver, self.nav, self.graph = ver, nav, CostGraph(nav) if usable else None
        if not usable:
            self.follower.set_path([])
            text = 'Новая карта от SLAM Toolbox непригодна для езды: на ней нет места, где робот может стоять'
            if self.mode in ('drive', 'home'):
                self._halt(f'{text}. Робот остановлен; когда карта восстановится, нажмите «Ехать» снова')
            elif had:
                self._say('bad', f'{text}. Ехать нельзя, пока не придёт пригодная')
            self._nav_ms = (time.perf_counter() - t0) * 1e3
            return
        if not had and ver > 1 and self._nav_cache[1] is not None:
            self._say('info', 'Карта от SLAM Toolbox снова пригодна для езды')
        if self.zones:
            self._apply_zones()             # новая сетка не отменяет ограничения оператора
        if self.follower.active and not self._path_ok(self.follower.pts[self.follower.i:]):
            self.follower.set_path([])      # цель и путь проверятся заново в _drive
        if self._explore and not self._explore['done']:
            # Границу, к которой ехали, уже видно издалека — незачем доезжать.
            now = frontiers(nav)
            for leg in self._pending():
                f = leg.get('frontier')
                if f and all(math.dist(f, g[:2]) > EXPLORE_NEAR for g in now):
                    leg['done'] = True
                    self.follower.set_path([])
        self._nav_ms = (time.perf_counter() - t0) * 1e3

    def _path_ok(self, pts):
        """Все точки пути и отрезки между ними проходят не ближе INFLATE к преградам нынешней карты."""
        nav = self.nav
        def safe(px, py):
            ix, iy = nav.w2g(px, py)
            return (nav.inside(ix, iy) and nav.clearance(px, py) >= INFLATE
                    and not self._zone_blocked[iy, ix])

        for k, (px, py) in enumerate(pts):
            if not safe(px, py):
                return False
            if k:
                qx, qy = pts[k - 1]
                n = int(math.ceil(math.hypot(px - qx, py - qy) / (0.5 * nav.res)))
                if any(not safe(qx + (px - qx) * j / n, qy + (py - qy) * j / n) for j in range(1, n)):
                    return False
        return True

    def _watch_slam(self):
        """Здоровье SLAM каждый такт: замолчал — езда прекращается, вернулся или передумал — путь строится заново.

        Признаки считает SlamPose (did/slam_map.py). В миссии по lost и unsure останавливается сам агент
        (did/agent.py::_guard — та же защита, что при потере положения по лидару; она же сообщает сторожам
        простоя, что это ожидание), пульт только показывает причину оператору.
        """
        st = self.tracker.stats
        fault = st['fault']
        if fault and not self._slam_fault:
            if self.mode in ('drive', 'home'):
                self._halt(f'{fault}. Робот остановлен; когда SLAM восстановится, нажмите «Ехать» снова')
            elif self.mode == 'mission':
                self._say('bad', f'{fault}. Агент стоит и ждёт; не дождётся — вернётся на базу медленно, по лидару')
            else:
                self._say('bad', f'{fault}. Ехать нельзя, пока он не восстановится')
        elif self._slam_fault and not fault:
            self.follower.set_path([])
            self._say('info', 'SLAM Toolbox снова на связи: путь строится заново'
                      + ('' if self.mode == 'mission' else '. Можно ехать'))
        self._slam_fault = fault
        if st['relocations'] != self._slam_jumps:
            self._slam_jumps = st['relocations']
            self.follower.set_path([])
            self._progress = None
            self._log(self.t, 'bad', 'SLAM Toolbox сдвинул позу скачком: пережидаю и строю путь заново')

    def _halt(self, text):
        """Прекратить ручную езду по причине, которую оператор должен увидеть."""
        self._command(0.0, 0.0)
        self.mode = 'idle'
        self.follower.set_path([])
        self._escape = self._stuck = self._progress = None
        self._drop_pending_mission()
        if self._explore and not self._explore['done']:
            self._explore, self.route = None, []
        self._say('bad', text)

    def _need_map(self):
        if self.graph is None:
            raise Refuse('Пригодной карты от SLAM Toolbox пока нет: подождите несколько секунд')
        st = self.tracker.stats
        if not st['ready']:
            raise Refuse('SLAM Toolbox ещё не прислал поправку позы: подождите несколько секунд')
        if st['fault']:
            raise Refuse(f"{st['fault']}. Ехать нельзя, пока он не восстановится")

    def _fresh_run(self):
        """Ручная езда судье не подотчётна: если его прогон закончился (заряд, время), начинаем новый.

        Возвращает предупреждение для оператора, если судья новый прогон не начал (езде это не мешает).
        """
        if not (self.world.score() or {}).get('finished') or time.monotonic() < self._rerun_at:
            return None
        self._rerun_at = time.monotonic() + 10.0            # судья без сервиса сброса: не спрашивать каждый такт
        self._command(0.0, 0.0)                             # new_run() ждёт на месте: робот в это время стоит
        ok, msg = self.world.new_run()
        self._escape = self._stuck = None
        if ok:
            self._log(0.0, 'info', 'Прежний прогон судьи закончился — начат новый: заряд снова полный')
            return None
        return f'Судья не начал новый прогон: {msg}. Ручная езда идёт на прежнем, законченном прогоне'

    def _say_go(self, text, warn):
        """Сообщение о начале езды вместе с предупреждением _fresh_run(), если оно есть."""
        self._say('bad' if warn else 'info', f'{text}. {warn}' if warn else text)

    def _to_map(self, x, y, th):
        return self.tracker.to_map(x, y, th) if self.tracker else (x, y, th)

    def _lag(self):
        """Запаздывание управления за последние 10 с: худший промежуток между тактами и замедлен ли робот."""
        now = time.monotonic()
        recent = [g for t, g in self._gaps if now - t <= 10.0]
        return {'max': round(max(recent, default=TICK_S), 2), 'late': sum(g > LAG_S for g in recent),
                'slow': now < self._slow_until}

    def _command(self, v, w):
        self._last_cmd = (v, w)
        self.world.command(v, w)

    def _on_event(self, ev):
        kind = ev.get('type')
        text = EVENT_TEXT.get(kind, kind)
        if kind == 'sample_collected':
            text = f"Образец собран: всего {ev.get('collected', '?')}"
        self._log(ev.get('t', self.t), 'bad' if kind in ('collision', 'hazard_hit', 'false_collect') else 'ok',
                  text, x=ev.get('x'), y=ev.get('y'))
        if kind == 'collision' and self.mode in ('drive', 'home'):
            self._escape = {'until': self.t + 1.5, 'v': self._escape_v(back=self._last_cmd[0] >= 0)}

    def _escape_v(self, back):
        """Отъезд после столкновения — только туда, где лидар видит свободное место (как у агента)."""
        if not self._scan:
            return -0.10 if back else 0.10
        # Скан снят из позы self._scan[:3], робот с тех пор мог повернуться: стороны считаются от нынешнего курса.
        v, turn = escape_plan(self._scan[3], self._scan[4], back, moved=moved_since(self._scan[:3], self.pose))
        return 0.0 if turn else v                # оператор рядом: разворот на месте оставляем ему

    def _log(self, t, kind, text, **data):
        self.log.append({'t': round(float(t), 1), 'kind': kind, 'text': text,
                         **{k: v for k, v in data.items() if v is not None}})
        del self.log[:-60]

    def _say(self, tone, text):
        """Сообщение оператору: видно на странице крупно и остаётся в списке событий."""
        self.note = {'tone': tone, 'text': text}
        self._log(self.t, tone, text)

    # ======================================================================================
    # езда по точкам оператора
    # ======================================================================================

    def _pending(self):
        return [p for p in self.route if not p['done']]

    def _drive(self, obs, x, y, th):
        if self.slam_nav and self._overdue(obs, x, y):
            return
        if self._escape:
            if obs.t < self._escape['until']:
                v = self._escape['v']
                if v and self._scan and corridor_room(self._scan[3], self._scan[4], back=v < 0, moved=moved_since(
                        self._scan[:3], (x, y, th))) < ESCAPE_STOP:
                    v = self._escape['v'] = 0.0         # преграда уже близко и с этой стороны: дальше стоим
                return self._command(v, 0.0)
            self._escape = self._stuck = None
            self.follower.set_path([])
        if self.slam_nav and self.tracker.stats['unsure']:
            self._stuck = None
            return self._command(0.0, 0.0)              # SLAM только что передумал, где робот: ждём новую карту
        while True:
            todo = self._pending()
            exploring = bool(self._explore) and not self._explore['done']
            if not todo and exploring and self._next_frontier(x, y):
                continue
            if not todo:
                self._command(0.0, 0.0)
                self.mode = 'idle'
                if exploring:
                    self._explore_done(obs)
                return
            leg = todo[0]
            if self.slam_nav and not self.follower.active and not self._target_ok(leg):
                # Карта изменилась: цель теперь в преграде или вне увиденного пола. Ехать «к ближайшему месту» нельзя.
                if exploring:
                    self._explore['dead'].append(leg['frontier'])
                    leg['done'] = True
                    continue
                return self._halt(f'Точка {self.route.index(leg) + 1} на новой карте SLAM оказалась в преграде или вне '
                                  'увиденного пола: маршрут остановлен. Поставьте точку заново')
            target = (leg['x'], leg['y'])
            if not self.follower.active:
                pts, _ = self.graph.plan((x, y), target)
                if pts is not None and self.slam_nav:
                    if not self._path_ok(pts + [target]):
                        pts = None
                    elif self._progress and self._progress['leg'] is leg and self._progress['limit'] is None:
                        self._progress['limit'] = LEG_BASE_S + LEG_S_PER_M * path_length([(x, y)] + pts + [target])
                if pts is None and exploring:
                    self._explore['dead'].append(leg['frontier'])   # карта изменилась, проезда туда больше нет
                    leg['done'] = True
                    continue
                if pts is None:
                    self._command(0.0, 0.0)
                    self.mode = 'idle'
                    return self._say('bad', 'Пути к точке нет: маршрут остановлен')
                self.follower.set_path(pts + [target])
                leg['pts'] = pts + [target]
            last = len(todo) == 1 and not exploring
            v, w, arrived = self.follower.step(x, y, th, tol=0.05 if last else 0.12,
                                               v_max=SLOW_V if time.monotonic() < self._slow_until else V_MAX)
            if not arrived:
                break
            if last and abs(obs.v) > 0.03:
                return self._command(0.0, 0.0)          # в конечной точке ждём полной остановки
            leg['done'] = True
            leg['pts'] = []
            self.follower.set_path([])
            self._stuck = None
            if last:
                self._command(0.0, 0.0)
                return self._arrived(obs, x, y, target)
            if exploring:
                self._explore['dead'].append(leg['frontier'])       # доехали, а граница осталась — второй раз не едем
                continue
            self._log(obs.t, 'info', f'Точка {self.route.index(leg) + 1} пройдена')     # и сразу к следующей
        if v > 0.0 and self._front < 0.15:
            v = 0.0                                     # лидар видит преграду вплотную по курсу
        self._watch_stuck(obs, x, y, v)
        self._command(v, w)

    def _target_ok(self, leg):
        """Цель подъезда на нынешней карте SLAM: свободна и не ближе INFLATE к преграде.

        Цель, оказавшуюся чуть ближе к стене (карта уточнилась на клетку), передвигаем на ближайшее разрешённое
        место, как при постановке точки; занятую, неувиденную или далёкую от разрешённого — отклоняем.
        """
        nav = self.nav
        ix, iy = nav.w2g(leg['x'], leg['y'])
        if not nav.inside(ix, iy) or not nav.free[iy, ix]:
            return False
        if nav.clear[iy, ix] >= INFLATE:
            return True
        node = self.graph.node(leg['x'], leg['y'])
        nx, ny = float(self.graph.xs[node]), float(self.graph.ys[node])
        if math.dist((leg['x'], leg['y']), (nx, ny)) > SNAP_MAX:
            return False
        leg['x'], leg['y'] = round(nx, 3), round(ny, 3)
        return True

    def _overdue(self, obs, x, y):
        """Сроки подъезда в режиме карты SLAM. True — езда прекращена (сообщение оператору уже показано).

        Два срока: робот не сдвинулся на PROGRESS_M за NO_PROGRESS_S (стоит перед преградой, которой нет на
        карте: лидар обнуляет скорость, и сторож застревания по времени команды «вперёд» молчит) и весь
        подъезд дольше LEG_BASE_S + LEG_S_PER_M на метр пути. В объезде такая граница бросается, и робот едет
        к следующей; у точки оператора езда останавливается.
        """
        todo = self._pending()
        leg = todo[0] if todo else None
        p = self._progress
        if leg is None:
            self._progress = None
            return False
        if p is None or p['leg'] is not leg or obs.t < p['t0']:
            p = self._progress = {'leg': leg, 't0': obs.t, 'limit': None, 'x': x, 'y': y, 'still': obs.t}
        if math.hypot(x - p['x'], y - p['y']) >= PROGRESS_M:
            p['x'], p['y'], p['still'] = x, y, obs.t
        if obs.t - p['still'] >= NO_PROGRESS_S:
            why = f'робот {NO_PROGRESS_S:.0f} с не продвигается (преграда, которой нет на карте, или путь не проходим)'
        elif obs.t - p['t0'] >= (p['limit'] or LEG_BASE_S):
            why = f"подъезд занял больше {p['limit'] or LEG_BASE_S:.0f} с"
        else:
            return False
        self._progress = None
        if self._explore and not self._explore['done']:
            self._log(obs.t, 'bad', f'Граница увиденного брошена: {why}')
            self._explore['dead'].append(leg['frontier'])
            leg['done'] = True
            self._escape = self._stuck = None
            self.follower.set_path([])
            return False
        self._halt(f'Езда остановлена: {why}. Поставьте точку заново или нажмите «Ехать»')
        return True

    def _arrived(self, obs, x, y, target):
        err = math.dist((x, y), target)
        self.arrivals.append(round(err, 3))
        home = self.mode == 'home'
        self.mode = 'idle'
        self.route = [] if home else self.route
        self._say('ok', f'Робот на базе: до её центра {err * 100:.0f} см' if home
                  else f'Приехал: до заданной точки {err * 100:.0f} см')
        after, self._after_home = self._after_home, None
        if home and after:
            after()

    def _watch_stuck(self, obs, x, y, v):
        if self._stuck is None:
            self._stuck = [obs.t, x, y, 0.0, obs.t]
        s = self._stuck
        if v > 0.03:
            s[3] += obs.t - s[4]
        s[4] = obs.t
        if obs.t - s[0] >= 4.0:
            if s[3] >= 3.0 and math.hypot(x - s[1], y - s[2]) < 0.03:
                self._log(obs.t, 'bad', 'Робот не движется: отъезжаю назад и строю путь заново')
                self._escape = {'until': obs.t + 1.5, 'v': self._escape_v(back=True)}
            self._stuck = None

    # --- объезд: достроить карту SLAM ---------------------------------------------------------

    def _next_frontier(self, x, y):
        """Выбрать ближайшую по пути границу увиденного и поставить её целью. False — ехать больше некуда."""
        ex = self._explore
        if ex['visited'] >= EXPLORE_MAX:
            return False
        dist, _ = self.graph.field(x, y)
        best = None
        for fx, fy, _n in frontiers(self.nav):
            if any(math.dist((fx, fy), d) <= EXPLORE_NEAR for d in ex['dead']):
                continue
            node = self.graph.node(fx, fy)
            if node < 0 or not math.isfinite(dist[node]):
                continue
            tx, ty = float(self.graph.xs[node]), float(self.graph.ys[node])
            if math.dist((tx, ty), (x, y)) < 0.15:
                ex['dead'].append((fx, fy))             # стоим вплотную, а границу не видно: её не достроить
                continue
            if best is None or dist[node] < best[0]:
                best = (dist[node], tx, ty, (fx, fy))
        if best is None:
            return False
        ex['visited'] += 1
        self.route = [{'x': round(best[1], 3), 'y': round(best[2], 3), 'done': False, 'pts': [], 'frontier': best[3]}]
        self.follower.set_path([])
        return True

    def _explore_done(self, obs):
        ex = self._explore
        ex['done'] = True
        ex['seconds'] = round(obs.t - ex['t0'], 1)
        ex['left'] = len(frontiers(self.nav))
        self.route = []
        self._say('ok', f"Карта построена по SLAM: {ex['visited']} {_plural(ex['visited'], 'подъезд', 'подъезда', 'подъездов')} "
                  f"к границам увиденного за {_num(ex['seconds'], 0)} с. Поставьте точку — поеду по этой карте")

    def _cmd_explore(self, cmd):
        if not self.slam_nav:
            raise Refuse('Достраивать карту нужно только в режиме карты SLAM: pixi run demo --slam-map')
        self._manual_only()
        self._need_map()
        warn = self._fresh_run()
        self._drop_pending_mission()
        self._explore = {'visited': 0, 'dead': [], 'done': False, 't0': self.t}
        self.route = []
        self.follower.set_path([])
        self.mode = 'drive'
        self._say_go('Строю карту: еду к границам того, что уже видел', warn)
        return 'Строю карту'

    def _plan_route(self, points):
        """Проверить точки и построить путь через них от текущей позы. Возвращает маршрут и заметки."""
        if self.slam_nav:
            self._need_map()
        arena = self.nav
        route, notes = [], []
        prev = self.pose[:2]
        for n, p in enumerate(points, 1):
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                raise Refuse(f'Точка {n}: нужны два числа — x и y в метрах') from None
            ix, iy = arena.w2g(x, y)
            if not (math.isfinite(x) and math.isfinite(y)) or not arena.inside(ix, iy):
                raise Refuse(f'Точка {n} за пределами арены')
            if self.slam_nav and not arena.free[iy, ix] and not arena.occupied[iy, ix]:
                raise Refuse(f'Точка {n} там, где SLAM ещё не видел пола: сначала достройте карту')
            if not arena.free[iy, ix]:
                raise Refuse(f'Точка {n} попала в стену или столб: поставьте её на свободный пол')
            if arena.clear[iy, ix] < INFLATE:
                # Центр робота не подходит к стене ближе 17 см: берём ближайшее разрешённое место.
                node = self.graph.node(x, y)
                nx, ny = float(self.graph.xs[node]), float(self.graph.ys[node])
                notes.append(f'точка {n} сдвинута от стены на {math.dist((x, y), (nx, ny)) * 100:.0f} см')
                x, y = nx, ny
            node = self.graph.node(x, y)
            if self._zone_blocked[self.graph.iy[node], self.graph.ix[node]]:
                raise Refuse(f'Точка {n} в опасной зоне или слишком близко к ней')
            pts, _ = self.graph.plan(prev, (x, y))
            if pts is None:
                raise Refuse(f'К точке {n} нет проезда')
            route.append({'x': round(x, 3), 'y': round(y, 3), 'done': False, 'pts': pts + [(x, y)]})
            prev = (x, y)
        return route, notes

    # ======================================================================================
    # команды оператора
    # ======================================================================================

    def command(self, cmd):
        """Исполнить команду. Возвращает {'ok', 'message'}; отказ — понятной фразой."""
        name = cmd.get('cmd')
        handler = getattr(self, f'_cmd_{name}', None) if isinstance(name, str) else None
        if handler is None:
            return {'ok': False, 'message': f'Нет такой команды: {name}'}
        try:
            message = handler(cmd) or 'Готово'
        except Refuse as e:
            self.note = {'tone': 'bad', 'text': str(e)}
            return {'ok': False, 'message': str(e)}
        return {'ok': True, 'message': message}

    def _manual_only(self):
        if self.mode == 'mission':
            raise Refuse('Идёт автономная миссия: роботом управляет агент. Сначала «Завершить миссию»')
        left = self.world.settle_left()
        if left > 0.0:
            raise Refuse(f'Робот ещё оседает после появления в мире: подождите {left:.0f} с')
        if not self.world.linked():
            raise Refuse('Нет связи со стендом: данные от робота не приходят')

    def _cmd_route(self, cmd):
        if self.mode == 'mission':
            raise Refuse('Идёт автономная миссия: маршрут задаёт агент')
        points = cmd.get('points')
        if not isinstance(points, list):
            raise Refuse('Маршрут — это список точек [[x, y], ...]')
        if len(points) > MAX_POINTS:
            raise Refuse(f'Слишком много точек: не больше {MAX_POINTS}')
        route, notes = self._plan_route(points)
        if self.mode == 'home':
            self.mode = 'drive' if route else 'idle'
        self._drop_pending_mission()
        self._explore = None
        if self.mission and self.mission.get('state') not in ('running', 'to_base'):
            self.mission = self.bot = self.rec = None
        self.route = route
        self.follower.set_path([])
        if not route:
            self.mode = 'idle' if self.mode == 'drive' else self.mode
            self._command(0.0, 0.0)
            self.note = {'tone': 'info', 'text': 'Маршрут очищен'}
            return 'Маршрут очищен'
        length = sum(path_length(p['pts']) for p in route)
        text = f'Маршрут из {len(route)} {_plural(len(route), "точки", "точек", "точек")}: путь {_num(length)} м'
        if notes:
            text += ' (' + '; '.join(notes) + ')'
        self.note = {'tone': 'info', 'text': text + ('' if self.mode == 'drive' else '. Нажмите «Ехать»')}
        return text

    def _apply_zones(self):
        X, Y = self.nav.cell_centers()
        bias = np.ones(X.shape)
        blocked = np.zeros(X.shape, dtype=bool)
        for z in self.zones:
            if z['kind'] == 'danger':
                # Запас для корпуса и отклонения от дискретного пути.
                blocked |= (X - z['x']) ** 2 + (Y - z['y']) ** 2 <= (z['r'] + INFLATE + 0.08) ** 2
            else:
                mask = (X - z['x']) ** 2 + (Y - z['y']) ** 2 <= z['r'] ** 2
                bias[mask] = 4.0
        self._zone_blocked = blocked
        self.graph.set_cost(bias=bias, hard_forbidden=blocked)

    def _cmd_zones(self, cmd):
        self._manual_only()
        if self.slam_nav:
            self._need_map()
        if self.mode != 'idle':
            raise Refuse('Сначала остановите робота, затем измените зоны')
        raw = cmd.get('zones')
        if not isinstance(raw, list) or len(raw) > 20:
            raise Refuse('Нужно не больше 20 зон')
        zones = []
        for z in raw:
            try:
                kind = z['kind']
                x, y, r = float(z['x']), float(z['y']), float(z['r'])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise Refuse('Зоне нужны тип, координаты и радиус') from None
            if kind not in ('yellow', 'danger') or not all(map(math.isfinite, (x, y, r))) or not 0.15 <= r <= 1.2:
                raise Refuse('Тип зоны: yellow или danger; радиус от 0,15 до 1,2 м')
            ix, iy = self.nav.w2g(x, y)
            if not self.nav.inside(ix, iy) or not self.nav.free[iy, ix]:
                raise Refuse('Центр зоны должен быть на свободном полу')
            if kind == 'danger' and any(math.dist((x, y), p) <= r + INFLATE + 0.08 for p in (BASE, self.pose[:2])):
                raise Refuse('Опасная зона не должна перекрывать робота или базу')
            zones.append({'kind': kind, 'x': x, 'y': y, 'r': r})
        previous = self.zones
        self.zones = zones
        self._apply_zones()
        try:
            route, _ = self._plan_route([[p['x'], p['y']] for p in self._pending()])
            # Обязателен доступный путь домой, даже когда заданных точек нет.
            back, _ = self.graph.plan(self.pose[:2], BASE)
            if back is None:
                raise Refuse('Зоны перекрыли возвращение на базу')
        except Refuse:
            self.zones = previous
            self._apply_zones()
            raise
        self.route = route
        self.follower.set_path([])
        self._say('info', 'Зоны сохранены, маршрут перестроен' if zones else 'Зоны убраны')
        return self.note['text']

    def _cmd_go(self, cmd):
        self._manual_only()
        if not self._pending():
            raise Refuse('Сначала поставьте точку на карте')
        if self.slam_nav:
            self._need_map()
        warn = self._fresh_run()
        self._progress = None
        self.mode = 'drive'
        self._drop_pending_mission()
        self.follower.set_path([])
        self._say_go(f'Еду по маршруту: точек {len(self._pending())}', warn)
        return 'Еду'

    def _drop_pending_mission(self):
        """Оператор передумал, пока робот ехал на базу перед миссией."""
        self._after_home = None
        if self.mission and self.mission.get('state') == 'to_base':
            self.mission = None

    def _cmd_stop(self, cmd):
        self._command(0.0, 0.0)
        self._drop_pending_mission()
        if self.mode == 'mission':
            self._end_mission('aborted', 'Миссия остановлена оператором')
            return 'Миссия остановлена'
        if self.mode in ('drive', 'home'):
            self.mode = 'idle'
            self.follower.set_path([])
            if self._explore and not self._explore['done']:
                self._explore, self.route = None, []
                self._say('info', 'Стоп: объезд прерван, карта остаётся какая есть')
                return 'Робот стоит'
            self._say('info', 'Стоп: робот остановлен, маршрут сохранён')
        return 'Робот стоит'

    def _cmd_home(self, cmd):
        self._manual_only()
        self._drop_pending_mission()
        self._go_home()
        return 'Еду на базу'

    def _go_home(self, after=None):
        warn = self._fresh_run()
        self._explore = None
        route, _ = self._plan_route([BASE])
        self.route, self.mode, self._after_home = route, 'home', after
        self.follower.set_path([])
        self._say_go('Возвращаюсь на базу', warn)

    def _cmd_reset(self, cmd):
        w = self.world
        self._command(0.0, 0.0)
        self.mode = 'idle'
        self.route, self.trail, self.arrivals = [], [], []
        self.mission = self.bot = self.rec = self._after_home = self._escape = self._stuck = self._explore = None
        self._progress = self._reset = None
        self.follower.set_path([])
        self.mapper.reset()
        self.distance = 0.0
        if getattr(w, 'can_respawn', False):
            w.respawn()
            if not self.slam_nav:
                self.tracker = PoseTracker(self.arena) if PoseTracker else None
            self._last_odom = None
            text = 'Сброс: робот на базе, карта и след стёрты, прогон начат заново'
        else:
            ok, msg = w.new_run()
            text = ('Сброс: след стёрт, карта SLAM Toolbox остаётся' if self.slam_nav else 'Сброс: карта и след стёрты') \
                + (', судья начал прогон заново' if ok else f'. Судья не начал новый прогон: {msg}')
            if math.dist(self.pose[:2], BASE) > BASE_NEAR:
                text += '. Робот не на базе — нажмите «Домой»'
            if not ok:
                self._say('bad', text)
                return text
        self._say('info', text)
        return text

    def _cmd_speed(self, cmd):
        raise Refuse('Скорость времени меняется только в быстром симуляторе')

    # ======================================================================================
    # автономная миссия
    # ======================================================================================

    def _cmd_mission(self, cmd):
        if self.zones:
            raise Refuse('Зоны оператора применяются к заданному маршруту. Уберите их перед автономной миссией')
        agent = cmd.get('agent', 'adaptive')
        if agent not in MISSION_AGENTS or agent not in PRESETS:
            raise Refuse(f'Нет агента «{agent}». Есть: {", ".join(MISSION_AGENTS)}')
        if self.mode == 'mission':
            raise Refuse('Миссия уже идёт')
        self._manual_only()
        if self.slam_nav:
            self._need_map()
            cfg = make_config(agent)
            if not (cfg.guard and cfg.localize):
                # Остановку при молчании SLAM в миссии исполняет защита агента: без неё в этом режиме не едем.
                raise Refuse(f'Агент «{MISSION_AGENTS[agent]}» работает без защиты от потери положения: '
                             'в режиме карты SLAM его запускать нельзя')
        self._explore = None
        self.mission = {'agent': agent, 'label': MISSION_AGENTS[agent], 'state': 'to_base',
                        'total': LEVELS[self.level]['samples'], 'collected': 0}
        if math.dist(self.pose[:2], BASE) > BASE_NEAR:
            self._go_home(after=lambda: self._start_mission(agent))
            self.note = {'tone': 'info', 'text': 'Миссия начнётся с базы: сначала возвращаюсь'}
            return 'Сначала на базу, затем миссия'
        self._start_mission(agent)
        if self.mission.get('state') == 'refused':
            raise Refuse(self.note['text'])
        return 'Миссия запущена' if self.bot else 'Жду новый прогон судьи, затем миссия'

    def _start_mission(self, agent):
        """Чистый прогон судьи (полный заряд, время с нуля, все образцы на месте) — и только затем агент.

        Сброс подтверждается счётом судьи; пока подтверждения нет, пульт остаётся в режиме миссии без агента
        (_reset_tick): робот стоит, старый счёт и старый признак «прогон закончен» ни на что не влияют.
        """
        self._command(0.0, 0.0)
        self.bot = self.rec = None                  # агент прошлой миссии не должен получить ни одного такта
        self.follower.set_path([])
        self.mode = 'mission'
        self._finish_at = None
        self._reset = agent
        self.mission = {'agent': agent, 'label': MISSION_AGENTS[agent], 'state': 'reset',
                        'total': LEVELS[self.level]['samples'], 'collected': 0}
        self.world.reset_begin()
        self._say('info', 'Жду, пока судья начнёт новый прогон: миссия стартует сразу после этого')
        self._reset_tick()                          # быстрый симулятор подтверждает сразу

    def _reset_tick(self):
        """Ждём подтверждения нового прогона. Подтверждён — запускаем агента; нет — миссия не начинается."""
        self._command(0.0, 0.0)
        res = self.world.reset_poll()
        if res is None:
            return
        agent, self._reset = self._reset, None
        if res[0]:
            return self._launch_mission(agent)
        # Без нового прогона агент получил бы остаток заряда и времени после ручной езды: не запускаем.
        self.mode = 'idle'
        self.mission.update(state='refused', fresh_run=False, reason=res[1])
        self._say('bad', f'Судья не начал новый прогон: {res[1]}. Миссия не запущена — '
                  'нажмите «Запустить миссию» ещё раз')

    def watch_reset(self):
        """Опрос ожидающего сброса из основного цикла пульта, по часам машины: такта может и не быть (/clock встал)."""
        if self._reset is not None:
            self._reset_tick()

    def _launch_mission(self, agent):
        cfg = make_config(agent)
        self.rec = Recorder()
        # В режиме карты SLAM агент получает снимок арены, собранной из сетки SLAM на эту секунду.
        self.bot = Agent(self.nav, cfg, n_samples=LEVELS[self.level]['samples'], rules=self.rules,
                         planner=make_planner(cfg, None, self.seed), recorder=self.rec)
        if self.tracker and cfg.localize:
            self.bot.tracker = self.tracker         # поправка позы не начинается с нуля
        self.route, self.trail = [], []            # на карте остаётся только путь самой миссии
        self.distance = 0.0                       # ручной маршрут и подъезд к базе не входят в новую миссию
        self._last_odom = None
        self.follower.set_path([])
        self.mode = 'mission'
        self._finish_at = None
        self.mission = {'agent': agent, 'label': MISSION_AGENTS[agent], 'state': 'running',
                        'total': LEVELS[self.level]['samples'], 'collected': 0, 'fresh_run': True}
        self._say('info', f'Автономная миссия: агент «{MISSION_AGENTS[agent]}» ищет образцы')

    def _mission_tick(self, obs):
        if self._reset is not None:
            return self._reset_tick()               # obs.done здесь — ещё от прежнего прогона: миссию не завершает
        bot = self.bot
        if bot is None:
            return self._command(0.0, 0.0)
        if self._finish_at is None and (obs.done or bot.finished):
            self._finish_at = time.monotonic() + self.world.score_delay     # дождаться итогового счёта судьи
        if self._finish_at is not None and time.monotonic() >= self._finish_at:
            self._end_mission('finished')

    def _cmd_finish(self, cmd):
        if self.mode != 'mission':
            raise Refuse('Миссия не запущена')
        self._command(0.0, 0.0)
        if self.bot is None:                        # миссия ещё ждёт нового прогона судьи: завершать у судьи нечего
            self._end_mission('aborted', 'Миссия отменена оператором до старта')
            return 'Миссия отменена'
        ok, msg = self.world.finish()
        self._end_mission('finished', f'Миссия завершена оператором: {msg}')
        return f'Миссия завершена: {msg}'

    def _end_mission(self, state, text=None):
        w, bot = self.world, self.bot
        self._command(0.0, 0.0)
        self.mode = 'idle'
        self._finish_at = self._reset = None
        score = dict(w.score() or {})
        m = self.mission or {}
        m.update(state=state, collected=score.get('samples_collected', bot.collected if bot else 0))
        if bot and score:
            try:
                metrics = run_metrics(score, self.rules, bot.journal, w.world_log(), self.rec.plans, self.rec.llm)
                m['result'] = {**score, **metrics}
                trace = self.rec.build(run_id=f'pilot/{bot.cfg.name}/{self.level}-{self.seed}', experiment='pilot',
                                       arm=bot.cfg.name, backend='gazebo' if w.backend == 'gazebo' else 'fastsim',
                                       agent={'name': bot.cfg.name, 'config': bot.cfg.to_dict()},
                                       scenario=w.scenario_dict(), rules=self.rules.to_dict(), result=m['result'],
                                       world=w.world_log(), journal=bot.journal)
                if bot.inv:
                    trace.update(bot.inv.export())
                path = save_trace(trace, RUNS / 'pilot' / bot.cfg.name / f'{self.level}-{self.seed}.json.gz')
                m['trace_file'] = str(path.relative_to(RUNS))
            except Exception as e:                          # noqa: BLE001 — итог важнее записи
                m['result'] = score
                self._log(self.t, 'bad', f'Запись миссии не сохранена: {type(e).__name__}: {e}')
        self.mission = m
        if text is None:
            r = m.get('result') or {}
            back = 'вернулся на базу' if r.get('returned') else 'на базу не вернулся'
            text = (f"Миссия окончена ({REASONS.get(r.get('reason'), 'итог судьи')}): собрано "
                    f"{r.get('samples_collected', 0)} из {r.get('samples_total', m.get('total'))}, {back}, "
                    f"счёт {_num(r.get('score', 0))}")
        self._say('ok' if state == 'finished' else 'info', text)

    def _mission_view(self):
        m = dict(self.mission)
        bot = self.bot
        if not bot:
            return m
        j = bot.journal
        m.update(collected=max(m.get('collected', 0), bot.collected), agent_mode=bot.mode,
                 journal=j.entries[-14:], journal_total=len(j.entries),
                 hypotheses=[{k: h.get(k) for k in ('id', 'statement', 'test', 'status', 'verdict')}
                             for h in j.hypotheses[-8:]],
                 queue=[_brief(s) for s in bot.queue[:5]],
                 hazards=[[round(x, 2), round(y, 2), round(r, 2)] for x, y, r in bot.hazards])
        if self.rec.plans:
            p = self.rec.plans[-1]
            m['plan'] = {'t': p['t'], 'source': p['source'], 'trigger': p['trigger'], 'reasoning': p['reasoning']}
        if self.t - self._belief_cache[0] >= 1.0 or self._belief_cache[1] is None or m['state'] != 'running':
            b = bot.belief
            self._belief_cache = (self.t, {'res': b.res, 'x0': b.x0, 'y0': b.y0, 'w': b.w, 'h': b.h, 'enc': 'sqrt',
                                           'data': encode_grid(np.round(np.sqrt(b.grid()) * 255))})
        m['belief'] = self._belief_cache[1]
        if self.rec.soil and self.rec.soil['snaps']:
            m['soil'] = {**{k: v for k, v in self.rec.soil.items() if k != 'snaps'},
                         'data': self.rec.soil['snaps'][-1]['data']}
        if bot.inv:
            m['inquiries'] = bot.inv.export()['inquiries'][-4:]
        return m

    # ======================================================================================
    # состояние для интерфейса
    # ======================================================================================

    def state(self):
        w = self.world
        x, y, th = self.pose
        ox, oy, oth = self.odom
        fix = (x - ox, y - oy, _wrap(th - oth))
        stats = self.tracker.stats if self.tracker else {}
        todo = self._pending()
        path = []
        if self.mode == 'mission' and self.bot:
            path = list(self.bot.follower.pts[self.bot.follower.i:])
        elif todo:
            first = self.follower.pts[self.follower.i:] if self.follower.active else todo[0]['pts']
            path = [(x, y)] + list(first) + [q for p in todo[1:] for q in p['pts']]
        if self._map_cache[0] != self.mapper.version:
            self._map_cache = (self.mapper.version, self.mapper.to_dict())
        scan = None
        if self._scan is not None:
            sx, sy, sth, r, step = self._scan
            k = max(1, len(r) // 120)
            scan = {'x': round(sx + LIDAR_OFFSET * math.cos(sth), 3), 'y': round(sy + LIDAR_OFFSET * math.sin(sth), 3),
                    'th': round(sth, 4), 'step': (step or 2 * math.pi / len(r)) * k,
                    'r': np.where(np.isfinite(r[::k]), np.round(r[::k] * 100.0), 0).astype(int).tolist()}
        score = w.score() or {}
        left = w.settle_left()
        k = max(1, len(self.trail) // 900)
        return {
            'active': True, 'backend': w.backend, 'level': self.level, 'seed': self.seed,
            'rules': getattr(w, 'rules_name', 'base'), 'linked': w.linked(),
            't': round(self.t, 1), 'wall': round(time.time(), 2), 'uptime': round(time.time() - self.wall0, 1),
            'mode': self.mode, 'settle_left': round(left, 1), 'note': self.note,
            'pose': [round(x, 3), round(y, 3), round(th, 4)],
            'odom': [round(ox, 3), round(oy, 3), round(oth, 4)],
            'fix': {'source': 'slam' if self.slam_nav else 'lidar' if self.tracker else 'odom', 'dx': round(fix[0], 4), 'dy': round(fix[1], 4),
                    'dth': round(fix[2], 4), 'shift': round(math.hypot(fix[0], fix[1]), 4),
                    'scans': stats.get('scans', 0), 'fixes': stats.get('fixes', 0),
                    'inliers': round(float(stats.get('inliers') or 0.0), 3)},
            'base': list(BASE), 'at_base': math.dist((x, y), BASE) <= BASE_NEAR,
            'trail': [[round(px, 2), round(py, 2)] for px, py in self.trail[::k]] + [[round(x, 2), round(y, 2)]],
            'route': [{'x': p['x'], 'y': p['y'], 'done': p['done']} for p in self.route],
            'zones': self.zones,
            'path': [[round(px, 2), round(py, 2)] for px, py in _thin(path, 2)],
            'path_len': round(path_length(path), 2) if len(path) > 1 else 0.0,
            'battery': round(float(self.battery), 2), 'battery_start': self.rules.battery_start,
            'sensor': round(float(self.sensor), 3), 'distance': round(self.distance, 2),
            'time_limit': self.rules.time_limit_s, 'arrivals': self.arrivals[-10:],
            'score': score, 'events': self.log[-30:], 'lag': self._lag(),
            'map': self._map_cache[1], 'scan': scan,
            'mission': self._mission_view() if self.mission else None,
            'agents': [{'id': k, 'label': v} for k, v in MISSION_AGENTS.items() if k in PRESETS],
            'truth': w.truth(),
            'slam': w.slam_view() if hasattr(w, 'slam_view') else None,
            'slam_nav': self.slam_nav,
            'nav': self._nav_view() if self.slam_nav else None,
        }

    def _nav_view(self):
        """Что робот знает о карте в режиме SLAM: сколько пола, сколько границ неувиденного, как идёт объезд,
        жив ли сам SLAM (возраст последней поправки позы и последней сетки)."""
        key = (self._nav_ver, self.graph is not None, self._grid_bad)
        if self._nav_cache[0] != key:
            self._nav_cache = (key, {
                'version': self._nav_ver, 'ready': self.graph is not None and not self._grid_bad,
                'free_m2': round(float(self.nav.free.sum()) * self.nav.res ** 2, 2),
                'frontiers': len(frontiers(self.nav)), 'ms': round(self._nav_ms, 1)})
        ex = self._explore
        st = self.tracker.stats
        age = lambda v: None if v is None or not math.isfinite(v) else round(v, 2)      # noqa: E731
        return {**self._nav_cache[1],
                'slam': {'ok': st['ready'] and not st['fault'], 'ready': st['ready'], 'fault': st['fault'],
                         'tf_age': age(st['tf_age']), 'grid_age': age(st['grid_age']), 'jumps': st['relocations']},
                'explore': None if ex is None else {k: ex[k] for k in ('visited', 'done', 'seconds', 'left') if k in ex}}


class _Guard:
    """RobotIO для агента в миссии: пока управление запаздывает, линейная скорость ограничена."""

    def __init__(self, pilot):
        self._p = pilot

    def command(self, v, w):
        if time.monotonic() < self._p._slow_until:
            v = max(-SLOW_V, min(SLOW_V, v))
        self._p.world.command(v, w)

    def collect(self):
        return self._p.world.collect()

    def finish(self):
        return self._p.world.finish()


def _thin(pts, k):
    return pts[::k] + ([pts[-1]] if pts and (len(pts) - 1) % k else [])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _num(v, digits=1):
    """Число для сообщений оператору: с запятой."""
    return f'{v:.{digits}f}'.replace('.', ',').replace('-', '−')


def _plural(n, one, few, many):
    a, b = abs(n) % 100, abs(n) % 10
    if 10 < a < 20:
        return many
    return one if b == 1 else few if 2 <= b <= 4 else many


def _brief(sg):
    names = {'investigate': 'проверить место', 'explore': 'разведка', 'goto': 'ехать', 'return_base': 'на базу'}
    if sg['type'] == 'return_base':
        return 'на базу'
    return f"{names.get(sg['type'], sg['type'])} ({sg['x']:.1f}; {sg['y']:.1f})"


def dumps(state):
    return json.dumps(state, ensure_ascii=False, separators=(',', ':'), default=_plain).encode('utf-8')


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, '__dataclass_fields__'):
        return asdict(o)
    return str(o)


# =================================================================================================
# быстрый симулятор: поток внутри сервера интерфейса
# =================================================================================================

class FastSession(threading.Thread):
    """Пульт на быстром симуляторе в реальном времени (с множителем скорости)."""

    SPEEDS = (1.0, 2.0, 4.0, 8.0)

    def __init__(self, level, seed, speed=1.0, rules='base'):
        super().__init__(daemon=True, name='did-pilot-fastsim')
        self.world = FastWorld(load_arena(), level, seed, rules=rules)
        self.pilot = Pilot(self.world.arena, self.world, level, seed)
        self.pilot._cmd_speed = self._cmd_speed
        self.speed = float(speed)
        self.snapshot = None
        self.error = None
        self._inbox = queue.Queue()
        self._halt = threading.Event()

    def _cmd_speed(self, cmd):
        try:
            value = float(cmd.get('value'))
        except (TypeError, ValueError):
            raise Refuse('Скорость — число: 1, 2, 4 или 8') from None
        if value not in self.SPEEDS:
            raise Refuse('Скорость времени: 1, 2, 4 или 8')
        self.speed = value
        return f'Скорость времени ×{value:g}'

    def submit(self, cmd, timeout=5.0):
        """Передать команду потоку пульта и дождаться ответа."""
        box = {'done': threading.Event(), 'result': None}
        self._inbox.put((cmd, box))
        if not box['done'].wait(timeout):
            return {'ok': False, 'message': 'Пульт не ответил: симулятор занят'}
        return box['result']

    def stop(self):
        self._halt.set()

    def run(self):
        world, pilot = self.world, self.pilot
        next_tick = time.monotonic()
        last_state = 0.0
        try:
            while not self._halt.is_set():
                while True:
                    try:
                        cmd, box = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    box['result'] = pilot.command(cmd)
                    box['done'].set()
                    last_state = 0.0                        # ответ на команду виден на странице сразу
                pilot.tick(world.observe())
                world.advance()
                now = time.monotonic()
                if now - last_state >= STATE_S * 0.8:
                    last_state = now
                    self.snapshot = {**pilot.state(), 'speed': self.speed, 'speeds': list(self.SPEEDS)}
                next_tick += world.dt / self.speed
                if next_tick < now - 0.5:                   # отстали (машина занята) — не навёрстываем рывком
                    next_tick = now
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except Exception as e:                              # noqa: BLE001 — причина уходит на страницу
            self.error = f'{type(e).__name__}: {e}'
            raise


# =================================================================================================
# что видит сервер интерфейса
# =================================================================================================

def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(data if isinstance(data, bytes) else dumps(data))
    os.replace(tmp, path)


class PilotHub:
    """Один пульт на сервер: либо быстрый симулятор (поток здесь же), либо Gazebo (соседний процесс).

    Методы возвращают (код HTTP, тело ответа).
    """

    FRESH_S = 3.0           # пульт Gazebo жив, если его состояние не старше
    ACK_S = 8.0             # сколько ждать ответа пульта Gazebo на команду

    def __init__(self):
        self.fast = None
        self._lock = threading.Lock()
        self._n = 0

    def gazebo_alive(self):
        try:
            return time.time() - STATE.stat().st_mtime < self.FRESH_S
        except OSError:
            return False

    def state(self):
        fast = self.fast
        if fast is not None:
            if fast.is_alive() and fast.snapshot:
                return 200, dumps({**fast.snapshot, 'gazebo_alive': self.gazebo_alive()})
            if fast.error:
                return 200, {'active': False, 'gazebo_alive': self.gazebo_alive(),
                             'error': f'Пульт быстрого симулятора остановился: {fast.error}'}
        if self.gazebo_alive():
            try:
                return 200, STATE.read_bytes()
            except OSError:
                pass
        return 200, {'active': False, 'gazebo_alive': False}

    def start(self, body):
        backend = body.get('backend', 'fastsim')
        with self._lock:
            if backend in ('off', 'gazebo'):
                # Выключить быстрый симулятор; если пульт Gazebo запущен, страница переключится на него.
                if self.fast is not None:
                    self.fast.stop()
                    self.fast = None
                if backend == 'gazebo' and not self.gazebo_alive():
                    return 409, {'error': 'Пульт Gazebo не запущен. Запустите показ командой: pixi run demo'}
                return 200, {'ok': True, 'backend': backend}
            if backend != 'fastsim':
                return 400, {'error': f'Нет такого источника: {backend}. Есть fastsim, gazebo, off'}
            level = body.get('level', 'medium')
            if level not in LEVELS:
                return 400, {'error': f'Нет уровня {level}. Есть: {", ".join(LEVELS)}'}
            try:
                seed = int(body.get('seed', 3))
                speed = float(body.get('speed', 1.0))
            except (TypeError, ValueError):
                return 400, {'error': 'Номер сценария и скорость — числа'}
            if speed not in FastSession.SPEEDS:
                return 400, {'error': 'Скорость времени: 1, 2, 4 или 8'}
            rules = body.get('rules', 'base')
            if rules not in ('base', 'science'):
                return 400, {'error': 'Правила: base или science'}
            if self.fast is not None:
                self.fast.stop()
                self.fast.join(timeout=2.0)
            self.fast = FastSession(level, seed, speed=speed, rules=rules)
            self.fast.start()
            return 200, {'ok': True, 'backend': 'fastsim', 'level': level, 'seed': seed}

    def command(self, body):
        if not isinstance(body, dict) or not isinstance(body.get('cmd'), str):
            return 400, {'error': 'Нужна команда: {"cmd": "route|go|stop|home|reset|mission|finish", ...}'}
        fast = self.fast
        if fast is not None and fast.is_alive():
            res = fast.submit(body)
            return (200 if res['ok'] else 409), ({**res} if res['ok'] else {'error': res['message']})
        if not self.gazebo_alive():
            return 409, {'error': 'Пульт не запущен: запустите показ (pixi run demo) или быстрый симулятор'}
        with self._lock:
            self._n += 1
            cid = f'{time.time_ns()}-{os.getpid()}-{self._n}'
        _write_json(CMD_DIR / f'{cid}.json', {**body, 'id': cid})
        ack = ACK_DIR / f'{cid}.json'
        deadline = time.monotonic() + self.ACK_S
        while time.monotonic() < deadline:
            if ack.exists():
                try:
                    res = json.loads(ack.read_text(encoding='utf-8'))
                    ack.unlink()
                except (OSError, ValueError):
                    time.sleep(0.02)
                    continue
                return (200 if res.get('ok') else 409), (res if res.get('ok') else {'error': res.get('message')})
            time.sleep(0.03)
        try:
            (CMD_DIR / f'{cid}.json').unlink()              # не исполнять команду, на которую уже ответили отказом
        except OSError:
            return 504, {'error': 'Пульт Gazebo принял команду, но не ответил вовремя: смотрите состояние на странице'}
        return 504, {'error': f'Пульт Gazebo не ответил за {self.ACK_S:.0f} с: команда отменена'}


HUB = PilotHub()


# =================================================================================================
# Gazebo: отдельный процесс поверх ROS 2
# =================================================================================================

SLAM_FRAME = 'slam_map'        # кадр карты SLAM Toolbox (env/slam_demo.yaml): /map и кадр map заняты судьёй
WORLD_FRAME = 'map'            # мировой кадр по одометрии: его SLAM Toolbox и поправляет (odom_frame в том же файле)


def run_ros(level, seed, rules='base', wait_s=600.0, slam_nav=False):
    """Пульт при запущенном стенде (pixi run stand-gui / pixi run demo). Работает до Ctrl+C."""
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage

    from .ros_agent import SETTLE_UNTIL_S, RosIO

    class PilotIO(RosIO):
        def __init__(self):
            super().__init__()
            self.reset_cli = self.create_client(Trigger, '/did/reset')
            self.clock_wall = 0.0
            # Последняя карта SLAM Toolbox: (номер, клетка, x0, y0, сетка, время симуляции при получении).
            self.slam = None
            # Его поправка позы: (dx, dy, поворот, время симуляции при получении) «мир по одометрии → карта SLAM».
            # Время — получения, а не из заголовка: SLAM ставит в заголовок время скана с запасом вперёд.
            self.slam_tf = None
            self.slam_alien = 0                 # сообщений карты не в том кадре (отброшены)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(OccupancyGrid, '/slam/map', self._on_slam, latched)
            self.create_subscription(TFMessage, '/tf', self._on_tf, 100)

        def _on_tf(self, msg):
            for tr in msg.transforms:
                if tr.header.frame_id.lstrip('/') == SLAM_FRAME and tr.child_frame_id.lstrip('/') == WORLD_FRAME:
                    t, q = tr.transform.translation, tr.transform.rotation
                    v = (t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
                    if all(math.isfinite(c) for c in v):
                        self.slam_tf = (*v, self.sim_time if self.sim_time is not None else -math.inf)

        def _on_slam(self, msg):
            i = msg.info
            if msg.header.frame_id.lstrip('/') != SLAM_FRAME or i.width * i.height != len(msg.data) or not msg.data:
                self.slam_alien += 1            # чужой кадр или битая сетка: по такой карте ехать нельзя
                return
            grid = np.asarray(msg.data, dtype=np.int8).reshape(i.height, i.width)
            self.slam = ((self.slam[0] + 1) if self.slam else 1, i.resolution, i.origin.position.x,
                         i.origin.position.y, grid, self.sim_time if self.sim_time is not None else -math.inf)

        def _on_clock(self, msg):
            super()._on_clock(msg)
            self.clock_wall = time.monotonic()

    class RosWorld:
        """Стенд Gazebo как мир пульта."""
        backend = 'gazebo'
        can_respawn = False
        score_delay = 1.5       # с: итоговый /did/score приходит чуть позже финиша

        def __init__(self, io):
            self.io = io
            self.level, self.seed, self.rules_name = level, seed, rules
            self.rules = Rules(**SCIENCE) if rules == 'science' else Rules()
            self.scenario = generate(level, seed, load_arena())    # только для подписи записи, если нет /did/truth
            self._slam = (0, None)
            self._grid = (0, None)              # сетка SLAM на решётке арены: (номер, значения)
            self._still = None                  # (поза, с какого времени симуляции она не меняется)
            self._settled = False
            self._t_spawn = None                # время симуляции, когда судья увидел робота
            self._reset = RunReset(self._reset_send, lambda: (io.score, io.battery), self.rules.battery_start)

        observe = property(lambda self: self.io.observe)
        command = property(lambda self: self.io.command)
        collect = property(lambda self: self.io.collect)
        finish = property(lambda self: self.io.finish)

        def linked(self):
            return time.monotonic() - self.io.clock_wall < 2.0

        def settle_left(self):
            """Сколько секунд ещё нельзя ехать: робот оседает на колёса после появления в мире."""
            if self._settled:
                return 0.0
            io = self.io
            now, pose = io.sim_time, io.odom[:3]
            if self._t_spawn is None:
                self._t_spawn = -io.clock_offset
            if self._still is None or max(abs(a - b) for a, b in zip(pose, self._still[0])) >= 1e-4:
                self._still = (pose, now)
            # Стенд работает давно (пульт перезапустили) — достаточно, что робот стоит.
            since_spawn = now - self._t_spawn if now < 60.0 else 1e9
            left = max(SETTLE_UNTIL_S - since_spawn, 4.0 - (now - self._still[1]), 0.0)
            self._settled = left <= 0.0
            return left

        def _reset_send(self):
            """Отправить вызов /did/reset, не дожидаясь ответа. None — сервис судьи пока не виден."""
            cli = self.io.reset_cli
            if not cli.service_is_ready():
                return None
            future = cli.call_async(Trigger.Request())

            def answer():
                if not future.done():
                    return None
                try:
                    r = future.result()
                except Exception as e:          # noqa: BLE001 — причина уходит оператору
                    return False, f'вызов /did/reset не удался ({type(e).__name__}: {e})'
                if r is None:
                    return False, 'вызов /did/reset не удался'
                return bool(r.success), r.message

            return answer

        def reset_begin(self, wait_s=RESET_WAIT_S):
            """Начать новый прогон судьи; подтверждение — через reset_poll(), основной цикл пульта не ждёт."""
            self._reset.start(wait_s)

        def reset_poll(self):
            res = self._reset.poll()
            if res is not None and res[0]:
                self.io.observe()               # события и показания прежнего прогона новому не нужны
            return res

        def new_run(self):
            """То же, но с коротким ожиданием на месте (кнопка «Сбросить прогон» и новый прогон в ручной езде)."""
            self.reset_begin(RESET_SYNC_S)
            while True:
                res = self.reset_poll()
                if res is not None:
                    return res
                time.sleep(0.02)

        def score(self):
            return self.io.score or {}

        def slam_grid(self):
            """Сетка SLAM Toolbox на решётке арены: (номер, значения) — −1 не видел, 0 свободно, 100 занято.

            Кадр карты SLAM совпадает с мировым (env/slam_demo.yaml), поэтому начало сетки берётся из
            сообщения как есть. Готовая карта здесь не участвует.
            """
            m = self.io.slam
            if m is not None and m[0] != self._grid[0]:
                ver, res, ox, oy, grid, _ = m
                self._grid = (ver, resample(grid, res, ox, oy))
            return self._grid

        def slam_tf(self):
            """Поправка позы от SLAM Toolbox и возраст его данных по часам симуляции (см. did/slam_map.py::SlamPose).

            Возраст меньше нуля бывает, если часы симуляции пошли заново: такое сообщение — из прошлого запуска.
            """
            tf, m, now = self.io.slam_tf, self.io.slam, self.io.sim_time
            if tf is None:
                return None
            return {'tf': tf[:3], 'tf_age': now - tf[3], 'grid_age': None if m is None else now - m[5]}

        def slam_view(self):
            """Карта SLAM Toolbox для страницы и её сверка с готовой картой. None, если SLAM не запущен."""
            ver, v = self.slam_grid()
            if v is None or ver == self._slam[0]:
                return self._slam[1]
            seen, occ = v >= 0, v > 50
            c = compare(v, arena)               # сверка с готовой картой — только числа на экране
            view = {'res': arena.res, 'x0': arena.x0, 'y0': arena.y0, 'w': arena.w, 'h': arena.h, 'enc': 'occ',
                    'version': ver, 'data': encode_grid(np.where(seen, np.where(occ, 255, 1), 0)),
                    'coverage': round(c['coverage'], 4), 'agreement': round(c['agreement'], 4)}
            self._slam = (ver, view)
            return view

        def world_log(self):
            return (self.io.truth or {}).get('world_log', [])

        def scenario_dict(self):
            return (self.io.truth or {}).get('scenario') or self.scenario.to_dict()

        def truth(self):
            tr = self.io.truth
            if not tr:
                return None
            samples = {i: p for i, p in enumerate(tr.get('scenario', {}).get('samples', []))}
            return {'soils': tr.get('soils', []), 'hazards': tr.get('hazards', []),
                    'remaining': [[r['x'], r['y']] for r in tr.get('remaining', [])],
                    'collected': [list(samples[c['id']]) for c in tr.get('collected', []) if c['id'] in samples]}

    arena = load_arena()
    for d in (CMD_DIR, ACK_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob('*.json'):              # команды прошлого запуска исполнять нельзя
            f.unlink(missing_ok=True)

    rclpy.init()
    io = PilotIO()
    executor = SingleThreadedExecutor()
    executor.add_node(io)
    threading.Thread(target=executor.spin, daemon=True).start()
    halt = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.set())

    base = {'active': True, 'backend': 'gazebo', 'level': level, 'seed': seed, 'rules': rules, 'mode': 'wait',
            'linked': False, 'base': list(BASE)}
    world = pilot = None
    last_tick = last_state = last_cmd = last_warn = last_watch = -1e9
    tick_wall, spent = time.monotonic(), [0.0, 0.0]     # когда был прошлый такт; сколько заняли такт и запись состояния
    t0 = time.monotonic()
    print(f'пульт: жду стенд (уровень {level}, сценарий {seed})', flush=True)
    try:
        while not halt.is_set() and rclpy.ok():
            wall = time.monotonic()
            if pilot is None:
                if io.ready() and not (io.score or {}).get('finished'):
                    world = RosWorld(io)
                    pilot = Pilot(arena, world, level, seed, slam_nav=slam_nav)
                    print('пульт: стенд на связи, робот оседает' + (
                        '; режим карты SLAM — готовая карта роботу не даётся' if slam_nav else ''), flush=True)
                    continue
                if io.ready() and wall - last_state >= 2.0 and io.reset_cli.service_is_ready():
                    io._call(io.reset_cli)      # стенд остался от прошлого показа, прогон судьи уже закончен
                if wall - last_state >= STATE_S:
                    last_state = wall
                    _write_json(STATE, {**base, 'wall': time.time(), 'note': {
                        'tone': 'info', 'text': 'Жду данные от стенда Gazebo: мир загружается'}})
                if wall - t0 > wait_s:
                    raise RuntimeError('нет данных от стенда: проверьте, что он запущен (pixi run stand-gui)')
                time.sleep(0.05)
                continue

            for f in (sorted(CMD_DIR.glob('*.json')) if wall - last_cmd >= 0.05 else ()):
                try:
                    cmd = json.loads(f.read_text(encoding='utf-8'))
                    f.unlink()
                except (OSError, ValueError):
                    continue                    # файл ещё пишется или уже отменён сервером
                res = pilot.command(cmd)
                _write_json(ACK_DIR / f"{cmd.get('id', f.stem)}.json", {**res, 'id': cmd.get('id')})
                last_state = -1e9
            if wall - last_cmd >= 0.05:
                last_cmd = wall

            now = io.sim_time
            if now < last_tick:                 # часы симуляции пошли заново: стенд перезапустили
                last_tick = now - TICK_S
            if now - last_tick >= TICK_S:
                gap = min(now - last_tick, 60.0) if last_tick > -1e8 else TICK_S
                if gap > LAG_S and wall - last_warn >= 1.0 and pilot.mode != 'settle':
                    last_warn = wall
                    print(f'пульт: управление запоздало на {gap:.2f} с времени симуляции (по часам машины с прошлого '
                          f'такта {wall - tick_wall:.2f} с; такт {spent[0] * 1e3:.0f} мс, состояние {spent[1] * 1e3:.0f} мс)',
                          flush=True)
                last_tick, tick_wall = now, wall
                was = pilot.mode
                pilot.tick(io.observe(), gap)
                spent[0] = time.monotonic() - wall
                if was == 'settle' and pilot.mode != 'settle':
                    print('пульт: робот готов, можно ставить точки', flush=True)
            else:
                time.sleep(0.002)
            if wall - last_watch >= TICK_S:
                # Миссия, которая ждёт нового прогона судьи, не должна зависеть от /clock: если Gazebo встал,
                # тактов нет, а срок ожидания по часам машины всё равно истекает отказом.
                last_watch = wall
                pilot.watch_reset()
            if wall - last_state >= STATE_S:
                last_state = wall
                t1 = time.monotonic()
                _write_json(STATE, pilot.state())
                spent[1] = time.monotonic() - t1
    finally:
        try:
            for _ in range(3):
                io.command(0.0, 0.0)
                time.sleep(0.03)
        except Exception:                       # noqa: BLE001 — узел мог уже закрыться
            pass
        try:
            STATE.unlink(missing_ok=True)
        except OSError:
            pass
        executor.shutdown()
        io.destroy_node()
        rclpy.try_shutdown()
        print('пульт: остановлен', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ros', action='store_true', help='пульт для стенда Gazebo (стенд должен быть запущен)')
    ap.add_argument('--level', default='medium', choices=list(LEVELS), help='уровень, с которым запущен стенд')
    ap.add_argument('--seed', type=int, default=3, help='номер сценария, с которым запущен стенд')
    ap.add_argument('--rules', default='base', choices=['base', 'science'])
    ap.add_argument('--slam-map', action='store_true',
                    help='ехать по карте SLAM Toolbox, без готовой карты (SLAM должен быть запущен)')
    args = ap.parse_args()
    if not args.ros:
        ap.error('быстрый симулятор запускается со страницы «Пульт» или командой pixi run demo --fast; '
                 'для Gazebo добавьте --ros')
    run_ros(args.level, args.seed, args.rules, slam_nav=args.slam_map)


if __name__ == '__main__':
    main()
