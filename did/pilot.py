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

Команды оператора — словари {'cmd': ...}:
  route {points: [[x, y], ...]}  задать маршрут          go      ехать по маршруту
  stop                           остановиться            home    вернуться на базу
  reset                          стереть карту и след, начать прогон судьи заново
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
from dataclasses import asdict

import numpy as np

from .agent import PRESETS, Agent, make_config
from .arena import load_arena
from .config import BASE, LEVELS, SCIENCE, Rules
from .fastsim import FastSim
from .judge import Judge
from .mapping import LIDAR_OFFSET, OccupancyMapper
from .metrics import run_metrics
from .nav import INFLATE, CostGraph, Follower, path_length
from .recorder import Recorder, encode_grid, save_trace
from .runner import RUNS, make_planner
from .scenario import generate

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
MAX_POINTS = 20
BASE_NEAR = 0.25               # м: ближе — робот «на базе»
# Агенты, которым не нужна языковая модель: их можно запускать с пульта.
MISSION_AGENTS = {'adaptive': 'С адаптацией', 'scientist': 'Исследователь', 'fixed': 'Фиксированный план'}
# Уход одометрии в быстром симуляторе — как в Gazebo, чтобы поправка по лидару была видна и на репетиции.
FAST_DRIFT = dict(odom_turn_slip=1.0, odom_turn_scale=0.03, odom_path_scale=0.03)
EVENT_TEXT = {'collision': 'Столкновение', 'hazard_hit': 'Заезд в опасную зону', 'false_collect': 'Ложный сбор',
              'sample_collected': 'Образец собран'}
REASONS = {'finish': 'миссия завершена', 'battery': 'села батарея', 'timeout': 'вышло время прогона'}


class Refuse(Exception):
    """Команду оператора выполнить нельзя; текст показывается на странице."""


# =================================================================================================
# мир: быстрый симулятор
# =================================================================================================

class FastWorld:
    """Быстрый симулятор как мир пульта: RobotIO и то, что в Gazebo делает стенд."""

    backend = 'fastsim'
    dt = TICK_S

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

    def new_run(self):
        """Новый прогон судьи с того места, где стоит робот: время, заряд и образцы — с начала."""
        s = self.sim
        s.judge = Judge(self.scenario, self.arena, self.rules, seed=self.seed)
        s.t = 0.0
        s._next_sensor = s._next_lidar = 0.0
        s.judge.step(0.0, s.x, s.y)
        return True, 'судья начал прогон заново'

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

    def __init__(self, arena, world, level, seed):
        self.arena, self.world, self.level, self.seed = arena, world, level, int(seed)
        self.rules = world.rules
        self.graph = CostGraph(arena)
        self.follower = Follower()
        self.mapper = OccupancyMapper(arena)
        self.tracker = PoseTracker(arena) if PoseTracker else None
        self.mode = 'idle'                  # settle | idle | drive | home | mission
        self.route = []                     # [{'x', 'y', 'done', 'pts'}]: точки оператора и путь до каждой
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
        self._belief_cache = (-1e9, None)
        self._finish_at = None
        self._last_odom = None
        self._rerun_at = 0.0
        self._say('info', 'Пульт запущен')

    # ======================================================================================
    # один такт
    # ======================================================================================

    def tick(self, obs):
        w = self.world
        self.t = obs.t
        self.battery = obs.battery
        if obs.sensor is not None:
            self.sensor = obs.sensor
        for ev in obs.events:
            self._on_event(ev)

        if self.mode == 'mission' and self.bot:
            self.bot.tick(obs, w)                   # агент сам поправляет позу (трекер у нас общий)
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
                self._fresh_run()
            return self._drive(obs, x, y, th)
        self._command(0.0, 0.0)

    def _fresh_run(self):
        """Ручная езда судье не подотчётна: если его прогон закончился (заряд, время), начинаем новый."""
        if not (self.world.score() or {}).get('finished') or time.monotonic() < self._rerun_at:
            return
        self._rerun_at = time.monotonic() + 10.0            # судья без сервиса сброса: не спрашивать каждый такт
        ok, _ = self.world.new_run()
        self._escape = self._stuck = None
        if ok:
            self._log(0.0, 'info', 'Прежний прогон судьи закончился — начат новый: заряд снова полный')

    def _to_map(self, x, y, th):
        return self.tracker.to_map(x, y, th) if self.tracker else (x, y, th)

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
            self._escape = {'until': self.t + 1.5, 'v': 0.10 if self._last_cmd[0] < 0 else -0.10}

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
        if self._escape:
            if obs.t < self._escape['until']:
                return self._command(self._escape['v'], 0.0)
            self._escape = self._stuck = None
            self.follower.set_path([])
        while True:
            todo = self._pending()
            if not todo:
                self._command(0.0, 0.0)
                self.mode = 'idle'
                return
            leg = todo[0]
            target = (leg['x'], leg['y'])
            if not self.follower.active:
                pts, _ = self.graph.plan((x, y), target)
                if pts is None:
                    self._command(0.0, 0.0)
                    self.mode = 'idle'
                    return self._say('bad', 'Пути к точке нет: маршрут остановлен')
                self.follower.set_path(pts + [target])
                leg['pts'] = pts + [target]
            last = len(todo) == 1
            v, w, arrived = self.follower.step(x, y, th, tol=0.05 if last else 0.12)
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
            self._log(obs.t, 'info', f'Точка {self.route.index(leg) + 1} пройдена')     # и сразу к следующей
        if v > 0.0 and self._front < 0.15:
            v = 0.0                                     # лидар видит преграду вплотную по курсу
        self._watch_stuck(obs, x, y, v)
        self._command(v, w)

    def _arrived(self, obs, x, y, target):
        err = math.dist((x, y), target)
        self.arrivals.append(round(err, 3))
        home = self.mode == 'home'
        self.mode = 'idle'
        self.route = [] if home else self.route
        self._say('ok', f'Робот на базе, до её центра {err * 100:.0f} см' if home
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
                self._escape = {'until': obs.t + 1.5, 'v': -0.10}
            self._stuck = None

    def _plan_route(self, points):
        """Проверить точки и построить путь через них от текущей позы. Возвращает маршрут и заметки."""
        route, notes = [], []
        prev = self.pose[:2]
        for n, p in enumerate(points, 1):
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                raise Refuse(f'Точка {n}: нужны два числа — x и y в метрах') from None
            ix, iy = self.arena.w2g(x, y)
            if not (math.isfinite(x) and math.isfinite(y)) or not self.arena.inside(ix, iy):
                raise Refuse(f'Точка {n} за пределами арены')
            if not self.arena.free[iy, ix]:
                raise Refuse(f'Точка {n} попала в стену или столб: поставьте её на свободный пол')
            if self.arena.clear[iy, ix] < INFLATE:
                # Центр робота не подходит к стене ближе 17 см: берём ближайшее разрешённое место.
                node = self.graph.node(x, y)
                nx, ny = float(self.graph.xs[node]), float(self.graph.ys[node])
                notes.append(f'точка {n} сдвинута от стены на {math.dist((x, y), (nx, ny)) * 100:.0f} см')
                x, y = nx, ny
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
        self.route = route
        self.follower.set_path([])
        if not route:
            self.mode = 'idle' if self.mode == 'drive' else self.mode
            self._command(0.0, 0.0)
            self.note = {'tone': 'info', 'text': 'Маршрут очищен'}
            return 'Маршрут очищен'
        length = sum(path_length(p['pts']) for p in route)
        text = f'Маршрут из {len(route)} {_plural(len(route), "точки", "точек", "точек")}: путь {length:.1f} м'
        if notes:
            text += ' (' + '; '.join(notes) + ')'
        self.note = {'tone': 'info', 'text': text + ('' if self.mode == 'drive' else '. Нажмите «Ехать»')}
        return text

    def _cmd_go(self, cmd):
        self._manual_only()
        if not self._pending():
            raise Refuse('Сначала поставьте точку на карте')
        self._fresh_run()
        self.mode = 'drive'
        self._drop_pending_mission()
        self.follower.set_path([])
        self._say('info', f'Еду по маршруту: точек {len(self._pending())}')
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
            self._say('info', 'Стоп: робот остановлен, маршрут сохранён')
        return 'Робот стоит'

    def _cmd_home(self, cmd):
        self._manual_only()
        self._drop_pending_mission()
        self._go_home()
        return 'Еду на базу'

    def _go_home(self, after=None):
        self._fresh_run()
        route, _ = self._plan_route([BASE])
        self.route, self.mode, self._after_home = route, 'home', after
        self.follower.set_path([])
        self._say('info', 'Возвращаюсь на базу')

    def _cmd_reset(self, cmd):
        w = self.world
        self._command(0.0, 0.0)
        self.mode = 'idle'
        self.route, self.trail, self.arrivals = [], [], []
        self.mission = self.bot = self.rec = self._after_home = self._escape = self._stuck = None
        self.follower.set_path([])
        self.mapper.reset()
        self.distance = 0.0
        if getattr(w, 'can_respawn', False):
            w.respawn()
            self.tracker = PoseTracker(self.arena) if PoseTracker else None
            self._last_odom = None
            text = 'Сброс: робот на базе, карта и след стёрты, прогон начат заново'
        else:
            ok, msg = w.new_run()
            text = 'Сброс: карта и след стёрты' + (', судья начал прогон заново' if ok else f' (судья: {msg})')
            if math.dist(self.pose[:2], BASE) > BASE_NEAR:
                text += '. Робот не на базе — нажмите «Домой»'
        self._say('info', text)
        return text

    def _cmd_speed(self, cmd):
        raise Refuse('Скорость времени меняется только в быстром симуляторе')

    # ======================================================================================
    # автономная миссия
    # ======================================================================================

    def _cmd_mission(self, cmd):
        agent = cmd.get('agent', 'adaptive')
        if agent not in MISSION_AGENTS or agent not in PRESETS:
            raise Refuse(f'Нет агента «{agent}». Есть: {", ".join(MISSION_AGENTS)}')
        if self.mode == 'mission':
            raise Refuse('Миссия уже идёт')
        self._manual_only()
        self.mission = {'agent': agent, 'label': MISSION_AGENTS[agent], 'state': 'to_base',
                        'total': LEVELS[self.level]['samples'], 'collected': 0}
        if math.dist(self.pose[:2], BASE) > BASE_NEAR:
            self._go_home(after=lambda: self._start_mission(agent))
            self.note = {'tone': 'info', 'text': 'Миссия начнётся с базы: сначала возвращаюсь'}
            return 'Сначала на базу, затем миссия'
        self._start_mission(agent)
        return 'Миссия запущена'

    def _start_mission(self, agent):
        w = self.world
        self._command(0.0, 0.0)
        ok, msg = w.new_run()                       # чистый прогон: полный заряд, время с нуля, все образцы на месте
        cfg = make_config(agent)
        self.rec = Recorder()
        self.bot = Agent(self.arena, cfg, n_samples=LEVELS[self.level]['samples'], rules=self.rules,
                         planner=make_planner(cfg, None, self.seed), recorder=self.rec)
        if self.tracker and cfg.localize:
            self.bot.tracker = self.tracker         # поправка позы не начинается с нуля
        self.route = []
        self.follower.set_path([])
        self.mode = 'mission'
        self._finish_at = None
        self.mission = {'agent': agent, 'label': MISSION_AGENTS[agent], 'state': 'running',
                        'total': LEVELS[self.level]['samples'], 'collected': 0, 'fresh_run': bool(ok)}
        self._say('info', f'Автономная миссия: агент «{MISSION_AGENTS[agent]}» ищет образцы'
                  + ('' if ok else f'. Судья прогон заново не начал ({msg}): заряд и время — какие остались'))

    def _mission_tick(self, obs):
        bot = self.bot
        if self._finish_at is None and (obs.done or bot.finished):
            self._finish_at = time.monotonic() + 1.5        # дождаться итогового счёта судьи
        if self._finish_at is not None and time.monotonic() >= self._finish_at:
            self._end_mission('finished')

    def _cmd_finish(self, cmd):
        if self.mode != 'mission':
            raise Refuse('Миссия не запущена')
        self._command(0.0, 0.0)
        ok, msg = self.world.finish()
        self._end_mission('finished', f'Миссия завершена оператором: {msg}')
        return f'Миссия завершена: {msg}'

    def _end_mission(self, state, text=None):
        w, bot = self.world, self.bot
        self._command(0.0, 0.0)
        self.mode = 'idle'
        self._finish_at = None
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
                    f"счёт {r.get('score', 0):.1f}")
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
            'fix': {'source': 'lidar' if self.tracker else 'odom', 'dx': round(fix[0], 4), 'dy': round(fix[1], 4),
                    'dth': round(fix[2], 4), 'shift': round(math.hypot(fix[0], fix[1]), 4),
                    'scans': stats.get('scans', 0), 'fixes': stats.get('fixes', 0),
                    'inliers': round(float(stats.get('inliers') or 0.0), 3)},
            'base': list(BASE), 'at_base': math.dist((x, y), BASE) <= BASE_NEAR,
            'trail': [[round(px, 2), round(py, 2)] for px, py in self.trail[::k]] + [[round(x, 2), round(y, 2)]],
            'route': [{'x': p['x'], 'y': p['y'], 'done': p['done']} for p in self.route],
            'path': [[round(px, 2), round(py, 2)] for px, py in _thin(path, 2)],
            'path_len': round(path_length(path), 2) if len(path) > 1 else 0.0,
            'battery': round(float(self.battery), 2), 'battery_start': self.rules.battery_start,
            'sensor': round(float(self.sensor), 3), 'distance': round(self.distance, 2),
            'time_limit': self.rules.time_limit_s, 'arrivals': self.arrivals[-10:],
            'score': score, 'events': self.log[-30:],
            'map': self._map_cache[1], 'scan': scan,
            'mission': self._mission_view() if self.mission else None,
            'agents': [{'id': k, 'label': v} for k, v in MISSION_AGENTS.items() if k in PRESETS],
            'truth': w.truth(),
        }


def _thin(pts, k):
    return pts[::k] + ([pts[-1]] if pts and (len(pts) - 1) % k else [])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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
