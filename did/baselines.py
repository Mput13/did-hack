"""Базовые стратегии поиска из условия задачи: подъём по сигналу и расширяющаяся спираль.

Карты вероятностей у них нет: решение принимается по самому показанию датчика. Нужны для сравнения
(опыт E9): сколько даёт карта вероятностей сверх простых правил. Интерфейс тот же, что у
did.agent.Agent: tick(obs, io), поля finished и journal, запись прогона через recorder.

  gradient — подъём по сигналу: шаг, остановка, среднее показание. Растёт — так и ехать, упало —
             поворот; по трём последним остановкам оценивается, в какую сторону сигнал растёт.
             Сигнала нет — переезд туда, где датчик ещё не слушал.
  spiral   — дуги растущего радиуса вокруг базы, туда и обратно; при сильном сигнале — тот же
             подъём до образца и возврат на маршрут.

Оба возвращаются на базу по жёсткому порогу заряда, как фиксированный агент, и ездят только по
разрешённым клеткам (CostGraph, Follower). Грунт не изучают; место штрафа запоминают и объезжают.
"""
import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .config import BASE, Rules
from .journal import Journal
from .nav import CostGraph, Follower


@dataclass(frozen=True)
class BaselineConfig:
    name: str = 'gradient'
    search: str = 'gradient'          # gradient | spiral
    planner: str = 'rule'             # языковой модели у базовых стратегий нет
    static_reserve: float = 18.0      # порог заряда для возврата — тот же, что у фиксированного агента
    leg: float = 0.35                 # м: шаг подъёма по сигналу
    pause_n: int = 5                  # показаний на остановке: по их среднему принимается решение (1 с)
    collect_dist: float = 0.25        # м: пробовать сбор, когда среднее показание говорит «образец ближе»
    floor: float = 0.08               # показание, ниже которого сигнала нет: образцы дальше ~1,8 м
    listen: float = 1.2               # м: без сигнала ехать слушать не ближе этого к уже пройденному пути
    strong: float = 0.55              # спираль: с такого сигнала (образец ближе ~0,9 м) заезжать за образцом
    pitch: float = 1.6                # м: спираль — расстояние между дугами; три дуги накрывают всю арену

    def to_dict(self):
        return asdict(self)


def _slope(verts):
    """Куда растёт сигнал: наклон плоскости через три остановки. None — они почти на одной прямой."""
    p = np.array(verts, dtype=float)
    d = p[:, :2] - p[:, :2].mean(axis=0)
    s = np.linalg.svd(d, compute_uv=False)
    if s[-1] < 0.04 or s[-1] < 0.2 * s[0]:
        return None
    g, *_ = np.linalg.lstsq(d, p[:, 2] - p[:, 2].mean(), rcond=None)
    return float(g[0]), float(g[1])


def spiral_route(arena, graph, pitch, step=0.4):
    """Дуги вокруг базы с шагом радиуса pitch, направление обхода чередуется: [(x, y), ...].

    База стоит у края арены, поэтому от спирали остаются дуги; ехать их все в одну сторону значило бы
    каждый раз возвращаться через всю арену.
    """
    out, r, flip = [], pitch / 2, False
    while True:
        n = max(8, int(2 * math.pi * r / step))
        arc = []
        for k in range(n):
            a = -math.pi + 2 * math.pi * k / n
            x, y = BASE[0] + r * math.cos(a), BASE[1] + r * math.sin(a)
            ix, iy = arena.w2g(x, y)
            if arena.inside(ix, iy) and graph.ok[iy, ix] and arena.clear[iy, ix] >= 0.25:
                arc.append((round(x, 2), round(y, 2)))
        if not arc:
            return out
        out += arc[::-1] if flip else arc
        r, flip = r + pitch, not flip


class Baseline:
    """Общее для базовых стратегий: езда, возврат, сбор, подъём по сигналу, запись прогона."""

    def __init__(self, arena, config, n_samples, rules=None, planner=None, recorder=None):
        self.arena = arena
        self.cfg = config
        self.rules = rules or Rules()
        self.n_samples = n_samples
        self.rec = recorder
        self.journal = Journal()
        self.base = BASE
        self.graph = CostGraph(arena)
        self.follower = Follower()
        self.mode = 'start'
        self.collected = 0
        self.finished = False
        self._returning = False
        self._readings = deque(maxlen=5)
        self._front = math.inf
        self._slow_t = -1e9
        self._path_goal = None
        self._escape = None
        self._stuck = None
        self._last_v = 0.0
        self._trail = []                                        # где робот уже был, через 0,3 м
        self._hazard = np.zeros(arena.free.shape, dtype=bool)   # где был штраф: туда больше не ехать
        self._X, self._Y = arena.cell_centers()
        # подъём по сигналу
        self._heading = 0.0
        self._verts = []                                        # остановки: (x, y, среднее показание)
        self._leg = None
        self._pause = None
        self._best = (-1.0, 0)

    # --- один такт -------------------------------------------------------------------------------

    def tick(self, obs, io):
        if obs.done or self.finished:
            io.command(0.0, 0.0)
            self._record(obs)
            return
        self._perceive(obs)
        if obs.t - self._slow_t >= 1.0:
            self._slow_t = obs.t
            self._check_return(obs)
        if self._escape and obs.t < self._escape['until']:
            self.mode = 'escape'
            self._command(io, self._escape['v'], 0.0)
        else:
            if self._escape:
                self._escape, self._path_goal, self._stuck = None, None, None
            if self._returning:
                self._do_return(obs, io)
            else:
                self._search(obs, io)
        self._record(obs)

    def _search(self, obs, io):
        raise NotImplementedError

    def _perceive(self, obs):
        if obs.scan is not None:
            self._front = float(min(obs.scan[:20].min(), obs.scan[-20:].min()))
        for ev in obs.events:
            kind = ev.get('type')
            if kind == 'collision':
                self.journal.add(obs.t, 'alarm', f'Штраф: столкновение в ({obs.x:.2f}; {obs.y:.2f})', tag='collision')
                self._escape = {'until': obs.t + 1.5, 'v': 0.10 if self._last_v < 0 else -0.10}
            elif kind == 'hazard_hit':
                self.journal.add(obs.t, 'alarm', f'Штраф: опасная зона в ({obs.x:.2f}; {obs.y:.2f}); '
                                 'это место дальше объезжаю', tag='hazard')
                cx, cy = obs.x + 0.3 * math.cos(obs.th), obs.y + 0.3 * math.sin(obs.th)
                self._hazard |= np.hypot(self._X - cx, self._Y - cy) <= 0.45
                self.graph.set_cost(forbidden=self._hazard)
                self._path_goal = None
                if self.rec:
                    self.rec.add_hazard(obs.t, cx, cy, 0.45)
            elif kind == 'sample_collected' and ev.get('collected', 0) > self.collected:
                self.collected = ev['collected']             # ответ сервиса сбора потерялся — сверка по событию
        if obs.sensor is not None:
            self._readings.append(obs.sensor)
            if self._pause is not None and abs(obs.v) < 0.03:
                self._pause['z'].append(obs.sensor)
        if not self._trail or math.dist(self._trail[-1], (obs.x, obs.y)) >= 0.3:
            self._trail.append((obs.x, obs.y))

    def _signal(self):
        """Среднее последних пяти показаний (1 с) или None, пока их меньше."""
        return float(np.mean(self._readings)) if len(self._readings) == self._readings.maxlen else None

    # --- возврат на базу: то же правило жёсткого порога, что у фиксированного агента --------------

    def _home_cost(self, x, y):
        _, cost = self.graph.plan((x, y), self.base)
        return cost * self.rules.drain_per_m

    def _check_return(self, obs):
        if self._returning:
            return
        need = self.cfg.static_reserve
        reason = None
        if self.collected >= self.n_samples:
            reason = 'все образцы собраны'
        elif obs.battery <= need:
            reason = f'заряд {obs.battery:.1f} ед. опустился до порога {need:.0f}'
        elif self.rules.time_limit_s - obs.t <= self._home_cost(obs.x, obs.y) / self.rules.drain_per_m / 0.15 + 10.0:
            reason = 'время прогона на исходе'
        if reason:
            self._go_home(obs.t, reason)

    def _go_home(self, t, reason):
        self._returning = True
        self._path_goal = None
        self._plan(t, 'return', f'Возвращаюсь на базу: {reason}.', [{'type': 'return_base'}])

    def _do_return(self, obs, io):
        self.mode = 'return'
        if self._drive_to(obs, io, self.base, tol=0.08) and abs(obs.v) < 0.03:
            _, msg = io.finish()
            self.finished = True
            self.mode = 'done'
            self.journal.add(obs.t, 'action', f'Финиш: {msg}. Собрано {self.collected} из {self.n_samples}, '
                             f'заряд {obs.battery:.1f}')

    # --- подъём по сигналу -----------------------------------------------------------------------

    def _climb_start(self, heading=None):
        """Начать подъём с остановки: сначала измерить сигнал там, где робот стоит."""
        self._verts, self._leg, self._best = [], None, (-1.0, 0)
        self._pause = {'z': [], 't0': None}
        if heading is not None:
            self._heading = heading

    def _climb(self, obs, io, floor):
        """Один такт подъёма. Возвращает None, пока подъём продолжается, иначе чем он кончился:
        'collected' — образец взят, 'lost' — сигнал пропал, 'stuck' — сигнал есть, но не растёт."""
        self.mode = 'approach'
        if self._leg is not None:
            if not self._drive_to(obs, io, self._leg['goal'], tol=0.04) and obs.t - self._leg['t0'] < 12.0:
                return None
            self._leg = None
            self._pause = {'z': [], 't0': None}
        self._command(io, 0.0, 0.0)
        pause = self._pause
        if pause['t0'] is None:
            pause['t0'] = obs.t
        if len(pause['z']) < self.cfg.pause_n and obs.t - pause['t0'] < 4.0:
            return None
        z = float(np.mean(pause['z'])) if pause['z'] else 0.0
        return self._vertex(obs, io, z, floor)

    def _vertex(self, obs, io, z, floor):
        """Остановка: по среднему показанию z решить — собирать, бросить подъём или куда шагнуть."""
        cfg, reach = self.cfg, self.rules.sensor_range_m
        self._pause = {'z': [], 't0': None}
        if z >= 1.0 - cfg.collect_dist / reach:
            self.mode = 'collect'
            ok, _ = io.collect()
            where = f'({obs.x:.2f}; {obs.y:.2f})'
            if ok:
                self.collected += 1
                self.journal.add(obs.t, 'action', f'Сбор в {where} при показании {z:.2f}: образец взят, '
                                 f'всего {self.collected} из {self.n_samples}')
                if self.collected >= self.n_samples:
                    self._go_home(obs.t, 'все образцы собраны')
                return 'collected'
            self.journal.add(obs.t, 'action', f'Сбор в {where} при показании {z:.2f}: промах, получен штраф')
        if z < floor:
            return 'lost'
        self._verts.append((obs.x, obs.y, z))
        n = len(self._verts)
        if z > self._best[0] + 0.02:
            self._best = (z, n)
        if n - self._best[1] >= 8 or n > 60:
            return 'stuck'                                   # восемь остановок без роста: топтаться нет смысла
        step = min(cfg.leg, max(0.12, 0.7 * (1.0 - z) * reach))   # у самого образца шаги короче
        goal, heading = self._leg_goal(obs.x, obs.y, self._next_heading(), step)
        if goal is None:
            return 'stuck'
        self._heading = heading
        self._leg = {'goal': goal, 't0': obs.t}
        self._path_goal = None
        return None

    def _next_heading(self):
        v = self._verts
        if len(v) < 2:
            return self._heading                             # первый шаг пробный: куда смотрит робот
        g = _slope(v[-3:]) if len(v) >= 3 else None
        if g is not None and math.hypot(*g) >= 0.1:
            return math.atan2(g[1], g[0])                    # три остановки не на прямой: ехать, куда растёт
        rise = v[-1][2] - v[-2][2]
        step = math.dist(v[-1][:2], v[-2][:2])
        if rise >= 0.5 * step / self.rules.sensor_range_m:   # растёт хотя бы вполовину возможного
            return self._heading
        # Упало или почти не растёт: поворот на 90° в сторону, где свободнее. Верный ли он, покажет
        # следующая остановка — с ней точек хватит, чтобы оценить направление роста.
        x, y = v[-1][:2]
        left, right = (self.arena.clearance(x + 0.4 * math.cos(self._heading + s * math.pi / 2),
                                            y + 0.4 * math.sin(self._heading + s * math.pi / 2)) for s in (1, -1))
        return self._heading + (math.pi / 2 if left >= right else -math.pi / 2)

    def _leg_goal(self, x, y, heading, step):
        """Конец шага по курсу; если там стена или место штрафа — с наименьшим отворотом."""
        for turn in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
            h = heading + math.radians(turn)
            c, s = math.cos(h), math.sin(h)
            if all(self._drivable(x + k * step * c, y + k * step * s) for k in (0.34, 0.67, 1.0)):
                return (x + step * c, y + step * s), h
        return None, heading

    def _drivable(self, x, y):
        ix, iy = self.arena.w2g(x, y)
        return self.arena.inside(ix, iy) and bool(self.graph.ok[iy, ix]) and not self._hazard[iy, ix]

    # --- езда ------------------------------------------------------------------------------------

    def _drive_to(self, obs, io, target, tol):
        """Едет к цели по разрешённым клеткам. Возвращает True по прибытии."""
        if self._path_goal is None or math.dist(self._path_goal, target) > 0.08:
            pts, cost = self.graph.plan((obs.x, obs.y), target)
            if pts is None:
                self._command(io, 0.0, 0.0)
                return False
            self.follower.set_path(pts)
            self._path_goal = target
            if self.rec:
                self.rec.add_path(obs.t, target, pts, cost * self.rules.drain_per_m)
        v, w, arrived = self.follower.step(obs.x, obs.y, obs.th, tol=tol)
        if arrived:
            self._stuck = None
            self._command(io, 0.0, 0.0)
            return True
        if v > 0.0 and self._front < 0.15:
            v = 0.0                                          # лидар видит преграду вплотную по курсу
        self._watch_stuck(obs, v)
        self._command(io, v, w)
        return False

    def _watch_stuck(self, obs, v):
        if self._stuck is None:
            self._stuck = [obs.t, obs.x, obs.y, 0.0, obs.t]
        s = self._stuck
        if v > 0.03:
            s[3] += obs.t - s[4]
        s[4] = obs.t
        if obs.t - s[0] >= 4.0:
            if s[3] >= 3.0 and math.hypot(obs.x - s[1], obs.y - s[2]) < 0.03:
                self.journal.add(obs.t, 'alarm', f'Робот не движется в ({obs.x:.2f}; {obs.y:.2f}): отъезжаю назад')
                self._escape = {'until': obs.t + 1.5, 'v': -0.10}
            self._stuck = None

    def _command(self, io, v, w):
        self._last_v = v
        io.command(v, w)

    # --- запись ----------------------------------------------------------------------------------

    def _plan(self, t, trigger, text, subgoals, source='rule'):
        self.journal.add(t, 'decision', text)
        if self.rec:
            self.rec.add_plan(t, source, trigger, text, subgoals)

    def _record(self, obs):
        if self.rec:
            self.rec.add_events(obs.events)
            self.rec.sample(obs, self.mode)


class GradientAgent(Baseline):
    """Подъём по сигналу; когда сигнала нет — переезд туда, где датчик ещё не слушал."""

    def __init__(self, arena, config, n_samples, rules=None, planner=None, recorder=None):
        super().__init__(arena, config, n_samples, rules, planner, recorder)
        ys, xs = np.nonzero(self.graph.ok & (arena.clear >= 0.3))
        keep = (ys % 10 == 5) & (xs % 10 == 5)                  # точки, куда можно поехать слушать: сетка 0,5 м
        self._anchors = [arena.g2w(i, j) for j, i in zip(ys[keep], xs[keep])]
        self._roam = None
        self._deaf = False                                      # подъём не удался: до следующей точки сигнал не слушать
        self._silent = []                                       # где сигнала не было: ближе ~1,8 м образцов нет

    def _search(self, obs, io):
        if self.mode == 'start':
            self._plan(obs.t, 'start', 'Подъём по сигналу датчика: шаг, остановка, сравнение показаний. '
                       'Нет сигнала — еду слушать туда, где ещё не был.', [])
            self._climb_start(heading=math.atan2(-obs.y, -obs.x))     # первый шаг — к середине арены
        if self._roam is not None:
            self.mode = 'explore'
            z = self._signal()
            if not self._deaf and z is not None and z >= self.cfg.floor + 0.07:
                self._plan(obs.t, 'candidate_found', f'Появился сигнал {z:.2f}: поднимаюсь по нему.', [])
            elif not self._drive_to(obs, io, self._roam['goal'], tol=0.15) and obs.t - self._roam['t0'] < 60.0:
                return
            self._roam, self._deaf = None, False
            self._climb_start(heading=obs.th)
        result = self._climb(obs, io, self.cfg.floor)
        if result == 'collected' and not self._returning:
            self._climb_start()                              # датчик теперь показывает следующий ближайший образец
        elif result in ('lost', 'stuck'):
            self._start_roam(obs, stuck=result == 'stuck')

    def _start_roam(self, obs, stuck):
        self._deaf = stuck
        if not stuck:
            self._silent.append((obs.x, obs.y))
        trail = np.array(self._trail)
        fresh = [p for p in self._anchors
                 if np.hypot(trail[:, 0] - p[0], trail[:, 1] - p[1]).min() >= self.cfg.listen and self._drivable(*p)]
        # Тишина значит, что вокруг пусто почти на дальность датчика: слушать стоит заметно дальше.
        fresh = [p for p in fresh if all(math.dist(p, q) >= 1.5 for q in self._silent)] or fresh
        if not fresh:
            return self._go_home(obs.t, 'сигнала нет, а слушать больше негде')
        goal = min(fresh, key=lambda p: math.dist(p, (obs.x, obs.y)))
        self._roam = {'goal': goal, 't0': obs.t}
        self._path_goal = None
        why = 'Сигнал есть, но не растёт' if stuck else 'Сигнала нет: образцы дальше 1,8 м'
        self._plan(obs.t, 'candidate_lost' if stuck else 'queue_empty',
                   f'{why}. Еду слушать в ({goal[0]:.1f}; {goal[1]:.1f}), где ещё не был.',
                   [{'type': 'explore', 'x': goal[0], 'y': goal[1]}])


class SpiralAgent(Baseline):
    """Дуги растущего радиуса вокруг базы; при сильном сигнале — заезд за образцом подъёмом по сигналу."""

    def __init__(self, arena, config, n_samples, rules=None, planner=None, recorder=None):
        super().__init__(arena, config, n_samples, rules, planner, recorder)
        self._route = spiral_route(arena, self.graph, config.pitch)
        self._i = 0
        self._t_point = 0.0
        self._homing = False
        self._again = False                                     # только что собрал образец и слушает, нет ли рядом ещё
        self._deaf_i = -1                                       # до этой точки маршрута сигнал не слушать

    def _search(self, obs, io):
        if self.mode == 'start':
            self._plan(obs.t, 'start', f'Расширяющаяся спираль от базы: дуги через {self.cfg.pitch:.1f} м, '
                       f'{len(self._route)} точек; при сильном сигнале заезжаю за образцом.',
                       [{'type': 'goto', 'x': x, 'y': y} for x, y in self._route] + [{'type': 'return_base'}],
                       source='fixed')
            self._t_point = obs.t
        if self._homing:
            # После сбора подъём продолжается, только если и следующий образец слышен сильно.
            again = self._again and not self._verts
            result = self._climb(obs, io, self.cfg.strong if again else self.cfg.strong - 0.2)
            if result == 'collected' and not self._returning:
                self._climb_start()
                self._again = True
            elif result in ('lost', 'stuck'):
                self._homing, self._path_goal, self._t_point, self._again = False, None, obs.t, False
                if result == 'stuck':
                    self._deaf_i = self._i                   # подъём не удался: до следующей точки сигнал не слушать
                self._plan(obs.t, 'subgoal_done', 'Возвращаюсь на спираль.', [])
            return
        self.mode = 'explore'
        if self._i >= len(self._route):
            return self._go_home(obs.t, 'спираль пройдена')
        z = self._signal()
        if z is not None and z >= self.cfg.strong and self._i > self._deaf_i:
            self._homing = True
            self._climb_start(heading=obs.th)
            return self._plan(obs.t, 'candidate_found', f'Сильный сигнал {z:.2f}: образец ближе '
                              f'{(1.0 - z) * self.rules.sensor_range_m:.1f} м, заезжаю за ним.', [])
        x, y = self._route[self._i]
        if not self._drivable(x, y) or self._drive_to(obs, io, (x, y), tol=0.15) or obs.t - self._t_point > 40.0:
            self._i += 1
            self._t_point = obs.t


BASELINES = {
    'gradient': (GradientAgent, BaselineConfig(name='gradient', search='gradient')),
    'spiral': (SpiralAgent, BaselineConfig(name='spiral', search='spiral')),
}
