"""Агентский цикл: измерил → обновил модель мира → выдвинул и проверил гипотезы → решил → поехал.

Один и тот же класс работает в быстром симуляторе и в Gazebo: наружу он видит только Observation
и четыре действия RobotIO. Варианты агента для опытов отличаются флагами AgentConfig — что из
измеренного ему разрешено использовать для смены плана.
"""
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np

from .belief import ChangeDetector, HazardMap, SampleBelief, SensorHealth, SoilModel
from .config import BASE, Rules
from .explore import rank_points
from .foresight import Advisor
from .inquiry import Investigator
from .journal import Journal
from .localize import PoseTracker
from .nav import CostGraph, Follower
from .planner import HeuristicPlanner, resolve_subgoals
from .route import survey_route


@dataclass(frozen=True)
class AgentConfig:
    name: str = 'adaptive'
    search: str = 'belief'            # belief — цели из карты вероятностей; route — фиксированный объезд
    explore: str = 'mass'             # точка разведки: mass — сколько вероятности вокруг; infogain — ожидаемая польза
    learn_soil: bool = True           # оценивать стоимость грунта по расходу батареи и объезжать дорогое
    detect_change: bool = True        # замечать, что модель расхода устарела, и переучиваться
    avoid_hazards: bool = True        # запоминать опасные зоны по штрафам и объезжать
    sensor_health: bool = True        # следить за шумом датчика и меньше верить шумным показаниям
    dynamic_reserve: bool = True      # возвращаться по оценке стоимости пути домой, а не по жёсткому порогу
    static_reserve: float = 18.0      # порог заряда для возврата, если dynamic_reserve выключен
    reserve_margin: float = 1.1
    reserve_abs: float = 4.0          # запас сверх дороги домой: одна потеря в опасной зоне и мелочи
    unknown_risk: float = 1.5         # насколько маршрут домой избегает пола, по которому робот ещё не ездил
    hazard_weight: float = 30.0       # во сколько раз дороже клетка, которая наверняка в опасной зоне
    planner: str = 'heuristic'        # heuristic | llm
    llm_strategy: str = 'single'      # single | critic | vote | scored | scored_calc
    candidate_mass: float = 0.35      # с какой уверенности место считается кандидатом
    collect_confidence: float = 0.85  # с какой уверенности пробовать сбор
    max_dv: float = 1.0               # предел изменения линейной скорости за такт, м/с (1.0 — без сглаживания)
    max_dw: float = 10.0              # то же для угловой, рад/с: сглаживание в Gazebo уход одометрии не уменьшило
    llm_min_interval_s: float = 4.0   # не дёргать модель чаще
    llm_wait_s: float = 0.0           # быстрый симулятор: сколько секунд робот стоит, «ожидая ответ модели»
    async_planner: bool = False       # Gazebo: модель думает в отдельном потоке, робот в это время стоит
    localize: bool = True             # поправлять позу одометрии по лидару и карте (did/localize.py)
    science: bool = False             # вести расследования: несколько объяснений странности и опыт (did/inquiry.py)
    foresight: bool = False           # выбирать цель и момент возврата сравнением вариантов плана (did/foresight.py)
    risk_limit: float = 0.05          # допустимая при таком сравнении вероятность не вернуться на базу
    foresight_choice: str = 'planner'  # среди прошедших по риску выбирает: planner — правило планировщика, score — счёт

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # Контрольный агент: объезд по заранее составленному маршруту, модель мира план не меняет.
    'fixed': AgentConfig(name='fixed', search='route', learn_soil=False, detect_change=False,
                         avoid_hazards=False, sensor_health=False, dynamic_reserve=False),
    # Полный агент.
    'adaptive': AgentConfig(name='adaptive'),
    # Тот же агент, но точку разведки выбирает по ожидаемой пользе измерений (did/explore.py).
    'adaptive_ig': AgentConfig(name='adaptive_ig', explore='infogain'),
    'adaptive_llm': AgentConfig(name='adaptive_llm', planner='llm'),
    # Исследователь: сам уточняет модель расхода и расследует странности опытами.
    'scientist': AgentConfig(name='scientist', science=True),
    'scientist_llm': AgentConfig(name='scientist_llm', science=True, planner='llm'),
    # Те же агенты, но цель и момент возврата выбираются сравнением будущих маршрутов с учётом риска.
    # Порог риска 15% подобран на отладочных сценариях 1–80: при 5% возврат почти всегда, но образцов меньше.
    'scientist_fs': AgentConfig(name='scientist_fs', science=True, foresight=True, risk_limit=0.15),
    'adaptive_fs': AgentConfig(name='adaptive_fs', foresight=True, risk_limit=0.15),
    # Отключение по одному механизму: что именно даёт выигрыш.
    'no_soil': AgentConfig(name='no_soil', learn_soil=False, detect_change=False),
    'no_change': AgentConfig(name='no_change', detect_change=False),
    'no_hazard': AgentConfig(name='no_hazard', avoid_hazards=False),
    'no_sensor_health': AgentConfig(name='no_sensor_health', sensor_health=False),
    'static_reserve': AgentConfig(name='static_reserve', dynamic_reserve=False),
    # Только поиск по карте вероятностей, остальная адаптация выключена.
    'belief_only': AgentConfig(name='belief_only', learn_soil=False, detect_change=False,
                               avoid_hazards=False, sensor_health=False, dynamic_reserve=False),
}


def make_config(name, **overrides):
    return replace(PRESETS[name], **overrides)


class Agent:

    def __init__(self, arena, config, n_samples, rules=None, planner=None, recorder=None, knowledge=None,
                 roles=None):
        self.arena = arena
        self.cfg = config
        self.rules = rules or Rules()
        self.n_samples = n_samples
        self.rec = recorder
        self.journal = Journal()
        self.planner = planner or HeuristicPlanner()
        self.base = BASE

        self.graph = CostGraph(arena)                       # дорога к целям: оценка как есть
        self.home_graph = CostGraph(arena)                  # дорога домой: предпочитает проверенный пол
        self.follower = Follower()
        self.tracker = PoseTracker(arena, enabled=config.localize)   # одометрия → поза на карте
        self.belief = SampleBelief(arena, n_samples, self.rules.sensor_range_m)
        self.soil = SoilModel(arena, self.rules.drain_per_m, self.rules.drain_idle_per_s)
        self.change = ChangeDetector()
        self.health = SensorHealth(self.rules.sensor_sigma)
        self.hazard_map = HazardMap()                       # гипотезы о том, где опасные зоны
        self.hazards = []                                   # [(x, y, r)] — их краткая сводка для журнала
        self._risk = np.zeros(arena.free.shape)             # вероятность опасной зоны по клеткам
        self._risk_full = self._risk                        # то же без временного исключения зоны, из которой выезжаем
        self._X, self._Y = arena.cell_centers()
        self._trail = deque(maxlen=14)                      # последние ~0,7 м пути: подход к месту штрафа
        self._trail_pause = -1e9
        self._risk_version = 0
        self._grace = (-1e9, None)                          # (до какого времени, какая зона) — выезд после штрафа

        self.queue = []                                     # подцели, первая — текущая
        self.mode = 'start'
        self.collected = 0
        self.finished = False
        self._trigger = 'start'                             # причина, по которой нужен новый план
        self._returning = False
        self._wait_until = 0.0
        self._future = None
        self._pool = ThreadPoolExecutor(max_workers=1) if config.async_planner else None
        self._last_llm_t = -1e9
        self._known_cands = []
        self._visited = deque(maxlen=8)                     # недавние точки разведки: [(t, x, y)]
        self._rng = np.random.default_rng(0)                # гипотезы для расчёта пользы измерений

        self._anchor = None                                 # начало текущего отрезка для оценки грунта
        self._skip_soil_until = -1.0
        self._hazard_t = -1e9
        self._readings = deque(maxlen=5)
        self._front = math.inf
        self._cost_dirty = False
        self._cost_t = -1e9
        self._cost_version = 0
        self._base_dist = None
        self._base_pred = None
        self._path_goal = None
        self._path_version = -1
        self._path_t = -1e9
        self._escape = None
        self._stuck = None                                  # (t, x, y, время с командой «вперёд»)
        self._cands_t = -1e9
        self._cands = []
        self._slow_t = -1e9
        self._soil_h = []                                   # гипотезы о грунтах: [{'key', 'x', 'y'}]
        self._n_sample_h = 0
        self._last_cmd = (0.0, 0.0)
        self._misses = deque(maxlen=6)                      # недавние ложные сборы: (t, x, y)
        self._no_collect_until = -1e9
        self.inv = Investigator(self, knowledge, roles) if config.science else None
        self.fs = Advisor(self) if config.foresight else None      # сравнение будущих маршрутов

    # ======================================================================================
    # один такт цикла
    # ======================================================================================

    def tick(self, obs, io):
        if self.cfg.localize:              # дальше весь агент работает с позой на карте, а не с одометрией
            x, y, th = self.tracker.update(obs.x, obs.y, obs.th, obs.scan, obs.scan_pose, obs.scan_step)
            obs = replace(obs, x=x, y=y, th=th)
        if obs.done or self.finished:
            io.command(0.0, 0.0)
            self._record(obs)
            return
        self._perceive(obs)
        self._refresh_costs(obs)
        if obs.t - self._slow_t >= 1.0:
            self._slow_t = obs.t
            self._soil_hypotheses(obs.t)
            self._check_return(obs)
        if self._trigger and obs.t >= self._wait_until:
            self._deliberate(obs)
        self._act(obs, io)
        self._record(obs)

    # ======================================================================================
    # восприятие: события, грунт, датчик образцов
    # ======================================================================================

    def _perceive(self, obs):
        if obs.scan is not None:
            self._front = float(min(obs.scan[:20].min(), obs.scan[-20:].min()))
        for ev in obs.events:
            self._on_event(ev, obs)

        if self.inv:
            self.inv.observe(obs)          # модель расхода и расследования вместо простой оценки «заряд на метр»
        else:
            if self._anchor is None:
                self._anchor = (obs.x, obs.y, obs.battery, obs.t)
            ax, ay, ab, at = self._anchor
            if math.hypot(obs.x - ax, obs.y - ay) >= 0.06:
                self._anchor = (obs.x, obs.y, obs.battery, obs.t)
                self._on_segment(ax, ay, obs.x, obs.y, ab - obs.battery, obs.t - at, obs.t)

        if obs.sensor is not None:
            self._on_reading(obs.sensor, obs)

    def _on_event(self, ev, obs):
        kind = ev.get('type')
        if kind == 'hazard_hit':
            self._skip_soil_until = obs.t + 0.6
            self._anchor = (obs.x, obs.y, obs.battery, obs.t)   # разовая потеря заряда — не свойство грунта
            self._hazard_t = obs.t
            self.journal.add(obs.t, 'alarm', f'Штраф: опасная зона в ({obs.x:.2f}; {obs.y:.2f})', tag='hazard')
            if self.inv:
                self.inv.on_penalty(obs)
            if self.cfg.avoid_hazards:
                back = self._last_cmd[0] < 0
                heading = obs.th + (math.pi if back else 0.0)
                known = len(self.hazard_map.zones)
                # Штраф уже получен, поэтому робот не пятится, а выезжает из зоны туда, куда ему нужно:
                # на несколько секунд эта зона для маршрута не считается, потом объезжается.
                self._grace = (obs.t + 4.0, self.hazard_map.hit(obs.x, obs.y, heading, list(self._trail)))
                self._trail.clear()
                self._trail_pause = obs.t + 4.0            # пока робот выезжает из зоны, его путь не «безопасный»
                self._sync_hazards(obs.t, new=len(self.hazard_map.zones) > known)
                self._path_goal = None
                self._request_plan('hazard')
        elif kind == 'sample_collected' and ev.get('collected', 0) > self.collected:
            # Ответ сервиса сбора потерялся, а образец засчитан: повторять сбор нельзя, сверяемся по событию.
            self.collected = ev['collected']
            self.belief.collected(obs.x, obs.y)
            self.journal.add(obs.t, 'action', f'Судья сообщил о сборе образца: всего {self.collected} из {self.n_samples}')
        elif kind == 'collision':
            self.journal.add(obs.t, 'alarm', f'Штраф: столкновение в ({obs.x:.2f}; {obs.y:.2f})', tag='collision')
            self._escape = {'until': obs.t + 1.5, 'v': 0.10 if self._last_cmd[0] < 0 else -0.10}

    def _sync_hazards(self, t, new=False):
        """Пересчитать карту риска и сводку по зонам после нового штрафа или уточнения."""
        self._risk = self.hazard_map.risk(self._X, self._Y, skip=self._grace[1])
        self._risk_full = self._risk if self._grace[1] is None else self.hazard_map.risk(self._X, self._Y)
        self._risk_version = self.hazard_map.version
        self.hazards = [(x, y, r) for x, y, r in self.hazard_map.summary()]
        self._cost_dirty = True
        self._cost_t = -1e9
        if new:
            cx, cy, r = self.hazards[-1]
            if self.rec:
                self.rec.add_hazard(t, cx, cy, r)
            self.journal.open(t, f'hazard:{len(self.hazards)}',
                              f'около ({cx:.1f}; {cy:.1f}) опасная зона радиусом около {r:.1f} м',
                              'объезжать; новых штрафов в этом месте быть не должно', x=cx, y=cy)

    def _trail_point(self, x, y, t):
        """Ещё одна точка, пройденная без штрафа: она уточняет, где опасных зон нет."""
        if self.cfg.avoid_hazards and t > self._trail_pause:
            self._trail.append((x, y))
            if self.hazard_map.zones:
                self.hazard_map.safe(x, y)                 # проехал без штрафа — часть гипотез о зоне отпадает
                if self.hazard_map.version != self._risk_version:
                    self._sync_hazards(t)

    def _on_segment(self, x0, y0, x1, y1, spent, dt, t):
        ds = math.hypot(x1 - x0, y1 - y0)
        self._trail_point(x1, y1, t)
        ratio = (spent - self.rules.drain_idle_per_s * dt) / (self.rules.drain_per_m * ds)
        if t <= self._skip_soil_until or ratio > 7.0:
            # Скачок расхода — это не грунт (разовая потеря в опасной зоне), в карту стоимостей не идёт.
            if ratio > 7.0 and t - self._hazard_t > 3.0:
                self.journal.add(t, 'alarm', f'Скачок расхода батареи: {spent:.1f} ед. на {ds * 100:.0f} см пути')
            return
        if not self.cfg.learn_soil:
            return
        ratio, predicted, conf = self.soil.observe(x0, y0, x1, y1, spent, dt)
        self._cost_dirty = True
        if self.cfg.detect_change:
            verdict = self.change.update(ratio, predicted, conf, ds)
            if verdict:
                self._on_model_mismatch(verdict, x1, y1, ratio, predicted, t)

    def _on_model_mismatch(self, verdict, x, y, ratio, predicted, t):
        self.journal.add(t, 'alarm',
                         f'Модель расхода не сходится около ({x:.1f}; {y:.1f}): ждал ×{predicted:.1f}, '
                         f'вижу ×{ratio:.1f} ({verdict} прогноза). Похоже, грунты изменились',
                         tag='model_mismatch', x=round(x, 2), y=round(y, 2))
        for h in self._soil_h:
            if math.hypot(h['x'] - x, h['y'] - y) <= 0.8:
                self.journal.close(t, h['key'], 'outdated', 'расход на участке перестал совпадать с оценкой')
        self._soil_h = [h for h in self._soil_h if math.hypot(h['x'] - x, h['y'] - y) > 0.8]
        # Рядом с местом расхождения старым данным не верим совсем, в остальных местах — меньше.
        self.soil.forget(x, y, radius=0.7, keep=0.1, keep_elsewhere=0.6)
        self._cost_dirty = True
        self._cost_t = -1e9
        self._request_plan('model_mismatch')

    def _on_reading(self, z, obs):
        self._readings.append(z)
        sigma = self.rules.sensor_sigma
        if self.cfg.sensor_health:
            change = self.health.update(z)
            if change == 'degraded' and self.inv:
                self.inv.on_noise(obs)         # не вывод, а повод для расследования
            elif change == 'degraded':
                self.journal.add(obs.t, 'alarm',
                                 f'Датчик образцов шумит: разброс {self.health.sigma:.2f} вместо '
                                 f'{self.health.nominal:.2f}', tag='sensor_degraded')
                self.journal.open(obs.t, 'sensor', 'датчик образцов неисправен, его показаниям нельзя верить как раньше',
                                  'снизить вес показаний и следить, вернётся ли разброс к норме')
                self.belief.relax(0.25)
                self._request_plan('sensor_degraded')
            elif change == 'recovered':
                self.journal.close(obs.t, 'sensor', 'confirmed',
                                   f'шум был повышен, сейчас вернулся к {self.health.sigma:.2f}; '
                                   'показания за время сбоя учтены с малым весом')
            sigma = self.health.effective_sigma()
        if self.inv:
            z, sigma = self.inv.reading(z, obs, sigma)
            if z is None:                      # датчик залип: карту образцов не трогаем
                return
        self.belief.update(obs.x, obs.y, z, sigma)

    def _soil_hypotheses(self, t):
        if not self.cfg.learn_soil:
            return
        for z in self.soil.zones():
            known = next((h for h in self._soil_h if math.hypot(h['x'] - z['x'], h['y'] - z['y']) <= 0.6), None)
            if known is None:
                key = f"soil:{t:.0f}:{z['x']}:{z['y']}"
                self.journal.open(t, key,
                                  f"около ({z['x']:.1f}; {z['y']:.1f}) грунт дороже обычного примерно в "
                                  f"{z['mult']:.1f} раза",
                                  'сравнить расход на следующих 0,3 м этого участка с прогнозом',
                                  x=z['x'], y=z['y'])
                self._soil_h.append({'key': key, 'x': z['x'], 'y': z['y']})
            elif z['evidence_m'] >= 0.3:
                self.journal.close(t, known['key'], 'confirmed',
                                   f"на {z['evidence_m']:.1f} м пути расход ×{z['mult']:.1f}; "
                                   'участок внесён в карту стоимостей')

    # ======================================================================================
    # модель стоимости пути и решение о возврате
    # ======================================================================================

    def _refresh_costs(self, obs):
        if self._grace[1] is not None and obs.t >= self._grace[0]:
            self._grace = (-1e9, None)                     # выезд закончен: зона снова учитывается в маршрутах
            self._sync_hazards(obs.t)
        if not self._cost_dirty or obs.t - self._cost_t < 1.0:
            return
        mult = self.soil.mult_grid() if self.cfg.learn_soil else None
        # Риск опасной зоны — штраф к стоимости клетки: чем вероятнее зона, тем дальше её объезжать.
        danger = 1.0 + self.cfg.hazard_weight * self._risk if self.hazards else None
        self.graph.set_cost(mult, bias=danger)
        # Домой — по проверенному: там, где робот уже ездил, нет ни опасных зон, ни сюрпризов с грунтом.
        if self.cfg.learn_soil:
            unknown = 1.0 + self.cfg.unknown_risk * (1.0 - self.soil.confidence_grid())
            danger = unknown if danger is None else danger * unknown
        self.home_graph.set_cost(mult, bias=danger)
        self._cost_dirty = False
        self._cost_t = obs.t
        self._cost_version += 1
        self._base_dist = None

    def _home_cost(self, x, y):
        """Оценка заряда на дорогу до базы по текущей карте стоимостей."""
        if self._base_dist is None:
            self._base_dist, self._base_pred = self.home_graph.field(*self.base)
        return self.home_graph.energy(self._base_dist, self._base_pred, x, y) * self._per_m()

    def _per_m(self):
        """Заряд на метр пути: номинал из правил либо то, что исследователь выяснил сам (груз, повороты)."""
        return self.inv.per_meter() if self.inv else self.rules.drain_per_m

    def _affordable(self, battery, cost_to, cost_back):
        if self.cfg.dynamic_reserve:
            return battery - cost_to - cost_back * self.cfg.reserve_margin - self.cfg.reserve_abs >= 0.0
        return battery - cost_to >= self.cfg.static_reserve

    def _check_return(self, obs):
        if self.fs is not None:
            return self.fs.check(obs)          # вместо жёсткого запаса — сравнение «ехать дальше» и «домой»
        if self._returning:
            return
        home = self._home_cost(obs.x, obs.y)
        if self.cfg.dynamic_reserve:
            need = home * self.cfg.reserve_margin + self.cfg.reserve_abs
        else:
            need = self.cfg.static_reserve
        if self.inv:
            need += self.inv.reserve(obs.t)    # идущая утечка или невыясненная причина расхода
        reason = None
        if self.collected >= self.n_samples:
            reason = 'все образцы собраны'
        elif obs.battery <= need:
            reason = (f'заряд {obs.battery:.1f} ед., дорога домой оценивается в {home:.1f} ед.'
                      if self.cfg.dynamic_reserve else f'заряд {obs.battery:.1f} ед. опустился до порога {need:.0f}')
        elif self.rules.time_limit_s - obs.t <= home / self.rules.drain_per_m / 0.15 + 10.0:
            reason = 'время прогона на исходе'
        if reason:
            self._go_home(obs.t, reason)

    def _go_home(self, t, reason):
        self._returning = True
        self.queue = [{'type': 'return_base'}]
        self._trigger = None
        self._path_goal = None
        self.journal.add(t, 'decision', f'Возвращаюсь на базу: {reason}')
        if self.rec:
            self.rec.add_plan(t, 'rule', 'return', f'Возвращаюсь на базу: {reason}.', [{'type': 'return_base'}])

    # ======================================================================================
    # планирование верхнего уровня
    # ======================================================================================

    def _request_plan(self, trigger):
        if not self._returning and self.cfg.search == 'belief':
            self._trigger = trigger

    def _candidates(self, t):
        if t - self._cands_t >= 0.5:
            self._cands_t = t
            self._cands = self.belief.candidates(min_mass=0.2)
        return self._cands

    def _state(self, obs, trigger):
        """Сводка для планировщика. Формат общий для правила и для языковой модели."""
        dist, pred = self.graph.field(obs.x, obs.y)
        battery = obs.battery

        def costs(p):
            to = self.graph.energy(dist, pred, p['x'], p['y']) * self._per_m()
            back = self._home_cost(p['x'], p['y'])
            ix, iy = self.arena.w2g(p['x'], p['y'])
            safe = self._risk[iy, ix] < 0.5
            return to, back, bool(safe and math.isfinite(to) and self._affordable(battery, to, back))

        cands = []
        for c in self.belief.candidates(min_mass=self.cfg.candidate_mass):
            to, back, ok = costs(c)
            cands.append({'id': f'C{len(cands) + 1}', 'x': round(c['x'], 2), 'y': round(c['y'], 2),
                          'confidence': round(c['mass'], 2), 'cost_to': round(to, 1), 'cost_back': round(back, 1),
                          'feasible': ok})

        left = max(1, self.n_samples - self.collected)
        points = []
        raw = self.belief.explore_points()
        if self.cfg.explore == 'infogain':
            raw = self._rank_by_gain(raw, obs, dist, pred)
        else:
            raw.sort(key=lambda p: -p['mass'] / (self.graph.cost_at(dist, p['x'], p['y']) + 1.0))
        for p in raw:
            if len(points) >= 4:
                break
            if any(math.hypot(p['x'] - q['x'], p['y'] - q['y']) < 0.7 for q in points):
                continue
            if any(obs.t - vt < 40.0 and math.hypot(p['x'] - vx, p['y'] - vy) < 0.4 for vt, vx, vy in self._visited):
                continue
            if math.hypot(p['x'] - obs.x, p['y'] - obs.y) < 0.3:
                continue
            to, back, ok = costs(p)
            points.append({'id': f'E{len(points) + 1}', 'x': round(p['x'], 2), 'y': round(p['y'], 2),
                           'unseen_share': round(min(1.0, p['mass'] / left), 2), 'cost_to': round(to, 1),
                           'cost_back': round(back, 1), 'feasible': ok})
            if 'gain' in p:
                points[-1]['gain_bits'] = round(p['gain'], 2)

        zones = [{'id': f'S{i + 1}', **z,
                  'status': 'confirmed' if z['evidence_m'] >= 0.3 else 'suspected'}
                 for i, z in enumerate(self.soil.zones())] if self.cfg.learn_soil else []
        alarms = [e['text'] for e in self.journal.entries[-12:] if e['kind'] == 'alarm' and obs.t - e['t'] <= 30.0]
        return {
            'trigger': trigger,
            'time_s': round(obs.t, 1), 'time_limit_s': self.rules.time_limit_s,
            'battery': round(battery, 1), 'battery_start': self.rules.battery_start,
            'pose': {'x': round(obs.x, 2), 'y': round(obs.y, 2)},
            'base': {'x': self.base[0], 'y': self.base[1]},
            'samples': {'collected': self.collected, 'total': self.n_samples},
            'return_cost': round(self._home_cost(obs.x, obs.y), 1),
            'candidates': cands,
            'explore_points': points,
            'soil_zones': zones,
            'hazards': [{'id': f'Z{i + 1}', 'x': round(x, 2), 'y': round(y, 2), 'radius': r}
                        for i, (x, y, r) in enumerate(self.hazards)],
            'sensor': {'reading': round(float(np.mean(self._readings)), 2) if self._readings else None,
                       'noise': round(self.health.effective_sigma() if self.cfg.sensor_health
                                      else self.rules.sensor_sigma, 3),
                       'status': 'degraded' if self.cfg.sensor_health and self.health.degraded else 'ok'},
            'recent_events': [],
            'alarms': alarms,
            'open_hypotheses': [{'id': h['id'], 'statement': h['statement']} for h in self.journal.open_list()][-6:],
        }

    def _rank_by_gain(self, raw, obs, dist, pred):
        """Точки разведки по ожидаемой пользе измерений на единицу заряда: считается и дорога до точки."""
        sigma = self.health.effective_sigma() if self.cfg.sensor_health else self.rules.sensor_sigma
        raw = [p for p in raw if math.hypot(p['x'] - obs.x, p['y'] - obs.y) >= 0.3 and not any(
            obs.t - vt < 40.0 and math.hypot(p['x'] - vx, p['y'] - vy) < 0.4 for vt, vx, vy in self._visited)]
        return rank_points(self.belief, raw,
                           cost_of=lambda p: self.graph.cost_at(dist, p['x'], p['y']) * self.rules.drain_per_m,
                           sigma=sigma, rng=self._rng, origin=(obs.x, obs.y),
                           path_of=lambda p: self.graph.trace(pred, p['x'], p['y']))

    def _deliberate(self, obs):
        if self.cfg.search == 'route':
            route = survey_route(self.arena)
            self.queue = [{'type': 'goto', 'x': x, 'y': y} for x, y in route] + [{'type': 'return_base'}]
            self._trigger = None
            text = f'Объезд по заранее составленному маршруту из {len(route)} точек, затем база.'
            self.journal.add(obs.t, 'decision', text)
            if self.rec:
                self.rec.add_plan(obs.t, 'fixed', 'start', text, self.queue)
            return

        if self._future is None:
            trigger = self._trigger
            state = self._state(obs, trigger)
            ahead = self.fs.plan(obs, state) if self.fs is not None else None    # выбор сравнением вариантов
            use_llm =self.cfg.planner == 'llm' and obs.t - self._last_llm_t >= self.cfg.llm_min_interval_s
            planner = self.planner if use_llm or self.cfg.planner != 'llm' else HeuristicPlanner()
            if use_llm:
                self._last_llm_t = obs.t
            self._pending = (state, trigger)
            if self._pool is not None and use_llm:
                self._future = self._pool.submit(planner.plan, state)
                return
            plan = ahead or planner.plan(state)
            if use_llm and self.cfg.llm_wait_s:
                self._wait_until = obs.t + self.cfg.llm_wait_s
        else:
            if not self._future.done():
                return
            plan, self._future = self._future.result(), None
        state, trigger = self._pending
        self._apply_plan(obs, plan, state, trigger)

    def _apply_plan(self, obs, plan, state, trigger):
        t = obs.t
        subgoals = resolve_subgoals(plan['subgoals'], state)
        source = plan['source']
        note = ''
        rule_goals = resolve_subgoals(HeuristicPlanner().plan(state)['subgoals'], state)
        if not subgoals:
            # План модели неисполним (цели исчезли или не по заряду) — решает правило.
            fallback = HeuristicPlanner().plan(state)
            subgoals = resolve_subgoals(fallback['subgoals'], state)
            note = ' План отклонён исполнителем: ни одна подцель не исполнима. Решение по правилу: ' + fallback['reasoning']
            source = 'fallback'
        self.queue = subgoals
        self._trigger = None
        self._path_goal = None
        self._known_cands = [(c['x'], c['y']) for c in state['candidates']]
        if subgoals and subgoals[0]['type'] == 'return_base':
            self._returning = True
        reasoning = plan['reasoning'] + note
        self.journal.add(t, 'decision', reasoning, source=source, trigger=trigger,
                         subgoals=[_brief(s) for s in subgoals], **(plan.get('data') or {}))
        for h in plan.get('hypotheses') or []:
            # Предположения модели идут в журнал заметкой: проверить их исполнитель сам не умеет.
            self.journal.add(t, 'llm', f"Модель предполагает: {h['statement']}. Как проверить: {h.get('test', '—')}")
        if self.rec:
            self.rec.add_plan(t, source, trigger, reasoning, subgoals)
            if plan.get('exchanges'):
                self.rec.plans[-1]['rule_match'] = bool(subgoals and rule_goals and subgoals[0] == rule_goals[0])
            for ex in plan.get('exchanges') or []:
                self.rec.add_llm(t, ex)

    # ======================================================================================
    # исполнение подцелей
    # ======================================================================================

    def _act(self, obs, io):
        if self._future is not None or obs.t < self._wait_until:
            self.mode = 'think'
            return self._command(io, 0.0, 0.0)
        if self._escape:
            if obs.t < self._escape['until']:
                self.mode = 'escape'
                return self._command(io, self._escape['v'], 0.0)
            self._escape = None
            self._path_goal = None
            self._stuck = None
        if self.inv and self.inv.act(obs, io):
            return                             # такт занят опытом расследования
        if not self.queue:
            if self.cfg.search == 'route':
                self._go_home(obs.t, 'маршрут пройден')
            elif not self._trigger:
                self._trigger = 'queue_empty'
            return self._command(io, 0.0, 0.0)
        sg = self.queue[0]
        if sg['type'] == 'investigate':
            self._do_investigate(sg, obs, io)
        elif sg['type'] == 'return_base':
            self._do_return(obs, io)
        else:
            self._do_goto(sg, obs, io)

    def _do_investigate(self, sg, obs, io):
        t = obs.t
        sg.setdefault('t0', t)
        near = [c for c in self._candidates(t) if math.hypot(c['x'] - sg['x'], c['y'] - sg['y']) <= 0.7]
        if not near or t - sg['t0'] > 120.0:
            self.journal.close(t, sg.get('key'), 'refuted', 'новые показания датчика место не подтвердили')
            return self._end_subgoal(obs, io, 'candidate_lost')
        best = max(near, key=lambda c: c['mass'])
        sg['x'], sg['y'] = best['x'], best['y']
        if 'key' not in sg and best['mass'] >= 0.6:
            # Пока уверенность мала, это уточнение, а не гипотеза: в журнал идёт только то, что можно проверить.
            self._n_sample_h += 1
            sg['key'] = f'sample:{self._n_sample_h}'
            self.journal.open(t, sg['key'], f"образец лежит около ({sg['x']:.1f}; {sg['y']:.1f})",
                              'подъехать вплотную и попробовать собрать', x=sg['x'], y=sg['y'])
        here = self.belief.prob_within(obs.x, obs.y, 0.25)
        if t < self._no_collect_until:
            here = min(here, 0.0)                  # пауза после промахов: только подъезжаем и слушаем
        if here >= self.cfg.collect_confidence:
            if any(t - mt < 30.0 and math.hypot(obs.x - mx, obs.y - my) < 0.35 for mt, mx, my in self._misses):
                # Здесь только что был промах: второй раз на том же месте не пробуем, как бы ни был уверен датчик.
                self.belief.clear_disc(obs.x, obs.y, 0.35, factor=0.02)
                self.journal.close(t, sg.get('key'), 'refuted', 'на этом месте уже был промах, повторно не собираю')
                return self._end_subgoal(obs, io, 'candidate_lost')
            return self._try_collect(sg, obs, io, here)
        self.mode = 'approach'
        if self._drive_to(obs, io, (sg['x'], sg['y']), tol=0.05):
            # Стоим на пике, но уверенности мало: копим показания. Не сошлось за 8 с — решаем по тому, что есть.
            sg.setdefault('arrived', t)
            if t - sg['arrived'] > 8.0:
                if here >= 0.5:
                    return self._try_collect(sg, obs, io, here)
                self.journal.close(t, sg.get('key'), 'refuted', f'на месте уверенность только {here:.0%}')
                self._end_subgoal(obs, io, 'candidate_lost')
        else:
            sg.pop('arrived', None)

    def _try_collect(self, sg, obs, io, confidence):
        self.mode = 'collect'
        self._command(io, 0.0, 0.0)
        if abs(obs.v) > 0.03:
            return
        ok, _ = io.collect()
        if ok:
            self.collected += 1
            self.belief.collected(obs.x, obs.y)
            if self.inv:
                self.inv.on_collect(obs.t)
            self.journal.add(obs.t, 'action', f'Сбор в ({obs.x:.2f}; {obs.y:.2f}) при уверенности {confidence:.0%}: '
                             f'образец взят, всего {self.collected} из {self.n_samples}')
            self.journal.close(obs.t, sg.get('key'), 'confirmed', f'образец собран в ({obs.x:.2f}; {obs.y:.2f})')
            self._end_subgoal(obs, io, 'sample_collected')
            self._wait_until = obs.t + 1.0      # дать датчику показать следующий ближайший образец
        else:
            self.belief.clear_disc(obs.x, obs.y, 0.30, factor=0.05)
            self._misses.append((obs.t, obs.x, obs.y))
            if sum(1 for mt, _, _ in self._misses if obs.t - mt < 25.0) >= 2:
                # Два промаха подряд: карте образцов сейчас верить нельзя. Пауза в сборе и частичный сброс карты.
                self._no_collect_until = obs.t + 15.0
                self.belief.relax(0.4)
                self.journal.add(obs.t, 'alarm', 'Два ложных сбора подряд: 15 секунд не собираю и заново набираю показания')
            if self.inv:
                self.inv.on_miss(obs)
            self.journal.add(obs.t, 'action', f'Сбор в ({obs.x:.2f}; {obs.y:.2f}) при уверенности {confidence:.0%}: '
                             'промах, получен штраф')
            self.journal.close(obs.t, sg.get('key'), 'refuted', 'образца в радиусе 0,3 м не оказалось')
            self._end_subgoal(obs, io, 'false_collect')

    def _do_goto(self, sg, obs, io):
        t = obs.t
        self.mode = 'explore' if sg['type'] == 'explore' else 'travel'
        strong = [c for c in self._candidates(t) if c['mass'] >= self.cfg.candidate_mass]
        if self.cfg.search == 'route':
            # Фиксированный маршрут не меняется; по дороге агент лишь подбирает то, что оказалось рядом.
            close = [c for c in strong if c['mass'] >= 0.5 and math.hypot(c['x'] - obs.x, c['y'] - obs.y) <= 0.9]
            if close:
                c = max(close, key=lambda c: c['mass'])
                self.queue.insert(0, {'type': 'investigate', 'x': c['x'], 'y': c['y']})
                self._path_goal = None
                self.journal.add(t, 'decision', f"Датчик показывает образец рядом с маршрутом, около "
                                 f"({c['x']:.1f}; {c['y']:.1f}): заезжаю и возвращаюсь на маршрут")
                return self._command(io, 0.0, 0.0)
        elif any(all(math.hypot(c['x'] - kx, c['y'] - ky) > 0.5 for kx, ky in self._known_cands) for c in strong):
            return self._end_subgoal(obs, io, 'candidate_found')
        if self._drive_to(obs, io, (sg['x'], sg['y']), tol=0.15):
            if sg['type'] == 'explore':
                self._visited.append((t, sg['x'], sg['y']))
            self._end_subgoal(obs, io, 'subgoal_done')

    def _do_return(self, obs, io):
        self.mode = 'return'
        if self._drive_to(obs, io, self.base, tol=0.08) and abs(obs.v) < 0.03:
            ok, msg = io.finish()
            self.finished = True
            self.mode = 'done'
            self.journal.add(obs.t, 'action', f'Финиш: {msg}. Собрано {self.collected} из {self.n_samples}, '
                             f'заряд {obs.battery:.1f}')

    def _end_subgoal(self, obs, io, trigger):
        self.queue.pop(0)
        self._path_goal = None
        self._command(io, 0.0, 0.0)
        if self.cfg.search == 'belief' and not self._returning:
            # После сбора, промаха и находки план пересматривается сразу; после обычной точки — когда очередь пуста.
            if trigger != 'subgoal_done' or not self.queue:
                self._trigger = trigger

    def _drive_to(self, obs, io, target, tol):
        """Едет к цели по кратчайшему по расходу пути. Возвращает True по прибытии."""
        t = obs.t
        moved_goal = self._path_goal is None or math.dist(self._path_goal, target) > 0.08
        stale = self._path_version != self._cost_version and t - self._path_t > 1.5
        if moved_goal or stale:
            graph = self.home_graph if self._returning else self.graph
            pts, cost = graph.plan((obs.x, obs.y), target)
            if pts is None:
                self._command(io, 0.0, 0.0)
                return False
            self.follower.set_path(pts)
            self._path_goal, self._path_version, self._path_t = target, self._cost_version, t
            if self.rec and (moved_goal or stale):
                self.rec.add_path(t, target, pts, cost * self.rules.drain_per_m)
        v, w, arrived = self.follower.step(obs.x, obs.y, obs.th, tol=tol)
        if arrived:
            self._stuck = None
            self._command(io, 0.0, 0.0)
            return True
        if v > 0.0 and self._front < 0.15:
            v = 0.0                                   # лидар видит преграду вплотную по курсу
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
        # При необходимости команду можно сглаживать (max_dv, max_dw); по умолчанию выключено.
        pv, pw = self._last_cmd
        v = pv + min(max(v - pv, -self.cfg.max_dv), self.cfg.max_dv)
        w = pw + min(max(w - pw, -self.cfg.max_dw), self.cfg.max_dw)
        self._last_cmd = (v, w)
        io.command(v, w)

    # ======================================================================================
    # запись
    # ======================================================================================

    def _record(self, obs):
        if not self.rec:
            return
        self.rec.add_events(obs.events)
        self.rec.sample(obs, self.mode, fix=self.tracker.stats['fix'] if self.cfg.localize else None)
        if self.rec.want_belief(obs.t):
            self.rec.add_belief(obs.t, self.belief)
            if self.cfg.learn_soil:
                self.rec.add_soil(obs.t, self.soil)


def _brief(sg):
    if sg['type'] == 'return_base':
        return 'return_base'
    if 'target' in sg:
        return f"{sg['type']} {sg['target']} ({sg['x']:.1f}; {sg['y']:.1f})"
    return f"{sg['type']} ({sg['x']:.1f}; {sg['y']:.1f})"
