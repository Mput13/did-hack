"""Проверки правок P1: спрямление пути, пережидание сбоя датчика, разбор потерь."""
import math
import sys
from pathlib import Path

import numpy as np

from did.agent import PRESETS, AgentConfig
from did.arena import load_arena
from did.config import BASE, Rules
from did.nav import INFLATE, CostGraph, path_length, straighten
from did.runner import run_episode
from did.scenario import Scenario, Zone, generate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import loss_breakdown as lb     # noqa: E402


# --- прежние агенты не тронуты --------------------------------------------------------------------

def test_new_behaviour_is_off_by_default():
    cfg = AgentConfig()
    assert not cfg.fault_wait and not cfg.straight_paths and cfg.soil_spread_m == 0.0
    for name in ('adaptive', 'scientist', 'fixed', 'adaptive_fs', 'scientist_fs'):
        assert not PRESETS[name].fault_wait and not PRESETS[name].straight_paths
    for name in ('adaptive_v2', 'scientist_v2'):
        assert PRESETS[name].fault_wait and PRESETS[name].straight_paths
    assert PRESETS['scientist_v2'].science and not PRESETS['adaptive_v2'].science


# --- спрямление пути ------------------------------------------------------------------------------

def test_straighten_is_shorter_and_stays_in_allowed_cells():
    arena = load_arena()
    g = CostGraph(arena)
    rng = np.random.default_rng(1)
    ys, xs = np.nonzero(g.ok)
    saved = []
    for _ in range(60):
        a, b = rng.integers(len(xs), size=2)
        pts, _ = g.plan(arena.g2w(xs[a], ys[a]), arena.g2w(xs[b], ys[b]))
        out = straighten(g, pts)
        assert out[0] == pts[0] and math.dist(out[-1], pts[-1]) < 1e-9
        assert path_length(out) <= path_length(pts) + 1e-9
        assert all(arena.clearance(x, y) >= INFLATE - arena.res for x, y in out)
        assert max(math.dist(p, q) for p, q in zip(out, out[1:])) <= 0.08       # шаг точек — как у пути по клеткам
        saved.append(1.0 - path_length(out) / max(path_length(pts), 1e-9))
    assert 0.01 < np.mean(saved) < 0.09           # путь по клеткам длиннее прямой не больше чем на 8%


def test_straighten_does_not_cut_through_expensive_floor():
    arena = load_arena()
    g = CostGraph(arena)
    X, Y = arena.cell_centers()
    zone = np.hypot(X + 1.5, Y - 0.2) <= 0.35                      # дорогой круг между стартом и целью
    g.set_cost(np.where(zone, 4.0, 1.0))
    pts, _ = g.plan((-2.0, 0.2), (-1.0, 0.2))
    out = straighten(g, pts)
    cells = lambda path: sum(bool(zone[arena.w2g(x, y)[::-1]]) for x, y in path)      # noqa: E731
    assert cells(out) <= cells(pts)
    assert cells(out) == 0


# --- пережидание сбоя датчика ----------------------------------------------------------------------

def _noisy_scenario():
    """Три образца и шумный датчик с 8-й по 40-ю секунду: без зон и грунтов, чтобы виден был один механизм."""
    return Scenario(level='hard', seed=1, samples=[[-1.0, -0.5], [0.5, -1.5], [0.5, 1.5]],
                    events=[{'t': 8.0, 'type': 'sensor_fault', 'duration': 32.0, 'sigma': 0.25}])


def test_agent_stands_still_while_sensor_is_noisy():
    from did.recorder import load_trace
    from did.runner import RUNS
    moved = {}
    for name in ('adaptive', 'adaptive_v2'):
        s = run_episode('hard', 1, name, experiment='_test', arm=f'p1_{name}', scenario=_noisy_scenario())
        tr = load_trace(RUNS / s['file'])
        t, x, y = (np.array(tr['track'][k]) for k in ('t', 'x', 'y'))
        fault = (t[1:] >= 14.0) & (t[1:] <= 38.0)         # сбой уже замечен и ещё не кончился
        moved[name] = float(np.hypot(np.diff(x), np.diff(y))[fault].sum())
        tags = [(e.get('data') or {}).get('tag') for e in tr['journal']]
        assert ('sensor_wait' in tags) == (name == 'adaptive_v2')
        if name == 'adaptive_v2':
            assert 'sensor_wait_end' in tags and 'wait' in tr['modes']
            assert s['metrics']['returned'] and s['metrics']['samples_collected'] == 3
    assert moved['adaptive_v2'] < 0.3 < moved['adaptive']


class _StubAgent:
    """Ровно то, на что смотрит SensorGuard: без симулятора, чтобы проверить его правила по одному."""

    def __init__(self):
        from types import SimpleNamespace as NS
        self.cfg = AgentConfig(fault_wait=True)
        self.rules = Rules()
        self.health = NS(degraded=False, nominal=0.05)
        self.inv = None
        self._returning = False
        self._hazard_t = -1e9
        self._grace = (-1e9, None)
        self.hazards = []
        self.notes, self.relaxed, self.plans = [], [], []
        self.journal = NS(add=lambda t, kind, text, **kw: self.notes.append((t, kw.get('tag'), text)))
        self.belief = NS(relax=self.relaxed.append)
        self._request_plan = self.plans.append


def _feed(guard, t0, t1, battery, value=lambda t: 0.4, x=lambda t: 0.0):
    """Показания 5 раз в секунду с t0 по t1; возвращает, стоял ли робот в каждый момент."""
    from types import SimpleNamespace as NS
    held = []
    for k in range(int(round((t1 - t0) * 5))):
        t = t0 + 0.2 * k
        obs = NS(t=t, x=x(t), y=0.0, battery=battery(t))
        guard.reading(value(t), obs)
        held.append(guard.hold(obs))
    return held


def test_guard_waits_for_noise_to_end_and_no_longer():
    from did.sensorguard import SensorGuard
    a = _StubAgent()
    g = SensorGuard(a)
    rng = np.random.default_rng(0)
    calm = lambda t: 0.4 + rng.normal(0.0, 0.05)                    # noqa: E731
    loud = lambda t: 0.4 + rng.normal(0.0, 0.25)                    # noqa: E731
    battery = lambda t: 50.0 - 0.01 * t                             # noqa: E731
    assert not any(_feed(g, 0.0, 5.0, battery, calm))               # датчик исправен — не стоим
    a.health.degraded = True
    assert all(_feed(g, 5.0, 20.0, battery, loud))                  # шумит — стоим, сколько шумит
    held = _feed(g, 20.0, 40.0, battery, calm)       # оценка шума в SensorHealth ещё «плохая» — второй раз не ждём
    assert held[0] and not held[-1]
    assert 2.0 <= held.count(True) * 0.2 <= 6.0                     # шум спал — через несколько секунд поехали
    assert [n[1] for n in a.notes] == ['sensor_wait', 'sensor_wait_end'] and a.plans == ['sensor_wait']
    assert a.relaxed                                                # показаниям за время сбоя доверия меньше


def test_guard_gives_up_when_standing_drains_battery_or_takes_too_long():
    from did.sensorguard import SensorGuard
    a = _StubAgent()
    a.health.degraded = True
    g = SensorGuard(a)
    rng = np.random.default_rng(0)
    loud = lambda t: 0.4 + rng.normal(0.0, 0.25)                    # noqa: E731
    held = _feed(g, 0.0, 20.0, lambda t: 50.0 - 0.15 * t, loud)     # утечка: стоять дорого
    assert held[0] and held.count(True) * 0.2 <= 4.0
    assert 'теряю заряд' in a.notes[-1][2]
    b = _StubAgent()
    b.health.degraded = True
    g = SensorGuard(b)
    held = _feed(g, 0.0, 90.0, lambda t: 50.0 - 0.01 * t, loud)     # шум не кончается: ждём не дольше предела
    assert abs(held.index(False) * 0.2 - g.MAX_WAIT['noise']) < 1.0
    assert not any(held[held.index(False):])                        # и тот же шум второй раз не пережидаем


def test_guard_never_waits_on_the_way_home_or_inside_a_hazard():
    from did.sensorguard import SensorGuard
    for spoil in ('returning', 'hazard', 'late'):
        a = _StubAgent()
        a.health.degraded = True
        t0 = 0.0
        if spoil == 'returning':
            a._returning = True
        elif spoil == 'hazard':
            a._hazard_t = 0.0                                       # только что был штраф: сначала выехать
            a._grace = (4.0, object())
        else:
            t0 = a.rules.time_limit_s - 100.0                       # времени осталось мало
        g = SensorGuard(a)
        assert not any(_feed(g, t0, t0 + 3.0, lambda t: 50.0, lambda t: 0.4))


def test_guard_stops_waiting_on_return_decision_and_on_penalty():
    from did.sensorguard import SensorGuard
    for what in ('return', 'penalty'):
        a = _StubAgent()
        a.health.degraded = True
        g = SensorGuard(a)
        rng = np.random.default_rng(0)
        loud = lambda t: 0.4 + rng.normal(0.0, 0.25)                # noqa: E731
        assert all(_feed(g, 0.0, 5.0, lambda t: 50.0, loud))
        if what == 'return':
            a._returning = True                                     # заряда осталось на дорогу домой
        else:
            a._hazard_t = 5.0                                       # под роботом появилась опасная зона
        assert not any(_feed(g, 5.0, 12.0, lambda t: 50.0, loud))


def test_guard_sees_stuck_sensor_only_while_moving():
    from did.sensorguard import SensorGuard
    a = _StubAgent()
    g = SensorGuard(a)
    assert not any(_feed(g, 0.0, 4.0, lambda t: 50.0, lambda t: 0.31))                       # стоит: одинаковые показания — норма
    held = _feed(g, 4.0, 8.0, lambda t: 50.0, lambda t: 0.31, x=lambda t: 0.2 * (t - 4.0))   # едет, а показание то же
    assert held[-1] and a.notes[-1][1] == 'sensor_wait'
    held = _feed(g, 8.0, 9.0, lambda t: 50.0, lambda t: 0.5, x=lambda t: 0.8)
    assert not held[-1] and 'меняются' in a.notes[-1][2]


# --- разбор потерь ---------------------------------------------------------------------------------

def test_oracle_on_one_sample_is_there_and_back():
    arena = load_arena()
    rules = Rules()
    sc = Scenario(level='easy', seed=0, samples=[[-1.0, -0.5]])
    top = lb.oracle(sc, rules)
    assert top['n'] == 1 and abs(top['distance'] - 2.0) < 0.05
    assert abs(top['energy'] - (2.0 * rules.drain_per_m + rules.drain_idle_per_s * 2.0 / lb.V_CRUISE)) < 0.2
    assert abs(top['score'] - (10 + 20 + 0.1 * (60 - top['energy']))) < 1e-6
    # Дорогой грунт на прямой: оракул платит либо за грунт, либо за объезд, но не меньше прежнего.
    sc.soils = [Zone('A', 'circle', -1.5, -0.5, r=0.3, mult=4.0)]
    assert lb.oracle(sc, rules)['energy'] > top['energy'] + 0.3
    # Заряда на всё не хватает: оракул берёт сколько влезает.
    far = generate('hard', 3, arena)
    assert lb.oracle(far, rules, budget=12.0)['n'] < len(far.samples)
    assert lb.oracle(far, rules)['energy'] <= rules.battery_start


def test_loss_breakdown_adds_up_to_gap():
    from did.recorder import load_trace
    from did.runner import RUNS
    for rules in (None, 'science'):
        s = run_episode('hard', 3, 'adaptive', experiment='_test', arm='p1_loss', rules=rules)
        row = lb.analyze(load_trace(RUNS / s['file']))                    # внутри — проверка, что сумма сходится
        assert abs(sum(row['loss'].values()) - row['gap']) < 0.05
        assert row['ceiling'] >= row['score'] - 1.0
        assert set(row['loss']) <= {k for k, _ in lb.CAUSES} | {'beyond_oracle'}
        assert abs(sum(row['ledger'].values()) - (60.0 - row['battery_left'])) < 0.05
