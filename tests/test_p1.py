"""Проверки правок P1: спрямление пути, пережидание сбоя датчика по измеренной цене простоя, разбор потерь."""
import math
import sys
from pathlib import Path

import numpy as np

from did.agent import PRESETS, AgentConfig
from did.arena import load_arena
from did.config import BASE, Rules
from did.belief import SensorHealth
from did.nav import CostGraph, _crossed_cells, path_length, straighten
from did.runner import run_episode
from did.scenario import Scenario, Zone, generate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import loss_breakdown as lb     # noqa: E402
import p1_ablation as ab       # noqa: E402


# --- прежние агенты не тронуты --------------------------------------------------------------------

def test_new_behaviour_is_off_by_default():
    cfg = AgentConfig()
    assert not cfg.fault_wait and not cfg.straight_paths
    for name in ('adaptive', 'scientist', 'fixed', 'adaptive_fs', 'scientist_fs'):
        assert not PRESETS[name].fault_wait and not PRESETS[name].straight_paths
    for name in ('adaptive_v2', 'scientist_v2'):
        assert PRESETS[name].fault_wait and PRESETS[name].fault_wait_lost == 0
        assert not PRESETS[name].straight_paths       # спрямление в v2 не входит: выигрыша у него не показано
    assert PRESETS['scientist_v2'].science and not PRESETS['adaptive_v2'].science


# --- спрямление пути ------------------------------------------------------------------------------

def _touched_cells(arena, a, b):
    """Клетки, которых касается отрезок, независимым способом: точки через 0,2 мм со сдвигами на 1 мкм."""
    n = int(math.dist(a, b) / 0.0002) + 2
    k = np.linspace(0.0, 1.0, n)
    x, y = a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])
    cells = set()
    for dx in (-1e-6, 0.0, 1e-6):
        for dy in (-1e-6, 0.0, 1e-6):
            ix = np.floor((x + dx - arena.x0) / arena.res).astype(int)
            iy = np.floor((y + dy - arena.y0) / arena.res).astype(int)
            cells |= set(zip(iy.tolist(), ix.tolist()))
    return cells


def test_crossed_cells_include_corner_touches_and_lengths_add_up():
    arena = load_arena()
    c = lambda ix, iy: (arena.x0 + (ix + 0.5) * arena.res, arena.y0 + (iy + 0.5) * arena.res)      # noqa: E731
    iy, ix, length = _crossed_cells(arena, c(10, 10), c(11, 11))          # по диагонали: через общий угол клеток
    assert {(10, 10), (11, 11), (10, 11), (11, 10)} == set(zip(iy.tolist(), ix.tolist()))
    assert abs(length.sum() - arena.res * math.sqrt(2)) < 1e-9
    rng = np.random.default_rng(5)
    for _ in range(200):
        a = (arena.x0 + rng.uniform(0.3, 2.0), arena.y0 + rng.uniform(0.3, 2.0))
        b = (a[0] + rng.uniform(-0.25, 0.25), a[1] + rng.uniform(-0.25, 0.25))
        iy, ix, length = _crossed_cells(arena, a, b)
        assert abs(length.sum() - math.dist(a, b)) < 1e-9
        assert _touched_cells(arena, a, b) <= set(zip(iy.tolist(), ix.tolist()))


def test_straighten_is_shorter_and_never_touches_a_forbidden_cell():
    arena = load_arena()
    g = CostGraph(arena)
    rng = np.random.default_rng(42)
    ys, xs = np.nonzero(g.ok)
    pairs = [((0.025, 1.475), (-2.025, -0.225))]      # из ревью: прежняя проверка по точкам через 2,5 см пропускала угол
    for _ in range(120):
        a, b = rng.integers(len(xs), size=2)
        pairs.append((arena.g2w(xs[a], ys[a]), arena.g2w(xs[b], ys[b])))
    saved = []
    for start, goal in pairs:
        pts, _ = g.plan(start, goal)
        out = straighten(g, pts)
        assert out[0] == pts[0] and math.dist(out[-1], pts[-1]) < 1e-9
        assert path_length(out) <= path_length(pts) + 1e-9
        for p, q in zip(out, out[1:]):                # весь отрезок, а не только его концы
            assert all(g.ok[iy, ix] for iy, ix in _touched_cells(arena, p, q)), (start, goal, p, q)
        assert max(math.dist(p, q) for p, q in zip(out, out[1:])) <= 0.08       # шаг точек — как у пути по клеткам
        saved.append(1.0 - path_length(out) / max(path_length(pts), 1e-9))
    assert 0.005 < np.mean(saved) < 0.09          # путь по клеткам длиннее прямой не больше чем на 8%


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


# --- пережидание сбоя датчика: сторож отдельно от симулятора ------------------------------------------

class _StubAgent:
    """Ровно то, на что смотрит SensorGuard: без симулятора, чтобы проверить его правила по одному."""

    def __init__(self, idle=0.01):
        from types import SimpleNamespace as NS
        self.cfg = AgentConfig(fault_wait=True)
        self.rules = Rules(drain_idle_per_s=idle)       # что агент думает о цене простоя до своих замеров
        self.health = SensorHealth(0.05)
        self.inv = None
        self._returning = False
        self._hazard_t = -1e9
        self._grace = (-1e9, None)
        self.hazards = []
        self.need = 10.0                                # заряд, при котором пора домой
        self.notes, self.relaxed, self.plans = [], [], []
        self.journal = NS(add=lambda t, kind, text, **kw: self.notes.append((t, kw.get('tag'), text)))
        self.belief = NS(relax=self.relaxed.append)
        self._request_plan = self.plans.append

    def _return_need(self, obs):
        return self.need - 4.0, self.need

    def tags(self):
        return [n[1] for n in self.notes]


class _World:
    """Такты по 0,1 с: робот едет, пока сторож не велит стоять; батарея тратится по настоящим ценам мира."""

    def __init__(self, agent, guard, *, hz=5.0, idle=0.01, move=0.45, burst=1, delay=0.0, battery=50.0, seed=0):
        self.a, self.g = agent, guard
        self.hz, self.idle, self.move, self.burst, self.delay = hz, idle, move, burst, delay
        self.t, self.x, self.battery, self.held = 0.0, 0.0, battery, False
        self.rng = np.random.default_rng(seed)
        self.queue, self.k = [], 0
        self.driving = True                              # False — робот стоит и без сторожа (например, на пике)

    def run(self, until, sigma=0.05, value=None, moving=True):
        """До момента until датчик шумит с разбросом sigma (или выдаёт value(t)). Возвращает, когда робот стоял."""
        from types import SimpleNamespace as NS
        held = []
        while self.t < until - 1e-9:
            self.t = round(self.t + 0.1, 6)
            go = moving and not self.held
            self.x += 0.02 if go else 0.0
            self.battery -= (self.move if go else self.idle) * 0.1
            obs = NS(t=self.t, x=self.x, y=0.0, v=0.2 if go else 0.0, w=0.0, battery=self.battery, events=[],
                     sensor_age=0.0)
            self.g.observe(obs)
            while self.k / self.hz <= self.t + 1e-9:     # измерения идут ровно, доставка — пачками и с задержкой
                tm = self.k / self.hz
                z = value(tm) if value else float(np.clip(0.4 + self.rng.normal(0.0, sigma), 0.0, 1.0))
                self.queue.append((tm, z))
                self.k += 1
            ready = [q for q in self.queue if q[0] + self.delay <= self.t + 1e-9]
            if len(ready) >= self.burst:
                for tm, z in ready:
                    self.a.health.update(z)
                    self.g.reading(z, NS(t=self.t, x=self.x, y=0.0, sensor_age=self.t - tm))
                self.queue = [q for q in self.queue if q not in ready]
            self.held = self.g.hold(obs)
            held.append(self.held)
        return held


def _guard(idle=0.01, lost=0, **world):
    from did.sensorguard import SensorGuard
    a = _StubAgent(idle=world.pop('believed_idle', idle))
    a.cfg = AgentConfig(fault_wait=True, fault_wait_lost=lost)
    g = SensorGuard(a)
    return a, g, _World(a, g, idle=idle, **world)


RATES = [dict(hz=2.0), dict(hz=5.0), dict(hz=10.0), dict(hz=5.0, burst=5, delay=0.6), dict(hz=2.0, burst=3, delay=1.0)]


def test_guard_waits_for_noise_to_end_at_any_reading_rate():
    """Находка 3: конец сбоя виден по здоровью датчика и по времени, а не по числу показаний за 4 секунды."""
    for rate in RATES:
        a, g, w = _guard(**rate)
        assert not any(w.run(20.0)), rate                       # датчик исправен — не стоим
        held = w.run(50.0, sigma=0.25)                          # 30 секунд шума
        assert held[-1] and held.count(True) * 0.1 >= 15.0, rate         # заметил и стоит до конца сбоя
        held = w.run(80.0)                                      # шум кончился
        gone = held.count(True) * 0.1
        assert not held[-1] and gone <= 14.0, (rate, gone)       # поехал через секунды, а не по исчерпании бюджета
        assert a.tags().count('sensor_wait') == 1 and a.tags().count('sensor_wait_end') == 1, (rate, a.notes)
        assert g.waited_s <= 45.0 and a.relaxed and a.plans == ['sensor_wait']
        assert g.time_budget == 120.0 and g.charge_budget == 3.0


def test_guard_total_wait_is_bounded_however_long_the_fault_lasts():
    """Находка 2: бесконечный шум. Раньше — шесть ожиданий на 250 с; теперь один бюджет на весь прогон."""
    for rate in RATES:
        a, g, w = _guard(**rate)
        w.run(10.0)
        held = w.run(500.0, sigma=0.25)
        assert g.waited_s <= g.time_budget + 0.2 and held.count(True) * 0.1 <= g.time_budget + 0.5, rate
        assert a.tags().count('sensor_wait') == 1, (rate, a.notes)       # и тот же шум второй раз не пережидаем
        assert not any(held[-1000:])


def test_guard_waits_again_only_after_stable_recovery():
    """Находка 2: краткое затихание оценки шума — не новый сбой; новый — после RECOVER_S исправных секунд."""
    a, g, w = _guard()
    w.run(10.0)
    w.run(40.0, sigma=0.25)
    w.run(43.0)                                                 # затишье на три секунды…
    held = w.run(70.0, sigma=0.25)                              # …и тот же шум снова
    assert a.tags().count('sensor_wait') == 1 and not any(held[60:])
    w.run(100.0)                                                # датчик исправен полминуты
    held = w.run(125.0, sigma=0.25)                             # это уже другой сбой
    assert a.tags().count('sensor_wait') == 2 and held[-1]
    assert g.waited_s <= g.time_budget


def test_guard_charge_budget_makes_expensive_idle_a_short_wait():
    """Находка 4: простой в 10 раз дороже — ждём, пока не потрачен бюджет заряда, а не весь сбой."""
    a, g, w = _guard(idle=0.10)
    w.run(10.0)
    b0 = w.battery
    held = w.run(200.0, sigma=0.25)
    stood = held.count(True) * 0.1
    assert 20.0 <= stood <= 31.0 and 'ед. заряда' in a.notes[-1][2]
    assert g.spent <= g.charge_budget + 0.05
    assert a.tags().count('sensor_wait') == 1
    assert b0 - w.battery <= g.charge_budget + 0.45 * (190.0 - stood) + 1.0


def test_guard_does_not_wait_when_standing_costs_as_much_as_blind_driving():
    a, g, w = _guard(idle=0.30)
    w.run(10.0)
    assert not any(w.run(60.0, sigma=0.25))
    assert a.tags().count('sensor_wait') == 0 and a.tags().count('sensor_wait_skip') == 1
    assert 'невыгодно' in a.notes[-1][2]


def test_guard_measures_idle_cost_itself_when_rules_are_wrong():
    """Агент думает, что простой стоит 0,01 ед./с, а судья берёт 0,10: цену показывает батарея."""
    a, g, w = _guard(idle=0.10, believed_idle=0.01)
    w.run(10.0)
    held = w.run(60.0, sigma=0.25)
    assert held.count(True) * 0.1 <= g.MIN_WAIT + 0.5 and 'ждать дорого' in a.notes[-1][2]
    w.run(70.0, moving=False)                                   # ещё постоял (на пике, после сбора) — замеров хватает
    assert abs(g.idle_rate() - 0.10) < 0.01
    assert abs(g.trip_idle(3.0) - 0.10 * 3.0 / g.TRIP_SPEED) < 0.25
    # и наоборот: правила обещают дорогой простой, а он дешёвый
    a, g, w = _guard(idle=0.01, believed_idle=0.10)
    w.run(10.0, moving=False)
    assert abs(g.idle_rate() - 0.01) < 0.005


def test_guard_gives_up_on_a_leak_while_standing():
    a, g, w = _guard()
    w.run(10.0)
    w.run(13.0, sigma=0.25)
    w.idle = 0.16                                               # стоя заряд уходит в 16 раз быстрее обычного
    held = w.run(60.0, sigma=0.25)
    assert held.count(True) * 0.1 <= 6.0 and 'ждать дорого' in a.notes[-1][2]
    assert g.idle_rate() < 0.05                                 # разовая утечка не становится «ценой простоя»


def test_return_home_has_priority_over_waiting():
    """Находка 4: ожидание не начинается, если после него не хватит на дорогу, и кончается по решению о возврате."""
    a, g, w = _guard(idle=0.10, battery=18.0)
    a.need = 13.0
    w.run(10.0)                                                 # осталось 13,5 ед. при пороге 13
    assert not any(w.run(40.0, sigma=0.25))
    assert 'не хватит на дорогу домой' in a.notes[-1][2]
    a, g, w = _guard()
    w.run(10.0)
    assert w.run(20.0, sigma=0.25)[-1]
    a._returning = True                                         # агент решил возвращаться
    assert not any(w.run(30.0, sigma=0.25))
    a, g, w = _guard()
    w.run(10.0)
    assert w.run(20.0, sigma=0.25)[-1]
    a._hazard_t = w.t                                           # под роботом появилась опасная зона
    assert not any(w.run(30.0, sigma=0.25))


def test_guard_never_waits_on_the_way_home_or_inside_a_hazard():
    for spoil in ('returning', 'hazard', 'late'):
        a, g, w = _guard()
        if spoil == 'returning':
            a._returning = True
        elif spoil == 'hazard':
            a._grace = (1e9, object())                          # только что был штраф: сначала выехать
        else:
            w.t = a.rules.time_limit_s - 140.0                  # времени осталось мало
            w.k = int(w.t * w.hz)
        assert not any(w.run(w.t + 30.0, sigma=0.25)), spoil


def test_three_lost_candidates_cost_a_short_probe_not_a_long_wait():
    """Находка 5: три промаха подряд на исправном датчике — постоять 2,5 с, и не больше двух раз за прогон."""
    a, g, w = _guard()
    w.run(10.0)
    for dt in (1.0, 2.0, 3.0):
        g.subgoal_ended('candidate_lost', w.t - 5.0 + dt)
    assert not any(w.run(20.0)) and g.n_probes == 0             # в v2 проверка выключена: робот не останавливается
    a, g, w = _guard(lost=3)
    w.run(10.0)
    stood = 0.0
    for k in range(4):
        for dt in (1.0, 2.0, 3.0):
            g.subgoal_ended('candidate_lost', w.t + dt)
        w.t += 3.0
        w.k = int(w.t * w.hz)
        stood += w.run(w.t + 20.0).count(True) * 0.1
    assert g.n_probes == g.MAX_PROBES and a.tags().count('sensor_probe') == g.MAX_PROBES
    assert stood <= g.MAX_PROBES * (g.PROBE_S + 0.3) and g.n_waits == 0
    assert not a.relaxed                                        # датчик исправен: карту образцов не трогаем
    # если стоя виден шум, проверка переходит в ожидание
    a, g, w = _guard(lost=3)
    w.run(10.0)
    for dt in (1.0, 2.0, 3.0):
        g.subgoal_ended('candidate_lost', w.t - 5.0 + dt)
    a.cfg = AgentConfig(fault_wait=True, fault_wait_lost=3, sensor_health=False)   # на ходу шум не оценивается
    held = w.run(30.0, sigma=0.25)
    assert held[-1] and a.tags().count('sensor_probe') == 1 and a.tags().count('sensor_wait') == 1


def test_guard_sees_stuck_sensor_only_while_moving():
    for rate in (dict(hz=2.0), dict(hz=5.0), dict(hz=10.0)):
        a, g, w = _guard(**rate)
        assert not any(w.run(6.0, value=lambda t: 0.31, moving=False)), rate      # стоит: одинаковые показания — норма
        held = w.run(12.0, value=lambda t: 0.31)                                 # едет, а показание то же
        assert held[-1] and a.notes[-1][1] == 'sensor_wait', rate
        held = w.run(16.0)
        assert not held[-1] and 'меняются' in a.notes[-1][2], rate


# --- пережидание сбоя датчика: агент целиком ---------------------------------------------------------

def _noisy_scenario(duration=32.0):
    """Три образца и шумный датчик с 8-й секунды: без зон и грунтов, чтобы виден был один механизм."""
    return Scenario(level='hard', seed=1, samples=[[-1.0, -0.5], [0.5, -1.5], [0.5, 1.5]],
                    events=[{'t': 8.0, 'type': 'sensor_fault', 'duration': duration, 'sigma': 0.25}])


def _waits(tr):
    """Сколько раз и сколько секунд робот стоял в ожидании датчика (по записи прогона)."""
    t, mode = np.array(tr['track']['t']), np.array(tr['track']['mode'])
    wait = tr['modes'].index('wait') if 'wait' in tr['modes'] else -1
    tags = [(e.get('data') or {}).get('tag') for e in tr['journal']]
    return tags.count('sensor_wait'), float(np.diff(t)[mode[1:] == wait].sum())


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


def test_agent_endless_fault_and_slow_sensor():
    """Находки 2 и 3 на агенте целиком: бесконечный шум и датчик 2 показания в секунду."""
    from did.recorder import load_trace
    from did.runner import RUNS
    s = run_episode('hard', 1, 'adaptive_v2', experiment='_test', arm='p1_endless', scenario=_noisy_scenario(10000.0))
    n, stood = _waits(load_trace(RUNS / s['file']))
    assert n == 1 and stood <= 121.0 and s['metrics']['returned']        # было: шесть ожиданий, 250 с
    s = run_episode('hard', 1, 'adaptive_v2', experiment='_test', arm='p1_hz2', scenario=_noisy_scenario(),
                    rules={'sensor_hz': 2.0})
    n, stood = _waits(load_trace(RUNS / s['file']))
    assert n == 1 and stood <= 40.0 and s['metrics']['returned']         # было: ждал все 60 с при исправном датчике


def test_trip_cost_counts_idle_drain_only_for_v2():
    """Находка 4: дорога домой у v2 стоит путь плюс расход «за включённость» за время пути."""
    from did.agent import Agent, make_config
    arena = load_arena()
    cost = {}
    for name in ('adaptive', 'adaptive_v2'):
        for idle in (0.01, 0.10):
            bot = Agent(arena, make_config(name), n_samples=3, rules=Rules(drain_idle_per_s=idle))
            cost[name, idle] = bot._home_cost(0.5, 1.0)
    path = cost['adaptive', 0.01] / Rules().drain_per_m
    assert cost['adaptive', 0.01] == cost['adaptive', 0.10]              # прежний агент: только метры
    assert cost['adaptive_v2', 0.10] - cost['adaptive', 0.10] > 0.09 * path / 0.20
    assert cost['adaptive_v2', 0.01] == cost['adaptive', 0.01]           # при обычной цене простоя маршруты v2 прежние


def test_agent_returns_when_idle_is_expensive():
    """Находка 4, пример из ревью: сценарий 5005 при простое в 5 раз дороже — прежний v2 не вернулся."""
    s = run_episode('hard', 5005, 'adaptive_v2', experiment='_test', arm='p1_idle5', rules={'drain_idle_per_s': 0.05},
                    save=False)
    assert s['metrics']['returned'] and s['metrics']['score'] >= 75.0


def test_fault_duration_changes_nothing_else_in_the_scenario():
    arena = load_arena()
    a, b = generate('hard', 7, arena), generate('hard', 7, arena, fault_duration=(90.0, 120.0))
    da = next(e for e in a.events if e['type'] == 'sensor_fault')
    db = next(e for e in b.events if e['type'] == 'sensor_fault')
    assert 25.0 <= da['duration'] <= 40.0 and 90.0 <= db['duration'] <= 120.0 and da['t'] == db['t']
    assert a.samples == b.samples and a.hazards == b.hazards and a.soils == b.soils


def test_ablation_cache_is_tied_to_settings_and_code(tmp_path):
    """Находка 7: запись варианта не подходит, если изменились настройки, условие или код."""
    import json
    key = ab.variant_key('adaptive', '', {'fault_wait': True}, 'base')
    path = tmp_path / 'v.json'
    path.write_text(json.dumps({'key': key, 'code': 'abc', 'runs': {'hard-1': {'score': 1.0}}}))
    assert ab.load_cache(path, key, 'abc') == {'hard-1': {'score': 1.0}}
    assert ab.load_cache(path, ab.variant_key('adaptive', '', {'fault_wait': False}, 'base'), 'abc') == {}
    assert ab.load_cache(path, ab.variant_key('adaptive', '', {'fault_wait': True}, 'idle5'), 'abc') == {}
    assert ab.load_cache(path, ab.variant_key('adaptive', 'science', {'fault_wait': True}, 'base'), 'abc') == {}
    assert ab.load_cache(path, key, 'другой код') == {}
    assert ab.load_cache(path, key, 'другой код', reuse=True)            # только по явной просьбе
    path.write_text(json.dumps({'hard-1': {'score': 1.0}}))               # запись прежнего вида, без ключа
    assert ab.load_cache(path, key, 'abc') == {}
    assert len(ab.code_version()) == 12


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


def test_one_approach_explains_one_missed_sample():
    """Находка 6: подъезд относится к одному образцу — ближайшему из ещё не собранных."""
    samples = [(0.0, 0.0), (0.5, 0.0), (3.0, 3.0)]
    attempt = (10.0, 20.0, 'candidate_lost', 0.2, 0.0)             # в 0,6 м от обоих образцов, ближе к первому
    assert dict(lb.assign_attempts([attempt], samples, {})) == {0: [attempt]}
    assert dict(lb.assign_attempts([attempt], samples, {0: 5.0})) == {1: [attempt]}      # первый уже собран
    assert dict(lb.assign_attempts([attempt], samples, {0: 15.0})) == {0: [attempt]}     # собран позже начала подъезда
    assert dict(lb.assign_attempts([(10.0, 20.0, 'candidate_lost', 1.5, 1.5)], samples, {})) == {}


def test_fault_is_blamed_only_when_it_overlaps_the_approach():
    faults = [('sensor_noise', 50.0, 80.0)]
    cause = lambda t0, t1, how='candidate_lost': lb.attempt_cause((t0, t1, how, 0.0, 0.0), faults)      # noqa: E731
    assert cause(60.0, 70.0) == 'sample_fault' and cause(40.0, 55.0) == 'sample_fault'
    assert cause(82.0, 86.0) == 'sample_unclear'                   # кончился вскоре после сбоя: карта могла быть испорчена
    assert cause(40.0, 49.5) == 'sample_lost' and cause(95.0, 99.0) == 'sample_lost'
    assert cause(20.0, 30.0, 'sensor_degraded') == 'sample_unclear'      # тревога о датчике без настоящего сбоя
    assert cause(60.0, 70.0, 'hazard') == 'sample_hazard' and cause(60.0, 70.0, 'return') == 'sample_cut'
    assert 'sample_unclear' in dict(lb.CAUSES)
