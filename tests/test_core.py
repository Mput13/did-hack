"""Проверки ядра стенда: сценарии, судья, модель мира агента, прогон целиком."""
import json
import math

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.belief import ChangeDetector, SampleBelief, SensorHealth, SoilModel
from did.config import BASE, LEVELS, ROBOT_RADIUS, Rules
from did.fastsim import FastSim
from did.judge import Judge
from did.nav import CostGraph, path_length
from did.route import survey_route
from did.runner import run_episode
from did.scenario import Scenario, Zone, generate


@pytest.fixture(scope='module')
def arena():
    return load_arena()


# --- арена и сценарии ----------------------------------------------------------------------------

def test_arena_matches_task(arena):
    assert arena.res == pytest.approx(0.05)
    assert arena.is_free(*BASE, margin=ROBOT_RADIUS)
    for x in (-1.1, 0.0, 1.1):          # девять столбов из условия
        for y in (-1.1, 0.0, 1.1):
            assert not arena.is_free(x, y)
    assert 19.0 < arena.free.sum() * arena.res ** 2 < 21.0


@pytest.mark.parametrize('level', list(LEVELS))
def test_scenario_follows_level_table(arena, level):
    for seed in range(1, 40):
        sc = generate(level, seed, arena)
        assert len(sc.samples) == LEVELS[level]['samples']
        assert len(sc.soils) == LEVELS[level]['soils']
        assert len(sc.hazards) == LEVELS[level]['hazards']
        assert {e['type'] for e in sc.events} == ({'soil_change', 'new_hazard', 'sensor_fault'}
                                                  if level == 'hard' else set())
        for p in sc.samples:
            assert arena.clearance(*p) >= 0.2 and math.dist(p, BASE) > 1.0
        for z in sc.hazards:
            assert all(math.dist((z.x, z.y), p) > z.r + 0.3 for p in sc.samples)   # образец достижим без штрафа


def test_scenario_is_reproducible_and_serializable(arena):
    a, b = generate('hard', 7, arena), generate('hard', 7, arena)
    assert a.to_dict() == b.to_dict()
    assert generate('hard', 8, arena).to_dict() != a.to_dict()
    assert Scenario.from_dict(json.loads(json.dumps(a.to_dict()))).to_dict() == a.to_dict()


# --- судья ---------------------------------------------------------------------------------------

def _scenario(**kw):
    base = dict(level='easy', seed=0, samples=[[-1.0, -0.5]], soils=[], hazards=[], events=[])
    base.update(kw)
    return Scenario(**base)


def test_battery_drains_by_distance_and_soil(arena):
    r = Rules()
    soil = Zone('A', 'rect', -1.0, -0.5, w=0.6, h=0.6, mult=3.0)
    j = Judge(_scenario(soils=[soil]), arena, r)
    j.step(0.0, -2.0, -0.5)
    j.step(1.0, -1.8, -0.5)                              # 0,2 м по обычному полу
    assert r.battery_start - j.battery == pytest.approx(0.2 * r.drain_per_m + r.drain_idle_per_s)
    before = j.battery
    j.step(2.0, -1.1, -0.5)
    j.step(3.0, -0.9, -0.5)                              # 0,2 м по грунту ×3
    assert before - j.battery == pytest.approx(0.7 * r.drain_per_m * 1.0 + 0.2 * r.drain_per_m * 3.0
                                               + 2 * r.drain_idle_per_s, rel=0.02)


def test_collect_finish_and_score(arena):
    r = Rules()
    j = Judge(_scenario(), arena, r)
    j.step(0.0, -2.0, -0.5)
    assert j.collect(-2.0, -0.5)[0] is False             # до образца 1 м
    assert j.pop_events()[0]['type'] == 'false_collect'
    j.step(5.0, -1.2, -0.5)
    assert j.collect(-1.2, -0.5)[0] is True              # 0,2 м — ближе 0,30
    assert j.pop_events()[0]['type'] == 'sample_collected'
    assert j.read_sensor(-1.2, -0.5) <= 0.3              # несобранных образцов не осталось
    j.step(10.0, -2.0, -0.5)
    assert j.finish(-2.0, -0.5)[0] is True
    s = j.score()
    assert s['returned'] and s['samples_collected'] == 1 and s['false_collects'] == 1
    assert s['score'] == pytest.approx(r.pts_sample + r.pts_false_collect + r.pts_return
                                       + r.pts_battery_left * j.battery, abs=0.01)


def test_sensor_is_proximity_without_direction(arena):
    j = Judge(_scenario(), arena, Rules(sensor_sigma=0.0))
    assert j.read_sensor(-1.0, -0.5) == pytest.approx(1.0)
    assert j.read_sensor(-2.0, -0.5) == pytest.approx(0.5)       # 1 м из 2 м дальности
    assert j.read_sensor(0.0, -0.5) == pytest.approx(j.read_sensor(-2.0, -0.5))   # слева и справа одинаково
    assert j.read_sensor(-1.0, 1.9) == pytest.approx(0.0)


def test_hidden_events_change_world_silently(arena):
    hazard = Zone('X1', 'circle', -1.5, -0.5, r=0.3)
    sc = _scenario(events=[{'t': 5.0, 'type': 'new_hazard', 'zone': hazard},
                           {'t': 6.0, 'type': 'sensor_fault', 'duration': 10.0, 'sigma': 0.3}])
    j = Judge(sc, arena)
    j.step(0.0, -2.0, -0.5)
    j.step(4.0, -1.5, -0.5)
    assert j.pop_events() == []                           # зоны ещё нет
    j.step(5.5, -1.5, -0.5)
    events = j.pop_events()
    assert [e['type'] for e in events] == ['hazard_hit']  # само появление зоны не объявляется
    j.step(6.5, -1.9, -0.5)
    assert j.sensor_sigma == pytest.approx(0.3)
    j.step(17.0, -1.9, -0.5)
    assert j.sensor_sigma == pytest.approx(Rules().sensor_sigma)
    assert [w['type'] for w in j.world_log] == ['new_hazard', 'sensor_fault', 'sensor_recovered']


def test_run_ends_when_battery_is_empty(arena):
    j = Judge(_scenario(), arena, Rules(battery_start=1.0))
    j.step(0.0, -2.0, -0.5)
    j.step(5.0, -1.0, -0.5)
    assert j.done and j.reason == 'battery' and not j.returned


# --- навигация -----------------------------------------------------------------------------------

def test_path_avoids_walls_and_expensive_ground(arena):
    g = CostGraph(arena)
    pts, cost = g.plan(BASE, (1.6, 0.5))
    assert pts and min(arena.clearance(*p) for p in pts) >= 0.17 - 1e-6
    straight = path_length(pts)
    mx, my = pts[len(pts) // 2]                                   # дорогой участок прямо на этом пути
    mult = np.ones(arena.free.shape)
    X, Y = arena.cell_centers()
    zone = (np.abs(X - mx) < 0.35) & (np.abs(Y - my) < 0.35)
    mult[zone] = 4.0
    g.set_cost(mult)
    detour, detour_cost = g.plan(BASE, (1.6, 0.5))
    in_zone = sum(1 for x, y in detour if abs(x - mx) < 0.35 and abs(y - my) < 0.35)
    assert in_zone == 0 and path_length(detour) >= straight - 1e-9 and detour_cost < 4.0 * cost


def test_survey_route_covers_arena(arena):
    route = survey_route(arena)
    assert 8 <= len(route) <= 25
    fy, fx = np.nonzero(arena.clear >= 0.2)
    pts = np.array(route + [BASE])
    for j, i in list(zip(fy, fx))[::7]:
        x, y = arena.g2w(i, j)
        assert np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() <= 0.95


# --- модель мира агента --------------------------------------------------------------------------

def test_belief_finds_sample_from_scalar_readings(arena):
    truth = (0.55, 0.6)
    b = SampleBelief(arena, n_expected=1, sensor_range=2.0)
    rng = np.random.default_rng(1)
    for x, y in [(-2.0, -0.5), (-1.6, -0.5), (-1.2, -0.3), (-0.6, -0.5), (-0.5, 0.2), (0.0, 0.55)]:
        for _ in range(4):
            z = float(np.clip(1.0 - math.dist((x, y), truth) / 2.0 + rng.normal(0, 0.05), 0, 1))
            b.update(x, y, z, 0.05)
    best = b.candidates(min_mass=0.3)[0]
    assert math.dist((best['x'], best['y']), truth) < 0.3
    assert b.total() == pytest.approx(1.0, abs=0.05)             # образец один — и вероятность одна


def test_belief_after_collect_drops_explained_rings(arena):
    b = SampleBelief(arena, n_expected=2, sensor_range=2.0)
    first, second = (-1.0, -0.5), (1.5, 0.5)
    for x in np.linspace(-2.0, -1.1, 10):
        z = 1.0 - min(math.dist((x, -0.5), first), math.dist((x, -0.5), second)) / 2.0
        b.update(float(x), -0.5, z, 0.05)
    b.collected(-1.1, -0.5)
    assert b.left == 1 and b.prob_within(-1.0, -0.5, 0.3) < 0.05
    assert all(math.dist((c['x'], c['y']), first) > 0.8 for c in b.candidates(min_mass=0.3))


def test_soil_model_learns_multiplier_and_detector_fires_on_change(arena):
    soil = SoilModel(arena, drain_per_m=2.5, idle_per_s=0.01)
    det = ChangeDetector()
    step, alarms = 0.06, []
    for k in range(20):                                           # 1,2 м по грунту ×3
        x0 = -0.5 + k * step
        ratio, pred, conf = soil.observe(x0, 0.5, x0 + step, 0.5, 2.5 * 3.0 * step + 0.01 * 0.3, 0.3)
        alarms.append(det.update(ratio, pred, conf, step))
    assert soil.at(0.1, 0.5)[0] == pytest.approx(3.0, rel=0.15)
    assert not any(alarms)                                        # открытие зоны — не «изменение»
    assert soil.zones()[0]['mult'] == pytest.approx(3.0, rel=0.1)
    for k in range(20):                                           # тот же участок стал обычным
        x0 = -0.5 + k * step
        ratio, pred, conf = soil.observe(x0, 0.5, x0 + step, 0.5, 2.5 * step + 0.01 * 0.3, 0.3)
        verdict = det.update(ratio, pred, conf, step)
        if verdict:
            break
    assert verdict == 'дешевле' and k * step < 0.5                # замечено меньше чем за полметра


def test_sensor_health_detects_noise_growth():
    h, rng = SensorHealth(0.05), np.random.default_rng(3)
    states = [h.update(0.5 + rng.normal(0, 0.05)) for _ in range(60)]
    assert 'degraded' not in states
    states = [h.update(float(np.clip(0.5 + rng.normal(0, 0.25), 0, 1))) for _ in range(40)]
    assert 'degraded' in states and states.index('degraded') < 25      # не дольше 5 с при 5 Гц
    states = [h.update(0.5 + rng.normal(0, 0.05)) for _ in range(60)]
    assert 'recovered' in states


# --- прогон целиком ------------------------------------------------------------------------------

@pytest.mark.parametrize('agent', ['fixed', 'adaptive'])
def test_episode_runs_and_returns(agent):
    s = run_episode('easy', 1, agent, save=False)
    m = s['metrics']
    assert m['returned'] and m['reason'] == 'finish'
    assert m['samples_collected'] >= 2 and m['collisions'] == 0


def test_episode_is_reproducible():
    a = run_episode('hard', 4, 'adaptive', save=False)['metrics']
    b = run_episode('hard', 4, 'adaptive', save=False)['metrics']
    assert a == b


def test_all_presets_survive_hard():
    for name in PRESETS:
        if PRESETS[name].planner == 'llm':
            continue
        m = run_episode('hard', 2, name, save=False)['metrics']
        assert m['reason'] in ('finish', 'battery', 'timeout')


def test_agent_never_drives_velocity_from_planner(arena):
    """Планировщик верхнего уровня отдаёт только подцели: команда скорости идёт из исполнителя."""
    sc = generate('easy', 1, arena)
    world = FastSim(arena, sc, seed=1)
    seen = []

    class Spy:
        def plan(self, state):
            seen.append(set(state))
            return {'reasoning': 'тест', 'hypotheses': [], 'subgoals': [{'type': 'return_base'}],
                    'source': 'heuristic', 'exchanges': [], 'error': None}

    bot = Agent(arena, make_config('adaptive'), n_samples=3, planner=Spy())
    while not world.done and world.t < 30:
        bot.tick(world.observe(), world)
        world.advance()
    assert seen and all('v' not in s and 'cmd_vel' not in s for s in seen)
    assert world.judge.returned
