"""Проверки выбора точки измерения по ожидаемой пользе и базовых стратегий поиска."""
import json
import math
import time
from dataclasses import replace

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.baselines import BASELINES, spiral_route
from did.belief import SampleBelief
from did.config import BASE, Rules
from did.explore import entropy_bits, expected_gain, path_stops, rank_points, sample_worlds
from did.fastsim import FastSim
from did.nav import CostGraph
from did.planner import HeuristicPlanner
from did.recorder import Recorder
from did.runner import run_episode
from did.scenario import generate

AXIS = math.atan2(0.0 - BASE[1], 0.0 - BASE[0])      # направление от базы к середине арены


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _mean_gain(belief, x, y, seeds=4, **kw):
    return float(np.mean([expected_gain(belief, x, y, 0.05, np.random.default_rng(s), draws=64, **kw)
                          for s in range(seeds)]))


def _from_base(dist, turn_deg=0.0):
    a = AXIS + math.radians(turn_deg)
    return BASE[0] + dist * math.cos(a), BASE[1] + dist * math.sin(a)


# --- пробное обновление карты --------------------------------------------------------------------

def test_trial_update_matches_real_update_and_leaves_belief_alone(arena):
    b = SampleBelief(arena, n_expected=3, sensor_range=2.0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        x, y = rng.uniform(-2.0, 2.0, 2)
        z = float(np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0))        # в том числе упёршиеся в 0 и в 1
        before, n_log, n_upd = b.p.copy(), len(b._log), b.updates
        tried = b.trial(x, y, z, 0.05)
        assert np.array_equal(b.p, before) and len(b._log) == n_log and b.updates == n_upd
        b.update(x, y, z, 0.05)
        assert np.allclose(tried[0], b.p, rtol=1e-9, atol=1e-12)
    many = b.trial(0.3, 0.2, np.array([0.0, 0.4, 1.0]), 0.05)        # три варианта показания разом
    assert many.shape == (3, b.n)
    assert np.allclose(many[1], b.trial(0.3, 0.2, 0.4, 0.05)[0])


def test_worlds_follow_the_map(arena):
    b = SampleBelief(arena, n_expected=2, sensor_range=2.0)
    for _ in range(3):
        b.update(*BASE, 0.5, 0.05)                                   # кольцо радиусом 1 м вокруг базы
    worlds = sample_worlds(b, np.random.default_rng(0), draws=400)
    assert worlds.shape == (400, 2)                                  # образцов в гипотезе столько, сколько осталось
    assert all(len(set(w)) == 2 for w in worlds)
    d = np.hypot(b.cx[worlds] - BASE[0], b.cy[worlds] - BASE[1])
    assert (d.min(axis=1) < 0.7).mean() < 0.02                       # внутри кольца образцов почти не бывает
    assert (np.abs(d.min(axis=1) - 1.0) < 0.25).mean() > 0.7         # ближайший обычно на кольце


# --- ожидаемая польза измерения ------------------------------------------------------------------

def test_gain_is_nonnegative_and_larger_where_map_is_uncertain(arena):
    b = SampleBelief(arena, n_expected=3, sensor_range=2.0)
    assert expected_gain(b, 0.5, 0.5, 0.05, np.random.default_rng(0)) > 0.0
    for x in (-2.0, -1.6, -1.2):                                     # слева робот уже слушал: там пусто
        for y in (-1.0, -0.5, 0.0, 0.5):
            b.update(x, y, 0.0, 0.05)
    known, unknown = _mean_gain(b, -1.6, -0.5), _mean_gain(b, 1.2, 0.5)
    assert known >= 0.0 and unknown > 3.0 * max(known, 0.05)
    rng = np.random.default_rng(1)
    for _ in range(20):                                              # нигде не отрицательна
        x, y = rng.uniform(-2.2, 2.2, 2)
        assert expected_gain(b, x, y, 0.05, rng) >= 0.0
        assert expected_gain(b, x, y, 0.05, rng, origin=(-1.6, -0.5)) >= 0.0
    b.left = 0                                                       # собирать больше нечего — и узнавать нечего
    assert expected_gain(b, 1.2, 0.5, 0.05, rng) == 0.0


def test_gain_accumulates_along_the_road(arena):
    b = SampleBelief(arena, n_expected=3, sensor_range=2.0)
    b.update(*BASE, 0.35, 0.05)
    x, y = _from_base(1.5)
    assert _mean_gain(b, x, y, origin=BASE) > 1.5 * _mean_gain(b, x, y)
    stops, piece = path_stops([BASE, (x, y)])
    assert len(stops) == 4 and piece == pytest.approx(1.5 / 4)
    assert tuple(stops[-1]) == pytest.approx((x, y))


def test_point_beside_ring_beats_point_on_axis(arena):
    """Игрушечный пример: образец один, после одного измерения на базе он где-то на дуге вокруг неё.

    Дуга симметрична относительно прямой «база — середина арены». Измерение на этой прямой не
    отличает образец выше оси от образца ниже неё, измерение сбоку — отличает.
    """
    radius = 1.3
    b = SampleBelief(arena, n_expected=1, sensor_range=2.0)
    b.update(*BASE, 1.0 - radius / 2.0, 0.05)
    ring = np.abs(np.hypot(b.cx - BASE[0], b.cy - BASE[1]) - radius) <= 0.25
    b.p[~ring] = 1e-7                                                # образец один, и он на кольце
    b._normalize()
    assert b.p[ring].sum() == pytest.approx(1.0, abs=0.01)

    on_axis = _mean_gain(b, *_from_base(0.6))
    centre = _mean_gain(b, *BASE)                                    # повтор измерения в центре кольца
    assert centre < 0.5 * on_axis
    for turn in (60, -60, 90, -90):
        assert _mean_gain(b, *_from_base(0.6, turn)) > 1.2 * on_axis
        assert _mean_gain(b, *_from_base(0.6, turn), origin=BASE) > 1.1 * _mean_gain(b, *_from_base(0.6), origin=BASE)

    # Прежняя прикидка этого не видит: массы вокруг точки на оси не меньше, чем вокруг точки сбоку.
    def mass(x, y):
        return float(b.p[np.hypot(b.cx - x, b.cy - y) <= 0.8].sum())
    assert mass(*_from_base(0.6)) >= mass(*_from_base(0.6, 90)) - 0.05


def test_rank_points_orders_by_gain_per_charge_and_is_fast(arena):
    b = SampleBelief(arena, n_expected=5, sensor_range=2.0)
    b.update(*BASE, 0.2, 0.05)
    points = b.explore_points()
    assert len(points) >= 12

    def cost_of(p):
        return 2.5 * math.dist(BASE, (p['x'], p['y']))

    ranked = rank_points(b, points, cost_of, 0.05, np.random.default_rng(0), origin=BASE)
    assert 0 < len(ranked) <= 12
    assert all({'x', 'y', 'gain', 'cost', 'score'} <= set(p) for p in ranked)
    assert [p['score'] for p in ranked] == sorted((p['score'] for p in ranked), reverse=True)
    assert all(p['gain'] >= 0.0 and p['score'] == pytest.approx(p['gain'] / (p['cost'] + 1.0)) for p in ranked)
    again = rank_points(b, points, cost_of, 0.05, np.random.default_rng(0), origin=BASE)
    assert [(p['x'], p['y']) for p in again] == [(p['x'], p['y']) for p in ranked]      # тот же rng — тот же порядок
    by_mass = rank_points(b, points, cost_of, 0.05, np.random.default_rng(0), mode='mass')
    assert len(by_mass) == len(points) and all(p['gain'] == p['mass'] for p in by_mass)
    unreachable = rank_points(b, points[:3], lambda p: math.inf, 0.05, np.random.default_rng(0), origin=BASE)
    assert all(p['score'] == 0.0 for p in unreachable)

    best = math.inf                                                  # одно решение — не дольше ~60 мс
    for _ in range(5):
        t0 = time.perf_counter()
        rank_points(b, points, cost_of, 0.05, np.random.default_rng(0), origin=BASE)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.06


# --- подключение к агенту ------------------------------------------------------------------------

def _first_states(agent, level='medium', seed=3, until=12.0):
    arena = load_arena()
    sc = generate(level, seed, arena)
    world = FastSim(arena, sc, seed=seed)
    seen = []

    class Spy(HeuristicPlanner):
        def plan(self, state):
            seen.append(state)
            return super().plan(state)

    bot = Agent(arena, make_config(agent), n_samples=len(sc.samples), planner=Spy())
    while not world.done and world.t < until:
        bot.tick(world.observe(), world)
        world.advance()
    return seen


def test_infogain_agent_reports_gain_and_default_agent_does_not():
    assert PRESETS['adaptive'].explore == 'mass' and PRESETS['adaptive_ig'].explore == 'infogain'
    assert PRESETS['adaptive_ig'] == replace(PRESETS['adaptive'], name='adaptive_ig', explore='infogain')
    for agent, has_gain in (('adaptive', False), ('adaptive_ig', True)):
        points = [p for s in _first_states(agent) for p in s['explore_points']]
        assert points
        for p in points:
            assert {'id', 'x', 'y', 'unseen_share', 'cost_to', 'cost_back', 'feasible'} <= set(p)
            assert ('gain_bits' in p) == has_gain


def test_planner_prefers_gain_when_it_is_given():
    point = dict(x=0.0, y=0.0, cost_to=4.0, cost_back=5.0, feasible=True)
    state = {'candidates': [], 'samples': {'total': 3, 'collected': 0}, 'battery': 50.0,
             'explore_points': [{**point, 'id': 'E1', 'unseen_share': 0.6, 'gain_bits': 1.0},
                                {**point, 'id': 'E2', 'unseen_share': 0.2, 'gain_bits': 5.0}]}
    plan = HeuristicPlanner().plan(state)
    assert plan['subgoals'] == [{'type': 'explore', 'target': 'E2'}] and 'бит' in plan['reasoning']
    for p in state['explore_points']:
        del p['gain_bits']
    assert HeuristicPlanner().plan(state)['subgoals'] == [{'type': 'explore', 'target': 'E1'}]


def test_infogain_episode_is_reproducible():
    a = run_episode('medium', 4, 'adaptive_ig', save=False)['metrics']
    b = run_episode('medium', 4, 'adaptive_ig', save=False)['metrics']
    assert a == b and a['returned'] and a['samples_collected'] >= 3


# --- базовые стратегии ---------------------------------------------------------------------------

def test_spiral_route_grows_from_base_over_allowed_cells(arena):
    graph = CostGraph(arena)
    route = spiral_route(arena, graph, pitch=1.6)
    assert 10 <= len(route) <= 40
    r = [math.dist(p, BASE) for p in route]
    assert r[0] == pytest.approx(0.8, abs=0.02) and max(r) > 3.9   # три дуги: 0,8, 2,4 и 4,0 м
    spots = [arena.g2w(i, j) for j, i in zip(*np.nonzero(arena.clear >= 0.22))][::5]
    pts = np.array(route + [BASE])                                  # любое место арены — рядом с какой-то дугой
    gaps = np.array([np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() for x, y in spots])
    assert gaps.max() <= 1.15 and (gaps > 0.9).mean() < 0.05
    assert all(b >= a - 0.05 for a, b in zip(r, r[1:]))             # радиус не убывает
    for x, y in route:
        ix, iy = arena.w2g(x, y)
        assert graph.ok[iy, ix]


@pytest.mark.parametrize('agent', ['gradient', 'spiral', 'adaptive_ig'])
def test_new_agents_survive_all_levels_and_return(agent):
    runs = [run_episode(level, seed, agent, save=False)['metrics']
            for level in ('easy', 'medium', 'hard') for seed in (1, 2, 3)]
    assert all(m['reason'] in ('finish', 'battery', 'timeout') for m in runs)
    assert sum(m['returned'] for m in runs) >= 7                     # возвращаются в большинстве прогонов
    assert sum(m['samples_collected'] for m in runs) >= 15           # и образцы находят: из 45 возможных
    assert sum(m['collisions'] for m in runs) == 0


@pytest.mark.parametrize('name', list(BASELINES))
def test_baseline_run_is_recorded_for_replay(arena, name):
    cls, cfg = BASELINES[name]
    sc = generate('easy', 2, arena)
    rules = Rules()
    world = FastSim(arena, sc, rules, seed=2)
    rec = Recorder()
    bot = cls(arena, cfg, n_samples=len(sc.samples), rules=rules, recorder=rec)
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    bot.tick(world.observe(), world)
    assert bot.finished and world.judge.returned
    assert cfg.static_reserve == PRESETS['fixed'].static_reserve    # то же правило возврата, что у фиксированного
    trace = rec.build(run_id='t', experiment='t', arm=name, backend='fastsim',
                      agent={'name': cfg.name, 'config': cfg.to_dict()}, scenario=sc.to_dict(),
                      rules=rules.to_dict(), result=world.judge.score(), world=world.judge.world_log,
                      journal=bot.journal)
    json.dumps(trace, default=lambda o: o.item() if isinstance(o, np.generic) else o.to_dict())
    assert len(trace['track']['t']) > 100 and trace['plans'] and trace['paths']
    assert set(trace['modes']) <= {'start', 'explore', 'approach', 'collect', 'return', 'escape', 'done'}
    assert trace['plans'][-1]['subgoals'] == [{'type': 'return_base'}]
    assert any(e['type'] == 'sample_collected' for e in trace['events'])
