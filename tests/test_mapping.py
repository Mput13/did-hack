"""Проверки карты занятости по лидару (did/mapping.py) в быстром симуляторе."""
import time

import numpy as np
import pytest

from did.arena import load_arena
from did.config import Rules
from did.fastsim import FastSim
from did.localize import PoseTracker
from did.mapping import OccupancyMapper
from did.nav import CostGraph, Follower
from did.recorder import decode_grid
from did.scenario import generate

TOUR = [(-1.6, 0.6), (-0.55, 1.6), (0.55, 0.55), (1.6, 0.55), (0.55, -0.55), (1.6, -1.2), (-0.55, -1.6),
        (-0.55, -0.55), (-1.6, -0.6), (-0.55, 0.55), (0.55, 1.6), (1.7, 0.0), (0.55, -1.6), (-1.7, -0.5)]
DRIFT = dict(odom_turn_slip=1.0, odom_turn_scale=0.03, odom_path_scale=0.03)   # уход одометрии как в Gazebo


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _tour(arena, sim=None, track=False):
    """Объезд арены по точкам; карта строится из той позы, которую знает робот. Возвращает карту и мс на скан."""
    world = FastSim(arena, generate('easy', 1, arena), Rules(battery_start=1e6, time_limit_s=1e6), seed=3,
                    lidar_hz=5.0, **(sim or {}))
    graph, follower, mapper = CostGraph(arena), Follower(), OccupancyMapper(arena)
    tracker = PoseTracker(arena, enabled=track)
    goals = list(TOUR)
    ms = []
    while world.t < 400.0:
        obs = world.observe()
        x, y, th = tracker.update(obs.x, obs.y, obs.th, obs.scan)
        if obs.scan is not None:
            t0 = time.perf_counter()
            mapper.update(x, y, th, obs.scan)
            ms.append((time.perf_counter() - t0) * 1e3)
        if not follower.active:
            if not goals:
                break
            pts, _ = graph.plan((x, y), goals.pop(0))
            follower.set_path(pts)
        v, w, arrived = follower.step(x, y, th, tol=0.1)
        if arrived:
            follower.set_path([])
        world.command(v, w)
        world.advance()
    assert not goals, 'объезд не закончен'
    return mapper, ms


def test_empty_map(arena):
    m = OccupancyMapper(arena)
    assert m.coverage() == 0.0 and m.agreement() == 0.0
    assert not decode_grid(m.encoded(), m.h, m.w).any()


def test_one_scan_marks_free_and_walls(arena):
    m = OccupancyMapper(arena)
    m.update(-2.0, -0.5, 0.0, arena.raycast(-2.0 - 0.032, -0.5, 0.0))
    ix, iy = arena.w2g(-1.7, -0.5)
    assert m.grid[iy, ix] < -m.KNOWN                     # пол перед роботом свободен
    assert m.occupied().sum() > 50                       # стены вокруг найдены
    assert 0.2 < m.coverage() < 0.9
    assert m.agreement() > 0.9
    grid = decode_grid(m.encoded(), m.h, m.w)
    assert grid[iy, ix] > 0 and (grid == 0).any()        # в показе есть и увиденное, и неизвестное


def test_tour_covers_arena(arena):
    m, ms = _tour(arena)
    print(f'покрытие {m.coverage():.3f}, совпадение {m.agreement():.3f}, скан {np.median(ms):.2f} мс')
    assert m.coverage() >= 0.90
    assert m.agreement() >= 0.95
    assert np.median(ms) < 6.0          # цель ~2 мс; запас на загруженную машину


def test_tour_with_drifting_odometry(arena):
    """Одометрия уходит, поза поправляется по лидару — карта всё равно сходится с эталонной."""
    m, _ = _tour(arena, sim=DRIFT, track=True)
    print(f'с уходом одометрии: покрытие {m.coverage():.3f}, совпадение {m.agreement():.3f}')
    assert m.coverage() >= 0.90
    assert m.agreement() >= 0.93
