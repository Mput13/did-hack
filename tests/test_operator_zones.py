"""Ограничения оператора: путь, исполнение и откат некорректной правки."""
import math

import pytest
import numpy as np

from did.arena import load_arena
from did.config import BASE
from did.pilot import FastWorld, Pilot
from did.slam_map import FREE, OCCUPIED


@pytest.fixture
def pilot():
    arena = load_arena()
    world = FastWorld(arena, 'medium', 3)
    bot = Pilot(arena, world, 'medium', 3)
    bot.tick(world.observe())
    return world, bot


def test_danger_zone_replans_and_robot_drives_around_it(pilot):
    world, bot = pilot
    goal = [0.55, 0.55]
    assert bot.command({'cmd': 'route', 'points': [goal]})['ok']
    path = bot.route[0]['pts']
    center = path[len(path) // 2]
    zone = {'kind': 'danger', 'x': center[0], 'y': center[1], 'r': 0.2}
    assert bot.command({'cmd': 'zones', 'zones': [zone]})['ok']
    assert min(math.dist(p, center) for p in bot.route[0]['pts']) > 0.4
    assert bot.command({'cmd': 'go'})['ok']
    for _ in range(3000):
        bot.tick(world.observe())
        world.advance()
        assert math.dist((world.sim.x, world.sim.y), center) > 0.2 + 0.105
        if bot.mode == 'idle':
            break
    assert bot.mode == 'idle'
    assert math.dist((world.sim.x, world.sim.y), goal) < 0.1


def test_yellow_zone_is_preference_not_battery_multiplier(pilot):
    world, bot = pilot
    assert bot.command({'cmd': 'route', 'points': [[0.55, 0.55]]})['ok']
    path = bot.route[0]['pts']
    center = path[len(path) // 2]
    truth_before = world.truth()
    zone = {'kind': 'yellow', 'x': center[0], 'y': center[1], 'r': 0.4}
    assert bot.command({'cmd': 'zones', 'zones': [zone]})['ok']
    assert bot.route[0]['pts'] != path
    assert world.truth() == truth_before
    assert bot.state()['zones'] == [zone]


def test_invalid_zone_edit_keeps_previous_route_and_zones(pilot):
    _, bot = pilot
    assert bot.command({'cmd': 'route', 'points': [[0.55, 0.55]]})['ok']
    zone = {'kind': 'yellow', 'x': -1.4, 'y': -0.5, 'r': 0.2}
    assert bot.command({'cmd': 'zones', 'zones': [zone]})['ok']
    previous = bot.route
    assert not bot.command({'cmd': 'zones', 'zones': [{'kind': 'danger', 'x': 0.55, 'y': 0.55, 'r': 0.3}]})['ok']
    assert bot.zones == [zone] and bot.route == previous
    assert not bot.command({'cmd': 'zones', 'zones': [{'kind': 'danger', 'x': BASE[0], 'y': BASE[1], 'r': 0.3}]})['ok']
    assert not bot.command({'cmd': 'mission'})['ok']
    assert bot.command({'cmd': 'zones', 'zones': []})['ok']
    assert bot.command({'cmd': 'mission'})['ok']


@pytest.mark.parametrize('zone', [None, {}, {'kind': 'danger', 'x': float('nan'), 'y': 1, 'r': 0.4},
                                  {'kind': 'yellow', 'x': 1, 'y': 1, 'r': 4}])
def test_malformed_zones_are_refused(pilot, zone):
    _, bot = pilot
    assert not bot.command({'cmd': 'zones', 'zones': [zone]})['ok']
    assert not bot.zones


def test_slam_refresh_keeps_operator_restrictions():
    class SlamWorld(FastWorld):
        ver = 1

        def slam_grid(self):
            return self.ver, np.where(self.arena.free, FREE, OCCUPIED).astype(np.int8)

        def slam_tf(self):
            return {'tf': (0., 0., 0.), 'tf_age': 0., 'grid_age': 0.}

    arena = load_arena()
    world = SlamWorld(arena, 'medium', 3, drift=False)
    bot = Pilot(arena, world, 'medium', 3, slam_nav=True)
    # No graph yet: an edit must be refused rather than crash.
    assert not bot.command({'cmd': 'zones', 'zones': []})['ok']
    bot.tick(world.observe())
    goal = [0.55, 0.55]
    assert bot.command({'cmd': 'route', 'points': [goal]})['ok']
    original = bot.route[0]['pts']
    center = original[len(original) // 2]
    zone = {'kind': 'danger', 'x': center[0], 'y': center[1], 'r': 0.2}
    assert bot.command({'cmd': 'zones', 'zones': [zone]})['ok']
    old_graph = bot.graph
    world.ver += 1
    bot.tick(world.observe())
    assert bot.graph is not old_graph
    assert not bot._path_ok(original)
    assert bot.command({'cmd': 'route', 'points': [goal]})['ok']
    assert min(math.dist(p, center) for p in bot.route[0]['pts']) > 0.4
