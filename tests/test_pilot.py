"""Проверки пульта (did/pilot.py) в быстром симуляторе: команды оператора, езда по точкам, миссия."""
import json
import math

import numpy as np
import pytest

from did.arena import load_arena
from did.config import BASE
from did.pilot import FastWorld, Pilot, PilotHub, dumps


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _pilot(arena, level='medium', seed=3):
    world = FastWorld(arena, level, seed)
    return world, Pilot(arena, world, level, seed)


def _run(world, pilot, until, limit=6000):
    """Крутить симулятор, пока не выполнится условие. Возвращает число тактов."""
    for n in range(limit):
        pilot.tick(world.observe())
        world.advance()
        if until():
            return n
    raise AssertionError(f'условие не выполнилось за {limit} тактов: режим {pilot.mode}, {pilot.note}')


def test_bad_points_are_refused_with_reason(arena):
    world, pilot = _pilot(arena)
    _run(world, pilot, lambda: True)
    assert 'в стену или столб' in pilot.command({'cmd': 'route', 'points': [[0.0, 0.0]]})['message']
    assert 'за пределами арены' in pilot.command({'cmd': 'route', 'points': [[9.0, 9.0]]})['message']
    assert not pilot.command({'cmd': 'route', 'points': [[-1.6, 0.6], ['x', 1]]})['ok']
    assert pilot.route == []                                # отклонённый маршрут ничего не меняет
    res = pilot.command({'cmd': 'go'})
    assert not res['ok'] and 'точку' in res['message']      # ехать некуда
    assert not pilot.command({'cmd': 'fly'})['ok']
    iy, ix = np.nonzero((arena.clear > 0.05) & (arena.clear < 0.11))        # свободно, но у самой стены
    near_wall = arena.g2w(ix[0], iy[0])
    res = pilot.command({'cmd': 'route', 'points': [list(near_wall)]})    # такая точка отодвигается от стены
    assert res['ok'] and 'сдвинута' in res['message']
    assert arena.clearance(pilot.route[0]['x'], pilot.route[0]['y']) >= 0.17


def test_drive_route_then_home(arena):
    world, pilot = _pilot(arena)
    _run(world, pilot, lambda: True)
    goal = (0.55, 0.55)
    assert pilot.command({'cmd': 'route', 'points': [[-1.6, 0.6], [-0.55, 1.6], list(goal)]})['ok']
    st = pilot.state()
    assert len(st['route']) == 3 and st['path_len'] > 3.0 and st['mode'] == 'idle'
    assert pilot.command({'cmd': 'go'})['ok']
    _run(world, pilot, lambda: pilot.mode == 'idle')
    assert math.dist((world.sim.x, world.sim.y), goal) < 0.08       # по истинной позе, а не по своей оценке
    assert all(p['done'] for p in pilot.route)
    assert pilot.mapper.coverage() > 0.6 and pilot.mapper.agreement() > 0.95
    assert pilot.command({'cmd': 'home'})['ok']
    assert pilot.command({'cmd': 'stop'})['ok'] and pilot.mode == 'idle'
    assert pilot.command({'cmd': 'home'})['ok']
    _run(world, pilot, lambda: pilot.mode == 'idle')
    assert math.dist((world.sim.x, world.sim.y), BASE) < 0.1
    json.loads(dumps(pilot.state()))                               # состояние целиком уходит в JSON


def test_mission_runs_from_base_with_fresh_judge(arena):
    world, pilot = _pilot(arena, 'easy', 1)
    _run(world, pilot, lambda: True)
    pilot.command({'cmd': 'route', 'points': [[-1.0, -1.6]]})
    pilot.command({'cmd': 'go'})
    _run(world, pilot, lambda: pilot.mode == 'idle')
    spent = world.rules.battery_start - pilot.battery
    assert spent > 1.0
    assert not pilot.command({'cmd': 'mission', 'agent': 'nobody'})['ok']
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    assert pilot.mode == 'home' and pilot.mission['state'] == 'to_base'     # сначала на базу
    _run(world, pilot, lambda: pilot.mode == 'mission')
    assert world.sim.judge.battery > world.rules.battery_start - 0.5         # судья начал прогон заново
    assert not pilot.command({'cmd': 'go'})['ok']                            # в миссии рулит агент
    _run(world, pilot, lambda: pilot.mode == 'idle', limit=9000)
    m = pilot.state()['mission']
    assert m['state'] == 'finished' and m['result']['returned']
    assert m['result']['samples_collected'] == m['result']['samples_total'] == 3
    assert m['journal'] and m['belief']['data'] and m['trace_file'].endswith('easy-1.json.gz')
    res = pilot.command({'cmd': 'reset'})
    assert res['ok'] and pilot.mission is None and pilot.mapper.scans == 0


def test_hub_reports_clear_errors():
    hub = PilotHub()
    hub.gazebo_alive = lambda: False
    assert hub.state()[1] == {'active': False, 'gazebo_alive': False}
    code, body = hub.command({'cmd': 'go'})
    assert code == 409 and 'не запущен' in body['error']
    assert hub.command({'nope': 1})[0] == 400
    assert hub.start({'backend': 'fastsim', 'level': 'nightmare'})[0] == 400
    assert hub.start({'backend': 'gazebo'})[0] == 409
    code, body = hub.start({'backend': 'fastsim', 'level': 'easy', 'seed': 1, 'speed': 8})
    assert code == 200 and body['backend'] == 'fastsim'
    try:
        assert hub.command({'cmd': 'route', 'points': [[-1.6, 0.6]]})[0] == 200
        assert hub.command({'cmd': 'route', 'points': [[0.0, 0.0]]})[0] == 409
        assert hub.command({'cmd': 'speed', 'value': 2})[0] == 200
        state = json.loads(hub.state()[1])
        assert state['active'] and state['backend'] == 'fastsim' and len(state['route']) == 1
    finally:
        assert hub.start({'backend': 'off'})[0] == 200


def test_escape_side_is_counted_from_current_heading(arena):
    """Ревью G2: пульт выбирает сторону отъезда от нынешнего курса, а не от курса в момент скана."""
    import numpy as np
    _, pilot = _pilot(arena)
    scan = np.full(360, np.inf)
    scan[170:191] = 0.15                                   # преграда вплотную позади — в момент скана
    pilot._scan = (0.0, 0.0, 0.0, scan, None)
    pilot.pose = (0.0, 0.0, 0.0)
    assert pilot._escape_v(back=True) == 0.10              # назад нельзя: вперёд
    pilot.pose = (0.0, 0.0, math.pi)                       # робот развернулся: преграда теперь спереди
    assert pilot._escape_v(back=True) == -0.10
