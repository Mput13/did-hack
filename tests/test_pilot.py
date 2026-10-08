"""Проверки пульта (did/pilot.py) в быстром симуляторе: команды оператора, езда по точкам, миссия."""
import json
import math

import numpy as np
import pytest

from did.arena import load_arena
from did.config import BASE
from did.mission_agents import MISSION_AGENTS
from did.pilot import PRESETS, FastWorld, Pilot, PilotHub, RunReset, dumps, fresh_score


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
    assert abs(pilot.distance - m['result']['distance']) < 0.3   # без прежнего ручного маршрута и подъезда к базе
    assert m['journal'] and m['belief']['data'] and m['trace_file'].endswith('easy-1.json.gz')
    res = pilot.command({'cmd': 'reset'})
    assert res['ok'] and pilot.mission is None and pilot.mapper.scans == 0


@pytest.mark.parametrize('agent', ['adaptive_v2', 'scientist_v2'])
def test_mission_with_fault_waiting_agents(arena, agent):
    """Агенты с пережиданием сбоя датчика запускаются с пульта; умолчание и прежние три агента — на своих местах."""
    world, pilot = _pilot(arena, 'hard', 1)
    _run(world, pilot, lambda: True)
    ids = [a['id'] for a in pilot.state()['agents']]
    assert ids[:3] == ['adaptive', 'scientist', 'fixed'] and agent in ids
    assert pilot.command({'cmd': 'mission', 'agent': agent})['ok']
    assert pilot.mode == 'mission' and 'пережидает сбой датчика' in pilot.mission['label']
    assert pilot.bot.cfg.name == agent and pilot.bot.cfg.fault_wait
    _run(world, pilot, lambda: pilot.mode == 'idle', limit=12000)
    m = pilot.state()['mission']
    assert m['state'] == 'finished' and m['result']['returned']
    assert m['trace_file'] == f'pilot/{agent}/hard-1.json.gz'


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


# --- новый прогон судьи перед миссией: агент стартует только после подтверждения ---------------------

class _StandWorld(FastWorld):
    """Быстрый симулятор со сбросом судьи как на стенде Gazebo: вызов, ответ и подтверждение счётом — порознь.

    lost — сколько первых вызовов сброса до судьи не дойдут. Часы ожидания — свои (self.now), чтобы проверка
    не ждала настоящие десять секунд. Счёт судьи приходит «сообщениями»: один и тот же объект до следующего такта.
    """

    def __init__(self, *args, lost=0, **kw):
        super().__init__(*args, **kw)
        self.lost, self.calls, self.now = lost, 0, 0.0
        self._msg = self.sim.judge.score()

    def advance(self):
        super().advance()
        self._msg = self.sim.judge.score()

    def _send(self):
        self.calls += 1
        if self.calls <= self.lost:
            return lambda: None                         # вызов ушёл, судья молчит
        FastWorld.reset_begin(self)
        self._msg = self.sim.judge.score()
        return lambda: (True, 'судья начал прогон заново')

    def reset_begin(self):
        self._reset = RunReset(self._send, lambda: (self._msg, self.sim.judge.battery), self.rules.battery_start,
                               clock=lambda: self.now)

    def reset_poll(self):
        return self._reset.poll()


def _stand(arena, lost, level='easy', seed=1):
    world = _StandWorld(arena, level, seed, lost=lost)
    pilot = Pilot(arena, world, level, seed)
    _run(world, pilot, lambda: True)
    return world, pilot


def test_mission_agents_are_known_presets():
    assert set(MISSION_AGENTS) <= set(PRESETS) and list(MISSION_AGENTS)[0] == 'adaptive'


def test_mission_starts_only_after_confirmed_reset(arena):
    world, pilot = _stand(arena, lost=0)
    res = pilot.command({'cmd': 'mission', 'agent': 'adaptive'})
    assert res['ok'] and res['message'] == 'Миссия запущена'
    assert pilot.mode == 'mission' and pilot.bot is not None and world.calls == 1
    m = pilot.state()['mission']
    assert m['state'] == 'running' and m['fresh_run'] is True


def test_mission_is_not_started_when_reset_is_refused(arena):
    world, pilot = _stand(arena, lost=99)
    pilot.bot = object()                                    # агент прошлой миссии: такт ему достаться не должен
    res = pilot.command({'cmd': 'mission', 'agent': 'adaptive'})
    assert res['ok'] and 'Жду' in res['message']
    assert pilot.mode == 'mission' and pilot.bot is None and pilot.mission['state'] == 'reset'
    assert not pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']     # вторая кнопка — «уже идёт»
    assert not pilot.command({'cmd': 'go'})['ok']
    for _ in range(3):                                      # первая попытка, вторая, отказ
        _run(world, pilot, lambda: True)
        assert pilot.bot is None
        world.now += 10.5
    _run(world, pilot, lambda: True)
    assert world.calls == 2 and pilot.mode == 'idle' and pilot.bot is None
    st = json.loads(dumps(pilot.state()))                   # то, что уходит странице показа
    assert st['mission']['state'] == 'refused' and st['mission']['fresh_run'] is False
    assert 'не ответил' in st['mission']['reason']
    assert st['note']['tone'] == 'bad' and 'Судья не начал новый прогон' in st['note']['text']
    assert 'ещё раз' in st['note']['text']
    world.lost = 0                                          # оператор нажал ещё раз, судья ожил
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok'] and pilot.bot is not None


def test_immediate_refusal_is_the_command_answer(arena):
    world, pilot = _pilot(arena, 'easy', 1)
    _run(world, pilot, lambda: True)
    world.reset_poll = lambda: (False, 'у судьи нет сервиса /did/reset')
    res = pilot.command({'cmd': 'mission', 'agent': 'adaptive'})
    assert not res['ok'] and 'Судья не начал новый прогон: у судьи нет сервиса' in res['message']
    assert pilot.mode == 'idle' and pilot.bot is None and pilot.note['tone'] == 'bad'


def test_mission_starts_when_reset_confirmed_on_second_try(arena):
    world, pilot = _stand(arena, lost=1)
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    _run(world, pilot, lambda: True)
    assert pilot.bot is None and pilot.mission['state'] == 'reset' and world.calls == 1
    world.now += 10.5
    _run(world, pilot, lambda: pilot.bot is not None, limit=5)
    assert world.calls == 2 and pilot.mode == 'mission'
    assert pilot.mission['state'] == 'running' and pilot.mission['fresh_run'] is True


def test_stale_finished_does_not_end_new_mission(arena):
    """Прежний прогон судьи закончен (finished=true), а сброс подтверждается не сразу: ни агент на старом счёте,
    ни завершение миссии старым итогом."""
    world, pilot = _stand(arena, lost=1)
    pilot.command({'cmd': 'route', 'points': [[-1.0, -1.6]]})
    pilot.command({'cmd': 'go'})
    _run(world, pilot, lambda: pilot.mode == 'idle')
    pilot.command({'cmd': 'home'})
    _run(world, pilot, lambda: pilot.mode == 'idle')
    world.finish()                                          # прежний прогон закончен
    assert world.observe().done and world.score()['finished']
    assert world.rules.battery_start - world.sim.judge.battery > 5.0
    pilot._rerun_at = 1e18                                  # ручная езда новый прогон сама не начнёт
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    for _ in range(20):
        _run(world, pilot, lambda: True)
        assert pilot.mode == 'mission' and pilot.bot is None and pilot.mission['state'] == 'reset'
    assert world.score()['finished']                        # судья всё ещё на старом прогоне
    world.now += 10.5
    _run(world, pilot, lambda: pilot.bot is not None, limit=5)
    for _ in range(30):
        _run(world, pilot, lambda: True)
    assert pilot.mode == 'mission' and pilot.mission['state'] == 'running' and pilot._finish_at is None
    assert world.sim.judge.battery > world.rules.battery_start - 3.0 and not world.score()['finished']
    assert pilot.command({'cmd': 'stop'})['ok'] and pilot.mode == 'idle'


def test_waiting_mission_can_be_cancelled(arena):
    world, pilot = _stand(arena, lost=99)
    pilot.command({'cmd': 'mission', 'agent': 'adaptive'})
    assert pilot.command({'cmd': 'finish'})['ok'] and pilot.mode == 'idle' and pilot._reset is None
    assert pilot.mission['state'] == 'aborted' and not world.score()['finished']    # судью «завершать» было нечего
    _run(world, pilot, lambda: True)
    assert pilot.bot is None and pilot.mode == 'idle'


def test_run_reset_contract():
    start = 60.0
    old = {'t': 150.0, 'finished': False}
    new = {'t': 0.1, 'finished': False}
    assert not fresh_score(old, start, old, start, True)                        # то же сообщение
    assert not fresh_score({'t': 0.1, 'finished': True}, start, old, start, True)
    assert not fresh_score(new, start - 5.0, old, start, True)                  # заряд ещё прежний
    assert not fresh_score(new, None, old, start, True)
    assert not fresh_score({'t': 151.0, 'finished': False}, start, old, start, True)   # прежний прогон идёт дальше
    assert fresh_score(new, start, old, start) and fresh_score(new, start, old, start, True)
    assert fresh_score({'t': 7.0, 'finished': False}, start - 0.5, old, start)  # заметили поздно, но прогон новый
    young = {'t': 0.5, 'finished': False}                                       # прежний прогон сам только начался
    assert not fresh_score({'t': 0.9, 'finished': False}, start, young, start)  # без ответа судьи это может быть он же
    assert fresh_score({'t': 0.9, 'finished': False}, start, young, start, True)

    box = {'now': 0.0, 'score': old, 'sent': 0}
    def send():
        box['sent'] += 1
        return lambda: (True, 'ок')
    r = RunReset(send, lambda: (box['score'], start), start, clock=lambda: box['now'])
    assert r.poll() is None and box['sent'] == 1
    box['now'] = 9.0
    assert r.poll() is None and box['sent'] == 1                                # ждём дольше прежних трёх секунд
    box['score'] = {'t': 0.4, 'finished': False}
    assert r.poll() == (True, 'ок') and r.poll() == (True, 'ок')

    box.update(now=0.0, score=old, sent=0)
    r = RunReset(send, lambda: (box['score'], start), start, clock=lambda: box['now'])
    for now in (0.0, 10.1, 10.2, 20.0):
        box['now'] = now
        assert r.poll() is None
    box['now'] = 20.4
    ok, why = r.poll()
    assert not ok and box['sent'] == 2 and 'счёт нового прогона' in why and '150' in why

    r = RunReset(lambda: None, lambda: (old, start), start, wait_s=1.0, clock=lambda: box['now'])   # сервиса нет
    for now in (30.0, 31.5, 31.6, 33.0):
        box['now'] = now
        res = r.poll()
    assert res[0] is False and '/did/reset' in res[1]

    r = RunReset(lambda: (lambda: (False, 'занят')), lambda: (old, start), start, clock=lambda: box['now'])
    assert r.poll() is None                                                     # отказ: сразу вторая попытка
    ok, why = r.poll()
    assert not ok and 'занят' in why
