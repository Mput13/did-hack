"""Проверки исследования G2: упор в преграду как в Gazebo и поведение агента после столкновения.

В Gazebo столкновение физическое: корпус стоит, колёса проскальзывают, одометрия уезжает; если давить
дальше, робот раскачивается и лидар слепнет. Быстрый симулятор умеет это изображать (bump=True), а
рывки курса мимо одометрии — параметром kick. Здесь проверяется, что агент в преграду не давит,
отъезжает туда, где свободно, и после потери положения находит себя, а не едет вслепую.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from did.agent import Agent, make_config
from did.arena import load_arena
from did.config import LIDAR_MAX, LIDAR_MIN, ROBOT_RADIUS, Rules
from did.fastsim import BUMP_ROCK_S, LIDAR_OFFSET, FastSim
from did.localize import PoseTracker
from did.nav import ESCAPE_ROOM, ESCAPE_V, corridor_room, escape_plan, free_ahead
from did.robot_io import Observation
from did.runner import run_episode
from did.scenario import generate

MODERATE = dict(odom_turn_slip=0.2, odom_turn_scale=0.01, odom_path_scale=0.01)
PILLAR = (-1.1, -1.1)                       # ближайший к базе столб
KICKS = dict(MODERATE, bump=True, kick=2.5, kick_every=15.0)      # курс рывком уходит до 143° мимо одометрии


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _scan(arena, x, y, th):
    r = arena.raycast(x + LIDAR_OFFSET * math.cos(th), y + LIDAR_OFFSET * math.sin(th), th)
    r[(r < LIDAR_MIN) | (r > LIDAR_MAX)] = np.inf
    return r


def _sim_at(arena, pose, **sim):
    """Быстрый симулятор с роботом в заданной позе (одометрия с ней согласована)."""
    world = FastSim(arena, generate('easy', 1, arena), Rules(battery_start=1e6), seed=1, **sim)
    world.x, world.y, world.th = pose
    if world._odom is not None:
        world._odom = list(pose)
    world._next_lidar = world.t
    world._sense()
    return world


# --- свободное место по лидару -------------------------------------------------------------------

def test_corridor_room_looks_at_robot_width(arena):
    """Столб сзади и сбоку в узкий сектор не попадает, а в полосу шириной с робота — попадает."""
    x, y = PILLAR[0] + 0.38, PILLAR[1] + 0.20                 # столб позади-справа, корпус его заденет
    scan = _scan(arena, x, y, 0.0)
    assert min(scan[170:191].min(), 9.0) > 0.25               # прямо назад (±10°) лидар видит дальше,
    assert corridor_room(scan, back=True) < 0.20              # чем корпусу на самом деле можно проехать
    assert corridor_room(scan) > 0.5
    assert corridor_room(scan, heading=math.pi) == pytest.approx(corridor_room(scan, back=True))
    assert corridor_room(np.full(360, np.inf)) == math.inf


def test_escape_goes_where_there_is_room(arena):
    x, y = PILLAR[0] + 0.38, PILLAR[1] + 0.20
    behind = _scan(arena, x, y, 0.0)                          # столб сзади: назад нельзя
    assert escape_plan(behind, back=True) == (ESCAPE_V, 0.0)
    assert escape_plan(behind, back=False) == (ESCAPE_V, 0.0)
    ahead = _scan(arena, x, y, math.pi)                       # столб спереди: как раньше, назад
    assert escape_plan(ahead, back=True) == (-ESCAPE_V, 0.0)
    assert escape_plan(None, back=True) == (-ESCAPE_V, 0.0)   # скана ещё нет: прежнее поведение
    # Тесно и спереди, и сзади (между столбом и стеной поперёк прохода): развернуться туда, где свободно.
    boxed = np.full(360, np.inf)
    boxed[:30] = boxed[-30:] = 0.22
    boxed[150:210] = 0.22
    v, turn = escape_plan(boxed, back=True)
    assert v == ESCAPE_V and abs(abs(turn) - math.pi / 2) < 0.3
    assert escape_plan(np.full(360, 0.2), back=True) == (0.0, 0.0)      # тесно всюду: стоять


def test_free_ahead_is_distance_to_touch(arena):
    scan = np.full(360, np.inf)
    scan[0] = 0.30 - LIDAR_OFFSET                             # преграда прямо по курсу в 0,30 м от центра
    assert free_ahead(scan) == pytest.approx(0.30 - ROBOT_RADIUS, abs=1e-6)
    side = np.full(360, np.inf)
    side[90] = 0.20                                           # преграда сбоку: вперёд ехать не мешает
    assert free_ahead(side) == math.inf
    assert free_ahead(side, heading=math.pi / 2) < 0.12       # а если повернуться к ней — мешает


# --- физический упор в быстром симуляторе ------------------------------------------------------------

def test_without_bump_collision_is_only_a_penalty(arena):
    world = _sim_at(arena, (PILLAR[0] - 0.5, PILLAR[1], 0.0))
    for _ in range(50):
        world.command(0.2, 0.0)
        world.advance()
    obs = world.observe()
    assert world._blocked and not world.rocking
    assert (obs.x, obs.y, obs.th) == (world.x, world.y, world.th)        # одометрия не уезжает


def test_bump_makes_odometry_run_away(arena):
    """Как в Gazebo (tools/gz_bump_test.py): в упоре одометрия считает путь, которого нет, корпус сползает вбок."""
    start = (PILLAR[0] - 0.5, PILLAR[1] + 0.03, 0.0)
    world = _sim_at(arena, start, bump=True)
    pushed = 0.0
    while pushed < 2.0:
        world.command(0.2, 0.0)
        world.advance()
        pushed += world.dt if world._blocked else 0.0
    obs = world.observe()
    assert world.x < PILLAR[0] - 0.2                          # корпус стоит перед столбом
    assert obs.x - world.x > 0.2                              # а по одометрии проехал ещё 20+ см
    assert abs(world.th - obs.th) > 0.15                      # и повернулся, чего одометрия не видела
    assert not world.rocking
    while pushed < BUMP_ROCK_S + 0.5:                         # давить дальше — раскачаться: лидар слепнет
        world.command(0.2, 0.0)
        world.advance()
        pushed += world.dt
    assert world.rocking
    tracker = PoseTracker(arena)
    tracker._calib = None
    for _ in range(30):
        world.advance()
        obs = world.observe()
        if obs.scan is not None:
            tracker.update(world.x, world.y, world.th, obs.scan)
    assert tracker.lost and tracker.stats['fixes'] == 0


def test_tracker_reports_slip_while_pushing(arena):
    """Поправка по лидару держит позу на месте и сообщает, что колёса проскальзывают — раньше, чем робот раскачается."""
    world = _sim_at(arena, (PILLAR[0] - 0.5, PILLAR[1] + 0.03, 0.0), bump=True)
    tracker = PoseTracker(arena)
    tracker._calib = None
    pushed, noticed = 0.0, None
    while pushed < 2.4:
        obs = world.observe()
        x, y, th = tracker.update(obs.x, obs.y, obs.th, obs.scan)
        if noticed is None and tracker.stats['slip'] >= Agent.SLIP_STOP:
            noticed = pushed
            assert tracker.stats['slip_vec'][0] < -0.03                  # поправка тянет назад, против хода
        world.command(0.2, 0.0)
        world.advance()
        pushed += world.dt if world._blocked else 0.0
    assert noticed is not None and noticed <= 1.5
    assert math.hypot(x - world.x, y - world.y) < 0.12 < math.hypot(obs.x - world.x, obs.y - world.y)


# --- агент после столкновения -------------------------------------------------------------------

def _agent(arena, guard=True):
    bot = Agent(arena, make_config('adaptive', guard=guard), n_samples=3)
    bot.tracker._calib = None
    return bot


def _obs(arena, pose, t=10.0, events=(), scan=True):
    return Observation(t=t, x=pose[0], y=pose[1], th=pose[2], v=0.0, w=0.0, battery=50.0, sensor=None,
                       scan=_scan(arena, *pose) if scan else None, events=list(events))


class _IO:
    def __init__(self):
        self.cmd = None

    def command(self, v, w):
        self.cmd = (v, w)

    def collect(self):
        return False, ''

    def finish(self):
        return True, ''


def test_agent_does_not_back_into_pillar(arena):
    """Причина срыва в Gazebo: после штрафа у столба робот вслепую отъезжал назад — прямо в этот столб."""
    pose = (PILLAR[0] + 0.38, PILLAR[1] + 0.20, 0.0)          # столб позади-справа
    hit = [{'type': 'collision', 't': 10.0, 'x': pose[0], 'y': pose[1]}]
    old, new = _agent(arena, guard=False), _agent(arena)
    for bot in (old, new):
        bot._last_cmd = (0.2, 0.0)
        bot._perceive(_obs(arena, pose, events=hit))
    assert old._escape['v'] == -0.10                           # как было: назад, не глядя
    assert new._escape['v'] == 0.10                            # теперь: туда, где лидар видит место
    assert new._alert_until > 10.0 and old._alert_until < 0.0


def test_escape_stops_when_obstacle_gets_close(arena):
    bot, io = _agent(arena), _IO()
    pose = (PILLAR[0] + 0.55, PILLAR[1], 0.0)                 # до столба сзади больше 0,3 м: отъезжать можно
    bot._last_cmd = (0.2, 0.0)
    bot.queue = [{'type': 'return_base'}]
    obs = _obs(arena, pose, events=[{'type': 'collision', 't': 10.0}])
    bot._perceive(obs)
    assert bot._escape['v'] == -0.10
    bot._act(obs, io)
    assert io.cmd == pytest.approx((-0.10, 0.0)) and bot.mode == 'escape'
    close = _obs(arena, (PILLAR[0] + 0.38, PILLAR[1], 0.0), t=10.5)      # подъехал: до столба уже меньше 0,2 м
    bot._perceive(close)
    bot._act(close, io)
    assert io.cmd == (0.0, 0.0)


def test_agent_stands_still_when_pose_is_lost(arena):
    """Сканы перестали сходиться с картой: агент не едет по позе, которой нельзя верить."""
    bot, io = _agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 0.5}]
    bot._trigger = None
    pose = (-1.6, 0.5, 0.4)
    rng = np.random.default_rng(0)
    moving = lost = 0
    for k in range(40):
        junk = np.where(rng.random(360) < 0.5, rng.uniform(0.15, 1.0, 360), np.inf) if k >= 10 and k % 2 == 0 else None
        obs = replace(_obs(arena, pose, t=0.1 * k, scan=(k < 10)), scan=junk if k >= 10 else _scan(arena, *pose))
        bot.tick(obs, io)
        if k < 10:
            moving += io.cmd != (0.0, 0.0)
        if k >= 22:                                           # после пяти плохих сканов
            lost += 1
            assert io.cmd == (0.0, 0.0) and bot.mode == 'lost'
    assert moving >= 8 and lost > 0
    assert any(e.get('data', {}).get('tag') == 'lost' for e in bot.journal.entries)
    for k in range(40, 50):                                   # сканы вернулись: нашёл себя и поехал
        bot.tick(_obs(arena, pose, t=0.1 * k), io)
    assert bot.mode != 'lost' and io.cmd != (0.0, 0.0)
    assert any(e.get('data', {}).get('tag') == 'relocated' for e in bot.journal.entries)


def test_quiet_run_is_the_same_with_and_without_guard():
    """Пока ничего не случилось, новая осторожность прогон не меняет."""
    for level, seed in (('medium', 1), ('hard', 3)):
        old = run_episode(level, seed, 'adaptive', save=False, config={'guard': False})['metrics']
        new = run_episode(level, seed, 'adaptive', save=False)['metrics']
        assert old['collisions'] == new['collisions'] == 0
        assert (old['score'], old['time'], old['distance']) == (new['score'], new['time'], new['distance'])


@pytest.mark.parametrize('seed', [4, 8])
def test_agent_survives_heading_kicks_with_physical_collisions(arena, seed):
    """Воспроизведение срыва из Gazebo без Gazebo: курс рывками уходит мимо одометрии, столкновения физические.

    Прежний агент в таком прогоне упирается в столбы, раскачивается и не возвращается; новый — возвращается.
    """
    old = run_episode('hard', seed, 'adaptive', save=False, sim=KICKS, config={'guard': False})['metrics']
    assert not old['returned'] and old['collisions'] >= 5

    scenario = generate('hard', seed, arena)
    world = FastSim(arena, scenario, Rules(), seed=seed, **KICKS)
    bot = Agent(arena, make_config('adaptive'), n_samples=len(scenario.samples))
    worst = 0.0
    while not world.done:
        obs = world.observe()
        bot.tick(obs, world)
        x, y, _ = bot.tracker.to_map(obs.x, obs.y, obs.th)
        if bot.mode not in ('lost', 'done') and not bot.tracker.stats['unsure']:
            worst = max(worst, math.hypot(x - world.x, y - world.y))      # ошибка позы, пока робот ей верит и едет
        world.advance()
    score = world.judge.score()
    assert score['returned'] and score['collisions'] <= 1 and not world.rocking
    assert score['samples_collected'] >= old['samples_collected']
    assert bot.tracker.stats['relocations'] >= 1              # положение терялось и было найдено заново
    assert worst < 0.45
