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
from did.nav import ESCAPE_ROOM, ESCAPE_STOP, ESCAPE_V, corridor_room, escape_plan, free_ahead, moved_since
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


# --- исправления по ревью G2 ------------------------------------------------------------------------

FAULTS = dict(KICKS, lidar_fault=20.0, lidar_fault_s=4.0)     # и рывки курса, и сбои лидара


def _echo_scan(points, pose, n=360):
    """Скан из позы pose, в котором видны только заданные точки мира (остальные лучи без отражения)."""
    x, y, th = pose
    lx, ly = x + LIDAR_OFFSET * math.cos(th), y + LIDAR_OFFSET * math.sin(th)
    r = np.full(n, np.inf)
    for px, py in points:
        k = round(math.remainder(math.atan2(py - ly, px - lx) - th, 2 * math.pi) / (2 * math.pi / n)) % n
        r[k] = min(r[k], math.hypot(px - lx, py - ly))
    return r


def _wall(x0, y0, x1, y1, n=25):
    return [(x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1)) for k in range(n)]


def _blind_agent(arena, **config):
    """Агент без поправки позы: в проверках ниже сканы выдуманные и с картой не сходятся."""
    bot = Agent(arena, make_config('adaptive', localize=False, **config), n_samples=3)
    bot._trigger = None
    return bot


def _see(pose, t, scan=None, events=(), scan_pose=None, battery=50.0):
    return Observation(t=t, x=pose[0], y=pose[1], th=pose[2], v=0.0, w=0.0, battery=battery, sensor=None,
                       scan=scan, events=list(events), scan_pose=scan_pose)


def test_room_accounts_for_motion_since_scan():
    scan = np.full(360, np.inf)
    scan[125] = 0.20
    assert corridor_room(scan, back=True) == math.inf
    turned = moved_since((0.0, 0.0, 0.0), (0.0, 0.0, -0.3))
    assert corridor_room(scan, back=True, moved=turned) == pytest.approx(0.1886, abs=1e-3)
    behind = np.full(360, np.inf)
    behind[180] = 0.50 + LIDAR_OFFSET                         # преграда в 0,5 м позади центра робота
    assert corridor_room(behind, back=True) == pytest.approx(0.50)
    assert corridor_room(behind, back=True, moved=moved_since((1.0, 1.0, 2.0), (1.0 - 0.2 * math.cos(2.0),
                         1.0 - 0.2 * math.sin(2.0), 2.0))) == pytest.approx(0.30)       # отъехал назад на 0,2 м
    assert free_ahead(behind, moved=(0.0, 0.0, math.pi)) == pytest.approx(0.50 - ROBOT_RADIUS)


def test_delayed_scan_is_tied_to_the_pose_it_was_taken_from(arena):
    """Ревью G2, находка 1: скан снят при курсе 0, а пришёл, когда робот уже повернулся на −0,3 рад.
    Позади-сбоку преграда: корпусу до неё 0,19 м, назад ехать нельзя. Раньше скан привязывался к
    текущему курсу, полоса назад выходила пустой, и робот отъезжал в преграду."""
    scan = np.full(360, np.inf)
    scan[125] = 0.20
    bot, io = _blind_agent(arena), _IO()
    bot._last_cmd = (0.2, 0.0)
    bot.queue = [{'type': 'return_base'}]
    obs = _see((0.0, 0.0, -0.3), 10.0, scan, [{'type': 'collision', 't': 10.0}], scan_pose=(0.0, 0.0, 0.0))
    bot._perceive(obs)
    bot._act(obs, io)
    assert bot.mode == 'escape' and io.cmd == pytest.approx((0.10, 0.0))        # вперёд: там свободно
    assert bot._room(obs, back=True) == pytest.approx(0.1886, abs=1e-3) and 0.1886 < ESCAPE_STOP


def test_escape_side_is_chosen_for_the_current_heading(arena):
    """Ревью G2, находка 1, второй случай: после скана робот развернулся. Сторона отъезда выбирается для
    нынешнего курса — иначе выбранную сторону тут же запрещала проверка остановки, и робот стоял,
    хотя с другой стороны свободно."""
    wall = _wall(-0.18, -0.15, -0.18, 0.15)                   # преграда в 0,18 м позади робота, стоящего в нуле
    bot, io = _blind_agent(arena), _IO()
    bot.queue = [{'type': 'return_base'}]
    first = _see((0.0, 0.0, 0.0), 9.8, _echo_scan(wall, (0.0, 0.0, 0.0)))
    bot._perceive(first)
    assert bot._room(first, back=True) < ESCAPE_STOP < bot._room(first)
    bot._last_cmd = (0.2, 0.0)
    now = _see((0.0, 0.0, math.pi), 10.0, None, [{'type': 'collision', 't': 10.0}])     # развернулся; скана нет
    bot._perceive(now)
    bot._act(now, io)
    assert io.cmd == pytest.approx((-0.10, 0.0))             # преграда теперь спереди: назад свободно


def test_turn_before_escape_then_route_resumes(arena):
    """Тесно и спереди, и сзади: робот разворачивается туда, где свободно, отъезжает и едет дальше по маршруту."""
    box = _wall(0.22, -0.2, 0.22, 0.2) + _wall(-0.22, -0.2, -0.22, 0.2)
    box = [(x - 1.0, y + 1.0) for x, y in box]                # в свободном месте арены
    pose = [-1.0, 1.0, 0.0]
    bot, io = _blind_agent(arena), _IO()
    bot._last_cmd = (0.2, 0.0)
    bot.queue = [{'type': 'return_base'}]
    modes, t = [], 10.0
    obs = _see(pose, t, _echo_scan(box, pose), [{'type': 'collision', 't': t}])
    for k in range(200):
        bot.tick(obs, io)
        modes.append(bot.mode)
        if k == 0:
            assert 'turn' in bot._escape and io.cmd[0] == 0.0 and io.cmd[1] != 0.0
        v, w = io.cmd
        pose[0] += v * math.cos(pose[2]) * 0.1
        pose[1] += v * math.sin(pose[2]) * 0.1
        pose[2] = math.remainder(pose[2] + w * 0.1, 2 * math.pi)
        t += 0.1
        obs = _see(pose, t, _echo_scan(box, pose) if k % 4 == 3 else None)
        if bot.mode == 'return' and math.hypot(pose[0] + 1.0, pose[1] - 1.0) > 0.5:
            break
    turning = modes.index('return') if 'return' in modes else len(modes)
    assert set(modes[:turning]) == {'escape'} and turning * 0.1 <= 7.5         # разворот и отъезд — не дольше 7,5 с
    assert abs(abs(pose[2]) - math.pi / 2) < 1.0 or math.hypot(pose[0] + 1.0, pose[1] - 1.0) > 0.5
    assert math.hypot(pose[0] + 1.0, pose[1] - 1.0) > 0.5     # выехал из тесного места и едет на базу
    assert min(math.hypot(pose[0] - x, pose[1] - y) for x, y in box) > ROBOT_RADIUS


def _lose(bot, io, arena, pose, t0, until, battery=50.0, scans=True):
    """Тики со сканами, которые не сходятся с картой: что командует агент и в каких режимах."""
    rng, out, t = np.random.default_rng(1), [], t0
    while t < until - 1e-9:
        k = round((t - t0) * 10)
        junk = np.where(rng.random(360) < 0.5, rng.uniform(0.15, 1.0, 360), np.inf) if scans and k % 4 == 0 else None
        bot.tick(_see(pose, t, junk, battery=battery), io)
        out.append((round(t - t0, 1), bot.mode, io.cmd))
        t += 0.1
    return out


def test_unsure_after_bad_scans_then_degenerate_scans_agent_drives(arena, monkeypatch):
    """Ревью G2, находка 2 вместе с командой агента: после двух плохих сканов и вырожденных хороших робот едет."""
    bot, io = _agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 0.5}]
    bot._trigger = None
    pose = (-1.6, 0.5, 0.4)
    bot.tick(_obs(arena, pose, t=0.0), io)
    for k in (1, 2):
        bot.tick(replace(_obs(arena, pose, t=0.4 * k), scan=np.full(360, np.inf)), io)
    assert bot.tracker.stats['unsure'] and io.cmd == (0.0, 0.0)
    fit = bot.tracker._fit
    monkeypatch.setattr(bot.tracker, '_fit', lambda *a, **kw: (lambda p, q: (p, (*q[:3], False)))(*fit(*a, **kw)))
    for k in range(3, 8):
        bot.tick(_obs(arena, pose, t=0.4 * k), io)
    assert not bot.tracker.stats['unsure'] and io.cmd != (0.0, 0.0)


def test_unsure_wait_has_a_deadline(arena):
    """Два скана не сошлись, и лидар замолчал: «не уверен» не может длиться вечно — робот едет дальше, медленно."""
    bot, io = _agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 0.5}]
    bot._trigger = None
    pose = (-1.6, 0.5, 0.4)
    bot.tick(_obs(arena, pose, t=0.0), io)
    for k in (1, 2):
        bot.tick(replace(_obs(arena, pose, t=0.4 * k), scan=np.full(360, np.inf)), io)
    log = _lose(bot, io, arena, pose, 1.0, 1.0 + Agent.UNSURE_MAX_S + 2.0, scans=False)
    stood = [t for t, mode, cmd in log if cmd == (0.0, 0.0)]
    assert stood and max(stood) <= Agent.UNSURE_MAX_S + 0.2
    assert log[-1][2] != (0.0, 0.0) and abs(log[-1][2][0]) <= Agent.ALERT_V
    assert any(e.get('data', {}).get('tag') == 'unsure_timeout' for e in bot.journal.entries)


def test_lost_wait_has_a_deadline(arena):
    """Положение не нашлось за LOST_MAX_S: робот не стоит до разряда, а едет на базу по той позе, какая есть."""
    bot, io = _agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 0.5}]
    bot._trigger = None
    pose = (0.5, 0.5, 0.4)
    bot.tick(_obs(arena, pose, t=0.0), io)
    log = _lose(bot, io, arena, pose, 0.4, 0.4 + Agent.LOST_MAX_S + 8.0)
    lost = [t for t, mode, cmd in log if mode == 'lost']
    assert lost and max(lost) - min(lost) <= Agent.LOST_MAX_S + 0.2
    assert bot._returning and log[-1][1] in ('return', 'escape')
    after = [cmd for t, mode, cmd in log if t > max(lost)]
    assert any(cmd != (0.0, 0.0) for cmd in after) and all(abs(cmd[0]) <= Agent.ALERT_V + 1e-9 for cmd in after)
    assert any(e.get('data', {}).get('tag') == 'blind' for e in bot.journal.entries)


def test_lost_wait_ends_when_charge_for_the_way_home_runs_out(arena):
    """Ревью G2: долгое ожидание при истекающем запасе. Если заряда осталось только на дорогу домой,
    робот не ждёт весь срок, а едет на базу сразу."""
    bot, io = _agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 0.5}]
    bot._trigger = None
    pose = (0.5, 0.5, 0.4)
    bot.tick(_obs(arena, pose, t=0.0), io)
    home = bot._home_cost(pose[0], pose[1])
    log = _lose(bot, io, arena, pose, 0.4, 8.0, battery=home * bot.cfg.reserve_margin + 0.3)
    lost = [t for t, mode, cmd in log if mode == 'lost']
    assert max(lost, default=0.0) <= 3.0 < Agent.LOST_MAX_S
    assert bot._returning and log[-1][1] in ('return', 'escape') and any(cmd != (0.0, 0.0) for t, mode, cmd in log[-30:])


def test_standing_without_a_reason_has_a_deadline(arena):
    """Сторож простоя: если робот по любой причине стоит дольше IDLE_MAX_S, он разворачивается туда, где
    свободно, и строит путь заново. Здесь преграда «вплотную по курсу» (старое правило: вперёд не ехать)."""
    pose = [-1.0, 1.0, 0.0]
    bot, io = _blind_agent(arena), _IO()
    bot.queue = [{'type': 'goto', 'x': 1.0, 'y': 1.0}]         # цель прямо по курсу, доворачивать нечего
    ahead = [(-1.0 + 0.14 + LIDAR_OFFSET, 1.0 + d) for d in (-0.04, -0.02, 0.0, 0.02, 0.04)]
    t, still, worst = 0.0, 0.0, 0.0
    for k in range(400):
        bot.tick(_see(pose, t, _echo_scan(ahead, pose) if k % 4 == 0 else None), io)
        v, w = io.cmd
        still = 0.0 if (abs(v) > 0.01 or abs(w) > 0.05) else still + 0.1
        worst = max(worst, still)
        pose[0] += v * math.cos(pose[2]) * 0.1
        pose[1] += v * math.sin(pose[2]) * 0.1
        pose[2] = math.remainder(pose[2] + w * 0.1, 2 * math.pi)
        t += 0.1
    assert worst <= Agent.IDLE_MAX_S + 0.3
    assert any(e.get('data', {}).get('tag') == 'idle' for e in bot.journal.entries)


@pytest.mark.parametrize('seed', [3, 4, 6, 8])
def test_no_endless_standing_with_scan_faults_and_kicks(arena, seed):
    """Рывки курса и сбои лидара в быстром симуляторе: в каждом прогоне робот либо вернулся, либо до конца
    продолжал двигаться; дольше предела он не стоит нигде."""
    limit = Agent.LOST_MAX_S + Agent.IDLE_MAX_S + 2.0
    scenario = generate('hard', seed, arena)
    world = FastSim(arena, scenario, Rules(), seed=seed, **FAULTS)
    bot = Agent(arena, make_config('adaptive'), n_samples=len(scenario.samples))
    mark, worst = (0.0, world.x, world.y, world.th), 0.0
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
        if math.hypot(world.x - mark[1], world.y - mark[2]) > 0.05 or abs(math.remainder(world.th - mark[3], 2 * math.pi)) > 0.3:
            mark = (world.t, world.x, world.y, world.th)
        worst = max(worst, world.t - mark[0])
    score = world.judge.score()
    assert worst <= limit, f'простой {worst:.0f} с'
    assert score['returned'] or score['reason'] != 'time', score


def test_obstacle_beside_the_hull_does_not_block_straight_escape():
    """Gazebo, hard-7: столб вплотную сбоку. Раньше полоса считалась занятой и вперёд, и назад, робот
    разворачивался на месте, задевал столб и раскачивался. Вдоль преграды можно уехать прямо."""
    pillar = [(0.27 * 0 + 0.15 * math.cos(a), 0.27 + 0.15 * math.sin(a)) for a in np.linspace(-2.6, -0.5, 40)]
    scan = _echo_scan(pillar, (0.0, 0.0, 0.0))               # столб радиусом 0,15 м, до его края 0,12 м влево
    assert np.nanmin(np.where(np.isfinite(scan), scan, np.nan)) < 0.13
    assert corridor_room(scan) == math.inf and corridor_room(scan, back=True) == math.inf
    assert escape_plan(scan, back=True) == (-ESCAPE_V, 0.0)
    assert corridor_room(scan, heading=math.pi / 2) < 0.15   # а повернуться к нему и ехать — нельзя
    ahead = np.full(360, np.inf)
    ahead[20] = 0.33                                         # преграда впереди на краю полосы — по-прежнему помеха
    assert 0.25 < corridor_room(ahead) < 0.33
