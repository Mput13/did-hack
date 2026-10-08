"""Проверки локализации по лидару (did/localize.py) и модели ухода одометрии в быстром симуляторе."""
import math
from dataclasses import replace

import numpy as np
import pytest

from did.agent import make_config
from did.arena import load_arena
from did.config import BASE, LIDAR_MAX, LIDAR_MIN, LIDAR_SIGMA, Rules
from did.fastsim import LIDAR_OFFSET, FastSim
from did.localize import PoseTracker
from did.nav import CostGraph, Follower
from did.recorder import Recorder
from did.robot_io import Observation
from did.runner import run_episode
from did.scenario import generate

# Уход, какой виден в Gazebo, когда колёса проскальзывают на каждом резком развороте.
STRONG = dict(odom_turn_slip=1.0, odom_turn_scale=0.03, odom_path_scale=0.03)
TOUR = [(-1.6, 0.6), (-0.55, 1.6), (0.55, 0.55), (1.6, 0.55), (0.55, -0.55), (1.6, -1.2), (-0.55, -1.6),
        (-0.55, -0.55), (-1.6, -0.6), (-0.55, 0.55), (0.55, 1.6), (1.7, 0.0), (0.55, -1.6), (-1.7, -0.5)]


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _scan(arena, x, y, th, rng=None):
    """Скан из истинной позы — так же, как его строит быстрый симулятор."""
    r = arena.raycast(x + LIDAR_OFFSET * math.cos(th), y + LIDAR_OFFSET * math.sin(th), th)
    if rng is not None:
        r = r + rng.normal(0.0, LIDAR_SIGMA, r.shape)
    r[(r < LIDAR_MIN) | (r > LIDAR_MAX)] = np.inf
    return r


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _drive(arena, sim, seconds=100.0, enabled=True):
    """Объезд арены по точкам: возвращает ошибки позы по одометрии и после поправки, трекер и время на скан.

    С поправкой робот едет по исправленной позе, как агент; без неё — по истинной (иначе он быстро
    упрётся в стену и уход одометрии будет не с чем сравнить).
    """
    rules = Rules(battery_start=1e6)
    world = FastSim(arena, generate('easy', 1, arena), rules, seed=3, **sim)
    graph, follower, tracker = CostGraph(arena), Follower(), PoseTracker(arena, enabled=enabled)
    goals = list(TOUR)
    raw, fixed, heading, ms = [], [], [], []
    while world.t < seconds:
        obs = world.observe()
        x, y, th = tracker.update(obs.x, obs.y, obs.th, obs.scan)
        if obs.scan is not None:
            ms.append(tracker.stats['ms'])
        raw.append(math.hypot(obs.x - world.x, obs.y - world.y))
        fixed.append(math.hypot(x - world.x, y - world.y))
        heading.append(abs(_wrap(th - world.th)))
        if not enabled:
            x, y, th = world.x, world.y, world.th
        if not follower.active or follower.step(x, y, th, tol=0.12)[2]:
            goal = goals.pop(0)
            goals.append(goal)
            follower.set_path(graph.plan((x, y), goal)[0])
        v, w, _ = follower.step(x, y, th, tol=0.12)
        world.command(v, w)
        world.advance()
    return np.array(raw), np.array(fixed), np.array(heading), tracker, np.array(ms)


# --- модель ухода одометрии ----------------------------------------------------------------------

def test_odometry_is_exact_without_drift(arena):
    raw, fixed, heading, tracker, _ = _drive(arena, {}, seconds=40.0, enabled=False)
    assert raw.max() == 0.0 and fixed.max() == 0.0 and heading.max() == 0.0


def test_drift_model_makes_odometry_wander(arena):
    slip, _, _, _, _ = _drive(arena, dict(odom_turn_slip=1.0), enabled=False)
    scale, _, _, _, _ = _drive(arena, dict(odom_turn_scale=0.03, odom_path_scale=0.03), enabled=False)
    walk, _, _, _, _ = _drive(arena, dict(odom_drift=0.3), enabled=False)
    assert slip.max() > 0.3          # проскальзывание при резких разворотах уводит курс, а с ним и положение
    assert scale.max() > 0.1
    assert walk.max() > 0.05

    # Плавный разворот одометрию не портит: колёса не проскальзывают, пока корпус успевает за ними.
    world = FastSim(arena, generate('easy', 1, arena), seed=1, odom_turn_slip=1.0)
    for k in range(60):
        world.command(0.0, 1.5 * math.sin(math.pi * k / 60))
        world.advance()
    smooth = abs(_wrap(world.observe().th - world.th))
    world = FastSim(arena, generate('easy', 1, arena), seed=1, odom_turn_slip=1.0)
    for k in range(60):
        world.command(0.0, 1.9 if k < 20 else -1.9 if k < 40 else 0.0)
        world.advance()
    sharp = abs(_wrap(world.observe().th - world.th))
    assert smooth < 0.03 < 0.3 < sharp


# --- трекер на отдельных сканах ------------------------------------------------------------------

def test_tracker_does_nothing_when_disabled_or_without_scan(arena):
    assert PoseTracker(arena, enabled=False).update(1.0, 2.0, 0.3, _scan(arena, *BASE, 0.0)) == (1.0, 2.0, 0.3)
    tracker = PoseTracker(arena)
    assert tracker.update(-2.0, -0.5, 0.0, None) == pytest.approx((-2.0, -0.5, 0.0))
    assert tracker.stats['scans'] == 0


@pytest.mark.parametrize('true, off', [
    ((-1.6, 0.5, 0.4), (0.10, -0.06, 0.08)),
    ((0.5, 0.6, 2.5), (-0.08, 0.10, -0.10)),
    ((1.5, -1.4, -1.2), (0.12, 0.05, 0.05)),
    ((-0.5, -1.7, 3.0), (0.00, 0.00, 0.30)),          # курс ушёл рывком: находится перебором
])
def test_tracker_pulls_wrong_odometry_to_true_pose(arena, true, off):
    rng = np.random.default_rng(1)
    tracker = PoseTracker(arena)
    tracker._calib = None                               # робот не на старте: сдвиг карты не измеряем
    odom = (true[0] + off[0], true[1] + off[1], true[2] + off[2])
    for _ in range(25):                                 # робот стоит, сканы идут: 10 с при 2,5 Гц
        x, y, th = tracker.update(*odom, _scan(arena, *true, rng))
    assert math.hypot(x - true[0], y - true[1]) < 0.02
    assert abs(_wrap(th - true[2])) < 0.015
    assert tracker.stats['fixes'] > 0 and tracker.stats['inliers'] > 0.9 and tracker.stats['residual'] < 0.02
    dx, dy, dth = tracker.stats['fix']
    assert (dx, dy, dth) == pytest.approx((x - odom[0], y - odom[1], _wrap(th - odom[2])))


def test_correction_is_limited_per_scan(arena):
    tracker = PoseTracker(arena)
    tracker._calib = None
    true = (-1.6, 0.5, 0.4)
    x, y, th = tracker.update(true[0] + 0.15, true[1], true[2] + 0.1, _scan(arena, *true))
    assert math.hypot(x - true[0] - 0.15, y - true[1]) <= PoseTracker.STEP_XY + 1e-9
    assert abs(_wrap(th - true[2] - 0.1)) <= PoseTracker.STEP_TH + 1e-9


def test_bad_scans_are_skipped(arena):
    rng = np.random.default_rng(2)
    pose = (-1.6, 0.5, 0.4)
    blind = np.full(360, np.inf)
    blind[:30] = _scan(arena, *pose)[:30]               # почти все лучи без отражения
    junk = rng.uniform(0.3, 3.0, 360)                   # отражения, которых на карте нет
    crowd = _scan(arena, *pose)
    crowd[::2] = rng.uniform(0.15, 0.5, 180)            # половину лучей закрыло что-то постороннее
    for scan in (blind, junk, crowd):
        tracker = PoseTracker(arena)
        tracker._calib = None
        for _ in range(8):
            assert tracker.update(pose[0] + 0.05, pose[1], pose[2], scan) == pytest.approx((pose[0] + 0.05, *pose[1:]))
        assert tracker.stats['fixes'] == 0 and tracker.stats['skipped'] == 8


def test_no_global_jump(arena):
    """Арена симметрична, похожих мест много: далеко от предсказанной позы трекер совпадение не ищет."""
    tracker = PoseTracker(arena)
    tracker._calib = None
    odom = (-1.6, 0.5, 0.4)
    for _ in range(10):
        x, y, th = tracker.update(*odom, _scan(arena, 0.9, 0.5, 0.4))     # робот «на самом деле» в 2,5 м
    assert math.hypot(x - odom[0], y - odom[1]) <= 10 * PoseTracker.STEP_XY
    assert math.hypot(x - 0.9, y - 0.5) > 2.0


def test_map_shift_is_measured_at_start(arena):
    """Карта сдвинута относительно мира на пару сантиметров (как в Gazebo): на старте это не уход одометрии."""
    shifted = replace_origin(arena, 0.025, 0.013)
    rng = np.random.default_rng(3)
    tracker = PoseTracker(arena)
    for _ in range(4):
        pose = tracker.update(*BASE, 0.0, _scan(shifted, *BASE, 0.0, rng))
        assert pose == pytest.approx((*BASE, 0.0))                          # на старте поза не трогается
    for k in range(1, 30):                                                  # поехали: калибровка закончена
        true = (BASE[0] + 0.02 * k, BASE[1], 0.0)
        pose = tracker.update(*true, _scan(shifted, *true, rng))
        assert math.hypot(pose[0] - true[0], pose[1] - true[1]) < 0.012
    assert tracker.shift == pytest.approx((-0.025, -0.013), abs=0.006)


def replace_origin(arena, dx, dy):
    """Та же арена, сдвинутая в мире на (dx, dy): «настоящие» стены для лидара."""
    import copy
    world = copy.copy(arena)
    world.x0, world.y0 = arena.x0 + dx, arena.y0 + dy
    return world


# --- трекер в движении ---------------------------------------------------------------------------

def test_drifting_odometry_is_corrected(arena):
    raw, fixed, heading, tracker, ms = _drive(arena, STRONG)
    assert raw[-1] > 0.4                                 # без поправки к концу 100 с робот «потерян»
    assert np.median(fixed) <= 0.05 and np.percentile(fixed, 95) <= 0.10
    assert np.median(heading) <= 0.05
    assert tracker.stats['skipped'] <= 0.1 * tracker.stats['scans']
    assert np.median(ms) < 3.0                           # прогоны идут сотнями: скан должен быть дешёвым


def test_ideal_odometry_is_not_degraded(arena):
    raw, fixed, heading, tracker, _ = _drive(arena, {})
    assert raw.max() == 0.0
    assert fixed.max() <= 0.02 and heading.max() <= 0.02
    assert np.median(fixed) <= 0.005


def test_mild_drift_and_random_walk_are_corrected(arena):
    for sim in (dict(odom_turn_slip=0.2, odom_turn_scale=0.01, odom_path_scale=0.01), dict(odom_drift=0.3)):
        raw, fixed, _, _, _ = _drive(arena, sim)
        assert raw.max() > 0.08
        assert np.median(fixed) <= 0.03 and np.percentile(fixed, 95) <= 0.06


# --- агент целиком -------------------------------------------------------------------------------

def test_agent_flag_and_record(arena):
    assert make_config('adaptive').localize is True
    assert make_config('adaptive', localize=False).localize is False
    rec = Recorder()
    obs = Observation(t=0.0, x=0.0, y=0.0, th=0.0, v=0.0, w=0.0, battery=60.0, sensor=None, scan=None)
    for k in range(25):
        rec.sample(replace(obs, t=0.1 * k), 'travel', fix=(0.01, -0.02, 0.003))
    assert [p['t'] for p in rec.pose_fix] == [0.0, 1.0, 2.0]
    assert rec.pose_fix[0] == {'t': 0.0, 'dx': 0.01, 'dy': -0.02, 'dth': 0.003}
    assert all(p['t'] in rec.track['t'] for p in rec.pose_fix)
    built = rec.build(run_id='x', experiment='x', arm='x', backend='fastsim', agent={}, scenario={}, rules={},
                      result={}, world=[], journal=type('J', (), {'entries': [], 'hypotheses': []})())
    assert built['pose_fix'] == rec.pose_fix
    assert 'pose_fix' not in Recorder().build(run_id='x', experiment='x', arm='x', backend='fastsim', agent={},
                                              scenario={}, rules={}, result={}, world=[],
                                              journal=type('J', (), {'entries': [], 'hypotheses': []})())


def test_agent_survives_drift_only_with_localization():
    base = run_episode('medium', 1, 'adaptive', save=False)['metrics']
    same = run_episode('medium', 1, 'adaptive', save=False, config={'localize': False})['metrics']
    fixed = run_episode('medium', 1, 'adaptive', save=False, sim=STRONG)['metrics']
    lost = run_episode('medium', 1, 'adaptive', save=False, sim=STRONG, config={'localize': False})['metrics']
    assert base['score'] == same['score']                # при точной одометрии локализация ничего не меняет
    assert fixed['samples_collected'] >= base['samples_collected'] - 1
    assert fixed['returned'] and fixed['collisions'] <= base['collisions'] + 1
    assert lost['score'] <= fixed['score'] - 15          # без поправки: не вернулся, потерял образцы, бился о стены


# --- симметрия арены, потеря и поиск положения (исследование G2) ----------------------------------

def _turned(pose, deg):
    """Та же поза, повёрнутая вокруг центра арены: стены шестиугольника при этом совпадают сами с собой."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * pose[0] - s * pose[1], s * pose[0] + c * pose[1], pose[2] + math.radians(deg)


@pytest.mark.parametrize('true, deg', [((-1.9, -1.2, 0.7), 120), ((-1.9, 1.25, 0.7), -120), ((-1.9, 0.2, 0.7), 180)])
def test_symmetric_false_match_is_not_accepted(arena, true, deg):
    """Одометрия думает, что робот в позе, повёрнутой вокруг центра арены. Стены оттуда выглядят так же,
    столбы — нет. Прежний трекер такое совмещение принимал и подтверждал ложную позу; теперь — нет."""
    guess = _turned(true, deg)
    scan = _scan(arena, *true)
    old, new = PoseTracker(arena, verify=False), PoseTracker(arena)
    old._calib = new._calib = None
    for _ in range(12):
        old.update(*guess, scan)
        pose = new.update(*guess, scan)
    assert old.stats['fixes'] > 0 and old.stats['skipped'] < 12        # прежний трекер считал, что всё сошлось
    assert new.stats['fixes'] == 0 and new.stats['skipped'] == 12
    assert new.lost and new.stats['lost']
    assert pose == pytest.approx((guess[0], guess[1], _wrap(guess[2])))   # и позу не трогает: искать негде


def test_two_way_check_sees_through_pillars(arena):
    true = (-1.9, -1.2, 0.7)
    tracker = PoseTracker(arena)
    assert tracker._agree(true, _scan(arena, *true), None) > 0.97
    assert tracker._agree(_turned(true, 120), _scan(arena, *true), None) < PoseTracker.MIN_AGREE
    half = _scan(arena, *true)[::2]                                      # скан с другим числом лучей
    assert tracker._agree(true, half, 2 * math.pi / 180) > 0.97


@pytest.mark.parametrize('true, off', [
    ((0.85, -0.85, -1.0), (0.30, -0.25, 2.3)),         # у столба, курс по одометрии ушёл на 130°
    ((-1.5, 0.4, 2.0), (-0.35, 0.30, -1.6)),
    ((1.6, 0.5, 0.3), (0.0, 0.45, 3.0)),
])
def test_lost_pose_is_found_again(arena, true, off):
    """Колёса прокрутились без сцепления: одометрия уехала на десятки сантиметров и больше чем на 90°."""
    rng = np.random.default_rng(4)
    tracker = PoseTracker(arena)
    tracker._calib = None
    for _ in range(3):
        tracker.update(*true, _scan(arena, *true, rng))
    odom = (true[0] + off[0], true[1] + off[1], true[2] + off[2])
    seen_lost = False
    for k in range(12):
        x, y, th = tracker.update(*odom, _scan(arena, *true, rng))
        seen_lost = seen_lost or tracker.stats['lost']
        if k < PoseTracker.LOST_SCANS - 1:              # пока не уверен, что потерялся, позу не трогает
            assert (x, y, th) == pytest.approx((odom[0], odom[1], _wrap(odom[2])))
    assert seen_lost and not tracker.lost and tracker.stats['relocations'] == 1
    assert math.hypot(x - true[0], y - true[1]) < 0.03 and abs(_wrap(th - true[2])) < 0.03
    assert tracker.stats['slip'] < 0.05                 # скачок после потери — не «проскальзывание»


def test_garbage_scans_do_not_move_pose(arena):
    """Робот раскачивается: лидар бьёт в пол и поверх стен. Поза стоит, а когда сканы вернутся — находится."""
    rng = np.random.default_rng(5)
    true = (0.85, -0.85, -1.0)
    tracker = PoseTracker(arena)
    tracker._calib = None
    tracker.update(*true, _scan(arena, *true, rng))
    odom = (true[0] + 0.2, true[1] - 0.2, true[2] + 1.2)
    for _ in range(40):
        junk = np.where(rng.random(360) < 0.5, rng.uniform(0.15, 1.0, 360), np.inf)
        pose = tracker.update(*odom, junk)
    assert tracker.lost and tracker.stats['relocations'] == 0 and tracker.stats['fixes'] <= 1
    assert pose == pytest.approx((odom[0], odom[1], _wrap(odom[2])))
    for _ in range(4):
        x, y, th = tracker.update(*odom, _scan(arena, *true, rng))
    assert not tracker.lost and math.hypot(x - true[0], y - true[1]) < 0.03 and abs(_wrap(th - true[2])) < 0.03


def test_degenerate_geometry_is_not_a_loss(arena, monkeypatch):
    """Скан лёг на карту, но сдвиг вдоль стены не определяет: поправлять нечего, но это и не потеря положения."""
    tracker = PoseTracker(arena)
    tracker._calib = None
    pose = (-1.6, 0.5, 0.4)
    fit = tracker._fit

    def loose(*args, **kw):                              # та же подгонка, но «положение держится только на привязке»
        found, quality = fit(*args, **kw)
        return found, (*quality[:3], False)

    monkeypatch.setattr(tracker, '_fit', loose)
    for _ in range(8):
        assert tracker.update(pose[0] + 0.03, pose[1], pose[2], _scan(arena, *pose)) == pytest.approx((pose[0] + 0.03, *pose[1:]))
    assert tracker.stats['skipped'] == 8 and tracker.stats['fixes'] == 0
    assert not tracker.stats['unsure'] and not tracker.lost


def test_good_degenerate_scans_clear_the_doubt(arena, monkeypatch):
    """Ревью G2: два плохих скана, а дальше сканы сходятся с картой, но сдвиг не определяют. Сомнение
    («не уверен») должно сняться: раньше счётчик плохих сканов оставался на двух навсегда и робот стоял."""
    tracker = PoseTracker(arena)
    tracker._calib = None
    pose = (-1.6, 0.5, 0.4)
    tracker.update(*pose, _scan(arena, *pose))
    for _ in range(2):
        tracker.update(*pose, np.full(360, np.inf))
    assert tracker.stats['unsure'] and not tracker.lost
    fit = tracker._fit

    def loose(*args, **kw):
        found, quality = fit(*args, **kw)
        return found, (*quality[:3], False)

    monkeypatch.setattr(tracker, '_fit', loose)
    for _ in range(100):
        tracker.update(*pose, _scan(arena, *pose))
    assert tracker._agree(pose, _scan(arena, *pose), None) == 1.0
    assert tracker._lost == 0 and not tracker.stats['unsure'] and not tracker.lost
