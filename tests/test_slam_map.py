"""Режим карты SLAM (бонусный трек F2): перенос сетки, арена из неё, сверка с миром, пульт без готовой карты."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from did.arena import load_arena
from did.nav import INFLATE
from did.pilot import LEG_BASE_S, NO_PROGRESS_S, FastWorld, Pilot
from did.slam_map import (FREE, GRID_STALE_S, JUMP_HOLD_S, OCCUPIED, PILLARS, TF_STALE_S, UNKNOWN, SlamArena, SlamPose,
                          compare, fit_circle, frontiers, origin_offset, outline as ideal_values, pillars, report,
                          resample)

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resample_uses_world_origin_of_the_grid():
    grid = np.full((40, 60), -1, dtype=np.int8)
    grid[10:20, 10:30] = 0
    grid[10, 10:30] = 100
    v = resample(grid, 0.05, -1.0, -0.5)            # угол сетки в мире (−1,0; −0,5)
    assert v.shape == (120, 120)
    iy, ix = np.nonzero(v == OCCUPIED)
    # Занятая строка сетки: y = −0,5 + 10·0,05 = 0,0; x от −0,5 до 0,5.
    assert set(iy) == {60} and ix.min() == 50 and ix.max() == 69
    assert (v == FREE).sum() == 9 * 20 and (v == UNKNOWN).sum() == 120 * 120 - 200
    # Сетка, уходящая за решётку арены, обрезается, а не ломает перенос.
    assert resample(np.zeros((400, 400), np.int8), 0.05, -10.0, -10.0).min() == FREE


def test_slam_arena_knows_only_what_slam_saw():
    ref = load_arena()
    v = ideal_values(ref)
    v[:, 70:] = UNKNOWN                              # правую часть арены SLAM ещё не видел
    a = SlamArena(v, near=(-2.0, -0.5))
    assert a.free[:, :70].sum() > 0 and not a.free[:, 70:].any()
    assert not (a.free & ~ref.free).any()            # ничего лишнего свободным не стало
    assert a.clearance(-2.0, -0.5) > 0.2 and a.clearance(1.5, 0.5) == 0.0
    assert len(frontiers(a)) >= 1                    # за границей увиденного есть куда ехать
    assert frontiers(SlamArena(ideal_values(ref), near=(-2.0, -0.5))) == []    # вся арена видна — границ нет
    assert SlamArena().empty


def test_slam_arena_fills_gaps_between_lidar_rays_but_not_walls():
    v = np.full((120, 120), UNKNOWN, dtype=np.int8)
    v[40:80, 40:80] = FREE
    v[40:80, 60] = UNKNOWN                           # щель между двумя лучами
    v[40:80, 81] = OCCUPIED                          # стена, за ней ничего не видно
    a = SlamArena(v, near=(-0.5, 0.0))
    assert a.free[50, 60] and not a.free[50, 81] and not a.free[50, 82]


def test_pillars_and_origin_on_an_ideal_map():
    ref = load_arena()
    rep = report(ideal_values(ref), ref)
    # Готовая карта сама сдвинута относительно мира на несколько сантиметров (did/localize.py это тоже видит).
    assert rep['pillars_found'] == 9 and rep['pillar_err_max'] < 0.06
    assert rep['agreement'] > 0.999 and rep['coverage'] == 1.0
    assert rep['origin']['shift'] < 0.05 and abs(rep['origin']['rot_deg']) < 0.5
    assert rep['best_shift']['dx'] == 0.0 and rep['best_shift']['dy'] == 0.0


def test_a_map_left_at_the_start_point_is_detected():
    """Если бы начало карты осталось в точке старта робота, столбы уехали бы — сверка это видит."""
    ref = load_arena()
    shifted = np.roll(ideal_values(ref), (2, 4), axis=(0, 1))      # карта сдвинута на (+0,20; +0,10) м
    rep = report(shifted, ref)
    o, o0 = rep['origin'], report(ideal_values(ref), ref)['origin']
    assert o['dx'] - o0['dx'] == pytest.approx(0.20, abs=0.02) and o["dy"] - o0["dy"] == pytest.approx(0.10, abs=0.02)
    assert rep['pillar_err_mean'] == pytest.approx(math.hypot(0.2, 0.1), abs=0.05)
    assert rep['best_shift']['dx'] == pytest.approx(-0.20) and rep['best_shift']['dy'] == pytest.approx(-0.10)
    assert compare(shifted, ref)['agreement'] < 0.9
    # Сдвиг на вектор старта (2,0; 0,5) уводит столбы из окна поиска: карта не проходит совсем.
    far = np.roll(ideal_values(ref), (10, 40), axis=(0, 1))
    assert report(far, ref)['pillars_found'] < 9 or report(far, ref)['pillar_err_max'] > 0.10


def test_fit_circle_from_an_arc():
    ang = np.linspace(0.2, 2.0, 12)                  # столб виден только с одной стороны
    cx, cy = fit_circle(0.3 + 0.15 * np.cos(ang), -0.2 + 0.15 * np.sin(ang))
    assert math.hypot(cx - 0.3, cy + 0.2) < 0.005
    assert len(PILLARS) == 9 and pillars(np.full((120, 120), UNKNOWN, np.int8))[0]['err'] is None
    assert origin_offset(pillars(np.full((120, 120), UNKNOWN, np.int8))) is None


def test_slam_pose_applies_the_transform_from_slam():
    t = [None]
    p = SlamPose(lambda: t[0])
    assert p.update(1.0, 2.0, 0.5) == pytest.approx((1.0, 2.0, 0.5))       # поправки ещё нет — одометрия
    t[0] = (0.1, -0.2, math.pi / 2)
    x, y, th = p.update(1.0, 0.0, 0.0, scan=np.ones(360))
    assert (x, y, th) == pytest.approx((0.1, 0.8, math.pi / 2))
    assert p.to_map(1.0, 0.0, 0.0) == pytest.approx((0.1, 0.8, math.pi / 2))
    assert p.stats['fixes'] == 1 and p.stats['scans'] == 1 and not p.stats['lost']


class _SlamWorld(FastWorld):
    """Быстрый симулятор, в котором «SLAM» изображает карта пульта по лидару: готовой карты в ней нет."""
    pilot = None

    def slam_grid(self):
        m = self.pilot.mapper if self.pilot else None
        if m is None or not m.scans:
            return 0, None
        return m.version // 5 + 1, np.where(m.seen(), np.where(m.occupied(), OCCUPIED, FREE), UNKNOWN).astype(np.int8)

    def slam_tf(self):
        return self.tf

    tf = (0.0, 0.0, 0.0)                    # что отвечает «SLAM» на вопрос о поправке позы; проверки его подменяют


class _GridWorld(FastWorld):
    """Быстрый симулятор, которому сетку «SLAM» и его поправку позы задаёт проверка."""

    def __init__(self, values=None):
        super().__init__(load_arena(), 'medium', 3, drift=False)
        self.ver, self.values = 1, ideal_values(self.arena) if values is None else values
        self.tf = {'tf': (0.0, 0.0, 0.0), 'tf_age': 0.0, 'grid_age': 0.0}

    def slam_grid(self):
        return self.ver, self.values

    def slam_tf(self):
        return self.tf


def _grid_pilot(values=None):
    world = _GridWorld(values)
    pilot = Pilot(world.arena, world, 'medium', 3, slam_nav=True)
    pilot.tick(world.observe())
    return pilot, world


def _hole_at(ref, x, y, r=0.42):
    """Карта, на которой столба в (x, y) нет: SLAM его «не увидел», пол там свободен."""
    v = ideal_values(ref)
    X, Y = ref.cell_centers()
    v[np.hypot(X - x, Y - y) < r] = FREE
    return v


def _run(pilot, world, until, limit):
    for n in range(limit):
        pilot.tick(world.observe())
        world.advance()
        if until():
            return n
    return None


def test_pilot_drives_by_the_slam_map_only():
    ref = load_arena()
    world = _SlamWorld(ref, 'medium', 3, drift=False)
    pilot = Pilot(ref, world, 'medium', 3, slam_nav=True)
    assert pilot.graph is None and pilot.nav.empty
    assert not pilot.command({'cmd': 'route', 'points': [[-1.5, -0.5]]})['ok']     # карты ещё нет
    world.pilot = pilot
    _run(pilot, world, lambda: False, 30)
    assert isinstance(pilot.nav, SlamArena) and pilot.nav is not ref and pilot.graph.arena is pilot.nav
    res = pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})
    assert not res['ok'] and 'не видел' in res['message']       # дальний угол с базы не виден: путь туда не строится

    assert pilot.command({'cmd': 'explore'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 3000) is not None
    assert pilot._explore['done'] and pilot.state()['nav']['explore']['done']
    assert compare(pilot.nav.values, ref)['coverage'] > 0.97
    assert frontiers(pilot.nav) == [] or pilot._explore['visited'] > 0
    assert world.score()['collisions'] == 0

    assert pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})['ok']
    assert pilot.command({'cmd': 'go'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 1500) is not None
    assert math.dist((world.sim.x, world.sim.y), (1.5, 1.2)) < 0.10
    assert world.score()['collisions'] == 0

    # Автономная миссия получает арену из сетки SLAM, а не готовую карту.
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'mission', 2000) is not None
    bot = pilot.bot
    assert isinstance(bot.arena, SlamArena) and bot.graph.arena is bot.arena and bot.belief is not None
    assert bot.tracker is pilot.tracker and isinstance(bot.tracker, SlamPose)
    _run(pilot, world, lambda: False, 100)
    assert pilot.mode == 'mission' and world.score()['collisions'] == 0


def test_pilot_without_the_flag_is_unchanged():
    ref = load_arena()
    world = FastWorld(ref, 'medium', 3)
    pilot = Pilot(ref, world, 'medium', 3)
    assert pilot.nav is ref and pilot.graph.arena is ref and not pilot.slam_nav
    res = pilot.command({'cmd': 'explore'})
    assert not res['ok'] and 'SLAM' in res['message']
    s = pilot.state()
    assert s['slam_nav'] is False and s['nav'] is None and s['fix']['source'] in ('lidar', 'odom')


# =================================================================================================
# ревью, находка 4: «найдено столбов» — только кольцо радиуса столба
# =================================================================================================

def _fake_pillars(ref, paint):
    v = ideal_values(ref)
    X, Y = ref.cell_centers()
    for x, y in PILLARS:
        v[np.hypot(X - x, Y - y) < 0.3] = FREE
        paint(v, x, y, X, Y)
    return v


def _square(v, x, y, X, Y):
    ix, iy = load_arena().w2g(x, y)
    v[iy - 1:iy + 2, ix - 1:ix + 2] = OCCUPIED


def _segment(v, x, y, X, Y):
    ix, iy = load_arena().w2g(x, y + 0.12)
    v[iy, ix - 3:ix + 4] = OCCUPIED


def _disc(v, x, y, X, Y):
    v[np.hypot(X - x, Y - y) < 0.15] = OCCUPIED


@pytest.mark.parametrize('paint', [_square, _segment, _disc])
def test_only_a_ring_of_pillar_radius_counts_as_a_pillar(paint):
    """Квадрат 15×15 см, отрезок стены и сплошное пятно на местах столбов — не столбы."""
    ref = load_arena()
    rep = report(_fake_pillars(ref, paint), ref)
    assert rep['pillars_found'] == 0 and rep['pillar_err_max'] is None and rep['origin'] is None
    assert all(p['found'] is None and p['err'] is None and p['why'] for p in rep['pillars'])
    assert not _load(ROOT / 'tools' / 'slam_check.py', 'slam_check').accepted(rep)


def test_random_cells_are_not_pillars():
    """Россыпь занятых клеток на местах столбов: 100 карт по 9 мест. Случайно кольцом выходит не больше 1 %
    мест (на этих картах — одно из 900), и ни одна карта не проходит приёмку."""
    ref = load_arena()
    check = _load(ROOT / 'tools' / 'slam_check.py', 'slam_check')
    X, Y = ref.cell_centers()
    base = ideal_values(ref)
    for x, y in PILLARS:
        base[np.hypot(X - x, Y - y) < 0.3] = FREE
    found = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        v = base.copy()
        for x, y in PILLARS:
            for _ in range(14):
                ix, iy = ref.w2g(x + rng.uniform(-0.25, 0.25), y + rng.uniform(-0.25, 0.25))
                v[iy, ix] = OCCUPIED
        rep = report(v, ref)
        found += rep['pillars_found']
        assert rep['pillars_found'] <= 1 and not check.accepted(rep)
    assert found <= 9


def test_a_half_seen_pillar_is_not_confirmed_and_real_maps_still_pass():
    ref = load_arena()
    v = ideal_values(ref)
    X, Y = ref.cell_centers()
    v[(v == OCCUPIED) & (X > -1.1) & (np.hypot(X + 1.1, Y) < 0.3)] = UNKNOWN     # столб (−1,1; 0) виден с одной стороны
    rep = report(v, ref)
    half = next(p for p in rep['pillars'] if (p['x'], p['y']) == (-1.1, 0.0))
    assert rep['pillars_found'] == 8 and half['found'] is None and 'три четверти' in half['why']
    check = _load(ROOT / 'tools' / 'slam_check.py', 'slam_check')
    assert not check.accepted(rep)
    # Три сетки, сохранённые в прогонах Gazebo, новой проверкой подтверждаются: числа отчёта не изменились.
    want = {'try1': (0.0155, 0.0205, 0.985), 'run1': (0.0440, 0.0623, 0.960), 'run2': (0.0184, 0.0207, 0.987)}
    for tag, (mean, worst, agree) in want.items():
        z = np.load(ROOT / 'research' / 'findings' / 'F2' / f'{tag}_map.npz')
        rep = report(resample(z['grid'], float(z['res']), float(z['ox']), float(z['oy'])), ref)
        assert rep['pillars_found'] == 9 and check.accepted(rep)
        assert rep['pillar_err_mean'] == pytest.approx(mean, abs=2e-4) and rep['pillar_err_max'] == pytest.approx(worst, abs=2e-4)
        assert rep['agreement'] == pytest.approx(agree, abs=1e-3)
        assert all(p['arc'] >= 10 and p['rms'] <= 0.03 for p in rep['pillars'])


# =================================================================================================
# ревью, находка 3: SLAM замолчал или передумал — это видно
# =================================================================================================

def test_slam_pose_reports_silence_instead_of_constant_ok():
    source = [(0.1, 0.2, 0)]
    p = SlamPose(lambda: source[0])
    p.update(0, 0, 0)
    source[0] = None                                 # преобразование пропало
    for _ in range(1000):
        p.update(1, 2, 0, scan=np.ones(360))
    assert p.to_map(1, 2, 0) == pytest.approx((1.1, 2.2, 0.0))     # последняя поправка остаётся, но верить ей нельзя
    assert p.stats['lost'] and p.lost and p.stats['fault'] and not p.stats['unsure']

    now = [0.0]
    src = {'tf': (0.0, 0.0, 0.0), 'tf_age': 0.02, 'grid_age': 0.5}
    p = SlamPose(lambda: src, clock=lambda: now[0])
    assert not p.stats['ready'] and not p.stats['lost']
    p.update(0, 0, 0)
    assert p.stats['ready'] and not p.stats['lost'] and p.stats['fault'] is None
    src['tf_age'] = TF_STALE_S + 0.1                 # /tf перестал приходить (в ROS последнее значение остаётся)
    p.update(0, 0, 0)
    assert p.stats['lost'] and 'поправки позы нет' in p.stats['fault']
    src['tf_age'], src['grid_age'] = 0.02, GRID_STALE_S + 0.1      # /tf идёт, а карта встала
    p.update(0, 0, 0)
    assert p.stats['lost'] and 'карта не обновлялась' in p.stats['fault']
    src['tf_age'] = -5.0                             # часы симуляции пошли заново: сообщение из прошлого запуска
    p.update(0, 0, 0)
    assert p.stats['lost']
    src['tf_age'], src['grid_age'] = 0.02, 0.5
    p.update(0, 0, 0)
    assert not p.stats['lost'] and p.stats['fault'] is None and p.stats['relocations'] == 0
    # Устаревшее преобразование не применяется, даже если оно другое.
    src['tf'], src['tf_age'] = (1.0, 0.0, 0.0), TF_STALE_S + 1.0
    assert p.update(0, 0, 0) == pytest.approx((0.0, 0.0, 0.0))


def test_slam_pose_sees_jumps_and_slip():
    now = [0.0]
    src = {'tf': (0.0, 0.0, 0.0), 'tf_age': 0.0, 'grid_age': 0.0}
    p = SlamPose(lambda: src, clock=lambda: now[0])
    scan = np.ones(360)
    src['tf'] = (0.3, 0.0, 0.0)                      # первое преобразование — начало отсчёта, не скачок
    p.update(0, 0, 0, scan=scan)
    assert p.stats['relocations'] == 0 and not p.stats['unsure'] and p.stats['slip'] == 0.0
    for k in range(1, 4):                            # обычная работа SLAM: по сантиметру за скан
        now[0] += 0.2
        src['tf'] = (0.3 - 0.01 * k, 0.0, 0.0)
        p.update(1.0, 0, 0, scan=scan)
    assert p.stats['relocations'] == 0 and not p.stats['unsure']
    assert p.stats['slip'] == pytest.approx(0.03) and p.stats['slip_vec'] == pytest.approx((-0.03, 0.0))
    now[0] += 0.2
    src['tf'] = (0.27 - 0.25, 0.0, 0.0)              # поза разом уехала на 25 см: SLAM передумал
    p.update(1.0, 0, 0, scan=scan)
    assert p.stats['relocations'] == 1 and p.stats['unsure'] and not p.stats['lost']
    assert p.stats['slip_vec'][0] == pytest.approx(-0.28)        # против хода: по одометрии едет, по SLAM — нет
    now[0] += JUMP_HOLD_S + 0.1
    p.update(1.0, 0, 0)
    assert not p.stats['unsure'] and p.stats['relocations'] == 1
    for _ in range(5):                               # поправка больше не меняется — признак проскальзывания уходит
        p.update(1.0, 0, 0, scan=scan)
    assert p.stats['slip'] == 0.0
    src['tf'] = (0.02, 0.0, 0.3)                     # поворот на 0,3 рад — тоже скачок, даже вблизи начала координат
    p.update(0.0, 0, 0)
    assert p.stats['relocations'] == 2


def test_silent_slam_stops_manual_driving_and_says_why():
    pilot, world = _grid_pilot()
    assert pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    _run(pilot, world, lambda: False, 30)
    assert pilot.mode == 'drive' and pilot._last_cmd != (0.0, 0.0)
    world.tf = {**world.tf, 'tf_age': TF_STALE_S + 0.5}
    _run(pilot, world, lambda: False, 1)
    assert pilot.mode == 'idle' and pilot._last_cmd == (0.0, 0.0) and not pilot.follower.active
    assert pilot.note['tone'] == 'bad' and 'SLAM Toolbox молчит' in pilot.note['text']
    assert pilot.state()['nav']['slam']['ok'] is False
    for cmd in ({'cmd': 'go'}, {'cmd': 'home'}, {'cmd': 'explore'}, {'cmd': 'mission', 'agent': 'adaptive'},
                {'cmd': 'route', 'points': [[0.5, 0.5]]}):
        res = pilot.command(cmd)
        assert not res['ok'] and 'SLAM Toolbox молчит' in res['message']
    _run(pilot, world, lambda: False, 50)
    assert pilot.mode == 'idle' and world.sim.v == 0.0
    world.tf = {**world.tf, 'tf_age': 0.0}           # SLAM вернулся: оператор видит это и едет дальше
    _run(pilot, world, lambda: False, 1)
    assert 'снова на связи' in pilot.note['text'] and pilot.state()['nav']['slam']['ok'] is True
    assert pilot.command({'cmd': 'go'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 1500) is not None
    assert pilot.arrivals and math.dist((world.sim.x, world.sim.y), (1.5, 1.2)) < 0.10


def test_slam_jump_drops_the_path_and_pauses():
    pilot, world = _grid_pilot()
    assert pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    _run(pilot, world, lambda: False, 30)
    world.tf = {**world.tf, 'tf': (0.2, 0.0, 0.0)}
    _run(pilot, world, lambda: False, 1)
    assert pilot.mode == 'drive' and pilot._last_cmd == (0.0, 0.0) and not pilot.follower.active
    assert pilot.tracker.stats['unsure'] and pilot.state()['nav']['slam']['jumps'] == 1
    _run(pilot, world, lambda: False, int(JUMP_HOLD_S * 10) + 3)
    assert pilot.mode == 'drive' and pilot.follower.active and pilot._last_cmd != (0.0, 0.0)


def _slam_mission():
    pilot, world = _grid_pilot()
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok'] and pilot.mode == 'mission'
    _run(pilot, world, lambda: False, 150)
    assert pilot.bot.mode not in ('lost', 'think') and abs(world.sim.v) > 0.02
    return pilot, world


def _tags(bot):
    return [e.get('data', {}).get('tag') for e in bot.journal.entries]


def test_silent_slam_holds_the_mission_without_a_false_escape():
    """SLAM молчит 15 с посреди миссии: агент стоит, это ожидание известно сторожам простоя, потом едет дальше."""
    pilot, world = _slam_mission()
    bot = pilot.bot
    world.tf = {**world.tf, 'grid_age': GRID_STALE_S + 1.0}
    moved = []
    for _ in range(150):
        pilot.tick(world.observe())
        world.advance()
        moved.append(abs(world.sim.v))
        assert bot.mode == 'lost' and bot._waiting and bot._stuck is None
    assert pilot.mode == 'mission' and pilot.note['tone'] == 'bad' and 'карта не обновлялась' in pilot.note['text']
    assert max(moved[10:]) <= 0.081                  # после торможения — только переползание защиты, не езда по маршруту
    world.tf = {**world.tf, 'grid_age': 0.0}
    _run(pilot, world, lambda: False, 150)
    tags = _tags(bot)
    assert 'lost' in tags and 'relocated' in tags and 'idle' not in tags and 'blind' not in tags
    assert bot.mode != 'lost' and pilot.mode == 'mission' and 'снова на связи' in pilot.note['text']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 6000) is not None        # и доводит миссию до конца
    assert pilot.mission['result']['returned'] and world.score()['collisions'] == 0


def test_slam_that_never_returns_does_not_hang_the_mission():
    """SLAM замолчал насовсем: агент не стоит вечно, а возвращается на базу, и миссия заканчивается."""
    pilot, world = _slam_mission()
    world.tf = None
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 6000) is not None
    assert 'blind' in _tags(pilot.bot)
    assert pilot.mission['state'] == 'finished' and world.score()['collisions'] == 0
    assert math.dist((world.sim.x, world.sim.y), (-2.0, -0.5)) < 0.3


# =================================================================================================
# ревью, находки 1, 2, 5: карта изменилась под едущим роботом
# =================================================================================================

def test_target_that_became_an_obstacle_stops_the_route():
    ref = load_arena()
    pilot, world = _grid_pilot(_hole_at(ref, -1.1, 0.0))
    assert pilot.command({'cmd': 'route', 'points': [[-1.1, 0]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    world.values, world.ver = ideal_values(ref), 2                   # SLAM увидел на месте цели столб
    n = _run(pilot, world, lambda: pilot.mode == 'idle', 800)
    assert n is not None and n <= 5
    assert pilot._last_cmd == (0.0, 0.0) and pilot.arrivals == [] and not pilot.follower.active
    assert pilot.note['tone'] == 'bad' and 'в преграде' in pilot.note['text']
    assert world.score()['collisions'] == 0
    res = pilot.command({'cmd': 'go'})                               # повтор без новой точки кончается тем же сразу
    _run(pilot, world, lambda: False, 2)
    assert res['ok'] and pilot.mode == 'idle' and 'в преграде' in pilot.note['text']


def test_target_slightly_too_close_to_a_wall_is_moved_not_dropped():
    ref = load_arena()
    pilot, world = _grid_pilot(_hole_at(ref, -1.1, 0.0))
    real = SlamArena(ideal_values(ref), near=(-2.0, -0.5))
    X, Y = ref.cell_centers()
    iy, ix = np.nonzero(real.free & (real.clear < INFLATE - 0.02) & (np.hypot(X + 1.1, Y) < 0.4) & (X < -1.1))
    goal = (round(float(X[iy[0], ix[0]]), 3), round(float(Y[iy[0], ix[0]]), 3))     # пол у самого столба
    assert pilot.command({'cmd': 'route', 'points': [list(goal)]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    assert (pilot.route[0]['x'], pilot.route[0]['y']) == goal        # пока столба на карте нет, место разрешённое
    world.values, world.ver = ideal_values(ref), 2                   # столб появился: цель слишком близко к нему
    _run(pilot, world, lambda: False, 2)
    leg = pilot.route[0]
    assert pilot.mode == 'drive' and (leg['x'], leg['y']) != goal
    assert pilot.nav.clearance(leg['x'], leg['y']) >= INFLATE and math.dist((leg['x'], leg['y']), goal) <= 0.15
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 800) is not None and pilot.arrivals


def test_obstacle_missing_from_the_map_ends_the_drive_by_deadline():
    """Столба нет на карте, а в мире он есть: лидар обнуляет скорость. Езда должна кончиться, а не висеть."""
    ref = load_arena()
    pilot, world = _grid_pilot(_hole_at(ref, -1.1, 0.0))
    assert pilot.command({'cmd': 'route', 'points': [[-1.1, 0]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    n = _run(pilot, world, lambda: pilot.mode == 'idle', 800)
    assert n is not None and pilot.arrivals == [] and pilot._last_cmd == (0.0, 0.0)
    assert pilot.note['tone'] == 'bad' and 'не продвигается' in pilot.note['text']
    assert world.score()['collisions'] == 0


class _DeadWheels(_GridWorld):
    def command(self, v, w):
        super().command(0.0, 0.0)                    # команды уходят, робот не едет


def test_every_leg_and_the_whole_exploration_are_bounded_in_time():
    ref = load_arena()
    v = ideal_values(ref)
    v[:, 70:] = UNKNOWN                              # есть границы увиденного, а робот до них не доедет
    world = _DeadWheels(v)
    pilot = Pilot(world.arena, world, 'medium', 3, slam_nav=True)
    pilot.tick(world.observe())
    assert pilot.command({'cmd': 'explore'})['ok']
    legs = len(frontiers(pilot.nav))
    n = _run(pilot, world, lambda: pilot.mode == 'idle', int((legs + 1) * NO_PROGRESS_S * 10) + 50)
    assert n is not None and pilot._explore['done'] and pilot._explore['visited'] <= legs
    assert sum('брошена' in e['text'] for e in pilot.log) == pilot._explore['visited'] >= 1

    # Срок всего подъезда: робот топчется (сдвигается, но не приближается) — подъезд всё равно кончается.
    pilot, world = _grid_pilot()
    assert pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    _run(pilot, world, lambda: False, 5)
    limit = pilot._progress['limit']
    assert LEG_BASE_S < limit < 150.0
    obs = world.observe()
    for k in range(int(limit * 10) + 20):
        if pilot._overdue(type(obs)(**{**obs.__dict__, 't': obs.t + 0.1 * k}), 0.1 * (k // 20 % 2), 0.0):
            break
    assert pilot.mode == 'idle' and 'подъезд занял больше' in pilot.note['text']


def test_unusable_new_map_takes_the_old_one_out_of_service():
    pilot, world = _grid_pilot()
    pilot.pose = (-2.0, -0.5, 0.0)
    old = pilot.graph
    pilot.follower.set_path([(-2.0, -0.5), (-1.5, -0.5)])
    world.ver, world.values = 2, np.full((120, 120), OCCUPIED, np.int8)
    pilot._refresh_nav()
    view = pilot._nav_view()
    assert pilot.graph is None and pilot.graph is not old and not pilot.follower.active
    assert view['version'] == 2 and view['ready'] is False and view['free_m2'] == 0.0
    res = pilot.command({'cmd': 'route', 'points': [[-1.5, -0.5]]})
    assert not res['ok'] and 'карты' in res['message']

    # То же на ходу: езда прекращается с сообщением; с пригодной картой оператор едет снова.
    pilot, world = _grid_pilot()
    assert pilot.command({'cmd': 'route', 'points': [[1.5, 1.2]]})['ok'] and pilot.command({'cmd': 'go'})['ok']
    _run(pilot, world, lambda: False, 30)
    world.ver, world.values = 2, np.full((120, 120), OCCUPIED, np.int8)
    _run(pilot, world, lambda: False, 1)
    assert pilot.mode == 'idle' and pilot._last_cmd == (0.0, 0.0) and pilot.graph is None
    assert pilot.note['tone'] == 'bad' and 'непригодна' in pilot.note['text']
    assert not pilot.command({'cmd': 'go'})['ok'] and not pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    assert pilot.state()['nav']['ready'] is False
    world.ver, world.values = 3, ideal_values(load_arena())
    _run(pilot, world, lambda: False, 1)
    assert pilot.state()['nav']['ready'] is True and 'снова пригодна' in pilot.note['text']
    assert pilot.command({'cmd': 'go'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 1500) is not None and pilot.arrivals


def test_unusable_map_during_a_mission_stops_the_agent():
    """Ревью, круг 2: в миссии агент едет по снимку карты, но свежая непригодная сетка запрещает движение."""
    pilot, world = _slam_mission()
    bot, good = pilot.bot, world.values
    world.ver, world.values = 2, np.full((120, 120), OCCUPIED, np.int8)
    moved = []
    for _ in range(50):
        pilot.tick(world.observe())
        world.advance()
        moved.append(abs(world.sim.v))
        assert bot.mode == 'lost' and bot._waiting and bot._stuck is None
    assert max(moved[10:]) <= 0.081                  # после торможения — только переползание защиты
    view = pilot.state()['nav']
    assert view['ready'] is False and view['slam']['ok'] is False and 'непригодна' in view['slam']['fault']
    assert pilot.mode == 'mission' and pilot.note['tone'] == 'bad' and 'непригодна' in pilot.note['text']
    world.ver, world.values = 3, good                # пригодная карта вернулась: агент едет дальше
    _run(pilot, world, lambda: False, 150)
    assert bot.mode != 'lost' and 'idle' not in _tags(bot) and pilot.state()['nav']['ready'] is True
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 6000) is not None
    assert pilot.mission['result']['returned'] and world.score()['collisions'] == 0


def test_map_that_stays_unusable_does_not_hang_the_mission():
    """Карта так и не стала пригодной: агент не стоит вечно, а возвращается на базу, как при молчании SLAM."""
    pilot, world = _slam_mission()
    world.ver, world.values = 2, np.full((120, 120), OCCUPIED, np.int8)
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 6000) is not None
    assert 'blind' in _tags(pilot.bot)
    assert pilot.mission['state'] == 'finished' and world.score()['collisions'] == 0
    assert math.dist((world.sim.x, world.sim.y), (-2.0, -0.5)) < 0.3
    # вне миссии та же сетка снимает карту с работы обычным порядком
    _run(pilot, world, lambda: False, 2)
    assert pilot.graph is None and not pilot.command({'cmd': 'route', 'points': [[-1.5, -0.5]]})['ok']


def test_path_is_dropped_by_the_same_margin_the_graph_is_built_with():
    ref = load_arena()
    pilot, world = _grid_pilot()
    clear = pilot.nav.clear
    X, Y = ref.cell_centers()
    iy, ix = np.nonzero((clear >= 0.11) & (clear < INFLATE - 0.01))
    k = int(np.argmin(np.hypot(X[iy, ix] + 1.5, Y[iy, ix] + 0.5)))
    near = (float(X[iy[k], ix[k]]), float(Y[iy[k], ix[k]]))        # 11–16 см до преграды: граф туда не ведёт
    assert not pilot._path_ok([(-2.0, -0.5), near])
    # Обе точки в порядке, а отрезок между ними идёт через столб.
    assert pilot.nav.clearance(-1.6, 0.0) >= INFLATE and pilot.nav.clearance(-0.6, 0.0) >= INFLATE
    assert not pilot._path_ok([(-1.6, 0.0), (-0.6, 0.0)])
    pts, _ = pilot.graph.plan((-2.0, -0.5), (1.5, 1.2))
    assert pilot._path_ok(pts + [(1.5, 1.2)])                        # путь самого графа проверку проходит
    pilot.follower.set_path([(-2.0, -0.5), near])
    world.ver = 2
    pilot._refresh_nav()
    assert not pilot.follower.active


# =================================================================================================
# замечания ревью о проверках: миссия до конца и обычный режим против main
# =================================================================================================

def test_mission_on_the_slam_map_runs_to_the_end():
    ref = load_arena()
    world = _SlamWorld(ref, 'medium', 3, drift=False)
    pilot = Pilot(ref, world, 'medium', 3, slam_nav=True)
    world.pilot = pilot
    _run(pilot, world, lambda: False, 30)
    assert pilot.command({'cmd': 'explore'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 3000) is not None
    assert pilot.command({'cmd': 'mission', 'agent': 'adaptive'})['ok']
    assert _run(pilot, world, lambda: pilot.mode == 'mission', 2000) is not None
    assert _run(pilot, world, lambda: pilot.mode == 'idle', 8000) is not None
    r = pilot.mission['result']
    assert pilot.mission['state'] == 'finished' and r['returned'] and r['reason'] == 'finish'
    assert r['samples_collected'] >= 3 and r['collisions'] == 0
    assert math.dist((world.sim.x, world.sim.y), (-2.0, -0.5)) < 0.3
    assert not {'lost', 'blind', 'slip', 'idle'} & set(_tags(pilot.bot))    # защита зря не срабатывала


def test_normal_mode_matches_main():
    """Обычный режим (готовая карта): те же команды дают те же позы, что код main, с которого снят эталон."""
    script = _load(ROOT / 'tests' / 'data' / 'pilot_main_poses.py', 'pilot_main_poses')
    want = json.loads((ROOT / 'tests' / 'data' / 'pilot_main_poses.json').read_text())['rows']
    got = script.run()
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.keys() == w.keys() and g['tick'] == w['tick']
        if 'cmd' in w:
            assert g == w
            continue
        assert g['mode'] == w['mode'] and g['arrivals'] == w['arrivals'], w['tick']
        for key in ('pose', 'true', 'cmd_vw'):
            assert g[key] == pytest.approx(w[key], abs=1e-6), (w['tick'], key)
    assert {r['mode'] for r in want if 'mode' in r} == {'idle', 'drive', 'home', 'mission'}
