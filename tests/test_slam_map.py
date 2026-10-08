"""Режим карты SLAM (бонусный трек F2): перенос сетки, арена из неё, сверка с миром, пульт без готовой карты."""
import math

import numpy as np
import pytest

from did.arena import load_arena
from did.pilot import FastWorld, Pilot
from did.slam_map import (FREE, OCCUPIED, PILLARS, UNKNOWN, SlamArena, SlamPose, compare, fit_circle, frontiers,
                          origin_offset, outline as ideal_values, pillars, report, resample)


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
        return None


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
