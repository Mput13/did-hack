"""P2: правки «бережного расхода» (did/frugal.py) — дорога по хоженому полу, свой замер расхода, сдвиг датчика."""
import math
from dataclasses import fields

import numpy as np
import pytest

from did.agent import PRESETS, V3, Agent, AgentConfig, make_config
from did.arena import load_arena
from did.belief import SampleBelief
from did.config import SCIENCE, Rules
from did.frugal import Frugal, OffsetWatch
from did.robot_io import Observation
from did.runner import run_episode


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _obs(t, x, y, z=None, battery=50.0):
    return Observation(t=t, x=x, y=y, th=0.0, v=0.0, w=0.0, battery=battery, sensor=z, scan=None, events=[],
                       done=False)


def _drive(bot, sample, path, shift=lambda t: 0.0, sigma=0.05, seed=0, dt=0.2, t0=0.0):
    """Провести робота по точкам path, подавая показания датчика от образца sample со сдвигом shift(t)."""
    rng = np.random.default_rng(seed)
    t = t0
    for x, y in path:
        clean = max(0.0, 1.0 - math.dist((x, y), sample) / bot.rules.sensor_range_m) - shift(t)
        z = float(np.clip(clean + rng.normal(0.0, sigma), 0.0, 1.0))
        bot._on_reading(z, _obs(t, x, y))
        t += dt
    return t


def _line(a, b, step=0.04):
    n = max(2, int(math.dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


# --- новое поведение выключено по умолчанию ----------------------------------------------------------

def test_new_behaviour_is_off_by_default():
    cfg = AgentConfig()
    assert not Frugal.wanted(cfg)
    assert cfg.known_ground == 0.0 and cfg.collect_reach == 0.25 and not cfg.own_drain and not cfg.sensor_offset
    for name in ('adaptive', 'adaptive_v2', 'scientist', 'scientist_v2', 'fixed'):
        assert not Frugal.wanted(PRESETS[name]), name
        assert PRESETS[name].collect_reach == 0.25


def test_v3_is_v2_plus_p2_flags():
    for old, new in (('adaptive_v2', 'adaptive_v3'), ('scientist_v2', 'scientist_v3')):
        a, b = PRESETS[old], PRESETS[new]
        changed = {f.name for f in fields(AgentConfig) if getattr(a, f.name) != getattr(b, f.name)}
        assert changed == {'name'} | set(V3), changed
    assert set(V3) <= {'known_ground', 'collect_reach', 'own_drain', 'sensor_offset'}


def test_old_presets_do_not_build_the_module(arena):
    assert Agent(arena, make_config('adaptive_v2'), n_samples=3).frugal is None
    assert Agent(arena, make_config('adaptive_v3'), n_samples=3).frugal is not None


def test_agent_keeps_its_own_rules(arena):
    """Агент v3 не получает правил судьи сверх своих допущений: при других правилах мира его правила прежние."""
    s = run_episode('medium', 3, 'adaptive_v3', rules={'sensor_range_m': 2.5, 'drain_idle_per_s': 0.05},
                    agent_rules={}, save=False)
    assert s['agent_rules'] == Rules().to_dict()


# --- дорога по хоженому полу -------------------------------------------------------------------------

def test_outbound_bias_prefers_driven_floor(arena):
    bot = Agent(arena, make_config('adaptive_v2', known_ground=0.5), n_samples=3)
    for i in range(30):                                # проехал полосу вдоль y = −0.5
        bot.soil.observe(-2.0 + 0.06 * i, -0.5, -2.0 + 0.06 * (i + 1), -0.5, 0.15, 0.3)
    bias = bot.frugal.outbound_bias(None)
    ix, iy = arena.w2g(-1.4, -0.5)
    jx, jy = arena.w2g(1.5, 1.5)
    assert bias[iy, ix] < 1.1 and bias[jy, jx] == pytest.approx(1.5)
    danger = np.full(arena.free.shape, 2.0)
    assert bot.frugal.outbound_bias(danger)[jy, jx] == pytest.approx(3.0)
    off = Agent(arena, make_config('adaptive_v2', sensor_offset=True), n_samples=3)
    assert off.frugal.outbound_bias(danger) is danger          # флажок выключен — цена дороги прежняя


# --- карта образцов: пересчёт сдвинутых показаний ---------------------------------------------------

def test_amend_equals_belief_built_from_corrected_readings(arena):
    rng = np.random.default_rng(1)
    pts = _line((-1.5, -0.5), (-0.5, -0.2))
    zs = [float(np.clip(1.0 - math.dist(p, (-0.6, 0.3)) / 2.0 - 0.2 + rng.normal(0, 0.05), 0, 1)) for p in pts]
    a, b = SampleBelief(arena, 3, 2.0), SampleBelief(arena, 3, 2.0)
    for (x, y), z in zip(pts, zs):
        a.update(x, y, z, 0.05)
    for k, ((x, y), z) in enumerate(zip(pts, zs)):
        b.update(x, y, z + 0.2 if k >= 5 else z, 0.05)
    a.amend(len(pts) - 5, 0.2)
    assert np.allclose(a.p, b.p)
    before = a.p.copy()
    a.amend(0, 0.3)
    assert np.array_equal(a.p, before)


# --- свой замер расхода ------------------------------------------------------------------------------

def test_drain_gauge_raises_trip_cost_only_on_persistent_excess(arena):
    bot = Agent(arena, make_config('adaptive_v2', own_drain=True), n_samples=3)
    f = bot.frugal
    assert f.drain_factor() == 1.0                     # замеров нет
    for _ in range(40):
        f.segment(0.06, 1.05, 1.0)                     # расход как обещала карта (в пределах обычного)
    assert f.drain_factor() == 1.0 and bot._per_m() == bot.rules.drain_per_m
    for _ in range(60):
        f.segment(0.06, 1.5, 1.0)                      # устойчиво в полтора раза выше
    assert f.drain_factor() == pytest.approx(1.5 / Frugal.GAUGE_FREE, rel=0.02)
    assert bot._per_m() == pytest.approx(bot.rules.drain_per_m * f.drain_factor())
    for _ in range(60):
        f.segment(0.06, 9.0, 1.0)
    assert f.drain_factor() == Frugal.GAUGE_MAX        # предел поправки


# --- сдвиг показаний датчика образцов ----------------------------------------------------------------

SAMPLE = (-0.6, 0.2)
PASS = _line((-1.9, -0.3), SAMPLE) + _line(SAMPLE, (0.3, 0.9))[1:]      # проезд прямо по образцу и дальше


def _watcher(arena):
    bot = Agent(arena, make_config('adaptive_v2', sensor_offset=True), n_samples=1)
    return bot, bot.frugal.offset


@pytest.mark.parametrize('seed', range(5))
def test_offset_is_found_when_sensor_reads_low(arena, seed):
    bot, w = _watcher(arena)
    _drive(bot, SAMPLE, PASS, shift=lambda t: 0.2, seed=seed)
    assert w.found == 1
    assert 0.08 <= w.shift <= 0.26                     # не больше настоящего сдвига с запасом на шум
    assert any(e.get('data', {}).get('tag') == 'sensor_offset' for e in bot.journal.entries)


@pytest.mark.parametrize('seed', range(5))
def test_no_offset_on_a_healthy_sensor(arena, seed):
    """Исправный датчик: ни проезд по образцу, ни проезд в 40 см от него сдвигом не считается."""
    for path in (PASS, _line((-1.9, -0.2), (0.5, -0.2)), _line((-1.9, -0.3), (-0.2, 0.9))):
        bot, w = _watcher(arena)
        _drive(bot, SAMPLE, path, seed=seed)
        assert w.found == 0 and w.shift == 0.0


def test_corrected_readings_put_the_sample_where_it_is(arena):
    """С поправкой карта видит образец на месте, без неё — нет: уверенность рядом с образцом заметно выше."""
    conf = {}
    for flag in (False, True):
        bot = Agent(arena, make_config('adaptive_v2', sensor_offset=flag), n_samples=1)
        _drive(bot, SAMPLE, _line((-1.9, -0.3), SAMPLE) + _line(SAMPLE, (-0.2, 0.5))[1:] + _line((-0.2, 0.5), SAMPLE)[1:],
               shift=lambda t: 0.2, seed=3)
        conf[flag] = bot.belief.prob_within(*SAMPLE, 0.25)
    assert conf[True] > 0.5 and conf[True] > conf[False] + 0.3, conf


def test_offset_is_dropped_when_it_stops_or_proves_wrong(arena):
    bot, w = _watcher(arena)
    t = _drive(bot, SAMPLE, PASS, shift=lambda t: 0.2, seed=1)
    assert w.shift > 0.0
    # Сбой кончился: робот возвращается к образцу, поправленное показание становится невозможным (> 1).
    _drive(bot, SAMPLE, _line(PASS[-1], SAMPLE), seed=2, t0=t)
    assert w.shift == 0.0
    # Ложный сбор при поправке снимает её сразу; давно не подтверждённая поправка снимается по времени.
    for end in ('miss', 'time'):
        bot, w = _watcher(arena)
        t = _drive(bot, SAMPLE, PASS, shift=lambda t: 0.2, seed=1)
        assert w.shift > 0.0
        if end == 'miss':
            w.missed(t)
        else:
            x, y = PASS[-1]
            for k in range(int(OffsetWatch.HOLD_S / 0.2) + 5):       # робот стоит: подтвердить нечем
                bot._on_reading(0.3, _obs(t + 0.2 * k, x, y))
        assert w.shift == 0.0


def test_offset_ignores_noisy_and_stuck_sensor(arena):
    bot, w = _watcher(arena)
    _drive(bot, SAMPLE, PASS, sigma=0.25, seed=4)       # сбой «шум»
    assert w.found == 0
    bot, w = _watcher(arena)
    for k, (x, y) in enumerate(PASS):                    # датчик залип
        bot._on_reading(0.62, _obs(0.2 * k, x, y))
    assert w.found == 0


def test_zero_reading_under_offset_is_not_evidence_of_absence(arena):
    bot, w = _watcher(arena)
    t = _drive(bot, SAMPLE, PASS, shift=lambda t: 0.2, seed=1)
    assert w.shift > 0.0
    n = bot.belief.updates
    bot._on_reading(0.0, _obs(t, *PASS[-1]))
    assert bot.belief.updates == n                       # «ноль» при заниженных показаниях в карту не идёт
