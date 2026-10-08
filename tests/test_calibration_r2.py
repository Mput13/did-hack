"""Самокалибровка (R14), второй круг: проверки на находки ревью."""
import math

import numpy as np
import pytest

from did.arena import load_arena
from did.belief import SampleBelief

SIGMA = 0.05


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def test_evidence_is_continuous_in_power(arena):
    """Находка 2: «рядом пусто» весит одинаково при любом законе — у p = 1 нет скачка правдоподобия."""
    ev = [SampleBelief(arena, 0, 2.0, power=p).evidence(0.0, 0.0, 0.1, SIGMA) for p in (0.999999, 1.0, 1.000001)]
    assert ev[1] == pytest.approx(ev[0], abs=1e-6) and ev[1] == pytest.approx(ev[2], abs=1e-6)
    noise = np.random.default_rng(1)
    maps = [SampleBelief(arena, 5, 2.0, power=p) for p in (0.999, 1.0, 1.001)]
    for i in range(60):
        x, y = noise.uniform(-2.0, 2.0, 2)
        z = (0.0, 1.0, float(noise.uniform(0.02, 0.98)), 0.05)[i % 4]
        e = [m.evidence(x, y, z, SIGMA) for m in maps]
        assert e[1] == pytest.approx(e[0], abs=0.02) and e[1] == pytest.approx(e[2], abs=0.02)
        assert maps[1].update(x, y, z, SIGMA) == pytest.approx(e[1], abs=1e-9)
        for m in maps:                          # карты держим одинаковыми: сравнивается только счёт показания
            m.p = maps[1].p.copy()


# --- находка 1: обычный запуск не сообщает калибрующемуся агенту правил мира ---------------------

def _first_tick(agent, **kw):
    """Агент и судья на первом такте прогона."""
    from did import runner
    seen = {}
    original = runner.make_agent

    def make_agent(name, config=None):
        cls, cfg = original(name, config)

        class Probe(cls):
            def tick(self, obs, io):
                seen.update(bot=self, judge=io.judge)
                raise RuntimeError('probe')

        return Probe, cfg

    runner.make_agent = make_agent
    try:
        with pytest.raises(RuntimeError, match='probe'):
            runner.run_episode('hard', 1101, agent, save=False, **kw)
    finally:
        runner.make_agent = original
    return seen['bot'], seen['judge']


WORLD = {'sensor_range_m': 1.5, 'drain_per_m': 3.75, 'sensor_law': 'sqrt'}


@pytest.mark.parametrize('agent', ['adaptive_cal', 'adaptive_cal_v2'])
def test_plain_run_does_not_tell_world_rules(agent):
    """Без agent_rules калибрующийся агент стартует с допущений, а не с дальности, закона и расхода мира."""
    bot, judge = _first_tick(agent, rules=WORLD)
    assert bot.rules is not judge.rules
    assert judge.rules.sensor_range_m == 1.5 and judge.rules.drain_per_m == 3.75
    assert (bot.rules.sensor_range_m, bot.rules.sensor_law, bot.rules.drain_per_m) == (2.0, 'linear', 2.5)
    assert bot.cal.law == (2.0, 1.0) and bot.cal.per_m == 2.5 and bot.cal.meter.nominal == 2.5
    assert bot.belief.range == 2.0 and bot.belief.power == 1.0 and bot.soil.per_m == 2.5


def test_plain_run_same_as_explicit_assumptions():
    """Запуск без agent_rules и запуск с agent_rules={} — один и тот же прогон."""
    from did.runner import run_episode
    a = run_episode('medium', 7, 'adaptive_cal', save=False, rules=WORLD)['metrics']
    b = run_episode('medium', 7, 'adaptive_cal', save=False, rules=WORLD, agent_rules={})['metrics']
    assert a == b


def test_other_variants_keep_their_rules():
    """Семантика прежних вариантов не тронута: без agent_rules они получают правила мира, ориентир — тоже."""
    for agent in ('adaptive', 'adaptive_v2', 'adaptive_known'):
        bot, judge = _first_tick(agent, rules=WORLD)
        assert bot.rules is judge.rules and bot.cal is None
    bot, _ = _first_tick('adaptive_known', rules=WORLD)
    assert bot.belief.range == 1.5 and bot.belief.power == 0.5
    bot, _ = _first_tick('adaptive_cal', rules=WORLD, agent_rules={'sensor_range_m': 1.8})
    assert bot.cal.law == (1.8, 1.0)            # явно заданное допущение — это допущение, его не трогаем


# --- находка 3: образец ищется в круге сбора, а не в квадрате --------------------------------------

def test_fit_law_keeps_sample_inside_collect_circle():
    from did.calibrate import fit_law, response
    noise = np.random.default_rng(5)
    sample = np.array([0.29, 0.29])                       # вне круга 0,3 м от точки сбора (0, 0): 0,41 м
    s = np.linspace(0.0, 1.0, 140)[:, None]
    path = np.array([-1.5, -1.2]) * (1.0 - s) + noise.normal(0.0, 0.01, (140, 2))
    z = np.clip(response(np.hypot(*(path - sample).T), 2.0, 1.0) + noise.normal(0.0, SIGMA, 140), 0.0, 1.0)
    fit = fit_law([{'x': 0.0, 'y': 0.0, 'pts': np.column_stack([path, z])}], SIGMA, radius=0.30)
    assert fit is not None
    assert all(math.hypot(ox, oy) <= 0.30 + 1e-9 for ox, oy in fit['offsets'])


def test_disc_map_covers_circle_only():
    from did.calibrate import _disc
    g = np.linspace(-1.0, 1.0, 41)
    r = [math.hypot(*_disc(u, v, 0.3)) for u in g for v in g]
    assert max(r) <= 0.3 + 1e-12 and max(r) == pytest.approx(0.3)
    assert _disc(0.0, 0.0, 0.3) == (0.0, 0.0) and _disc(1.0, 0.0, 0.3) == pytest.approx((0.3, 0.0))


# --- находка 4: дешёвые метры пути — ещё не обычный пол ---------------------------------------------

def test_drain_meter_does_not_refute_on_soil_only():
    """Весь путь по грунту ×2 при верном допущении 2,5: оценка 5,0 — осторожная, допущение не опровергнуто."""
    from did.calibrate import DrainMeter
    m = DrainMeter(2.5)
    out = [m.add(0.6, 3.0, 0.6 * (i + 1), 0.0) for i in range(3)]
    assert out == [None, 5.0, 5.0]
    assert m.verdict() is None and m.base is None and m.low == 5.0
    for i in range(20):                                   # выехал на обычный пол: теперь он выделен
        m.add(0.1, 0.25, 1.8 + 0.1 * (i + 1), 0.0)
    assert m.low == pytest.approx(2.5) and m.verdict() == 'confirmed' and m.base == pytest.approx(2.5)


def test_drain_meter_verdicts():
    from did.calibrate import DrainMeter
    m = DrainMeter(2.5)                                   # мир дороже: 3,75 на всём пути
    for i in range(30):
        m.add(0.1, 0.375, 0.1 * i, 0.0)
    assert m.low == pytest.approx(3.75) and m.verdict() is None          # 3 м: могло быть пятно грунта
    for i in range(30, 60):
        m.add(0.1, 0.375, 0.1 * i, 0.0)
    assert m.verdict() == 'refuted' and m.base == pytest.approx(3.75)   # 6 м подряд на размахе 6 м — не пятно
    assert m.low_se == pytest.approx(0.0, abs=1e-9)
    m = DrainMeter(2.5)                                   # то же, но на пятачке: размаха нет — вывода нет
    for i in range(60):
        m.add(0.1, 0.375, 0.1 * (i % 5), 0.0)
    assert m.verdict() is None
    m = DrainMeter(2.5)                                   # мир дешевле допущения: грунт расход не понижает
    for i in range(15):
        m.add(0.1, 0.2, 0.1 * i, 0.0)
    assert m.verdict() == 'refuted' and m.base == pytest.approx(2.0)
    m = DrainMeter(2.5)                                   # долгая езда по грунту после обычного пола оценку не поднимает
    for i in range(20):
        m.add(0.1, 0.25, 0.1 * i, 0.0)
    for i in range(200):
        m.add(0.1, 0.75, 2.0 + 0.01 * i, 0.5)
    assert m.low == pytest.approx(2.5) and m.verdict() == 'confirmed'
    noise = np.random.default_rng(2)                      # шум батареи: погрешность оценки не нулевая
    m = DrainMeter(2.5)
    for i in range(80):
        m.add(0.1, 0.25 + noise.normal(0.0, 0.005), 0.1 * i, 0.0)
    assert 0.0 < m.low_se < 0.05 and abs(m.low - 2.5) < 0.1


def test_agent_on_soil_keeps_assumption_open():
    """В агенте: путь только по грунту ×2 — гипотеза о расходе открыта, а цена метра для запаса уже осторожная."""
    from did.agent import Agent, make_config
    bot = Agent(load_arena(), make_config('adaptive_cal'), n_samples=3)
    bot.journal.open(0.0, 'cal:drain', 'метр обычного пола стоит 2.50', 'мерить')
    for i in range(3):
        bot.cal.segment(0.6, 3.0, 1.0 + i, 0.6 * (i + 1), 0.0)
    assert bot.journal.get('cal:drain')['status'] == 'open'
    assert bot.cal.base is None and bot.cal.per_m == 5.0 and bot._floor_per_m() == 5.0
    for i in range(20):
        bot.cal.segment(0.1, 0.25, 5.0 + i, 1.8 + 0.1 * (i + 1), 0.0)
    assert bot.journal.get('cal:drain')['status'] == 'confirmed' and bot.cal.base == pytest.approx(2.5)
    assert bot.cal.per_m == pytest.approx(2.5)


# --- совместимость со сторожем датчика (P1) ------------------------------------------------------

def test_fault_readings_do_not_vote_on_law(arena):
    """Показания при сбое датчика идут в карты, но не в сравнение законов и не в сверку по парам."""
    from types import SimpleNamespace
    from did.agent import Agent, make_config
    bot = Agent(arena, make_config('adaptive_cal_v2'), n_samples=3)
    noise = np.random.default_rng(4)
    obs = lambda t: SimpleNamespace(t=t, x=0.1 * t, y=0.0)
    for i in range(10):
        bot.cal.reading(obs(i), 0.3, SIGMA)
    score, n_log, n_upd = bot.cal.bank.score.copy(), len(bot.cal.log), bot.belief.updates
    bot.health.degraded = True
    for i in range(10, 60):
        bot.cal.reading(obs(i), float(np.clip(noise.normal(0.4, 0.25), 0.0, 1.0)), 0.25)
    assert np.array_equal(bot.cal.bank.score, score) and len(bot.cal.log) == n_log
    assert bot.belief.updates == n_upd + 50 and bot.cal.law == (2.0, 1.0)
    bot.health.degraded = False
    bot.guard.wait = {'why': 'stuck'}
    bot.cal.reading(obs(60), 0.3, SIGMA)
    assert np.array_equal(bot.cal.bank.score, score)
    bot.guard.wait = None
    bot.cal.reading(obs(61), 0.3, SIGMA)
    assert not np.array_equal(bot.cal.bank.score, score)
    fine = bot.cal.bank.zoom()                              # сужение сетки проигрывает историю с тем же счётом
    assert fine is None or fine.score[fine.null] == pytest.approx(bot.cal.bank.score[bot.cal.bank.null])


def test_gazebo_run_does_not_hand_stand_rules_to_calibrating_agent(monkeypatch):
    """Ревью, круг 2: в Gazebo калибрующийся агент тоже стартует с допущений, а не с объекта правил стенда."""
    pytest.importorskip('rclpy')
    import did.ros_agent as ra
    from did.config import Rules

    class Seen(Exception):
        pass

    class FakeIO:
        score, odom, truth = {}, (0.0, 0.0, 0.0), None
        _t = 0.0

        def ready(self):
            return True

        @property
        def sim_time(self):
            FakeIO._t += 1.0
            return FakeIO._t

        def now(self):
            return 1e9

        def command(self, v, w):
            pass

        def restart_clock(self):
            pass

        def destroy_node(self):
            pass

    class FakeExecutor:
        def add_node(self, node):
            pass

        def spin(self):
            pass

        def shutdown(self):
            pass

    stand = {}

    def fake_rules(**kw):
        # правила стенда отличаются от допущений команды по всем трём калибруемым величинам
        stand['rules'] = Rules(sensor_range_m=1.5, sensor_law='sqrt', drain_per_m=3.75, **kw)
        return stand['rules']

    def fake_agent(arena, cfg, **kw):
        raise Seen(cfg.name, kw['rules'])

    monkeypatch.setattr(ra, 'Rules', fake_rules)
    monkeypatch.setattr(ra, 'RosIO', FakeIO)
    monkeypatch.setattr(ra, 'SingleThreadedExecutor', FakeExecutor)
    monkeypatch.setattr(ra, 'Agent', fake_agent)
    monkeypatch.setattr(ra.rclpy, 'init', lambda *a, **k: None)
    monkeypatch.setattr(ra.rclpy, 'shutdown', lambda *a, **k: None)
    monkeypatch.setattr(ra.time, 'sleep', lambda s: None)
    default = Rules()
    for name in ('adaptive_cal', 'adaptive_cal_v2'):
        with pytest.raises(Seen) as e:
            ra.run('hard', 1101, name, 'test', settle_s=1.0, quiet=True)
        got = e.value.args[1]
        assert got is not stand['rules']
        assert (got.sensor_range_m, got.sensor_law, got.drain_per_m) == \
            (default.sensor_range_m, default.sensor_law, default.drain_per_m)
    with pytest.raises(Seen) as e:                   # обычный агент по-прежнему получает правила стенда
        ra.run('hard', 1101, 'adaptive', 'test', settle_s=1.0, quiet=True)
    assert e.value.args[1] is stand['rules']
