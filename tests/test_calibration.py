"""Самокалибровка датчика и расхода (R14): подбор закона, карта с другим законом, прежнее поведение."""
import math

import numpy as np
import pytest

from did.agent import PRESETS
from did.arena import load_arena
from did.belief import SampleBelief
from did.calibrate import DrainMeter, LawBank, fit_law, response, shape_name
from did.config import Rules
from did.experiments import _job, load_spec
from did.runner import run_episode

SIGMA = 0.05


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _readings(points, sample, rng_m, power, noise):
    d = np.hypot(points[:, 0] - sample[0], points[:, 1] - sample[1])
    z = np.clip(response(d, rng_m, power) + noise.normal(0.0, SIGMA, len(d)), 0.0, 1.0)
    return np.column_stack([points, z])


def _approach(noise, start, end, n=120):
    """Путь робота к месту сбора: дуга с боковым смещением, как при объезде, и стояние на месте в конце."""
    s = np.linspace(0.0, 1.0, n)[:, None]
    side = np.array([-(end - start)[1], (end - start)[0]]) * 0.35
    path = start + (end - start) * s + side * np.sin(math.pi * s)
    return np.vstack([path, np.repeat(end[None, :], 25, axis=0)]) + noise.normal(0.0, 0.01, (n + 25, 2))


@pytest.mark.parametrize('rng_m,power', [(2.0, 1.0), (1.5, 1.0), (2.5, 1.0), (2.0, 2.0), (2.0, 0.5)])
def test_fit_law_recovers_range_and_shape(rng_m, power):
    """По рукотворным парам «расстояние — показание» с шумом восстанавливаются дальность и форма закона."""
    noise = np.random.default_rng(7)
    sites = []
    for k, (sample, start) in enumerate([((0.4, 0.3), (-1.6, -0.9)), ((-0.8, 1.0), (1.0, 0.2))]):
        stop = np.array(sample) + [0.12, -0.10]            # сбор не точно над образцом, а в 0,16 м от него
        pts = _readings(_approach(noise, np.array(start, dtype=float), stop), sample, rng_m, power, noise)
        sites.append({'x': float(stop[0]), 'y': float(stop[1]), 'pts': pts})
    fit = fit_law(sites, SIGMA, start=(2.0, 1.0))
    assert fit is not None and fit['n'] >= 60
    assert abs(fit['range'] / rng_m - 1.0) < 0.10
    assert abs(fit['power'] / power - 1.0) < 0.15
    assert shape_name(fit['power']) == shape_name(power)


def test_fit_law_needs_data():
    assert fit_law([], SIGMA) is None
    assert fit_law([{'x': 0.0, 'y': 0.0, 'pts': np.zeros((3, 3))}], SIGMA) is None


def test_belief_default_law_is_linear(arena):
    """Без явного показателя карта работает по прямой — как до R14."""
    b = SampleBelief(arena, 5, 2.0)
    assert b.power == 1.0
    d = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert np.allclose(b._resp(d), 1.0 - d / 2.0)
    assert np.allclose(SampleBelief(arena, 5, 2.0, power=2.0)._resp(d), 1.0 - (d / 2.0) ** 2)


@pytest.mark.parametrize('power', [2.0, 0.5])
def test_belief_with_other_law_finds_sample(arena, power):
    """Карта с верным (не прямым) законом ставит образец на место; карта «по прямой» по тем же показаниям — нет."""
    noise = np.random.default_rng(3)
    sample = (0.5, 0.4)
    path = _approach(noise, np.array([-1.4, -0.8]), np.array([0.2, 1.3]), n=90)    # проезд мимо, без остановки над образцом
    pts = _readings(path, sample, 2.0, power, noise)
    right, wrong = SampleBelief(arena, 1, 2.0, power=power), SampleBelief(arena, 1, 2.0)
    for x, y, z in pts:
        right.update(x, y, z, SIGMA)
        wrong.update(x, y, z, SIGMA)
    peak = lambda b: (b.cx[b.p.argmax()], b.cy[b.p.argmax()])
    assert math.dist(peak(right), sample) <= 0.15
    assert right.prob_within(*sample, 0.15) > 0.6
    assert math.dist(peak(wrong), sample) > 0.3
    assert wrong.prob_within(*sample, 0.15) < 0.01


def test_update_returns_evidence(arena):
    """update возвращает то же, что evidence до обновления, — при любом законе и у краёв шкалы."""
    noise = np.random.default_rng(0)
    for rng_m, power in ((2.0, 1.0), (1.6, 0.5), (2.5, 2.0)):
        b = SampleBelief(arena, 5, rng_m, power=power)
        for i in range(40):
            x, y = noise.uniform(-2.0, 2.0, 2)
            z = (0.0, 1.0, float(noise.uniform(0.02, 0.98)))[i % 3]
            before = b.evidence(x, y, z, SIGMA)
            assert b.update(x, y, z, SIGMA) == pytest.approx(before, abs=1e-9)


def test_trial_matches_apply_for_other_law(arena):
    """Пробное обновление и настоящее дают одну карту и при кривом законе."""
    b = SampleBelief(arena, 3, 2.0, power=2.0)
    b.update(0.0, 0.0, 0.4, SIGMA)
    for z in (0.0, 0.37, 1.0):
        trial = b.trial(0.3, -0.2, z, SIGMA)[0]
        c = SampleBelief(arena, 3, 2.0, power=2.0)
        c.p = b.p.copy()
        c._apply(0.3, -0.2, z, SIGMA)
        c._normalize()
        assert np.allclose(trial, c.p, rtol=1e-9, atol=1e-12)


def _bank_after_drive(arena, law, seed=0):
    """Сетка гипотез после проезда мимо образца, подъезда к нему и отъезда: три отрезка под разными углами."""
    noise = np.random.default_rng(seed)
    sample = np.array([0.57, -0.33])
    path = np.vstack([_approach(noise, sample + [-2.1, -0.7], sample + [-0.3, 1.2], n=80),
                      _approach(noise, sample + [-0.3, 1.2], sample + [0.1, 0.05], n=60),
                      _approach(noise, sample + [0.1, 0.05], sample + [1.0, -0.9], n=60)])
    bank = LawBank(arena, 1, (2.0, 1.0))
    for x, y, z in _readings(path, sample, *law, noise):
        bank.apply('update', x, y, z, SIGMA)
    return bank


@pytest.mark.parametrize('law', [(2.0, 0.5), (2.0, 2.0)])
def test_law_bank_prefers_true_law(arena, law):
    """Из сетки гипотез выигрывает настоящий закон, с большим перевесом над допущением «по прямой»."""
    bank = _bank_after_drive(arena, law)
    best, _, over_null = bank.lead()
    assert bank.laws[best] == pytest.approx(law)
    assert over_null > 20.0


def test_law_bank_keeps_assumption_when_it_is_true(arena):
    bank = _bank_after_drive(arena, (2.0, 1.0))
    best, _, over_null = bank.lead()
    assert best == bank.null and over_null == 0.0


def test_law_bank_zoom(arena):
    """Сужение сетки: лучший узел остаётся, допущение — тоже, наблюдения проиграны заново с тем же счётом."""
    bank = _bank_after_drive(arena, (2.0, 2.0))
    best = int(bank.score.argmax())
    fine = bank.zoom()
    # по дальности лучший узел в середине сетки — шаг мельче; по показателю он с краю — шаг прежний
    assert fine.steps == pytest.approx((math.sqrt(1.25), math.sqrt(2.0))) and fine.center == bank.laws[best]
    assert fine.laws[fine.null] == (2.0, 1.0) and len(fine.ops) == len(bank.ops)
    assert fine.score[4] == pytest.approx(bank.score[best])                  # центр новой сетки — прежний лучший узел
    assert fine.score[fine.null] == pytest.approx(bank.score[bank.null])
    assert fine.score.max() >= bank.score.max() - 1e-9


def test_drain_meter_ignores_expensive_soil():
    """Расход обычного пола берётся по нижней части распределения: дорогой грунт его не завышает."""
    m = DrainMeter(2.5)
    value = None
    for i in range(60):
        mult = 3.0 if 10 <= i < 22 else 1.0
        value = m.add(0.06, 3.75 * mult * 0.06)
    assert value == pytest.approx(3.75)
    assert DrainMeter(2.5).add(0.5, 1.9) is None            # пути пока мало — оценки нет


# Счёт, доля образцов, расход, путь и время на исходном коде ветки (коммит 53db31c, до правок R14).
BEFORE = {
    ('adaptive', 'medium', 3): [72.23, 1.0, 37.65, 14.706, 88.7],
    ('adaptive', 'hard', 5): [65.32, 0.714, 56.78, 18.283, 109.0],
    ('adaptive', 'hard', 12): [70.32, 0.714, 56.81, 17.51, 111.1],
    # исследователь: значение после исправления учёта штрафа в основной ветке (работа V1), а не с коммита 53db31c
    ('scientist', 'hard', 5): [65.16, 0.714, 58.39, 17.093, 117.3],
    ('gradient', 'hard', 5): [61.17, 0.571, 48.29, 12.811, 125.5],
}


@pytest.mark.parametrize('key', list(BEFORE), ids=lambda k: f'{k[0]}-{k[1]}-{k[2]}')
def test_default_behaviour_unchanged(key):
    """Прежние варианты дают те же числа, что до правок: калибровка — только за флажком."""
    agent, level, seed = key
    m = run_episode(level, seed, agent, save=False)['metrics']
    assert [m['score'], m['samples_share'], m['battery_used'], m['distance'], m['time']] == BEFORE[key]


def test_default_under_other_law_unchanged():
    m = run_episode('hard', 5, 'adaptive', save=False, rules={'sensor_law': 'sqrt'}, agent_rules={})['metrics']
    assert [m['score'], m['samples_share'], m['battery_used']] == [15.37, 0.0, 56.26]


def test_flag_is_off_by_default():
    assert all(not cfg.calibrate and not cfg.law_known for name, cfg in PRESETS.items()
               if name not in ('adaptive_cal', 'adaptive_cal_v2', 'adaptive_known'))
    assert PRESETS['adaptive_cal'].calibrate and not PRESETS['adaptive_cal'].law_known


@pytest.mark.parametrize('condition', load_spec('E20')['conditions'], ids=lambda c: c['id'])
def test_e20_agent_is_not_told_world_rules(monkeypatch, condition):
    """Полный путь YAML → _job → runner: калибрующийся агент стартует с допущений, а не с правил мира."""
    from did import runner
    captured = {}
    original = runner.make_agent

    def make_agent(name, config=None):
        cls, cfg = original(name, config)

        class Probe(cls):
            def tick(self, obs, io):
                captured[name] = (self, io.judge)
                raise RuntimeError('R14 first-tick probe')

        return Probe, cfg

    monkeypatch.setattr(runner, 'make_agent', make_agent)
    arms = {a['id']: a for a in load_spec('E20')['arms']}
    for arm in ('adaptive_cal', 'adaptive_known'):
        assert _job(('E20', arms[arm], condition, 'medium', 1001))['error'] == 'RuntimeError: R14 first-tick probe'
    world = Rules(**condition.get('rules', {}))
    bot, judge = captured['adaptive_cal']
    assert judge.rules == world and bot.rules == Rules()
    assert bot.cal.law == (2.0, 1.0) and bot.cal.per_m == 2.5 and bot.belief.range == 2.0 and bot.belief.power == 1.0
    assert bot.soil.per_m == 2.5
    known, _ = captured['adaptive_known']
    assert known.rules == world and known.cal is None
    assert known.belief.range == world.sensor_range_m
    assert known.belief.power == {'linear': 1.0, 'quadratic': 2.0, 'sqrt': 0.5}[world.sensor_law]


def test_calibration_is_journaled():
    """Калибровка видна в журнале как гипотеза с проверкой и выводом, а закон «по корню» агент называет сам."""
    from did import runner
    keep = {}
    original = runner.make_agent

    def make_agent(name, config=None):
        cls, cfg = original(name, config)

        class Probe(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                keep['bot'] = self

        return Probe, cfg

    runner.make_agent = make_agent
    try:
        s = run_episode('medium', 3, 'adaptive_cal', save=False, rules={'sensor_law': 'sqrt', 'drain_per_m': 3.75},
                        agent_rules={})
    finally:
        runner.make_agent = original
    bot = keep['bot']
    hyp = [h for h in bot.journal.hypotheses if h['key'].startswith('law:cal:')]
    assert hyp[0]['status'] == 'refuted' and 'по прямой' in hyp[0]['statement']
    assert any(h['status'] == 'confirmed' and 'по корню' in h['statement'] for h in hyp)
    assert shape_name(bot.cal.law[1]) == 'по корню' and abs(bot.cal.law[0] / 2.0 - 1.0) < 0.15
    drain = bot.journal.get('cal:drain')
    assert drain['status'] == 'refuted' and bot.cal.per_m == pytest.approx(3.75, rel=0.02)
    assert s['metrics']['samples_collected'] >= 3
