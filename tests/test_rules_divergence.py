"""Проверки разделения правил мира и правил агента, а также законов датчика (R8)."""
import math
import pytest
from did.arena import load_arena
from did.config import Rules
from did.judge import Judge
from did.runner import run_episode
from did.scenario import Scenario
from did.experiments import load_spec, _job, summarize_experiment


@pytest.mark.parametrize('arm', ['adaptive', 'fixed', 'gradient'])
@pytest.mark.parametrize('condition', load_spec('E18')['conditions'], ids=lambda c: c['id'])
def test_e18_actual_objects(monkeypatch, arm, condition):
    """Полный путь YAML → _job → runner → реальные объекты, остановка на первом tick."""
    from did import runner
    captured = {}
    original_make_agent = runner.make_agent

    def make_agent(name, config=None):
        cls, cfg = original_make_agent(name, config)

        class Probe(cls):
            def tick(self, obs, io):
                captured['bot'] = self
                captured['judge'] = io.judge
                raise RuntimeError('R8 first-tick probe')

        return Probe, cfg

    monkeypatch.setattr(runner, 'make_agent', make_agent)
    result = _job(('E18', {'id': arm, 'agent': arm}, condition, 'medium', 1001))
    assert result['error'] == 'RuntimeError: R8 first-tick probe'
    bot, judge = captured['bot'], captured['judge']
    assert bot.rules == Rules()
    assert judge.rules == Rules(**condition.get('rules', {}))
    if arm != 'gradient':
        assert bot.belief.range == 2.0
        assert bot.soil.per_m == 2.5
        assert bot.health.nominal == 0.05


def test_control_differences_are_paired():
    spec = {'arms': [{'id': 'a'}], 'conditions': [{'id': 'base'}, {'id': 'changed'}],
            'levels': ['medium'], 'metrics': ['score'], 'claims': [], 'control_condition': 'base'}
    runs = [{'arm': 'a', 'condition': c, 'level': 'medium', 'seed': seed,
             'metrics': {'score': value}}
            for c, seed, value in [('base', 1001, 10), ('base', 1002, 30),
                                   ('changed', 1002, 27), ('changed', 1001, 7)]]
    summary = summarize_experiment(spec, runs, 2, 0)
    pair = summary['control_differences'][1]['pairs']['score']
    assert pair['mean'] == -3
    assert pair['ci'] == [-3, -3]


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _scenario(sample_x=-1.0, sample_y=-0.5):
    return Scenario(level='easy', seed=0, samples=[[sample_x, sample_y]], soils=[], hazards=[], events=[])


def test_sensor_laws(arena):
    """Закон датчика считает правильно: linear, quadratic, sqrt."""
    # sample at (-1.0, -0.5), R = 2.0, sigma = 0.0
    r_lin = Rules(sensor_range_m=2.0, sensor_sigma=0.0, sensor_law='linear')
    r_quad = Rules(sensor_range_m=2.0, sensor_sigma=0.0, sensor_law='quadratic')
    r_sqrt = Rules(sensor_range_m=2.0, sensor_sigma=0.0, sensor_law='sqrt')

    j_lin = Judge(_scenario(-1.0, -0.5), arena, r_lin)
    j_quad = Judge(_scenario(-1.0, -0.5), arena, r_quad)
    j_sqrt = Judge(_scenario(-1.0, -0.5), arena, r_sqrt)

    # d = 0.0
    assert j_lin.read_sensor(-1.0, -0.5) == pytest.approx(1.0)
    assert j_quad.read_sensor(-1.0, -0.5) == pytest.approx(1.0)
    assert j_sqrt.read_sensor(-1.0, -0.5) == pytest.approx(1.0)

    # d = 1.0 (u = 0.5)
    assert j_lin.read_sensor(-1.0, 0.5) == pytest.approx(0.5)
    assert j_quad.read_sensor(-1.0, 0.5) == pytest.approx(0.75)
    assert j_sqrt.read_sensor(-1.0, 0.5) == pytest.approx(1.0 - math.sqrt(0.5))

    # d = 2.0 (u = 1.0)
    assert j_lin.read_sensor(-1.0, 1.5) == pytest.approx(0.0)
    assert j_quad.read_sensor(-1.0, 1.5) == pytest.approx(0.0)
    assert j_sqrt.read_sensor(-1.0, 1.5) == pytest.approx(0.0)

    # d = 2.5 (u > 1.0, вне зоны)
    assert j_lin.read_sensor(-1.0, 2.0) == pytest.approx(0.0)
    assert j_quad.read_sensor(-1.0, 2.0) == pytest.approx(0.0)
    assert j_sqrt.read_sensor(-1.0, 2.0) == pytest.approx(0.0)


def test_unknown_sensor_law_rejected():
    with pytest.raises(ValueError, match='Unknown sensor_law'):
        Rules(sensor_law='quadractic')


def test_agent_and_judge_rules_divergence(arena):
    """Агент получает свои правила, а судья — свои."""
    world_rules = {'drain_per_m': 3.75, 'sensor_range_m': 1.5, 'sensor_law': 'quadratic'}
    agent_rules = {'drain_per_m': 2.5, 'sensor_range_m': 2.0, 'sensor_law': 'linear'}

    # 1. При явном agent_rules
    s = run_episode('easy', 1, 'adaptive', rules=world_rules, agent_rules=agent_rules, save=False)
    assert 'agent_rules' in s
    assert s['agent_rules']['drain_per_m'] == 2.5
    assert s['agent_rules']['sensor_range_m'] == 2.0
    assert s['agent_rules']['sensor_law'] == 'linear'

    # 2. При пустом agent_rules={} (агент получает значения по умолчанию)
    s_def = run_episode('easy', 1, 'adaptive', rules=world_rules, agent_rules={}, save=False)
    assert 'agent_rules' in s_def
    assert s_def['agent_rules']['drain_per_m'] == 2.5
    assert s_def['agent_rules']['sensor_range_m'] == 2.0
    assert s_def['agent_rules']['sensor_law'] == 'linear'

    # 3. При agent_rules=None (поведение прежнее: агент и судья получают одни и те же правила)
    s_same = run_episode('easy', 1, 'adaptive', rules=world_rules, agent_rules=None, save=False)
    assert 'agent_rules' not in s_same


def test_default_regression():
    """При значениях по умолчанию run_episode даёт тот же итог, что и до правок."""
    res_m1 = run_episode('medium', 1, 'adaptive', save=False)
    assert res_m1['metrics']['score'] == pytest.approx(71.64, abs=0.01)
    assert res_m1['metrics']['samples_share'] == pytest.approx(1.0)
    assert res_m1['metrics']['returned'] is True

    res_m2 = run_episode('medium', 2, 'adaptive', save=False)
    assert res_m2['metrics']['score'] == pytest.approx(72.02, abs=0.01)
    assert res_m2['metrics']['samples_share'] == pytest.approx(1.0)
    assert res_m2['metrics']['returned'] is True

    res_h1 = run_episode('hard', 1, 'adaptive', save=False)
    assert res_h1['metrics']['score'] == pytest.approx(75.21, abs=0.01)
    assert res_h1['metrics']['returned'] is True
