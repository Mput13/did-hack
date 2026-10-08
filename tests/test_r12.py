"""R12: способы выбора опыта в расследовании (did.science.CHOICES). Прежний способ — по умолчанию и не меняется."""
import numpy as np
import pytest

from did.agent import AgentConfig, make_config
from did.runner import run_episode
from did.science import CHOICES, OTHER, Alternative, Inquiry
from did.science import TestOption as Option


def _inquiry(policy='gain', seed=0, **kw):
    """Три опыта: дешёвый и слабый, дорогой и сильный, средний. Объявлены в порядке rest, straight, spin."""
    alts = [Alternative('leak', 'утечка', 0.4), Alternative('soil', 'грунт', 0.4), Alternative(OTHER, 'другое', 0.2)]
    tests = [Option('rest', 'постоять', cost=0.07, duration_s=2.0, unit='ед/с',
                        predictions={'leak': (0.16, 0.14), 'soil': (0.01, 0.012)}, sigma=0.015),
             Option('straight', 'прямо', cost=1.2, duration_s=2.0, unit='×',
                        predictions={'leak': (1.4, 0.35), 'soil': (3.0, 0.2)}, sigma=0.04),
             Option('spin', 'разворот', cost=0.2, duration_s=1.5, unit='ед/рад',
                        predictions={'leak': (0.27, 0.09), 'soil': (0.12, 0.05)}, sigma=0.02)]
    q = Inquiry('Q1', 0.0, 'energy', {'text': 'x'}, alts, tests, **kw)
    q.policy, q.rng = policy, np.random.default_rng([seed, 1])
    return q


def _old_choose(q):
    """Выбор опыта, каким он был до R12 (did/science.py в main), слово в слово."""
    if q.conclusion or q.settled or q.maneuvers >= q.max_tests:
        return None
    best, score = None, 0.0
    for test in q.tests:
        if test.measured is not None and (not test.repeatable or test.repeats >= 2):
            continue
        if q.spent + test.cost > q.budget:
            continue
        test.gain_bits = round(q.gain(test), 3)
        if test.gain_bits < q.min_gain:
            continue
        value = test.gain_bits / (test.cost + 0.05)
        if value > score:
            best, score = test, value
    return best


def test_default_is_gain():
    assert AgentConfig().inquiry_choice == 'gain'
    assert make_config('scientist').inquiry_choice == 'gain'
    assert _inquiry().policy == 'gain'
    assert Inquiry('Q', 0.0, 'energy', {}, [Alternative('a', 'a', 0.5), Alternative(OTHER, 'o', 0.5)], []).policy == 'gain'


@pytest.mark.parametrize('budget', [2.5, 1.0, 0.1, 0.05])
def test_gain_choice_is_the_old_choice(budget):
    a, b = _inquiry(energy_budget=budget), _inquiry(energy_budget=budget)
    for value in (0.02, 0.1, 2.0):
        new, old = a.choose(), _old_choose(b)
        assert (new.id if new else None) == (old.id if old else None)
        assert [x.gain_bits for x in a.tests] == [x.gain_bits for x in b.tests]
        if new is None:
            break
        a.record(new.id, value, new.sigma, 1.0)
        b.record(old.id, value, old.sigma, 1.0)
        assert a.posterior == b.posterior


def test_rules_pick_what_they_say():
    picks = {p: _inquiry(p).choose() for p in ('gain', 'bits', 'cheapest', 'fixed', 'worst')}
    gains = {x.id: x.gain_bits for x in _inquiry().tests}
    assert picks['cheapest'].id == 'rest' and picks['fixed'].id == 'rest'
    assert picks['bits'].id == max(gains, key=gains.get)
    assert picks['worst'].id == min(gains, key=gains.get)
    assert picks['bits'].id != picks['worst'].id
    assert set(CHOICES) == {'gain', 'bits', 'random', 'cheapest', 'fixed', 'worst', 'blind'}
    with pytest.raises(ValueError):
        _inquiry('nonsense').choose()


def test_random_is_reproducible_and_stays_admissible():
    seq = lambda seed: [_inquiry('random', seed).choose().id for _ in range(1)]      # noqa: E731
    assert [seq(s) for s in range(30)] == [seq(s) for s in range(30)]
    assert len({seq(s)[0] for s in range(30)}) == 3            # все три допустимых опыта встречаются
    # бюджет: дорогой опыт недопустим ни при каком способе выбора
    for policy in CHOICES:
        for seed in range(10):
            assert _inquiry(policy, seed, energy_budget=1.0).choose().id != 'straight'


def test_blind_ignores_gain_but_not_budget():
    # порог пользы выше любой возможной: остальным способам выбирать не из чего, «вслепую» — есть
    assert _inquiry('random', min_gain=5.0).choose() is None
    q = _inquiry('random', min_gain=5.0)
    q.choose()
    assert q.stop == 'no_gain'
    assert _inquiry('blind', min_gain=5.0).choose() is not None
    q = _inquiry('blind', energy_budget=0.01)
    assert q.choose() is None and q.stop == 'budget'


def test_stop_reasons_and_steps():
    q = _inquiry(energy_budget=0.1)
    first = q.choose()
    assert first.id == 'rest' and q.steps[-1]['chosen'] == 'rest'
    assert [x['id'] for x in q.steps[-1]['over_budget']] == ['straight', 'spin']
    q.record('rest', 0.05, 0.015, 1.0, cost=0.07)
    if not q.settled:
        assert q.choose() is None and q.stop == 'budget'
    q = _inquiry()
    q.record('rest', 0.16, 0.015, 1.0)
    q.record('straight', 1.4, 0.04, 2.0)
    assert q.settled and q.choose() is None and q.stop == 'settled'
    q = _inquiry(max_tests=1, accept=0.999)
    q.record('rest', 0.08, 0.015, 1.0)
    assert q.choose() is None and q.stop == 'max_tests'


def test_runs_gain_flag_changes_nothing_and_random_repeats():
    kw = dict(rules='science', save=False, knowledge={})
    base = run_episode('hard', 7, 'scientist', **kw)
    same = run_episode('hard', 7, 'scientist', config={'inquiry_choice': 'gain'}, **kw)
    for s in (base, same):
        s.pop('wall_s')
    assert base == same
    assert all(c['policy'] == 'gain' for c in base['science']['choices'])
    assert len(base['science']['choices']) == len(base['science']['inquiries'])
    r1 = run_episode('hard', 7, 'scientist', config={'inquiry_choice': 'random'}, **kw)
    r2 = run_episode('hard', 7, 'scientist', config={'inquiry_choice': 'random'}, **kw)
    for s in (r1, r2):
        s.pop('wall_s')
    assert r1 == r2
    assert all(c['policy'] == 'random' for c in r1['science']['choices'])
