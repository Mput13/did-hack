"""R12: способы выбора опыта в расследовании (did.science.CHOICES). Прежний способ — по умолчанию и не меняется."""
import numpy as np
import pytest

from did.agent import AgentConfig, make_config
from did.recorder import load_trace
from did.runner import RUNS, run_episode
from did.science import CHOICES, OTHER, Alternative, Inquiry
from did.science import TestOption as Option
from tools.r12_table import STATS, _rule_pick, clusters, diff, first_forks, inquiry_row, stat


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


# --- подсчёт (tools/r12_table.py): правки по ревью ---------------------------------------------------------


def _choice(q):
    """Запись выбора в том виде, в каком её выгружает Investigator (did/inquiry.py, поле choices)."""
    return {'id': q.id, 'policy': q.policy, 'stop': q.stop, 'maneuvers': q.maneuvers,
            'spent': round(q.spent, 3), 'budget': q.budget, 'steps': q.steps}


def test_repeated_test_keeps_accumulated_cost():
    q = _inquiry(accept=0.9999)
    q.tests[0].repeatable = True
    assert q.choose().id == 'rest'
    q.record('rest', 0.05, 0.015, 1.0, cost=0.057)
    assert q.choose().id == 'rest'                      # повтор той же паузы
    q.record('rest', 0.05, 0.015, 3.0, cost=0.085)
    q.close(3.0)
    # цена опыта перезаписана последним замером, накопленный расход — нет
    assert q.tests[0].cost == pytest.approx(0.085) and q.spent == pytest.approx(0.142)
    row = inquiry_row(q.to_dict(), _choice(q))
    assert row['maneuvers'] == 2 and row['order'] == ['rest']
    assert row['energy'] == pytest.approx(0.142)
    last_cost_only = sum(x['cost'] for x in q.to_dict()['tests'] if x['measured'])       # так считалось раньше
    assert last_cost_only < row['energy'] - 0.05
    # без повтора оба способа дают одно и то же (с точностью до округления цены в записи)
    q = _inquiry(accept=0.9999)
    q.record('rest', 0.05, 0.015, 1.0, cost=0.06)
    q.record('spin', 0.2, 0.02, 2.0, cost=0.11)
    q.close(2.0)
    assert inquiry_row(q.to_dict(), _choice(q))['energy'] == pytest.approx(0.17)


def test_repeated_test_in_recorded_run():
    """На записях итоговой серии, если они есть: расследования, на которые указало ревью."""
    path = RUNS / 'E27' / 'gain@science' / 'hard-12037.json.gz'
    if not path.exists():
        pytest.skip('записей E27 в этом дереве нет')
    tr = load_trace(path)
    c = next(x for x in tr['choices'] if x['id'] == 'Q5')
    row = inquiry_row(next(x for x in tr['inquiries'] if x['id'] == 'Q5'), c)
    assert c['maneuvers'] == 2 and row['energy'] == pytest.approx(0.152)


def _fake_run(correct, n=10):
    """Прогон с n закрытыми расследованиями, из них correct верных."""
    return {'m': {}, 'q': [{'closed': True, 'maneuvers': 1, 'topic': 'energy', 'truth': ['soil'], 'choice': True,
                            'verdict': 'correct' if i < correct else 'insufficient'} for i in range(n)]}


def test_bootstrap_keeps_both_rule_sets_of_a_scenario_together():
    # у каждого сценария два прогона (базовые и научные правила), и они зеркальны: 10 − k и k верных из 10.
    # сумма по сценарию всегда 10 из 20, так что при совместной перевыборке доля не меняется вовсе
    ks = [0, 2, 5, 9, 10, 3]
    data = {('base', 'a'): {('hard', s): _fake_run(10 - k) for s, k in enumerate(ks)},
            ('science', 'a'): {('hard', s): _fake_run(k) for s, k in enumerate(ks)},
            ('base', 'b'): {('hard', s): _fake_run(5) for s, _ in enumerate(ks)},
            ('science', 'b'): {('hard', s): _fake_run(5) for s, _ in enumerate(ks)}}
    data[('science', 'b')].pop(('hard', 5))            # сценарий без пары в срез не попадает
    a, b = (clusters(data, ['base', 'science'], arm) for arm in 'ab')
    assert sorted(a) == [('hard', s) for s in range(6)] and all(len(v) == 2 for v in a.values())
    assert sorted(b) == [('hard', s) for s in range(5)]
    assert clusters(data, ['base'], 'a')[('hard', 1)] == [data[('base', 'a')][('hard', 1)]]
    keys = sorted(set(a) & set(b))
    f = STATS['correct'][1]
    idx = np.random.default_rng(1).integers(len(keys), size=(500, len(keys)))
    s = stat(a, keys, f, idx)
    assert (s['value'], s['num'], s['den'], s['ci']) == (0.5, 50.0, 100, [0.5, 0.5])
    d = diff(a, b, keys, f, idx)
    assert (d['value'], d['ci']) == (0.0, [0.0, 0.0])
    # если бы 10 прогонов перевыбирались по одному (как было), тот же набор дал бы широкий интервал
    apart = {(c, *k): [v] for c in ('base', 'science') for k, v in data[(c, 'a')].items() if k in keys}
    keys2 = sorted(apart)
    wide = stat(apart, keys2, f, np.random.default_rng(1).integers(len(keys2), size=(500, len(keys2))))['ci']
    assert wide[1] - wide[0] > 0.3


def test_exact_tie_takes_the_first_option():
    """Две одинаковые по оценке и цене паузы: любое правило, кроме случайного, берёт объявленную первой."""
    def twins(policy, flip=False):
        q = _inquiry(policy)
        twin = Option('rest2', 'постоять ещё', cost=0.07, duration_s=2.0, unit='ед/с',
                      predictions=dict(q.tests[0].predictions), sigma=0.015)
        q.tests = [twin, q.tests[0]] if flip else [q.tests[0], twin]
        return q
    for policy in ('gain', 'bits', 'cheapest', 'fixed', 'worst'):
        for flip, first in ((False, 'rest'), (True, 'rest2')):
            q = twins(policy, flip)
            assert q.choose().id == first, policy
            a, b = q.tests
            assert a.gain_bits == b.gain_bits and a.cost == b.cost
            # таблица пересчитывает правило по записанному шагу так же
            assert _rule_pick(policy, q.steps[-1]['options']) == first, policy
    q, old = twins('gain', flip=True), twins('gain', flip=True)
    assert q.choose().id == _old_choose(old).id == 'rest2'


def test_first_fork_counts_the_same_inquiry_in_two_arms():
    def row(first, verdict, start='s', energy=0.1):
        return {'start': start, 'verdict': verdict, 'energy': energy,
                'steps': [{'options': [{'id': 'rest'}, {'id': 'straight'}], 'chosen': first}]}
    same = row('rest', 'correct')
    a = {1: [{'q': [same, row('rest', 'correct'), row('rest', 'wrong')]}],          # расхождение на втором
         2: [{'q': [row('rest', 'partial')]}],                                       # расхождение на первом
         3: [{'q': [row('rest', 'correct', start='x')]}],                            # начало разное — не пара
         4: [{'q': [same]}]}                                                         # расхождения нет
    b = {1: [{'q': [same, row('straight', 'insufficient', energy=2.0), row('straight', 'correct')]}],
         2: [{'q': [row('straight', 'correct')]}],
         3: [{'q': [row('straight', 'correct', start='y')]}],
         4: [{'q': [same]}]}
    out = first_forks(a, b)
    assert (out['n'], out['a_correct'], out['b_correct'], out['a_full'], out['b_full']) == (2, 2, 1, 1, 1)
    assert (out['both'], out['only_a'], out['only_b'], out['neither'], out['n_m']) == (1, 1, 0, 0, 2)
    assert out['a_energy'] == pytest.approx(0.2) and out['b_energy'] == pytest.approx(2.1)
