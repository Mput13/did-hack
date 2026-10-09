"""L5: мерило «цена решения задним числом» (did/hindsight.py) и отбор состояний в банк."""
import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from did import hindsight as hs
from did.planner import HeuristicPlanner
from did.runner import run_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))

LEVEL, SEED = 'medium', 6


@pytest.fixture(scope='module')
def run():
    rows, result = hs.decisions(LEVEL, SEED)
    return rows, result


def test_plain_run_is_unchanged(run):
    """Без инструмента агент ведёт себя как раньше, а наблюдатель без подстановки прогон не меняет."""
    rows, result = run
    plain = run_episode(LEVEL, SEED, hs.AGENT, save=False)['metrics']
    own = run_episode(LEVEL, SEED, hs.AGENT, save=False, planner=HeuristicPlanner())['metrics']
    assert plain['score'] == own['score'] == result['score']
    assert plain['samples_collected'] == result['samples_collected']
    assert len(rows) > 5


def test_replay_with_original_choice_reproduces_score(run):
    rows, result = run
    for row in rows[1::3]:
        rule = hs.rule_choice(row['state'])
        out, _ = hs.replay(LEVEL, SEED, at=row['k'], subgoals=[rule])
        assert out['state_hash'] == row['hash']
        assert abs(out['score'] - result['score']) < 0.005


def test_substitution_changes_only_that_decision(run):
    rows, result = run
    row = next(r for r in rows if r['k'] >= 2 and len(r['options']) >= 3)
    other = next(o for o in hs.options(row['state']) if hs.option_key(o) not in (row['rule'], 'return_base'))
    out, planner = hs.replay(LEVEL, SEED, at=row['k'], subgoals=[other], keep=True)
    # до решения и в момент решения планировщик видел те же состояния, что в исходном прогоне
    assert planner.hashes[:row['k'] + 1] == [r['hash'] for r in rows[:row['k'] + 1]]
    assert planner.state == row['state']
    # все остальные вызовы отвечает правило: подстановка одна
    forced = hs.ForcedPlanner(at=1, subgoals=[other])
    states = [r['state'] for r in rows[:4]]
    plans = [forced.plan(copy.deepcopy(s)) for s in states]
    assert plans[1]['subgoals'] == [other]
    for i in (0, 2, 3):
        assert plans[i]['subgoals'] == HeuristicPlanner().plan(states[i])['subgoals']
    assert out['calls'] >= row['k'] + 1


def test_return_base_option_ends_run(run):
    rows, _ = run
    row = rows[3]
    out, _ = hs.replay(LEVEL, SEED, at=row['k'], subgoals=[{'type': 'return_base'}])
    assert out['returned'] and out['next_t'] is None
    assert out['samples_collected'] == row['state']['samples']['collected']


def test_reseed_keeps_history_and_changes_only_future(run):
    rows, result = run
    row = rows[2]
    a, pa = hs.replay(LEVEL, SEED, at=row['k'], rep=3)
    b, pb = hs.replay(LEVEL, SEED, at=row['k'], rep=3)
    assert a == b                                              # продолжение с тем же номером повторяется
    assert pa.hashes[:row['k'] + 1] == [r['hash'] for r in rows[:row['k'] + 1]]
    zero, _ = hs.replay(LEVEL, SEED, at=row['k'], rep=0)
    assert zero['score'] == result['score']


def _state(**kw):
    state = {'time_s': 50.0, 'time_limit_s': 600, 'battery': 30.0, 'pose': {'x': 0.0, 'y': 0.0},
             'samples': {'collected': 2, 'total': 5}, 'candidates': [], 'explore_points': [], 'soil_zones': [],
             'hazards': [], 'sensor': {'status': 'ok'}}
    state.update(kw)
    return state


def _cand(i, conf, cost, ok=True, x=1.0, y=1.0):
    return {'id': f'C{i}', 'x': x, 'y': y, 'confidence': conf, 'cost_to': cost, 'cost_back': 5.0, 'feasible': ok}


def _point(i, ok=True, x=-1.0, y=-1.0):
    return {'id': f'E{i}', 'x': x, 'y': y, 'unseen_share': 0.3, 'cost_to': 3.0, 'cost_back': 5.0, 'feasible': ok}


def test_options_and_tags():
    state = _state(candidates=[_cand(1, 0.4, 1.0), _cand(2, 0.9, 6.0), _cand(3, 0.9, 1.0, ok=False)],
                   explore_points=[_point(1), _point(2, ok=False)])
    assert [hs.option_key(o) for o in hs.options(state)] == ['investigate:C1', 'investigate:C2', 'explore:E1',
                                                              'return_base']
    assert hs.tags(state) == ['weak_near_strong_far']           # 0.4/1.5 > 0.9/6.5, а второй увереннее на 0,5
    assert hs.tags(_state(battery=45.0, candidates=[_cand(1, 0.6, 1.0), _cand(2, 0.7, 1.2)])) == ['high_battery']
    assert hs.tags(_state(battery=15.0, candidates=[_cand(1, 0.4, 1.0)])) == ['low_battery', 'weak_only']
    assert hs.tags(_state(time_s=500.0, sensor={'status': 'degraded'}, samples={'collected': 4, 'total': 5},
                          explore_points=[_point(1), _point(2)])) == ['time_short', 'sensor_bad', 'almost_done',
                                                                      'explore_only']
    near = _state(explore_points=[_point(1)], hazards=[{'x': -1.0, 'y': -0.2, 'radius': 0.3}])
    assert hs.tags(near) == ['danger_near']
    far = _state(explore_points=[_point(1)], hazards=[{'x': 2.0, 'y': 2.0, 'radius': 0.3}],
                 soil_zones=[{'x': 0.1, 'y': 0.0, 'radius': 0.2, 'mult': 2.0, 'status': 'suspected'}])
    assert hs.tags(far) == []


def _rows(n_runs=12, per_run=14):
    rng = random.Random(1)
    rows = []
    for seed in range(n_runs):
        for k in range(per_run):
            rows.append({'id': f'hard-{seed}#{k}', 'level': 'hard', 'seed': seed, 'k': k, 't': 4.0 * k,
                         'options': ['explore:E1', 'return_base'] if k % 5 else ['return_base'],
                         'tags': rng.sample(hs.TAGS, rng.randint(0, 2))})
    return rows


def test_selection_is_deterministic_and_follows_limits():
    rows = _rows()
    bank = hs.select(rows, total=40, per_tag=3)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    assert [r['id'] for r in hs.select(shuffled, total=40, per_tag=3)] == [r['id'] for r in bank]
    assert len(bank) == 40 and len({r['id'] for r in bank}) == 40
    assert all(r['t'] > 0 and len(r['options']) >= 2 for r in bank)
    by_run = {}
    for r in bank:
        by_run.setdefault(r['seed'], []).append(r)
    for mine in by_run.values():
        assert len(mine) <= 6
        times = sorted(r['t'] for r in mine)
        assert all(b - a >= 5.0 for a, b in zip(times, times[1:]))
    for tag in hs.TAGS:
        picked = [r for r in bank if r['picked_for'] == tag]
        assert len(picked) <= 3 and all(tag in r['tags'] for r in picked)
        assert max([0, *[sum(1 for r in picked if r['seed'] == s) for s in by_run]]) <= 2


def test_crossfit_loss_is_not_inflated_by_noise():
    """У равноценных шумных вариантов прямая потеря положительна всегда, перекрёстная в среднем около нуля."""
    import l5_bank
    rng = np.random.default_rng(0)
    direct, cross = [], []
    for _ in range(400):
        table = {f'o{i}': rng.normal(70.0, 8.0, l5_bank.REPS) for i in range(6)}
        direct.append(max(v.mean() for v in table.values()) - table['o0'].mean())
        cross.append(l5_bank.crossfit_loss(table, table['o0']))
    assert np.mean(direct) > 2.0
    assert abs(np.mean(cross)) < 0.6
    better = {'a': np.full(l5_bank.REPS, 60.0), 'b': np.full(l5_bank.REPS, 75.0)}
    assert l5_bank.crossfit_loss(better, better['a']) == pytest.approx(15.0)


def test_index_survives_summary_written_by_own_tool(tmp_path, monkeypatch):
    """Сводка банка L5 пишется своей программой и вердиктов не содержит: список опытов на ней не падает."""
    import json

    from did import experiments
    monkeypatch.setattr(experiments, 'RUNS', tmp_path)
    (tmp_path / 'L5').mkdir()
    (tmp_path / 'L5' / 'summary.json').write_text(json.dumps({'experiment': 'L5', 'generated': '2026-10-09T05:00:00'}))
    item = next(i for i in experiments.build_index()['experiments'] if i['id'] == 'L5')
    assert item['generated'] == '2026-10-09T05:00:00' and item['runs'] == 0 and item['claims'] == []

