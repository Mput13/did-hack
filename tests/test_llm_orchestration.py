"""Способы обращения к модели (did/llm_orchestration.py), характеры имитатора и плата за ожидание."""
import copy
import json

import pytest

from did.llm import EXAMPLE_STATE, LocalClient, Plan, build_messages
from did.llm_mock import MockResponder, TEMPERAMENTS, mock_plan, temper_choice, temper_plan, temper_review
from did.llm_orchestration import (PROPOSE_MARK, calc_choice, consequences, parse_choice, parse_proposals,
                                   parse_review)
from did.planner import MISSION, HeuristicPlanner, LLMPlanner

C1 = {'type': 'investigate', 'target': 'C1'}
E1 = {'type': 'explore', 'target': 'E1'}
HOME = {'type': 'return_base'}


def make_state(**changes):
    state = copy.deepcopy(EXAMPLE_STATE)
    state.update(changes)
    return state


def low_battery():
    return make_state(battery=30.0, battery_start=50.0)      # 30 < 70% от 50: робкий характер хочет домой


def plan_json(*subgoals, why='потому что'):
    return json.dumps({'reasoning': why, 'hypotheses': [], 'subgoals': list(subgoals)}, ensure_ascii=False)


class Script:
    """Ответчик по списку заранее заданных ответов; запоминает, что его спрашивали."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []

    def __call__(self, messages):
        self.asked.append(messages)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def planner(responder, strategy):
    return LLMPlanner(LocalClient(responder), strategy=strategy)


def roles(result):
    return [ex['role'] for ex in result['exchanges']]


# --- один запрос остаётся прежним -------------------------------------------------------------

def test_single_is_the_default_and_unchanged():
    state = make_state()
    res = LLMPlanner(LocalClient(MockResponder(seed=42))).plan(state)
    assert res == {'reasoning': res['reasoning'], 'hypotheses': res['hypotheses'], 'subgoals': [C1, E1],
                   'source': 'llm', 'exchanges': res['exchanges'], 'error': None}       # без новых полей
    assert roles(res) == ['user']


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        planner(MockResponder(), 'typo')


def test_first_request_is_the_same_as_single():
    """Автор и первый голос спрашивают дословно то же, что single: при кэше это один и тот же ответ."""
    state = make_state()
    single = build_messages({'mission': MISSION, **state})
    for strategy in ('critic', 'vote'):
        script = Script(plan_json(C1), {'verdict': 'accept', 'issues': [], 'advice': ''}, plan_json(C1))
        planner(script, strategy).plan(state)
        assert script.asked[0] == single


# --- автор, критик, исправление ---------------------------------------------------------------

def test_critic_accepts_and_plan_stays():
    script = Script(plan_json(C1), {'verdict': 'accept', 'issues': [], 'advice': ''})
    res = planner(script, 'critic').plan(make_state())
    assert res['subgoals'] == [C1] and res['source'] == 'llm'
    assert roles(res) == ['author', 'critic']
    assert [s['outcome'] for s in res['orchestration']['steps']] == ['investigate C1', 'accept']
    assert res['exchanges'][-1]['outcome'] == 'accept'              # итог шага виден и в обмене


def test_critic_revise_leads_to_revision():
    script = Script(plan_json(HOME), {'verdict': 'revise', 'issues': [], 'advice': 'Возьми C1'}, plan_json(C1))
    res = planner(script, 'critic').plan(make_state())
    assert res['subgoals'] == [C1]
    assert roles(res) == ['author', 'critic', 'revision']
    assert res['orchestration']['revised'] and res['orchestration']['changed']
    revision = script.asked[2]
    assert revision[-2]['role'] == 'assistant' and 'return_base' in revision[-2]['content']
    assert 'Возьми C1' in revision[-1]['content']


def test_critic_sees_state_plan_and_planner_instruction():
    script = Script(plan_json(C1), {'verdict': 'accept'})
    planner(script, 'critic').plan(make_state())
    system, user = script.asked[1]
    assert 'критик' in system['content'] and '# Как выбирать' in system['content']
    payload = json.loads(user['content'][user['content'].index('{'):user['content'].rindex('}') + 1])
    assert payload['plan']['subgoals'] == [C1] and payload['state']['battery'] == EXAMPLE_STATE['battery']


def test_broken_critic_or_revision_keeps_author_plan():
    res = planner(Script(plan_json(C1), 'не JSON', 'опять не JSON'), 'critic').plan(make_state())
    assert res['subgoals'] == [C1] and res['source'] == 'llm'
    assert roles(res) == ['author', 'critic', 'critic']             # отзыв просили исправить один раз
    assert 'отзыва нет' in res['orchestration']['steps'][-1]['outcome']
    bad = plan_json({'type': 'investigate', 'target': 'C99'})
    res = planner(Script(plan_json(C1), {'verdict': 'revise', 'issues': ['x']}, bad, bad), 'critic').plan(make_state())
    assert res['subgoals'] == [C1] and not res['orchestration']['revised']


def test_parse_review_needs_a_verdict():
    assert parse_review('{"advice": "без вердикта"}')[0] is None
    assert parse_review('{"verdict": "maybe"}')[0] is None
    assert parse_review('{"verdict": "REVISE", "issues": "одна строка"}')[0] == {
        'verdict': 'revise', 'issues': ['одна строка'], 'advice': ''}


# --- голосование ------------------------------------------------------------------------------

def test_vote_majority_and_distinct_requests():
    script = Script(plan_json(E1), plan_json(C1), plan_json(C1))
    res = planner(script, 'vote').plan(make_state())
    assert res['subgoals'] == [C1]
    assert roles(res) == ['voter_1', 'voter_2', 'voter_3']
    assert res['orchestration']['votes'] == {'explore E1': 1, 'investigate C1': 2}
    assert len({m[-1]['content'] for m in script.asked}) == 3       # иначе кэш вернул бы один ответ трижды


def test_vote_tie_goes_to_the_earliest_and_goto_counts_by_point():
    there = {'type': 'goto', 'x': 1.0, 'y': 0.0}
    res = planner(Script(plan_json(E1), plan_json(C1), plan_json(there)), 'vote').plan(make_state())
    assert res['subgoals'] == [E1] and not res['orchestration']['unanimous']
    res = planner(Script(plan_json({'type': 'goto', 'x': -1.0, 'y': 0.0}), plan_json(there), plan_json(there)),
                  'vote').plan(make_state())
    assert res['subgoals'] == [there]


def test_vote_survives_one_bad_voter():
    res = planner(Script('мусор', 'мусор', plan_json(C1), plan_json(C1)), 'vote').plan(make_state())
    assert res['subgoals'] == [C1] and res['source'] == 'llm'
    assert len(res['exchanges']) == 4


# --- предложения, таблица, выбор --------------------------------------------------------------

def proposals(*firsts):
    return {'plans': [json.loads(plan_json(sg)) for sg in firsts]}


def test_scored_model_chooses_from_the_table():
    script = Script(proposals(C1, E1, HOME), {'choice': 2, 'reasoning': 'так хочу'})
    res = planner(script, 'scored').plan(make_state())
    assert res['subgoals'] == [E1]
    assert roles(res) == ['proposals', 'choice']
    info = res['orchestration']
    assert info['proposals'] == ['investigate C1', 'explore E1', 'return_base']
    assert (info['model_choice'], info['calc_choice'], info['chosen_by']) == (2, 1, 'model')
    ask, table = script.asked
    assert PROPOSE_MARK in ask[-1]['content'] and 'Верни план' not in ask[-1]['content']
    assert '"sample_per_charge": 0.2' in table[-1]['content']       # 0.74 / (3.2 + 0.5)


def test_scored_calc_takes_the_same_proposals_and_never_asks_to_choose():
    state = make_state()
    asked = []
    for strategy in ('scored', 'scored_calc'):
        script = Script(proposals(HOME, E1, C1), {'choice': 1, 'reasoning': 'домой'})
        res = planner(script, strategy).plan(state)
        asked.append(script.asked[0])
        assert res['subgoals'] == ([HOME] if strategy == 'scored' else [C1])
    assert asked[0] == asked[1]                                      # один и тот же первый запрос
    assert roles(res) == ['proposals'] and res['orchestration']['chosen_by'] == 'calc'


def test_scored_bad_choice_keeps_models_first_plan_not_the_calculation():
    script = Script(proposals(HOME, C1), {'choice': 7}, 'мусор')
    res = planner(script, 'scored').plan(make_state())
    assert res['subgoals'] == [HOME] and res['orchestration']['chosen_by'] == 'first'
    assert res['orchestration']['calc_choice'] == 2


def test_scored_single_proposal_needs_no_choice():
    res = planner(Script(proposals(C1)), 'scored').plan(make_state())
    assert res['subgoals'] == [C1] and roles(res) == ['proposals']


def test_parse_proposals_drops_bad_and_repeated_plans():
    state = make_state()
    reply = proposals(C1, {'type': 'investigate', 'target': 'C99'}, C1, E1, HOME)
    plans, errors = parse_proposals(json.dumps(reply), state, k=3)
    assert [p.subgoals[0].type for p in plans] == ['investigate', 'explore', 'return_base'] and not errors
    assert parse_proposals(json.dumps(proposals({'type': 'explore', 'target': 'E9'})), state)[0] is None
    assert parse_proposals('{"plans": []}', state)[0] is None
    assert len(parse_proposals(plan_json(C1), state)[0]) == 1        # один план без обёртки тоже годится


def test_parse_choice_bounds():
    assert parse_choice('{"choice": 2, "reasoning": "x"}', 3)[0]['choice'] == 2
    assert parse_choice('{"choice": "2"}', 3)[0]['choice'] == 2
    for bad in ('{"choice": 4}', '{"choice": 0}', '{"choice": "второй"}', '{"reasoning": "без выбора"}'):
        assert parse_choice(bad, 3)[0] is None


def test_consequences_come_from_the_state_only():
    state = make_state()
    row = consequences(Plan.model_validate(json.loads(plan_json(C1, E1))), state)
    assert (row['sample_probability'], row['cost_to'], row['cost_back']) == (0.74, 3.2, 5.0)
    assert row['battery_after_return'] == 33.0 and row['enough_battery'] is True
    home = consequences(Plan.model_validate(json.loads(plan_json(HOME))), state)
    assert home['ends_run'] and home['battery_after_return'] == 36.1 and home['sample_probability'] == 0.0
    far = consequences(Plan.model_validate(json.loads(plan_json({'type': 'goto', 'x': 0, 'y': 0}))), state)
    assert far['cost_to'] is None and far['enough_battery'] is None


def test_calc_choice_is_the_rule_applied_to_the_proposals():
    state = make_state()
    state['candidates'].append({'id': 'C2', 'x': 0.0, 'y': 0.0, 'confidence': 0.6, 'cost_to': 1.0,
                                'cost_back': 4.0, 'feasible': True})
    rule = HeuristicPlanner().plan(state)['subgoals'][0]
    options = [HOME, E1, C1, {'type': 'investigate', 'target': 'C2'}]
    rows = [consequences(Plan.model_validate(json.loads(plan_json(sg))), state) for sg in options]
    assert options[calc_choice(rows, state)] == rule
    assert options[calc_choice(rows[:2], state)] == E1               # правила среди вариантов нет — лучший из них
    done = make_state(samples={'collected': 5, 'total': 5})
    assert options[calc_choice(rows, done)] == HOME


# --- запасное правило --------------------------------------------------------------------------

@pytest.mark.parametrize('strategy', ['critic', 'vote', 'scored', 'scored_calc'])
def test_garbage_replies_fall_back_to_the_rule(strategy):
    state = make_state()
    p = planner(lambda messages: 'это не JSON', strategy)
    res = p.plan(state)
    assert res['source'] == 'fallback' and p.failures == 1
    assert res['subgoals'] == HeuristicPlanner().plan(state)['subgoals']
    assert res['exchanges'] and not any(ex['ok'] for ex in res['exchanges'])
    assert res['error'] and 'Решение по правилу' in res['reasoning']


@pytest.mark.parametrize('strategy', ['critic', 'vote', 'scored', 'scored_calc'])
@pytest.mark.parametrize('fault', ['malformed', 'empty', 'invalid_target', 'http_500', 'wrapped'])
def test_mock_faults_never_break_a_strategy(strategy, fault):
    for seed in range(5):
        res = planner(MockResponder(seed=seed, faults={fault: 0.5}), strategy).plan(make_state())
        assert res['source'] in ('llm', 'fallback') and res['subgoals']
        assert all('role' in ex for ex in res['exchanges'])


def test_calls_per_decision():
    """Сколько обращений стоит одно решение, когда модель отвечает без ошибок."""
    counts = {s: len(planner(MockResponder(), s).plan(make_state())['exchanges'])
              for s in ('single', 'critic', 'vote', 'scored', 'scored_calc')}
    assert counts == {'single': 1, 'critic': 2, 'vote': 3, 'scored': 2, 'scored_calc': 1}


# --- характеры имитатора -----------------------------------------------------------------------

def test_normal_temperament_is_the_old_mock():
    for state in (make_state(), low_battery(), make_state(candidates=[]), {}):
        assert temper_plan(state) == temper_plan(state, 'normal') == mock_plan(state)
    with pytest.raises(ValueError):
        MockResponder(temperament='brave')
    assert set(TEMPERAMENTS) == {'normal', 'timid', 'random', 'careless'}


def test_timid_is_timid_in_every_role():
    state = low_battery()
    assert temper_plan(state, 'timid')['subgoals'] == [HOME]
    assert temper_plan(make_state(battery=45.0, battery_start=50.0), 'timid')['subgoals'][0] == C1
    assert temper_review(state, {'subgoals': [C1]}, 'timid')['verdict'] == 'revise'
    assert temper_review(state, {'subgoals': [HOME]}, 'timid')['verdict'] == 'accept'
    rows = [consequences(Plan.model_validate(json.loads(plan_json(sg))), state) for sg in (C1, HOME)]
    assert temper_choice(state, rows, 'timid')['choice'] == 2
    assert temper_choice(state, rows, 'normal')['choice'] == 1


def test_wrappers_do_not_cure_the_timid_mock_but_calculation_overrides_it():
    state = low_battery()
    first = {s: planner(MockResponder(temperament='timid'), s).plan(state)['subgoals'][0]
             for s in ('single', 'critic', 'vote', 'scored', 'scored_calc')}
    assert first == {'single': HOME, 'critic': HOME, 'vote': HOME, 'scored': HOME, 'scored_calc': C1}


def test_careless_and_random_temperaments():
    import random
    state = make_state()
    state['candidates'].append({'id': 'C9', 'x': 2.0, 'y': 2.0, 'confidence': 0.99, 'cost_to': 30.0,
                                'cost_back': 30.0, 'feasible': False})
    rng = random.Random(1)
    picks = [temper_plan(state, 'careless', rng)['subgoals'][0].get('target') for _ in range(60)]
    assert 5 < picks.count('C9') < 40 and set(picks) == {'C1', 'C9'}
    picks = {temper_plan(state, 'random', rng)['subgoals'][0]['target'] for _ in range(40)}
    assert picks == {'C1', 'E1'}                                     # наугад, но только из достижимых
    verdicts = {temper_review(state, {'subgoals': [C1]}, 'random', rng)['verdict'] for _ in range(40)}
    assert verdicts == {'accept', 'revise'}


# --- запись прогона и плата за ожидание ---------------------------------------------------------

def test_run_records_steps_calls_and_rule_match(tmp_path, monkeypatch):
    from did import runner
    from did.recorder import load_trace
    monkeypatch.setattr(runner, 'RUNS', tmp_path)
    out = runner.run_episode('medium', 1001, 'adaptive_llm', experiment='t', arm='scored',
                             llm={'kind': 'mock', 'temperament': 'timid'}, config={'llm_strategy': 'scored'})
    trace = load_trace(tmp_path / out['file'])
    decisions = [p for p in trace['plans'] if 'rule_match' in p]
    assert decisions and all(p['orchestration']['strategy'] == 'scored' for p in decisions)
    assert out['metrics']['llm_calls'] == len(trace['llm']) == sum(
        len([s for s in p['orchestration']['steps'] if s['role'] in ('proposals', 'choice')]) for p in decisions)
    assert any(not p['rule_match'] for p in decisions)              # робкий имитатор уехал домой раньше правила
    assert {ex['role'] for ex in trace['llm']} == {'proposals', 'choice'}


def test_default_strategy_gives_the_same_run_as_before(tmp_path, monkeypatch):
    from did import runner
    monkeypatch.setattr(runner, 'RUNS', tmp_path)
    plain = runner.run_episode('hard', 1002, 'adaptive_llm', experiment='t', arm='a', llm={'kind': 'mock'})
    named = runner.run_episode('hard', 1002, 'adaptive_llm', experiment='t', arm='b', llm={'kind': 'mock'},
                               config={'llm_strategy': 'single'})
    rule = runner.run_episode('hard', 1002, 'adaptive', experiment='t', arm='c')
    for key in ('score', 'samples_collected', 'battery_left', 'time', 'llm_calls'):
        assert plain['metrics'][key] == named['metrics'][key]
    assert plain['metrics']['score'] == rule['metrics']['score']    # имитатор отвечает правилом


def test_llm_wait_keeps_the_robot_standing_while_the_world_goes_on():
    from did.agent import Agent, make_config
    from did.arena import load_arena
    from did.fastsim import FastSim
    from did.runner import make_planner
    from did.scenario import generate
    arena = load_arena()
    cfg = make_config('adaptive_llm', llm_wait_s=5.0)
    scenario = generate('medium', 1001, arena)
    scenario.events = [{'t': 2.0, 'type': 'sensor_fault', 'sigma': 0.2, 'duration': 10.0}]
    world = FastSim(arena, scenario, seed=1001)
    bot = Agent(arena, cfg, n_samples=len(scenario.samples), planner=make_planner(cfg, {'kind': 'mock'}, 1001))
    start, battery = (world.x, world.y), world.judge.battery
    while world.t < 4.0:
        bot.tick(world.observe(), world)
        world.advance()
    assert (world.x, world.y) == start and bot.mode == 'think'       # робот стоит и «ждёт ответа»
    assert bot.queue                                                 # решение принято по состоянию на момент запроса
    assert world.judge.battery < battery                             # простой расходует заряд
    assert any(e['type'] == 'sensor_fault' for e in world.judge.world_log)   # события среды идут
    while world.t < 6.0:
        bot.tick(world.observe(), world)
        world.advance()
    assert (world.x, world.y) != start                               # ожидание кончилось — поехал
