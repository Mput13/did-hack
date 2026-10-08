"""Тесты способов оркестрации языковой модели (single, critic, vote, scored) и характеров имитатора."""
import copy
import json
import pytest

from did.llm import EXAMPLE_STATE, LocalClient, Plan
from did.llm_mock import MockResponder, mock_plan, mock_critique, mock_score_choice
from did.llm_orchestration import (
    OrchestratedLLMPlanner, plan_single, plan_critic, plan_vote, plan_scored,
    evaluate_plan_consequences, Critique, ScoredChoice
)
from did.planner import HeuristicPlanner


def test_single_orchestration():
    state = copy.deepcopy(EXAMPLE_STATE)
    client = LocalClient(MockResponder(seed=42))
    res = plan_single(client, state)
    assert res['source'] == 'llm'
    assert len(res['subgoals']) > 0
    assert len(res['exchanges']) == 1
    assert res['error'] is None


def test_critic_orchestration():
    state = copy.deepcopy(EXAMPLE_STATE)
    client = LocalClient(MockResponder(seed=42))
    res = plan_critic(client, state)
    assert res['source'] == 'llm'
    assert len(res['subgoals']) > 0
    # Автор + Критик (2) или с доработкой (3)
    assert len(res['exchanges']) in (2, 3)
    assert res['error'] is None


def test_vote_orchestration():
    state = copy.deepcopy(EXAMPLE_STATE)
    client = LocalClient(MockResponder(seed=42))
    res = plan_vote(client, state, n=3)
    assert res['source'] == 'llm'
    assert len(res['subgoals']) > 0
    assert len(res['exchanges']) == 3
    assert res['error'] is None


def test_scored_orchestration():
    state = copy.deepcopy(EXAMPLE_STATE)
    client = LocalClient(MockResponder(seed=42))
    res = plan_scored(client, state, k=3)
    assert res['source'] == 'llm'
    assert len(res['subgoals']) > 0
    # 3 генератора + 1 выбор из таблицы
    assert len(res['exchanges']) == 4
    assert res['error'] is None


def test_orchestration_limits():
    state = copy.deepcopy(EXAMPLE_STATE)
    
    # single: 1 вызов (при починке до 2)
    c1 = LocalClient(MockResponder(seed=1))
    r1 = plan_single(c1, state, max_repairs=1)
    assert len(r1['exchanges']) in (1, 2)
    
    # critic: 2 или 3 вызова
    c2 = LocalClient(MockResponder(seed=1))
    r2 = plan_critic(c2, state)
    assert len(r2['exchanges']) in (2, 3)
    
    # vote: N=3 вызова
    c3 = LocalClient(MockResponder(seed=1))
    r3 = plan_vote(c3, state, n=3)
    assert len(r3['exchanges']) == 3
    
    # scored: K+1 = 4 вызова
    c4 = LocalClient(MockResponder(seed=1))
    r4 = plan_scored(c4, state, k=3)
    assert len(r4['exchanges']) == 4


def test_graceful_fallback():
    # Имитатор со сплошными поломками
    state = copy.deepcopy(EXAMPLE_STATE)
    faulty_client = LocalClient(MockResponder(seed=1, script=['malformed'] * 10, faults={'malformed': 1.0}))
    
    # Все стратегии должны перейти на fallback без исключений
    r_single = plan_single(faulty_client, state, max_repairs=0)
    assert r_single['source'] == 'fallback'
    assert len(r_single['subgoals']) > 0
    
    r_critic = plan_critic(faulty_client, state, max_repairs=0)
    assert r_critic['source'] == 'fallback'
    assert len(r_critic['subgoals']) > 0
    
    r_vote = plan_vote(faulty_client, state, n=3, max_repairs=0)
    assert r_vote['source'] == 'fallback'
    assert len(r_vote['subgoals']) > 0
    
    r_scored = plan_scored(faulty_client, state, k=3, max_repairs=0)
    assert r_scored['source'] == 'fallback'
    assert len(r_scored['subgoals']) > 0


def test_temperament_timid():
    # При заряде < 70% timid возвращается на базу
    state_low = copy.deepcopy(EXAMPLE_STATE)
    state_low['battery'] = 30.0
    state_low['battery_start'] = 50.0  # 30 < 35 (70%)
    
    p = mock_plan(state_low, temperament='timid')
    assert p['subgoals'][0]['type'] == 'return_base'
    
    # При высоком заряде (> 70%) timid исследует цель
    state_high = copy.deepcopy(EXAMPLE_STATE)
    state_high['battery'] = 45.0
    state_high['battery_start'] = 50.0
    p_high = mock_plan(state_high, temperament='timid')
    assert p_high['subgoals'][0]['type'] != 'return_base'


def test_temperament_careless():
    state = copy.deepcopy(EXAMPLE_STATE)
    # Сделаем одного кандидата заведомо невыполнимым
    state['candidates'].append({
        'id': 'C99', 'x': 10.0, 'y': 10.0, 'cost_to': 50.0,
        'cost_back': 50.0, 'feasible': False, 'confidence': 0.99
    })
    
    # За 50 вызовов careless выберет невыполнимую цель хотя бы несколько раз
    import random
    rng = random.Random(42)
    infeasible_count = 0
    for _ in range(50):
        p = mock_plan(state, temperament='careless', rng=rng)
        if p['subgoals'] and p['subgoals'][0].get('target') == 'C99':
            infeasible_count += 1
    assert infeasible_count > 0, "Беспечный характер должен хотя бы иногда выбирать недостижимую цель"


def test_planner_class_orchestration():
    state = copy.deepcopy(EXAMPLE_STATE)
    client = LocalClient(MockResponder(seed=42))
    
    for strat in ('single', 'critic', 'vote', 'scored'):
        planner = OrchestratedLLMPlanner(client, strategy=strat)
        res = planner.plan(state)
        assert res['source'] == 'llm'
        assert len(res['subgoals']) > 0
        assert planner.calls == 1
        assert planner.failures == 0


def test_scored_does_not_inject_rule_and_calc_skips_selector():
    state = copy.deepcopy(EXAMPLE_STATE)
    state.update(battery=30, battery_start=50)
    for calculate in (False, True):
        res = plan_scored(LocalClient(MockResponder(temperament='timid')), state, calculate=calculate)
        assert res['subgoals'][0]['type'] == 'return_base'
        assert len(res['exchanges']) == (3 if calculate else 4)


def test_table_uses_first_step_and_values_exploration():
    state = copy.deepcopy(EXAMPLE_STATE)
    plan = Plan.model_validate({'reasoning': 'проверка', 'subgoals': [
        {'type': 'investigate', 'target': 'C1'}, {'type': 'investigate', 'target': 'C1'}]})
    cons = evaluate_plan_consequences(plan, state)
    assert cons['expected_samples'] == state['candidates'][0]['confidence']
    assert cons['cost_to'] == state['candidates'][0]['cost_to']
    returned = Plan.model_validate({'reasoning': 'домой', 'subgoals': [{'type': 'return_base'}]})
    assert evaluate_plan_consequences(returned, state)['expected_samples'] == 0


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        OrchestratedLLMPlanner(LocalClient(MockResponder()), strategy='typo')


def test_single_preserves_legacy_planner_result():
    from did.planner import LLMPlanner
    state = copy.deepcopy(EXAMPLE_STATE)
    old = LLMPlanner(LocalClient(MockResponder(seed=42))).plan(state)
    new = plan_single(LocalClient(MockResponder(seed=42)), state)
    assert old['subgoals'] == new['subgoals']
    assert old['reasoning'] == new['reasoning']


def test_invalid_selector_and_critic_are_not_silent_accepts():
    from did.llm_orchestration import parse_choice, parse_critique, parse_table_choice
    assert parse_choice({'reasoning': 'без выбора'})[0] is None
    assert parse_choice({'choice': 'oops'})[0] is None
    assert parse_table_choice({'choice': 4}, 3)[0] is None
    assert parse_critique({'advice': 'без вердикта'})[0] is None


def test_wait_advances_world_clock_while_robot_stands():
    from did.agent import Agent, make_config
    from did.arena import load_arena
    from did.fastsim import FastSim
    from did.scenario import generate
    from did.runner import make_planner
    arena = load_arena()
    cfg = make_config('adaptive_llm', llm_wait_s=5)
    scenario = generate('medium', 1001, arena)
    scenario.events = [{'t': 2.0, 'type': 'sensor_fault', 'sigma': 0.2, 'duration': 10.0}]
    world = FastSim(arena, scenario, seed=1001)
    bot = Agent(arena, cfg, n_samples=len(world.judge.scenario.samples),
                planner=make_planner(cfg, {'kind': 'mock'}, 1001))
    start = (world.x, world.y)
    for _ in range(40):
        bot.tick(world.observe(), world)
        world.advance()
    assert world.t == pytest.approx(4)
    assert (world.x, world.y) == start
    assert bot.mode == 'think'
    assert any(e['type'] == 'sensor_fault' for e in world.judge.world_log)


def test_vote_counts_distinct_goto_coordinates():
    answers = iter([{'type': 'goto', 'x': -1, 'y': 0},
                    {'type': 'goto', 'x': 1, 'y': 0},
                    {'type': 'goto', 'x': 1, 'y': 0}])
    client = LocalClient(lambda messages: json.dumps({'reasoning': 'голос', 'subgoals': [next(answers)]}))
    result = plan_vote(client, copy.deepcopy(EXAMPLE_STATE))
    assert result['subgoals'][0]['x'] == 1


def test_critic_revises_on_verdict_even_without_issues():
    answers = iter([
        {'reasoning': 'автор', 'subgoals': [{'type': 'return_base'}]},
        {'verdict': 'revise', 'issues': [], 'advice': 'Выбери кандидата'},
        {'reasoning': 'исправлено', 'subgoals': [{'type': 'investigate', 'target': 'C1'}]},
    ])
    result = plan_critic(LocalClient(lambda messages: json.dumps(next(answers))), copy.deepcopy(EXAMPLE_STATE))
    assert result['subgoals'][0]['target'] == 'C1'
    assert len(result['exchanges']) == 3
