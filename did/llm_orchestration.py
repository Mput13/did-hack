# flake8: noqa
"""Способы оркестрации языковой модели на одной и той же модели:
- single: один запрос к модели с валидацией и довызовом при ошибке (обёртка над request_plan).
- critic: автор (запрос 1) -> критик (запрос 2) -> исправление при замечаниях (запрос 3).
- vote: N запросов (по умолчанию 3) с разными индексами в запросе (temperature клиента 0.2), выбор большинства по первой подцели (тай-брейкер по confidence / cost_to).
- scored: модель предлагает до K планов -> детерминированный расчёт метрик последствий -> модель выбирает лучший план из таблицы.

При отсутствии пригодных планов — fallback на HeuristicPlanner. Ошибка критика оставляет план автора;
ошибка выбора из таблицы включает детерминированный выбор.
Все обращения к модели логируются в exchanges для llm_stats.
"""
from __future__ import annotations

import json
from typing import Literal
from pydantic import BaseModel, Field

from .llm import (
    Plan, request_plan, request_json, parse_plan,
    plan_schema, build_messages, _jsonable, extract_json
)
from .planner import HeuristicPlanner, LLMPlanner, MISSION

STRATEGIES = ('single', 'critic', 'vote', 'scored', 'scored_calc')


class Critique(BaseModel):
    verdict: Literal['accept', 'revise'] = 'accept'
    issues: list[str] = Field(default_factory=list)
    advice: str = ""


def parse_critique(raw):
    try:
        if isinstance(raw, str):
            raw = extract_json(raw)
        if not isinstance(raw, dict):
            return None, ["критик вернул не JSON-объект"]
        verdict = str(raw.get('verdict', '')).lower().strip()
        if verdict not in ('accept', 'revise'):
            return None, ['нужен verdict accept или revise']
        issues = raw.get('issues') or []
        if isinstance(issues, str):
            issues = [issues]
        elif not isinstance(issues, list):
            issues = []
        advice = str(raw.get('advice') or "")
        return Critique(verdict=verdict, issues=issues, advice=advice), []
    except Exception as e:
        return None, [f"ошибка разбора критики: {e}"]


class ScoredChoice(BaseModel):
    choice: int = 1
    reasoning: str = ""


def parse_choice(raw):
    try:
        if isinstance(raw, str):
            raw = extract_json(raw)
        if not isinstance(raw, dict):
            return None, ["выбор из таблицы вернул не JSON-объект"]
        choice = raw.get('choice')
        if type(choice) is not int or choice < 1:
            return None, ['choice должен быть положительным целым числом']
        reasoning = str(raw.get('reasoning') or "")
        return ScoredChoice(choice=choice, reasoning=reasoning), []
    except Exception as e:
        return None, [f"ошибка разбора выбора: {e}"]


def consequence_rank(cons):
    """Фиксированный порядок: допустимость, ожидаемые образцы, разведка/заряд, остаток."""
    return (cons['is_feasible'], cons['expected_samples'], cons['exploration_value'], cons['battery_left'])


def evaluate_plan_consequences(plan: Plan, state: dict) -> dict:
    """Приближение для первой подцели + возврата, а не сумма путей из одной исходной точки.

    Для goto нет оценки пути в состоянии: такой вариант помечаем неоценимым.
    Разведка имеет информационную ценность, но не обещает конкретного образца.
    """
    sg = plan.subgoals[0] if plan.subgoals else None
    items = {c['id']: c for c in state.get('candidates', []) + state.get('explore_points', [])}
    item = items.get(getattr(sg, 'target', None), {}) if sg else {}
    returning = bool(sg and sg.type == 'return_base')
    known = returning or bool(item)
    cost_to = float(item.get('cost_to', 0))
    cost_back = float(item.get('cost_back', state.get('return_cost', 0)))
    left = float(state.get('battery', 0)) - cost_to - cost_back
    samples = float(item.get('confidence', 0)) if sg and sg.type == 'investigate' else 0.0
    exploration = (float(item.get('gain_bits', item.get('unseen_share', 0))) / (cost_to + 1)
                   if sg and sg.type == 'explore' else 0.0)
    return {'expected_samples': samples, 'exploration_value': exploration,
            'cost_to': cost_to, 'cost_back': cost_back, 'total_cost': cost_to + cost_back,
            'battery_left': left, 'is_feasible': known and item.get('feasible', True) and left >= 2,
            'description': f'{sg.type} {getattr(sg, "target", "") or ""}' if sg else 'пустой план'}


def plan_single(client, state, system_prompt=None, fallback=None, max_repairs=1, mission=MISSION):
    fallback = fallback or HeuristicPlanner()
    full_state = {'mission': mission, **state}
    res = request_plan(client, full_state, system_prompt=system_prompt, max_repairs=max_repairs)
    if res.plan is None:
        out = fallback.plan(state)
        out.update(source='fallback', exchanges=res.exchanges, error=res.error, latency_ms=res.latency_ms)
        out['reasoning'] = f"Модель не дала пригодного плана ({res.error}). Решение по правилу: " + out['reasoning']
        return out
    plan = res.plan.model_dump(exclude_none=True)
    return {
        'reasoning': plan['reasoning'],
        'hypotheses': plan.get('hypotheses', []),
        'subgoals': plan['subgoals'],
        'source': 'llm',
        'exchanges': res.exchanges,
        'latency_ms': res.latency_ms,
        'error': None
    }


def plan_critic(client, state, system_prompt=None, fallback=None, max_repairs=1, mission=MISSION):
    fallback = fallback or HeuristicPlanner()
    full_state = {'mission': mission, **state}
    exchanges = []
    total_latency = 0

    # Шаг 1: Автор предлагает план
    author_res = request_plan(client, full_state, system_prompt=system_prompt, max_repairs=max_repairs, role='author')
    exchanges.extend(author_res.exchanges)
    total_latency += author_res.latency_ms or 0
    if author_res.plan is None:
        out = fallback.plan(state)
        out.update(source='fallback', exchanges=exchanges, error=author_res.error, latency_ms=total_latency)
        out['reasoning'] = f"Автор не дал пригодного плана ({author_res.error}). Решение по правилу: " + out['reasoning']
        return out

    current_plan = author_res.plan

    # Шаг 2: Критик оценивает план
    critic_system = (
        "Ты — строгий критик плана мобильного робота. Проверь предложенный план на ошибки:\n"
        "1. Достижимы ли цели (feasible=True)? Хватит ли заряда battery на cost_to и cost_back с запасом?\n"
        "2. Нет ли преждевременного возврата на базу (return_base), когда заряд высок и есть доступные кандидаты?\n"
        "3. Верни вердикт JSON: {\"verdict\": \"accept\" | \"revise\", \"issues\": [\"...\"], \"advice\": \"...\"}"
    )
    critic_input = {
        'role': 'critic',
        'state': {k: v for k, v in full_state.items() if k != 'mission'},
        'plan': current_plan.model_dump(exclude_none=True)
    }
    critic_messages = [
        {'role': 'system', 'content': critic_system},
        {'role': 'user', 'content': f"Критика плана:\n{json.dumps(critic_input, ensure_ascii=False, default=_jsonable)}\nВерни JSON с вердиктом."}
    ]

    critique_schema = Critique.model_json_schema()
    critique_obj, critic_err, critic_ex, critic_lat = request_json(
        client, critic_messages, parse_critique, schema=critique_schema, max_repairs=0, role='critic'
    )
    exchanges.extend(critic_ex)
    total_latency += critic_lat or 0

    # Шаг 3: Исправление при наличии замечаний
    if critique_obj and critique_obj.verdict == 'revise':
        issues_str = "\n".join(f"- {iss}" for iss in critique_obj.issues)
        revise_user_msg = (
            f"Критик нашёл замечания в твоём плане:\n{issues_str}\n"
            f"Совет критика: {critique_obj.advice}\n"
            f"Сформируй исправленный план с учётом замечаний. Верни один JSON-объект по схеме."
        )
        revise_messages = build_messages(full_state, system_prompt) + [
            {'role': 'assistant', 'content': json.dumps(current_plan.model_dump(exclude_none=True), ensure_ascii=False)},
            {'role': 'user', 'content': revise_user_msg}
        ]
        revised_plan, rev_err, rev_ex, rev_lat = request_json(
            client, revise_messages, lambda text: parse_plan(text, full_state),
            schema=plan_schema(full_state), max_repairs=max_repairs, role='author_revise'
        )
        exchanges.extend(rev_ex)
        total_latency += rev_lat or 0
        if revised_plan is not None:
            current_plan = revised_plan

    plan_dict = current_plan.model_dump(exclude_none=True)
    return {
        'reasoning': plan_dict['reasoning'],
        'hypotheses': plan_dict.get('hypotheses', []),
        'subgoals': plan_dict['subgoals'],
        'source': 'llm',
        'exchanges': exchanges,
        'latency_ms': total_latency,
        'error': None
    }


def plan_vote(client, state, system_prompt=None, fallback=None, n=3, max_repairs=1, mission=MISSION):
    fallback = fallback or HeuristicPlanner()
    full_state = {'mission': mission, **state}
    exchanges = []
    total_latency = 0
    valid_plans = []

    for i in range(n):
        voter_res = request_plan(client, {**full_state, 'voter_index': i + 1}, system_prompt=system_prompt, max_repairs=max_repairs,
                                 role=f"voter_{i+1}")
        exchanges.extend(voter_res.exchanges)
        total_latency += voter_res.latency_ms or 0
        if voter_res.plan is not None:
            valid_plans.append(voter_res.plan)

    if not valid_plans:
        out = fallback.plan(state)
        out.update(source='fallback', exchanges=exchanges, error='Все избиратели вернули негодные планы', latency_ms=total_latency)
        out['reasoning'] = "Голосование не дало пригодного плана. Решение по правилу: " + out['reasoning']
        return out

    if len(valid_plans) == 1:
        chosen = valid_plans[0]
    else:
        def sg_key(plan):
            if not plan.subgoals:
                return ('return_base', None)
            sg = plan.subgoals[0]
            return (sg.type, getattr(sg, 'target', None), getattr(sg, 'x', None), getattr(sg, 'y', None))

        from collections import Counter
        counts = Counter(sg_key(p) for p in valid_plans)
        max_count = max(counts.values())
        top_keys = [k for k, v in counts.items() if v == max_count]

        candidates_by_id = {c['id']: c for c in state.get('candidates', [])}
        points_by_id = {p['id']: p for p in state.get('explore_points', [])}

        def plan_score(plan):
            if not plan.subgoals:
                return -10.0
            sg = plan.subgoals[0]
            if sg.type == 'investigate':
                c = candidates_by_id.get(sg.target)
                if c:
                    return float(c.get('confidence', 0.0)) / (float(c.get('cost_to', 0.0)) + 0.5)
            elif sg.type == 'explore':
                p = points_by_id.get(sg.target)
                if p:
                    return float(p.get('unseen_share', 0.0)) / (float(p.get('cost_to', 0.0)) + 1.0)
            elif sg.type == 'return_base':
                return 0.0
            return -1.0

        candidate_plans = [p for p in valid_plans if sg_key(p) in top_keys]
        chosen = max(candidate_plans, key=plan_score)

    plan_dict = chosen.model_dump(exclude_none=True)
    return {
        'reasoning': plan_dict['reasoning'],
        'hypotheses': plan_dict.get('hypotheses', []),
        'subgoals': plan_dict['subgoals'],
        'source': 'llm',
        'exchanges': exchanges,
        'latency_ms': total_latency,
        'error': None
    }


def parse_table_choice(raw, count):
    choice, errors = parse_choice(raw)
    if choice is not None and choice.choice > count:
        return None, ['choice выходит за границы таблицы']
    return choice, errors


def plan_scored(client, state, system_prompt=None, fallback=None, k=3, max_repairs=1, mission=MISSION, calculate=False):
    fallback = fallback or HeuristicPlanner()
    full_state = {'mission': mission, **state}
    exchanges = []
    total_latency = 0
    candidate_plans = []

    # Шаг 1: Сбор до K вариантов плана от модели
    for i in range(k):
        cand_res = request_plan(client, {**full_state, 'proposal': {'index': i + 1, 'total': k,
                                'instruction': 'Предложи альтернативный план; рассматривай другие первые подцели.',
                                'previous': [p.model_dump(exclude_none=True)['subgoals'] for p in candidate_plans]}},
                                system_prompt=system_prompt, max_repairs=max_repairs, role=f"generator_{i+1}")
        exchanges.extend(cand_res.exchanges)
        total_latency += cand_res.latency_ms or 0
        if cand_res.plan is not None:
            cand_dump = cand_res.plan.model_dump(exclude_none=True)
            if not any(p.model_dump(exclude_none=True)['subgoals'] == cand_dump['subgoals'] for p in candidate_plans):
                candidate_plans.append(cand_res.plan)

    if not candidate_plans:
        out = fallback.plan(state)
        out.update(source='fallback', exchanges=exchanges, error='Модель не предложила вариантов плана', latency_ms=total_latency)
        out['reasoning'] = "Схема scored не получила пригодных планов. Решение по правилу: " + out['reasoning']
        return out

    # Шаг 2: Детерминированная оценка последствий каждого плана
    evaluated = [(p, evaluate_plan_consequences(p, state)) for p in candidate_plans]

    # Шаг 3: Формирование таблицы и выбор лучшего плана
    table_lines = [
        "№ | План | Ожидаемые образцы | Польза разведки/заряд | Расход (туда+обратно) | Остаток на базе | Хватает заряда?"
    ]
    for idx, (p, cons) in enumerate(evaluated, 1):
        ok_str = "Да" if cons['is_feasible'] else "Нет"
        table_lines.append(
            f"{idx} | {cons['description']} | образцы: {cons['expected_samples']} | "
            f"разведка: {cons['exploration_value']} | расход: {cons['total_cost']} | остаток: {cons['battery_left']} | хватает: {ok_str}"
        )
    table_text = "\n".join(table_lines)

    scorer_system = (
        "Ты — аналитик планов мобильного робота. Тебе дана таблица с детерминированным расчётом последствий "
        "для предложенных вариантов плана. Выбери наиболее выгодный и безопасный вариант:\n"
        "- Варианты с 'хватает: Нет' опасны аварийной остановкой из-за разряда батареи.\n"
        "- Порядок выбора: допустимость, ожидаемые образцы, польза разведки/заряд, остаток на базе (по убыванию).\n"
        "Верни JSON: {\"choice\": номер_варианта, \"reasoning\": \"краткое объяснение\"}"
    )
    scorer_input = {
        'role': 'scorer',
        'state': {k: v for k, v in full_state.items() if k != 'mission'},
        'table': table_lines,
        'candidates': [cons for _, cons in evaluated]
    }
    scorer_messages = [
        {'role': 'system', 'content': scorer_system},
        {'role': 'user', 'content': f"Таблица вариантов:\n{table_text}\n\nДанные:\n{json.dumps(scorer_input, ensure_ascii=False, default=_jsonable)}\nВыбери лучший вариант плана из таблицы и верни JSON."}
    ]

    choice_schema = ScoredChoice.model_json_schema()
    choice_obj = None
    if not calculate:
        choice_obj, choice_err, choice_ex, choice_lat = request_json(
            client, scorer_messages, lambda raw: parse_table_choice(raw, len(candidate_plans)), schema=choice_schema, max_repairs=max_repairs, role='scorer'
        )
        exchanges.extend(choice_ex)
        total_latency += choice_lat or 0
    chosen_idx = max(range(len(evaluated)), key=lambda i: consequence_rank(evaluated[i][1]))
    if choice_obj and 1 <= choice_obj.choice <= len(candidate_plans):
        chosen_idx = choice_obj.choice - 1
    chosen_plan = candidate_plans[chosen_idx]
    plan_dict = chosen_plan.model_dump(exclude_none=True)
    if choice_obj and choice_obj.reasoning:
        reasoning = f"[Выбор {chosen_idx + 1}: {choice_obj.reasoning}] " + plan_dict['reasoning']
    else:
        reasoning = plan_dict['reasoning']

    return {
        'reasoning': reasoning,
        'hypotheses': plan_dict.get('hypotheses', []),
        'subgoals': plan_dict['subgoals'],
        'source': 'llm',
        'exchanges': exchanges,
        'latency_ms': total_latency,
        'error': None
    }


def plan_orchestrated(strategy, client, state, fallback=None, system_prompt=None, max_repairs=1, mission=MISSION):
    if strategy == 'critic':
        return plan_critic(client, state, system_prompt=system_prompt, fallback=fallback, max_repairs=max_repairs, mission=mission)
    elif strategy == 'vote':
        return plan_vote(client, state, system_prompt=system_prompt, fallback=fallback, n=3, max_repairs=max_repairs, mission=mission)
    elif strategy in ('scored', 'scored_calc'):
        return plan_scored(client, state, system_prompt=system_prompt, fallback=fallback, k=3, max_repairs=max_repairs, mission=mission, calculate=strategy == 'scored_calc')
    else:
        return plan_single(client, state, system_prompt=system_prompt, fallback=fallback, max_repairs=max_repairs, mission=mission)


class OrchestratedLLMPlanner(LLMPlanner):
    def __init__(self, client, strategy='single', fallback=None, system_prompt=None, max_repairs=1):
        super().__init__(client, fallback=fallback, system_prompt=system_prompt, max_repairs=max_repairs, strategy=strategy)

    def plan(self, state):
        self.calls += 1
        res = plan_orchestrated(self.strategy, self.client, state, fallback=self.fallback,
                                system_prompt=self.system_prompt, max_repairs=self.max_repairs,
                                mission=MISSION)
        if res.get('source') == 'fallback':
            self.failures += 1
        return res
