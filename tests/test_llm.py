"""Транспорт и обвязка планировщика: разбор ответа, схема, проверки плана, клиент, имитатор модели."""
import copy
import json
import logging
import os
import random
import time

import httpx
import numpy as np
import pytest

from did.llm import (EXAMPLE_STATE, ChatClient, LLMError, LocalClient, Plan, PlanResult, build_messages,
                     extract_json, load_env, load_system_prompt, parse_plan, request_plan, validate_plan)
from did.llm_mock import FAULTS, MockResponder, _parse_faults, mock_plan, start_mock_server

PLAN = {
    'reasoning': 'Еду к C1: уверенность 0.74, заряда хватает с запасом. Потом разведка E1 и база.',
    'hypotheses': [{'statement': 'S1 дороже обычного пола в 3 раза', 'test': 'сравнить расход на 0.5 м по краю'}],
    'subgoals': [{'type': 'investigate', 'target': 'C1'}, {'type': 'explore', 'target': 'E1'},
                 {'type': 'return_base'}],
}
PLAN_JSON = json.dumps(PLAN, ensure_ascii=False)
C1 = {'type': 'investigate', 'target': 'C1'}
E1 = {'type': 'explore', 'target': 'E1'}
HOME = {'type': 'return_base'}
KEY = 'sk-test-secret-0123456789'


def make_state(**changes):
    state = copy.deepcopy(EXAMPLE_STATE)
    state.update(changes)
    return state


def make_plan(*subgoals):
    return Plan.model_validate({'reasoning': 'проверка', 'subgoals': list(subgoals)})


def goto(x, y):
    return {'type': 'goto', 'x': x, 'y': y}


class Scripted:
    """Ответчик по списку: строка — ответ, исключение — сбой."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# --- extract_json ------------------------------------------------------------------------------

WRAPPERS = {
    'как есть': PLAN_JSON,
    'пробелы вокруг': f'\n\n  {PLAN_JSON}  \n',
    'с отступами': json.dumps(PLAN, ensure_ascii=False, indent=2),
    'блок json': f'```json\n{PLAN_JSON}\n```',
    'блок без языка': f'```\n{PLAN_JSON}\n```',
    'текст до и после': f'Вот план:\n{PLAN_JSON}\nГотов уточнить.',
    'think перед ответом': f'<think>\nСначала {{"черновик": 1}}, потом ответ.\n</think>\n{PLAN_JSON}',
    'think заглавными': f'<THINK>шум {{</THINK>{PLAN_JSON}',
    'think без открывающего тега': f'Думаю про {{"x": 1}}...\n</think>\n\n{PLAN_JSON}',
    'скобки в тексте до ответа': f'Формат {{type, target}} соблюдён. {PLAN_JSON}',
    'всё сразу': f'<think>план {{"a": [1, 2}}</think>\nИтог:\n```json\n{PLAN_JSON}\n```\nКонец {{не json}}.',
    'лишняя запятая': PLAN_JSON[:-1] + ',\n}',
}


@pytest.mark.parametrize('text', WRAPPERS.values(), ids=WRAPPERS.keys())
def test_extract_json_wrappers(text):
    assert extract_json(text) == PLAN


def test_extract_json_braces_and_quotes_inside_strings():
    obj = {'reasoning': 'скобки } { и кавычка " и \\ внутри строки', 'вложено': {'глубже': [{'a': 1}]}}
    assert extract_json('ответ: ' + json.dumps(obj, ensure_ascii=False) + ' } хвост') == obj


def test_extract_json_takes_first_object():
    assert extract_json('{"a": 1} и ещё {"b": 2}') == {'a': 1}


@pytest.mark.parametrize('text', ['', '  \n', None, 'плана нет', PLAN_JSON[:40], '[1, 2, 3]', '{не json}',
                                  '<think>{"a": 1}</think>', '<think>не закончил {"a": 1}'])
def test_extract_json_failure(text):
    with pytest.raises(LLMError):
        extract_json(text)


# --- схема -------------------------------------------------------------------------------------

def test_schema_accepts_limits_and_extras():
    data = {'reasoning': 'я' * 600, 'hypotheses': [{'statement': 'a', 'test': 'b'}] * 3,
            'subgoals': [goto(0.1 * i, 0) for i in range(5)] + [HOME], 'confidence': 0.9}
    plan, errors = parse_plan(json.dumps(data), EXAMPLE_STATE)
    assert errors == [] and len(plan.subgoals) == 6 and len(plan.hypotheses) == 3
    plan, errors = parse_plan('{"reasoning": " ок ", "hypotheses": null, "subgoals": '
                              '[{"type": "investigate", "target": " C1 ", "x": 0.5, "note": "лишнее"}]}', EXAMPLE_STATE)
    assert errors == [] and plan.model_dump() == {'reasoning': 'ок', 'hypotheses': [], 'subgoals': [C1]}
    assert Plan.model_validate({'reasoning': 'ок', 'subgoals': [HOME]}).hypotheses == []


SCHEMA_ERRORS = [
    ({'subgoals': [HOME]}, 'reasoning: нет обязательного поля'),
    ({'reasoning': '  ', 'subgoals': [HOME]}, 'reasoning: пустая строка'),
    ({'reasoning': 'я' * 601, 'subgoals': [HOME]}, 'reasoning: строка длиннее 600 символов (сейчас 601)'),
    ({'reasoning': 'ок'}, 'subgoals: нет обязательного поля'),
    ({'reasoning': 'ок', 'subgoals': []}, 'subgoals: нужно не меньше 1 элементов'),
    ({'reasoning': 'ок', 'subgoals': [goto(0.1 * i, 0) for i in range(7)]}, 'subgoals: нужно не больше 6 элементов'),
    ({'reasoning': 'ок', 'subgoals': [HOME], 'hypotheses': [{'statement': 'a', 'test': 'b'}] * 4},
     'hypotheses: нужно не больше 3 элементов'),
    ({'reasoning': 'ок', 'subgoals': [HOME], 'hypotheses': [{'statement': 'a'}]},
     'hypotheses[0].test: нет обязательного поля'),
    ({'reasoning': 'ок', 'subgoals': [{'type': 'fly', 'target': 'C1'}]}, 'subgoals[0]: неизвестный тип "fly"'),
    ({'reasoning': 'ок', 'subgoals': [C1, {'target': 'C1'}]}, 'subgoals[1]: у подцели нет поля "type"'),
    ({'reasoning': 'ок', 'subgoals': [{'type': 'investigate'}]}, 'subgoals[0].target: нет обязательного поля'),
    ({'reasoning': 'ок', 'subgoals': [{'type': 'goto', 'x': 1.0}]}, 'subgoals[0].y: нет обязательного поля'),
    ({'reasoning': 'ок', 'subgoals': [goto('справа', 0)]}, 'subgoals[0].x: нужно число'),
    ({'reasoning': 'ок', 'subgoals': 'return_base'}, 'subgoals: нужен список'),
]


@pytest.mark.parametrize('data, expected', SCHEMA_ERRORS, ids=[e for _, e in SCHEMA_ERRORS])
def test_schema_errors_are_readable(data, expected):
    plan, errors = parse_plan(json.dumps(data, ensure_ascii=False), EXAMPLE_STATE)
    assert plan is None
    assert any(e.startswith(expected) for e in errors), errors


def test_schema_rejects_infinite_coordinates():
    plan, errors = parse_plan('{"reasoning": "ок", "subgoals": [{"type": "goto", "x": Infinity, "y": NaN}]}', {})
    assert plan is None and len(errors) == 2 and all('конечное число' in e for e in errors)


# --- validate_plan -----------------------------------------------------------------------------

def test_validate_ok():
    assert validate_plan(Plan.model_validate(PLAN), EXAMPLE_STATE) == []
    assert validate_plan(PLAN, EXAMPLE_STATE) == []                          # словарь тоже годится
    assert validate_plan(make_plan(C1, E1, C1, goto(2.6, -2.6), HOME), EXAMPLE_STATE) == []


def test_validate_unknown_target():
    errors = validate_plan(make_plan({'type': 'investigate', 'target': 'C7'}), EXAMPLE_STATE)
    assert errors == ['subgoals[0]: цели "C7" нет в candidates (C1)']
    errors = validate_plan(make_plan(C1, {'type': 'explore', 'target': 'E9'}), EXAMPLE_STATE)
    assert errors == ['subgoals[1]: цели "E9" нет в explore_points (E1)']


def test_validate_target_from_wrong_list():
    errors = validate_plan(make_plan({'type': 'investigate', 'target': 'E1'}), EXAMPLE_STATE)
    assert len(errors) == 1 and 'explore_points' in errors[0] and 'тип "explore"' in errors[0]
    errors = validate_plan(make_plan({'type': 'explore', 'target': 'C1'}), EXAMPLE_STATE)
    assert len(errors) == 1 and 'тип "investigate"' in errors[0]


@pytest.mark.parametrize('x, y, ok', [(2.6, -2.6, True), (0.0, 0.0, True), (2.61, 0.0, False), (0.0, -2.7, False),
                                      (-5.0, 5.0, False)])
def test_validate_goto_inside_arena(x, y, ok):
    errors = validate_plan(make_plan(goto(x, y)), EXAMPLE_STATE)
    assert (errors == []) is ok
    assert ok or ('вне арены' in errors[0] and '2.6' in errors[0])


def test_validate_return_base_only_last():
    errors = validate_plan(make_plan(HOME, C1), EXAMPLE_STATE)
    assert errors == ['subgoals[0]: return_base может быть только последней подцелью']
    assert validate_plan(make_plan(C1, HOME), EXAMPLE_STATE) == []
    assert validate_plan(make_plan(HOME), EXAMPLE_STATE) == []


def test_validate_no_repeat_in_a_row():
    errors = validate_plan(make_plan(C1, C1, E1), EXAMPLE_STATE)
    assert len(errors) == 1 and errors[0].startswith('subgoals[1]: повтор')
    errors = validate_plan(make_plan(goto(1.0, 1.0), goto(1.0, 1.0)), EXAMPLE_STATE)
    assert len(errors) == 1 and errors[0].startswith('subgoals[1]: повтор')
    assert validate_plan(make_plan(goto(1.0, 1.0), goto(1.0, 1.2), C1, E1, C1), EXAMPLE_STATE) == []


@pytest.mark.parametrize('flag', [False, np.False_, 0])
def test_validate_infeasible_target(flag):
    state = make_state()
    state['candidates'][0]['feasible'] = flag
    errors = validate_plan(make_plan(C1, E1), state)
    assert len(errors) == 1 and errors[0].startswith('subgoals[0]: цель "C1" недостижима')
    state = make_state()
    state['explore_points'][0]['feasible'] = flag
    errors = validate_plan(make_plan(C1, E1), state)
    assert len(errors) == 1 and errors[0].startswith('subgoals[1]: цель "E1" недостижима')


def test_validate_empty_lists():
    for state in ({}, None, make_state(candidates=[], explore_points=[])):
        assert validate_plan(make_plan(HOME), state) == []
        assert validate_plan(make_plan(C1), state) == ['subgoals[0]: цели "C1" нет в candidates (список пуст)']


def test_validate_reports_every_error():
    errors = validate_plan(make_plan(HOME, {'type': 'investigate', 'target': 'C7'}, goto(9, 9)), EXAMPLE_STATE)
    assert len(errors) == 3


# --- промпт и окружение ------------------------------------------------------------------------

def test_prompt_matches_schema():
    prompt = load_system_prompt()
    assert len(prompt.splitlines()) <= 90
    for key in EXAMPLE_STATE:
        assert f'`{key}`' in prompt, key
    # Три примера: кандидат и разведка; кандидатов нет; ни одна цель не достижима.
    stuck = make_state(battery=9.0)
    stuck['candidates'][0]['feasible'] = stuck['explore_points'][0]['feasible'] = False
    cases = [(EXAMPLE_STATE, [C1, E1], 1), (make_state(candidates=[]), [E1], 0), (stuck, [HOME], 0)]
    answers = prompt.split('# Примеры')[1].split('Ответ:')[1:]
    assert len(answers) == len(cases)
    for text, (state, subgoals, n_hypotheses) in zip(answers, cases):
        plan, errors = parse_plan(text, state)
        assert errors == [] and plan.model_dump()['subgoals'] == subgoals
        assert len(plan.hypotheses) == n_hypotheses and len(plan.reasoning) <= 400
    old = load_system_prompt('planner_system_v1')                # прежняя версия: с ней сравниваются прогоны
    assert old != prompt and parse_plan(old.split('# Пример')[1], EXAMPLE_STATE)[1] == []


@pytest.fixture
def env():
    saved = dict(os.environ)
    for name in [n for n in os.environ if n.startswith('DID_LLM_')]:
        del os.environ[name]
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)


def test_load_env(tmp_path, env):
    path = tmp_path / '.env'
    path.write_text('# комментарий\n\nDID_LLM_BASE_URL=http://example.test/v1   # адрес\n'
                    'export DID_LLM_MODEL="qwen-plus"  # модель\n'
                    f"DID_LLM_API_KEY='{KEY}'\nDID_LLM_TIMEOUT_S = 12\nстрока без знака равенства\n", encoding='utf-8')
    env['DID_LLM_API_KEY'] = 'ключ-из-окружения'
    assert sorted(load_env(path)) == ['DID_LLM_BASE_URL', 'DID_LLM_MODEL', 'DID_LLM_TIMEOUT_S']
    assert env['DID_LLM_BASE_URL'] == 'http://example.test/v1'
    assert env['DID_LLM_MODEL'] == 'qwen-plus'
    assert env['DID_LLM_TIMEOUT_S'] == '12'
    assert env['DID_LLM_API_KEY'] == 'ключ-из-окружения'            # заданное раньше не перетирается
    assert load_env(tmp_path / 'нет-такого') == []


def test_from_env(tmp_path, env):
    assert ChatClient.from_env(env_path=None) is None
    assert ChatClient.from_env(env_path=tmp_path / 'нет-такого') is None
    env.update(DID_LLM_BASE_URL='http://127.0.0.1:1/v1/', DID_LLM_API_KEY=KEY, DID_LLM_MODEL='deepseek-chat',
               DID_LLM_TIMEOUT_S='7.5', DID_LLM_MAX_RETRIES='4', DID_LLM_JSON_MODE='0',
               DID_LLM_EXTRA_BODY='{"enable_thinking": false}')
    with ChatClient.from_env(env_path=None) as client:
        assert (client.base_url, client.model, client.timeout_s) == ('http://127.0.0.1:1/v1', 'deepseek-chat', 7.5)
        assert (client.max_retries, client.json_mode, client.extra_body) == (4, False, {'enable_thinking': False})
        assert KEY not in repr(client) and 'deepseek-chat' in repr(client)
    for name, value in [('DID_LLM_TIMEOUT_S', 'быстро'), ('DID_LLM_EXTRA_BODY', '[1]'), ('DID_LLM_BASE_URL', 'ai.mai.ru')]:
        old, env[name] = env[name], value
        with pytest.raises(LLMError, match=value if name == 'DID_LLM_TIMEOUT_S' else None):
            ChatClient.from_env(env_path=None)
        env[name] = old


def test_from_env_reads_file(tmp_path, env):
    path = tmp_path / '.env'
    path.write_text('DID_LLM_BASE_URL=http://127.0.0.1:1/v1/chat/completions\nDID_LLM_MODEL=qwen\n')
    with ChatClient.from_env(env_path=path) as client:
        assert (client.base_url, client.model, client.timeout_s, client.json_mode) == ('http://127.0.0.1:1/v1', 'qwen',
                                                                                    30.0, True)


# --- request_plan через LocalClient ------------------------------------------------------------

def test_mock_plan_rule():
    assert mock_plan(EXAMPLE_STATE)['subgoals'] == [C1, E1]
    assert mock_plan(make_state(battery=9.0))['subgoals'] == [HOME]                     # не хватит на возврат
    assert mock_plan(make_state(candidates=[]))['subgoals'] == [E1]
    assert mock_plan(make_state(samples={'collected': 5, 'total': 5}))['subgoals'] == [HOME]
    assert mock_plan({})['subgoals'] == [HOME]
    state = make_state()
    state['candidates'] += [{'id': 'C2', 'confidence': 0.9, 'cost_to': 1.0, 'cost_back': 4.0, 'feasible': False},
                            {'id': 'C3', 'confidence': 0.6, 'cost_to': 1.0, 'cost_back': 4.0, 'feasible': True}]
    assert mock_plan(state)['subgoals'][0] == {'type': 'investigate', 'target': 'C3'}   # 0.6/1.5 > 0.74/3.7


def test_request_plan_ok():
    res = request_plan(LocalClient(MockResponder()), EXAMPLE_STATE)
    assert isinstance(res, PlanResult) and res.ok and res.error is None
    assert res.plan.model_dump()['subgoals'] == [C1, E1]
    assert 'C1' in res.plan.reasoning and len(res.plan.hypotheses) == 1
    assert len(res.exchanges) == 1
    ex = res.exchanges[0]
    assert ex['ok'] and ex['errors'] == [] and ex['attempt'] == 1 and ex['role'] == 'user'
    assert ex['request'].startswith('Состояние:') and len(ex['request']) <= 301
    assert json.loads(ex['response'])['subgoals'] == [C1, E1]
    assert res.latency_ms >= ex['latency_ms'] >= 0
    json.dumps(res.to_dict())                                                 # журнал пишется в JSON как есть


def test_request_plan_sends_prompt_and_state():
    responder = Scripted(PLAN_JSON, PLAN_JSON)
    request_plan(LocalClient(responder), {'mission': 'собери образцы', **EXAMPLE_STATE})
    system, user = responder.seen[0]
    assert system == {'role': 'system', 'content': load_system_prompt()}
    assert user['role'] == 'user' and extract_json(user['content'])['mission'] == 'собери образцы'
    request_plan(LocalClient(responder), EXAMPLE_STATE, system_prompt='Свой промпт. Ответ в JSON.')
    assert responder.seen[1][0]['content'] == 'Свой промпт. Ответ в JSON.'


def test_request_plan_repairs_after_error():
    bad = json.dumps({**PLAN, 'subgoals': [{'type': 'investigate', 'target': 'C9'}, goto(3.0, 0.0)]})
    responder = Scripted(bad, f'```json\n{PLAN_JSON}\n```')
    res = request_plan(LocalClient(responder), EXAMPLE_STATE)
    assert res.plan == Plan.model_validate(PLAN) and res.error is None
    assert [ex['ok'] for ex in res.exchanges] == [False, True]
    assert [ex['attempt'] for ex in res.exchanges] == [1, 2]
    first, second = res.exchanges
    assert len(first['errors']) == 2 and 'C9' in first['errors'][0] and 'вне арены' in first['errors'][1]
    assert first['response'] == bad and second['request'].startswith('Ответ не принят')
    repair = responder.seen[1]
    assert [m['role'] for m in repair] == ['system', 'user', 'assistant', 'user']
    assert repair[2]['content'] == bad
    assert all(e in repair[3]['content'] for e in first['errors'])


@pytest.mark.parametrize('fault', ['empty', 'malformed', 'invalid_target', 'out_of_arena'])
def test_request_plan_repairs_every_mock_fault(fault):
    responder = MockResponder(seed=3, faults={fault: 1.0})
    res = request_plan(LocalClient(responder), EXAMPLE_STATE)
    assert res.ok and res.plan.model_dump()['subgoals'] == [C1, E1]
    assert [ex['ok'] for ex in res.exchanges] == [False, True] and res.exchanges[0]['errors']
    assert responder.stats == {fault: 1, 'ok': 1}


def test_request_plan_reads_wrapped_answer_at_once():
    responder = MockResponder(faults={'wrapped': 1.0})
    res = request_plan(LocalClient(responder), EXAMPLE_STATE)
    assert res.ok and len(res.exchanges) == 1 and '<think>' in res.exchanges[0]['response']


@pytest.mark.parametrize('max_repairs', [0, 1, 3])
def test_request_plan_gives_up(max_repairs):
    responder = Scripted(*['Не могу составить план.'] * 5)
    res = request_plan(LocalClient(responder), EXAMPLE_STATE, max_repairs=max_repairs)
    assert res.plan is None and not res.ok
    assert f'после {max_repairs + 1} попыток' in res.error and 'нет JSON-объекта' in res.error
    assert len(res.exchanges) == len(responder.seen) == max_repairs + 1
    assert not any(ex['ok'] for ex in res.exchanges)


def test_request_plan_empty_answer_is_not_sent_back_empty():
    responder = Scripted('<think>думаю</think>', PLAN_JSON)
    res = request_plan(LocalClient(responder), EXAMPLE_STATE)
    assert res.ok and responder.seen[1][2] == {'role': 'assistant', 'content': '(пустой ответ)'}


def test_request_plan_never_raises():
    def boom(messages):
        raise RuntimeError('ответчик упал')

    for client in (LocalClient(boom), LocalClient(lambda messages: 42), LocalClient(Scripted(LLMError('нет сети'))),
                   None, object()):
        res = request_plan(client, EXAMPLE_STATE)
        assert res.plan is None and res.error and res.latency_ms >= 0
    assert 'ответчик упал' in request_plan(LocalClient(boom), EXAMPLE_STATE).error
    for state in (None, {}, [], 'строка', make_state(candidates=None), {'candidates': [None, 5, {}]}):
        assert isinstance(request_plan(LocalClient(MockResponder()), state), PlanResult)
    assert request_plan(LocalClient(MockResponder()), EXAMPLE_STATE, max_repairs='много').plan is None


def test_request_plan_accepts_numpy_state():
    state = make_state(battery=np.float32(41.2), samples={'collected': np.int64(2), 'total': 5},
                       pose={'x': np.float64(0.42), 'y': np.array(1.1)})
    state['candidates'][0]['feasible'] = np.True_
    state['explore_points'][0]['cost_to'] = float('inf')                      # недостижимая точка в оценках агента
    res = request_plan(LocalClient(MockResponder()), state)
    assert res.ok and res.plan.model_dump()['subgoals'] == [C1]


def test_mock_is_deterministic():
    faults = dict.fromkeys(FAULTS, 0.3)
    messages = build_messages(EXAMPLE_STATE)

    def kinds(seed):
        responder = MockResponder(seed=seed, faults=faults)
        return [responder.draw(messages) for _ in range(40)]

    assert kinds(5) == kinds(5) and kinds(5) != kinds(6)
    assert {kind for kind, _ in kinds(5)} > {'ok', 'http_500', 'timeout', 'malformed'}
    with pytest.raises(ValueError, match='опечатка'):
        MockResponder(faults={'опечатка': 0.5})
    assert _parse_faults('malformed=0.2, http_500=0.1') == {'malformed': 0.2, 'http_500': 0.1}


# --- ChatClient по настоящему HTTP к имитатору -------------------------------------------------

@pytest.fixture
def mock():
    servers = []

    def start(**options):
        server, url = start_mock_server(**options)
        servers.append(server)
        return server, url

    yield start
    for server in servers:
        server.stop()


def connect(url, **options):
    options.setdefault('retry_pause_s', 0.0)
    return ChatClient(url, KEY, 'mock-planner', **options)


def fake_network(client, handler):
    """Вместо сети — функция handler(request) -> httpx.Response: для ответов, которых имитатор не даёт."""
    headers = client._http.headers
    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(handler), headers=headers)


def test_http_ok(mock):
    server, url = mock()
    with connect(url) as client:
        assert client.models() == ['mock-planner']
        res = request_plan(client, EXAMPLE_STATE)
    assert res.ok and res.plan.model_dump()['subgoals'] == [C1, E1]
    ex = res.exchanges[0]
    assert ex['http_attempts'] == 1 and ex['usage']['total_tokens'] > 0
    sent = server.requests[0]
    assert sent['model'] == 'mock-planner' and sent['response_format'] == {'type': 'json_object'}
    assert (sent['temperature'], sent['max_tokens']) == (0.2, 800)
    assert [m['role'] for m in sent['messages']] == ['system', 'user']
    assert 'Миссия' in sent['messages'][0]['content']                         # кириллица доходит как есть


def test_http_extra_body_and_parameters(mock):
    server, url = mock()
    with connect(url, json_mode=False, extra_body={'enable_thinking': False, 'max_tokens': 2000}) as client:
        reply = client.chat([{'role': 'user', 'content': 'привет'}], temperature=None)
    assert reply.attempts == 1 and json.loads(reply.text)['subgoals'] == [HOME]   # состояния нет — на базу
    sent = server.requests[0]
    assert sent['enable_thinking'] is False and sent['max_tokens'] == 2000
    assert 'response_format' not in sent and 'temperature' not in sent


def test_http_500_then_retry(mock):
    server, url = mock(script=['http_500', 'ok'])
    with connect(url) as client:
        res = request_plan(client, EXAMPLE_STATE)
    assert res.ok and len(res.exchanges) == 1 and res.exchanges[0]['http_attempts'] == 2
    assert server.responder.stats == {'http_500': 1, 'ok': 1}


def test_http_500_exhausts_retries(mock):
    server, url = mock(faults={'http_500': 1.0})
    with connect(url, max_retries=2) as client:
        with pytest.raises(LLMError, match='за 3 попыток: HTTP 500'):
            client.chat(build_messages(EXAMPLE_STATE))
        assert server.responder.stats['http_500'] == 3
        res = request_plan(client, EXAMPLE_STATE)
    assert res.plan is None and 'HTTP 500' in res.error and len(res.exchanges) == 1
    assert res.exchanges[0]['response'] is None and 'HTTP 500' in res.exchanges[0]['errors'][0]


def test_http_429_waits_retry_after(mock):
    server, url = mock(script=['http_429', 'ok'], retry_after_s=0.25)
    with connect(url) as client:
        reply = client.chat(build_messages(EXAMPLE_STATE))
    assert reply.attempts == 2 and reply.latency_ms >= 250


def test_http_timeout_then_retry(mock):
    server, url = mock(script=['timeout', 'ok'], timeout_sleep_s=2.0)
    with connect(url, timeout_s=0.2) as client:
        res = request_plan(client, EXAMPLE_STATE)
    assert res.ok and res.exchanges[0]['http_attempts'] == 2
    assert 180 <= res.latency_ms < 1500


def test_http_timeout_exhausts_retries(mock):
    server, url = mock(faults={'timeout': 1.0}, timeout_sleep_s=2.0)
    with connect(url, timeout_s=0.1, max_retries=1) as client:
        t0 = time.monotonic()
        with pytest.raises(LLMError, match='за 2 попыток: таймаут 0.1 с'):
            client.chat(build_messages(EXAMPLE_STATE))
        assert 0.18 <= time.monotonic() - t0 < 1.0                             # два таймаута, без сна сервера
        res = request_plan(client, EXAMPLE_STATE)
    assert res.plan is None and 'таймаут' in res.error and res.exchanges[0]['http_attempts'] == 2


def test_http_response_format_fallback(mock):
    server, url = mock(reject_response_format=True)
    with connect(url) as client:
        assert client.json_mode
        reply = client.chat(build_messages(EXAMPLE_STATE))
        assert reply.attempts == 2 and json.loads(reply.text)['subgoals'] == [C1, E1]
        assert not client.json_mode                                           # клиент запомнил
        assert ['response_format' in r for r in server.requests] == [True, False]
        assert request_plan(client, EXAMPLE_STATE).exchanges[0]['http_attempts'] == 1
        assert ['response_format' in r for r in server.requests] == [True, False, False]


def test_http_400_for_another_reason(mock):
    server, url = mock()
    with connect(url) as client:
        with pytest.raises(LLMError, match='HTTP 400.*messages'):
            client.chat('не список сообщений')
        assert client.json_mode                                               # дело было не в response_format
        with pytest.raises(LLMError, match='не превращается в JSON'):
            client.chat([{'role': 'user', 'content': object()}])


def test_http_auth_error_hides_key(mock, caplog):
    caplog.set_level(logging.DEBUG)
    server, url = mock(api_key='верный-ключ')
    with connect(url) as client:
        with pytest.raises(LLMError) as err:                                  # имитатор возвращает ключ в теле 401
            client.chat(build_messages(EXAMPLE_STATE))
        res = request_plan(client, EXAMPLE_STATE)
        with pytest.raises(LLMError) as err_models:
            client.models()
    assert 'HTTP 401' in str(err.value) and '***' in str(err.value) and '***' in str(err_models.value)
    assert res.plan is None and 'HTTP 401' in res.error and len(res.exchanges) == 1
    assert len(server.requests) == 0 and server.responder.stats == {}        # 401 не повторяется и до модели не доходит
    for text in (str(err.value), repr(err.value), res.error, json.dumps(res.to_dict()), str(err_models.value),
                 caplog.text):
        assert KEY not in text


def test_http_retry_log_hides_key(caplog):
    caplog.set_level(logging.INFO, logger='did.llm')

    def handler(request):
        return httpx.Response(503, text=f'перегрузка, ключ {request.headers["authorization"]}')

    with connect('http://llm.test/v1', max_retries=1) as client:
        fake_network(client, handler)
        with pytest.raises(LLMError, match='HTTP 503') as err:
            client.chat([{'role': 'user', 'content': 'привет'}])
    assert 'повтор' in caplog.text and KEY not in caplog.text and KEY not in str(err.value)


def test_http_no_server(mock):
    server, url = mock()
    server.stop()
    with connect(url, max_retries=1) as client:
        with pytest.raises(LLMError, match='нет связи'):
            client.chat(build_messages(EXAMPLE_STATE))
        with pytest.raises(LLMError, match='нет связи'):
            client.models()
        res = request_plan(client, EXAMPLE_STATE)
    assert res.plan is None and 'модель недоступна' in res.error
    with pytest.raises(LLMError, match='http://'):
        ChatClient('ai.mai.ru/v1', KEY, 'qwen')


def answer(content, finish='stop', **extra):
    return {'choices': [{'message': {'role': 'assistant', 'content': content, **extra}, 'finish_reason': finish}]}


@pytest.mark.parametrize('body, expected', [
    ('<html>шлюз</html>', 'не JSON'),
    ('{"choices": []}', 'нет choices'),
    ('{"error": {"message": "перегрузка"}}', 'нет choices'),
    ('[]', 'нет choices'),
], ids=['html', 'нет ответа', 'ошибка с кодом 200', 'список'])
def test_http_200_with_foreign_body(body, expected):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=body)

    with connect('http://llm.test/v1', max_retries=2) as client:
        fake_network(client, handler)
        with pytest.raises(LLMError, match=expected):
            client.chat([{'role': 'user', 'content': 'привет'}])
    assert len(calls) == 3


def test_http_redirect_is_reported_not_followed():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(308, headers={'Location': 'https://llm.test/api/chat/completions'})

    with connect('http://llm.test/v1') as client:
        fake_network(client, handler)
        with pytest.raises(LLMError, match='HTTP 308.*https://llm.test/api'):
            client.chat([{'role': 'user', 'content': 'привет'}], max_tokens=None)
    assert len(calls) == 1 and b'max_tokens' not in calls[0].content


def test_http_timeout_covers_trickling_answer():
    def trickle(empty_lines):                # сервер держит соединение пустыми строками, как DeepSeek в очереди
        for _ in range(empty_lines):
            time.sleep(0.02)
            yield b'\n'
        yield json.dumps(answer(PLAN_JSON)).encode()

    queue = [150, 150, 3]                    # два ответа дольше таймаута, третий укладывается
    with connect('http://llm.test/v1', timeout_s=0.15, max_retries=1) as client:
        fake_network(client, lambda request: httpx.Response(200, content=trickle(queue.pop(0))))
        t0 = time.monotonic()
        with pytest.raises(LLMError, match='за 2 попыток: таймаут 0.15 с'):
            client.chat([{'role': 'user', 'content': 'привет'}])
        assert 0.28 <= time.monotonic() - t0 < 1.0
        assert json.loads(client.chat([{'role': 'user', 'content': 'привет'}]).text) == PLAN


def test_http_reply_variants():
    replies = [answer([{'type': 'text', 'text': PLAN_JSON[:50]}, {'type': 'text', 'text': PLAN_JSON[50:]}]),
               answer(None, reasoning_content='долго думал'),
               answer(PLAN_JSON[:60], finish='length'), answer(PLAN_JSON[:60], finish='length')]

    with connect('http://llm.test/v1') as client:
        fake_network(client, lambda request: httpx.Response(200, json=replies.pop(0)))
        assert request_plan(client, EXAMPLE_STATE).ok                         # содержимое частями
        reply = client.chat([{'role': 'user', 'content': 'привет'}])
        assert (reply.text, reply.usage, reply.finish_reason) == ('', {}, 'stop')
        res = request_plan(client, EXAMPLE_STATE)
    assert res.plan is None and 'обрезан по лимиту токенов' in res.error


# --- устойчивость под сбоями -------------------------------------------------------------------

def random_state(rng):
    def items(prefix, extra):
        return [{'id': f'{prefix}{i + 1}', 'x': round(rng.uniform(-2.5, 2.5), 2), 'y': round(rng.uniform(-2.5, 2.5), 2),
                 extra: round(rng.random(), 2), 'cost_to': round(rng.uniform(0.5, 9), 1),
                 'cost_back': round(rng.uniform(0.5, 9), 1), 'feasible': rng.random() < 0.8}
                for i in range(rng.randrange(4))]

    return make_state(battery=round(rng.uniform(2, 60), 1), candidates=items('C', 'confidence'),
                      explore_points=items('E', 'unseen_share'),
                      samples={'collected': rng.randrange(6), 'total': 5})


def run_under_faults(client, n=200):
    rng = random.Random(11)
    good = 0
    for _ in range(n):
        state = random_state(rng)
        res = request_plan(client, state)                                     # исключение уронит тест
        assert isinstance(res, PlanResult) and (res.plan is None) == bool(res.error)
        assert 1 <= len(res.exchanges) <= 2
        if res.plan is not None:
            assert validate_plan(res.plan, state) == []
            good += 1
    return good / n


def test_200_requests_under_faults_local(capsys):
    responder = MockResponder(seed=1, faults=dict.fromkeys(FAULTS, 0.3))
    share = run_under_faults(LocalClient(responder))
    with capsys.disabled():
        print(f'\n  в процессе, 30% каждого сбоя: успешных планов {share:.0%}, ответы {dict(responder.stats)}')
    assert share > 0.15 and responder.stats['timeout'] == 0


def test_200_requests_under_faults_http(mock, capsys):
    server, url = mock(seed=1, faults=dict.fromkeys(FAULTS, 0.3), timeout_sleep_s=0.2)
    t0 = time.monotonic()
    with connect(url, timeout_s=0.02, max_retries=2) as client:
        share = run_under_faults(client)
    with capsys.disabled():
        print(f'\n  по HTTP, 30% каждого сбоя: успешных планов {share:.0%} за {time.monotonic() - t0:.1f} с, '
              f'ответы {dict(server.responder.stats)}')
    assert share > 0.35 and all(server.responder.stats[kind] > 0 for kind in FAULTS)
