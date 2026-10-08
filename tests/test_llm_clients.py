"""Клиенты настоящих моделей: Codex CLI (на подставной программе), кэш ответов, фабрика, схема плана."""
import copy
import json
import os
import stat
import sys
import time

import pytest

from did.llm import (EXAMPLE_STATE, PLAN_SCHEMA, ChatClient, ChatReply, LLMError, LocalClient, error_kind, llm_stats,
                     make_client, parse_plan, plan_schema, request_plan)
from did.llm_cache import CachedClient, ReplyCache
from did.llm_codex import CodexCliClient, render_prompt
from did.llm_mock import MockResponder, mock_plan, start_mock_server

PLAN_JSON = json.dumps(mock_plan(EXAMPLE_STATE), ensure_ascii=False)
MESSAGES = [{'role': 'system', 'content': 'Ты планировщик. Ответ в JSON.'}, {'role': 'user', 'content': 'Состояние: {}'}]

# Подставной codex: пишет ответ в файл из -o и события в stdout, как настоящий; поведение — FAKE_CODEX.
FAKE = r'''
import json, os, sys, time
args = sys.argv[1:]
log = os.environ['FAKE_CODEX_LOG']
out = args[args.index('-o') + 1]
schema = json.load(open(args[args.index('--output-schema') + 1])) if '--output-schema' in args else None
cwd = args[args.index('-C') + 1]
with open(log, 'a') as f:
    f.write(json.dumps({'args': args, 'schema': schema, 'cwd_files': os.listdir(cwd), 'stdin': sys.stdin.read()}) + '\n')
mode = os.environ.get('FAKE_CODEX', 'ok')
calls = sum(1 for _ in open(log))
if mode == 'hang':
    time.sleep(30)
if mode == 'fail' or (mode == 'flaky' and calls == 1):
    print(json.dumps({'type': 'error', 'message': 'stream disconnected, Bearer sk-secret-0123456789abcdef'}))
    sys.exit(1)
if mode == 'fatal':
    print(json.dumps({'type': 'turn.failed', 'error': {'message': 'You have hit your usage limit'}}))
    sys.exit(1)
if mode == 'silent':
    sys.stderr.write('Reading additional input from stdin...\nчто-то пошло не так\n')
    sys.exit(0)
print('не JSON: строка, которую надо пропустить')
print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'черновик'}}))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 15000, 'output_tokens': 120, 'note': 'x'}}))
open(out, 'w').write(os.environ.get('FAKE_CODEX_REPLY', '{"answer": "готов"}') + '\n')
'''


@pytest.fixture
def codex(tmp_path, monkeypatch):
    """Фабрика клиентов с подставной программой; у фабрики .log() — что программа получила."""
    script = tmp_path / 'codex'
    script.write_text(f'#!{sys.executable}\n{FAKE}', encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / 'calls.log'
    monkeypatch.setenv('FAKE_CODEX_LOG', str(log))
    for name in [n for n in os.environ if n.startswith('DID_CODEX_')]:
        monkeypatch.delenv(name)

    def make(mode='ok', reply=None, **options):
        monkeypatch.setenv('FAKE_CODEX', mode)
        if reply is not None:
            monkeypatch.setenv('FAKE_CODEX_REPLY', reply)
        options.setdefault('retry_pause_s', 0.0)
        return CodexCliClient(binary=str(script), cache_dir=tmp_path / 'cache', **options)

    make.log = lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return make


# --- Codex CLI ---------------------------------------------------------------------------------

def test_codex_reply_and_command(codex):
    client = codex()
    schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
    reply = client.chat(MESSAGES, schema=schema)
    assert isinstance(reply, ChatReply) and json.loads(reply.text) == {'answer': 'готов'}
    assert (reply.attempts, reply.cached) == (1, False) and reply.latency_ms > 0
    assert reply.usage['prompt_tokens'] == 15000 and reply.usage['total_tokens'] == 15120 and 'note' not in reply.usage
    call, = codex.log()
    args = call['args']
    assert args[0] == 'exec' and args[args.index('-m') + 1] == 'gpt-6-luna'
    assert 'model_reasoning_effort=low' in args and args[args.index('-s') + 1] == 'read-only'
    assert {'--ephemeral', '--skip-git-repo-check', '--ignore-user-config', '--ignore-rules', '--json'} <= set(args)
    assert call['schema'] == schema and call['cwd_files'] == [] and call['stdin'] == ''   # пустая папка, stdin закрыт
    prompt = args[-1]
    assert 'Не запускай команды' in prompt and '<инструкция>\nТы планировщик' in prompt and '<запрос>\nСостояние' in prompt


def test_codex_cache_saves_quota(codex):
    client = codex()
    first = client.chat(MESSAGES)
    again = client.chat(MESSAGES)
    assert again.cached and again.attempts == 0 and again.text == first.text
    assert again.latency_ms == first.latency_ms                              # время настоящего вызова
    assert len(codex.log()) == 1 and (client.calls, client.hits) == (1, 1)
    assert client.cache.calls() == {'gpt-6-luna': 1} and client.cache.saved() == {'gpt-6-luna': 1}
    assert codex().chat(MESSAGES).cached                                     # другой процесс найдёт тот же файл
    assert not codex(effort='none').chat(MESSAGES).cached                    # другой запрос — другой ключ
    assert not codex().chat(MESSAGES, schema={'type': 'object'}).cached
    assert not codex(use_cache=False).chat(MESSAGES).cached
    assert len(codex.log()) == 4


def test_codex_timeout_kills_process(codex):
    client = codex('hang', timeout_s=0.5, max_retries=1)
    t0 = time.monotonic()
    with pytest.raises(LLMError, match='таймаут 0.5 с') as err:
        client.chat(MESSAGES)
    assert time.monotonic() - t0 < 5.0 and err.value.attempts == 2
    assert client.cache.calls() == {'gpt-6-luna': 2} and client.cache.saved() == {}


def test_codex_retries_transient_failure(codex):
    client = codex('flaky')
    reply = client.chat(MESSAGES)
    assert reply.attempts == 2 and json.loads(reply.text) == {'answer': 'готов'}
    with pytest.raises(LLMError, match='codex: код 1: stream disconnected') as err:
        codex('fail', max_retries=1).chat([{'role': 'user', 'content': 'другой запрос'}])
    assert err.value.attempts == 2 and 'sk-secret' not in str(err.value) and '***' in str(err.value)


def test_codex_fatal_errors_are_not_retried(codex, tmp_path):
    with pytest.raises(LLMError, match='usage limit') as err:
        codex('fatal', max_retries=3).chat(MESSAGES)
    assert err.value.attempts == 1 and len(codex.log()) == 1
    with pytest.raises(LLMError, match='пустой ответ|что-то пошло не так'):
        codex('silent', max_retries=0).chat(MESSAGES)
    missing = CodexCliClient(binary=str(tmp_path / 'нет-такой-программы'), cache_dir=tmp_path / 'cache', max_retries=3)
    with pytest.raises(LLMError, match='не запускается') as err:
        missing.chat(MESSAGES)
    assert err.value.attempts == 1
    with pytest.raises(LLMError, match='список сообщений'):
        codex().chat('не список')


def test_codex_call_limit(codex):
    client = codex(max_calls=2)
    client.chat([{'role': 'user', 'content': 'первый'}])
    client.chat([{'role': 'user', 'content': 'второй'}])
    with pytest.raises(LLMError, match='квота вызовов gpt-6-luna исчерпана: сделано 2 из 2'):
        client.chat([{'role': 'user', 'content': 'третий'}])
    assert len(codex.log()) == 2
    assert client.chat([{'role': 'user', 'content': 'первый'}]).cached     # кэш работает и при исчерпанной квоте
    res = request_plan(client, EXAMPLE_STATE)
    assert res.plan is None and 'квота' in res.error
    assert error_kind(res.exchanges[0]['errors'][0]) == 'квота вызовов исчерпана'
    assert llm_stats(res.exchanges) == {**llm_stats(res.exchanges), 'requests': 1, 'failed': 1, 'quota': 1}


def test_codex_settings_from_env(codex, monkeypatch):
    monkeypatch.setenv('DID_CODEX_MODEL', 'gpt-test')
    monkeypatch.setenv('DID_CODEX_EFFORT', 'none')
    monkeypatch.setenv('DID_CODEX_TIMEOUT_S', '7')
    monkeypatch.setenv('DID_CODEX_MAX_CALLS', '9')
    client = codex(isolated=False)
    assert (client.model, client.effort, client.timeout_s, client.max_calls) == ('gpt-test', 'none', 7.0, 9)
    assert 'gpt-test' in repr(client) and 'потолок вызовов 9' in repr(client)
    client.chat(MESSAGES)
    assert '--ignore-user-config' not in codex.log()[0]['args']
    assert codex(model='gpt-other', max_calls=1).model == 'gpt-other'


def test_codex_plan_request_uses_state_schema(codex):
    client = codex(reply=PLAN_JSON)
    res = request_plan(client, EXAMPLE_STATE)
    assert res.ok and [sg.type for sg in res.plan.subgoals] == ['investigate', 'explore']
    assert codex.log()[0]['schema'] == plan_schema(EXAMPLE_STATE)
    first = res.exchanges[0]
    assert not first['cached'] and first['usage']['completion_tokens'] == 120
    again = request_plan(client, EXAMPLE_STATE)                              # из кэша: время — исходное
    ex = again.exchanges[0]
    assert again.ok and ex['cached'] and ex['http_attempts'] == 0 and ex['latency_ms'] == first['latency_ms'] > 0
    assert again.latency_ms >= ex['latency_ms'] and len(codex.log()) == 1


def test_render_prompt_keeps_the_whole_dialogue():
    text = render_prompt(MESSAGES + [{'role': 'assistant', 'content': '{"плохо": 1}'},
                                     {'role': 'user', 'content': 'Ответ не принят.'}])
    assert text.index('<инструкция>') < text.index('<запрос>') < text.index('<твой_прежний_ответ>')
    assert text.count('<запрос>') == 2 and 'один JSON-объект по заданной схеме' in text
    assert 'только текст ответа' in render_prompt(MESSAGES, structured=False)


# --- кэш ---------------------------------------------------------------------------------------

def test_reply_cache_counts_and_limits(tmp_path):
    cache = ReplyCache(tmp_path / 'cache')
    assert cache.calls() == {} and cache.saved() == {} and cache.get('m', 'нет') is None
    assert [cache.charge('m'), cache.charge('m'), cache.charge('qwen2.5:3b')] == [1, 2, 1]
    with pytest.raises(LLMError, match='сделано 2 из 2'):
        cache.charge('m', limit=2)
    assert cache.calls() == {'m': 2, 'qwen2.5:3b': 1}
    key = cache.key('m', MESSAGES, None)
    assert key == cache.key('m', copy.deepcopy(MESSAGES), None) != cache.key('m', MESSAGES, {'type': 'object'})
    cache.put('qwen2.5:3b', key, {'text': 'ответ', 'latency_ms': 5})
    assert cache.get('qwen2.5:3b', key)['text'] == 'ответ' and cache.saved() == {'qwen2.5-3b': 1}
    (tmp_path / 'cache' / 'qwen2.5-3b' / f'{key}.json').write_text('{испорчен')
    assert cache.get('qwen2.5:3b', key) is None


def test_cached_client_wraps_any_client(tmp_path):
    seen = []

    def responder(messages):
        seen.append(messages)
        time.sleep(0.02)
        return PLAN_JSON

    client = CachedClient(LocalClient(responder), cache_dir=tmp_path / 'cache')
    first = client.chat(MESSAGES)
    again = client.chat(MESSAGES)
    assert not first.cached and again.cached and again.text == PLAN_JSON and len(seen) == 1
    assert again.latency_ms == first.latency_ms >= 20
    assert not client.chat(MESSAGES, temperature=0.9).cached and len(seen) == 2
    res = request_plan(client, EXAMPLE_STATE)
    assert res.ok and request_plan(client, EXAMPLE_STATE).exchanges[0]['cached']
    client.close()


# --- фабрика клиентов --------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch):
    for name in [n for n in os.environ if n.startswith(('DID_LLM_', 'DID_OLLAMA_', 'DID_CODEX_'))]:
        monkeypatch.delenv(name)
    return monkeypatch


def test_make_client_kinds(env, tmp_path):
    mock = make_client('mock', seed=3, faults={'empty': 1.0})
    assert isinstance(mock, LocalClient) and mock.responder.faults == {'empty': 1.0}
    assert request_plan(make_client(), EXAMPLE_STATE).ok
    ollama = make_client('ollama')
    assert (ollama.base_url, ollama.model, ollama.timeout_s, ollama.use_schema) == (
        'http://127.0.0.1:11434/v1', 'qwen2.5:3b', 120.0, True)
    ollama.close()
    env.setenv('DID_OLLAMA_MODEL', 'qwen3:4b')
    assert make_client('ollama').model == 'qwen3:4b' and make_client('ollama', model='qwen2.5:7b').model == 'qwen2.5:7b'
    codex = make_client('codex', cache_dir=tmp_path, max_calls=5)
    assert isinstance(codex, CodexCliClient) and codex.max_calls == 5 and codex.use_schema
    with pytest.raises(LLMError, match='DID_LLM_BASE_URL'):
        make_client('http')
    with pytest.raises(LLMError, match='неизвестный вид клиента «gpt»'):
        make_client('gpt')
    env.setenv('DID_LLM_BASE_URL', 'http://127.0.0.1:1/v1')
    env.setenv('DID_LLM_MODEL', 'qwen-plus')
    http = make_client('http', timeout_s=5)
    assert (http.model, http.timeout_s, http.use_schema) == ('qwen-plus', 5.0, False)
    env.setenv('DID_LLM_JSON_SCHEMA', '1')
    assert make_client('http').use_schema
    cached = make_client('ollama', cache=True)
    assert isinstance(cached, CachedClient) and cached.model == 'qwen3:4b' and cached.use_schema


def test_runner_makes_planner_for_every_kind(env, tmp_path):
    from did.agent import make_config
    from did.planner import HeuristicPlanner, LLMPlanner
    from did.runner import make_planner
    cfg = make_config('adaptive_llm')
    assert isinstance(make_planner(make_config('adaptive'), {'kind': 'codex'}), HeuristicPlanner)
    assert isinstance(make_planner(cfg).client.responder, MockResponder)
    planner = make_planner(cfg, {'kind': 'mock', 'faults': {'malformed': 0.5}}, seed=7)
    assert isinstance(planner, LLMPlanner) and planner.client.responder.faults == {'malformed': 0.5}
    planner = make_planner(cfg, {'kind': 'codex', 'model': 'gpt-x', 'cache_dir': tmp_path, 'prompt': 'planner_system_v1'})
    assert planner.client.model == 'gpt-x' and planner.system_prompt.startswith('# Роль')
    assert make_planner(cfg, {'kind': 'ollama'}).client.model == 'qwen2.5:3b'
    with pytest.raises(LLMError):
        make_planner(cfg, {'kind': 'http'})


# --- JSON-схема ответа по HTTP -----------------------------------------------------------------

def test_http_sends_schema_when_asked():
    server, url = start_mock_server()
    try:
        with ChatClient(url, 'key', 'mock-planner', use_schema=True) as client:
            assert request_plan(client, EXAMPLE_STATE).ok
            assert client.chat(MESSAGES).attempts == 1                       # без схемы — обычный json_object
        with ChatClient(url, 'key', 'mock-planner') as client:               # use_schema выключен
            assert request_plan(client, EXAMPLE_STATE).ok
    finally:
        server.stop()
    formats = [r['response_format'] for r in server.requests]
    assert formats[0] == {'type': 'json_schema', 'json_schema': {'name': 'reply', 'strict': True,
                                                                'schema': plan_schema(EXAMPLE_STATE)}}
    assert formats[1] == formats[2] == {'type': 'json_object'}


def test_http_schema_falls_back_step_by_step():
    server, url = start_mock_server(reject_response_format=True)
    try:
        with ChatClient(url, 'key', 'mock-planner', use_schema=True, retry_pause_s=0.0) as client:
            res = request_plan(client, EXAMPLE_STATE)
            assert res.ok and res.exchanges[0]['http_attempts'] == 3         # схема → json_object → без формата
            assert not client.use_schema and not client.json_mode
            assert request_plan(client, EXAMPLE_STATE).exchanges[0]['http_attempts'] == 1
    finally:
        server.stop()
    with pytest.raises(LLMError, match='ключ доступа'):
        ChatClient(url, 'ключ-кириллицей', 'mock-planner')


# --- схема плана под состояние -----------------------------------------------------------------

def strict(schema):
    """Все объекты схемы строгие: лишних полей нет, все поля обязательны."""
    if isinstance(schema, dict):
        if schema.get('type') == 'object':
            assert schema['additionalProperties'] is False and schema['required'] == list(schema['properties'])
        return all(strict(v) for v in schema.values())
    return all(strict(v) for v in schema) if isinstance(schema, list) else True


def variants(schema):
    return {v['properties']['type']['enum'][0]: v for v in schema['properties']['subgoals']['items']['anyOf']}


def test_plan_schema_lists_only_feasible_targets():
    state = copy.deepcopy(EXAMPLE_STATE)
    state['candidates'] += [{'id': 'C2', 'feasible': False}, {'id': 'C3', 'feasible': True}, {'id': 'C4'}, 'мусор']
    by_type = variants(plan_schema(state))
    assert list(by_type) == ['investigate', 'explore', 'goto', 'return_base']
    assert by_type['investigate']['properties']['target']['enum'] == ['C1', 'C3', 'C4']
    assert by_type['explore']['properties']['target']['enum'] == ['E1']
    assert strict(plan_schema(state)) and strict(PLAN_SCHEMA) and PLAN_SCHEMA == plan_schema(EXAMPLE_STATE)
    for empty in ({}, None, 'строка', {'candidates': None, 'explore_points': [{'id': 'E1', 'feasible': False}]}):
        assert list(variants(plan_schema(empty))) == ['goto', 'return_base']


@pytest.mark.parametrize('target', ['E1ъ', 'E1}]} <>', 'E1) ... nope? Wait target only id. use E1.', ' E1 ',
                                    'E1った'])
def test_garbage_after_target_id_is_trimmed(target):
    """Так ошибался GPT на первой проверке: идентификатор верный, после него мусор."""
    text = json.dumps({'reasoning': 'ок', 'subgoals': [{'type': 'explore', 'target': target, 'x': None, 'y': None}]})
    plan, errors = parse_plan(text, EXAMPLE_STATE)
    assert errors == [] and plan.subgoals[0].target == 'E1'


@pytest.mark.parametrize('target', ['E12', 'E9ъ', 'точка E1', 'C1'])
def test_unknown_target_is_not_guessed(target):
    text = json.dumps({'reasoning': 'ок', 'subgoals': [{'type': 'explore', 'target': target}]})
    plan, errors = parse_plan(text, EXAMPLE_STATE)
    assert plan is None and len(errors) == 1


# --- сводка по обменам -------------------------------------------------------------------------

def test_llm_stats_groups_requests():
    ok = {'attempt': 1, 'ok': True, 'errors': [], 'latency_ms': 100}
    bad = {'attempt': 1, 'ok': False, 'errors': ['subgoals[0]: цели "C9" нет в candidates (C1)',
                                                 'subgoals[1]: точка (4; 4) вне арены, нужно |x| ≤ 2.6'], 'latency_ms': 300}
    fixed = {'attempt': 2, 'ok': True, 'errors': [], 'latency_ms': 200, 'cached': True}
    worse = {'attempt': 2, 'ok': False, 'errors': ['пустой ответ'], 'latency_ms': 50}
    down = {'attempt': 1, 'ok': False, 'errors': ['модель не ответила за 3 попыток: HTTP 500'], 'latency_ms': 900}
    stats = llm_stats([ok, bad, fixed, bad, worse, down, ok])
    assert (stats['requests'], stats['exchanges']) == (5, 7)
    assert (stats['first_ok'], stats['repaired'], stats['failed'], stats['cached'], stats['quota']) == (2, 1, 2, 1, 0)
    assert stats['errors'] == {'цель не из списка': 2, 'точка вне арены': 2, 'ответ не JSON': 1, 'нет ответа модели': 1}
    assert stats['latency_ms'] == {'mean': 279, 'median': 200, 'max': 900, 'total': 1950}
    assert stats['transport_retries'] == 0 and stats['tokens'] == {'prompt': 0, 'completion': 0}
    usage = {'prompt_tokens': 17000, 'completion_tokens': 130}
    stats = llm_stats([{**ok, 'usage': usage, 'http_attempts': 2}, {**ok, 'usage': {**usage, 'prompt_tokens': 16000}}, ok])
    assert stats['transport_retries'] == 1 and stats['tokens'] == {'prompt': 16500, 'completion': 130}
    assert llm_stats([])['requests'] == 0 and llm_stats(None)['latency_ms']['mean'] == 0
    kinds = {'reasoning: нет обязательного поля': 'нарушена схема',
             'subgoals[0]: цель "C1" недостижима по заряду (feasible=false)': 'цель недостижима по заряду',
             'subgoals[0]: "E1" есть в explore_points, а не в candidates': 'цель не из списка',
             'subgoals[0]: return_base может быть только последней подцелью': 'return_base не последняя',
             'subgoals[1]: повтор предыдущей подцели': 'повтор цели подряд',
             'ответ обрезан по лимиту токенов, отвечай короче': 'ответ обрезан',
             'codex: таймаут 120 с, процесс остановлен': 'нет ответа модели'}
    assert {text: error_kind(text) for text in kinds} == kinds


def test_run_episode_records_llm_stats():
    from did.runner import run_episode
    summary = run_episode('medium', 1003, 'adaptive_llm', llm={'kind': 'mock', 'faults': {'malformed': 0.6}}, save=False)
    stats = summary['metrics']['llm_stats']
    assert stats['requests'] == stats['first_ok'] + stats['repaired'] + stats['failed'] > 0
    assert stats['repaired'] > 0 and stats['errors'].get('ответ не JSON') == stats['repaired'] + stats['failed']
    assert 'llm_stats' not in run_episode('medium', 1003, 'adaptive', save=False)['metrics']
