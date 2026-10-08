"""J1: клиент платной модели решений Jev и сторож миссии. Сеть не используется: ответы сервера подменены.

Журнал расходов, ключ и кэш — только во временном каталоге проверки; домашний каталог подменён.
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import httpx
import pytest

import did.llm_jev as llm_jev
import did.runner as runner
from did.llm import CacheMiss
from did.llm_jev import (BUDGET_CAP_USD, BUDGET_STOP_USD, CACHE_ONLY_ENV, ENDPOINT, MODEL, REASON_CODES, Budget,
                         JevBudgetExceeded, JevClient, JevCostOverrun, JevError, JevJournalError, JevModelRefused,
                         JevReply, JevWrongModel, max_cost_usd)
from did.mission_guard import DISABLE_REASONS, GuardedPlanner, build_request, make_guarded
from did.runner import run_episode

ROOT = Path(__file__).resolve().parent.parent
SECRET = 'sk-or-test-0123456789abcdef'
Q = {'q': {'type': 'noul', 'instructions': 'Is it so?'}}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Ни одна проверка не видит настоящих ключа и журнала и не пишет в домашний каталог."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv(llm_jev.KEY_FILE_ENV, raising=False)
    monkeypatch.delenv(llm_jev.BUDGET_FILE_ENV, raising=False)
    monkeypatch.delenv(CACHE_ONLY_ENV, raising=False)
    monkeypatch.setattr(llm_jev.time, 'sleep', lambda s: None)        # паузы между повторами не нужны
    yield
    assert not any(home.iterdir())                                    # домашний каталог не тронут


def _budget(tmp_path, spent=0.0, **opts):
    budget = Budget(tmp_path / 'budget' / 'jev_budget.json', **opts)
    if not budget.path.exists():
        budget.init(spent)
    return budget


def _client(tmp_path, handler, budget=None, **opts):
    tmp_path.mkdir(parents=True, exist_ok=True)
    key_file = tmp_path / 'key.env'
    key_file.write_text(f'OTHER=1\nOPENROUTER_API_KEY={SECRET}\n', encoding='utf-8')
    seen = []

    def transport(request):
        seen.append(request)
        return handler(request)
    opts.setdefault('max_retries', 0)
    client = JevClient(cache_dir=tmp_path / 'cache', budget=budget or _budget(tmp_path),
                       key_file=key_file, transport=httpx.MockTransport(transport), **opts)
    return client, seen


def _reply(request, model='typesafe/jev-1.13-20260917', cost=0.00002, tokens=300, p=0.9, **extra):
    names = json.loads(request.content)['questions']
    usage = {'input_tokens': tokens, 'output_tokens': 20, **({'cost': cost} if cost is not None else {})}
    return httpx.Response(200, json={'id': 'gen-1', 'model': model, 'provider': 'TypeSafe', 'usage': usage,
                                     'answers': {k: {'type': 'noul', 'noul': p} for k in names}, **extra})


def _ok(**kw):
    return lambda request: _reply(request, **kw)


def _script(*steps):
    """Сервер, который на каждую следующую попытку отвечает следующим шагом: исключение, код ответа или 'ok'."""
    left = list(steps)

    def handler(request):
        step = left.pop(0)
        if isinstance(step, Exception):
            raise step
        return _reply(request) if step == 'ok' else httpx.Response(step, text='busy')
    return handler


def _all_files(root):
    return '\n'.join(p.read_text(encoding='utf-8') for p in root.rglob('*') if p.is_file() and p.name != 'key.env')


def _body(state):
    return json.dumps({'model': MODEL, 'state': state, 'questions': Q}, ensure_ascii=False).encode('utf-8')


def test_other_model_is_refused_before_anything_is_sent(tmp_path):
    client, seen = _client(tmp_path, _ok())
    before = client.budget.path.read_bytes()
    for model in ('typesafe/jev-router', 'typesafe/jev-latest', '~typesafe/jev-latest', 'jev-1.13', 'openrouter/auto',
                  'typesafe/jev-1.13:free', 'TYPESAFE/JEV-1.13', '', None):
        with pytest.raises(JevModelRefused):
            client.ask('state', Q, model=model)
    assert seen == []
    assert client.budget.path.read_bytes() == before              # даже журнал не тронут
    client.ask('state', Q)                                        # разрешённая модель уходит по адресу Jev
    assert len(seen) == 1 and str(seen[0].url) == ENDPOINT
    assert json.loads(seen[0].content)['model'] == MODEL


# --- журнал расходов ---------------------------------------------------------------------------

def test_budget_journal_lives_outside_the_repository_and_can_be_moved_by_env(tmp_path, monkeypatch):
    default = Budget().path
    assert default == tmp_path / 'home' / '.did' / 'jev_budget.json'          # ~/.did, один на все деревья
    assert ROOT not in default.parents and not hasattr(llm_jev, 'BUDGET_PATH')
    monkeypatch.setenv(llm_jev.BUDGET_FILE_ENV, str(tmp_path / 'other' / 'b.json'))
    assert Budget().path == tmp_path / 'other' / 'b.json'


def test_missing_empty_or_broken_journal_forbids_paid_calls(tmp_path):
    path = tmp_path / 'budget' / 'jev_budget.json'
    client, seen = _client(tmp_path, _ok(), budget=Budget(path))
    with pytest.raises(JevBudgetExceeded, match='--init-budget'):  # журнала нет — это не «нулевой расход»
        client.ask('state', Q)
    assert seen == [] and not path.parent.exists()                # и ничего не создано
    assert Budget(path).read() is None and not path.parent.exists()
    path.parent.mkdir()
    for text in ('', '{}', 'не json', '{"model": "typesafe/jev-1.13", "initialized": "x", "reserved": {}, '
                 '"cost_usd": "много", "unknown_cost_usd": 0}', '{"model": "openai/gpt-6", "initialized": "x", '
                 '"reserved": {}, "cost_usd": 0, "unknown_cost_usd": 0}'):
        path.write_text(text, encoding='utf-8')
        with pytest.raises(JevBudgetExceeded):
            client.ask('state', Q)
        assert path.read_text(encoding='utf-8') == text           # испорченный журнал клиент не «чинит»
    assert seen == []


def test_init_command_carries_over_spent_money_and_never_resets_the_journal(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'private' / 'jev_budget.json'
    monkeypatch.setenv(llm_jev.BUDGET_FILE_ENV, str(path))
    assert llm_jev.main([]) == 0 and 'журнала нет' in capsys.readouterr().out
    assert not path.parent.exists()                               # просмотр ничего не создаёт
    with pytest.raises(SystemExit):                               # сумму надо назвать явно
        llm_jev.main(['--init-budget'])
    assert llm_jev.main(['--init-budget', '--spent', '0.226192974', '--calls', '3964']) == 0
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700 and stat.S_IMODE(path.stat().st_mode) == 0o600
    b = Budget().read()
    assert b['cost_usd'] == 0.226192974 and b['calls'] == 3964 and b['carried'] == {'cost_usd': 0.226192974,
                                                                                   'calls': 3964}
    assert Budget.spent(b) == Budget.committed(b) == 0.226192974 and b['halted'] is None
    before = path.read_bytes()
    assert llm_jev.main(['--init-budget', '--spent', '0']) == 2   # второй раз — отказ: обнулить счёт нельзя
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        Budget().init(0.0)
    assert llm_jev.main(['--init-budget', '--spent', '-1']) == 2
    assert '0.226193' in capsys.readouterr().out


def test_reservation_keeps_spent_plus_reserved_under_the_cap(tmp_path, monkeypatch):
    assert BUDGET_STOP_USD == 0.60 and BUDGET_CAP_USD - BUDGET_STOP_USD == pytest.approx(0.40)   # запас до потолка
    monkeypatch.setattr(llm_jev, 'max_cost_usd', lambda body: 0.2)
    client, seen = _client(tmp_path, _ok(cost=0.15))
    for state in ('s1', 's2', 's3'):
        client.ask(state, Q)
    b = client.budget.read()
    assert Budget.spent(b) == pytest.approx(0.45) and b['reserved'] == {} and b['calls'] == 3
    with pytest.raises(JevBudgetExceeded):                        # 0,45 + 0,20 > 0,60: запрос не отправлен
        client.ask('s4', Q)
    assert len(seen) == 3 and Budget.committed(client.budget.read()) == pytest.approx(0.45)
    assert client.ask('s1', Q).cached and len(seen) == 3          # кэш отдаёт и после потолка, денег не тратит
    # Случай из ревью: до порога остаётся 0,001, запрос может стоить 0,04 — раньше проходил, теперь нет.
    near, seen2 = _client(tmp_path / 'near', _ok(cost=0.04), budget=_budget(tmp_path / 'near', spent=0.599))
    monkeypatch.setattr(llm_jev, 'max_cost_usd', lambda body: 0.04)
    with pytest.raises(JevBudgetExceeded):
        near.ask('s', Q)
    assert seen2 == [] and Budget.spent(near.budget.read()) == 0.599
    # Порог, записанный в журнале, обязателен и для клиента, которому в коде задали порог выше.
    loose, seen3 = _client(tmp_path / 'near', _ok(cost=0.04), budget=Budget(near.budget.path, stop_usd=1.0))
    with pytest.raises(JevBudgetExceeded):
        loose.ask('s', Q)
    assert seen3 == []


def test_concurrent_clients_cannot_both_take_the_last_money(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_jev, 'max_cost_usd', lambda body: 0.04)
    budget = _budget(tmp_path, spent=0.55)
    in_flight = []

    def slow(request):
        in_flight.append(Budget.committed(budget.read()))         # пока запрос в пути, резерв уже в журнале
        threading.Event().wait(0.3)
        return _reply(request, cost=0.04)
    clients = [_client(tmp_path, slow, budget=Budget(budget.path)) for _ in range(4)]
    outcomes = []

    def work(i):
        try:
            clients[i][0].ask(f'state {i}', Q)
            outcomes.append('ok')
        except JevBudgetExceeded:
            outcomes.append('refused')
    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ['ok', 'refused', 'refused', 'refused']
    assert sum(len(seen) for _, seen in clients) == 1 and in_flight == [pytest.approx(0.59)]
    b = budget.read()
    assert Budget.spent(b) == pytest.approx(0.59) and Budget.spent(b) <= BUDGET_STOP_USD and b['reserved'] == {}


def test_separate_processes_share_one_journal_and_stop_at_the_cap(tmp_path):
    budget = _budget(tmp_path, spent=0.15)
    code = ('import sys\nfrom did.llm_jev import Budget, JevBudgetExceeded\nb = Budget(sys.argv[1])\nn = 0\n'
            'for _ in range(30):\n    try:\n        b.settle(b.reserve(0.01), cost=0.01)\n        n += 1\n'
            '    except JevBudgetExceeded:\n        pass\nprint(n)')
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    procs = [subprocess.Popen([sys.executable, '-c', code, str(budget.path)], cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE) for _ in range(2)]
    done = [int(p.communicate(timeout=120)[0]) for p in procs]
    assert sum(done) == 45                                        # 0,15 + 45 × 0,01 = 0,60 и ни запросом больше
    b = budget.read()
    assert Budget.spent(b) == pytest.approx(0.60) and b['calls'] == 45 and b['reserved'] == {}


def test_every_possibly_delivered_attempt_is_counted_even_before_a_success(tmp_path):
    limit = max_cost_usd(_body('state'))
    timeout = httpx.ReadTimeout('поздно')
    # Случай из ревью: первая попытка — таймаут чтения, вторая — успех. Раньше в учёте была одна попытка.
    client, seen = _client(tmp_path / 'a', _script(timeout, 'ok'), budget=_budget(tmp_path / 'a'), max_retries=1)
    reply = client.ask('state', Q)
    b = client.budget.read()
    assert len(seen) == 2 and b['calls'] == 2 and b['failed_calls'] == 1 and b['unknown_attempts'] == 1
    assert b['unknown_cost_usd'] == pytest.approx(limit) and b['cost_usd'] == pytest.approx(0.00002)
    assert Budget.spent(b) == pytest.approx(limit + 0.00002) and b['reserved'] == {}
    assert reply.cost_usd == pytest.approx(0.00002) and reply.cost_known
    # Связь не установилась — запрос не дошёл и не считается; отказ сервера (429) считается, но стоит ноль.
    client, seen = _client(tmp_path / 'b', _script(httpx.ConnectError('нет связи'), 429, 'ok'),
                           budget=_budget(tmp_path / 'b'), max_retries=2)
    client.ask('state', Q)
    b = client.budget.read()
    assert len(seen) == 3 and b['calls'] == 2 and b['unknown_attempts'] == 0 and b['unknown_cost_usd'] == 0
    assert b['cost_usd'] == pytest.approx(0.00002) and b['reserved'] == {}
    # Ошибка сервера и ответ не в JSON могли быть оплачены: оценка сверху. Все попытки неудачны — учтены все.
    client, seen = _client(tmp_path / 'c', _script(500, timeout, 200), budget=_budget(tmp_path / 'c'), max_retries=2)
    with pytest.raises(JevError):
        client.ask('state', Q)
    b = client.budget.read()
    assert len(seen) == 3 and b['calls'] == 3 and b['failed_calls'] == 3 and b['unknown_attempts'] == 3
    assert Budget.spent(b) == pytest.approx(3 * limit) and b['cost_usd'] == 0 and b['reserved'] == {}
    # Повторы тоже резервируют деньги: когда на следующую попытку не хватает, она не отправляется.
    tight = _budget(tmp_path / 'd', stop_usd=1.5 * limit)
    client, seen = _client(tmp_path / 'd', _script(timeout, 'ok'), budget=tight, max_retries=3)
    with pytest.raises(JevBudgetExceeded):
        client.ask('state', Q)
    assert len(seen) == 1 and Budget.spent(tight.read()) == pytest.approx(limit)


def test_unknown_cost_is_kept_apart_and_counted_by_the_upper_bound(tmp_path):
    client, _ = _client(tmp_path, _ok(cost=None, tokens=1000))
    reply = client.ask('state', Q)
    limit = max_cost_usd(_body('state'))
    assert limit > 1000 * llm_jev.PRICE_PER_TOKEN                 # оценка сверху, а не цена по счётчику токенов
    assert reply.cost_usd == pytest.approx(limit) and not reply.cost_known
    b = client.budget.read()
    assert b['unknown_attempts'] == 1 and b['unknown_cost_usd'] == pytest.approx(limit) and b['cost_usd'] == 0
    assert b['cost_from_response'] == 0 and b['input_tokens'] == 1000
    for bad in (-1.0, True, 'дёшево', float('nan')):              # нелепая стоимость — тоже неизвестная
        handler = lambda request, bad=bad: httpx.Response(200, content=json.dumps(     # noqa: E731
            {'model': MODEL, 'usage': {'cost': bad}, 'answers': {'q': {'type': 'noul', 'noul': 0.5}}}))
        other, _ = _client(tmp_path, handler, budget=client.budget)
        assert not other.ask(f'state {bad}', Q).cost_known
    assert client.budget.read()['unknown_attempts'] == 5


def test_cost_above_the_reservation_is_recorded_stops_the_work_and_is_never_returned(tmp_path):
    client, seen = _client(tmp_path, _ok(cost=0.4))               # сервер насчитал больше оценки сверху
    with pytest.raises(JevCostOverrun) as err:                    # ответ оплачен, но как успех не возвращается
        client.ask('s1', Q)
    assert err.value.code == 'budget' and not isinstance(err.value, JevBudgetExceeded)
    b = client.budget.read()
    assert b['cost_usd'] == pytest.approx(0.4) and 'оценки сверху' in b['halted']
    assert b['calls'] == 1 and b['cost_from_response'] == 1 and b['reserved'] == {}      # деньги учтены
    assert not list((tmp_path / 'cache').glob('*.json'))          # и в кэш такой ответ не попал
    for state in ('s2', 's1'):                                    # дальше — отказ без сети, в том числе на тот же вопрос
        with pytest.raises(JevBudgetExceeded):
            client.ask(state, Q)
    assert len(seen) == 1
    # Случай из ревью: журнал у самого порога, обычный небольшой запрос, сервер насчитал 0,06.
    near, seen2 = _client(tmp_path / 'near', _ok(cost=0.06),
                          budget=_budget(tmp_path / 'near', spent=round(BUDGET_STOP_USD - 0.001, 9)))
    assert max_cost_usd(_body('s')) < 0.001
    with pytest.raises(JevCostOverrun):
        near.ask('s', Q)
    b = near.budget.read()
    assert len(seen2) == 1 and b['halted'] and not list((tmp_path / 'near' / 'cache').glob('*.json'))
    assert Budget.spent(b) == pytest.approx(BUDGET_STOP_USD + 0.059) and Budget.spent(b) < BUDGET_CAP_USD
    # На показе сторож от такого ответа отключается, а не пользуется им.
    shown, _ = _client(tmp_path / 'показ', _ok(cost=0.4), budget=_budget(tmp_path / 'показ'))
    guard = GuardedPlanner(shown, fallback=True)
    assert guard.plan(_state())['guard']['reason'] == 'budget' and guard.calls == 0


def test_journal_is_replaced_atomically_and_reading_never_writes(tmp_path, monkeypatch):
    budget = _budget(tmp_path, spent=0.3)
    budget.settle(budget.reserve(0.01), 100, 5, 0.004)
    before, mtime = budget.path.read_bytes(), budget.path.stat().st_mtime_ns
    assert Budget.spent(budget.read()) == pytest.approx(0.304)
    assert budget.path.read_bytes() == before and budget.path.stat().st_mtime_ns == mtime     # чтение не пишет

    def crash(src, dst):
        raise OSError('диск отвалился посреди записи')
    with monkeypatch.context() as m:
        m.setattr(llm_jev.os, 'replace', crash)
        with pytest.raises(JevJournalError):
            budget.reserve(0.01)
    assert budget.path.read_bytes() == before                     # прежний журнал цел, счёт не потерян
    assert sorted(p.name for p in budget.path.parent.iterdir()) == ['jev_budget.json', 'jev_budget.json.lock']
    assert Budget.spent(budget.read()) == pytest.approx(0.304)


def test_answer_from_another_model_stops_all_further_calls(tmp_path):
    client, seen = _client(tmp_path, _ok(model='openai/gpt-6'))
    with pytest.raises(JevWrongModel):
        client.ask('state', Q)
    assert client.budget.read()['halted'] and client.budget.read()['calls'] == 1     # деньги уже учтены
    assert not list((tmp_path / 'cache').glob('*.json'))          # чужой ответ в кэш не попал
    with pytest.raises(JevBudgetExceeded):                        # дальше — отказ без сети
        client.ask('other', Q)
    assert len(seen) == 1
    for model in ('typesafe/jev-router', 'typesafe/jev-1.14', 'typesafe/jev-1.13-preview', 'typesafe/jev-1.130'):
        assert not JevClient._answered_by_jev(model)
    assert JevClient._answered_by_jev('typesafe/jev-1.13') and JevClient._answered_by_jev('typesafe/jev-1.13-20260917')


# --- ключ --------------------------------------------------------------------------------------

def test_key_never_reaches_cache_budget_errors_or_client_object(tmp_path):
    def echo_key(request):                                        # сервер, который возвращает ключ в тексте ошибки
        return httpx.Response(403, text=f"bad key {request.headers['authorization']}")
    client, seen = _client(tmp_path, echo_key)
    with pytest.raises(JevError) as err:
        client.ask('state', Q)
    assert seen[0].headers['authorization'] == f'Bearer {SECRET}'
    assert SECRET not in str(err.value) and '***' in str(err.value)
    # Ключ на границе обрезки: сначала вымарывание, потом обрезка — куска ключа в сообщении не остаётся.
    edge, _ = _client(tmp_path, lambda request: httpx.Response(403, text='x' * 190 + SECRET + ' хвост'))
    with pytest.raises(JevError) as err:
        edge.ask('state', Q)
    assert SECRET[:8] not in str(err.value)
    good, _ = _client(tmp_path, _ok())
    reply = good.ask('state', Q)
    assert SECRET not in _all_files(tmp_path)
    assert SECRET not in repr(vars(good)) and SECRET not in repr(reply) and SECRET not in repr(good._http.headers)


def _answer_200(**fields):
    def handler(request):
        data = {'model': 'typesafe/jev-1.13-20260917', 'provider': 'TypeSafe',
                'usage': {'input_tokens': 300, 'output_tokens': 20, 'cost': 0.00002},
                'answers': {'q': {'type': 'noul', 'noul': 0.9}}, **fields}
        return httpx.Response(200, content=json.dumps(data, ensure_ascii=True))
    return handler


@pytest.mark.parametrize('fields, error', [
    ({'model': SECRET}, JevWrongModel),                                             # ключ в поле model
    ({'model': f'typesafe/jev-1.13 {SECRET}'}, JevWrongModel),
    ({'provider': f'Bearer {SECRET}'}, JevWrongModel),                              # ключ в поле provider
    ({'answers': f'no answers for {SECRET}'}, JevError),                            # ключ вместо ответов
    ({'answers': {SECRET: {'type': 'noul', 'noul': 0.9}}}, JevError),               # ключ в имени ответа
    ({'answers': {'q': {'type': 'noul', 'noul': SECRET}}}, JevError),               # ключ вместо вероятности
    ({'answers': {'q': {'type': SECRET, 'noul': 0.9}}}, JevError),
])
def test_key_echoed_in_a_successful_response_is_scrubbed_everywhere(tmp_path, fields, error):
    client, seen = _client(tmp_path, _answer_200(**fields))
    with pytest.raises(error) as err:
        client.ask('state', Q)
    assert len(seen) == 1 and SECRET in seen[0].headers['authorization']
    assert SECRET not in str(err.value) and SECRET not in repr(err.value.args)
    assert SECRET not in _all_files(tmp_path)                     # журнал (в том числе halted) и кэш
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_only_the_checked_schema_leaves_the_client(tmp_path):
    extra = {'debug': {'authorization': f'Bearer {SECRET}'}, 'id': SECRET,
             'answers': {'q': {'type': 'noul', 'noul': 1, 'note': f'key was {SECRET}', 'trace': [SECRET]}}}
    client, _ = _client(tmp_path, _answer_200(**extra))
    reply = client.ask('state', Q)
    assert reply.answers == {'q': {'type': 'noul', 'noul': 1.0}} and reply.model == 'typesafe/jev-1.13-20260917'
    assert SECRET not in repr(reply) and SECRET not in _all_files(tmp_path)
    record = json.loads(next((tmp_path / 'cache').glob('*.json')).read_text(encoding='utf-8'))
    assert record['answers'] == {'q': {'type': 'noul', 'noul': 1.0}} and 'debug' not in record
    # Не JSON при HTTP 200 и «почти схема» с вероятностью вне [0, 1] — отказ с постоянным текстом.
    raw, _ = _client(tmp_path, lambda request: httpx.Response(200, text=f'<html>{SECRET}</html>'))
    with pytest.raises(JevError) as err:
        raw.ask('другое', Q)
    assert SECRET not in str(err.value) and '<html>' not in str(err.value)
    for wrong in ({'type': 'noul', 'noul': 1.7}, {'type': 'noul', 'noul': True}, {'type': 'choice', 'choice': 'a'},
                  {'type': 'noul'}, [0.9]):
        odd, _ = _client(tmp_path, _answer_200(answers={'q': wrong}))
        with pytest.raises(JevError, match='не по схеме'):
            odd.ask(f'state {wrong}', Q)
    # Строки в ответах choice остаются, но без ключа; неизвестные поля отбрасываются.
    choice_q = {'c': {'type': 'choice', 'instructions': 'Which?', 'options': ['a', 'b']}}
    ans = {'c': {'type': 'choice', 'choice': f'a {SECRET}', 'probabilities': {'a': 0.7, 'b': 0.3, SECRET: 0.0},
                 'confidence': 0.8, 'why': SECRET}}
    chooser, _ = _client(tmp_path, _answer_200(answers=ans))
    got = chooser.ask('выбор', choice_q).answers['c']
    assert got == {'type': 'choice', 'choice': 'a ***', 'probabilities': {'a': 0.7, 'b': 0.3, '***': 0.0},
                   'confidence': 0.8}
    assert SECRET not in _all_files(tmp_path)


# --- третья правка по ревью: схема журнала, порядок чтения ключа, исключения транспорта ---------

JOURNAL_FIELDS = ('model', 'cap_usd', 'stop_usd', 'initialized', 'calls', 'failed_calls', 'input_tokens',
                  'output_tokens', 'cost_usd', 'cost_from_response', 'unknown_cost_usd', 'unknown_attempts',
                  'reserved', 'halted')
JOURNAL_DAMAGE = (
    [(f'нет поля {k}', lambda d, k=k: d.pop(k)) for k in JOURNAL_FIELDS]
    + [(f'{k} = {v!r}', lambda d, k=k, v=v: d.update({k: v})) for k, v in [
        # неверный тип
        ('calls', '3'), ('calls', 1.5), ('calls', True), ('calls', None), ('failed_calls', []),
        ('input_tokens', '0'), ('cost_from_response', 0.5), ('unknown_attempts', {}), ('cost_usd', None),
        ('cost_usd', True), ('unknown_cost_usd', '0'), ('reserved', []), ('reserved', None),
        ('reserved', {'a': 5}), ('reserved', {'a': {'usd': '0.01'}}), ('reserved', {'a': {}}), ('halted', 0),
        ('halted', False), ('initialized', None), ('initialized', ''), ('initialized', True), ('model', None),
        ('cap_usd', '1'), ('stop_usd', None),
        # отрицательное значение
        ('calls', -1), ('failed_calls', -1), ('input_tokens', -5), ('output_tokens', -1),
        ('cost_from_response', -1), ('unknown_attempts', -2), ('cost_usd', -0.1), ('unknown_cost_usd', -0.001),
        ('reserved', {'a': {'usd': -0.01}}), ('stop_usd', -0.1), ('cap_usd', -1.0),
        # не число, чужая модель, потолок выше разрешённого или порог выше потолка
        ('cost_usd', float('nan')), ('unknown_cost_usd', float('inf')), ('model', 'typesafe/jev-router'),
        ('cap_usd', 2.0), ('cap_usd', 0), ('stop_usd', 1.5)]]
    + [('порог выше потолка', lambda d: d.update(cap_usd=0.5, stop_usd=0.6))])
JOURNAL_TEXT = [('пустой файл', lambda text: ''), ('пробелы', lambda text: ' \n'),
                ('усечённый JSON', lambda text: text[:len(text) // 2]),
                ('без последней скобки', lambda text: text.rstrip()[:-1]),
                ('список', lambda text: '[' + text + ']'), ('null', lambda text: 'null'),
                ('не UTF-8', lambda text: None)]


@pytest.fixture
def key_reads(monkeypatch):
    """Сколько раз клиент читал файл с ключом."""
    reads, real = [], llm_jev._read_key
    monkeypatch.setattr(llm_jev, '_read_key', lambda path: reads.append(path) or real(path))
    return reads


@pytest.mark.parametrize('damage', [d for _, d in JOURNAL_DAMAGE], ids=[n for n, _ in JOURNAL_DAMAGE])
def test_incomplete_or_wrong_journal_forbids_sending(tmp_path, monkeypatch, key_reads, damage):
    client, seen = _client(tmp_path, _ok())
    d = json.loads(client.budget.path.read_text(encoding='utf-8'))
    damage(d)
    text = json.dumps(d)
    client.budget.path.write_text(text, encoding='utf-8')
    with pytest.raises(JevBudgetExceeded) as err:                 # отказ до резерва и до отправки
        client.ask('state', Q)
    assert isinstance(err.value, JevJournalError) and err.value.code == 'journal'
    assert seen == [] and key_reads == []                         # ни запроса, ни чтения ключа
    assert client.budget.path.read_text(encoding='utf-8') == text  # негодный журнал клиент не «чинит»
    with pytest.raises(JevJournalError):
        client.budget.read()
    for call in (lambda: client.budget.settle('нет', cost=0.01), lambda: client.budget.release('нет'),
                 lambda: client.budget.halt('стоп')):
        with pytest.raises(JevJournalError):                      # KeyError и подобное наружу не выходят
            call()
    monkeypatch.setenv(llm_jev.BUDGET_FILE_ENV, str(client.budget.path))
    assert llm_jev.main([]) == 2                                  # и просмотр журнала отвечает отказом


@pytest.mark.parametrize('spoil', [f for _, f in JOURNAL_TEXT], ids=[n for n, _ in JOURNAL_TEXT])
def test_empty_or_truncated_journal_forbids_sending(tmp_path, key_reads, spoil):
    client, seen = _client(tmp_path, _ok())
    raw = spoil(client.budget.path.read_text(encoding='utf-8'))
    raw = b'\xff\xfe{' if raw is None else raw.encode('utf-8')
    client.budget.path.write_bytes(raw)
    with pytest.raises(JevJournalError):
        client.ask('state', Q)
    assert seen == [] and key_reads == [] and client.budget.path.read_bytes() == raw


def test_complete_journal_is_accepted_and_unused_fields_are_not_required(tmp_path, key_reads):
    client, seen = _client(tmp_path, _ok())
    d = json.loads(client.budget.path.read_text(encoding='utf-8'))
    assert set(JOURNAL_FIELDS) <= set(d)                          # журнал, созданный командой, схему проходит
    for unused in ('carried', 'updated'):                         # проверяется то, чем клиент пользуется
        d.pop(unused, None)
    client.budget.path.write_text(json.dumps(d), encoding='utf-8')
    assert client.ask('state', Q).answers['q']['noul'] == 0.9
    assert len(seen) == 1 and len(key_reads) == 1 and client.budget.read()['calls'] == 1
    with pytest.raises(ValueError):                               # журнал с отрицательным счётчиком не создаётся
        Budget(tmp_path / 'другой' / 'b.json').init(0.0, calls=-1)


def test_key_file_is_read_only_after_a_successful_reservation(tmp_path, key_reads):
    limit = max_cost_usd(_body('state'))
    forbidden = {'исчерпан': _budget(tmp_path / 'a', spent=BUDGET_STOP_USD),
                 'не хватает на запрос': _budget(tmp_path / 'b', spent=round(BUDGET_STOP_USD - limit / 2, 9)),
                 'отсутствует': Budget(tmp_path / 'c' / 'budget' / 'jev_budget.json'),
                 'остановлен': _budget(tmp_path / 'd')}
    forbidden['остановлен'].halt('остановлено человеком')
    for name, budget in forbidden.items():
        client, seen = _client(tmp_path / name, _ok(), budget=budget, max_retries=3)
        with pytest.raises(JevBudgetExceeded):
            client.ask('state', Q)
        assert seen == [] and key_reads == [], name               # файл ключа не прочитан ни разу
    # Разрешённый запрос читает ключ один раз, сколько бы ни было повторов.
    client, seen = _client(tmp_path / 'ok', _script(httpx.ReadTimeout('поздно'), 503, 'ok'),
                           budget=_budget(tmp_path / 'ok'), max_retries=2)
    client.ask('state', Q)
    assert len(seen) == 3 and len(key_reads) == 1
    # Ключ не прочитался: резерв снят до отправки, попытка не учтена, запрос не ушёл.
    for i, make in enumerate((lambda p: p.unlink(), lambda p: p.write_text('OTHER=1\n', encoding='utf-8'),
                              lambda p: p.write_bytes(b'\xff\xfe' + f'OPENROUTER_API_KEY={SECRET}'.encode()))):
        root = tmp_path / f'key{i}'
        client, seen = _client(root, _ok(), budget=_budget(root))
        make(root / 'key.env')
        before = client.budget.read()
        with pytest.raises(JevError) as err:
            client.ask('state', Q)
        after = client.budget.read()
        assert err.value.code == 'key' and seen == [] and after['reserved'] == {} and after['calls'] == 0
        assert Budget.committed(after) == Budget.committed(before) == 0
        assert err.value.__cause__ is None and err.value.__context__ is None
        assert SECRET not in ''.join(traceback.format_exception(err.value))
    client.key_file = None
    with pytest.raises(JevError, match='ключ не задан'):
        client.ask('state', Q)
    assert client.budget.read()['reserved'] == {}


class OddTransportError(Exception):
    pass


LEAKS = {'JevError': lambda auth: JevError('transport ' + auth),
         'Exception': lambda auth: OddTransportError(f'headers were {auth!r}', {'authorization': auth}),
         'RuntimeError': lambda auth: RuntimeError('x' * 190 + auth),            # ключ на границе обрезки
         'ReadError': lambda auth: httpx.ReadError('reset while sending ' + auth),
         'ReadTimeout': lambda auth: httpx.ReadTimeout('timeout, ' + auth),
         'ConnectError': lambda auth: httpx.ConnectError('no route with ' + auth),
         'LLMError': lambda auth: llm_jev.LLMError(auth, attempts=3)}


def _leaking(kind):
    def transport(request):
        raise LEAKS[kind](request.headers['authorization'])
    return transport


def _no_key_in_exception(error):
    """Ни в тексте, ни в цепочке исключений, ни в кадрах клиента ключа нет."""
    assert error.__cause__ is None and error.__context__ is None
    shown = ''.join(traceback.format_exception(error)) + repr(error) + repr(error.args) + repr(vars(error))
    assert SECRET not in shown and SECRET[:8] not in shown
    frames = [f for f, _ in traceback.walk_tb(error.__traceback__) if f.f_code.co_filename == llm_jev.__file__]
    assert any(f.f_code.co_name == '_send' for f in frames)
    for frame in frames:
        if frame.f_code.co_name == '_send':
            assert not {'key', 'headers', 'r', 'e'} & set(frame.f_locals)
        assert SECRET not in repr(frame.f_locals)


@pytest.mark.parametrize('kind', list(LEAKS))
def test_transport_exception_cannot_carry_the_key_out(tmp_path, kind):
    client, seen = _client(tmp_path, _leaking(kind), max_retries=1)
    with pytest.raises(JevError) as err:
        client.ask('state', Q)
    assert type(err.value) is JevError and err.value.code in REASON_CODES
    assert SECRET in seen[0].headers['authorization']             # ключ у транспорта был — и наружу не вышел
    _no_key_in_exception(err.value)
    b = client.budget.read()
    assert b['reserved'] == {} and SECRET not in _all_files(tmp_path)      # журнал расходов и кэш
    if kind == 'ConnectError':
        assert len(seen) == 2 and b['calls'] == 0 and err.value.code == 'network'      # не дошёл: денег не стоило
    elif kind in ('ReadError', 'ReadTimeout'):
        assert len(seen) == 2 and b['unknown_attempts'] == 2      # сбой связи повторяется, каждая попытка учтена
        assert err.value.code == ('timeout' if kind == 'ReadTimeout' else 'network')
    else:
        assert len(seen) == 1 and b['unknown_attempts'] == 1      # чужое исключение не повторяется, но учтено
        assert err.value.code == 'other' and ('***' in str(err.value) or kind == 'RuntimeError')   # там метка за обрезкой
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_error_reply_leaves_no_key_in_the_frames_of_the_client(tmp_path):
    client, _ = _client(tmp_path, lambda request: httpx.Response(403, text=request.headers['authorization']))
    with pytest.raises(JevError) as err:
        client.ask('state', Q)
    assert err.value.code == 'server'
    _no_key_in_exception(err.value)


@pytest.mark.parametrize('stop', [KeyboardInterrupt, SystemExit])
def test_process_interruption_is_not_swallowed_but_the_attempt_is_counted(tmp_path, stop):
    def interrupted(request):
        raise stop()
    client, seen = _client(tmp_path, interrupted, max_retries=3)
    with pytest.raises(stop):
        client.ask('state', Q)
    b = client.budget.read()
    assert len(seen) == 1 and b['unknown_attempts'] == 1 and b['reserved'] == {}
    assert b['unknown_cost_usd'] == pytest.approx(max_cost_usd(_body('state')))


def test_leaking_transport_leaves_no_key_in_decisions_run_record_or_journal(tmp_path, monkeypatch, capsys):
    for kind in LEAKS:
        client, seen = _client(tmp_path / kind, _leaking(kind), budget=_budget(tmp_path / kind))
        guard = GuardedPlanner(client, fallback=True)
        plan = guard.plan(_state())
        code = plan['guard']['reason']
        assert seen and code in DISABLE_REASONS and plan['guard']['disabled'] == f'{code}: {DISABLE_REASONS[code]}'
        assert SECRET not in json.dumps(plan, ensure_ascii=False) and '***' not in plan['reasoning'], kind
        assert kind not in plan['reasoning'] and 'transport' not in plan['reasoning']   # текста исключения нет вовсе
    # Прогон целиком на показе: запись прогона (всё, что собрал регистратор) и сводка — без ключа.
    records = []

    class Keeping(runner.Recorder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            records.append(self)
    monkeypatch.setattr(runner, 'Recorder', Keeping)
    client, seen = _client(tmp_path / 'показ', _leaking('JevError'), budget=_budget(tmp_path / 'показ'))
    res = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'}, llm={'jev': {'client': client}})
    record = json.dumps(vars(records[-1]), ensure_ascii=False, default=repr)
    assert len(seen) == 1 and res['metrics']['returned'] and '"decision": "disabled"' in record
    assert SECRET not in record and SECRET not in json.dumps(res, ensure_ascii=False, default=repr)
    # Исследовательский прогон: обработчик печатает исключение и traceback — ключа нет и там.
    mod = _jev_eval()
    monkeypatch.setattr(mod, 'run_episode', lambda *a, **k: run_episode(*a, **{**k, 'save': False}))
    client, seen = _client(tmp_path / 'серия', _leaking('JevError'), budget=_budget(tmp_path / 'серия'))
    assert mod.run_single(('M1', 'jev', 'medium', 3), '_j1b_test', False, client) is None
    printed = capsys.readouterr()
    assert len(seen) == 1 and 'ОШИБКА: JevError' in printed.err and 'Traceback' in printed.err
    assert SECRET not in printed.err + printed.out
    assert SECRET not in _all_files(tmp_path)                     # журналы расходов и кэши всех клиентов


def test_journal_file_errors_are_client_errors_and_the_guard_falls_back(tmp_path, monkeypatch):
    def denied(*a, **k):
        raise PermissionError(13, 'Permission denied')
    limit = max_cost_usd(_body('s1'))
    # До резерва (нет прав на файл блокировки, на каталог, на замену файла): запрос не отправляется.
    for i, (target, name) in enumerate([(Budget, '_lock'), (llm_jev.tempfile, 'mkstemp'), (llm_jev.os, 'replace'),
                                        (llm_jev.fcntl, 'flock')]):
        root = tmp_path / f'до{i}'
        client, seen = _client(root, _ok(), budget=_budget(root))
        before = client.budget.path.read_bytes()
        with monkeypatch.context() as m:
            m.setattr(target, name, denied)
            with pytest.raises(JevJournalError) as err:
                client.ask('s1', Q)
            assert err.value.code == 'journal' and seen == []
            guard = GuardedPlanner(client, fallback=True)         # на показе сторож отключается, прогон не падает
            assert guard.plan(_state())['guard']['reason'] == 'journal' and seen == []
        assert client.budget.path.read_bytes() == before
    # После ответа: запись итога не удалась — резерв остаётся занятым, ответ не отдаётся и в кэш не идёт.
    client, seen = _client(tmp_path, _ok())
    writes, real = [], Budget._write

    def second_write_fails(self, d):
        writes.append(1)
        if len(writes) == 2:
            raise OSError(28, 'No space left on device')
        return real(self, d)
    with monkeypatch.context() as m:
        m.setattr(Budget, '_write', second_write_fails)
        with pytest.raises(JevJournalError):
            client.ask('s1', Q)
    b = client.budget.read()
    assert len(seen) == 1 and len(b['reserved']) == 1 and b['calls'] == 0
    assert Budget.committed(b) == pytest.approx(limit) and not list((tmp_path / 'cache').glob('*.json'))
    # Связь не установилась, а снять резерв не удалось: резерв тоже остаётся (деньги считаются занятыми).
    client, seen = _client(tmp_path / 'связь', _script(httpx.ConnectError('нет связи')), budget=_budget(tmp_path / 'связь'))
    writes.clear()
    with monkeypatch.context() as m:
        m.setattr(Budget, '_write', second_write_fails)
        with pytest.raises(JevJournalError) as err:
            client.ask('s1', Q)
    assert len(client.budget.read()['reserved']) == 1 and err.value.__context__ is None


# --- только кэш --------------------------------------------------------------------------------

def test_cache_replays_without_network_and_strict_mode_never_reads_the_key(tmp_path, monkeypatch):
    client, seen = _client(tmp_path, _ok(p=0.37))
    first = client.ask({'a': 1}, Q)
    again = client.ask({'a': 1}, Q)
    assert len(seen) == 1 and again.cached and again.answers == first.answers and again.latency_ms == first.latency_ms
    journal = client.budget.path.read_bytes()

    def forbidden(*a, **k):
        raise AssertionError('режим «только кэш» не должен читать ключ, трогать сеть и журнал расходов')
    monkeypatch.setattr(llm_jev, '_read_key', forbidden)
    monkeypatch.setattr(Budget, 'reserve', forbidden)
    monkeypatch.setenv(llm_jev.KEY_FILE_ENV, str(tmp_path / 'key.env'))       # ключ доступен — и всё равно не нужен
    strict = JevClient(cache_dir=tmp_path / 'cache', budget=Budget(tmp_path / 'нет' / 'журнала.json'), strict=True,
                       key_file=tmp_path / 'key.env', transport=httpx.MockTransport(forbidden))
    assert strict.key_file is None
    assert strict.ask({'a': 1}, Q).answers['q']['noul'] == 0.37
    with pytest.raises(CacheMiss):
        strict.ask({'a': 2}, Q)
    with pytest.raises(CacheMiss):                                # и прямой вызов отправки закрыт
        strict._send(b'{}', Q)
    assert client.budget.path.read_bytes() == journal and not (tmp_path / 'нет').exists()
    # Переменная окружения делает «только кэш» любым клиентом процесса, как бы его ни создали.
    monkeypatch.setenv(CACHE_ONLY_ENV, '1')
    forced = JevClient(cache_dir=tmp_path / 'cache', transport=httpx.MockTransport(forbidden))
    assert forced.strict and forced.key_file is None
    with pytest.raises(CacheMiss):
        forced.ask({'a': 3}, Q)


def _jev_eval():
    spec = importlib.util.spec_from_file_location('jev_eval_under_test', ROOT / 'tools' / 'jev_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cache_only_reaches_every_place_where_the_tool_creates_a_client(tmp_path, monkeypatch):
    mod = _jev_eval()
    created, clients = [], []

    def forbidden(*a, **k):
        pytest.fail('режим «только кэш» не должен создавать или менять журнал расходов')
    for name in ('init', 'reserve', 'release', 'settle', 'halt', '_update', '_write', '_lock'):
        monkeypatch.setattr(Budget, name, forbidden)              # поведение, а не текст исходника

    class Recording(JevClient):
        def __init__(self, **kw):
            created.append(kw)
            clients.append(self)
            super().__init__(cache_dir=tmp_path / 'cache', **kw,
                             transport=httpx.MockTransport(lambda r: pytest.fail('запрос ушёл в сеть')))
    monkeypatch.setattr(mod, 'JevClient', Recording)
    monkeypatch.setattr(llm_jev, '_read_key', lambda path: pytest.fail('прочитан ключ'))
    monkeypatch.setenv(llm_jev.KEY_FILE_ENV, str(tmp_path / 'key.env'))       # «ключ доступен»
    monkeypatch.setenv(CACHE_ONLY_ENV, '')                        # восстановится после проверки
    state = {'scenario': 'medium-1', 't': 1.0,
             'facts': {'samples_collected': 0, 'samples_total': 5, 'penalties': 0, 'battery': 50.0,
                       'battery_if_return_now': 41.6, 'x': 0.0, 'y': 0.0, 'trigger': 'start'},
             'options': [{'id': 'C1', 'kind': 'investigate', 'x': 1.2, 'y': 0.0, 'battery_after': 38.5}]}
    monkeypatch.setattr(mod, 'load_states', lambda: [state])
    # Случай из ревью: подбор с --cache-only. Промах кэша — ошибка, а не сетевой запрос.
    with pytest.raises(CacheMiss):
        mod.main(['--tune', 'v5', '--per-mission', '2', '--missions', 'M1', '--cache-only'])
    assert created == [{'strict': True}] and os.environ[CACHE_ONLY_ENV] == '1'
    # Прогоны с --cache-only: клиент, который получают прогоны, — тоже «только кэш».
    got = []
    monkeypatch.setattr(mod, 'run_single', lambda task, out, cache_only, jev: got.append((cache_only, jev)))
    monkeypatch.setattr(mod, 'compile_summary', lambda out, tasks: ({}, tmp_path / 'summary.json'))
    monkeypatch.setattr(mod, 'print_tables', lambda summary: None)
    monkeypatch.setattr(mod, 'load_env', lambda path: pytest.fail('в режиме «только кэш» окружение МАИ не читается'))
    assert mod.main(['--out', '_j1_test', '--seeds', '1', '--levels', 'medium', '--missions', 'M1', '--variants',
                     'jev', '--cache-only']) == 0
    assert created == [{'strict': True}] * 2 and len(got) == 1 and got[0][0] is True and got[0][1].strict
    assert got[0][1] is clients[-1] and len(clients) == 2
    # Без флажка strict не выставляется (а создание клиента само ничего не читает и не отправляет).
    monkeypatch.setenv(CACHE_ONLY_ENV, '')
    assert mod.make_jev(False).strict is False and created[-1] == {'strict': False}


def test_paired_sign_test_ignores_runs_where_the_outcome_was_not_checked():
    mod = _jev_eval()
    cell = lambda v: {'success': v}                                                       # noqa: E731
    # M4: штраф был не во всех прогонах. None — «не проверено», это не поражение.
    a = {'s1': cell(True), 's2': cell(True), 's3': cell(True), 's4': cell(None), 's5': cell(True), 's6': cell(False)}
    b = {'s1': cell(False), 's2': cell(False), 's3': cell(False), 's4': cell(False), 's5': cell(None), 's6': cell(False)}
    got = mod.pair_counts(a, b, sorted(a))
    assert got == {'pairs': 6, 'checked': 4, 'both_success': 0, 'a_only': 3, 'b_only': 0, 'p_sign': 0.25}
    same = mod.pair_counts(a, a, sorted(a))
    assert same['checked'] == 5 and same['both_success'] == 4 and same['a_only'] == same['b_only'] == 0
    assert same['p_sign'] == 1.0


# --- сторож ------------------------------------------------------------------------------------

class FakeJev:
    """Отвечает по функции от (состояние, имя вопроса, вопрос)."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def ask(self, state, questions, model=MODEL):
        self.requests.append((state, questions))
        return JevReply({k: {'type': 'noul', 'noul': self.answer(state, k, q)} for k, q in questions.items()},
                        400, {'input_tokens': 500}, 0.00002, MODEL)


def _state(**over):
    cand = lambda i, x, conf: {'id': i, 'x': x, 'y': 0.0, 'confidence': conf, 'cost_to': 2.0, 'cost_back': 5.0,   # noqa: E731
                               'feasible': True}
    state = {'mission': 'Не заезжай в правую половину арены (x больше 0,5 м): работай только в левой.',
             'trigger': 'candidate_found', 'battery': 50.0, 'return_cost': 4.0, 'pose': {'x': 0.0, 'y': 0.0},
             'samples': {'collected': 1, 'total': 5}, 'penalties': {'total': 0},
             'candidates': [cand('C1', 1.2, 0.9), cand('C2', -1.0, 0.5)],
             'explore_points': [{'id': 'E1', 'x': -1.5, 'y': 1.0, 'unseen_share': 0.3, 'cost_to': 3.0,
                                 'cost_back': 6.0, 'feasible': True}]}
    state.update(over)
    return state


def test_guard_sends_mission_and_generic_facts_in_one_request():
    jev_state, questions, opts = build_request(_state(), 'v1')
    assert jev_state['mission'].startswith('Не заезжай') and jev_state['robot']['samples_collected_so_far'] == 1
    text_state, text_q, _ = build_request(_state())                # итоговая формулировка: факты фразами
    assert 'collected 1 samples' in text_state['robot'] and 'x = ' not in text_state['robot']
    assert 'x = 1.2 m' in text_q['C1']['instructions']['proposed_action']
    assert set(questions) == {'done', 'C1', 'C2', 'E1'} and [o['id'] for o in opts] == ['C1', 'C2', 'E1']
    assert questions['C1']['instructions']['proposed_action']['battery_left_at_base_after_this_action'] == 38.5
    assert all(q['type'] == 'noul' for q in questions.values())


def test_guard_rejects_violating_subgoal_and_rule_proposes_the_next_one():
    jev = FakeJev(lambda s, k, q: 0.97 if k == 'C1' else 0.03)
    plan = GuardedPlanner(jev).plan(_state())
    assert plan['subgoals'] == [{'type': 'investigate', 'target': 'C2'}] and plan['source'] == 'heuristic'
    assert len(jev.requests) == 1 and plan['guard']['asked'] == ['C1', 'C2'] and plan['guard']['decision'] == 'go'
    assert plan['wait_s'] == pytest.approx(0.4)


def test_guard_returns_home_when_mission_is_done_or_every_goal_violates():
    done = GuardedPlanner(FakeJev(lambda s, k, q: 0.95 if k == 'done' else 0.0)).plan(_state())
    assert done['subgoals'] == [{'type': 'return_base'}] and done['guard']['decision'] == 'return'
    none = GuardedPlanner(FakeJev(lambda s, k, q: 0.0 if k == 'done' else 0.9)).plan(_state())
    assert none['subgoals'] == [{'type': 'return_base'}] and none['guard']['decision'] == 'exhausted'
    assert none['guard']['asked'] == ['C1', 'C2', 'E1']


def test_guard_does_not_ask_when_rule_already_returns():
    jev = FakeJev(lambda s, k, q: 0.0)
    plan = GuardedPlanner(jev).plan(_state(candidates=[], explore_points=[]))
    assert plan['subgoals'] == [{'type': 'return_base'}] and jev.requests == [] and 'guard' not in plan


def test_unsure_answer_is_rejected_by_cautious_guard_and_escalated_with_big_model():
    unsure = lambda s, k, q: 0.5 if k == 'C1' else 0.05                                  # noqa: E731
    cautious = GuardedPlanner(FakeJev(unsure)).plan(_state())
    assert cautious['subgoals'][0]['target'] == 'C2'

    class Big:
        calls = 0

        def plan(self, state):
            Big.calls += 1
            return {'reasoning': 'Еду в E1.', 'hypotheses': [], 'subgoals': [{'type': 'explore', 'target': 'E1'}],
                    'source': 'llm', 'exchanges': [{'latency_ms': 15000}], 'error': None}
    guard = GuardedPlanner(FakeJev(unsure), mode='jev_llm', big=Big())
    plan = guard.plan(_state())
    assert Big.calls == 1 and plan['subgoals'][0]['target'] == 'E1' and plan['source'] == 'llm'
    assert plan['guard']['escalated'] and plan['wait_s'] == pytest.approx(15.4) and guard.escalations == 1
    sure = GuardedPlanner(FakeJev(lambda s, k, q: 0.02), mode='jev_llm', big=Big()).plan(_state())
    assert Big.calls == 1 and sure['subgoals'][0]['target'] == 'C1'                     # уверен — большую не зовёт


def test_guard_off_by_default_and_run_with_guard_follows_the_mission():
    from did.agent import AgentConfig
    assert AgentConfig().mission_guard == '' and AgentConfig().guard_wait is False
    base = run_episode('medium', 3, 'adaptive', save=False)
    # Сторож, который всё разрешает, не меняет поведения правила...
    same = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'},
                       llm={'jev': {'client': FakeJev(lambda s, k, q: 0.0)}})
    assert same['metrics']['score'] == base['metrics']['score']
    # ...а сторож «после второго образца — домой» останавливает сбор ровно на двух.
    two = FakeJev(lambda s, k, q: 0.01 if 'collected 0 samples' in s['robot'] or 'collected 1 samples' in s['robot']
                  else 0.99)
    res = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'}, llm={'jev': {'client': two}})
    assert res['metrics']['samples_collected'] == 2 and res['metrics']['returned']


def test_guard_wait_keeps_the_robot_standing():
    slow = FakeJev(lambda s, k, q: 0.0)
    slow.ask = lambda state, questions, model=MODEL: JevReply(                          # noqa: E731
        {k: {'type': 'noul', 'noul': 0.0} for k in questions}, 3000, {'input_tokens': 500}, 0.0, MODEL)
    fast = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'}, llm={'jev': {'client': slow}})
    wait = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev', 'guard_wait': True},
                       llm={'jev': {'client': slow}})
    assert wait['metrics']['time'] > fast['metrics']['time'] + 10.0


# --- запасной режим: Jev недоступен ------------------------------------------------------------

def _failing_clients(tmp_path):
    """Клиенты Jev на каждый вид отказа: (название, клиент, ожидаемый вид ошибки)."""
    def no_network(request):
        raise httpx.ConnectError('нет сети')

    def timeout(request):
        raise httpx.ReadTimeout('ответ не пришёл в срок')
    cases = [('network', no_network, JevError, {}), ('timeout', timeout, JevError, {}),
             ('server', lambda r: httpx.Response(503, text=f'down {SECRET}'), JevError, {}),
             ('wrong_model', _ok(model='openai/gpt-6'), JevWrongModel, {}),
             ('bad_reply', _answer_200(answers={'done': 'да'}), JevError, {}),
             ('budget', _ok(), JevBudgetExceeded, {'spent': BUDGET_STOP_USD}),
             ('journal', _ok(), JevBudgetExceeded, {'missing': True}),
             ('key', _ok(), JevError, {'no_key': True}),
             ('other', _leaking('JevError'), JevError, {})]
    for i, (name, handler, error, how) in enumerate(cases):
        root = tmp_path / str(i)
        budget = Budget(root / 'budget' / 'jev_budget.json') if how.get('missing') else _budget(root, how.get('spent', 0.0))
        client = _client(root, handler, budget=budget)[0]
        if how.get('no_key'):
            client.key_file = None
        yield name, client, error


def test_guard_switches_itself_off_and_the_rule_goes_on_when_jev_fails(tmp_path):
    rule_plan = GuardedPlanner(FakeJev(lambda s, k, q: 0.0)).rule.plan(_state())
    seen = []
    for name, jev, error in _failing_clients(tmp_path):
        # Исследовательский прогон (fallback выключен): отказ Jev останавливает прогон, как раньше.
        with pytest.raises(error):
            GuardedPlanner(jev).plan(_state())
    for name, jev, error in _failing_clients(tmp_path / 'показ'):
        guard = GuardedPlanner(jev, fallback=True)
        plan = guard.plan(_state())
        assert plan['subgoals'] == rule_plan['subgoals'] and plan['source'] == rule_plan['source'], name
        assert 'Сторож миссии отключён: ' in plan['reasoning'], name
        assert 'миссия словами дальше не гарантируется' in plan['reasoning'], name
        assert plan['guard']['decision'] == 'disabled' and plan['guard']['disabled'] == guard.disabled, name
        # Причина — код из закрытого списка и постоянное пояснение; текста исключения в журнале решений нет.
        assert plan['guard']['reason'] == name and guard.disabled == f'{name}: {DISABLE_REASONS[name]}', name
        assert f'Сторож миссии отключён: {name}: {DISABLE_REASONS[name]}; миссия' in plan['reasoning'], name
        assert SECRET not in json.dumps(plan, ensure_ascii=False), name
        jev.ask = lambda *a, **k: pytest.fail('отключённый сторож не должен спрашивать Jev')
        later = guard.plan(_state())                              # до конца миссии — только правило
        assert later['subgoals'] == rule_plan['subgoals'] and 'guard' not in later, name
        assert 'Сторож миссии отключён' in later['reasoning'], name
        seen.append(guard.disabled)
    assert len(set(seen)) == len(seen) == len(REASON_CODES)       # причины разные, и это весь закрытый список
    assert set(DISABLE_REASONS) == set(REASON_CODES)
    # Незнакомая ошибка клиента (чужой код, нестроковый код, код с текстом) — «other», текст не копируется.
    for odd in (JevError(f'bad {SECRET}', code=f'leak {SECRET}'), JevError('x', code=['network'])):
        class Odd(FakeJev):
            def ask(self, state, questions, model=MODEL, odd=odd):
                raise odd
        plan = GuardedPlanner(Odd(None), fallback=True).plan(_state())
        assert plan['guard']['reason'] == 'other' and SECRET not in json.dumps(plan, ensure_ascii=False)


def test_cache_miss_still_stops_a_research_replay_even_with_fallback(tmp_path):
    strict = JevClient(cache_dir=tmp_path / 'cache', strict=True)
    guard = GuardedPlanner(strict, fallback=True)
    with pytest.raises(CacheMiss):
        guard.plan(_state())
    assert guard.disabled is None


def test_fallback_is_on_for_shows_and_off_for_research_runs(tmp_path):
    from did.agent import AgentConfig
    cfg = AgentConfig(mission_guard='jev')

    class Broken(FakeJev):
        def ask(self, state, questions, model=MODEL):
            self.requests.append((state, questions))
            if len(self.requests) > 2:
                raise JevError('Jev не ответил за 5 попыток: ReadTimeout')
            return super().ask(state, questions, model)
    assert make_guarded(cfg, {'jev': {'client': Broken(None)}}, None).fallback is True          # показ
    assert make_guarded(cfg, {'jev': {'client': Broken(None), 'fallback': False}}, None).fallback is False
    assert GuardedPlanner(Broken(None)).fallback is False
    # Инструмент исследования передаёт в прогон выключенный запасной режим (поведение, а не текст исходника).
    mod, passed = _jev_eval(), []
    with pytest.MonkeyPatch.context() as m:
        m.setattr(mod, 'run_episode', lambda *a, **k: passed.append(k['llm']) or {'metrics': {'score': 0}, 'file': ''})
        mod.run_single(('M1', 'jev', 'medium', 3), '_j1b_test', False, Broken(None))
    assert passed[0]['jev']['fallback'] is False and make_guarded(cfg, passed[0], None).fallback is False
    # Прогон целиком: Jev отказал на третьем вопросе — робот доезжает миссию по правилу и возвращается.
    base = run_episode('medium', 3, 'adaptive', save=False)
    jev = Broken(lambda s, k, q: 0.0)
    shown = run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'}, llm={'jev': {'client': jev}})
    assert len(jev.requests) == 3                                 # после отказа сторожа больше не спрашивали
    assert shown['metrics']['score'] == base['metrics']['score'] and shown['metrics']['returned']
    with pytest.raises(JevError):
        run_episode('medium', 3, 'adaptive', save=False, config={'mission_guard': 'jev'},
                    llm={'jev': {'client': Broken(lambda s, k, q: 0.0), 'fallback': False}})
