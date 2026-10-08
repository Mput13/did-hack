"""Имитатор модели: отвечает по простому правилу и по заказу ломается.

Отвечает и за планировщик, и за роли расследования (автор, критик, вывод — см. did/llm_roles.py):
роль узнаётся по полю task во входе.

Нужен, чтобы проверять транспорт и обвязку без ключей и сети. Два режима:
  в процессе — LocalClient(MockResponder(seed, faults)), для быстрых пакетных прогонов;
  HTTP-сервер, совместимый с OpenAI /chat/completions:
      python -m did.llm_mock --port 8791 --faults malformed=0.2,http_500=0.1 --seed 1
"""
import argparse
import json
import random
import signal
import socketserver
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .llm import LLMError, extract_json
from .llm_roles import find_task, mock_reply

# Сбои проверяются в этом порядке, срабатывает первый выпавший: сначала транспорт, потом содержимое.
FAULTS = ('http_500', 'http_429', 'timeout', 'empty', 'malformed', 'invalid_target', 'out_of_arena', 'wrapped')
TRANSPORT = ('http_500', 'http_429', 'timeout')
MODEL_ID = 'mock-planner'
RESERVE = 2.0    # запас заряда сверх пути к цели и возврата на базу


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reachable(items, budget):
    return [it for it in items or []
            if isinstance(it, dict) and it.get('id') and (it.get('feasible') is None or it['feasible'])
            and _num(it.get('cost_to')) + _num(it.get('cost_back')) + RESERVE <= budget]


def _hypotheses(state):
    out = []
    for z in state.get('soil_zones') or []:
        if z.get('status') != 'confirmed':
            out.append({'statement': f"В зоне {z.get('id')} около ({_num(z.get('x')):.1f}; {_num(z.get('y')):.1f}) "
                                     f"грунт дороже примерно в {_num(z.get('mult'), 1.0):.1f} раза",
                        'test': 'проехать 0.5 м по краю зоны и сравнить расход с прогнозом для обычного пола'})
    for alarm in state.get('alarms') or []:
        out.append({'statement': f'Тревога «{alarm}» вызвана неучтённой зоной дорогого грунта',
                    'test': 'измерить расход на коротком отрезке рядом и сравнить с прогнозом'})
    return out[:2]


def mock_plan(state):
    """Корректный план по простому правилу: выгодный кандидат, затем разведка, иначе — на базу."""
    battery = _num(state.get('battery'))
    samples = state.get('samples') or {}
    subgoals, why = [], []
    if samples.get('total') and _num(samples.get('collected')) >= _num(samples.get('total')):
        why.append(f"Все образцы собраны ({samples.get('collected')} из {samples.get('total')}), еду на базу.")
        return {'reasoning': why[0], 'hypotheses': [], 'subgoals': [{'type': 'return_base'}]}

    budget = battery
    cands = _reachable(state.get('candidates'), budget)
    if cands:
        c = max(cands, key=lambda c: _num(c.get('confidence')) / (_num(c.get('cost_to')) + 0.5))
        subgoals.append({'type': 'investigate', 'target': c['id']})
        why.append(f"Кандидат {c['id']}: уверенность {_num(c.get('confidence')):.2f}, путь "
                   f"{_num(c.get('cost_to')):.1f} и возврат {_num(c.get('cost_back')):.1f} при заряде {battery:.1f}.")
        budget -= _num(c.get('cost_to'))
    points = _reachable(state.get('explore_points'), budget)
    if points:
        p = max(points, key=lambda p: _num(p.get('unseen_share')) / (_num(p.get('cost_to')) + 1.0))
        subgoals.append({'type': 'explore', 'target': p['id']})
        why.append(f"{'Затем разведка' if cands else 'Достижимых кандидатов нет, разведка'} {p['id']}: "
                   f"не проверено {_num(p.get('unseen_share')):.0%}, путь {_num(p.get('cost_to')):.1f}.")
    if not subgoals:
        subgoals.append({'type': 'return_base'})
        why.append(f"Заряда {battery:.1f} не хватает ни на одну цель с возвратом и запасом {RESERVE:g} "
                   f"(до базы сейчас {_num(state.get('return_cost')):.1f}), еду на базу.")
    return {'reasoning': ' '.join(why)[:600], 'hypotheses': _hypotheses(state), 'subgoals': subgoals}


def find_state(messages):
    """Состояние из последнего сообщения пользователя, в котором оно есть; иначе {}."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get('role') == 'user':
            try:
                obj = extract_json(m.get('content'))
            except LLMError:
                continue
            if 'battery' in obj or 'candidates' in obj:
                return obj
    return {}


class MockResponder:
    """Ответчик для LocalClient и HTTP-сервера. Сбои выпадают по вероятностям, детерминированно по seed.

    faults — {имя из FAULTS: вероятность}; script — виды первых ответов по порядку, например
    ['http_500', 'ok'], потом снова работают вероятности. На запрос исправления (в переписке уже
    есть ответ модели) приходит корректный план: сбои содержимого на него не действуют.
    """

    def __init__(self, seed=0, faults=None, script=None):
        self.faults = dict(faults or {})
        self.script = list(script or [])
        unknown = sorted((set(self.faults) | set(self.script)) - set(FAULTS) - {'ok'})
        if unknown:
            raise ValueError(f"неизвестные сбои: {', '.join(unknown)}; есть: {', '.join(FAULTS)}")
        self.rng = random.Random(seed)
        self.stats = Counter()               # сколько ответов каждого вида выдано
        self._lock = threading.Lock()

    def __call__(self, messages):
        """Текст ответа. Сбой http_500 или http_429 в процессе — это LLMError; timeout не выпадает."""
        kind, text = self.draw(messages, http=False)
        if kind in TRANSPORT:
            raise LLMError(f'имитатор: сбой транспорта {kind}')
        return text

    def draw(self, messages, http=True):
        """(вид, текст): вид — 'ok' или имя сбоя; для сбоев транспорта текст — корректный план."""
        with self._lock:
            rolls = {name: self.rng.random() for name in FAULTS}   # всегда 8 чисел: ряд не зависит от настроек
            repair = any(isinstance(m, dict) and m.get('role') == 'assistant' for m in messages or [])
            kind = 'ok'
            if self.script:
                kind = self.script.pop(0)
            else:
                for name in FAULTS:
                    if rolls[name] >= self.faults.get(name, 0.0):
                        continue
                    if (name == 'timeout' and not http) or (repair and name not in TRANSPORT):
                        continue
                    kind = name
                    break
            self.stats[kind] += 1
        task = find_task(messages)           # запрос роли расследования (автор, критик, вывод), а не плана
        if task is not None:
            return kind, mock_reply(kind, task, repair)
        return kind, self._render(kind, find_state(messages))

    def _render(self, kind, state):
        if kind == 'empty':
            return ''
        plan = mock_plan(state)
        if kind == 'invalid_target':
            plan['subgoals'][0] = {'type': 'investigate', 'target': 'C99'}
        elif kind == 'out_of_arena':
            plan['subgoals'][0] = {'type': 'goto', 'x': 4.2, 'y': -3.7}
        text = json.dumps(plan, ensure_ascii=False)
        if kind == 'malformed':
            return text[:len(text) * 2 // 3]
        if kind == 'wrapped':
            pretty = json.dumps(plan, ensure_ascii=False, indent=2)
            return ('<think>\nСравню цели по выгоде. Формат подцели: {"type": ..., "target": ...}.\n</think>\n\n'
                    f'Вот план на ближайшие шаги:\n\n```json\n{pretty}\n```\n\nЕсли состояние изменится, пересчитаю.')
        return text


# --- HTTP-сервер -------------------------------------------------------------------------------

def _error(message, kind):
    return {'error': {'message': message, 'type': kind, 'code': kind}}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'did-llm-mock'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload, headers=()):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
        except OSError:                      # клиент не дождался и закрыл соединение
            self.close_connection = True

    def _path(self):
        return self.path.split('?')[0].rstrip('/')

    def _refused(self, suffix):
        """Отвечает 404 на чужой путь и 401 на неверный ключ; True — запрос дальше не обрабатывать."""
        auth = self.headers.get('Authorization', '')
        if not self._path().endswith(suffix):
            self._send(404, _error(f'нет такого пути: {self.path}', 'not_found'))
        elif self.server.api_key and auth != f'Bearer {self.server.api_key}':
            self._send(401, _error(f'Incorrect API key provided: {auth}', 'invalid_api_key'))
        else:
            return False
        return True

    def do_GET(self):
        if not self._refused('/v1/models'):
            self._send(200, {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model', 'owned_by': 'did'}]})

    def do_POST(self):
        srv = self.server
        try:
            raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        except (ValueError, OSError):
            raw = b''
        if self._refused('/v1/chat/completions'):
            return
        try:
            req = json.loads(raw)
            messages = req['messages']
            if not isinstance(messages, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return self._send(400, _error('нужен JSON-объект со списком messages', 'invalid_request_error'))
        srv.requests.append(req)
        if srv.reject_response_format and 'response_format' in req:
            return self._send(400, _error('Unrecognized request argument supplied: response_format',
                                          'invalid_request_error'))
        kind, text = srv.responder.draw(messages, http=True)
        if srv.verbose:
            print(f"{time.strftime('%H:%M:%S')} chat: сообщений {len(messages)}, ответ {kind}", flush=True)
        if kind == 'http_500':
            return self._send(500, _error('имитатор: внутренняя ошибка сервера', 'server_error'))
        if kind == 'http_429':
            headers = [] if srv.retry_after_s is None else [('Retry-After', f'{srv.retry_after_s:g}')]
            return self._send(429, _error('имитатор: слишком много запросов', 'rate_limit_exceeded'), headers)
        if kind == 'timeout':
            srv.stopping.wait(srv.timeout_sleep_s)
        prompt = sum(len(str(m.get('content', ''))) for m in messages if isinstance(m, dict)) // 4 + 1
        answer = len(text) // 4 + 1
        self._send(200, {
            'id': f'chatcmpl-mock-{len(srv.requests)}', 'object': 'chat.completion', 'created': int(time.time()),
            'model': req.get('model') or MODEL_ID,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': answer, 'total_tokens': prompt + answer},
        })


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, responder, reject_response_format=False, api_key=None, timeout_sleep_s=60.0,
                 retry_after_s=None, verbose=False):
        super().__init__(address, _Handler)
        self.responder = responder
        self.reject_response_format = reject_response_format   # 400 на response_format, как старые серверы
        self.api_key = api_key                                 # задан — проверяется заголовок Authorization
        self.timeout_sleep_s = timeout_sleep_s                 # сколько молчать при сбое timeout
        self.retry_after_s = retry_after_s                     # заголовок Retry-After у ответа 429
        self.verbose = verbose
        self.requests = deque(maxlen=500)                      # тела последних запросов chat/completions
        self.stopping = threading.Event()

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)   # в обход HTTPServer: его getfqdn бывает очень медленным
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # ушедший клиент — не ошибка сервера
            super().handle_error(request, client_address)

    @property
    def base_url(self):
        return f'http://{self.server_name}:{self.server_port}/v1'

    def stop(self):
        """Остановить сервер, запущенный через start_mock_server."""
        self.stopping.set()
        self.shutdown()
        self.server_close()


def start_mock_server(port=0, seed=0, faults=None, script=None, host='127.0.0.1', reject_response_format=False,
                      api_key=None, timeout_sleep_s=60.0, retry_after_s=None, verbose=False):
    """Имитатор в потоке-демоне -> (server, base_url). port=0 — любой свободный. Конец: server.stop()."""
    server = MockServer((host, port), MockResponder(seed, faults, script),
                        reject_response_format=reject_response_format, api_key=api_key,
                        timeout_sleep_s=timeout_sleep_s, retry_after_s=retry_after_s, verbose=verbose)
    threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, name='did-llm-mock',
                     daemon=True).start()
    return server, server.base_url


def _parse_faults(text):
    faults = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        name, sep, value = part.partition('=')
        if not sep or name.strip() not in FAULTS:
            raise argparse.ArgumentTypeError(f"«{part}»: нужно имя=вероятность, имена: {', '.join(FAULTS)}")
        try:
            faults[name.strip()] = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'«{part}»: вероятность должна быть числом') from None
    return faults


def main(argv=None):
    ap = argparse.ArgumentParser(description='Имитатор модели-планировщика с интерфейсом OpenAI.')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8791)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--faults', type=_parse_faults, default={}, metavar='ИМЯ=P,...',
                    help='вероятности сбоев: ' + ', '.join(FAULTS))
    ap.add_argument('--timeout-sleep', type=float, default=60.0, help='сколько секунд молчать при сбое timeout')
    ap.add_argument('--reject-response-format', action='store_true', help='отвечать 400 на response_format')
    ap.add_argument('--api-key', help='требовать этот ключ в заголовке Authorization')
    ap.add_argument('--quiet', action='store_true', help='не печатать строку на каждый запрос')
    args = ap.parse_args(argv)

    server = MockServer((args.host, args.port), MockResponder(args.seed, args.faults),
                        reject_response_format=args.reject_response_format, api_key=args.api_key,
                        timeout_sleep_s=args.timeout_sleep, verbose=not args.quiet)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(f'имитатор модели: {server.base_url}, сбои: {args.faults or "нет"}, seed {args.seed}. Ctrl+C — стоп',
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    for sig in (signal.SIGINT, signal.SIGTERM):      # повторный сигнал не должен сорвать закрытие
        signal.signal(sig, signal.SIG_IGN)
    server.stopping.set()
    server.server_close()
    print(f'остановлен, ответов по видам: {dict(server.responder.stats)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
