"""Сервер лаборатории: отдаёт интерфейс из lab/ и данные прогонов из runs/, запускает прогоны.

    pixi run lab            # http://127.0.0.1:8765

Только стандартная библиотека и только localhost: демонстрация не должна зависеть от сети.

API (всё JSON):
  GET  /api/index                     список опытов и их статусы
  GET  /api/arena                     геометрия арены
  GET  /api/options                   уровни, варианты агента, правила судьи по умолчанию
  GET  /api/experiment/<id>           описание и сводка опыта (status: not_run, если ещё не запускался)
  GET  /api/trace?file=<путь в runs>  запись прогона целиком
  GET  /api/scenario?level=&seed=     сценарий без прогона (показ генератора)
  POST /api/run {level, seed, agent, rules?}  один прогон в быстром симуляторе, сразу возвращает сводку;
                                      rules: 'science' — правила с несколькими причинами расхода и сбоями
  POST /api/experiment/<id>/run {seeds?}  запустить серию в фоне
  GET  /api/jobs                      фоновые серии: состояние и последние строки вывода
  GET  /api/live                      текущее состояние прогона в Gazebo (если идёт)
  GET  /api/knowledge                 знания, накопленные между прогонами (runs/_knowledge/kb.json);
                                      пока файла нет — {"rules": [], "runs": 0}
  GET  /api/study/presets             конструктор исследования (did/study.py): готовые задания и словари формы
  POST /api/study/plan {level, seed, spec}  проверка задания и черновой план, без прогона
  POST /api/study {level, seed, spec} исследование по заданию в быстром симуляторе: сводка с отчётом и файлом записи
  GET  /api/pilot/state               пульт (did/pilot.py): поза, карта по лидару, маршрут, миссия
  POST /api/pilot/command {cmd, ...}  команда оператора: route | go | stop | home | reset | mission | finish
  POST /api/pilot/start {backend: 'fastsim', level, seed}  пульт на быстром симуляторе; 'off' — выключить
"""
import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .. import ROOT
from ..agent import PRESETS
from ..arena import load_arena
from ..config import LEVELS, Rules
from ..experiments import build_index, list_specs, load_spec
from ..runner import RUNS, run_episode
from ..scenario import generate
from ..pilot import HUB as PILOT
from .. import study as STUDY             # конструктор исследования: /api/study, /api/study/plan, /api/study/presets

STATIC = ROOT / 'lab'
LIVE = RUNS / '_live' / 'state.json'
KNOWLEDGE = RUNS / '_knowledge' / 'kb.json'
JOBS = {}
_run_lock = threading.Lock()

AGENT_LABELS = {
    'fixed': 'Фиксированный план',
    'adaptive': 'С адаптацией',
    'adaptive_llm': 'С адаптацией, планирует языковая модель',
    'no_soil': 'Без обучения грунтам',
    'no_change': 'Без обнаружения изменений',
    'no_hazard': 'Без памяти об опасных зонах',
    'no_sensor_health': 'Без контроля датчика',
    'static_reserve': 'Возврат по жёсткому порогу',
    'belief_only': 'Только карта образцов',
    'adaptive_ig': 'С адаптацией, разведка по ожидаемой пользе',
    'scientist': 'Исследователь: ведёт расследования',
    'scientist_llm': 'Исследователь с языковой моделью',
    'scientist_fs': 'Исследователь, сравнивает будущие маршруты',
    'adaptive_fs': 'С адаптацией, сравнивает будущие маршруты',
    'spiral': 'Спираль',
    'gradient': 'Подъём по сигналу',
}


def options():
    flags = [f.name for f in fields(PRESETS['adaptive']) if isinstance(getattr(PRESETS['adaptive'], f.name), bool)]
    return {
        'levels': [{'id': k, **v} for k, v in LEVELS.items()],
        'agents': [{'id': k, 'label': AGENT_LABELS.get(k, k), 'search': v.search, 'planner': v.planner,
                    'flags': {f: getattr(v, f) for f in flags}} for k, v in PRESETS.items()] + _baseline_agents(),
        'rules': Rules().to_dict(),
    }


def _baseline_agents():
    """Простые стратегии поиска (did/baselines.py): их тоже можно запускать со страницы сценариев."""
    try:
        from ..baselines import BASELINES
    except Exception:                         # noqa: BLE001 — модуля может не быть в старой копии кода
        return []
    return [{'id': k, 'label': AGENT_LABELS.get(k, k), 'search': cfg.search, 'planner': cfg.planner, 'flags': {}}
            for k, (_, cfg) in BASELINES.items() if k not in PRESETS]


def knowledge():
    """Накопленные знания. Файл пишет агент между прогонами; пока его нет или он пишется — пустой список."""
    empty = {'rules': [], 'runs': 0}
    try:
        data = json.loads(KNOWLEDGE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    data.setdefault('rules', [])
    data.setdefault('runs', 0)
    return data


def start_job(exp_id, seeds=None):
    cmd = [sys.executable, '-u', '-m', 'did.experiments', exp_id]
    if seeds:
        cmd += ['--seeds', str(int(seeds))]
    job = {'id': f'{exp_id}-{int(time.time())}', 'experiment': exp_id, 'state': 'running', 'started': time.time(),
           'lines': [], 'done': 0, 'total': 0}
    JOBS[job['id']] = job

    def work():
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        buf = ''
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in '\r\n':
                line, buf = buf.strip(), ''
                if not line:
                    continue
                if '/' in line and line.replace('/', '').isdigit():
                    job['done'], job['total'] = (int(x) for x in line.split('/'))
                else:
                    job['lines'] = (job['lines'] + [line])[-20:]
            else:
                buf += ch
        job['state'] = 'done' if proc.wait() == 0 else 'failed'
        job['finished'] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    server_version = 'DIDLab/1'

    def log_message(self, fmt, *args):       # тишина в консоли, ошибки видны в ответах
        pass

    # --- ответы ------------------------------------------------------------------------------

    def _send(self, code, body, ctype='application/json; charset=utf-8', gz=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if gz:
            self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, text):
        self._send(code, {'error': text})

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}') if n else {}

    # --- маршруты ----------------------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        try:
            if path == '/api/index':
                return self._send(200, build_index())
            if path == '/api/arena':
                return self._send(200, load_arena().to_dict())
            if path == '/api/options':
                return self._send(200, options())
            if path.startswith('/api/experiment/'):
                exp_id = path.split('/')[3]
                if exp_id not in list_specs():
                    return self._error(404, f'нет опыта {exp_id}')
                summary = RUNS / exp_id / 'summary.json'
                if summary.exists():
                    return self._send(200, summary.read_bytes())
                return self._send(200, {'spec': load_spec(exp_id), 'status': 'not_run', 'runs': [], 'groups': [],
                                        'claims': [], 'errors': []})
            if path == '/api/trace':
                file = (RUNS / q.get('file', '')).resolve()
                if RUNS.resolve() not in file.parents or not file.is_file():
                    return self._error(404, 'нет такой записи')
                return self._send(200, file.read_bytes(), gz=True)
            if path == '/api/scenario':
                sc = generate(q.get('level', 'easy'), int(q.get('seed', 1)), load_arena())
                return self._send(200, sc.to_dict())
            if path == '/api/jobs':
                return self._send(200, sorted(JOBS.values(), key=lambda j: -j['started']))
            if path == '/api/live':
                if LIVE.exists() and time.time() - LIVE.stat().st_mtime < 5.0:
                    return self._send(200, LIVE.read_bytes())
                return self._send(200, {'active': False})
            if path == '/api/study/presets':       # готовые задания и словари для формы конструктора
                return self._send(200, STUDY.api_presets())
            if path == '/api/knowledge':
                return self._send(200, knowledge())
            if path == '/api/pilot/state':
                return self._send(*PILOT.state())
            return self._static(path)
        except Exception as exc:              # noqa: BLE001
            return self._error(500, f'{type(exc).__name__}: {exc}')

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == '/api/pilot/command':
                return self._send(*PILOT.command(body))
            if path == '/api/pilot/start':
                return self._send(*PILOT.start(body))
            if path == '/api/study/plan':          # {level, seed, spec} → проверка задания и черновой план
                return self._send(*STUDY.api_plan(body))
            if path == '/api/study':               # {level, seed, spec} → прогон в быстром симуляторе и отчёт
                with _run_lock:
                    return self._send(*STUDY.api_run(body))
            if path == '/api/run':
                agent = body.get('agent', 'adaptive')
                if agent not in PRESETS and agent not in {a['id'] for a in _baseline_agents()}:
                    return self._error(400, f'нет агента {agent}')
                rules = 'science' if body.get('rules') == 'science' else None
                with _run_lock:
                    summary = run_episode(body.get('level', 'easy'), int(body.get('seed', 1)), agent,
                                          experiment='adhoc', llm=body.get('llm'), rules=rules)
                return self._send(200, summary)
            if path.startswith('/api/experiment/') and path.endswith('/run'):
                exp_id = path.split('/')[3]
                if exp_id not in list_specs():
                    return self._error(404, f'нет опыта {exp_id}')
                if any(j['experiment'] == exp_id and j['state'] == 'running' for j in JOBS.values()):
                    return self._error(409, 'эта серия уже считается')
                return self._send(200, start_job(exp_id, body.get('seeds')))
            return self._error(404, 'нет такого адреса')
        except Exception as exc:              # noqa: BLE001
            return self._error(500, f'{type(exc).__name__}: {exc}')

    def _static(self, path):
        rel = 'index.html' if path in ('', '/') else path.lstrip('/')
        file = (STATIC / rel).resolve()
        if STATIC.resolve() not in file.parents or not file.is_file():
            return self._error(404, 'нет такой страницы')
        ctype = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        if ctype.startswith('text/') or ctype in ('application/javascript', 'application/json'):
            ctype += '; charset=utf-8'
        self._send(200, file.read_bytes(), ctype)


def main():
    ap = argparse.ArgumentParser(description='Лаборатория DID: интерфейс и данные прогонов')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    load_arena()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Лаборатория: http://127.0.0.1:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
