"""Запуск показа (tools/demo.py): выбор порта сервера интерфейса и проверка имени агента — без настоящего сервера."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('demo_launch', ROOT / 'tools' / 'demo.py')
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


class _Proc:
    def __init__(self, dies):
        self.dies, self.pid = dies, 0

    def poll(self):
        return 1 if self.dies else None


@pytest.fixture
def ports(tmp_path, monkeypatch):
    """Порты: 'http' — отвечает чужой сервер без пульта, 'busy' — занят и по HTTP молчит, 'pilot' — наш сервер
    уже работает, 'broken' — порт свободен, но сервер падает по другой причине; остальные свободны."""
    kinds, spawned, up = {}, [], set()

    def http(url, body=None, timeout=3.0):
        port = int(url.split(':')[2].split('/')[0])
        if kinds.get(port) == 'pilot' or port in up:
            return 200, {}
        return (404, None) if kinds.get(port) == 'http' else (None, None)

    def spawn(cmd, log_name, echo=None):
        port = int(cmd[-1])
        spawned.append(port)
        kind = kinds.get(port)
        (tmp_path / log_name).write_text({'busy': 'OSError: [Errno 48] Address already in use\n',
                                          'broken': 'ImportError: no module\n'}.get(kind, 'Лаборатория\n'))
        if kind not in ('busy', 'broken'):
            up.add(port)
        return _Proc(dies=kind in ('busy', 'broken'))

    monkeypatch.setattr(demo, 'LOGS', tmp_path)
    monkeypatch.setattr(demo, 'http', http)
    monkeypatch.setattr(demo, 'spawn', spawn)
    monkeypatch.setattr(demo, 'stop', lambda *a, **k: None)
    monkeypatch.setattr(demo, 'say', lambda text: None)
    monkeypatch.setattr(demo.time, 'sleep', lambda s: None)
    return kinds, spawned


def test_free_port_and_running_server(ports):
    kinds, spawned = ports
    url, proc = demo.ensure_server(8765)
    assert url == 'http://127.0.0.1:8765' and proc is not None and spawned == [8765]
    kinds[8770] = 'pilot'
    assert demo.ensure_server(8770) == ('http://127.0.0.1:8770', None)


def test_busy_port_without_http_is_skipped(ports):
    kinds, spawned = ports
    kinds.update({8765: 'busy', 8766: 'http', 8767: 'busy'})
    url, proc = demo.ensure_server(8765)
    assert url == 'http://127.0.0.1:8768' and proc is not None and spawned == [8765, 8767, 8768]


def test_all_ports_busy_and_other_failures_stop_the_show(ports):
    kinds, spawned = ports
    kinds.update({p: 'busy' for p in range(8765, 8770)})
    with pytest.raises(SystemExit, match='не нашёл свободного порта'):
        demo.ensure_server(8765)
    kinds.clear()
    spawned.clear()
    kinds[8765] = 'broken'                                  # не порт: следующий порт не поможет, нужна причина из журнала
    with pytest.raises(SystemExit, match='не запустился'):
        demo.ensure_server(8765)
    assert spawned == [8765]


def test_unknown_agent_stops_demo_at_once():
    assert demo.agent_error('scientist_v2') is None and demo.agent_error('adaptive') is None
    assert 'scientist_v2' in demo.agent_error('scientist_v3')
    # Настоящий запуск: до сервера, стенда и пульта дело не доходит.
    res = subprocess.run([sys.executable, str(ROOT / 'tools' / 'demo.py'), '--agent', 'scientist_v3', '--port', '1'],
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 2 and 'нет агента «scientist_v3»' in res.stderr and 'adaptive_v2' in res.stderr
