#!/usr/bin/env python3
"""Показ одной командой: сервер интерфейса, стенд Gazebo с окном, пульт — и адрес страницы.

    pixi run demo                          # Gazebo, уровень medium, сценарий 3
    pixi run demo --level hard --seed 1    # другой сценарий
    pixi run demo --fast                   # то же без Gazebo, на быстром симуляторе
    pixi run demo --open                   # ещё и открыть страницу в браузере
    pixi run demo --slam                   # рядом с нашей картой — карта от SLAM Toolbox
    pixi run demo --slam-map               # без готовой карты: робот едет по карте SLAM Toolbox

Что происходит: поднимается сервер лаборатории (если он ещё не запущен), стенд (мир, робот, судья),
затем пульт (did/pilot.py). Пульт сам ждёт, пока робот осядет на колёса, — на странице виден
обратный отсчёт. Ctrl+C останавливает всё, что запущено этой командой, и только это.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / 'runs' / '_logs'
STATE = ROOT / 'runs' / '_live' / 'pilot.json'


def say(text):
    print(f'[показ] {text}', flush=True)


def http(url, body=None, timeout=3.0):
    """(код, JSON) или (None, None), если сервер не отвечает."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b'null')
        except ValueError:
            return e.code, None
    except (OSError, ValueError):
        return None, None


def spawn(cmd, log_name, echo=None):
    """Дочерний процесс в своей группе: по ней он потом и останавливается, чужие процессы не трогаем."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / log_name, 'w')
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    if echo is None:
        return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env, text=True)

    def pump():
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith(echo):
                say(line.strip())

    threading.Thread(target=pump, daemon=True).start()
    return proc


def stop(proc, name, grace=12.0):
    if proc is None:
        return
    if proc.poll() is None:
        say(f'останавливаю: {name}')
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)              # в группе ещё кто-то жив (launch иногда оставляет детей)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def ensure_server(port):
    """Сервер лаборатории с пультом: уже работающий или свой. Возвращает (адрес, процесс или None)."""
    for p in range(port, port + 5):
        url = f'http://127.0.0.1:{p}'
        code, _ = http(f'{url}/api/pilot/state')
        if code == 200:
            say(f'сервер интерфейса уже работает: {url}')
            return url, None
        if code is None:
            proc = spawn([sys.executable, '-m', 'did.lab.server', '--port', str(p)], f'demo-lab-{p}.log')
            for _ in range(100):
                if http(f'{url}/api/pilot/state')[0] == 200:
                    say(f'сервер интерфейса запущен: {url}')
                    return url, proc
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            stop(proc, 'сервер интерфейса')
            raise SystemExit(f'сервер интерфейса не запустился: см. {LOGS / f"demo-lab-{p}.log"}')
        say(f'на порту {p} работает сервер старой версии, без пульта — пробую порт {p + 1}')
    raise SystemExit('не нашёл свободного порта для сервера интерфейса')


def settled(url):
    """Пульт на связи со стендом, и робот уже осел на колёса."""
    _, s = http(f'{url}/api/pilot/state')
    return bool(s and s.get('active') and s.get('linked') and s.get('mode') != 'settle' and s.get('settle_left', 1) <= 0)


def start_slam(cmd, slam_map):
    proc = spawn(cmd, 'demo-slam.log')
    say('SLAM Toolbox запущен: робот едет по его карте, готовая карта ему не даётся' if slam_map
        else 'SLAM Toolbox запущен: карта появится на странице, переключатель под картой')
    return proc


def tail(path, n=15):
    try:
        return '\n'.join(Path(path).read_text(errors='replace').splitlines()[-n:])
    except OSError:
        return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='medium', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--fast', action='store_true', help='без Gazebo: быстрый симулятор внутри сервера интерфейса')
    ap.add_argument('--rules', default='base', choices=['base', 'science'])
    ap.add_argument('--port', type=int, default=8765, help='порт сервера интерфейса')
    ap.add_argument('--no-gui', action='store_true', help='не открывать окно Gazebo')
    ap.add_argument('--no-hints', action='store_true', help='не рисовать скрытую правду на полу Gazebo')
    ap.add_argument('--open', action='store_true', help='открыть страницу пульта в браузере')
    ap.add_argument('--slam', action='store_true',
                    help='запустить ещё и SLAM Toolbox: его карта появится на странице рядом с нашей')
    ap.add_argument('--slam-map', action='store_true',
                    help='без готовой карты: SLAM Toolbox строит карту, робот планирует путь и едет по ней')
    ap.add_argument('--slam-start', choices=['stand', 'settle'], default='stand',
                    help='когда запускать SLAM Toolbox: stand — вместе со стендом, settle — когда робот осел на колёса')
    ap.add_argument('--launch-arg', action='append', default=[], metavar='ИМЯ:=ЗНАЧЕНИЕ',
                    help='дополнительный аргумент для stand.launch.py (можно несколько раз)')
    args = ap.parse_args()

    halt = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.set())

    server = stand = pilot = slam = None
    slam_cmd = None                 # SLAM Toolbox ещё не запущен: ждём, пока робот осядет
    fast_started = False
    code = 0
    try:
        url, server = ensure_server(args.port)
        page = f'{url}/#/pilot'
        if args.fast and args.slam_map:
            raise SystemExit('--slam-map работает только в Gazebo: в быстром симуляторе SLAM Toolbox нет')
        if args.fast:
            status, res = http(f'{url}/api/pilot/start', {'backend': 'fastsim', 'level': args.level,
                                                          'seed': args.seed, 'rules': args.rules}, timeout=15.0)
            if status != 200:
                raise SystemExit(f'быстрый симулятор не запустился: {(res or {}).get("error", status)}')
            fast_started = True
            say(f'пульт на быстром симуляторе: уровень {args.level}, сценарий {args.seed}')
        else:
            if STATE.exists() and time.time() - STATE.stat().st_mtime < 3.0:
                raise SystemExit('пульт Gazebo уже работает (другой pixi run demo?) — сначала остановите его')
            http(f'{url}/api/pilot/start', {'backend': 'off'})      # страница должна показывать Gazebo
            launch = ['ros2', 'launch', 'did_bringup', 'stand.launch.py', f'level:={args.level}',
                      f'seed:={args.seed}', f'gui:={"false" if args.no_gui else "true"}', 'rviz:=false',
                      f'show_truth:={"false" if args.no_hints else "true"}', f'rules:={args.rules}', *args.launch_arg]
            say('стенд: ' + ' '.join(launch))
            stand = spawn(launch, 'demo-stand.log')
            pilot = spawn([sys.executable, '-m', 'did.pilot', '--ros', '--level', args.level, '--seed', str(args.seed),
                           '--rules', args.rules, *(['--slam-map'] if args.slam_map else [])],
                          'demo-pilot.log', echo='пульт:')
            if args.slam or args.slam_map:
                # Своя тема и свой кадр карты (env/slam_demo.yaml): /map и кадр map заняты эталонной картой судьи.
                slam_cmd = ['ros2', 'launch', 'slam_toolbox', 'online_async_launch.py', 'use_sim_time:=true',
                            f'slam_params_file:={ROOT / "env" / "slam_demo.yaml"}']
                if args.slam_start == 'stand':
                    slam = start_slam(slam_cmd, args.slam_map)
                    slam_cmd = None
                else:
                    say('SLAM Toolbox запустится, когда робот осядет на колёса')
        say(f'страница пульта: {page}')
        say('Ctrl+C — остановить показ')
        if args.open:
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', page], check=False)

        while not halt.wait(0.5):
            if slam_cmd is not None and settled(url):
                slam = start_slam(slam_cmd, args.slam_map)
                slam_cmd = None
            if stand is not None and stand.poll() is not None:
                say(f'стенд остановился сам (код {stand.returncode}). Последние строки журнала:\n'
                    + tail(LOGS / 'demo-stand.log'))
                code = 1
                break
            if pilot is not None and pilot.poll() is not None:
                say(f'пульт остановился сам (код {pilot.returncode}). Последние строки журнала:\n'
                    + tail(LOGS / 'demo-pilot.log'))
                code = 1
                break
            if slam is not None and slam.poll() is not None:
                say(f'SLAM Toolbox остановился сам (код {slam.returncode}). Последние строки журнала:\n'
                    + tail(LOGS / 'demo-slam.log'))
                if args.slam_map:
                    code = 1
                    break
                slam = None
            if server is not None and server.poll() is not None:
                say('сервер интерфейса остановился сам')
                code = 1
                break
    finally:
        if fast_started:
            http(f'{url}/api/pilot/start', {'backend': 'off'})
        stop(pilot, 'пульт', grace=6.0)
        stop(slam, 'SLAM Toolbox', grace=6.0)
        stop(stand, 'стенд Gazebo', grace=15.0)
        stop(server, 'сервер интерфейса', grace=4.0)
        say('показ остановлен')
    return code


if __name__ == '__main__':
    sys.exit(main())
