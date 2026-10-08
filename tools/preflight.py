#!/usr/bin/env python3
"""Проверка перед показом: всё ли на месте и запускается ли.

    pixi run preflight            # быстро, без Gazebo (около 20 секунд)
    pixi run preflight --gazebo   # плюс запуск мира и судьи в Gazebo (около двух минут)

Печатает список «готово / не готово» и возвращает ненулевой код, если показ под угрозой.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = 'http://127.0.0.1:8765'
STALE = ('gz sim', 'parameter_bridge', 'robot_state_publisher', 'did_ros/judge', 'did.ros_agent', 'did.pilot')
results = []


def check(name, ok, hint=''):
    results.append(ok)
    print(f"  {'готово  ' if ok else 'НЕ ГОТОВО'}  {name}" + (f' — {hint}' if hint and not ok else ''))
    return ok


def get(path, timeout=5):
    with urllib.request.urlopen(LAB + path, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def running(pattern):
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gazebo', action='store_true', help='проверить и запуск Gazebo со стендом')
    args = ap.parse_args()
    print('Проверка перед показом\n')

    print('Код и данные')
    sys.path.insert(0, str(ROOT))
    try:
        from did.runner import run_episode
        m = run_episode('easy', 1, 'adaptive', save=False)['metrics']
        check('агент проходит сценарий в быстром симуляторе', m['returned'] and m['samples_collected'] >= 2,
              f"собрано {m['samples_collected']}, вернулся: {m['returned']}")
    except Exception as exc:      # noqa: BLE001
        check('агент проходит сценарий в быстром симуляторе', False, f'{type(exc).__name__}: {exc}')
    tests = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_core.py', '-q', '-x'], cwd=ROOT,
                           capture_output=True, text=True)
    check('базовые автоматические проверки проходят', tests.returncode == 0, tests.stdout.strip().splitlines()[-1:])
    check('страница-объяснение собрана (explain.html)', (ROOT / 'explain.html').exists(), 'pixi run explain')
    check('презентация собрана', (ROOT / 'presentation' / 'checkpoint2.pptx').exists(),
          'нет presentation/checkpoint2.pptx')
    check('сценарий показа записан', (ROOT / 'docs' / 'demo_script.md').exists(), 'нет docs/demo_script.md')

    print('\nИнтерфейс')
    try:
        index = get('/api/index')
        done = [e['id'] for e in index['experiments'] if e['status'] != 'not_run']
        check('сервер интерфейса отвечает', True)
        check(f'серии опытов посчитаны ({len(done)} из {len(index["experiments"])})', len(done) >= 8,
              'pixi run exp all')
    except Exception as exc:      # noqa: BLE001
        check('сервер интерфейса отвечает', False, f'запустите: pixi run lab ({type(exc).__name__})')
    try:
        get('/api/pilot/state')
        check('пульт доступен', True)
    except Exception as exc:      # noqa: BLE001
        check('пульт доступен', False, f'{type(exc).__name__}: перезапустите pixi run lab')

    print('\nПроцессы')
    stale = [p for p in STALE if running(p)]
    check('нет оставшихся процессов симуляции', not stale or not args.gazebo,
          'остались: ' + ', '.join(stale) + ' — завершите их перед показом')

    if args.gazebo:
        print('\nGazebo (запуск мира и судьи без окна)')
        log = open('/tmp/did_preflight_stand.log', 'w')
        stand = subprocess.Popen(['ros2', 'launch', 'did_bringup', 'stand.launch.py', 'level:=easy', 'seed:=1',
                                  'gui:=false'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            time.sleep(8)
            smoke = subprocess.run([sys.executable, str(ROOT / 'tools' / 'smoke_judge.py')], cwd=ROOT,
                                   capture_output=True, text=True, timeout=240)
            check('мир, робот и судья работают в Gazebo', smoke.returncode == 0,
                  (smoke.stdout.strip().splitlines() or ['нет вывода'])[-1])
        except Exception as exc:  # noqa: BLE001
            check('мир, робот и судья работают в Gazebo', False, f'{type(exc).__name__}: {exc}')
        finally:
            import os
            import signal
            try:
                os.killpg(stand.pid, signal.SIGINT)
                stand.wait(timeout=15)
            except Exception:     # noqa: BLE001
                pass
            for name in STALE[:4]:
                subprocess.run(['pkill', '-f', name], capture_output=True)
            log.close()

    bad = results.count(False)
    print(f"\nИтог: {'всё готово к показу' if not bad else f'не готово пунктов: {bad}'}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
