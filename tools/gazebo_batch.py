#!/usr/bin/env python3
"""Несколько прогонов в Gazebo подряд — сверка с быстрым симулятором на новых сценариях (опыт E7).

    pixi run python tools/gazebo_batch.py --level hard --seeds 2 3 4 5 6 7 8 9

Каждый сценарий проходит один раз в Gazebo (без окна) и один раз в быстром симуляторе; записи ложатся в runs/E7.
Свой номер сети ROS и своё имя раздела Gazebo, чтобы не мешать показу, запущенному рядом. Перед каждым прогоном
проверяется пауза (research/PAUSE) и не идёт ли показ: пока он идёт, очередной прогон ждёт.
"""
import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def busy():
    if (ROOT / 'research' / 'PAUSE').exists():
        return 'пауза'
    if subprocess.run(['pgrep', '-f', 'gz sim'], stdout=subprocess.DEVNULL).returncode == 0:
        return 'работает другой Gazebo'
    if os.getloadavg()[0] > float(os.environ.get('DID_MAX_LOAD', 'inf')):
        return f'загрузка {os.getloadavg()[0]:.1f}'
    if subprocess.run(['pgrep', '-f', 'tools/demo.py'], stdout=subprocess.DEVNULL).returncode == 0:
        return 'идёт показ'
    return None


def lagged(path):
    """Доля тактов записи, пришедших с опозданием: признак того, что компьютер был перегружен."""
    try:
        t = json.load(gzip.open(path))['track']['t']
    except (OSError, ValueError, KeyError):
        return None
    gaps = [b - a for a, b in zip(t, t[1:])]
    return sum(1 for g in gaps if g > 0.45) / max(1, len(gaps))


RUN_LIMIT_S = 480.0          # прогон идёт 2–3,5 минуты; дольше — сервер Gazebo завис (бывает: взаимная блокировка в gz-transport)


def run_one(cmd, env):
    """Один прогон с потолком по времени. Зависший прогон прерывается так же, как Ctrl+C: стенд убирает gazebo_run.py."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=RUN_LIMIT_S)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate()
        err += f'\nпрогон прерван: не закончился за {RUN_LIMIT_S:.0f} с'
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='hard')
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='E7')
    args = ap.parse_args()
    # Своя сеть по умолчанию; если ROS_DOMAIN_ID и GZ_PARTITION заданы снаружи, берутся они.
    env = dict(os.environ)
    env.setdefault('ROS_DOMAIN_ID', '23')
    env.setdefault('GZ_PARTITION', 'did_batch')
    arm = 'gazebo' if args.exp == 'E7' else args.agent      # так же, как кладёт запись did/ros_agent.py
    for seed in args.seeds:
        while (why := busy()):
            print(f'жду: {why}', flush=True)
            time.sleep(30)
        for attempt in (1, 2):
            stale = ROOT / 'runs' / args.exp / arm / f'{args.level}-{seed}.json.gz'
            stale.unlink(missing_ok=True)       # запись прежней попытки не должна сойти за новую
            t0 = time.time()
            r = run_one([sys.executable, str(ROOT / 'tools' / 'gazebo_run.py'), '--level', args.level, '--seed', str(seed),
                         '--agent', args.agent, '--exp', args.exp], env)
            tail = [line for line in r.stdout.strip().splitlines() if line.strip()][-1:] or ['нет вывода']
            lag = lagged(ROOT / 'runs' / args.exp / arm / f'{args.level}-{seed}.json.gz')
            note = '' if lag is None else f', опоздавших тактов {lag:.0%}'
            print(f'{args.level}-{seed}: код {r.returncode}, {time.time() - t0:.0f} с{note} — {tail[0][:100]}', flush=True)
            if r.returncode == 0 and (lag is None or lag <= 0.03):
                break
            why = 'компьютер был перегружен, прогон не годится' if r.returncode == 0 else \
                'прогон не состоялся: ' + ' | '.join(r.stderr.strip().splitlines()[-2:])[:300]
            print(f'{args.level}-{seed}: {why}' + ('; повторяю' if attempt == 1 else ''), flush=True)
    print('BATCH_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
