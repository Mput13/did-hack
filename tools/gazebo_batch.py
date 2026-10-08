#!/usr/bin/env python3
"""Несколько прогонов в Gazebo подряд — сверка с быстрым симулятором на новых сценариях (опыт E7).

    pixi run python tools/gazebo_batch.py --level hard --seeds 2 3 4 5 6 7 8 9

Каждый сценарий проходит один раз в Gazebo (без окна) и один раз в быстром симуляторе; записи ложатся в runs/E7.
Свой номер сети ROS и своё имя раздела Gazebo, чтобы не мешать показу, запущенному рядом. Перед каждым прогоном
проверяется пауза (research/PAUSE) и не идёт ли показ: пока он идёт, очередной прогон ждёт.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def busy():
    if (ROOT / 'research' / 'PAUSE').exists():
        return 'пауза'
    if subprocess.run(['pgrep', '-f', 'tools/demo.py'], stdout=subprocess.DEVNULL).returncode == 0:
        return 'идёт показ'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='hard')
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='E7')
    args = ap.parse_args()
    env = dict(os.environ, ROS_DOMAIN_ID='23', GZ_PARTITION='did_batch')
    for seed in args.seeds:
        while (why := busy()):
            print(f'жду: {why}', flush=True)
            time.sleep(30)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'gazebo_run.py'), '--level', args.level, '--seed', str(seed),
                            '--agent', args.agent, '--exp', args.exp], cwd=ROOT, env=env, capture_output=True, text=True)
        tail = [line for line in r.stdout.strip().splitlines() if line.strip()][-1:] or ['нет вывода']
        print(f'{args.level}-{seed}: код {r.returncode}, {time.time() - t0:.0f} с — {tail[0][:120]}', flush=True)
    print('BATCH_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
