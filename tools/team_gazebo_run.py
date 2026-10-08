#!/usr/bin/env python3
"""Прогон двух роботов в Gazebo одной командой (исследование M1): мир, судья на двоих, два агента, запись.

    ./px python tools/team_gazebo_run.py --level medium --seed 3            # без окна
    ./px python tools/team_gazebo_run.py --level medium --seed 3 --gui      # с окном Gazebo
    ./px python tools/team_gazebo_run.py --level medium --seed 3 --mode pair_lidar   # без связи между роботами

Запись ложится в runs/<exp>/<mode>/<уровень>-<seed>.json.gz и открывается в лаборатории, как прогон быстрого
симулятора: оба пути, граница участков, лента переговоров. Своя сеть ROS и свой раздел Gazebo, чтобы не мешать
показу и чужим прогонам; если Gazebo уже идёт или стоит пауза research/PAUSE — команда ждёт.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path('/Users/a/MAI/DID')


def busy():
    if (MAIN / 'research' / 'PAUSE').exists() or (ROOT / 'research' / 'PAUSE').exists():
        return 'пауза research/PAUSE'
    for name, why in (('gz sim', 'уже идёт Gazebo'), ('tools/gazebo_run.py', 'идёт чужой прогон в Gazebo'),
                      ('tools/gazebo_batch.py', 'идёт чужая серия в Gazebo'), ('tools/demo.py', 'идёт показ')):
        if subprocess.run(['pgrep', '-f', name], stdout=subprocess.DEVNULL).returncode == 0:
            return why
    return None


def stop(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='medium', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--mode', default='team', choices=['team', 'pair_lidar'])
    ap.add_argument('--exp', default='M1gz')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--timeout', type=float, default=1500.0, help='предел по часам машины, с')
    args = ap.parse_args()
    free = 0                            # чужие прогоны идут сериями: стартуем, когда тихо две проверки подряд
    while free < 2:
        why = busy()
        free = 0 if why else free + 1
        if why:
            print(f'жду: {why}', flush=True)
        time.sleep(20)
    env = dict(os.environ, ROS_DOMAIN_ID=os.environ.get('ROS_DOMAIN_ID', '37'),
               GZ_PARTITION=os.environ.get('GZ_PARTITION', 'did_m1'),
               PYTHONPATH=f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")     # библиотека did — из этого дерева
    logs = ROOT / 'runs' / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    tag = f'{args.level}-{args.seed}'
    common = ['--level', args.level, '--seed', str(args.seed), '--exp', args.exp, '--mode', args.mode]

    def start(name, cmd):
        log = open(logs / f'team-{name}-{tag}.log', 'w')
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    procs = [start('world', ['ros2', 'launch', str(ROOT / 'ws/src/did_bringup/launch/team.launch.py'),
                             f'gui:={"true" if args.gui else "false"}'])]
    code = 1
    try:
        time.sleep(8)
        procs.append(start('judge', [sys.executable, '-m', 'did.ros_team', 'judge'] + common))
        agents = [start(r, [sys.executable, '-m', 'did.ros_team', 'agent', '--robot', r] + common) for r in ('tb1', 'tb2')]
        procs += agents
        t0 = time.monotonic()
        while any(a.poll() is None for a in agents):
            if time.monotonic() - t0 > args.timeout:
                print('время вышло: останавливаю', flush=True)
                break
            time.sleep(1.0)
        codes = [a.poll() for a in agents]
        print(f'агенты закончили за {time.monotonic() - t0:.0f} с, коды {codes}', flush=True)
        if codes == [0, 0]:
            code = subprocess.run([sys.executable, '-m', 'did.ros_team', 'merge'] + common, cwd=ROOT, env=env).returncode
    finally:
        stop(procs)
        for p in procs:                 # launch иногда оставляет дочерние процессы: добиваем только свои группы
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return code


if __name__ == '__main__':
    sys.exit(main())
