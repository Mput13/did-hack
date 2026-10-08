#!/usr/bin/env python3
"""Один прогон в Gazebo одной командой: поднять стенд, провести агента, сохранить запись, всё остановить.

    pixi run gazebo-run --level hard --seed 3                 # без окна
    pixi run gazebo-run --level hard --seed 3 --gui           # с окном Gazebo
    pixi run gazebo-run --level easy --seed 1 --exp E7        # ещё и такой же прогон в быстром симуляторе

Пока прогон идёт, он виден в лаборатории на странице «Живой прогон».
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEFTOVERS = ('gz sim', 'parameter_bridge', 'robot_state_publisher', 'judge_node', 'ros_gz_bridge', 'rviz2')


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    for name in LEFTOVERS:                      # launch иногда оставляет дочерние процессы
        subprocess.run(['pkill', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='easy', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='E7', help='папка в runs/; для E7 дополнительно считается пара в быстром симуляторе')
    ap.add_argument('--gui', action='store_true', help='открыть окно Gazebo')
    ap.add_argument('--rviz', action='store_true')
    ap.add_argument('--no-hints', action='store_true', help='не рисовать скрытую правду на полу Gazebo')
    ap.add_argument('--llm', default=None, choices=['mock', 'http', 'ollama', 'codex'])
    ap.add_argument('--rules', default=None, choices=['science'],
                    help='science — правила с несколькими причинами расхода и сбоями (для агента scientist)')
    ap.add_argument('--pose-log', action='store_true',
                    help='писать журнал поз (одометрия против истины) в runs/<exp>/pose-<уровень>-<seed>.csv')
    args = ap.parse_args()

    logs = ROOT / 'runs' / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / f'stand-{args.level}-{args.seed}.log', 'w')
    launch = ['ros2', 'launch', 'did_bringup', 'stand.launch.py', f'level:={args.level}', f'seed:={args.seed}',
              f'gui:={"true" if args.gui else "false"}', f'rviz:={"true" if args.rviz else "false"}',
              f'show_truth:={"false" if args.no_hints else "true"}', f'rules:={args.rules or "base"}']
    print('стенд:', ' '.join(launch))
    stand = subprocess.Popen(launch, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    code = 1
    try:
        arm = 'gazebo' if args.exp == 'E7' else None
        cmd = [sys.executable, '-m', 'did.ros_agent', '--level', args.level, '--seed', str(args.seed),
               '--agent', args.agent, '--exp', args.exp] + (['--arm', arm] if arm else []) \
            + (['--llm', args.llm] if args.llm else []) + (['--rules', args.rules] if args.rules else [])
        pose_log = None
        if args.pose_log:                       # сам завершится, когда судья закончит прогон
            out = ROOT / 'runs' / args.exp / f'pose-{args.level}-{args.seed}.csv'
            pose_log = subprocess.Popen([sys.executable, str(ROOT / 'tools' / 'gz_pose_log.py'), '--out', str(out),
                                         '--scans', str(out.with_suffix('.npz'))], cwd=ROOT)
        t0 = time.monotonic()
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if pose_log:
            try:
                pose_log.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pose_log.send_signal(signal.SIGINT)
        print(f'прогон занял {time.monotonic() - t0:.0f} с, код {code}')
    finally:
        stop(stand)
        log.close()
    if code == 0 and args.exp == 'E7':
        sys.path.insert(0, str(ROOT))
        from did.experiments import rebuild_from_traces
        from did.runner import run_episode
        run_episode(args.level, args.seed, args.agent, experiment='E7', arm='fastsim')
        s = rebuild_from_traces('E7')
        print(f'E7: записей {len(s["runs"])}, итог: {s["status"]}')
    return code


if __name__ == '__main__':
    sys.exit(main())
