#!/usr/bin/env python3
"""Локализация без Gazebo: журнал сканов (tools/gz_pose_log.py --scans) проигрывается через PoseTracker.

    ./px python tools/g2_replay.py runs/g2_bump/pose.npz runs/gz_loc/pose-hard-1.npz
    ./px python tools/g2_replay.py --old runs/g2_bump/pose.npz        # трекер как до исследования G2

В журнале лежат сканы лидара, поза по одометрии и истинная поза Gazebo в момент каждого скана. Печатается ошибка
исправленной позы против истины, сколько сканов пропущено, сколько раз положение терялось и находилось заново и
какой была ошибка сразу после каждой такой находки (ложная находка — главная опасность).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.arena import load_arena                      # noqa: E402
from did.config import LIDAR_MAX, LIDAR_MIN            # noqa: E402
from did.localize import PoseTracker                   # noqa: E402


def replay(path, verify=True, until=None):
    z = np.load(path)
    tracker = PoseTracker(load_arena(), verify=verify)
    step = float(z['step'])
    err, eth, ms, slip, found = [], [], [], [], []
    lost_scans = relocations = 0
    for t, odom, truth, ranges in zip(z['t'], z['odom'], z['truth'], z['ranges']):
        if until is not None and t > until:
            break
        r = np.asarray(ranges, dtype=float)
        r[~np.isfinite(r) | (r < LIDAR_MIN) | (r > LIDAR_MAX)] = np.inf
        x, y, th = tracker.update(*odom, r, tuple(odom), step)
        e = math.hypot(x - truth[0], y - truth[1])
        a = abs((th - truth[2] + math.pi) % (2 * math.pi) - math.pi)
        ms.append(tracker.stats['ms'])
        slip.append(tracker.stats['slip'])
        if tracker.stats['relocations'] != relocations:
            relocations = tracker.stats['relocations']
            found.append((e, a))
        if tracker.stats['lost']:
            lost_scans += 1
        else:                                      # пока положение потеряно, поза и не обязана быть верной
            err.append(e)
            eth.append(a)
    err, eth = np.array(err), np.array(eth)
    st = tracker.stats
    print(f'{path}: сканов {st["scans"]}, пропущено {st["skipped"]}, из них в потере {lost_scans}')
    print(f'  ошибка позы, пока трекер ей верит: медиана {np.median(err) * 100:.1f} см, 95% '
          f'{np.percentile(err, 95) * 100:.1f} см, наибольшая {err.max() * 100:.1f} см; курс — медиана '
          f'{math.degrees(np.median(eth)):.1f}°, 95% {math.degrees(np.percentile(eth, 95)):.1f}°')
    print(f'  сдвиг позы поправкой за {PoseTracker.SLIP_SCANS} сканов: 99% {np.percentile(slip, 99) * 100:.1f} см, '
          f'наибольший {max(slip) * 100:.1f} см; время на скан: медиана {np.median(ms):.2f} мс, наибольшее {max(ms):.0f} мс')
    if found:
        print(f'  найдено заново {len(found)} раз; ошибка сразу после: наибольшая {max(f[0] for f in found) * 100:.1f} см '
              f'и {math.degrees(max(f[1] for f in found)):.1f}°')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--old', action='store_true', help='трекер без проверки в обе стороны и поиска заново')
    ap.add_argument('--until', type=float, default=None, help='только сканы до этого времени симуляции, с')
    args = ap.parse_args()
    for path in args.logs:
        replay(path, verify=not args.old, until=args.until)


if __name__ == '__main__':
    main()
