#!/usr/bin/env python3
"""Где робот был на самом деле: истинная поза по редким сканам записи прогона (исследование G2).

    ./px python tools/g2_trace_truth.py /Users/a/MAI/DID/runs/E7/gazebo/hard-5.json.gz --from 70 --to 100

В записи прогона нет истинной позы, но есть скан раз в секунду (120 лучей). Для каждого скана поза ищется
полным перебором места (±1,2 м вокруг позы агента) и курса по карте, затем уточняется подгонкой. Печатается поза
агента, найденная поза, расхождение, доля лучей, лёгших на карту при лучшей позе (мала — скан не похож на карту
ни из какой позы: робот наклонён), расстояние от найденной позы до преграды на карте и ближайшее отражение по курсу.
Последние столбцы — что по этому скану решает новая осторожность агента: сколько свободно позади в полосе шириной
с робота, сколько корпусу осталось до касания преграды впереди и куда бы он отъехал после столкновения.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.arena import load_arena                       # noqa: E402
from did.localize import LIDAR_OFFSET, PoseTracker      # noqa: E402
from did.nav import corridor_room, escape_plan, free_ahead   # noqa: E402
from did.recorder import load_trace                     # noqa: E402


def locate(arena, tracker, scan, reach=1.2):
    r = np.array(scan['r'], dtype=float) / 100.0
    ang = np.arange(len(r)) * 2 * math.pi / len(r)
    ok = r > 0.12
    qx, qy = LIDAR_OFFSET + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
    X, Y = (a.ravel() for a in np.meshgrid(np.arange(scan['x'] - reach, scan['x'] + reach, 0.05),
                                           np.arange(scan['y'] - reach, scan['y'] + reach, 0.05)))
    free = np.array([arena.clearance(x, y) > 0.08 for x, y in zip(X, Y)])
    X, Y = X[free], Y[free]
    best = (math.inf, None)
    for th in np.arange(-math.pi, math.pi, math.radians(3)):
        c, s = math.cos(th), math.sin(th)
        dist, _, _ = tracker._field((X[:, None] + c * qx - s * qy).ravel(), (Y[:, None] + s * qx + c * qy).ravel())
        cost = np.minimum(np.abs(dist - tracker.bias), 0.15).reshape(len(X), -1).mean(axis=1)
        i = int(cost.argmin())
        if cost[i] < best[0]:
            best = (float(cost[i]), (float(X[i]), float(Y[i]), float(th)))
    pose, quality = tracker._fit(best[1], qx, qy, iters=10)
    return pose, quality


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trace')
    ap.add_argument('--from', dest='t0', type=float, default=0.0)
    ap.add_argument('--to', dest='t1', type=float, default=1e9)
    args = ap.parse_args()
    arena = load_arena()
    tracker = PoseTracker(arena)
    d = load_trace(args.trace)
    tr, modes = d['track'], d['modes']
    print('  t, с  режим     поза агента            по скану               расхождение   легло  до преграды  по курсу'
          '  позади  до касания  отъезд')
    for s in d['scans']:
        if not args.t0 <= s['t'] <= args.t1:
            continue
        pose, quality = locate(arena, tracker, s)
        r = np.array(s['r'], dtype=float) / 100.0
        r[r < 0.12] = np.inf
        i = min(range(len(tr['t'])), key=lambda k: abs(tr['t'][k] - s['t']))
        err = math.hypot(pose[0] - s['x'], pose[1] - s['y'])
        eth = math.degrees((s['th'] - pose[2] + math.pi) % (2 * math.pi) - math.pi)
        front = min([v for v in s['r'][:7] + s['r'][-7:] if v] or [0])
        print(f"{s['t']:6.1f}  {modes[tr['mode'][i]]:8s}  ({s['x']:5.2f}; {s['y']:5.2f}; {math.degrees(s['th']):5.0f}°)  "
              f"({pose[0]:5.2f}; {pose[1]:5.2f}; {math.degrees(pose[2]):5.0f}°)  {err * 100:4.0f} см {eth:5.0f}°  "
              f"{quality[0]:5.2f}  {arena.clearance(pose[0], pose[1]):6.2f} м  {front or '—':>5} см"
              f"  {min(corridor_room(r, back=True), 9.99):5.2f} м  {min(free_ahead(r), 9.99):6.2f} м  "
              f"{'вперёд' if escape_plan(r, back=True)[0] > 0 else 'назад' if escape_plan(r, back=True)[0] < 0 else 'стоять'}")


if __name__ == '__main__':
    main()
