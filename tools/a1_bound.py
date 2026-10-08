"""A1: оценка сверху для счёта на сценарии — сколько очков набрал бы тот, кто знает всё и не ошибается.

    ./px python tools/a1_bound.py --seeds 6001-6040 --levels medium,hard --out runs/E23_bound.json

Это не агент, а расчёт по правде сценария: точный перебор подмножеств и порядков образцов (задача
ориентирования) при оптимистичных допущениях, каждое из которых только завышает счёт:
  * грунт в клетке — самый дешёвый из всех, что там будут за прогон; опасных зон и сбоев нет;
  * путь по сетке укорочен в 1,082 раза (ломаная по клеткам длиннее прямой не больше чем на столько);
  * до образца и до базы достаточно не доехать 0,30 м (радиус сбора и радиус базы), и эти метры
    списываются по цене самого дорогого грунта сценария;
  * расход — только метры и простой на скорости 0,22 м/с; повороты, груз, разгон не считаются.
Настоящий агент этот счёт превысить не может; оракул oracle_all должен быть к нему близко.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.arena import load_arena  # noqa: E402
from did.config import V_MAX, Rules  # noqa: E402
from did.nav import CostGraph  # noqa: E402
from did.scenario import generate  # noqa: E402

GRID_SLACK = 1.082        # путь по восьми направлениям длиннее прямой не больше чем в 1/cos(22,5°) раза


def bound(level, seed, arena, rules=None):
    r = rules or Rules()
    sc = generate(level, seed, arena)
    X, Y = arena.cell_centers()
    maps = [sc.soils] + [ev['soils'] for ev in sc.events if ev['type'] == 'soil_change']
    grid = None
    for soils in maps:
        g = np.ones(X.shape)
        for z in soils:
            g = np.where(z.mask(X, Y), np.maximum(g, z.mult), g)
        grid = g if grid is None else np.minimum(grid, g)
    graph = CostGraph(arena)
    graph._wall[:] = 1.0                      # без надбавки у стен: нужна самая дешёвая дорога
    graph.set_cost(grid)
    pts = [tuple(sc.base)] + [tuple(p) for p in sc.samples]
    n = len(pts)
    per_m = r.drain_per_m / GRID_SLACK + r.drain_idle_per_s / V_MAX
    cost = np.zeros((n, n))
    for i in range(n):
        dist, pred = graph.field(*pts[i])
        for j in range(n):
            if i != j:
                reach = r.collect_radius_m + (r.base_radius_m if 0 in (i, j) else r.collect_radius_m)
                cost[i, j] = max(0.0, graph.energy(dist, pred, *pts[j]) - reach * grid.max()) * per_m
    k = n - 1                                 # образцы — узлы 1..k
    dp = np.full((1 << k, k), np.inf)
    for i in range(k):
        dp[1 << i, i] = cost[0, i + 1]
    for mask in range(1, 1 << k):
        for last in range(k):
            c = dp[mask, last]
            if not math.isfinite(c):
                continue
            for nxt in range(k):
                if not mask >> nxt & 1:
                    c2 = c + cost[last + 1, nxt + 1]
                    if c2 < dp[mask | 1 << nxt, nxt]:
                        dp[mask | 1 << nxt, nxt] = c2
    best = r.pts_return + r.pts_battery_left * r.battery_start      # никуда не ездить
    best_n = 0
    for mask in range(1, 1 << k):
        total = (dp[mask] + cost[1:, 0]).min()
        if total <= r.battery_start:
            count = bin(mask).count('1')
            score = r.pts_sample * count + r.pts_return + r.pts_battery_left * (r.battery_start - total)
            if score > best:
                best, best_n = score, count
    return {'level': level, 'seed': seed, 'score': round(float(best), 2), 'samples': best_n, 'samples_total': k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='6001-6040')
    ap.add_argument('--levels', default='medium,hard')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.seeds.split('-'))
    arena = load_arena()
    rows = [bound(level, seed, arena) for level in args.levels.split(',') for seed in range(lo, hi + 1)]
    for level in args.levels.split(','):
        sel = [x for x in rows if x['level'] == level]
        print(f"{level}: оценка сверху {np.mean([x['score'] for x in sel]):.1f} очка, образцов "
              f"{np.mean([x['samples'] / x['samples_total'] for x in sel]):.1%} (сценариев {len(sel)})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
