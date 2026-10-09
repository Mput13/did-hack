"""Данные для слайдов о математике (presentation/team): считаются кодом робота, а не рисуются от руки.

    pixi run python presentation/team/mathdata.py      # → presentation/team/data/math.json

1. Карта вероятностей: did.belief.SampleBelief получает три настоящих показания из прогона в Gazebo
   (runs/F1demo/scientist_v2-hard-2-f1b.json.gz): карта после первого, после двух и после трёх.
2. Путь: did.nav.CostGraph на арене сценария 2 (трудный уровень) — кратчайший по длине и кратчайший по расходу,
   если цена грунта роботу уже известна.
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from did.arena import load_arena          # noqa: E402
from did.belief import SampleBelief       # noqa: E402
from did.nav import CostGraph             # noqa: E402

rec = json.load(gzip.open(ROOT / 'runs' / 'F1demo' / 'scientist_v2-hard-2-f1b.json.gz'))
rules, sc, tr = rec['rules'], rec['scenario'], rec['track']
arena = load_arena()


def at(t):
    i = min(range(len(tr['t'])), key=lambda k: abs(tr['t'][k] - t))
    return {'t': round(tr['t'][i], 1), 'x': tr['x'][i], 'y': tr['y'][i], 'z': tr['sensor'][i]}


def grid(b):
    g = np.full((b.h, b.w), -1.0)
    g[b.iy, b.ix] = b.p
    return [[round(float(v), 5) for v in row] for row in g]


def fresh():
    return SampleBelief(arena, len(sc['samples']), rules['sensor_range_m'])


# Три показания из первых секунд прогона: карта после первого, после двух и после трёх.
reads = [at(0.5), at(5.5), at(8.0)]
b, steps = fresh(), []
first = next(e for e in rec['events'] if e['type'] == 'sample_collected')
for r in reads:
    b.update(r['x'], r['y'], r['z'], rules['sensor_sigma'])
    top = b.candidates(min_mass=0.03)[:2]
    steps.append({'reading': {**r, 'd': (1.0 - r['z']) * rules['sensor_range_m']}, 'grid': grid(b),
                  'top': [{'x': c['x'], 'y': c['y'], 'mass': c['mass'], 'to_sample': math.dist((c['x'], c['y']), (first['x'], first['y']))} for c in top]})
belief = {'res': b.res, 'x0': b.x0, 'y0': b.y0, 'w': b.w, 'h': b.h, 'prior': b.prior, 'steps': steps,
          'sample': [first['x'], first['y']], 'range': rules['sensor_range_m'], 'sigma': rules['sensor_sigma']}

# ---- путь: грунты сценария после смены (как на слайде «Сценарий на карте»)
soils = sc['soils']
for e in sc['events']:
    if e['type'] == 'soil_change':
        soils = e['soils']
mult = np.ones(arena.free.shape)
for iy in range(arena.h):
    for ix in range(arena.w):
        x, y = arena.x0 + (ix + 0.5) * arena.res, arena.y0 + (iy + 0.5) * arena.res
        for z in soils:
            inside = math.dist((x, y), (z['x'], z['y'])) <= z['r'] if (z.get('shape') or 'circle') == 'circle' \
                else abs(x - z['x']) <= z['w'] / 2 and abs(y - z['y']) <= z['h'] / 2
            if inside:
                mult[iy, ix] = max(mult[iy, ix], z['mult'])
graph = CostGraph(arena)


def route(start, goal, use_soil):
    graph.set_cost(mult if use_soil else None)
    dist, pred = graph.field(*start)
    pts = graph.trace(pred, *goal)
    graph.set_cost(mult)                       # расход любого пути считаем по настоящей цене грунта
    meters = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
    energy = sum(math.dist(p, q) * 0.5 * (mult[arena.w2g(*p)[1], arena.w2g(*p)[0]] + mult[arena.w2g(*q)[1], arena.w2g(*q)[0]])
                 for p, q in zip(pts, pts[1:])) * rules['drain_per_m']
    return {'pts': [[round(x, 3), round(y, 3)] for x, y in pts], 'meters': round(meters, 2), 'energy': round(energy, 2)}


best = None
for start, goal in (((-1.6, 0.0), (0.9, -0.1)), ((-1.5, -0.2), (0.8, -0.3)), ((-1.2, 0.0), (0.4, 0.0)),
                    ((0.3, 0.9), (1.5, 2.2)), ((-2.0, -0.5), (1.0, -0.2)), ((0.0, 1.0), (1.7, 2.0))):
    short, cheap = route(start, goal, False), route(start, goal, True)
    gain = short['energy'] - cheap['energy']
    print(start, goal, 'короткий', short['meters'], short['energy'], '| по расходу', cheap['meters'], cheap['energy'], '| выигрыш', round(gain, 2))
    if best is None or gain > best['gain']:
        best = {'start': start, 'goal': goal, 'short': short, 'cheap': cheap, 'gain': round(gain, 2)}
path = {**best, 'soils': soils, 'drain_per_m': rules['drain_per_m']}
(ROOT / 'presentation' / 'team' / 'data' / 'math.json').write_text(json.dumps({'belief': belief, 'path': path}, ensure_ascii=False))
for st in steps:
    print('показание', {k: round(v, 2) for k, v in st['reading'].items()}, '| лучшие места', [(round(c['x'], 2), round(c['y'], 2), round(c['mass'], 2), round(c['to_sample'], 2)) for c in st['top']])
print('лучший пример пути', best['start'], best['goal'], best['gain'])
