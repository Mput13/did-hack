"""Таблицы R8 из сводки, сверка E1 и разбор самых больших потерь.

Запуск из корня: ./px python -m tools.analyze_r8 > tmp/R8-analysis.md
Никаких прогонов и изменений исходных сводок.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from did.recorder import decode_grid, load_trace

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / 'runs/E18/summary.json').read_text())
e1 = json.loads((ROOT / 'runs/E1/summary.json').read_text())
assert not summary['errors'] and not e1['errors']
assert len(summary['runs']) == 2160 and len(e1['runs']) == 240
control = {(r['arm'], r['level'], r['seed']): r for r in summary['runs'] if r['condition'] == 'base'}
checked = 0
for r in e1['runs']:
    if r['level'] not in ('medium', 'hard'):
        continue
    assert r['metrics'] == control[r['arm'], r['level'], r['seed']]['metrics']
    checked += 1
assert checked == 160
print(f'Сверка E1: {checked} из {checked} итогов medium/hard совпали целиком, включая счёт.\n')

groups = {(g['arm'], g['condition'], g['level']): g['stats'] for g in summary['groups']}
deltas = {(g['arm'], g['condition'], g['level']): g['pairs'] for g in summary['control_differences']}
metrics = ['samples_share', 'returned', 'score', 'false_collects']
conditions = [c['id'] for c in summary['spec']['conditions']]
arms = [a['id'] for a in summary['spec']['arms']]


def fmt(stat, metric):
    scale = 100 if metric in ('samples_share', 'returned') else 1
    return f"{stat['mean'] * scale:.1f} [{stat['ci'][0] * scale:.1f}; {stat['ci'][1] * scale:.1f}]"


for level in ('medium', 'hard'):
    print(f'### {level}: средние и 95% интервалы\n')
    print('| Условие | Вариант | Образцы, % | Возврат, % | Счёт | Ложные сборы |')
    print('|---|---|---:|---:|---:|---:|')
    for c in conditions:
        for a in arms:
            print('| ' + ' | '.join([c, a] + [fmt(groups[a, c, level][m], m) for m in metrics]) + ' |')
    print(f'\n### {level}: парное изменение относительно своего контроля\n')
    print('Δ = отклонение − контроль; отрицательные Δ образцов, возврата и счёта — потеря; положительные Δ ложных сборов — ухудшение. Доли — в процентных пунктах.\n')
    print('| Условие | Вариант | Δ образцов, п.п. | Δ возврата, п.п. | Δ счёта | Δ ложных сборов |')
    print('|---|---|---:|---:|---:|---:|')
    for c in conditions[1:]:
        for a in arms:
            print('| ' + ' | '.join([c, a] + [fmt(deltas[a, c, level][m], m) for m in metrics]) + ' |')
    print()

print('### Преимущество adaptive: парные разности\n')
print('В каждой ячейке: среднее [95% интервал], статус; «?» означает, что различие не показано.\n')
print('| Уровень | Условие | Образцы против fixed, п.п. | Образцы против gradient, п.п. | Счёт против fixed | Счёт против gradient |')
print('|---|---|---:|---:|---:|---:|')
for level in ('medium', 'hard'):
    for cond in conditions:
        vals = []
        for metric, other in [('samples_share', 'fixed'), ('samples_share', 'gradient'), ('score', 'fixed'), ('score', 'gradient')]:
            claim = next(c for c in summary['claims'] if c['metric'] == metric and c['b'] == other)
            cell = next(c for c in claim['cells'] if c['level'] == level and c['condition'] == cond)
            mark = {'supported': 'выше', 'refuted': 'ниже', 'inconclusive': '?'}[cell['verdict']]
            vals.append(fmt(cell['pair'], metric) + ', ' + mark)
        print('| ' + ' | '.join([level, cond] + vals) + ' |')

worst = sorted(conditions[1:], key=lambda c: sum(deltas['adaptive', c, l]['score']['mean'] for l in ('medium', 'hard')))[:2]
print('\n### Разбор записей: два самых болезненных отклонения\n')
print('Выбраны по наибольшей средней потере счёта adaptive при равном весе medium и hard: ' + ', '.join(worst) + '. Это описательный выбор после опыта, не отдельная подтверждающая проверка.\n')
for cond in worst:
    print(f'#### {cond}\n')
    for level in ('medium', 'hard'):
        runs = [r for r in summary['runs'] if r['arm'] == 'adaptive' and r['condition'] == cond and r['level'] == level]
        reasons = Counter(r['metrics']['reason'] for r in runs)
        false = sum(r['metrics']['false_collects'] for r in runs)
        print(f"{level}: причины завершения {dict(reasons)}, ложных сборов суммарно {false}; невозвратов {sum(not r['metrics']['returned'] for r in runs)}/40.")
        components = Counter()
        for r in runs:
            m, b = r['metrics'], control[r['arm'], level, r['seed']]['metrics']
            for key, weight in [('samples_collected', 10), ('returned', 20), ('false_collects', -3),
                                ('collisions', -2), ('hazard_hits', -5)]:
                components[key] += weight * (m[key] - b[key]) / len(runs)
            components['battery_bonus'] += 0.1 * (m['battery_left'] * m['returned'] - b['battery_left'] * b['returned']) / len(runs)
        print('Разложение изменения среднего счёта, очки: ' + ', '.join(f'{k}={v:+.2f}' for k, v in components.items()) + '; бонус заряда вычислен по округлённым метрикам.')
        chosen = sorted(runs, key=lambda r: r['metrics']['score'] - control[r['arm'], level, r['seed']]['metrics']['score'])[:2]
        for r in chosen:
            tr = load_trace(ROOT / 'runs' / r['file'])
            assert tr['agent_rules']['sensor_law'] == 'linear'
            assert tr['rules'] == {**tr['agent_rules'], **next(c for c in summary['spec']['conditions'] if c['id'] == cond)['rules']}
            m, b = r['metrics'], control[r['arm'], level, r['seed']]['metrics']
            track = tr['track']
            end = math.dist((track['x'][-1], track['y'][-1]), tr['scenario']['base'])
            approach = sum(tr['modes'][mode] == 'approach' for mode in track['mode']) * 0.2
            print(f"- `{r['file']}`: образцы {m['samples_collected']}/{m['samples_total']} (контроль {b['samples_collected']}), счёт {m['score']:.2f} (контроль {b['score']:.2f}), ложные сборы {m['false_collects']} (контроль {b['false_collects']}), завершение {m['reason']}, заряд {m['battery_left']:.2f}, до базы в конце {end:.2f} м; время режима approach ≈{approach:.1f} с.")
            misses = [e for e in tr['events'] if e['type'] == 'false_collect']
            if misses:
                ev = misses[0]
                taken = {e['sample'] for e in tr['events'] if e['type'] == 'sample_collected' and e['t'] <= ev['t']}
                remaining = [p for i, p in enumerate(tr['scenario']['samples']) if i not in taken]
                dist = min(math.dist((ev['x'], ev['y']), p) for p in remaining)
                actions = [j for j in tr['journal'] if j['kind'] == 'action' and abs(j['t'] - ev['t']) < 0.11]
                print(f"  Первый промах t={ev['t']:.1f} с, ближайший несобранный образец в {dist:.3f} м (радиус судьи {tr['rules']['collect_radius_m']:.2f} м). Журнал: " + '; '.join(j['text'] for j in actions) + '.')
                belief = tr['belief']
                snap = max((s for s in belief['snaps'] if s['t'] <= ev['t']), key=lambda s: s['t'])
                grid = (decode_grid(snap['data'], belief['h'], belief['w']) / 255.0) ** 2
                iy, ix = np.unravel_index(grid.argmax(), grid.shape)
                peak = (belief['x0'] + (ix + 0.5) * belief['res'], belief['y0'] + (iy + 0.5) * belief['res'])
                print(f"  На последнем снимке карты t={snap['t']:.1f} с глобальный пик {peak[0]:.2f}, {peak[1]:.2f}; до ближайшего оставшегося образца {min(math.dist(peak, p) for p in remaining):.3f} м. Снимок дискретный и не обязан совпадать с текущей целью.")
            else:
                sample_h = [h for h in tr['hypotheses'] if 'образец лежит' in h['statement']]
                print(f"  Гипотез о месте образца {len(sample_h)}, из них опровергнуты {sum(h['status'] == 'refuted' for h in sample_h)}.")
                probes = [(p, s) for p in tr['plans'] for s in p['subgoals'] if s['type'] == 'investigate'][:2]
                for p, sg in probes:
                    taken = {e['sample'] for e in tr['events'] if e['type'] == 'sample_collected' and e['t'] <= p['t']}
                    remaining = [pos for i, pos in enumerate(tr['scenario']['samples']) if i not in taken]
                    distance = min(math.dist((sg['x'], sg['y']), pos) for pos in remaining)
                    print(f"  Цель investigate t={p['t']:.1f} с, ({sg['x']:.2f}, {sg['y']:.2f}), до ближайшего несобранного образца {distance:.3f} м; план: {p['reasoning']}")
            returning = [p for p in tr['plans'] if any(s['type'] == 'return_base' for s in p['subgoals'])]
            if returning:
                print('  Решение о возврате: ' + returning[-1]['reasoning'])
        print()
