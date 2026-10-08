"""Таблицы для отчёта R14 по готовой серии E20: результат, парные разности, качество калибровки.

    ./px python -m tools.analyze_r14 > tmp/R14-analysis.md

Числа берутся из runs/E20/summary.json; что назвала калибровка — из журналов в записях прогонов
runs/E20/adaptive_cal*/. Ничего не пересчитывает и не запускает.
"""
import json

import numpy as np

from did.agent import LAW_POWER
from did.calibrate import shape_name
from did.config import Rules
from did.metrics import paired, summarize
from did.recorder import load_trace
from did.runner import RUNS

EXP = 'E20'
ARMS = ['adaptive', 'adaptive_cal', 'gradient', 'adaptive_known']
NAMES = {'adaptive': 'адаптивный', 'adaptive_cal': 'с калибровкой', 'gradient': 'подъём по сигналу',
         'adaptive_known': 'правила известны'}
TABLE = [('samples_share', 'образцы, %', 100.0, 1), ('returned', 'возврат, %', 100.0, 0), ('score', 'счёт', 1.0, 1),
         ('false_collects', 'ложные сборы', 1.0, 2), ('battery_used', 'заряд', 1.0, 1)]


def fmt(stat, scale=1.0, digits=1):
    if stat is None:
        return '—'
    return f"{stat['mean'] * scale:.{digits}f} [{stat['ci'][0] * scale:.{digits}f}; {stat['ci'][1] * scale:.{digits}f}]"


def calibration(journal):
    """Что агент назвал за прогон: закон в конце, когда отказался от допущения, сверка по парам, расход."""
    cal = [e for e in journal if (e.get('data') or {}).get('tag') == 'calibration']
    laws = [e for e in cal if e['data']['what'] == 'law']
    left = [e for e in laws if e['data']['source'] == 'maps' and (e['data']['range'], e['data']['power']) != (2.0, 1.0)]
    pairs = [e for e in cal if e['data']['what'] == 'pairs']
    drains = [e for e in cal if e['data']['what'] == 'drain']
    last = laws[-1]['data'] if laws else {'range': 2.0, 'power': 1.0}
    return {'range': last['range'], 'power': last['power'], 't_left': left[0]['t'] if left else None,
            'changed': (last['range'], last['power']) != (2.0, 1.0),
            't_pairs': next((e['t'] for e in pairs if e['data']['agree']), None),
            'pairs_range': pairs[-1]['data']['range'] if pairs else None,
            'pairs_power': pairs[-1]['data']['power'] if pairs else None,
            'per_m': drains[-1]['data']['per_m'] if drains else None, 't_drain': drains[0]['t'] if drains else None}


def q(values, digits=2):
    v = [x for x in values if x is not None]
    if not v:
        return '—'
    return f'{np.median(v):.{digits}f} [{np.percentile(v, 10):.{digits}f}; {np.percentile(v, 90):.{digits}f}]'


def main():
    s = json.loads((RUNS / EXP / 'summary.json').read_text(encoding='utf-8'))
    spec, runs = s['spec'], s['runs']
    conds, levels = spec['conditions'], spec['levels']
    rng = np.random.default_rng(0)
    pick = lambda arm, cond, level: [r for r in runs if r['arm'] == arm and r['condition'] == cond and r['level'] == level]
    group = {(g['arm'], g['condition'], g['level']): g for g in s['groups']}

    print(f"Серия {EXP}: прогонов {len(runs)}, ошибок {len(s['errors'])}, сценариев на уровень {s['seeds']}, "
          f"посчитано {s['generated']}, {s['wall_s']} с. Итог по утверждениям: {s['status']}.")
    for c in s['claims']:
        print(f"- {c['text']}: {c['status']}")

    print('\n### Результат: условие × вариант (среднее [95% интервал])\n')
    print('| условие | уровень | вариант | ' + ' | '.join(t[1] for t in TABLE) + ' |')
    print('|---|---|---|' + '---|' * len(TABLE))
    for c in conds:
        for lv in levels:
            for arm in ARMS:
                g = group.get((arm, c['id'], lv))
                if g:
                    print(f"| {c['id']} | {lv} | {NAMES[arm]} | "
                          + ' | '.join(fmt(g['stats'][m], k, d) for m, _, k, d in TABLE) + ' |')

    for b, title in (('adaptive', 'с калибровкой − адаптивный'), ('adaptive_known', 'с калибровкой − правила известны'),
                     ('gradient', 'с калибровкой − подъём по сигналу')):
        print(f'\n### Парные разности: {title} (на одинаковых сценариях, среднее [95% интервал])\n')
        print('| условие | уровень | образцы, п.п. | счёт | возврат, п.п. | ложные сборы | выше/ниже/равно по образцам |')
        print('|---|---|---|---|---|---|---|')
        for c in conds:
            for lv in levels:
                a, o = pick('adaptive_cal', c['id'], lv), pick(b, c['id'], lv)
                p = {m: paired(a, o, m, rng) for m in ('samples_share', 'score', 'returned', 'false_collects')}
                ps = p['samples_share']
                print(f"| {c['id']} | {lv} | {fmt(ps, 100)} | {fmt(p['score'])} | {fmt(p['returned'], 100, 0)} | "
                      f"{fmt(p['false_collects'], 1, 2)} | {ps['a_higher']}/{ps['b_higher']}/{ps['ties']} |")

    print('\n### Разность разностей: (с калибровкой − адаптивный) в условии минус то же в контроле\n')
    print('| условие | уровень | образцы, п.п. | счёт |')
    print('|---|---|---|---|')
    for c in conds[1:]:
        for lv in levels:
            row = []
            for m, k in (('samples_share', 100.0), ('score', 1.0)):
                d = {}
                for cond in (c['id'], 'base'):
                    by = {r['seed']: r['metrics'][m] for r in pick('adaptive', cond, lv)}
                    d[cond] = {r['seed']: r['metrics'][m] - by[r['seed']] for r in pick('adaptive_cal', cond, lv)
                               if r['seed'] in by}
                v = np.array([d[c['id']][seed] - d['base'][seed] for seed in d[c['id']] if seed in d['base']])
                boot = rng.choice(v, size=(4000, len(v)), replace=True).mean(axis=1)
                row.append(f'{v.mean() * k:.1f} [{np.percentile(boot, 2.5) * k:.1f}; {np.percentile(boot, 97.5) * k:.1f}]')
            print(f"| {c['id']} | {lv} | {row[0]} | {row[1]} |")

    print('\n### Какую долю уровня «правила известны» даёт каждый вариант (образцы; отношение средних [95% интервал])\n')
    print('| условие | уровень | адаптивный | с калибровкой | известны, % образцов |')
    print('|---|---|---|---|---|')
    for c in conds:
        for lv in levels:
            known = {r['seed']: r['metrics']['samples_share'] for r in pick('adaptive_known', c['id'], lv)}
            cells = []
            for arm in ('adaptive', 'adaptive_cal'):
                mine = {r['seed']: r['metrics']['samples_share'] for r in pick(arm, c['id'], lv)}
                seeds = sorted(set(known) & set(mine))
                a, k = np.array([mine[x] for x in seeds]), np.array([known[x] for x in seeds])
                idx = rng.integers(0, len(seeds), size=(4000, len(seeds)))
                boot = a[idx].mean(axis=1) / np.maximum(k[idx].mean(axis=1), 1e-9)
                cells.append(f'{100 * a.mean() / k.mean():.0f}% [{100 * np.percentile(boot, 2.5):.0f}; '
                             f'{100 * np.percentile(boot, 97.5):.0f}]')
            print(f"| {c['id']} | {lv} | {cells[0]} | {cells[1]} | {100 * np.mean(list(known.values())):.1f} |")

    base = paired(pick('adaptive_cal', 'base', 'medium') + pick('adaptive_cal', 'base', 'hard'),
                  pick('adaptive', 'base', 'medium') + pick('adaptive', 'base', 'hard'), 'score', rng)
    print('\n### «Не хуже» в контроле: счёт, с калибровкой − адаптивный, допуск 1 очко\n')
    for lv in levels + ['оба уровня']:
        p = base if lv == 'оба уровня' else paired(pick('adaptive_cal', 'base', lv), pick('adaptive', 'base', lv), 'score', rng)
        ok = 'интервал целиком выше −1: «не хуже» показано' if p['ci'][0] > -1.0 else \
            'нижняя граница интервала ниже −1: «не хуже» не показано'
        print(f"- {lv}: {fmt(p)} — {ok}")

    print('\n### Качество калибровки (вариант «с калибровкой»; медиана [10%; 90%] по прогонам)\n')
    print('| условие | уровень | истина: R, p, расход | назвал R, м | назвал p | форма названа верно | отказался от допущения, '
          'прогонов | на какой секунде | пары сошлись, прогонов | на какой секунде | по парам R | по парам p | расход, ед/м | '
          'на какой секунде |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    wall = {}
    for c in conds:
        truth = Rules(**c.get('rules', {}))
        power = LAW_POWER[truth.sensor_law]
        for lv in levels:
            rows = []
            for r in pick('adaptive_cal', c['id'], lv):
                rows.append(calibration(load_trace(RUNS / r['file'])['journal']))
            n = len(rows)
            right = sum(shape_name(k['power']) == shape_name(power) for k in rows)
            print(f"| {c['id']} | {lv} | {truth.sensor_range_m}; {power}; {truth.drain_per_m} | {q([k['range'] for k in rows])} | "
                  f"{q([k['power'] for k in rows])} | {right}/{n} | {sum(k['changed'] for k in rows)}/{n} | "
                  f"{q([k['t_left'] for k in rows], 0)} | {sum(k['t_pairs'] is not None for k in rows)}/{n} | "
                  f"{q([k['t_pairs'] for k in rows], 0)} | {q([k['pairs_range'] for k in rows])} | "
                  f"{q([k['pairs_power'] for k in rows])} | {q([k['per_m'] for k in rows])} | "
                  f"{q([k['t_drain'] for k in rows], 1)} |")
    for arm in ARMS:
        wall[arm] = np.mean([r['wall_s'] for r in runs if r['arm'] == arm])
    print('\nВремя расчёта одного прогона, с (в среднем): ' + ', '.join(f'{NAMES[a]} {v:.1f}' for a, v in wall.items()))


if __name__ == '__main__':
    main()
