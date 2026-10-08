"""A1: таблицы отчёта research/findings/A1.md из сводки опыта E23.

    ./px python tools/a1_table.py            # читает runs/E23/summary.json, печатает таблицы в Markdown
    ./px python tools/a1_table.py E23_pilot  # то же для подбора на отладочных сценариях

Все числа — из сводки: средние по прогонам группы и парные разности счёта с adaptive на одних и тех же
сценариях. Разность и её 95% интервал берутся из утверждений сводки (claims); для вариантов без своего
утверждения считаются тем же способом (бутстреп по сценариям, did.metrics.paired).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.config import Rules  # noqa: E402
from did.metrics import paired  # noqa: E402

BASE = 'adaptive'


def num(v, digits=1):
    return f'{v:.{digits}f}'.replace('.', ',').replace('-', '−')


def signed(v):
    return ('+' if v >= 0 else '−') + f'{abs(v):.1f}'.replace('.', ',')


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'E23'
    path = Path(__file__).resolve().parent.parent / 'runs' / exp / 'summary.json'
    s = json.loads(path.read_text(encoding='utf-8'))
    spec, runs = s['spec'], s['runs']
    labels = {a['id']: a['label'] for a in spec['arms']}
    print(f"прогонов {len(runs)}, ошибок {len(s['errors'])}, сценарии {spec['seed_start']}–"
          f"{spec['seed_start'] + s['seeds'] - 1}, посчитано {s['generated']}\n")

    def pick(arm, cond, level):
        return [r for r in runs if r['arm'] == arm and r['condition'] == cond and r['level'] == level]

    def mean(sel, key):
        return float(np.mean([float(r['metrics'][key]) for r in sel]))

    claimed = {}                                  # (вариант, условие, уровень) → разность счёта с adaptive из сводки
    for c in s['claims']:
        if c['metric'] == 'score' and c['b'] == BASE and not c.get('kind') and not c.get('minus'):
            for cell in c['cells']:
                claimed[c['a'], cell['condition'], cell['level']] = cell['pair']
    rules = Rules()                               # очки за образец, возврат и штрафы одни при любых правилах среды

    for cond in spec['conditions']:
        for level in spec['levels']:
            print(f"### {cond['label']}, уровень {level}\n")
            print('| Вариант | Счёт | Разность с adaptive [95%] | Собрано | Вернулся | Штрафов | Потрачено заряда |')
            print('|---|---|---|---|---|---|---|')
            base = pick(BASE, cond['id'], level)
            for arm in spec['arms']:
                sel = pick(arm['id'], cond['id'], level)
                if not sel:
                    continue
                if arm['id'] == BASE:
                    diff = '—'
                else:
                    p = claimed.get((arm['id'], cond['id'], level)) or paired(sel, base, 'score',
                                                                                np.random.default_rng(0))
                    diff = f"{signed(p['mean'])} [{signed(p['ci'][0])}; {signed(p['ci'][1])}]"
                print(f"| {labels[arm['id']]} (`{arm['id']}`) | {num(mean(sel, 'score'))} | {diff} | "
                      f"{mean(sel, 'samples_share'):.0%} | {mean(sel, 'returned'):.0%} | "
                      f"{num(mean(sel, 'penalties'), 2)} | {num(mean(sel, 'battery_used'))} |")
            print()

    if any(a['id'] == 'oracle_all' for a in spec['arms']):
        print('### Из чего складывается счёт: adaptive и потолок\n')
        print('| Условие, уровень | Вариант | За образцы | За возврат | За остаток заряда | Штрафы | Счёт |')
        print('|---|---|---|---|---|---|---|')
        for cond in spec['conditions']:
            for level in spec['levels']:
                for arm in (BASE, 'oracle_all'):
                    sel = pick(arm, cond['id'], level)
                    m = [r['metrics'] for r in sel]
                    parts = (np.mean([rules.pts_sample * x['samples_collected'] for x in m]),
                             np.mean([rules.pts_return * x['returned'] for x in m]),
                             np.mean([rules.pts_battery_left * x['battery_left'] * x['returned'] for x in m]),
                             np.mean([rules.pts_collision * x['collisions'] + rules.pts_false_collect * x['false_collects']
                                      + rules.pts_hazard_hit * x['hazard_hits'] for x in m]))
                    print(f"| {cond['label']}, {level} | `{arm}` | " + ' | '.join(num(v) for v in parts)
                          + f" | {num(mean(sel, 'score'))} |")
        print()

    print('### Утверждения опыта (как посчитаны в сводке)\n')
    print('| Утверждение | Условие | Уровень | Разность [95%] | Вывод |')
    print('|---|---|---|---|---|')
    words = {'supported': 'подтверждено', 'refuted': 'опровергнуто', 'inconclusive': 'различие не показано',
             'no_data': 'нет данных'}
    for c in s['claims']:
        for cell in c['cells']:
            p = cell['pair']
            digits = 1 if c['metric'] == 'score' else 3
            diff = f"{p['mean']:+.{digits}f} [{p['ci'][0]:+.{digits}f}; {p['ci'][1]:+.{digits}f}]" if p else '—'
            print(f"| {c['text']} | {cell['condition']} | {cell['level']} | {diff.replace('.', ',').replace('-', '−')} | "
                  f"{words[cell['verdict']]} |")


if __name__ == '__main__':
    main()
