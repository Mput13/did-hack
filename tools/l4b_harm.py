"""L4b: где память помогает и где вредит — разбор парных разностей второго прогона по лабораториям.

    ./px python tools/l4b_harm.py L4b              # итоговая серия
    ./px python tools/l4b_harm.py L4b_dev_d80      # отладочная

Берутся все прогоны второго шага (основная раскладка и повторы) агента adaptive_v2_lab при базовых правилах на
трудном уровне и сравниваются с прогоном без памяти на том же сценарии. Прогоны делятся по тому, что лежало в
памяти и что с ней стало; для каждой группы — число прогонов, средняя разность счёта и из чего она сложилась.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.runner import RUNS                             # noqa: E402


def rows(s, arm, cond, level, chain):
    base = {(r['seed'], r['layout']): r for r in s['runs'] if r['chain'] is None and r['arm'] == arm[:-4]
            and r['condition'] == cond and r['level'] == level}
    first = {r['seed']: r for r in s['runs'] if r['chain'] == chain and r['step'] == 2 and not r.get('repeat')
             and r['arm'] == arm and r['condition'] == cond and r['level'] == level}
    out = []
    for r in s['runs']:
        if (r['arm'], r['condition'], r['level'], r['chain'], r['step']) != (arm, cond, level, chain, 2):
            continue
        b = base[(r['seed'], r['layout'])]['metrics']
        m = r['metrics']
        out.append({'seed': r['seed'], 'layout': r['layout'], 'mem': first[r['seed']]['memory'],
                    'd': {k: float(m[k]) - float(b[k]) for k in ('score', 'samples_share', 'returned', 'hazard_hits',
                                                                 'hazard_first', 'battery_used', 'false_collects')},
                    'base': b['score'], 'with': m['score']})
    return out


def line(name, sel):
    if not sel:
        return f'| {name} | 0 | — | — | — | — | — |'
    d = {k: float(np.mean([r['d'][k] for r in sel])) for k in sel[0]['d']}
    f = lambda v, n=2: f'{v:+.{n}f}'.replace('.', ',').replace('-', '−')      # noqa: E731
    return (f"| {name} | {len(sel)} | {f(d['score'])} | {f(d['samples_share'] * 100, 1)} | {f(d['returned'] * 100, 1)} | "
            f"{f(d['hazard_first'])} | {f(d['battery_used'])} |")


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'L4b'
    s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
    for arm, cond, level in (('adaptive_v2_lab', 'base', 'hard'), ('adaptive_v2_lab', 'science', 'hard')):
        for chain in ('same_lab', 'moved'):
            sel = rows(s, arm, cond, level, chain)
            if not sel:
                continue
            print(f'\n**{arm}, {cond}, {level}, {chain}: второй прогон, {len(sel)} прогонов '
                  f'({len({r["seed"] for r in sel})} лабораторий)**\n')
            print('| Группа прогонов | Прогонов | Счёт, очки | Образцы, п. п. | Возврат, п. п. | Первых въездов, шт. | Расход, ед. |')
            print('|---|---|---|---|---|---|---|')
            print(line('все', sel))
            print(line('в памяти есть опасная зона', [r for r in sel if r['mem']['hazards'] > 0]))
            print(line('в памяти нет опасных зон (только грунт)', [r for r in sel if r['mem']['hazards'] == 0]))
            print(line('в первом прогоне робот видел, что среда менялась', [r for r in sel if r['mem']['volatile']]))
            print(line('не видел', [r for r in sel if not r['mem']['volatile']]))
            d = np.array([r['d']['score'] for r in sel])
            print(f'\nРазность счёта по прогонам: хуже чем на 10 очков — {int((d < -10).sum())}, от −10 до −1 — '
                  f'{int(((d >= -10) & (d < -1)).sum())}, в пределах ±1 — {int((abs(d) <= 1).sum())}, от +1 до +10 — '
                  f'{int(((d > 1) & (d <= 10)).sum())}, лучше чем на 10 — {int((d > 10).sum())}; '
                  f'стандартное отклонение {d.std(ddof=1):.1f}.')
            worst = sorted(sel, key=lambda r: r['d']['score'])[:5]
            print('Пять худших: ' + '; '.join(
                f"{level}-{r['seed']}, раскладка {r['layout']}: {r['base']:.1f} → {r['with']:.1f} "
                f"(образцы {r['d']['samples_share'] * 100:+.0f} п. п., возврат {r['d']['returned']:+.0f}, "
                f"въезды {r['d']['hazard_first']:+.0f})" for r in worst) + '.')


if __name__ == '__main__':
    main()
