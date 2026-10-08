"""Таблица приёмки P1: «условие → разность счёта и возврата «v2 − прежний» с интервалами».

    ./px python tools/p1_acceptance.py                 # по runs/E22b/summary.json
    ./px python tools/p1_acceptance.py E22 --pair scientist_v2:scientist

Критерий записан в experiments/E22b.yaml до запуска: при наших правилах (base, science) на hard прибавка
счёта не меньше 5 очков и интервал выше нуля; на medium потери больше 0,5 очка нет; при чужих правилах
возвратов не меньше, чем у прежнего агента, а счёт ниже не больше чем на 1 очко.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.metrics import paired      # noqa: E402

OURS = ('base', 'science')


def judge(cond, level, score, back):
    """Проходит ли ячейка критерий приёмки: (да/нет, какое требование)."""
    if level != 'hard':
        return score['mean'] >= -0.5, 'счёт не ниже чем на 0,5'
    if cond in OURS:
        return score['mean'] >= 5.0 and score['ci'][0] > 0.0, 'счёт +5 и больше, интервал выше нуля'
    return back['mean'] >= -1e-9 and score['mean'] >= -1.0, 'возвратов не меньше, счёт не ниже чем на 1'


def table(summary, new, old):
    runs = summary['runs']
    labels = {c['id']: c['label'] for c in summary['spec']['conditions']}
    out = [f'**{old} → {new}**, сценарии {summary["spec"]["seed_start"]}–'
           f'{summary["spec"]["seed_start"] + summary["seeds"] - 1}', '',
           '| Условие | Уровень | Счёт: прежний → v2 | Разность счёта [95%] | Возврат: прежний → v2 | '
           'Разность возврата, п.п. [95%] | Время, с | Критерий |', '|---|---|---:|---:|---:|---:|---:|---|']
    ok_all = True
    for cond in labels:
        for level in ('hard', 'medium'):
            a = [r for r in runs if r['arm'] == new and r['condition'] == cond and r['level'] == level]
            b = [r for r in runs if r['arm'] == old and r['condition'] == cond and r['level'] == level]
            if not a or not b:
                continue
            score = paired(a, b, 'score', np.random.default_rng(0))
            back = paired(a, b, 'returned', np.random.default_rng(0))
            mean = lambda rs, m: float(np.mean([r['metrics'][m] for r in rs]))      # noqa: E731
            ok, rule = judge(cond, level, score, back)
            ok_all &= ok
            n = len(a)
            out.append(
                f"| {labels[cond]} | {level} | {mean(b, 'score'):.2f} → {mean(a, 'score'):.2f} | "
                f"{score['mean']:+.2f} [{score['ci'][0]:+.2f}; {score['ci'][1]:+.2f}] | "
                f"{round(mean(b, 'returned') * n)}/{n} → {round(mean(a, 'returned') * n)}/{n} | "
                f"{100 * back['mean']:+.1f} [{100 * back['ci'][0]:+.1f}; {100 * back['ci'][1]:+.1f}] | "
                f"{mean(b, 'time'):.0f} → {mean(a, 'time'):.0f} | {'да' if ok else '**нет**'}: {rule} |")
    out += ['', f"Критерий приёмки для пары выполнен во всех ячейках: {'да' if ok_all else 'нет'}."]
    return '\n'.join(out), ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp', nargs='?', default='E22b')
    ap.add_argument('--pair', action='append', help='новый:прежний; по умолчанию обе пары')
    args = ap.parse_args()
    summary = json.loads((ROOT / 'runs' / args.exp / 'summary.json').read_text(encoding='utf-8'))
    print(f"{args.exp}: прогонов {len(summary['runs'])}, ошибок {len(summary['errors'])}, "
          f"посчитан {summary['generated']}\n")
    for pair in args.pair or ['adaptive_v2:adaptive', 'scientist_v2:scientist']:
        new, old = pair.split(':')
        print(table(summary, new, old)[0], '\n')


if __name__ == '__main__':
    main()
