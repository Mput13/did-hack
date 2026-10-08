"""Таблица приёмки P2: парные разности счёта и возврата по условиям и уровням.

    ./px python tools/p2_acceptance.py                                   # runs/E26/summary.json, v3 против v2
    ./px python tools/p2_acceptance.py E26 --pair adaptive_v3:adaptive   # против исходного
    ./px python tools/p2_acceptance.py E26r --pair scientist_v3:scientist_v2

Критерий записан в experiments/E26.yaml до запуска: трудный уровень, базовые правила — счёт +3 и больше с
интервалом выше нуля, возврат не ниже; средний уровень — не хуже чем на 1 очко; без событий среды — не хуже
чем на 1 очко. Для остальных ячеек (научные правила, чужие правила) печатается мерка «не хуже»: возвратов не
меньше, счёт не ниже чем на 1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.metrics import paired      # noqa: E402


def judge(cond, level, score, back):
    """Проходит ли ячейка: (да/нет, какое требование)."""
    if cond == 'base' and level == 'hard':
        return (score['mean'] >= 3.0 and score['ci'][0] > 0.0 and back['mean'] >= -1e-9,
                'счёт +3 и больше, интервал выше нуля, возврат не ниже')
    if cond in ('base', 'no_events'):
        return score['mean'] >= -1.0, 'счёт не ниже чем на 1'
    return back['mean'] >= -1e-9 and score['mean'] >= -1.0, 'возвратов не меньше, счёт не ниже чем на 1 (справочно)'


def table(summary, new, old):
    runs = summary['runs']
    labels = {c['id']: c['label'] for c in summary['spec']['conditions']}
    first = summary['spec']['seed_start']
    out = [f'**{old} → {new}**, сценарии {first}–{first + summary["seeds"] - 1}', '',
           '| Условие | Уровень | Счёт | Разность счёта [95%] | Возврат | Разность возврата, п.п. [95%] | '
           'Собрано, % | Штрафы зон | Ложные сборы | Время, с | Критерий |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
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
            n = len(a)
            out.append(
                f"| {labels[cond]} | {level} | {mean(b, 'score'):.2f} → {mean(a, 'score'):.2f} | "
                f"{score['mean']:+.2f} [{score['ci'][0]:+.2f}; {score['ci'][1]:+.2f}] | "
                f"{round(mean(b, 'returned') * n)}/{n} → {round(mean(a, 'returned') * n)}/{n} | "
                f"{100 * back['mean']:+.1f} [{100 * back['ci'][0]:+.1f}; {100 * back['ci'][1]:+.1f}] | "
                f"{100 * mean(b, 'samples_share'):.1f} → {100 * mean(a, 'samples_share'):.1f} | "
                f"{mean(b, 'hazard_hits'):.2f} → {mean(a, 'hazard_hits'):.2f} | "
                f"{mean(b, 'false_collects'):.2f} → {mean(a, 'false_collects'):.2f} | "
                f"{mean(b, 'time'):.0f} → {mean(a, 'time'):.0f} | {'да' if ok else '**нет**'}: {rule} |")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp', nargs='?', default='E26')
    ap.add_argument('--pair', action='append', help='новый:прежний; можно несколько раз')
    args = ap.parse_args()
    summary = json.loads((ROOT / 'runs' / args.exp / 'summary.json').read_text(encoding='utf-8'))
    for pair in args.pair or ['adaptive_v3:adaptive_v2']:
        new, old = pair.split(':')
        print(table(summary, new, old))
        print()
    if summary['errors']:
        print(f"ошибок в прогонах: {len(summary['errors'])}")


if __name__ == '__main__':
    main()
