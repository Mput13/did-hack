"""Таблица приёмки P3: парные разности счёта, возврата и ложных сборов по условиям и уровням.

    ./px python tools/p3_acceptance.py                                   # runs/E28/summary.json, v4 против v2
    ./px python tools/p3_acceptance.py E28 --pair scientist_v4:scientist_v2

Критерий записан в experiments/E28.yaml до запуска: трудный уровень, базовые правила — счёт +1,5 и больше с
интервалом выше нуля, возврат не ниже, ложных сборов не больше; средний уровень — не хуже чем на 1 очко; без
событий среды — не хуже чем на 1 очко. Для научных правил печатается мерка «не хуже» (справочно).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import p2_acceptance      # noqa: E402

from did.metrics import paired      # noqa: E402


def judge(cond, level, score, back):
    if cond == 'base' and level == 'hard':
        return (score['mean'] >= 1.5 and score['ci'][0] > 0.0 and back['mean'] >= -1e-9,
                'счёт +1,5 и больше, интервал выше нуля, возврат не ниже')
    if cond in ('base', 'no_events'):
        return score['mean'] >= -1.0, 'счёт не ниже чем на 1'
    return back['mean'] >= -1e-9 and score['mean'] >= -1.0, 'возвратов не меньше, счёт не ниже чем на 1 (справочно)'


def false_collects(summary, new, old):
    out = ['Ложные сборы, парная разность на прогон [95%] (и сколько всего: прежний → новый):']
    for cond in [c['id'] for c in summary['spec']['conditions']]:
        for level in ('hard', 'medium'):
            a = [r for r in summary['runs'] if r['arm'] == new and r['condition'] == cond and r['level'] == level]
            b = [r for r in summary['runs'] if r['arm'] == old and r['condition'] == cond and r['level'] == level]
            if a and b:
                d = paired(a, b, 'false_collects', np.random.default_rng(0))
                total = lambda rs: int(sum(r['metrics']['false_collects'] for r in rs))      # noqa: E731
                out.append(f"  {cond:10s} {level:6s} {d['mean']:+.3f} [{d['ci'][0]:+.3f}; {d['ci'][1]:+.3f}]   "
                           f"{total(b)} → {total(a)}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp', nargs='?', default='E28')
    ap.add_argument('--pair', action='append', help='новый:прежний; можно несколько раз')
    args = ap.parse_args()
    p2_acceptance.judge = judge
    summary = json.loads((ROOT / 'runs' / args.exp / 'summary.json').read_text(encoding='utf-8'))
    for pair in args.pair or ['adaptive_v4:adaptive_v2', 'scientist_v4:scientist_v2']:
        new, old = pair.split(':')
        print(p2_acceptance.table(summary, new, old))
        print()
        print(false_collects(summary, new, old))
        print()
    if summary['errors']:
        print(f"ошибок в прогонах: {len(summary['errors'])}")


if __name__ == '__main__':
    main()
