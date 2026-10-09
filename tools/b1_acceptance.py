"""Таблица приёмки B1: парные разности счёта, возврата и ложных сборов по условиям и уровням.

    ./px python tools/b1_acceptance.py                  # runs/B1/summary.json, *_v5 против *_v2

Критерий записан в experiments/B1.yaml до запуска: трудный уровень, базовые правила — счёт +1,5 и больше с
интервалом выше нуля, возврат не ниже, ложных сборов не больше; трудный уровень, научные правила — интервал разности
счёта не целиком ниже −1 и возврат не ниже; средний уровень и условие без событий среды — счёт не ниже чем на 1.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import p2_acceptance      # noqa: E402
import p3_acceptance      # noqa: E402


def judge(cond, level, score, back, miss):
    """Проходит ли ячейка: (да/нет, какое требование)."""
    same_back, no_more_misses = back['mean'] >= -1e-9, miss['mean'] <= 1e-9
    if level == 'medium' or cond == 'no_events':
        return score['mean'] >= -1.0, 'счёт не ниже чем на 1'
    if cond == 'base':
        return (score['mean'] >= 1.5 and score['ci'][0] > 0.0 and same_back and no_more_misses,
                'счёт +1,5 и больше, интервал выше нуля, возврат не ниже, ложных сборов не больше')
    return score['ci'][1] >= -1.0 and same_back, 'интервал счёта не целиком ниже −1, возврат не ниже'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp', nargs='?', default='B1')
    ap.add_argument('--pair', action='append', help='новый:прежний; можно несколько раз')
    args = ap.parse_args()
    p2_acceptance.judge = judge
    summary = json.loads((ROOT / 'runs' / args.exp / 'summary.json').read_text(encoding='utf-8'))
    for pair in args.pair or ['adaptive_v5:adaptive_v2', 'scientist_v5:scientist_v2']:
        new, old = pair.split(':')
        print(p2_acceptance.table(summary, new, old))
        print()
        print(p3_acceptance.false_collects(summary, new, old))
        print()
    if summary['errors']:
        print(f"ошибок в прогонах: {len(summary['errors'])}")


if __name__ == '__main__':
    main()
