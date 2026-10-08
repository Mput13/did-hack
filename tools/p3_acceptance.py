"""Таблица приёмки P3: парные разности счёта, возврата и ложных сборов по условиям и уровням.

    ./px python tools/p3_acceptance.py                                   # runs/E28/summary.json, v4 против v2
    ./px python tools/p3_acceptance.py E28 --pair scientist_v4:scientist_v2
    ./px python tools/p3_acceptance.py E28_first --pickups               # первый прогон; с числом попутных сборов

Критерий записан в experiments/E28.yaml до запуска: трудный уровень, базовые правила — счёт +1,5 и больше с
интервалом выше нуля, возврат не ниже, ложных сборов не больше; средний уровень — не хуже чем на 1 очко; без
событий среды — не хуже чем на 1 очко. Для научных правил печатается мерка «не хуже» (справочно). Все названные
условия, включая ложные сборы (парная разность на прогон не выше нуля), входят в «да/нет».
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


def judge(cond, level, score, back, miss):
    """Проходит ли ячейка. miss — парная разность ложных сборов: условие «не больше» входит в решение."""
    same_back, no_more_misses = back['mean'] >= -1e-9, miss['mean'] <= 1e-9
    if cond == 'base' and level == 'hard':
        return (score['mean'] >= 1.5 and score['ci'][0] > 0.0 and same_back and no_more_misses,
                'счёт +1,5 и больше, интервал выше нуля, возврат не ниже, ложных сборов не больше')
    if cond in ('base', 'no_events'):
        return score['mean'] >= -1.0, 'счёт не ниже чем на 1'
    return (same_back and no_more_misses and score['mean'] >= -1.0,
            'возвратов не меньше, ложных сборов не больше, счёт не ниже чем на 1 (справочно)')


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


def pickups(exp, summary, new, old):
    """Попутные сборы по журналам записей: сколько начато, сколько удалось, сколько промахов — и чистая прибавка
    образцов к прежнему варианту (сбор может лишь заменить образец, который прежний агент взял бы подъездом)."""
    from did.recorder import load_trace
    out = ['Попутные сборы: начато / образец взят / промах / отказ после остановки; чистая прибавка образцов и '
           'сценарии, где число образцов изменилось:']
    for cond in [c['id'] for c in summary['spec']['conditions']]:
        for level in ('hard', 'medium'):
            a = {r['seed']: r for r in summary['runs'] if (r['arm'], r['condition'], r['level']) == (new, cond, level)}
            b = {r['seed']: r for r in summary['runs'] if (r['arm'], r['condition'], r['level']) == (old, cond, level)}
            if not a or not b:
                continue
            begun = taken = missed = 0
            where = []
            for seed, r in sorted(a.items()):
                folder = new if cond == 'base' else f'{new}@{cond}'
                journal = load_trace(ROOT / 'runs' / exp / folder / f'{level}-{seed}.json.gz')['journal']
                for i, e in enumerate(journal):
                    if 'собираю попутно' not in e['text']:
                        continue
                    begun += 1
                    end = next((x['text'] for x in journal[i + 1:] if x['kind'] == 'action'
                                and x['text'].startswith('Сбор в') and x['t'] - e['t'] <= 3.0), '')
                    taken += 'образец взят' in end
                    missed += 'промах' in end
                    where.append(f"{seed} ({e['t']:.1f} с: {'взят' if 'взят' in end else 'промах' if 'промах' in end else 'отказ'})")
            got = lambda rs: int(sum(r['metrics']['samples_collected'] for r in rs.values()))      # noqa: E731
            diff = [f"{s}: {int(a[s]['metrics']['samples_collected'] - b[s]['metrics']['samples_collected']):+d}"
                    for s in sorted(a) if s in b and a[s]['metrics']['samples_collected'] != b[s]['metrics']['samples_collected']]
            out.append(f"  {cond:10s} {level:6s} {begun} / {taken} / {missed} / {begun - taken - missed};  "
                       f"образцов {got(b)} → {got(a)} ({got(a) - got(b):+d});  сборы: {', '.join(where) or '—'};  "
                       f"изменилось: {', '.join(diff) or '—'}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp', nargs='?', default='E28')
    ap.add_argument('--pair', action='append', help='новый:прежний; можно несколько раз')
    ap.add_argument('--pickups', action='store_true', help='счёт попутных сборов по записям прогонов')
    args = ap.parse_args()
    p2_acceptance.judge = judge
    summary = json.loads((ROOT / 'runs' / args.exp / 'summary.json').read_text(encoding='utf-8'))
    for pair in args.pair or ['adaptive_v4:adaptive_v2', 'scientist_v4:scientist_v2']:
        new, old = pair.split(':')
        print(p2_acceptance.table(summary, new, old))
        print()
        print(false_collects(summary, new, old))
        print()
        if args.pickups:
            print(pickups(args.exp, summary, new, old))
            print()
    if summary['errors']:
        print(f"ошибок в прогонах: {len(summary['errors'])}")


if __name__ == '__main__':
    main()
