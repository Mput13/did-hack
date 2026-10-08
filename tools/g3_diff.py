"""Сравнение двух сводок опыта по прогонам: какие прогоны изменились и как (исследование G3).

    ./px python tools/g3_diff.py /Users/a/MAI/DID/tmp/E15-before-G2.json /Users/a/MAI/DID/runs/E15/summary.json
    ./px python tools/g3_diff.py ДО ПОСЛЕ --now runs/E15/summary.json      # три колонки: до, после, теперь
    ./px python tools/g3_diff.py ДО ПОСЛЕ --list                           # перечислить изменившиеся прогоны

Прогоны сопоставляются по ключу (вариант, условие, уровень, номер сценария). «Изменился» — отличается счёт,
число образцов, возврат, столкновения или штрафы зон между ДО и ПОСЛЕ.
"""
import argparse
import json

KEYS = ('score', 'samples_collected', 'returned', 'collisions', 'hazard_hits')


def load(path):
    s = json.load(open(path, encoding='utf-8'))
    return {(r['arm'], r.get('condition', 'base'), r['level'], r['seed']): r['metrics'] for r in s['runs']}


def changed(a, b):
    return sorted(k for k in a.keys() & b.keys() if any(a[k][m] != b[k][m] for m in KEYS))


def totals(runs, keys):
    n = max(1, len(keys))
    return {'n': len(keys), 'score': sum(runs[k]['score'] for k in keys) / n,
            'returned': sum(bool(runs[k]['returned']) for k in keys),
            'samples': sum(runs[k]['samples_collected'] for k in keys),
            'collisions': sum(runs[k]['collisions'] for k in keys),
            'hazard_hits': sum(runs[k]['hazard_hits'] for k in keys)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('before')
    ap.add_argument('after')
    ap.add_argument('--now', default=None, help='третья сводка: нынешний код')
    ap.add_argument('--list', action='store_true', help='перечислить изменившиеся прогоны')
    ap.add_argument('--worse', action='store_true', help='в списке только ухудшившиеся (счёт ниже)')
    args = ap.parse_args()
    cols = [('до', load(args.before)), ('после', load(args.after))]
    if args.now:
        cols.append(('теперь', load(args.now)))
    keys = changed(cols[0][1], cols[1][1])
    keys = [k for k in keys if all(k in runs for _, runs in cols)]
    print(f'прогонов {len(cols[0][1])}, изменились {len(keys)}')
    for name, runs in cols:
        t = totals(runs, keys)
        print(f"  {name:7s} счёт {t['score']:6.2f}  возвратов {t['returned']:3d}  образцов {t['samples']:4d}  "
              f"столкновений {t['collisions']:3d}  штрафов зон {t['hazard_hits']:3d}")
    if args.now:
        same = sum(all(cols[0][1][k][m] == cols[2][1][k][m] for m in KEYS) for k in cols[0][1] if k in cols[2][1])
        print(f'  теперь совпадает с «до» в {same} прогонах из {len(cols[0][1])}')
        rest = [k for k in changed(cols[0][1], cols[2][1])]
        t0, t2 = totals(cols[0][1], rest), totals(cols[2][1], rest)
        print(f"  отличаются от «до» теперь: {len(rest)}; в них счёт {t0['score']:.2f} → {t2['score']:.2f}, "
              f"возвратов {t0['returned']} → {t2['returned']}")
        if args.list:
            keys = sorted(set(keys) | set(rest))
    if args.list:
        for k in keys:
            if args.worse and cols[-1][1][k]['score'] >= cols[0][1][k]['score']:
                continue
            cells = ['{score:6.1f} обр {samples_collected} возв {r} столк {collisions} зон {hazard_hits}'.format(
                r='да ' if runs[k]['returned'] else 'нет', **runs[k]) for _, runs in cols]
            print('  ' + ' '.join(str(x) for x in k) + ': ' + '  →  '.join(cells))


if __name__ == '__main__':
    main()
