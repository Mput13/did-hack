"""Таблицы приёмки G3 целиком: каждый опыт по всем прогонам, а не только по изменившимся после G2.

    ./px python tools/g3_table.py --before /Users/a/MAI/DID/tmp --main tmp/main_tree/runs --now tmp/r3
    ./px python tools/g3_table.py ... --exps E10 E11 --list          # перечислить прогоны, отличающиеся от main

Три набора сводок: «до G2» (файлы E*-before-G2.json), нынешний main (каталог с E*/summary.json) и нынешний
код (файлы E*.json). По каждому опыту: сколько прогонов отличается, переходы возврата в обе стороны, у скольких
счёт ниже и выше, средний счёт по всем прогонам. Отдельно — прогоны, изменившиеся после G2 (до G2 → main).
«Отличается» — как в g3_diff.py (счёт, образцы, возврат, столкновения, штрафы зон); в скобках — если
считать ещё время и путь.
"""
import argparse
from pathlib import Path

from g3_diff import KEYS, changed, load, totals

STRICT = KEYS + ('time', 'distance')
EXPS = ('E10', 'E11', 'E13', 'E14', 'E15', 'E17')


def compare(a, b, keys=None):
    keys = sorted(a.keys() & b.keys()) if keys is None else keys
    diff = [k for k in keys if any(a[k][m] != b[k][m] for m in KEYS)]
    return {'n': len(keys), 'diff': len(diff),
            'strict': sum(any(a[k][m] != b[k][m] for m in STRICT) for k in keys),
            'lost': sum(bool(a[k]['returned']) and not b[k]['returned'] for k in keys),
            'gained': sum(not a[k]['returned'] and bool(b[k]['returned']) for k in keys),
            'lower': sum(b[k]['score'] < a[k]['score'] for k in keys),
            'higher': sum(b[k]['score'] > a[k]['score'] for k in keys)}


def cell(c):
    return f"{c['diff']} ({c['strict']}) | {c['lost']} / {c['gained']} | {c['lower']} / {c['higher']}"


def row(runs, keys):
    t = totals(runs, keys)
    return f"{t['score']:.2f}", t['returned'], t['collisions'], t['hazard_hits']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--before', required=True, help='каталог с E*-before-G2.json')
    ap.add_argument('--main', required=True, help='каталог runs нынешнего main')
    ap.add_argument('--now', required=True, help='каталог с E*.json нынешнего кода')
    ap.add_argument('--exps', nargs='+', default=list(EXPS))
    ap.add_argument('--list', action='store_true', help='перечислить прогоны, отличающиеся от main')
    args = ap.parse_args()
    data = {e: (load(Path(args.before) / f'{e}-before-G2.json'), load(Path(args.main) / e / 'summary.json'),
                load(Path(args.now) / f'{e}.json')) for e in args.exps}

    print('Все прогоны. Ячейка сравнения: отличается (с временем и путём) | возврат потерян / появился | счёт ниже / выше')
    print('| Опыт | Прогонов | теперь против «до G2» | теперь против main | main против «до G2» | Счёт: до G2 / main / теперь '
          '| Возвратов: до G2 / main / теперь |')
    print('|---|---:|---|---|---|---|---|')
    total = [dict.fromkeys(('n', 'diff', 'strict', 'lost', 'gained', 'lower', 'higher'), 0) for _ in range(3)]
    sums = [[0.0, 0] for _ in range(3)]
    for e, (b, m, n) in data.items():
        keys = sorted(b.keys() & m.keys() & n.keys())
        cs = [compare(b, n, keys), compare(m, n, keys), compare(b, m, keys)]
        for t, c in zip(total, cs):
            for k in t:
                t[k] += c[k]
        rows = [row(r, keys) for r in (b, m, n)]
        for s, r, runs in zip(sums, rows, (b, m, n)):
            s[0] += sum(runs[k]['score'] for k in keys)
            s[1] += r[1]
        print(f"| {e} | {len(keys)} | {cell(cs[0])} | {cell(cs[1])} | {cell(cs[2])} | "
              f"{' / '.join(r[0] for r in rows)} | {' / '.join(str(r[1]) for r in rows)} |")
    n_all = total[0]['n']
    print(f"| **все** | {n_all} | {cell(total[0])} | {cell(total[1])} | {cell(total[2])} | "
          f"{' / '.join(f'{s[0] / n_all:.2f}' for s in sums)} | {' / '.join(str(s[1]) for s in sums)} |")

    print('\nПрогоны, изменившиеся после G2 (до G2 → main)')
    print('| Опыт | Изменились | Возвратов: до G2 / main / теперь | Счёт: до G2 / main / теперь | Столкновений | Штрафов зон '
          '| Теперь совпадают с «до G2» | с main |')
    print('|---|---:|---|---|---|---|---:|---:|')
    acc = [[0.0, 0, 0, 0] for _ in range(3)]
    n_set = same_b = same_m = 0
    for e, (b, m, n) in data.items():
        keys = [k for k in changed(b, m) if k in n]
        rows = [totals(r, keys) for r in (b, m, n)]
        sb = len(keys) - compare(b, n, keys)['diff']
        sm = len(keys) - compare(m, n, keys)['diff']
        n_set, same_b, same_m = n_set + len(keys), same_b + sb, same_m + sm
        for a, r in zip(acc, rows):
            a[0] += r['score'] * r['n']
            a[1] += r['returned']
            a[2] += r['collisions']
            a[3] += r['hazard_hits']
        print(f"| {e} | {len(keys)} | {' / '.join(str(r['returned']) for r in rows)} | "
              f"{' / '.join(format(r['score'], '.2f') for r in rows)} | {' / '.join(str(r['collisions']) for r in rows)} | "
              f"{' / '.join(str(r['hazard_hits']) for r in rows)} | {sb} | {sm} |")
    print(f"| **все** | {n_set} | {' / '.join(str(a[1]) for a in acc)} | "
          f"{' / '.join(format(a[0] / max(1, n_set), '.2f') for a in acc)} | {' / '.join(str(a[2]) for a in acc)} | "
          f"{' / '.join(str(a[3]) for a in acc)} | {same_b} | {same_m} |")

    if args.list:
        for e, (b, m, n) in data.items():
            print(f'\n{e}: отличаются от main')
            for k in sorted(m.keys() & n.keys()):
                if any(m[k][x] != n[k][x] for x in STRICT):
                    cells = ['{score:6.2f} обр {samples_collected} возв {r} столк {collisions} зон {hazard_hits} t {time:5.1f}'.format(
                        r='да ' if runs[k]['returned'] else 'нет', **runs[k]) for runs in (b, m, n) if k in runs]
                    print('  ' + ' '.join(str(x) for x in k) + ': ' + '  →  '.join(cells))


if __name__ == '__main__':
    main()
