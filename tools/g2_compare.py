#!/usr/bin/env python3
"""Таблица «до и после» для исследования G2: записи Gazebo прежнего агента и нового.

    ./px python tools/g2_compare.py --before /Users/a/MAI/DID/runs/E7/gazebo --after runs/g2_all/adaptive \\
        --runs hard-1 hard-2 ... --extra hard-5=runs/g2_r1/adaptive/hard-5.json.gz

По каждой записи: собрано, возврат, штрафы, столкновения, наибольшая поправка позы (м и градусы), сколько раз
агент терял положение, замечал проскальзывание и останавливался перед преградой, доля опоздавших тактов. Если рядом с
записью лежит журнал поз (pose-<прогон>.csv), добавляется ошибка позы агента против истины Gazebo.
"""
import argparse
import contextlib
import gzip
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def row(path):
    path = Path(path)
    if not path.exists():
        return None
    d = json.load(gzip.open(path))
    r, fix = d['result'], d.get('pose_fix') or [{'dx': 0, 'dy': 0, 'dth': 0}]
    t = d['track']['t']
    gaps = [b - a for a, b in zip(t, t[1:])]
    tags = [(e.get('data') or {}).get('tag') for e in d['journal']]
    out = {'samples': f"{r['samples_collected']}/{r['samples_total']}", 'returned': 'да' if r['returned'] else 'НЕТ',
           'penalties': r['penalties'], 'collisions': r['collisions'], 'score': r['score'], 'time': round(r['t']),
           'fix_m': max(math.hypot(f['dx'], f['dy']) for f in fix),
           'fix_deg': max(abs(math.degrees(f['dth'])) for f in fix),
           'lost': tags.count('lost'), 'slip': tags.count('slip'), 'blocked': tags.count('blocked'),
           'lag': sum(1 for g in gaps if g > 0.45) / max(1, len(gaps)), 'err': None}
    pose = path.parent.parent / f'pose-{path.name.split(".")[0]}.csv'
    if pose.exists() and pose.stat().st_size > 20000:
        from gz_pose_log import report
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                s = report(str(pose), str(path), out=io.StringIO())
                out['err'] = (s['agent']['median'] * 100, s['agent']['max'] * 100, math.degrees(s['agent_th']['max']))
            except Exception:      # noqa: BLE001 — журнал поз оборвался: таблица строится без него
                pass
    return out


def cell(r):
    if r is None:
        return '— | — | — | — | —'
    return (f"{r['samples']} | {r['returned']} | {r['penalties']} | {r['collisions']} | "
            f"{r['fix_m']:.2f} м, {r['fix_deg']:.0f}°")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--extra', nargs='*', default=[], help='прогон=путь: взять запись «после» из другого места')
    args = ap.parse_args()
    extra = dict(e.split('=', 1) for e in args.extra)
    print('| Сценарий | До: собрано | вернулся | штрафов | столкн. | наиб. поправка | После: собрано | вернулся | штрафов '
          '| столкн. | наиб. поправка | потерь / проскальз. / стоп перед преградой | ошибка позы: медиана, наиб., курс |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    total = {'b': [0, 0, 0, 0, 0], 'a': [0, 0, 0, 0, 0]}
    for name in args.runs:
        b = row(Path(args.before) / f'{name}.json.gz')
        a = row(extra.get(name) or Path(args.after) / f'{name}.json.gz')
        for key, r in (('b', b), ('a', a)):
            if r:
                got, of = map(int, r['samples'].split('/'))
                total[key] = [x + y for x, y in zip(total[key], (got, of, r['returned'] == 'да', r['penalties'], r['collisions']))]
        notes = f"{a['lost']} / {a['slip']} / {a['blocked']}" if a else '—'
        err = '—' if not a or not a['err'] else f"{a['err'][0]:.1f} см, {a['err'][1]:.1f} см, {a['err'][2]:.0f}°"
        lag = f" (опозд. тактов {a['lag']:.1%})" if a and a['lag'] > 0.03 else ''
        print(f'| {name}{lag} | {cell(b)} | {cell(a)} | {notes} | {err} |')
    n = len(args.runs)
    print(f"| **всего** | {total['b'][0]}/{total['b'][1]} | {total['b'][2]} из {n} | {total['b'][3]} | {total['b'][4]} | | "
          f"{total['a'][0]}/{total['a'][1]} | {total['a'][2]} из {n} | {total['a'][3]} | {total['a'][4]} | | | |")


if __name__ == '__main__':
    main()
