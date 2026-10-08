"""Разбор одного прогона по записи: что агент знал, куда ехал и на чём потерял очки.

    ./px python tools/p2_inspect.py runs/P2/adaptive_v2@base/hard-17.json.gz
    ./px python tools/p2_inspect.py runs/P2/adaptive_v2@base/hard-17.json.gz --brief

Печатает правду сценария (образцы, зоны, события), ход решений агента с зарядом и местом, штрафы и —
для каждого несобранного образца — как менялась уверенность агента рядом с ним и как близко он проезжал.
Инструмент читает скрытую правду из записи; агенту она недоступна.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.recorder import decode_grid, load_trace      # noqa: E402


def conf_series(tr, point, radius=0.35):
    """[(t, уверенность агента, что в круге radius у точки лежит образец)] по снимкам карты."""
    b = tr.get('belief')
    if not b:
        return []
    ys = b['y0'] + (np.arange(b['h']) + 0.5) * b['res']
    xs = b['x0'] + (np.arange(b['w']) + 0.5) * b['res']
    m = np.hypot(xs[None, :] - point[0], ys[:, None] - point[1]) <= radius
    out = []
    for snap in b['snaps']:
        p = (decode_grid(snap['data'], b['h'], b['w']).astype(float) / 255.0) ** 2
        out.append((snap['t'], 1.0 - math.exp(-float(p[m].sum()))))
    return out


def at(tr, t):
    k = min(int(np.searchsorted(tr['track']['t'], t)), len(tr['track']['t']) - 1)
    return tr['track']['x'][k], tr['track']['y'][k], tr['track']['battery'][k]


def describe(tr, brief=False):
    sc, res = tr['scenario'], tr['result']
    out = [f"== {tr['id']}: счёт {res['score']}, собрано {res['samples_collected']}/{res['samples_total']}, "
           f"возврат {res['returned']} ({res['reason']}), заряд {res['battery']:.1f}, путь {res['distance']:.1f} м, "
           f"{res['t']:.0f} с; штрафы: зоны {res['hazard_hits']}, ложные сборы {res['false_collects']}, "
           f"столкновения {res['collisions']}"]
    got = {e['sample']: e['t'] for e in tr['events'] if e['type'] == 'sample_collected'}
    out.append('образцы: ' + '; '.join(f"#{i} ({p[0]:.2f};{p[1]:.2f})" + (f' взят {got[i]:.0f}с' if i in got else ' НЕТ')
                                       for i, p in enumerate(sc['samples'])))
    out.append('зоны на старте: ' + '; '.join(f"{z['id']} ({z['x']:.2f};{z['y']:.2f}) r={z['r']}" for z in sc['hazards']))
    for ev in sc['events']:
        extra = ''
        if ev['type'] == 'new_hazard':
            z = ev['zone']
            extra = f" ({z['x']:.2f};{z['y']:.2f}) r={z['r']}"
        elif ev['type'] == 'sensor_fault':
            extra = f" {ev.get('kind')} {ev['duration']} с"
        out.append(f"  событие t={ev['t']}: {ev['type']}{extra}")
    for w in tr['world']:
        if w['type'] in ('fault', 'fault_end'):
            out.append(f"  мир t={w['t']}: {w['type']} {w.get('kind')}")
    timeline = [(p['t'], 'план', f"[{p['trigger']}] " + (', '.join(
        s['type'] + (f"({s['x']:.1f};{s['y']:.1f})" if 'x' in s else '') for s in p['subgoals'][:2]))
                 + ('' if brief else ' — ' + p['reasoning'][:150])) for p in tr['plans']]
    timeline += [(e['t'], 'СУДЬЯ', f"{e['type']} ({e.get('x', 0):.2f};{e.get('y', 0):.2f})") for e in tr['events']]
    if not brief:
        timeline += [(e['t'], e['kind'], e['text'][:170]) for e in tr['journal'] if e['kind'] in ('alarm', 'action')]
    for t, kind, text in sorted(timeline, key=lambda r: r[0]):
        x, y, b = at(tr, t)
        out.append(f'  {t:6.1f}с ({x:5.2f};{y:5.2f}) заряд {b:5.1f}  {kind}: {text}')
    tx, ty, tt = (np.array(tr['track'][k]) for k in ('x', 'y', 't'))
    for i, p in enumerate(sc['samples']):
        if i in got:
            continue
        d = np.hypot(tx - p[0], ty - p[1])
        k = int(d.argmin())
        series = conf_series(tr, p)
        peak = max(series, key=lambda s: s[1], default=(0, 0))
        known = [t for t, c in series if c >= 0.35]
        out.append(f"  образец #{i} ({p[0]:.2f};{p[1]:.2f}) не взят: ближе всего {d[k]:.2f} м в t={tt[k]:.0f}с "
                   f"(заряд {tr['track']['battery'][k]:.1f}); уверенность макс {peak[1]:.2f} в t={peak[0]:.0f}с; "
                   f"знал (≥0,35) с {known[0]:.0f} по {known[-1]:.0f}с, снимков {len(known)}" if known else
                   f"  образец #{i} ({p[0]:.2f};{p[1]:.2f}) не взят: ближе всего {d[k]:.2f} м в t={tt[k]:.0f}с "
                   f"(заряд {tr['track']['battery'][k]:.1f}); уверенность макс {peak[1]:.2f} в t={peak[0]:.0f}с; не знал")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--brief', action='store_true', help='только планы и события судьи')
    args = ap.parse_args()
    for f in args.files:
        print(describe(load_trace(f), args.brief))
        print()


if __name__ == '__main__':
    main()
