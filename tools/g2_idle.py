#!/usr/bin/env python3
"""Простои в записях прогонов (исследование G2): наибольшее время без движения и чем робот был занят.

    ./px python tools/g2_idle.py runs/g2_fix/adaptive/*.json.gz

«Без движения» — поза в записи не сдвинулась на 5 см и не повернулась на 0,3 рад. Рядом печатаются итог
прогона, записи журнала о потере положения и сроках (метки lost, relocated, blind, unsure_timeout, idle,
slip, blocked) и доля опоздавших тактов: при доле выше 3% машина была перегружена и прогон не годится.
"""
import gzip
import json
import math
import sys
from pathlib import Path

TAGS = ('collision', 'lost', 'relocated', 'blind', 'unsure_timeout', 'idle', 'slip', 'blocked')


def idle(track, modes):
    """Наибольший простой: (длительность, с какого времени, режим)."""
    t, x, y, th, mode = (track[k] for k in ('t', 'x', 'y', 'th', 'mode'))
    best, mark = (0.0, 0.0, None), 0
    for k in range(len(t)):
        if math.hypot(x[k] - x[mark], y[k] - y[mark]) > 0.05 or abs(math.remainder(th[k] - th[mark], 2 * math.pi)) > 0.3:
            mark = k
        if t[k] - t[mark] > best[0]:
            best = (t[k] - t[mark], t[mark], modes[mode[k]] if modes else mode[k])
    return best


def main():
    print('| Прогон | Собрано | Вернулся | Столкн. | Штрафов | Время, с | Наиб. простой, с | когда и в каком режиме | Журнал | Опоздавших тактов |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for path in sys.argv[1:]:
        tr = json.load(gzip.open(path))
        r, track = tr['result'], tr['track']
        gaps = [b - a for a, b in zip(track['t'], track['t'][1:])]
        lag = sum(g > 0.45 for g in gaps) / max(1, len(gaps))
        dur, since, mode = idle(track, tr.get('modes'))
        tags = [e.get('data', {}).get('tag') for e in tr['journal']]
        notes = ', '.join(f'{k}×{tags.count(k)}' for k in TAGS if k in tags) or '—'
        name = f'{Path(path).parent.parent.name}/{Path(path).name.split(".")[0]}'
        print(f"| {name} | {r['samples_collected']}/{r['samples_total']} | {'да' if r['returned'] else '**нет**'} | "
              f"{r['collisions']} | {r['penalties']} | {r['time']:.0f} | {dur:.1f} | с {since:.0f}-й с, {mode} | {notes} | {lag:.1%} |")


if __name__ == '__main__':
    main()
