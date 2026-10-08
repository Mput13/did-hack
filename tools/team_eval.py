"""Таблица опыта с командой роботов по сводке runs/<опыт>/summary.json (исследование M1).

    ./px python tools/team_eval.py E25            # таблица вариантов и парные разности
    ./px python tools/team_eval.py E25 --md       # то же в виде таблиц Markdown для отчёта

Сам опыт считается обычной командой: ./px python -m did.experiments E25 --jobs 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from did.metrics import METRICS, paired                      # noqa: E402
from did.runner import RUNS                                  # noqa: E402

SHOW = ['samples_share', 'mean_sample_time', 't_last_collect', 'time', 'score', 'score_per_robot', 'returned',
        'battery_used', 'penalties', 'robot_contacts', 'same_target_s']
PAIRS = [('team', 'pair_lidar'), ('team', 'pair'), ('team', 'solo'), ('pair_lidar', 'solo'), ('pair', 'solo')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp')
    ap.add_argument('--md', action='store_true')
    ap.add_argument('--pairs', default=None, help='пары вариантов через запятую: team:pair,team:solo')
    args = ap.parse_args()
    s = json.loads((RUNS / args.exp / 'summary.json').read_text(encoding='utf-8'))
    spec, runs = s['spec'], s['runs']
    arms = [a['id'] for a in spec['arms']]
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',')] if args.pairs else \
        [p for p in PAIRS if p[0] in arms and p[1] in arms]
    sep = ' | ' if args.md else '  '
    print(f"{args.exp}: прогонов {len(runs)}, ошибок {len(s['errors'])}, сценарии с {spec['seed_start']}, "
          f"по {s['seeds']} на уровень")
    for e in s['errors'][:5]:
        print('  ошибка:', e)
    for cond in spec['conditions']:
        for level in spec['levels'] + ['all']:
            def pick(arm):
                return [r for r in runs if r['arm'] == arm and r['condition'] == cond['id']
                        and (level == 'all' or r['level'] == level)]
            print(f"\n### {cond.get('label', cond['id'])}, уровень {level}, сценариев {len(pick(arms[0]))}\n")
            head = ['показатель'] + arms
            rows = []
            for m in SHOW:
                row = [f'{METRICS[m][0]}, {METRICS[m][1]}']
                for a in arms:
                    g = next((g for g in s['groups'] if g['arm'] == a and g['condition'] == cond['id']
                              and g['level'] == level), None)
                    st = g and g['stats'].get(m)
                    row.append(f"{st['mean']:.2f} [{st['ci'][0]:.2f}; {st['ci'][1]:.2f}]" if st else '—')
                rows.append(row)
            _table(head, rows, sep, args.md)
            print()
            head = ['разность'] + [f'{a} − {b}' for a, b in pairs]
            rows = []
            rng = np.random.default_rng(0)
            for m in SHOW:
                row = [f'{METRICS[m][0]}, {METRICS[m][1]}']
                for a, b in pairs:
                    p = paired(pick(a), pick(b), m, rng)
                    row.append(f"{p['mean']:+.2f} [{p['ci'][0]:+.2f}; {p['ci'][1]:+.2f}] n={p['n']}" if p else '—')
                rows.append(row)
            _table(head, rows, sep, args.md)


def _table(head, rows, sep, md):
    width = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    def line(r):
        body = sep.join(str(c).ljust(w) for c, w in zip(r, width))
        return f'| {body} |' if md else body
    print(line(head))
    if md:
        print('|' + '|'.join('-' * (w + 2) for w in width) + '|')
    for r in rows:
        print(line(r))


if __name__ == '__main__':
    main()
