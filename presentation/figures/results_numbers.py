#!/usr/bin/env python3
"""Числа для слайда «Реальные результаты: шесть опытов» → presentation/data/results.json.

    pixi run python presentation/figures/results_numbers.py

Числа считаются теми же функциями, что страница опытов (tools/build_showcase.py), из сводок runs/<опыт>/summary.json.
Миссии словами (R13) в основном каталоге не пересчитываются: числа — из таблицы MISSIONS той же страницы.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.build_showcase import MISSIONS, R13_RUNS, RUNS, exp1, exp5, load, pair, stat, summary  # noqa: E402

OUT = ROOT / 'presentation' / 'data' / 'results.json'


def diff(p):
    return {'mean': p['mean'], 'ci': p['ci']}


def main():
    _, ph, ah, fh = exp1()                       # (счёт, доля образцов, возврат, расход) на трудном уровне
    e1 = summary('E1')
    e9 = summary('E9')
    share = {a: stat(e9, a, 'samples_share', level='medium')['mean'] for a in ('adaptive', 'gradient', 'fixed', 'spiral')}
    p1 = json.loads((RUNS / '_showcase' / 'p1_check.json').read_text(encoding='utf-8'))
    e2 = summary('E2')
    _, val = exp5()                              # (доля верных, верных, всего) среди подтверждённых гипотез
    e17 = summary('E17')
    score = {a: float(np.mean([r['metrics']['score'] for r in e17['runs'] if r['arm'] == a]))
             for a in ('adaptive', 'scientist')}       # средний счёт по всем прогонам опыта

    pairs = []
    for g in sorted((RUNS / 'E7' / 'gazebo').glob('*.json.gz')):
        f = RUNS / 'E7' / 'fastsim' / g.name
        if f.exists():
            pairs.append((load(f)['result'], load(g)['result']))
    d = [gz['score'] - fs['score'] for fs, gz in pairs]
    after = [load(p)['result'] for p in sorted((RUNS / 'F1_v2').glob('*/*.json.gz'))]

    # Задания с ограничением (M1–M3): правило против модели, отвечающей на каждый повод.
    done = lambda col: sum(int(m[col].split()[0]) for m in MISSIONS[1:])       # noqa: E731
    total = sum(int(m[2].split()[-1]) for m in MISSIONS[1:])

    out = {
        'e1': {'runs': len(e1['runs']), 'fixed': fh[0], 'adaptive': ah[0], 'diff': diff(ph),
               'samples': [fh[1], ah[1]], 'returned': [fh[2], ah[2]]},
        'e9': {'runs': len(e9['runs']), 'share': share,
               'diff': diff(pair(e9, 'samples_share', 'adaptive', 'gradient', level='medium')[0]),
               'battery': pair(e9, 'battery_used', 'adaptive', 'gradient', level='medium')[0]['mean']},
        'p1': {'runs': 2 * p1['pair']['score']['n'], 'old': p1['means']['adaptive']['score'],
               'new': p1['means']['adaptive_v2']['score'], 'diff': diff(p1['pair']['score'])},
        'e2': {'change': diff(pair(e2, 'score', 'adaptive', 'no_change')[0]),
               'soil': diff(pair(e2, 'score', 'adaptive', 'no_soil')[0])},
        'r13': {'total': total, 'rule': done(2), 'model': done(3), 'model_every': done(4),
                'plain': {'rule': MISSIONS[0][2], 'model': MISSIONS[0][3]}},
        'e17': {'runs': len(e17['runs']),
                'soil': {a: dict(zip(('share', 'ok', 'n'), val[('soil', 'science', a)])) for a in ('adaptive', 'scientist')},
                'score': score},
        'e7': {'pairs': len(pairs), 'close': sum(abs(x) <= 6 for x in d), 'diff': float(np.mean(d)),
               'fast': float(np.mean([fs['score'] for fs, _ in pairs])), 'gazebo': float(np.mean([gz['score'] for _, gz in pairs])),
               'after': {'n': len(after), 'returned': sum(r['returned'] for r in after),
                         'collisions': sum(r['collisions'] for r in after)}},
    }
    # Прогоны, которые идут в роликах на слайдах (presentation/figures/experiment_clips.py): их итоги для подписей.
    res = lambda path: {k: load(path)['result'][k] for k in ('score', 'samples_collected', 'samples_total', 'returned', 'collisions')}  # noqa: E731
    m = R13_RUNS / 'missions'
    out['clips'] = {
        'e1': [res(RUNS / 'E1' / a / 'hard-1026.json.gz') for a in ('fixed', 'adaptive')],
        'e9': [res(RUNS / 'E9' / a / 'medium-1005.json.gz') for a in ('spiral', 'gradient', 'adaptive')],
        'p1': [res(RUNS / '_showcase' / a / 'hard-8024.json.gz') for a in ('adaptive', 'adaptive_v2')],
        'm1': [res(m / a / 'hard-1001.json.gz') for a in ('M1_rule', 'M1_llm')],
        'm3': [res(m / a / 'hard-1002.json.gz') for a in ('M3_rule', 'M3_llm_ask')],
        'e7': [res(RUNS / 'E7' / a / 'hard-4.json.gz') for a in ('fastsim', 'gazebo')],
        'g2': [res(RUNS / 'E7' / 'gazebo' / 'hard-5.json.gz'), res(RUNS / 'F1_v2' / 'adaptive_v2' / 'hard-5.json.gz')],
    }
    out['r13']['missions'] = {mm[0]: {'rule': mm[2], 'model': mm[3], 'model_every': mm[4]} for mm in MISSIONS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
