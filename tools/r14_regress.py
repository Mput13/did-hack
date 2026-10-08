"""Прежние варианты дают прежние числа: 20 прогонов E1 и 20 прогонов adaptive_v2 (E22b) против сводок основной ветки.

    ./px python -m tools.r14_regress /Users/a/MAI/DID/runs

Сравниваются все числовые метрики прогона до цифры. Считает в один поток.
"""
import json
import sys
from pathlib import Path

import did
from did.experiments import _job, load_spec


def main(runs_dir):
    print('код:', Path(did.__file__).parent)
    bad = total = 0
    for exp, arm_id, cond_id, picks in (
            ('E1', 'adaptive', 'base', [(lv, s) for lv in ('medium', 'hard') for s in range(1001, 1006)]),
            ('E1', 'fixed', 'base', [(lv, s) for lv in ('medium', 'hard') for s in range(1001, 1006)]),
            ('E22b', 'adaptive_v2', 'base', [(lv, s) for lv in ('medium', 'hard') for s in range(8001, 8011)])):
        ref = json.loads((Path(runs_dir) / exp / 'summary.json').read_text(encoding='utf-8'))
        spec = load_spec(exp)
        arm = next(a for a in spec['arms'] if a['id'] == arm_id)
        cond = next(c for c in spec.get('conditions') or [{'id': 'base'}] if c['id'] == cond_id)
        by = {(r['level'], r['seed']): r['metrics'] for r in ref['runs']
              if r['arm'] == arm_id and r.get('condition', 'base') == cond_id}
        for lv, seed in picks:
            got = _job((f'_regress_{exp}', arm, cond, lv, seed))
            want = by[(lv, seed)]
            diff = {k: (want[k], got['metrics'].get(k)) for k in want
                    if isinstance(want[k], (int, float)) and want[k] != got['metrics'].get(k)} if 'metrics' in got else got
            total += 1
            bad += bool(diff)
            if diff:
                print(f'РАСХОЖДЕНИЕ {exp} {arm_id} {lv}-{seed}: {diff}')
    print(f'прогонов {total}, расхождений {bad}')
    return bad


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1]) else 0)
