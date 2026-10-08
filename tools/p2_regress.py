"""Прежние варианты агента дают прежние числа: сверка прогонов этой ветки со сводками опытов основного каталога.

    ./px python tools/p2_regress.py            # 20 прогонов E1 и 20 прогонов adaptive_v2 из E22b

Сравниваются счёт, образцы, возврат, остаток заряда, путь, время и штрафы — до последней цифры сводки.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.runner import run_episode      # noqa: E402

MAIN = Path('/Users/a/MAI/DID/runs')
KEYS = ('score', 'samples_collected', 'returned', 'battery_left', 'distance', 'time', 'hazard_hits',
        'false_collects', 'collisions')
# (опыт, вариант, условие, уровень, сценарии)
CHECKS = [('E1', 'adaptive', 'base', 'hard', range(1001, 1011)), ('E1', 'adaptive', 'base', 'medium', range(1001, 1006)),
          ('E1', 'fixed', 'base', 'hard', range(1001, 1006)),
          ('E22b', 'adaptive_v2', 'base', 'hard', range(8001, 8011)),
          ('E22b', 'adaptive_v2', 'science', 'hard', range(8001, 8006)),
          ('E22b', 'scientist_v2', 'science', 'hard', range(8001, 8006))]


def main():
    bad = total = 0
    for exp, arm, cond, level, seeds in CHECKS:
        summary = json.loads((MAIN / exp / 'summary.json').read_text(encoding='utf-8'))
        ref = {r['seed']: r['metrics'] for r in summary['runs']
               if r['arm'] == arm and r.get('condition', 'base') == cond and r['level'] == level}
        for seed in seeds:
            m = run_episode(level, seed, arm, rules='science' if cond == 'science' else None, save=False)['metrics']
            diff = {k: (ref[seed].get(k), m.get(k)) for k in KEYS if ref[seed].get(k) != m.get(k)}
            total += 1
            if diff:
                bad += 1
                print(f'РАСХОЖДЕНИЕ {exp} {arm}@{cond} {level}-{seed}: {diff}')
    print(f'сверено прогонов: {total}, расхождений: {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
