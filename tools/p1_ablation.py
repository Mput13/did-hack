"""Проверка правок P1 по одной на отладочных сценариях: несколько вариантов настроек на одних сценариях.

    ./px python tools/p1_ablation.py 'old=adaptive::{}' 'wait=adaptive::{"fault_wait":true}' \
        'both=adaptive::{"fault_wait":true,"straight_paths":true}' --levels=hard --seeds=1-80

Вариант задаётся как имя=агент:правила:настройки — правила пустые (базовые) или science, настройки —
поля AgentConfig в JSON. Первый вариант — точка отсчёта: для остальных печатается парная разность счёта
с 95% интервалом. Результаты складываются в tmp/dev/<имя>.json и при повторном запуске не пересчитываются
(--fresh=1 — пересчитать). Сценарии по умолчанию 1–80: на них разрешено подбирать настройки.
"""
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.runner import run_episode      # noqa: E402

CACHE = ROOT / 'tmp' / 'dev'
KEYS = ('score', 'samples_share', 'returned', 'hazard_hits', 'false_collects', 'distance', 'time', 'battery_left')


def _job(task):
    name, agent, rules, config, level, seed = task
    try:
        m = run_episode(level, seed, agent, experiment='P1dev', arm=name, rules=rules or None, config=config,
                        save=False)['metrics']
        return name, level, seed, {k: m[k] for k in KEYS}
    except Exception:      # noqa: BLE001 — один упавший прогон не должен ронять серию
        return name, level, seed, {'error': traceback.format_exc()}


def main():
    variants = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    first, last = (int(v) for v in opts.get('seeds', '1-80').split('-'))
    levels = opts.get('levels', 'hard').split(',')
    CACHE.mkdir(parents=True, exist_ok=True)

    results, tasks = {}, []
    for variant in variants:
        name, spec = variant.split('=', 1)
        agent, rules, config = spec.split(':', 2)
        path = CACHE / f'{name}.json'
        results[name] = json.loads(path.read_text()) if path.exists() and not opts.get('fresh') else {}
        tasks += [(name, agent, rules, json.loads(config or '{}'), level, seed)
                  for level in levels for seed in range(first, last + 1) if f'{level}-{seed}' not in results[name]]
    with ProcessPoolExecutor(3) as pool:
        for name, level, seed, metrics in pool.map(_job, tasks, chunksize=2):
            if 'error' in metrics:
                print(name, level, seed, metrics['error'])
            else:
                results[name][f'{level}-{seed}'] = metrics
    for name, runs in results.items():
        (CACHE / f'{name}.json').write_text(json.dumps(runs))

    names = list(results)
    for level in levels:
        ids = [f'{level}-{seed}' for seed in range(first, last + 1)]
        print(f'--- {level}, сценарии {first}–{last}')
        print(f"{'':22}" + ''.join(f'{k[:10]:>11}' for k in KEYS) + '   разность счёта [95%]')
        base = np.array([float(results[names[0]][i]['score']) for i in ids])
        for name in names:
            values = {k: np.array([float(results[name][i][k]) for i in ids]) for k in KEYS}
            diff = values['score'] - base
            boot = np.random.default_rng(0).choice(diff, (4000, len(diff))).mean(axis=1)
            print(f'{name:22}' + ''.join(f'{values[k].mean():11.2f}' for k in KEYS)
                  + f'   {diff.mean():+.2f} [{np.percentile(boot, 2.5):+.2f}; {np.percentile(boot, 97.5):+.2f}]')


if __name__ == '__main__':
    main()
