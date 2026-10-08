"""A1: отладочные серии на сценариях 1–80 (подбор параметров). Итоговый опыт — experiments/E23.yaml.

    ./px python tools/a1_dev.py --level hard --seeds 1-40 adaptive adaptive_tour 'adaptive_tour:{"scheme_opts":{"explore_value":0.7}}'

Вариант — имя агента, по желанию с поправками настроек после двоеточия (JSON). Первый вариант — база для
парных разностей. Записи прогонов не сохраняются.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.runner import run_episode  # noqa: E402


def job(args):
    label, agent, config, level, seed, rules = args
    try:
        s = run_episode(level, seed, agent, config=config, rules=rules, save=False)
        return label, f'{level}-{seed}', s['metrics'], s['wall_s']
    except Exception as exc:  # noqa: BLE001
        import traceback
        return label, f'{level}-{seed}', {'error': traceback.format_exc()}, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--level', default='hard,medium')
    ap.add_argument('--seeds', default='1-80')
    ap.add_argument('--rules', default=None)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--dump', default=None)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.seeds.split('-'))
    assert hi <= 80, 'подбор — только на сценариях 1–80'
    arms = []
    for spec in args.arms:
        name, _, cfg = spec.partition(':')
        arms.append((spec, name, json.loads(cfg) if cfg else None))
    tasks = [(label, agent, cfg, level, seed, args.rules) for label, agent, cfg in arms
             for level in args.level.split(',') for seed in range(lo, hi + 1)]
    res = {label: {} for label, _, _ in arms}
    wall = {label: 0.0 for label, _, _ in arms}
    with ProcessPoolExecutor(max_workers=min(args.jobs, 3)) as pool:
        for label, seed, m, w in pool.map(job, tasks, chunksize=2):
            if 'error' in m:
                print(label, seed, m['error'])
                continue
            res[label][seed] = m
            wall[label] += w
    rng = np.random.default_rng(0)
    base = arms[0][0]
    print(f"{'вариант':60s} счёт  образцы возврат штрафы остаток  с/прогон  разность со счётом базы [95%]; по уровням")
    for label, _, _ in arms:
        r = res[label]
        if not r:
            continue
        by = ' '.join(f"{lv}: {np.mean([r[k]['score'] - res[base][k]['score'] for k in r if k.startswith(lv) and k in res[base]]):+.1f}"
                      for lv in args.level.split(','))
        v = lambda k: np.mean([float(m[k]) for m in r.values()])  # noqa: E731
        keys = sorted(set(r) & set(res[base]))
        d = np.array([r[k]['score'] - res[base][k]['score'] for k in keys])
        boot = d[rng.integers(len(d), size=(4000, len(d)))].mean(axis=1)
        print(f"{label[:60]:60s} {v('score'):5.1f} {v('samples_share'):6.2f} {v('returned'):6.2f} {v('penalties'):6.2f} "
              f"{v('battery_left'):6.1f} {wall[label] / len(r):8.1f}  {d.mean():+5.1f} [{np.percentile(boot, 2.5):+.1f}; "
              f"{np.percentile(boot, 97.5):+.1f}]  {by}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
