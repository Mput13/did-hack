"""Отладка самокалибровки (R14) на сценариях 1–80: счёт вариантов по условиям и что назвала калибровка.

    ./px python -m tools.r14_dev --seeds 1-20 --arms adaptive,adaptive_cal --conds base,range_15 --jobs 4

Только для подбора параметров: итог считается опытом E20 на отложенных сценариях.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from did.agent import LAW_POWER
from did.config import Rules
from did.runner import run_episode

CONDS = {
    'base': {},
    'drain_x15': {'drain_per_m': 3.75},
    'range_15': {'sensor_range_m': 1.5},
    'range_25': {'sensor_range_m': 2.5},
    'law_quad': {'sensor_law': 'quadratic'},
    'law_sqrt': {'sensor_law': 'sqrt'},
    'range_15_drain_x15': {'sensor_range_m': 1.5, 'drain_per_m': 3.75},
    # дальности не из сетки гипотез и не из опыта E20: проверка, что калибровка не «угадывает узлы»
    'range_17': {'sensor_range_m': 1.7},
    'range_23': {'sensor_range_m': 2.3},
    'range_30_drain_x2': {'sensor_range_m': 3.0, 'drain_per_m': 5.0},
}


def calibration(journal):
    """Что агент назвал: последние закон и расход из журнала и когда закон назван впервые."""
    laws = [e for e in journal if (e.get('data') or {}).get('tag') == 'calibration' and e['data']['what'] == 'law']
    pairs = [e for e in journal if (e.get('data') or {}).get('tag') == 'calibration' and e['data']['what'] == 'pairs']
    drains = [e for e in journal if (e.get('data') or {}).get('tag') == 'calibration' and e['data']['what'] == 'drain']
    out = {'n_law': len(laws), 'law_t': None, 'range': None, 'power': None, 'source': None, 'per_m': None,
           'drain_t': None}
    if laws:
        d = laws[-1]['data']
        out.update(law_t=laws[0]['t'], last_t=laws[-1]['t'], range=d['range'], power=d['power'], source=d['source'])
    if drains:
        out.update(per_m=drains[-1]['data']['per_m'], drain_t=drains[0]['t'])
    return out


def job(args):
    level, seed, arm, cond, tweak = args
    from did import runner
    from did.calibrate import Calibrator
    for k, v in tweak:                      # пробные значения параметров калибровки
        setattr(Calibrator, k, v)
    keep = {}
    orig = runner.make_agent

    def make_agent(name, config=None):
        cls, cfg = orig(name, config)

        class Probe(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                keep['bot'] = self
        return Probe, cfg

    runner.make_agent = make_agent
    try:
        known = arm == 'adaptive_known'
        s = run_episode(level, seed, arm, save=False, rules=CONDS[cond], agent_rules=None if known else {})
    finally:
        runner.make_agent = orig
    m = s['metrics']
    row = {'level': level, 'seed': seed, 'arm': arm, 'cond': cond, 'wall': s['wall_s'],
           **{k: m[k] for k in ('score', 'samples_share', 'returned', 'false_collects', 'battery_used', 'time')}}
    row['cal'] = calibration(keep['bot'].journal.entries)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1-20')
    ap.add_argument('--levels', default='medium,hard')
    ap.add_argument('--arms', default='adaptive,adaptive_cal')
    ap.add_argument('--conds', default=','.join(list(CONDS)[:7]))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--detail', action='store_true')
    ap.add_argument('--set', default='', help='пробные параметры Calibrator: LEAVE=5,BACK=2')
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.seeds.split('-'))
    assert hi <= 80, 'подбор параметров — только на сценариях 1–80'
    tweak = tuple((k, float(v)) for k, v in (kv.split('=') for kv in args.set.split(',') if kv))
    tasks = [(lv, seed, arm, c, tweak) for c in args.conds.split(',') for arm in args.arms.split(',')
             for lv in args.levels.split(',') for seed in range(lo, hi + 1)]
    with ProcessPoolExecutor(max_workers=min(args.jobs, 4)) as pool:
        rows = list(pool.map(job, tasks, chunksize=2))
    for c in args.conds.split(','):
        truth = Rules(**CONDS[c])
        print(f'== {c}: дальность {truth.sensor_range_m}, показатель {LAW_POWER[truth.sensor_law]}, расход {truth.drain_per_m}')
        for arm in args.arms.split(','):
            for lv in args.levels.split(','):
                sel = [r for r in rows if r['cond'] == c and r['arm'] == arm and r['level'] == lv]
                mean = lambda k: float(np.mean([r[k] for r in sel]))
                line = (f"  {arm:15s} {lv:6s} образцы {mean('samples_share'):.3f} возврат {mean('returned'):.2f} "
                        f"счёт {mean('score'):6.1f} ложных {mean('false_collects'):.2f} заряд {mean('battery_used'):.1f} "
                        f"время {mean('time'):.0f} с, расчёт {mean('wall'):.1f} с")
                cal = [r['cal'] for r in sel if r['cal']['range'] is not None]
                if arm == 'adaptive_cal':
                    dr = [r['cal']['per_m'] for r in sel if r['cal']['per_m'] is not None]
                    line += f"\n      закон назван в {len(cal)}/{len(sel)}"
                    if cal:
                        line += (f": дальность {np.median([k['range'] for k in cal]):.2f} "
                                 f"[{np.percentile([k['range'] for k in cal], 10):.2f}; {np.percentile([k['range'] for k in cal], 90):.2f}], "
                                 f"показатель {np.median([k['power'] for k in cal]):.2f} "
                                 f"[{np.percentile([k['power'] for k in cal], 10):.2f}; {np.percentile([k['power'] for k in cal], 90):.2f}], "
                                 f"впервые на {np.median([k['law_t'] for k in cal]):.0f} с, "
                                 f"подтверждено парами {sum(k['source'] == 'pairs' for k in cal)}")
                    if dr:
                        line += f"; расход {np.median(dr):.2f} [{min(dr):.2f}; {max(dr):.2f}]"
                print(line)
        if args.detail:
            for r in rows:
                if r['cond'] == c and r['arm'] == 'adaptive_cal':
                    print('     ', r['level'], r['seed'], r['samples_share'], r['false_collects'], r['cal'])


if __name__ == '__main__':
    main()
