"""Черновая серия для отладки: несколько агентов на одних сценариях, сводка по уровням."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from did.runner import run_episode


def job(a):
    agent, level, seed, kw = a
    try:
        return run_episode(level, seed, agent, experiment='dev', save=kw.get('save', False),
                           rules=kw.get('rules'), llm=kw.get('llm'))
    except Exception as e:      # noqa: BLE001
        import traceback
        return {'error': traceback.format_exc(), 'arm': agent, 'level': level, 'seed': seed}


def main(agents, n=20, levels=('easy', 'medium', 'hard'), **kw):
    jobs = [(a, lv, s, kw) for a in agents for lv in levels for s in range(1, n + 1)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(job, jobs, chunksize=4))
    errs = [r for r in res if 'error' in r]
    for e in errs[:3]:
        print(e['arm'], e['level'], e['seed'], e['error'])
    print(f'ошибок {len(errs)} из {len(res)}')
    keys = ('score', 'samples_share', 'returned', 'battery_used', 'distance', 'time', 'false_collects',
            'hazard_hits', 'collisions')
    print(f"{'агент':18}{'уровень':8}" + ''.join(f'{k[:11]:>12}' for k in keys) + '   refuted  причины')
    for a in agents:
        for lv in levels:
            rs = [r for r in res if 'error' not in r and r['arm'] == a and r['level'] == lv]
            if not rs:
                continue
            row = ''.join(f"{np.mean([float(r['metrics'][k]) for r in rs]):12.2f}" for k in keys)
            ref = np.mean([r['metrics']['hypotheses']['refuted'] for r in rs])
            reasons = {}
            for r in rs:
                reasons[r['metrics']['reason']] = reasons.get(r['metrics']['reason'], 0) + 1
            print(f'{a:18}{lv:8}{row}   {ref:5.1f}  {reasons}')
    return res


if __name__ == '__main__':
    main(sys.argv[1].split(','), n=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
