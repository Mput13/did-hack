"""A1: на что уходят путь и заряд — по режимам агента (подъезд к кандидату, разведка, возврат) и поводам пересчёта плана.

    ./px python tools/a1_modes.py adaptive oracle_samples oracle_env oracle_all

Уровень hard, отладочные сценарии 1–40, базовые правила. Записи прогонов кладутся в runs/_a1dbg (в git не идут).
m_* — метры пути, e_* — единицы заряда, s_* — секунды в режиме; trig_* — сколько раз за прогон план пересчитан
по этому поводу (candidate_lost — подъезд к месту, где образца не оказалось); всё — в среднем на прогон.
"""
import collections
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
from did.runner import run_episode, RUNS
from did.recorder import load_trace
def job(a):
    agent, level, seed = a
    s = run_episode(level, seed, agent, experiment='_a1dbg', arm=agent)
    tr = load_trace(RUNS / s['file'])
    return agent, seed, s['metrics'], tr['track'], tr['plans'], tr['modes']
if __name__ == '__main__':
    agents = sys.argv[1:]
    tasks = [(a,'hard',s) for a in agents for s in range(1,41)]
    agg = {a: collections.Counter() for a in agents}
    with ProcessPoolExecutor(3) as pool:
        for agent, seed, m, track, plans, tr_modes in pool.map(job, tasks):
            c = agg[agent]
            modes = tr_modes
            for i in range(1, len(track['t'])):
                mo = modes[track['mode'][i]]
                c['m_'+mo] += math.hypot(track['x'][i]-track['x'][i-1], track['y'][i]-track['y'][i-1])
                c['e_'+mo] += track['battery'][i-1]-track['battery'][i]
                c['s_'+mo] += track['t'][i]-track['t'][i-1]
            c['n'] += 1
            for p in plans:
                c['trig_'+p['trigger']] += 1
            c['plans'] += len(plans)
            c['left'] += m['battery_left']; c['samples'] += m['samples_collected']
    for a, c in agg.items():
        n = c['n']
        print(a, {k: round(v/n, 2) for k, v in sorted(c.items()) if k != 'n'})
