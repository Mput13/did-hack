"""Причина отличия от main для каждого прогона, который отличается между двумя сводками опыта (G3, круг 2).

    ./px python tools/g3_vs_main.py G2 tmp/main_tree/runs/G2/summary.json tmp/r3/G2.json

Каждый отличающийся прогон повторяется дважды в одном процессе: с агентом из main (git show main:did/agent.py) и
с нынешним. Печатается итог обоих, момент, когда истинные позы разошлись, и записи сторожей в журнале обоих
прогонов до этого момента, которых нет в другом («Робот не движется», штрафы, потеря положения и т. д.).
Варианты с памятью между прогонами не разбираются: их прогоны зависят от всех предыдущих.
"""
import argparse
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from did import agent as agent_mod                               # noqa: E402
from did.experiments import load_spec                            # noqa: E402
import g3_diff                                                   # noqa: E402
import g3_probe                                                  # noqa: E402
from g3_table import STRICT                                      # noqa: E402

_main_agent = None


def main_agent():
    global _main_agent
    if _main_agent is None:
        mod = types.ModuleType('did._agent_main')
        mod.__package__ = 'did'
        sys.modules[mod.__name__] = mod
        exec(subprocess.check_output(['git', 'show', 'main:did/agent.py'], text=True, cwd=ROOT), mod.__dict__)
        _main_agent = mod.Agent
    return _main_agent


def guard_log(bot):
    return [(e['t'], e.get('data', {}).get('tag') or 'stuck') for e in bot.journal.entries
            if e.get('data', {}).get('tag') in g3_probe.GUARD_TAGS or 'не движется' in e['text']]


def one(job):
    exp, key = job
    current = agent_mod.Agent
    out = []
    try:
        for cls in (main_agent(), current):
            agent_mod.Agent = cls
            s, log, bot = g3_probe.run(exp, *key)
            out.append((g3_probe.brief(s), log, guard_log(bot)))
    finally:
        agent_mod.Agent = current
    t_div = g3_probe.diverge(out[0][1], out[1][1])
    lim = 1e9 if t_div is None else t_div + 0.5
    only = [[x for x in a if x[0] <= lim and x not in b] for a, b in ((out[0][2], out[1][2]), (out[1][2], out[0][2]))]
    return key, out[0][0], out[1][0], t_div, only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp')
    ap.add_argument('main')
    ap.add_argument('now')
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    a, b = g3_diff.load(args.main), g3_diff.load(args.now)
    memory = {arm['id'] for arm in load_spec(args.exp)['arms'] if arm.get('memory')}
    keys = sorted(k for k in a.keys() & b.keys() if any(a[k][m] != b[k][m] for m in STRICT))
    print(f'{args.exp}: прогонов {len(a)}, отличаются от main {len(keys)}'
          + (f", из них с памятью между прогонами {sum(k[0] in memory for k in keys)} (не разбираются)" if memory else ''))
    jobs = [(args.exp, k) for k in keys if k[0] not in memory]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for key, old, new, t_div, only in pool.map(one, jobs):
            print(' '.join(str(x) for x in key))
            print(f'   main:   {old}\n   теперь: {new}')
            print(f'   расходятся с t={t_div}; до этого только в main: '
                  + (', '.join(f'{w}@{t:.1f}' for t, w in only[0]) or '—') + '; только теперь: '
                  + (', '.join(f'{w}@{t:.1f}' for t, w in only[1]) or '—'))


if __name__ == '__main__':
    main()
