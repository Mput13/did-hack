"""Причина расхождения с guard=False для каждого прогона, изменившегося между двумя сводками (G3).

    ./px python tools/g3_causes.py E15 /Users/a/MAI/DID/tmp/E15-before-G2.json /Users/a/MAI/DID/runs/E15/summary.json

Каждый изменившийся прогон повторяется с guard=False и как есть; печатается итог обоих, момент, когда истинные
позы разошлись, и какие защиты сработали до этого момента (кто и когда включил осторожную езду, остановки).
Варианты с памятью между прогонами (memory) не разбираются: их прогоны зависят от всех предыдущих.
"""
import argparse
import inspect
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from did import agent as agent_mod                               # noqa: E402
from did.experiments import load_spec                            # noqa: E402
import g3_diff                                                   # noqa: E402
import g3_probe                                                  # noqa: E402

TAGS = ('lost', 'blind', 'unsure_timeout', 'slip', 'idle', 'blocked')


def one(job):
    exp, key = job
    arm, cond, level, seed = key
    calls = []
    real = agent_mod.Agent._alert

    def spy(self, t, *a, **kw):
        if self.cfg.guard:
            calls.append((round(t, 1), inspect.stack()[1].function))
        return real(self, t, *a, **kw)

    off, log_off, _ = g3_probe.run(exp, arm, cond, level, seed, {'guard': False})
    agent_mod.Agent._alert = spy
    try:
        on, log_on, bot = g3_probe.run(exp, arm, cond, level, seed)
    finally:
        agent_mod.Agent._alert = real
    t_div = g3_probe.diverge(log_off, log_on)
    seen = []
    for t, who in calls:
        if (t_div is None or t <= t_div + 0.5) and (not seen or seen[-1][1] != who or t - seen[-1][0] > 5.0):
            seen.append((t, who))
    seen += [(e['t'], e['data']['tag']) for e in bot.journal.entries
             if e.get('data', {}).get('tag') in TAGS and (t_div is None or e['t'] <= t_div + 0.5)]
    waits = sum(1 for r in log_on if r[4] == 'lost') * 0.1
    return key, g3_probe.brief(off), g3_probe.brief(on), t_div, sorted(set(seen)), waits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp')
    ap.add_argument('before')
    ap.add_argument('after')
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    a, b = g3_diff.load(args.before), g3_diff.load(args.after)
    memory = {arm['id'] for arm in load_spec(args.exp)['arms'] if arm.get('memory')}
    jobs = [(args.exp, k) for k in g3_diff.changed(a, b) if k[0] not in memory]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for key, off, on, t_div, seen, waits in pool.map(one, jobs):
            print(' '.join(str(x) for x in key))
            print(f'   без защиты: {off}\n   с защитой:  {on}')
            print(f'   расходятся с t={t_div}; до этого: ' + (', '.join(f'{w}@{t}' for t, w in seen[-6:]) or '—')
                  + (f'; в потере {waits:.0f} с' if waits else ''))


if __name__ == '__main__':
    main()
