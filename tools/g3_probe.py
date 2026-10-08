"""Разбор одного прогона опыта: что сделала защита G2 и с какого момента прогон разошёлся с guard=False (G3).

    ./px python tools/g3_probe.py E15 adaptive c2_2 hard 1024
    ./px python tools/g3_probe.py E10 scientist base hard 1036 --arms off on no_idle

Прогон берётся из описания опыта (вариант, условие, уровень, номер сценария), запись не сохраняется.
Руки: off — guard=False; on — как есть; остальные — как есть, но с одной выключенной частью защиты
(перечень в ABLATE). Для каждой руки: итог, журнал защиты, момент расхождения с рукой off.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from did import agent as agent_mod, runner                       # noqa: E402
from did.config import SCIENCE                                   # noqa: E402
from did.experiments import load_spec                            # noqa: E402
from did.localize import PoseTracker                             # noqa: E402

GUARD_TAGS = ('collision', 'hazard', 'lost', 'blind', 'relocated', 'unsure_timeout', 'slip', 'idle', 'blocked',
              'cautious_off')
# Что выключить: (класс, поле, значение)
ABLATE = {
    'no_idle': [(agent_mod.Agent, 'IDLE_MAX_S', 1e9)],
    'no_alert': [(agent_mod.Agent, 'ALERT_S', 0.0)],
    'no_slip': [(agent_mod.Agent, 'SLIP_STOP', 1e9)],
    'no_unsure': [(agent_mod.Agent, 'UNSURE_MAX_S', 0.0)],
    'no_verify': [(PoseTracker, 'MIN_AGREE', 0.0)],
}


def run(exp, arm_id, cond_id, level, seed, config=None, patches=()):
    spec = load_spec(exp)
    arm = next(a for a in spec['arms'] if a['id'] == arm_id)
    cond = next(c for c in spec['conditions'] if c['id'] == cond_id)
    base_rules = dict(SCIENCE) if spec.get('rules') == 'science' else dict(spec.get('rules') or {})
    rules = {**base_rules, **cond.get('rules', {})}
    agent_rules = cond.get('agent_rules', spec.get('agent_rules'))
    log, bots = [], []

    class Probe(agent_mod.Agent):
        def tick(self, obs, io):
            if not bots:
                bots.append(self)
            super().tick(obs, io)
            log.append((round(obs.t, 2), io.x, io.y, io.th, self.mode, self._last_cmd, io.judge.battery))

    saved = [(c, k, getattr(c, k)) for c, k, _ in patches]
    real = runner.Agent
    runner.Agent = Probe
    try:
        for c, k, v in patches:
            setattr(c, k, v)
        s = runner.run_episode(level, seed, arm['agent'], experiment='g3_probe', arm=arm_id,
                               scenario_args={**cond.get('scenario', {})},
                               config={**(arm.get('config') or {}), **(config or {})}, rules=rules,
                               agent_rules=agent_rules,
                               llm={**arm['llm'], **(cond.get('llm') or {})} if arm.get('llm') else None,
                               sim=cond.get('sim'), study=arm.get('study'), save=False)
    finally:
        runner.Agent = real
        for c, k, v in saved:
            setattr(c, k, v)
    return s, log, bots[0]


def diverge(a, b):
    """Первый момент, когда истинные позы двух прогонов разошлись больше чем на сантиметр."""
    for p, q in zip(a, b):
        if math.hypot(p[1] - q[1], p[2] - q[2]) > 0.01 or abs(math.remainder(p[3] - q[3], 2 * math.pi)) > 0.05:
            return p[0]
    return None if len(a) == len(b) else min(a[-1][0], b[-1][0])


def brief(s):
    m = s['metrics']
    return (f"счёт {m['score']:5.1f}  образцов {m['samples_collected']}  возврат {'да ' if m['returned'] else 'нет'}  "
            f"столкн {m['collisions']}  зон {m['hazard_hits']}  заряд {m['battery_left']:5.1f}  t {m['time']:5.1f}  "
            f"конец: {m['reason']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp')
    ap.add_argument('arm')
    ap.add_argument('cond')
    ap.add_argument('level')
    ap.add_argument('seed', type=int)
    ap.add_argument('--arms', nargs='+', default=['off', 'on'], help='off, on, ' + ', '.join(ABLATE))
    ap.add_argument('--journal', action='store_true', help='весь журнал, а не только записи защиты и решения')
    ap.add_argument('--around', type=float, default=None, help='показать такты ±6 с вокруг этого момента')
    args = ap.parse_args()
    ref = None
    for name in args.arms:
        cfg = {'guard': False} if name == 'off' else None
        s, log, bot = run(args.exp, args.arm, args.cond, args.level, args.seed, cfg, ABLATE.get(name, ()))
        if name == 'off':
            ref = log
        t_div = diverge(ref, log) if ref is not None and name != 'off' else None
        print(f'--- {name}: {brief(s)}' + (f'   расходится с off с t={t_div}' if t_div is not None else ''))
        if len(args.arms) > 3 and not args.journal:
            continue
        for e in bot.journal.entries:
            tag = e.get('data', {}).get('tag')
            if args.journal or tag in GUARD_TAGS or 'не движется' in e['text'] \
                    or e['kind'] == 'decision' and 'базу' in e['text']:
                if t_div is None or e['t'] >= t_div - 15 or tag in GUARD_TAGS:
                    print(f"   {e['t']:6.1f} {e['kind']:8s} {e['text'][:150]}")
        if args.around is not None:
            for row in log:
                if abs(row[0] - args.around) <= 6 and round(row[0] * 10) % 5 == 0:
                    print(f'   t={row[0]:6.1f} ({row[1]:+.2f}; {row[2]:+.2f}) th={row[3]:+.2f} {row[4]:9s} '
                          f'v={row[5][0]:+.2f} w={row[5][1]:+.2f} заряд {row[6]:.1f}')


if __name__ == '__main__':
    main()
