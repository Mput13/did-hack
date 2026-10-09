#!/usr/bin/env python3
"""L5, часть 4: простые правки правила выбора цели — оценка на банке и на отладочных сценариях.

    ./px python tools/l5_fix.py bank          # что дала бы каждая правка в состояниях банка (цены — из runs/L5)
    ./px python tools/l5_fix.py dev           # парно с adaptive_v2 на отладочных сценариях 1–80, hard и medium

Правки не видят ни цен вариантов, ни будущего: только сводку состояния, как и само правило. Итог обеих команд
дописывается в research/findings/L5-fix.json. Приёмка выбранной правки — опыт L5b (experiments/L5b.yaml).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l3_common import RUNS, read_json, write_json     # noqa: E402
from l5_bank import FINDINGS, boot, fmt, load         # noqa: E402

from did import hindsight as hs                       # noqa: E402
from did.planner import HeuristicPlanner              # noqa: E402
from did.runner import run_episode                    # noqa: E402

OUT = FINDINGS / 'L5-fix.json'
DEV = {'w=0.25': {'plan_home_weight': 0.25}, 'w=0.5': {'plan_home_weight': 0.5}, 'w=1.0': {'plan_home_weight': 1.0},
       'p=2': {'plan_conf_power': 2.0}, 'p=4': {'plan_conf_power': 4.0}}


def _feasible(state):
    return ([c for c in state['candidates'] if c['feasible']], [p for p in state['explore_points'] if p['feasible']])


def _planner(**kw):
    return lambda state: hs.option_key(HeuristicPlanner(**kw).plan(state)['subgoals'][0])


def _weak_to_explore(limit):
    """Лучший по правилу кандидат слабее limit, а точки разведки есть — разведка (близко к adaptive_picky из A1)."""
    def choose(state):
        cands, points = _feasible(state)
        if cands and points and max(cands, key=lambda c: c['confidence'] / (c['cost_to'] + 0.5))['confidence'] < limit:
            return 'explore:' + max(points, key=lambda p: p['unseen_share'] / (p['cost_to'] + 1.0))['id']
        return _planner()(state)
    return choose


def _home_if(slack):
    """Запас сверх порога возврата (заряд − 1,1 · дорога домой − 4) не больше slack — домой."""
    def choose(state):
        return 'return_base' if state['battery'] - 1.1 * state['return_cost'] - 4.0 <= slack else _planner()(state)
    return choose


def _explore_by(key):
    def choose(state):
        cands, points = _feasible(state)
        return 'explore:' + max(points, key=key)['id'] if points and not cands else _planner()(state)
    return choose


def _other(state):
    """Для масштаба: любая другая годная цель (первая по списку, не выбранная правилом)."""
    rule = _planner()(state)
    rest = [hs.option_key(o) for o in hs.options(state) if hs.option_key(o) not in (rule, 'return_base')]
    return rest[0] if rest else rule


BANK = {
    'цель дальше от базы дороже, вес 0,25 (adaptive_v5)': _planner(home_weight=0.25),
    'цель дальше от базы дороже, вес 0,5': _planner(home_weight=0.5),
    'цель дальше от базы дороже, вес 1,0': _planner(home_weight=1.0),
    'уверенность в квадрате': _planner(conf_power=2.0),
    'уверенность в четвёртой степени': _planner(conf_power=4.0),
    'кандидат слабее 0,5 — разведка': _weak_to_explore(0.5),
    'кандидат слабее 0,4 — разведка': _weak_to_explore(0.4),
    'запас до 4 ед. — домой': _home_if(4.0),
    'запас до 6 ед. — домой': _home_if(6.0),
    'заряд до 20 ед. — домой': lambda s: 'return_base' if s['battery'] <= 20.0 else _planner()(s),
    'разведка: самая непроверенная точка': _explore_by(lambda p: p['unseen_share']),
    'разведка: самая дешёвая точка': _explore_by(lambda p: -p['cost_to']),
    'для масштаба: другая годная цель': _other,
}


def cmd_bank(args):
    _, rows = load(RUNS / 'L5', with_models=False)
    out = {}
    for name, choose in BANK.items():
        picks = [choose(r['state']) for r in rows]
        d = np.array([r['table'][p].mean() - r['table'][r['rule']].mean() for r, p in zip(rows, picks)])
        differs = np.array([p != r['rule'] for r, p in zip(rows, picks)])
        out[name] = {'differs': int(differs.sum()), 'n': len(rows), 'minus_rule': boot(d),
                     'better': int((d > 1.0).sum()), 'worse': int((d < -1.0).sum())}
        print(f"{name:52s} иначе в {differs.sum():2d} из {len(rows)}; цена − правило {fmt(out[name]['minus_rule'])}; "
              f"лучше {out[name]['better']}, хуже {out[name]['worse']}")
    write_json(OUT, {**read_json(OUT, {}), 'bank': out})


def _run(cell):
    level, seed, config = cell
    m = run_episode(level, seed, hs.AGENT, config=config, save=False)['metrics']
    return m['score'], bool(m['returned']), m['samples_collected']


def cmd_dev(args):
    out = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for level in ('hard', 'medium'):
            cells = lambda config: [(level, s, config) for s in range(1, 81)]      # noqa: E731
            base = list(pool.map(_run, cells({}), chunksize=4))
            score = np.array([r[0] for r in base])
            out[level] = {'adaptive_v2': {'score': round(float(score.mean()), 2), 'returned': sum(r[1] for r in base)}}
            print(f"{level}: adaptive_v2 счёт {score.mean():.2f}, возврат {sum(r[1] for r in base)} из 80", flush=True)
            for name, config in DEV.items():
                res = list(pool.map(_run, cells(config), chunksize=4))
                d = np.array([r[0] for r in res]) - score
                out[level][name] = {'score': round(float(np.mean([r[0] for r in res])), 2),
                                    'returned': sum(r[1] for r in res), 'minus_v2': boot(d),
                                    'higher': int((d > 0.5).sum()), 'lower': int((d < -0.5).sum())}
                print(f"  {name:7s} счёт {out[level][name]['score']:.2f}, возврат {out[level][name]['returned']} из 80, "
                      f"− v2 {fmt(out[level][name]['minus_v2'])}, выше {out[level][name]['higher']}, "
                      f"ниже {out[level][name]['lower']}", flush=True)
    write_json(OUT, {**read_json(OUT, {}), 'dev': out})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['bank', 'dev'])
    args = ap.parse_args()
    {'bank': cmd_bank, 'dev': cmd_dev}[args.what](args)


if __name__ == '__main__':
    raise SystemExit(main())
