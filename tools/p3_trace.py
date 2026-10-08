"""P3: по тактам восстановить, что агент знал о несобранном образце и почему к нему не поехал.

    ./px python tools/p3_trace.py --agent adaptive_v2 --level hard --seeds 5,17 --samples unclear
    ./px python tools/p3_trace.py --from runs/P3/dev_v2.json --cause sample_unclear --out runs/P3/unclear.json

Прогон повторяется с тем же зерном (он воспроизводим до цифры), а к агенту снаружи приставлен наблюдатель:
он ничего не меняет и пишет каждые полсекунды, что карта образцов говорит о месте настоящего образца, и
каждое решение планировщика — со всеми кандидатами, их ценой и флагом «по заряду». Инструмент читает скрытую
правду (места образцов); агенту она недоступна.

Для каждого образца печатается хронология и ставится одна причина (см. classify).
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did import agent as agent_mod                    # noqa: E402
from did.arena import load_arena                      # noqa: E402
from did.runner import run_episode                    # noqa: E402
from did.scenario import generate                     # noqa: E402

NEAR = 0.50          # кандидат агента «про этот образец», если он не дальше, м
STEP = 0.5           # шаг наблюдения, с

REASONS = {
    'ring': 'место не было выделено: вероятность размазана кольцом, кандидата у агента не было',
    'late': 'кандидат появился только на обратном пути (или после решения о возврате)',
    'vanished': 'кандидат был, но исчез до ближайшего решения (объяснён соседним образцом или новыми показаниями)',
    'unaffordable': 'кандидат был на решении, но по расчёту агента не по заряду',
    'risky': 'кандидат был на решении, но место считалось опасной зоной',
    'other_first': 'кандидат был и был по заряду, планировщик выбрал другую цель, потом кандидат пропал или заряд кончился',
    'assigned': 'подъезд к нему назначался (разбор потерь его не засчитал)',
}


class Watch:
    """Наблюдатель за одним прогоном: карта образцов возле настоящих образцов и решения планировщика."""

    def __init__(self, samples):
        self.samples = [tuple(p) for p in samples]
        self.rows = []           # по времени: {t, x, y, battery, returning, goal, per: [по образцу]}
        self.plans = []          # {t, trigger, candidates, explore_points, chosen}
        self.got = {}            # номер образца → время сбора (по событиям судьи в наблюдении)
        self.reach = {}          # номер образца → [(t, уверенность у робота, режим)]: робот стоял в радиусе сбора
        self.erased = {}         # номер образца → [(t сбора соседа, уверенность до, после)]
        self._next = 0.0

    def before_tick(self, bot, obs):
        self._n = bot.collected
        self._p35 = [bot.belief.prob_within(sx, sy, 0.35) for sx, sy in self.samples]

    def after_tick(self, bot, obs, io):
        for e in obs.events:
            if e.get('type') == 'sample_collected':
                self.got.setdefault(e.get('sample'), obs.t)
        if bot.collected > self._n:          # такт со сбором: что пересчёт карты сделал с остальными местами
            for i, (sx, sy) in enumerate(self.samples):
                self.erased.setdefault(i, []).append(
                    (round(obs.t, 1), round(self._p35[i], 2), round(bot.belief.prob_within(sx, sy, 0.35), 2)))
        for i, (sx, sy) in enumerate(self.samples):
            # Робот в радиусе сбора по правилам (0,3 м) от ещё не собранного образца: сбор здесь удался бы.
            if i not in self.got and math.hypot(obs.x - sx, obs.y - sy) <= 0.30:
                here = bot.belief.prob_within(obs.x, obs.y, bot.cfg.collect_reach)
                self.reach.setdefault(i, []).append((round(obs.t, 1), round(here, 2), bot.mode))
        if obs.t + 1e-9 < self._next:
            return
        self._next = obs.t + STEP
        b = bot.belief
        strong = b.candidates(min_mass=bot.cfg.candidate_mass)
        weak = b.candidates(min_mass=0.2, limit=12)
        field = None
        per = []
        for sx, sy in self.samples:
            row = {'d': round(math.hypot(obs.x - sx, obs.y - sy), 2), 'p35': round(b.prob_within(sx, sy, 0.35), 2),
                   'p25': round(b.prob_within(sx, sy, 0.25), 2)}
            for key, cands in (('cand', strong), ('weak', weak)):
                near = [c for c in cands if math.hypot(c['x'] - sx, c['y'] - sy) <= NEAR]
                if near:
                    c = max(near, key=lambda c: c['mass'])
                    row[key] = [round(c['x'], 2), round(c['y'], 2), round(c['mass'], 2)]
            if 'cand' in row:
                if field is None:
                    field = bot.graph.field(obs.x, obs.y)
                cx, cy = row['cand'][:2]
                to = bot._trip_cost(bot.graph, field[0], field[1], cx, cy)
                back = bot._home_cost(cx, cy)
                ix, iy = bot.arena.w2g(cx, cy)
                row.update(to=round(float(to), 1), back=round(float(back), 1), risk=round(float(bot._risk[iy, ix]), 2),
                           extra=round(float(to + back - bot._home_cost(obs.x, obs.y)), 1),
                           ok=bool(math.isfinite(to) and bot._affordable(obs.battery, to, back)))
            per.append(row)
        goal = bot.queue[0] if bot.queue else None
        self.rows.append({'t': round(obs.t, 1), 'x': round(obs.x, 2), 'y': round(obs.y, 2),
                          'battery': round(obs.battery, 1), 'returning': bool(bot._returning),
                          'goal': [goal['type'], round(goal.get('x', math.nan), 2), round(goal.get('y', math.nan), 2)]
                          if goal else None, 'per': per})

    def on_plan(self, bot, obs, state, subgoals):
        self.plans.append({'t': round(obs.t, 1), 'trigger': state['trigger'], 'battery': state['battery'],
                           'candidates': state['candidates'], 'explore_points': state['explore_points'],
                           'chosen': subgoals[0] if subgoals else None})


def run_watched(agent, level, seed, rules=None):
    """Прогнать сценарий с наблюдателем. Возвращает (сводка прогона, Watch)."""
    scenario = generate(level, seed, load_arena())
    watch = Watch(scenario.samples)
    cls = agent_mod.Agent
    tick, apply_plan = cls.tick, cls._apply_plan

    def tick_w(self, obs, io):
        watch.before_tick(self, obs)
        out = tick(self, obs, io)
        watch.after_tick(self, obs, io)
        return out

    def apply_w(self, obs, plan, state, trigger):
        out = apply_plan(self, obs, plan, state, trigger)
        watch.on_plan(self, obs, state, list(self.queue))
        return out

    cls.tick, cls._apply_plan = tick_w, apply_w
    try:
        summary = run_episode(level, seed, agent, rules=rules, save=False)
    finally:
        cls.tick, cls._apply_plan = tick, apply_plan
    return summary, watch, scenario


def classify(watch, i):
    """Одна причина, почему образец i не собран, и числа к ней."""
    sx, sy = watch.samples[i]
    rows = watch.rows
    back_t = next((r['t'] for r in rows if r['returning']), math.inf)
    with_cand = [r for r in rows if 'cand' in r['per'][i]]
    before = [r for r in with_cand if r['t'] < back_t]
    info = {'peak_p35': max(r['per'][i]['p35'] for r in rows), 'nearest': min(r['per'][i]['d'] for r in rows),
            'cand_s': round(len(before) * STEP, 1), 'cand_s_return': round((len(with_cand) - len(before)) * STEP, 1),
            'first_cand_t': with_cand[0]['t'] if with_cand else None, 'return_t': back_t if back_t < math.inf else None,
            'peak_p35_before': max((r['per'][i]['p35'] for r in rows if r['t'] < back_t), default=0.0)}
    # Сбор соседнего образца стёр это место из карты: до пересчёта уверенность была не ниже 0,35, после — вдвое меньше.
    info['erased'] = [e for e in watch.erased.get(i, []) if e[1] >= 0.35 and e[2] <= 0.5 * e[1]]
    # Робот был в радиусе сбора и карта была достаточно уверена, но сбор не делался (режим не «подъезд»).
    sure = [r for r in watch.reach.get(i, []) if r[1] >= 0.85]
    info['in_reach_s'] = len(watch.reach.get(i, []))
    info['pickup'] = [sure[0], sure[-1]] if sure else []
    # Самый дешёвый крюк к этому месту за прогон, пока место было кандидатом с уверенностью от 0,5:
    # крюк = туда + оттуда домой − отсюда домой, в единицах заряда по расчёту агента.
    hooks = [(r['per'][i]['extra'], r['t'], r['per'][i]['cand'][2], r['battery'], r['returning'], r['per'][i]['ok'])
             for r in with_cand if r['per'][i]['cand'][2] >= 0.5 and r['per'][i]['risk'] < 0.5]
    info['hook'] = min(hooks) if hooks else None
    mine = lambda c: math.hypot(c['x'] - sx, c['y'] - sy) <= NEAR      # noqa: E731
    seen = []                                 # решения, на которых кандидат про этот образец был в списке
    for p in watch.plans:
        c = next((c for c in p['candidates'] if mine(c)), None)
        if c is None:
            continue
        ch = p['chosen'] or {}
        seen.append({'t': p['t'], 'trigger': p['trigger'], 'battery': p['battery'], 'cand': c,
                     'chosen': ch.get('type'), 'chosen_xy': [ch.get('x'), ch.get('y')],
                     'chosen_it': ch.get('type') == 'investigate' and mine(ch)})
    info['plans_with_cand'] = seen
    if any(s['chosen_it'] for s in seen):
        return 'assigned', info
    if not before:
        return ('late' if with_cand else 'ring'), info
    if not seen:
        return 'vanished', info
    if any(s['cand']['feasible'] for s in seen):
        return 'other_first', info
    risky = [r for r in before if r['per'][i].get('risk', 0.0) >= 0.5]
    return ('risky' if len(risky) > len(before) / 2 else 'unaffordable'), info


def timeline(watch, i, every=2.0):
    """Строки хронологии по образцу i: что о нём знала карта и что делал робот."""
    out, last = [], -1e9
    plans = {p['t']: p for p in watch.plans}
    for r in watch.rows:
        q = r['per'][i]
        p = next((plans[t] for t in plans if abs(t - r['t']) < STEP / 2 + 1e-6), None)
        if r['t'] - last < every and p is None:
            continue
        last = r['t']
        cand = (f"кандидат ({q['cand'][0]:.1f};{q['cand'][1]:.1f}) {q['cand'][2]:.2f} туда {q['to']} обратно {q['back']} "
                f"риск {q['risk']} {'по заряду' if q['ok'] else 'НЕ по заряду'}") if 'cand' in q else \
            (f"слабый пик {q['weak'][2]:.2f}" if 'weak' in q else '—')
        goal = f"{r['goal'][0]}({r['goal'][1]:.1f};{r['goal'][2]:.1f})" if r['goal'] else 'нет цели'
        line = (f"  {r['t']:6.1f}с робот ({r['x']:5.2f};{r['y']:5.2f}) заряд {r['battery']:5.1f} до образца {q['d']:.2f} м | "
                f"круг 0,35: {q['p35']:.2f}, круг 0,25: {q['p25']:.2f} | {cand} | {'ДОМОЙ ' if r['returning'] else ''}{goal}")
        if p is not None:
            cs = ', '.join(f"{c['id']}({c['x']:.1f};{c['y']:.1f}) {c['confidence']:.2f}/{c['cost_to']}"
                           f"{'' if c['feasible'] else ' нельзя'}" for c in p['candidates']) or 'кандидатов нет'
            line += f"\n          решение [{p['trigger']}]: {cs} → {(p['chosen'] or {}).get('type')}"
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', default='adaptive_v2')
    ap.add_argument('--level', default='hard')
    ap.add_argument('--rules', default=None, choices=['science'])
    ap.add_argument('--seeds', default=None, help='номера сценариев через запятую')
    ap.add_argument('--samples', default='missed', help='missed — все несобранные; или номера через запятую')
    ap.add_argument('--from', dest='src', default=None, help='разбор потерь (--json из loss_breakdown.py): взять образцы оттуда')
    ap.add_argument('--cause', default='sample_unclear', help='какая статья разбора потерь (для --from)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--quiet', action='store_true', help='без хронологий, только причины')
    args = ap.parse_args()

    todo = {}                                 # зерно → номера образцов или None (все несобранные)
    if args.src:
        dump = json.loads(Path(args.src).read_text(encoding='utf-8'))
        for key, block in dump.items():
            if not key.endswith('/' + args.level):
                continue
            for run in block['runs']:
                ids = [int(i) for i, m in run['detail']['missed'].items() if args.cause in ('any', m['cause'])]
                if ids:
                    todo[run['seed']] = ids
    else:
        for s in args.seeds.split(','):
            todo[int(s)] = None if args.samples == 'missed' else [int(v) for v in args.samples.split(',')]

    result, tally = [], Counter()
    for seed, ids in sorted(todo.items()):
        summary, watch, scenario = run_watched(args.agent, args.level, seed, args.rules)
        m = summary['metrics']
        if ids is None:
            ids = [i for i in range(len(watch.samples)) if i not in watch.got]
        print(f"== {args.level}-{seed}: счёт {m.get('score')}, собрано {len(watch.got)}/{len(watch.samples)}, "
              f"возврат {m.get('returned')}")
        for i in ids:
            if i in watch.got:
                print(f'  образец #{i} в повторном прогоне собран — прогон не воспроизвёлся')
                continue
            reason, info = classify(watch, i)
            tally[reason] += 1
            sx, sy = watch.samples[i]
            print(f"  образец #{i} ({sx:.2f};{sy:.2f}): {REASONS[reason]}\n"
                  f"    пик в круге 0,35 м: {info['peak_p35']:.2f} (до возврата {info['peak_p35_before']:.2f}); "
                  f"ближе всего {info['nearest']:.2f} м; кандидатом был {info['cand_s']} с до возврата и "
                  f"{info['cand_s_return']} с после; возврат с {info['return_t']} с\n"
                  f"    стёрт сбором соседа: {info['erased'] or 'нет'}; в радиусе сбора при уверенности ≥ 0,85 "
                  f"(t, уверенность, режим): {info['pickup'] or 'не был'}; тактов в радиусе сбора: {info['in_reach_s']}")
            tally['  из них: место стёрто сбором соседа'] += bool(info['erased'])
            tally['  из них: был в радиусе сбора с уверенностью ≥ 0,85 и не собрал'] += bool(info['pickup'])
            if not args.quiet:
                print('\n'.join(timeline(watch, i)))
            print(f"    самый дешёвый крюк (ед. заряда, t, уверенность, заряд, уже домой, по заряду): {info['hook']}; "
                  f"привезено заряда {m.get('battery_left')}")
            result.append({'seed': seed, 'sample': i, 'xy': [sx, sy], 'reason': reason, 'score': m.get('score'),
                           'battery_left': m.get('battery_left'), 'returned': m.get('returned'), **info})
    print('\nПричины:')
    for reason, n in tally.most_common():
        print(f'  {n:3d}  {REASONS.get(reason, reason)}')
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
