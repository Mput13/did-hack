"""B1: что пересчёт карты образцов после сбора делает с остальными местами — на одном и том же входе.

    ./px python tools/b1_trace.py --rules science --seeds 1-80 --replays old,exact --out tmp/b1_collects.json
    ./px python tools/b1_trace.py --rules science --seeds 17 --config '{"exact_replay":true}' --journal

Прогон идёт как обычно (поведение агента не меняется), а на каждом сборе наблюдатель берёт копию карты до
пересчёта и считает на ней несколько пересчётов рядом (--replays). Так пересчёты сравниваются на одинаковых
показаниях, а не на разошедшихся прогонах. Для каждого пересчёта пишется: уверенность у каждого оставшегося
настоящего образца, список кандидатов и какие из них ложные (дальше 0,5 м от любого оставшегося образца).
Пересчёты: old — прежний; exact — проба P3; trusted — точный без показаний, которым нельзя верить (маску строит сам
агент по журналу); oracle — точный без показаний, снятых при настоящем сбое (маска из скрытой правды, только для
разбора); gated — пересчёт пресетов *_v5. О каждом показании журнала известно, какой сбой датчика шёл на самом деле и считал ли датчик неисправным сам
агент. Инструмент читает скрытую правду судьи; агенту она недоступна.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did import agent as agent_mod                    # noqa: E402
from did import fastsim                               # noqa: E402
from did.belief import SampleBelief                   # noqa: E402
from did.replay import exact_replay, trusted_replay   # noqa: E402
from did.runner import run_episode                    # noqa: E402

NEAR = 0.50          # кандидат «про настоящий образец», если он не дальше, м
SENSOR_FAULTS = ('sensor_noise', 'sensor_stuck', 'sensor_bias')


def replays():
    """Имя → функция пересчёта (b, sx, sy) или None для прежнего."""
    from did.replay import NeighborReplay, trusted_replay
    return {'old': None, 'exact': exact_replay,
            'trusted': lambda b, sx, sy: trusted_replay(b, sx, sy, 0.05),
            'gated': NeighborReplay(0.05, 0.35, 'gated'), 'gated_clean': NeighborReplay(0.05, 0.35, 'gated_clean')}


def redo(belief, x, y, fn):
    """Копия карты после пересчёта fn (None — прежний)."""
    b = copy.copy(belief)
    b.p, b._epoch, b._log = belief.p.copy(), belief._epoch.copy(), list(belief._log)
    for name in ('_tags', '_marks'):
        if hasattr(belief, name):
            setattr(b, name, list(getattr(belief, name)))
    b.replay = fn
    b.collected(x, y)
    return b


class Watch:

    def __init__(self, names):
        self.names = names
        self.world = None
        self.bot = None
        self.tags = []           # по показанию журнала карты: (t, настоящие сбои датчика, агент считает датчик шумным)
        self.collects = []
        self.t = 0.0
        self.acts = {}           # занятие → [секунд, метров, ед. заряда, штрафов зон]
        self.trips = []          # подъезды: {t0, x, y, true, mass, end, t1}
        self.hits = []           # штрафы зон: {t, mode, repeat — повтор в той же зоне, since — секунд с прошлого штрафа}
        self._prev = None
        self._sg = None
        self.trail = []          # (t, x, y) раз в такт
        self.back = None         # решение о возврате: {t, x, y, battery, home — оценка дороги, collected}

    def after_tick(self, bot, obs):
        """Чем робот был занят этот такт и чем кончился каждый подъезд к кандидату."""
        remaining = list(self.world.judge.remaining.values())
        sg = bot.queue[0] if bot.queue else None
        if sg is not self._sg:
            if self.trips and 'end' not in self.trips[-1]:
                last = self.trips[-1]
                last['t1'] = round(obs.t, 1)
                last['end'] = 'collected' if bot.collected > last['n'] else ('return' if bot._returning else (bot._trigger or 'other'))
            self._sg = sg
            if sg is not None and sg['type'] == 'investigate':
                d = min((math.hypot(sg['x'] - sx, sg['y'] - sy) for sx, sy in remaining), default=9.9)
                self.trips.append({'t0': round(obs.t, 1), 'x': round(sg['x'], 2), 'y': round(sg['y'], 2), 'd': round(d, 2),
                                   'true': d <= NEAR, 'n': bot.collected, 'fault': bool(self.tag()[1]),
                                   'battery': round(obs.battery, 1)})
        key = bot.mode
        if key in ('approach', 'collect') and self.trips:
            key = 'approach:' + ('true' if self.trips[-1]['true'] else 'false')
        prev, self._prev = self._prev, (obs.t, obs.x, obs.y, obs.battery)
        self.trail.append((obs.t, obs.x, obs.y))
        if bot._returning and self.back is None:
            self.back = {'t': round(obs.t, 1), 'x': round(obs.x, 2), 'y': round(obs.y, 2), 'battery': round(obs.battery, 1),
                         'home': round(float(bot._home_cost(obs.x, obs.y)), 1), 'collected': bot.collected,
                         'd_base': round(math.hypot(obs.x - bot.base[0], obs.y - bot.base[1]), 2),
                         'cands': len(bot.belief.candidates(min_mass=bot.cfg.candidate_mass))}
        if prev is None:
            return
        a = self.acts.setdefault(key, [0.0, 0.0, 0.0, 0])
        a[0] += obs.t - prev[0]
        a[1] += math.hypot(obs.x - prev[1], obs.y - prev[2])
        a[2] += prev[3] - obs.battery
        for e in obs.events:
            if e.get('type') == 'hazard_hit':
                a[3] += 1
                zone = min(self.world.judge.hazards, key=lambda z: math.hypot(z.x - obs.x, z.y - obs.y) - z.r).id
                old = [h for h in self.hits if h['zone'] == zone]
                self.hits.append({'t': round(obs.t, 1), 'mode': key, 'zone': zone, 'repeat': bool(old),
                                  'since': round(obs.t - old[-1]['t'], 1) if old else None,
                                  'returning': bool(bot._returning), 'v': round(obs.v, 2),
                                  'x': round(obs.x, 2), 'y': round(obs.y, 2),
                                  'trail_d': round(min((math.hypot(obs.x - x, obs.y - y) for t, x, y in self.trail
                                                        if t < obs.t - 5.0), default=9.9), 2),
                                  'zone_new': any(w['type'] == 'new_hazard' for w in self.world.judge.world_log)
                                  and zone == self.world.judge.hazards[-1].id})

    def tag(self):
        j = self.world.judge
        bot = self.bot
        true = tuple(k for k in SENSOR_FAULTS if k in j.faults)
        inv = bot.inv.sensor['mode'] if bot.inv else 'ok'
        known = bool(bot.cfg.sensor_health and bot.health.degraded) or inv != 'ok' \
            or bool(bot.guard and bot.guard.wait is not None)
        return (round(self.t, 2), true, known)

    def on_collect(self, belief, x, y):
        fns = replays()
        remaining = list(self.world.judge.remaining.items())
        tags = self.tags[-len(belief._log):] if belief._log else []
        row = {'t': round(self.t, 1), 'x': round(x, 2), 'y': round(y, 2), 'log': len(belief._log),
               'left_before': belief.left, 'true_left': len(remaining),
               'fault_readings': sum(1 for g in tags if g[1]), 'fault_known': sum(1 for g in tags if g[1] and g[2]),
               'flagged': sum(1 for g in tags if g[2]),
               'by_kind': {k: sum(1 for g in tags if k in g[1]) for k in SENSOR_FAULTS},
               'before': {'p35': [round(belief.prob_within(sx, sy, 0.35), 3) for _, (sx, sy) in remaining]},
               'samples': [[i, round(sx, 2), round(sy, 2)] for i, (sx, sy) in remaining], 'after': {}}
        from did.replay import suspects
        bad = suspects(belief._log, self.bot.rules.sensor_sigma)
        row['suspect'] = {'marked': int(bad.sum()), 'marked_true': sum(1 for g, m in zip(tags, bad) if m and g[1]),
                          'by_kind_marked': {k: sum(1 for g, m in zip(tags, bad) if m and k in g[1]) for k in SENSOR_FAULTS}}
        truth = np.array([bool(g[1]) for g in tags], dtype=bool)
        fns['oracle'] = lambda b, sx, sy: trusted_replay(b, sx, sy, self.bot.rules.sensor_sigma, skip=truth)
        for name in self.names:
            b = redo(belief, x, y, fns[name])
            cands = b.candidates(min_mass=self.bot.cfg.candidate_mass)
            for c in cands:
                c['d'] = round(min((math.hypot(c['x'] - sx, c['y'] - sy) for _, (sx, sy) in remaining), default=9.9), 2)
            row['after'][name] = {
                'p35': [round(b.prob_within(sx, sy, 0.35), 3) for _, (sx, sy) in remaining],
                'cands': [[round(c['x'], 2), round(c['y'], 2), round(c['mass'], 2), c['d']] for c in cands],
                'false': sum(1 for c in cands if c['d'] > NEAR), 'false_mass': round(sum(c['mass'] for c in cands if c['d'] > NEAR), 2),
                'total': round(b.total(), 2), 'left': b.left}
        self.collects.append(row)


def run_watched(agent, level, seed, rules=None, config=None, names=('old', 'exact')):
    watch = Watch(list(names))
    cls = agent_mod.Agent
    tick, init = cls.tick, fastsim.FastSim.__init__
    update, collected = SampleBelief.update, SampleBelief.collected
    depth = [0]

    def init_w(self, *a, **k):
        init(self, *a, **k)
        watch.world = self

    def tick_w(self, obs, io):
        watch.bot, watch.t = self, obs.t
        out = tick(self, obs, io)
        watch.after_tick(self, obs)
        return out

    def update_w(self, x, y, z, sigma):
        if depth[0] == 0 and watch.bot is not None and self is watch.bot.belief:
            watch.tags.append(watch.tag())
        return update(self, x, y, z, sigma)

    def collected_w(self, x, y):
        if depth[0] == 0 and watch.bot is not None and self is watch.bot.belief:
            depth[0] += 1
            try:
                watch.on_collect(self, x, y)
            finally:
                depth[0] -= 1
        return collected(self, x, y)

    cls.tick, fastsim.FastSim.__init__ = tick_w, init_w
    SampleBelief.update, SampleBelief.collected = update_w, collected_w
    try:
        summary = run_episode(level, seed, agent, rules=rules, config=config, save=False)
    finally:
        cls.tick, fastsim.FastSim.__init__ = tick, init
        SampleBelief.update, SampleBelief.collected = update, collected
    return summary, watch


def _seg_d(a, b, p):
    ax, ay, bx, by, px, py = *a, *b, *p
    dx, dy = bx - ax, by - ay
    k = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-12)))
    return math.hypot(px - ax - k * dx, py - ay - k * dy)


def _job(task):
    agent, level, seed, rules, config, names = task
    summary, watch = run_watched(agent, level, seed, rules, config, names)
    m = summary['metrics']
    if watch.back:                       # обратная дорога: сколько её прошло по полу, где робот ещё не ездил
        t0 = watch.back['t']
        before = np.array([(x, y) for t, x, y in watch.trail if t < t0])
        after = np.array([(x, y) for t, x, y in watch.trail if t >= t0])
        if len(before) and len(after) > 1:
            step = np.hypot(*np.diff(after, axis=0).T)
            off = np.array([np.hypot(*(before - q).T).min() for q in after[1:]]) > 0.2
            watch.back.update(path=round(float(step.sum()), 2), off=round(float(step[off].sum()), 2))
        # все зоны на прямой от места возврата до базы
        bx, by = watch.bot.base
        watch.back['zones_on_line'] = sum(1 for z in watch.world.judge.hazards if _seg_d(
            (watch.back['x'], watch.back['y']), (bx, by), (z.x, z.y)) <= z.r)
    return {'seed': seed, 'score': m['score'], 'returned': m['returned'], 'false_collects': m['false_collects'],
            'collects': watch.collects, 'acts': watch.acts, 'trips': watch.trips, 'hits': watch.hits, 'back': watch.back,
            'replay': getattr(watch.bot.belief.replay, 'stats', None), 'world': watch.world.judge.world_log,
            'journal': [e for e in watch.bot.journal.entries]}


def summarize(runs, names):
    """Сводка по всем сборам: сколько ложных кандидатов оставляет каждый пересчёт и сколько соседей теряет."""
    groups = {'все сборы': lambda c: True,
              'в журнале нет показаний при сбое': lambda c: c['fault_readings'] == 0,
              'в журнале есть показания при сбое': lambda c: c['fault_readings'] > 0,
              '  из них шум': lambda c: c['by_kind']['sensor_noise'] > 0,
              '  из них залипание': lambda c: c['by_kind']['sensor_stuck'] > 0,
              '  из них сдвиг': lambda c: c['by_kind']['sensor_bias'] > 0}
    for title, pick in groups.items():
        cs = [c for r in runs for c in r['collects'] if pick(c) and c['true_left'] > 0]
        if not cs:
            continue
        print(f'{title}: сборов {len(cs)}')
        for name in names:
            false = sum(c['after'][name]['false'] for c in cs)
            mass = sum(c['after'][name]['false_mass'] for c in cs)
            kept = lost = known = 0
            for c in cs:
                for before, after in zip(c['before']['p35'], c['after'][name]['p35']):
                    if before >= 0.35:
                        known += 1
                        kept += after >= 0.35
                        lost += after <= 0.5 * before
            true = sum(1 for c in cs for cand in c['after'][name]['cands'] if cand[3] <= NEAR)
            print(f'   {name:12s} ложных кандидатов {false:4d} (сумма уверенности {mass:6.1f}); настоящих кандидатов {true:4d}; '
                  f'из {known} известных соседей (≥0,35 до сбора) осталось {kept}, упало вдвое и больше {lost}')


def distrust(runs):
    """Насколько маска недоверия (did.replay.suspects) совпадает с настоящими сбоями датчика."""
    cs = [c for r in runs for c in r['collects'] if 'suspect' in c]
    total = sum(c['log'] for c in cs)
    marked = sum(c['suspect']['marked'] for c in cs)
    true = sum(c['fault_readings'] for c in cs)
    hit = sum(c['suspect']['marked_true'] for c in cs)
    print(f'Недоверие: показаний в журналах на сборах {total}, снято при настоящем сбое {true}, помечено {marked}, '
          f'из помеченных при настоящем сбое {hit}')
    for k in SENSOR_FAULTS:
        n = sum(c['by_kind'][k] for c in cs)
        print(f"   {k:14s} показаний {n:5d}, помечено {sum(c['suspect']['by_kind_marked'][k] for c in cs):5d}")


def activity(runs):
    """На что ушли время, путь и заряд и чем кончались подъезды — в сумме по прогонам."""
    acts = {}
    for r in runs:
        for k, v in r['acts'].items():
            a = acts.setdefault(k, [0.0, 0.0, 0.0, 0])
            for i in range(4):
                a[i] += v[i]
    n = len(runs)
    print(f'Занятия, на прогон (прогонов {n}): секунд, метров, ед. заряда; штрафов зон всего')
    for k, v in sorted(acts.items(), key=lambda kv: -kv[1][2]):
        print(f'   {k:16s} {v[0] / n:7.1f} {v[1] / n:6.2f} {v[2] / n:6.2f}   {v[3]:3d}')
    for label, pick in (('к настоящему образцу', True), ('к ложному месту', False)):
        trips = [t for r in runs for t in r['trips'] if t['true'] == pick]
        ends = {}
        for t in trips:
            ends[t.get('end', 'конец прогона')] = ends.get(t.get('end', 'конец прогона'), 0) + 1
        print(f'Подъездов {label}: {len(trips)} (начато при сбое датчика {sum(t["fault"] for t in trips)}); чем кончились: {ends}')
    stats = [r['replay'] for r in runs if r.get('replay')]
    if stats:
        print(f"Пересчётов после сбора {sum(s['collects'] for s in stats)}, из них взята новая карта "
              f"{sum(s['switched'] for s in stats)} (в {sum(1 for s in stats if s['switched'])} прогонах из {len(stats)}); "
              f"показаний отброшено как ненадёжные {sum(s['skipped'] for s in stats)}")
    hits = [h for r in runs for h in r['hits']]
    print(f"Штрафов зон {len(hits)}: первый въезд в зону {sum(not h['repeat'] for h in hits)}, повтор в той же зоне "
          f"{sum(h['repeat'] for h in hits)} (из них не позже 6 с после прошлого — робот из зоны не выехал: "
          f"{sum(1 for h in hits if h['repeat'] and h['since'] <= 6.0)}); по дороге домой {sum(h['returning'] for h in hits)}, "
          f"из них повторов {sum(1 for h in hits if h['returning'] and h['repeat'])}")
    print(f"Счёт {sum(r['score'] for r in runs) / n:.2f}, возвратов {sum(bool(r['returned']) for r in runs)}, "
          f"ложных сборов {sum(r['false_collects'] for r in runs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', default='adaptive_v2')
    ap.add_argument('--level', default='hard')
    ap.add_argument('--rules', default=None, choices=['science'])
    ap.add_argument('--seeds', default='1-80')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--replays', default='old,exact')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--journal', action='store_true', help='печатать журнал решений и события судьи')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if '-' in args.seeds:
        a, b = (int(v) for v in args.seeds.split('-'))
        seeds = list(range(a, b + 1))
    else:
        seeds = [int(v) for v in args.seeds.split(',')]
    names = args.replays.split(',')
    tasks = [(args.agent, args.level, s, args.rules, json.loads(args.config), names) for s in seeds]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as pool:
            runs = list(pool.map(_job, tasks))
    else:
        runs = [_job(t) for t in tasks]
    if len(runs) <= 3 or args.journal:
        for r in runs:
            print(f"== {args.level}-{r['seed']}: счёт {r['score']}, возврат {r['returned']}, ложных сборов {r['false_collects']}")
            if args.journal:
                lines = [(w['t'], f"   СУДЬЯ {w}") for w in r['world']]
                lines += [(e['t'], f"   {e['t']:6.1f} [{e['kind']}] {e['text']}") for e in r['journal']
                          if e['kind'] in ('decision', 'alarm', 'action')]
                for _, line in sorted(lines, key=lambda v: v[0]):
                    print(line[:260])
            for c in r['collects']:
                print(f"  сбор {c['t']} с в ({c['x']};{c['y']}): в журнале {c['log']} показаний, при сбое {c['fault_readings']} "
                      f"(агент знал о {c['fault_known']}), {c['by_kind']}; осталось {c['true_left']}")
                print(f"      до:    {c['before']['p35']}")
                for name in names:
                    a = c['after'][name]
                    print(f"      {name:8s} {a['p35']}  кандидаты {a['cands']}  ложных {a['false']}")
    summarize(runs, names)
    distrust(runs)
    activity(runs)
    if args.out:
        Path(args.out).write_text(json.dumps(runs, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
