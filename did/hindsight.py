"""L5: цена решения планировщика «задним числом» — мерило, а не часть агента.

Быстрый симулятор детерминирован по номеру сценария, поэтому прогон можно повторить до того же вызова
планировщика, подставить вместо его решения любой допустимый вариант первой подцели и дать агенту доиграть
прогон как обычно, правилом. Итоговый счёт судьи такого прогона — цена варианта.

В цене участвует скрытая правда сценария (через симулятор): это оценка с подглядыванием в будущее. Она годится
только для сравнения решений между собой; агент этот модуль не импортирует и получить из него ничего не может.

    replay('hard', 5, at=7, subgoals=[{'type': 'explore', 'target': 'E2'}])   # цена варианта
    decisions('hard', 5)                                                     # все решения прогона со сводками
"""
import copy
import hashlib
import json
import math

import numpy as np

from .planner import HeuristicPlanner
from .runner import run_episode

AGENT = 'adaptive_v2'
# Виды состояний в порядке отбора (редкие — раньше). Считаются только по сводке, которую видит планировщик.
TAGS = ('time_short', 'sensor_bad', 'almost_done', 'low_battery', 'weak_near_strong_far', 'weak_only',
        'danger_near', 'high_battery', 'explore_only')
RESULT_KEYS = ('score', 'samples_collected', 'returned', 'battery', 'collisions', 'false_collects', 'hazard_hits',
               't', 'reason')


def option_key(sg):
    """Короткое имя первой подцели: 'investigate:C1', 'explore:E2', 'return_base', 'goto:0.50,-1.00'."""
    kind = sg.get('type')
    if kind in ('investigate', 'explore'):
        return f"{kind}:{sg.get('target')}"
    if kind == 'goto':
        return f"goto:{float(sg['x']):.2f},{float(sg['y']):.2f}"
    return 'return_base'


def options(state):
    """Допустимые варианты первой подцели: годные кандидаты, годные точки разведки и возврат на базу."""
    out = [{'type': 'investigate', 'target': c['id']} for c in state['candidates'] if c['feasible']]
    out += [{'type': 'explore', 'target': p['id']} for p in state['explore_points'] if p['feasible']]
    return out + [{'type': 'return_base'}]


def rule_choice(state):
    """Первая подцель правила на этом состоянии (у правила она всегда одна)."""
    return HeuristicPlanner().plan(state)['subgoals'][0]


def state_hash(state):
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]


class ForcedPlanner(HeuristicPlanner):
    """Правило, у которого одно решение — вызов номер `at` — заменено заданными подцелями.

    Остальные вызовы отвечает правило, поэтому без подстановки (subgoals=None) прогон совпадает с обычным.
    rep > 0 — в момент этого решения шум датчика образцов и лидара пересеивается: мир и скрытая правда те же,
    а продолжение прогона — другая реализация шума. keep=True — запоминать все сводки состояния.
    """

    def __init__(self, at=None, subgoals=None, rep=0, keep=False):
        self.at, self.subgoals, self.rep, self.keep = at, subgoals, int(rep), keep
        self.n = 0
        self.times, self.hashes, self.states = [], [], []
        self.state = None                    # сводка состояния в момент подстановки
        self._world = None

    def bind_run(self, world, bot):
        self._world = world

    def plan(self, state):
        k, self.n = self.n, self.n + 1
        self.times.append(state['time_s'])
        self.hashes.append(state_hash(state))
        if self.keep:
            self.states.append(copy.deepcopy(state))
        if k != self.at:
            return super().plan(state)
        self.state = copy.deepcopy(state)
        if self.rep and self._world is not None:
            seed = int(self._world.judge.scenario.seed)
            self._world.judge.rng = np.random.default_rng([seed, 7, self.rep])
            self._world.rng = np.random.default_rng([seed, 11, self.rep])
        if self.subgoals is None:
            return super().plan(state)
        text = 'Подстановка мерила L5: ' + ', '.join(option_key(sg) for sg in self.subgoals)
        return self._result(text, copy.deepcopy(self.subgoals))


def replay(level, seed, at=None, subgoals=None, rep=0, agent=AGENT, keep=False, rules=None):
    """Повторить прогон; на вызове планировщика номер `at` подставить subgoals (None — оставить выбор правила).

    Возвращает итог судьи, число вызовов планировщика, хэши сводок по вызовам, время следующего вызова после
    подстановки (сколько прожило решение) и сам планировщик-наблюдатель.
    """
    planner = ForcedPlanner(at, subgoals, rep, keep)
    summary = run_episode(level, seed, agent, planner=planner, rules=rules, save=False)
    m = summary['metrics']
    out = {key: m.get(key) for key in RESULT_KEYS}
    out.update(calls=planner.n, reached=at is None or planner.state is not None,
               state_hash=planner.hashes[at] if at is not None and at < planner.n else None,
               next_t=planner.times[at + 1] if at is not None and at + 1 < planner.n else None)
    return out, planner


def decisions(level, seed, agent=AGENT, rules=None):
    """Все решения планировщика в обычном прогоне: [{level, seed, k, t, state, options, rule}] и итог прогона."""
    result, planner = replay(level, seed, agent=agent, keep=True, rules=rules)
    rows = []
    for k, state in enumerate(planner.states):
        rows.append({'id': f'{level}-{seed}#{k}', 'level': level, 'seed': seed, 'k': k, 't': state['time_s'],
                     'state': state, 'hash': planner.hashes[k], 'options': [option_key(o) for o in options(state)],
                     'rule': option_key(rule_choice(state)), 'tags': tags(state)})
    return rows, result


# --- виды состояний и отбор в банк -------------------------------------------------------------------

def tags(state):
    """Виды состояния (experiments/L5.yaml). Только по сводке: ни ответов моделей, ни цены вариантов здесь нет."""
    cands = [c for c in state['candidates'] if c['feasible']]
    points = [p for p in state['explore_points'] if p['feasible']]
    out = []
    if state['time_limit_s'] - state['time_s'] <= 120.0:
        out.append('time_short')
    if state['sensor']['status'] == 'degraded':
        out.append('sensor_bad')
    if state['samples']['total'] - state['samples']['collected'] == 1:
        out.append('almost_done')
    if state['battery'] <= 20.0:
        out.append('low_battery')
    if len(cands) >= 2:
        by_rule = max(cands, key=lambda c: c['confidence'] / (c['cost_to'] + 0.5))
        surest = max(cands, key=lambda c: c['confidence'])
        if surest['id'] != by_rule['id'] and surest['confidence'] - by_rule['confidence'] >= 0.15:
            out.append('weak_near_strong_far')
    if cands and all(c['confidence'] < 0.5 for c in cands):
        out.append('weak_only')
    zones = list(state['hazards']) + [z for z in state['soil_zones']
                                      if z.get('status') == 'confirmed' and z.get('mult', 1.0) >= 1.5]
    places = [state['pose']] + cands + points
    if any(math.hypot(z['x'] - p['x'], z['y'] - p['y']) <= z['radius'] + 0.6 for z in zones for p in places):
        out.append('danger_near')
    if state['battery'] >= 40.0 and len(cands) >= 2:
        out.append('high_battery')
    if not cands and len(points) >= 2:
        out.append('explore_only')
    return out


def eligible(row):
    """Годится в банк: решение после старта и хотя бы два допустимых варианта первой подцели."""
    return row['t'] > 0.0 and len(row['options']) >= 2


def _order(row):
    return hashlib.sha256(f"L5|{row['level']}|{row['seed']}|{row['k']}".encode('utf-8')).hexdigest()


def select(rows, total=90, per_tag=10, per_run_tag=2, per_run=6, gap_s=5.0):
    """Отбор состояний в банк по правилу из experiments/L5.yaml. Детерминирован: зависит только от rows."""
    pool = sorted((r for r in rows if eligible(r)), key=_order)
    taken, by_run = [], {}

    def free(row, tag_count):
        run = (row['level'], row['seed'])
        mine = by_run.get(run, [])
        if any(r['k'] == row['k'] for r in mine) or len(mine) >= per_run:
            return False
        if tag_count is not None and tag_count.get(run, 0) >= per_run_tag:
            return False
        return all(abs(r['t'] - row['t']) >= gap_s for r in mine)

    def take(row, why):
        taken.append({**row, 'picked_for': why})
        by_run.setdefault((row['level'], row['seed']), []).append(row)

    for tag in TAGS:
        count, n = {}, 0
        for row in pool:
            if n >= per_tag or len(taken) >= total:
                break
            if tag in row['tags'] and free(row, count):
                take(row, tag)
                run = (row['level'], row['seed'])
                count[run] = count.get(run, 0) + 1
                n += 1
    for row in pool:
        if len(taken) >= total:
            break
        if free(row, None):
            take(row, 'any')
    return taken
