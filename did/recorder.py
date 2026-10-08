"""Запись прогона: всё, что нужно, чтобы потом показать его по секундам и посчитать метрики.

Формат один для быстрого симулятора и для Gazebo. Файл — JSON в gzip, поля:

  version, id, experiment, arm, backend, created
  agent      {name, config}
  scenario   сценарий целиком (истина: образцы, грунты, опасные зоны, события)
  rules      правила судьи
  result     итог судьи и метрики
  track      {t, x, y, th, battery, sensor, mode}: массивы одной длины, шаг ~0,2 с; mode — индекс в modes
  modes      названия режимов агента
  events     сообщения судьи (/did/events): [{t, type, x, y, ...}]
  world      скрытые изменения среды, применённые судьёй: [{t, type}]
  journal    записи журнала агента: [{t, kind, text, data?}]
  hypotheses гипотезы агента: [{id, t_open, statement, test, status, t_close, verdict}]
  plans      решения планировщика: [{t, source, trigger, reasoning, subgoals}]
  paths      построенные пути: [{t, goal: [x, y], pts: [[x, y], ...], cost}]
  belief     {res, x0, y0, w, h, enc: 'sqrt', snaps: [{t, data}]}: вероятность образца, uint8 в base64,
             строки снизу вверх, p = (v / 255)²
  soil       {res, x0, y0, w, h, scale: 40, snaps: [{t, data}]}: оценка множителя расхода, uint8 в base64,
             mult = v / 40; 0 — на этом участке агент не ездил
  hazards    опасные зоны, которые запомнил агент: [{t, x, y, r}]
  scans      [{t, x, y, th, r}]: редкие сканы лидара, r в сантиметрах, 0 — нет отражения
  pose_fix   [{t, dx, dy, dth}]: поправка позы по лидару раз в секунду — исправленная поза (она же в
             track) минус поза по одометрии; поля нет в записях без локализации
  llm        обмены с языковой моделью: [{t, ok, latency_ms, request, response, errors}]
  foresight  сравнения будущих маршрутов (did/foresight.py): [{t, trigger, worlds, risk_limit, battery, chosen,
             ms, unknowns, columns: [{key, label, unit}], rows: [{id, label, samples, score,
             risk, battery_p5, battery_p50, battery_p95, cost, hits, unknown_m, ok, worth, chosen}]}]; ok — риск
             не выше порога, worth — ожидаемые образцы окупают прибавку риска; поля нет в записях агентов без
             такого сравнения
"""
import base64
import gzip
import json
import time
from pathlib import Path

import numpy as np

VERSION = 1


def encode_grid(values):
    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode('ascii')


def decode_grid(data, h, w):
    return np.frombuffer(base64.b64decode(data), dtype=np.uint8).reshape(h, w)


class Recorder:

    def __init__(self, track_dt=0.2, belief_dt=2.0, scan_dt=1.0, scan_rays=120):
        self.track_dt, self.belief_dt, self.scan_dt, self.scan_rays = track_dt, belief_dt, scan_dt, scan_rays
        self.track = {k: [] for k in ('t', 'x', 'y', 'th', 'battery', 'sensor', 'mode')}
        self.modes = []
        self.events, self.plans, self.paths, self.hazards, self.scans, self.llm = [], [], [], [], [], []
        self.belief = None
        self.soil = None
        self._next_track = self._next_belief = self._next_scan = 0.0
        self._last_sensor = 0.0
        self._last_soil = None
        self.pose_fix, self._next_fix = [], 0.0
        self.foresight = []

    # --- во время прогона --------------------------------------------------------------------

    def sample(self, obs, mode, fix=None):
        if obs.sensor is not None:
            self._last_sensor = obs.sensor
        if obs.scan is not None and obs.t + 1e-9 >= self._next_scan:
            self._next_scan = obs.t + self.scan_dt
            step = max(1, len(obs.scan) // self.scan_rays)
            r = np.where(np.isfinite(obs.scan[::step]), obs.scan[::step] * 100.0, 0.0)
            self.scans.append({'t': round(obs.t, 2), 'x': round(obs.x, 3), 'y': round(obs.y, 3),
                               'th': round(obs.th, 3), 'r': np.round(r).astype(int).tolist()})
        if obs.t + 1e-9 < self._next_track:
            return
        self._next_track = obs.t + self.track_dt
        if mode not in self.modes:
            self.modes.append(mode)
        tr = self.track
        tr['t'].append(round(obs.t, 2))
        tr['x'].append(round(obs.x, 3))
        tr['y'].append(round(obs.y, 3))
        tr['th'].append(round(obs.th, 3))
        tr['battery'].append(round(obs.battery, 2))
        tr['sensor'].append(round(self._last_sensor, 3))
        tr['mode'].append(self.modes.index(mode))
        if fix is not None and obs.t + 1e-9 >= self._next_fix:      # всегда вместе с точкой трека
            self._next_fix = obs.t + 1.0
            self.pose_fix.append({'t': round(obs.t, 2), 'dx': round(fix[0], 4), 'dy': round(fix[1], 4),
                                  'dth': round(fix[2], 4)})

    def want_belief(self, t):
        if t + 1e-9 < self._next_belief:
            return False
        self._next_belief = t + self.belief_dt
        return True

    def add_belief(self, t, belief):
        if self.belief is None:
            self.belief = {'res': belief.res, 'x0': belief.x0, 'y0': belief.y0, 'w': belief.w, 'h': belief.h,
                           'enc': 'sqrt', 'snaps': []}
        data = encode_grid(np.round(np.sqrt(belief.grid()) * 255))
        self.belief['snaps'].append({'t': round(t, 1), 'data': data})

    def add_soil(self, t, soil):
        if soil.version == self._last_soil:
            return
        self._last_soil = soil.version
        if self.soil is None:
            self.soil = {'res': soil.res, 'x0': soil.x0, 'y0': soil.y0, 'w': soil.w, 'h': soil.h,
                         'scale': 40, 'snaps': []}
        mult, conf = soil._estimate()
        values = np.where(conf > 0.15, np.clip(np.round(mult * 40), 1, 255), 0)
        self.soil['snaps'].append({'t': round(t, 1), 'data': encode_grid(values)})

    def add_events(self, events):
        self.events.extend(events)

    def add_plan(self, t, source, trigger, reasoning, subgoals):
        self.plans.append({'t': round(t, 1), 'source': source, 'trigger': trigger, 'reasoning': reasoning,
                           'subgoals': [dict(s) for s in subgoals]})   # копия: очередь агента потом меняется

    def add_path(self, t, goal, pts, cost):
        thin = pts[::3] + ([pts[-1]] if (len(pts) - 1) % 3 else [])
        self.paths.append({'t': round(t, 1), 'goal': [round(goal[0], 2), round(goal[1], 2)],
                           'pts': [[round(x, 2), round(y, 2)] for x, y in thin], 'cost': round(cost, 2)})

    def add_hazard(self, t, x, y, r):
        self.hazards.append({'t': round(t, 1), 'x': round(x, 2), 'y': round(y, 2), 'r': round(r, 2)})

    def add_llm(self, t, exchange):
        self.llm.append({'t': round(t, 1), **exchange})

    def add_foresight(self, decision):
        self.foresight.append(decision)

    # --- итог --------------------------------------------------------------------------------

    def build(self, *, run_id, experiment, arm, backend, agent, scenario, rules, result, world, journal):
        return {
            'version': VERSION, 'id': run_id, 'experiment': experiment, 'arm': arm, 'backend': backend,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'agent': agent, 'scenario': scenario, 'rules': rules, 'result': result,
            'track': self.track, 'modes': self.modes, 'events': self.events, 'world': world,
            'journal': journal.entries, 'hypotheses': journal.hypotheses,
            'plans': self.plans, 'paths': self.paths, 'belief': self.belief, 'soil': self.soil,
            'hazards': self.hazards, 'scans': self.scans, 'llm': self.llm,
            **({'pose_fix': self.pose_fix} if self.pose_fix else {}),
            **({'foresight': self.foresight} if self.foresight else {}),
        }


def save_trace(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(trace, f, ensure_ascii=False, separators=(',', ':'), default=_plain)
    return path


def load_trace(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _plain(o):
    if hasattr(o, 'to_dict'):
        return o.to_dict()
    if hasattr(o, '__dataclass_fields__'):
        return {k: getattr(o, k) for k in o.__dataclass_fields__}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
