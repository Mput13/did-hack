"""Генератор сценариев: где лежат образцы, где дорогой грунт, что случится по ходу прогона.

Один и тот же (уровень, seed) всегда даёт один и тот же сценарий — на этом держится сравнение
агентов: оба получают одинаковую арену.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import BASE, FAULT_KINDS, LEVELS

SOIL_MULTS = {'easy': [3.0], 'medium': [2.0, 3.0, 4.0], 'hard': [2.0, 2.5, 3.0, 4.0]}
ALL_EVENTS = ('soil_change', 'new_hazard', 'sensor_fault')


@dataclass
class Zone:
    id: str
    shape: str              # 'circle' или 'rect' (стороны вдоль осей)
    x: float
    y: float
    r: float = 0.0
    w: float = 0.0
    h: float = 0.0
    mult: float = 1.0       # множитель расхода батареи; для опасных зон не используется
    fault: str = ''         # опасная зона: какой сбой она вызывает (при правилах faults=True)
    fault_s: float = 0.0    # и на сколько секунд

    def contains(self, px, py):
        if self.shape == 'circle':
            return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.r ** 2
        return abs(px - self.x) <= self.w / 2 and abs(py - self.y) <= self.h / 2

    def mask(self, X, Y):
        if self.shape == 'circle':
            return (X - self.x) ** 2 + (Y - self.y) ** 2 <= self.r ** 2
        return (np.abs(X - self.x) <= self.w / 2) & (np.abs(Y - self.y) <= self.h / 2)

    def reach(self):
        """Радиус круга, в который зона помещается целиком."""
        return self.r if self.shape == 'circle' else math.hypot(self.w, self.h) / 2


@dataclass
class Scenario:
    level: str
    seed: int
    samples: list                      # [[x, y], ...]
    soils: list = field(default_factory=list)
    hazards: list = field(default_factory=list)
    events: list = field(default_factory=list)   # [{'t': с, 'type': ..., ...}], по возрастанию t
    base: tuple = BASE

    @property
    def name(self):
        return f'{self.level}-{self.seed}'

    def to_dict(self):
        d = asdict(self)
        d['base'] = list(self.base)
        d['name'] = self.name
        return d

    @staticmethod
    def from_dict(d):
        def zones(items):
            return [Zone(**z) for z in items]
        events = []
        for ev in d.get('events', []):
            ev = dict(ev)
            if 'soils' in ev:
                ev['soils'] = zones(ev['soils'])
            if 'zone' in ev:
                ev['zone'] = Zone(**ev['zone'])
            events.append(ev)
        return Scenario(level=d['level'], seed=d['seed'], samples=[list(p) for p in d['samples']],
                        soils=zones(d.get('soils', [])), hazards=zones(d.get('hazards', [])),
                        events=events, base=tuple(d.get('base', BASE)))


def soil_mult(soils, x, y):
    m = 1.0
    for z in soils:
        if z.contains(x, y):
            m = max(m, z.mult)
    return m


def _free_points(arena, rng, margin):
    """Бесконечный поток случайных точек арены с зазором до стен не меньше margin."""
    ys, xs = np.nonzero(arena.clear >= margin)
    while True:
        i = rng.integers(len(xs))
        x, y = arena.g2w(xs[i], ys[i])
        yield x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)


def _place(points, accept, tries=4000):
    for _ in range(tries):
        p = next(points)
        if accept(p):
            return p
    raise RuntimeError('не удалось разместить объект сценария: слишком жёсткие ограничения')


def _soil_zone(zid, mult, arena, rng, others, base):
    points = _free_points(arena, rng, 0.10)

    def make(p):
        if rng.random() < 0.5:
            return Zone(zid, 'circle', round(p[0], 2), round(p[1], 2), r=round(rng.uniform(0.38, 0.60), 2),
                        mult=mult)
        return Zone(zid, 'rect', round(p[0], 2), round(p[1], 2), w=round(rng.uniform(0.7, 1.2), 2),
                    h=round(rng.uniform(0.7, 1.2), 2), mult=mult)

    for _ in range(4000):
        z = make(next(points))
        if math.dist((z.x, z.y), base) < z.reach() + 0.45:
            continue
        if any(math.dist((z.x, z.y), (o.x, o.y)) < 0.8 * (z.reach() + o.reach()) for o in others):
            continue
        return z
    raise RuntimeError('не удалось разместить зону грунта')


def _hazard_zone(zid, arena, rng, samples, base, others):
    """Опасная зона ставится на оживлённом месте: между двумя образцами или между образцом и базой."""
    points = _free_points(arena, rng, 0.15)
    anchors = [tuple(s) for s in samples] + [tuple(base)]

    def ok(p):
        return (math.dist(p, base) > 1.0
                and all(math.dist(p, s) > 0.65 for s in samples)
                and all(math.dist(p, (o.x, o.y)) > 0.9 for o in others))

    for _ in range(400):
        i, j = rng.choice(len(anchors), size=2, replace=False)
        k = rng.uniform(0.35, 0.65)
        p = (anchors[i][0] + k * (anchors[j][0] - anchors[i][0]) + rng.uniform(-0.2, 0.2),
             anchors[i][1] + k * (anchors[j][1] - anchors[i][1]) + rng.uniform(-0.2, 0.2))
        if arena.clearance(*p) >= 0.15 and ok(p):
            break
    else:
        p = _place(points, ok)
    return Zone(zid, 'circle', round(p[0], 2), round(p[1], 2), r=round(rng.uniform(0.25, 0.33), 2))


def generate(level, seed, arena, n_samples=None, n_soils=None, n_hazards=None, events=None,
             event_window=(20.0, 70.0)):
    """Сценарий уровня easy/medium/hard. Параметры n_* и events переопределяют таблицу уровней.

    event_window — в какие секунды прогона случаются события hard: окно подобрано под длительность
    прогона (1,5–3 минуты), чтобы изменения застали агента в работе.
    """
    spec = LEVELS[level]
    rng = np.random.default_rng([seed, sum(level.encode())])
    base = BASE
    n_samples = spec['samples'] if n_samples is None else n_samples
    n_soils = spec['soils'] if n_soils is None else n_soils
    n_hazards = spec['hazards'] if n_hazards is None else n_hazards
    if events is None:
        events = ALL_EVENTS if level == 'hard' else ()

    samples = []
    points = _free_points(arena, rng, 0.22)
    for _ in range(n_samples):
        p = _place(points, lambda q: math.dist(q, base) > 1.0
                   and all(math.dist(q, s) > 0.9 for s in samples))
        samples.append([round(p[0], 2), round(p[1], 2)])

    mults = list(SOIL_MULTS[level])
    rng.shuffle(mults)
    soils = []
    for i in range(n_soils):
        soils.append(_soil_zone(chr(ord('A') + i), mults[i % len(mults)], arena, rng, soils, base))

    hazards = []
    for i in range(n_hazards):
        hazards.append(_hazard_zone(f'X{i + 1}', arena, rng, samples, base, hazards))

    timeline = []
    t0, t1 = event_window
    span = t1 - t0
    if 'soil_change' in events and soils:
        # Один грунт меняет стоимость, один переезжает: карта стоимостей агента устаревает.
        changed = [Zone(**asdict(z)) for z in soils]
        i = int(rng.integers(len(changed)))
        changed[i].mult = 4.0 if changed[i].mult <= 2.5 else 1.5
        if len(changed) > 1:
            j = int((i + 1 + rng.integers(len(changed) - 1)) % len(changed))
            others = [z for k, z in enumerate(changed) if k != j]
            moved = _soil_zone(changed[j].id, changed[j].mult, arena, rng, others, base)
            changed[j] = moved
        timeline.append({'t': round(float(rng.uniform(t0, t0 + 0.4 * span)), 1), 'type': 'soil_change',
                         'soils': changed})
    if 'new_hazard' in events:
        zone = _hazard_zone(f'X{len(hazards) + 1}', arena, rng, samples, base, hazards)
        timeline.append({'t': round(float(rng.uniform(t0 + 0.2 * span, t0 + 0.7 * span)), 1), 'type': 'new_hazard',
                         'zone': zone})
    if 'sensor_fault' in events:
        timeline.append({'t': round(float(rng.uniform(t0 + 0.4 * span, t1)), 1), 'type': 'sensor_fault',
                         'duration': round(float(rng.uniform(25, 40)), 1), 'sigma': 0.25})
    timeline.sort(key=lambda e: e['t'])

    # Виды сбоев назначаются отдельным генератором случайных чисел: расстановка прежних серий не меняется.
    rng2 = np.random.default_rng([seed, sum(level.encode()), 2])
    for z in hazards + [e['zone'] for e in timeline if e['type'] == 'new_hazard']:
        z.fault = str(rng2.choice(FAULT_KINDS))
        z.fault_s = round(float(rng2.uniform(20, 30)), 1)
    for e in timeline:
        if e['type'] == 'sensor_fault':
            e['kind'] = str(rng2.choice(FAULT_KINDS[1:]))

    return Scenario(level=level, seed=int(seed), samples=samples, soils=soils, hazards=hazards,
                    events=timeline, base=base)
