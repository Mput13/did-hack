"""Судья: скрытое состояние мира, батарея, датчик образцов, штрафы и счёт.

Одна и та же логика работает в быстром симуляторе и в узле ROS 2 (/did/*). Судье сообщают позу
робота, он считает расход и события. Изменения среды из сценария применяются молча.
"""
import math

import numpy as np

from .config import Rules
from .scenario import soil_mult


class Judge:

    def __init__(self, scenario, arena, rules=None, seed=0):
        self.scenario = scenario
        self.arena = arena
        self.rules = rules or Rules()
        self.rng = np.random.default_rng([int(seed), 7])
        self.rng_battery = np.random.default_rng([int(seed), 8])   # отдельный поток: шум батареи не сдвигает шум датчика
        self.soils = list(scenario.soils)
        self.hazards = list(scenario.hazards)
        self.pending = sorted(scenario.events, key=lambda e: e['t'])
        self.remaining = {i: tuple(p) for i, p in enumerate(scenario.samples)}
        self.collected = []            # [(номер образца, t)]
        self.battery = self.rules.battery_start
        self.t = 0.0
        self.distance = 0.0
        self.faults = {}               # действующие сбои: вид -> до какого времени
        self._stuck = None             # показание, на котором «залип» датчик
        self._noise_sigma = self.rules.sensor_fault_sigma
        self._last_reading = 0.0
        self._th = None
        self.counts = {'collision': 0, 'false_collect': 0, 'hazard_hit': 0}
        self.done = False
        self.reason = None             # finish | battery | timeout
        self.returned = False
        self.world_log = []            # применённые изменения среды: для записи прогона, агенту не видны
        self._events = []              # очередь на публикацию в /did/events
        self._pose = None
        self._colliding = False
        self._hazard_hit_t = {}

    # --- ход времени -------------------------------------------------------------------------

    @property
    def sensor_sigma(self):
        return self._noise_sigma if 'sensor_noise' in self.faults else self.rules.sensor_sigma

    def step(self, t, x, y, blocked=False, th=None):
        """Робот в момент t (секунды от начала прогона) находится в (x, y), курс th (если известен)."""
        if self.done:
            return
        dt = max(0.0, t - self.t)
        ds = 0.0 if self._pose is None else math.dist(self._pose, (x, y))
        mx, my = (x, y) if self._pose is None else ((x + self._pose[0]) / 2, (y + self._pose[1]) / 2)
        self._apply_world_events(t)

        r = self.rules
        load = 1.0 + r.load_drain * len(self.collected)             # несомые образцы утяжеляют робота
        self.battery -= r.drain_per_m * soil_mult(self.soils, mx, my) * load * ds + r.drain_idle_per_s * dt
        if th is not None:
            if self._th is not None:
                self.battery -= r.drain_per_rad * abs((th - self._th + math.pi) % (2 * math.pi) - math.pi)
            self._th = th
        if 'leak' in self.faults:
            self.battery -= r.leak_per_s * dt
        self.distance += ds
        self.t = t
        self._pose = (x, y)

        clearance = self.arena.clearance(x, y)
        if not self._colliding and (blocked or clearance < r.collision_clearance_m):
            self._colliding = True
            self._penalty('collision', x, y)
        elif self._colliding and not blocked and clearance > r.collision_clearance_m + 0.03:
            self._colliding = False

        for z in self.hazards:
            if z.contains(x, y):
                last = self._hazard_hit_t.get(z.id)
                if last is None or t - last >= r.hazard_repeat_s:
                    self._hazard_hit_t[z.id] = t
                    self.battery -= r.hazard_battery_hit
                    self._penalty('hazard_hit', x, y)
                    if r.faults and z.fault:
                        self._start_fault(z.fault, t, z.fault_s, source=z.id)
            else:
                self._hazard_hit_t.pop(z.id, None)

        if self.battery <= 0.0:
            self.battery = 0.0
            self._end('battery')
        elif t >= r.time_limit_s:
            self._end('timeout')

    def _start_fault(self, kind, t, duration, source):
        if '+' in kind:                 # два сбоя разом: батарея и датчик ломаются независимо
            for part in kind.split('+'):
                self._start_fault(part, t, duration, source)
            return
        if kind == 'sensor_stuck' and kind not in self.faults:
            self._stuck = self._last_reading
        if kind == 'sensor_noise':
            self._noise_sigma = self.rules.sensor_fault_sigma
        self.faults[kind] = max(self.faults.get(kind, 0.0), t + duration)
        self.world_log.append({'t': round(t, 2), 'type': 'fault', 'kind': kind, 'until': round(t + duration, 2),
                               'source': source})

    def _apply_world_events(self, t):
        for kind, until in list(self.faults.items()):
            if t >= until:
                del self.faults[kind]
                # прежнее имя события сохранено для сбоя датчика из расписания сценария
                self.world_log.append({'t': round(t, 2), 'type': 'sensor_recovered' if kind.startswith('sensor')
                                       else 'fault_end', 'kind': kind})
        while self.pending and self.pending[0]['t'] <= t:
            ev = self.pending.pop(0)
            entry = {'t': round(t, 2), 'type': ev['type']}
            if ev['type'] == 'soil_change':
                self.soils = list(ev['soils'])
            elif ev['type'] == 'new_hazard':
                self.hazards.append(ev['zone'])
            elif ev['type'] == 'sensor_fault':
                kind = ev.get('kind', 'sensor_noise') if self.rules.faults else 'sensor_noise'
                if kind == 'sensor_stuck':
                    self._stuck = self._last_reading
                if kind == 'sensor_noise':
                    self._noise_sigma = ev.get('sigma', self.rules.sensor_fault_sigma)
                self.faults[kind] = ev['t'] + ev['duration']
                entry['kind'] = kind
            self.world_log.append(entry)

    # --- интерфейс агента --------------------------------------------------------------------

    def read_sensor(self, x, y):
        """Близость к ближайшему несобранному образцу: 1 вплотную, 0 дальше дальности датчика."""
        if 'sensor_stuck' in self.faults:
            return float(self._stuck)
        if self.remaining:
            d = min(math.dist((x, y), p) for p in self.remaining.values())
            law = self.rules.sensor_law
            u = d / self.rules.sensor_range_m
            if law == 'quadratic':
                clean = max(0.0, 1.0 - u * u)
            elif law == 'sqrt':
                clean = max(0.0, 1.0 - math.sqrt(u))
            else:
                clean = max(0.0, 1.0 - u)
        else:
            clean = 0.0
        if 'sensor_bias' in self.faults:
            clean -= self.rules.sensor_bias
        self._last_reading = float(np.clip(clean + self.rng.normal(0.0, self.sensor_sigma), 0.0, 1.0))
        return self._last_reading

    def read_battery(self):
        """Показание /did/battery: заряд с шумом измерения."""
        if not self.rules.battery_sigma:
            return self.battery
        return max(0.0, self.battery + float(self.rng_battery.normal(0.0, self.rules.battery_sigma)))

    def collect(self, x, y):
        if self.done:
            return False, 'прогон завершён'
        best = min(self.remaining.items(), key=lambda kv: math.dist((x, y), kv[1]), default=None)
        if best and math.dist((x, y), best[1]) <= self.rules.collect_radius_m:
            idx, pos = best
            del self.remaining[idx]
            self.collected.append((idx, round(self.t, 2)))
            self._events.append({'type': 'sample_collected', 't': round(self.t, 2), 'x': round(x, 3),
                                 'y': round(y, 3), 'sample': idx, 'collected': len(self.collected)})
            return True, f'образец собран, всего {len(self.collected)}'
        self._penalty('false_collect', x, y)
        return False, 'рядом нет образца'

    def finish(self, x, y):
        if self.done:
            return False, 'прогон уже завершён'
        self.returned = math.dist((x, y), self.scenario.base) <= self.rules.base_radius_m
        self._end('finish')
        return self.returned, 'возврат на базу засчитан' if self.returned else 'робот не на базе'

    def pop_events(self):
        out, self._events = self._events, []
        return out

    def score(self):
        r, c = self.rules, self.counts
        points = (r.pts_sample * len(self.collected)
                  + r.pts_collision * c['collision']
                  + r.pts_false_collect * c['false_collect']
                  + r.pts_hazard_hit * c['hazard_hit'])
        if self.returned:
            points += r.pts_return + r.pts_battery_left * self.battery
        return {
            'level': self.scenario.level,
            't': round(self.t, 2),
            'battery': round(self.battery, 3),
            'samples_collected': len(self.collected),
            'samples_total': len(self.scenario.samples),
            'collisions': c['collision'],
            'false_collects': c['false_collect'],
            'hazard_hits': c['hazard_hit'],
            'distance': round(self.distance, 3),
            'finished': self.done,
            'reason': self.reason,
            'returned': self.returned,
            'score': round(points, 2),
        }

    # --- внутреннее --------------------------------------------------------------------------

    def _penalty(self, kind, x, y):
        self.counts[kind] += 1
        self._events.append({'type': kind, 't': round(self.t, 2), 'x': round(x, 3), 'y': round(y, 3)})

    def _end(self, reason):
        self.done = True
        self.reason = reason
