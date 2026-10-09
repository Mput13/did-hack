"""Память о лаборатории между прогонами (L4b): где были опасные зоны и дорогой грунт.

did/memory.py хранит числа модели расхода и длительности сбоев; мест в ней нет. Здесь — места: робот,
который уже работал на этой арене, начинает следующий прогон не с чистого листа.

  LabMemory     — файл рядом с прогонами. В него идёт только то, что робот узнал сам: зоны, в которых он получил
                  штраф (вместе с оставшимися гипотезами «центр и радиус»), и клетки пола, по которым он ездил,
                  с измеренной кратностью расхода. У каждого знания — прогоны, из которых оно взято. Скрытой
                  правды сценария и мест образцов в памяти нет.
  LabPrior      — то же внутри агента на время прогона: предположения с уверенностью, а не истина. Зона из
                  памяти удорожает клетки маршрута, но не запрещает их; проезд без штрафа отбрасывает гипотезы
                  о зоне, и когда не осталось ни одной — зона снята. Штраф рядом зону подтверждает.
  LabSoilModel  — карта грунта, в которой память — слабое исходное предположение: оно гаснет там, где робот
                  набрал свои замеры, и стирается вместе с остальным, когда расход разошёлся с прогнозом.

Уверенность падает и между прогонами (знание, которое не перепроверялось, стареет), и по ходу прогона — если
в прошлых прогонах этой лаборатории робот сам замечал, что среда меняется.
"""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .belief import HazardMap, SoilModel


@dataclass(frozen=True)
class Settings:
    """Числа подобраны на отладочных сценариях 1–80 (research/findings/L4b.md)."""
    hazard_conf: float = 0.75       # уверенность в зоне, где штраф был в одном прогоне
    hazard_conf_many: float = 0.9   # и в двух и больше
    hazard_margin: float = 0.05     # запас вокруг зоны из памяти сверх обычного (HazardMap.MARGIN), м
    soil_conf: float = 0.7          # уверенность в клетке пола, измеренной в одном прогоне
    soil_conf_many: float = 0.9     # и в двух и больше, если замеры сошлись
    soil_weight_m: float = 0.2      # вес клетки из памяти при полной уверенности — как один проезд по ней, м
    soil_agree: float = 0.5         # замеры двух прогонов сошлись, если кратности отличаются не больше
    floor: bool = True              # помнить и обычный пол (False — только дорогие клетки)
    age_keep: float = 0.85          # во сколько раз падает уверенность за каждый прогон без перепроверки
    decay_s: float = 240.0          # по ходу прогона уверенность падает в e раз за столько секунд, если среда
                                    # в этой лаборатории уже менялась на глазах робота (0 — не падает)
    hazards: bool = True            # что из памяти использовать (для разбора, откуда выигрыш)
    soils: bool = True


EXPENSIVE = 1.45        # с какой кратности клетка считается дорогим грунтом (как SoilModel.zones)
MIN_CELL_M = 0.03       # клетка измерена, если по ней пройдено не меньше


class LabMemory:
    """Память о лаборатории в файле. learn — после прогона, priors — перед следующим."""

    def __init__(self, path=None, settings=None):
        self.path = Path(path) if path else None
        self.s = settings or Settings()
        self.data = {'version': 1, 'updated': None, 'runs': [], 'grid': None, 'hazards': [], 'dropped': [],
                     'soil': {}, 'soil_zones': []}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))

    # --- накопление --------------------------------------------------------------------------

    def learn(self, export, run_id):
        """export — то, что агент выгрузил после прогона (LabPrior.export)."""
        if not export:
            return
        d = self.data
        k = len(d['runs'])                       # номер этого прогона в памяти
        if d['grid'] is None:
            d['grid'] = export['grid']
        elif d['grid'] != export['grid']:
            raise ValueError('память о лаборатории снята на другой сетке карты')
        confirmed = {z['confirms'] for z in export['hazards'] if z.get('confirms')}
        refuted = 0
        for p in export['priors']:               # что стало с зонами, взятыми из памяти
            h = next((h for h in d['hazards'] if h['id'] == p['id']), None)
            if h is None or h['id'] in confirmed:
                continue
            if p['status'] == 'refuted':
                refuted += 1
                d['hazards'].remove(h)
                d['dropped'].append({**{q: h[q] for q in ('id', 'x', 'y', 'r', 'runs')}, 'refuted_by': run_id,
                                     'why': 'робот проехал это место без штрафа'})
                continue
            h['cloud'] = p['cloud']              # проезды рядом сузили гипотезы о месте
            h['x'], h['y'], h['r'] = p['x'], p['y'], p['r']
            h['age'] += 1
            h['untested'] = h['untested'] + [run_id]
        for z in export['hazards']:              # зоны, где в этом прогоне пришёл штраф
            h = next((h for h in d['hazards'] if h['id'] == z.get('confirms')), None)
            if h is None:                        # та же зона могла быть в памяти, но снята с учёта раньше штрафа
                h = next((h for h in d['hazards'] if math.hypot(h['x'] - z['x'], h['y'] - z['y']) <= 0.4), None)
            if h is None:
                h = {'id': f"Z{len(d['hazards']) + len(d['dropped']) + 1}", 'evidence': 'penalty', 'hits': 0,
                     'runs': [], 'untested': [], 't_hit': z['t_hit'], 'appeared_after': None}
                d['hazards'].append(h)
            h.update(x=z['x'], y=z['y'], r=z['r'], cloud=z['cloud'], age=0)
            h['hits'] += z['hits']
            h['runs'] = h['runs'] + [run_id]
            h['t_hit'] = min(h['t_hit'], z['t_hit'])
            if z.get('appeared_after') is not None:
                h['appeared_after'] = max(h['appeared_after'] or 0.0, z['appeared_after'])
        changed = 0
        seen = set()
        for iy, ix, metres, mult in export['soil']:
            key = f'{iy},{ix}'
            seen.add(key)
            c = d['soil'].get(key)
            if c is not None and abs(c['k'] - mult) <= self.s.soil_agree:
                w = c['m'] + metres
                c.update(k=round((c['k'] * c['m'] + mult * metres) / w, 2), m=round(min(w, 1.0), 2), n=c['n'] + 1,
                         age=0, last=k)
            else:
                if c is not None and max(c['k'], mult) >= EXPENSIVE:
                    changed += 1                 # пол в этой клетке оказался не таким, каким запомнен
                d['soil'][key] = {'k': round(mult, 2), 'm': round(metres, 2), 'n': 1, 'age': 0, 'first': k, 'last': k}
        for key in [q for q in d['soil'] if q not in seen]:
            c = d['soil'][key]
            c['age'] += 1
            if self.s.age_keep ** c['age'] < 0.2:
                del d['soil'][key]
        d['runs'].append({'id': run_id, 't_end': export.get('t_end'), 'penalty_zones': len(export['hazards']),
                          'mismatch': export['changes']['mismatch'], 'zones_refuted': refuted,
                          'soil_refuted': export['changes'].get('soil_refuted', 0),
                          'soil_cells_changed': changed})
        d['soil_zones'] = self._soil_zones()
        d['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    def _soil_conf(self, c):
        base = self.s.soil_conf if c['n'] < 2 else self.s.soil_conf_many
        return base * min(1.0, c['m'] / 0.1) * self.s.age_keep ** c['age']

    def _hazard_conf(self, h):
        base = self.s.hazard_conf if len(h['runs']) < 2 else self.s.hazard_conf_many
        return base * self.s.age_keep ** h['age']

    def _soil_zones(self):
        """Связные участки дорогого грунта по клеткам памяти — вид для человека и для планировщика."""
        g = self.data['grid']
        if not g or not self.data['soil']:
            return []
        hot = np.zeros((g['h'], g['w']), dtype=bool)
        for key, c in self.data['soil'].items():
            iy, ix = map(int, key.split(','))
            hot[iy, ix] = c['k'] >= EXPENSIVE
        labels, n = ndimage.label(hot, structure=np.ones((3, 3)))
        ids = [r['id'] for r in self.data['runs']]
        out = []
        for i in range(1, n + 1):
            iy, ix = np.nonzero(labels == i)
            cells = [self.data['soil'][f'{a},{b}'] for a, b in zip(iy, ix)]
            w = np.array([c['m'] for c in cells])
            if w.sum() < 0.12:
                continue
            runs = sorted({q for c in cells for q in (c['first'], c['last'])})
            out.append({'x': round(float(g['x0'] + (np.average(ix, weights=w) + 0.5) * g['res']), 2),
                        'y': round(float(g['y0'] + (np.average(iy, weights=w) + 0.5) * g['res']), 2),
                        'radius': round(float(max(g['res'], math.sqrt(len(ix) * g['res'] ** 2 / math.pi))), 2),
                        'mult': round(float(sum(c['k'] * c['m'] for c in cells) / w.sum()), 2),
                        'evidence_m': round(float(w.sum()), 2),
                        'confidence': round(float(np.mean([self._soil_conf(c) for c in cells])), 2),
                        'runs': [ids[q] if q < len(ids) else None for q in runs]})
        return out

    # --- использование -----------------------------------------------------------------------

    def priors(self):
        """Исходные предположения для следующего прогона (LabPrior). None — память пуста."""
        d = self.data
        if not d['runs']:
            return None
        hazards = [{'id': h['id'], 'x': h['x'], 'y': h['y'], 'r': h['r'], 'cloud': h['cloud'],
                    'conf': round(self._hazard_conf(h), 3), 'hits': h['hits'], 'runs': h['runs'],
                    # проезды раньше этого времени зону не опровергают: в прошлом прогоне её тогда ещё не было
                    't_from': h['t_hit'] if h['appeared_after'] is not None else 0.0}
                   for h in d['hazards']]
        soil = [[*map(int, key.split(',')), c['k'], round(self._soil_conf(c), 3)] for key, c in d['soil'].items()]
        return {'runs': len(d['runs']), 'grid': d['grid'], 'hazards': hazards, 'soil': soil,
                'soil_zones': self._soil_zones(),
                # среда в этой лаборатории менялась на глазах робота: память по ходу прогона стареет
                'volatile': any(r['mismatch'] or r['zones_refuted'] or r.get('soil_refuted') or r['soil_cells_changed']
                                for r in d['runs'])}

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


class LabSoilModel(SoilModel):
    """Карта грунта с исходным предположением из памяти. Без предположения считает ровно как SoilModel."""

    OWN_M = 0.09          # столько своих замеров (после сглаживания — один проезд), и память в клетке не учитывается
    REFUTE_M = 0.15       # на стольких метрах пути расход разошёлся с памятью — участок из памяти снимается целиком
    REFUTE_R = 0.4        # для обычного пола из памяти (участка нет) — в каком радиусе, м

    def __init__(self, arena, drain_per_m, idle_per_s):
        super().__init__(arena, drain_per_m, idle_per_s)
        self.prior_w = None           # вес памяти по клеткам, м
        self.prior_wk = None          # вес × кратность
        self.fade = 1.0               # общий множитель уверенности: падает по ходу прогона
        self.refuted = []             # снятые участки памяти: (x, y, кратность по памяти, измеренная)
        self._smooth = None
        self._patch = None            # номер связного дорогого участка памяти по клеткам (0 — обычный пол)
        self._off = {}                # сколько метров подряд расход расходится с памятью, по участкам

    def set_prior(self, cells, weight_m):
        """cells — [[iy, ix, кратность, уверенность]]."""
        self.prior_w = np.zeros((self.h, self.w))
        self.prior_wk = np.zeros((self.h, self.w))
        for iy, ix, mult, conf in cells:
            if 0 <= iy < self.h and 0 <= ix < self.w:
                self.prior_w[iy, ix] = weight_m * conf
                self.prior_wk[iy, ix] = weight_m * conf * mult
        mult = np.divide(self.prior_wk, self.prior_w, out=np.ones_like(self.prior_w), where=self.prior_w > 0)
        self._patch, _ = ndimage.label(mult >= EXPENSIVE, structure=np.ones((3, 3)))
        self._touch()

    def observe(self, x0, y0, x1, y1, spent, dt):
        out = super().observe(x0, y0, x1, y1, spent, dt)
        self._compare((x0 + x1) / 2, (y0 + y1) / 2, math.hypot(x1 - x0, y1 - y0), out[0])
        return out

    def add(self, x, y, ds, ratio):
        super().add(x, y, ds, ratio)
        self._compare(x, y, ds, ratio)

    def _compare(self, x, y, ds, ratio):
        """Свой замер против памяти: робот проехал по участку, а расход не тот, что запомнен, — участок снят."""
        if self.prior_w is None:
            return
        c = self.cell(x, y)
        if self.prior_w[c] <= 0:
            return
        k = float(self.prior_wk[c] / self.prior_w[c])
        patch = int(self._patch[c])
        off = self._off.get(patch, 0.0) + (ds if abs(ratio - k) > max(0.5, 0.3 * k) else -ds)
        self._off[patch] = max(off, 0.0)
        if off < self.REFUTE_M:
            return
        if patch:
            gone = self._patch == patch
        else:
            ys = self.y0 + (np.arange(self.h) + 0.5) * self.res
            xs = self.x0 + (np.arange(self.w) + 0.5) * self.res
            gone = (np.hypot(xs[None, :] - x, ys[:, None] - y) <= self.REFUTE_R) & (self._patch == 0)
        self.prior_w[gone] = 0.0
        self.prior_wk[gone] = 0.0
        self._off[patch] = 0.0
        self.refuted.append((x, y, k, ratio))
        self._touch()

    def set_fade(self, fade):
        if self.prior_w is not None and fade != self.fade:
            self.fade = fade
            self.version += 1
            self._cache = None

    def _touch(self):
        self._smooth = None
        self._cache = None
        self.version += 1

    def _estimate(self):
        if self.prior_w is None:
            return super()._estimate()
        if self._cache is None:
            if self._smooth is None:
                self._smooth = (ndimage.gaussian_filter(self.prior_w, self.SMOOTH, mode='constant'),
                                ndimage.gaussian_filter(self.prior_wk, self.SMOOTH, mode='constant'))
            gd = ndimage.gaussian_filter(self.dist, self.SMOOTH, mode='constant')
            gr = ndimage.gaussian_filter(self.drain, self.SMOOTH, mode='constant')
            keep = self.fade * np.clip(1.0 - gd / self.OWN_M, 0.0, 1.0)      # свои замеры вытесняют память
            pw, pwk = self._smooth[0] * keep, self._smooth[1] * keep
            self._cache = ((gr + pwk + self.PRIOR_M) / (gd + pw + self.PRIOR_M), (gd + pw) / (gd + pw + self.PRIOR_M))
        return self._cache

    def prior_at(self, x, y):
        """(кратность по памяти, её вес) в клетке — для записи в журнал, что именно опровергнуто."""
        if self.prior_w is None:
            return 1.0, 0.0
        c = self.cell(x, y)
        w = float(self.prior_w[c])
        return (float(self.prior_wk[c]) / w if w > 0 else 1.0), w

    def forget(self, x, y, radius, keep=0.1, keep_elsewhere=1.0):
        near = super().forget(x, y, radius, keep, keep_elsewhere)
        if self.prior_w is not None:
            # Расход разошёлся с прогнозом: рядом память снимается совсем, в остальных местах ей веры меньше.
            scale = np.where(near, 0.0, keep_elsewhere)
            self.prior_w *= scale
            self.prior_wk *= scale
            self._touch()
        return near


class LabPrior:
    """Память о лаборатории внутри агента на время одного прогона."""

    REFRESH_S = 10.0      # как часто пересчитывать падение уверенности по ходу прогона

    def __init__(self, agent, priors=None):
        self.a = agent
        self.s = Settings(**(agent.cfg.lab_opts or {}))
        priors = priors or {}
        self.runs = priors.get('runs', 0)
        self.volatile = bool(priors.get('volatile'))
        self.zones = []               # зоны из памяти: гипотезы о месте и уверенность, что зона вообще есть
        self.mismatch = 0
        self.soil_refuted = 0
        self.soil_zones = priors.get('soil_zones', []) if self.s.soils else []
        self._found = []              # зоны этого прогона: [{'z': зона HazardMap, 't', 'hits', 'confirms'}]
        self._trail = []              # все точки, пройденные без штрафа: (x, y, t)
        self._fade = 1.0
        self._fade_t = 0.0
        self._grid = None             # кэш карты риска
        soil = agent.soil
        if priors and priors.get('grid') != self.grid():
            raise ValueError('память о лаборатории снята на другой сетке карты')
        if self.s.soils and priors.get('soil') and isinstance(soil, LabSoilModel):
            cells = [c for c in priors['soil'] if self.s.floor or c[2] >= EXPENSIVE]
            soil.set_prior(cells, self.s.soil_weight_m)
            agent._cost_dirty = True
        for h in (priors.get('hazards', []) if self.s.hazards else []):
            cloud = np.array(h['cloud'], dtype=float).reshape(-1, 4)
            if not len(cloud):
                continue
            self.zones.append({'id': h['id'], 'c': cloud[:, :2], 'r': cloud[:, 2], 'w': cloud[:, 3],
                               'w0': float(cloud[:, 3].sum()), 'conf': float(h['conf']), 'status': 'untested',
                               't_from': float(h.get('t_from') or 0.0), 'x': h['x'], 'y': h['y'], 'radius': h['r'],
                               'runs': h.get('runs', [])})
        if self.zones:
            agent._cost_dirty = True
        if self.runs:
            agent.journal.add(0.0, 'observe',
                              f'Память о лаборатории из {self.runs} прошлых прогонов: опасных зон — {len(self.zones)}, '
                              f'участков дорогого грунта — {len(self.soil_zones)}. Это предположения: объезжаю и '
                              'сверяю с тем, что увижу сам', tag='lab_memory')

    def grid(self):
        s = self.a.soil
        return {'res': round(float(s.res), 4), 'x0': round(float(s.x0), 4), 'y0': round(float(s.y0), 4),
                'w': int(s.w), 'h': int(s.h)}

    # --- опасные зоны ------------------------------------------------------------------------

    def active(self):
        return [z for z in self.zones if z['status'] in ('untested', 'narrowed')]

    def _exists(self, z):
        """Вероятность, что зона на месте: исходная уверенность, поправленная на отпавшие гипотезы о месте."""
        left = float(z['w'].sum()) / z['w0']
        return z['conf'] * left / (z['conf'] * left + 1.0 - z['conf'])

    def safe(self, x, y, t):
        """Робот проехал (x, y) без штрафа: гипотезы о зоне из памяти, накрывающие эту точку, отпадают."""
        self._trail.append((x, y, t))
        for z in self.active():
            if t < z['t_from']:
                continue
            keep = np.hypot(z['c'][:, 0] - x, z['c'][:, 1] - y) >= z['r'] - 0.02
            if keep.all():
                continue
            z['c'], z['r'], z['w'] = z['c'][keep], z['r'][keep], z['w'][keep]
            z['status'] = 'narrowed'
            if not keep.any():
                z['status'] = 'refuted'
                self.a.journal.add(t, 'verdict', f"Зоны из памяти около ({z['x']:.1f}; {z['y']:.1f}) нет: проехал это "
                                   'место без штрафа. Предположение снято', tag='lab_refuted', zone=z['id'])
            self._changed()

    def on_hit(self, obs, zone):
        """Штраф: zone — зона HazardMap, которую он создал или уточнил."""
        rec = next((f for f in self._found if f['z'] is zone), None)
        if rec is None:
            rec = {'z': zone, 't': obs.t, 'hits': 0, 'confirms': None}
            self._found.append(rec)
        rec['hits'] += 1
        for z in self.active():
            near = np.hypot(z['c'][:, 0] - obs.x, z['c'][:, 1] - obs.y) <= z['r'] + 0.10
            if near.any() or math.hypot(z['x'] - obs.x, z['y'] - obs.y) <= 0.6:
                z['status'] = 'confirmed'
                rec['confirms'] = rec['confirms'] or z['id']
                self.a.journal.add(obs.t, 'verdict', f"Зона из памяти около ({z['x']:.1f}; {z['y']:.1f}) подтверждена "
                                   'штрафом: она на месте', tag='lab_confirmed', zone=z['id'])
                self._changed()

    def _cover(self, z, x, y):
        """Какая доля гипотез о зоне накрывает точку (с обычным запасом)."""
        inside = np.hypot(z['c'][:, 0] - x, z['c'][:, 1] - y) <= z['r'] + HazardMap.MARGIN
        return float(z['w'][inside].sum() / z['w'].sum())

    def _open_needed(self, t):
        """Въезд по необходимости: цель — место образца, выбранное по датчику, или база — лежит внутри зоны из
        памяти. Объезжать такую зону бессмысленно: в неё всё равно въезжать. Образец стоит дороже штрафа, а
        без штрафа проезд зону опровергнет — поэтому до конца прогона она маршруту не мешает."""
        sg = self.a.queue[0] if self.a.queue else None
        if sg is None or sg['type'] not in ('investigate', 'return_base'):
            return
        x, y = self.a.base if sg['type'] == 'return_base' else (sg['x'], sg['y'])
        for z in self.active():
            if not z.get('open') and self._cover(z, x, y) >= 0.5:
                z['open'] = True
                what = 'база' if sg['type'] == 'return_base' else 'место образца'
                self.a.journal.add(t, 'decision', f"Зона из памяти около ({z['x']:.1f}; {z['y']:.1f}) накрывает цель "
                                   f'({what}): еду проверять, объезд здесь ничего не даёт', tag='lab_open', zone=z['id'])
                self._changed()

    def _changed(self):
        self._grid = None
        self.a._cost_dirty = True
        self.a._cost_t = -1e9

    def risk(self, own):
        """Риск по клеткам: свой (по штрафам этого прогона) и по памяти — что больше."""
        zones = [z for z in self.active() if not z.get('open')]
        if not zones:
            return own
        if self._grid is None:
            X, Y = self.a._X, self.a._Y
            out = np.zeros(X.shape)
            reach = HazardMap.MARGIN + self.s.hazard_margin
            for z in zones:
                p = np.zeros(X.shape)
                for (cx, cy), r, w in zip(z['c'], z['r'], z['w']):
                    p += w * (np.hypot(X - cx, Y - cy) <= r + reach)
                out = np.maximum(out, self._exists(z) * self._fade * p / z['w'].sum())
            self._grid = out
        return np.maximum(own, self._grid)

    # --- ход прогона -------------------------------------------------------------------------

    def tick(self, t):
        """Раз в секунду: уверенность в памяти падает со временем прогона, если среда здесь уже менялась."""
        self._open_needed(t)
        soil = self.a.soil
        while isinstance(soil, LabSoilModel) and len(soil.refuted) > self.soil_refuted:
            x, y, k, seen = soil.refuted[self.soil_refuted]
            self.soil_refuted += 1
            self.a._cost_dirty = True
            self.a.journal.add(t, 'verdict', f'Память о поле около ({x:.1f}; {y:.1f}) не подтвердилась: помнил ×{k:.1f}, '
                               f'вижу ×{seen:.1f}. Участок из памяти снят', tag='lab_soil_refuted')
        if not self.volatile or self.s.decay_s <= 0 or t - self._fade_t < self.REFRESH_S:
            return
        self._fade_t = t
        self._fade = math.exp(-t / self.s.decay_s)
        if isinstance(soil, LabSoilModel):
            soil.set_fade(self._fade)
        self._grid = None
        self.a._cost_dirty = True

    def on_mismatch(self, x, y, t, prior):
        """Расход разошёлся с прогнозом. prior — (кратность по памяти, вес) в этом месте до стирания."""
        self.mismatch += 1
        if prior[1] > 0:
            self.a.journal.add(t, 'verdict', f'Память о грунте около ({x:.1f}; {y:.1f}) (×{prior[0]:.1f}) не сошлась с '
                               'расходом: рядом она стёрта, в остальных местах ей веры меньше', tag='lab_soil_refuted')

    def summary(self):
        """Короткая сводка для планировщика (поле lab_memory в снимке состояния)."""
        word = {'untested': 'не проверялась', 'narrowed': 'место уточнено проездами рядом',
                'confirmed': 'подтверждена штрафом в этом прогоне', 'refuted': 'снята: проехал без штрафа'}
        return {'past_runs': self.runs,
                'note': 'сведения из прошлых прогонов в этой лаборатории; это предположения, а не истина',
                'changes_seen_before': self.volatile,
                # свои зоны этого прогона в снимке зовутся Z1, Z2…; у зон из памяти другая буква
                'hazards': [{'id': 'M' + z['id'][1:], 'x': round(z['x'], 2), 'y': round(z['y'], 2), 'radius': round(z['radius'], 2),
                             'confidence': round(self._exists(z) * self._fade, 2) if z['status'] in
                             ('untested', 'narrowed') else (1.0 if z['status'] == 'confirmed' else 0.0),
                             'penalty_in_runs': len(z['runs']), 'status': word[z['status']]} for z in self.zones],
                'soil_zones': [{'x': q['x'], 'y': q['y'], 'radius': q['radius'], 'mult': q['mult'],
                                'confidence': round(q['confidence'] * self._fade, 2)} for q in self.soil_zones]}

    # --- выгрузка после прогона --------------------------------------------------------------

    @staticmethod
    def _cloud(z):
        w = z['w'] / z['w'].sum()
        cx, cy = float(w @ z['c'][:, 0]), float(w @ z['c'][:, 1])
        spread = float(w @ np.hypot(z['c'][:, 0] - cx, z['c'][:, 1] - cy))
        cloud = [[round(float(a), 3), round(float(b), 3), round(float(r), 3), round(float(q), 4)]
                 for (a, b), r, q in zip(z['c'], z['r'], w)]
        return {'x': round(cx, 2), 'y': round(cy, 2), 'r': round(float(w @ z['r']) + spread, 2), 'cloud': cloud}

    def _appeared_after(self, rec):
        """Когда робот в последний раз до штрафа проезжал место, которое накрывают все оставшиеся гипотезы о
        зоне: значит, тогда её ещё не было. None — такого проезда не было."""
        z = rec['z']
        best = None
        for x, y, t in self._trail:
            if t < rec['t'] - 1.0 and (np.hypot(z['c'][:, 0] - x, z['c'][:, 1] - y) < z['r'] - 0.02).all():
                best = t
        return best

    def export(self, t_end):
        """Что робот узнал о лаборатории сам: для LabMemory.learn."""
        soil = self.a.soil
        iy, ix = np.nonzero(soil.dist > MIN_CELL_M)
        cells = [[int(a), int(b), round(float(soil.dist[a, b]), 3),
                  round(float(np.clip(soil.drain[a, b] / soil.dist[a, b], 0.5, 6.0)), 2)] for a, b in zip(iy, ix)]
        hazards = []
        for rec in self._found:
            if not len(rec['z']['w']):
                continue
            after = self._appeared_after(rec)
            hazards.append({**self._cloud(rec['z']), 't_hit': round(rec['t'], 1), 'hits': rec['hits'],
                            'confirms': rec['confirms'], 'appeared_after': None if after is None else round(after, 1)})
        priors = [{'id': z['id'], 'status': z['status'],
                   **(self._cloud(z) if len(z['w']) else {'x': z['x'], 'y': z['y'], 'r': z['radius'], 'cloud': []})}
                  for z in self.zones]
        return {'grid': self.grid(), 't_end': round(float(t_end), 1), 'changes': {'mismatch': self.mismatch,
                                                                              'soil_refuted': len(getattr(
                                                                                  soil, 'refuted', []))},
                'hazards': hazards, 'priors': priors, 'soil': cells if self.a.cfg.learn_soil else []}


def settings_dict(**over):
    return asdict(Settings(**over))
