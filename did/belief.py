"""Модель мира агента: где могут лежать образцы, сколько стоит грунт, исправен ли датчик.

Всё строится только из того, что агент измеряет сам: показаний /did/sample_sensor, расхода
/did/battery на пройденный путь и собственных координат.
"""
import math
from collections import deque

import numpy as np
from scipy import ndimage
from scipy.special import ndtr


class SampleBelief:
    """Сетка 0,1 м: вероятность, что в клетке лежит несобранный образец.

    Датчик сообщает близость только к ближайшему образцу. Одно показание значит две вещи:
    ближе измеренного расстояния образцов нет, а на этом расстоянии (на кольце) хотя бы один есть.
    Формула Байеса ниже учитывает обе, и «кольцо» делит между клетками ровно один образец.
    """

    SUB = 2   # клетка убеждений = 2×2 клетки карты

    def __init__(self, arena, n_expected, sensor_range, margin=0.15):
        s = self.SUB
        self.res = arena.res * s
        self.x0, self.y0 = arena.x0, arena.y0
        self.h, self.w = arena.h // s, arena.w // s
        clear = arena.clear[:self.h * s, :self.w * s].reshape(self.h, s, self.w, s).max(axis=(1, 3))
        self.mask = clear >= margin
        self.iy, self.ix = np.nonzero(self.mask)
        self.cx = self.x0 + (self.ix + 0.5) * self.res
        self.cy = self.y0 + (self.iy + 0.5) * self.res
        self.n = len(self.ix)
        self.range = sensor_range
        self.prior = min(0.5, n_expected / self.n)
        self.p = np.full(self.n, self.prior)
        self.left = n_expected          # сколько образцов ещё не собрано
        self.updates = 0
        self._epoch = self.p.copy()     # убеждения на момент последнего сбора
        self._log = []                  # показания с того момента: (x, y, z, sigma)

    def update(self, x, y, z, sigma):
        """Учесть показание z в точке (x, y) при шуме датчика sigma."""
        self._log.append((x, y, z, sigma))
        self._apply(x, y, z, sigma)
        self._normalize()
        self.updates += 1

    def _normalize(self):
        """Образцов осталось ровно left: суммарная вероятность по арене не может быть другой.

        Без этого «кольца» от старых показаний копятся и дают ложные пики.
        """
        total = self.p.sum()
        if total > 0:
            self.p = np.clip(self.p * (self.left / total), 1e-7, 0.995)

    def _apply(self, x, y, z, sigma):
        sigma = math.hypot(sigma, 0.4 * self.res / self.range)   # клетка конечного размера — тоже шум
        d = np.hypot(self.cx - x, self.cy - y)
        f = 1.0 - d / self.range
        idx = np.nonzero(f > 0.0)[0]
        if len(idx) == 0:
            return
        order = idx[np.argsort(-f[idx], kind='stable')]      # от ближних клеток к дальним
        p, f = self.p[order], f[order]
        if z <= 1e-6:            # показание упёрлось в ноль: вероятность, а не плотность
            like, like_none = ndtr(-f / sigma), 0.5
        elif z >= 1.0 - 1e-6:
            like, like_none = ndtr((f - 1.0) / sigma), ndtr(-1.0 / sigma)
        else:
            like = np.exp(-0.5 * ((z - f) / sigma) ** 2)
            like_none = math.exp(-0.5 * (z / sigma) ** 2)
        cum = np.cumsum(np.log1p(-p))
        closer_empty = np.exp(np.concatenate(([0.0], cum[:-1])))   # ближе этой клетки образцов нет
        a = p * closer_empty * like                                # «ближайший образец — в этой клетке»
        before = np.concatenate(([0.0], np.cumsum(a)[:-1]))
        total = a.sum() + math.exp(cum[-1]) * like_none
        num = closer_empty * like + before
        den = before + (total - before - a) / (1.0 - p)
        lr = np.clip(num / np.maximum(den, 1e-300), 1e-4, 1e4)
        odds = p * lr
        self.p[order] = np.clip(odds / (odds + 1.0 - p), 1e-7, 0.995)

    def trial(self, x, y, z, sigma, p=None, normalize=True):
        """Пробное обновление: какой стала бы карта после показания z в точке (x, y).

        Сама карта и журнал показаний не меняются. z — число или массив (D,): сразу D вариантов
        показания; p — исходная карта (n,) или (D, n), по умолчанию текущая. Возвращает (D, n).
        Формулы те же, что в _apply и _normalize (совпадение проверяет tests/test_explore.py).
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))[:, None]
        out = np.array(np.broadcast_to(self.p if p is None else p, (len(z), self.n)))
        sigma = math.hypot(sigma, 0.4 * self.res / self.range)
        f = 1.0 - np.hypot(self.cx - x, self.cy - y) / self.range
        idx = np.nonzero(f > 0.0)[0]
        if len(idx):
            order = idx[np.argsort(-f[idx], kind='stable')]
            q, f = out[:, order], f[order]
            like = np.exp(-0.5 * ((z - f) / sigma) ** 2)
            like_none = np.exp(-0.5 * (z / sigma) ** 2)
            lo, hi = z <= 1e-6, z >= 1.0 - 1e-6
            if lo.any():
                like, like_none = np.where(lo, ndtr(-f / sigma), like), np.where(lo, 0.5, like_none)
            if hi.any():
                like = np.where(hi, ndtr((f - 1.0) / sigma), like)
                like_none = np.where(hi, ndtr(-1.0 / sigma), like_none)
            cum = np.cumsum(np.log1p(-q), axis=1)
            zero = np.zeros((len(z), 1))
            closer_empty = np.exp(np.hstack([zero, cum[:, :-1]]))
            a = q * closer_empty * like
            before = np.hstack([zero, np.cumsum(a, axis=1)[:, :-1]])
            total = a.sum(axis=1, keepdims=True) + np.exp(cum[:, -1:]) * like_none
            den = before + (total - before - a) / (1.0 - q)
            lr = np.clip((closer_empty * like + before) / np.maximum(den, 1e-300), 1e-4, 1e4)
            odds = q * lr
            out[:, order] = np.clip(odds / (odds + 1.0 - q), 1e-7, 0.995)
        if normalize:
            out = np.clip(out * (self.left / out.sum(axis=1, keepdims=True)), 1e-7, 0.995)
        return out

    def collected(self, x, y):
        """Образец взят рядом с (x, y). Пересчитать убеждения с учётом того, что он там был.

        Показания, которые этот образец объясняет, больше не говорят «на таком расстоянии что-то
        есть» — только «ближе ничего нет». Поэтому убеждения откатываются к моменту прошлого сбора
        и показания проигрываются заново уже с этим знанием.
        """
        near = np.hypot(self.cx - x, self.cy - y) <= 0.35
        if near.any():
            k = np.nonzero(near)[0][self.p[near].argmax()]
            sx, sy = self.cx[k], self.cy[k]
        else:
            sx, sy = x, y
        self.p = self._epoch.copy()
        f_all = None
        for (rx, ry, z, sigma) in self._log:
            sig = math.hypot(sigma, 0.06)
            f_known = max(0.0, 1.0 - math.hypot(sx - rx, sy - ry) / self.range)
            if abs(z - f_known) > 3.5 * sig:
                self._apply(rx, ry, z, sigma)        # тогда ближайшим был другой образец
                continue
            f = 1.0 - np.hypot(self.cx - rx, self.cy - ry) / self.range
            closer = f > f_known
            lr = np.exp(np.minimum(0.0, -0.5 * ((z - f[closer]) / sig) ** 2 + 0.5 * ((z - f_known) / sig) ** 2))
            self.p[closer] = np.maximum(self.p[closer] * lr, 1e-7)
        self.left = max(0, self.left - 1)
        self.clear_disc(sx, sy, 0.35)
        self._normalize()
        self._epoch = self.p.copy()
        self._log = []

    def clear_disc(self, x, y, r, factor=0.0):
        """После сбора или промаха: в круге радиуса r образцов (почти) нет."""
        near = np.hypot(self.cx - x, self.cy - y) <= r
        self.p[near] = np.maximum(self.p[near] * factor, 1e-7)

    def relax(self, alpha):
        """Частично вернуться к исходному незнанию: показаниям за последнее время нельзя верить."""
        self.p = (1.0 - alpha) * self.p + alpha * self.prior

    def prob_within(self, x, y, r):
        near = np.hypot(self.cx - x, self.cy - y) <= r
        return float(1.0 - np.exp(np.log1p(-self.p[near]).sum())) if near.any() else 0.0

    def total(self):
        return float(self.p.sum())

    def grid(self):
        g = np.zeros((self.h, self.w))
        g[self.iy, self.ix] = self.p
        return g

    def _disc_sum(self, radius):
        k = int(math.ceil(radius / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        kernel = (np.hypot(xx, yy) * self.res <= radius).astype(float)
        return ndimage.convolve(self.grid(), kernel, mode='constant')

    def candidates(self, min_mass=0.3, radius=0.25, limit=5):
        """Места, где образец вероятен: локальные максимумы суммарной вероятности в круге radius."""
        mass = self._disc_sum(radius)
        peaks = (mass == ndimage.maximum_filter(mass, size=7)) & (mass >= min_mass) & self.mask
        iy, ix = np.nonzero(peaks)
        out = [{'x': float(self.x0 + (i + 0.5) * self.res), 'y': float(self.y0 + (j + 0.5) * self.res),
                'mass': float(1.0 - math.exp(-mass[j, i]))} for j, i in zip(iy, ix)]
        out.sort(key=lambda c: -c['mass'])
        return out[:limit]

    def explore_points(self, radius=0.8, step=0.5, limit=6):
        """Куда ехать за новой информацией: точки, вокруг которых осталось больше всего неясного."""
        mass = self._disc_sum(radius)
        stride = max(1, round(step / self.res))
        out = []
        for j in range(stride // 2, self.h, stride):
            for i in range(stride // 2, self.w, stride):
                if self.mask[j, i] and mass[j, i] > 0.02:
                    out.append({'x': float(self.x0 + (i + 0.5) * self.res),
                                'y': float(self.y0 + (j + 0.5) * self.res), 'mass': float(mass[j, i])})
        out.sort(key=lambda c: -c['mass'])
        return out[:limit * 3]


class SoilModel:
    """Сетка 0,2 м: во сколько раз расход батареи на метр выше обычного.

    В каждой клетке копятся пройденный путь и потраченный заряд. Оценка сглаживается по соседям:
    одна дорогая клетка делает подозрительными клетки рядом — это и есть гипотеза «здесь зона».
    """

    SUB = 4
    PRIOR_M = 0.02        # вес исходного предположения «пол обычный» (после сглаживания по соседям)
    SMOOTH = 0.9          # в клетках

    def __init__(self, arena, drain_per_m, idle_per_s):
        s = self.SUB
        self.arena = arena
        self.res = arena.res * s
        self.x0, self.y0 = arena.x0, arena.y0
        self.h, self.w = arena.h // s, arena.w // s
        self.dist = np.zeros((self.h, self.w))
        self.drain = np.zeros((self.h, self.w))
        self.per_m = drain_per_m
        self.idle = idle_per_s
        self.version = 0
        self._cache = None

    def cell(self, x, y):
        ix = min(max(int((x - self.x0) / self.res), 0), self.w - 1)
        iy = min(max(int((y - self.y0) / self.res), 0), self.h - 1)
        return iy, ix

    def observe(self, x0, y0, x1, y1, spent, dt):
        """Отрезок пути и заряд, потраченный на нём. Возвращает (множитель, прогноз, уверенность)."""
        ds = math.hypot(x1 - x0, y1 - y0)
        ratio = (spent - self.idle * dt) / (self.per_m * ds)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        predicted, confidence = self.at(mx, my)
        c = self.cell(mx, my)
        self.dist[c] += ds
        self.drain[c] += ratio * ds
        self.version += 1
        self._cache = None
        return ratio, predicted, confidence

    def add(self, x, y, ds, ratio):
        """Готовая оценка «во сколько раз дороже» на отрезке длиной ds около (x, y)."""
        c = self.cell(x, y)
        self.dist[c] += ds
        self.drain[c] += ratio * ds
        self.version += 1
        self._cache = None

    def _estimate(self):
        if self._cache is None:
            gd = ndimage.gaussian_filter(self.dist, self.SMOOTH, mode='constant')
            gr = ndimage.gaussian_filter(self.drain, self.SMOOTH, mode='constant')
            self._cache = ((gr + self.PRIOR_M) / (gd + self.PRIOR_M), gd / (gd + self.PRIOR_M))
        return self._cache

    def at(self, x, y):
        mult, conf = self._estimate()
        c = self.cell(x, y)
        return float(mult[c]), float(conf[c])

    def mult_grid(self):
        """Оценка множителя на сетке карты (для планировщика пути)."""
        mult, _ = self._estimate()
        return self._fine(np.clip(mult, 0.5, 6.0), fill=1.0)

    def confidence_grid(self):
        """Насколько участок проверен колёсами: 0 — там не ездили, ближе к 1 — ездили."""
        _, conf = self._estimate()
        return self._fine(conf, fill=0.0)

    def _fine(self, coarse, fill):
        fine = np.kron(coarse, np.ones((self.SUB, self.SUB)))
        out = np.full(self.arena.free.shape, fill)
        out[:fine.shape[0], :fine.shape[1]] = fine
        return out

    def forget(self, x, y, radius, keep=0.1, keep_elsewhere=1.0):
        """Модель устарела: вокруг (x, y) почти забыть накопленное, в остальных местах — ослабить."""
        ys = self.y0 + (np.arange(self.h) + 0.5) * self.res
        xs = self.x0 + (np.arange(self.w) + 0.5) * self.res
        near = np.hypot(xs[None, :] - x, ys[:, None] - y) <= radius
        scale = np.where(near, keep, keep_elsewhere)
        self.dist *= scale
        self.drain *= scale
        self.version += 1
        self._cache = None

    def zones(self, threshold=1.45, min_evidence=0.12):
        """Связные участки дорогого грунта, по которым агент уже ездил."""
        mult, _ = self._estimate()
        raw = np.divide(self.drain, self.dist, out=np.ones_like(self.drain), where=self.dist > 0.03)
        hot = (raw >= threshold) & (self.dist > 0.03)
        labels, n = ndimage.label(hot, structure=np.ones((3, 3)))
        out = []
        for k in range(1, n + 1):
            iy, ix = np.nonzero(labels == k)
            evidence = float(self.dist[iy, ix].sum())
            if evidence < min_evidence:
                continue
            w = self.dist[iy, ix]
            cx = self.x0 + (np.average(ix, weights=w) + 0.5) * self.res
            cy = self.y0 + (np.average(iy, weights=w) + 0.5) * self.res
            out.append({'x': round(float(cx), 2), 'y': round(float(cy), 2),
                        'radius': round(float(max(self.res, math.sqrt(len(ix) * self.res ** 2 / math.pi))), 2),
                        'mult': round(float(self.drain[iy, ix].sum() / evidence), 2),
                        'evidence_m': round(evidence, 2)})
        return out


class ChangeDetector:
    """Замечает, что расход батареи перестал совпадать с прогнозом модели (накопленная сумма, CUSUM).

    Считаются только участки, где агент уже был уверен в оценке: расхождение на незнакомом полу —
    это открытие новой зоны, а не изменение среды.
    """

    def __init__(self, slack=0.5, threshold=0.25, min_confidence=0.6):
        self.slack = slack              # расхождение в множителе, которое считается нормой
        self.threshold = threshold      # порог тревоги, множитель × метр
        self.min_confidence = min_confidence
        self.up = self.down = 0.0

    def update(self, ratio, predicted, confidence, ds):
        if confidence < self.min_confidence:
            self.up = self.down = 0.0
            return None
        err = ratio - predicted
        self.up = max(0.0, self.up + (err - self.slack) * ds)
        self.down = max(0.0, self.down + (-err - self.slack) * ds)
        if self.up > self.threshold or self.down > self.threshold:
            kind = 'дороже' if self.up > self.threshold else 'дешевле'
            self.up = self.down = 0.0
            return kind
        return None


class SensorHealth:
    """Оценка шума датчика образцов по разбросу соседних показаний."""

    def __init__(self, nominal_sigma, window=25):
        self.nominal = nominal_sigma
        self.buf = deque(maxlen=window)
        self.sigma = nominal_sigma
        self.degraded = False

    def update(self, z):
        """Возвращает 'degraded' или 'recovered' при смене состояния, иначе None."""
        self.buf.append(z)
        if len(self.buf) < 12:
            return None
        arr = np.array(self.buf)
        if arr.mean() < 0.08:      # сигнал у нуля обрезан, по нему шум не оценить
            return None
        diff = np.diff(arr)
        self.sigma = float(1.4826 * np.median(np.abs(diff - np.median(diff))) / math.sqrt(2))
        if not self.degraded and self.sigma > 2.5 * self.nominal:
            self.degraded = True
            return 'degraded'
        if self.degraded and self.sigma < 1.5 * self.nominal:
            self.degraded = False
            return 'recovered'
        return None

    def effective_sigma(self):
        return max(self.nominal, self.sigma) if self.degraded else self.nominal


class HazardMap:
    """Где опасные зоны, если известны только точки, в которых пришёл штраф.

    Штраф приходит на границе зоны, поэтому центр лежит где-то впереди по ходу на расстоянии её
    радиуса. Каждая зона — набор гипотез «центр и радиус». Гипотезу отбрасывает любое место, где
    робот проехал без штрафа, и уточняет каждый новый штраф. Клетка опасна настолько, какая доля
    оставшихся гипотез её накрывает.
    """

    RADII = (0.25, 0.29, 0.33)       # какими бывают зоны: предположение агента о размере
    MARGIN = 0.07                    # запас вокруг зоны на неточность управления

    def __init__(self):
        self.zones = []              # [{'c': Nx2, 'r': N, 'w': N}]
        self.version = 0

    def hit(self, x, y, heading, approach):
        """Штраф в точке (x, y) при движении по курсу heading. approach — недавний путь до штрафа."""
        e = np.array([x, y])
        for z in self.zones:         # штраф на границе уже известной зоны уточняет её
            keep = np.abs(np.hypot(*(z['c'] - e).T) - z['r']) <= 0.08
            if keep.any():
                self._filter(z, keep)
                return z
        phi = np.radians(np.linspace(-80, 80, 17))
        c, r, w = [], [], []
        for radius in self.RADII:
            c.append(e + radius * np.column_stack([np.cos(heading + phi), np.sin(heading + phi)]))
            r.append(np.full(len(phi), radius))
            w.append(np.cos(phi))    # лобовой въезд вероятнее касательного
        z = {'c': np.vstack(c), 'r': np.concatenate(r), 'w': np.concatenate(w)}
        self.zones.append(z)
        for px, py in approach:
            self._safe(z, px, py)
        self.version += 1
        return z

    def safe(self, x, y):
        """Робот проехал (x, y) без штрафа: зоны, накрывающие эту точку, невозможны."""
        for z in self.zones:
            self._safe(z, x, y)

    def _safe(self, z, x, y):
        keep = np.hypot(z['c'][:, 0] - x, z['c'][:, 1] - y) >= z['r'] - 0.02
        if keep.any() and not keep.all():
            self._filter(z, keep)

    def _filter(self, z, keep):
        z['c'], z['r'], z['w'] = z['c'][keep], z['r'][keep], z['w'][keep]
        self.version += 1

    def risk(self, X, Y, skip=None):
        """Вероятность по клеткам, что клетка лежит в опасной зоне (с запасом MARGIN).

        skip — зона, которую сейчас не учитывать: робот уже внутри неё и штраф получен.
        """
        out = np.zeros(X.shape)
        for z in self.zones:
            if z is skip:
                continue
            p = np.zeros(X.shape)
            for (cx, cy), r, w in zip(z['c'], z['r'], z['w']):
                p += w * (np.hypot(X - cx, Y - cy) <= r + self.MARGIN)
            out = np.maximum(out, p / z['w'].sum())
        return out

    def summary(self):
        """Для журнала и показа: оценка центра и радиус круга, накрывающего основную часть гипотез."""
        out = []
        for z in self.zones:
            w = z['w'] / z['w'].sum()
            cx, cy = float(w @ z['c'][:, 0]), float(w @ z['c'][:, 1])
            spread = float(w @ np.hypot(z['c'][:, 0] - cx, z['c'][:, 1] - cy))
            out.append((cx, cy, float(w @ z['r']) + spread))
        return out
