"""Карта занятости по лидару: что робот увидел сам, без готовой карты.

Сетка та же, что у эталонной арены (0,05 м). Каждый луч скана говорит две вещи: до конца луча
свободно, в конце — преграда. Клетка хранит логарифм шансов «занято»; каждое свидетельство
сдвигает его на постоянную величину, поэтому один шумный луч карту не портит.

Эталонная карта (did/arena.py) нужна только для двух чисел на экране: какая доля арены уже
увидена и насколько построенная карта совпадает с эталонной.
"""
import math

import numpy as np

from .config import LIDAR_MAX, LIDAR_MIN
from .recorder import encode_grid

LIDAR_OFFSET = -0.032     # м: лидар Burger стоит позади центра робота


class OccupancyMapper:

    L_FREE, L_OCC = -0.45, 0.9      # сдвиг логарифма шансов: луч прошёл сквозь клетку / упёрся в неё
    L_LIMIT = 4.0                   # предел уверенности: карту можно переубедить за несколько сканов
    KNOWN = 0.4                     # |логарифм шансов| больше — клетка считается увиденной
    STEP = 0.5                      # шаг точек вдоль луча в долях клетки
    INTO = 0.01                     # м: конец луча чуть углубляется в преграду, чтобы попасть в её клетку

    def __init__(self, arena):
        self.arena = arena
        self.res, self.x0, self.y0, self.w, self.h = arena.res, arena.x0, arena.y0, arena.w, arena.h
        self._inv = 1.0 / self.res
        self._r = np.arange(0.0, LIDAR_MAX, self.STEP * self.res)
        self._rays = {}                  # (число лучей, шаг) → (cos, sin) углов лучей
        self.reset()

    def reset(self):
        self.logodds = np.zeros(self.h * self.w, dtype=np.float32)
        self.scans = 0
        self.version = 0

    # --- обновление --------------------------------------------------------------------------

    def update(self, x, y, th, scan, scan_step=None):
        """Скан из позы робота (x, y, th) на карте. scan — дальности от курса против часовой, inf — нет отражения."""
        scan = np.asarray(scan, dtype=float)
        ok = np.isfinite(scan) & (scan >= LIDAR_MIN) & (scan <= LIDAR_MAX)
        if not ok.any():
            return
        cos, sin = self._ray_dirs(len(scan), scan_step)
        c, s = math.cos(th), math.sin(th)
        r = scan[ok]
        dx, dy = c * cos[ok] - s * sin[ok], s * cos[ok] + c * sin[ok]      # направления лучей в мире
        lx, ly = x + LIDAR_OFFSET * c, y + LIDAR_OFFSET * s

        # Свободно: точки вдоль луча, не доходя полклетки до его конца.
        along = self._r[:int(r.max() * self._inv / self.STEP) + 1]
        ix = ((lx - self.x0) + dx[:, None] * along) * self._inv
        iy = ((ly - self.y0) + dy[:, None] * along) * self._inv
        keep = (along < (r - 0.5 * self.res)[:, None]) & (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        free = np.zeros(self.h * self.w, dtype=bool)
        free[(iy[keep].astype(np.intp)) * self.w + ix[keep].astype(np.intp)] = True

        # Занято: клетка, в которую пришёл конец луча.
        ex = ((lx - self.x0) + dx * (r + self.INTO)) * self._inv
        ey = ((ly - self.y0) + dy * (r + self.INTO)) * self._inv
        inside = (ex >= 0) & (ex < self.w) & (ey >= 0) & (ey < self.h)
        occ = np.zeros(self.h * self.w, dtype=bool)
        occ[ey[inside].astype(np.intp) * self.w + ex[inside].astype(np.intp)] = True

        free &= ~occ
        self.logodds[free] += self.L_FREE
        self.logodds[occ] += self.L_OCC
        np.clip(self.logodds, -self.L_LIMIT, self.L_LIMIT, out=self.logodds)
        self.scans += 1
        self.version += 1

    def _ray_dirs(self, n, step):
        key = (n, step)
        if key not in self._rays:
            ang = np.arange(n) * (step or 2 * math.pi / n)
            self._rays[key] = (np.cos(ang), np.sin(ang))
        return self._rays[key]

    # --- что получилось ----------------------------------------------------------------------

    @property
    def grid(self):
        return self.logodds.reshape(self.h, self.w)

    def seen(self):
        return np.abs(self.grid) > self.KNOWN

    def occupied(self):
        return self.grid > self.KNOWN

    def coverage(self):
        """Доля свободной площади арены, которую робот уже увидел."""
        return float(self.seen()[self.arena.free].mean())

    def agreement(self):
        """Доля увиденных клеток, где построенная карта согласна с эталонной (свободно / занято)."""
        seen = self.seen()
        if not seen.any():
            return 0.0
        return float(((self.grid > 0) == self.arena.solid)[seen].mean())

    def encoded(self):
        """Сетка для показа, как в did/recorder.py: uint8 в base64, строки снизу вверх.

        0 — клетку ещё не видели; иначе вероятность «занято»: p = (v − 1) / 254.
        """
        p = 1.0 / (1.0 + np.exp(-self.grid))
        return encode_grid(np.where(self.seen(), 1 + np.round(p * 254), 0))

    def to_dict(self):
        return {'res': self.res, 'x0': self.x0, 'y0': self.y0, 'w': self.w, 'h': self.h, 'enc': 'occ',
                'version': self.version, 'data': self.encoded(),
                'coverage': round(self.coverage(), 4), 'agreement': round(self.agreement(), 4)}
