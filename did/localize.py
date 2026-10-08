"""Положение робота по лидару и карте: поправка к одометрии.

Одометрия колёс уходит: при резком развороте колёса проскальзывают, при упоре в преграду крутятся
на месте. PoseTracker хранит преобразование «система одометрии → система карты» и на каждом скане
уточняет его: концы лучей должны лечь на преграды карты. Мера несовпадения — расстояние со знаком
от конца луча до границы свободных клеток (положительное в свободном месте, отрицательное внутри
преграды), минимум ищется методом Гаусса — Ньютона вокруг предсказанной позы.

Особенности, ради которых всё не в три строки:
- карта грубая (клетка 5 см), а настоящая поверхность стены лежит в глубине граничной клетки,
  поэтому общий сдвиг «стена дальше, чем граница на карте» оценивается на ходу (bias);
- карта может быть сдвинута относительно мира на долю клетки: пока робот стоит на старте, где
  поза известна, этот сдвиг измеряется (shift) и дальше вычитается;
- поиск только локальный: арена симметрична, далёкое совпадение — почти наверняка ложное;
- плохое совпадение (мало лучей, большой остаток, вырожденная геометрия) пропускается;
- поправка применяется плавно и с порогом нечувствительности: при точной одометрии поза не дрожит.
"""
import math
import time

import numpy as np
from scipy import ndimage

from .config import LIDAR_MAX, LIDAR_MIN

LIDAR_OFFSET = -0.032     # м: лидар Burger стоит позади центра робота


class PoseTracker:

    GAIN = 0.5                    # доля найденного расхождения, которая применяется за один скан
    STEP_XY, STEP_TH = 0.04, 0.06     # предел поправки за скан: м и рад
    STEP_TH_SWEEP = 0.35          # рад: предел, когда курс потерян и найден перебором
    DEAD_XY, DEAD_TH = 0.008, 0.004   # расхождение меньше этого — шум совпадения, не поправляем
    WINDOW_XY, WINDOW_TH = 0.30, 0.70     # дальше от предсказанной позы совпадение не ищется
    LOST_SCANS = 5                # столько плохих сканов подряд — курс ищется по всей окружности
    MIN_RAYS = 60                 # меньше годных лучей — скан пропускается
    MIN_INLIERS = 0.75            # доля лучей, лёгших на карту
    MAX_RESIDUAL = 0.030          # м: средний остаток по лёгшим лучам
    INLIER = 0.06                 # м: луч «лёг на карту», если его конец ближе к преграде
    KERNEL = 0.05                 # м: масштаб весовой функции (дальние концы почти не тянут)
    CUT = 0.25                    # м: концы дальше от преград в подгонке не участвуют
    BIAS_RANGE = (-0.06, 0.02)    # м: возможный сдвиг поверхности относительно границы на карте
    CALIB_SCANS = 10              # по стольким сканам на старте измеряется сдвиг карты
    SHIFT_RANGE = (0.010, 0.040)  # м: меньший сдвиг — шум, больший — уже не сдвиг карты

    def __init__(self, arena, enabled=True):
        self.arena = arena
        self.enabled = enabled
        res = arena.res
        # Расстояние со знаком в центрах клеток: на границе свободной и занятой клетки — ноль.
        sdf = np.where(arena.free, ndimage.distance_transform_edt(arena.free) - 0.5,
                       0.5 - ndimage.distance_transform_edt(arena.solid)) * res
        self._sdf = np.ascontiguousarray(sdf, dtype=float).ravel()
        self._w, self._h = arena.w, arena.h
        self._x0, self._y0, self._inv = arena.x0, arena.y0, 1.0 / res
        self._rays = {}                  # (число лучей, шаг) → (cos, sin) углов лучей
        self._t = (0.0, 0.0, 0.0)        # одометрия → карта: сдвиг и поворот вокруг начала координат
        self.bias = -0.02
        self._bias_n = 0
        self.shift = (0.0, 0.0)          # сдвиг карты относительно мира
        self._calib = []                 # расхождения на старте; None — калибровка закончена
        self._home = None                # первая поза одометрии
        self._lost = 0                   # плохих сканов подряд
        self.stats = {'scans': 0, 'fixes': 0, 'skipped': 0, 'fix': (0.0, 0.0, 0.0), 'residual': None,
                      'valid': 0.0, 'inliers': 0.0, 'bias': self.bias, 'shift': self.shift, 'ms': 0.0}

    # --- наружу ------------------------------------------------------------------------------

    def update(self, x, y, th, scan=None, scan_pose=None, scan_step=None):
        """Поза по одометрии и свежий скан (или None) → исправленная поза в системе карты.

        scan_pose — поза по одометрии в момент скана, если она известна точнее текущей (в Gazebo скан
        приходит с задержкой); scan_step — угол между лучами, рад (по умолчанию 2π / число лучей).
        """
        if not self.enabled:
            return x, y, th
        if scan is not None:
            t0 = time.perf_counter()
            self._on_scan(tuple(scan_pose) if scan_pose is not None else (x, y, th),
                          np.asarray(scan, dtype=float), scan_step)
            self.stats['ms'] = (time.perf_counter() - t0) * 1e3
        mx, my, mth = self.to_map(x, y, th)
        self.stats['fix'] = (mx - x, my - y, _wrap(mth - th))
        return mx, my, mth

    def to_map(self, x, y, th):
        tx, ty, dth = self._t
        c, s = math.cos(dth), math.sin(dth)
        return c * x - s * y + tx, s * x + c * y + ty, _wrap(th + dth)

    # --- один скан ---------------------------------------------------------------------------

    def _on_scan(self, odom_pose, scan, step):
        st = self.stats
        st['scans'] += 1
        ok = np.isfinite(scan) & (scan >= LIDAR_MIN) & (scan <= LIDAR_MAX - 0.05)
        n = int(ok.sum())
        st['valid'] = n / len(scan)
        if n < self.MIN_RAYS:
            return self._skip()
        cos, sin = self._ray_dirs(len(scan), step)
        r = scan[ok]
        qx, qy = LIDAR_OFFSET + r * cos[ok], r * sin[ok]       # концы лучей в системе робота

        guess = self.to_map(*odom_pose)
        pose, quality = self._fit(guess, qx, qy)
        swept = not self._good(quality)
        if swept:
            # Курс мог уйти рывком (колёса проскользнули): перебираем курс около предсказанного,
            # а если совпадения нет уже давно — по всей окружности.
            window = math.pi if self._lost >= self.LOST_SCANS else self.WINDOW_TH
            pose, quality = self._fit(self._sweep(guess, qx, qy, window), qx, qy)
        st['inliers'], st['residual'] = quality[0], quality[1]
        dx, dy, dth = pose[0] - guess[0], pose[1] - guess[1], _wrap(pose[2] - guess[2])
        if not self._good(quality) or math.hypot(dx, dy) > self.WINDOW_XY:
            return self._skip()
        self._lost = 0

        # Сдвиг поверхности относительно карты: сначала среднее по первым сканам, потом медленно.
        self._bias_n += 1
        self.bias += (quality[2] - self.bias) * max(0.05, 1.0 / self._bias_n)
        self.bias = min(max(self.bias, self.BIAS_RANGE[0]), self.BIAS_RANGE[1])
        st['bias'] = self.bias

        if self._calib is not None and self._calibrate(odom_pose, dx, dy):
            return

        d = math.hypot(dx, dy)
        k = min(self.GAIN * max(0.0, d - self.DEAD_XY), self.STEP_XY) / d if d > 0.0 else 0.0
        dx, dy = dx * k, dy * k
        limit = self.STEP_TH_SWEEP if swept else self.STEP_TH
        dth = math.copysign(min(self.GAIN * max(0.0, abs(dth) - self.DEAD_TH), limit), dth)
        if dx == 0.0 and dy == 0.0 and dth == 0.0:
            return
        # Новая поза робота на карте в момент скана → новое преобразование для всей одометрии.
        ox, oy, oth = odom_pose
        mx, my, mth = guess[0] + dx, guess[1] + dy, guess[2] + dth
        new = _wrap(mth - oth)
        c, s = math.cos(new), math.sin(new)
        self._t = (mx - (c * ox - s * oy), my - (s * ox + c * oy), new)
        st['fixes'] += 1

    def _skip(self):
        self._lost += 1
        self.stats['skipped'] += 1

    def _calibrate(self, odom_pose, dx, dy):
        """Сдвиг карты относительно мира: пока робот стоит на старте, его поза известна без лидара."""
        if self._home is None:
            self._home = odom_pose
        hx, hy, hth = self._home
        moved = math.hypot(odom_pose[0] - hx, odom_pose[1] - hy) > 0.01 or abs(_wrap(odom_pose[2] - hth)) > 0.01
        if not moved:
            self._calib.append((dx, dy))
        if moved or len(self._calib) >= self.CALIB_SCANS:
            if self._calib:
                sx, sy = np.mean(self._calib, axis=0)
                if self.SHIFT_RANGE[0] <= math.hypot(sx, sy) <= self.SHIFT_RANGE[1]:
                    self.shift = (float(sx), float(sy))
                    self._x0, self._y0 = self.arena.x0 - self.shift[0], self.arena.y0 - self.shift[1]
                    self.stats['shift'] = self.shift
            self._calib = None
        return not moved                     # на старте поза не поправляется: расхождение — это сдвиг карты

    def _good(self, quality):
        inliers, residual, _, firm = quality
        return firm and inliers >= self.MIN_INLIERS and residual <= self.MAX_RESIDUAL

    def _ray_dirs(self, n, step):
        key = (n, step)
        if key not in self._rays:
            ang = np.arange(n) * (step or 2 * math.pi / n)
            self._rays[key] = (np.cos(ang), np.sin(ang))
        return self._rays[key]

    def _field(self, ex, ey):
        """Расстояние со знаком и его градиент в точках (билинейно между центрами клеток)."""
        fx = (ex - self._x0) * self._inv - 0.5
        fy = (ey - self._y0) * self._inv - 0.5
        ix = np.clip(fx.astype(np.intp), 0, self._w - 2)
        iy = np.clip(fy.astype(np.intp), 0, self._h - 2)
        u = np.clip(fx - ix, 0.0, 1.0)
        v = np.clip(fy - iy, 0.0, 1.0)
        k = iy * self._w + ix
        s00, s10 = self._sdf.take(k), self._sdf.take(k + 1)
        s01, s11 = self._sdf.take(k + self._w), self._sdf.take(k + self._w + 1)
        low = s00 + (s10 - s00) * u
        high = s01 + (s11 - s01) * u
        gx = ((s10 - s00) + ((s11 - s01) - (s10 - s00)) * v) * self._inv
        return low + (high - low) * v, gx, (high - low) * self._inv

    def _fit(self, pose, qx, qy, iters=5):
        """Подгонка позы к скану. Возвращает позу и качество: (доля лёгших лучей, остаток, сдвиг, устойчиво ли)."""
        x, y, th = pose
        n = len(qx)
        for _ in range(iters):
            c, s = math.cos(th), math.sin(th)
            rx, ry = c * qx - s * qy, s * qx + c * qy
            dist, gx, gy = self._field(x + rx, y + ry)
            r = dist - self.bias
            w = 1.0 / (1.0 + (r / self.KERNEL) ** 2)
            w[np.abs(r) > self.CUT] = 0.0
            J = np.stack((gx, gy, gy * rx - gx * ry), axis=1)
            JW = J * w[:, None]
            H = JW.T @ J
            # Слабая привязка к предсказанной позе: куда скан не тянет, туда поза и не сдвинется.
            H[0, 0] += 0.02 * n
            H[1, 1] += 0.02 * n
            H[2, 2] += 0.02 * n
            step = np.linalg.solve(H, -(JW.T @ r))
            x, y, th = x + step[0], y + step[1], th + step[2]
            if abs(step[0]) < 5e-4 and abs(step[1]) < 5e-4 and abs(step[2]) < 5e-4:
                break
        # Сдвиг вдоль стены скан не определяет: если положение держится только на привязке — не верим.
        firm = np.linalg.eigvalsh(H[:2, :2])[0] > 0.02 * n + 0.04 * w.sum()
        c, s = math.cos(th), math.sin(th)
        dist, _, _ = self._field(x + c * qx - s * qy, y + s * qx + c * qy)
        r = dist - self.bias
        inl = np.abs(r) < self.INLIER
        m = int(inl.sum())
        if m == 0:
            return (x, y, th), (0.0, math.inf, self.bias, False)
        return (x, y, th), (m / n, float(np.abs(r[inl]).mean()), float(np.median(dist[inl])), bool(firm))

    def _sweep(self, pose, qx, qy, window):
        """Лучший курс из перебора вокруг предсказанного, положение то же."""
        x, y, th = pose
        ths = th + np.arange(-window, window, 0.05)
        c, s = np.cos(ths)[:, None], np.sin(ths)[:, None]
        qx, qy = qx[::2], qy[::2]
        dist, _, _ = self._field((x + c * qx - s * qy).ravel(), (y + s * qx + c * qy).ravel())
        cost = np.minimum(np.abs(dist - self.bias), 0.15).reshape(len(ths), -1).mean(axis=1)
        return x, y, float(ths[int(cost.argmin())])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
