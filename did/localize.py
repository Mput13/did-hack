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
- поправка применяется плавно и с порогом нечувствительности: при точной одометрии поза не дрожит;
- совпадение проверяется в обе стороны: мало того, что концы лучей легли на преграды, — карта из
  этой позы должна обещать те же дальности. Шестиугольник стен совпадает сам с собой при повороте
  на 60°, а столбы — нет: луч, прошедший «сквозь столб» карты, выдаёт ложное совмещение;
- если совпадения нет уже LOST_SCANS сканов, положение считается потерянным (stats['lost']): курс
  и место ищутся заново вокруг позы по одометрии, принимается только единственный кандидат,
  прошедший проверку на двух сканах подряд. До тех пор поза не трогается: лучше честное «не знаю»,
  чем уверенная ошибка, по которой робот уедет в столб;
- stats['slip'] — на сколько поправка сдвинула позу за последние сканы: если это десятки
  сантиметров в секунду, колёса крутятся, а робот стоит (упёрся в преграду).
"""
import math
import time
from collections import deque

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
    UNSURE_SCANS = 2              # столько плохих сканов подряд — ехать дальше пока не стоит
    MIN_RAYS = 60                 # меньше годных лучей — скан пропускается
    MIN_INLIERS = 0.75            # доля лучей, лёгших на карту
    MAX_RESIDUAL = 0.030          # м: средний остаток по лёгшим лучам
    INLIER = 0.06                 # м: луч «лёг на карту», если его конец ближе к преграде
    KERNEL = 0.05                 # м: масштаб весовой функции (дальние концы почти не тянут)
    CUT = 0.25                    # м: концы дальше от преград в подгонке не участвуют
    BIAS_RANGE = (-0.06, 0.02)    # м: возможный сдвиг поверхности относительно границы на карте
    CALIB_SCANS = 10              # по стольким сканам на старте измеряется сдвиг карты
    SHIFT_RANGE = (0.010, 0.040)  # м: меньший сдвиг — шум, больший — уже не сдвиг карты
    AGREE_TOL = 0.12              # м: измеренная и обещанная картой дальность луча «сходятся»
    MIN_AGREE = 0.80              # доля сошедшихся лучей, с которой совпадению можно верить
    RELOC_XY, RELOC_STEP = 0.60, 0.10     # м: где и с каким шагом ищется потерянное положение
    RELOC_TH = 0.10               # рад: шаг перебора курса при таком поиске
    RELOC_AGREE = 0.85            # доля сошедшихся лучей у найденной заново позы
    RELOC_LEAD = 0.08             # настолько она должна быть лучше любого другого кандидата
    SLIP_SCANS = 5                # за столько сканов считается сдвиг позы поправкой (stats['slip'])

    def __init__(self, arena, enabled=True, verify=True):
        self.arena = arena
        self.enabled = enabled
        self.verify = verify             # False — прежнее поведение: без проверки в обе стороны и поиска заново
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
        self._pending = None             # найденное заново преобразование, ждёт подтверждения вторым сканом
        self._moves = deque(maxlen=self.SLIP_SCANS)     # сдвиги позы поправкой на последних сканах
        self._move = (0.0, 0.0)
        self.stats = {'scans': 0, 'fixes': 0, 'skipped': 0, 'fix': (0.0, 0.0, 0.0), 'residual': None,
                      'valid': 0.0, 'inliers': 0.0, 'bias': self.bias, 'shift': self.shift, 'ms': 0.0,
                      'lost': False, 'unsure': False, 'slip': 0.0, 'slip_vec': (0.0, 0.0), 'relocations': 0}

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
            self._move = (0.0, 0.0)
            self._on_scan(tuple(scan_pose) if scan_pose is not None else (x, y, th),
                          np.asarray(scan, dtype=float), scan_step)
            self._moves.append(self._move)
            self.stats['slip_vec'] = (sum(m[0] for m in self._moves), sum(m[1] for m in self._moves))
            self.stats['slip'] = math.hypot(*self.stats['slip_vec'])
            self.stats['lost'], self.stats['unsure'] = self.lost, self.unsure
            self.stats['ms'] = (time.perf_counter() - t0) * 1e3
        mx, my, mth = self.to_map(x, y, th)
        self.stats['fix'] = (mx - x, my - y, _wrap(mth - th))
        return mx, my, mth

    def to_map(self, x, y, th):
        tx, ty, dth = self._t
        c, s = math.cos(dth), math.sin(dth)
        return c * x - s * y + tx, s * x + c * y + ty, _wrap(th + dth)

    @property
    def lost(self):
        """Совпадения скана с картой нет уже давно: позе верить нельзя."""
        return self.verify and self._lost >= self.LOST_SCANS

    @property
    def unsure(self):
        """Последние сканы с картой не сошлись: может, случайность, а может, начало потери."""
        return self.verify and self._lost >= self.UNSURE_SCANS

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
        good = self._good(quality) and self._trust(pose, scan, step)
        if self.verify and not good and not quality[3] and self._good((*quality[:3], True)) \
                and self._trust(pose, scan, step):
            # Скан лёг на карту, но сдвиг вдоль стены не определяет: это не потеря, просто нечего поправить.
            st['inliers'], st['residual'] = quality[0], quality[1]
            st['skipped'] += 1
            return
        swept = not good
        if swept:
            if self.lost:
                return self._relocate(odom_pose, guess, qx, qy, scan, step)
            # Курс мог уйти рывком (колёса проскользнули): перебираем курс около предсказанного,
            # а если совпадения нет уже давно — по всей окружности (только без проверки в обе стороны).
            window = math.pi if self._lost >= self.LOST_SCANS else self.WINDOW_TH
            pose, quality = self._fit(self._sweep(guess, qx, qy, window), qx, qy)
            good = self._good(quality) and self._trust(pose, scan, step)
        st['inliers'], st['residual'] = quality[0], quality[1]
        dx, dy, dth = pose[0] - guess[0], pose[1] - guess[1], _wrap(pose[2] - guess[2])
        if not good or math.hypot(dx, dy) > self.WINDOW_XY:
            return self._skip()
        self._lost = 0
        self._pending = None

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
        self._t = _transform(odom_pose, (mx, my, mth))
        self._move = (dx, dy)
        st['fixes'] += 1

    def _skip(self):
        self._lost += 1
        self.stats['skipped'] += 1

    # --- проверка совпадения и поиск потерянного положения -------------------------------------

    def _trust(self, pose, scan, step):
        """Проверка в обе стороны: карта из этой позы обещает те же дальности, что намерил лидар.

        Делается на каждом скане, даже когда все концы лучей легли на преграды: у раскачавшегося робота
        половина лучей уходит в небо, а оставшиеся короткие ложатся на карту где угодно.
        """
        return not self.verify or self._agree(pose, scan, step) >= self.MIN_AGREE

    def _agree(self, pose, scan, step):
        """Доля лучей, у которых измеренная дальность сходится с обещанной картой из этой позы."""
        x, y, th = pose
        x, y = x + self.shift[0], y + self.shift[1]
        want = self.arena.raycast(x + LIDAR_OFFSET * math.cos(th), y + LIDAR_OFFSET * math.sin(th), th)
        n = len(scan)
        if n != len(want) or step:
            k = np.round(np.arange(n) * (step or 2 * math.pi / n) / (2 * math.pi / len(want))).astype(int)
            want = want[k % len(want)]
        far = LIDAR_MAX - 0.3
        seen, due = np.isfinite(scan) & (scan <= far), np.isfinite(want) & (want <= far)
        # Ближе LIDAR_MIN лидар молчит, у предела дальности отражение то есть, то нет: такие лучи не судьи.
        edge = (np.isfinite(scan) & ~seen) | (np.isfinite(want) & ~due) | (want < LIDAR_MIN + 0.05)
        judge = (seen | due) & ~edge
        if judge.sum() < self.MIN_RAYS:
            return 0.0
        same = seen & due & (np.abs(np.where(seen, scan, 0.0) - np.where(due, want, 0.0)) <= self.AGREE_TOL)
        return float(same[judge].mean())

    def _relocate(self, odom_pose, guess, qx, qy, scan, step):
        """Положение потеряно: ищем его заново вокруг позы по одометрии, по всем курсам.

        Принимается только кандидат, который прошёл проверку в обе стороны, заметно лучше остальных
        (симметричных двойников в окрестности нет) и подтвердился на следующем скане.
        """
        scored = []
        for start in self._search(guess, qx, qy):
            pose, quality = self._fit(start, qx, qy, iters=8)
            if self._good(quality) and math.hypot(pose[0] - guess[0], pose[1] - guess[1]) <= self.RELOC_XY + 0.1:
                scored.append((self._agree(pose, scan, step), pose, quality))
        scored.sort(key=lambda s: -s[0])
        found = None
        if scored and scored[0][0] >= self.RELOC_AGREE:
            best, pose, quality = scored[0]
            rivals = [a for a, p, _ in scored[1:] if _far(p, pose)]
            if all(a <= best - self.RELOC_LEAD for a in rivals):
                found = _transform(odom_pose, pose)
                self.stats['inliers'], self.stats['residual'] = quality[0], quality[1]
        if found is None or self._pending is None or _far(_apply(found, odom_pose), _apply(self._pending, odom_pose)):
            self._pending = found
            return self._skip()
        self._t, self._pending, self._lost = found, None, 0
        self._moves.clear()                  # скачок позы после потери — не проскальзывание колёс
        self.stats['fixes'] += 1
        self.stats['relocations'] += 1

    def _search(self, guess, qx, qy, keep=6):
        """Грубый перебор места и курса вокруг предсказанной позы: лучшие непохожие друг на друга кандидаты."""
        gx, gy, gth = guess
        span = np.arange(-self.RELOC_XY, self.RELOC_XY + 1e-9, self.RELOC_STEP)
        X, Y = (a.ravel() for a in np.meshgrid(gx + span, gy + span))
        near = np.hypot(X - gx, Y - gy) <= self.RELOC_XY + 1e-9
        dist, _, _ = self._field(X, Y)
        near &= dist > 0.08                  # центр робота не может быть в стене
        X, Y = X[near], Y[near]
        if not len(X):
            return []
        k = max(1, len(qx) // 72)
        qx, qy = qx[::k], qy[::k]
        ths = gth + np.arange(-math.pi, math.pi, self.RELOC_TH)
        cost = np.empty((len(ths), len(X)))
        for i, th in enumerate(ths):
            c, s = math.cos(th), math.sin(th)
            d, _, _ = self._field((X[:, None] + c * qx - s * qy).ravel(), (Y[:, None] + s * qx + c * qy).ravel())
            cost[i] = np.minimum(np.abs(d - self.bias), 0.15).reshape(len(X), -1).mean(axis=1)
        picked = []
        for flat in np.argsort(cost, axis=None)[:400]:
            i, j = divmod(int(flat), len(X))
            pose = (float(X[j]), float(Y[j]), float(ths[i]))
            if all(_far(pose, p) for p in picked):
                picked.append(pose)
                if len(picked) >= keep:
                    break
        return picked

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


def _transform(odom_pose, map_pose):
    """Преобразование «одометрия → карта», при котором поза одометрии ложится в заданную позу карты."""
    ox, oy, oth = odom_pose
    mx, my, mth = map_pose
    new = _wrap(mth - oth)
    c, s = math.cos(new), math.sin(new)
    return mx - (c * ox - s * oy), my - (s * ox + c * oy), new


def _apply(t, pose):
    c, s = math.cos(t[2]), math.sin(t[2])
    return c * pose[0] - s * pose[1] + t[0], s * pose[0] + c * pose[1] + t[1], _wrap(pose[2] + t[2])


def _far(a, b, xy=0.15, th=0.25):
    """Две позы — разные места или курсы, а не одно и то же с точностью подгонки."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) > xy or abs(_wrap(a[2] - b[2])) > th


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
