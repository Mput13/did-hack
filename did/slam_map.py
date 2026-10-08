"""Карта от SLAM Toolbox вместо готовой: та же арена для навигации, но собранная из сетки SLAM.

Что здесь есть:
- resample — сетка SLAM (своя решётка, растёт по мере езды) переносится на решётку арены 0,05 м;
- SlamArena — то же, что did/arena.py::Arena (свободные клетки, расстояние до преград, лучи), но
  построенное из сетки SLAM: свободно только то, что SLAM увидел свободным и что связано с роботом,
  всё неувиденное считается преградой;
- frontiers — границы увиденного: куда ехать, чтобы достроить карту;
- SlamPose — поправка позы от самого SLAM Toolbox (его преобразование «карта → одометрия») с тем же
  интерфейсом, что у did/localize.py::PoseTracker;
- compare и pillars — сверка карты SLAM с готовой картой и с известными координатами столбов мира.
  Это измерение качества для отчёта и экрана, роботу оно не передаётся.

Начало координат. SLAM Toolbox ставит начало карты туда, где робот был на первом скане, по той
одометрии, которую ему дали. Мы даём ему одометрию в мировом кадре (env/slam_demo.yaml: odom_frame —
кадр, сдвинутый от одометрии на старт (−2,0; −0,5), как в условии: «поза в мире = старт + одометрия»).
Поэтому первая поза в графе SLAM — (−2,0; −0,5), и вся карта сразу в мировых координатах: поле
origin сообщения OccupancyGrid — мировые координаты угла сетки, пересчитывать ничего не нужно.
"""
import math

import numpy as np
from scipy import ndimage

from .arena import CROP, Arena
from .config import LIDAR_MAX

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
OCC_FROM = 50                       # значение клетки OccupancyGrid, с которого она считается занятой
# Столбы мира turtlebot3_world (models/turtlebot3_world/model.sdf): девять цилиндров радиусом 0,15 м.
PILLARS = [(x, y) for x in (-1.1, 0.0, 1.1) for y in (-1.1, 0.0, 1.1)]
PILLAR_R = 0.15


def lattice(res=0.05):
    """Решётка арены: (res, x0, y0, w, h) — та же, что у готовой карты, но без самой карты."""
    return res, CROP[0], CROP[1], round((CROP[2] - CROP[0]) / res), round((CROP[3] - CROP[1]) / res)


def resample(grid, res, ox, oy, lat=None):
    """Сетка SLAM → значения на решётке арены: −1 не видел, 0 свободно, 100 занято.

    grid[iy, ix] — клетка с углом (ox + ix*res, oy + iy*res) в мировых координатах. Берётся клетка SLAM,
    в которую попал центр клетки арены: шаг решёток одинаковый, так что ни одна клетка не теряется.
    """
    lres, x0, y0, w, h = lat or lattice()
    grid = np.asarray(grid)
    xs = x0 + (np.arange(w) + 0.5) * lres
    ys = y0 + (np.arange(h) + 0.5) * lres
    ix = np.floor((xs - ox) / res + 1e-6).astype(np.intp)
    iy = np.floor((ys - oy) / res + 1e-6).astype(np.intp)
    okx, oky = (ix >= 0) & (ix < grid.shape[1]), (iy >= 0) & (iy < grid.shape[0])
    out = np.full((h, w), UNKNOWN, dtype=np.int8)
    sub = grid[np.ix_(iy[oky], ix[okx])]
    out[np.ix_(oky, okx)] = np.where(sub < 0, UNKNOWN, np.where(sub >= OCC_FROM, OCCUPIED, FREE))
    return out


class SlamArena(Arena):
    """Арена для навигации, собранная из сетки SLAM. Готовая карта сюда не попадает никак."""

    def __init__(self, values=None, near=None, res=0.05):
        """values — сетка на решётке арены (см. resample); near — (x, y) робота: свободным считается
        только тот увиденный пол, который с этим местом связан."""
        self.res, self.x0, self.y0, self.w, self.h = lattice(res)
        if values is None:
            values = np.full((self.h, self.w), UNKNOWN, dtype=np.int8)
        self.values = values
        self.occupied = values == OCCUPIED
        # Лучи лидара расходятся веером: вдали между соседними лучами остаются неувиденные щели в одну-две
        # клетки. Щель между двумя полосами увиденного пола — тоже пол (занятые клетки при этом не трогаются).
        seen_free = ndimage.binary_closing(values == FREE, structure=np.ones((3, 3), bool)) & ~self.occupied
        self.free = np.zeros_like(seen_free)
        if seen_free.any():
            labels, _ = ndimage.label(seen_free)
            ix, iy = self.w2g(*(near or (0.0, 0.0)))
            ix, iy = min(max(ix, 0), self.w - 1), min(max(iy, 0), self.h - 1)
            if not seen_free[iy, ix]:
                # Клетку под самим роботом лидар не видит: берём ближайшую увиденную.
                _, (ny, nx) = ndimage.distance_transform_edt(~seen_free, return_indices=True)
                iy, ix = int(ny[iy, ix]), int(nx[iy, ix])
            self.free = labels == labels[iy, ix]
        self.solid = ~self.free
        self.unknown = ~seen_free & ~self.occupied
        self.clear = ndimage.distance_transform_edt(self.free) * self.res
        self._ray_r = np.arange(0.02, LIDAR_MAX + 0.02, 0.02)

    @property
    def empty(self):
        return not self.free.any()


def frontiers(arena, min_cells=5):
    """Границы увиденного: группы свободных клеток, за которыми начинается неувиденное.

    Возвращает список (x, y, число клеток): клетка группы, ближайшая к её центру (сам центр изогнутой
    группы может оказаться где угодно). Столб и то, что за стеной, границей не считаются: между
    свободным полом и неувиденным там лежит занятая клетка.
    """
    edge = arena.free & ndimage.binary_dilation(arena.unknown, structure=np.ones((3, 3), bool))
    labels, n = ndimage.label(edge, structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        iy, ix = np.nonzero(labels == k)
        if len(ix) >= min_cells:
            j = int(np.argmin((ix - ix.mean()) ** 2 + (iy - iy.mean()) ** 2))
            out.append((arena.x0 + (ix[j] + 0.5) * arena.res, arena.y0 + (iy[j] + 0.5) * arena.res, len(ix)))
    return out


class SlamPose:
    """Поза робота на карте SLAM: одометрия, исправленная преобразованием от SLAM Toolbox.

    Интерфейс — как у PoseTracker (update, to_map, stats, shift), чтобы пульт и агент не различали,
    кто поправляет позу. source() возвращает (dx, dy, dth) — преобразование «мировой кадр одометрии →
    карта SLAM», которое SLAM Toolbox публикует в /tf, или None, пока его нет.
    """

    shift = (0.0, 0.0)
    lost = unsure = False

    def __init__(self, source):
        self._source = source
        self._t = (0.0, 0.0, 0.0)
        self.stats = {'scans': 0, 'fixes': 0, 'skipped': 0, 'fix': (0.0, 0.0, 0.0), 'residual': None,
                      'valid': 0.0, 'inliers': 0.0, 'bias': 0.0, 'shift': self.shift, 'ms': 0.0,
                      'lost': False, 'unsure': False, 'slip': 0.0, 'slip_vec': (0.0, 0.0), 'relocations': 0}

    def update(self, x, y, th, scan=None, scan_pose=None, scan_step=None):
        t = self._source()
        if scan is not None:
            self.stats['scans'] += 1
        if t is not None and tuple(t) != self._t:
            self._t = tuple(t)
            self.stats['fixes'] += 1
        mx, my, mth = self.to_map(x, y, th)
        self.stats['fix'] = (mx - x, my - y, _wrap(mth - th))
        return mx, my, mth

    def to_map(self, x, y, th):
        tx, ty, dth = self._t
        c, s = math.cos(dth), math.sin(dth)
        return c * x - s * y + tx, s * x + c * y + ty, _wrap(th + dth)


# =================================================================================================
# сверка с готовой картой и с миром: только для отчёта и экрана
# =================================================================================================

def compare(values, ref):
    """Карта SLAM (на решётке арены) против готовой карты ref (did/arena.py::Arena).

    coverage — какую долю свободного пола готовой карты SLAM увидел;
    agreement — доля увиденных SLAM клеток, где он согласен с готовой картой (свободно / занято);
    free_ok — доля клеток, свободных у SLAM, которые свободны и на готовой карте;
    occ_ok — доля занятых у SLAM клеток, которые на готовой карте заняты или граничат с занятой.
    """
    seen, occ = values != UNKNOWN, values == OCCUPIED
    if not seen.any():
        return {'coverage': 0.0, 'agreement': 0.0, 'free_ok': 0.0, 'occ_ok': 0.0, 'seen_cells': 0, 'occ_cells': 0}
    near_solid = ndimage.binary_dilation(ref.solid, structure=np.ones((3, 3), bool))
    free = seen & ~occ
    return {'coverage': float(seen[ref.free].mean()),
            'agreement': float((occ == ref.solid)[seen].mean()),
            'free_ok': float(ref.free[free].mean()) if free.any() else 0.0,
            'occ_ok': float(near_solid[occ].mean()) if occ.any() else 0.0,
            'seen_cells': int(seen.sum()), 'occ_cells': int(occ.sum())}


def best_shift(values, ref, reach=4):
    """На сколько клеток надо сдвинуть карту SLAM, чтобы она лучше всего легла на готовую.

    Возвращает (dx, dy, совпадение при этом сдвиге) в метрах. Ноль — карты уже совмещены с точностью до клетки.
    """
    seen, occ = values != UNKNOWN, values == OCCUPIED
    best = (0.0, 0.0, -1.0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            s = np.roll(seen, (dy, dx), axis=(0, 1))
            o = np.roll(occ, (dy, dx), axis=(0, 1))
            a = float((o == ref.solid)[s].mean()) if s.any() else 0.0
            if a > best[2] + 1e-9 or (abs(a - best[2]) <= 1e-9 and abs(dx) + abs(dy) < (abs(best[0]) + abs(best[1])) / ref.res):
                best = (dx * ref.res, dy * ref.res, a)
    return best


def fit_circle(xs, ys, r=PILLAR_R):
    """Центр окружности известного радиуса по точкам на ней (или на её части)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    cx, cy = xs.mean(), ys.mean()
    for _ in range(30):
        dx, dy = xs - cx, ys - cy
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        # Гаусс — Ньютон: остаток d − r, производные по центру — (−dx/d, −dy/d).
        J = np.stack((-dx / d, -dy / d), axis=1)
        H = J.T @ J + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, -(J.T @ (d - r)))
        cx, cy = cx + step[0], cy + step[1]
        if abs(step[0]) < 1e-5 and abs(step[1]) < 1e-5:
            break
    return float(cx), float(cy)


def pillars(values, lat=None, window=0.32):
    """Где на карте SLAM стоят девять столбов и на сколько это отличается от мира.

    Для каждого столба берутся занятые клетки в окне вокруг его мирового места, по их центрам
    подгоняется окружность радиуса столба. Возвращает список словарей: мировое место, найденный
    центр, ошибка в метрах, число клеток; столб без клеток — с ошибкой None.
    """
    res, x0, y0, w, h = lat or lattice()
    iy, ix = np.nonzero(values == OCCUPIED)
    X, Y = x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res
    out = []
    for px, py in PILLARS:
        m = np.hypot(X - px, Y - py) <= window
        n = int(m.sum())
        if n < 6:
            out.append({'x': px, 'y': py, 'found': None, 'err': None, 'cells': n})
            continue
        cx, cy = fit_circle(X[m], Y[m])
        out.append({'x': px, 'y': py, 'found': [round(cx, 4), round(cy, 4)], 'dx': round(cx - px, 4),
                    'dy': round(cy - py, 4), 'err': round(math.hypot(cx - px, cy - py), 4), 'cells': n})
    return out


def origin_offset(found):
    """Сдвиг и поворот карты относительно мира по найденным столбам (жёсткое совмещение).

    Возвращает {'dx', 'dy', 'shift', 'rot_deg', 'residual'}: на сколько начало координат карты смещено
    от мирового и на какой угол карта повёрнута; residual — средняя ошибка столбов после вычитания этого.
    """
    pts = [(p['x'], p['y'], *p['found']) for p in found if p.get('found')]
    if len(pts) < 3:
        return None
    a = np.array([(p[0], p[1]) for p in pts])
    b = np.array([(p[2], p[3]) for p in pts])
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    A, B = a - ca, b - cb
    rot = math.atan2(float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum()), float((A * B).sum()))
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    t = cb - R @ ca                         # куда на карте попало мировое начало координат
    resid = np.hypot(*(b - (a @ R.T + t)).T)
    return {'dx': float(t[0]), 'dy': float(t[1]), 'shift': float(math.hypot(*t)), 'rot_deg': math.degrees(rot),
            'residual': float(resid.mean()), 'pillars': len(pts)}


def outline(ref):
    """Готовая карта в том же виде, что сетка SLAM: пол свободен, граница преград занята, остальное не видно.

    Нужна, чтобы теми же функциями измерить саму готовую карту: она тоже стоит в мире не идеально.
    """
    edge = ref.solid & ndimage.binary_dilation(ref.free, structure=np.ones((3, 3), bool))
    return np.where(ref.free, FREE, np.where(edge, OCCUPIED, UNKNOWN)).astype(np.int8)


def report(values, ref):
    """Все числа сверки одной структурой (её печатает tools/slam_check.py)."""
    found = pillars(values, (ref.res, ref.x0, ref.y0, ref.w, ref.h))
    errs = [p['err'] for p in found if p['err'] is not None]
    sx, sy, sa = best_shift(values, ref)
    return {**compare(values, ref), 'pillars': found,
            'pillars_found': len(errs), 'pillar_err_mean': float(np.mean(errs)) if errs else None,
            'pillar_err_max': float(np.max(errs)) if errs else None,
            'origin': origin_offset(found),
            'best_shift': {'dx': sx, 'dy': sy, 'agreement': sa}}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
