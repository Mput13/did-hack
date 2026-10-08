"""Навигация уровня 1: сетка со стоимостями клеток, кратчайший по расходу путь, ведение по пути.

Стены и запас вокруг них запрещены. Стоимость клетки — оценка расхода батареи на метр в ней,
поэтому «длина» пути в графе измеряется в единицах заряда.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .config import V_MAX

INFLATE = 0.17        # м: ближе к стене центр робота не планируется (радиус 0,105 + запас)
WALL_SOFT = 0.30      # м: до этой дистанции клетки чуть дороже, чтобы путь не жался к стенам
FORBIDDEN_COST = 60.0


class CostGraph:

    def __init__(self, arena, inflate=INFLATE):
        self.arena = arena
        self.ok = arena.clear >= inflate
        self.node_of = np.full(arena.free.shape, -1, dtype=np.int32)
        iy, ix = np.nonzero(self.ok)
        self.iy, self.ix = iy, ix
        self.n = len(ix)
        self.node_of[iy, ix] = np.arange(self.n)
        self.xs = arena.x0 + (ix + 0.5) * arena.res
        self.ys = arena.y0 + (iy + 0.5) * arena.res
        # Ближайшая разрешённая клетка для любой точки арены (старт у стены, цель в столбе).
        _, (ny, nx) = ndimage.distance_transform_edt(~self.ok, return_indices=True)
        self._near = self.node_of[ny, nx]

        us, vs, ls = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            jx, jy = ix + dx, iy + dy
            inside = (jx >= 0) & (jx < arena.w) & (jy >= 0) & (jy < arena.h)
            a = np.nonzero(inside)[0]
            b = self.node_of[jy[a], jx[a]]
            keep = b >= 0
            if dx and dy:   # по диагонали нельзя срезать угол запрещённой клетки
                keep &= self.ok[iy[a], jx[a]] & self.ok[jy[a], ix[a]]
            us.append(a[keep])
            vs.append(b[keep])
            ls.append(np.full(keep.sum(), arena.res * math.hypot(dx, dy)))
        u, v, length = np.concatenate(us), np.concatenate(vs), np.concatenate(ls)
        self._eu = np.concatenate([u, v])
        self._ev = np.concatenate([v, u])
        self._len = np.concatenate([length, length])
        order = csr_matrix((np.arange(1, len(self._eu) + 1), (self._eu, self._ev)), shape=(self.n, self.n))
        self._perm = order.data - 1
        self._indices, self._indptr = order.indices, order.indptr
        soft = np.clip((WALL_SOFT - arena.clear[iy, ix]) / WALL_SOFT, 0.0, 1.0)
        self._wall = 1.0 + 0.6 * soft
        self.cost = np.ones(self.n)
        self.set_cost(None)

    def set_cost(self, mult_grid=None, forbidden=None, bias=None):
        """mult_grid — оценка множителя расхода по клеткам арены; forbidden — маска запретных клеток;
        bias — во сколько раз клетка нежелательна для маршрута сверх расхода (например, непроверенный пол)."""
        c = np.ones(self.n) if mult_grid is None else np.asarray(mult_grid)[self.iy, self.ix].astype(float)
        self.mult = c                      # чистый множитель расхода: по нему считается заряд на путь
        c = c * self._wall
        if bias is not None:
            c = c * np.asarray(bias)[self.iy, self.ix]
        if forbidden is not None:
            c = np.where(np.asarray(forbidden)[self.iy, self.ix], FORBIDDEN_COST, c)
        self.cost = c
        w = self._len * 0.5 * (c[self._eu] + c[self._ev])
        self._graph = csr_matrix((w[self._perm], self._indices, self._indptr), shape=(self.n, self.n))

    def node(self, x, y):
        ix, iy = self.arena.w2g(x, y)
        ix = min(max(ix, 0), self.arena.w - 1)
        iy = min(max(iy, 0), self.arena.h - 1)
        return int(self._near[iy, ix])

    def field(self, x, y):
        """Стоимость пути от (x, y) до каждой разрешённой клетки и дерево путей."""
        dist, pred = dijkstra(self._graph, directed=True, indices=self.node(x, y), return_predecessors=True)
        return dist, pred

    def cost_at(self, dist, x, y):
        return float(dist[self.node(x, y)])

    def energy(self, dist, pred, x, y):
        """Расход на путь из источника поля в (x, y), в метрах обычного пола.

        Путь выбирается по стоимости со штрафами (стены, запретные клетки), а заряд считается без
        них: штраф — это нежелание туда ехать, а не трата батареи.
        """
        node = self.node(x, y)
        if not math.isfinite(dist[node]):
            return math.inf
        total = 0.0
        while pred[node] >= 0:
            prev = pred[node]
            total += math.hypot(self.xs[node] - self.xs[prev], self.ys[node] - self.ys[prev]) \
                * 0.5 * (self.mult[node] + self.mult[prev])
            node = prev
        return total

    def trace(self, pred, x, y):
        """Путь из источника поля в (x, y): список точек, включая конечную."""
        node = self.node(x, y)
        cells = []
        while node >= 0:
            cells.append(node)
            node = pred[node]
        cells.reverse()
        return [(float(self.xs[i]), float(self.ys[i])) for i in cells]

    def plan(self, start, goal):
        dist, pred = self.field(*start)
        if not math.isfinite(self.cost_at(dist, *goal)):
            return None, math.inf
        return self.trace(pred, *goal), self.energy(dist, pred, *goal)

    def grid(self, values, fill=np.nan):
        """Значения по узлам → сетка арены (для сглаживания и показа)."""
        out = np.full(self.arena.free.shape, fill, dtype=float)
        out[self.iy, self.ix] = values
        return out


def path_length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


class Follower:
    """Ведение по пути: смотрит на точку впереди, при большой ошибке курса разворачивается на месте."""

    LOOKAHEAD = 0.22
    TURN_IN_PLACE = 1.0     # рад
    K_W = 2.4
    W_LIMIT = 1.9

    def __init__(self):
        self.pts = []
        self.i = 0

    def set_path(self, pts):
        self.pts = list(pts)
        self.i = 0

    @property
    def active(self):
        return bool(self.pts)

    def remaining(self, x, y):
        if not self.pts:
            return 0.0
        return math.dist((x, y), self.pts[self.i]) + path_length(self.pts[self.i:])

    def step(self, x, y, th, tol=0.06, v_max=V_MAX):
        """Возвращает (v, w, приехал)."""
        if not self.pts:
            return 0.0, 0.0, True
        goal = self.pts[-1]
        to_goal = math.dist((x, y), goal)
        if to_goal <= tol:
            return 0.0, 0.0, True
        # Ближайшая точка пути не раньше текущей, затем точка на расстоянии LOOKAHEAD.
        end = min(len(self.pts), self.i + 12)
        self.i = min(range(self.i, end), key=lambda k: math.dist((x, y), self.pts[k]))
        k = self.i
        while k < len(self.pts) - 1 and math.dist((x, y), self.pts[k]) < self.LOOKAHEAD:
            k += 1
        tx, ty = self.pts[k]
        err = _wrap(math.atan2(ty - y, tx - x) - th)
        w = float(np.clip(self.K_W * err, -self.W_LIMIT, self.W_LIMIT))
        if abs(err) > self.TURN_IN_PLACE:
            return 0.0, w, False
        v = v_max * max(0.25, 1.0 - abs(err) / 1.2)
        v = min(v, 0.05 + 0.9 * to_goal)
        return v, w, False


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
