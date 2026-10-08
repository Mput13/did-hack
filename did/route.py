"""Фиксированный маршрут объезда арены: план, составленный до старта и не меняющийся в прогоне.

Точки выбираются жадно так, чтобы любое место арены оказалось не дальше COVER от маршрута,
порядок — ближайший сосед с улучшением перестановками (2-opt). Это контрольный агент в опытах.
"""
import math
from functools import lru_cache

import numpy as np

from .config import BASE

COVER = 0.8      # м: на такой дистанции образец уже уверенно «слышен» датчиком
LATTICE = 0.25   # м: шаг сетки точек-кандидатов


@lru_cache(maxsize=4)
def survey_route(arena, cover=COVER):
    """Список точек [(x, y), ...] без базы: объезд начинается и заканчивается на базе."""
    step = max(1, round(LATTICE / arena.res))
    ys, xs = np.nonzero(arena.clear >= 0.30)
    keep = (ys % step == 0) & (xs % step == 0)
    cand = np.array([arena.g2w(i, j) for j, i in zip(ys[keep], xs[keep])])
    fy, fx = np.nonzero(arena.clear >= 0.15)
    cells = np.array([arena.g2w(i, j) for j, i in zip(fy[::2], fx[::2])])
    near = np.hypot(cand[:, None, 0] - cells[None, :, 0], cand[:, None, 1] - cells[None, :, 1]) <= cover

    uncovered = np.ones(len(cells), dtype=bool)
    uncovered &= np.hypot(cells[:, 0] - BASE[0], cells[:, 1] - BASE[1]) > cover   # стартовая точка уже «слышит»
    chosen = []
    while uncovered.sum() > 0.01 * len(cells):
        gain = (near & uncovered[None, :]).sum(axis=1)
        k = int(gain.argmax())
        if gain[k] == 0:
            break
        chosen.append(k)
        uncovered &= ~near[k]
    pts = [tuple(cand[k]) for k in chosen]

    # Порядок обхода: ближайший сосед от базы, затем 2-opt по евклидовой длине замкнутого тура.
    tour, rest, cur = [], pts[:], BASE
    while rest:
        nxt = min(rest, key=lambda p: math.dist(cur, p))
        rest.remove(nxt)
        tour.append(nxt)
        cur = nxt

    def length(order):
        chain = [BASE] + order + [BASE]
        return sum(math.dist(a, b) for a, b in zip(chain, chain[1:]))

    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                alt = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                if length(alt) + 1e-9 < length(tour):
                    tour, improved = alt, True
    return [(round(float(x), 2), round(float(y), 2)) for x, y in tour]
