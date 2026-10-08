"""Куда ехать за измерением: ожидаемая польза показаний датчика образцов, в битах.

Прикидка «сколько вероятности вокруг точки» не отличает место, где показание что-то решит, от
места, где оно повторит уже известное. Здесь польза считается прямо: карта вероятностей сама
говорит, какие показания возможны, и для каждого из них видно, насколько карта станет определённее.

  1. Из карты берутся D гипотез о мире: где лежат оставшиеся образцы (клетки — по их вероятности).
  2. Для каждой гипотезы считается, что покажет датчик в точках по дороге и в конце пути.
  3. Этими показаниями пробно обновляется копия карты (SampleBelief.trial).
  4. Польза — насколько в среднем по гипотезам уменьшилась неопределённость карты (энтропия).

Точки сравниваются на одних и тех же гипотезах и одном и том же шуме, поэтому разница между
точками не тонет в случайности выборки.
"""
import math

import numpy as np

STEP = 0.4          # м: шаг пробных измерений вдоль дороги
STOPS = 4           # не больше стольких пробных измерений на одну дорогу, последнее — в конце
READ_EVERY = 0.04   # м пути между настоящими показаниями: 5 Гц при скорости около 0,2 м/с
MAX_POINTS = 12     # столько точек оценивается точно; остальные отсеивает дешёвая прикидка
DRAWS = 16          # гипотез о мире на одно решение
COST_FLOOR = 1.0    # ед. заряда: добавка к цене дороги, чтобы ближняя точка не выигрывала одной близостью


def entropy_bits(p):
    """Неопределённость карты: сумма энтропий клеток «образец есть / нет», биты. p — (n,) или (D, n)."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)).sum(axis=-1)


def sample_worlds(belief, rng, draws=DRAWS):
    """D гипотез о мире: номера клеток, где лежат оставшиеся образцы, массив (D, left).

    Клетки выбираются без повторов с весом p / (1 − p): образцов в каждой гипотезе ровно столько,
    сколько осталось собрать, — как и считает сама карта.
    """
    k = int(min(max(belief.left, 0), belief.n))
    if k == 0:
        return np.zeros((draws, 0), dtype=int)
    p = np.clip(belief.p, 1e-12, 1.0 - 1e-9)
    keys = np.log(p / (1.0 - p)) + rng.gumbel(size=(draws, belief.n))
    return np.argpartition(-keys, k - 1, axis=1)[:, :k]


def readings(belief, worlds, x, y, noise):
    """Показание датчика в (x, y) в каждой гипотезе: близость к ближайшему образцу плюс шум."""
    if worlds.shape[1] == 0:
        return np.clip(noise, 0.0, 1.0)
    d = np.hypot(belief.cx[worlds] - x, belief.cy[worlds] - y).min(axis=1)
    return np.clip(np.maximum(0.0, 1.0 - d / belief.range) + noise, 0.0, 1.0)


def path_stops(path, step=None, stops=None):
    """Точки пробных измерений вдоль дороги через равные отрезки, последняя — конец пути.

    Возвращает (точки (k, 2), длина отрезка дороги на одну точку).
    """
    pts = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts[-1:], 0.0
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    k = int(min(stops or STOPS, max(1, math.ceil(s[-1] / (step or STEP)))))
    at = s[-1] * np.arange(1, k + 1) / k
    return np.column_stack([np.interp(at, s, pts[:, 0]), np.interp(at, s, pts[:, 1])]), float(s[-1] / k)


def expected_gain(belief, x, y, sigma, rng, draws=DRAWS, origin=None, path=None, worlds=None, noise=None):
    """На сколько бит измерение в (x, y) в среднем уменьшит неопределённость карты.

    origin — откуда робот поедет: тогда польза набирается и по дороге, в нескольких точках вдоль
    прямой (или вдоль path — уже построенной дороги [(x, y), ...]). Без origin и path — одно
    показание в самой точке. worlds и noise (D, STOPS), единичный шум — общие для сравнения точек.
    """
    if belief.left <= 0:
        return 0.0
    if worlds is None:
        worlds = sample_worlds(belief, rng, draws)
    if noise is None:
        noise = rng.standard_normal((len(worlds), STOPS))
    if path is None and origin is not None:
        path = [origin, (x, y)]
    stops, piece = path_stops(path) if path is not None else (np.array([[x, y]]), 0.0)
    # Датчик работает и в пути: пробное измерение заменяет все показания своего отрезка дороги,
    # поэтому его шум меньше в корень из их числа.
    s = sigma / math.sqrt(max(1.0, piece / READ_EVERY))
    spread = math.hypot(s, 0.4 * belief.res / belief.range)  # тот же шум, что закладывает карта
    p = belief.p
    for j, (sx, sy) in enumerate(stops):
        z = readings(belief, worlds, sx, sy, spread * noise[:, j])
        p = belief.trial(sx, sy, z, s, p=p)
    return float(max(0.0, entropy_bits(belief.p) - entropy_bits(p).mean()))


def rank_points(belief, points, cost_of, sigma, rng, mode='infogain', origin=None, path_of=None,
                draws=None, cost_floor=None):
    """Упорядочить точки разведки по пользе на единицу заряда.

    points — [{'x', 'y', 'mass', ...}] из SampleBelief.explore_points; cost_of(point) — заряд на
    дорогу; path_of(point) — сама дорога, если она известна (иначе прямая от origin).
    mode: 'infogain' — ожидаемое уменьшение неопределённости, биты; 'mass' — прежняя прикидка,
    сумма вероятностей вокруг точки. Возвращает копии точек с полями gain, cost, score, лучшие первыми.
    """
    draws, cost_floor = draws or DRAWS, COST_FLOOR if cost_floor is None else cost_floor
    points = [(p, float(cost_of(p))) for p in points]
    if mode == 'infogain':
        # Точный расчёт стоит ~2 мс на точку, поэтому он делается только для лучших по прикидке.
        points.sort(key=lambda pc: -pc[0].get('mass', 1.0) / (pc[1] + cost_floor))
        points = points[:MAX_POINTS]
        worlds = sample_worlds(belief, rng, draws)
        noise = rng.standard_normal((draws, STOPS))
    out = []
    for p, cost in points:
        if mode == 'infogain':
            path = path_of(p) if path_of else None
            gain = expected_gain(belief, p['x'], p['y'], sigma, rng, origin=origin, path=path,
                                 worlds=worlds, noise=noise) if math.isfinite(cost) else 0.0
        else:
            gain = float(p['mass'])
        out.append({**p, 'gain': gain, 'cost': cost,
                    'score': gain / (cost + cost_floor) if math.isfinite(cost) else 0.0})
    out.sort(key=lambda q: -q['score'])
    return out
