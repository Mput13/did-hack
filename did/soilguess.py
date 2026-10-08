"""Догадка о размере дорогого участка: измерен дорогой пол — значит, и рядом, скорее всего, дорого.

Модель грунта (did.belief.SoilModel) знает стоимость только там, где робот проехал, и чуть шире.
Поэтому маршрут, задев край дорогой зоны, продолжает идти сквозь неё: непроверенный пол впереди
считается обычным. Здесь непроверенные клетки вокруг измеренного дорогого места получают наценку,
убывающую с расстоянием, и маршрут обходит зону стороной, если объезд короче ожидаемой переплаты.
Наценка — нежелание ехать, а не оценка расхода: в запас на возврат она не входит и пропадает там,
где робот проехал и увидел обычный расход.
"""
import numpy as np


def suspect_grid(soil, radius, threshold=1.45, min_m=0.03):
    """Множитель «подозрения» по клеткам карты: 1 — нет причин считать пол дорогим."""
    raw = np.divide(soil.drain, soil.dist, out=np.ones_like(soil.drain), where=soil.dist > min_m)
    hot = (raw >= threshold) & (soil.dist > min_m)
    if radius <= 0.0 or not hot.any():
        return None
    _, conf = soil._estimate()
    ys, xs = np.mgrid[0:soil.h, 0:soil.w]
    extra = np.zeros(raw.shape)
    for j, i in zip(*np.nonzero(hot)):
        d = np.hypot(xs - i, ys - j) * soil.res
        extra = np.maximum(extra, (min(raw[j, i], 6.0) - 1.0) * np.clip(1.0 - d / radius, 0.0, 1.0))
    return soil._fine(1.0 + extra * (1.0 - conf), fill=1.0)
