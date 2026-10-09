"""Проба P3: пересчёт карты образцов после сбора без потери соседнего образца (research/findings/P3.md).

За флажком настройки `exact_replay`, по умолчанию выключен.

После сбора карта откатывается к прошлому сбору, и показания проигрываются заново: теперь известно, что
один образец лежал здесь. Прежний пересчёт (SampleBelief.collected) делит показания на два сорта жёстким
порогом: показание, которое собранный образец объясняет с точностью 3,5 разброса (это ±0,55 м по
расстоянию), считается только свидетельством «ближе ничего нет». Если то же показание гораздо лучше
объясняет другой образец, лежавший чуть ближе, это свидетельство о нём теряется, и уверенный кандидат
исчезает с карты.

Здесь показания проигрываются по той же формуле Байеса, что и при обычном обновлении, но с собранным
образцом на карте: его клетка на время пересчёта занята наверняка. Тогда каждое показание само делится
между «это был собранный образец» и «это был другой, ближе» — по тому, что его лучше объясняет.
"""
import math

import numpy as np

PIN = 0.995            # клетка собранного образца на время пересчёта (верхний предел вероятности на карте)
PLACE_SIGMA = 0.06     # неточность места собранного образца в единицах показания (как в прежнем пересчёте)


def exact_replay(b, sx, sy):
    """Пересчитать карту b (SampleBelief) после сбора образца у (sx, sy). Замена тела SampleBelief.collected."""
    k = int(np.hypot(b.cx - sx, b.cy - sy).argmin())
    rest = max(0, b.left - 1)
    b.p = b._epoch.copy()
    for rx, ry, z, sigma in b._log:
        b.p[k] = PIN
        b._apply(rx, ry, z, math.hypot(sigma, PLACE_SIGMA))
        b.p[k] = 0.0
        total = b.p.sum()
        if total > 0:                              # остальных образцов — на один меньше
            b.p = np.clip(b.p * (rest / total), 1e-7, 0.995)
    b.left = rest
    b.clear_disc(sx, sy, 0.35)
    b._normalize()
    b._epoch = b.p.copy()
    b._log = []


# ======================================================================================================
# B1 (research/findings/B1.md): пересчёт, который не теряет соседа и не верит показаниям при сбое датчика.
# За настройкой `neighbor_replay` ('' — прежний пересчёт), пресеты *_v5.
# ======================================================================================================

NOISE_LAG = 12         # столько показаний нужно оценке шума (SensorHealth), чтобы заметить сбой: они уже шумные
STUCK_RUN = 3          # столько одинаковых до последнего знака показаний подряд — датчик залип
KNOWN_NEAR = 0.5       # м: место «про собранный образец», если оно не дальше
LOST = 0.5             # место стёрто, если уверенность упала до этой доли прежней и ниже


def suspects(log, nominal):
    """Каким показаниям журнала нельзя верить: маска длиной в журнал.

    Признаки — только те, что видны агенту в самом журнале показаний:
      шум      — показание записано с шумом выше обычного (оценка шума уже подняла тревогу), а также
                 NOISE_LAG показаний перед первым таким: тревога запаздывает, они сняты уже при сбое;
      залип    — STUCK_RUN и больше показаний подряд, одинаковых до последнего знака и не упёршихся в 0
                 или 1: у исправного датчика шум есть всегда. Первое из них настоящее, остальные — нет.
    """
    n = len(log)
    bad = np.zeros(n, dtype=bool)
    if n == 0:
        return bad
    z = np.array([r[2] for r in log])
    loud = np.array([r[3] for r in log]) > nominal * (1.0 + 1e-9)
    bad |= loud
    for i in np.nonzero(loud & ~np.concatenate(([False], loud[:-1])))[0]:      # начала шумных отрезков
        bad[max(0, i - NOISE_LAG):i] = True
    start = 0
    for i in range(1, n + 1):
        if i == n or abs(z[i] - z[start]) > 1e-9:
            if i - start >= STUCK_RUN and 0.0 < z[start] < 1.0:
                bad[start + 1:i] = True
            start = i
    return bad


def legacy_replay(b, sx, sy, skip=None):
    """Прежний пересчёт — то же, что тело SampleBelief.collected (совпадение проверяет tests/test_b1.py).

    skip — маска показаний, которые в карту не идут (по умолчанию идут все, как раньше)."""
    b.p = b._epoch.copy()
    for i, (rx, ry, z, sigma) in enumerate(b._log):
        if skip is not None and skip[i]:
            continue
        sig = math.hypot(sigma, 0.06)
        f_known = max(0.0, float(b._resp(math.hypot(sx - rx, sy - ry))))
        if abs(z - f_known) > 3.5 * sig:
            b._apply(rx, ry, z, sigma)
            continue
        f = b._resp(np.hypot(b.cx - rx, b.cy - ry))
        closer = f > f_known
        lr = np.exp(np.minimum(0.0, -0.5 * ((z - f[closer]) / sig) ** 2 + 0.5 * ((z - f_known) / sig) ** 2))
        b.p[closer] = np.maximum(b.p[closer] * lr, 1e-7)
    b.left = max(0, b.left - 1)
    b.clear_disc(sx, sy, 0.35)
    b._normalize()
    b._epoch = b.p.copy()
    b._log = []


def trusted_replay(b, sx, sy, nominal, skip=None):
    """Точный пересчёт (как exact_replay), но показания, которым нельзя верить, в карту не идут вовсе.

    skip — готовая маска недоверия (для разбора в инструментах); по умолчанию её строит suspects.
    Число оставшихся образцов и нормировка согласованы: после каждого показания сумма вероятностей по
    арене равна числу образцов без собранного, в конце b.left на один меньше.
    """
    k = int(np.hypot(b.cx - sx, b.cy - sy).argmin())
    rest = max(0, b.left - 1)
    bad = suspects(b._log, nominal) if skip is None else skip
    b.p = b._epoch.copy()
    for (rx, ry, z, sigma), distrust in zip(b._log, bad):
        if distrust:
            continue
        b.p[k] = PIN
        b._apply(rx, ry, z, math.hypot(sigma, PLACE_SIGMA))
        b.p[k] = 0.0
        total = b.p.sum()
        if total > 0:
            b.p = np.clip(b.p * (rest / total), 1e-7, 0.995)
    b.left = rest
    b.clear_disc(sx, sy, 0.35)
    b._normalize()
    b._epoch = b.p.copy()
    b._log = []


class NeighborReplay:
    """Пересчёт карты после сбора для пресетов *_v5. Вызывается как replay(b, sx, sy) из SampleBelief.collected.

    mode='gated_clean' (пресеты *_v5) — прежний пересчёт, но без показаний, которым нельзя верить (suspects).
                   Если при этом стирается известное место — до сбора на карте был кандидат (уверенность не ниже
                   candidate_mass, дальше KNOWN_NEAR от собранного образца), а после пересчёта от него осталась
                   доля LOST и меньше, — и точный пересчёт trusted_replay это место сохраняет, берётся его карта.
                   Если показаний при сбое в журнале нет и известное место не стирается, карта совпадает с
                   прежней до последней цифры.
    mode='gated' — то же, но прежний пересчёт идёт по всем показаниям (на отладке — столько же очков и больше
                   ложных кандидатов после сбоя датчика; в пресеты не вошёл).
    mode='full'  — всегда trusted_replay (на отладке хуже: меняет почти каждый прогон и прибавляет въезды в
                   опасные зоны; в пресеты не вошёл).
    """

    def __init__(self, nominal, candidate_mass=0.35, mode='gated_clean'):
        if mode not in ('full', 'gated', 'gated_clean'):
            raise ValueError(f'Unknown neighbor_replay: {mode}')
        self.nominal, self.candidate_mass, self.mode = nominal, candidate_mass, mode
        self.stats = {'collects': 0, 'switched': 0, 'skipped': 0}      # для разбора: сколько раз карта взята новая

    def __call__(self, b, sx, sy):
        self.stats['collects'] += 1
        if self.mode == 'full':
            self.stats['switched'] += 1
            self.stats['skipped'] += int(suspects(b._log, self.nominal).sum())
            return trusted_replay(b, sx, sy, self.nominal)
        known = [(c['x'], c['y'], b.prob_within(c['x'], c['y'], 0.25))
                 for c in b.candidates(min_mass=self.candidate_mass)
                 if math.hypot(c['x'] - sx, c['y'] - sy) > KNOWN_NEAR]
        epoch, log, left = b._epoch, b._log, b.left
        legacy_replay(b, sx, sy, suspects(log, self.nominal) if self.mode == 'gated_clean' else None)
        lost = [(x, y, m) for x, y, m in known if b.prob_within(x, y, 0.25) <= LOST * m]
        if not lost:
            return
        old = (b.p, b._epoch, b._log, b.left)
        b._epoch, b._log, b.left = epoch, log, left
        trusted_replay(b, sx, sy, self.nominal)
        if any(b.prob_within(x, y, 0.25) > LOST * m for x, y, m in lost):
            self.stats['switched'] += 1
            self.stats['skipped'] += int(suspects(log, self.nominal).sum())
        else:
            b.p, b._epoch, b._log, b.left = old


VARIANTS = {'legacy': legacy_replay}      # для tools/b1_trace.py: пересчёты, которые можно сравнить на одном входе
