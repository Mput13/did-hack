"""Счёт команды из нескольких роботов и правило столкновения роботов друг с другом.

Определено до прогонов (исследование M1) и одинаково для быстрого симулятора и для судьи в Gazebo.

Счёт команды — сумма счетов роботов по обычным правилам (did/config.py):
  10 за каждый образец, кто бы его ни собрал; 20 за возврат каждого робота на своё место базы;
  штрафы каждого робота; 0,1 за единицу остатка заряда у каждого вернувшегося.
Два робота могут набрать больше одного просто потому, что возвратов и остатков заряда два, поэтому
для сравнения с одним роботом главные показатели — доля собранных образцов и время, а счёт даётся
и на команду, и на одного робота (счёт команды, делённый на число роботов).

Столкновение роботов: центры ближе суммы радиусов (плюс тот же запас 5 мм, что и у стен). Штраф
получает каждый участник контакта, который ещё в прогоне: судья не разбирает, кто в кого въехал.
"""
import math

import numpy as np

from .config import BASE, ROBOT_RADIUS

# Места на базе. Первое — база из условия задачи. Второе — в 0,4 м от неё к оси арены:
# генератор сценариев держит круг 0,45 м вокруг базы свободным от грунтов, а образцы и опасные зоны
# не ближе 1 м, поэтому второе место чисто в любом сценарии (проверено на сценариях 1–80 и
# 10001–10040 уровней medium и hard). Между центрами 0,40 м — вдвое больше суммы радиусов (0,21 м):
# на старте и на стоянке роботы друг другу не мешают, а зазор до стен у второго места больше (0,72 м).
SLOTS = (BASE, (-2.0, -0.1))
ROBOT_NAMES = ('tb1', 'tb2')
CONTACT_M = 2 * ROBOT_RADIUS            # центры ближе — роботы упёрлись друг в друга
CONTACT_MARGIN_M = 0.005                # запас судьи, как у стен (collision_clearance_m − радиус)
CONTACT_RELEASE_M = 0.03                # разъехались на столько сверх контакта — следующий контакт новый
UNCOLLECTED_T = 600.0                   # время, которое приписывается несобранному образцу (предел прогона)


def team_score(scores, collected_times, n_samples, time_limit=UNCOLLECTED_T):
    """Итог команды по итогам судей роботов.

    scores — словари Judge.score() по роботам; collected_times — времена всех сборов, секунды.
    """
    n = len(scores)
    times = sorted(collected_times)
    got = len(times)
    total = sum(s['score'] for s in scores)
    return {
        'robots': n,
        'score': round(total, 2),
        'score_per_robot': round(total / n, 2),
        'samples_collected': got,
        'samples_total': n_samples,
        'samples_share': round(got / max(1, n_samples), 3),
        # Доля вернувшихся РОБОТОВ, а не прогонов: 0,5 — вернулся один из двух. «Вернулись все» — returned_all.
        'returned': round(sum(bool(s['returned']) for s in scores) / n, 3),
        'returned_all': all(s['returned'] for s in scores),
        'battery_left': round(sum(s['battery'] for s in scores), 2),
        'distance': round(sum(s['distance'] for s in scores), 3),
        'time': round(max(s['t'] for s in scores), 2),                          # когда закончил последний
        'collisions': sum(s['collisions'] for s in scores),
        'false_collects': sum(s['false_collects'] for s in scores),
        'hazard_hits': sum(s['hazard_hits'] for s in scores),
        'penalties': sum(s['collisions'] + s['false_collects'] + s['hazard_hits'] for s in scores),
        # Время до последнего сбора; None — ничего не собрано.
        't_last_collect': round(times[-1], 2) if times else None,
        # Среднее время до образца: несобранный считается как предел времени прогона. Один показатель и
        # для «сколько собрано», и для «как быстро»: его нельзя улучшить, собрав меньше.
        'mean_sample_time': round((sum(times) + time_limit * (n_samples - got)) / max(1, n_samples), 2),
    }


def min_gap(track_a, track_b):
    """Наименьшее расстояние между центрами двух роботов по их путям {t, x, y}, м; None — пути пусты.

    Пути пишут разные процессы: времена точек и длины у них разные, поэтому сравнивать точки по номеру
    нельзя. Оба пути приводятся к общим временам (линейно между точками), и на каждом отрезке берётся
    ближайшее сближение, а не только концы. До первой и после последней точки робот стоит на месте:
    закончивший прогон остаётся неподвижным препятствием для второго.
    """
    ta, tb = np.asarray(track_a['t'], float), np.asarray(track_b['t'], float)
    if not len(ta) or not len(tb):
        return None
    t = np.union1d(ta, tb)
    dx = np.interp(t, ta, track_a['x']) - np.interp(t, tb, track_b['x'])
    dy = np.interp(t, ta, track_a['y']) - np.interp(t, tb, track_b['y'])
    best = float(np.hypot(dx, dy).min())
    for i in range(len(t) - 1):                 # между общими временами разность положений линейна
        ux, uy = dx[i + 1] - dx[i], dy[i + 1] - dy[i]
        uu = ux * ux + uy * uy
        if uu > 0.0:
            s = min(1.0, max(0.0, -(dx[i] * ux + dy[i] * uy) / uu))
            best = min(best, math.hypot(dx[i] + s * ux, dy[i] + s * uy))
    return best
