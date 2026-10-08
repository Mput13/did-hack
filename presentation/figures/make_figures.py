#!/usr/bin/env python3
"""Рисунки и числа для презентации чекпоинта 1.

Источники: реальная карта арены (maps/map.pgm), реальные сканы лидара из симуляции
(presentation/data/scans.json) и офлайн-модели. Картинки пишутся в presentation/assets,
числа для слайдов — в presentation/data/results.json.

Запуск: python3 presentation/figures/make_figures.py
"""
import heapq
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'presentation' / 'assets'
DATA = ROOT / 'presentation' / 'data'

RES = 0.05
ORIGIN = (-10.0, -10.0)
START = (-2.0, -0.5)
ROBOT_R = 0.105
LIDAR_OFFSET_X = -0.032   # base_scan относительно центра робота (модель Burger)
VIEW = (-3.15, 2.95, -2.85, 2.9)

# Палитра: проверена scripts/validate_palette.js (категориальные 1–3 по всем парам,
# порядковая оранжевая шкала для грунтов).
INK, SECOND, MUTED, AXIS = '#0b0b0b', '#52514e', '#898781', '#c3c2b7'
WALL, FLOOR, SURFACE = '#2b3138', '#eef0f2', '#ffffff'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SOIL = {2: '#f09d78', 3: '#eb6834', 4: '#b8481b'}
CRITICAL = '#d03b3b'
GRAY_CONTEXT = '#9a9891'
BLUE_SEQ = LinearSegmentedColormap.from_list(
    'blue_seq', ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 13, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': SECOND,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'legend.frameon': False,
})


# ---------- карта ----------

IMG = np.array(Image.open(ROOT / 'maps' / 'map.pgm'))
H, W = IMG.shape
OCC = IMG < 100
FREE = IMG > 250
EXTENT = (ORIGIN[0], ORIGIN[0] + W * RES, ORIGIN[1], ORIGIN[1] + H * RES)
DIST_TO_OBST = ndi.distance_transform_edt(~OCC) * RES   # м до ближайшей занятой клетки
SOLID = OCC | (ndi.binary_fill_holes(OCC | FREE) & ~FREE)   # стены и столбы целиком, для отрисовки


def ru(value, digits=1):
    return f'{value:.{digits}f}'.replace('.', ',').replace('-', '−')


def cell(x, y):
    return H - 1 - int(math.floor((y - ORIGIN[1]) / RES)), int(math.floor((x - ORIGIN[0]) / RES))


def center(r, c):
    return ORIGIN[0] + (c + 0.5) * RES, ORIGIN[1] + (H - 1 - r + 0.5) * RES


XS = ORIGIN[0] + (np.arange(W) + 0.5) * RES
YS = ORIGIN[1] + (H - 1 - np.arange(H) + 0.5) * RES
GX, GY = np.meshgrid(XS, YS)


def new_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def draw_map(ax, ticks=True):
    rgb = np.ones((H, W, 3))
    rgb[FREE] = to_rgb(FLOOR)
    rgb[SOLID] = to_rgb(WALL)
    ax.imshow(rgb, extent=EXTENT, origin='upper', interpolation='nearest', zorder=0)
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect('equal')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if ticks:
        ax.set_xticks(range(-3, 3))
        ax.set_yticks(range(-2, 3))
        ax.set_xlabel('x, м')
        ax.set_ylabel('y, м')
        ax.tick_params(length=3, width=0.8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)


def draw_walls(ax, zorder=2.5):
    """Стены поверх заливок, чтобы зоны и карты убеждений их не перекрывали."""
    rgba = np.zeros((H, W, 4))
    rgba[SOLID] = (*to_rgb(WALL), 1.0)
    ax.imshow(rgba, extent=EXTENT, origin='upper', interpolation='nearest', zorder=zorder)


def legend_below(ax, handles, ncol=1):
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.02, -0.14), ncol=ncol, fontsize=12.5,
              labelcolor=INK, borderaxespad=0.0, handletextpad=0.5, columnspacing=1.4)


def draw_robot(ax, x, y, yaw=0.0, color=WALL, zorder=6):
    ax.add_patch(Circle((x, y), ROBOT_R, facecolor=color, edgecolor=SURFACE, linewidth=2, zorder=zorder))
    ax.plot([x, x + 0.2 * math.cos(yaw)], [y, y + 0.2 * math.sin(yaw)], color=color, linewidth=2.2,
            solid_capstyle='round', zorder=zorder)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


# ---------- 1. арена ----------

def fig_arena():
    fig, ax = new_axes((6.0, 5.6))
    draw_map(ax)
    draw_robot(ax, *START)
    ax.annotate('старт и база\n(−2,0; −0,5)', xy=(START[0] - 0.09, START[1] - 0.09), xytext=(-3.12, -2.15),
                color=INK, fontsize=13, arrowprops=dict(arrowstyle='-', color=SECOND, linewidth=0.8),
                ha='left', va='top')
    ax.annotate('9 столбов,\nшаг 1,1 м', xy=(1.14, 1.08), xytext=(1.7, 2.4), color=INK, fontsize=13,
                arrowprops=dict(arrowstyle='-', color=SECOND, linewidth=0.8), ha='left', va='center')
    save(fig, 'arena.png')


def fig_title_art():
    """Тёмная арена с реальными точками лидара — для титульного слайда."""
    shots = json.loads((DATA / 'scans.json').read_text())['shots']
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    rgba = np.zeros((H, W, 4))
    rgba[FREE] = (*to_rgb('#1f262d'), 1.0)
    rgba[SOLID] = (*to_rgb('#4a545e'), 1.0)
    ax.imshow(rgba, extent=EXTENT, origin='upper', interpolation='nearest', zorder=0)
    for shot in shots:
        x, y, yaw = shot['world']
        lx, ly = x + LIDAR_OFFSET_X * math.cos(yaw), y + LIDAR_OFFSET_X * math.sin(yaw)
        pts = [(lx + r * math.cos(yaw + shot['angle_min'] + k * shot['angle_increment']),
                ly + r * math.sin(yaw + shot['angle_min'] + k * shot['angle_increment']))
               for k, r in enumerate(shot['ranges']) if r is not None and shot['range_min'] < r < shot['range_max']]
        ax.scatter(*zip(*pts), s=9, color=S2, linewidths=0, zorder=3)
    ax.add_patch(Circle(START, ROBOT_R, facecolor='#ffffff', edgecolor='none', zorder=5))
    ax.plot([START[0], START[0] + 0.22], [START[1], START[1]], color='#ffffff', linewidth=2.4,
            solid_capstyle='round', zorder=5)
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect('equal')
    ax.axis('off')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'arena_dark.png', dpi=220, bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.close(fig)


# ---------- 2. лидар на карте ----------

def fig_lidar(results):
    shots = json.loads((DATA / 'scans.json').read_text())['shots']
    fig, ax = new_axes((6.0, 5.6))
    draw_map(ax)
    errors, handles = [], []
    for i, (shot, color) in enumerate(zip(shots, (S1, S2, S3)), start=1):
        x, y, yaw = shot['world']
        lx, ly = x + LIDAR_OFFSET_X * math.cos(yaw), y + LIDAR_OFFSET_X * math.sin(yaw)
        px, py = [], []
        for k, rng in enumerate(shot['ranges']):
            if rng is None or not (shot['range_min'] < rng < shot['range_max']):
                continue
            a = yaw + shot['angle_min'] + k * shot['angle_increment']
            px.append(lx + rng * math.cos(a))
            py.append(ly + rng * math.sin(a))
        for qx, qy in zip(px, py):
            r, c = cell(qx, qy)
            if 0 <= r < H and 0 <= c < W:
                errors.append(DIST_TO_OBST[r, c])
        ax.scatter(px, py, s=13, color=color, edgecolors=SURFACE, linewidths=0.4, zorder=4)
        draw_robot(ax, x, y, yaw, color=color)
        ax.text(x - 0.02, y - 0.2, str(i), color=INK, fontsize=14, fontweight='bold', ha='center', va='top', zorder=7)
        handles.append(Line2D([], [], marker='o', linestyle='', color=color, markersize=8,
                              label=f'снимок {i}'))
    legend_below(ax, handles, ncol=3)
    errors = np.array(errors)
    results['lidar'] = {
        'shots': len(shots), 'points': int(errors.size),
        'median_cm': round(float(np.median(errors)) * 100, 1),
        'p95_cm': round(float(np.percentile(errors, 95)) * 100, 1),
        'within_10cm_pct': round(float((errors <= 0.10).mean()) * 100, 1),
        'poses': [[round(v, 3) for v in s['world']] for s in shots],
    }
    save(fig, 'lidar_on_map.png')


# ---------- 3. A* по карте стоимостей ----------

INFLATE_M = 0.20
_r = int(round(INFLATE_M / RES))
_yy, _xx = np.mgrid[-_r:_r + 1, -_r:_r + 1]
BLOCKED = ndi.binary_dilation(~FREE, structure=(_xx ** 2 + _yy ** 2 <= _r ** 2))
NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(blocked, mult, start, goal):
    """A* по 8-связной сетке. Цена шага = длина (м) × средний множитель двух клеток."""
    g = np.full(blocked.shape, np.inf)
    parent = {}
    g[start] = 0.0
    heap = [(0.0, start)]
    closed = np.zeros(blocked.shape, dtype=bool)
    while heap:
        _, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        if cur == goal:
            break
        closed[cur] = True
        r, c = cur
        for dr, dc, d in NEIGH:
            nr, nc = r + dr, c + dc
            if blocked[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue   # не срезаем углы препятствий
            cand = g[cur] + d * RES * 0.5 * (mult[cur] + mult[nr, nc])
            if cand < g[nr, nc]:
                g[nr, nc] = cand
                parent[(nr, nc)] = cur
                dy, dx = abs(nr - goal[0]), abs(nc - goal[1])
                h = RES * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))
                heapq.heappush(heap, (cand + h, (nr, nc)))
    if not np.isfinite(g[goal]):
        return None, math.inf
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], float(g[goal])


def path_xy(path):
    return np.array([center(r, c) for r, c in path])


def path_length(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def segment_cost(a, b, blocked, mult):
    n = max(2, int(math.dist(a, b) / (RES / 2)))
    xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
    rows = (H - 1 - np.floor((ys - ORIGIN[1]) / RES)).astype(int)
    cols = np.floor((xs - ORIGIN[0]) / RES).astype(int)
    if blocked[rows, cols].any():
        return math.inf
    return float(mult[rows, cols].mean() * math.dist(a, b))


def to_waypoints(xy, blocked, mult):
    """Прореживание пути до путевых точек: прямой отрезок берём, если он не дороже пути."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    rows = (H - 1 - np.floor((xy[:, 1] - ORIGIN[1]) / RES)).astype(int)
    cols = np.floor((xy[:, 0] - ORIGIN[0]) / RES).astype(int)
    m = mult[rows, cols]
    cum = np.concatenate([[0.0], np.cumsum(seg * 0.5 * (m[:-1] + m[1:]))])
    out, i = [xy[0]], 0
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1 and segment_cost(xy[i], xy[j], blocked, mult) > (cum[j] - cum[i]) * 1.02:
            j -= 1
        out.append(xy[j])
        i = j
    return np.array(out)


def ellipse_mask(cx, cy, a, b, angle_deg=0.0):
    t = math.radians(angle_deg)
    dx, dy = GX - cx, GY - cy
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def draw_zones(ax, zones, label_shift=(0.0, 0.0)):
    rgba = np.zeros((H, W, 4))
    for cx, cy, a, b, ang, m in sorted(zones, key=lambda z: z[5]):
        rgba[ellipse_mask(cx, cy, a, b, ang) & FREE] = (*to_rgb(SOIL[m]), 0.92)
    ax.imshow(rgba, extent=EXTENT, origin='upper', interpolation='nearest', zorder=1)
    draw_walls(ax)
    for cx, cy, a, b, ang, m in zones:
        ax.text(cx + label_shift[0], cy + label_shift[1], f'×{m}', color='#ffffff' if m >= 3 else INK,
                fontsize=15, fontweight='bold', ha='center', va='center', zorder=3)


def fig_astar(results):
    zone = (-0.55, -0.45, 0.80, 0.62, 0.0, 3)
    mult = np.ones((H, W))
    mult[ellipse_mask(*zone[:5])] = zone[5]
    start, goal_xy = cell(*START), (1.85, -0.45)
    goal = cell(*goal_xy)
    assert not BLOCKED[start] and not BLOCKED[goal]

    naive, _ = astar(BLOCKED, np.ones((H, W)), start, goal)
    t = []
    for _ in range(15):
        t0 = time.perf_counter()
        smart, smart_cost = astar(BLOCKED, mult, start, goal)
        t.append((time.perf_counter() - t0) * 1000)
    naive_xy, smart_xy = path_xy(naive), path_xy(smart)
    naive_cost = sum(math.dist(a, b) * 0.5 * (mult[cell(*a)] + mult[cell(*b)]) for a, b in zip(naive_xy, naive_xy[1:]))
    wps = to_waypoints(smart_xy, BLOCKED, mult)

    fig, ax = new_axes((6.6, 5.6))
    draw_map(ax)
    band = np.zeros((H, W, 4))
    band[BLOCKED & FREE] = (*to_rgb('#c9ced3'), 1.0)   # запас вокруг стен: клетки запрещены
    ax.imshow(band, extent=EXTENT, origin='upper', interpolation='nearest', zorder=0.5)
    draw_zones(ax, [zone], label_shift=(0.0, -0.3))
    ax.plot(naive_xy[:, 0], naive_xy[:, 1], color=GRAY_CONTEXT, linewidth=2.2, linestyle=(0, (4, 3)), zorder=3)
    ax.plot(smart_xy[:, 0], smart_xy[:, 1], color=S1, linewidth=2.6, solid_capstyle='round', zorder=4)
    ax.scatter(wps[1:-1, 0], wps[1:-1, 1], s=46, color=S1, edgecolors=SURFACE, linewidths=1.6, zorder=5)
    draw_robot(ax, *START)
    ax.scatter(*goal_xy, s=150, marker='*', color=INK, edgecolors=SURFACE, linewidths=1.2, zorder=6)
    ax.text(goal_xy[0], goal_xy[1] - 0.2, 'цель', color=INK, fontsize=13, ha='center', va='top')
    legend_below(ax, [
        Line2D([], [], color=S1, linewidth=2.6, marker='o', markersize=7, markeredgecolor=SURFACE,
               label=f'с учётом грунта: {ru(path_length(smart_xy))} м, {ru(smart_cost)} ед. батареи'),
        Line2D([], [], color=GRAY_CONTEXT, linewidth=2.2, linestyle=(0, (4, 3)),
               label=f'кратчайший по длине: {ru(path_length(naive_xy))} м, {ru(naive_cost)} ед. батареи'),
        Line2D([], [], color='#c9ced3', linewidth=7, label=f'запас у стен {ru(INFLATE_M, 2)} м: клетки запрещены'),
    ])
    # Компактный вариант для слайда с гипотезой: те же маршруты, подписи в терминах гипотезы.
    fig2, ax2 = new_axes((6.0, 5.4))
    draw_map(ax2, ticks=False)
    draw_zones(ax2, [zone], label_shift=(0.0, -0.3))
    ax2.plot(naive_xy[:, 0], naive_xy[:, 1], color=GRAY_CONTEXT, linewidth=3.0, linestyle=(0, (4, 3)), zorder=3)
    ax2.plot(smart_xy[:, 0], smart_xy[:, 1], color=S1, linewidth=3.4, solid_capstyle='round', zorder=4)
    draw_robot(ax2, *START)
    ax2.scatter(*goal_xy, s=260, marker='*', color=INK, edgecolors=SURFACE, linewidths=1.2, zorder=6)
    ax2.legend(handles=[
        Line2D([], [], color=S1, linewidth=3.4, label=f'с адаптацией: {ru(smart_cost)} ед. батареи'),
        Line2D([], [], color=GRAY_CONTEXT, linewidth=3.0, linestyle=(0, (4, 3)),
               label=f'фиксированный план: {ru(naive_cost)} ед.'),
    ], loc='upper center', bbox_to_anchor=(0.5, 0.04), fontsize=21, labelcolor=INK, borderaxespad=0.0,
        handletextpad=0.6, handlelength=1.6)
    save(fig2, 'astar_hypothesis.png')

    results['astar'] = {
        'free_cells': int((~BLOCKED).sum()),
        'plan_ms_median': round(float(np.median(t)), 1),
        'naive_len_m': round(path_length(naive_xy), 2), 'naive_energy': round(float(naive_cost), 2),
        'smart_len_m': round(path_length(smart_xy), 2), 'smart_energy': round(float(smart_cost), 2),
        'saving_pct': round((1 - smart_cost / naive_cost) * 100, 1),
        'waypoints': int(len(wps)), 'path_cells': int(len(smart)), 'inflate_m': INFLATE_M,
    }
    save(fig, 'astar.png')


# ---------- 4. пример сценария и маршрут оракула ----------

BATTERY = 60.0
LEVELS = {'easy': (3, 1, 0), 'medium': (5, 3, 0), 'hard': (7, 4, 1)}   # образцы, грунты, опасные зоны


def generate_scenario(seed, level):
    """Прототип генератора: случайные грунты, образцы и опасная зона с проверкой размещения."""
    rng = np.random.default_rng(seed)
    n_samples, n_zones, n_hazards = LEVELS[level]
    free_cells = np.argwhere(~BLOCKED)

    def random_point(min_clear):
        while True:
            r, c = free_cells[rng.integers(len(free_cells))]
            if DIST_TO_OBST[r, c] >= min_clear:
                return center(r, c)

    zones = []
    while len(zones) < n_zones:
        cx, cy = random_point(0.3)
        a, b = rng.uniform(0.5, 0.95), rng.uniform(0.38, 0.65)
        if math.dist((cx, cy), START) < a + 0.5:
            continue
        if any(math.dist((cx, cy), z[:2]) < 1.2 for z in zones):
            continue
        zones.append((cx, cy, a, b, float(rng.uniform(0, 180)), int(rng.choice([2, 3, 4]))))
    hazards = []
    while len(hazards) < n_hazards:
        cx, cy = random_point(0.45)
        if math.dist((cx, cy), START) > 1.4 and all(math.dist((cx, cy), z[:2]) > 0.9 for z in zones):
            hazards.append((cx, cy, 0.38))
    samples = []
    while len(samples) < n_samples:
        p = random_point(0.35)
        if math.dist(p, START) < 1.0 or any(math.dist(p, s) < 0.8 for s in samples):
            continue
        if any(math.dist(p, h[:2]) < h[2] + 0.35 for h in hazards):
            continue
        samples.append(p)
    return {'seed': seed, 'level': level, 'zones': zones, 'hazards': hazards, 'samples': samples}


def oracle(scn):
    """Лучший возможный результат при полном знании мира: перебор подмножеств образцов."""
    mult = np.ones((H, W))
    for z in scn['zones']:
        mult[ellipse_mask(*z[:5])] = np.maximum(mult[ellipse_mask(*z[:5])], z[5])
    blocked = BLOCKED.copy()
    for cx, cy, rad in scn['hazards']:
        blocked |= (GX - cx) ** 2 + (GY - cy) ** 2 <= (rad + ROBOT_R) ** 2
    nodes = [START] + scn['samples']
    n = len(nodes)
    energy = np.zeros((n, n))
    paths = {}
    for i in range(n):
        for j in range(i + 1, n):
            p, c = astar(blocked, mult, cell(*nodes[i]), cell(*nodes[j]))
            energy[i, j] = energy[j, i] = c
            paths[(i, j)] = path_xy(p)
            paths[(j, i)] = paths[(i, j)][::-1]
    m = n - 1
    dp = np.full((1 << m, m), np.inf)
    prev = {}
    for j in range(m):
        dp[1 << j, j] = energy[0, j + 1]
    for mask in range(1 << m):
        for j in range(m):
            if not np.isfinite(dp[mask, j]):
                continue
            for k in range(m):
                if mask >> k & 1:
                    continue
                cand = dp[mask, j] + energy[j + 1, k + 1]
                if cand < dp[mask | 1 << k, k]:
                    dp[mask | 1 << k, k] = cand
                    prev[(mask | 1 << k, k)] = j
    best = (0, 0.0, None, None)
    for mask in range(1, 1 << m):
        for j in range(m):
            total = dp[mask, j] + energy[j + 1, 0]
            count = bin(mask).count('1')
            if total <= BATTERY and (count > best[0] or (count == best[0] and total < best[1])):
                best = (count, float(total), mask, j)
    count, total, mask, j = best
    order = []
    while mask:
        order.append(j + 1)
        nxt = prev.get((mask, j))
        mask ^= 1 << j
        j = nxt
    order = [0] + order[::-1] + [0]
    route = np.vstack([paths[(a, b)] for a, b in zip(order, order[1:])])
    return {'collected': count, 'energy': total, 'order': order, 'route': route,
            'length_m': path_length(route), 'mult': mult}


def fig_scenario(results):
    scn = generate_scenario(seed=2026, level='hard')
    best = oracle(scn)
    fig, ax = new_axes((6.6, 5.6))
    draw_map(ax)
    draw_zones(ax, scn['zones'])
    for cx, cy, rad in scn['hazards']:
        ax.add_patch(Circle((cx, cy), rad, facecolor=CRITICAL, alpha=0.16, edgecolor='none', zorder=1))
        ax.add_patch(Circle((cx, cy), rad, facecolor='none', edgecolor=CRITICAL, linewidth=2, zorder=2))
        ax.text(cx, cy, '!', color=CRITICAL, fontsize=15, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(cx, cy - rad - 0.08, 'опасная зона', color=INK, fontsize=12.5, ha='center', va='top', zorder=3)
    route = best['route']
    ax.plot(route[:, 0], route[:, 1], color=WALL, linewidth=2.0, solid_capstyle='round', zorder=3)
    for k, node in enumerate(best['order'][1:-1], start=1):
        x, y = scn['samples'][node - 1]
        ax.scatter(x, y, s=250, color=S1, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.text(x, y, str(k), color='#ffffff', fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=6)
    draw_robot(ax, *START)
    ax.text(START[0], START[1] - 0.22, 'база', color=INK, fontsize=12.5, ha='center', va='top')
    legend_below(ax, [
        Line2D([], [], marker='o', linestyle='', color=S1, markersize=10, markeredgecolor=SURFACE,
               label='образец (номер — порядок обхода у оракула)'),
        Line2D([], [], color=WALL, linewidth=2.0, label='маршрут оракула: мир известен заранее'),
        Line2D([], [], marker='s', linestyle='', color=SOIL[3], markersize=10,
               label='грунт: расход батареи ×2, ×3, ×4'),
    ])
    results['scenario'] = {
        'seed': scn['seed'], 'level': scn['level'], 'samples': len(scn['samples']),
        'zones': [int(z[5]) for z in scn['zones']], 'hazards': len(scn['hazards']),
        'oracle_collected': best['collected'], 'oracle_energy': round(best['energy'], 1),
        'oracle_length_m': round(best['length_m'], 1), 'battery': BATTERY,
    }
    save(fig, 'scenario.png')


# ---------- 5. поиск образца: байесовская карта против градиента и спирали ----------

SENSOR_R, SENSOR_SIGMA = 2.0, 0.05
STEP = 0.10
MAX_PATH = 60.0
COLLECT_R = 0.30


def sensor_mean(d):
    return np.clip(1.0 - d / SENSOR_R, 0.0, 1.0)


class World:
    """Упрощённый стенд: один образец, движение по прямой, шумный скалярный датчик."""

    def __init__(self, sample, rng):
        self.sample, self.rng = np.array(sample), rng
        self.pos, self.path, self.false_collects, self.trace = np.array(START), 0.0, 0, [np.array(START)]

    def read(self):
        d = float(np.hypot(*(self.pos - self.sample)))
        return float(np.clip(sensor_mean(d) + self.rng.normal(0.0, SENSOR_SIGMA), 0.0, 1.0))

    def move_towards(self, target, dist):
        v = np.array(target) - self.pos
        n = float(np.hypot(*v))
        if n < 1e-9:
            return
        step = min(dist, n)
        self.pos = self.pos + v / n * step
        self.path += step
        self.trace.append(self.pos.copy())

    def collect(self):
        if np.hypot(*(self.pos - self.sample)) <= COLLECT_R:
            return True
        self.false_collects += 1
        return False


_cand = np.argwhere(~BLOCKED)[::1]
BELIEF_CELLS = np.array([center(r, c) for r, c in _cand if r % 2 == 0 and c % 2 == 0])   # сетка 0.1 м
ARENA_PTS = np.array([center(r, c) for r, c in np.argwhere(~BLOCKED)])


def inside_arena(p):
    r, c = cell(*p)
    return 0 <= r < H and 0 <= c < W and FREE[r, c]


def run_bayes(world, snapshots=None):
    logp = np.zeros(len(BELIEF_CELLS))
    target, moved_since_pick = None, 0.0
    while world.path < MAX_PATH:
        s = world.read()
        d = np.hypot(*(BELIEF_CELLS - world.pos).T)
        logp += -((s - sensor_mean(d)) ** 2) / (2 * (1.5 * SENSOR_SIGMA) ** 2)
        post = np.exp(logp - logp.max())
        post /= post.sum()
        if snapshots is not None:
            snapshots.append((world.path, world.pos.copy(), post.copy(), np.array(world.trace)))
        if post[d <= 0.25].sum() >= 0.8:
            if world.collect():
                return True
            logp[d <= COLLECT_R] = -np.inf
            target = None
            continue
        best = int(np.argmax(post))
        if target is None or moved_since_pick >= 0.3 or post[target] < 0.5 * post[best]:
            target, moved_since_pick = best, 0.0
        world.move_towards(BELIEF_CELLS[target], STEP)
        moved_since_pick += STEP
    return False


def gradient_step(world, heading):
    """Оценка градиента двумя пробами по 0.2 м и шаг 0.3 м вдоль него."""
    def avg(n=3):
        return float(np.mean([world.read() for _ in range(n)]))
    s0 = avg()
    if s0 >= 0.86:
        return 'collect', heading
    if s0 < 0.04:
        return 'lost', heading
    h = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-h[1], h[0]])
    world.move_towards(world.pos + h * 0.2, 0.2)
    s1 = avg()
    world.move_towards(world.pos + left * 0.2, 0.2)
    s2 = avg()
    g = (s1 - s0) * h + (s2 - s1) * left
    if np.hypot(*g) < 1e-6:
        return 'move', heading
    heading = math.atan2(g[1], g[0])
    world.move_towards(world.pos + np.array([math.cos(heading), math.sin(heading)]) * 0.3, 0.3)
    return 'move', heading


def run_gradient(world):
    rng = world.rng
    goal = np.array([0.0, 0.0])
    heading = math.atan2(*(goal - world.pos)[::-1])
    while world.path < MAX_PATH:
        state, heading = gradient_step(world, heading)
        if state == 'collect':
            if world.collect():
                return True
            heading += math.pi / 2
            world.move_towards(world.pos + np.array([math.cos(heading), math.sin(heading)]) * 0.3, 0.3)
        elif state == 'lost':   # сигнала нет: идём к случайной точке арены, пока он не появится
            if np.hypot(*(goal - world.pos)) < 0.3:
                goal = ARENA_PTS[rng.integers(len(ARENA_PTS))]
            world.move_towards(goal, 0.3)
            heading = math.atan2(*(goal - world.pos)[::-1])
    return False


def spiral_points(pitch=0.5):
    pts, theta = [], 0.0
    while True:
        r = pitch * theta / (2 * math.pi)
        if r > 3.4:
            return pts
        p = (r * math.cos(theta), r * math.sin(theta))
        if inside_arena(p) and not BLOCKED[cell(*p)]:
            pts.append(p)
        theta += STEP / max(r, 0.15)


SPIRAL = spiral_points()


def run_spiral(world):
    """Спираль от центра арены с шагом витка 0.5 м; при сильном сигнале — добор градиентом."""
    for p in SPIRAL:
        while np.hypot(*(np.array(p) - world.pos)) > 1e-6 and world.path < MAX_PATH:
            world.move_towards(p, STEP)
            if world.read() >= 0.75:
                heading = 0.0
                while world.path < MAX_PATH:
                    state, heading = gradient_step(world, heading)
                    if state == 'collect':
                        if world.collect():
                            return True
                        heading += math.pi / 2
                        world.move_towards(world.pos + np.array([math.cos(heading), math.sin(heading)]) * 0.3, 0.3)
                    elif state == 'lost':
                        break
        if world.path >= MAX_PATH:
            return False
    return False


def search_experiment(results, n_runs=300):
    rng = np.random.default_rng(7)
    candidates = np.array([center(r, c) for r, c in np.argwhere(~BLOCKED) if DIST_TO_OBST[r, c] >= 0.35])
    candidates = candidates[np.hypot(*(candidates - np.array(START)).T) >= 1.0]
    samples = candidates[rng.choice(len(candidates), n_runs, replace=False)]
    out = {}
    for name, runner in (('bayes', run_bayes), ('gradient', run_gradient), ('spiral', run_spiral)):
        paths, fails, false_collects = [], 0, []
        for i, sample in enumerate(samples):
            world = World(sample, np.random.default_rng(1000 + i))
            ok = runner(world)
            fails += not ok
            paths.append(world.path if ok else MAX_PATH)
            false_collects.append(world.false_collects)
        paths = np.array(paths)
        out[name] = {
            'median_m': round(float(np.median(paths)), 2), 'mean_m': round(float(paths.mean()), 2),
            'p90_m': round(float(np.percentile(paths, 90)), 2),
            'fail_pct': round(fails / n_runs * 100, 1),
            'false_collects_per_run': round(float(np.mean(false_collects)), 2),
        }
    out['runs'] = n_runs
    out['model'] = {'range_m': SENSOR_R, 'sigma': SENSOR_SIGMA, 'step_m': STEP, 'max_path_m': MAX_PATH}
    results['search'] = out


def fig_belief(results):
    sample = (0.55, 1.75)
    world = World(sample, np.random.default_rng(11))
    snaps = []
    assert run_bayes(world, snaps)
    total = world.path
    picks = [snaps[0], min(snaps, key=lambda s: abs(s[0] - 0.45 * total)), snaps[-1]]
    captions = ('рядом пусто', 'осталась дуга', 'образец найден')
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.25))
    for ax, (path, pos, post, trace), caption in zip(axes, picks, captions):
        draw_map(ax, ticks=False)
        field = np.full((H, W), np.nan)
        for (x, y), p in zip(BELIEF_CELLS, post):
            r, c = cell(x, y)
            field[r:r + 2, c:c + 2] = p
        ax.imshow(np.sqrt(field / np.nanmax(field)), extent=EXTENT, origin='upper', cmap=BLUE_SEQ,
                  vmin=0, vmax=1, interpolation='nearest', zorder=1)
        draw_walls(ax, zorder=2)
        ax.plot(trace[:, 0], trace[:, 1], color=WALL, linewidth=1.8, zorder=3)
        ax.scatter(*sample, s=120, marker='x', color=S2, linewidths=2.6, zorder=5)
        draw_robot(ax, pos[0], pos[1], color=WALL)
        ax.set_title(f'путь {path:.1f} м: '.replace('.', ',') + caption, color=INK, fontsize=19,
                     loc='left', pad=4)
    fig.legend(handles=[
        Line2D([], [], marker='s', linestyle='', color='#256abf', markersize=10, label='вероятность положения образца'),
        Line2D([], [], marker='x', linestyle='', color=S2, markersize=9, markeredgewidth=2.6,
               label='истинное положение (агенту неизвестно)'),
        Line2D([], [], color=WALL, linewidth=1.8, marker='o', markersize=8, label='робот и его путь'),
    ], loc='lower center', ncol=3, fontsize=17, labelcolor=INK, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6)
    fig.subplots_adjust(wspace=0.02, bottom=0.13, top=0.9)
    results['belief_demo'] = {'sample': list(sample), 'path_m': round(total, 2),
                              'snap_paths_m': [round(p[0], 2) for p in picks]}
    save(fig, 'belief.png')


# ---------- 6. обнаружение смены грунта (CUSUM) ----------

def cusum_experiment(results):
    """Робот едет по зоне с выученным множителем ×2; на 12-м метре грунт становится ×3."""
    step, sigma, k_ref, h = 0.25, 0.15, 0.5, 5.0   # шаг пути, шум замера расхода, параметры CUSUM
    change_at, total = 12.0, 24.0
    n = int(total / step)

    def run(rng, change=True):
        g, trace, alarm, believed = 0.0, [], None, 2.0
        for i in range(n):
            dist = (i + 1) * step
            true_mult = 3.0 if (change and dist > change_at) else 2.0
            observed = true_mult * step + rng.normal(0.0, sigma)
            g = max(0.0, g + (observed - believed * step) / sigma - k_ref)
            trace.append(g)
            if alarm is None and g > h:
                alarm = dist
                believed, g = true_mult, 0.0   # тревога: агент переоценивает грунт, невязка уходит
        return trace, alarm

    trace, alarm = run(np.random.default_rng(3))
    delays, false_alarms = [], 0
    for seed in range(2000):
        _, a = run(np.random.default_rng(100 + seed))
        if a is not None and a <= change_at:
            false_alarms += 1
        elif a is not None:
            delays.append(a - change_at)
    results['cusum'] = {
        'step_m': step, 'sigma': sigma, 'k': k_ref, 'h': h, 'change_at_m': change_at,
        'from_mult': 2, 'to_mult': 3,
        'distance': [round((i + 1) * step, 2) for i in range(n)],
        'stat': [round(v, 2) for v in trace],
        'demo_alarm_m': alarm,
        'median_delay_m': round(float(np.median(delays)), 2),
        'p90_delay_m': round(float(np.percentile(delays, 90)), 2),
        'false_alarm_pct': round(false_alarms / 2000 * 100, 1),
        'trials': 2000,
    }


def main():
    results = {}
    fig_arena()
    fig_title_art()
    fig_lidar(results)
    fig_astar(results)
    fig_scenario(results)
    search_experiment(results)
    fig_belief(results)
    cusum_experiment(results)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    brief = {k: v for k, v in results.items()}
    brief['cusum'] = {k: v for k, v in results['cusum'].items() if k not in ('distance', 'stat')}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
