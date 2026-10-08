#!/usr/bin/env python3
"""Рисунки и числа для презентации чекпоинта 2.

Всё берётся из настоящих записей прогонов и кода проекта:
  runs/gz_loc/adaptive/*.json.gz, runs/gz_loc/pose-*.csv  — прогоны в Gazebo с поправкой по лидару
  did/mapping.py, did/nav.py, did/arena.py                — карта по лидару, запрет у стен, арена
  did/runner.py                                           — те же сценарии в быстром симуляторе (в памяти)

Картинки пишутся в presentation/assets/c2, числа — в presentation/data/checkpoint2.json.
Папки вне presentation/ не меняются.

Запуск:  python3 presentation/figures/checkpoint2_figures.py            # рисунки и числа
         python3 presentation/figures/checkpoint2_figures.py --tests    # ещё и прогнать автоматические проверки
"""
import argparse
import contextlib
import gzip
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'presentation' / 'assets' / 'c2'
DATA = ROOT / 'presentation' / 'data' / 'checkpoint2.json'
RUNS = ROOT / 'runs'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

INK, WALL, FLOOR, MUTED, UNSEEN = '#14181C', '#2B3138', '#EEF0F2', '#5B6168', '#C9CDD2'
ORANGE, BLUE, GREEN, YELLOW, RED = '#EB6834', '#2A78D6', '#1BAF7A', '#EDA100', '#D03B3B'
SOIL = {2.0: '#F6DCCB', 2.5: '#F0C7AE', 3.0: '#E9B291', 4.0: '#DD9469'}
BAND = '#DADDE1'
BLUE_SEQ = LinearSegmentedColormap.from_list(
    'blue_seq', ['#FFFFFF', '#CDE2FB', '#9EC5F4', '#6DA7EC', '#3987E5', '#256ABF', '#184F95'])
VIEW = (-3.0, 2.85, -2.8, 2.8)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 13, 'savefig.facecolor': 'none',
                     'figure.facecolor': 'none'})


def ru(v, d=1):
    return f'{v:.{d}f}'.replace('.', ',')


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return json.load(f)


# ---------- основа рисунка ----------

def axes(size=(5.0, 4.79)):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def extent(g):
    return (g.x0, g.x0 + g.w * g.res, g.y0, g.y0 + g.h * g.res)


def wall_mask(arena):
    return arena.solid & ndimage.binary_dilation(arena.free, iterations=2)


def draw_base(ax, arena, floor=FLOOR, wall=WALL, zorder=1):
    img = np.zeros((arena.h, arena.w, 4))
    img[arena.free] = to_rgba(floor)
    ax.imshow(img, origin='lower', extent=extent(arena), interpolation='nearest', zorder=zorder)
    draw_walls(ax, arena, wall)


def draw_walls(ax, arena, wall=WALL, zorder=5):
    img = np.zeros((arena.h, arena.w, 4))
    img[wall_mask(arena)] = to_rgba(wall)
    ax.imshow(img, origin='lower', extent=extent(arena), interpolation='nearest', zorder=zorder)


def draw_robot(ax, x, y, th, color=INK, edge='white', zorder=9):
    ax.add_patch(Circle((x, y), 0.105, facecolor=color, edgecolor=edge, linewidth=1.6, zorder=zorder))
    ax.plot([x, x + 0.2 * math.cos(th)], [y, y + 0.2 * math.sin(th)], color=color, linewidth=2.6,
            solid_capstyle='round', zorder=zorder)


def soil_mask(arena, s):
    X, Y = arena.cell_centers()
    if s['shape'] == 'circle':
        m = np.hypot(X - s['x'], Y - s['y']) <= s['r']
    else:
        m = (np.abs(X - s['x']) <= s['w'] / 2) & (np.abs(Y - s['y']) <= s['h'] / 2)
    return m & arena.free


def draw_soils(ax, arena, soils, label=True, zorder=2):
    """Зоны дорогого грунта: только та часть, что лежит на свободном полу; подпись — в её середине."""
    X, Y = arena.cell_centers()
    for s in soils:
        m = soil_mask(arena, s)
        if not m.any():
            continue
        img = np.zeros((arena.h, arena.w, 4))
        img[m] = to_rgba(SOIL.get(float(s['mult']), SOIL[3.0]))
        ax.imshow(img, origin='lower', extent=extent(arena), interpolation='nearest', zorder=zorder)
        if label:
            deep = ndimage.distance_transform_edt(m)
            iy, ix = np.unravel_index(int(deep.argmax()), deep.shape)
            ax.text(X[iy, ix], Y[iy, ix], '×' + ru(s['mult'], 0 if float(s['mult']).is_integer() else 1),
                    ha='center', va='center', fontsize=15, fontweight='bold', color='#7A3F1D', zorder=zorder + 0.5)


def draw_hazard(ax, z, zorder=3):
    ax.add_patch(Circle((z['x'], z['y']), z['r'], facecolor='#F7D4D4', edgecolor=RED, linewidth=2.2, zorder=zorder))
    ax.text(z['x'], z['y'], '!', ha='center', va='center', fontsize=17, fontweight='bold', color=RED, zorder=zorder + 0.5)


def draw_sample(ax, x, y, collected=False, hidden=False, zorder=7):
    if hidden:      # где образец лежит на самом деле: робот этого не знает
        ax.add_patch(Circle((x, y), 0.09, facecolor='none', edgecolor=YELLOW, linewidth=2.2, zorder=zorder))
    elif collected:
        ax.add_patch(Circle((x, y), 0.10, facecolor=GREEN, edgecolor='white', linewidth=1.6, zorder=zorder))
    else:
        ax.add_patch(Circle((x, y), 0.10, facecolor=YELLOW, edgecolor=INK, linewidth=1.4, zorder=zorder))


def save(fig, name, dpi=200):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=dpi, transparent=True)
    plt.close(fig)
    return name


# ---------- слайд «задача»: как на самом деле и что видит робот ----------

def fig_task(arena, trace):
    scn = trace['scenario']
    fig, ax = axes()
    draw_base(ax, arena)
    draw_soils(ax, arena, scn['soils'])
    for z in scn['hazards']:
        draw_hazard(ax, z)
    for x, y in scn['samples']:
        draw_sample(ax, x, y)
    draw_robot(ax, *scn['base'], 0.0)
    truth = save(fig, 'task_truth.png')

    scan = trace['scans'][0]
    tr = trace['track']
    fig, ax = axes()
    # Робот знает стены: карта арены дана, лидар её подтверждает. Больше ничего на карте нет.
    draw_base(ax, arena, floor='#F6F7F8', wall='#B9BEC5')
    r = np.array(scan['r'], dtype=float) / 100.0
    ang = scan['th'] + np.arange(len(r)) * 2 * math.pi / len(r)
    ok = r > 0
    lx, ly = scan['x'] - 0.032 * math.cos(scan['th']), scan['y'] - 0.032 * math.sin(scan['th'])
    ax.scatter(lx + r[ok] * np.cos(ang[ok]), ly + r[ok] * np.sin(ang[ok]), s=16, color=ORANGE, zorder=6, linewidths=0)
    z = float(np.mean(tr['sensor'][:5]))
    dist = (1.0 - z) * trace['rules']['sensor_range_m']
    ring = Circle((scan['x'], scan['y']), dist, facecolor='none', edgecolor=BLUE, linewidth=2.2, linestyle=(0, (5, 4)), zorder=6.5)
    ax.add_patch(ring)
    draw_robot(ax, scan['x'], scan['y'], scan['th'])
    view = save(fig, 'task_robot_view.png')
    return {'truth': truth, 'robot_view': view, 'sensor': round(z, 2), 'ring_m': round(dist, 2),
            'samples': len(scn['samples']), 'soils': len(scn['soils']),
            'hazards': len(scn['hazards']) + sum(1 for e in scn['events'] if e['type'] == 'new_hazard'),
            'level': scn['level'], 'seed': scn['seed']}


# ---------- слайд «сценарий»: кадры одного прогона в Gazebo ----------

def belief_at(trace, t):
    b = trace['belief']
    snaps = [s for s in b['snaps'] if s['t'] <= t + 1e-6] or b['snaps'][:1]
    from did.recorder import decode_grid
    v = decode_grid(snaps[-1]['data'], b['h'], b['w']).astype(float)
    return (v / 255.0) ** 2, (b['x0'], b['x0'] + b['w'] * b['res'], b['y0'], b['y0'] + b['h'] * b['res'])


def frame(arena, trace, t, name):
    scn, tr = trace['scenario'], trace['track']
    tt = np.array(tr['t'])
    k = int(np.searchsorted(tt, t, side='right')) - 1
    k = max(0, min(k, len(tt) - 1))
    fig, ax = axes()
    draw_base(ax, arena)
    p, ext = belief_at(trace, t)
    shade = np.clip(p / max(0.02, np.percentile(p[p > 0], 99.5) if (p > 0).any() else 1.0), 0, 1) ** 0.6
    rgba = BLUE_SEQ(shade)
    rgba[..., 3] = np.where(shade > 0.03, 0.9, 0.0)
    ax.imshow(rgba, origin='lower', extent=ext, interpolation='nearest', zorder=2)
    # вероятность вне свободной площади не показываем
    mask = np.zeros((arena.h, arena.w, 4))
    mask[~arena.free] = (1, 1, 1, 1)
    ax.imshow(mask, origin='lower', extent=extent(arena), interpolation='nearest', zorder=2.5)
    draw_walls(ax, arena)
    got = [e for e in trace['events'] if e['type'] == 'sample_collected' and e['t'] <= t]
    got_ids = {e.get('sample') for e in got}
    for i, (x, y) in enumerate(scn['samples']):       # номер образца в событиях судьи — с нуля
        draw_sample(ax, x, y, collected=i in got_ids, hidden=i not in got_ids)
    for hz in trace.get('hazards') or []:
        if hz['t'] <= t:
            ax.add_patch(Circle((hz['x'], hz['y']), hz['r'], facecolor='none', edgecolor=RED, linewidth=2.0,
                                linestyle=(0, (4, 3)), zorder=6))
    for e in trace['events']:
        if e['type'] == 'hazard_hit' and e['t'] <= t:
            ax.plot([e['x']], [e['y']], marker='x', color=RED, markersize=11, markeredgewidth=3, zorder=8)
    ax.plot(tr['x'][:k + 1], tr['y'][:k + 1], color=INK, linewidth=2.0, solid_capstyle='round',
            solid_joinstyle='round', zorder=6.5)
    draw_robot(ax, tr['x'][k], tr['y'][k], tr['th'][k], color=ORANGE)
    save(fig, name)
    return {'file': name, 't': round(float(tt[k]), 1), 'since_start': round(float(tt[k] - tt[0]), 0),
            'battery': round(float(tr['battery'][k]), 1), 'collected': len(got),
            'total': len(scn['samples'])}


def journal_line(trace, kind, pattern, after=0.0):
    for j in trace['journal']:
        if j['kind'] == kind and j['t'] >= after and re.search(pattern, j['text']):
            return {'t': j['t'], 'text': j['text']}
    return None


def fig_story(arena, trace):
    """Четыре кадра трудного прогона в Gazebo и строки журнала агента к ним."""
    tr = trace['track']
    t0, t1 = tr['t'][0], tr['t'][-1]
    first = next(e for e in trace['events'] if e['type'] == 'sample_collected')
    hit = next((e for e in trace['events'] if e['type'] == 'hazard_hit'), None)
    alarm = journal_line(trace, 'alarm', 'Датчик образцов шумит')
    hyp1 = journal_line(trace, 'hypothesis', 'образец лежит около')
    hyp_h = journal_line(trace, 'hypothesis', 'опасная зона')
    hyp_s = journal_line(trace, 'hypothesis', 'датчик образцов неисправен')
    back = journal_line(trace, 'decision', 'Возвращаюсь на базу')
    fin = journal_line(trace, 'action', 'Финиш')
    moments = [
        ('поиск', (hyp1 or {'t': first['t'] - 2})['t'] + 1.0, hyp1),
        ('сбор и штраф', (hit or first)['t'] + 0.6, hyp_h or journal_line(trace, 'action', 'Сбор')),
        ('сбой датчика', (alarm or {'t': (t0 + t1) / 2})['t'] + 6.0, hyp_s or alarm),
        ('возврат', t1, fin or back),
    ]
    frames = []
    for i, (label, t, quote) in enumerate(moments, 1):
        f = frame(arena, trace, t, f'story_{i}.png')
        f.update(label=label, quote=quote)
        frames.append(f)
    res = trace['result']
    return {'run': trace['id'], 'backend': trace['backend'], 'level': trace['scenario']['level'],
            'frames': frames, 'settle_s': round(t0, 0),
            'result': {k: res.get(k) for k in ('samples_collected', 'samples_total', 'returned', 'score',
                                               'battery_left', 'time', 'penalties', 'hazard_hits')},
            'hypotheses': res.get('hypotheses'), 'detect': res.get('detect'),
            'back': back, 'journal_records': len(trace['journal'])}


# ---------- слайд «карта и движение» ----------

def fig_mapping(arena, trace):
    from did.mapping import OccupancyMapper
    m = OccupancyMapper(arena)
    for s in trace['scans']:
        r = np.array(s['r'], dtype=float) / 100.0
        r[r <= 0] = np.inf
        m.update(s['x'], s['y'], s['th'], r)
    fig, ax = axes()
    img = np.zeros((arena.h, arena.w, 4))
    inside = ndimage.binary_dilation(arena.free, iterations=2)
    img[inside] = to_rgba(UNSEEN)
    seen, occ = m.seen(), m.occupied()
    img[seen & ~occ] = to_rgba(FLOOR)
    img[occ] = to_rgba(WALL)
    ax.imshow(img, origin='lower', extent=extent(arena), interpolation='nearest', zorder=1)
    tr = trace['track']
    ax.plot(tr['x'], tr['y'], color=ORANGE, linewidth=1.8, solid_capstyle='round', solid_joinstyle='round', zorder=4)
    draw_robot(ax, tr['x'][0], tr['y'][0], tr['th'][0])
    name = save(fig, 'map_lidar.png')
    return {'file': name, 'scans': int(m.scans), 'rays': len(trace['scans'][0]['r']),
            'coverage': round(m.coverage(), 4), 'agreement': round(m.agreement(), 4), 'run': trace['id']}


def path_len(pts):
    return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))


def fig_path(arena, trace):
    from did.nav import INFLATE, Follower
    scn, tr = trace['scenario'], trace['track']
    fig, ax = axes()
    draw_base(ax, arena)
    band = np.zeros((arena.h, arena.w, 4))
    band[arena.free & (arena.clear < INFLATE)] = to_rgba(BAND)
    ax.imshow(band, origin='lower', extent=extent(arena), interpolation='nearest', zorder=1.5)
    draw_soils(ax, arena, scn['soils'], zorder=1.2)
    ax.imshow(band, origin='lower', extent=extent(arena), interpolation='nearest', zorder=1.5)
    draw_walls(ax, arena)
    ax.plot(tr['x'], tr['y'], color=INK, linewidth=1.5, alpha=0.55, solid_capstyle='round', zorder=6)
    best = max(trace['paths'], key=lambda p: path_len(p['pts']))
    xs, ys = zip(*best['pts'])
    ax.plot(xs, ys, color=ORANGE, linewidth=3.4, solid_capstyle='round', solid_joinstyle='round', zorder=7)
    ax.plot(xs, ys, linestyle='none', marker='o', markersize=4.5, color='white', markeredgecolor=ORANGE,
            markeredgewidth=1.4, zorder=7.5)
    ax.plot([xs[-1]], [ys[-1]], marker='*', markersize=20, color=ORANGE, markeredgecolor='white',
            markeredgewidth=1.2, zorder=8)
    tt = np.array(tr['t'])
    k = int(np.searchsorted(tt, best['t'], side='right')) - 1
    draw_robot(ax, tr['x'][k], tr['y'][k], tr['th'][k])
    for e in trace['events']:
        if e['type'] == 'sample_collected':
            draw_sample(ax, e['x'], e['y'], collected=True)
    name = save(fig, 'path_grid.png')
    free = int(arena.free.sum())
    return {'file': name, 'inflate_m': INFLATE, 'lookahead_m': Follower.LOOKAHEAD, 'grid_m': arena.res,
            'robot_radius_m': 0.105, 'belief_cell_m': trace['belief']['res'], 'soil_cell_m': trace['soil']['res'],
            'path_m': round(path_len(best['pts']), 2), 'path_points': len(best['pts']),
            'path_cost': best.get('cost'), 'paths_in_run': len(trace['paths']),
            'free_cells': free, 'allowed_cells': int((arena.free & (arena.clear >= INFLATE)).sum()),
            'run': trace['id']}


# ---------- титул ----------

def fig_title(arena, trace):
    fig, ax = axes()
    img = np.zeros((arena.h, arena.w, 4))
    img[arena.free] = to_rgba('#20262D')
    ax.imshow(img, origin='lower', extent=extent(arena), interpolation='nearest', zorder=1)
    draw_walls(ax, arena, wall='#4A535E')
    tr = trace['track']
    ax.plot(tr['x'], tr['y'], color=ORANGE, linewidth=2.4, solid_capstyle='round', solid_joinstyle='round', zorder=6)
    for e in trace['events']:
        if e['type'] == 'sample_collected':
            ax.add_patch(Circle((e['x'], e['y']), 0.11, facecolor=YELLOW, edgecolor='#14181C', linewidth=1.6, zorder=7))
    draw_robot(ax, tr['x'][0], tr['y'][0], 0.0, color='white', edge='#14181C')
    return save(fig, 'title_run.png')


# ---------- числа: Gazebo против быстрого симулятора, ошибка положения ----------

def gazebo_numbers():
    import gz_pose_log
    from did.runner import run_episode
    rows, series = [], {}
    for level in ('easy', 'medium', 'hard'):
        tfile = RUNS / 'gz_loc' / 'adaptive' / f'{level}-1.json.gz'
        pfile = RUNS / 'gz_loc' / f'pose-{level}-1.csv'
        if not tfile.exists():
            continue
        trace = load(tfile)
        res = trace['result']
        row = {'level': level, 'seed': trace['scenario']['seed'], 'file': str(tfile.relative_to(ROOT)),
               'gazebo': {k: res.get(k) for k in ('samples_collected', 'samples_total', 'returned', 'score',
                                                  'battery_used', 'time', 'collisions', 'false_collects',
                                                  'hazard_hits', 'distance')},
               'pose_fixes': len(trace.get('pose_fix') or [])}
        with contextlib.redirect_stdout(io.StringIO()):
            fast = run_episode(level, trace['scenario']['seed'], 'adaptive', scenario=trace['scenario'],
                               rules=trace['rules'], save=False)
        fm = fast['metrics']
        row['fastsim'] = {k: fm.get(k) for k in ('samples_collected', 'samples_total', 'returned', 'score',
                                                 'battery_used', 'time', 'collisions', 'false_collects',
                                                 'hazard_hits', 'distance')}
        if pfile.exists():
            st = gz_pose_log.report(str(pfile), str(tfile), out=io.StringIO())
            cm = lambda s: {k: round(v * 100, 1) for k, v in s.items()}   # noqa: E731
            row['agent_err_cm'], row['odom_err_cm'] = cm(st['agent']), cm(st['odom'])
            d = gz_pose_log.load(str(pfile))
            ax_, ay_, _ = gz_pose_log.agent_pose(d, trace)
            t = d['tj']
            run = (t >= trace['track']['t'][0]) & (t <= trace['track']['t'][-1])
            odom = np.hypot(d['ox'] - d['gx'], d['oy'] - d['gy'])[run] * 100
            agent = np.hypot(ax_ - d['gx'], ay_ - d['gy'])[run] * 100
            tt = t[run] - t[run][0]
            step = 4.0
            edges = np.arange(0.0, tt[-1] + step, step)
            ts, o, a = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mm = (tt >= lo) & (tt < hi)
                if mm.any():
                    ts.append(int(round(lo)))
                    o.append(round(float(odom[mm].max()), 1))
                    a.append(round(float(agent[mm].max()), 1))
            series[level] = {'t': ts, 'odom_cm': o, 'agent_cm': a, 'step_s': step}
        rows.append(row)
    out = {'runs': rows, 'series': series}
    if rows and all('agent_err_cm' in r for r in rows):
        out['agent_median_cm'] = [min(r['agent_err_cm']['median'] for r in rows), max(r['agent_err_cm']['median'] for r in rows)]
        out['agent_max_cm'] = max(r['agent_err_cm']['max'] for r in rows)
        out['odom_max_cm'] = max(r['odom_err_cm']['max'] for r in rows)
    base = RUNS / 'gz_base' / 'pose-medium-1.csv'
    if base.exists():       # прогон до исправлений: робот поехал, пока ещё покачивался
        d = gz_pose_log.load(str(base))
        err = np.hypot(d['ox'] - d['gx'], d['oy'] - d['gy']) * 100
        out['before_fix'] = {'file': str(base.relative_to(ROOT)), 'odom_median_cm': round(float(np.median(err)), 1),
                             'odom_max_cm': round(float(err.max()), 1)}
    m = re.search(r'^SETTLE_UNTIL_S\s*=\s*([\d.]+)', (ROOT / 'did' / 'ros_agent.py').read_text(), re.M)
    if m:                   # до этой секунды агент в Gazebo стоит: робот оседает на колёса
        out['settle_until_s'] = float(m.group(1))
    return out


def run_tests():
    pixi = shutil.which('pixi') or str(Path.home() / '.pixi' / 'bin' / 'pixi')
    cmd = [pixi, 'run', 'pytest', 'tests', '-q'] if Path(pixi).exists() else [sys.executable, '-m', 'pytest', 'tests', '-q']
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    tail = (p.stdout or '').strip().splitlines()[-1] if p.stdout else ''
    m = re.search(r'(\d+) passed', tail)
    out = {'passed': int(m.group(1)) if m else None, 'line': tail, 'ok': p.returncode == 0,
           'when': time.strftime('%Y-%m-%dT%H:%M:%S')}
    for key in ('failed', 'skipped', 'error'):
        mm = re.search(rf'(\d+) {key}', tail)
        out[key] = int(mm.group(1)) if mm else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tests', action='store_true', help='прогнать автоматические проверки и записать их число')
    args = ap.parse_args()

    from did.arena import Arena
    arena = Arena()
    old = json.loads(DATA.read_text()) if DATA.exists() else {}
    hard = load(RUNS / 'gz_loc' / 'adaptive' / 'hard-1.json.gz')
    medium = load(RUNS / 'gz_loc' / 'adaptive' / 'medium-1.json.gz')

    data = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S')}
    data['task'] = fig_task(arena, hard)
    data['story'] = fig_story(arena, hard)
    data['mapping'] = fig_mapping(arena, medium)
    data['nav'] = fig_path(arena, medium)
    data['title'] = fig_title(arena, medium)
    data['gazebo'] = gazebo_numbers()
    data['tests'] = run_tests() if args.tests else old.get('tests')
    data['rules'] = {k: hard['rules'].get(k) for k in ('battery_start', 'drain_per_m', 'collect_radius_m',
                                                       'sensor_range_m', 'sensor_hz', 'sensor_sigma')}
    data['levels'] = {}
    for level in ('easy', 'medium', 'hard'):
        f = RUNS / 'gz_loc' / 'adaptive' / f'{level}-1.json.gz'
        if f.exists():
            scn = load(f)['scenario']
            data['levels'][level] = {'samples': len(scn['samples']), 'soils': len(scn['soils']),
                                     'events': [e['type'] for e in scn['events']]}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print('рисунки:', OUT)
    print('числа:  ', DATA)
    g = data['gazebo']
    for r in g['runs']:
        print(f"  {r['level']}: Gazebo {r['gazebo']['samples_collected']}/{r['gazebo']['samples_total']}, "
              f"быстрый {r['fastsim']['samples_collected']}/{r['fastsim']['samples_total']}, "
              f"ошибка положения {r.get('agent_err_cm')}, одометрия {r.get('odom_err_cm')}")
    print('  карта по лидару:', data['mapping'])
    print('  проверки:', data['tests'])


if __name__ == '__main__':
    main()
