"""Рисунки для колоды команды (presentation/team/build_team.js): карты прогонов из записей runs/.

    /usr/local/bin/python3 presentation/team/figures.py

Всё рисуется из сохранённых записей прогонов (ничего не пересчитывается): арена — presentation/team/data/arena.json
(выгружена из did.arena), записи — runs/<опыт>/<вариант>/<сценарий>.json.gz. Подписи на слайдах — текст PowerPoint,
на рисунках только то, что привязано к карте (множители грунта, номера событий).
Цвета: наш робот — синий, сравнение — серый, языковая модель — фиолетовый; оранжевый и красный — только события
среды (сбой, штраф). Набор «синий, зелёный, фиолетовый» проверен скриптом validate_palette.js.
"""
import base64
import gzip
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / 'runs'
OUT = HERE / 'fig'
OUT.mkdir(exist_ok=True)

INK, MUTED, BG, FLOOR = '#182B45', '#50627B', '#F8F9FC', '#FFFFFF'
BLUE, GREEN, VIOLET, GREY = '#3D63DD', '#2E9E73', '#7C5CD6', '#8A94A6'
AMBER, AMBER_FILL, RED, RED_FILL, ORANGE = '#C98A1B', '#FBE7BF', '#D64545', '#F8D2D2', '#E8590C'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'svg.fonttype': 'none'})

A = json.loads((HERE / 'data' / 'arena.json').read_text())
FREE = np.array([[c == '1' for c in row] for row in A['free']])
EXT = (A['x0'], A['x0'] + A['w'] * A['res'], A['y0'], A['y0'] + A['h'] * A['res'])
YS, XS = np.nonzero(FREE)
PAD = 0.12
LIM = (A['x0'] + XS.min() * A['res'] - PAD, A['x0'] + (XS.max() + 1) * A['res'] + PAD,
       A['y0'] + YS.min() * A['res'] - PAD, A['y0'] + (YS.max() + 1) * A['res'] + PAD)
BASE = A['base']


def load(*parts):
    with gzip.open(RUNS.joinpath(*parts)) as f:
        return json.load(f)


def num(x, d=1):
    return f'{x:.{d}f}'.replace('.', ',')


def arena(ax, floor=FLOOR, wall=INK, lw=1.3, outside=BG):
    ax.imshow(np.where(FREE, 1.0, np.nan), origin='lower', extent=EXT, cmap=LinearSegmentedColormap.from_list('f', [floor, floor]),
              interpolation='nearest', zorder=0)
    ax.imshow(np.where(FREE, np.nan, 1.0), origin='lower', extent=EXT, cmap=LinearSegmentedColormap.from_list('o', [outside, outside]),
              interpolation='nearest', zorder=2.5)
    ax.contour(np.linspace(EXT[0] + A['res'] / 2, EXT[1] - A['res'] / 2, A['w']),
               np.linspace(EXT[2] + A['res'] / 2, EXT[3] - A['res'] / 2, A['h']),
               FREE.astype(float), levels=[0.5], colors=[wall], linewidths=lw, zorder=3)
    ax.set_xlim(LIM[0], LIM[1])
    ax.set_ylim(LIM[2], LIM[3])
    ax.set_aspect('equal')
    ax.axis('off')


def zone(ax, z, kind, label=None, z0=1):
    face, edge = (AMBER_FILL, AMBER) if kind == 'soil' else (RED_FILL, RED)
    if (z.get('shape') or 'circle') == 'circle':
        ax.add_patch(Circle((z['x'], z['y']), z['r'], facecolor=face, edgecolor=edge, lw=0.9, zorder=z0))
    else:
        ax.add_patch(Rectangle((z['x'] - z['w'] / 2, z['y'] - z['h'] / 2), z['w'], z['h'], facecolor=face, edgecolor=edge,
                               lw=0.9, zorder=z0))
    if label:
        ax.text(z['x'], z['y'], label, ha='center', va='center', fontsize=8, color=edge, fontweight='bold', zorder=z0 + 1, clip_on=True)


def soils_at(sc, t=None):
    """Грунты сценария: начальные или после смены (t позже события soil_change)."""
    soils = sc.get('soils') or []
    for e in sc.get('events') or []:
        if e.get('type') == 'soil_change' and (t is None or t >= e['t']):
            soils = e['soils']
    return soils


def hazards_at(sc, t=None):
    out = list(sc.get('hazards') or [])
    for e in sc.get('events') or []:
        if e.get('type') == 'new_hazard' and (t is None or t >= e['t']):
            out.append(e['zone'])
    return out


def truth(ax, sc, t=None, labels=True):
    for z in soils_at(sc, t):
        zone(ax, z, 'soil', '×' + num(z['mult']).replace(',0', '') if labels else None)
    for z in hazards_at(sc, t):
        zone(ax, z, 'haz', '!' if labels else None)


def collected(rec):
    got, samples = set(), rec['scenario']['samples']
    for e in rec.get('events') or []:
        if e.get('type') != 'sample_collected':
            continue
        if e.get('sample') is not None:
            got.add(e['sample'])
        elif e.get('x') is not None:
            got.add(min(range(len(samples)), key=lambda i: math.dist(samples[i], (e['x'], e['y']))))
    return got


def samples(ax, rec, got=None, size=5.2):
    got = collected(rec) if got is None else got
    for i, (x, y) in enumerate(rec['scenario']['samples']):
        if i in got:
            ax.plot(x, y, 'o', ms=size + 1.2, mfc=GREEN, mec='white', mew=1.1, zorder=7)
        else:
            ax.plot(x, y, 'o', ms=size, mfc='white', mec=INK, mew=1.3, zorder=7)


def base(ax):
    ax.add_patch(FancyBboxPatch((BASE[0] - 0.11, BASE[1] - 0.11), 0.22, 0.22, boxstyle='round,pad=0,rounding_size=0.05',
                                facecolor=INK, edgecolor='white', lw=1.0, zorder=8))


def track(ax, rec, color, lo=None, hi=None, lw=2.0, z0=5, alpha=1.0):
    tr = rec['track']
    idx = [i for i, t in enumerate(tr['t']) if (lo is None or t >= lo) and (hi is None or t <= hi)]
    if idx:
        ax.plot([tr['x'][i] for i in idx], [tr['y'][i] for i in idx], '-', color=color, lw=lw, solid_capstyle='round',
                solid_joinstyle='round', zorder=z0, alpha=alpha)
    return idx


def at(rec, t):
    tr = rec['track']
    i = min(range(len(tr['t'])), key=lambda k: abs(tr['t'][k] - t))
    return tr['x'][i], tr['y'][i], i


def badge(ax, x, y, n, color=INK, dx=0.0, dy=0.0):
    ax.plot(x + dx, y + dy, 'o', ms=13, mfc=color, mec='white', mew=1.4, zorder=10)
    ax.text(x + dx, y + dy - 0.005, str(n), ha='center', va='center', color='white', fontsize=8.5, fontweight='bold', zorder=11)


def cross(ax, x, y):
    ax.plot(x, y, 'o', ms=12, mfc='white', mec=RED, mew=1.6, zorder=9)
    ax.plot(x, y, 'x', ms=6, mec=RED, mew=1.8, zorder=10)


def save(fig, name, face=BG):
    fig.savefig(OUT / name, dpi=240, facecolor=face,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('  ', name)


def panels(n, size=3.3):
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    fig.subplots_adjust(wspace=0.04, left=0, right=1, top=1, bottom=0)
    return fig, ([axes] if n == 1 else list(axes))


# ---------------------------------------------------------------- 1. сценарий показа на карте
def fig_scenario():
    r = load('F1demo', 'scientist_v2-hard-2-f1b.json.gz')
    sc = r['scenario']
    fig, (ax,) = panels(1, 4.6)
    arena(ax)
    truth(ax, sc)
    track(ax, r, BLUE, lw=2.2)
    samples(ax, r)
    base(ax)
    q = r['inquiries'][0]
    wx, wy, _ = at(r, 62.0)                      # стоит, пока шумит датчик
    first = next(e for e in r['events'] if e['type'] == 'sample_collected')
    last = [e for e in r['events'] if e['type'] == 'sample_collected'][-1]
    badge(ax, first['x'], first['y'], 1, dx=-0.24, dy=0.2)
    badge(ax, q['anomaly']['x'], q['anomaly']['y'], 2, color=AMBER, dx=0.27, dy=-0.2)
    badge(ax, wx, wy, 3, color=ORANGE, dx=0.26, dy=0.2)
    badge(ax, last['x'], last['y'], 4, dx=0.05, dy=0.3)
    badge(ax, BASE[0], BASE[1], 5, dx=-0.32, dy=0.0)
    save(fig, 'scenario.png')
    info = {'wait_xy': [round(wx, 2), round(wy, 2)], 'result': r['result'], 'q1': q['anomaly'],
            'collect_t': [round(e['t'], 1) for e in r['events'] if e['type'] == 'sample_collected']}
    # карта для титула: без подписей, прозрачный фон
    fig, (ax,) = panels(1, 4.2)
    arena(ax, floor='#FFFFFF', lw=1.5, outside='#DCE8FF')
    truth(ax, sc, labels=False)
    track(ax, r, BLUE, lw=2.6)
    samples(ax, r, size=6)
    base(ax)
    save(fig, 'title_map.png', face='#DCE8FF')
    # Всё, что снаружи арены, делаем прозрачным: карта ложится на круг титула в любой программе показа.
    from PIL import Image, ImageDraw
    im = Image.open(OUT / 'title_map.png').convert('RGBA')
    for corner in ((0, 0), (im.width - 1, 0), (0, im.height - 1), (im.width - 1, im.height - 1)):
        ImageDraw.floodfill(im, corner, (0, 0, 0, 0), thresh=40)
    im.save(OUT / 'title_map.png')
    return info


# ---------------------------------------------------------------- 2. что скрыто и что видит робот
def fig_truth_vs_robot():
    r = load('F1demo', 'scientist_v2-hard-2-f1b.json.gz')
    sc = r['scenario']
    fig, (a, b) = panels(2, 3.4)
    arena(a)
    truth(a, sc, t=0)
    samples(a, r, got=set())
    base(a)
    arena(b)
    x, y, i = at(r, 5.0)
    z = r['track']['sensor'][i]
    d = (1.0 - z) * r['rules']['sensor_range_m']
    track(b, r, BLUE, hi=5.0)
    b.add_patch(Circle((x, y), d, facecolor='none', edgecolor=BLUE, lw=1.6, ls=(0, (4, 3)), zorder=6))
    b.plot(x, y, 'o', ms=8, mfc=BLUE, mec='white', mew=1.3, zorder=9)
    base(b)
    save(fig, 'truth_vs_robot.png')
    return {'t': 5.0, 'z': round(z, 2), 'd': round(d, 2)}


# ---------------------------------------------------------------- 3. карта вероятностей по шагам
def belief_at(rec, t):
    bel = rec['belief']
    snap = min(bel['snaps'], key=lambda s: abs(s['t'] - t))
    v = np.frombuffer(base64.b64decode(snap['data']), dtype=np.uint8).reshape(bel['h'], bel['w']).astype(float) / 255.0
    ext = (bel['x0'], bel['x0'] + bel['w'] * bel['res'], bel['y0'], bel['y0'] + bel['h'] * bel['res'])
    return (v ** 2 if bel.get('enc') == 'sqrt' else v), ext, snap['t']


def fig_belief():
    r = load('F1demo', 'scientist_v2-hard-2-f1b.json.gz')
    times = (2.2, 6.3, 8.3)
    cmap = LinearSegmentedColormap.from_list('b', ['#E9F0FF', '#B9CEFA', '#7F9FEF', BLUE, '#1F3C9C'])
    fig, axes = panels(3, 3.0)
    out = []
    first = next(e for e in r['events'] if e['type'] == 'sample_collected')
    for ax, t in zip(axes, times):
        arena(ax)
        p, ext, ts = belief_at(r, t)
        v = np.sqrt(np.minimum(p / 0.5, 1.0))               # одна шкала на все кадры; корень: видно и слабое кольцо, и пик
        ax.imshow(np.where(v > 0.13, v, np.nan), origin='lower', extent=ext, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest',
                  zorder=1)
        track(ax, r, INK, hi=ts, lw=1.6)
        x, y, i = at(r, ts)
        ax.plot(x, y, 'o', ms=7.5, mfc=INK, mec='white', mew=1.2, zorder=9)
        ax.plot(first['x'], first['y'], 'o', ms=11, mfc='none', mec=GREEN, mew=2.0, zorder=8)
        base(ax)
        out.append({'t': ts, 'z': round(r['track']['sensor'][i], 2), 'pmax': round(float(np.nanmax(p)), 2)})
    save(fig, 'belief.png')
    return out


# ---------------------------------------------------------------- 4–6. пары «обычный — наш» на одном сценарии
def pair(name, left, right, lc, rc, *, t_truth=None, deco=None):
    fig, axes = panels(2, 3.4)
    for ax, rec, color, side in zip(axes, (left, right), (lc, rc), ('left', 'right')):
        arena(ax)
        truth(ax, rec['scenario'], t_truth)
        track(ax, rec, color, lw=2.1)
        if deco:
            deco(ax, rec, side)
        samples(ax, rec)
        base(ax)
    save(fig, name)
    return [rec['result'] for rec in (left, right)]


def fig_pair_e1():
    return pair('pair_fixed_adaptive.png', load('E1', 'fixed', 'hard-1026.json.gz'), load('E1', 'adaptive', 'hard-1026.json.gz'), GREY, BLUE)


def fig_pair_fault():
    a, v = load('_showcase', 'adaptive', 'hard-8024.json.gz'), load('_showcase', 'adaptive_v2', 'hard-8024.json.gz')
    t0 = next(w['t'] for w in a['world'] if w['type'] == 'sensor_fault')
    t1 = next(w['t'] for w in a['world'] if w['type'] == 'sensor_recovered')
    info = {'fault': [t0, t1]}

    def deco(ax, rec, side):
        idx = track(ax, rec, ORANGE, lo=t0, hi=t1, lw=2.6, z0=6)
        tr = rec['track']
        dist = sum(math.dist((tr['x'][i], tr['y'][i]), (tr['x'][j], tr['y'][j])) for i, j in zip(idx, idx[1:]))
        info[side] = {'dist_in_fault': round(dist, 2), 'battery_in_fault': round(tr['battery'][idx[0]] - tr['battery'][idx[-1]], 2)}
        if dist < 1.0:
            x, y, _ = at(rec, (t0 + t1) / 2)
            ax.plot(x, y, 'o', ms=13, mfc=ORANGE, mec='white', mew=1.5, zorder=9)
    info['results'] = pair('pair_sensor_fault.png', a, v, GREY, BLUE, deco=deco)
    return info


def fig_pair_llm(mission, seed, name):
    rule = load('L3d', f'{mission}_rule', f'{seed}.json.gz')
    llm = load('L3d', f'{mission}_qwen3.8-flash-next', f'{seed}.json.gz')
    info = {}

    def deco(ax, rec, side):
        pens = [e for e in rec['events'] if e['type'] in ('hazard_hit', 'false_collect', 'collision')]
        if mission == 'M4' and pens:
            track(ax, rec, GREY if side == 'left' else VIOLET, lo=pens[0]['t'], lw=2.6, z0=6)
            cross(ax, pens[0]['x'], pens[0]['y'])
        info[side] = {'penalties': [(round(e['t'], 1), e['type']) for e in pens],
                      'collect_t': [round(e['t'], 1) for e in rec['events'] if e['type'] == 'sample_collected']}
    info['results'] = pair(name, rule, llm, GREY, VIOLET, deco=deco)
    info['plans'] = [(p['t'], p['source'], p.get('trigger'), p.get('reasoning')) for p in llm['plans'] if p['source'] == 'llm'][-3:]
    return info


# ---------------------------------------------------------------- 7. два робота (запись из Gazebo)
def fig_team():
    rec = load('M1gz', 'team', 'medium-3.json.gz')
    fig, (ax,) = panels(1, 3.4)
    arena(ax)
    sc = rec['scenario']
    truth(ax, sc)
    robots = rec.get('robots') or []
    for rb, color in zip(robots, (BLUE, VIOLET)):
        tr = rb['track']
        ax.plot(tr['x'], tr['y'], '-', color=color, lw=2.0, solid_capstyle='round', zorder=5)
        ax.plot(tr['x'][-1], tr['y'][-1], 'o', ms=7, mfc=color, mec='white', mew=1.2, zorder=9)
    for x, y in sc['samples']:
        ax.plot(x, y, 'o', ms=6.4, mfc=GREEN, mec='white', mew=1.1, zorder=7)
    save(fig, 'two_robots.png')
    return {'keys': list(rec.keys()), 'robots': len(robots), 'result': rec.get('result')}


# ---------------------------------------------------------------- 8. математика: карта вероятностей по шагам
# Данные считает presentation/team/mathdata.py кодом робота (did.belief.SampleBelief) на трёх настоящих показаниях.
def fig_bayes():
    m = json.loads((HERE / 'data' / 'math.json').read_text())['belief']
    cmap = LinearSegmentedColormap.from_list('b', ['#E9F0FF', '#B9CEFA', '#7F9FEF', BLUE, '#1F3C9C'])
    ext = (m['x0'], m['x0'] + m['w'] * m['res'], m['y0'], m['y0'] + m['h'] * m['res'])
    fig, axes = panels(3, 3.0)
    info = []
    for k, ax in enumerate(axes):
        arena(ax)
        st = m['steps'][k]
        p = np.array(st['grid'])
        v = np.clip(p / 0.35, 0.0, 1.0) ** 0.35           # степень: видно и слабое кольцо первого показания, и пик
        ax.imshow(np.where(p > 0.0068, v, np.nan), origin='lower', extent=ext, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest', zorder=1)
        for j, q in enumerate(m['steps'][:k + 1]):                       # кольца всех учтённых показаний
            r = q['reading']
            new = j == k
            ax.add_patch(Circle((r['x'], r['y']), r['d'], facecolor='none', edgecolor=INK if new else MUTED,
                                lw=1.5 if new else 0.9, ls=(0, (4, 3)), zorder=6, alpha=1.0 if new else 0.55))
            ax.plot(r['x'], r['y'], 'o', ms=7 if new else 4.5, mfc=INK if new else MUTED, mec='white', mew=1.1, zorder=9)
        ax.plot(*m['sample'], 'o', ms=12, mfc='none', mec=GREEN, mew=2.2, zorder=8)
        base(ax)
        ax.set_xlim(LIM[0], 0.9)                                            # левая часть арены крупнее: всё происходит у базы
        ax.set_ylim(-2.0, 1.6)
        info.append({'t': st['reading']['t'], 'z': round(st['reading']['z'], 2), 'd': round(st['reading']['d'], 2),
                     'top': round(st['top'][0]['mass'], 2), 'to_sample': round(st['top'][0]['to_sample'], 2)})
    save(fig, 'bayes_steps.png')
    return info


def fig_sensor_law():
    m = json.loads((HERE / 'data' / 'math.json').read_text())['belief']
    r = m['steps'][0]['reading']
    fig, ax = plt.subplots(figsize=(2.5, 1.9))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.24)
    ax.set_facecolor(BG)
    d = np.linspace(0, m['range'], 50)
    ax.fill_betweenx([0, r['z']], r['d'] - 2 * m['sigma'] * m['range'], r['d'] + 2 * m['sigma'] * m['range'], color='#DCE8FF', lw=0, zorder=1)
    ax.plot(d, 1 - d / m['range'], '-', color=INK, lw=2.0, zorder=3)
    ax.plot([0, r['d']], [r['z'], r['z']], '--', color=BLUE, lw=1.3, zorder=2)
    ax.plot([r['d'], r['d']], [r['z'], 0], '--', color=BLUE, lw=1.3, zorder=2)
    ax.plot(r['d'], r['z'], 'o', ms=7, mfc=BLUE, mec='white', mew=1.2, zorder=4)
    ax.set_xlim(0, m['range'])
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0,5', '1'])
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#B9C6DA')
    ax.set_xlabel('расстояние до образца, м', color=MUTED, fontsize=8.5, labelpad=2)
    ax.set_ylabel('показание z', color=MUTED, fontsize=8.5, labelpad=2)
    save(fig, 'sensor_law.png')
    return {'z': round(r['z'], 2), 'd': round(r['d'], 2)}


# ---------------------------------------------------------------- 9. математика: путь Дейкстры по карте стоимости
def fig_dijkstra():
    m = json.loads((HERE / 'data' / 'math.json').read_text())['path']
    fig, (ax,) = panels(1, 3.6)
    arena(ax)
    for z in m['soils']:
        zone(ax, z, 'soil', '×' + num(z['mult']).replace(',0', ''))
    for key, color, ls, lw in (('short', GREY, (0, (3, 2.5)), 2.0), ('cheap', BLUE, '-', 2.4)):
        pts = m[key]['pts']
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=color, ls=ls, lw=lw, solid_capstyle='round', zorder=6)
    ax.plot(*m['start'], 'o', ms=9, mfc=INK, mec='white', mew=1.3, zorder=9)
    ax.plot(*m['goal'], 'o', ms=10, mfc=GREEN, mec='white', mew=1.3, zorder=9)
    ax.set_xlim(-2.02, 1.45)
    ax.set_ylim(-1.27, 1.27)
    save(fig, 'dijkstra.png')
    return {k: {'meters': m[k]['meters'], 'energy': m[k]['energy']} for k in ('short', 'cheap')}


# ---------------------------------------------------------------- 10. миссия словами: запретная половина арены
def fig_pair_llm_zone(seed='medium-1002', limit=0.5, exp='L3d', arms=('M3_rule', 'M3_qwen3.8-flash-next'), out='pair_llm_zone.png'):
    rule = load(exp, arms[0], f'{seed}.json.gz')
    llm = load(exp, arms[1], f'{seed}.json.gz')
    info = {}
    fig, axes = panels(2, 3.4)
    for ax, rec, color, side in zip(axes, (rule, llm), (GREY, VIOLET), ('left', 'right')):
        arena(ax)
        ax.add_patch(Rectangle((limit, LIM[2]), LIM[1] - limit, LIM[3] - LIM[2], facecolor='#F8D2D2', edgecolor='none', alpha=0.55, zorder=0.8))
        ax.imshow(np.where(FREE, np.nan, 1.0), origin='lower', extent=EXT, cmap=LinearSegmentedColormap.from_list('o', [BG, BG]),
                  interpolation='nearest', zorder=0.9)
        ax.plot([limit, limit], [LIM[2] + 0.35, LIM[3] - 0.35], color=RED, lw=1.6, ls=(0, (5, 3)), zorder=4)
        truth(ax, rec['scenario'])
        tr = rec['track']
        ax.plot(tr['x'], tr['y'], '-', color=color, lw=2.1, solid_capstyle='round', zorder=5)
        inside = [i for i, x in enumerate(tr['x']) if x > limit]
        if inside:                                           # участки пути в запретной половине
            xs = [x if x > limit else np.nan for x in tr['x']]
            ax.plot(xs, tr['y'], '-', color=RED, lw=2.6, solid_capstyle='round', zorder=6)
        samples(ax, rec)
        base(ax)
        left_total = sum(1 for sx, _ in rec['scenario']['samples'] if sx <= limit)
        info[side] = {'collected': rec['result']['samples_collected'], 'total': rec['result']['samples_total'], 'left_total': left_total,
                      'share_inside': round(len(inside) / len(tr['x']), 2), 'returned': rec['result']['returned'], 'max_x': round(max(tr['x']), 2)}
    save(fig, out)
    info['seed'] = seed
    return info


# Составная миссия словами (tools/l3_combo.py): пара карт одного сценария и итог серии из runs/L3combo/summary.json.
COMBO_SEED = 'medium-1002'


def fig_pair_llm_combo():
    arms = ('rule', 'qwen3.8-flash-next')
    info = fig_pair_llm_zone(COMBO_SEED, exp='L3combo', arms=arms, out='pair_llm_combo.png')
    summary = json.loads((RUNS / 'L3combo' / 'summary.json').read_text())
    info['mission'] = summary['mission']
    info['series'] = {('rule' if a == 'rule' else 'llm'): {k: summary['arms'][a][k] for k in ('success', 'verified', 'n')} for a in arms}
    return info


if __name__ == '__main__':
    facts = {}
    for key, fn in (('scenario', fig_scenario), ('truth_vs_robot', fig_truth_vs_robot), ('belief', fig_belief),
                    ('pair_e1', fig_pair_e1), ('pair_fault', fig_pair_fault),
                    ('pair_llm_m4', lambda: fig_pair_llm('M4', 'hard-1003', 'pair_llm_m4.png')),
                    ('pair_llm_m1', lambda: fig_pair_llm('M1', 'hard-1006', 'pair_llm_m1.png')),
                    ('team', fig_team), ('bayes', fig_bayes), ('sensor_law', fig_sensor_law), ('dijkstra', fig_dijkstra),
                    ('pair_llm_zone', fig_pair_llm_zone),
                    ('pair_llm_combo', fig_pair_llm_combo)):
        try:
            facts[key] = fn()
        except Exception as e:  # рисунок не получился — остальные всё равно нужны
            facts[key] = {'error': repr(e)}
            print('ОШИБКА', key, repr(e))
    (HERE / 'data' / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: facts[k] for k in ('bayes', 'sensor_law', 'dijkstra', 'pair_llm_zone')}, ensure_ascii=False, default=str)[:2500])
