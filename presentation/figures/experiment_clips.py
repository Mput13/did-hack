#!/usr/bin/env python3
"""Короткие анимации прогонов (GIF) для слайдов и страницы разбора опытов.

В каждом ролике одна и та же миссия идёт в двух-трёх вариантах на общих часах: роботы едут одновременно,
образец закрашивается цветом робота, когда тот его собрал, внизу — время миссии и события сценария.
Всё рисуется по записям настоящих прогонов, тем же, что читает tools/build_showcase.py:

    runs/E1, runs/E9, runs/E17, runs/E7, runs/F1_v2   — pixi run exp <опыт>
    runs/_showcase                                    — pixi run python -m tools.showcase_p1_check
    ../DID-research/R13/runs/missions                 — миссии словами (переменная DID_R13_RUNS)

Запуск:  python3 presentation/figures/experiment_clips.py            # все ролики
         python3 presentation/figures/experiment_clips.py e1 r13     # только те, чьё имя начинается так

Нужны matplotlib, Pillow и ffmpeg. Пишет presentation/assets/clips/<имя>.gif и <имя>.png (итоговый кадр).
Первый кадр гифки — тоже итоговый: его показывают PDF, печать и миниатюры слайдов.
"""
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from did.arena import load_arena  # noqa: E402

RUNS = ROOT / 'runs'
R13_RUNS = Path(os.environ.get('DID_R13_RUNS', ROOT.parent / 'DID-research' / 'R13' / 'runs'))
OUT = ROOT / 'presentation' / 'assets' / 'clips'

# Цвета и вид карты — как на слайдах чекпоинта (presentation/figures/checkpoint2_figures.py).
INK, WALL, FLOOR, MUTED, LINE = '#14181C', '#2B3138', '#EEF0F2', '#5B6168', '#C9CDD2'
ORANGE, BLUE, GREEN, YELLOW, RED = '#EB6834', '#2A78D6', '#1BAF7A', '#EDA100', '#D03B3B'
SOIL = {1.5: '#F9E8DC', 2.0: '#F6DCCB', 2.5: '#F0C7AE', 3.0: '#E9B291', 4.0: '#DD9469'}
FAULT = '#FBE3A6'

W = 1200                    # ширина кадра, пикселей
DELAY_CS = 5                # задержка кадра в сотых долях секунды: 20 кадров в секунду
FPS = 100 / DELAY_CS
HOLD_S = 2.6                # сколько стоит итоговый кадр
PAD, GAP = 16, 22
ROW_CAPTION, ROW_HEAD, ROW_STATUS, ROW_FOOT = 38, 38, 38, 66

ARENA = load_arena()
EXTENT = (ARENA.x0, ARENA.x0 + ARENA.w * ARENA.res, ARENA.y0, ARENA.y0 + ARENA.h * ARENA.res)
WALLS = ARENA.solid & ndimage.binary_dilation(ARENA.free, iterations=2)
_ys, _xs = np.nonzero(WALLS)
VIEW = (ARENA.x0 + _xs.min() * ARENA.res - 0.05, ARENA.x0 + (_xs.max() + 1) * ARENA.res + 0.05,
        ARENA.y0 + _ys.min() * ARENA.res - 0.05, ARENA.y0 + (_ys.max() + 1) * ARENA.res + 0.05)

plt.rcParams.update({'font.family': 'Arial'})

EVENT = {'soil_change': 'грунт сменился', 'new_hazard': 'новая опасная зона', 'sensor_fault': 'сбой датчика'}
STATUS = {'explore': 'ищет образцы', 'travel': 'едет по плану', 'think': None, 'approach': 'подъезжает к образцу',
          'collect': 'подъезжает к образцу', 'return': 'возвращается на базу', 'wait': 'ждёт: датчик неисправен',
          'experiment': 'ставит опыт', 'escape': 'отъезжает от стены', 'done': None}


def px(n):
    """Размер шрифта и толщина линии в пикселях кадра."""
    return n * 72 / 100


def ru(v, d=1):
    return f'{v:.{d}f}'.replace('.', ',').replace('-', '−')


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return json.load(f)


def clock(t):
    return f'{int(t) // 60}:{int(t) % 60:02d}'


# ---------- запись прогона как функция времени ----------

class Track:
    def __init__(self, rec):
        tr = rec['track']
        self.t = np.asarray(tr['t'], float)
        self.x, self.y = np.asarray(tr['x'], float), np.asarray(tr['y'], float)
        self.th = np.unwrap(np.asarray(tr['th'], float))
        self.battery = np.asarray(tr['battery'], float)
        self.mode = [rec['modes'][i] for i in tr['mode']]
        self.end = float(self.t[-1])

    def at(self, t):
        t = min(t, self.end)
        return tuple(float(np.interp(t, self.t, a)) for a in (self.x, self.y, self.th, self.battery))

    def path(self, t0, t1):
        """Путь за время от t0 до t1; последняя точка — положение ровно в t1."""
        t1 = min(t1, self.end)
        if t1 <= t0:
            return [], []
        m = (self.t >= t0) & (self.t < t1)
        x0, y0, _, _ = self.at(t0)
        x1, y1, _, _ = self.at(t1)
        return [x0, *self.x[m], x1], [y0, *self.y[m], y1]

    def mode_at(self, t):
        return self.mode[max(0, min(len(self.mode) - 1, int(np.searchsorted(self.t, t, 'right')) - 1))]


def soils_at(sc, t):
    cur = sc.get('soils') or []
    for e in sc.get('events') or []:
        if e['type'] == 'soil_change' and e['t'] <= t:
            cur = e['soils']
    return cur


def zone_mask(z):
    X, Y = ARENA.cell_centers()
    if (z.get('shape') or 'circle') == 'circle':
        m = np.hypot(X - z['x'], Y - z['y']) <= z['r']
    else:
        m = (np.abs(X - z['x']) <= z['w'] / 2) & (np.abs(Y - z['y']) <= z['h'] / 2)
    return m & ARENA.free


def layer(mask, color):
    img = np.zeros((ARENA.h, ARENA.w, 4))
    img[mask] = to_rgba(color)
    return img


# ---------- описание ролика ----------

@dataclass
class Run:
    rec: dict
    label: str
    color: str
    ghost: dict = None            # путь другого прогона бледной линией под своим
    ghost_color: str = BLUE
    notes: list = field(default_factory=list)     # [(t0, t1, текст)]: подпись под картой на это время
    sub: str = ''                 # приписка к названию, мелко
    lost_at: float = None         # с этой секунды робот потерял положение: путь дальше — его оценка, а не правда


@dataclass
class Clip:
    name: str
    caption: str
    runs: list
    speed: int = 10               # во сколько раз ускорено
    vline: float = None           # граница из текста миссии, x в метрах
    dark: tuple = None            # (t0, t1): путь за это время рисуется чёрным
    journal: bool = False         # справа вместо второй карты — журнал гипотез первого прогона
    mission: bool = False         # подпись — текст задания, крупнее


# ---------- одна карта ----------

class MapView:
    def __init__(self, fig, ui, box, run, clip, small):
        self.run, self.clip, self.tr, self.sc = run, clip, Track(run.rec), run.rec['scenario']
        self.ghost = Track(run.ghost) if run.ghost else None
        x, y, w, h = box
        H = fig.get_figheight() * 100
        ax = self.ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.set_xlim(VIEW[0], VIEW[1])
        ax.set_ylim(VIEW[2], VIEW[3])
        ax.axis('off')
        k = self.k = w / (VIEW[1] - VIEW[0])                     # пикселей на метр
        fs = 12.5 if small else 15
        im = dict(origin='lower', extent=EXTENT, interpolation='nearest')
        ax.imshow(layer(ARENA.free, FLOOR), zorder=1, **im)
        ax.imshow(layer(WALLS, WALL), zorder=5, **im)

        # Грунт: набор зон до смены и после неё; виден тот, что действует сейчас.
        self.soil_sets, X, Y = [], *ARENA.cell_centers()
        busy = []                                                   # где подпись закроют образец или опасная зона
        for near in (0.36, 0.26):
            b = np.zeros_like(ARENA.free)
            for sx, sy in self.sc['samples']:
                b |= np.hypot(X - sx, Y - sy) < near
            for z in (self.sc.get('hazards') or []) + [e['zone'] for e in self.sc.get('events') or [] if e['type'] == 'new_hazard']:
                b |= np.hypot(X - z['x'], Y - z['y']) < z['r'] + 0.14
            busy.append(b)
        times = [0.0] + [e['t'] for e in self.sc.get('events') or [] if e['type'] == 'soil_change']
        for t0 in times:
            arts = []
            for z in soils_at(self.sc, t0):
                m = zone_mask(z)
                if not m.any():
                    continue
                arts.append(ax.imshow(layer(m, SOIL.get(float(z['mult']), SOIL[3.0])), zorder=2, **im))
                clear = next((m & ~b for b in busy if (m & ~b).any()), m)
                deep = ndimage.distance_transform_edt(clear)
                iy, ix = np.unravel_index(int(deep.argmax()), deep.shape)
                mult = ru(z['mult'], 0 if float(z['mult']).is_integer() else 1)
                arts.append(ax.text(X[iy, ix], Y[iy, ix], '×' + mult, ha='center', va='center', fontsize=px(fs),
                                    fontweight='bold', color='#7A3F1D', zorder=2.5))
            self.soil_sets.append((t0, arts))

        self.hazards = []                                           # (с какого времени, рисунки)
        zones = [(0.0, z) for z in self.sc.get('hazards') or []]
        zones += [(e['t'], e['zone']) for e in self.sc.get('events') or [] if e['type'] == 'new_hazard']
        self.pulses = []                                            # (время, кольцо, наибольший радиус)
        for t0, z in zones:
            arts = [ax.add_patch(Circle((z['x'], z['y']), z['r'], facecolor='#F7D4D4', edgecolor=RED,
                                        linewidth=px(2.2), zorder=3)),
                    ax.text(z['x'], z['y'], '!', ha='center', va='center', fontsize=px(fs + 2), fontweight='bold',
                            color=RED, zorder=3.5)]
            self.hazards.append((t0, arts))
            if t0 > 0:
                self._pulse(t0, z['x'], z['y'], RED, z['r'] + 0.35)

        if clip.vline is not None:
            ax.plot([clip.vline] * 2, [VIEW[2] + 0.2, VIEW[3] - 0.2], color=RED, linewidth=px(2.4),
                    dashes=(5, 3), zorder=6.5)
            ax.imshow(layer(ARENA.free & (X > clip.vline), to_rgba(RED, 0.09)), zorder=4, **im)

        lw = px(2.2 if small else 2.8)
        self.ghost_line, = ax.plot([], [], color=run.ghost_color, linewidth=lw, alpha=0.4, zorder=5.8,
                                   solid_capstyle='round', solid_joinstyle='round')
        self.trail, = ax.plot([], [], color=run.color, linewidth=lw, zorder=6, solid_capstyle='round',
                              solid_joinstyle='round')
        self.dark_line, = ax.plot([], [], color=INK, linewidth=lw * 1.35, zorder=6.2, solid_capstyle='round',
                                  solid_joinstyle='round')
        self.guess_line, = ax.plot([], [], color=run.color, linewidth=lw * 0.8, dashes=(2.5, 2.5), zorder=6)

        r = 0.075 if small else 0.085
        self.samples = [ax.add_patch(Circle((sx, sy), r, facecolor=YELLOW, edgecolor=INK, linewidth=px(1.4), zorder=7))
                        for sx, sy in self.sc['samples']]
        self.got = {}                                               # номер образца -> время сбора
        for e in run.rec.get('events') or []:
            if e['type'] == 'sample_collected':
                i = e.get('sample')
                if i is None:
                    i = min(range(len(self.samples)), key=lambda j: math.dist(self.sc['samples'][j], (e['x'], e['y'])))
                self.got[i] = e['t']
                self._pulse(e['t'], *self.sc['samples'][i], run.color, 0.38)

        self.marks = []                                             # штрафы: (время, рисунки)
        for e in run.rec.get('events') or []:
            if e['type'] in ('hazard_hit', 'collision') and e.get('x') is not None:
                d = 0.075
                arts = [ax.add_patch(Circle((e['x'], e['y']), 0.14, facecolor='white', edgecolor=RED,
                                            linewidth=px(1.6), zorder=8))]
                for sgn in (1, -1):
                    arts += ax.plot([e['x'] - d, e['x'] + d], [e['y'] - sgn * d, e['y'] + sgn * d], color=RED,
                                    linewidth=px(2.6), solid_capstyle='round', zorder=8.1)
                self.marks.append((e['t'], arts))
                self._pulse(e['t'], e['x'], e['y'], RED, 0.45)

        bx, by = self.sc.get('base') or (self.tr.x[0], self.tr.y[0])
        ax.add_patch(Rectangle((bx - 0.13, by - 0.13), 0.26, 0.26, facecolor='white', edgecolor=INK,
                               linewidth=px(1.8), zorder=6.6))
        self.robots = []
        for tr, color, z in ((self.ghost, run.ghost_color, 8.5), (self.tr, run.color, 9)):
            if tr is None:
                continue
            body = ax.add_patch(Circle((0, 0), 0.125 if small else 0.13, facecolor=color, edgecolor='white',
                                       linewidth=px(2), zorder=z, alpha=1.0 if tr is self.tr else 0.45))
            nose, = ax.plot([], [], color=color, linewidth=px(3.2), solid_capstyle='round', zorder=z,
                            alpha=1.0 if tr is self.tr else 0.45)
            self.robots.append((tr, body, nose))
        self.waiting = ax.add_patch(Circle((0, 0), 0.2, facecolor='none', edgecolor=YELLOW, linewidth=px(2.6),
                                           zorder=8.8, visible=False))

    def _pulse(self, t, x, y, color, rmax):
        ring = self.ax.add_patch(Circle((x, y), 0.1, facecolor='none', edgecolor=color, linewidth=px(2.4),
                                        zorder=7.5, visible=False))
        self.pulses.append((t, ring, rmax))

    def update(self, t, frame_s):
        cur = max(i for i, (t0, _) in enumerate(self.soil_sets) if t0 <= t)
        for i, (_, arts) in enumerate(self.soil_sets):
            for a in arts:
                a.set_visible(i == cur)
        for group in (self.hazards, self.marks):
            for t0, arts in group:
                for a in arts:
                    a.set_visible(t0 <= t)
        for i, s in enumerate(self.samples):
            done = i in self.got and self.got[i] <= t
            s.set_facecolor(self.run.color if done else YELLOW)
            s.set_edgecolor('white' if done else INK)
            s.set_linewidth(px(1.8 if done else 1.4))
        for t0, ring, rmax in self.pulses:                          # кольцо расходится полсекунды экранного времени
            age = (t - t0) / frame_s / (0.55 * FPS)
            ring.set_visible(0 <= age < 1)
            if 0 <= age < 1:
                ring.set_radius(0.1 + (rmax - 0.1) * age)
                ring.set_alpha(1 - age)
        lost = self.run.lost_at is not None and t >= self.run.lost_at
        self.trail.set_data(*self.tr.path(0, min(t, self.run.lost_at) if lost else t))
        if lost:
            self.guess_line.set_data(*self.tr.path(self.run.lost_at, t))
        if self.clip.dark:
            self.dark_line.set_data(*self.tr.path(self.clip.dark[0], min(t, self.clip.dark[1])))
        if self.ghost:
            self.ghost_line.set_data(*self.ghost.path(0, t))
        for tr, body, nose in self.robots:
            x, y, th, _ = tr.at(t)
            body.set_center((x, y))
            nose.set_data([x, x + 0.25 * math.cos(th)], [y, y + 0.25 * math.sin(th)])
            if tr is self.tr and lost:
                body.set_facecolor('white')
                body.set_edgecolor(self.run.color)
                body.set_linestyle((0, (2, 1.6)))
        still = t < self.tr.end and self.tr.mode_at(t) == 'wait'
        self.waiting.set_visible(still)
        if still:
            age = (t / frame_s / FPS) % 0.9 / 0.9                    # одно кольцо за 0,9 с экранного времени
            x, y, _, _ = self.tr.at(t)
            self.waiting.set_center((x, y))
            self.waiting.set_radius(0.16 + 0.3 * age)
            self.waiting.set_alpha(1 - age)

    def status(self, t):
        """Что робот делает сейчас, словами; None — оставить прежнюю подпись."""
        for t0, t1, text in self.run.notes:
            if t0 <= t <= min(t1, self.tr.end):
                return text
        return STATUS.get(self.tr.mode_at(t))


def final_text(rec, short=False):
    r = rec['result']
    tr, base = rec['track'], rec['scenario']['base']
    near = math.dist((tr['x'][-1], tr['y'][-1]), base) < 0.5
    state = 'на базе' if r['returned'] else 'не вернулся'
    if not r['returned'] and r.get('reason') == 'battery':
        state = 'разрядился у базы' if near else 'заряд кончился'
    mid = '' if short else f" · {r['samples_collected']} из {r['samples_total']}"
    return f"Итог: {ru(r['score'])} очка{mid} · {state}"


# ---------- журнал гипотез ----------

GOOD, BAD = '#0E6F4C', '#A52A2A'
CHIP = {'open': ('проверяется', '#E3E6EA', MUTED), 'left': ('не проверена', '#E3E6EA', MUTED),
        'confirmed': ('подтверждена', '#D5F0E4', GOOD), 'refuted': ('опровергнута', '#F7D9D9', BAD),
        'outdated': ('устарела', '#E3E6EA', MUTED)}
LAW = {'per_rad': 'поворот стоит заряда: {} ед. на радиан', 'per_m_load': 'груз удорожает путь: {} ед. на метр'}


def check(h):
    """Вывод робота против скрытой правды. Судья оценивает само утверждение гипотезы, поэтому
    «опровергнута» при неверном утверждении — вывод верный."""
    truth = h.get('truth_verdict') or 'unverifiable'
    if truth == 'unverifiable':
        return 'не проверить', MUTED
    if h['status'] not in ('confirmed', 'refuted'):
        return 'без вывода', MUTED
    return ('верно', GOOD) if (h['status'] == 'confirmed') == (truth == 'correct') else ('ошибка', BAD)


def short_statement(h):
    d = h.get('data') or {}
    at = f"({ru(d['x'])}; {ru(d['y'])})" if 'x' in d else ''
    if h['kind'] == 'sample':
        return f'образец около {at}'
    if h['kind'] == 'soil':
        return f"грунт около {at} дороже в {ru(d.get('mult', 0))} раза"
    if h['kind'] == 'hazard':
        return f'опасная зона около {at}'
    value = re.search(r'около ([\d.]+)', h['statement'])
    if d.get('law') in LAW and value:
        return LAW[d['law']].format(value[1].replace('.', ','))
    return h['statement'] if len(h['statement']) < 42 else h['statement'][:40] + '…'


class Journal:
    """Список гипотез робота: строка появляется, когда гипотеза выдвинута; после финиша — сверка с правдой."""

    def __init__(self, ui, box, rec):
        x, y, w, h = box
        self.hyps, self.end = rec['hypotheses'], rec['track']['t'][-1]
        row = min(38.0, (h - 30) / max(1, len(self.hyps)))
        self.cols = (x, x + 42, x + w - 226, x + w - 104)
        ui.text(self.cols[2] + 56, y + 9, 'вывод робота', fontsize=px(14), color=MUTED, ha='center', va='center')
        ui.text(self.cols[3] + 52, y + 9, 'сверка с правдой', fontsize=px(14), color=MUTED, ha='center', va='center')
        renderer = ui.figure.canvas.get_renderer()
        ui.plot([x, x + w], [y + 24, y + 24], color=LINE, linewidth=px(1))
        self.rows = []
        for i, hyp in enumerate(self.hyps):
            cy = y + 30 + row * (i + 0.5)
            arts = {'id': ui.text(self.cols[0], cy, hyp['id'], fontsize=px(15.5), fontweight='bold', color=INK, va='center'),
                    'text': ui.text(self.cols[1], cy, short_statement(hyp), fontsize=px(15.5), color=INK, va='center'),
                    'chip': ui.add_patch(FancyBboxPatch((self.cols[2], cy - 12), 112, 24, linewidth=0,
                                                        boxstyle='round,pad=0,rounding_size=6')),
                    'state': ui.text(self.cols[2] + 56, cy, '', fontsize=px(14), va='center', ha='center'),
                    'truth': ui.text(self.cols[3] + 52, cy, '', fontsize=px(14.5), fontweight='bold', va='center', ha='center')}
            room = self.cols[2] - self.cols[1] - 10                  # длинное утверждение ужимается до своей колонки
            width = arts['text'].get_window_extent(renderer).width
            if width > room:
                arts['text'].set_fontsize(px(15.5) * room / width)
            self.rows.append(arts)

    def update(self, t, hold):
        """hold — сколько кадров прошло после финиша: строки сверки открываются по одной."""
        for i, (hyp, a) in enumerate(zip(self.hyps, self.rows)):
            seen = hyp['t_open'] <= t
            closed = hyp.get('t_close') is not None and hyp['t_close'] <= t
            state = hyp['status'] if closed or t >= self.end else 'open'
            word, bg, fg = CHIP['left' if state == 'open' and t >= self.end else state]
            for key in ('id', 'text', 'chip', 'state'):
                a[key].set_visible(seen)
            a['chip'].set_facecolor(bg)
            a['state'].set_text(word)
            a['state'].set_color(fg)
            word, color = check(hyp)
            a['truth'].set_text(word if seen and hold > 1 + i else '')
            a['truth'].set_color(color)


# ---------- кадр целиком ----------

def legend(ui, x_right, y, clip):
    """Условные знаки в строке подписи, справа налево."""
    has_haz = any(r.rec['scenario'].get('hazards') or any(e['type'] == 'new_hazard' for e in r.rec['scenario'].get('events') or [])
                  for r in clip.runs)
    has_soil = any(r.rec['scenario'].get('soils') for r in clip.runs)
    items = [('sample', 'образец'), ('got', 'собран')] + ([('soil', 'дорогой грунт')] if has_soil else []) \
        + ([('haz', 'опасная зона')] if has_haz else [])
    x = x_right
    for kind, word in reversed(items):
        t = ui.text(x, y, word, fontsize=px(15), color=MUTED, ha='right', va='center')
        x -= t.get_window_extent(ui.figure.canvas.get_renderer()).width + 8
        if kind == 'sample':
            ui.add_patch(Circle((x - 8, y), 7, facecolor=YELLOW, edgecolor=INK, linewidth=px(1.4)))
            x -= 16
        elif kind == 'got':
            for run in reversed(clip.runs):
                ui.add_patch(Circle((x - 8, y), 7, facecolor=run.color, edgecolor='white', linewidth=px(1.4)))
                x -= 15
            x -= 1
        elif kind == 'soil':
            ui.add_patch(Rectangle((x - 20, y - 8), 20, 16, facecolor=SOIL[3.0], linewidth=0))
            x -= 20
        else:
            ui.add_patch(Circle((x - 9, y), 8, facecolor='#F7D4D4', edgecolor=RED, linewidth=px(1.6)))
            x -= 18
        x -= 22
    return x + 22                                                    # левый край условных знаков


def timebar(ui, y, t_max, clip):
    """Шкала времени миссии с событиями сценария; возвращает функцию обновления."""
    x0, x1 = PAD + 74, W - PAD - 136
    yb = y + 36
    to_x = lambda t: x0 + (x1 - x0) * t / t_max      # noqa: E731
    ui.plot([x0, x1], [yb, yb], color=LINE, linewidth=px(6), solid_capstyle='round')
    events = sorted(clip.runs[0].rec['scenario'].get('events') or [], key=lambda e: e['t'])
    placed = [[], []]                                                # занятые отрезки над шкалой и под ней
    for e in events:
        ex = to_x(e['t'])
        if e['type'] == 'sensor_fault':
            ui.add_patch(Rectangle((ex, yb - 9), to_x(min(t_max, e['t'] + e['duration'])) - ex, 18, facecolor=FAULT,
                                   linewidth=0, zorder=0.5))
        word = EVENT[e['type']]
        width = 7.4 * len(word)
        ex1 = to_x(min(t_max, e['t'] + e.get('duration', 0)))
        left = max(ex + 4, (ex + ex1 - width) / 2) if e['type'] == 'sensor_fault' else ex + 4
        row = next((i for i in (0, 1) if all(left > b + 14 or left + width + 14 < a for a, b in placed[i])), 1)
        placed[row].append((left, left + width))
        ui.plot([ex, ex], [yb - 9 if row == 0 else yb + 9, yb], color=MUTED, linewidth=px(1.6))
        ui.text(left, yb - 20 if row == 0 else yb + 20, word, fontsize=px(14.5), color=MUTED, va='center')
    done, = ui.plot([x0, x0], [yb, yb], color=INK, linewidth=px(6), solid_capstyle='round')
    knob = ui.add_patch(Circle((x0, yb), 7, facecolor=INK, edgecolor='white', linewidth=px(2), zorder=5))
    now = ui.text(PAD, yb, '0:00', fontsize=px(21), fontweight='bold', color=INK, va='center')
    ui.text(W - PAD, yb, f'ускорено ×{clip.speed}', fontsize=px(15), color=MUTED, va='center', ha='right')

    def update(t):
        done.set_data([x0, to_x(t)], [yb, yb])
        knob.set_center((to_x(t), yb))
        now.set_text(clock(t))
    return update


def render(clip, out=OUT):
    n = 1 if clip.journal else len(clip.runs)
    cols = 2 if clip.journal else n
    small = cols >= 3
    pw = (W - 2 * PAD - GAP * (cols - 1)) / cols
    ph = round(pw * (VIEW[3] - VIEW[2]) / (VIEW[1] - VIEW[0]))
    H = ROW_CAPTION + ROW_HEAD + ph + ROW_STATUS + ROW_FOOT
    H += H % 2
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor='white')
    ui = fig.add_axes([0, 0, 1, 1])
    ui.set_xlim(0, W)
    ui.set_ylim(H, 0)
    ui.axis('off')
    fig.canvas.draw()

    size = 17 if clip.mission else 15.5
    caption = ui.text(PAD, ROW_CAPTION / 2 + 2, clip.caption, fontsize=px(size), va='center',
                      color=INK if clip.mission else MUTED, fontweight='bold' if clip.mission else 'normal')
    room = legend(ui, W - PAD, ROW_CAPTION / 2 + 2, clip) - PAD - 24
    width = caption.get_window_extent(fig.canvas.get_renderer()).width
    if width > room:
        caption.set_fontsize(px(size) * room / width)
    ui.plot([PAD, W - PAD], [ROW_CAPTION, ROW_CAPTION], color=LINE, linewidth=px(1))

    y_head, y_map = ROW_CAPTION + ROW_HEAD / 2 + 3, ROW_CAPTION + ROW_HEAD
    y_stat = y_map + ph + ROW_STATUS / 2
    views, panels = [], []
    for i, run in enumerate(clip.runs[:n]):
        x = PAD + i * (pw + GAP)
        view = MapView(fig, ui, (x, y_map, pw, ph), run, clip, small)
        views.append(view)
        fs = 18 if small else 21
        ui.add_patch(Circle((x + 9, y_head), 8, facecolor=run.color, linewidth=0))
        name = ui.text(x + 26, y_head, run.label, fontsize=px(fs), fontweight='bold', color=INK, va='center')
        if run.sub:
            ui.text(x + 26 + name.get_window_extent(fig.canvas.get_renderer()).width + 12, y_head + 1, run.sub,
                    fontsize=px(14.5), color=MUTED, va='center')
        total = len(view.samples)
        count = ui.text(x + pw, y_head, f'0/{total}', fontsize=px(fs - 3), color=INK, va='center', ha='right')
        step = 17 if small else 19
        x_pip = x + pw - (44 if small else 50) - step * (total - 1)
        pips = [ui.add_patch(Circle((x_pip + step * j, y_head), 6.5, facecolor='white', edgecolor=MUTED, linewidth=px(1.4)))
                for j in range(total)]
        status = ui.text(x, y_stat, '', fontsize=px(15.5 if small else 17), color=MUTED, va='center')
        bw = 64 if small else 90
        ui.text(x + pw - bw - 46, y_stat, 'заряд', fontsize=px(14 if small else 15), color=MUTED, va='center', ha='right')
        ui.add_patch(Rectangle((x + pw - bw - 38, y_stat - 6), bw, 12, facecolor='#E3E6EA', linewidth=0))
        bar = ui.add_patch(Rectangle((x + pw - bw - 38, y_stat - 6), bw, 12, facecolor=INK, linewidth=0))
        left = ui.text(x + pw, y_stat, '', fontsize=px(15 if small else 16), color=INK, va='center', ha='right')
        panels.append((view, count, pips, status, bar, bw, left))

    journal = None
    if clip.journal:
        x = PAD + pw + GAP
        ui.text(x, y_head, 'Журнал гипотез робота', fontsize=px(21), fontweight='bold', color=INK, va='center')
        journal = Journal(ui, (x, y_map + 4, pw, ph + ROW_STATUS - 8), clip.runs[0].rec)

    t_max = max(max(v.tr.end, v.ghost.end if v.ghost else 0) for v in views)
    tick = timebar(ui, y_map + ph + ROW_STATUS, t_max, clip)
    frame_s = clip.speed / FPS                                      # секунд миссии на кадр
    times = [i * frame_s for i in range(int(t_max / frame_s) + 1)] + [t_max]
    hold = round((HOLD_S + (1.2 if clip.journal else 0)) * FPS)

    # Подпись под картой меняется не чаще чем раз в полсекунды экранного времени, иначе она мигает.
    texts = []
    for view in views:
        raw, last = [], 'ищет образцы'
        for t in times:
            last = view.status(t) or last
            raw.append(last)
        k = max(1, round(0.3 * FPS))
        texts.append([max(set(raw[max(0, i - k):i + k + 1]), key=raw[max(0, i - k):i + k + 1].count) for i in range(len(raw))])

    tmp = Path(tempfile.mkdtemp(prefix='did_clip_'))
    frames = [(t, 0, i) for i, t in enumerate(times)] + [(t_max, j + 1, len(times) - 1) for j in range(hold)]
    last, fitted = None, {}
    for f, (t, held, ti) in enumerate(frames):
        for vi, (view, count, pips, status, bar, bw, left) in enumerate(panels):
            view.update(t, frame_s)
            got = sum(1 for tt in view.got.values() if tt <= t)
            count.set_text(f'{got}/{len(pips)}')
            for j, p in enumerate(pips):
                p.set_facecolor(view.run.color if j < got else 'white')
                p.set_edgecolor(view.run.color if j < got else MUTED)
            over = t >= view.tr.end
            status.set_text(final_text(view.run.rec, short=small) if over else texts[vi][ti])
            status.set_color(INK if over else MUTED)
            status.set_fontweight('bold' if over else 'normal')
            if over and vi not in fitted:
                room = pw - bw - 104
                width = status.get_window_extent(fig.canvas.get_renderer()).width
                fitted[vi] = status.get_fontsize() * min(1.0, room / width)
                status.set_fontsize(fitted[vi])
            battery = view.tr.at(t)[3]
            bar.set_width(bw * max(0.0, battery) / view.run.rec['rules']['battery_start'])
            bar.set_facecolor(RED if battery < 6 else INK)
            left.set_text(f'{battery:.0f}')
        if journal:
            journal.update(t, held)
        tick(t)
        fig.canvas.draw()
        last = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        last.save(tmp / f'f{f + 1:04d}.png', compress_level=1)
    plt.close(fig)
    last.save(tmp / 'f0000.png', compress_level=1)                  # первый кадр — итоговая картинка
    out.mkdir(parents=True, exist_ok=True)
    last.save(out / f'{clip.name}.png', optimize=True)

    src = ['-framerate', f'100/{DELAY_CS}', '-i', str(tmp / 'f%04d.png')]
    run = lambda *a: subprocess.run(['ffmpeg', '-v', 'error', '-y', *a], check=True)      # noqa: E731
    run(*src, '-vf', 'palettegen=max_colors=160:stats_mode=full', str(tmp / 'palette.png'))
    run(*src, '-i', str(tmp / 'palette.png'), '-lavfi', 'paletteuse=dither=none:diff_mode=rectangle',
        '-loop', '0', str(out / f'{clip.name}.gif'))
    shutil.rmtree(tmp)
    size = (out / f'{clip.name}.gif').stat().st_size
    print(f'{clip.name}.gif  {W}×{H}  {len(frames) / FPS:.1f} с  {size / 1e6:.2f} МБ', flush=True)
    return {'name': clip.name, 'width': W, 'height': H, 'seconds': round(len(frames) / FPS, 1), 'bytes': size,
            'speed': clip.speed, 'caption': clip.caption}


# ---------- ролики ----------

def clips():
    out = []
    seed = 1026
    out.append(Clip('e1_plan_vs_adaptive', f'Трудный уровень, сценарий {seed}: один мир, два агента', [
        Run(load(RUNS / 'E1' / 'fixed' / f'hard-{seed}.json.gz'), 'Фиксированный план', BLUE),
        Run(load(RUNS / 'E1' / 'adaptive' / f'hard-{seed}.json.gz'), 'Адаптивный агент', ORANGE)]))

    seed = 1005
    e9 = {a: load(RUNS / 'E9' / a / f'medium-{seed}.json.gz') for a in ('spiral', 'gradient', 'adaptive')}
    out.append(Clip('e9_search', f'Средний уровень, сценарий {seed}: три способа искать образцы', [
        Run(e9['spiral'], 'Спираль', BLUE), Run(e9['gradient'], 'Подъём по сигналу', GREEN),
        Run(e9['adaptive'], 'Карта вероятностей', ORANGE)]))

    seed = 8024
    old = load(RUNS / '_showcase' / 'adaptive' / f'hard-{seed}.json.gz')
    new = load(RUNS / '_showcase' / 'adaptive_v2' / f'hard-{seed}.json.gz')
    ev = next(e for e in old['scenario']['events'] if e['type'] == 'sensor_fault')
    f0, f1 = ev['t'], ev['t'] + ev['duration']
    out.append(Clip('p1_sensor_fault', f'Трудный уровень, сценарий {seed}: датчик образцов ломается на {ru(ev["duration"], 0)} секунд', [
        Run(old, 'Прежний агент', BLUE, notes=[(f0, f1, 'датчик врёт — едет по показаниям')]),
        Run(new, 'Новый агент', ORANGE, notes=[(f0, f0 + 4.0, 'заметил сбой датчика')])], dark=(f0, f1)))

    m = R13_RUNS / 'missions'
    out.append(Clip('r13_two_samples', 'Задание словами: «Собери ровно два образца и сразу возвращайся на базу»', [
        Run(load(m / 'M1_rule' / 'hard-1001.json.gz'), 'Правило: текст не читает', BLUE),
        Run(load(m / 'M1_llm' / 'hard-1001.json.gz'), 'Модель читает задание', ORANGE)], speed=8, mission=True))
    out.append(Clip('r13_boundary', 'Задание словами: «Не заезжай в правую половину арены (x больше 0,5 м)»', [
        Run(load(m / 'M3_rule' / 'hard-1002.json.gz'), 'Правило: текст не читает', BLUE),
        Run(load(m / 'M3_llm_ask' / 'hard-1002.json.gz'), 'Модель читает задание', ORANGE)], vline=0.5, mission=True))

    out.append(Clip('e17_journal', 'Трудный уровень, усложнённые правила, сценарий 1001: что робот предположил и что вышло', [
        Run(load(RUNS / 'E17' / 'scientist@science' / 'hard-1001.json.gz'), 'Исследователь', ORANGE)], journal=True))

    f4, g4 = (load(RUNS / 'E7' / b / 'hard-4.json.gz') for b in ('fastsim', 'gazebo'))
    out.append(Clip('e7_fast_vs_gazebo', 'Трудный уровень, сценарий 4: один агент в двух симуляторах', [
        Run(f4, 'Быстрый симулятор', BLUE),
        Run(g4, 'Gazebo', ORANGE, ghost=f4, ghost_color=BLUE, sub='бледным — быстрый симулятор')], speed=12))
    # В Gazebo в записи лежит положение по оценке самого робота. В hard-5 до правки она разошлась с правдой
    # на 81–83-й секунде (research/findings/G2.md, «цепочка срыва»): дальше путь — то, что робот думал.
    lost = 82.0
    before = load(RUNS / 'E7' / 'gazebo' / 'hard-5.json.gz')
    out.append(Clip('g2_gazebo_fix', 'Gazebo, трудный уровень, сценарий 5: до правки и после неё', [
        Run(before, 'До правки', BLUE, lost_at=lost, sub='пунктир — где он себя считал',
            notes=[(lost, before['track']['t'][-1], 'потерял положение: думает, что едет домой')]),
        Run(load(RUNS / 'F1_v2' / 'adaptive_v2' / 'hard-5.json.gz'), 'После правки', ORANGE)], speed=12))
    return out


def render_named(name):
    return render(next(c for c in clips() if c.name == name))


def main():
    want = sys.argv[1:]
    names = [c.name for c in clips() if not want or any(c.name.startswith(w) for w in want)]
    with ProcessPoolExecutor(max_workers=min(len(names), os.cpu_count() or 2)) as pool:      # ролики собираются одновременно
        meta = list(pool.map(render_named, names))
    if not want:
        (OUT / 'clips.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
