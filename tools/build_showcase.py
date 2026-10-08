#!/usr/bin/env python3
"""Шесть опытов для показа: две страницы из одних и тех же чисел и рисунков.

    pixi run python tools/build_showcase.py

  docs/experiments.html        — для жюри: вопрос, устройство, результат, пример прогона, ограничения;
  docs/experiments_guide.html  — для докладчика: то же плюс вопросы с ответами и порядок рассказа.

Анимации прогонов (presentation/assets/clips/*.gif) собирает presentation/figures/experiment_clips.py;
они вшиты прямо в страницы, поэтому каждая страница — один файл, который открывается откуда угодно.

Числа берутся из сводок опытов (runs/<опыт>/summary.json), рисунки — из записей прогонов, тексты — в этом файле.
Чего нет в основном каталоге:
  - опыт E22b (пережидание сбоя датчика) и миссии R13 считались в рабочих деревьях исследований; их сводка и
    записи читаются оттуда (переменные DID_P1_RUNS и DID_R13_RUNS, по умолчанию ../DID-research/<имя>/runs);
  - главное число E22b перепроверяется на текущем коде отдельно: pixi run python -m tools.showcase_p1_check.
Кроме llm_calls.html, внешних ссылок на страницах нет.
"""
import base64
import gzip
import html
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from did.arena import load_arena  # noqa: E402
from did.config import BASE  # noqa: E402

RUNS = ROOT / 'runs'
P1_RUNS = Path(os.environ.get('DID_P1_RUNS', ROOT.parent / 'DID-research' / 'P1' / 'runs'))
R13_RUNS = Path(os.environ.get('DID_R13_RUNS', ROOT.parent / 'DID-research' / 'R13' / 'runs'))
OUT = ROOT / 'docs' / 'experiments_guide.html'
OUT_JURY = ROOT / 'docs' / 'experiments.html'
CLIPS = ROOT / 'presentation' / 'assets' / 'clips'

VERDICT = {'supported': 'подтверждено', 'inconclusive': 'различие не показано', 'refuted': 'опровергнуто'}
LEVEL = {'easy': 'лёгкий', 'medium': 'средний', 'hard': 'трудный'}


# ---------- числа ----------
def num(x, d=1):
    return f'{x:.{d}f}'.replace('.', ',').replace('-', '−')


def sgn(x, d=1):
    return f'{x:+.{d}f}'.replace('.', ',').replace('-', '−')


def pct(x, d=0):
    return num(100 * x, d) + '%'


def ci_text(p, d=1, k=1.0):
    return f"{sgn(k * p['mean'], d)} [{num(k * p['ci'][0], d)}; {num(k * p['ci'][1], d)}]"


def esc(s):
    return html.escape(str(s), quote=True)


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def summary(exp, runs=RUNS):
    return json.loads((runs / exp / 'summary.json').read_text(encoding='utf-8'))


def pair(s, metric, a, b, cond='base', level='hard'):
    """Парная разность a − b из сводки опыта: (статистика, вердикт)."""
    for c in s['claims']:
        if c['metric'] == metric and c['a'] == a and c['b'] == b:
            for cell in c['cells']:
                if cell.get('condition', 'base') == cond and cell['level'] == level:
                    return cell['pair'], cell['verdict']
    raise KeyError((metric, a, b, cond, level))


def stat(s, arm, metric, cond='base', level='hard'):
    for g in s['groups']:
        if g['arm'] == arm and g.get('condition', 'base') == cond and g['level'] == level:
            return g['stats'][metric]
    raise KeyError((arm, metric, cond, level))


def runs_of(s, arm, cond='base', level='hard'):
    return {r['seed']: r for r in s['runs'] if r['arm'] == arm and r.get('condition', 'base') == cond and r['level'] == level}


# ---------- карта арены ----------
ARENA = load_arena()
K = 52.0                                    # пикселей на метр


def _arena_path():
    a, parts = ARENA, []
    ys, xs = np.nonzero(a.free)
    for iy in range(a.h):
        row, ix = a.free[iy], 0
        while ix < a.w:
            if row[ix]:
                j = ix
                while j < a.w and row[j]:
                    j += 1
                parts.append(f'M{a.x0 + ix * a.res:.2f} {a.y0 + iy * a.res:.2f}h{(j - ix) * a.res:.2f}'
                             f'v{a.res * 1.08:.3f}h{-(j - ix) * a.res:.2f}z')
                ix = j
            else:
                ix += 1
    pad = 0.1
    return ''.join(parts), (a.x0 + xs.min() * a.res - pad, a.y0 + ys.min() * a.res - pad,
                            a.x0 + (xs.max() + 1) * a.res + pad, a.y0 + (ys.max() + 1) * a.res + pad)


ARENA_D, (XMIN, YMIN, XMAX, YMAX) = _arena_path()
MW, MH = (XMAX - XMIN) * K, (YMAX - YMIN) * K


def sx(x):
    return (x - XMIN) * K


def sy(y):
    return (YMAX - y) * K


def _zone(z, cls, tip, label=None):
    if (z.get('shape') or 'circle') == 'circle':
        shape = f'<circle cx="{sx(z["x"]):.1f}" cy="{sy(z["y"]):.1f}" r="{z["r"] * K:.1f}"'
    else:
        shape = (f'<rect x="{sx(z["x"] - z["w"] / 2):.1f}" y="{sy(z["y"] + z["h"] / 2):.1f}" '
                 f'width="{z["w"] * K:.1f}" height="{z["h"] * K:.1f}" rx="2"')
    out = f'{shape} class="{cls}" data-tip="{esc(tip)}"/>'
    if label:
        out += f'<text x="{sx(z["x"]):.1f}" y="{sy(z["y"]) + 3.5:.1f}" class="zl">{esc(label)}</text>'
    return out


def collected_set(rec):
    got, samples = set(), rec['scenario']['samples']
    for e in rec.get('events') or []:
        if e.get('type') != 'sample_collected':
            continue
        if e.get('sample') is not None:
            got.add(e['sample'])
        elif e.get('x') is not None and samples:
            got.add(min(range(len(samples)), key=lambda i: math.dist(samples[i], (e['x'], e['y']))))
    return got


def _poly(tr, lo=None, hi=None, step=None):
    idx = [i for i, t in enumerate(tr['t']) if (lo is None or t >= lo) and (hi is None or t <= hi)]
    if not idx:
        return ''
    step = step or max(1, len(idx) // 320)
    idx = idx[::step] + ([idx[-1]] if idx[-1] not in idx[::step] else [])
    return ' '.join(f'{sx(tr["x"][i]):.1f},{sy(tr["y"][i]):.1f}' for i in idx)


def run_map(rec, title, sub, *, color='s1', highlight=None, vline=None, second=None, second_color='s2'):
    """Карта одного прогона. highlight=(t0, t1) — участок пути другим цветом; second — второй прогон поверх."""
    sc = rec['scenario']
    got = collected_set(rec)
    p = [f'<svg viewBox="0 0 {MW:.0f} {MH:.0f}" role="img" aria-label="{esc(title)}">',
         f'<use href="#arena" transform="translate({-XMIN * K:.1f},{YMAX * K:.1f}) scale({K},{-K})"/>']
    for z in sc.get('soils') or []:
        p.append(_zone(z, 'soil', f'Дорогой грунт {z["id"]}: метр стоит в {num(z["mult"])} раза больше',
                       '×' + num(z['mult']).rstrip('0').rstrip(',')))
    for z in sc.get('hazards') or []:
        p.append(_zone(z, 'haz', f'Опасная зона {z["id"]}: заезд — штраф', '!'))
    for e in sc.get('events') or []:
        if e.get('type') == 'new_hazard':
            z = e['zone']
            p.append(_zone(z, 'haz', f'Опасная зона {z["id"]}: появилась на {num(e["t"], 0)}-й секунде', '!'))
    if vline is not None:
        p.append(f'<line x1="{sx(vline):.1f}" y1="{sy(YMAX - 0.25):.1f}" x2="{sx(vline):.1f}" y2="{sy(YMIN + 0.25):.1f}" '
                 f'class="ban" data-tip="Граница из текста миссии: x = {num(vline)} м"/>')
    if second is not None:
        p.append(f'<polyline points="{_poly(second["track"])}" class="tr {second_color}"/>')
    p.append(f'<polyline points="{_poly(rec["track"])}" class="tr {color}"/>')
    if highlight:
        p.append(f'<polyline points="{_poly(rec["track"], *highlight, step=1)}" class="tr hl"/>')
    for i, (x, y) in enumerate(sc['samples']):
        cls, word = ('sm got', 'собран') if i in got else ('sm', 'не собран')
        p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.6" class="{cls}" data-tip="Образец {i + 1}: {word}"/>')
    for e in rec.get('events') or []:
        if e.get('type') in ('hazard_hit', 'collision') and e.get('x') is not None:
            x, y, word = sx(e['x']), sy(e['y']), 'заезд в опасную зону' if e['type'] == 'hazard_hit' else 'столкновение'
            p.append(f'<g class="pen" data-tip="Штраф на {num(e["t"], 0)}-й секунде: {word}">'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"/><path d="M{x - 4:.1f} {y - 4:.1f}l8 8M{x + 4:.1f} {y - 4:.1f}l-8 8"/></g>')
    p.append(f'<rect x="{sx(BASE[0]) - 5:.1f}" y="{sy(BASE[1]) - 5:.1f}" width="10" height="10" rx="2" class="base" data-tip="База"/>')
    for r, cls in ((second, second_color), (rec, color)):
        if r is not None:
            p.append(f'<circle cx="{sx(r["track"]["x"][-1]):.1f}" cy="{sy(r["track"]["y"][-1]):.1f}" r="4" class="end {cls}" '
                     f'data-tip="Конец пути"/>')
    p.append('</svg>')
    return f'<figure class="map">{"".join(p)}<figcaption><b>{title}</b><span>{sub}</span></figcaption></figure>'


def result_line(rec):
    r = rec['result']
    back = 'вернулся' if r['returned'] else 'не вернулся'
    pen = (r.get('hazard_hits') or 0) + (r.get('collisions') or 0)
    return (f"{num(r['score'])} очка · {r['samples_collected']} из {r['samples_total']} · {back} · "
            f"{num(r['time'], 0)} с" + (f" · штрафов {pen}" if pen else ''))


MAP_LEGEND = ('<div class="legend maplegend">'
              '<span class="key"><i class="dot got"></i>образец собран</span>'
              '<span class="key"><i class="dot open"></i>не собран</span>'
              '<span class="key"><i class="sq soil"></i>дорогой грунт, × цена метра</span>'
              '<span class="key"><i class="sq haz"></i>опасная зона</span>'
              '<span class="key"><i class="x">×</i>штраф</span>'
              '<span class="key"><i class="sq base"></i>база</span></div>')


def legend(items):
    return '<div class="legend">' + ''.join(f'<span class="key"><i class="dot {c}"></i>{esc(n)}</span>' for c, n in items) + '</div>'


# ---------- график «точка и интервал» ----------
def interval_chart(rows, series, lo, hi, ticks, *, zero=None, unit='', lanes=True, connect=False,
                   label_w=255, value_w=165, width=800, d=1, head=('Строка', 'Значение'), aria=''):
    """rows: [{'label', 'sub'?, 'pts': [{'s', 'm', 'lo'?, 'hi'?, 'val', 'tip', 'muted'?}]}]."""
    px0, px1 = label_w, width - value_w

    def X(v):
        return px0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (px1 - px0)

    geo, y = [], 8
    for r in rows:
        n = len(r['pts']) if lanes else 1
        h = max(30 + 18 * (n - 1), 40 if r.get('sub') else 30)
        geo.append((y, h, n))
        y += h
    bottom = y + 4
    H = bottom + 26
    o = [f'<svg viewBox="0 0 {width} {H}" class="chart" role="img" aria-label="{esc(aria)}">']
    for t in ticks:
        cls = 'zero' if zero is not None and abs(t - zero) < 1e-9 else 'grid'
        o.append(f'<line x1="{X(t):.1f}" y1="4" x2="{X(t):.1f}" y2="{bottom}" class="{cls}"/>')
        o.append(f'<text x="{X(t):.1f}" y="{bottom + 16}" class="tick">{num(t, 0 if float(t).is_integer() else d)}{unit}</text>')
    for r, (ry, rh, n) in zip(rows, geo):
        cy = ry + rh / 2
        if r.get('sub'):
            o.append(f'<text x="{px0 - 14}" y="{cy - 3:.1f}" class="rl">{esc(r["label"])}</text>')
            o.append(f'<text x="{px0 - 14}" y="{cy + 12:.1f}" class="rs">{esc(r["sub"])}</text>')
        else:
            o.append(f'<text x="{px0 - 14}" y="{cy + 4:.1f}" class="rl">{esc(r["label"])}</text>')
        if connect and len(r['pts']) == 2:
            a, b = (X(pt['m']) for pt in r['pts'])
            o.append(f'<line x1="{a:.1f}" y1="{cy:.1f}" x2="{b:.1f}" y2="{cy:.1f}" class="conn"/>')
        for i, pt in enumerate(r['pts']):
            py = (ry + rh / 2 - 9 * (n - 1) + 18 * i) if lanes else cy
            cls = 'mut' if pt.get('muted') else f's{pt["s"] + 1}'
            if pt.get('lo') is not None:
                o.append(f'<line x1="{X(pt["lo"]):.1f}" y1="{py:.1f}" x2="{X(pt["hi"]):.1f}" y2="{py:.1f}" class="ci {cls}"/>')
            o.append(f'<circle cx="{X(pt["m"]):.1f}" cy="{py:.1f}" r="5" class="pt {cls}"/>')
            if lanes or i == 0:
                vx = px1 + 16
                if len(series) > 1 and lanes:
                    o.append(f'<circle cx="{vx + 4}" cy="{py:.1f}" r="3.5" class="pt {cls} bare"/>')
                    vx += 13
                text = pt['val'] if lanes else r.get('val', pt['val'])
                o.append(f'<text x="{vx}" y="{py + 4:.1f}" class="val">{esc(text)}</text>')
            o.append(f'<rect x="{px0}" y="{py - 9:.1f}" width="{px1 - px0}" height="18" class="hit" data-tip="{esc(pt["tip"])}"/>')
    o.append('</svg>')
    table = ['<details class="tbl"><summary>Таблица с числами</summary><table><thead><tr>'
             f'<th>{head[0]}</th>' + ('<th>Кто</th>' if len(series) > 1 else '') + f'<th>{head[1]}</th></tr></thead><tbody>']
    for r in rows:
        for pt in r['pts']:
            who = f'<td>{esc(series[pt["s"]])}</td>' if len(series) > 1 else ''
            table.append(f'<tr><td>{esc(r["label"])}{(" — " + esc(r["sub"])) if r.get("sub") else ""}</td>{who}<td>{esc(pt["val"])}</td></tr>')
    table.append('</tbody></table></details>')
    keys = legend([(f's{i + 1}', n) for i, n in enumerate(series)]) if len(series) > 1 else ''
    return keys + ''.join(o) + ''.join(table)


# ---------- временной ряд датчика ----------
def timeline(rec, name, color, tmax, ymax, bands, width=800, height=150):
    tr = rec['track']
    px0, px1, py0, py1 = 46, width - 34, height - 26, 14

    def X(t):
        return px0 + min(t, tmax) / tmax * (px1 - px0)

    def Y(v):
        return py0 - min(max(v, 0.0), ymax) / ymax * (py0 - py1)

    pts = [(t, v) for t, v in zip(tr['t'], tr['sensor']) if v is not None]
    pts = pts[::max(1, len(pts) // 420)]
    o = [f'<svg viewBox="0 0 {width} {height}" class="chart ts" role="img" aria-label="{esc(name)}" '
         f'data-name="{esc(name)}" data-x0="{px0}" data-x1="{px1}" data-y0="{py0}" data-y1="{py1}" data-tmax="{tmax}" '
         f'data-ymax="{ymax}" data-t="{esc(json.dumps([round(t, 1) for t, _ in pts]))}" '
         f'data-v="{esc(json.dumps([round(v, 3) for _, v in pts]))}">']
    for t0, t1, cls, label in bands:
        o.append(f'<rect x="{X(t0):.1f}" y="{py1}" width="{max(1.0, X(t1) - X(t0)):.1f}" height="{py0 - py1}" class="band {cls}" '
                 f'data-tip="{esc(label)}: с {num(t0, 0)}-й по {num(t1, 0)}-ю секунду"/>')
    for v in (0, 0.5, 1.0):
        o.append(f'<line x1="{px0}" y1="{Y(v):.1f}" x2="{px1}" y2="{Y(v):.1f}" class="grid"/>')
        o.append(f'<text x="{px0 - 8}" y="{Y(v) + 4:.1f}" class="tick e">{num(v, 1)}</text>')
    for t in range(0, int(tmax) + 1, 20):
        o.append(f'<text x="{X(t):.1f}" y="{py0 + 16}" class="tick">{t} с</text>')
    o.append('<polyline points="' + ' '.join(f'{X(t):.1f},{Y(v):.1f}' for t, v in pts) + f'" class="ln {color}"/>')
    for e in rec.get('events') or []:
        if e.get('type') == 'sample_collected':
            o.append(f'<circle cx="{X(e["t"]):.1f}" cy="{py1 + 1}" r="4" class="sm got" data-tip="Образец собран на {num(e["t"], 0)}-й секунде"/>')
    o.append(f'<line class="cross" x1="0" x2="0" y1="{py1}" y2="{py0}" visibility="hidden"/>'
             f'<circle class="pt {color} crossdot" r="4.5" visibility="hidden"/></svg>')
    return f'<div class="tsname"><i class="dot {color}"></i>{esc(name)}</div>' + ''.join(o)


def intervals(tr, pred):
    out, start = [], None
    for t, m in zip(tr['t'], tr['mode']):
        if pred(m) and start is None:
            start = t
        elif not pred(m) and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, tr['t'][-1]))
    return out


# ---------- куски разметки ----------
def clip(name, caption):
    """Анимация прогона; первый кадр гифки — итоговая картинка, она же остаётся при печати."""
    meta = {c['name']: c for c in json.loads((CLIPS / 'clips.json').read_text(encoding='utf-8'))}[name]
    data = base64.b64encode((CLIPS / f'{name}.gif').read_bytes()).decode('ascii')
    return (f'<figure class="clip"><img src="data:image/gif;base64,{data}" width="{meta["width"]}" '
            f'height="{meta["height"]}" loading="lazy" alt="{esc(caption)}"><figcaption>{caption}</figcaption></figure>')


def qa(items):
    return '<div class="qa">' + ''.join(f'<details><summary>{q}</summary><p>{a}</p></details>' for q, a in items) + '</div>'


def code(text):
    return f'<pre><code>{esc(text.strip())}</code></pre>'


def section(n, anchor, title, tag, body):
    return (f'<section id="{anchor}"><div class="num">Опыт {n}</div><h2>{title}</h2>'
            f'<p class="tag">{tag}</p>{body}</section>')


# =====================================================================================
#  Опыт 1. Адаптивный агент против фиксированного плана
# =====================================================================================
def exp1():
    s = summary('E1')
    rows, facts = [], {}
    for lv in ('easy', 'medium', 'hard'):
        pts = []
        for i, arm in enumerate(('fixed', 'adaptive')):
            st = stat(s, arm, 'score', level=lv)
            sh, rt, bt = (stat(s, arm, m, level=lv)['mean'] for m in ('samples_share', 'returned', 'battery_used'))
            facts[(lv, arm)] = (st['mean'], sh, rt, bt)
            pts.append({'s': i, 'm': st['mean'], 'lo': st['ci'][0], 'hi': st['ci'][1],
                        'val': f"{num(st['mean'])} [{num(st['ci'][0])}; {num(st['ci'][1])}]",
                        'tip': f"{LEVEL[lv].capitalize()} уровень, {'адаптивный' if i else 'фиксированный план'}: счёт "
                               f"{num(st['mean'])}, собрано {pct(sh)}, вернулся в {pct(rt)} прогонов, потрачено {num(bt)} ед. заряда"})
        p, _ = pair(s, 'score', 'adaptive', 'fixed', level=lv)
        rows.append({'label': f'{LEVEL[lv].capitalize()} уровень', 'sub': f'разность {ci_text(p)}', 'pts': pts})
    chart = interval_chart(rows, ['Фиксированный план', 'Адаптивный агент'], 40, 80, [40, 50, 60, 70, 80],
                           head=('Уровень', 'Счёт, среднее и 95% интервал'), aria='Счёт по уровням: фиксированный план и адаптивный агент')
    ph, _ = pair(s, 'score', 'adaptive', 'fixed', level='hard')
    pm, _ = pair(s, 'score', 'adaptive', 'fixed', level='medium')
    pe, _ = pair(s, 'score', 'adaptive', 'fixed', level='easy')
    pb, _ = pair(s, 'battery_used', 'adaptive', 'fixed', level='hard')
    # Сценарий для рисунка: трудный уровень, оба вернулись, разность счёта ближе всего к средней.
    ra, rf = runs_of(s, 'adaptive'), runs_of(s, 'fixed')
    seed = min((k for k in ra if ra[k]['metrics']['returned'] and rf[k]['metrics']['returned']),
               key=lambda k: abs(ra[k]['metrics']['score'] - rf[k]['metrics']['score'] - ph['mean']))
    a, f = load(RUNS / 'E1' / 'adaptive' / f'hard-{seed}.json.gz'), load(RUNS / 'E1' / 'fixed' / f'hard-{seed}.json.gz')
    maps = ('<div class="maps two">' + run_map(f, 'Фиксированный план', result_line(f), color='s1')
            + run_map(a, 'Адаптивный агент', result_line(a), color='s2') + '</div>' + MAP_LEGEND)
    fh, ah = facts[('hard', 'fixed')], facts[('hard', 'adaptive')]
    fm, am = facts[('medium', 'fixed')], facts[('medium', 'adaptive')]
    fe, ae = facts[('easy', 'fixed')], facts[('easy', 'adaptive')]
    body = f'''
<div class="answer"><b>Вопрос.</b> Даёт ли «умная» часть агента что-нибудь по сравнению с роботом, который просто объезжает
арену по заранее заданному маршруту?<br><b>Ответ.</b> Да. На среднем уровне {sgn(pm['mean'])} очка, на трудном {sgn(ph['mean'])}:
адаптивный собирает {pct(am[1], 1)} и {pct(ah[1])} образцов против {pct(fm[1], 1)} и {pct(fh[1])}.</div>

<h3>Как устроен</h3>
<ul>
<li>Два варианта робота, три уровня, по 40 сценариев на уровень (номера 1001–1040): всего {len(s['runs'])} прогонов в быстром симуляторе.</li>
<li>Оба варианта проходят <b>одни и те же</b> сценарии, поэтому сравнивается разность на каждом сценарии, а не два средних.</li>
<li><b>Фиксированный план</b> едет по заранее заданному маршруту вокруг арены и подбирает образцы, рядом с которыми оказался.
Не запоминает ничего и маршрут не меняет.</li>
<li><b>Адаптивный агент</b> по показаниям датчика ведёт карту «где может лежать образец», по расходу батареи учит цену грунта,
запоминает опасные зоны и сам считает, когда пора домой.</li>
</ul>

<h3>Результат</h3>
{chart}
<p class="cap">Точка — средний счёт за 40 сценариев, отрезок — 95% интервал. Под названием уровня — разность «адаптивный минус
фиксированный» на одних и тех же сценариях.</p>

<h3>Как читать</h3>
<ul>
<li><b>Лёгкий уровень: разница мала</b> ({sgn(pe['mean'])}). Три образца фиксированный маршрут почти всегда находит и сам
({pct(fe[1])}). Выигрыш адаптивного здесь — заряд: он тратит {num(ae[3])} единицы против {num(fe[3])}.</li>
<li><b>Средний и трудный: разница большая</b> и интервал далеко от нуля. На трудном уровне адаптивный выиграл в
{ph['a_higher']} сценариях из {ph['n']}, проиграл в {ph['b_higher']}.</li>
<li><b>Возврат на базу одинаковый</b>: на трудном уровне оба вернулись в {pct(ah[2])} прогонов. Этот опыт не про надёжность
возврата.</li>
<li><b>На трудном уровне адаптивный тратит больше заряда</b>: {num(ah[3])} против {num(fh[3])} ({ci_text(pb)}). Он едет за дальними
образцами, фиксированный мимо них просто проезжает. Гипотеза «тратит меньше» на трудном уровне опровергнута, и это записано.</li>
</ul>

<h3>Прогон, который стоит посмотреть</h3>
<p>Трудный уровень, сценарий {seed}: разность счёта здесь ближе всего к средней по опыту. Слева робот едет по заданному кругу,
справа — к местам, на которые указывает датчик.</p>
{clip('e1_plan_vs_adaptive', 'Оба робота едут одновременно по одному сценарию. Образец закрашивается цветом робота, когда тот его собрал; внизу — время миссии и события сценария.')}
<p class="cap">Те же два прогона целиком: наведите курсор на образец или зону.</p>
{maps}

<h3>Что могут спросить</h3>
{qa([
    ('Фиксированный план — не слишком ли слабый соперник?',
     'Это намеренно простая точка отсчёта: робот с навигацией и датчиком, но без памяти и расчётов. С соперниками из условия — '
     'спиралью и подъёмом по сигналу — агент сравнивается в опыте 2. Более сложные устройства агента проверялись в отдельном '
     'исследовании: два лучших перепроверены на 160 новых сценариях и оказались не лучше нынешнего (−0,4 и −2,2 очка, различие не показано).'),
    ('Почему на трудном уровне собрано только четыре пятых образцов?',
     'Не хватает заряда. Батарея — 60 единиц, метр по обычному полу стоит 2,5: это 24 метра на семь образцов и дорогу домой, а часть '
     'образцов лежит за дорогим грунтом. Агент, которому заранее сообщили всё о сценарии, набирает на трудном уровне 92,5 очка — это '
     'потолок при наших правилах; наш набирает около 73.'),
    ('Откуда уверенность, что разница не случайна?',
     f'Интервал разности на трудном уровне — от {num(ph["ci"][0])} до {num(ph["ci"][1])}, ноль в него не попадает. Сценарии итоговой '
     'серии при отладке агента не использовались: отлаживали на номерах 1–80.'),
    ('Это симулятор. А в Gazebo?',
     'Серии идут в быстром симуляторе, потому что один прогон в Gazebo занимает около трёх минут. Что тот же агент ведёт себя в Gazebo '
     'так же, проверяет опыт 6.'),
])}

<h3>Оговорки</h3>
<ul>
<li>Правила подсчёта очков — нашего судьи (<code>did/config.py</code>, класс <code>Rules</code>).</li>
<li>Опыт говорит «весь агент лучше простого маршрута». Какая именно часть агента даёт выигрыш, он не говорит — это опыты 2 и 3.</li>
</ul>
<p class="where">Описание опыта: <code>experiments/E1.yaml</code> · пересчитать: <code>pixi run exp E1</code> · записи: <code>runs/E1</code></p>
<div class="say"><b>Одной фразой для доклада.</b> На {len(s['runs'])} прогонах адаптивный агент набирает на трудном уровне на
{num(ph['mean'], 0)} очков больше, чем робот с заранее заданным маршрутом, и собирает {pct(ah[1])} образцов вместо {pct(fh[1])}.</div>
'''
    return section(1, 'e1', 'Адаптивный агент против фиксированного плана',
                   'Критерий «Работоспособность агента», 30% оценки', body), ph, ah, fh


# =====================================================================================
#  Опыт 2. Способы поиска образцов
# =====================================================================================
def exp2():
    s = summary('E9')
    names = [('adaptive', 'Карта вероятностей', 'основной способ агента'),
             ('adaptive_ig', 'Карта вероятностей + выбор по пользе', 'разведка по ожидаемой пользе измерений'),
             ('gradient', 'Подъём по сигналу', 'едет туда, где показание растёт'),
             ('fixed', 'Заданный маршрут', 'для сравнения'),
             ('spiral', 'Спираль', 'расширяющиеся дуги вокруг базы')]
    rows, share = [], {}
    for arm, label, sub in names:
        st, bt = stat(s, arm, 'samples_share', level='medium'), stat(s, arm, 'battery_used', level='medium')['mean']
        share[arm] = (st['mean'], bt, stat(s, arm, 'samples_share', level='easy')['mean'])
        rows.append({'label': label, 'sub': sub, 'pts': [{
            's': 0, 'm': 100 * st['mean'], 'lo': 100 * st['ci'][0], 'hi': 100 * st['ci'][1],
            'val': f"{pct(st['mean'], 1)} · заряд {num(bt)}",
            'tip': f"{label}: на среднем уровне собрано {pct(st['mean'], 1)} образцов, потрачено {num(bt)} ед. заряда из 60; "
                   f"на лёгком собрано {pct(share[arm][2], 1)}"}]})
    chart = interval_chart(rows, ['Доля собранных образцов'], 50, 100, [50, 60, 70, 80, 90, 100], unit='%',
                           head=('Способ поиска', 'Собрано на среднем уровне · потрачено заряда'), label_w=290,
                           aria='Доля собранных образцов по способам поиска, средний уровень')
    pg, _ = pair(s, 'samples_share', 'adaptive', 'gradient', level='medium')
    ps, _ = pair(s, 'samples_share', 'adaptive', 'spiral', level='medium')
    pbat, _ = pair(s, 'battery_used', 'adaptive', 'gradient', level='medium')
    e18 = summary('E18')
    ph, _ = pair(e18, 'samples_share', 'adaptive', 'gradient', level='hard')
    # Куда не доезжает каждая стратегия: доля собранных ближе и дальше 2,5 м от базы.
    reach = {}
    for arm in ('spiral', 'gradient', 'adaptive'):
        near, far = [0, 0], [0, 0]
        for p in sorted((RUNS / 'E9' / arm).glob('medium-*.json.gz')):
            r = load(p)
            got = collected_set(r)
            for i, sp in enumerate(r['scenario']['samples']):
                b = near if math.dist(sp, BASE) < 2.5 else far
                b[0] += i in got
                b[1] += 1
        reach[arm] = (near[0] / near[1], far[0] / far[1])
    # Сценарий для рисунка: спираль собрала 3 из 5, подъём 4, карта 5 — как в среднем.
    rs = {arm: runs_of(s, arm, level='medium') for arm in ('spiral', 'gradient', 'adaptive')}
    want = {'spiral': 3, 'gradient': 4, 'adaptive': 5}
    seed = min(rs['adaptive'], key=lambda k: (sum(abs(rs[a][k]['metrics']['samples_collected'] - want[a]) for a in want), k))
    recs = {arm: load(RUNS / 'E9' / arm / f'medium-{seed}.json.gz') for arm in want}
    maps = ('<div class="maps three">' + run_map(recs['spiral'], 'Спираль', result_line(recs['spiral']), color='s1')
            + run_map(recs['gradient'], 'Подъём по сигналу', result_line(recs['gradient']), color='s3')
            + run_map(recs['adaptive'], 'Карта вероятностей', result_line(recs['adaptive']), color='s2') + '</div>' + MAP_LEGEND)
    belief = '''
# did/belief.py, смысл обновления. belief — сетка: вероятность «образец лежит в этой клетке».
# Робот в точке (x, y) получил показание z. Датчик показывает 1 вплотную и 0 дальше 2 метров.
f = 1 - dist(cells, (x, y)) / 2.0                 # что показал бы датчик, лежи образец в каждой клетке
like = exp(-0.5 * ((z - f) / sigma) ** 2)         # насколько показание согласуется с этой клеткой
belief = belief * like / (belief * like + (1 - belief) * like_none)
# Клетки на «правильном» расстоянии от робота становятся вероятнее, остальные — менее вероятными.
# Через несколько показаний из разных точек кольца пересекаются, и остаётся одно место.
'''
    body = f'''
<div class="answer"><b>Вопрос.</b> Датчик образцов показывает только «близко или далеко», без направления. Условие называет три
способа искать по такому датчику: поиск максимума, спираль, градиентный подъём. Что лучше и что добавляет наша карта вероятностей?<br>
<b>Ответ.</b> На среднем уровне карта вероятностей собирает {pct(share['adaptive'][0], 1)} образцов, подъём по сигналу —
{pct(share['gradient'][0])}, спираль — {pct(share['spiral'][0], 1)}. И тратит при этом меньше заряда.</div>

<h3>Как устроен</h3>
<ul>
<li>Пять способов поиска, два уровня (лёгкий и средний), по 40 сценариев: всего {len(s['runs'])} прогонов.</li>
<li><b>Спираль</b> и <b>подъём по сигналу</b> — простые стратегии из условия. Решают по самому показанию датчика, карты не ведут
(<code>did/baselines.py</code>).</li>
<li><b>Карта вероятностей</b> хранит, где образец может лежать с учётом всех показаний сразу. Это и есть «поиск максимума», только
не показания, а вероятности.</li>
<li>Пятый вариант проверяет более сложную идею: выбирать точку разведки по ожидаемой пользе будущих измерений.</li>
</ul>
{code(belief)}

<h3>Результат</h3>
{chart}
<p class="cap">Средний уровень, 5 образцов на сценарий. Точка — средняя доля собранных, отрезок — 95% интервал. Справа — сколько
заряда потрачено из 60.</p>

<h3>Как читать</h3>
<ul>
<li><b>Карта вероятностей против подъёма по сигналу:</b> {ci_text(pg, 0, 100)} процентных пунктов собранных образцов и
{ci_text(pbat)} единицы заряда. Больше собрано и дешевле.</li>
<li><b>Спираль хуже всех</b> ({ci_text(ps, 0, 100)} п. п. в пользу карты). По записям прогонов видно почему: образцы ближе 2,5 м
от базы она собирает в {pct(reach['spiral'][0])} случаев, а дальние — только в {pct(reach['spiral'][1])}. Заряд кончается раньше,
чем спираль доходит до края арены.</li>
<li><b>Сложная идея не помогла:</b> выбор точки разведки по ожидаемой пользе даёт {pct(share['adaptive_ig'][0], 1)} против
{pct(share['adaptive'][0], 1)} — различие не показано. Оставлен простой вариант.</li>
<li><b>На лёгком уровне разницы с подъёмом почти нет</b> ({pct(share['adaptive'][2])} против {pct(share['gradient'][2])}): когда
образцов три и они рядом, хватает и простого правила.</li>
</ul>

<h3>Прогон, который стоит посмотреть</h3>
<p>Средний уровень, сценарий {seed}, один и тот же для трёх способов.</p>
{clip('e9_search', f"Спираль собрала {recs['spiral']['result']['samples_collected']} образца из {recs['spiral']['result']['samples_total']}, подъём по сигналу — {recs['gradient']['result']['samples_collected']}, карта вероятностей — {recs['adaptive']['result']['samples_collected']}.")}
{maps}

<h3>Что могут спросить</h3>
{qa([
    ('Что у вас считается «поиском максимума»?',
     'Подъём по сигналу — это и есть поиск максимума показаний: едем туда, где сигнал растёт. Карта вероятностей решает ту же задачу '
     'иначе: одно показание задаёт кольцо возможных мест вокруг робота, несколько показаний из разных точек оставляют одно место.'),
    ('Почему нет трудного уровня?',
     'В этом опыте его нет. Подъём по сигналу на трудном уровне есть в другом опыте (с изменёнными правилами судьи): там карта '
     f'вероятностей собирает на {num(100 * ph["mean"], 0)} процентных пунктов больше '
     f'(интервал от {num(100 * ph["ci"][0], 0)} до {num(100 * ph["ci"][1], 0)}). Спираль на трудном уровне не проверялась.'),
    ('Зачем показывать вариант, который ничего не дал?',
     'Потому что это результат: более сложный метод не обязан быть лучше. Мы его реализовали, измерили и оставили простой.'),
    ('Датчик у судьи может быть устроен иначе. Что тогда?',
     'Карта вероятностей опирается на формулу датчика «1 вплотную, 0 дальше двух метров, линейно». Если закон другой, она ошибается: '
     'при законе «по корню» агент в наших проверках собирал около 1% образцов. Правка, которая определяет закон по первому найденному '
     'образцу, написана, но ревью не прошла и в основной код не влита.'),
])}

<h3>Оговорки</h3>
<ul>
<li>Только лёгкий и средний уровни; изменений среды в этих прогонах нет.</li>
<li>Карта вероятностей знает формулу датчика заранее; простые стратегии от неё не зависят.</li>
</ul>
<p class="where">Описание: <code>experiments/E9.yaml</code> · пересчитать: <code>pixi run exp E9</code> · код: <code>did/belief.py</code>,
<code>did/baselines.py</code></p>
<div class="say"><b>Одной фразой для доклада.</b> Из трёх способов поиска, названных в условии, мы реализовали все и сравнили на
{len(s['runs'])} прогонах: карта вероятностей собирает почти все образцы, подъём по сигналу — четыре из пяти, спираль — два из трёх.</div>
'''
    return section(2, 'e2', 'Способы поиска образцов по датчику без направления',
                   'Уровень 3 «Научный цикл»: «поиск максимума, спираль, градиентный подъём» названы в условии', body), share


# =====================================================================================
#  Опыт 3. Адаптация: что из неё даёт очки
# =====================================================================================
def exp3():
    e2, e15, e19, e22 = summary('E2'), summary('E15'), summary('E19s'), summary('E22b', P1_RUNS)
    chk = json.loads((RUNS / '_showcase' / 'p1_check.json').read_text(encoding='utf-8'))
    pv = chk['pair']['score']

    def row(label, sub, p, extra='', d=1):
        muted = p['ci'][0] <= 0 <= p['ci'][1]
        return {'label': label, 'sub': sub, 'pts': [{
            's': 0, 'm': p['mean'], 'lo': p['ci'][0], 'hi': p['ci'][1], 'muted': muted,
            'val': ci_text(p, d) + (' · не показано' if muted else ''),
            'tip': f"{label}: {ci_text(p, 2)} очка на {p['n']} сценариях трудного уровня. "
                   + ('Интервал включает ноль: различие не показано.' if muted else 'Интервал выше нуля: подтверждено.') + extra}]}

    rows = [row('Пережидание сбоя датчика', 'новый агент против прежнего', pv)]
    for arm, label in (('no_sensor_health', 'Контроль исправности датчика'), ('static_reserve', 'Расчётный запас на возврат'),
                       ('no_soil', 'Обучение цене грунта'), ('no_hazard', 'Память об опасных зонах'),
                       ('no_change', 'Обнаружение изменений среды')):
        p, _ = pair(e2, 'score', 'adaptive', arm)
        rows.append(row(label, 'полный агент против агента без неё', p))
    pr, _ = pair(e19, 'score', 'adaptive_fresh', 'no_change', cond='route')
    rows.append(row('Смена грунта прямо на пути', 'выигрыш от обнаружения, отдельный опыт', pr, d=2))
    chart_a = interval_chart(rows, ['Прибавка к счёту'], -6, 12, [-6, -3, 0, 3, 6, 9, 12], zero=0,
                             head=('Способность агента', 'Прибавка к счёту, очков'), label_w=290, value_w=215, width=860,
                             aria='Сколько очков даёт каждая способность агента на трудном уровне')
    rows_b = []
    for c in e15['spec']['conditions']:
        p, _ = pair(e15, 'score', 'adaptive', 'no_change', cond=c['id'])
        rows_b.append(row(c['label'], None, p))
    chart_b = interval_chart(rows_b, ['Прибавка к счёту'], -6, 12, [-6, -3, 0, 3, 6, 9, 12], zero=0,
                             head=('Сила изменений в сценарии', 'Обнаружение изменений: прибавка к счёту'), label_w=290, value_w=215,
                             width=860, aria='Прибавка от обнаружения изменений при разной силе изменений')
    rows_c = [row('Наши правила', 'сбой 25–40 секунд', pv)]
    for cid, label, sub in (('science', 'Научные правила', 'несколько причин расхода'), ('idle5', 'Простой в 5 раз дороже', None),
                            ('idle10', 'Простой в 10 раз дороже', None), ('long_fault', 'Сбой 90–120 секунд', None),
                            ('hz2', 'Датчик 2 показания в секунду', 'вместо пяти'), ('no_fault', 'Сбоев нет вовсе', None)):
        p, _ = pair(e22, 'score', 'adaptive_v2', 'adaptive', cond=cid)
        r = row(label, sub, p)
        if cid == 'no_fault':
            r['pts'][0].update(muted=True, val='0,0 — прогоны совпадают')
        rows_c.append(r)
    chart_c = interval_chart(rows_c, ['Прибавка к счёту'], -6, 14, [-6, -3, 0, 3, 6, 9, 12], zero=0,
                             head=('Условие', 'Новый агент минус прежний, очков'), label_w=290, value_w=215, width=860,
                             aria='Прибавка от пережидания сбоя при других правилах судьи')
    pmed, _ = pair(e22, 'score', 'adaptive_v2', 'adaptive', cond='idle10', level='medium')
    po, _ = pair(e19, 'score', 'oracle', 'no_change', cond='route')
    ph3, _ = pair(e15, 'score', 'adaptive', 'no_hazard', cond='hazard_x3')
    e3 = summary('E3')
    p_none, _ = pair(e3, 'score', 'adaptive', 'fixed', cond='none')
    p_soil, _ = pair(e3, 'score', 'adaptive', 'fixed', cond='soil')
    # Сценарий для рисунка: сбой «шум», прежний агент недобрал образец, новый переждал.
    seed = 8024
    a = load(RUNS / '_showcase' / 'adaptive' / f'hard-{seed}.json.gz')
    v = load(RUNS / '_showcase' / 'adaptive_v2' / f'hard-{seed}.json.gz')
    ev = next(e for e in a['scenario']['events'] if e['type'] == 'sensor_fault')
    f0, f1 = ev['t'], ev['t'] + ev['duration']
    wi = v['modes'].index('wait') if 'wait' in v['modes'] else -1
    waits = intervals(v['track'], lambda m: m == wi)
    tmax = 20 * math.ceil(max(a['track']['t'][-1], v['track']['t'][-1]) / 20)
    vals = [x for x in a['track']['sensor'] + v['track']['sensor'] if x is not None]
    ymax = max(1.0, round(max(vals) + 0.05, 1))
    ts = (timeline(a, 'Прежний агент: едет по показаниям неисправного датчика', 's1', tmax, ymax, [(f0, f1, 'fault', 'датчик неисправен')])
          + timeline(v, 'Новый агент: заметил сбой, стоит и ждёт', 's2', tmax, ymax,
                     [(f0, f1, 'fault', 'датчик неисправен')] + [(w0, w1, 'wait', 'робот стоит и ждёт') for w0, w1 in waits])
          + '<div class="legend"><span class="key"><i class="sq fault"></i>датчик неисправен</span>'
            '<span class="key"><i class="sq waitb"></i>робот стоит и ждёт</span>'
            '<span class="key"><i class="dot got"></i>образец собран</span></div>')
    dist_a = sum(math.dist((a['track']['x'][i], a['track']['y'][i]), (a['track']['x'][i + 1], a['track']['y'][i + 1]))
                 for i in range(len(a['track']['t']) - 1) if f0 <= a['track']['t'][i] <= f1)
    maps = ('<div class="maps two">'
            + run_map(a, 'Прежний агент', result_line(a), color='s1', highlight=(f0, f1))
            + run_map(v, 'Новый агент', result_line(v), color='s2', highlight=(f0, f1)) + '</div>'
            + '<div class="legend"><span class="key"><i class="ln hl"></i>путь за время сбоя датчика</span></div>' + MAP_LEGEND)
    wait_s = sum(w1 - w0 for w0, w1 in waits)
    guard = '''
# did/sensorguard.py, смысл решения (не дословный код)
idle  = расход_стоя()           # измерен по своей батарее, а не взят из правил: около 0,01 ед./с
blind = расход_в_езде() * 0.5   # езда по неисправному датчику: около половины пути впустую, 0,25 ед./с

ждать = (датчик_неисправен and idle < blind      # стоять дешевле, чем ехать вслепую
         and простоял < 120                      # не дольше двух минут за прогон
         and потрачено_на_простой < 3            # не больше трёх единиц заряда
         and заряда_хватает_домой)               # иначе — домой, не дожидаясь датчика
'''
    means = chk['means']
    body = f'''
<div class="answer"><b>Вопрос.</b> На трудном уровне среда меняется по ходу прогона: грунт дорожает, появляется новая опасная
зона, датчик образцов на полминуты выходит из строя. На что из этого агенту выгодно реагировать?<br>
<b>Ответ.</b> На сбой датчика — да: агент, который его пережидает, набирает на {num(pv['mean'])} очка больше. Реакция на смену
грунта и новую зону работает, но очков почти не добавляет. Мы показываем и то, и другое.</div>

<h3>Как устроен</h3>
<p>Здесь не один опыт, а три вида измерений, и все они отвечают на один вопрос: «сколько очков даёт эта способность».</p>
<ul>
<li><b>Отключение по одному.</b> Берём полного агента и агента, у которого выключена одна способность. Разность счёта на одних и тех
же 60 сценариях — цена этой способности.</li>
<li><b>Сила изменений.</b> Генератор сценариев умеет делать изменения слабее и сильнее: без событий, одна смена грунта и одна зона,
по две, по три, только грунт, только зоны. На каждом варианте сравниваем агента с обнаружением изменений и без него.</li>
<li><b>Новый агент против прежнего.</b> Прежний при сбое датчика продолжает ездить за образцами, которых нет. Новый останавливается
и ждёт. Сравнение — на 40 сценариях (номера 8001–8040), которых при отладке никто не видел, и ещё при шести вариантах правил судьи.</li>
</ul>

<h3>Результат: что даёт каждая способность</h3>
{chart_a}
<p class="cap">Трудный уровень. Точка — прибавка к счёту, отрезок — 95% интервал. Серым показаны строки, где интервал включает ноль:
про них можно сказать только «различие не показано».</p>

<h3>Как читать</h3>
<ul>
<li><b>Платит всё, что связано с датчиком и зарядом:</b> пережидание сбоя, контроль исправности датчика, расчёт запаса на дорогу
домой.</li>
<li><b>Не видно выигрыша от трёх способностей,</b> которые отвечают за изменения среды: обучение цене грунта, память об опасных
зонах, обнаружение изменений. Интервалы у них широкие и накрывают ноль.</li>
<li><b>«Различие не показано» не значит «разницы нет».</b> Это значит, что на таком числе сценариев она не видна. Если она есть, то
не больше нескольких очков.</li>
</ul>

<h3>Проверка: может, изменения слишком слабые?</h3>
{chart_b}
<p class="cap">Агент с обнаружением изменений против такого же без него, при шести вариантах сценариев. По 40 сценариев на вариант.</p>
<p>Нет: усиление изменений втрое картину не меняет. Единственное, что стало заметно, — память об опасных зонах при трёх зонах
вместо одной: {ci_text(ph3)} очка. Отдельный опыт, где грунт меняется прямо на пути робота, дал {ci_text(pr, 2)}. Там же посчитан
потолок: агент, которому новую карту грунтов сообщают заранее, выигрывает {ci_text(po)} очка. То есть даже идеальная реакция на
смену грунта в наших сценариях стоит около очка.</p>

<h3>Что платит: пережидание сбоя датчика</h3>
<p>Раз за прогон датчик образцов на 25–40 секунд начинает шуметь, залипать или занижать показания. Прежний агент замечал это, но
продолжал ездить. Новый сравнивает, что дешевле: стоять или ехать вслепую.</p>
{code(guard)}
<p>На 40 новых сценариях трудного уровня счёт вырос с {num(means['adaptive']['score'])} до {num(means['adaptive_v2']['score'])}:
<b>{ci_text(pv)}</b>. Собрано {pct(means['adaptive_v2']['samples_share'], 1)} образцов вместо {pct(means['adaptive']['samples_share'])},
возврат на базу {pct(means['adaptive_v2']['returned'])} против {pct(means['adaptive']['returned'])}. Плата — время: прогон длиннее
на {num(chk['pair']['time']['mean'], 0)} секунд. Это число пересчитано на текущем коде 8 октября и совпало с отчётом исследования.</p>

<h3>Прогон, который стоит посмотреть</h3>
<p>Трудный уровень, сценарий {seed}. На {num(f0, 0)}-й секунде датчик начинает шуметь и шумит {num(ev['duration'], 0)} секунд.
Прежний агент за это время проехал {num(dist_a)} м по ложным показаниям. Новый простоял {num(wait_s, 0)} секунд и поехал дальше, когда
шум прекратился.</p>
{clip('p1_sensor_fault', 'Чёрным — путь за время сбоя датчика. Прежний агент едет по ложным показаниям; новый стоит (жёлтое кольцо) и едет дальше, когда датчик восстановился.')}
{ts}
<p class="cap">Показание датчика образцов по времени: 1 — образец вплотную, 0 — дальше двух метров. Кружки сверху — моменты сбора
образца. Наведите курсор, чтобы увидеть значение.</p>
{maps}

<h3>Не подогнано ли под наши правила?</h3>
{chart_c}
<p class="cap">Новый агент минус прежний, трудный уровень, те же 40 сценариев при других правилах судьи. Агент о смене правил не знает.</p>
<p>Выигрыш держится, когда простой стоит впятеро и вдесятеро дороже, когда сбой втрое длиннее и когда датчик работает реже. Если
сбоев нет, новый агент ведёт себя в точности как прежний. Нашлось одно условие, где он уступает: средний уровень при простое
вдесятеро дороже, {ci_text(pmed)} очка.</p>

<h3>Что могут спросить</h3>
{qa([
    ('Почему реакция на смену грунта не даёт очков?',
     'По двум причинам. Первая измерена: даже агент, которому новую карту грунтов сообщают заранее, выигрывает около очка — грунт '
     'меняется на участках, через которые маршрут проходит редко, и объезд экономит мало. Вторая следует из устройства агента: цену '
     'грунта он и так постоянно переоценивает по расходу батареи, поэтому отдельная «тревога об изменении» добавляет немного.'),
    ('Тогда зачем эти механизмы в агенте?',
     'Ради объяснимости и на случай сильных изменений. Гипотезы агента о грунте верны в 99% случаев (опыт 5), а память об опасных зонах '
     f'при трёх зонах уже даёт {num(ph3["mean"])} очка. Но приписывать им выигрыш в обычных сценариях мы не будем.'),
    ('У вас был опыт «Неожиданности по одной». Почему его нет?',
     f'Потому что он не про адаптацию, хотя так называется. Адаптивный агент обгоняет фиксированный план на {num(p_soil["mean"])} очка '
     f'при смене грунта — и на {num(p_none["mean"])} вообще без событий. Преимущество одинаковое, значит, его даёт поиск по датчику, '
     'а не реакция на событие.'),
    ('А если сбой датчика не закончится?',
     'Агент ждёт не дольше двух минут за прогон и тратит на простой не больше трёх единиц заряда; если пора домой — едет домой. Но при '
     'сбое длиннее двух минут он возвращается реже прежнего: в проверке ревьюера 16 раз из 20 против 19. В наших сценариях сбой длится '
     '25–40 секунд.'),
    ('Это проверено в Gazebo?',
     'Да, пять прогонов трудного уровня 8 октября: во всех робот заметил сбой, простоял 33–37 секунд, потратил на это 0,33–0,47 единицы '
     'заряда, вернулся на базу и ничего не задел. Это проверка работоспособности, а не измерение выигрыша: прежний агент рядом не запускался.'),
])}

<h3>Оговорки</h3>
<ul>
<li>Все серии — в быстром симуляторе.</li>
<li>В графике «что даёт каждая способность» первая строка измерена на других сценариях (8001–8040), чем остальные (1001–1060).</li>
<li>Интервалы посчитаны для каждой строки отдельно, без поправки на то, что строк много.</li>
<li>Сторож видит шум и залипание датчика. Сбой «показания занижены» стоя не заметен, его агент не пережидает.</li>
</ul>
<p class="where">Отчёты: <code>research/findings/P1.md</code>, <code>R10.md</code> · описания: <code>experiments/E2.yaml</code>,
<code>E15.yaml</code>, <code>E19s.yaml</code>, <code>E22b.yaml</code> · код: <code>did/sensorguard.py</code> · перепроверка числа:
<code>pixi run python -m tools.showcase_p1_check</code></p>
<div class="say"><b>Одной фразой для доклада.</b> Мы измерили, какая адаптация приносит очки: пережидание сбоя датчика даёт
{sgn(pv['mean'])} на трудном уровне, а реакция на смену грунта в наших сценариях — около нуля, и мы это не скрываем.</div>
'''
    return section(3, 'e3', 'Адаптация: что из неё даёт очки',
                   'Критерий «Адаптивность», 25% оценки: «демонстрация генерации сценариев и адаптивности агента»', body), pv


# =====================================================================================
#  Опыт 4. Миссия, заданная словами
# =====================================================================================
MISSIONS = [
    ('M0', '«Собери как можно больше образцов, избегай дорогих зон, вернись на базу до разрядки»', '5 из 6', '5 из 6', '6 из 6'),
    ('M1', '«Собери ровно два образца и сразу возвращайся на базу»', '0 из 6', '6 из 6', '6 из 6'),
    ('M2', '«Заряд важнее образцов: вернись, сохранив не меньше 30 единиц заряда»', '0 из 6', '5 из 6', '5 из 6'),
    ('M3', '«Не заезжай в правую половину арены (x больше 0,5 м)»', '0 из 6', '1 из 6', '5 из 6'),
]


def exp4():
    m = R13_RUNS / 'missions'
    r_rule, r_llm = load(m / 'M1_rule' / 'hard-1001.json.gz'), load(m / 'M1_llm' / 'hard-1001.json.gz')
    b_rule, b_llm = load(m / 'M3_rule' / 'hard-1002.json.gz'), load(m / 'M3_llm_ask' / 'hard-1002.json.gz')
    maps1 = ('<div class="maps two">' + run_map(r_rule, 'Правило: текста не читает', result_line(r_rule), color='s1')
             + run_map(r_llm, 'Модель: «ровно два образца»', result_line(r_llm), color='s2') + '</div>')
    maps3 = ('<div class="maps two">'
             + run_map(b_rule, 'Правило: текста не читает', result_line(b_rule) + f' · дальше всего вправо: x = {num(max(b_rule["track"]["x"]), 2)} м', color='s1', vline=0.5)
             + run_map(b_llm, 'Модель: «не заезжай правее 0,5 м»', result_line(b_llm) + f' · дальше всего вправо: x = {num(max(b_llm["track"]["x"]), 2)} м', color='s2', vline=0.5)
             + '</div><div class="legend"><span class="key"><i class="ln ban"></i>граница из текста миссии</span></div>' + MAP_LEGEND)
    call = r_llm['llm'][-1]
    req = call['request']
    get = lambda k: (re.search(rf'"{k}": ("[^"]*"|[\d.]+)', req) or [None, '?'])[1].strip('"')  # noqa: E731
    col = re.search(r'"collected": (\d+), "total": (\d+)', req)
    ans = json.loads(call['response'])
    dialog = f'''
<div class="dialog">
<div class="turn"><div class="who">Робот → модель, {num(call['t'], 1)}-я секунда</div>
<pre><code>{esc(json.dumps({'mission': get('mission'), 'trigger': get('trigger'), 'time_s': float(get('time_s')), 'battery': float(get('battery')),
                         'samples': {'collected': int(col[1]), 'total': int(col[2])}, '…': 'кандидаты, точки разведки, цена дороги домой'},
                        ensure_ascii=False, indent=1))}</code></pre></div>
<div class="turn"><div class="who">Модель → робот ({esc(r_llm['agent']['config'].get('llm_model') or 'qwen3.8-flash-next')})</div>
<pre><code>{esc(json.dumps(ans, ensure_ascii=False, indent=1))}</code></pre></div>
</div>'''
    rows = ''.join(f'<tr><td><b>{k}</b></td><td>{t}</td><td class="c">{a}</td><td class="c">{b}</td><td class="c">{c}</td></tr>'
                   for k, t, a, b, c in MISSIONS)
    body = f'''
<div class="answer"><b>Вопрос.</b> Условие требует, чтобы языковая модель по тексту миссии и состоянию робота составляла список
подцелей. Меняется ли поведение робота, если поменять только текст?<br>
<b>Ответ.</b> Да. Миссию «собери ровно два образца и возвращайся» робот с моделью выполнил в 6 прогонах из 6. Робот с правилом
вместо модели — ни разу: правило текста не читает.</div>

<h3>Как устроен</h3>
<ul>
<li>Модель стоит <b>только на верхнем уровне</b>: получает состояние (заряд, сколько собрано, кандидаты, цена дороги домой) и текст
миссии, отвечает списком подцелей. Ехать к цели, объезжать столбы, собирать — дело обычного кода, десять раз в секунду.</li>
<li>Ответ модели — JSON по схеме: рассуждение, гипотезы, подцели. Код его проверяет: такая цель существует, заряда на неё хватает.
Если ответ не прошёл проверку, модель спрашивают повторно с текстом ошибки, а потом решает правило.</li>
<li>Четыре миссии, у каждой свой критерий «выполнено», записанный до прогонов. Шесть сценариев: по три среднего и трудного уровня.</li>
<li>Модель — <code>qwen3.8-flash-next</code> через сервер МАИ. «Правило» — тот же агент без модели.</li>
</ul>

<h3>Результат</h3>
<table class="missions"><thead><tr><th></th><th>Текст миссии</th><th>Правило</th><th>Модель</th><th>Модель на каждый повод</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="cap">Сколько прогонов из шести выполнили миссию. «Модель на каждый повод» — режим, в котором правило не подменяет модель
ни в одном решении.</p>

<h3>Как читать</h3>
<ul>
<li><b>Обычная миссия (M0): модель и правило равны.</b> По очкам тоже: 71,3 против 71,2. Модель не делает робота результативнее.</li>
<li><b>Миссии с ограничением (M1, M2): правило не может, модель может.</b> Это и есть её место: поведение меняется от текста, без
новой строки кода.</li>
<li><b>M3 сорвалась в обычном режиме (1 из 6).</b> Чтобы не ждать модель на каждое решение, на треть поводов отвечает правило, а оно
запрета не знает. Когда на каждый повод отвечает модель — 5 из 6.</li>
<li><b>Выполнение миссии стоит очков.</b> В M1 робот набирает 44 очка вместо 71: он и должен был собрать только два образца.</li>
</ul>

<h3>Прогон, который стоит посмотреть</h3>
<p>Миссия M1, трудный уровень, сценарий 1001. Слева правило собирает всё, до чего дотянется. Справа модель останавливает сбор после
второго образца.</p>
{clip('r13_two_samples', 'Один сценарий, одно задание. Робот на правиле продолжает собирать; робот с моделью после второго образца поворачивает на базу.')}
{maps1}
<p>Вот обмен, после которого робот повернул домой. Это настоящий запрос и настоящий ответ из записи прогона.</p>
{dialog}
<p>Миссия M3, трудный уровень, сценарий 1002, модель отвечает на каждый повод. Робот ни разу не пересёк границу.</p>
{clip('r13_boundary', 'Красная черта — граница из текста задания. Робот на правиле её пересекает, робот с моделью — нет.')}
{maps3}
<p>Все настоящие вызовы модели с полными текстами — на странице <a href="../llm_calls.html">llm_calls.html</a>.</p>

<h3>Что могут спросить</h3>
{qa([
    ('Модель делает робота лучше?',
     'По очкам обычной миссии — нет, счёт тот же, что у правила. Модель нужна там, где задание меняют словами. Говорить «с LLM робот '
     'собирает больше» нельзя.'),
    ('Что будет, если модель ответит чепуху или не ответит?',
     'Ответ проверяется по схеме и по состоянию робота. Не прошёл — повторный запрос с текстом ошибки; не помогло — решает правило. '
     'Пример неверного ответа и его исправления есть на странице вызовов. Долгий ответ тоже опасен: ожидание модели по 15 секунд на '
     'решение стоит 73 очка на трудном уровне, поэтому на показе модель идёт в записи.'),
    ('Модель управляет колёсами?',
     'Нет. Она выбирает, что делать дальше: разведать участок, подъехать к кандидату, вернуться. Путь по сетке, скорость и поворот '
     'считает обычный код. Так и требует условие: модель только на верхнем уровне.'),
    ('Шесть прогонов — это мало.',
     'Да. Это счёт прогонов, а не доля. Расширенная серия (48 прогонов на модель, ещё две модели) посчитана: 45 и 46 из 48, правило — '
     '0 из 48. Но она ещё не прошла ревью, и на слайды эти числа пока не идут.'),
    ('Какие модели вы пробовали?',
     'Шесть моделей с сервера МАИ на шести сценариях. По счёту обычной миссии все шесть равны правилу: в 96–98% решений они выбирают ту '
     'же цель. Годные по формату планы в 98–100% случаев дают четыре модели из шести. Рабочая — qwen3.8-flash-next: 100% годных '
     'планов. Локальные модели на ноутбуке и платные сервисы в итоговых числах не участвуют.'),
])}

<h3>Оговорки</h3>
<ul>
<li>Шесть сценариев, по одному прогону на сценарий; ответы модели при повторе берутся из кэша.</li>
<li>Не показано, что модель лучше правила, в которое такое же ограничение записали бы руками. Показано одно: поведение меняется по
тексту без нового кода.</li>
<li>Прогоны сделаны до последних правок агента (защита от потери положения, пережидание сбоя) и не пересчитывались.</li>
</ul>
<p class="where">Отчёт: <code>research/findings/R13.md</code> · проверка миссий: <code>tools/mission_eval.py</code>,
<code>did/mission_criteria.py</code> · подключение модели: <code>did/llm.py</code></p>
<div class="say"><b>Одной фразой для доклада.</b> Меняем один текст миссии — «собери ровно два образца» — и тот же робот без новой
строки кода выполняет её 6 раз из 6; робот без модели — ни разу.</div>
'''
    return section(4, 'e4', 'Миссия, заданная словами',
                   'Критерий «Качество LLM-планирования и промптов», 15% оценки; уровень 2 «LLM-планировщик»', body)


# =====================================================================================
#  Опыт 5. Журнал гипотез против скрытой правды
# =====================================================================================
def _pooled(runs, kind, status, rng):
    """Доля верных среди гипотез вида kind со статусом status по всем прогонам; интервал — бутстрепом по прогонам."""
    c = np.array([[((r['metrics'].get('hypotheses_truth') or {}).get('by_kind') or {}).get(kind, {}).get('confusion', {}).get(status, {}).get(k, 0)
                   for k in ('correct', 'wrong')] for r in runs], dtype=float)
    tot = c.sum(axis=0)
    idx = rng.integers(0, len(c), size=(2000, len(c)))
    boot = c[idx].sum(axis=1)
    share = boot[:, 0] / np.maximum(1.0, boot.sum(axis=1))
    return tot[0] / max(1.0, tot.sum()), float(np.percentile(share, 2.5)), float(np.percentile(share, 97.5)), int(tot[0]), int(tot.sum())


def exp5():
    s = summary('E17')
    rng = np.random.default_rng(0)
    arms = [('adaptive', 'Адаптивный агент'), ('scientist', 'Исследователь')]
    conds = [('base', 'базовые правила'), ('science', 'научные правила')]
    grab = {(a, c): [r for r in s['runs'] if r['arm'] == a and r.get('condition', 'base') == c] for a, _ in arms for c, _ in conds}
    rows, val = [], {}
    for kind, klabel in (('sample', '«Здесь лежит образец»'), ('soil', '«Здесь грунт дороже»')):
        for c, clabel in conds:
            pts = []
            for i, (a, alabel) in enumerate(arms):
                m, lo, hi, ok, n = _pooled(grab[(a, c)], kind, 'confirmed', rng)
                val[(kind, c, a)] = (m, ok, n)
                pts.append({'s': i, 'm': 100 * m, 'lo': 100 * lo, 'hi': 100 * hi, 'val': f'{pct(m)} · {ok} из {n}',
                            'tip': f'{alabel}, {clabel}: из {n} подтверждённых гипотез {klabel} верны {ok} ({pct(m, 1)})'})
            rows.append({'label': klabel, 'sub': clabel, 'pts': pts})
    chart = interval_chart(rows, [n for _, n in arms], 50, 100, [50, 60, 70, 80, 90, 100], unit='%',
                           head=('Гипотеза и правила', 'Верны среди подтверждённых'), aria='Доля верных среди подтверждённых гипотез')
    ref = {(a, c): _pooled(grab[(a, c)], 'sample', 'refuted', rng) for a, _ in arms for c, _ in conds}
    inq = {}
    for c, _ in conds:
        t = {k: sum((r['metrics'].get('inquiries') or {}).get(k, 0) for r in grab[('scientist', c)])
             for k in ('total', 'identified', 'insufficient', 'correct', 'partial', 'wrong', 'unverifiable')}
        inq[c] = t
    err = {(a, c): stat(s, a, 'hyp_soil_error', cond=c, level='all')['mean'] for a, _ in arms for c, _ in conds}
    score = {(a, c): stat(s, a, 'score', cond=c, level='all')['mean'] for a, _ in arms for c, _ in conds}
    # Журнал: исследователь, научные правила, трудный уровень, сценарий 1001 — первые полторы минуты.
    r = load(RUNS / 'E17' / 'scientist@science' / 'hard-1001.json.gz')
    truth = {h['id']: h.get('truth_verdict') for h in r['hypotheses']}
    mark = {'correct': '<span class="tv ok">по скрытой правде: верно</span>', 'wrong': '<span class="tv bad">по скрытой правде: неверно</span>',
            'unverifiable': '<span class="tv na">сверить не с чем</span>'}
    lines = []
    for e in r['journal']:
        if e['kind'] not in ('inquiry', 'hypothesis', 'verdict') or not 19 <= e['t'] <= 62:
            continue
        kind = {'inquiry': 'расследование', 'hypothesis': 'гипотеза', 'verdict': 'вывод'}[e['kind']]
        hid = re.match(r'(H\d+)', e['text'])
        extra = ' ' + mark.get(truth.get(hid[1]), '') if hid and e['kind'] == 'hypothesis' else ''
        lines.append(f'<li class="k-{e["kind"]}"><span class="t">{num(e["t"], 1)} с</span><span class="k">{kind}</span>'
                     f'<span class="x">{esc(e["text"])}{extra}</span></li>')
    journal = '<ol class="journal">' + ''.join(lines) + '</ol>'
    bayes = '''
# did/inquiry.py и did/science.py, смысл одного расследования
странность:  расход 5,6 ед./м при прогнозе 2,7
объяснения:  дорогой грунт 73%, утечка заряда 15%, другое 12%        # вероятности до опыта
опыты:       постоять 2 с; проехать 30 см по прямой; развернуться на месте
выбор:       опыт с наибольшим отношением «польза / цена»;
             польза — на сколько бит он уменьшит неопределённость, цена — заряд
измерение:   расход стоя 0,01 ед./с                                  # при утечке было бы 0,61
пересчёт:    дорогой грунт 100%, утечка 0%, другое 0%                 # формула Байеса
действие:    участок внесён в карту стоимостей, маршрут строится в объезд
'''
    b, sc_ = inq['base'], inq['science']
    body = f'''
<div class="answer"><b>Вопрос.</b> Робот пишет в журнал «гипотеза подтверждена». А она на самом деле верна?<br>
<b>Ответ.</b> Чаще всего да: гипотезы о грунте, которые робот подтвердил, верны в {pct(val[('soil', 'base', 'adaptive')][0])} случаев,
об образцах — в {pct(val[('sample', 'base', 'adaptive')][0])}. Когда у одного симптома несколько причин, обычный агент ошибается
про грунт в четырёх случаях из десяти, а исследователь — в двух из ста. Слабое место — поспешные опровержения.</div>

<h3>Как устроен</h3>
<ul>
<li>Каждая гипотеза в журнале имеет вид, проверку и статус: открыта, подтверждена, опровергнута, устарела.</li>
<li>После прогона отдельный код сверяет каждую гипотезу со <b>скрытой правдой</b> сценария: где на самом деле лежали образцы, какие
были грунты и зоны. Агент во время прогона этого не видит.</li>
<li>Два агента, два набора правил, два уровня, по 40 сценариев: {len(s['runs'])} прогонов.</li>
<li><b>Научные правила</b> — усложнённый судья: заряд уходит ещё и на повороты и на груз, батарея шумит, опасная зона вызывает
сбой. Высокий расход теперь может значить и дорогой грунт, и утечку, и крутой поворот.</li>
<li><b>Исследователь</b> — адаптивный агент с расследованиями: заметив странность, он перечисляет объяснения с вероятностями,
выбирает опыт, который их различит, измеряет и пересчитывает вероятности.</li>
</ul>
{code(bayes)}

<h3>Результат</h3>
{chart}
<p class="cap">Какая доля гипотез, помеченных роботом «подтверждена», верна по скрытой правде. Отрезок — 95% интервал.</p>

<h3>Как читать</h3>
<ul>
<li><b>При базовых правилах агенты равны.</b> Причина высокого расхода там одна — грунт, расследовать нечего.</li>
<li><b>При научных правилах они расходятся на гипотезах о грунте:</b> {pct(val[('soil', 'science', 'adaptive')][0])} у адаптивного
против {pct(val[('soil', 'science', 'scientist')][0])} у исследователя. Адаптивный любой лишний расход списывает на грунт.
Исследователь сначала проверяет другие объяснения. Ошибка в оценке «во сколько раз грунт дороже» — {num(err[('adaptive', 'science')], 2)}
против {num(err[('scientist', 'science')], 2)}.</li>
<li><b>На счёте это не сказывается:</b> {num(score[('scientist', 'science')])} против {num(score[('adaptive', 'science')])} очка.
Исследователь точнее в выводах, но не результативнее.</li>
<li><b>Расследования.</b> При базовых правилах из {b['total']} расследований вывод сделан в {b['identified']}, верен в
{b['correct']}; в {b['insufficient']} случаях робот записал «данных недостаточно». При научных: {sc_['total']} расследований,
вывод в {sc_['identified']}, верен в {sc_['correct']}, частично верен в {sc_['partial']}, неверен в {sc_['wrong']}.</li>
<li><b>Слабое место.</b> Среди гипотез об образце, которые робот <i>опроверг</i>, от {pct(min(v[0] for v in ref.values()))} до
{pct(max(v[0] for v in ref.values()))} на самом деле были верны: образец лежал в пределах 35 см. Робот слишком рано сдаётся. Это
не исправлено.</li>
</ul>

<h3>Прогон, который стоит посмотреть</h3>
<p>Исследователь, научные правила, трудный уровень, сценарий 1001. Кусок журнала с 19-й по 62-ю секунду, как он записан в прогоне.
В нём три расследования:</p>
<ul>
<li><b>Q1</b> — расход вдвое выше прогноза. Робот ставит опыт «постоять две секунды» и отличает дорогой грунт от утечки заряда.</li>
<li><b>Q2</b> — то же, но объяснение одно, и вывод сделан по исключению, без опыта.</li>
<li><b>Q3</b> — датчик образцов десять раз подряд выдал одно значение. Робот выясняет, что датчик залип, и перестаёт ему верить.</li>
</ul>
<p>Обратите внимание на гипотезу <b>H6</b>: робот её подтвердил, а по скрытой правде она неверна. На 23-й секунде в сценарии
сменились грунты, и дорогой участок с этого места исчез — как раз когда робот заканчивал измерение. Таких ошибок у исследователя
при научных правилах {val[('soil', 'science', 'scientist')][2] - val[('soil', 'science', 'scientist')][1]} из
{val[('soil', 'science', 'scientist')][2]}.</p>
{clip('e17_journal', 'Весь прогон. Строка журнала появляется, когда робот выдвигает гипотезу. После финиша справа открывается сверка: совпал ли вывод робота со скрытой правдой среды.')}
{journal}

<h3>Что могут спросить</h3>
{qa([
    ('Гипотезы выдвигает языковая модель?',
     'Нет. Объяснения, выбор опыта и вероятности считает программа. Модель может выступать автором плана опытов и критиком — такой код '
     'есть, и серия на трёх моделях МАИ посчитана (ревью ещё не прошла): результат совпал с программой, 64,8 против 65,0 очка. '
     'Говорить «гипотезы выдвигает LLM» мы не будем.'),
    ('Откуда вы знаете правду?',
     'Сценарий создаёт наш генератор, он знает всё. Агент получает только то, что есть в интерфейсе из условия: одометрию, лидар, '
     'батарею, датчик образцов. Сверка идёт после прогона отдельным кодом (did/metrics.py).'),
    ('А сам код сверки проверен?',
     'Да, и это было нужно: независимое ревью нашло в первой версии семь ошибок — шесть в подсчёте и одну в самом агенте. Два вывода '
     'после исправления перевернулись. Числа на этой странице — после исправлений.'),
    ('Чем исследователь лучше, если счёт тот же?',
     'Объяснимостью. Он не просто объезжает участок, а записывает, почему: что наблюдал, какие объяснения рассматривал, какой опыт '
     'поставил и что измерил. И его выводы о причинах точнее, когда причин несколько. Очков это не добавляет, и мы так и говорим.'),
    ('Почему среди опровергнутых так много верных?',
     'Робот подъезжает к месту, сигнал не подтверждается — гипотеза закрывается как опровергнутая, хотя образец был рядом. По отдельным '
     'прогонам мы это не разбирали и не исправляли.'),
])}

<h3>Оговорки</h3>
<ul>
<li>Гипотезы о коэффициентах расхода («поворот стоит 0,11 единицы на радиан») со скрытой правдой не сверяются: в подсчёт не входят.</li>
<li>Гипотеза об образце считается верной, если несобранный образец лежит ближе 35 см от названной точки. Порог выбран нами.</li>
<li>На других страницах проекта встречается число 98,5% верных расследований: оно из другого опыта (исследователь, трудный уровень).
Числа здесь посчитаны по опыту E17 на текущем коде.</li>
</ul>
<p class="where">Описание: <code>experiments/E17.yaml</code> · отчёты: <code>research/findings/R4.md</code>, <code>V1.md</code> · код:
<code>did/journal.py</code>, <code>did/inquiry.py</code>, сверка — <code>did/metrics.py</code></p>
<div class="say"><b>Одной фразой для доклада.</b> Мы сверили журнал робота со скрытой правдой среды: подтверждённые гипотезы о грунте
верны в {pct(val[('soil', 'base', 'adaptive')][0])} случаев, а там, где причин несколько, робот-исследователь ошибается в
{pct(1 - val[('soil', 'science', 'scientist')][0])} случаев против {pct(1 - val[('soil', 'science', 'adaptive')][0])} у обычного агента.</div>
'''
    return section(5, 'e5', 'Журнал гипотез против скрытой правды',
                   'Критерий «Научный подход», 15% оценки; бонусный трек «Лучший научный агент»', body), val


# =====================================================================================
#  Опыт 6. Перенос в Gazebo
# =====================================================================================
def exp6():
    pairs = []
    for p in sorted((RUNS / 'E7' / 'gazebo').glob('*.json.gz'),
                    key=lambda q: (['easy', 'medium', 'hard'].index(q.name.split('-')[0]), int(q.name.split('-')[1].split('.')[0]))):
        f = RUNS / 'E7' / 'fastsim' / p.name
        if f.exists():
            pairs.append((p.name[:-8], load(f), load(p)))
    rows = []
    for name, fs, gz in pairs:
        lv, n = name.split('-')
        notes = []
        for word, r in (('быстрый', fs['result']), ('Gazebo', gz['result'])):
            if not r['returned']:
                notes.append(f'{word}: не вернулся' + (f', столкновений {r["collisions"]}' if r.get('collisions') else ''))
        rows.append({'label': f'{LEVEL[lv].capitalize()}, сценарий {n}', 'sub': '; '.join(notes) or None,
                     'val': f"{num(fs['result']['score'])} и {num(gz['result']['score'])} · {sgn(gz['result']['score'] - fs['result']['score'])}",
                     'pts': [{'s': i, 'm': r['result']['score'], 'val': num(r['result']['score']),
                              'tip': f"{word}: счёт {num(r['result']['score'])}, собрано {r['result']['samples_collected']} из "
                                     f"{r['result']['samples_total']}, {'вернулся' if r['result']['returned'] else 'не вернулся'}, "
                                     f"{num(r['result']['time'], 0)} с"}
                             for i, (word, r) in enumerate((('Быстрый симулятор', fs), ('Gazebo', gz)))]})
    chart = interval_chart(rows, ['Быстрый симулятор', 'Gazebo'], 30, 100, [30, 40, 50, 60, 70, 80, 90, 100], lanes=False, connect=True,
                           head=('Сценарий', 'Счёт'), value_w=190, aria='Счёт одного и того же сценария в быстром симуляторе и в Gazebo')
    d = [gz['result']['score'] - fs['result']['score'] for _, fs, gz in pairs]
    same = sum(fs['result']['samples_collected'] == gz['result']['samples_collected'] for _, fs, gz in pairs)
    close = sum(abs(x) <= 6 for x in d)
    ret_f, ret_g = (sum(r[i]['result']['returned'] for r in pairs) for i in (1, 2))
    hard = [(fs, gz) for name, fs, gz in pairs if name.startswith('hard')]
    by = {name: (fs, gz) for name, fs, gz in pairs}
    f4, g4 = by['hard-4']
    g5 = by['hard-5'][1]
    after = load(RUNS / 'F1_v2' / 'adaptive_v2' / 'hard-5.json.gz')
    v2 = [load(p) for p in sorted((RUNS / 'F1_v2').glob('*/*.json.gz'))]
    maps = ('<div class="maps three">'
            + run_map(g4, 'Сценарий 4: два симулятора', f"быстрый {num(f4['result']['score'])} · Gazebo {num(g4['result']['score'])} очка",
                      color='s2', second=f4, second_color='s1')
            + run_map(g5, 'Сценарий 5 в Gazebo, до правки', result_line(g5), color='s2')
            + run_map(after, 'Сценарий 5 в Gazebo, после правки', result_line(after), color='s2') + '</div>'
            + '<div class="legend"><span class="key"><i class="ln s1"></i>быстрый симулятор</span>'
              '<span class="key"><i class="ln s2"></i>Gazebo</span></div>' + MAP_LEGEND)
    body = f'''
<div class="answer"><b>Вопрос.</b> Тысячи прогонов сделаны в быстром симуляторе. Ведёт ли себя тот же агент так же в Gazebo, где
его будут показывать?<br>
<b>Ответ.</b> В основном да: в {close} парах из {len(pairs)} счёт отличается не больше чем на 6 очков, в {same} из {len(pairs)}
собрано одинаково образцов. Один прогон в Gazebo вскрыл дефект, которого быстрый симулятор не показывал: потерю положения после
столкновения. Он исправлен.</div>

<h3>Как устроен</h3>
<ul>
<li>Один и тот же сценарий проходит один раз в быстром симуляторе и один раз в Gazebo. Агент и правила судьи — один и тот же код.</li>
<li><b>Быстрый симулятор</b> — наш код на Python: робот — точка без инерции, шаг 0,1 секунды, лидар считается лучами по карте.
Прогон занимает полсекунды.</li>
<li><b>Gazebo</b> — физический симулятор из условия: колёса, инерция, настоящий лидар, одометрия с ошибкой. Прогон — около трёх
минут, и одновременно может идти только один.</li>
<li>{len(pairs)} пар: по одной на лёгком и среднем уровне, {len(hard)} на трудном. Прогоны сделаны 8 октября до правки защиты от
потери положения.</li>
</ul>

<h3>Результат</h3>
{chart}
<p class="cap">Каждая строка — один сценарий. Две точки — счёт в быстром симуляторе и в Gazebo; справа оба числа и разность
«Gazebo минус быстрый».</p>

<h3>Как читать</h3>
<ul>
<li><b>Систематического сдвига нет:</b> средняя разность {sgn(float(np.mean(d)))} очка, Gazebo то выше, то ниже.</li>
<li><b>Возврат на базу:</b> {ret_g} из {len(pairs)} в Gazebo и {ret_f} из {len(pairs)} в быстром симуляторе — но на разных
сценариях. На трудном сценарии 3 не вернулся робот в быстром симуляторе, на сценарии 5 — в Gazebo.</li>
<li><b>Отдельные сценарии расходятся сильно.</b> Прогон — цепочка решений: чуть другая траектория, другой образец найден первым,
дальше прогоны разные. Поэтому сравнивать надо серии, а не один прогон с одним.</li>
<li><b>Сценарий 5 — настоящий дефект.</b> Робот задел столб, получил штраф, отъехал назад не глядя и ударился ещё раз. От удара
корпус в Gazebo закачался, лидар и одометрия перестали сходиться с картой, робот потерял положение. В быстром симуляторе корпус не
качается, и там этого не было видно.</li>
</ul>

<h3>Прогон, который стоит посмотреть</h3>
{clip('e7_fast_vs_gazebo', 'Трудный сценарий 4 в двух симуляторах. На правой карте бледной линией повторён путь из быстрого симулятора.')}
{clip('g2_gazebo_fix', 'Трудный сценарий 5 в Gazebo до правки и после. В записи Gazebo лежит положение по оценке самого робота. До правки она разошлась с правдой примерно на 82-й секунде: дальше путь показан пунктиром — робот считал, что вернулся на базу, судья возврат не засчитал.')}
<p>Те же прогоны целиком. Слева — трудный сценарий 4: путь в Gazebo и в быстром симуляторе на одной карте. В центре — сценарий 5
до правки: четыре столкновения, робот не вернулся; конец пути на этой карте — там, где робот себя считал, положение он к тому
времени потерял. Справа — тот же сценарий после правки.</p>
{maps}
<p>После правки робот отъезжает только туда, где лидар видит свободное место, не давит в преграду и, потеряв положение, ищет себя
не дольше двадцати секунд. В проверке исследования — 7 возвратов из 7. В проверке нового агента 8 октября — {sum(r['result']['returned'] for r in v2)}
из {len(v2)}, столкновений {sum(r['result']['collisions'] for r in v2)}, счёт от {num(min(r['result']['score'] for r in v2))} до
{num(max(r['result']['score'] for r in v2))}.</p>

<h3>Что могут спросить</h3>
{qa([
    ('Почему не считать всё в Gazebo?',
     'Один прогон — около трёх минут, и идти может только один одновременно. Восемь с лишним тысяч наших прогонов заняли бы больше двух '
     'недель. В быстром симуляторе это часы.'),
    ('Насколько можно верить быстрому симулятору?',
     'Он верно передаёт то, от чего зависит счёт: расход заряда, показания датчика, правила судьи — это один и тот же код. Он не передаёт '
     'физику корпуса: столкновения, качание, проскальзывание колёс, уход одометрии. Поэтому всё, что касается столкновений и потери '
     'положения, проверяется только в Gazebo.'),
    ('Одиннадцать пар — это мало.',
     'Да. Это проверка, что нет систематического сдвига, а не измерение разности. Интервал для неё мы не заявляем.'),
    ('Правка для Gazebo ничего не сломала?',
     'Сломала, и мы это нашли: после неё пересчитали все серии в быстром симуляторе, и в нескольких опытах появились прогоны со '
     'столкновениями и невозвратами, которых раньше не было. Это исправляется прямо сейчас; после исправления серии пересчитаем ещё раз.'),
    ('На карте «после» другой агент?',
     'Да: до правки ездил адаптивный агент, после — он же с пережиданием сбоя датчика. Это не чистое сравнение «до и после» одной '
     'правки, а два прогона одного сценария с разницей в день работы.'),
])}

<h3>Оговорки</h3>
<ul>
<li>По одному прогону на сценарий в каждом симуляторе; Gazebo не полностью повторяем — второй прогон того же сценария даст другой счёт.</li>
<li>Пары сняты до правки защиты от потери положения; после неё парная серия не повторялась.</li>
<li>На загруженной машине Gazebo пропускает такты, и робот задевает столбы при любой защите. Такие прогоны не засчитываются.</li>
</ul>
<p class="where">Описание: <code>experiments/E7.yaml</code> · запуск пар: <code>pixi run python tools/gazebo_batch.py --level hard --seeds 1 2 3</code>
· отчёт о потере положения: <code>research/findings/G2.md</code></p>
<div class="say"><b>Одной фразой для доклада.</b> Серии мы считаем в быстром симуляторе и сверяем с Gazebo на парах одинаковых
сценариев: систематического сдвига нет, а единственный серьёзный дефект — потерю положения после удара о столб — нашёл именно Gazebo.</div>
'''
    return section(6, 'e6', 'Перенос в Gazebo: можно ли верить быстрым прогонам',
                   'Критерий «Работоспособность агента», 30% оценки; показ идёт в Gazebo', body), (close, len(pairs))


# =====================================================================================
#  Страница
# =====================================================================================
CSS = '''
:root{color-scheme:light;--page:#f5f5f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8b8a84;--line:#e3e2dd;
--floor:#e9e8e2;--code:#f0efeb;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--crit:#d03b3b;--warn:#fab219;--good:#0ca30c;--accent:#eb6834}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#111110;--surface:#1a1a19;
--ink:#ffffff;--ink2:#c3c2b7;--muted:#8e8d85;--line:#34342f;--floor:#2b2b27;--code:#232320;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--crit:#e66767;--accent:#d95926}}
:root[data-theme="dark"]{color-scheme:dark;--page:#111110;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8e8d85;
--line:#34342f;--floor:#2b2b27;--code:#232320;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--crit:#e66767;--accent:#d95926}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:34px;line-height:1.15;margin:8px 0 10px}
h2{font-size:26px;line-height:1.2;margin:2px 0 6px}
h3{font-size:18px;margin:26px 0 8px}
p{margin:8px 0}
ul{margin:6px 0;padding-left:22px}
li{margin:5px 0}
a{color:var(--s1)}
code{font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:1px 5px;border-radius:4px}
pre{background:var(--code);border-radius:8px;padding:12px 14px;overflow-x:auto;margin:10px 0}
pre code{background:none;padding:0;white-space:pre}
.lead{font-size:18px;color:var(--ink2);max-width:780px}
.note{font-size:14px;color:var(--ink2);border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:12px 0}
section{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px 26px 24px;margin:22px 0}
.num{font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
.tag{color:var(--ink2);font-size:14.5px;margin:0 0 12px}
.answer{background:var(--code);border-radius:10px;padding:12px 16px;margin:10px 0 4px}
.say{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:10px 14px;margin:18px 0 0}
.cap{font-size:13.5px;color:var(--ink2);margin:4px 0 12px}
.where{font-size:13.5px;color:var(--ink2);margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:13px}
td.c{white-space:nowrap}
.sum td:first-child{white-space:nowrap}
details.tbl{margin:2px 0 6px;font-size:13.5px}
details.tbl summary{cursor:pointer;color:var(--ink2)}
details.tbl td{font-variant-numeric:tabular-nums}
.qa details{border-bottom:1px solid var(--line);padding:9px 0}
.qa summary{cursor:pointer;font-weight:600}
.qa p{color:var(--ink2);margin:6px 0 2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13.5px;color:var(--ink2);margin:8px 0 4px}
.key{display:inline-flex;align-items:center;gap:6px}
.key i{display:inline-block;flex:none}
i.dot{width:10px;height:10px;border-radius:50%}
i.dot.s1{background:var(--s1)} i.dot.s2{background:var(--s2)} i.dot.s3{background:var(--s3)}
i.dot.got{background:var(--ink)} i.dot.open{border:1.5px solid var(--ink);background:var(--surface)}
i.sq{width:12px;height:12px;border-radius:3px}
i.sq.soil{background:color-mix(in srgb,var(--ink) 16%,transparent)}
i.sq.haz{border:1.5px solid var(--crit);background:color-mix(in srgb,var(--crit) 14%,transparent);border-radius:50%}
i.sq.base{border:1.5px solid var(--ink);background:var(--surface)}
i.sq.fault{background:color-mix(in srgb,var(--warn) 45%,transparent)}
i.sq.waitb{background:color-mix(in srgb,var(--ink) 30%,transparent)}
i.x{color:var(--crit);font-style:normal;font-weight:700;line-height:1}
i.ln{width:20px;height:3px;border-radius:2px}
i.ln.s1{background:var(--s1)} i.ln.s2{background:var(--s2)} i.ln.hl{background:var(--ink)} i.ln.ban{background:var(--crit)}
svg{display:block;width:100%;height:auto}
svg.chart{margin:6px 0 2px}
.grid{stroke:var(--line);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--ink2);font-size:12px;text-anchor:middle;font-variant-numeric:tabular-nums}
.tick.e{text-anchor:end}
.rl{fill:var(--ink);font-size:13.5px;text-anchor:end}
.rs{fill:var(--muted);font-size:11.5px;text-anchor:end}
.val{fill:var(--ink);font-size:12.5px;font-variant-numeric:tabular-nums}
.ci{stroke-width:2;stroke-linecap:round}
.pt{stroke:var(--surface);stroke-width:2}
.pt.bare{stroke:none}
.s1{stroke:var(--s1)} .s2{stroke:var(--s2)} .s3{stroke:var(--s3)} .mut{stroke:var(--muted)}
circle.s1{fill:var(--s1);stroke:var(--surface)} circle.s2{fill:var(--s2);stroke:var(--surface)}
circle.s3{fill:var(--s3);stroke:var(--surface)} circle.mut{fill:var(--muted);stroke:var(--surface)}
.conn{stroke:var(--line);stroke-width:3;stroke-linecap:round}
.hit{fill:transparent}
.ln{fill:none;stroke-width:1.6;stroke-linejoin:round}
.band.fault{fill:color-mix(in srgb,var(--warn) 22%,transparent)}
.band.wait{fill:color-mix(in srgb,var(--ink) 14%,transparent)}
.bl{fill:var(--ink2);font-size:11.5px;text-anchor:middle}
.cross{stroke:var(--muted);stroke-width:1}
.tsname{display:flex;align-items:center;gap:7px;font-size:13.5px;font-weight:600;margin:12px 0 0}
.maps{display:grid;gap:14px;margin:10px 0 4px}
.maps.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.maps.three{grid-template-columns:repeat(3,minmax(0,1fr))}
@media (max-width:720px){.maps.two,.maps.three{grid-template-columns:1fr}}
figure.clip{margin:12px 0 16px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
figure.clip img{display:block;width:100%;height:auto}
figure.clip figcaption{font-size:13.5px;line-height:1.4;padding:8px 12px;color:var(--ink2);background:var(--surface);border-top:1px solid var(--line)}
figure.map{margin:0;border:1px solid var(--line);border-radius:10px;padding:8px 8px 10px;background:var(--surface)}
figure.map figcaption{font-size:13px;line-height:1.35;padding:6px 4px 0}
figure.map figcaption span{display:block;color:var(--ink2)}
.floor{fill:var(--floor)}
.soil{fill:color-mix(in srgb,var(--ink) 15%,transparent)}
.haz{fill:color-mix(in srgb,var(--crit) 14%,transparent);stroke:var(--crit);stroke-width:1.5}
.zl{fill:var(--ink2);font-size:10.5px;text-anchor:middle;pointer-events:none}
.ban{stroke:var(--crit);stroke-width:2}
.tr{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.tr.hl{stroke:var(--ink);stroke-width:3.2}
.sm{fill:var(--surface);stroke:var(--ink);stroke-width:1.6}
.sm.got{fill:var(--ink);stroke:var(--surface);stroke-width:2}
.pen circle{fill:var(--surface);opacity:.85}
.pen path{stroke:var(--crit);stroke-width:2.2;stroke-linecap:round}
.base{fill:var(--surface);stroke:var(--ink);stroke-width:1.6}
.end{stroke-width:2}
.dialog{display:grid;gap:10px;margin:10px 0}
.turn .who{font-size:13px;font-weight:600;color:var(--ink2)}
.turn pre{margin:4px 0 0}
.turn pre code{white-space:pre-wrap;word-break:break-word}
ol.journal{list-style:none;padding:0;margin:10px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden}
ol.journal li{display:grid;grid-template-columns:62px 112px 1fr;gap:10px;padding:7px 12px;margin:0;border-bottom:1px solid var(--line);font-size:14px}
ol.journal li:last-child{border-bottom:none}
ol.journal .t{color:var(--muted);font-variant-numeric:tabular-nums}
ol.journal .k{color:var(--ink2);font-weight:600}
ol.journal li.k-inquiry{background:color-mix(in srgb,var(--s1) 7%,transparent)}
.tv{font-size:12.5px;white-space:nowrap;border-radius:4px;padding:1px 6px;margin-left:4px;border:1px solid var(--line);color:var(--ink2)}
.tv.ok::before{content:"✓ "} .tv.bad::before{content:"✗ "}
.toc a{text-decoration:none}
#tip{position:fixed;z-index:10;max-width:340px;background:var(--ink);color:var(--surface);font-size:13px;line-height:1.4;
padding:7px 10px;border-radius:7px;pointer-events:none}
@media print{section{break-inside:avoid-page}.qa details{display:block}}
'''

JS = '''
(function(){
  var tip=document.getElementById('tip');
  function place(e){var p=14,w=tip.offsetWidth,h=tip.offsetHeight,x=e.clientX+p,y=e.clientY+p;
    if(x+w>innerWidth-8)x=e.clientX-w-p; if(y+h>innerHeight-8)y=e.clientY-h-p; tip.style.left=Math.max(4,x)+'px'; tip.style.top=Math.max(4,y)+'px';}
  function ru(v,d){return v.toFixed(d).replace('.',',');}
  var live=null;
  function clear(){if(live){live.querySelector('.cross').setAttribute('visibility','hidden');live.querySelector('.crossdot').setAttribute('visibility','hidden');live=null;}}
  document.addEventListener('mousemove',function(e){
    var el=e.target.closest&&e.target.closest('[data-tip]');
    if(el){clear();tip.textContent=el.dataset.tip;tip.hidden=false;place(e);return;}
    var ts=e.target.closest&&e.target.closest('svg.ts');
    if(ts){
      if(live&&live!==ts)clear(); live=ts;
      if(!ts._t){ts._t=JSON.parse(ts.dataset.t);ts._v=JSON.parse(ts.dataset.v);}
      var r=ts.getBoundingClientRect(),vb=ts.viewBox.baseVal,vx=(e.clientX-r.left)*vb.width/r.width;
      var x0=+ts.dataset.x0,x1=+ts.dataset.x1,y0=+ts.dataset.y0,y1=+ts.dataset.y1,tm=+ts.dataset.tmax,ym=+ts.dataset.ymax;
      var t=Math.min(tm,Math.max(0,(vx-x0)/(x1-x0)*tm)),a=ts._t,i=0,best=1e9;
      for(var k=0;k<a.length;k++){var d=Math.abs(a[k]-t);if(d<best){best=d;i=k;}}
      var cx=x0+a[i]/tm*(x1-x0),cy=y0-Math.min(ym,Math.max(0,ts._v[i]))/ym*(y0-y1);
      var c=ts.querySelector('.cross'),dot=ts.querySelector('.crossdot');
      c.setAttribute('x1',cx);c.setAttribute('x2',cx);c.setAttribute('visibility','visible');
      dot.setAttribute('cx',cx);dot.setAttribute('cy',cy);dot.setAttribute('visibility','visible');
      tip.textContent=ru(a[i],0)+'-я секунда: датчик показывает '+ru(ts._v[i],2);tip.hidden=false;place(e);return;
    }
    clear();tip.hidden=true;
  });
  document.addEventListener('mouseleave',function(){clear();tip.hidden=true;});
})();
'''


def main():
    s1, ph, ah, fh = exp1()
    s2, share = exp2()
    s3, pv = exp3()
    s4 = exp4()
    s5, val = exp5()
    s6, (close, npairs) = exp6()
    made = __import__('datetime').datetime.now().strftime('%d.%m.%Y, %H:%M')
    rows = [
        ('e1', 'Адаптивный агент против фиксированного плана', 'Работоспособность, 30%',
         f'{sgn(ph["mean"])} очка на трудном уровне [{num(ph["ci"][0])}; {num(ph["ci"][1])}]', 'возврат на базу одинаковый'),
        ('e2', 'Способы поиска образцов', 'Уровень 3: способы названы в условии',
         f'собрано {pct(share["adaptive"][0], 1)}, {pct(share["gradient"][0])} и {pct(share["spiral"][0], 1)}', 'только лёгкий и средний уровни'),
        ('e3', 'Адаптация: что из неё даёт очки', 'Адаптивность, 25%',
         f'пережидание сбоя датчика: {sgn(pv["mean"])} [{num(pv["ci"][0])}; {num(pv["ci"][1])}]', 'реакция на смену грунта очков не даёт'),
        ('e4', 'Миссия, заданная словами', 'LLM-планирование, 15%; уровень 2', 'модель 6 из 6, правило 0 из 6', 'шесть прогонов; по очкам модель не лучше'),
        ('e5', 'Журнал гипотез против скрытой правды', 'Научный подход, 15%; бонус',
         f'гипотезы о грунте верны в {pct(val[("soil", "base", "adaptive")][0])}', 'треть опровержений поспешны'),
        ('e6', 'Перенос в Gazebo', 'Работоспособность; показ в Gazebo', f'{close} пар из {npairs} сходятся в пределах 6 очков', 'пар мало; сняты до последней правки'),
    ]
    toc = ''.join(f'<tr><td>{i}</td><td><a href="#{a}">{t}</a></td><td>{c}</td><td>{n}</td><td>{w}</td></tr>'
                  for i, (a, t, c, n, w) in enumerate(rows, 1))
    boot = '''
# did/metrics.py: как получается запись «+6,4 [2,9; 9,8]»
diffs = [score_new[s] - score_old[s] for s in scenarios]   # один и тот же сценарий у обоих агентов
mean = diffs.mean()                                        # +6,4 — средняя разность
# 4000 раз набираем те же 40 сценариев заново, с повторами, и каждый раз считаем среднее
boot = rng.choice(diffs, size=(4000, len(diffs))).mean(axis=1)
lo, hi = percentile(boot, 2.5), percentile(boot, 97.5)     # [2,9; 9,8] — где лежат 95% этих средних
'''
    svg = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><symbol id="arena" overflow="visible">'
           f'<path class="floor" d="{ARENA_D}"/></symbol></defs></svg>')
    levels = '''<h3>1. Сценарий и уровень</h3>
<p>Арена всегда одна: шестиугольник с девятью столбами, база слева. <b>Сценарий</b> — это номер, по которому генератор расставляет
образцы, участки дорогого грунта и опасные зоны. Один номер всегда даёт один и тот же сценарий.</p>
<ul>
<li><b>Лёгкий:</b> 3 образца, 1 участок дорогого грунта.</li>
<li><b>Средний:</b> 5 образцов, 3 участка.</li>
<li><b>Трудный:</b> 7 образцов, 4 участка, опасная зона — и среда меняется по ходу прогона: грунт дорожает, появляется новая зона,
датчик образцов на время выходит из строя.</li>
</ul>
<p>Робот ничего из этого заранее не знает. У него есть лидар, одометрия, остаток батареи и датчик образцов.</p>
<h3>2. Счёт</h3>
<p>10 очков за образец, 20 за возврат на базу, 0,1 за каждую оставшуюся единицу заряда (только если вернулся). Штрафы: 5 очков и 3 единицы заряда за
заезд в опасную зону, 2 очка за столкновение, 3 за ложную попытку сбора. Батарея — 60 единиц, метр по обычному полу стоит 2,5: это 24 метра.</p>
<h3>3. Кто с кем сравнивается</h3>
<ul>
<li><b>Фиксированный план</b> — едет по заранее заданному маршруту, ничего не запоминает.</li>
<li><b>Адаптивный агент</b> — ищет образцы по карте вероятностей, учит цену грунта, помнит опасные зоны, считает запас на дорогу домой.</li>
<li><b>Исследователь</b> — адаптивный агент, который расследует странности опытами.</li>
<li><b>Новый агент</b> (в коде <code>adaptive_v2</code>) — адаптивный, который пережидает сбой датчика.</li>
</ul>
<h3>4. Запись «+6,4 [2,9; 9,8]»</h3>
<p>Это <b>парная разность</b> и её <b>95% интервал</b>. Оба варианта проходят одни и те же сценарии, на каждом считается разность
счёта. Первое число — средняя разность. В скобках — диапазон, в котором она лежала бы, возьми мы другие сценарии.</p>
''' + code(boot) + '''
<ul>
<li>Интервал целиком выше нуля — <b>подтверждено</b>.</li>
<li>Интервал включает ноль — <b>различие не показано</b>. Это не «разницы нет»: это «на таком числе сценариев её не видно».</li>
<li>Интервал целиком на «плохой» стороне — <b>опровергнуто</b>.</li>
</ul>
<p>Границы интервала считаются случайным перебором и при пересчёте могут отличаться во втором знаке.</p>
<h3>5. Отладка и итог — на разных сценариях</h3>
<p>Агента отлаживали на сценариях с номерами 1–80. Итоговые серии идут на других номерах, которых при отладке никто не смотрел.
Иначе можно подогнать агента под конкретные сценарии и принять это за улучшение.</p>'''
    story = f'''<ol>
<li><b>Робот работает.</b> На трудном уровне адаптивный агент набирает на {num(ph['mean'], 0)} очков больше, чем робот с заданным
маршрутом (опыт 1). Образцы он ищет по карте вероятностей — она собирает больше, чем спираль и подъём по сигналу из условия (опыт 2).</li>
<li><b>Робот адаптируется — и мы знаем, что из этого работает.</b> Пережидание сбоя датчика даёт {sgn(pv['mean'])} очка; реакция на
смену грунта в наших сценариях почти ничего (опыт 3).</li>
<li><b>Языковая модель на своём месте.</b> Меняем текст миссии — меняется поведение, без нового кода (опыт 4).</li>
<li><b>Журнал можно проверить.</b> Гипотезы робота сверены со скрытой правдой среды (опыт 5).</li>
<li><b>Это не только симулятор.</b> Те же сценарии пройдены в Gazebo, и один дефект нашёл именно он (опыт 6).</li>
</ol>'''
    sections = s1 + s2 + s3 + s4 + s5 + s6

    def page(title, lead, note, toc_head, basics_title, body, closing):
        return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body>{svg}
<main>
<div class="num">DID Hack · робот-исследователь</div>
<h1>{title}</h1>
<p class="lead">{lead}</p>
<p class="note">{note}</p>

<section class="toc"><h2>Коротко</h2>
<table class="sum"><thead><tr><th></th>{''.join(f'<th>{h}</th>' for h in toc_head)}</tr></thead>
<tbody>{toc}</tbody></table></section>

<section id="basics"><h2>{basics_title}</h2>
{levels}
</section>

{body}

{closing}
</main>
<div id="tip" hidden></div>
<script>{JS}</script>
</body></html>'''

    guide = page(
        'Шесть опытов для показа',
        'Что в каждом проверено, как он устроен, как читать числа и что могут спросить. По одному опыту на каждый критерий оценки.',
        f'Собрано {made}. Числа взяты из сводок опытов на текущем коде, рисунки и анимации — из записей настоящих прогонов. '
        'Сейчас исправляется побочный эффект защиты от потери положения; после исправления серии пересчитаем, и числа опытов 1, 3, 5 и 6 '
        'могут сдвинуться в последнем знаке. Пересобрать страницу: <code>pixi run python tools/build_showcase.py</code>. '
        'Та же страница без вопросов и подсказок докладчику — <a href="experiments.html">experiments.html</a>.',
        ('Опыт', 'Что закрывает', 'Главное число', 'Оговорка'), 'Пять вещей, без которых числа не прочитать', sections,
        f'''<section id="order"><h2>Как это уложить в доклад</h2>
<p>В пяти минутах показа на опыты уйдёт секунд сорок–шестьдесят. Порядок, в котором один результат подводит к следующему:</p>
{story}
<h3>Что не показываем и почему</h3>
<ul>
<li><b>«Неожиданности по одной».</b> Преимущество над фиксированным планом одинаково с событием и без него — опыт не про адаптацию.</li>
<li><b>Исследователь против адаптивного по очкам.</b> Различие не показано; его сила — в точности выводов, это опыт 5.</li>
<li><b>Чужие правила судьи.</b> «Чужой судья» — наше допущение, в условии его нет. Проверка есть, но в рассказ она не входит.</li>
<li><b>Числа с пометкой «ревью не прошло»:</b> расширенная серия миссий, модель в роли автора гипотез, езда во время ожидания
модели, карта через SLAM Toolbox.</li>
</ul>
</section>''')

    # Страница для жюри: те же разделы без вопросов с ответами и без подсказок докладчику.
    plain = re.sub(r'<h3>Что могут спросить</h3>\s*<div class="qa">.*?</div>', '', sections, flags=re.S)
    plain = re.sub(r'<div class="say">.*?</div>', '', plain, flags=re.S)
    plain = (plain.replace('<h3>Прогон, который стоит посмотреть</h3>', '<h3>Пример прогона</h3>')
             .replace('<h3>Оговорки</h3>', '<h3>Ограничения</h3>'))
    jury = page(
        'Шесть опытов: что проверяли и что получили',
        'По одному опыту на каждый критерий оценки. В каждом — вопрос, как устроена проверка, результат с интервалом, '
        'пример настоящего прогона и ограничения.',
        'Числа взяты из сводок серий прогонов, рисунки и анимации — из записей этих прогонов. Любой опыт пересчитывается одной '
        'командой, она указана в конце его раздела.',
        ('Опыт', 'Критерий оценки', 'Главное число', 'Ограничение'), 'Как читать числа', plain,
        f'<section id="summary"><h2>Итог</h2>\n{story}\n</section>')

    for out, text in ((OUT, guide), (OUT_JURY, jury)):
        out.write_text(text, encoding='utf-8')
        print(f'{out.relative_to(ROOT)}: {len(text.encode("utf-8")) / 1024:.0f} КБ')


if __name__ == '__main__':
    main()
