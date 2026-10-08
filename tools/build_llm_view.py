#!/usr/bin/env python3
"""Страница «Что получает языковая модель и что она отвечает»: llm_calls.html в корне репозитория.

    pixi run python tools/build_llm_view.py

Каждая карточка — один настоящий обмен из кэша ответов (runs/_llm_cache и кэш исследования R13): слева карта,
нарисованная только по тому, что было в сообщении модели, справа — её ответ; ниже оба текста дословно.
К модели не обращается. Оформление и карта арены — из tools/build_showcase.py.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_showcase import ARENA_D, CSS, JS, K, MH, MW, XMIN, YMAX, esc, num, sx, sy  # noqa: E402

MAIN = ROOT / 'runs' / '_llm_cache'
R13 = ROOT.parent / 'DID-research' / 'R13' / 'runs' / '_llm_cache'
OUT = ROOT / 'llm_calls.html'

TRIGGER = {'start': 'старт прогона', 'candidate_found': 'датчик нашёл место, похожее на образец',
           'candidate_lost': 'место, к которому ехали, не подтвердилось', 'sample_collected': 'образец собран',
           'sensor_degraded': 'датчик образцов начал сбоить', 'subgoal_done': 'подцель выполнена',
           'hazard': 'штраф: заезд в опасную зону', 'battery_threshold': 'заряд дошёл до порога',
           'model_mismatch': 'расход не сходится с прогнозом', 'collision': 'столкновение', 'false_collect': 'ложная попытка сбора'}
GOAL = {'explore': 'разведать', 'investigate': 'подъехать и собрать', 'return_base': 'вернуться на базу', 'goto': 'доехать до точки'}

# (кэш, модель, начало имени файла, заголовок, что здесь смотреть)
CARDS = [
    (MAIN, 'qwen3.8-flash-next', '0a81e0a6', 'Старт: робот ещё ничего не знает',
     'Кандидатов нет, датчик молчит. Модели остаётся выбрать, какой участок разведать первым.'),
    (MAIN, 'qwen3.8-flash-next', '2ed71a24', 'Датчик нашёл место, похожее на образец',
     'В списке появился кандидат. Модель должна решить, ехать ли к нему и что делать после.'),
    (MAIN, 'qwen3.8-flash-next', '027c0822', 'Образец собран — что дальше',
     'Самый частый повод: после каждого сбора робот спрашивает, куда ехать теперь.'),
    (MAIN, 'qwen3.8-flash-next', '01034f00', 'Датчик сбоит, кандидат пропал',
     'В состоянии стоит sensor.status = degraded и тревога. По правилам подсказки слабому кандидату при неисправном датчике верить нельзя.'),
    (MAIN, 'qwen3.8-flash-next', '2ec08117', 'Штраф: робот заехал в опасную зону',
     'Робот стоит внутри опасной зоны, датчик при этом неисправен. Модель решает, ехать ли к кандидату с уверенностью 62%.'),
    (R13, 'qwen3.8-flash-next', '823d7190', 'Другая миссия: «собери ровно два образца»',
     'Код тот же, изменён только текст миссии. Второй образец собран — модель должна остановить сбор.'),
    (R13, 'qwen3.8-flash-next', '0c0a3622', 'Другая миссия: «не заезжай в правую половину»',
     'Все кандидаты и точки разведки лежат в запрещённой половине. Модель сама должна от них отказаться.'),
    (R13, 'qwen3.8-flash-next', '195b5bf2', 'Другая миссия: «после первого штрафа — домой»',
     'В состояние добавлен счётчик штрафов. Штраф получен — модель должна прекратить сбор.'),
    (R13, 'qwen3.8-flash-next', '3072122a', 'Другая миссия: «сохрани не меньше 30 единиц заряда»',
     'Рядом кандидат с уверенностью 80%, до него полторы единицы заряда. Модель считает, что с ним на базу привезёт меньше 30, и отказывается.'),
    (MAIN, 'DeepSeek-V4-Flash', '9b77a487', 'Модель ошиблась: пустой ответ и повторный запрос',
     'Первый ответ не прошёл проверку. Робот отправил модели текст ошибки и получил годный план со второй попытки.'),
]


def find(root, model, prefix):
    return json.loads(next((root / model).glob(prefix + '*.json')).read_text(encoding='utf-8'))


def state_of(text):
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def pretty(text):
    try:
        return json.dumps(json.loads(text[text.index('{'):text.rindex('}') + 1]), ensure_ascii=False, indent=1)
    except ValueError:
        return text


def targets(st, ans):
    """Подцели ответа → точки на карте: [(номер, x, y, подпись)]."""
    by = {p['id']: p for p in st.get('candidates', []) + st.get('explore_points', [])}
    out = []
    for i, g in enumerate(ans.get('subgoals', []), 1):
        if g['type'] == 'return_base':
            out.append((i, st['base']['x'], st['base']['y'], 'вернуться на базу'))
        elif g['type'] == 'goto':
            out.append((i, g['x'], g['y'], f"доехать до точки ({num(g['x'])}; {num(g['y'])})"))
        elif g.get('target') in by:
            p = by[g['target']]
            out.append((i, p['x'], p['y'], f"{GOAL[g['type']]}: {g['target']}"))
    return out


def state_map(st, ans, ban=None):
    """Карта только из того, что написано в сообщении модели, плюс её ответ стрелками."""
    p = [f'<svg viewBox="0 0 {MW:.0f} {MH:.0f}" role="img" aria-label="Что известно модели">',
         f'<use href="#arena" transform="translate({-XMIN * K:.1f},{YMAX * K:.1f}) scale({K},{-K})"/>']
    if ban is not None:
        p.append(f'<rect x="{sx(ban):.1f}" y="0" width="{MW - sx(ban):.1f}" height="{MH:.0f}" class="banarea" '
                 f'data-tip="Запрещено текстом миссии: x больше {num(ban)} м"/>')
    for z in st.get('soil_zones', []):
        p.append(f'<circle cx="{sx(z["x"]):.1f}" cy="{sy(z["y"]):.1f}" r="{max(z.get("radius", 0.25), 0.18) * K:.1f}" class="soil" '
                 f'data-tip="{z["id"]}: дорогой грунт ×{num(z["mult"])}, измерено на {num(z["evidence_m"])} м пути, {z["status"]}"/>'
                 f'<text x="{sx(z["x"]):.1f}" y="{sy(z["y"]) + 3.5:.1f}" class="zl">×{num(z["mult"])}</text>')
    for z in st.get('hazards', []):
        p.append(f'<circle cx="{sx(z["x"]):.1f}" cy="{sy(z["y"]):.1f}" r="{z["radius"] * K:.1f}" class="haz" data-tip="{z["id"]}: опасная зона"/>'
                 f'<text x="{sx(z["x"]):.1f}" y="{sy(z["y"]) + 3.5:.1f}" class="zl">!</text>')
    x0, y0 = st['pose']['x'], st['pose']['y']
    prev = (x0, y0)
    for i, x, y, word in targets(st, ans):
        if math.dist(prev, (x, y)) > 0.05:
            p.append(f'<line x1="{sx(prev[0]):.1f}" y1="{sy(prev[1]):.1f}" x2="{sx(x):.1f}" y2="{sy(y):.1f}" class="plan" marker-end="url(#arrow)"/>')
        prev = (x, y)
    for e in st.get('explore_points', []):
        ok = '' if e['feasible'] else ', заряда не хватит'
        p.append(f'<rect x="{sx(e["x"]) - 5:.1f}" y="{sy(e["y"]) - 5:.1f}" width="10" height="10" rx="2" class="ep" '
                 f'data-tip="{e["id"]}: точка разведки, не проверено {num(100 * e["unseen_share"], 0)}% окрестности, дорога {num(e["cost_to"])} ед.{ok}"/>'
                 f'<text x="{sx(e["x"]):.1f}" y="{sy(e["y"]) - 9:.1f}" class="idl">{e["id"]}</text>')
    for c in st.get('candidates', []):
        ok = '' if c['feasible'] else ', заряда не хватит'
        p.append(f'<circle cx="{sx(c["x"]):.1f}" cy="{sy(c["y"]):.1f}" r="6.5" class="cand" '
                 f'data-tip="{c["id"]}: возможно, образец; уверенность {num(100 * c["confidence"], 0)}%, дорога {num(c["cost_to"])} ед., оттуда домой {num(c["cost_back"])} ед.{ok}"/>'
                 f'<text x="{sx(c["x"]):.1f}" y="{sy(c["y"]) - 10:.1f}" class="idl">{c["id"]} · {num(100 * c["confidence"], 0)}%</text>')
    for i, x, y, word in targets(st, ans):
        p.append(f'<g data-tip="Подцель {i}: {esc(word)}"><circle cx="{sx(x) + 11:.1f}" cy="{sy(y) + 11:.1f}" r="8" class="stepc"/>'
                 f'<text x="{sx(x) + 11:.1f}" y="{sy(y) + 15:.1f}" class="stepn">{i}</text></g>')
    bx, by_ = st['base']['x'], st['base']['y']
    p.append(f'<rect x="{sx(bx) - 5:.1f}" y="{sy(by_) - 5:.1f}" width="10" height="10" rx="2" class="base" data-tip="База"/>')
    p.append(f'<circle cx="{sx(x0):.1f}" cy="{sy(y0):.1f}" r="6" class="robot" data-tip="Робот: ({num(x0)}; {num(y0)})"/></svg>')
    return ''.join(p)


def facts(st):
    s = st['sensor']
    rows = [('Миссия', esc(st['mission'])),
            ('Повод', TRIGGER.get(st['trigger'], st['trigger']) + f' <code>{st["trigger"]}</code>'),
            ('Время', f"{num(st['time_s'], 0)} с из {num(st['time_limit_s'], 0)}"),
            ('Заряд', f"{num(st['battery'])} из {num(st['battery_start'], 0)}; до базы нужно {num(st['return_cost'])}"),
            ('Образцы', f"собрано {st['samples']['collected']} из {st['samples']['total']}"),
            ('Датчик', f"показание {num(s['reading'], 2)}, " + ('<b>неисправен</b>' if s['status'] != 'ok' else 'исправен')),
            ('Кандидаты', ', '.join(f"{c['id']} ({num(100 * c['confidence'], 0)}%)" for c in st.get('candidates', [])) or 'нет'),
            ('Точки разведки', ', '.join(e['id'] for e in st.get('explore_points', [])) or 'нет')]
    if st.get('penalties'):
        rows.append(('Штрафы', f"всего {st['penalties']['total']}"))
    if st.get('alarms'):
        rows.append(('Тревоги', '<br>'.join(esc(a) for a in st['alarms'])))
    if st.get('recent_events'):
        rows.append(('События', '<br>'.join(esc(json.dumps(e, ensure_ascii=False) if not isinstance(e, str) else e) for e in st['recent_events'][-3:])))
    return '<table class="facts">' + ''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in rows) + '</table>'


def answer(st, ans):
    steps = ''.join(f'<li><span class="stepb">{i}</span>{esc(word)}</li>' for i, _, _, word in targets(st, ans))
    hyp = ''.join(f'<li><b>{esc(h["statement"])}</b><br><span class="test">Проверка: {esc(h["test"])}</span></li>' for h in ans.get('hypotheses', []))
    return (f'<div class="said"><div class="lbl">Почему (поле reasoning)</div><p>{esc(ans.get("reasoning", ""))}</p>'
            f'<div class="lbl">Подцели</div><ol class="steps">{steps}</ol>'
            + (f'<div class="lbl">Гипотезы модели</div><ul class="hyp">{hyp}</ul>' if hyp else
               '<div class="lbl">Гипотезы модели</div><p class="muted">нет</p>') + '</div>')


def card(n, root, model, prefix, title, lead):
    d = find(root, model, prefix)
    msgs = d['messages']
    st = state_of(msgs[1]['content'])
    ans = json.loads(pretty(d['text']))
    ban = 0.5 if 'правую половину' in st['mission'] else None
    retry = ''
    if len(msgs) > 2:
        retry = (f'<div class="retry"><div class="pane bad"><div class="who">Первый ответ модели — не принят</div><pre><code>{esc(msgs[2]["content"])}</code></pre></div>'
                 f'<div class="pane"><div class="who">Робот → модель: сообщение об ошибке</div><pre><code>{esc(msgs[3]["content"])}</code></pre></div></div>')
    usage = d.get('usage') or {}
    meta = (f"{model} · ответ за {num(d['latency_ms'] / 1000, 0)} с · {usage.get('prompt_tokens', '?')} токенов на входе, "
            f"{usage.get('completion_tokens', '?')} на выходе")
    return f'''<section id="c{n}"><div class="num">Ситуация {n}</div><h2>{title}</h2><p class="tag">{lead}</p>
<div class="duo"><figure class="map">{state_map(st, ans, ban)}<figcaption><b>Карта по сообщению модели</b>
<span>Только то, что робот ей написал. Стрелки и номера — её ответ.</span></figcaption></figure>
<div><h3>Что модель узнала</h3>{facts(st)}<h3>Что она ответила</h3>{answer(st, ans)}</div></div>
<h3>Дословно</h3>
<div class="raw"><div class="pane"><div class="who">Робот → модель</div><pre><code>{esc(pretty(msgs[1]["content"]))}</code></pre></div>
<div class="pane"><div class="who">Модель → робот</div><pre><code>{esc(pretty(d["text"]))}</code></pre></div></div>
{retry}<p class="where">{meta} · файл кэша <code>{prefix}…</code></p></section>'''


EXTRA_CSS = '''
main{max-width:1120px}
.duo{display:grid;grid-template-columns:minmax(0,430px) minmax(0,1fr);gap:22px;align-items:start;margin-top:10px}
.duo h3:first-child{margin-top:0}
@media (max-width:860px){.duo,.raw,.retry{grid-template-columns:1fr !important}}
table.facts th{width:132px;white-space:nowrap;color:var(--ink2);font-weight:600;font-size:13.5px}
table.facts td{font-size:14.5px}
.said{background:var(--code);border-radius:10px;padding:10px 14px}
.said p{margin:2px 0 10px}
.lbl{font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--ink2);margin-top:4px}
ol.steps{list-style:none;padding:0;margin:4px 0 10px}
ol.steps li{display:flex;align-items:center;gap:8px;margin:4px 0}
.stepb{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;background:var(--s2);color:#fff;font-size:12px;font-weight:700;flex:none}
ul.hyp{margin:4px 0 2px;padding-left:18px;font-size:14.5px}
.test{color:var(--ink2)}
.muted{color:var(--muted)}
.raw,.retry{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:6px}
.retry{margin-top:14px}
.pane .who{font-size:13px;font-weight:700;color:var(--ink2);margin-bottom:4px}
.pane pre{margin:0;max-height:430px;overflow:auto}
.pane pre code{white-space:pre-wrap;word-break:break-word;font-size:12.5px}
.pane.bad pre{border:1.5px solid var(--crit)}
.prompt pre{max-height:520px;overflow:auto}
.prompt pre code{white-space:pre-wrap}
.robot{fill:var(--s1);stroke:var(--surface);stroke-width:2.5}
.cand{fill:var(--surface);stroke:var(--ink);stroke-width:2;stroke-dasharray:3 2}
.ep{fill:var(--surface);stroke:var(--muted);stroke-width:1.5}
.idl{fill:var(--ink2);font-size:10.5px;text-anchor:middle}
.plan{stroke:var(--s2);stroke-width:2.5;stroke-linecap:round}
.stepc{fill:var(--s2);stroke:var(--surface);stroke-width:2}
.stepn{fill:#fff;font-size:11px;font-weight:700;text-anchor:middle;pointer-events:none}
.banarea{fill:color-mix(in srgb,var(--crit) 13%,transparent)}
.toc2 a{display:block;padding:3px 0;text-decoration:none}
i.sq.cand{border:1.5px dashed var(--ink);border-radius:50%;background:var(--surface)}
i.sq.ep{border:1.5px solid var(--muted);background:var(--surface)}
i.dot.robot{background:var(--s1)}
'''


def main():
    cards = ''.join(card(i, *c) for i, c in enumerate(CARDS, 1))
    toc = ''.join(f'<a href="#c{i}">{i}. {c[3]}</a>' for i, c in enumerate(CARDS, 1))
    system = find(*CARDS[0][:3])['messages'][0]['content']
    page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Что получает языковая модель и что она отвечает</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><symbol id="arena" overflow="visible">
<path class="floor" d="{ARENA_D}"/></symbol><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
<path d="M0 0L10 5L0 10z" style="fill:var(--s2)"/></marker></defs></svg>
<main>
<div class="num">DID Hack · робот-исследователь</div>
<h1>Что получает языковая модель и что она отвечает</h1>
<p class="lead">Десять настоящих обменов из прогонов робота. В каждом: ситуация на карте, что робот написал модели, что она
ответила — и оба текста дословно.</p>

<section><h2>Как это устроено</h2>
<ul>
<li><b>Когда робот обращается к модели.</b> Не постоянно, а по поводу: старт, найден кандидат, образец собран, кандидат пропал,
датчик сбоит, штраф, заряд у порога. Между поводами робот едет сам.</li>
<li><b>Что он ей пишет.</b> Два сообщения. Первое — постоянная инструкция (она ниже, целиком). Второе — состояние на эту секунду
в виде JSON: заряд, сколько собрано, кандидаты, точки разведки, известные грунты и зоны, датчик, тревоги.</li>
<li><b>Что она отвечает.</b> Один JSON: рассуждение, гипотезы и список из 1–3 подцелей. Подцель — это «куда ехать дальше», а не
команда колёсам.</li>
<li><b>Чего модель не видит.</b> Где на самом деле лежат образцы и какие грунты ещё не найдены. На картах ниже нарисовано только
то, что было в сообщении.</li>
<li><b>Что делает код с ответом.</b> Проверяет: цель есть в списке, заряда хватает, формат верный. Не прошло — отправляет модели
текст ошибки (ситуация 10). Потом едет к первой подцели обычным кодом.</li>
</ul>
<div class="legend"><span class="key"><i class="dot robot"></i>робот</span><span class="key"><i class="sq cand"></i>кандидат: возможно, образец, и уверенность</span>
<span class="key"><i class="sq ep"></i>точка разведки</span><span class="key"><i class="sq soil"></i>известный дорогой грунт</span>
<span class="key"><i class="sq haz"></i>известная опасная зона</span><span class="key"><i class="sq base"></i>база</span>
<span class="key"><span class="stepb">1</span>подцели из ответа модели, по порядку</span></div>
<h3>Ситуации</h3><div class="toc2">{toc}</div>
</section>

<section class="prompt"><h2>Постоянная инструкция модели</h2>
<p class="tag">Первое сообщение каждого обмена, одно и то же. Меняется только строка «Миссия». Файл: <code>did/prompts/</code>.</p>
<pre><code>{esc(system)}</code></pre></section>

{cards}
</main><div id="tip" hidden></div><script>{JS}</script></body></html>'''
    OUT.write_text(page, encoding='utf-8')
    print(f'{OUT.name}: {len(page.encode("utf-8")) / 1024:.0f} КБ')


if __name__ == '__main__':
    main()
