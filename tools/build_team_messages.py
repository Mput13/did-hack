#!/usr/bin/env python3
"""Страница «Как два робота разговаривают»: team_messages.html в корне репозитория.

    /usr/local/bin/python3 tools/build_team_messages.py      # нужен только stdlib

Берёт запись прогона двух роботов в Gazebo (runs/M1gz/team/medium-3.json.gz): канал сообщений team.messages и
журналы обоих роботов — и показывает сами строки JSON рядом с тем, что записал у себя получатель.
Оформление то же, что у llm_calls.html (стили берутся из него).
"""
import gzip
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / 'runs' / 'M1gz' / 'team' / 'medium-3.json.gz'
OUT = ROOT / 'team_messages.html'

# вид сообщения -> (что в нём, что делает получатель, функция-обработчик в did/team.py, группа цвета)
KINDS = {
    'obs': ('раз в секунду: положение, заряд, показания датчика образцов, пройденные отрезки с расходом, текущая цель',
            'вносит показания напарника в свою карту вероятностей как свои; дополняет карту грунта; объезжает напарника',
            '_rx_obs', 'obs'),
    'claim': ('«эту цель беру я»: вид цели (образец или разведка), точка, сколько заряда до неё',
              'туда не едет; разведку рядом уступает тому, кто едет за образцом; за один образец едет тот, кому дешевле',
              '_rx_claim', 'claim'),
    'collected': ('где собран образец и сколько осталось на арене',
                  'убирает образец со своей карты, бросает цель, если ехал туда же, строит новый план', '_rx_collected', 'collected'),
    'sector': ('граница участков и чья какая сторона — либо «вся арена моя» — и причина',
               'разведку на чужом участке берёт неохотно', '_rx_sector', 'sector'),
    'soil': ('гипотеза «здесь дорогой грунт» и во сколько раз', 'записывает в журнал', '_rx_soil', 'soil'),
    'soil_changed': ('«расход перестал сходиться с оценкой»', 'старым данным о грунте в этом месте больше не верит',
                     '_rx_soil_changed', 'soil'),
    'hazard': ('точка штрафа опасной зоны, курс и подход', 'заносит зону в свою карту и объезжает, не получив штрафа сам',
               '_rx_hazard', 'status'),
    'status': ('еду на базу / на базе / встал', 'оставшийся забирает всю арену', '_rx_status', 'status'),
    'hello': ('я на связи, вот моё место на базе', 'запоминает напарника; робот с меньшим именем предлагает деление арены',
              '_rx_hello', 'status'),
}
GROUPS = {'claim': ('var(--s1)', 'заявка на цель'), 'collected': ('var(--s3)', 'образец собран'),
          'sector': ('var(--s2)', 'деление арены'), 'soil': ('var(--warn)', 'грунт'), 'status': ('var(--muted)', 'на связи, состояние')}

E = html.escape


def ru(x, d=1):
    return f'{x:.{d}f}'.replace('.', ',')


def js(m, wide=False):
    """Сообщение как в канале: одна строка JSON; длинные массивы — по строке на поле."""
    if not wide:
        return E(json.dumps(m, ensure_ascii=False))
    lines = ['{'] + [f'  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)},' for k, v in m.items()]
    lines[-1] = lines[-1].rstrip(',')
    return E('\n'.join(lines + ['}']))


def main():
    with gzip.open(RECORD) as f:
        rec = json.load(f)
    msgs = rec['team']['messages']
    robots = {r['name']: r for r in rec['robots']}
    res = rec['result']
    counts = Counter(m['type'] for m in msgs)
    t0, t1 = msgs[0]['t'], max(m['t'] for m in msgs)

    def other(name):
        return next(n for n in robots if n != name)

    def heard(name, a, b, *tags):
        """Что робот name записал в журнал между a и b (по меткам записей)."""
        out = []
        for j in robots[name]['journal']:
            tag = (j.get('data') or {}).get('tag')
            if a <= j['t'] <= b and (tag in tags or (not tags and tag)):
                out.append(j)
        return out

    def first(kind, sender=None, after=0.0, where=lambda m: True):
        return next(m for m in msgs if m['type'] == kind and m['t'] >= after and (sender is None or m['from'] == sender) and where(m))

    def card(n, title, tag, items, say=None):
        """items: [(сообщение, wide, [записи журнала получателя], пояснение)]"""
        parts = [f'<section id="e{n}"><div class="num">Эпизод {n}</div><h2>{E(title)}</h2><p class="tag">{tag}</p>']
        for m, wide, lines, note in items:
            to = other(m['from'])
            parts.append(f'<div class="msg"><div class="who"><b>{m["from"]}</b> → {to} · {ru(m["t"], 2)} с · '
                         f'<code>{m["type"]}</code> · разбирает <code>{KINDS[m["type"]][2]}</code></div>'
                         f'<pre><code>{js(m, wide)}</code></pre>')
            if lines:
                parts.append(f'<div class="said"><div class="lbl">Что {to} записал в свой журнал</div>'
                             + ''.join(f'<p><span class="t">{ru(j["t"])} с</span> {E(j["text"])}</p>' for j in lines) + '</div>')
            if note:
                parts.append(f'<p class="cap">{note}</p>')
            parts.append('</div>')
        if say:
            parts.append(f'<div class="say"><b>Что это значит.</b> {say}</div>')
        parts.append('</section>')
        return ''.join(parts)

    # ---- шкала прогона: кто и когда что сказал
    W, X0, X1 = 920, 70, 900

    def X(t):
        return X0 + (t - t0) / (t1 - t0) * (X1 - X0)

    lane = {'tb1': 52, 'tb2': 122}
    svg = [f'<svg class="chart" viewBox="0 0 {W} 180" role="img" aria-label="Сообщения двух роботов по времени прогона">']
    for name, y in lane.items():
        svg.append(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" class="grid"/><text x="{X0 - 12}" y="{y + 4}" class="rl">{name}</text>')
    for t in range(40, int(t1) + 1, 10):
        svg.append(f'<line x1="{X(t):.1f}" y1="22" x2="{X(t):.1f}" y2="150" class="grid"/><text x="{X(t):.1f}" y="168" class="tick">{t} с</text>')
    for m in msgs:
        y, x = lane[m['from']], X(m['t'])
        if m['type'] == 'obs':
            svg.append(f'<line x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" y2="{y + 5}" stroke="var(--muted)" stroke-width="1" opacity=".45"/>')
    taken = {name: [] for name in lane}              # точки, попавшие в одно место, разводим по высоте
    for m in msgs:
        if m['type'] != 'obs':
            x = X(m['t'])
            level = next(k for k in (0, -1, 1, -2, 2, -3, 3) if all(abs(x - px) > 11 or pk != k for px, pk in taken[m['from']]))
            taken[m['from']].append((x, level))
            y = lane[m['from']] + level * 11
            color = GROUPS[KINDS[m['type']][3]][0]
            svg.append(f'<circle cx="{x:.1f}" cy="{y}" r="6.5" fill="{color}" stroke="var(--surface)" stroke-width="2">'
                       f'<title>{ru(m["t"], 2)} с · {m["from"]} · {m["type"]}: {E(json.dumps({k: v for k, v in m.items() if k not in ("t", "from", "type")}, ensure_ascii=False))}</title></circle>')
    svg.append('</svg>')
    legend = ('<div class="legend">' + ''.join(f'<span class="key"><i class="dot" style="background:{c}"></i>{E(label)}</span>' for c, label in GROUPS.values())
              + '<span class="key"><i class="ln" style="background:var(--muted);width:2px;height:10px"></i>ежесекундная сводка obs</span></div>')

    # ---- эпизоды
    hello = first('hello', 'tb2')
    sector0 = first('sector')
    claim1, claim2 = first('claim', 'tb1', where=lambda m: m['kind'] == 'investigate'), first('claim', 'tb2', where=lambda m: m['kind'] == 'investigate')
    obs = first('obs', 'tb2', after=36.0)
    soil, changed = first('soil', 'tb1'), first('soil_changed', 'tb1')
    got = first('collected', 'tb2')
    yield_line = heard('tb2', 60, 80, 'yield')[0]
    # Заявка ходит двумя путями: отдельным сообщением claim и полем claim в ежесекундной сводке. Здесь сработала сводка.
    claim_y = [m for m in msgs if m['from'] == 'tb1' and m['t'] <= yield_line['t'] + 0.05
               and (m['type'] == 'claim' or m['type'] == 'obs' and m['claim'])][-1]
    claim_back = first('claim', 'tb2', after=yield_line['t'] - 0.05)
    mine_before = [m for m in msgs if m['from'] == 'tb2' and m['type'] == 'obs' and m['t'] < claim_y['t'] and m['claim']][-1]['claim']
    sector1 = first('sector', 'tb2')
    last = [m for m in msgs if m['type'] == 'collected'][-1]
    status = first('status', 'tb2', after=last['t'])
    sector2 = first('sector', 'tb1', after=last['t'])

    cards = [
        card(1, 'Старт: познакомились и поделили арену',
             'Первое, что делает робот, — сообщает, что он на связи. Робот с меньшим именем сразу предлагает, кому какая половина.',
             [(hello, False, heard('tb1', 35, 35.3, None)[:0] or [j for j in robots['tb1']['journal'] if 'На связи' in j['text']][:1], ''),
              (sector0, False, heard('tb2', sector0['t'], sector0['t'] + 0.5, 'sector'),
               'В <code>line</code> — прямая: точка <code>c</code> и направление <code>n</code>. В <code>owners</code> — чья сторона. '
               'Граница проведена так, чтобы на каждой половине была половина оставшейся вероятности образцов.')],
             'Главного робота нет: код у обоих одинаковый. Деление предлагает тот, чьё имя меньше по алфавиту, — иначе они предложили бы два разных деления одновременно.'),
        card(2, 'Заявка на цель: «эту беру я»',
             'Как только робот выбрал, куда ехать, он сообщает точку и цену пути в единицах заряда.',
             [(claim1, False, [], ''), (claim2, False, [], 'Цели разные, спорить не о чем: в журналах пусто. Каждый просто запомнил цель напарника и не выбирает её.')]),
        card(3, 'Ежесекундная сводка: показания датчика',
             f'Таких сообщений больше всего — {counts["obs"]} из {len(msgs)}. В них главное, ради чего нужна связь.',
             [(obs, True, [],
               'В <code>readings</code> каждая пятёрка — одно показание датчика образцов: <code>[x, y, z, σ, время]</code>, то есть где стоял робот, '
               'что показал датчик, какой у датчика шум и когда это снято. Получатель делает с каждой пятёркой то же, что со своим показанием: '
               '<code>self.belief.update(x, y, z, sigma)</code> — пересчитывает карту вероятностей по правилу Байеса. '
               'В журнал это не пишется: так происходит раз в секунду. '
               'В <code>soil</code> — отрезки пути и во сколько раз расход на них выше обычного; в <code>claim</code> — текущая цель.')],
             'Два робота видят арену с двух точек и складывают показания в одну и ту же карту. Кольца от двух датчиков пересекаются быстрее, чем от одного.'),
        card(4, 'Грунт: гипотеза одного — предупреждение другому',
             'Робот tb1 заметил перерасход заряда и сообщил об этом дважды: сначала гипотезу, потом — что прежние оценки больше не годятся.',
             [(soil, False, heard('tb2', soil['t'] - 0.1, soil['t'] + 0.15, 'partner_soil'), ''),
              (changed, False, heard('tb2', changed['t'] - 0.1, changed['t'] + 0.3, 'partner_soil_changed'), '')]),
        card(5, 'Образец собран',
             'Самое короткое и самое важное сообщение: место сбора и сколько образцов осталось на арене.',
             [(got, False, heard('tb1', got['t'], got['t'] + 0.5, 'partner_collected'),
               'Получатель убирает этот образец со своей карты (<code>self.belief.collected(x, y)</code>): показания, которые его «слышали», больше не означают «образец где-то рядом». '
               'Показания, снятые до сбора, но пришедшие после этого сообщения, выбрасываются — иначе на карте появилось бы ложное кольцо.')],
             'Поле <code>left</code> — общий счётчик команды. Когда он доходит до нуля, оба знают, что искать больше нечего.'),
        card(6, 'Цели пересеклись: один уступает',
             f'Робот tb2 ехал на разведку в точку ({ru(mine_before["x"], 2)}; {ru(mine_before["y"], 2)}). В очередной сводке tb1 сообщил, что едет '
             f'за образцом в ({ru(claim_y["claim"]["x"], 2)}; {ru(claim_y["claim"]["y"], 2)}) — это рядом.',
             [(claim_y, True, [yield_line],
               'Заявка пришла не отдельным сообщением, а полем <code>claim</code> в ежесекундной сводке: цель повторяется каждую секунду вместе с остатком пути до неё. '
               'Правило в <code>_rx_claim</code>: разведку рядом с образцом, за которым едет напарник, робот уступает. '
               'В журнале это записано коротко — «цель взял tb1»: на самом деле tb1 едет за образцом в 0,9 м от точки разведки tb2. '
               'Если же оба едут за одним и тем же образцом, сравниваются пары «цена, имя»: <code>(m["cost"], p["name"]) &lt; (cost, self.name)</code> — '
               'едет тот, кому дешевле, при равной цене — у кого имя меньше.'),
              (claim_back, False, [], 'Ответ tb2 через шесть сотых секунды: новая точка разведки, в стороне от цели напарника.')],
             'Договариваться не нужно: оба считают одно и то же правило по одним и тем же числам и приходят к одному ответу.'),
        card(7, 'Передел арены по ходу прогона',
             'Свой участок проверен, а на чужом по карте ещё остался образец — робот предлагает поделить оставшееся заново.',
             [(sector1, False, heard('tb1', sector1['t'], sector1['t'] + 0.5, 'sector'), '')]),
        card(8, 'Конец: последний образец и дорога домой',
             'Счётчик дошёл до нуля — три сообщения за десятую долю секунды.',
             [(last, False, heard('tb2', last['t'], last['t'] + 0.3, 'partner_collected'), ''),
              (status, False, heard('tb1', status['t'], status['t'] + 0.3, 'partner_returning'), ''),
              (sector2, False, heard('tb2', sector2['t'], sector2['t'] + 0.3, 'sector'), '')]),
    ]

    # ---- таблицы
    kinds_rows = ''.join(f'<tr><td class="c"><code>{k}</code></td><td class="c">{counts.get(k, 0)}</td><td>{E(v[0])}</td><td>{E(v[1])}</td></tr>'
                         for k, v in KINDS.items())
    log_rows = ''.join(f'<tr><td class="c">{ru(m["t"], 2)}</td><td class="c">{m["from"]}</td><td class="c"><code>{m["type"]}</code></td>'
                       f'<td><code>{E(json.dumps({k: v for k, v in m.items() if k not in ("t", "from", "type")}, ensure_ascii=False))}</code></td></tr>'
                       for m in msgs if m['type'] != 'obs')
    plans = {n: Counter(p['source'] for p in r.get('plans', [])) for n, r in robots.items()}
    llm_calls = sum(len(r.get('llm') or []) for r in robots.values())

    style = (ROOT / 'llm_calls.html').read_text(encoding='utf-8')
    style = style[style.index('<style>') + 7:style.index('</style>')]
    extra = """
.msg{margin:16px 0 0}.who{font-size:13.5px;color:var(--ink2);margin:0 0 4px}.who b{color:var(--ink)}
.msg pre{margin:4px 0 8px}.msg pre code{white-space:pre-wrap;word-break:break-word}
.said p{margin:4px 0}.said .t{color:var(--muted);font-variant-numeric:tabular-nums;margin-right:6px}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:14px 0}
.flow div{flex:1 1 150px;background:var(--code);border-radius:10px;padding:10px 12px;font-size:14px}
.flow div b{display:block;font-size:13px;color:var(--accent);margin-bottom:2px}
.slide{background:#F8F9FC;color:#182B45;border:1px solid var(--line);border-radius:12px;padding:22px 26px;margin:14px 0;font-family:Inter,-apple-system,Arial,sans-serif}
.slide h4{font-size:22px;font-weight:400;margin:0 0 4px}.slide .sub{color:#40536B;font-size:13.5px;margin:0 0 14px}
.slide .row{display:flex;gap:12px;align-items:center;margin:8px 0}
.slide .json{flex:1 1 48%;background:#DCE8FF;border-radius:14px;padding:9px 14px;font:12px/1.45 ui-monospace,Menlo,monospace;word-break:break-word}
.slide .json.v{background:#E7DFF6}.slide .act{flex:1 1 48%;font-size:13.5px}.slide .act small{display:block;color:#40536B}
.slide .foot{background:#fff;border-radius:14px;padding:9px 14px;font-size:12.5px;color:#40536B;margin-top:12px}
"""
    def mock_row(m, line, violet=False):
        body = {k: v for k, v in m.items() if k != 't'}
        text = json.dumps(body, ensure_ascii=False)
        if m['type'] == 'obs':                       # сводка длинная: на макете — только отправитель и заявка
            text = json.dumps({'from': m['from'], 'type': 'obs', 'claim': m['claim']}, ensure_ascii=False)[:-1] + ', …}'
        return (f'<div class="row"><div class="json{" v" if violet else ""}">{E(text)}</div><div>→</div>'
                f'<div class="act"><small>{other(m["from"])} записал в журнал</small>{E(line)}</div></div>')

    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Как два робота разговаривают: настоящие сообщения одного прогона</title><style>{style}{extra}</style></head>
<body><main>
<div class="num">Два робота · прогон в Gazebo</div>
<h1>Как два робота разговаривают: настоящие сообщения одного прогона</h1>
<p class="lead">Запись прогона в Gazebo с двумя TurtleBot3 (средний уровень, сценарий 3). За {ru(res['time'], 0)} секунд роботы сказали друг другу
{len(msgs)} сообщений. Ниже — сами строки JSON из записи и то, что робот-получатель после них записал в свой журнал.</p>

<section id="short"><div class="num">Коротко</div><h2>Три вещи, которые важно понимать</h2>
<div class="answer"><ul>
<li><b>Языковой модели здесь нет.</b> Сообщения разбирает обычный код робота (<code>did/team.py</code>): у каждого вида сообщения своя функция-обработчик.
В этом прогоне планы строило правило ({E(', '.join(f'{n}: {sum(c.values())} планов' for n, c in plans.items()))}), обращений к языковой модели — {llm_calls}.</li>
<li><b>Общей памяти нет.</b> Есть один канал ROS 2 — топик <code>/did/team</code>, тип <code>std_msgs/String</code>. Одно сообщение — одна строка JSON.
Оба робота в него и пишут, и читают; свои сообщения робот отбрасывает.</li>
<li><b>Раз в секунду — сводка, остальное — по событию.</b> Сводка <code>obs</code> несёт показания датчика; заявка на цель, сбор, грунт и деление арены уходят сразу, как случились.</li>
</ul></div>
<p class="note">Итог этого прогона: собрано {res['samples_collected']} из {res['samples_total']}, оба вернулись, столкновений {res['collisions']}, штрафов {res['penalties']};
наименьшее расстояние между роботами {ru(res['min_gap_m'], 2)} м. Это один прогон — показ, что схема работает в Gazebo. Числа «−26 с» и «80 из 80» со слайда — из серии в быстром симуляторе.</p></section>

<section id="how"><div class="num">Устройство</div><h2>Путь одного сообщения</h2>
<p class="tag">От решения одного робота до изменения в голове другого — пять шагов, без посредников.</p>
<div class="flow"><div><b>1. Событие</b>tb1 выбрал цель, собрал образец или прошла секунда</div><div><b>2. Словарь → JSON</b><code>channel.post(имя, t, вид, **поля)</code></div>
<div><b>3. Топик ROS 2</b>строка уходит в <code>/did/team</code></div><div><b>4. Приём</b>tb2 кладёт её во входящие; своё — отбрасывает</div>
<div><b>5. Обработчик</b>на следующем такте (0,1 с) вызывается <code>_rx_&lt;вид&gt;</code>: меняет карту и план</div></div>
<h3>Отправка и приём (did/ros_team.py)</h3>
<pre><code>{E('''def post(self, sender, t, kind, /, **body):
    msg = {'t': round(float(t), 2), 'from': sender, 'type': kind, **body}
    self._pub.publish(String(data=_dumps(msg)))        # одна строка JSON в топик /did/team

def _on_msg(self, msg):
    m = json.loads(msg.data)
    if m.get('from') == self.name:                     # своё сообщение — не читаем
        return
    self._inbox.append(m)''')}</code></pre>
<h3>Разбор входящих (did/team.py)</h3>
<pre><code>{E('''def _receive(self, obs):
    for m in self.channel.read(self.name):
        p = self.partners.setdefault(m['from'], {...})   # всё, что я знаю о напарнике
        handler = getattr(self, '_rx_' + m['type'], None) # obs -> _rx_obs, claim -> _rx_claim, ...
        if handler:
            handler(m, p, obs)''')}</code></pre>
<p class="cap">В быстром симуляторе вместо топика — список в памяти с тем же интерфейсом (<code>TeamChannel</code>), остальной код тот же.</p></section>

<section id="kinds"><div class="num">Словарь</div><h2>Какие бывают сообщения</h2>
<p class="tag">У всех есть три общих поля: <code>t</code> — время, <code>from</code> — кто сказал, <code>type</code> — вид.</p>
<table class="sum"><tr><th>Вид</th><th>В этом прогоне</th><th>Что внутри</th><th>Что делает получатель</th></tr>{kinds_rows}</table>
<p class="cap">Сообщения <code>hazard</code> в этом прогоне не было: ни один робот не получил штрафа опасной зоны.</p></section>

<section id="timeline"><div class="num">Весь прогон</div><h2>Кто и когда говорил</h2>
<p class="tag">Верхняя строка — tb1, нижняя — tb2. Серые чёрточки — ежесекундные сводки, цветные точки — сообщения по событию. Наведите на точку, чтобы увидеть содержимое.</p>
{''.join(svg)}{legend}
<p class="cap">Роботы начинают говорить на 35-й секунде: до этого они стоят, пока модель в Gazebo не успокоится после появления в мире.</p></section>

{''.join(cards)}

<section id="log"><div class="num">Лента</div><h2>Все сообщения по событию</h2>
<p class="tag">{len(msgs) - counts['obs']} сообщений, кроме ежесекундных сводок. Общие поля вынесены в столбцы.</p>
<details class="tbl" open><summary>Показать или скрыть</summary><table><tr><th>Время, с</th><th>Кто</th><th>Вид</th><th>Поля</th></tr>{log_rows}</table></details></section>

<section id="qa" class="qa"><div class="num">К защите</div><h2>Что могут спросить</h2>
<details open><summary>Роботы общаются через языковую модель?</summary><p>Нет. Сообщение — словарь с числами, его разбирает код. Языковая модель в проекте выбирает подцели одного робота по заданию словами; к разговору двух роботов она отношения не имеет.</p></details>
<details open><summary>Что будет, если сообщение потеряется или связь пропадёт?</summary><p>Если напарник молчит дольше 5 секунд, робот считает, что его нет, и забирает всю арену себе. Отдельные потери и задержки сообщений мы не моделировали: в быстром симуляторе сообщение приходит в тот же такт, в Gazebo оно идёт по настоящему топику ROS 2, но это один прогон, а не серия.</p></details>
<details open><summary>Кто из роботов главный?</summary><p>Никто. Код одинаковый. Если оба едут за одним образцом, едет тот, кому дешевле, при равной цене — у кого имя меньше. Если один едет за образцом, а второй собирался разведывать рядом, уступает разведчик. Деление арены на старте предлагает робот с меньшим именем.</p></details>
<details open><summary>Откуда известно, что выигрыш даёт именно связь, а не второй робот?</summary><p>Сравнение — с такой же парой роботов без канала: те же сценарии, те же две батареи. Со связью прогон короче на 26 секунд, заряда уходит на четверть меньше, на базу вернулись оба в 80 прогонах из 80 против 75. Сравнивать счёт команды со счётом одного робота нельзя: у команды два заряда.</p></details>
<details open><summary>Чем робот делится, а чем нет?</summary><p>Делится измерениями и намерениями: показания датчика, расход на пройденных отрезках, цель, сбор, штраф. Готовую карту вероятностей не пересылает — каждый строит свою из общих показаний.</p></details>
</section>

<section id="slide"><div class="num">Набросок</div><h2>Как это можно показать на слайде</h2>
<p class="tag">Сейчас на слайде пять общих строк «что сообщает → что делает напарник». Вариант: показать три настоящих сообщения из этого прогона и что после них записал получатель.</p>
<div class="slide"><h4>Два робота с координацией</h4><p class="sub">Общей памяти нет: один канал ROS 2, одна строка JSON на сообщение. Разбирает код робота, не языковая модель</p>
{mock_row(got, heard('tb1', got['t'], got['t'] + 0.5, 'partner_collected')[0]['text'], True)}
{mock_row(claim_y, yield_line['text'])}
{mock_row(sector1, heard('tb1', sector1['t'], sector1['t'] + 0.5, 'sector')[0]['text'], True)}
<div class="foot">Ещё раз в секунду — показания датчика: напарник вносит их в свою карту вероятностей как свои. За прогон {len(msgs)} сообщений · −26 с · −26% заряда · оба вернулись в 80 из 80</div></div>
<p class="cap">Это макет в браузере, не слайд из колоды: в колоду он пока не внесён.</p></section>
</main></body></html>"""
    OUT.write_text(page, encoding='utf-8')
    print(OUT, len(page) // 1024, 'КБ; сообщений', len(msgs), dict(counts))
    print('заявка в споре:', claim_y, '|', yield_line['text'])


if __name__ == '__main__':
    main()
