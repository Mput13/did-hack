#!/usr/bin/env python3
"""Страница «Память между прогонами: как хранится и в каком виде приходит к модели»: memory_between_runs.html.

    pixi run python tools/build_memory_page.py

Всё на странице берётся из настоящих данных одной лаборатории (трудный уровень, сценарий 2):
  первый прогон (adaptive_v2_lab, раскладка образцов 1) считается заново в быстром симуляторе — он детерминирован;
  файл памяти — то, что LabMemory записала после него;
  второй прогон с языковой моделью — запись runs/L4b_llm/adaptive_llm_lab/hard-2.json.gz. В записи запросы
  обрезаны до 300 знаков, поэтому прогон повторяется по записанным ответам модели (сеть не нужна), и начала всех
  запросов сверяются с записью.
Законы среды — файл runs/_knowledge/kb_state.json. Оформление то же, что у llm_calls.html.
"""
import html
import json
import tempfile
from collections import Counter
from pathlib import Path

from did.labmemory import LabMemory
from did.llm import LLMError, LocalClient, _brief
from did.memory import KB_PATH, KnowledgeBase
from did.recorder import load_trace
from did.runner import RUNS, run_episode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'memory_between_runs.html'
DECK = ROOT / 'presentation' / 'team' / 'data' / 'memory_example.json'     # те же записи для слайда колоды
LEVEL, SEED = 'hard', 2
RECORD = RUNS / 'L4b_llm' / 'adaptive_llm_lab' / f'{LEVEL}-{SEED}.json.gz'
E = html.escape

TRIGGERS = {
    'start': 'начало прогона',
    'sample_collected': 'образец собран',
    'candidate_found': 'по дороге датчик показал новое место-кандидат',
    'candidate_lost': 'кандидат на месте не подтвердился',
    'false_collect': 'попытка сбора промахнулась (штраф)',
    'subgoal_done': 'доехал до точки, очередь подцелей пуста',
    'queue_empty': 'очередь подцелей пуста',
    'hazard': 'штраф за опасную зону',
    'collision': 'столкновение',
    'model_mismatch': 'расход заряда разошёлся с прогнозом',
    'sensor_degraded': 'датчик образцов начал сбоить',
    'battery_threshold': 'заряд дошёл до порога миссии',
}
WHO = {'llm': 'модель', 'heuristic': 'правило', 'fallback': 'правило (модель не справилась)', 'rule': 'исполнитель'}


def ru(x, d=1):
    return f'{x:.{d}f}'.replace('.', ',')


def pretty(obj, cut=None, width=100):
    """JSON для чтения: короткие объекты и списки — в одну строку, длинные — по полю на строку.
    cut = {ключ: сколько элементов оставить} — длинные списки и словари обрезаются с пометкой."""
    cut = cut or {}

    def trim(v, key=None):
        if isinstance(v, dict):
            items = list(v.items())
            keep = cut.get(key)
            out = {k: trim(x, k) for k, x in (items[:keep] if keep else items)}
            if keep and len(items) > keep:
                out[f'… ещё {len(items) - keep}'] = '…'
            return out
        if isinstance(v, list):
            keep = cut.get(key)
            out = [trim(x) for x in (v[:keep] if keep else v)]
            if keep and len(v) > keep:
                out.append(f'… ещё {len(v) - keep}')
            return out
        return v

    def dumps(v, ind=0):
        flat = json.dumps(v, ensure_ascii=False)
        if not isinstance(v, (dict, list)) or len(flat) <= width - ind:
            return flat
        pad = ' ' * (ind + 2)
        if isinstance(v, dict):
            rows = [f'{pad}{json.dumps(k, ensure_ascii=False)}: {dumps(x, ind + 2)}' for k, x in v.items()]
            return '{\n' + ',\n'.join(rows) + '\n' + ' ' * ind + '}'
        return '[\n' + ',\n'.join(pad + dumps(x, ind + 2) for x in v) + '\n' + ' ' * ind + ']'
    return dumps(trim(obj))


def first_json(text):
    """Первый JSON-объект в тексте: у снимка после него идёт напоминание о формате, у ответа бывает хвост."""
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]


def first_run():
    """Первый прогон без памяти и файл, который после него записала память о лаборатории."""
    first = run_episode(LEVEL, SEED, 'adaptive_v2_lab', save=False, scenario_args={'sample_seed': 1})
    path = Path(tempfile.mkdtemp()) / f'{LEVEL}-{SEED}.json'
    mem = LabMemory(path)
    mem.learn(first['lab'], first['id'])
    mem.save()
    return first, json.loads(path.read_text(encoding='utf-8')), path.stat().st_size, mem.priors()


def replay(prior, record):
    """Второй прогон заново по записанным ответам модели: возвращает полные запросы и сводку прогона."""
    answers = [ex['response'] for ex in record['llm']]
    sent = []

    def responder(messages):
        sent.append(messages)
        text = answers[len(sent) - 1]
        if text is None:
            raise LLMError('в записи на этот запрос сервер не ответил')
        return text

    s = run_episode(LEVEL, SEED, 'adaptive_llm_lab', llm={'client': LocalClient(responder)}, save=False,
                    scenario_args={'sample_seed': 2}, lab=prior)
    # Сверяются запросы со снимком состояния (первая попытка). Просьба исправить ответ в повторе короче записанной:
    # в записи нет признака «ответ обрезан по лимиту», а на ход прогона её текст не влияет.
    asked = [(m, ex) for m, ex in zip(sent, record['llm']) if ex['attempt'] == 1]
    same = sum(_brief(m[-1]['content']) == ex['request'] for m, ex in asked)
    return sent, s, (same, len(asked))


def main():
    first, file, size, prior = first_run()
    record = load_trace(RECORD)
    sent, second, same = replay(prior, record)
    res = record['result']
    same, asked = same
    if same != asked or len(sent) != len(record['llm']) or abs(second['metrics']['score'] - res['score']) > 1e-6:
        raise SystemExit(f'повтор не совпал с записью: запросов {len(sent)} из {len(record["llm"])}, совпало начал {same} из {asked}, '
                         f'счёт {second["metrics"]["score"]} против {res["score"]}')
    states = [first_json(m[-1]['content']) for m, ex in zip(sent, record['llm']) if ex['attempt'] == 1]
    replies = [first_json(ex['response']) for ex in record['llm'] if ex['attempt'] == 1 and ex['response']]
    st0, rep0 = states[0], replies[0]
    system = sent[0][0]['content']
    usage0 = record['llm'][0]['usage']

    plans = record['plans']
    # что случилось у запомненной зоны во втором прогоне — с памятью и без неё (та же лаборатория, те же образцы)
    import math
    plain = load_trace(RUNS / 'L4b_llm' / 'adaptive_llm' / f'{LEVEL}-{SEED}.json.gz')
    true_zone = min(record['scenario']['hazards'], key=lambda z: math.hypot(z['x'] - file['hazards'][0]['x'], z['y'] - file['hazards'][0]['y']))

    def in_zone(tr):
        return [e for e in tr['events'] if e['type'] == 'hazard_hit'
                and math.hypot(e['x'] - true_zone['x'], e['y'] - true_zone['y']) <= true_zone['r'] + 0.15]
    closest = min(math.hypot(x - file['hazards'][0]['x'], y - file['hazards'][0]['y'])
                  for x, y in zip(record['track']['x'], record['track']['y']))
    statuses = sorted({st['lab_memory']['hazards'][0]['status'] for st in states})
    by = Counter(p['source'] for p in plans)
    hz = file['hazards'][0]
    sz = file['soil_zones']
    kb = json.loads(KB_PATH.read_text(encoding='utf-8'))
    kb_priors = KnowledgeBase(KB_PATH).priors()
    llm_rows = json.loads((RUNS / 'L4b_llm' / 'summary.json').read_text(encoding='utf-8'))['rows']

    def stat(key):
        s = kb['stats'][key]
        return {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ('support', 'contradictions')},
                'support': s['support'][:2] + [f'… ещё {len(s["support"]) - 2}'], 'contradictions': s['contradictions']}

    plan_rows = ''.join(
        f'<tr><td class="c">{ru(p["t"])}</td><td>{E(TRIGGERS.get(p["trigger"], p["trigger"]))}</td>'
        f'<td><b>{WHO.get(p["source"], p["source"])}</b></td><td>{E(p["reasoning"][:150])}{"…" if len(p["reasoning"]) > 150 else ""}</td></tr>'
        for p in plans[:9])
    lab_rows = ''.join(
        f'<tr><td class="c">{r["seed"]}</td><td class="c">{ru(r["adaptive_llm"]["score"])}</td><td class="c">{ru(r["adaptive_llm_lab"]["score"])}</td>'
        f'<td class="c">{r["adaptive_llm_lab"]["plans_citing_memory"]} из {r["adaptive_llm_lab"]["llm_plans"]}</td></tr>' for r in llm_rows)
    keys = ''.join(f'<span class="{"k hit" if k == "lab_memory" else "k"}">{k}</span>' for k in st0)
    trig_rows = ''.join(f'<tr><td><code>{k}</code></td><td>{E(v)}</td></tr>' for k, v in TRIGGERS.items())
    after4 = sum(1 for p in plans if p['source'] == 'heuristic')

    style = (ROOT / 'llm_calls.html').read_text(encoding='utf-8')
    style = style[style.index('<style>') + 7:style.index('</style>')]
    extra = """
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:14px 0}
.flow div{flex:1 1 150px;background:var(--code);border-radius:10px;padding:10px 12px;font-size:14px}
.flow div b{display:block;font-size:13px;color:var(--accent);margin-bottom:2px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}@media(max-width:820px){.two{grid-template-columns:1fr}}
.two h4{margin:0 0 4px;font-size:14px;color:var(--ink2);font-weight:600}
pre code{white-space:pre-wrap;word-break:break-word;font-size:12.5px}
.k{display:inline-block;font:12.5px ui-monospace,Menlo,monospace;background:var(--code);border-radius:6px;padding:2px 7px;margin:2px 3px 2px 0}
.k.hit{background:var(--accent);color:#fff}
.mem3 td:first-child{white-space:nowrap}
td.c{text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap}
.quote{background:var(--code);border-left:3px solid var(--s1);border-radius:0 8px 8px 0;padding:10px 14px;margin:10px 0;font-size:14.5px}
"""
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Память между прогонами: как хранится и в каком виде приходит к модели</title><style>{style}{extra}</style></head>
<body><main>
<div class="num">Память робота · настоящие файлы и запросы</div>
<h1>Память между прогонами: как она хранится и в каком виде приходит к модели</h1>
<p class="lead">Один пример от начала до конца: лаборатория трудного уровня, сценарий {SEED}. Робот проехал её один раз, записал файл, и во второй
прогон пошёл уже с ним. Ниже — сам файл, поле из настоящего запроса к языковой модели и её ответ. В конце — когда агент вообще
зовёт модель и кто решает без неё.</p>

<section id="short"><div class="num">Коротко</div><h2>Что нужно понимать</h2>
<div class="answer"><ul>
<li><b>Между прогонами живут два файла JSON.</b> Их пишет и читает обычный код на Python. Языковая модель файлов не видит и ничего в них не пишет.</li>
<li><b>Память о лаборатории</b> — места: где робот получил штраф за зону и где пол дорогой. Один файл на одну арену. Мест образцов и «правды судьи» в нём нет.</li>
<li><b>Законы среды</b> — числа: сколько стоит метр, поворот и простой, сколько длится сбой. Один общий файл на все арены.</li>
<li><b>К модели память о лаборатории приходит одним полем <code>lab_memory</code></b> внутри того же снимка состояния, что и всё остальное. Законы среды в запрос не попадают вовсе.</li>
<li><b>Память работает и без модели.</b> Число «+4,3 очка» со слайда получено у агента на правиле: зону из памяти объезжает исполнитель, который строит маршрут.</li>
</ul></div>
<table class="mem3">
<tr><th></th><th>Карта мира</th><th>Законы среды</th><th>Память о лаборатории</th></tr>
<tr><td><b>Где живёт</b></td><td>в памяти программы, один прогон</td><td>файл <code>runs/_knowledge/kb_state.json</code></td><td>файл на каждую арену, например <code>…/lab/…/hard-{SEED}.json</code></td></tr>
<tr><td><b>Что хранит</b></td><td>вероятные места образцов, цена грунта, опасные зоны</td><td>числа: цена метра, поворота, простоя; длительность сбоя; утечка</td><td>места: зона штрафа, клетки пола с измеренной ценой</td></tr>
<tr><td><b>Годится для другой карты</b></td><td>—</td><td>да</td><td>нет, только та же арена</td></tr>
<tr><td><b>Как доходит до модели</b></td><td>поля <code>candidates</code>, <code>soil_zones</code>, <code>hazards</code> снимка</td><td>никак: меняет исходные вероятности в расследованиях агента-исследователя</td><td>поле <code>lab_memory</code> снимка</td></tr>
<tr><td><b>Что дала по измерениям</b></td><td>основа всех решений</td><td>выигрыш не показан (опыт E11)</td><td>+4,3 очка [2,4; 6,3] во втором прогоне, агент на правиле, быстрый симулятор</td></tr>
</table></section>

<section id="file"><div class="num">Шаг 1 · как хранится</div><h2>Первый прогон и файл, который после него остался</h2>
<p class="tag">Первый прогон идёт без памяти. Робот собрал {first['metrics']['samples_collected']} образцов из 7, набрал {ru(first['metrics']['score'])} очка
и на {ru(hz['t_hit'])}-й секунде получил штраф за невидимую опасную зону. После прогона агент выгружает то, что узнал сам,
и память дописывает это в файл.</p>
<pre><code>{E(pretty(file, {'cloud': 2, 'soil': 3}))}</code></pre>
<p class="cap">Файл целиком — {size} байт. Сокращены только два длинных списка: гипотезы о зоне и клетки пола. Файл получен повтором первого прогона при сборке страницы
(быстрый симулятор воспроизводит прогон точно), поэтому дата в нём — дата сборки.</p>
<table>
<tr><th>Поле</th><th>Что это</th></tr>
<tr><td><code>runs</code></td><td>Из каких прогонов собрана память. У каждого знания есть ссылка на прогон, из которого оно взято</td></tr>
<tr><td><code>hazards</code></td><td>Опасные зоны, в которых робот получил штраф. Зону он не видит: знает только точку штрафа. Поэтому хранит не круг, а
{len(hz['cloud'])} гипотез «центр и радиус», которые с этой точкой согласуются (<code>cloud</code>: x, y, радиус, вес). <code>x</code>, <code>y</code>, <code>r</code> — их среднее с запасом:
({ru(hz['x'])}; {ru(hz['y'])}), {ru(hz['r'], 2)} м</td></tr>
<tr><td><code>soil</code></td><td>Клетки пола 0,2 × 0,2 м, по которым робот ездил: <code>k</code> — во сколько раз расход выше обычного, <code>m</code> — сколько метров по клетке пройдено. Всего {len(file['soil'])} клетки</td></tr>
<tr><td><code>soil_zones</code></td><td>Те же клетки, собранные в пятна дорогого грунта: {len(sz)} пятно, дороже в {ru(sz[0]['mult'], 2)} раза, измерено на {ru(sz[0]['evidence_m'], 2)} м пути</td></tr>
<tr><td><code>age</code></td><td>Сколько прогонов знание не перепроверялось. С каждым таким прогоном уверенность умножается на 0,85</td></tr>
</table>
<h3>Кто пишет и читает файл (tools/l4b_run.py, did/labmemory.py)</h3>
<pre><code>{E('''mem = LabMemory(path)              # файл этой лаборатории; нет файла — память пустая
prior = mem.priors()               # что взять в прогон: зоны и грунт с уверенностью, а не как истину
r = run_episode(level, seed, 'adaptive_v2_lab', lab=prior)
mem.learn(r['lab'], r['id'])       # дописать то, что робот узнал сам в этом прогоне
mem.save()                         # json.dumps(...) в файл''')}</code></pre>
<p>Перед вторым прогоном <code>priors()</code> превращает запись в предположение: зоне, где штраф был в одном прогоне, даётся уверенность
<b>{ru(prior['hazards'][0]['conf'], 2)}</b>, пятну грунта — <b>{ru(prior['soil_zones'][0]['confidence'], 2)}</b>. Внутри прогона зона из памяти не запрещает клетки, а удорожает их для маршрута.
Проехал запомненное место без штрафа — гипотезы о зоне отбрасываются, и когда не осталось ни одной, зона снята.</p></section>

<section id="model"><div class="num">Шаг 2 · в каком виде получает модель</div><h2>Второй прогон: настоящий запрос и ответ</h2>
<p class="tag">Та же лаборатория, образцы в других местах, агент с языковой моделью (<code>adaptive_llm_lab</code>, модель {E(json.loads((RUNS / 'L4b_llm' / 'summary.json').read_text())['model'])}, сервер МАИ).
Запрос состоит из двух сообщений: постоянная инструкция ({len(system)} знаков) и снимок состояния в JSON. Вот поля снимка первого запроса; память — одно из них:</p>
<p>{keys}</p>
<div class="two"><div><h4>Поле lab_memory в запросе, t = 0 с</h4>
<pre><code>{E(pretty(st0['lab_memory'], width=58))}</code></pre></div>
<div><h4>Для сравнения: то, что робот узнал сам, в том же запросе</h4>
<pre><code>{E(pretty({'hazards': st0['hazards'], 'soil_zones': st0['soil_zones'], 'candidates': st0['candidates'], 'explore_points': st0['explore_points'][:2] + ['…']}, width=58))}</code></pre>
<p class="cap">На старте своих зон и грунтов ещё нет: списки пустые. Всё, что робот знает о зонах в эту секунду, пришло из файла.</p></div></div>
<p>Первый запрос — {usage0['prompt_tokens']} токенов (единиц текста), из них поле <code>lab_memory</code> — {len(json.dumps(st0['lab_memory'], ensure_ascii=False))} знаков.
В инструкции модели про это поле ничего не сказано: что это и насколько этому верить, объясняет строка <code>note</code> внутри самого поля.
Зоны из памяти названы <code>M1</code>, <code>M2</code>…, чтобы не путать их с зонами <code>Z1</code>, которые робот нашёл в этом прогоне.</p>
<h3>Что модель ответила</h3>
<div class="quote">{E(rep0['reasoning'])}</div>
<p>Выбранная подцель: <code>{E(json.dumps(rep0['subgoals'], ensure_ascii=False))}</code>. {'Правило в этой же ситуации выбрало ту же точку.' if plans[0].get('rule_match') else 'Правило в этой же ситуации выбрало бы другую цель.'}</p>
<h3>Что было у этой зоны во втором прогоне</h3>
<p>Настоящую зону знает только судья: центр ({ru(true_zone['x'], 2)}; {ru(true_zone['y'], 2)}), радиус {ru(true_zone['r'], 2)} м. Память после первого прогона оценила её как
({ru(hz['x'])}; {ru(hz['y'])}), {ru(hz['r'], 2)} м — с запасом, потому что знает лишь точку штрафа.</p>
<table><tr><th></th><th>Без памяти</th><th>С памятью</th></tr>
<tr><td>Штрафов в запомненной зоне</td><td class="c">{len(in_zone(plain))}{f" (на {ru(in_zone(plain)[0]['t'])}-й секунде)" if in_zone(plain) else ""}</td><td class="c">{len(in_zone(record))}</td></tr>
<tr><td>Всего штрафов за зоны</td><td class="c">{plain['result']['hazard_hits']}</td><td class="c">{res['hazard_hits']}</td></tr>
<tr><td>Ближе всего к центру зоны из памяти</td><td class="c">—</td><td class="c">{ru(closest, 2)} м</td></tr>
<tr><td>Счёт</td><td class="c">{ru(plain['result']['score'])}</td><td class="c">{ru(res['score'])}</td></tr></table>
<p>Поле <code>lab_memory</code> собирается заново перед каждым вопросом к модели. В этом прогоне статус зоны все {len(states)} раз был «{E(statuses[0]) if len(statuses) == 1 else E(", ".join(statuses))}»:
робот к зоне не подъезжал. Другие значения статуса: «место уточнено проездами рядом», «подтверждена штрафом в этом прогоне», «снята: проехал без штрафа».
Второй штраф в обоих прогонах — в зоне, которая появляется на {ru(next(e['t'] for e in record['scenario']['events'] if e['type'] == 'new_hazard'))}-й секунде; в первом прогоне робот в неё не попал, и в памяти её нет.
Счёт с памятью здесь ниже, хотя штрафов меньше: это один прогон, причина не разбиралась.</p>
<h3>Что это дало</h3>
<table><tr><th>Лаборатория</th><th>Счёт без памяти</th><th>Счёт с памятью</th><th>Решений модели со ссылкой на память</th></tr>{lab_rows}</table>
<p class="note">Четыре лаборатории, восемь прогонов — это наблюдение, а не измерение. Модель читает память и ссылается на неё, но только в первом
решении на старте; выигрыш у агента с моделью не показан. В этой лаборатории второй прогон кончился счётом {ru(res['score'])}
(собрано {res['samples_collected']} из {res['samples_total']}, штрафов за зону {res['hazard_hits']}). Измеренный выигрыш «+4,3 очка [2,4; 6,3]» — у агента на правиле,
160 пар прогонов в 40 лабораториях.</p></section>

<section id="laws"><div class="num">Вторая память</div><h2>Законы среды: числа, а не места</h2>
<p class="tag">Агент-исследователь после каждого прогона записывает, что выяснил о самой среде. Файл один на все арены: сейчас в нём {kb['runs']} прогонов.</p>
<div class="two"><div><h4>Одна запись файла kb_state.json</h4>
<pre><code>{E(pretty({'energy.per_m': stat('energy.per_m')}, width=58))}</code></pre>
<p class="cap"><code>w</code>, <code>wx</code>, <code>wxx</code> — суммы для среднего и разброса; <code>support</code> — прогоны-доказательства; <code>contradictions</code> — прогоны, которые правилу противоречат. Много противоречий — правило снимается.</p></div>
<div><h4>Что из файла берёт следующий прогон: priors()</h4>
<pre><code>{E(pretty({k: ({a: {b: round(c, 3) for b, c in v2.items()} for a, v2 in v.items()} if k == 'energy' else ({a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else round(v, 3))) for k, v in kb_priors.items()}, width=58))}</code></pre>
<p class="cap">Это исходные предположения, а не истина: значение и разброс между прогонами.</p></div></div>
<p>Куда это идёт. Числа получает расследователь внутри агента (<code>did/inquiry.py</code>): когда расход расходится с прогнозом, он сравнивает объяснения
«дорогой грунт», «утечка батареи», «сбой датчика», и исходные вероятности этих объяснений берутся из файла. В запрос к языковой модели эти числа не входят.
В опыте E11 выигрыш от этой памяти не показан.</p></section>

<section id="when"><div class="num">Кто решает</div><h2>Когда агент зовёт модель и как решает без неё</h2>
<p class="tag">Модель не управляет роботом. Она отвечает на один вопрос — «куда ехать дальше» — и только когда для этого есть повод.</p>
<div class="flow"><div><b>1. Исполнитель, 10 раз в секунду</b>всегда код: едет по пути, объезжает препятствия и зоны, собирает образец, следит за зарядом</div>
<div><b>2. Повод</b>что-то случилось: собрал образец, кандидат не подтвердился, штраф, расход разошёлся с прогнозом</div>
<div><b>3. Планировщик</b>правило или модель выбирает следующую цель. Модель — если с прошлого вопроса к ней прошло 4 с</div>
<div><b>4. Проверка ответа</b>форма и смысл. Не прошёл — один запрос на исправление, затем правило</div>
<div><b>5. Очередь подцелей</b>исполнитель едет к цели до следующего повода</div></div>
<table>
<tr><th>Решение</th><th>Кто принимает</th></tr>
<tr><td>Скорость, поворот, путь в обход препятствий и опасных зон</td><td>всегда код (исполнитель)</td></tr>
<tr><td>Пробовать ли сбор на месте</td><td>всегда код: уверенность карты вероятностей не ниже 0,85</td></tr>
<tr><td>Возврат на базу</td><td>всегда код: все образцы собраны, заряда осталось на дорогу домой с запасом, или время на исходе</td></tr>
<tr><td>Куда ехать дальше (какой кандидат, какая точка разведки)</td><td>планировщик: правило или модель</td></tr>
</table>
<h3>Поводы, по которым спрашивают планировщик</h3>
<table><tr><th>Повод</th><th>Когда</th></tr>{trig_rows}</table>
<h3>Правило, которое решает без модели (did/planner.py)</h3>
<pre><code>{E('''if cands:    # есть места, где датчик видит признаки образца
    best = max(cands, key=lambda c: c['confidence'] / (c['cost_to'] + 0.5))     # выгода на единицу заряда
    return [{'type': 'investigate', 'target': best['id']}]
if points:   # кандидатов нет — разведка
    best = max(points, key=lambda p: p['unseen_share'] / (p['cost_to'] + 1.0))
    return [{'type': 'explore', 'target': best['id']}]
return [{'type': 'return_base'}]                                                # достижимых целей не осталось''')}</code></pre>
<h3>Тот же второй прогон: кто отвечал на каждый повод</h3>
<p>За прогон планировщик спрашивали {len(plans)} раз: {by['llm']} раз ответила модель, {after4} — правило, потому что повод пришёл раньше чем через 4 секунды после прошлого вопроса к модели
(<code>llm_min_interval_s = 4.0</code>). Первые девять решений:</p>
<table><tr><th>Время, с</th><th>Повод</th><th>Кто решал</th><th>Объяснение</th></tr>{plan_rows}</table>
<table>
<tr><th>Вариант агента</th><th>Когда спрашивают модель</th></tr>
<tr><td><code>fixed</code></td><td>никогда: едет по заранее составленному маршруту, планировщика нет</td></tr>
<tr><td><code>adaptive</code>, <code>scientist</code> и их версии <code>_v2</code> (в том числе агент показа <code>scientist_v2</code>)</td><td>никогда: на каждый повод отвечает правило</td></tr>
<tr><td><code>adaptive_llm</code>, <code>scientist_llm</code>, <code>adaptive_llm_lab</code></td><td>на каждый повод, но не чаще раза в 4 секунды; между вопросами отвечает правило</td></tr>
<tr><td>серии L3c, L3d, L4a (сравнение решений модели и правила)</td><td>на каждый повод: интервал выставлен в ноль, мир стоит, пока модель думает</td></tr>
</table>
<p class="note">Если сервер не ответил или ответ негоден и после исправления, решение принимает то же правило, и в записи прогона оно помечено как «запасное». Если модель
назвала цель, которой уже нет или до которой не хватит заряда, план отклоняет исполнитель, и решает правило.</p></section>

<section id="qa"><div class="num">К защите</div><h2>Вопросы, которые могут задать</h2>
<details open><summary>Память формируется на разных картах или на одной?</summary><p>Памятей между прогонами две. Законы среды — числа, они копятся со всех карт и годятся для любой. Память о лаборатории — места, она относится к одной арене:
если арену переставили, файл нужно стереть. В опыте с чужой памятью выигрыш не показан (+1,9 [−0,4; 4,0]).</p></details>
<details open><summary>Не подсматривает ли робот правду через память?</summary><p>Нет. В файл идёт только то, что робот узнал сам: точка штрафа и расход на клетках, по которым он ездил. Мест образцов и настоящих границ зон там нет:
зона хранится как набор гипотез вокруг точки штрафа.</p></details>
<details open><summary>Что, если среда изменилась, а память осталась?</summary><p>Память — предположение с уверенностью меньше единицы. Проезд без штрафа снимает зону; расход, разошедшийся с прогнозом, стирает память о грунте рядом.
Но вред измерен: чужая память на среднем уровне при научных правилах — −1,8 очка [−3,0; −0,6].</p></details>
<details open><summary>Модель учится между прогонами?</summary><p>Нет. Модель не меняется и ничего не помнит: каждый вызов — отдельный запрос. Учится код: он пишет файл и в следующем прогоне кладёт его содержимое в запрос.</p></details>
</section>
<p class="cap">Собрано программой <code>tools/build_memory_page.py</code>. Второй прогон повторён по записанным ответам модели без сети: начала всех {asked} запросов со снимком состояния и итоговый счёт совпали с записью
<code>runs/L4b_llm/adaptive_llm_lab/{LEVEL}-{SEED}.json.gz</code>.</p>
</main></body></html>"""
    OUT.write_text(page, encoding='utf-8')
    # Для слайда: по одной строке из файла и из запроса. Поля взяты как есть, пропуски помечены многоточием.
    one = lambda v: json.dumps(v, ensure_ascii=False, separators=(', ', ':'))                    # noqa: E731
    m1 = st0['lab_memory']['hazards'][0]
    DECK.write_text(json.dumps({
        'lab': f'{LEVEL}-{SEED}', 'made_by': 'tools/build_memory_page.py',
        'file': '{"hazards":[' + one({k: hz[k] for k in ('x', 'y', 'r', 'hits')})[:-1] + ', …}], "soil_zones":['
                + one({k: sz[0][k] for k in ('x', 'y', 'mult')})[:-1] + ', …}], …}',
        'request': '"lab_memory":{"hazards":[' + one({k: m1[k] for k in ('id', 'x', 'y', 'confidence', 'status')})[:-1]
                   + ', …}], …}',
        'hypotheses': len(hz['cloud']), 'file_bytes': size, 'prompt_tokens': usage0['prompt_tokens'],
        'reply': rep0['reasoning']}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(OUT, len(page) // 1024, 'КБ; запросов', len(sent), 'совпало начал', same, '; решений', len(plans), dict(by))


if __name__ == '__main__':
    main()
