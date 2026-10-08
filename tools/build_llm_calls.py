#!/usr/bin/env python3
"""Страница «Настоящие вызовы языковой модели»: что агент отправил модели и что получил, по записям прогонов.

    pixi run python tools/build_llm_calls.py            # пишет llm_calls.html в корень репозитория

Берёт записи прогонов (runs/llm_real, прогоны миссий исследования R13) и кэш ответов (runs/_llm_cache): в записи
запрос хранится обрезанным, полный текст — в кэше. Ничего не запускает и к модели не обращается.
"""
import gzip
import html
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUNS = ROOT / 'runs'
R13 = ROOT.parent / 'DID-research' / 'R13' / 'runs'          # прогоны миссий лежат в дереве исследования

TRIGGER = {'start': 'старт', 'candidate_found': 'найден кандидат', 'candidate_lost': 'кандидат не подтвердился',
           'sample_collected': 'образец собран', 'sensor_degraded': 'датчик сбоит', 'subgoal_done': 'подцель выполнена',
           'hazard_hit': 'штраф: опасная зона', 'hazard': 'штраф: опасная зона', 'collision': 'столкновение', 'battery_threshold': 'заряд у порога',
           'model_mismatch': 'расход не сходится с прогнозом', 'new_hazard': 'новая опасная зона'}
GOAL = {'explore': 'разведка', 'investigate': 'проверить кандидата', 'return_base': 'на базу', 'goto': 'ехать в точку'}


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def esc(text):
    return html.escape(str(text))


def pretty(text):
    """JSON из текста — с отступами; не JSON — как есть."""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=1)
    except (TypeError, ValueError):
        return str(text)


def cached(cache_dir, response):
    """Запись кэша с этим ответом: в ней полный запрос (messages)."""
    for path in sorted(Path(cache_dir).glob('*.json')):
        try:
            entry = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get('text') == response:
            return entry
    return None


def state_of(content):
    """Сводка состояния из текста запроса: «Состояние: {...}»."""
    i = content.find('{')
    try:
        obj, _ = json.JSONDecoder().raw_decode(content[i:])
        return json.dumps(obj, ensure_ascii=False, indent=1)
    except ValueError:
        return content


def exchange(title, lead, entry, reply, meta, after):
    """Один обмен целиком: подсказка, состояние, ответ, что сделал код."""
    users = [m['content'] for m in entry['messages'] if m['role'] == 'user']
    usage = entry.get('usage') or {}
    return f'''<div class="call">
<h3>{esc(title)}</h3>
<p>{lead}</p>
<div class="meta">{meta} · ответ за <b>{entry.get('latency_ms', 0) / 1000:.1f} с</b> · запрос {usage.get('prompt_tokens', '—')} токенов, ответ {usage.get('completion_tokens', '—')}</div>
<div class="cols">
 <div><h4>Агент → модель: сводка состояния</h4><pre class="tall">{esc(state_of(users[0]) if users else '')}</pre></div>
 <div><h4>Модель → агент: план</h4><pre class="tall">{esc(pretty(reply))}</pre></div>
</div>
<p class="after"><b>Что сделал код.</b> {after}</p>
</div>'''


def timeline(tr):
    rows = []
    calls = {round(e['t'], 1): e for e in tr['llm'] if e['attempt'] == 1}
    for p in tr['plans']:
        call = calls.get(round(p['t'], 1))
        first = (p.get('subgoals') or [{}])[0]
        goal = GOAL.get(first.get('type'), first.get('type', '—')) + (f" {first['target']}" if first.get('target') else '')
        who = '<span class="tag llm">модель</span>' if p['source'] == 'llm' else '<span class="tag rule">правило</span>'
        wait = f"{call['latency_ms'] / 1000:.0f} с" if call and p['source'] == 'llm' else '—'
        rows.append(f"<tr><td>{p['t']:.1f} с</td>".replace('.', ',') + f"<td>{esc(TRIGGER.get(p['trigger'], p['trigger']))}</td><td>{who}</td>"
                    f"<td>{wait}</td><td>{esc(goal)}</td><td>{esc((p.get('reasoning') or '')[:150])}</td></tr>")
    return '\n'.join(rows)


def source(obj, limit=28):
    lines = inspect.getsource(obj).rstrip().split('\n')
    return '\n'.join(lines[:limit]) + ('\n    ...' if len(lines) > limit else '')


def main():
    from did.llm import ChatClient
    from did.planner import LLMPlanner

    parts = []
    # --- обычный прогон: агент adaptive_llm, модель МАИ
    tr = load(RUNS / 'llm_real' / 'mai-qwen3.8-flash-next' / 'hard-1001.json.gz')
    cache = RUNS / '_llm_cache' / 'qwen3.8-flash-next'
    res = tr['result']
    by_model = sum(p['source'] == 'llm' for p in tr['plans'])
    total_wait = sum(e['latency_ms'] for e in tr['llm']) / 1000
    parts.append(f'''<h2>1. Один прогон целиком: когда агент спрашивал модель</h2>
<p>Агент <code>adaptive_llm</code>, модель <b>qwen3.8-flash-next</b> с сервера МАИ, трудный сценарий 1001. Робот собрал
{res['samples_collected']} образцов из {res['samples_total']}, вернулся на базу, счёт {res['score']}. За прогон длиной {res['t']:.0f} секунд
было {len(tr['plans'])} решений о том, куда ехать дальше: {by_model} приняла модель, {len(tr['plans']) - by_model} — правило.
Модель в сумме думала {total_wait:.0f} секунд; в этом прогоне время раздумий роботу не засчитывалось.</p>
<table><thead><tr><th>Время</th><th>Повод</th><th>Кто решал</th><th>Ответ за</th><th>Первая подцель</th><th>Обоснование (начало)</th></tr></thead>
<tbody>{timeline(tr)}</tbody></table>
<p class="note">Почему часть решений принимает правило: если модель спрашивали меньше четырёх секунд назад, на новый повод отвечает
правило — иначе робот стоял бы в ожидании почти весь прогон. Это настройка <code>llm_min_interval_s</code>.</p>''')

    first = tr['llm'][0]
    entry = cached(cache, first['response'])
    if entry:
        system = next((x['content'] for x in entry['messages'] if x['role'] == 'system'), '')
        parts.append(f'''<h2>2. Вызовы целиком</h2>
<p>Каждый запрос состоит из двух частей. Первая — постоянная инструкция: кто ты, что означают поля сводки, как выбирать, в каком виде
отвечать. Вторая — сводка состояния на этот момент. Инструкция одна и та же во всех вызовах планировщика:</p>
<details><summary>Инструкция модели целиком ({len(system)} знаков)</summary><pre>{esc(system)}</pre></details>''' + exchange(
            'Вызов на старте: куда ехать, когда ещё ничего не известно',
            'Первая секунда прогона. Кандидатов нет, поэтому в сводке только точки разведки (<code>explore_points</code>) с долей '
            'неосмотренной площади и ценой пути до каждой.',
            entry, first['response'], f"агент <code>adaptive_llm</code> · модель {esc(entry['model'])} · сценарий hard-1001 · время 0,0 с",
            'Ответ прошёл проверку формы (это план по схеме) и смысла (такие точки есть, заряда хватает). Исполнитель повёл '
            'робота к первой подцели сам: путь, скорости и объезд препятствий модель не задаёт.'))
    fault = next((e for e in tr['llm'] if round(e['t'], 1) == 60.8), None)
    entry = fault and cached(cache, fault['response'])
    if entry:
        parts.append(exchange(
            'Вызов при сбое датчика',
            'На 61-й секунде агент заметил, что датчик образцов ведёт себя не так, как предсказывает карта. В сводке это видно '
            'в полях <code>sensor</code> и <code>alarms</code>.',
            entry, fault['response'], f"агент <code>adaptive_llm</code> · модель {esc(entry['model'])} · сценарий hard-1001 · время 60,8 с",
            'План принят и исполнен. Что датчик сбоит, определил код (сравнение показаний с прогнозом); модель получила это '
            'готовым фактом и учла в выборе цели.'))

    # --- миссия словами (R13)
    try:
        m = load(R13 / 'missions_strict' / 'M1_llm' / 'hard-1001.json.gz')
        call = next(e for e in m['llm'] if 'return_base' in (e['response'] or ''))
        entry = cached(R13 / '_llm_cache' / 'qwen3.8-flash-next', call['response'])
        if entry:
            parts.append(exchange(
                'Миссия, заданная словами: «Собери ровно два образца и сразу возвращайся на базу»',
                f"Тот же агент и тот же сценарий, но текст миссии другой (поле <code>mission</code> в сводке). На {call['t']:.1f}-й секунде "
                'собран второй образец. На арене ещё пять, заряда много — правило поехало бы дальше.',
                entry, call['response'],
                f"агент <code>adaptive_llm</code> · модель {esc(entry['model'])} · сценарий hard-1001 · время {call['t']:.1f} с",
                f"Робот вернулся на базу с двумя образцами из {m['result']['samples_total']}. Кода, который знает про «ровно два», в "
                'программе нет: поведение изменила одна фраза в миссии. Это исследование R13.'))
    except (OSError, StopIteration):
        pass

    # --- негодный ответ и исправление
    bad_tr = load(RUNS / 'llm_real' / 'mai-deepseek-v4-flash' / 'hard-1003.json.gz')
    i = next((k for k, e in enumerate(bad_tr['llm']) if not e['ok']), None)
    if i is not None and i + 1 < len(bad_tr['llm']):
        bad, fix = bad_tr['llm'][i], bad_tr['llm'][i + 1]
        parts.append(f'''<div class="call"><h3>Негодный ответ и что с ним происходит</h3>
<p>Модель <b>DeepSeek-V4-Flash</b>, трудный сценарий 1003, время {bad['t']:.1f} с. Модель рассуждала так долго, что упёрлась в лимит длины
и не успела написать сам план.</p>
<div class="cols">
 <div><h4>Первый ответ модели</h4><pre>{esc(bad['response'] or '(пусто)')}</pre>
 <h4>Что нашла проверка</h4><pre>{esc(chr(10).join('— ' + x for x in bad['errors']))}</pre></div>
 <div><h4>Агент → модель: просьба исправить</h4><pre>{esc(fix['request'])}</pre>
 <h4>Второй ответ модели ({fix['latency_ms'] / 1000:.0f} с)</h4><pre class="tall">{esc(pretty(fix['response']))}</pre></div>
</div>
<p class="after"><b>Что сделал код.</b> Второй ответ прошёл проверку, план исполнен. Если бы и он не прошёл, решение принял бы запасной
вариант — правило; робот без плана не остаётся никогда.</p></div>''')

    # --- роли в расследовании
    roles = RUNS / 'llm_real' / 'roles'
    try:
        d = json.loads((roles / 'gpt-6-luna_deliberate.json').read_text(encoding='utf-8'))
        e = json.loads((roles / 'gpt-6-luna_explain.json').read_text(encoding='utf-8'))
        ctx = d['context']
        alts = ''.join(f"<tr><td>{esc(a['statement'])}</td><td>{a['prior']:.0%}</td></tr>" for a in ctx['alternatives'])
        tests = ''.join(f"<tr><td>{esc(t['name'])}</td><td>{t.get('cost', '—')}</td><td>{t.get('gain_bits', '—')}</td></tr>" for t in ctx['tests'])
        parts.append(f'''<h2>3. Модель в расследовании: автор, критик, рассказчик</h2>
<p>Расследование начинается, когда агент <code>scientist</code> заметил странность. Здесь: «{esc(ctx['anomaly']['text'])}».
Код заранее составил список объяснений с начальными вероятностями и список опытов с их ценой и пользой — это вход для модели.
Модель <b>GPT-6 Luna</b>. Это записанный отдельный пример, а не прогон: серия на моделях МАИ сейчас идёт (исследование L3).</p>
<div class="cols">
 <div><h4>Объяснения, которые подготовил код</h4><table><thead><tr><th>Объяснение</th><th>Вероятность до опытов</th></tr></thead><tbody>{alts}</tbody></table></div>
 <div><h4>Опыты, которые можно поставить</h4><table><thead><tr><th>Опыт</th><th>Цена, ед. заряда</th><th>Польза, бит</th></tr></thead><tbody>{tests}</tbody></table></div>
</div>
<div class="cols">
 <div><h4>Автор (модель): какие объяснения проверять и какими опытами</h4><pre class="tall">{esc(json.dumps(d['proposal'], ensure_ascii=False, indent=1))}</pre></div>
 <div><h4>Критик (та же модель в другой роли): что не так с планом</h4><pre class="tall">{esc(json.dumps(d['critique'], ensure_ascii=False, indent=1))}</pre></div>
</div>
<h4>Рассказчик (модель): вывод по завершённому расследованию</h4><pre>{esc(e['text'])}</pre>
<p class="after"><b>Что остаётся за кодом.</b> Вероятности объяснений после каждого опыта и итоговый вердикт считает программа по измерениям.
В ответах модели таких полей нет; если модель напишет в тексте вероятность, которой не было во входе, ответ считается ошибочным.</p>''')
    except (OSError, KeyError):
        pass

    if '--embed' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--embed') + 1])
        mission = next((x for x in parts if 'Миссия, заданная словами' in x), '')
        body = INTRO.split('<p>Вызов модели — это один HTTP-запрос')[0] + parts[0] + mission
        out.write_text(EMBED.replace('{{BODY}}', body), encoding='utf-8')
        print(f'{out}: {out.stat().st_size / 1024:.0f} КБ')
        return
    page = PAGE.replace('{{INTRO}}', INTRO).replace('{{BODY}}', '\n'.join(parts)).replace('{{PLAN_SRC}}', esc(source(LLMPlanner.plan))) \
               .replace('{{POST_SRC}}', esc(source(ChatClient._post, 22)))
    out = ROOT / 'llm_calls.html'
    out.write_text(page, encoding='utf-8')
    print(f'{out.name}: {out.stat().st_size / 1024:.0f} КБ, разделов {len(parts)}')


PAGE = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DID Hack — настоящие вызовы языковой модели</title>
<style>
 :root { --acc: #eb6834; --ok: #1baf7a; }
 body { margin: 0 auto; max-width: 1100px; padding: 18px 22px 60px; font: 15.5px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--foreground, #14181c); background: var(--background, #fff); }
 h1 { font-size: 26px; margin: 8px 0 6px; } h2 { font-size: 20px; margin: 38px 0 8px; } h2:first-child { margin-top: 0; } h3 { font-size: 17px; margin: 0 0 6px; } h4 { font-size: 13.5px; margin: 12px 0 5px; color: var(--muted-foreground, #5b6570); }
 p { margin: 7px 0; } .sub, .note { color: var(--muted-foreground, #5b6570); } .note { font-size: 14px; }
 code { font-family: ui-monospace, Menlo, monospace; font-size: 13px; background: rgba(127,127,127,.13); padding: 1px 5px; border-radius: 4px; }
 pre { font: 12.5px/1.45 ui-monospace, Menlo, monospace; background: rgba(127,127,127,.09); border-radius: 8px; padding: 10px 12px; margin: 0; white-space: pre-wrap; word-break: break-word; overflow: auto; }
 pre.tall { max-height: 440px; }
 table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 8px 0; }
 th, td { text-align: left; padding: 6px 10px 6px 0; vertical-align: top; border-top: 1px solid rgba(127,127,127,.25); }
 th { font-size: 12.5px; color: var(--muted-foreground, #5b6570); border-top: 0; }
 .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; } @media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
 .call { border: 1px solid rgba(127,127,127,.3); border-radius: 12px; padding: 16px 18px; margin: 16px 0; }
 .meta { font-size: 13.5px; color: var(--muted-foreground, #5b6570); margin: 6px 0 10px; }
 .after { margin-top: 12px; } details { margin: 8px 0 12px; } summary { cursor: pointer; font-size: 14px; color: var(--muted-foreground, #5b6570); }
 .tag { font-size: 12.5px; font-weight: 650; padding: 1px 8px; border-radius: 10px; white-space: nowrap; }
 .tag.llm { background: rgba(235,104,52,.16); color: #b8481b; } .tag.rule { background: rgba(127,127,127,.18); }
 .flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; margin: 12px 0; } .flow div { flex: 1 1 150px; border: 1px solid rgba(127,127,127,.35); border-radius: 10px; padding: 9px 11px; font-size: 13.5px; }
 .flow div b { display: block; font-size: 14.5px; } .flow .hl { border-color: var(--acc); border-width: 2px; }
</style></head><body>
<h1>Настоящие вызовы языковой модели</h1>
<p class="sub">Что именно агент отправляет модели, что получает в ответ и что делает с ответом. Всё ниже — из записей настоящих прогонов, без пересказа.</p>

{{INTRO}}{{BODY}}
</body></html>
'''

INTRO = '''<h2>0. На чём построен агент и где в нём модель</h2>
<p>Агент — обычная программа на Python, написанная командой; готовых «агентных» библиотек в ней нет. Десять раз в секунду она делает одно и то же:</p>
<div class="flow">
 <div><b>1. Читает датчики</b>одометрия, лидар, батарея, датчик образцов</div>
 <div><b>2. Обновляет картину мира</b>где могут быть образцы, сколько стоит пол, где опасно. Это расчёт, без модели</div>
 <div class="hl"><b>3. Если есть повод — выбирает подцели</b>собран образец, пропал кандидат, сбой датчика… Здесь стоит сменная деталь: <b>правило</b> или <b>языковая модель</b></div>
 <div><b>4. Едет</b>путь по карте, скорости, объезд, сбор, возврат по заряду. Без модели</div>
</div>
<table><thead><tr><th style="width:20%">Вариант агента</th><th style="width:30%">Кто выбирает подцели</th><th>Что ещё</th></tr></thead><tbody>
<tr><td><code>fixed</code></td><td>никто: маршрут посчитан заранее</td><td>база для сравнения</td></tr>
<tr><td><code>adaptive</code></td><td>правило: «уверенность, делённая на цену пути»</td><td>основной агент, на нём посчитано большинство опытов</td></tr>
<tr><td><code>scientist</code></td><td>правило</td><td>плюс расследования странностей: объяснения и опыты выбирает код</td></tr>
<tr><td><code>adaptive_llm</code></td><td><span class="tag llm">модель</span>, при отказе — правило</td><td>всё остальное как у <code>adaptive</code></td></tr>
<tr><td><code>scientist_llm</code></td><td><span class="tag llm">модель</span>, при отказе — правило</td><td>та же модель — автор и критик в расследованиях</td></tr>
</tbody></table>
<p>Вызов модели — это один HTTP-запрос к серверу МАИ в формате, общем для большинства моделей. Вот код, который его делает (файлы <code>did/planner.py</code> и <code>did/llm.py</code>):</p>
<div class="cols"><div><h4>Планировщик: спросить модель, проверить ответ, при отказе — правило</h4><pre class="tall">{{PLAN_SRC}}</pre></div>
<div><h4>Сам запрос</h4><pre class="tall">{{POST_SRC}}</pre></div></div>
'''

EMBED = PAGE.split('</style>')[0].replace('max-width: 1100px; padding: 18px 22px 60px;', 'padding: 0;').replace('background: var(--background, #fff);', '') + '</style></head><body>\n{{BODY}}\n</body></html>\n'

if __name__ == '__main__':
    main()
