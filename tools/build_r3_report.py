"""Собрать R3.md только из двух summary.json; сохраняет компактный снимок исходных чисел."""
import json
from pathlib import Path
from did.metrics import paired, summarize

ROOT = Path(__file__).resolve().parents[1]
e16 = json.loads((ROOT / 'runs/E16/summary.json').read_text())
real = json.loads((ROOT / 'runs/R3_real/summary.json').read_text())
assert len(e16['runs']) == 960 and not e16['errors']
assert len(real['runs']) == 48
out = ROOT / 'research/findings'; out.mkdir(exist_ok=True)
wrap = lambda rows: [{'level': r['level'], 'seed': r['seed'], 'metrics': {'score': r['score']}} for r in rows]
f = lambda n: f'{n:.2f}'
interval = lambda c: f'[{f(c[0])}; {f(c[1])}]'
def conclusion():
    parts = []
    for mode in ('frozen', 'charged'):
        rows = [r for r in real['runs'] if r['mode'] == mode]
        base = wrap([r for r in rows if r['strategy'] == 'single'])
        diffs = {arm: paired(wrap([r for r in rows if r['strategy'] == arm]), base, 'score')
                 for arm in ('critic', 'scored', 'scored_calc')}
        descriptions = ', '.join(f"{arm}: {d['mean']:+.2f} {interval(d['ci'])}" for arm, d in diffs.items())
        parts.append(f"{mode}: разности к single — {descriptions}; " +
                     ('положительное различие в этой серии показано для ' + ', '.join(k for k,d in diffs.items() if d['ci'][0]>0)
                      if any(d['ci'][0]>0 for d in diffs.values()) else 'положительный выигрыш не показан'))
    a = wrap([r for r in real['runs'] if r['mode']=='frozen' and r['strategy']=='scored'])
    b = wrap([r for r in real['runs'] if r['mode']=='frozen' and r['strategy']=='scored_calc'])
    d = paired(a,b,'score')
    attribution = ('Дополнительная заслуга выбора модели относительно расчёта не показана; предложения остаются вкладом модели, а контроль проверяет только шаг выбора'
                   if d['ci'][0] <= 0 <= d['ci'][1] else 'Выбор модели отличается от выбора расчёта в этой серии')
    return '; '.join(parts) + '. ' + attribution + '; обобщать на другие сценарии и модели по шести парам нельзя.'

lines = ['# R3. Оркестрация одной модели и цена ожидания',
'## Вопрос',
'Меняет ли обвязка качество решений одной и той же модели, кто создаёт выигрыш — модель или расчёт — и сохраняется ли он при оплате времени ожидания?',
'## Гипотеза и что её опровергнет',
'До сравнения задана гипотеза: scored повышает счёт относительно single; critic и vote дают промежуточный эффект. Контроль scored_calc отделяет выбор модели от выбора расчётом. Если интервалы парных разностей включают ноль, различие не показано. Эквивалентность не проверялась: допуск заранее не установлен.',
'## Что сделано',
'В research/R3 влит main (478bbed), включая openrouter, min_tokens и тест стабильности генератора. Прямая инструкция заказчика разрешила сеть и модели МАИ поверх старого запрета протокола; Gazebo, ROS, Ollama не запускались.',
'`did/llm_orchestration.py`: single оставлен на прежнем request_plan; critic — автор, критик, исправление при revise (даже без списка issues); vote — три запроса с разными индексами и голосованием по первой подцели, включая координаты goto. Температура клиента 0.2, роль в журнале сама по себе не меняет запрос. Генераторы scored получают номер предложения и предыдущие варианты, поэтому кэш не превращает K запросов в один.',
'Из scored удалён скрыто добавлявшийся план HeuristicPlanner: теперь и scored, и scored_calc имеют только предложения модели. scored_calc не спрашивает модель о выборе; порядок расчёта зафиксирован до реальных результатов: допустимость, ожидаемые образцы, информационная польза разведки на заряд, остаток заряда. При негодном выборе scored использует этот же расчёт. При полном отсутствии планов все схемы переходят к правилу.',
'Таблица оценивает первую подцель плюс путь домой. Прежняя сумма cost_to нескольких целей считала каждый путь от текущей позиции, могла повторно учитывать один образец и приписывала goto нулевой расход. Сейчас goto без оценки пути считается неоценимым, разведка имеет явную информационную ценность. Это приближение, а не прогноз всего маршрута: агент может выполнить последующие подцели очереди до очередного триггера.',
'`did/llm_mock.py`: normal сохраняет прежний алгоритм, timid возвращается при <70% заряда, random случайно выбирает допустимую цель, careless иногда игнорирует допустимость. Убран отдельный порог 10 ед., который делал timid смелее именно при выборе таблицы; timid теперь максимизирует остаток. Частота беспечности в выборе приведена к 1/3. Характеры заданы вручную и не моделируют распределение ошибок настоящей модели; E16 проверяет обвязку, не качество LLM.',
'`did/agent.py`, `did/planner.py`, `did/runner.py`: флаг стратегии, отказ на опечатке, запись совпадения первой исполнимой подцели с правилом на том же состоянии (включая fallback). `did/llm.py`: роль запроса и характер mock. `did/experiments.py`, `did/metrics.py`, `experiments/E16.yaml`: условия mock, счётчики вызовов, контроль scored_calc. `tests/test_llm_orchestration.py`: схемы, ошибки, отсутствие внедрения правила, первая подцель и совместимость single.',
'`did/orchestration_eval.py`: воспроизводимые 48 прогонов (4 способа × 6 сценариев × 2 режима), модель qwen3.8-flash-next, http, use_schema=True, min_tokens=3000, cache=True, максимум три параллельных запроса. Ключ только в окружении. Таймаут клиента 30 с, до двух транспортных повторов; они входят во время ответа. E16 пересчитан полностью: 960 прогонов, по 20 сценариев на уровень и характер. `tools/build_r3_report.py` создаёт отчёт и компактный снимок `R3-results.json` из сводок; новые runs в коммит не входят.',
'Проверка: ./px python -m pytest tests -q — 326 passed, 1 skipped. Первый запуск под параллельной нагрузкой упал на существующем тесте скорости rank_points; повтор после E16 прошёл, код этой функции не менялся.',
'## Результат',
'Источник имитатора — `runs/E16/summary.json`. Среднее и 95% бутстреп-интервал среднего; Δ — парная разность со single того же характера и уровня (20 пар). Доля образцов и возвратов — средние по прогонам.',
'| Уровень / характер | Способ | Счёт [95%] | Δ к single [95%] | Доля образцов | Возвраты | Обращений/прогон |',
'|---|---|---:|---:|---:|---:|---:|']
for level in ('medium', 'hard'):
 for cond in ('normal','timid','random','careless'):
  base = [r for r in e16['runs'] if r['level']==level and r['condition']==cond and r['arm']=='single']
  for arm in ('rule','single','critic','vote','scored','scored_calc'):
   rs = [r for r in e16['runs'] if r['level']==level and r['condition']==cond and r['arm']==arm]
   score=next(g['stats']['score'] for g in e16['groups'] if g['arm']==arm and g['condition']==cond and g['level']==level)
   diff=paired(rs,base,'score')
   claim=next((c for c in e16['claims'] if c['a']==arm and c['b']=='single' and c['metric']=='score'), None)
   if claim:
    diff=next(c['pair'] for c in claim['cells'] if c['condition']==cond and c['level']==level)
   lines.append(f'| {level}/{cond} | {arm} | {f(score["mean"])} {interval(score["ci"])} | {f(diff["mean"])} {interval(diff["ci"])} | {f(summarize(rs,"samples_share")["mean"])} | {f(summarize(rs,"returned")["mean"])} | {f(summarize(rs,"llm_calls")["mean"])} |')
lines += ['', 'Сверка single@normal с rule (разности single − rule по каждому сценарию 1001–1020):']
for level in ('medium','hard'):
 a=[r for r in e16['runs'] if r['level']==level and r['condition']=='normal' and r['arm']=='single']
 b={r['seed']:r for r in e16['runs'] if r['level']==level and r['condition']=='normal' and r['arm']=='rule'}
 lines.append(f'- {level}: '+', '.join(f"{r['metrics']['score']-b[r['seed']]['metrics']['score']:+.2f}" for r in a)+'.')
lines += ['Различие возможно без изменения single: normal-mock выдаёт очередь investigate + explore, дополнительно проверяет запас 2 ед. на округлённых cost_to/cost_back и ранжирует разведку по unseen_share. HeuristicPlanner выдаёт одну цель, доверяет feasible и при наличии gain_bits использует его. Однако на всех 40 сценариях E16 счёт совпал точно: в журналах первая исполнимая подцель normal-mock совпадает с правилом, а в adaptive используется mass, а не gain_bits. Дополнительная очередь не изменила итог при этих триггерах перепланирования; это не доказывает тождество алгоритмов на всех состояниях. Ветка single и её системный промпт не менялись; тест сравнивает legacy LLMPlanner с plan_single.',
'', 'Источник настоящей модели — `runs/R3_real/summary.json`. В каждой строке шесть сценариев: medium-1001, medium-1002, medium-1003, hard-1001, hard-1002, hard-1003. Обращения включают исправления; время — сумма измеренных исходных задержек ответов, в том числе восстановленных из кэша. Совпадение с правилом — суммарные совпадения / все решения с обращением к модели, а не среднее процентов.',
'| Режим | Способ | Счёт [95%] | Δ к single [95%] | Собрано/прогон | Возвратов/6 | Обращений/прогон | Ответы с/прогон | Задано ожидания с/прогон | Совпало с правилом | Негодных обменов |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for mode in ('frozen','charged'):
 base=[r for r in real['runs'] if r['mode']==mode and r['strategy']=='single']
 for arm in ('single','critic','scored','scored_calc'):
  rs=[r for r in real['runs'] if r['mode']==mode and r['strategy']==arm]
  score=summarize(wrap(rs),'score'); diff=paired(wrap(rs),wrap(base),'score')
  lines.append(f'| {mode} | {arm} | {f(score["mean"])} {interval(score["ci"])} | {f(diff["mean"])} {interval(diff["ci"])} | {f(sum(r["collected"] for r in rs)/6)} | {sum(r["returned"] for r in rs)}/6 | {f(sum(r["requests"] for r in rs)/6)} | {f(sum(r["wait_s"] for r in rs)/6)} | {f(sum(r["decisions"] * r["llm_wait_s"] for r in rs)/6)} | {sum(r["rule_matches"] for r in rs)}/{sum(r["decisions"] for r in rs)} | {sum(r["failed_exchanges"] for r in rs)} |')
lines += ['', 'Парные разности счёта по шести сценариям в указанном порядке; это малая описательная серия, а интервалы не учитывают повторные проверки нескольких способов:']
for mode in ('frozen','charged'):
 base={ (r['level'],r['seed']):r for r in real['runs'] if r['mode']==mode and r['strategy']=='single'}
 for arm in ('critic','scored','scored_calc'):
  rs=[r for r in real['runs'] if r['mode']==mode and r['strategy']==arm]
  lines.append(f'- {mode}, {arm} − single: '+', '.join(f"{r['score']-base[r['level'],r['seed']]['score']:+.2f}" for r in rs)+'.')
for mode in ('frozen','charged'):
 a=wrap([r for r in real['runs'] if r['mode']==mode and r['strategy']=='scored'])
 b=wrap([r for r in real['runs'] if r['mode']==mode and r['strategy']=='scored_calc'])
 d=paired(a,b,'score'); lines.append(f'- {mode}, scored − scored_calc: среднее {f(d["mean"])} {interval(d["ci"])}.')
lines += ['', 'Калибровка llm_wait_s (секунд на решение, сумма задержек / число решений в frozen): '+', '.join(f'{k}={v:.3f}' for k,v in real['calibration'].items())+'.',
'В frozen мир стоит во время HTTP. В charged именно llm_wait_s удерживает нулевую скорость после синхронного ответа; FastSim продолжает такты, часы, события и расход стоящего робота, лимит 600 с. План применяется сразу, но движение блокируется до wait_until; правило принудительного возврата продолжает работать. Задано ожидания — число решений × llm_wait_s; фактически до окончания прогона может пройти меньше (ожидание последнего решения обрезается лимитом). Это средняя фиксированная задержка способа, а не точное проигрывание каждого измеренного ответа, и число решений/запросов может измениться. Измеренные задержки новых запросов charged могут отличаться от frozen, в частности из-за серверного кэша или нагрузки; это ограничивает вывод о плате за реальное ожидание.',
f'Всего полученных некэшированных ответов модели в локальном счётчике кэша: {real["cache_calls_total"]}. Счётчик считает HTTP-ответы, включая пустые; сетевые попытки без ответа не считаются в нём. Он включает прерванный подготовительный запуск; повторный запрос из кэша не является новым вызовом. В законченных записях негодных обменов: {sum(r["failed_exchanges"] for r in real["runs"])}; HTTP-попытки хранятся отдельно в снимке.',
'## Вывод',
conclusion(),
'## Ограничения',
'Шесть сценариев и один сохранённый ответ на запрос не оценивают устойчивость к случайности декодирования и другим моделям. Парные интервалы описательны, уровни различны, поправки на множественные сравнения нет. Таблица не знает скрытого положения образцов, не оценивает вероятность поиска в единицах будущего счёта и не строит полный маршрут; её лексикографический порядок — наше допущение. Совпадение с правилом не доказывает, что рассуждение модели полезно: совпавшее решение может быть fallback. Имитатор специально создан из правил и не является слабой настоящей LLM. Результаты быстрого симулятора не проверены в Gazebo. Vote на настоящей модели не запускался: обязательная серия ограничена четырьмя способами. В реальной серии встречались SSL/EOF и обрыв JSON по лимиту: неудачная критика сохраняет план автора, а неудачные предложения и выбор могут включить fallback. Поэтому нулевой эффект critic нельзя целиком приписывать качеству модели. Кэш сохраняет исходные успешные ответы и задержки, но новый запуск на другом сервере может дать другие числа.',
'## Как повторить',
'```bash',
'cd /Users/a/MAI/DID-research/R3',
'./px python -m pytest tests -q',
'./px python -m did.experiments E16 --jobs 4',
'set -a; . /Users/a/MAI/DID/.env; set +a',
'./px python -m did.orchestration_eval --jobs 3',
'./px python tools/build_r3_report.py',
'```',
'Кэш: runs/_llm_cache; записи: runs/R3_real и runs/E16. Сохранить эти папки для бесплатного повторения. По умолчанию законченные записи с той же калибровкой читаются без сети: это пересборка сводки. Для повторения симуляции добавить --rerun: успешные ответы берутся из кэша; сетевые сбои не кэшируются и могут привести к новым обращениям. Для полностью независимого запуска нужен отдельный рабочий каталог/кэш, и это уже новые обращения. Снимок чисел в research/findings/R3-results.json содержит результаты и пути, без ключа и ответов модели.']
text = ''
for i, line in enumerate(lines):
    separator = '\n' if i and ((line.startswith('|') and lines[i-1].startswith('|')) or (line.startswith('- ') and lines[i-1].startswith('- '))) else '\n\n'
    text += (separator if i else '') + line
(out/'R3.md').write_text(text, encoding='utf-8')
(out/'R3-results.json').write_text(json.dumps({'E16':{'generated':e16['generated'], 'groups':[{'arm':g['arm'],'condition':g['condition'],'level':g['level'],'stats':{k:g['stats'][k] for k in ('score','samples_share','returned','llm_calls')}} for g in e16['groups']], 'claims':e16['claims'], 'errors':e16['errors'], 'runs':[{'arm':r['arm'],'condition':r['condition'],'level':r['level'],'seed':r['seed'],'metrics':{k:r['metrics'][k] for k in ('score','samples_collected','samples_share','returned','llm_calls')}} for r in e16['runs']]}, 'real':real},ensure_ascii=False,indent=1))
