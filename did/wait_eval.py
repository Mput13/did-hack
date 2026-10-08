"""R16: вспомогательные проверки режима «действовать, пока модель думает» (did/waiting.py).

    ./px python -m did.wait_eval missions            # миссии словами (M1–M4) на имитаторе с задержкой 15 с, сети нет
    ./px python -m did.wait_eval cached              # те же миссии на записанных ответах настоящей модели: только
                                                     # кэш, сеть не трогается; сколько запросов кэш не покрыл
    ./px python -m did.wait_eval real                # шесть сценариев R3 на настоящей модели, время ответа настоящее
    ./px python -m did.wait_eval real --cache-only   # строгий повтор: ответы только из кэша, сеть не трогается
    ./px python -m did.wait_eval real --no-run       # пересобрать сводку по записям

Основное измерение — опыт E24 (experiments/E24.yaml); здесь то, что в его форму не укладывается.

missions. Правило текст миссии не читает, поэтому нужен имитатор, который её «понимает»: MissionResponder
отвечает как правило, но для M1 после второго образца зовёт домой, для M2 бережёт 30 единиц заряда, для M3
не берёт целей правее границы, для M4 после первого штрафа зовёт домой (счётчик штрафов — state_penalties).
Варианты — правило, имитатор без ожидания, «стоять и ждать», «весь план правила в ожидании», «только
обратимое в ожидании» с привязью 0,5 м и без привязи; обвязка — как у варианта llm_ask из R13 (модель спрашивают по каждому поводу).
Записи — runs/R16_missions/<миссия>_<вариант>/, сводка — runs/R16_missions/summary.json.

real. Те же шесть сценариев и та же модель, что в R3. Время ответа — не среднее, а своё у каждого ответа
(llm_wait_measured): столько секунд прогона робот ждёт. Ответы кэшируются (runs/_llm_cache); сколько было
настоящих обращений к сети, пишется в сводку runs/R16_real/summary.json.

cached. Миссии M1–M4, шесть сценариев R13, варианты «весь план правила» и «привязь», обвязка llm_ask. Ответы —
только те, что уже записаны (кэши рабочих деревьев R16, R13, R3, J1 и основного каталога, только чтение).
Запроса нет в кэше — в сеть не иду: он считается непокрытым, на него отвечает правило (как при отказе
модели), и прогон идёт дальше, чтобы посчитать все непокрытые. Такой прогон — не измерение модели; в сводке
runs/R16_cached/summary.json главное — счёт покрытых и непокрытых запросов.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from .llm import CacheMiss, LLMError, LocalClient, llm_stats, load_env
from .llm_cache import CachedClient, OfflineClient, ReplyCache
from .llm_mock import find_state, mock_plan
from .mission_criteria import MISSIONS, RETURN_DELAY_S, RIGHT_HALF_X, verify_mission
from .recorder import load_trace
from .runner import RUNS, run_episode

ENV_FILE = '/Users/a/MAI/DID/.env'
MODEL = 'qwen3.8-flash-next'
LLM = {'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True, 'timeout_s': 90}
REAL_SCENARIOS = [(level, seed) for level in ('medium', 'hard') for seed in (1001, 1002, 1003)]
REAL_ARMS = {'stand': {'llm_wait_measured': True},
             'act': {'llm_wait_measured': True, 'llm_act_while_waiting': 'rule'},
             'safe': {'llm_wait_measured': True, 'llm_act_while_waiting': 'leash'}}

WAIT_S = 15.0
MISSION_IDS = ('M1', 'M2', 'M3', 'M4')
MISSION_EXTRA = {'M4': {'state_penalties': True}}      # M4: модель видит счётчик штрафов (вариант llm_ask_pen из R13)
MISSION_SCENARIOS = [(level, seed) for level in ('medium', 'hard') for seed in range(1001, 1011)]
ASK = {'mission_triggers': True, 'llm_min_interval_s': 0.0}       # обвязка llm_ask из R13
MISSION_ARMS = {'rule': None,
                'nowait': {**ASK},
                'stand': {**ASK, 'llm_wait_s': WAIT_S},
                'act': {**ASK, 'llm_wait_s': WAIT_S, 'llm_act_while_waiting': 'rule'},
                'safe': {**ASK, 'llm_wait_s': WAIT_S, 'llm_act_while_waiting': 'leash'},
                # привязь нулевой длины: до ответа стоять на месте (кроме выезда из опасной зоны)
                'still': {**ASK, 'llm_wait_s': WAIT_S, 'llm_act_while_waiting': 'leash', 'llm_wait_leash_m': 0.0}}
M3_MARGIN_M = 0.2         # имитатор M3 берёт цели не ближе этого к границе: робот не точка
M2_KEEP = 30.0 + 1.0      # имитатор M2 берёт цели, после которых с дорогой домой останется не меньше этого
CACHE_ROOTS = ['/Users/a/MAI/DID-research/R16/runs/_llm_cache', '/Users/a/MAI/DID-research/R13/runs/_llm_cache',
               '/Users/a/MAI/DID-research/R3/runs/_llm_cache', '/Users/a/MAI/DID-research/J1/runs/_llm_cache',
               '/Users/a/MAI/DID/runs/_llm_cache']
CACHED_SCENARIOS = [(level, seed) for level in ('medium', 'hard') for seed in (1001, 1002, 1003)]
CACHED_ARMS = {'act': {**ASK, 'llm_wait_measured': True, 'llm_act_while_waiting': 'rule'},
               'leash': {**ASK, 'llm_wait_measured': True, 'llm_act_while_waiting': 'leash'}}


class MissionResponder:
    """Имитатор, который «понимает» миссию: правило mock_plan с одним ограничением из текста.

    M1 — после второго образца только return_base; M2 — цели, после которых с дорогой домой останется меньше
    30 единиц заряда, не рассматриваются; M3 — цели правее границы не рассматриваются; M4 — после первого
    штрафа только return_base. Остальные миссии — обычный mock_plan.
    """

    def __init__(self, mission_id):
        self.mission_id = mission_id

    def __call__(self, messages):
        state = find_state(messages)
        if self.mission_id == 'M1' and (state.get('samples') or {}).get('collected', 0) >= 2:
            plan = {'reasoning': 'Два образца собраны — по миссии возвращаюсь на базу.', 'hypotheses': [],
                    'subgoals': [{'type': 'return_base'}]}
        elif self.mission_id == 'M4' and ((state.get('penalties') or {}).get('total') or 0) > 0:
            plan = {'reasoning': 'Получен штраф — по миссии возвращаюсь на базу.', 'hypotheses': [],
                    'subgoals': [{'type': 'return_base'}]}
        elif self.mission_id == 'M2':
            plan = mock_plan({**state, 'battery': float(state.get('battery') or 0.0) - M2_KEEP})
            plan['reasoning'] = 'Берегу 30 единиц заряда. ' + plan['reasoning']
        elif self.mission_id == 'M3':
            left = {name: [it for it in state.get(name) or [] if it.get('x', 9.0) <= RIGHT_HALF_X - M3_MARGIN_M]
                    for name in ('candidates', 'explore_points')}
            plan = mock_plan({**state, **left})
            plan['reasoning'] = 'В правую половину не еду. ' + plan['reasoning']
        else:
            plan = mock_plan(state)
        return json.dumps(plan, ensure_ascii=False)


def mission_row(m_id, arm, level, seed, file):
    """Строка сводки: исход по критерию R13 и то, что в нём зависит от самого ожидания."""
    trace = load_trace(RUNS / file)
    res = trace['result']
    check = verify_mission(m_id, trace.get('track', {}), trace.get('events', []), res, plans=trace.get('plans'),
                           modes=trace.get('modes'), truth=trace.get('truth'))
    d = check['details']
    row = {'mission': m_id, 'arm': arm, 'level': level, 'seed': seed, 'file': file,
           'success': check['success'], 'outcome': check['outcome'], 'score': res['score'],
           'collected': res['samples_collected'], 'returned': bool(res['returned']), 'time_s': res['time'],
           'idle_s': res.get('idle_s'), 'llm_calls': res.get('llm_calls'), 'llm_wait': res.get('llm_wait')}
    if m_id == 'M1':
        # Критерий R13 требует поворота домой за 2 с после второго сбора. Ответ модели идёт 15 с, поэтому
        # отдельно считается суть миссии: ровно два образца и возврат (сколько бы ни шёл ответ).
        row.update(exactly_two=res['samples_collected'] == 2 and bool(res['returned']),
                   return_delay_s=d.get('return_delay_s'), late_limit_s=RETURN_DELAY_S)
    if m_id == 'M2':
        row.update(battery=d.get('battery_remaining'))
    if m_id == 'M3':
        row.update(time_right_s=d.get('time_right_s'), max_x=d.get('max_x'), samples_right=d.get('samples_right'))
    if m_id == 'M4':
        # Критерий R13 требует поворота домой за 2 с после штрафа — при ответе через 15 с это невыполнимо.
        # Суть миссии: штраф был, после него робот ничего не собрал, домой повернул и вернулся.
        row.update(had_penalty=d.get('had_penalty'), reaction_s=d.get('reaction_s'),
                   collected_after=d.get('collected_after_penalty'),
                   home_after_penalty=bool(d.get('had_penalty') and res['returned'] and d.get('reaction_s') is not None
                                           and d['reaction_s'] >= 0 and not d.get('collected_after_penalty')))
    return row


def _mission_run(task):
    m_id, arm, level, seed, run, out = task
    file = f'{out}/{m_id}_{arm}/{level}-{seed}.json.gz'
    if run:
        cfg = MISSION_ARMS[arm]
        config = {'mission': MISSIONS[m_id]['text'], **(cfg or {}), **(MISSION_EXTRA.get(m_id) or {} if cfg else {})}
        if cfg is None:
            run_episode(level, seed, 'adaptive', experiment=out, arm=f'{m_id}_{arm}', config=config, truth=True)
        else:
            run_episode(level, seed, 'adaptive_llm', experiment=out, arm=f'{m_id}_{arm}', config=config, truth=True,
                        llm={'client': LocalClient(MissionResponder(m_id))})
    elif not (RUNS / file).exists():
        return None
    return mission_row(m_id, arm, level, seed, file)


def mission_table(rows):
    table = []
    for m_id in MISSION_IDS:
        for arm in MISSION_ARMS:
            sel = [r for r in rows if r['mission'] == m_id and r['arm'] == arm]
            if not sel:
                continue
            n = len(sel)
            outcomes = {}
            for r in sel:
                outcomes[r['outcome']] = outcomes.get(r['outcome'], 0) + 1
            entry = {'mission': m_id, 'arm': arm, 'n': n, 'success': sum(r['success'] is True for r in sel),
                     'outcomes': outcomes, 'score_mean': round(sum(r['score'] for r in sel) / n, 2),
                     'collected_mean': round(sum(r['collected'] for r in sel) / n, 2),
                     'returned': sum(r['returned'] for r in sel),
                     'idle_s_mean': round(sum(r['idle_s'] or 0 for r in sel) / n, 1),
                     'time_s_mean': round(sum(r['time_s'] for r in sel) / n, 1)}
            if m_id == 'M1':
                entry['exactly_two'] = sum(r['exactly_two'] for r in sel)
                entry['more_than_two'] = sum(r['collected'] > 2 for r in sel)
            if m_id == 'M2':
                entry['battery_mean'] = round(sum(r['battery'] or 0 for r in sel) / n, 1)
            if m_id == 'M4':
                pen = [r for r in sel if r['had_penalty']]
                entry.update(with_penalty=len(pen), home_after_penalty=sum(r['home_after_penalty'] for r in pen),
                             by_criterion=sum(r['success'] is True for r in pen),
                             collected_after=sum(r['collected_after'] or 0 for r in pen))
            if m_id == 'M3':
                entry['entered_right'] = sum((r['time_right_s'] or 0) > 0 or (r['samples_right'] or 0) > 0 for r in sel)
                entry['time_right_s_mean'] = round(sum(r['time_right_s'] or 0 for r in sel) / n, 1)
                entry['max_x'] = max(r['max_x'] for r in sel)        # как глубоко за границу 0,5 м (центр робота)
            table.append(entry)
    return table


def run_missions(args):
    out = args.out or 'R16_missions'
    tasks = [(m, arm, lv, sd, not args.no_run, out) for m in MISSION_IDS for arm in MISSION_ARMS
             for lv, sd in MISSION_SCENARIOS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for r in pool.map(_mission_run, tasks) if r]
    summary = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'wait_s': WAIT_S,
               'scenarios': [f'{lv}-{sd}' for lv, sd in MISSION_SCENARIOS],
               'missions': {m: MISSIONS[m]['text'] for m in MISSION_IDS},
               'arms': {k: v for k, v in MISSION_ARMS.items()}, 'table': mission_table(rows), 'runs': rows}
    path = RUNS / out / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    for e in summary['table']:
        extra = {'M1': lambda: f"ровно два и вернулся {e['exactly_two']}, больше двух {e['more_than_two']}",
                 'M2': lambda: f"остаток заряда в среднем {e['battery_mean']}",
                 'M3': lambda: f"заезжал вправо {e['entered_right']}, там в среднем {e['time_right_s_mean']} с",
                 'M4': lambda: f"со штрафом {e['with_penalty']}, из них домой без сбора {e['home_after_penalty']}"
                 }[e['mission']]()
        print(f"{e['mission']} {e['arm']:7s} по критерию R13 {e['success']:2d} из {e['n']}  {extra}  счёт "
              f"{e['score_mean']:6.2f}  собрано {e['collected_mean']:.2f}  простой {e['idle_s_mean']:5.1f} с  "
              f"{e['outcomes']}")
    print(f'сводка: {path}')


class _Roots(ReplyCache):
    """Несколько папок кэша только для чтения: ответ ищется по очереди во всех."""

    def __init__(self, roots):
        super().__init__(roots[0])
        self.caches = [ReplyCache(r) for r in roots]

    def get(self, model, key):
        return next((hit for c in self.caches if (hit := c.get(model, key))), None)

    def get_failure(self, model, key):
        return next((hit for c in self.caches if (hit := c.get_failure(model, key))), None)


class CoverageClient(CachedClient):
    """Строгий кэш со счётом: что покрыто записанными ответами, а что нет. В сеть не ходит никогда."""

    def __init__(self, roots=None):
        super().__init__(OfflineClient(MODEL, use_schema=True), strict=True)
        self.cache = _Roots(roots or CACHE_ROOTS)
        self.hits = self.misses = 0
        self.first_miss = None               # номер запроса (с единицы), на котором кэш кончился

    def chat(self, messages, **kw):
        try:
            reply = super().chat(messages, **kw)
        except CacheMiss as e:
            self.misses += 1
            if self.first_miss is None:
                self.first_miss = self.hits + self.misses
            raise LLMError('нет записанного ответа; сеть не трогаю') from e
        self.hits += 1
        return reply


def _cached_run(task):
    m_id, arm, level, seed, out = task
    client = CoverageClient()
    res = run_episode(level, seed, 'adaptive_llm', experiment=out, arm=f'{m_id}_{arm}', truth=True,
                      config={'mission': MISSIONS[m_id]['text'], **CACHED_ARMS[arm]}, llm={'client': client})
    row = mission_row(m_id, arm, level, seed, res['file'])
    row.update(hits=client.hits, misses=client.misses, first_miss=client.first_miss)
    return row


def run_cached(args):
    out = args.out or 'R16_cached'
    tasks = [(m, arm, lv, sd, out) for m in MISSION_IDS for arm in CACHED_ARMS for lv, sd in CACHED_SCENARIOS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(_cached_run, tasks))
    table = []
    for m_id in MISSION_IDS:
        for arm in CACHED_ARMS:
            sel = [r for r in rows if r['mission'] == m_id and r['arm'] == arm]
            table.append({'mission': m_id, 'arm': arm, 'n': len(sel), 'hits': sum(r['hits'] for r in sel),
                          'misses': sum(r['misses'] for r in sel), 'fully_covered': sum(r['misses'] == 0 for r in sel),
                          'covered_before_first_miss': sum((r['first_miss'] or r['hits'] + 1) - 1 for r in sel),
                          'success_fully_covered': sum(r['success'] is True for r in sel if r['misses'] == 0)})
    summary = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': MODEL, 'cache_roots': CACHE_ROOTS,
               'scenarios': [f'{lv}-{sd}' for lv, sd in CACHED_SCENARIOS], 'arms': CACHED_ARMS,
               'requests': sum(r['hits'] + r['misses'] for r in rows), 'hits': sum(r['hits'] for r in rows),
               'misses': sum(r['misses'] for r in rows), 'table': table, 'runs': rows}
    path = RUNS / out / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    for e in table:
        print(f"{e['mission']} {e['arm']:6s} прогонов {e['n']}  запросов из кэша {e['hits']}, не покрыто {e['misses']}; "
              f"до первого промаха {e['covered_before_first_miss']}; прогонов целиком из кэша {e['fully_covered']}")
    print(f"всего запросов {summary['requests']}: из кэша {summary['hits']}, не покрыто {summary['misses']}; сеть не "
          f"трогалась; сводка: {path}")


def real_row(arm, level, seed, file):
    trace = load_trace(RUNS / file)
    res = trace['result']
    stats = llm_stats(trace.get('llm') or [])
    return {'arm': arm, 'level': level, 'seed': seed, 'file': file, 'score': res['score'],
            'collected': res['samples_collected'], 'total': res['samples_total'], 'returned': bool(res['returned']),
            'hazard_hits': res.get('hazard_hits', 0), 'time_s': res['time'], 'idle_s': res.get('idle_s'),
            'llm_calls': res.get('llm_calls'), 'llm_wait': res.get('llm_wait'), 'plans': res.get('plans'),
            'requests': stats['requests'], 'failed': stats['failed'], 'cached': stats['cached'],
            'latency_s': {k: round(v / 1000, 1) for k, v in stats['latency_ms'].items()}}


def _real_run(task):
    arm, level, seed, run, out, cache_only = task
    file = f'{out}/{arm}/{level}-{seed}.json.gz'
    if run:
        try:
            run_episode(level, seed, 'adaptive_llm', experiment=out, arm=arm, config=REAL_ARMS[arm],
                        llm={**LLM, 'cache': 'only'} if cache_only else dict(LLM))
        except CacheMiss as e:
            return {'miss': str(e), 'arm': arm, 'level': level, 'seed': seed}
    elif not (RUNS / file).exists():
        return None
    return real_row(arm, level, seed, file)


def run_real(args):
    out = args.out or 'R16_real'
    run = not args.no_run
    if run and not args.cache_only:
        load_env(ENV_FILE)
    before = ReplyCache().calls().get(MODEL, 0)
    t0 = time.time()
    tasks = [(arm, lv, sd, run, out, args.cache_only) for arm in args.arms for lv, sd in REAL_SCENARIOS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for r in pool.map(_real_run, tasks) if r]
    missed = [r for r in rows if 'miss' in r]
    rows = [r for r in rows if 'miss' not in r]
    made = ReplyCache().calls().get(MODEL, 0) - before
    path = RUNS / out / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    keys = {(r['arm'], r['level'], r['seed']) for r in rows}
    rows = [r for r in old.get('runs') or [] if (r['arm'], r['level'], r['seed']) not in keys] + rows
    summary = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': MODEL, 'llm': LLM,
               'scenarios': [f'{lv}-{sd}' for lv, sd in REAL_SCENARIOS],
               # Настоящие обращения к сети по запускам: кэш считает их сам, повтор из кэша прибавляет ноль.
               'network_calls': (old.get('network_calls') or []) + ([{'arms': args.arms, 'calls': made}] if run else []),
               'runs': rows}
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    for arm in REAL_ARMS:
        sel = [r for r in rows if r['arm'] == arm]
        if sel:
            n = len(sel)
            print(f"{arm:6s} прогонов {n}  счёт {sum(r['score'] for r in sel) / n:6.2f}  штрафов за зоны "
                  f"{sum(r['hazard_hits'] for r in sel)}  вернулся {sum(r['returned'] for r in sel)}  простой "
                  f"{sum(r['idle_s'] or 0 for r in sel) / n:5.1f} с  запросов {sum(r['requests'] for r in sel)} "
                  f"(отказов {sum(r['failed'] for r in sel)})")
    print(f'настоящих обращений к сети за этот запуск: {made}; {time.time() - t0:.0f} с; сводка: {path}')
    for m in missed:
        print(f"  не досчитан (нет ответа в кэше): {m['arm']} {m['level']}-{m['seed']}: {m['miss']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('missions', 'cached', 'real'))
    ap.add_argument('--jobs', type=int, choices=(1, 2, 3), default=2)
    ap.add_argument('--arms', nargs='+', choices=list(REAL_ARMS), default=['stand', 'act'], help='только для real')
    ap.add_argument('--no-run', action='store_true', help='не считать прогоны, собрать сводку по записям')
    ap.add_argument('--cache-only', action='store_true', help='real: ответы только из кэша, сеть не трогается')
    ap.add_argument('--out', default=None, help='папка в runs/ (по умолчанию R16_missions и R16_real)')
    args = ap.parse_args(argv)
    {'missions': run_missions, 'cached': run_cached, 'real': run_real}[args.what](args)


if __name__ == '__main__':
    main()
