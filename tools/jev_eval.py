"""J1: модель решений Jev в роли сторожа миссии. Прогоны, сводка, калибровка, подбор формулировок.

Варианты:
  rule      — правило (HeuristicPlanner): текст миссии не читает;
  llm       — большая модель (qwen3.8-flash-next) на каждый повод, как лучший вариант R13: поводы «столкновение»
              и «заряд у порога», счётчик штрафов в сводке;
  jev       — правило под присмотром сторожа Jev; сомнительную подцель сторож отбрасывает;
  jev_llm   — то же, но сомнительный случай целиком решает большая модель;
  jev_wait, jev_llm_wait — те же два, и робот в симуляторе стоит столько, сколько сторож ждал ответов.

Записи — runs/<--out>/<миссия>_<вариант>/<сценарий>.json.gz, сводка — runs/<--out>/summary.json.

    ./px python tools/jev_eval.py --collect                      # состояния для подбора: правило на сценариях 1–20
    ./px python tools/jev_eval.py --tune v1 --per-mission 150    # формулировка против правды на этих состояниях
    ./px python tools/jev_eval.py --out J1_dev --seeds 1 2 3 --variants rule jev      # отладочные прогоны
    ./px python tools/jev_eval.py --final --variants rule jev jev_llm llm --jobs 3    # итог: сценарии 1101–1110
    ./px python tools/jev_eval.py --final --cache-only --out J1_replay ...   # повтор без сети и без денег
    ./px python tools/jev_eval.py --out J1 --report              # только пересобрать сводку и таблицы
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import CacheMiss, load_env                                  # noqa: E402
from did.llm_jev import CACHE_ONLY_ENV, Budget, JevClient, JevReply      # noqa: E402
from did.mission_criteria import MISSIONS, verify_mission                # noqa: E402
from did.mission_guard import (HI, LO, PROFILE, PROFILES, relevance_request, relevant_groups,   # noqa: E402
                               request_from)
from did.recorder import load_trace                                      # noqa: E402
from did.runner import RUNS, run_episode                                 # noqa: E402

MODEL = 'qwen3.8-flash-next'
ENV_FILE = '/Users/a/MAI/DID/.env'
LLM = {'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True}
LLM_TIMEOUT_S = None                   # срок ответа большой модели; None — из окружения (30 с)
ASK = {'mission_triggers': True}
VARIANTS = {
    'rule': ('adaptive', False, False, {}),
    'llm': ('adaptive_llm', True, False, {**ASK, 'llm_min_interval_s': 0.0, 'state_penalties': True}),
    'jev': ('adaptive', False, True, {**ASK, 'mission_guard': 'jev'}),
    'jev_llm': ('adaptive', True, True, {**ASK, 'mission_guard': 'jev_llm'}),
    'jev_wait': ('adaptive', False, True, {**ASK, 'mission_guard': 'jev', 'guard_wait': True}),
    'jev_llm_wait': ('adaptive', True, True, {**ASK, 'mission_guard': 'jev_llm', 'guard_wait': True}),
}
NAMES = {'rule': 'правило', 'llm': 'большая модель', 'jev': 'сторож Jev + правило',
         'jev_llm': 'сторож Jev + большая модель', 'jev_wait': 'сторож Jev + правило, ожидание в симуляторе',
         'jev_llm_wait': 'сторож Jev + большая модель, ожидание в симуляторе'}
FINAL_SEEDS = list(range(1101, 1111))
DEV_SEEDS = list(range(1, 21))
LEVELS = ('medium', 'hard')
FINAL_PLAN = ([(m, v) for m in ('M1', 'M2', 'M3', 'M4') for v in ('rule', 'llm', 'jev', 'jev_llm')]
              + [('M0', 'rule'), ('M0', 'jev')]
              + [(m, v) for m in ('M1', 'M2', 'M3', 'M4') for v in ('jev_wait', 'jev_llm_wait')])
BINS = (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0000001)


def make_jev(cache_only):
    """Единственное место, где создаётся клиент Jev. cache_only — только кэш: ни сети, ни ключа, ни расходов."""
    return JevClient(strict=bool(cache_only))


def budget_now():
    """Журнал расходов для сводки; None — журнала нет или он не читается (сводке это не мешает)."""
    try:
        b = Budget().read()
    except Exception:
        return None
    return b and {**b, 'spent_usd': Budget.spent(b)}


# --- правда для калибровки ---------------------------------------------------------------------

def truth_violates(m_id, f, o):
    """Нарушит ли подцель o миссию — по критерию миссии (did/mission_criteria.py) и числам состояния.

    M1: после второго сбора любая работа — нарушение («сразу возвращайся»); M2: по оценке агента на базе
    останется меньше 30; M3: цель правее x = 0,5 м; M4: штраф уже был. M0 ничего не запрещает.
    """
    return {'M0': False, 'M1': f['samples_collected'] >= 2, 'M2': o['battery_after'] < 30.0,
            'M3': o['x'] > 0.5, 'M4': f['penalties'] >= 1}[m_id]


def truth_done(m_id, f, opts):
    """Пора ли по миссии немедленно на базу. M2: любая оставшаяся цель привела бы на базу с зарядом ниже 30."""
    return {'M0': f['samples_collected'] >= f['samples_total'], 'M1': f['samples_collected'] >= 2,
            'M2': all(o['battery_after'] < 30.0 for o in opts), 'M3': False, 'M4': f['penalties'] >= 1}[m_id]


def answers_of(m_id, guard):
    """Ответы Jev одного решения с правдой: [(вопрос, вероятность «да», правда, спрошена ли подцель по делу)]."""
    f, opts = guard['facts'], guard['options']
    rows = [('done', guard['done_p'], truth_done(m_id, f, opts), True)]
    rows += [('violates', o['p'], truth_violates(m_id, f, o), o['id'] in guard['asked']) for o in opts
             if o.get('q', o['id']) == o['id']]         # дословно совпавшие вопросы заданы один раз
    return rows


def calibration(rows):
    """Таблица по корзинам вероятности «да»: сколько ответов и какая доля из них на самом деле «да»."""
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        part = [r for r in rows if lo <= r[1] < hi]
        out.append({'from': lo, 'to': min(hi, 1.0), 'n': len(part),
                    'mean_p': round(statistics.mean(r[1] for r in part), 3) if part else None,
                    'true_share': round(sum(r[2] for r in part) / len(part), 3) if part else None})
    return out


def confusion(rows, lo=LO, hi=HI):
    """Исходы по порогам: уверенно верно, уверенно неверно (по видам ошибки), сомнение."""
    c = {'n': len(rows), 'yes_right': 0, 'yes_wrong': 0, 'no_right': 0, 'no_wrong': 0, 'unsure': 0,
         'unsure_true': 0}
    for row in rows:
        p, truth = row[1], row[2]
        if p >= hi:
            c['yes_right' if truth else 'yes_wrong'] += 1
        elif p <= lo:
            c['no_wrong' if truth else 'no_right'] += 1
        else:
            c['unsure'] += 1
            c['unsure_true'] += bool(truth)
    return c


# --- прогоны -----------------------------------------------------------------------------------

def cell_path(out, task):
    m_id, variant, level, seed = task
    return RUNS / out / f'{m_id}_{variant}' / f'{level}-{seed}.json.gz'


def run_single(task, out, cache_only=False, jev=None):
    m_id, variant, level, seed = task
    agent, big, guard, cfg = VARIANTS[variant]
    llm = dict({**LLM, 'cache': 'only'} if cache_only else LLM) if big else {}
    if big and LLM_TIMEOUT_S and not cache_only:
        llm['timeout_s'] = LLM_TIMEOUT_S
    if guard:
        # Исследовательский прогон: отказ Jev останавливает прогон, а не подменяется ездой по правилу.
        llm['jev'] = {'client': jev, 'fallback': False}
    t0 = time.time()
    try:
        res = run_episode(level, seed, agent, experiment=out, arm=f'{m_id}_{variant}',
                          config={'mission': MISSIONS[m_id]['text'], **cfg}, llm=llm or None, save=True, truth=True)
    except Exception as e:
        if not isinstance(e, CacheMiss):
            traceback.print_exc()
        print(f'[{m_id}][{variant}][{level}-{seed}] ОШИБКА: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        cell_path(out, task).unlink(missing_ok=True)          # старая запись не должна сойти за результат
        return None
    print(f"[{m_id}][{variant}][{level}-{seed}] счёт {res['metrics'].get('score')} ({time.time() - t0:.0f} с)",
          flush=True)
    return res['file']


def _delayed(plans):
    """Те же планы, но решение о возврате (последний план, если он начинается с return_base) отнесено на момент,
    когда ответ моделей пришёл: время вопроса плюс длительность ответа."""
    if not plans or not plans[-1]['subgoals'] or plans[-1]['subgoals'][0].get('type') != 'return_base':
        return plans
    last = plans[-1]
    wait = last.get('wait_s') if 'guard' in last else (last.get('latency_ms') or 0) / 1000.0
    return plans[:-1] + [{**last, 't': last['t'] + (wait or 0.0)}]


def read_run(task, out):
    m_id, variant, level, seed = task
    path = cell_path(out, task)
    cell = {'mission': m_id, 'variant': variant, 'scenario': f'{level}-{seed}', 'status': 'ok'}
    if not path.is_file():
        return {**cell, 'status': 'no_record'}
    trace = load_trace(path)
    want = {'mission': MISSIONS[m_id]['text'], **VARIANTS[variant][3]}
    got = (trace.get('agent') or {}).get('config') or {}
    if any(got.get(k) != v for k, v in want.items()):
        return {**cell, 'status': 'other_settings'}
    result, plans = trace['result'], trace.get('plans', [])
    args = dict(track=trace.get('track', {}), events=trace.get('events', []), result=result,
                modes=trace.get('modes'), truth=trace.get('truth'))
    check = verify_mission(m_id, plans=plans, **args)
    # Проверка «с учётом времени ответа»: момент поворота домой берётся только из планов (в записи режимов
    # движения робот повернул сразу — в этих прогонах мир на время ответа стоял).
    late = verify_mission(m_id, plans=_delayed(plans),
                          **{**args, 'track': {k: v for k, v in args['track'].items() if k != 'mode'}})
    guards = [p['guard'] for p in plans if p.get('guard') and p['guard'].get('decision') != 'disabled']
    if len(guards) != sum(bool(p.get('guard')) for p in plans):
        return {**cell, 'status': 'guard_disabled'}       # сторож отключился: это прогон правила, а не сторожа
    llm_plans = [p for p in plans if p.get('latency_ms') is not None and (p['source'] in ('llm', 'fallback'))]
    llm_ms = [p['latency_ms'] for p in llm_plans]
    jev_ms = [g['latency_ms'] for g in guards]
    return {
        **cell, 'success': check['success'], 'outcome': check['outcome'], 'details': check['details'],
        'success_delayed': late['success'], 'outcome_delayed': late['outcome'],
        'score': result['score'], 'samples_collected': result['samples_collected'],
        'samples_total': result['samples_total'], 'returned': result['returned'],
        'battery': round(result['battery'], 1), 't_end': result.get('t'),
        'penalties': result['collisions'] + result['false_collects'] + result['hazard_hits'],
        'decisions': len(plans),
        'jev_calls': len(guards), 'jev_ms': jev_ms, 'jev_keys': [g['key'] for g in guards],
        'jev_fresh': sum(not g['cached'] for g in guards), 'jev_cost_usd': round(sum(g['cost_usd'] for g in guards), 7),
        'jev_tokens': sum(g['input_tokens'] for g in guards),
        'llm_calls': len(llm_plans), 'llm_ms': llm_ms,
        'escalated': sum(g['escalated'] for g in guards),
        'guard_decisions': {k: sum(g['decision'] == k for g in guards)
                            for k in ('go', 'return', 'exhausted', 'escalated')},
        'idle_s': round((sum(jev_ms) + sum(llm_ms)) / 1000.0, 2),
        'answers': [(g['key'], *row) for g in guards for row in answers_of(m_id, g)],
    }


def _sign_p(win, lose):
    """Точный двусторонний критерий знаков по парам, где исходы различаются."""
    from math import comb
    n = win + lose
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(win, lose) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pair_counts(ra, rb, both):
    """Критерий знаков по общим сценариям. Сравниваются только сценарии, где исход проверен у обоих вариантов:
    «не проверено» (success is None — в M4 штрафа не было) не выигрыш и не проигрыш, такие пары исключаются."""
    checked = [k for k in both if ra[k]['success'] is not None and rb[k]['success'] is not None]
    win = sum(ra[k]['success'] and not rb[k]['success'] for k in checked)
    lose = sum(rb[k]['success'] and not ra[k]['success'] for k in checked)
    return {'pairs': len(both), 'checked': len(checked),
            'both_success': sum(bool(ra[k]['success'] and rb[k]['success']) for k in checked),
            'a_only': win, 'b_only': lose, 'p_sign': _sign_p(win, lose)}


def _stat(values):
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'max': max(values) if values else None,
            'p90': sorted(values)[int(0.9 * (len(values) - 1))] if values else None}


def compile_summary(out, tasks):
    cells = [read_run(t, out) for t in tasks]
    table = []
    for m_id in MISSIONS:
        for variant in VARIANTS:
            arm = [c for c in cells if c['mission'] == m_id and c['variant'] == variant]
            if not arm:
                continue
            done = [c for c in arm if c['status'] == 'ok']
            n = max(1, len(done))
            entry = {'mission': m_id, 'variant': variant, 'expected': len(arm), 'missing': len(arm) - len(done),
                     'success': sum(c['success'] is True for c in done),
                     'failed': sum(c['success'] is False for c in done),
                     'unverified': sum(c['success'] is None for c in done),
                     'success_delayed': sum(c['success_delayed'] is True for c in done),
                     'outcomes': {}, 'returned': sum(c['returned'] for c in done),
                     'samples_mean': round(sum(c['samples_collected'] for c in done) / n, 2),
                     'battery_mean': round(sum(c['battery'] for c in done) / n, 1),
                     'score_mean': round(sum(c['score'] for c in done) / n, 1),
                     'jev_calls': sum(c['jev_calls'] for c in done), 'llm_calls': sum(c['llm_calls'] for c in done),
                     'escalated': sum(c['escalated'] for c in done),
                     'jev_ms': _stat([x for c in done for x in c['jev_ms']]),
                     'llm_ms': _stat([x for c in done for x in c['llm_ms']]),
                     'idle_s_mean': round(sum(c['idle_s'] for c in done) / n, 1),
                     'idle_s_max': max([c['idle_s'] for c in done], default=0.0),
                     'jev_cost_usd': round(sum(c['jev_cost_usd'] for c in done), 6),
                     'jev_tokens': sum(c['jev_tokens'] for c in done)}
            for c in done:
                entry['outcomes'][c['outcome']] = entry['outcomes'].get(c['outcome'], 0) + 1
            if m_id == 'M4':
                hit = [c for c in done if c['details']['had_penalty']]
                entry['with_penalty'] = {'runs': len(hit), 'success': sum(c['success'] is True for c in hit),
                                         'success_delayed': sum(c['success_delayed'] is True for c in hit)}
            table.append(entry)
    # Калибровка: один и тот же запрос (тот же хэш) в разных вариантах и прогонах — одно наблюдение.
    by_mission = {}
    uniq, seen_req = [], set()
    for c in cells:
        per_req = {}
        for key, kind, p, truth, asked in c.get('answers', []):
            per_req.setdefault((c['mission'], key), []).append((kind, p, truth, asked))
        for req, items in per_req.items():
            if req in seen_req:
                continue
            seen_req.add(req)
            uniq += [(kind, p, truth, asked, req[0]) for kind, p, truth, asked in items]
    for kind, p, truth, asked, m_id in uniq:
        by_mission.setdefault(m_id, {}).setdefault(kind, []).append((kind, p, truth, asked))
    calib = {
        'answers': len(uniq), 'requests': len(seen_req),
        'all': {'bins': calibration(uniq), 'confusion': confusion(uniq)},
        'done': {'bins': calibration([r for r in uniq if r[0] == 'done']),
                 'confusion': confusion([r for r in uniq if r[0] == 'done'])},
        'violates': {'bins': calibration([r for r in uniq if r[0] == 'violates']),
                     'confusion': confusion([r for r in uniq if r[0] == 'violates'])},
        'by_mission': {m: {k: confusion(v) for k, v in kinds.items()} for m, kinds in sorted(by_mission.items())},
    }
    paired = []
    for m_id in MISSIONS:
        for a, b in (('jev', 'rule'), ('jev', 'llm'), ('jev_llm', 'llm'), ('jev_llm', 'jev'), ('llm', 'rule'),
                     ('jev_wait', 'jev')):
            ra = {c['scenario']: c for c in cells if (c['mission'], c['variant'], c['status']) == (m_id, a, 'ok')}
            rb = {c['scenario']: c for c in cells if (c['mission'], c['variant'], c['status']) == (m_id, b, 'ok')}
            both = sorted(set(ra) & set(rb))
            if both:
                paired.append({'mission': m_id, 'a': a, 'b': b, **pair_counts(ra, rb, both)})
    guard_cells = [c for c in cells if c['status'] == 'ok' and c['jev_calls']]
    all_jev = [x for c in guard_cells for x in c['jev_ms']]
    # Один и тот же запрос встречается в записях много раз (варианты, прогоны, кэш): по разу на запрос.
    by_key = {k: x for c in guard_cells for k, x in zip(c['jev_keys'], c['jev_ms'])}
    summary = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'experiment': out, 'model_big': MODEL,
        'model_guard': 'typesafe/jev-1.13', 'profile': PROFILE, 'lo': LO, 'hi': HI,
        'cells': {'expected': len(cells), **{k: sum(c['status'] == k for c in cells)
                                              for k in ('ok', 'no_record', 'other_settings', 'guard_disabled')}},
        'missions': {k: v['text'] for k, v in MISSIONS.items()},
        'jev_latency_ms': _stat(all_jev),
        'jev_latency_ms_unique': _stat(list(by_key.values())),
        'jev_answers_fresh': sum(c['jev_fresh'] for c in guard_cells),
        'jev_cached_note': ('jev_latency_ms — по всем использованным ответам, включая взятые из кэша (у них время '
                            'первого, настоящего вызова); jev_latency_ms_unique — по разу на разный запрос; '
                            'jev_answers_fresh — сколько использованных ответов получено по сети в этом прогоне'),
        'budget': budget_now(),
        'table': table, 'paired': paired, 'calibration': calib,
        'runs': [{k: v for k, v in c.items() if k not in ('answers', 'jev_ms', 'llm_ms', 'jev_keys')} for c in cells],
    }
    path = RUNS / out / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary, path


# --- таблицы -----------------------------------------------------------------------------------

def _num(x, fmt='.0f'):
    return '—' if x is None else format(x, fmt)


def print_tables(s):
    c = s['cells']
    print(f"Ячеек по плану {c['expected']}: с записью {c['ok']}, нет записи {c['no_record']}, запись от другой "
          f"настройки {c['other_settings']}. Формулировка {s['profile']}, пороги {s['lo']} и {s['hi']}.\n")
    print('| Миссия | Вариант | Выполнено | с учётом времени ответа | Не выполнено | Не проверено | Не посчитано | '
          'Собрано | Остаток заряда | Вернулся | Обращений к Jev | к большой модели | Простой на прогон, с (средний / '
          'наибольший) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for e in s['table']:
        n = e['expected'] - e['missing']
        if not n:
            print(f"| {e['mission']} | {NAMES[e['variant']]} | — | — | — | — | {e['missing']} | — | — | — | — | — | — |")
            continue
        print(f"| {e['mission']} | {NAMES[e['variant']]} | {e['success']} из {e['expected']} | {e['success_delayed']} | "
              f"{e['failed']} | {e['unverified']} | {e['missing']} | {e['samples_mean']:.2f} | {e['battery_mean']:.1f} | "
              f"{e['returned']} из {n} | {e['jev_calls'] or '—'} | {e['llm_calls'] or '—'} | "
              f"{e['idle_s_mean']:.1f} / {e['idle_s_max']:.1f} |")
    print('\n**M4 по прогонам со штрафом**\n\n| Вариант | Прогонов со штрафом | Вернулся из-за штрафа | с учётом времени '
          'ответа |\n|---|---|---|---|')
    for e in s['table']:
        if 'with_penalty' in e and e['expected'] > e['missing']:
            w = e['with_penalty']
            print(f"| {NAMES[e['variant']]} | {w['runs']} | {w['success']} | {w['success_delayed']} |")
    print('\n**Парное сравнение по сценариям** (выполнено у первого и не выполнено у второго / наоборот; критерий '
          'знаков; «проверено у обоих» — общие сценарии, где исход проверен у обоих вариантов: только они и '
          'сравниваются)\n\n| Миссия | Сравнение | Общих сценариев | Проверено у обоих | Выполнено у обоих | '
          'Только первый | Только второй | p |\n|---|---|---|---|---|---|---|---|')
    for x in s['paired']:
        print(f"| {x['mission']} | {NAMES[x['a']]} против: {NAMES[x['b']]} | {x['pairs']} | {x['checked']} | "
              f"{x['both_success']} | {x['a_only']} | "
              f"{x['b_only']} | {'< 0.001' if x['p_sign'] < 0.001 else format(x['p_sign'], '.3f')} |")
    j, u = s['jev_latency_ms'], s['jev_latency_ms_unique']
    print(f"\nВремя ответа Jev по всем использованным ответам (включая взятые из кэша): {j['n']} ответов, медиана "
          f"{_num(j['median'])} мс, 90% — {_num(j['p90'])} мс, наибольшее {_num(j['max'])} мс. По разу на разный "
          f"запрос: {u['n']} запросов, медиана {_num(u['median'])} мс, 90% — {_num(u['p90'])} мс.")
    print('\n**Время ответа**\n\n| Миссия | Вариант | Jev: ответов | медиана, мс | 90%, мс | наибольшее, мс | '
          'Большая модель: ответов | медиана, с | наибольшее, с | Цена Jev, долл. |\n|---|---|---|---|---|---|---|---|---|---|')
    for e in s['table']:
        if e['jev_calls'] or e['llm_calls']:
            j, b = e['jev_ms'], e['llm_ms']
            print(f"| {e['mission']} | {NAMES[e['variant']]} | {j['n']} | {_num(j['median'])} | {_num(j['p90'])} | "
                  f"{_num(j['max'])} | {b['n']} | {_num(b['median'] and b['median'] / 1000, '.1f')} | "
                  f"{_num(b['max'] and b['max'] / 1000, '.1f')} | {e['jev_cost_usd']:.5f} |")
    cal = s['calibration']
    print(f"\n**Калибровка** (разных запросов {cal['requests']}, ответов «да/нет» {cal['answers']})\n")
    for name, title in (('all', 'все вопросы'), ('violates', '«нарушит ли подцель миссию»'),
                        ('done', '«пора ли на базу»')):
        print(f'{title}:\n\n| Jev сказал «да» с вероятностью | Ответов | Средняя вероятность | На самом деле «да» |\n'
              '|---|---|---|---|')
        for b in cal[name]['bins']:
            share = '—' if b['true_share'] is None else f"{b['true_share']:.0%}"
            print(f"| {b['from']:.0%}–{b['to']:.0%} | {b['n']} | {_num(b['mean_p'], '.1%')} | {share} |")
        x = cal[name]['confusion']
        print(f"\nПо порогам: уверенное «да» — {x['yes_right'] + x['yes_wrong']} (неверно {x['yes_wrong']}), уверенное "
              f"«нет» — {x['no_right'] + x['no_wrong']} (неверно {x['no_wrong']}), сомнение — {x['unsure']} из {x['n']}"
              f" ({x['unsure'] / max(1, x['n']):.0%}).\n")
    print('| Миссия | Вопрос | Ответов | «да» верно | «да» неверно | «нет» верно | «нет» неверно | Сомнение |\n'
          '|---|---|---|---|---|---|---|---|')
    for m, kinds in cal['by_mission'].items():
        for k, x in kinds.items():
            print(f"| {m} | {'пора на базу' if k == 'done' else 'нарушит ли подцель'} | {x['n']} | {x['yes_right']} | "
                  f"{x['yes_wrong']} | {x['no_right']} | {x['no_wrong']} | {x['unsure']} |")
    b = s.get('budget')
    if b:
        print(f"\nЖурнал расходов на Jev (всё время работы): обращений {b['calls']}, входных токенов "
              f"{b['input_tokens']}, {b['spent_usd']:.5f} долл.")


# --- подбор формулировок -----------------------------------------------------------------------

class _SilentJev:
    """Вместо Jev: на всё отвечает «нет». Нужен, чтобы сторож записал состояния, а робот ехал по правилу."""

    def ask(self, state, questions, model=None):
        return JevReply({k: {'type': 'noul', 'noul': 0.0} for k in questions}, 0, {'input_tokens': 0}, 0.0, 'none', True)


def collect(jobs):
    """Состояния для подбора: правило под «немым» сторожем на сценариях 1–20. В сеть не ходит."""
    tasks = [(lvl, seed) for lvl in LEVELS for seed in DEV_SEEDS]

    def one(task):
        level, seed = task
        return run_episode(level, seed, 'adaptive', experiment='J1_states', arm='rule',
                           config={**ASK, 'mission_guard': 'jev'}, llm={'jev': {'client': _SilentJev()}}, save=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, tasks))
    return load_states()


def load_states():
    states = []
    for path in sorted((RUNS / 'J1_states' / 'rule').glob('*.json.gz')):
        for p in load_trace(path).get('plans', []):
            if p.get('guard'):
                states.append({'scenario': path.name[:-len('.json.gz')], 't': p['t'], 'facts': p['guard']['facts'],
                               'options': [{k: v for k, v in o.items() if k != 'p'} for o in p['guard']['options']]})
    return states


def tune(profile, per_mission, missions, seed=0, max_options=6, show=12, cache_only=False):
    """Формулировка profile против правды на собранных состояниях; ответы Jev — через кэш.

    cache_only — только кэш: промах останавливает подбор ошибкой CacheMiss, в сеть ничего не уходит."""
    states = load_states()
    jev = make_jev(cache_only)
    rng = random.Random(seed)
    print(f'состояний {len(states)}; формулировка {profile}; по {per_mission} на миссию')
    total = []
    for m_id in missions:
        # Половина выборки — состояния, где ответ на «пора на базу» или хотя бы на одну подцель «да»: иначе
        # в M1 и M4 почти все примеры одного класса.
        pos = [s for s in states if truth_done(m_id, s['facts'], s['options'])
               or any(truth_violates(m_id, s['facts'], o) for o in s['options'])]
        neg = [s for s in states if s not in pos]
        pick = rng.sample(pos, min(len(pos), per_mission // 2))
        pick += rng.sample(neg, min(len(neg), per_mission - len(pick)))
        rows, wrong, ms = [], [], []
        groups = None
        if PROFILES[profile].get('filter'):
            groups = relevant_groups(jev.ask(*relevance_request(MISSIONS[m_id]['text'])).answers)
            print(f'{m_id}: миссия говорит о: {groups}')
        for s in pick:
            opts = [dict(o) for o in s['options'][:max_options]]
            jev_state, questions = request_from(MISSIONS[m_id]['text'], s['facts'], opts, profile, groups)
            reply = jev.ask(jev_state, questions)
            ms.append(reply.latency_ms)
            row = ('done', float(reply.answers['done']['noul']), truth_done(m_id, s['facts'], s['options']), True)
            rows.append(row)
            if (row[1] >= 0.5) != row[2]:
                wrong.append((row[1], 'done', s['facts'], None))
            for o in opts:
                row = ('violates', float(reply.answers[o['q']]['noul']), truth_violates(m_id, s['facts'], o), True)
                rows.append(row)
                if (row[1] >= 0.5) != row[2]:
                    wrong.append((row[1], 'violates', s['facts'], o))
        total += rows
        for kind in ('done', 'violates'):
            x = confusion([r for r in rows if r[0] == kind])
            print(f"{m_id} {kind:<9} n={x['n']:<4} да верно {x['yes_right']:<4} да НЕВЕРНО {x['yes_wrong']:<4} нет верно "
                  f"{x['no_right']:<4} нет НЕВЕРНО {x['no_wrong']:<4} сомнение {x['unsure']} (из них «да» {x['unsure_true']})")
        print(f'   время ответа: медиана {statistics.median(ms):.0f} мс, наибольшее {max(ms)} мс; запросов {len(ms)}')
        for p, kind, f, o in wrong[:show]:
            print(f'   ошибка: {kind} p={p:.2f} факты={ {k: f[k] for k in ("samples_collected", "penalties", "battery", "battery_if_return_now")} }'
                  + (f" цель=({o['x']}; {o['y']}) заряд после {o['battery_after']} {o['kind']}" if o else ''))
    print('все миссии, подробно (вероятность: ответов с правдой «нет» / с правдой «да»):')
    for i in range(20):
        part = [r for r in total if i * 0.05 <= r[1] < (i + 1) * 0.05 + (i == 19)]
        if part:
            print(f"   {i * 0.05:.2f}–{(i + 1) * 0.05:.2f}: {sum(not r[2] for r in part):>4} / {sum(r[2] for r in part):<4}")
    print('все миссии:')
    for b in calibration(total):
        print(f"   {b['from']:.2f}–{b['to']:.2f}: {b['n']:>5} ответов, на самом деле «да» {_num(b['true_share'], '.0%')}")
    b = budget_now()
    print(f"расходы: обращений {b['calls']}, {b['spent_usd']:.5f} долл." if b else 'журнала расходов нет')


# --- запуск ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--missions', nargs='+', default=['M1', 'M2', 'M3', 'M4'], choices=list(MISSIONS))
    ap.add_argument('--levels', nargs='+', default=list(LEVELS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--variants', nargs='+', default=['rule', 'jev'], choices=list(VARIANTS))
    ap.add_argument('--final', action='store_true', help='итоговая серия: сценарии 1101–1110, каталог runs/J1')
    ap.add_argument('--cache-only', action='store_true',
                    help='ответы Jev и большой модели только из кэша (и в прогонах, и при --tune)')
    ap.add_argument('--out', help='каталог записей внутри runs/ (по умолчанию J1_dev, с --final — J1)')
    ap.add_argument('--report', action='store_true', help='не считать, только сводка и таблицы')
    ap.add_argument('--collect', action='store_true', help='собрать состояния для подбора (сценарии 1–20, без сети)')
    ap.add_argument('--tune', metavar='ФОРМУЛИРОВКА', choices=list(PROFILES))
    ap.add_argument('--per-mission', type=int, default=100)
    ap.add_argument('--llm-timeout', type=float, help='срок ответа большой модели, с (по умолчанию из окружения: 30)')
    ap.add_argument('--sample-seed', type=int, default=0, help='какую выборку состояний брать при --tune')
    args = ap.parse_args(argv)
    if args.cache_only:
        os.environ[CACHE_ONLY_ENV] = '1'       # вторая защита: любой клиент Jev этого процесса — только кэш
    jobs = max(1, min(3, args.jobs))
    global LLM_TIMEOUT_S
    LLM_TIMEOUT_S = args.llm_timeout

    if args.collect:
        print(f'состояний собрано: {len(collect(jobs))}')
        return 0
    if args.tune:
        tune(args.tune, args.per_mission, args.missions, seed=args.sample_seed, cache_only=args.cache_only)
        return 0
    if args.final:
        args.seeds = FINAL_SEEDS
    args.out = args.out or ('J1' if args.final else 'J1_dev')
    tasks = [(m, v, lvl, seed) for v in VARIANTS if v in args.variants for m in args.missions
             for lvl in args.levels for seed in args.seeds]
    if not args.report:
        if not args.cache_only:
            load_env(ENV_FILE)
        jev = make_jev(args.cache_only)
        print(f"Прогонов: {len(tasks)}, одновременно: {jobs}, каталог runs/{args.out}"
              f"{', только кэш' if args.cache_only else ''}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(lambda t: run_single(t, args.out, args.cache_only, jev), tasks))
        print(f'Готово за {time.time() - t0:.0f} с, сбоев: {done.count(None)}', flush=True)
    # Сводка — по плану серии (для runs/J1 — весь итоговый план) и по всем записям каталога: непосчитанная
    # ячейка видна как «нет записи», а не пропадает.
    cells = set(tasks) if not args.report else set()
    if args.out == 'J1':
        cells |= {(m, v, lvl, seed) for m, v in FINAL_PLAN for lvl in LEVELS for seed in FINAL_SEEDS}
    for p in (RUNS / args.out).glob('M*_*/*.json.gz'):
        m, v = p.parent.name.split('_', 1)
        lvl, seed = p.name[:-len('.json.gz')].rsplit('-', 1)
        cells.add((m, v, lvl, int(seed)))
    tasks = sorted(cells, key=lambda t: (t[0], list(VARIANTS).index(t[1]), t[2], t[3]))
    summary, path = compile_summary(args.out, tasks)
    print_tables(summary)
    print(f'\nСводка: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
