#!/usr/bin/env python3
"""L4a: память планировщика на языковой модели — свои решения и их исходы внутри прогона.

    ./px python tools/l4a_memory.py run main                    # правило, модель без памяти, модель с памятью
    ./px python tools/l4a_memory.py shadow A                    # тень: в состояниях руки mem спросить подсказку без памяти
    ./px python tools/l4a_memory.py shadow B                    # тень: в состояниях руки nomem, где память указывает на цель
    ./px python tools/l4a_memory.py shadow C                    # контроль шума: тот же вопрос без памяти второй раз
    ./px python tools/l4a_memory.py shadow C --retry-failed     # досчёт: сохранённые обрывы связи спросить заново
    ./px python tools/l4a_memory.py shadow Call                 # контроль шума на всех состояниях руки mem (сверх плана)
    ./px python tools/l4a_memory.py shadow CB                   # контроль шума на состояниях тени B (сверх плана)
    ./px python tools/l4a_memory.py run missions --retry-failed --out L4a_redo   # прогоны с обрывами — заново
    ./px python tools/l4a_memory.py select                      # сценарии миссий N1–N3 по прогонам правила
    ./px python tools/l4a_memory.py run missions --missions N2  # миссии словами: правило, без памяти, с памятью
    ./px python tools/l4a_memory.py report                      # сводка runs/<--out>/summary.json и снимок чисел
    ./px python tools/l4a_memory.py run main --dev              # отладка на сценариях hard 1–4, папка L4a_dev
    ./px python tools/l4a_memory.py run main --cache-only --out L4a_replay     # строгий повтор без сети
    ./px python tools/l4a_memory.py compare --out L4a_replay    # сверка повтора с runs/L4a
    ./px python tools/l4a_memory.py run main --model DeepSeek-V4-Flash --out L4a_deepseek --seeds 14001-14006

Записи — runs/<--out>/<рука>/<сценарий>.json.gz; готовые записи не пересчитываются. К серверу — не больше
--jobs запросов сразу (по умолчанию 2): каждый прогон и каждый теневой вопрос ждёт ответа по очереди.
В быстром симуляторе мир стоит, пока модель думает: время ответа только измеряется.
"""
import argparse
import math
import time
from collections import Counter

from l3_common import (ENV_FILE, MAIN, RUNS, CacheFirst, client as plain_client, exchange_stats, fmt_share, llm_opts, mean_ci, paired_binary,
                       paired_diff, read_json, run_cells, share, share_diff, write_json)

from did.llm import load_env, load_system_prompt, request_plan
from did.llm_cache import ReplyCache
from did.plan_memory import MISSIONS, PROMPT, approaches, revisits, verify, without_memory
from did.planner import resolve_subgoals
from did.recorder import load_trace
from did.runner import run_episode

ROOT = RUNS.parent
EXPERIMENT = 'L4a'
LEVEL = 'hard'
FINAL = list(range(14001, 14011))          # основная пара; до заморозки кода не запускались
POOL = list(range(14011, 14041))           # отсюда берутся сценарии миссий — по прогонам правила
DEV = [1, 2, 3, 4]                         # отладка
PER_MISSION = 5
EVERY = {'llm_min_interval_s': 0.0}        # на каждый повод отвечает модель, а не правило
ARMS = ('rule', 'nomem', 'mem')
NOISE_N = 40                               # контроль шума: столько первых состояний руки mem
NOISE = ('C', 'Call', 'CB')                # тени, где тот же запрос без памяти задаётся второй раз


def arm_dir(group, arm):
    return f'{group}_{arm}'


def record(out, group, arm, seed):
    return RUNS / out / arm_dir(group, arm) / f'{LEVEL}-{seed}.json.gz'


def make_llm(model, cache_only, retry=False):
    """Клиент модели: строгий повтор — только кэш; иначе что есть в кэше, повторяется как было, остальное — по сети.
    retry — досчёт после обрывов связи: сохранённые отказы сервера спрашиваются заново (прогон с этого места
    идёт своей дорогой, поэтому пишется в отдельную папку --out)."""
    if cache_only:
        return llm_opts(model, True)
    return {'client': plain_client(model) if retry else CacheFirst(model)}


def run_cell(cell, out, model, cache_only, retry=False):
    group, arm, seed = cell
    path = record(out, group, arm, seed)
    if path.is_file():
        return 'kept'
    mission = {'mission': MISSIONS[group]['text']} if group in MISSIONS else {}
    asks = {'mission_triggers': True} if group in MISSIONS else {}
    if arm == 'rule':
        run_episode(LEVEL, seed, 'adaptive', experiment=out, arm=arm_dir(group, arm), config=mission, truth=True)
    elif arm == 'nomem':
        # Память ведётся и пишется в запись (для тени B), но в запрос не попадает: запросы — как у adaptive_llm.
        run_episode(LEVEL, seed, 'adaptive_llm', experiment=out, arm=arm_dir(group, arm), truth=True,
                    config={**EVERY, **mission, **asks, 'planner_memory': 'track'},
                    llm=make_llm(model, cache_only, retry))
    else:
        run_episode(LEVEL, seed, 'adaptive_llm_mem', experiment=out, arm=arm_dir(group, arm), truth=True,
                    config={**EVERY, **mission, **asks}, llm=make_llm(model, cache_only, retry))
    return 'ok'


def mission_seeds(out):
    return (read_json(RUNS / out / 'mission_scenarios.json') or {}).get('scenarios') or {}


def with_calls(out, what, model, cache_only, fn):
    """Выполнить fn и записать в runs/<out>/launches.json, сколько настоящих обращений к серверу ушло."""
    before = ReplyCache().calls().get(model, 0)
    t0 = time.time()
    result = fn()
    made = ReplyCache().calls().get(model, 0) - before
    log = RUNS / out / 'launches.json'
    old = read_json(log, [])
    old.append({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'what': what, 'model': model, 'cache_only': cache_only,
                'network_calls': made, 'wall_s': round(time.time() - t0)})
    write_json(log, old)
    print(f'{what}: готово за {time.time() - t0:.0f} с, настоящих обращений к серверу: {made}', flush=True)
    return result


# --- теневое сравнение -----------------------------------------------------------------------------

class Again:
    """Тот же запрос второй раз: другой ключ кэша (max_tokens на единицу больше), а на сервер уходит то же самое —
    клиент поднимает лимит ответа до min_tokens. Нужен, чтобы узнать, как часто модель расходится сама с собой."""

    def __init__(self, client):
        self.client, self.model, self.use_schema = client, client.model, client.use_schema

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        return self.client.chat(messages, temperature=temperature, max_tokens=max_tokens + 1, schema=schema)


def first_goal(subgoals, state):
    done = resolve_subgoals(subgoals, state)
    return done[0] if done else None


def same_goal(a, b):
    if a is None or b is None:
        return a is b
    if a['type'] != b['type']:
        return False
    return a['type'] == 'return_base' or math.hypot(a['x'] - b['x'], a['y'] - b['y']) <= 0.05


def with_memory(state, memory):
    """Снимок, каким его получил бы планировщик с памятью: как в Agent._state при planner_memory='on'."""
    out = dict(state)
    out.update(memory)
    return out


def near_ids(memory, field):
    return {i for p in memory.get(field) or [] for i in p.get('near') or []}


def feasible_ids(state):
    return {c['id'] for c in state['candidates'] + state['explore_points'] if c['feasible']}


def shadow_cases(out, kind):
    """Состояния для тени: [(ключ, запись решения)]. A и C — рука mem, B — рука nomem, где память указывает на цель.
    Сверх плана: Call — все состояния руки mem (C — первые NOISE_N из них), CB — состояния тени B."""
    cases = []
    arm = 'nomem' if kind in ('B', 'CB') else 'mem'
    for seed in sorted(int(p.name.split('-')[1].split('.')[0]) for p in (RUNS / out / arm_dir('main', arm)).glob('*.json.gz')):
        tr = load_trace(record(out, 'main', arm, seed))
        for i, p in enumerate(tr['plans']):
            if 'state' not in p or 'rule_match' not in p:          # решение без участия модели
                continue
            if kind in ('B', 'CB'):
                ok = feasible_ids(p['state'])
                if not (near_ids(p['memory'], 'checked_empty') | near_ids(p['memory'], 'penalty_places')) & ok:
                    continue
            cases.append((f'{LEVEL}-{seed}#{i}', p))
    return cases[:NOISE_N] if kind == 'C' else cases


def shadow_one(case, kind, client):
    key, p = case
    state = p['state']
    if kind == 'B':
        ask, prompt = with_memory(state, p['memory']), load_system_prompt(PROMPT)
    else:
        ask, prompt = without_memory(state), None
    res = request_plan(Again(client) if kind in NOISE else client, ask, system_prompt=prompt)
    goal = first_goal(res.plan.model_dump(exclude_none=True)['subgoals'], ask) if res.plan else None
    return {'key': key, 't': p['t'], 'trigger': p['trigger'], 'ok': res.plan is not None, 'error': res.error,
            'goal': goal, 'reasoning': res.plan.reasoning if res.plan else None,
            'exchanges': [{k: v for k, v in ex.items() if k != 'request'} for ex in res.exchanges]}


def cmd_shadow(args):
    kind = args.kind
    cases = shadow_cases(args.out, kind)
    if args.retry_failed and not args.cache_only:
        # Досчёт после обрывов связи: готовые ответы — из кэша, сохранённые отказы и новое — по сети.
        client = plain_client(args.model)
    else:
        client = make_llm(args.model, args.cache_only)
        if 'client' not in client:
            from did.llm import make_client
            client = {'client': make_client(**client)}
        client = client['client']
    by_key = dict(cases)
    print(f'Тень {kind}: состояний {len(cases)}', flush=True)
    done = with_calls(args.out, f'shadow {kind}', args.model, args.cache_only,
                      lambda: run_cells(list(by_key), lambda k: shadow_one((k, by_key[k]), kind, client), args.jobs))
    rows = [v for v in done.values() if v is not None]
    write_json(RUNS / args.out / f'shadow_{kind}.json', sorted(rows, key=lambda r: r['key']))
    print(f'Тень {kind}: ответов {len(rows)} из {len(cases)}')


# --- сводка ----------------------------------------------------------------------------------------

def revisits_by_memory(trace):
    """Вторая, проверочная мера повторных отправок — по спискам самой памяти (они ведутся и в руке nomem):
    решения, первая подцель которых стоит в near у checked_empty на момент решения. Основная мера —
    did.plan_memory.revisits: она памяти не касается и годится для любого агента. None — память не велась."""
    n, seen = 0, False
    for p in trace.get('plans') or []:
        memory = p.get('memory') or (p.get('state') if 'checked_empty' in (p.get('state') or {}) else None)
        if memory is None:
            continue
        seen = True
        sg = (p.get('subgoals') or [None])[0]
        n += bool(sg and sg.get('target') in near_ids(memory, 'checked_empty'))
    return n if seen else None


def fisher_p(k1, n1, k2, n2):
    """Точный двусторонний критерий Фишера для двух долей k1 из n1 и k2 из n2."""
    if not n1 or not n2:
        return None
    k, n = k1 + k2, n1 + n2
    prob = lambda i: math.comb(n1, i) * math.comb(n2, k - i) / math.comb(n, k)          # noqa: E731
    p0 = prob(k1)
    return round(min(1.0, sum(prob(i) for i in range(max(0, k - n2), min(k, n1) + 1) if prob(i) <= p0 + 1e-12)), 4)


def read_run(path):
    tr = load_trace(path)
    res = tr['result']
    plans = tr['plans']
    asked = [p for p in plans if 'rule_match' in p]                      # решения, где спрашивали модель
    by_model = [p for p in asked if p['source'] == 'llm']
    rv = revisits(tr)
    seq = Counter(a[1] for a in approaches(tr))
    first = [ex for ex in tr.get('llm') or [] if ex.get('attempt', 1) == 1]
    tokens = [ex['usage']['prompt_tokens'] for ex in first if (ex.get('usage') or {}).get('prompt_tokens')]
    return {'score': res['score'], 'collected': res['samples_collected'], 'total': res['samples_total'],
            'returned': bool(res['returned']), 'battery': round(res['battery'], 1), 'time_s': res['t'],
            'hazard_hits': res['hazard_hits'], 'false_collects': res['false_collects'], 'collisions': res['collisions'],
            'penalties': res['collisions'] + res['false_collects'] + res['hazard_hits'],
            'decisions': len(plans), 'asked': len(asked), 'by_model': len(by_model),
            'same_as_rule': sum(p['rule_match'] for p in by_model),
            'empty_places': rv['empty_places'], 'to_empty': rv['to_empty'], 'to_empty_collected': rv['to_empty_collected'],
            'to_penalty': rv['to_penalty'], 'revisit_list': rv['list'],
            'to_empty_by_memory': revisits_by_memory(tr),
            'fallback': sum(p['source'] == 'fallback' for p in plans),
            'no_answer': sum(1 for ex in tr.get('llm') or [] if ex.get('response') is None),
            'lost': seq['lost'], 'refuted': seq['candidate_refuted'],
            'prompt_tokens': round(sum(tokens) / len(tokens)) if tokens else None,
            'exchanges': tr.get('llm') or [], 'trace': tr}


METRICS = ('score', 'collected', 'penalties', 'decisions', 'to_empty', 'to_penalty', 'empty_places', 'lost')


def load_group(out, group, seeds):
    data = {}
    for arm in ARMS:
        runs = {}
        for seed in seeds:
            path = record(out, group, arm, seed)
            if path.is_file():
                runs[f'{LEVEL}-{seed}'] = read_run(path)
        data[arm] = runs
    return data


def arm_summary(runs, arm):
    n = len(runs)
    vals = list(runs.values())
    e = {'runs': n, 'score': mean_ci([r['score'] for r in vals]),
         'collected_share': round(sum(r['collected'] for r in vals) / sum(r['total'] for r in vals), 3),
         'returned': sum(r['returned'] for r in vals),
         'hazard_hits': sum(r['hazard_hits'] for r in vals), 'false_collects': sum(r['false_collects'] for r in vals),
         'collisions': sum(r['collisions'] for r in vals)}
    for m in ('decisions', 'to_empty', 'to_penalty', 'empty_places', 'lost', 'to_empty_collected'):
        e[m] = {'total': sum(r[m] for r in vals), 'per_run': round(sum(r[m] for r in vals) / n, 2)}
    e['runs_with_empty_place'] = sum(r['empty_places'] > 0 for r in vals)
    e['runs_with_to_empty'] = sum(r['to_empty'] > 0 for r in vals)
    e['collected'] = sum(r['collected'] for r in vals)
    e['penalties'] = sum(r['penalties'] for r in vals)
    if arm != 'rule':
        k = sum(r['by_model'] for r in vals)
        e['asked'] = sum(r['asked'] for r in vals)
        e['same_as_rule'] = share(sum(r['same_as_rule'] for r in vals), k)
        e['llm'] = exchange_stats([ex for r in vals for ex in r['exchanges']])
        tok = [r['prompt_tokens'] for r in vals if r['prompt_tokens']]
        e['prompt_tokens_per_run_mean'] = round(sum(tok) / len(tok)) if tok else None
        e['to_empty_by_memory'] = sum(r['to_empty_by_memory'] or 0 for r in vals)
        e['fallback'] = sum(r['fallback'] for r in vals)
    return e


def shadow_summary(out, data):
    """Тени A, B и C: доли расхождений в одних и тех же состояниях."""
    plans = {}
    for arm in ('mem', 'nomem'):
        for sc, r in data[arm].items():
            for i, p in enumerate(r['trace']['plans']):
                plans[(arm, f'{sc}#{i}')] = p
    res = {}
    a = read_json(RUNS / out / 'shadow_A.json') or []
    if a:
        rows = []
        for s in a:
            p = plans.get(('mem', s['key']))
            if p is None:
                continue
            mem = p['subgoals'][0] if p['subgoals'] and p['source'] == 'llm' else None
            ok = feasible_ids(p['state'])
            near = near_ids(p['state'], 'checked_empty') & ok
            near_pen = near_ids(p['state'], 'penalty_places') & ok
            rows.append({'key': s['key'], 't': s['t'], 'trigger': s['trigger'], 'mem_ok': p['source'] == 'llm',
                         'nomem_ok': s['ok'], 'mem': mem, 'nomem': s['goal'], 'rule': p.get('rule_first'),
                         'near_empty': sorted(near), 'near_penalty': sorted(near_pen),
                         'mem_reasoning': p['reasoning'], 'nomem_reasoning': s['reasoning']})
        both = [r for r in rows if r['mem_ok'] and r['nomem_ok']]
        pick = lambda r, who, ids: bool(r[who] and r[who].get('target') in r[ids])          # noqa: E731
        info = [r for r in both if r['near_empty']]
        info_p = [r for r in both if r['near_penalty']]
        res['A'] = {
            'states': len(rows), 'both_answered': len(both),
            'mem_differs_from_nomem': share(sum(not same_goal(r['mem'], r['nomem']) for r in both), len(both)),
            'mem_differs_from_rule': share(sum(not same_goal(r['mem'], r['rule']) for r in both), len(both)),
            'nomem_differs_from_rule': share(sum(not same_goal(r['nomem'], r['rule']) for r in both), len(both)),
            'diff_kinds': dict(Counter(f"{(r['mem'] or {}).get('type')} вместо {(r['nomem'] or {}).get('type')}"
                                       for r in both if not same_goal(r['mem'], r['nomem']))),
            'states_memory_points_to_goal': len(info),
            'choose_checked_empty': paired_binary({r['key']: pick(r, 'mem', 'near_empty') for r in info},
                                                  {r['key']: pick(r, 'nomem', 'near_empty') for r in info}),
            'rule_chooses_checked_empty': share(sum(pick(r, 'rule', 'near_empty') for r in info), len(info)),
            'states_near_penalty': len(info_p),
            'choose_near_penalty': paired_binary({r['key']: pick(r, 'mem', 'near_penalty') for r in info_p},
                                                 {r['key']: pick(r, 'nomem', 'near_penalty') for r in info_p}),
            'differing': [r for r in both if not same_goal(r['mem'], r['nomem'])],
            'rows_all': both,
        }
    b = read_json(RUNS / out / 'shadow_B.json') or []
    if b:
        rows = []
        for s in b:
            p = plans.get(('nomem', s['key']))
            if p is None or not s['ok'] or p['source'] != 'llm':
                continue
            ok = feasible_ids(p['state'])
            near = near_ids(p['memory'], 'checked_empty') & ok
            near_pen = near_ids(p['memory'], 'penalty_places') & ok
            nomem = p['subgoals'][0] if p['subgoals'] else None
            rows.append({'key': s['key'], 't': s['t'], 'nomem': nomem, 'mem': s['goal'], 'near_empty': sorted(near),
                         'near_penalty': sorted(near_pen), 'mem_reasoning': s['reasoning'],
                         'nomem_reasoning': p['reasoning']})
        pick = lambda r, who, ids: bool(r[who] and r[who].get('target') in r[ids])          # noqa: E731
        e_rows = [r for r in rows if r['near_empty']]
        p_rows = [r for r in rows if r['near_penalty']]
        res['B'] = {'states': len(rows),
                    'mem_differs_from_nomem': share(sum(not same_goal(r['mem'], r['nomem']) for r in rows), len(rows)),
                    'states_near_empty': len(e_rows),
                    'choose_checked_empty': paired_binary({r['key']: pick(r, 'mem', 'near_empty') for r in e_rows},
                                                          {r['key']: pick(r, 'nomem', 'near_empty') for r in e_rows}),
                    'states_near_penalty': len(p_rows),
                    'choose_near_penalty': paired_binary({r['key']: pick(r, 'mem', 'near_penalty') for r in p_rows},
                                                         {r['key']: pick(r, 'nomem', 'near_penalty') for r in p_rows}),
                    'rows': rows}
    first = {s['key']: s for s in a}
    for kind in ('C', 'Call'):
        c = read_json(RUNS / out / f'shadow_{kind}.json') or []
        if not (c and a):
            continue
        pairs = [(first[s['key']], s) for s in c if s['key'] in first and s['ok'] and first[s['key']]['ok']]
        keys = {s['key'] for _, s in pairs}
        # На тех же состояниях: «с памятью против без памяти» (тень A) рядом с «без памяти против себя».
        same = [r for r in res['A']['rows_all'] if r['key'] in keys]
        again = {s['key']: s['goal'] for _, s in pairs}
        res[kind] = {'asked': len(c), 'no_answer': sum(not s['ok'] for s in c), 'states': len(pairs),
                     'nomem_differs_from_itself': share(sum(not same_goal(x['goal'], y['goal']) for x, y in pairs),
                                                        len(pairs)),
                     'mem_differs_from_nomem_same_states': share(sum(not same_goal(r['mem'], r['nomem']) for r in same),
                                                                 len(same)),
                     'mem_differs_from_second_nomem': share(sum(not same_goal(r['mem'], again[r['key']]) for r in same),
                                                            len(same)),
                     'paired': paired_binary({r['key']: not same_goal(r['mem'], r['nomem']) for r in same},
                                             {x['key']: not same_goal(x['goal'], y['goal']) for x, y in pairs})}
    if 'A' in res:
        del res['A']['rows_all']
    cb = read_json(RUNS / out / 'shadow_CB.json') or []
    if cb and 'B' in res:
        again = {s['key']: s['goal'] for s in cb if s['ok']}
        rows = [r for r in res['B']['rows'] if r['key'] in again]
        pick = lambda goal, ids: bool(goal and goal.get('target') in ids)                    # noqa: E731
        e_rows = [r for r in rows if r['near_empty']]
        p_rows = [r for r in rows if r['near_penalty']]
        res['CB'] = {'asked': len(cb), 'no_answer': sum(not s['ok'] for s in cb), 'states': len(rows),
                     'nomem_differs_from_itself': share(sum(not same_goal(r['nomem'], again[r['key']]) for r in rows),
                                                        len(rows)),
                     'mem_differs_from_nomem_same_states': share(sum(not same_goal(r['mem'], r['nomem']) for r in rows),
                                                                 len(rows)),
                     'states_near_empty': len(e_rows),
                     'second_nomem_chooses_checked_empty': share(
                         sum(pick(again[r['key']], r['near_empty']) for r in e_rows), len(e_rows)),
                     'states_near_penalty': len(p_rows),
                     'second_nomem_chooses_near_penalty': share(
                         sum(pick(again[r['key']], r['near_penalty']) for r in p_rows), len(p_rows))}
    return res


def missions_summary(out):
    table = {}
    seeds = mission_seeds(out)
    for m_id, own in seeds.items():
        data = load_group(out, m_id, own)
        if not data['rule']:
            # Правило миссию не читает: его прогоны для миссии — те же, по которым отбирались сценарии.
            data['rule'] = {f'{LEVEL}-{s}': read_run(record(out, 'pool', 'rule', s)) for s in own
                            if record(out, 'pool', 'rule', s).is_file()}
        cell = {'text': MISSIONS[m_id]['text'], 'scenarios': own, 'arms': {}}
        verdicts = {}
        for arm, runs in data.items():
            if not runs:
                continue
            checks = {sc: verify(m_id, r['trace']) for sc, r in runs.items()}
            verdicts[arm] = {sc: c['success'] for sc, c in checks.items()}
            checked = [c for c in checks.values() if c['success'] is not None]
            vals = list(runs.values())
            e = {'runs': len(runs), 'success': share(sum(c['success'] for c in checked), len(checked)),
                 'no_occasion': len(runs) - len(checked), 'outcomes': dict(Counter(c['outcome'] for c in checks.values())),
                 'score': round(sum(r['score'] for r in vals) / len(vals), 1),
                 'collected': round(sum(r['collected'] for r in vals) / len(vals), 2),
                 'returned': sum(r['returned'] for r in vals),
                 'by_scenario': {sc: {'success': c['success'], 'outcome': c['outcome'],
                                      'details': {k: v for k, v in c['details'].items() if k != 'list'}}
                                 for sc, c in checks.items()}}
            if arm != 'rule':
                e['llm'] = exchange_stats([ex for r in vals for ex in r['exchanges']])
                e['rule_decisions'] = sum(r['decisions'] - r['by_model'] for r in vals)
                # Прогоны, где сервер не ответил и решало запасное правило: {сценарий: сколько таких решений}.
                e['fallback_by_scenario'] = {sc: r['fallback'] for sc, r in runs.items() if r['fallback']}
                for sc, r in runs.items():
                    e['by_scenario'][sc]['fallback'] = r['fallback']
            cell['arms'][arm] = e
        a = cell['arms']
        if 'mem' in a and 'nomem' in a:
            cell['mem_minus_nomem'] = share_diff(a['mem']['success']['k'], a['mem']['success']['n'],
                                                 a['nomem']['success']['k'], a['nomem']['success']['n'])
            cell['paired'] = paired_binary(verdicts['mem'], verdicts['nomem'])
            cell['fisher_p'] = fisher_p(a['mem']['success']['k'], a['mem']['success']['n'],
                                        a['nomem']['success']['k'], a['nomem']['success']['n'])
            for m in ('score', 'collected'):
                cell[f'{m}_mem_minus_nomem'] = paired_diff({k: r[m] for k, r in data['mem'].items()},
                                                           {k: r[m] for k, r in data['nomem'].items()})
        if 'mem' in a and 'rule' in a:
            cell['mem_minus_rule'] = share_diff(a['mem']['success']['k'], a['mem']['success']['n'],
                                                a['rule']['success']['k'], a['rule']['success']['n'])
            cell['fisher_p_mem_rule'] = fisher_p(a['mem']['success']['k'], a['mem']['success']['n'],
                                                 a['rule']['success']['k'], a['rule']['success']['n'])
        table[m_id] = cell
    return table


def summarize(out, seeds):
    data = load_group(out, 'main', seeds)
    s = {'experiment': out, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'level': LEVEL, 'seeds': seeds,
         'arms': {arm: arm_summary(runs, arm) for arm, runs in data.items() if runs}, 'diffs': {}}
    for a, b in (('mem', 'nomem'), ('mem', 'rule'), ('nomem', 'rule')):
        if data[a] and data[b]:
            s['diffs'][f'{a}-{b}'] = {m: paired_diff({k: r[m] for k, r in data[a].items()},
                                                     {k: r[m] for k, r in data[b].items()}) for m in METRICS}
    if data['mem'] and data['nomem']:
        x, y = s['arms']['mem'], s['arms']['nomem']
        s['same_as_rule_mem_minus_nomem'] = share_diff(x['same_as_rule']['k'], x['same_as_rule']['n'],
                                                       y['same_as_rule']['k'], y['same_as_rule']['n'])
        fx, fy = x['llm'], y['llm']
        s['first_ok_mem_minus_nomem'] = share_diff(fx['first_ok'], fx['requests'], fy['first_ok'], fy['requests'])
        if x['llm']['tokens']['prompt'] and y['llm']['tokens']['prompt']:
            s['prompt_tokens_ratio'] = round(x['llm']['tokens']['prompt'] / y['llm']['tokens']['prompt'], 3)
    s['shadow'] = shadow_summary(out, data)
    s['missions'] = missions_summary(out)
    # Цена памяти по всем прогонам с моделью (основная пара и миссии): больше запросов — уже интервал.
    every = {arm: [] for arm in ('nomem', 'mem')}
    for group, own in [('main', seeds)] + list(mission_seeds(out).items()):
        for arm in every:
            for seed in own:
                if record(out, group, arm, seed).is_file():
                    every[arm] += load_trace(record(out, group, arm, seed)).get('llm') or []
    if every['mem'] and every['nomem']:
        x, y = exchange_stats(every['mem']), exchange_stats(every['nomem'])
        s['llm_all_runs'] = {'mem': x, 'nomem': y,
                             'first_ok_mem_minus_nomem': share_diff(x['first_ok'], x['requests'], y['first_ok'],
                                                                    y['requests']),
                             'prompt_tokens_ratio': round(x['tokens']['prompt'] / y['tokens']['prompt'], 3)}
    s['launches'] = read_json(RUNS / out / 'launches.json', [])
    s['network_calls'] = sum(x['network_calls'] for x in s['launches'] if not x['cache_only'])
    s['runs'] = [{'arm': arm, 'scenario': k, **{f: r[f] for f in r if f not in ('exchanges', 'trace')}}
                 for arm, runs in data.items() for k, r in runs.items()]
    return s


def _ci(d, digits=2):
    if not d:
        return '—'
    return f"{d['mean']:+.{digits}f} [{d['ci'][0]:+.{digits}f}; {d['ci'][1]:+.{digits}f}]"


def report(s):
    lines = []
    for arm, e in s['arms'].items():
        line = (f"{arm}: прогонов {e['runs']}, счёт {e['score']['mean']:.2f} [{e['score']['ci'][0]:.2f}; "
                f"{e['score']['ci'][1]:.2f}], собрано {e['collected_share']:.0%}, вернулся {e['returned']}, штрафы "
                f"зона/ложный/столкн. {e['hazard_hits']}/{e['false_collects']}/{e['collisions']}; решений на прогон "
                f"{e['decisions']['per_run']}, пустых мест {e['empty_places']['total']}, повторных отправок в пустое "
                f"{e['to_empty']['total']} (кончились сбором {e['to_empty_collected']['total']}), к месту штрафа "
                f"{e['to_penalty']['total']}, пропавших по дороге кандидатов {e['lost']['total']}")
        if 'llm' in e:
            x = e['llm']
            line += (f"\n   решений модели {e['same_as_rule']['n']}, как правило {fmt_share(e['same_as_rule'], 1)}; запросов "
                     f"{x['requests']}: годных сразу {x['first_ok_share']:.1%}, после исправления {x['repaired']}, отказов "
                     f"{x['failed']}; ответ: медиана {x['latency_s']['median']} с, среднее {x['latency_s']['mean']} с; "
                     f"токенов в запросе {x['tokens']['prompt']}, в ответе {x['tokens']['completion']}")
        lines.append(line)
    for name, d in s['diffs'].items():
        lines.append(f"{name}: счёт {_ci(d['score'])}; собрано {_ci(d['collected'])}; штрафы {_ci(d['penalties'])}; "
                     f"повторных в пустое {_ci(d['to_empty'])}; к штрафу {_ci(d['to_penalty'])}; решений "
                     f"{_ci(d['decisions'], 1)}; пропавших {_ci(d['lost'], 1)}")
    for k in ('same_as_rule_mem_minus_nomem', 'first_ok_mem_minus_nomem', 'prompt_tokens_ratio', 'llm_all_runs'):
        if k in s:
            lines.append(f'{k}: {s[k]}')
    for kind, e in s['shadow'].items():
        lines.append(f'тень {kind}: ' + '; '.join(f'{k}: {v}' for k, v in e.items() if k not in ('differing', 'rows')))
    for m_id, cell in s['missions'].items():
        for arm, e in cell['arms'].items():
            lines.append(f"{m_id}/{arm}: выполнено {fmt_share(e['success'])}, без повода {e['no_occasion']} из {e['runs']}; "
                         f"исходы {e['outcomes']}; счёт {e['score']}, собрано {e['collected']}, вернулся {e['returned']}")
        lines.append(f"{m_id}: mem − nomem {cell.get('mem_minus_nomem')}, Фишер p={cell.get('fisher_p')}; парно "
                     f"{cell.get('paired')}; счёт {_ci(cell.get('score_mem_minus_nomem'))}, образцы "
                     f"{_ci(cell.get('collected_mem_minus_nomem'))}")
    lines.append(f"настоящих обращений к серверу: {s['network_calls']}")
    return '\n'.join(lines)


def cmd_select(args):
    """Сценарии миссий: первые PER_MISSION из POOL, где повод случился в прогоне правила."""
    cells = [('pool', 'rule', seed) for seed in POOL]
    run_cells(cells, lambda c: run_cell(c, args.out, args.model, True), args.jobs, label=lambda c: f'правило {c[2]}')
    chosen, occasions = {}, {}
    for m_id in MISSIONS:
        have = []
        for seed in POOL:
            v = verify(m_id, load_trace(record(args.out, 'pool', 'rule', seed)))
            if v['success'] is not None:
                have.append(seed)
        occasions[m_id] = have
        chosen[m_id] = have[:PER_MISSION]
    write_json(RUNS / args.out / 'mission_scenarios.json',
               {'pool': POOL, 'per_mission': PER_MISSION, 'occasion_in_rule_run': occasions, 'scenarios': chosen})
    for m_id in MISSIONS:
        print(f'{m_id}: повод у правила в {len(occasions[m_id])} из {len(POOL)}; взяты {chosen[m_id]}')


def cmd_run(args, seeds):
    if args.what == 'main':
        cells = [('main', arm, seed) for seed in seeds for arm in args.arms]
    else:
        own = mission_seeds(args.out)
        cells = [(m, arm, seed) for m in args.missions for seed in own[m] for arm in args.arms if arm != 'rule']
    print(f'Прогонов: {len(cells)}, одновременно {args.jobs}, runs/{args.out}', flush=True)
    done = with_calls(args.out, f'run {args.what}', args.model, args.cache_only,
                      lambda: run_cells(cells, lambda c: run_cell(c, args.out, args.model, args.cache_only,
                                                                    args.retry_failed), args.jobs,
                                        label=lambda c: f'{c[0]} {c[1]} {c[2]}'))
    print(f'Без результата: {sum(v is None for v in done.values())} из {len(cells)}')


RESULT_KEYS = ('t', 'battery', 'samples_collected', 'returned', 'score', 'collisions', 'false_collects', 'hazard_hits')


def cmd_compare(args):
    """Сверка строгого повтора с исходными записями: итог, события, планы, ответы модели, трек."""
    same = total = 0
    for path in sorted((RUNS / args.other).glob('*/*.json.gz')):
        twin = RUNS / args.out / path.parent.name / path.name
        if not twin.is_file():
            continue
        total += 1
        x, y = load_trace(twin), load_trace(path)
        diff = [k for k in ('events', 'plans', 'track') if x.get(k) != y.get(k)]
        diff += [k for k in RESULT_KEYS if x['result'].get(k) != y['result'].get(k)]
        if [(e['t'], e.get('response')) for e in x.get('llm') or []] != [(e['t'], e.get('response')) for e in y.get('llm') or []]:
            diff.append('llm')
        same += not diff
        if diff:
            print(f'{path.parent.name}/{path.name}: отличается {diff}')
    print(f'Совпало с runs/{args.other} до цифры: {same} из {total} прогонов')
    write_json(RUNS / args.out / 'compare.json', {'same': same, 'total': total, 'with': args.other})


def parse_seeds(text):
    a, _, b = text.partition('-')
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'shadow', 'select', 'report', 'compare'])
    ap.add_argument('what', nargs='?', default='main', help='run: main | missions; shadow: A | B | C | Call | CB')
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    ap.add_argument('--missions', nargs='+', default=list(MISSIONS), choices=list(MISSIONS))
    ap.add_argument('--model', default=MAIN)
    ap.add_argument('--seeds', default=None, help='например 14001-14006')
    ap.add_argument('--dev', action='store_true', help='отладочные сценарии, папка L4a_dev')
    ap.add_argument('--out', default=None)
    ap.add_argument('--with', dest='other', default=EXPERIMENT, help='compare: с какой папкой сверять')
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=2, help='к серверу — не больше двух запросов сразу')
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--retry-failed', action='store_true',
                    help='сохранённые в кэше обрывы связи спросить заново (ответы из кэша не трогаются); для run — в '
                         'отдельную папку --out с её mission_scenarios.json')
    args = ap.parse_args()
    args.out = args.out or ('L4a_dev' if args.dev else EXPERIMENT)
    seeds = parse_seeds(args.seeds) if args.seeds else (DEV if args.dev else FINAL)
    if args.cmd in ('run', 'shadow') and not args.cache_only:
        load_env(ENV_FILE)
    if args.cmd == 'run':
        cmd_run(args, seeds)
    elif args.cmd == 'shadow':
        args.kind = args.what
        cmd_shadow(args)
    elif args.cmd == 'select':
        cmd_select(args)
    elif args.cmd == 'compare':
        cmd_compare(args)
    if args.cmd in ('run', 'shadow', 'report'):
        have = sorted({int(p.name.split('-')[1].split('.')[0]) for p in (RUNS / args.out).glob('main_*/*.json.gz')})
        s = summarize(args.out, parse_seeds(args.seeds) if args.seeds else have)
        print(report(s))
        write_json(RUNS / args.out / 'summary.json', s)
        if args.cmd == 'report' and args.out == EXPERIMENT:
            write_json(ROOT / 'research' / 'findings' / 'L4a-results.json', s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
