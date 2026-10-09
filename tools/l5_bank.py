#!/usr/bin/env python3
"""L5: банк состояний с настоящим выбором, цена вариантов задним числом, ответы правила и шести моделей МАИ.

    ./px python tools/l5_bank.py bank                 # 80 прогонов adaptive_v2 (16001–16040), отбор 90 состояний
    ./px python tools/l5_bank.py values --jobs 2      # цена каждого варианта: 8 продолжений на вариант
    ./px python tools/l5_bank.py ask --jobs 2         # один запрос на состояние и модель, ответы в кэш
    ./px python tools/l5_bank.py ask --cache-only     # строгий повтор без сети
    ./px python tools/l5_bank.py values --jobs 2      # ещё раз: цена выборов вне таблицы (goto) и полных планов
    ./px python tools/l5_bank.py report               # таблицы, runs/L5/summary.json, research/findings/L5-results.json

Всё лежит в runs/<--out> (по умолчанию L5): bank.json, values.json, answers.json, summary.json. Расчёты досчитываются:
готовые цены и ответы не пересчитываются. Цена варианта — оценка с подглядыванием в будущее (did/hindsight.py):
мерило, агенту она недоступна. Модели — только сервер МАИ, не больше двух запросов сразу.
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l3_common import ENV_FILE, RUNS, llm_opts, read_json, share, slug, write_json     # noqa: E402

from did import hindsight as hs                                                      # noqa: E402
from did.llm import CacheMiss, load_env, load_system_prompt, make_client, request_plan                   # noqa: E402

FINDINGS = Path(__file__).resolve().parent.parent / 'research' / 'findings'
MODELS = ('qwen3.8-flash-next', 'qwen3.8-27b', 'qwen3.6-35b-a3b', 'Qwen3.5-122B-A10B', 'deepseek-v4.1-flash',
          'DeepSeek-V4-Flash')
LEVELS = ('hard', 'medium')
SEEDS = (16001, 16040)
REPS = 8                 # продолжений на вариант: 0 — исходный шум, 1–7 — шум после решения пересеян
TOTAL = 90
VARIANTS = {'planner_system_goal': 'goal'}      # подсказка без формулы правила (L3c)
NEAR = 1.0               # цена выбора выше или ниже, чем у правила, больше чем на столько очков — «лучше» / «хуже»
TAG_NAMES = {'time_short': 'мало времени', 'sensor_bad': 'датчик неисправен', 'almost_done': 'не собран один образец',
             'low_battery': 'мало заряда (до 20)', 'weak_near_strong_far': 'слабый близко, сильный далеко',
             'weak_only': 'только слабые кандидаты', 'danger_near': 'рядом опасная зона или дорогой грунт',
             'high_battery': 'много заряда (от 40), кандидатов два и больше', 'explore_only': 'только разведка'}


# --- банк ----------------------------------------------------------------------------------------------

def _decisions(cell):
    level, seed = cell
    rows, result = hs.decisions(level, seed)
    return rows, {'level': level, 'seed': seed, **result}


def cmd_bank(args):
    cells = [(lv, s) for lv in LEVELS for s in range(args.seeds[0], args.seeds[1] + 1)]
    rows, runs = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for got, run in pool.map(_decisions, cells):
            rows += got
            runs.append(run)
    bank = hs.select(rows, total=args.total)
    population = {'decisions': len(rows), 'eligible': sum(hs.eligible(r) for r in rows),
                  'by_tag': dict(Counter(t for r in rows if hs.eligible(r) for t in r['tags'])),
                  'options': dict(Counter(len(r['options']) for r in rows if hs.eligible(r)))}
    write_json(args.dir / 'bank.json', {'agent': hs.AGENT, 'levels': LEVELS, 'seeds': list(args.seeds),
                                        'runs': runs, 'population': population, 'states': bank})
    print(f"прогонов {len(runs)}, решений {len(rows)}, годных {population['eligible']}, в банке {len(bank)}")
    print('годных по видам:', population['by_tag'])
    print('в банке по видам:', dict(Counter(t for r in bank for t in r['tags'])))
    print('взято за вид:', dict(Counter(r['picked_for'] for r in bank)))


# --- цена вариантов ------------------------------------------------------------------------------------

def plan_key(subgoals):
    """Имя полного плана модели (все подцели) в таблице цен."""
    return 'plan:' + '>'.join(hs.option_key(sg) for sg in subgoals)


def _value(job):
    sid, level, seed, k, want, key, subgoals, rep = job
    out, _ = hs.replay(level, seed, at=k, subgoals=subgoals, rep=rep)
    if out['state_hash'] != want:
        raise RuntimeError(f'{sid}: повтор прогона пришёл к другому состоянию')
    return f'{sid}|{key}|{rep}', {key_: out[key_] for key_ in ('score', 'samples_collected', 'returned', 'battery',
                                                               'false_collects', 'hazard_hits', 'collisions', 'next_t')}


def first_valid(state, subgoals):
    """Первая исполнимая подцель плана и весь исполнимый план — так, как их принял бы агент."""
    from did.planner import resolve_subgoals
    done = resolve_subgoals(subgoals, state)
    clean = [{k: v for k, v in sg.items() if k in ('type', 'target') or sg['type'] == 'goto'} for sg in done]
    return clean


def wanted(bank, answers):
    """Что нужно оценить: все варианты из таблицы, а после ответов — выборы вне таблицы и полные планы."""
    jobs = {}
    for row in bank['states']:
        todo = {hs.option_key(o): [o] for o in hs.options(row['state'])}
        for model in MODELS:
            ans = (answers.get(row['id']) or {}).get(model) or {}
            plan = first_valid(row['state'], ans['subgoals']) if ans.get('ok') else []
            if plan:
                todo.setdefault(hs.option_key(plan[0]), [plan[0]])
                if len(plan) > 1:
                    todo.setdefault(plan_key(plan), plan)
        for key, subgoals in todo.items():
            for rep in range(REPS):
                jobs[f"{row['id']}|{key}|{rep}"] = (row['id'], row['level'], row['seed'], row['k'], row['hash'], key,
                                                    subgoals, rep)
    return jobs


def cmd_values(args):
    bank = read_json(args.dir / 'bank.json')
    values = read_json(args.dir / 'values.json', {})
    jobs = {}
    for name in ('answers.json', *[f'answers-{v}.json' for v in VARIANTS.values()]):
        jobs.update(wanted(bank, read_json(args.dir / name, {})))
    jobs = [j for name, j in jobs.items() if name not in values]
    print(f'нужно прогонов: {len(jobs)} (готово {len(values)})', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (name, out) in enumerate(pool.map(_value, jobs, chunksize=4), 1):
            values[name] = out
            if i % 200 == 0 or i == len(jobs):
                write_json(args.dir / 'values.json', values)
                print(f'  {i}/{len(jobs)} за {time.time() - t0:.0f} с', flush=True)
    # Проверка мерила: выбор правила с исходным шумом обязан дать исходный счёт прогона до сотых.
    base = {(r['level'], r['seed']): r['score'] for r in bank['runs']}
    bad = [row['id'] for row in bank['states']
           if abs(values[f"{row['id']}|{row['rule']}|0"]['score'] - base[(row['level'], row['seed'])]) > 0.005]
    print('повтор с выбором правила воспроизводит исходный счёт:', 'да, во всех состояниях' if not bad else f'НЕТ: {bad}')


# --- ответы моделей ------------------------------------------------------------------------------------

def ask_one(client, state, prompt=None):
    """Один запрос плана без исправления — подсказка и схема нынешнего планировщика (prompt — другая подсказка)."""
    res = request_plan(client, state, system_prompt=load_system_prompt(prompt) if prompt else None, max_repairs=0)
    ex = res.exchanges[-1] if res.exchanges else {}
    out = {'ok': res.plan is not None, 'error': res.error, 'answered': ex.get('response') is not None,
           'latency_ms': ex.get('latency_ms'), 'cached': ex.get('cached', False), 'response': ex.get('response'),
           'errors': ex.get('errors') or [], 'http_attempts': ex.get('http_attempts', 0), 'usage': ex.get('usage') or {}}
    if res.plan is not None:
        plan = res.plan.model_dump(exclude_none=True)
        out.update(reasoning=plan['reasoning'], subgoals=plan['subgoals'], hypotheses=plan.get('hypotheses', []))
    return out


def cmd_ask(args):
    bank = read_json(args.dir / 'bank.json')
    answers = read_json(args.dir / args.answers, {})
    if not args.cache_only:
        load_env(ENV_FILE)
    clients = {m: make_client(**llm_opts(m, args.cache_only)) for m in args.models}
    cells = []
    for row in bank['states']:
        for model in args.models:
            old = (answers.get(row['id']) or {}).get(model)
            # Нет ответа сервера — переспросить, но не больше одного раза (tries считает попытки).
            if args.cache_only or old is None or (not old['answered'] and old.get('tries', 1) < 2):
                cells.append((row, model))
    np.random.default_rng(0).shuffle(cells)            # модели вперемешку: очередь сервера делится между ними
    print(f'запросов: {len(cells)}', flush=True)
    sent = 0

    def one(cell):
        row, model = cell
        try:
            return row['id'], model, ask_one(clients[model], row['state'], args.prompt)
        except CacheMiss as e:
            return row['id'], model, {'ok': False, 'answered': False, 'error': f'нет в кэше: {e}', 'cached': True}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(2, args.jobs)) as pool:      # серверу — не больше двух запросов сразу
        for i, (sid, model, out) in enumerate(pool.map(one, cells), 1):
            old = (answers.get(sid) or {}).get(model) or {}
            out['tries'] = old.get('tries', 0) + (0 if out.get('cached') else 1)
            out['net_calls'] = old.get('net_calls', 0) + (0 if out.get('cached') else max(1, out.get('http_attempts') or 1))
            sent += 0 if out.get('cached') else 1
            answers.setdefault(sid, {})[model] = out
            if i % 10 == 0 or i == len(cells):
                write_json(args.dir / args.answers, answers)
                print(f'  {i}/{len(cells)} за {time.time() - t0:.0f} с, по сети {sent}', flush=True)
    write_json(args.dir / args.answers, answers)


# --- сводка --------------------------------------------------------------------------------------------

def boot(x, level=0.95, clusters=None, seed=0, n=4000):
    """Среднее и интервал уровня level: бутстреп по элементам или, если заданы clusters, по группам целиком."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    else:
        groups = [x[np.asarray(clusters) == c] for c in sorted(set(clusters))]
        sums, sizes = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
        pick = rng.integers(0, len(groups), size=(n, len(groups)))
        means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return {'n': int(len(x)), 'mean': round(float(x.mean()), 2), 'ci': [round(float(lo), 2), round(float(hi), 2)]}


def crossfit_loss(table, choice):
    """Потеря выбора относительно лучшего варианта без завышения: лучший выбирается по одной половине
    продолжений (чётные номера), цена считается по другой (нечётные), и наоборот; итог — среднее двух."""
    halves = (np.arange(REPS) % 2 == 0, np.arange(REPS) % 2 == 1)
    out = []
    for pick, check in (halves, halves[::-1]):
        best = max(table, key=lambda k: table[k][pick].mean())
        out.append(table[best][check].mean() - choice[check].mean())
    return float(np.mean(out))


def load(directory, with_models=True, answers_file='answers.json'):
    """Банк с ценами и выборами. with_models=False — только правило (пока модели ещё отвечают)."""
    bank = read_json(directory / 'bank.json')
    values = read_json(directory / 'values.json', {})
    answers = read_json(directory / answers_file, {}) if with_models else {}
    rows = []
    for row in bank['states']:
        state = row['state']

        def vec(key, field='score'):
            return np.array([float(values[f"{row['id']}|{key}|{rep}"][field]) for rep in range(REPS)])

        table = {hs.option_key(o): vec(hs.option_key(o)) for o in hs.options(state)}
        item = {**row, 'table': table, 'choice': {'rule': row['rule']}, 'valid': {}, 'plan': {}, 'answer': {},
                'returned': {k: vec(k, 'returned').mean() for k in table},
                'collected': {k: vec(k, 'samples_collected').mean() for k in table},
                'hold_s': {k: (values[f"{row['id']}|{k}|0"]['next_t'] or np.nan) - row['t'] for k in table}}
        for model in MODELS:
            ans = (answers.get(row['id']) or {}).get(model)
            if ans is None:
                continue
            plan = first_valid(state, ans['subgoals']) if ans.get('ok') else []
            item['answer'][model] = ans
            item['valid'][model] = bool(plan)
            # Негодный ответ, отсутствие ответа или неисполнимый план: агент берёт выбор правила.
            item['choice'][model] = hs.option_key(plan[0]) if plan else row['rule']
            full = plan_key(plan) if len(plan) > 1 else item['choice'][model]
            item['plan'][model] = full
            for key in (item['choice'][model], full):
                if key not in table and f"{row['id']}|{key}|0" in values:
                    item.setdefault('extra', {})[key] = vec(key)
        rows.append(item)
    return bank, rows


def value_of(item, key):
    return item['table'][key] if key in item['table'] else item['extra'][key]


def chooser_stats(rows, who, level=0.95):
    """Сводка по одному выбирающему (правило или модель) на наборе состояний."""
    have = [r for r in rows if who in r['choice']]
    if not have:
        return None
    runs = [f"{r['level']}-{r['seed']}" for r in have]
    val = np.array([value_of(r, r['choice'][who]).mean() for r in have])
    rule = np.array([r['table'][r['rule']].mean() for r in have])
    best = np.array([max(v.mean() for v in r['table'].values()) for r in have])
    cross = np.array([crossfit_loss(r['table'], value_of(r, r['choice'][who])) for r in have])
    d = val - rule
    out = {'n': len(have), 'value': boot(val), 'loss_direct': boot(best - val), 'loss_crossfit': boot(cross),
           'loss_crossfit_by_scenario': boot(cross, clusters=runs),
           'minus_rule': boot(d, level), 'minus_rule_95': boot(d), 'minus_rule_by_scenario': boot(d, clusters=runs),
           'same_as_rule': share(sum(r['choice'][who] == r['rule'] for r in have), len(have)),
           'better': int((d > NEAR).sum()), 'worse': int((d < -NEAR).sum()),
           'best_pick': share(sum(abs(v - b) < 1e-9 for v, b in zip(val, best)), len(have)),
           'kinds': dict(Counter(r['choice'][who].split(':')[0] for r in have))}
    if who != 'rule':
        ans = [r['answer'][who] for r in have]
        ms = sorted(a['latency_ms'] for a in ans if a.get('answered') and a.get('latency_ms'))
        full = np.array([value_of(r, r['plan'][who]).mean() if r['plan'][who] in r['table'] or
                         r['plan'][who] in r.get('extra', {}) else np.nan for r in have])
        out.update(valid=share(sum(r['valid'][who] for r in have), len(have)),
                   answered=sum(bool(a.get('answered')) for a in ans),
                   errors=dict(Counter(_error_kind(a) for a in ans if not a.get('ok'))),
                   latency_s={'median': round(ms[len(ms) // 2] / 1000, 1) if ms else None,
                              'p90': round(ms[int(len(ms) * 0.9)] / 1000, 1) if ms else None,
                              'max': round(ms[-1] / 1000, 1) if ms else None},
                   net_calls=sum(a.get('net_calls', 0) for a in ans),
                   full_plan_minus_rule=boot((full - rule)[~np.isnan(full)]) if (~np.isnan(full)).any() else None,
                   plans_longer=sum(r['plan'][who] != r['choice'][who] for r in have),
                   plans_end_base=sum(r['plan'][who].startswith('plan:') and r['plan'][who].endswith('return_base')
                                      for r in have))
        valid = [r for r in have if r['valid'][who]]
        if valid:
            dv = np.array([value_of(r, r['choice'][who]).mean() - r['table'][r['rule']].mean() for r in valid])
            out['minus_rule_valid_only'] = boot(dv)
    return out


def _error_kind(ans):
    from did.llm import error_kind
    if not ans.get('answered'):
        return 'нет ответа сервера'
    return error_kind('; '.join(str(e) for e in ans.get('errors') or []) or str(ans.get('error')))


def summarize(directory, with_models=True, answers_file='answers.json'):
    bank, rows = load(directory, with_models, answers_file)
    who_all = ['rule', *[m for m in MODELS if any(m in r['choice'] for r in rows)]]
    models = who_all[1:]
    s = {'experiment': 'L5', 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'agent': bank['agent'],
         'states': len(rows), 'reps': REPS, 'population': bank['population'],
         'bank_tags': dict(Counter(t for r in rows for t in r['tags'])),
         'picked_for': dict(Counter(r['picked_for'] for r in rows)),
         'options_per_state': boot([len(r['table']) for r in rows])['mean']}
    # погрешность цены одного варианта и разброс цен внутри состояния
    se = [float(v.std(ddof=1) / np.sqrt(REPS)) for r in rows for k, v in r['table'].items() if k != 'return_base']
    spread = [max(v.mean() for k, v in r['table'].items() if k != 'return_base')
              - min(v.mean() for k, v in r['table'].items() if k != 'return_base') for r in rows
              if len(r['table']) > 2]
    s['noise'] = {'option_se_median': round(float(np.median(se)), 2), 'option_se_mean': round(float(np.mean(se)), 2),
                  'spread_without_base': boot(spread) if spread else None,
                  'base_minus_rule': boot([r['table']['return_base'].mean() - r['table'][r['rule']].mean() for r in rows])}
    hold = [r['hold_s'][r['rule']] for r in rows if np.isfinite(r['hold_s'][r['rule']])]
    s['hold_s'] = {'median': round(float(np.median(hold)), 1), 'share_under_5s': round(float(np.mean(np.array(hold) < 5)), 2)}
    s['choosers'] = {w: chooser_stats(rows, w, 0.95 if w == 'rule' else 1 - 0.05 / 6) for w in who_all}
    # согласие первой подцели попарно
    s['agreement'] = {a: {b: share(sum(r['choice'][a] == r['choice'][b] for r in rows if a in r['choice'] and b in r['choice']),
                                   sum(1 for r in rows if a in r['choice'] and b in r['choice']))['share']
                          for b in who_all} for a in who_all}
    s['all_models_same'] = share(sum(len({r['choice'][m] for m in models}) == 1 for r in rows), len(rows)) if models else None
    s['all_same_as_rule'] = share(sum(all(r['choice'][m] == r['rule'] for m in models) for r in rows), len(rows)) if models else None
    # пары моделей: разность цены выбора с поправкой на 15 сравнений
    pairs = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            d = [value_of(r, r['choice'][a]).mean() - value_of(r, r['choice'][b]).mean() for r in rows]
            pairs[f'{a} − {b}'] = boot(d, 1 - 0.05 / 15)
    s['pairs'] = pairs
    # по видам состояний
    s['by_tag'] = {}
    for tag in (*hs.TAGS, 'none'):
        sub = [r for r in rows if (tag in r['tags'] if tag != 'none' else not r['tags'])]
        if len(sub) >= 3:
            s['by_tag'][tag] = {w: chooser_stats(sub, w) for w in who_all}
    # где правило теряет: какой вид варианта оказывается лучшим (по одной половине), сколько это даёт на другой
    s['rule_loss'] = rule_loss_table(rows)
    s['rows'] = [{'id': r['id'], 'level': r['level'], 'seed': r['seed'], 'k': r['k'], 't': r['t'], 'tags': r['tags'],
                  'picked_for': r['picked_for'], 'choice': r['choice'], 'valid': r['valid'], 'plan': r['plan'],
                  'table': {k: round(float(v.mean()), 2) for k, v in {**r['table'], **r.get('extra', {})}.items()},
                  'se': {k: round(float(v.std(ddof=1) / np.sqrt(REPS)), 2) for k, v in r['table'].items()},
                  'rule_loss_crossfit': round(crossfit_loss(r['table'], r['table'][r['rule']]), 2)} for r in rows]
    return s, rows


def rule_loss_table(rows):
    """Перекрёстно: лучший вариант по одной половине продолжений — какого он вида против выбора правила и
    сколько очков даёт на другой половине."""
    halves = (np.arange(REPS) % 2 == 0, np.arange(REPS) % 2 == 1)
    cells = {}
    for r in rows:
        for pick, check in (halves, halves[::-1]):
            best = max(r['table'], key=lambda k: r['table'][k][pick].mean())
            kind = 'то же' if best == r['rule'] else f"{r['rule'].split(':')[0]} → {best.split(':')[0]}"
            cells.setdefault(kind, []).append(float(r['table'][best][check].mean() - r['table'][r['rule']][check].mean()))
    return {k: {'half_states': len(v), 'gain_mean': round(float(np.mean(v)), 2), 'gain_sum': round(float(np.sum(v)) / 2, 1)}
            for k, v in sorted(cells.items(), key=lambda kv: -np.sum(kv[1]))}


def fmt(b, sign=True):
    f = '{:+.2f}' if sign else '{:.2f}'
    return (f + ' [' + f + '; ' + f + ']').format(b['mean'], *b['ci']).replace('.', ',')


def print_report(s):
    print(f"Банк: {s['states']} состояний, вариантов на состояние в среднем {s['options_per_state']}, "
          f"продолжений на вариант {s['reps']}")
    print(f"Погрешность цены одного варианта (медиана): {s['noise']['option_se_median']} очка; решение правила живёт "
          f"до следующего вызова планировщика (медиана) {s['hold_s']['median']} с")
    print('\n| Кто | Годных сразу | Как правило | Цена выбора − правило | Потеря: перекрёстная | Потеря: прямая | Лучше / хуже правила | Медиана ответа |')
    print('|---|---|---|---|---|---|---|---|')
    for who, c in s['choosers'].items():
        valid = f"{c['valid']['k']} из {c['valid']['n']}" if 'valid' in c else '—'
        lat = f"{c['latency_s']['median']} с" if 'latency_s' in c else '—'
        print(f"| {who} | {valid} | {c['same_as_rule']['k']} из {c['n']} | {fmt(c['minus_rule_95'])} | "
              f"{fmt(c['loss_crossfit'])} | {fmt(c['loss_direct'], False)} | {c['better']} / {c['worse']} | {lat} |")
    print('\nПо видам состояний (цена выбора − правило, 95 %):')
    for tag, block in s['by_tag'].items():
        r = block['rule']
        line = f"  {tag} (n={r['n']}): потеря правила перекр. {fmt(r['loss_crossfit'])}"
        for who, c in block.items():
            if who != 'rule':
                line += f"\n      {who}: {fmt(c['minus_rule_95'])}, как правило {c['same_as_rule']['k']}/{c['n']}"
        print(line)
    print('\nГде правило теряет (перекрёстно):')
    for kind, v in s['rule_loss'].items():
        print(f"  {kind}: полусостояний {v['half_states']}, в среднем {v['gain_mean']:+.2f}, всего {v['gain_sum']:+.1f}")


def cmd_report(args):
    s, _ = summarize(args.dir, not args.rule_only, args.answers)
    s['prompt'] = args.prompt or 'planner_system'
    if args.rule_only:
        return print_report(s)
    tail = f'-{VARIANTS[args.prompt]}' if args.prompt else ''
    write_json(args.dir / f'summary{tail}.json', s)
    if args.out == 'L5':
        write_json(FINDINGS / f'L5-results{tail}.json', s)
        answers = read_json(args.dir / args.answers, {})
        bank = read_json(args.dir / 'bank.json')
        keep = ('ok', 'answered', 'error', 'errors', 'latency_ms', 'response', 'reasoning', 'subgoals', 'tries', 'net_calls')
        write_json(FINDINGS / f'L5-bank{tail}.json', {
            'about': 'Банк состояний L5: сводка состояния, варианты, ответы моделей (как пришли). Цены — в L5-results.json.',
            'states': [{k: r[k] for k in ('id', 'level', 'seed', 'k', 't', 'tags', 'picked_for', 'options', 'rule', 'state')}
                       for r in bank['states']],
            'answers': {sid: {m: {k: a.get(k) for k in keep} for m, a in by.items()} for sid, by in answers.items()}})
    print_report(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['bank', 'values', 'ask', 'report'])
    ap.add_argument('--out', default='L5', help='папка в runs/')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--seeds', type=int, nargs=2, default=SEEDS, help='первый и последний номер сценария')
    ap.add_argument('--total', type=int, default=TOTAL)
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--rule-only', action='store_true', help='report: только правило, без ответов моделей')
    ap.add_argument('--prompt', default=None, choices=list(VARIANTS),
                    help='дополнительный опыт: другая подсказка из did/prompts (ответы и сводка — в отдельных файлах)')
    args = ap.parse_args()
    args.dir = RUNS / args.out
    args.answers = f'answers-{VARIANTS[args.prompt]}.json' if args.prompt else 'answers.json'
    args.jobs = max(1, min(2, args.jobs))            # расчёты — не больше двух потоков
    {'bank': cmd_bank, 'values': cmd_values, 'ask': cmd_ask, 'report': cmd_report}[args.what](args)


if __name__ == '__main__':
    raise SystemExit(main())
