"""R3: способы обращения к одной и той же настоящей модели на шести сценариях.

    set -a; . /Users/a/MAI/DID/.env; set +a                    # адрес и ключ сервера МАИ
    ./px python -m did.orchestration_eval                       # single, critic, scored, scored_calc
    ./px python -m did.orchestration_eval --strategies vote     # досчитать ещё способ
    ./px python -m did.orchestration_eval --no-run              # пересобрать сводку по записям, без сети
    ./px python -m did.orchestration_eval --cache-only --out R3_replay --strategies single critic scored scored_calc vote
                                                                # строгий повтор: ответы только из кэша, сеть не трогается

Два режима времени на каждый способ:
  frozen  — мир стоит, пока модель думает (так устроен быстрый симулятор): качество решений в чистом виде;
  charged — после каждого решения робот стоит llm_wait_s секунд, а часы прогона, расход простоя и события
            среды идут. llm_wait_s — среднее измеренное время одного решения этим способом в режиме frozen.

Ответы модели кэшируются (runs/_llm_cache): повтор не тратит запросов и даёт те же числа. Одновременно идёт
не больше --jobs прогонов (не больше трёх), в каждом запросы идут по одному.
Записи — runs/R3_real/<способ>-<режим>/<сценарий>.json.gz, сводка — runs/R3_real/summary.json.
--cache-only — режим «только кэш» (did/llm.py, cache='only'): адрес и ключ сервера не нужны, запрос, которого в
кэше нет, останавливает свой прогон (CacheMiss), и такие прогоны перечисляются в конце. --out — другая папка
в runs/, чтобы повтор не затирал исходные записи.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from .llm import CacheMiss
from .llm_cache import ReplyCache
from .recorder import load_trace
from .runner import RUNS, run_episode

EXPERIMENT = 'R3_real'
MODEL = 'qwen3.8-flash-next'
# Запас на рассуждение 3000 токенов; срок 90 с: длинное рассуждение идёт дольше обычных 30 с.
LLM = {'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True, 'timeout_s': 90}
SCENARIOS = [(level, seed) for level in ('medium', 'hard') for seed in (1001, 1002, 1003)]
STRATEGIES = ('single', 'critic', 'scored', 'scored_calc')
MODES = ('frozen', 'charged')
# Сервер отдаёт уже виденный запрос из своего кэша за доли секунды. Такое время — не время решения:
# ответ в сотни токенов быстрее этого порога модель не пишет.
SERVER_CACHE_MS, SERVER_CACHE_TOKENS = 1500, 200


def _label(subgoal):
    if not subgoal:
        return None
    if subgoal['type'] == 'goto':
        return f"goto ({subgoal['x']:g}; {subgoal['y']:g})"
    return ' '.join(str(subgoal[k]) for k in ('type', 'target') if subgoal.get(k) is not None)


def _row(strategy, mode, level, seed, wait, file):
    """Строка сводки по одной записи прогона."""
    trace = load_trace(RUNS / file)
    res = trace['result']
    decisions = [p for p in trace['plans'] if 'rule_match' in p]        # решения, где спрашивали модель
    row = {'strategy': strategy, 'mode': mode, 'level': level, 'seed': seed, 'llm_wait_s': wait, 'file': file,
           'score': res['score'], 'collected': res['samples_collected'], 'total': res['samples_total'],
           'returned': bool(res['returned']), 'time_s': res['time'], 'battery_left': res.get('battery_left'),
           'reason': res.get('reason'), 'hazard_hits': res.get('hazard_hits', 0),
           'decisions': len(decisions),
           'fallbacks': sum(p['source'] == 'fallback' for p in decisions),
           'rule_matches': sum(p['rule_match'] for p in decisions),
           'mismatches': [{'t': p['t'], 'chosen': _label((p['subgoals'] or [None])[0]), 'rule': _label(p['rule_first'])}
                          for p in decisions if not p['rule_match']],
           'exchanges': [{'role': ex.get('role'), 'ok': bool(ex.get('ok')), 'cached': bool(ex.get('cached')),
                          'latency_ms': int(ex.get('latency_ms') or 0),
                          'tokens': (ex.get('usage') or {}).get('completion_tokens') or 0,
                          'errors': ex.get('errors') or []} for ex in trace.get('llm') or []]}
    steps = [p['orchestration'] for p in decisions if p.get('orchestration')]
    if strategy == 'critic':
        row['critic'] = {'reviews': sum(s.get('verdict') is not None for s in steps),
                         'revise': sum(s.get('verdict') == 'revise' for s in steps),
                         'revised': sum(bool(s.get('revised')) for s in steps),
                         'changed': sum(bool(s.get('changed')) for s in steps)}
    if strategy == 'vote':
        row['vote'] = {'split': sum(not s.get('unanimous', True) for s in steps)}
    if strategy in ('scored', 'scored_calc'):
        tables = [(p, p['orchestration']) for p in decisions if (p.get('orchestration') or {}).get('proposals')]
        row['scored'] = {'tables': len(tables), 'proposals': sum(len(s['proposals']) for _, s in tables),
                         'rule_offered': sum(_label(p['rule_first']) in s['proposals'] for p, s in tables),
                         'asked': sum(s['chosen_by'] == 'model' for _, s in tables),
                         'model_differs_from_calc': sum(s['chosen_by'] == 'model'
                                                        and s['model_choice'] != s['calc_choice'] for _, s in tables),
                         'calc_differs_from_first': sum(s['calc_choice'] != 1 for _, s in tables)}
    return row


def _run(task):
    strategy, mode, level, seed, wait, run, out, cache_only = task
    arm = f'{strategy}-{mode}'
    file = f'{out}/{arm}/{level}-{seed}.json.gz'
    if run:
        if strategy == 'rule':
            run_episode(level, seed, 'adaptive', experiment=out, arm=arm)
        else:
            try:
                run_episode(level, seed, 'adaptive_llm', experiment=out, arm=arm,
                            llm={**LLM, 'cache': 'only'} if cache_only else dict(LLM),
                            config={'llm_strategy': strategy, 'llm_wait_s': wait})
            except CacheMiss as e:           # строгий повтор: ответа нет в кэше — прогон не досчитан, сеть не тронута
                return {'miss': str(e), 'strategy': strategy, 'mode': mode, 'level': level, 'seed': seed}
    elif not (RUNS / file).exists():
        return None
    return _row(strategy, mode, level, seed, wait, file)


def decision_time(rows):
    """Время ответов модели по строкам одного способа: (секунд на решение, секунд по прогонам, замен).

    Ответ, отданный сервером из его кэша, считается по медиане настоящих ответов того же шага.
    """
    def fast(ex):
        return ex['ok'] and ex['latency_ms'] < SERVER_CACHE_MS and ex['tokens'] >= SERVER_CACHE_TOKENS

    real = {}
    for r in rows:
        for ex in r['exchanges']:
            if ex['ok'] and not fast(ex):
                real.setdefault(ex['role'], []).append(ex['latency_ms'])
    medians = {role: statistics.median(ms) for role, ms in real.items()}
    overall = statistics.median([ms for v in real.values() for ms in v]) if real else 0
    per_run, replaced = [], 0
    for r in rows:
        total = 0
        for ex in r['exchanges']:
            replaced += fast(ex)
            total += medians.get(ex['role'], overall) if fast(ex) else ex['latency_ms']
        per_run.append(total / 1000)
    decisions = sum(r['decisions'] for r in rows)
    return (sum(per_run) / decisions if decisions else 0.0), per_run, replaced


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strategies', nargs='+', choices=(*STRATEGIES, 'vote'), default=list(STRATEGIES))
    ap.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3, help='одновременных прогонов (и запросов)')
    ap.add_argument('--no-run', action='store_true', help='не считать прогоны, собрать сводку по записям')
    ap.add_argument('--cache-only', action='store_true',
                    help='ответы модели только из кэша; чего в кэше нет — прогон не считается, сеть не трогается')
    ap.add_argument('--out', default=EXPERIMENT, help=f'папка записей и сводки в runs/ (по умолчанию {EXPERIMENT})')
    args = ap.parse_args(argv)
    path = RUNS / args.out / 'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    summary = {'model': MODEL, 'llm': LLM, 'scenarios': [f'{lv}-{sd}' for lv, sd in SCENARIOS],
               'wait_s': dict(old.get('wait_s') or {}),
               'server_cached_replaced': dict(old.get('server_cached_replaced') or {}),
               'runs': list(old.get('runs') or [])}
    run = not args.no_run
    calls_before = ReplyCache().calls().get(MODEL, 0)
    t0 = time.time()
    missed = []                              # прогоны строгого повтора, которым не хватило ответа в кэше

    def put(rows):
        keys = {(r['strategy'], r['mode'], r['level'], r['seed']) for r in rows}
        summary['runs'] = [r for r in summary['runs']
                           if (r['strategy'], r['mode'], r['level'], r['seed']) not in keys] + rows
        summary['created'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        summary['real_calls_total'] = ReplyCache().calls().get(MODEL, 0)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        def batch(strategy, mode, wait):
            rows = [r for r in pool.map(_run, [(strategy, mode, lv, sd, wait, run, args.out, args.cache_only)
                                               for lv, sd in SCENARIOS]) if r]
            missed.extend(r for r in rows if 'miss' in r)
            return [r for r in rows if 'miss' not in r]

        put(batch('rule', 'frozen', 0.0))                    # правило без модели: точка отсчёта, запросов нет
        for strategy in args.strategies:
            frozen = batch(strategy, 'frozen', 0.0)
            if len(frozen) < len(SCENARIOS):
                print(f'{strategy}: записей режима frozen не хватает ({len(frozen)} из {len(SCENARIOS)}), пропуск',
                      flush=True)
                missed.extend({'miss': 'не считался: нет времени ожидания (режим frozen неполон)', 'strategy': strategy,
                               'mode': 'charged', 'level': lv, 'seed': sd} for lv, sd in SCENARIOS if args.cache_only)
                continue
            per_decision, per_run, replaced = decision_time(frozen)
            wait = round(per_decision, 1)
            summary['wait_s'][strategy] = wait
            summary['server_cached_replaced'][strategy] = replaced
            for r, s in zip(frozen, per_run):
                r['answer_s'] = round(s, 1)
            put(frozen)
            charged = batch(strategy, 'charged', wait)
            for r, s in zip(charged, decision_time(charged)[1] if charged else []):
                r['answer_s'] = round(s, 1)
            put(charged)
            for mode, rows in (('frozen', frozen), ('charged', charged)):
                if rows:
                    print(f"{strategy:12s} {mode:8s} счёт {sum(r['score'] for r in rows) / len(rows):6.2f}  "
                          f"решений {sum(r['decisions'] for r in rows):3d}  обращений "
                          f"{sum(len(r['exchanges']) for r in rows):3d}  совпало с правилом "
                          f"{sum(r['rule_matches'] for r in rows)}/{sum(r['decisions'] for r in rows)}  "
                          f"ожидание на решение {wait if mode == 'charged' else 0:.1f} с", flush=True)
    made = ReplyCache().calls().get(MODEL, 0) - calls_before
    print(f'настоящих обращений к модели за этот запуск: {made}; всего по счётчику кэша: '
          f"{summary['real_calls_total']}; {time.time() - t0:.0f} с; сводка: {path}")
    if args.cache_only:
        print(f'только кэш: не досчитано прогонов — {len(missed)}')
        for m in missed:
            print(f"  {m['strategy']}-{m['mode']} {m['level']}-{m['seed']}: {m['miss']}")


if __name__ == '__main__':
    main()
