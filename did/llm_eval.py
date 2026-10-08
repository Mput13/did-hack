"""Проверка планировщика на настоящей модели: те же сценарии проходят правило и модель.

    pixi run python -m did.llm_eval --kind codex                      # GPT по подписке, 6 сценариев
    pixi run python -m did.llm_eval --kind ollama --arm qwen2.5-3b    # локальная Qwen
    pixi run python -m did.llm_eval --kind codex --prompt planner_system_v1 --arm gpt-6-luna_v1
    pixi run python -m did.llm_eval --arm gpt-6-luna --no-run         # пересчитать сводку по записям

Записи прогонов — runs/llm_real/<arm>/<сценарий>.json.gz, сводка — runs/llm_real/<arm>.summary.json.
В быстром симуляторе время ответа модели роботу ничего не стоит (мир стоит, пока она думает),
поэтому оно только измеряется: в записи это latency_ms каждого обмена и result.llm_stats.
Запросы, на которые модель не спрашивали из-за исчерпанной квоты вызовов, в доли не входят.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from .llm import CLIENT_KINDS, error_kind, llm_stats
from .recorder import load_trace
from .runner import RUNS, run_episode

EXPERIMENT = 'llm_real'
BLOCKED = 'квота вызовов исчерпана'


def _run(task):
    level, seed, arm, llm = task
    agent = 'adaptive_llm' if llm else 'adaptive'
    return run_episode(level, seed, agent, experiment=EXPERIMENT, arm=arm, llm=llm)


def _row(level, seed, metrics, rule, file):
    """Строка сводки по одному сценарию: итог модели, итог правила, ответы модели из записи прогона."""
    log = load_trace(RUNS / file).get('llm') or []
    st = llm_stats(log)
    row = {'level': level, 'seed': seed, 'score': metrics['score'], 'rule_score': rule['score'],
           'samples': f"{metrics['samples_collected']}/{metrics['samples_total']}", 'rule_samples': rule['samples'],
           'returned': metrics['returned'], 'rule_returned': rule['returned'], 'plans': metrics['plans'],
           'requests': st['requests'], 'first_ok': st['first_ok'], 'repaired': st['repaired'],
           'failed': st['failed'] - st['quota'], 'quota_blocked': st['quota'], 'latency_ms': st['latency_ms'],
           'file': file}
    bad = [{'level': level, 'seed': seed, 't': ex['t'], 'attempt': ex['attempt'], 'errors': ex['errors'],
            'kinds': sorted({error_kind(str(e)) for e in ex['errors']}), 'response': ex['response']}
           for ex in log if ex.get('errors') and BLOCKED not in {error_kind(str(e)) for e in ex['errors']}]
    return row, log, bad


def summarize(arm, llm, rows, wall_s=None, progress=None):
    """Сводка по строкам _row; пишется в runs/llm_real/<arm>.summary.json."""
    exchanges, rejected = [], []
    for row, log, bad in rows:
        exchanges += log
        rejected += bad
        if progress:
            mean_s = row['latency_ms']['mean'] / 1000
            progress(f"  {row['level']}-{row['seed']}: счёт {row['score']:.1f} (правило {row['rule_score']:.1f}), "
                     f"запросов {row['requests'] - row['quota_blocked']}: сразу {row['first_ok']}, после исправления "
                     f"{row['repaired']}, отказов {row['failed']}, ответ в среднем {mean_s:.1f} с"
                     + (f"; без модели из-за квоты ещё {row['quota_blocked']}" if row['quota_blocked'] else ''))
    rows = [row for row, _, _ in rows]
    total = llm_stats(exchanges)
    asked = total['requests'] - total['quota']           # запросы, на которые модель действительно спрашивали
    n = max(1, asked)
    errors = {k: v for k, v in total['errors'].items() if k != BLOCKED}
    whole = [r for r in rows if not r['quota_blocked']]  # прогоны, которые модель вела до конца
    mean = lambda rs, key: round(sum(r[key] for r in rs) / len(rs), 2) if rs else None      # noqa: E731
    summary = {
        'arm': arm, 'llm': llm, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'wall_s': wall_s,
        'scenarios': len(rows), 'requests': asked, 'quota_blocked': total['quota'],
        'first_ok_share': round(total['first_ok'] / n, 3), 'repaired_share': round(total['repaired'] / n, 3),
        'fallback_share': round((total['failed'] - total['quota']) / n, 3), 'errors': errors,
        'latency_ms': total['latency_ms'], 'transport_retries': total['transport_retries'],
        'tokens': total['tokens'], 'cached_exchanges': total['cached'],
        'score_mean': mean(rows, 'score'), 'rule_score_mean': mean(rows, 'rule_score'),
        'score_diff': [round(r['score'] - r['rule_score'], 2) for r in rows],
        'complete_runs': len(whole), 'score_mean_complete': mean(whole, 'score'),
        'rule_score_mean_complete': mean(whole, 'rule_score'),
        'runs': rows, 'rejected': rejected,
    }
    out = RUNS / EXPERIMENT / f'{arm}.summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def evaluate(kind, arm, levels, seeds, llm_opts=None, jobs=3, progress=print):
    """Прогоны правила и модели на одних сценариях -> сводка (она же пишется на диск)."""
    llm = {'kind': kind, **(llm_opts or {})}
    tasks = [(level, seed) for level in levels for seed in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:    # прогон в основном ждёт модель
        rule = list(pool.map(_run, [(*t, 'rule', None) for t in tasks]))
        done = list(pool.map(_run, [(*t, arm, llm) for t in tasks]))
    rows = []
    for (level, seed), base, s in zip(tasks, rule, done):
        b = base['metrics']
        rule = {'score': b['score'], 'returned': b['returned'], 'samples': f"{b['samples_collected']}/{b['samples_total']}"}
        rows.append(_row(level, seed, s['metrics'], rule, s['file']))
    return summarize(arm, llm, rows, round(time.perf_counter() - t0, 1), progress)


def resummarize(arm, progress=print):
    """Пересчитать сводку по уже записанным прогонам: итоги правила берутся из прежней сводки."""
    old = json.loads((RUNS / EXPERIMENT / f'{arm}.summary.json').read_text(encoding='utf-8'))
    rows = []
    for r in old['runs']:
        result = load_trace(RUNS / r['file'])['result']
        rule = {'score': r['rule_score'], 'returned': r['rule_returned'], 'samples': r['rule_samples']}
        rows.append(_row(r['level'], r['seed'], result, rule, r['file']))
    return summarize(arm, old.get('llm'), rows, old.get('wall_s'), progress)


def report(s):
    """Сводка одной строкой на показатель — для отчёта и презентации."""
    lat = s['latency_ms']
    lines = [f"{s['arm']}: сценариев {s['scenarios']}, запросов плана к модели {s['requests']}",
             f"  годных с первого раза {s['first_ok_share']:.0%}, после исправления ещё {s['repaired_share']:.0%}, "
             f"отказов в запасное правило {s['fallback_share']:.0%}",
             f"  время ответа: среднее {lat['mean'] / 1000:.1f} с, медиана {lat['median'] / 1000:.1f} с, "
             f"наибольшее {lat['max'] / 1000:.1f} с; повторов из-за таймаута или обрыва {s['transport_retries']}",
             f"  токенов на запрос {s['tokens']['prompt']}, на ответ {s['tokens']['completion']}",
             f"  счёт: модель {s['score_mean']:.1f}, правило {s['rule_score_mean']:.1f}; разности по сценариям "
             f"{', '.join(f'{d:+.1f}' for d in s['score_diff'])}",
             f"  ошибки ответов: {s['errors'] or 'нет'}"]
    if s['quota_blocked']:
        lines.append(f"  квота вызовов кончилась: {s['quota_blocked']} запросов ушли правилу без модели и в доли не "
                     f"входят; до конца модель вела {s['complete_runs']} прогонов из {s['scenarios']}"
                     + (f" — в них счёт {s['score_mean_complete']:.1f} против {s['rule_score_mean_complete']:.1f}"
                        if s['complete_runs'] else ''))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', default='codex', choices=CLIENT_KINDS)
    ap.add_argument('--model', default=None, help='модель вместо заданной по умолчанию')
    ap.add_argument('--arm', default=None, help='подпапка в runs/llm_real; по умолчанию — имя модели')
    ap.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    ap.add_argument('--prompt', default=None, help='другой системный промпт из did/prompts, без .md')
    ap.add_argument('--jobs', type=int, default=3, help='сколько прогонов ждут модель одновременно')
    ap.add_argument('--max-calls', type=int, default=None, help='codex: потолок настоящих вызовов за всё время')
    ap.add_argument('--no-schema', action='store_true', help='ollama, http: не слать JSON-схему ответа')
    ap.add_argument('--cache', action='store_true', help='ollama, http: класть ответы в кэш на диске')
    ap.add_argument('--min-tokens', type=int, default=None, help='http: нижняя граница лимита ответа')
    ap.add_argument('--no-run', action='store_true', help='не запускать прогоны: пересчитать сводку по записям')
    args = ap.parse_args(argv)
    default_arm = {'codex': 'gpt-6-luna', 'ollama': 'qwen2.5-3b'}.get(args.kind, args.kind)
    arm = args.arm or (args.model or default_arm).replace(':', '-')
    if args.no_run:
        print(report(resummarize(arm)))
        return 0
    opts = {}
    if args.model:
        opts['model'] = args.model
    if args.prompt:
        opts['prompt'] = args.prompt
    if args.kind == 'codex' and args.max_calls is not None:
        opts['max_calls'] = args.max_calls
    if args.kind in ('ollama', 'http'):
        opts.update({'use_schema': False} if args.no_schema else {}, **({'cache': True} if args.cache else {}))
    if args.kind == 'http':
        opts.update({} if args.no_schema else {'use_schema': True},
                    **({'min_tokens': args.min_tokens} if args.min_tokens else {}))
    print(report(evaluate(args.kind, arm, args.levels, args.seeds, opts, jobs=args.jobs)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
