"""Проверка планировщика на настоящей модели: те же сценарии проходят правило и модель.

    pixi run python -m did.llm_eval --kind codex                      # GPT по подписке, 6 сценариев
    pixi run python -m did.llm_eval --kind ollama --arm qwen2.5-3b    # локальная Qwen
    pixi run python -m did.llm_eval --kind codex --prompt planner_system_v1 --arm gpt-6-luna_v1

Записи прогонов — runs/llm_real/<arm>/<сценарий>.json.gz, сводка — runs/llm_real/<arm>.summary.json.
В быстром симуляторе время ответа модели роботу ничего не стоит (мир стоит, пока она думает),
поэтому оно только измеряется: в записи это latency_ms каждого обмена и result.llm_stats.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from .llm import CLIENT_KINDS, error_kind, llm_stats
from .recorder import load_trace
from .runner import RUNS, run_episode

EXPERIMENT = 'llm_real'


def _run(task):
    level, seed, arm, llm = task
    agent = 'adaptive_llm' if llm else 'adaptive'
    summary = run_episode(level, seed, agent, experiment=EXPERIMENT, arm=arm, llm=llm)
    trace = load_trace(RUNS / summary['file'])
    return summary, trace.get('llm') or []


def evaluate(kind, arm, levels, seeds, llm_opts=None, jobs=3, progress=print):
    """Прогоны правила и модели на одних сценариях -> сводка (она же пишется на диск)."""
    llm = {'kind': kind, **(llm_opts or {})}
    tasks = [(level, seed) for level in levels for seed in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:    # прогон в основном ждёт модель
        rule = {t: r[0] for t, r in zip(tasks, pool.map(_run, [(*t, 'rule', None) for t in tasks]))}
        done = list(pool.map(_run, [(*t, arm, llm) for t in tasks]))
    rows, exchanges, bad = [], [], []
    for (level, seed), (s, llm_log) in zip(tasks, done):
        m, st = s['metrics'], s['metrics'].get('llm_stats') or llm_stats([])
        base = rule[(level, seed)]['metrics']
        rows.append({'level': level, 'seed': seed, 'score': m['score'], 'rule_score': base['score'],
                     'samples': f"{m['samples_collected']}/{m['samples_total']}",
                     'rule_samples': f"{base['samples_collected']}/{base['samples_total']}",
                     'returned': m['returned'], 'rule_returned': base['returned'], 'plans': m['plans'],
                     'requests': st['requests'], 'first_ok': st['first_ok'], 'repaired': st['repaired'],
                     'failed': st['failed'], 'latency_ms': st['latency_ms'], 'file': s['file']})
        exchanges += llm_log
        bad += [{'level': level, 'seed': seed, 't': ex['t'], 'attempt': ex['attempt'], 'errors': ex['errors'],
                 'kinds': sorted({error_kind(str(e)) for e in ex['errors']}), 'response': ex['response']}
                for ex in llm_log if ex.get('errors')]
        if progress:
            progress(f"  {level}-{seed}: счёт {m['score']:.1f} (правило {base['score']:.1f}), запросов "
                     f"{st['requests']}: сразу {st['first_ok']}, после исправления {st['repaired']}, "
                     f"отказов {st['failed']}, ответ в среднем {st['latency_ms']['mean'] / 1000:.1f} с")
    total = llm_stats(exchanges)
    n = max(1, total['requests'])
    summary = {
        'arm': arm, 'llm': llm, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'wall_s': round(time.perf_counter() - t0, 1),
        'scenarios': len(rows), 'requests': total['requests'],
        'first_ok_share': round(total['first_ok'] / n, 3), 'repaired_share': round(total['repaired'] / n, 3),
        'fallback_share': round(total['failed'] / n, 3), 'errors': total['errors'],
        'latency_ms': total['latency_ms'], 'cached_exchanges': total['cached'],
        'score_mean': round(sum(r['score'] for r in rows) / len(rows), 2),
        'rule_score_mean': round(sum(r['rule_score'] for r in rows) / len(rows), 2),
        'score_diff': [round(r['score'] - r['rule_score'], 2) for r in rows],
        'runs': rows, 'rejected': bad,
    }
    out = RUNS / EXPERIMENT / f'{arm}.summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def report(s):
    """Сводка одной строкой на показатель — для отчёта и презентации."""
    lat = s['latency_ms']
    lines = [f"{s['arm']}: сценариев {s['scenarios']}, запросов плана {s['requests']}",
             f"  годных с первого раза {s['first_ok_share']:.0%}, после исправления ещё {s['repaired_share']:.0%}, "
             f"отказов в запасное правило {s['fallback_share']:.0%}",
             f"  время ответа: среднее {lat['mean'] / 1000:.1f} с, медиана {lat['median'] / 1000:.1f} с, "
             f"наибольшее {lat['max'] / 1000:.1f} с",
             f"  счёт: модель {s['score_mean']:.1f}, правило {s['rule_score_mean']:.1f}; разности по сценариям "
             f"{', '.join(f'{d:+.1f}' for d in s['score_diff'])}",
             f"  ошибки ответов: {s['errors'] or 'нет'}"]
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
    args = ap.parse_args(argv)
    opts = {}
    if args.model:
        opts['model'] = args.model
    if args.prompt:
        opts['prompt'] = args.prompt
    if args.kind == 'codex' and args.max_calls is not None:
        opts['max_calls'] = args.max_calls
    if args.kind in ('ollama', 'http'):
        opts.update({'use_schema': False} if args.no_schema else {}, **({'cache': True} if args.cache else {}))
    default_arm = {'codex': 'gpt-6-luna', 'ollama': 'qwen2.5-3b'}.get(args.kind, args.kind)
    arm = args.arm or (args.model or default_arm).replace(':', '-')
    summary = evaluate(args.kind, arm, args.levels, args.seeds, opts, jobs=args.jobs)
    print(report(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
