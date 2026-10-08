"""R3: одна модель, шесть сценариев, замороженный мир и измеренная цена решения.

./px python -m did.orchestration_eval --jobs 3
Ответы кэшируются; llm_wait_s калибруется отдельно для каждого способа по режиму frozen.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from .runner import RUNS, run_episode
from .recorder import load_trace
from .llm_cache import ReplyCache

MODEL = 'qwen3.8-flash-next'
STRATEGIES = ('single', 'critic', 'scored', 'scored_calc')


def _run(task):
    strategy, mode, level, seed, wait, rerun = task
    file = f'R3_real/{strategy}-{mode}/{level}-{seed}.json.gz'
    trace = load_trace(RUNS / file) if (RUNS / file).exists() and not rerun else None
    if trace is not None and trace['agent']['config']['llm_wait_s'] != wait:
        trace = None
    if trace is None:
        out = run_episode(level, seed, 'adaptive_llm', experiment='R3_real', arm=f'{strategy}-{mode}',
                          llm={'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True},
                          config={'llm_strategy': strategy, 'llm_wait_s': wait})
        trace = load_trace(RUNS / out['file'])
    else:
        out = {'metrics': trace['result'], 'file': file}
    ex = trace.get('llm', [])
    decisions = [p for p in trace['plans'] if 'rule_match' in p]
    m = out['metrics']
    return {'strategy': strategy, 'mode': mode, 'level': level, 'seed': seed,
            'score': m['score'], 'collected': m['samples_collected'], 'returned': m['returned'],
            'requests': len(ex), 'uncached': sum(not e.get('cached') for e in ex),
            'http_attempts': sum(e.get('http_attempts', 0) for e in ex),
            'wait_s': sum(e['latency_ms'] for e in ex) / 1000,
            'imposed_wait_s': len(decisions) * wait,
            'decisions': len(decisions), 'rule_matches': sum(p['rule_match'] for p in decisions),
            'failed_exchanges': sum(not e.get('ok') for e in ex),
            'errors': [e['errors'] for e in ex if e.get('errors')], 'llm_wait_s': wait, 'file': out['file']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', type=int, choices=[1, 2, 3], default=3)
    ap.add_argument('--strategies', nargs='+', choices=(*STRATEGIES, 'vote'), default=list(STRATEGIES))
    ap.add_argument('--rerun', action='store_true', help='пересчитать симуляцию; успешные ответы из кэша, сбои сети могут потребовать новых запросов')
    args = ap.parse_args()
    folder = RUNS / 'R3_real'; folder.mkdir(exist_ok=True)
    path = folder / 'summary.json'
    summary = json.loads(path.read_text()) if path.exists() else {'model': MODEL, 'runs': [], 'calibration': {}}
    before = ReplyCache().calls().get(MODEL, 0)
    scenarios = [(level, seed) for level in ('medium', 'hard') for seed in (1001, 1002, 1003)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for strategy in args.strategies:
            frozen = None
            for mode in ('frozen', 'charged'):
                wait = 0 if mode == 'frozen' else sum(r['wait_s'] for r in frozen) / sum(r['decisions'] for r in frozen)
                rows = list(pool.map(_run, [(strategy, mode, *s, wait, args.rerun) for s in scenarios]))
                if mode == 'frozen':
                    frozen = rows
                    summary['calibration'][strategy] = sum(r['wait_s'] for r in rows) / sum(r['decisions'] for r in rows)
                summary['runs'] = [r for r in summary['runs'] if (r['strategy'], r['mode']) != (strategy, mode)] + rows
                summary['cache_calls_total'] = ReplyCache().calls().get(MODEL, 0)
                summary['last_invocation_calls'] = summary['cache_calls_total'] - before
                path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                print(f'{strategy} {mode}: score={sum(r["score"] for r in rows)/6:.2f}, calls={sum(r["requests"] for r in rows)}, wait/decision={wait:.2f}', flush=True)


if __name__ == '__main__':
    main()
