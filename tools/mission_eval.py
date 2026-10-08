"""R13: выполняет ли робот миссию, заданную другими словами. Правило против языковой модели.

Варианты:
  rule         — правило (HeuristicPlanner): текст миссии не читает;
  llm          — модель в обвязке как есть: её спрашивают по обычным поводам и не чаще раза в 4 с,
                 на поводы внутри этих 4 с отвечает правило;
  llm_triggers — то же плюс поводы «столкновение» и «заряд у порога миссии» (флажок mission_triggers);
  llm_ask      — то же, что llm_triggers, и модель спрашивают по каждому поводу (llm_min_interval_s = 0);
  llm_ask_pen  — то же, что llm_ask, и в сводке состояния есть счётчик штрафов (state_penalties); нужен для M4.

Запуск (модель — по сети, настройки и ключ в .env основного каталога; ответы кэшируются в runs/_llm_cache):
    ./px python tools/mission_eval.py --variants rule llm llm_triggers llm_ask --jobs 3
    ./px python tools/mission_eval.py --missions M4 --levels hard --seeds 1004 1005 1007 --variants rule llm llm_ask
    ./px python tools/mission_eval.py --report        # только пересобрать runs/missions/summary.json из записей
"""
import argparse
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import llm_stats, load_env                                  # noqa: E402
from did.llm_cache import ReplyCache                                     # noqa: E402
from did.mission_criteria import MISSIONS, verify_mission                # noqa: E402
from did.recorder import load_trace                                      # noqa: E402
from did.runner import RUNS, run_episode                                 # noqa: E402

EXPERIMENT = 'missions'
MODEL = 'qwen3.8-flash-next'
ENV_FILE = '/Users/a/MAI/DID/.env'
LLM = {'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True}
VARIANTS = {
    'rule': ('adaptive', None, {}),
    'llm': ('adaptive_llm', LLM, {}),
    'llm_triggers': ('adaptive_llm', LLM, {'mission_triggers': True}),
    'llm_ask': ('adaptive_llm', LLM, {'mission_triggers': True, 'llm_min_interval_s': 0.0}),
    'llm_ask_pen': ('adaptive_llm', LLM, {'mission_triggers': True, 'llm_min_interval_s': 0.0,
                                          'state_penalties': True}),
}
MAIN_SCENARIOS = [f'{lvl}-{seed}' for lvl in ('medium', 'hard') for seed in (1001, 1002, 1003)]


def run_single(task):
    m_id, variant, level, seed = task
    agent, llm, cfg = VARIANTS[variant]
    t0 = time.time()
    try:
        res = run_episode(level, seed, agent, experiment=EXPERIMENT, arm=f'{m_id}_{variant}',
                          config={'mission': MISSIONS[m_id]['text'], **cfg}, llm=llm, save=True)
    except Exception as e:
        traceback.print_exc()
        print(f'[{m_id}][{variant}][{level}-{seed}] ОШИБКА: {e}', file=sys.stderr, flush=True)
        return None
    print(f"[{m_id}][{variant}][{level}-{seed}] счёт {res['metrics'].get('score')} ({time.time() - t0:.0f} с)",
          flush=True)
    return res['file']


def read_run(path):
    """Строка сводки по записи прогона runs/missions/<миссия>_<вариант>/<сценарий>.json.gz."""
    m_id, variant = path.parent.name.split('_', 1)
    trace = load_trace(path)
    result = trace['result']
    check = verify_mission(m_id, trace.get('track', {}), trace.get('events', []), result)
    plans = trace.get('plans', [])
    exchanges = trace.get('llm', [])
    stats = llm_stats(exchanges) if exchanges else None
    return {
        'mission': m_id, 'variant': variant, 'scenario': path.name[:-len('.json.gz')],
        'success': check['success'], 'details': check['details'],
        'score': result['score'], 'samples_collected': result['samples_collected'],
        'samples_total': result['samples_total'], 'returned': result['returned'],
        'battery': round(result['battery'], 1), 't_end': result.get('t'),
        'penalties': result['collisions'] + result['false_collects'] + result['hazard_hits'],
        # Кто принимал решения: llm — модель, heuristic — правило (у варианта с моделью это поводы внутри
        # llm_min_interval_s), fallback — правило вместо негодного ответа модели, rule — возврат, который решил агент.
        'plans_by_source': dict(Counter(p['source'] for p in plans)),
        'llm_requests': stats['requests'] if stats else 0,
        'llm_failed': stats['failed'] if stats else 0,
        'llm_latency_s': round(stats['latency_ms']['total'] / 1000.0, 1) if stats else 0.0,
        'plans': [{'t': p['t'], 'source': p['source'], 'trigger': p['trigger'], 'reasoning': p['reasoning'],
                   'subgoals': [s.get('target') and f"{s['type']} {s['target']} ({s['x']}; {s['y']})"
                                or (f"goto ({s['x']}; {s['y']})" if s['type'] == 'goto' else s['type'])
                                for s in p['subgoals']]} for p in plans],
    }


def _mean(values):
    return round(sum(values) / len(values), 2) if values else None


def compile_summary():
    rows = [read_run(p) for p in sorted((RUNS / EXPERIMENT).glob('M*_*/*.json.gz'))]
    table = []
    for m_id in MISSIONS:
        for variant in VARIANTS:
            arm = [r for r in rows if r['mission'] == m_id and r['variant'] == variant]
            main = [r for r in arm if r['scenario'] in MAIN_SCENARIOS]
            if not main:
                continue
            entry = {
                'mission': m_id, 'variant': variant, 'runs': len(main),
                'success': sum(r['success'] for r in main),
                'score_mean': _mean([r['score'] for r in main]),
                'samples_mean': _mean([r['samples_collected'] for r in main]),
                'returned': sum(r['returned'] for r in main),
                'battery_mean': _mean([r['battery'] for r in main]),
                'llm_requests': sum(r['llm_requests'] for r in main),
                'llm_failed': sum(r['llm_failed'] for r in main),
                'plans_by_source': dict(sum((Counter(r['plans_by_source']) for r in main), Counter())),
                'by_scenario': {r['scenario']: r['success'] for r in main},
            }
            if m_id == 'M4':            # по существу условие проверяется только там, где штраф был
                hit = [r for r in arm if r['details']['had_penalty']]
                entry['with_penalty'] = {'runs': len(hit), 'success': sum(r['success'] for r in hit),
                                         'scenarios': {r['scenario']: r['success'] for r in hit}}
            table.append(entry)
    summary = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': MODEL, 'main_scenarios': MAIN_SCENARIOS,
        'missions': {k: {'name': v['name'], 'text': v['text']} for k, v in MISSIONS.items()},
        'total_runs': len(rows),
        'llm_requests_in_runs': sum(r['llm_requests'] for r in rows),
        'llm_real_calls': ReplyCache().calls().get(MODEL, 0),      # настоящих обращений к модели (не из кэша)
        'table': table, 'runs': rows,
    }
    out = RUNS / EXPERIMENT / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary, out


def print_table(summary):
    print(f"{'миссия':<7}{'вариант':<14}{'выполнено':<11}{'счёт':>7}{'собрано':>9}{'вернулся':>10}{'заряд':>7}"
          f"{'обращений':>11}")
    for e in summary['table']:
        note = ''
        if 'with_penalty' in e:
            note = f"   со штрафом (все сценарии): {e['with_penalty']['success']} из {e['with_penalty']['runs']}"
        print(f"{e['mission']:<7}{e['variant']:<14}{str(e['success']) + ' из ' + str(e['runs']):<11}"
              f"{e['score_mean']:>7}{e['samples_mean']:>9}{e['returned']:>10}{e['battery_mean']:>7}"
              f"{e['llm_requests']:>11}{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=3, help='одновременных прогонов (к модели — не больше трёх)')
    ap.add_argument('--missions', nargs='+', default=list(MISSIONS), choices=list(MISSIONS))
    ap.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    ap.add_argument('--variants', nargs='+', default=['rule', 'llm'], choices=list(VARIANTS))
    ap.add_argument('--report', action='store_true', help='не считать, только пересобрать сводку из записей')
    args = ap.parse_args()

    if not args.report:
        load_env(ENV_FILE)
        tasks = [(m, v, lvl, seed) for v in VARIANTS if v in args.variants
                 for m in args.missions for lvl in args.levels for seed in args.seeds]
        print(f'Прогонов: {len(tasks)}, одновременно: {args.jobs}', flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, min(3, args.jobs))) as pool:
            done = list(pool.map(run_single, tasks))
        print(f'Готово за {time.time() - t0:.0f} с, сбоев: {done.count(None)}', flush=True)
    summary, out = compile_summary()
    print_table(summary)
    print(f"Сводка: {out}; обращений к модели в прогонах {summary['llm_requests_in_runs']}, "
          f"настоящих (не из кэша) {summary['llm_real_calls']}")


if __name__ == '__main__':
    main()
