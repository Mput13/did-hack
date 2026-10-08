"""Сравнение выполнения миссий M0–M4: эвристика (rule) vs LLM (qwen3.8-flash-next).

Запуск:
    python tools/mission_eval.py --jobs 3
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Подключение к корню проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import load_env, llm_stats
from did.mission_criteria import MISSIONS, verify_mission
from did.recorder import load_trace
from did.runner import RUNS, run_episode

EXPERIMENT = 'missions'


def run_single(task):
    m_id, variant, level, seed = task
    mission_info = MISSIONS[m_id]
    mission_text = mission_info['text']

    if variant == 'rule':
        agent = 'adaptive'
        llm = None
        arm = f'{m_id}_rule'
        cfg = {'mission': mission_text, 'mission_triggers': False}
    elif variant == 'llm':
        agent = 'adaptive_llm'
        llm = {'kind': 'http', 'model': 'qwen3.8-flash-next', 'use_schema': True, 'min_tokens': 3000, 'cache': True}
        arm = f'{m_id}_llm'
        cfg = {'mission': mission_text, 'mission_triggers': False}
    elif variant == 'llm_triggers':
        agent = 'adaptive_llm'
        llm = {'kind': 'http', 'model': 'qwen3.8-flash-next', 'use_schema': True, 'min_tokens': 3000, 'cache': True}
        arm = f'{m_id}_llm_triggers'
        cfg = {'mission': mission_text, 'mission_triggers': True}
    else:
        raise ValueError(f'Unknown variant: {variant}')

    t0 = time.time()
    try:
        res = run_episode(level, seed, agent, experiment=EXPERIMENT, arm=arm,
                          config=cfg, llm=llm, save=True)
        wall = time.time() - t0
        trace_file = RUNS / res['file']
        trace = load_trace(trace_file)
        check = verify_mission(m_id, trace.get('track', {}), trace.get('events', []), trace.get('result', {}))
        plans = trace.get('plans', [])
        llm_log = trace.get('llm', [])

        row = {
            'mission_id': m_id,
            'mission_name': mission_info['name'],
            'variant': variant,
            'arm': arm,
            'level': level,
            'seed': seed,
            'scenario': f'{level}_{seed}',
            'success': check['success'],
            'check_details': check['details'],
            'score': trace['result'].get('score', 0.0),
            'samples_collected': trace['result'].get('samples_collected', 0),
            'samples_total': trace['result'].get('samples_total', 0),
            'returned': trace['result'].get('returned', False),
            'battery': trace['result'].get('battery', 0.0),
            'distance': trace['result'].get('distance', 0.0),
            'wall_s': round(wall, 2),
            'file': res['file'],
            'plans_count': len(plans),
            'reasoning_quotes': [
                {'trigger': p.get('trigger'), 't': p.get('t'), 'reasoning': p.get('reasoning'),
                 'subgoals': p.get('subgoals')}
                for p in plans if p.get('reasoning')
            ],
            'llm_stats': llm_stats(llm_log) if llm_log else None,
        }
        status = '✓ УСПЕХ' if check['success'] else '✗ НЕУДАЧА'
        print(f'[{m_id}][{variant}][{level}-{seed}] {status} | Счёт: {row["score"]} | Образцы: {row["samples_collected"]}/{row["samples_total"]} | Возврат: {row["returned"]} | Батарея: {row["battery"]} | ({wall:.1f}с)')
        return row
    except Exception as e:
        print(f'[{m_id}][{variant}][{level}-{seed}] ОШИБКА: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {
            'mission_id': m_id,
            'variant': variant,
            'level': level,
            'seed': seed,
            'scenario': f'{level}_{seed}',
            'error': str(e),
            'success': False,
        }


def compile_summary(rows):
    missions_summary = {}
    for m_id, m_info in MISSIONS.items():
        missions_summary[m_id] = {
            'name': m_info['name'],
            'text': m_info['text'],
            'variants': {}
        }

    # Группировка по (m_id, variant)
    grouped = {}
    for r in rows:
        key = (r['mission_id'], r['variant'])
        grouped.setdefault(key, []).append(r)

    table_data = []
    for (m_id, variant), items in sorted(grouped.items()):
        valid = [r for r in items if 'error' not in r]
        n = len(valid)
        if n == 0:
            continue
        success_count = sum(1 for r in valid if r['success'])
        success_rate = round(success_count / n, 3)
        mean_score = round(sum(r['score'] for r in valid) / n, 2)
        mean_samples = round(sum(r['samples_collected'] for r in valid) / n, 2)
        return_rate = round(sum(1 for r in valid if r['returned']) / n, 3)
        mean_battery = round(sum(r['battery'] for r in valid) / n, 2)
        total_llm_calls = sum(r['llm_stats']['requests'] for r in valid if r.get('llm_stats'))

        entry = {
            'mission_id': m_id,
            'mission_name': MISSIONS[m_id]['name'],
            'variant': variant,
            'runs': n,
            'success_count': success_count,
            'success_rate': success_rate,
            'score_mean': mean_score,
            'samples_mean': mean_samples,
            'return_rate': return_rate,
            'battery_mean': mean_battery,
            'llm_calls_total': total_llm_calls,
        }
        table_data.append(entry)
        missions_summary[m_id]['variants'][variant] = entry

    # Поиск характерных кейсов (цитат reasoning) для каждой миссии
    case_studies = {}
    for m_id in MISSIONS:
        case_studies[m_id] = []
        llm_runs = [r for r in rows if r['mission_id'] == m_id and r['variant'] == 'llm' and 'error' not in r]
        for r in llm_runs:
            for quote in r.get('reasoning_quotes', []):
                case_studies[m_id].append({
                    'scenario': r['scenario'],
                    'success': r['success'],
                    't': quote['t'],
                    'trigger': quote['trigger'],
                    'reasoning': quote['reasoning'],
                    'subgoals': quote['subgoals'],
                })

    return {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_runs': len(rows),
        'table': table_data,
        'missions': missions_summary,
        'case_studies': case_studies,
        'runs': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=3, help='Параллельных задач (макс 3 к модели)')
    parser.add_argument('--missions', nargs='+', default=['M0', 'M1', 'M2', 'M3', 'M4'])
    parser.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    parser.add_argument('--variants', nargs='+', default=['rule', 'llm'])
    args = parser.parse_args()

    # Загрузка .env
    load_env('/Users/a/MAI/DID/.env')

    tasks = []
    # Сначала эвристику (она выполняется мгновенно)
    if 'rule' in args.variants:
        for m in args.missions:
            for lvl in args.levels:
                for seed in args.seeds:
                    tasks.append((m, 'rule', lvl, seed))

    # Затем LLM
    llm_variants = [v for v in args.variants if v != 'rule']
    for v in llm_variants:
        for m in args.missions:
            for lvl in args.levels:
                for seed in args.seeds:
                    tasks.append((m, v, lvl, seed))

    print(f'Всего запланировано прогонов: {len(tasks)} (потоков: {args.jobs})')
    t_start = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for r in pool.map(run_single, tasks):
            rows.append(r)

    total_time = round(time.time() - t_start, 1)
    print(f'\nВсе прогоны завершены за {total_time} с.')

    summary = compile_summary(rows)
    summary['wall_total_s'] = total_time

    out_file = RUNS / EXPERIMENT / 'summary.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Итоговая сводка сохранена в: {out_file}')


if __name__ == '__main__':
    main()
