"""Проверка правок P2 на отладочных сценариях: несколько вариантов настроек на одних сценариях.

    ./px python tools/p2_ablation.py 'v2=adaptive_v2::{}' 'v3=adaptive_v3::{}' --levels=hard --seeds=1-80
    ./px python tools/p2_ablation.py 'v2=adaptive_v2::{}' 'v3=adaptive_v3::{}' --cond=range25 --seeds=1-20
    ./px python tools/p2_ablation.py 'v2=adaptive_v2:science:{}' 'near=adaptive_v2:science:{"near_collect":true}'

То же, что tools/p1_ablation.py (вариант — имя=агент:правила:настройки, первый — точка отсчёта, записи в
tmp/dev2/ привязаны к настройкам, условию и версии кода), но по умолчанию в один поток (--jobs=N) и с условиями, где правила судьи
расходятся с допущениями агента: другая дальность датчика образцов, другой его шум, дорогой простой.
--reuse=имя,имя — взять записи этих вариантов, даже если код с тех пор менялся (для прежних агентов).
"""
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from p1_ablation import code_version, diff_ci, load_cache, run_rules, variant_key      # noqa: E402

from did.runner import run_episode      # noqa: E402

CACHE = ROOT / 'tmp' / 'dev2'
KEYS = ('score', 'samples_share', 'returned', 'hazard_hits', 'false_collects', 'distance', 'time', 'battery_left')
# Условия среды: rules — правила судьи поверх выбранного набора; scenario — параметры генератора;
# agent_rules — во что при этом верит агент ({} — в правила по умолчанию; без ключа — в правила судьи).
CONDITIONS = {
    'base': {},
    'idle5': {'rules': {'drain_idle_per_s': 0.05}},                          # простой в 5 раз дороже
    'long_fault': {'scenario': {'fault_duration': [90.0, 120.0]}},           # сбой датчика 90–120 с
    'no_events': {'scenario': {'events': []}},                               # событий среды нет вовсе
    'hz2': {'rules': {'sensor_hz': 2.0}},                                    # датчик 2 показания в секунду
    # Судья считает иначе, чем думает агент: дальность датчика 1,6 и 2,5 м при вере в 2,0; шум вдвое больше.
    'range16': {'rules': {'sensor_range_m': 1.6}, 'agent_rules': {}},
    'range25': {'rules': {'sensor_range_m': 2.5}, 'agent_rules': {}},
    'sigma2': {'rules': {'sensor_sigma': 0.10}, 'agent_rules': {}},
    'collect25': {'rules': {'collect_radius_m': 0.25}, 'agent_rules': {}},   # сбор засчитывают с 0,25 м, а не с 0,30
    'battery65': {'rules': {'battery_start': 65.0}},                         # мерка: сколько очков даёт единица заряда
}


def _job(task):
    name, agent, rules, config, cond, level, seed = task
    c = CONDITIONS[cond]
    world = {**(run_rules(rules)), **c.get('rules', {})}
    bot = None
    if 'agent_rules' in c:                 # агент верит в выбранный набор правил без поправок условия
        bot = {**run_rules(rules), **c['agent_rules']}
    try:
        m = run_episode(level, seed, agent, experiment='P2dev', arm=name, rules=world or None, config=config,
                        scenario_args=c.get('scenario'), agent_rules=bot, save=False)['metrics']
        return name, level, seed, {k: m[k] for k in KEYS}
    except Exception:      # noqa: BLE001 — один упавший прогон не должен ронять серию
        return name, level, seed, {'error': traceback.format_exc()}


def main():
    variants = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    first, last = (int(v) for v in opts.get('seeds', '1-80').split('-'))
    levels = opts.get('levels', 'hard').split(',')
    cond = opts.get('cond', 'base')
    reuse = set(filter(None, opts.get('reuse', '').split(',')))     # эти варианты брать и с другой версии кода
    CACHE.mkdir(parents=True, exist_ok=True)
    code = code_version()

    results, keys, tasks = {}, {}, []
    for variant in variants:
        name, spec = variant.split('=', 1)
        agent, rules, config = spec.split(':', 2)
        config = json.loads(config or '{}')
        keys[name] = variant_key(agent, rules, config, cond)
        path = CACHE / f'{name}@{cond}.json'
        results[name] = {} if opts.get('fresh') else load_cache(path, keys[name], code, name in reuse)
        tasks += [(name, agent, rules, config, cond, level, seed)
                  for level in levels for seed in range(first, last + 1) if f'{level}-{seed}' not in results[name]]
    with ProcessPoolExecutor(int(opts.get('jobs', 1))) as pool:
        for name, level, seed, metrics in pool.map(_job, tasks, chunksize=2):
            if 'error' in metrics:
                print(name, level, seed, metrics['error'])
            else:
                results[name][f'{level}-{seed}'] = metrics
    for name, runs in results.items():
        if any(t[0] == name for t in tasks):
            (CACHE / f'{name}@{cond}.json').write_text(json.dumps({'key': keys[name], 'code': code, 'runs': runs}))

    names = list(results)
    for level in levels:
        ids = [f'{level}-{seed}' for seed in range(first, last + 1)]
        print(f'--- {level}, сценарии {first}–{last}, условие {cond}, посчитано заново {len(tasks)}')
        print(f"{'':22}" + ''.join(f'{k[:10]:>11}' for k in KEYS) + '   разность счёта [95%]   разность возврата, п.п. [95%]')
        base = {k: np.array([float(results[names[0]][i][k]) for i in ids]) for k in KEYS}
        for name in names:
            values = {k: np.array([float(results[name][i][k]) for i in ids]) for k in KEYS}
            s, r = diff_ci(values['score'], base['score']), diff_ci(values['returned'], base['returned'])
            print(f'{name:22}' + ''.join(f'{values[k].mean():11.2f}' for k in KEYS)
                  + f'   {s[0]:+.2f} [{s[1]:+.2f}; {s[2]:+.2f}]   {100 * r[0]:+.1f} [{100 * r[1]:+.1f}; {100 * r[2]:+.1f}]')


if __name__ == '__main__':
    main()
