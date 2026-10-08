"""Проверка правок P1 на отладочных сценариях: несколько вариантов настроек на одних сценариях.

    ./px python tools/p1_ablation.py 'old=adaptive::{}' 'v2=adaptive::{"fault_wait":true}' --levels=hard --seeds=1-80
    ./px python tools/p1_ablation.py 'old=adaptive::{}' 'v2=adaptive_v2::{}' --cond=idle10

Вариант задаётся как имя=агент:правила:настройки — правила пустые (базовые) или science, настройки —
поля AgentConfig в JSON. --cond — условие среды из таблицы CONDITIONS (дорогой простой, долгий сбой,
редкие показания…), одно на все варианты. Первый вариант — точка отсчёта: для остальных печатаются
парные разности счёта и возврата с 95% интервалом.

Результаты складываются в tmp/dev/ и при повторном запуске не пересчитываются. Запись привязана к
варианту целиком: агент, правила, настройки, условие и версия кода (содержимое did/*.py). Изменилось
что-то из этого — вариант считается заново. --reuse=имя,имя разрешает взять записи этих вариантов,
посчитанные на другой версии кода (для прежних агентов, которых правка не касается); --fresh=1 —
пересчитать всё. Сценарии по умолчанию 1–80: на них разрешено подбирать настройки.
"""
import hashlib
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.runner import run_episode      # noqa: E402

CACHE = ROOT / 'tmp' / 'dev'
KEYS = ('score', 'samples_share', 'returned', 'hazard_hits', 'false_collects', 'distance', 'time', 'battery_left')
# Условия среды: правила судьи поверх выбранного набора, параметры генератора сценариев.
CONDITIONS = {
    'base': {},
    'idle5': {'rules': {'drain_idle_per_s': 0.05}},                          # простой в 5 раз дороже
    'idle10': {'rules': {'drain_idle_per_s': 0.10}},                         # в 10 раз
    'long_fault': {'scenario': {'fault_duration': [90.0, 120.0]}},           # сбой датчика 90–120 с
    'no_fault': {'scenario': {'events': ['soil_change', 'new_hazard']}},     # сбоев датчика нет вовсе
    'hz2': {'rules': {'sensor_hz': 2.0}},                                    # датчик 2 показания в секунду
    'hz10': {'rules': {'sensor_hz': 10.0}},
}


def code_version():
    """Отпечаток кода агента и симулятора: содержимое всех did/*.py."""
    h = hashlib.sha1()
    for path in sorted((ROOT / 'did').glob('*.py')):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def variant_key(agent, rules, config, cond):
    """Чем вариант определяется, кроме версии кода."""
    return json.dumps({'agent': agent, 'rules': rules, 'config': config, 'cond': cond}, sort_keys=True)


def load_cache(path, key, code, reuse=False):
    """Записи варианта, если они посчитаны для тех же настроек и (если не reuse) той же версии кода."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('key') != key:
        return {}
    if data.get('code') != code and not reuse:
        return {}
    return data['runs']


def _job(task):
    name, agent, rules, config, cond, level, seed = task
    c = CONDITIONS[cond]
    world = {**(run_rules(rules)), **c.get('rules', {})}
    try:
        m = run_episode(level, seed, agent, experiment='P1dev', arm=name, rules=world or None, config=config,
                        scenario_args=c.get('scenario'), save=False)['metrics']
        return name, level, seed, {k: m[k] for k in KEYS}
    except Exception:      # noqa: BLE001 — один упавший прогон не должен ронять серию
        return name, level, seed, {'error': traceback.format_exc()}


def run_rules(rules):
    from did.config import SCIENCE
    return dict(SCIENCE) if rules == 'science' else {}


def diff_ci(a, b):
    """Парная разность a − b: среднее и 95% интервал (бутстреп по сценариям)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    variants = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    first, last = (int(v) for v in opts.get('seeds', '1-80').split('-'))
    levels = opts.get('levels', 'hard').split(',')
    cond = opts.get('cond', 'base')
    reuse = set(filter(None, opts.get('reuse', '').split(',')))
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
    with ProcessPoolExecutor(3) as pool:
        for name, level, seed, metrics in pool.map(_job, tasks, chunksize=2):
            if 'error' in metrics:
                print(name, level, seed, metrics['error'])
            else:
                results[name][f'{level}-{seed}'] = metrics
    for name, runs in results.items():
        if any(t[0] == name for t in tasks):      # записи, взятые через --reuse целиком, остаются со своей версией
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
