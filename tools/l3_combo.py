#!/usr/bin/env python3
"""Составная миссия словами: «ровно два образца, только в левой половине арены, и домой» (для колоды команды).

    pixi run python tools/l3_combo.py              # 12 сценариев × (правило, qwen3.8-flash-next) → runs/L3combo
    pixi run python tools/l3_combo.py --report     # только сводка по готовым записям
    pixi run python tools/l3_combo.py --cache-only # повтор без сети

Обвязка и сценарии — как в L3d (tools/l3_planner.py): модель отвечает на каждый повод, medium и hard 1001–1006.
Выполнение — обе проверки did/mission_criteria.py на одной записи: M1 (собрано ровно два, сразу домой) и
M3 (истинная координата x ни разу не больше 0,5 м). В быстром симуляторе мир стоит, пока модель думает.
"""
import argparse

from l3_common import ENV_FILE, MAIN, RUNS, llm_opts, run_cells, slug, write_json

from did.llm import load_env
from did.mission_criteria import verify_mission
from did.recorder import load_trace
from did.runner import run_episode

EXPERIMENT = 'L3combo'
TEXT = 'Собери ровно два образца, не заезжая в правую половину арены (x больше 0,5 м), и сразу возвращайся на базу.'
SCENARIOS = [(lv, s) for lv in ('medium', 'hard') for s in range(1001, 1007)]
ARMS = ('rule', slug(MAIN))


def run_cell(cell, cache_only=False):
    arm, level, seed = cell
    if arm == 'rule':
        res = run_episode(level, seed, 'adaptive', experiment=EXPERIMENT, arm=arm, config={'mission': TEXT}, truth=True)
    else:
        res = run_episode(level, seed, 'adaptive_llm', experiment=EXPERIMENT, arm=arm, truth=True, llm=llm_opts(MAIN, cache_only),
                          config={'mission': TEXT, 'mission_triggers': True, 'llm_min_interval_s': 0.0})
    return res['file']


def read_run(path):
    tr = load_trace(path)
    res = tr['result']
    checks = {m: verify_mission(m, tr.get('track', {}), tr.get('events', []), res, plans=tr.get('plans', []),
                                modes=tr.get('modes'), truth=tr.get('truth')) for m in ('M1', 'M3')}
    ok = {m: c['success'] for m, c in checks.items()}
    return {'success': None if None in ok.values() else all(ok.values()), 'two_samples': ok['M1'], 'left_half': ok['M3'],
            'outcomes': {m: c['outcome'] for m, c in checks.items()}, 'collected': res['samples_collected'],
            'total': res['samples_total'], 'returned': bool(res['returned']), 'score': res['score'], 't': round(res['t'], 1),
            'llm_calls': res.get('llm_calls', 0), 'llm_failed': res.get('llm_failed', 0)}


def summary():
    out = {'experiment': EXPERIMENT, 'mission': TEXT, 'model': MAIN, 'arms': {}}
    for arm in ARMS:
        runs = {}
        for level, seed in SCENARIOS:
            path = RUNS / EXPERIMENT / arm / f'{level}-{seed}.json.gz'
            if path.exists():
                runs[f'{level}-{seed}'] = read_run(path)
        done = [r for r in runs.values() if r['success'] is not None]
        out['arms'][arm] = {'n': len(runs), 'verified': len(done), 'success': sum(r['success'] for r in done), 'runs': runs}
    write_json(RUNS / EXPERIMENT / 'summary.json', out)
    for arm, a in out['arms'].items():
        print(f"{arm}: миссия выполнена в {a['success']} из {a['verified']} (записей {a['n']})")
        for name, r in a['runs'].items():
            print(f"   {name}: {'да ' if r['success'] else 'нет'} собрано {r['collected']}/{r['total']}, "
                  f"два и домой: {r['outcomes']['M1']}, левая половина: {r['outcomes']['M3']}, вызовов {r['llm_calls']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if not a.report:
        if not a.cache_only:
            load_env(ENV_FILE)
        cells = [(arm, lv, s) for arm in ARMS for lv, s in SCENARIOS]
        run_cells(cells, lambda c: run_cell(c, a.cache_only), a.jobs, label=lambda c: f'{c[0]} {c[1]}-{c[2]}')
    summary()


if __name__ == '__main__':
    main()
