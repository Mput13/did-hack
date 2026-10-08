#!/usr/bin/env python3
"""L3c и L3d: модель МАИ как планировщик — подсказка без формулы правила (R15) и миссии словами шире (R13).

    ./px python tools/l3_planner.py prompt                      # L3c: правило, подсказка с формулой, без формулы
    ./px python tools/l3_planner.py missions                    # L3d: миссии M1–M4 × три модели и правило
    ./px python tools/l3_planner.py missions --models DeepSeek-V4-Flash --missions M1 M2
    ./px python tools/l3_planner.py prompt --cache-only --out L3c_replay     # строгий повтор без сети
    ./px python tools/l3_planner.py prompt --report             # только пересобрать сводку
Записи — runs/<опыт>/<вариант>/<сценарий>.json.gz, сводка — runs/<опыт>/summary.json.
В быстром симуляторе мир стоит, пока модель думает: время ответа только измеряется.
"""
import argparse
import random
import time
from collections import Counter

from l3_common import (ENV_FILE, MAIN, MODELS, RUNS, exchange_stats, fmt_share, llm_opts, paired_diff, run_cells,
                       share, share_diff, slug, wait_for_idle, write_json)

from did.llm import load_env
from did.mission_criteria import MISSIONS, verify_mission
from did.recorder import load_trace
from did.runner import run_episode

EVERY = {'llm_min_interval_s': 0.0}            # на каждый повод отвечает модель, а не правило

# --- L3c: подсказка -----------------------------------------------------------------------------
PROMPT_SCENARIOS = [(lv, s) for lv in ('medium', 'hard') for s in range(1001, 1011)]
PROMPT_ARMS = {'rule': None, 'formula': None, 'goal': 'planner_system_goal'}     # вариант -> файл подсказки

# --- L3d: миссии --------------------------------------------------------------------------------
MAIN_SCENARIOS = [(lv, s) for lv in ('medium', 'hard') for s in range(1001, 1007)]
# M4 проверяется только там, где штраф был. Среди hard 1001–1012 правило получает штраф всего в трёх, поэтому до
# запуска моделей набор заменён на первые 12 сценариев hard (с 1001), где штраф получает правило — как в R13.
M4_SCENARIOS = [('hard', s) for s in (1001, 1003, 1009, 1013, 1015, 1016, 1018, 1019, 1020, 1022, 1023, 1024)]
MISSION_IDS = ('M1', 'M2', 'M3', 'M4')


def mission_scenarios(m_id):
    return M4_SCENARIOS if m_id == 'M4' else MAIN_SCENARIOS


def mission_config(m_id):
    cfg = {'mission': MISSIONS[m_id]['text'], 'mission_triggers': True, **EVERY}
    return {**cfg, 'state_penalties': True} if m_id == 'M4' else cfg


def run_cell(cell, experiment, cache_only):
    kind, arm, level, seed = cell
    if kind == 'prompt':
        if arm == 'rule':
            res = run_episode(level, seed, 'adaptive', experiment=experiment, arm=arm)
        else:
            llm = llm_opts(MAIN, cache_only)
            if PROMPT_ARMS[arm]:
                llm['prompt'] = PROMPT_ARMS[arm]
            res = run_episode(level, seed, 'adaptive_llm', experiment=experiment, arm=arm, llm=llm, config=EVERY)
    else:
        m_id, model = arm
        if model == 'rule':
            res = run_episode(level, seed, 'adaptive', experiment=experiment, arm=f'{m_id}_rule',
                              config={'mission': MISSIONS[m_id]['text']}, truth=True)
        else:
            res = run_episode(level, seed, 'adaptive_llm', experiment=experiment, arm=f'{m_id}_{slug(model)}',
                              config=mission_config(m_id), llm=llm_opts(model, cache_only), truth=True)
    return res['file']


# --- сводка L3c ---------------------------------------------------------------------------------

def _first(sg):
    if not sg:
        return '—'
    return sg['type']


def read_prompt_run(path):
    tr = load_trace(path)
    res = tr['result']
    decisions = [p for p in tr['plans'] if 'rule_match' in p]            # решения, где спрашивали модель
    by_model = [p for p in decisions if p['source'] == 'llm']            # и ответ модели принят
    diffs = [{'t': p['t'], 'trigger': p['trigger'], 'model': p['subgoals'][0] if p['subgoals'] else None,
              'rule': p.get('rule_first'), 'reasoning': p['reasoning']} for p in by_model if not p['rule_match']]
    return {'score': res['score'], 'collected': res['samples_collected'], 'total': res['samples_total'],
            'returned': bool(res['returned']), 'battery': round(res['battery'], 1),
            'penalties': res['collisions'] + res['false_collects'] + res['hazard_hits'],
            'decisions': len(decisions), 'by_model': len(by_model),
            'same': sum(p['rule_match'] for p in by_model), 'diffs': diffs,
            'sources': dict(Counter(p['source'] for p in tr['plans'])), 'exchanges': tr.get('llm') or []}


def _load(experiment, arm_dir, scenarios, reader):
    out = {}
    for level, seed in scenarios:
        path = RUNS / experiment / arm_dir / f'{level}-{seed}.json.gz'
        if path.is_file():
            out[f'{level}-{seed}'] = reader(path)
    return out


def prompt_summary(experiment):
    data = {arm: _load(experiment, arm, PROMPT_SCENARIOS, read_prompt_run) for arm in PROMPT_ARMS}
    table = {}
    for arm, runs in data.items():
        if not runs:
            continue
        scores = {k: r['score'] for k, r in runs.items()}
        e = {'runs': len(runs), 'score': round(sum(scores.values()) / len(runs), 2),
             'collected_share': round(sum(r['collected'] for r in runs.values())
                                      / sum(r['total'] for r in runs.values()), 3),
             'returned': sum(r['returned'] for r in runs.values()),
             'penalties': sum(r['penalties'] for r in runs.values()),
             'missing': [f'{lv}-{s}' for lv, s in PROMPT_SCENARIOS if f'{lv}-{s}' not in runs]}
        if arm != 'rule':
            n = sum(r['by_model'] for r in runs.values())
            e['decisions'] = sum(r['decisions'] for r in runs.values())
            e['same_as_rule'] = share(sum(r['same'] for r in runs.values()), n)
            e['diff_kinds'] = dict(Counter(f"{_first(d['model'])} вместо {_first(d['rule'])}"
                                           for r in runs.values() for d in r['diffs']))
            e['llm'] = exchange_stats([ex for r in runs.values() for ex in r['exchanges']])
            for other in ('rule', 'formula'):
                if other != arm and data.get(other):
                    e[f'score_minus_{other}'] = paired_diff(scores, {k: r['score'] for k, r in data[other].items()})
        table[arm] = e
    if 'goal' in table and 'formula' in table:
        a, b = table['goal']['same_as_rule'], table['formula']['same_as_rule']
        table['goal']['same_minus_formula'] = share_diff(a['k'], a['n'], b['k'], b['n'])
    rows = [{'arm': arm, 'scenario': k, **{f: r[f] for f in r if f != 'exchanges'}}
            for arm, runs in data.items() for k, r in runs.items()]
    return {'experiment': experiment, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'done',
            'model': MAIN, 'arms': table, 'claims': [], 'errors': [], 'runs': rows}


def prompt_report(s):
    lines = []
    for arm, e in s['arms'].items():
        line = (f"{arm}: прогонов {e['runs']}, счёт {e['score']}, собрано {e['collected_share']:.0%}, вернулся "
                f"{e['returned']}, штрафов {e['penalties']}")
        if 'same_as_rule' in e:
            x = e['llm']
            line += (f"\n   решений модели {e['same_as_rule']['n']}, как правило: {fmt_share(e['same_as_rule'], 1)}; "
                     f"иначе: {e['diff_kinds']}\n   запросов {x['requests']}: годных сразу {x['first_ok_share']:.1%}, "
                     f"отказов {x['failed']}, медиана ответа {x['latency_s']['median']} с")
            for other in ('rule', 'formula'):
                d = e.get(f'score_minus_{other}')
                if d:
                    line += (f"\n   счёт − {other}: {d['mean']:+.2f} [{d['ci'][0]:+.2f}; {d['ci'][1]:+.2f}] "
                             f"(выше {d['a_higher']}, ниже {d['b_higher']}, поровну {d['ties']})")
        lines.append(line)
    return '\n'.join(lines)


# --- сводка L3d ---------------------------------------------------------------------------------

def read_mission_run(path, m_id):
    tr = load_trace(path)
    res = tr['result']
    check = verify_mission(m_id, tr.get('track', {}), tr.get('events', []), res, plans=tr.get('plans', []),
                           modes=tr.get('modes'), truth=tr.get('truth'))
    return {'success': check['success'], 'outcome': check['outcome'], 'details': check['details'],
            'score': res['score'], 'collected': res['samples_collected'], 'total': res['samples_total'],
            'returned': bool(res['returned']), 'battery': round(res['battery'], 1),
            'sources': dict(Counter(p['source'] for p in tr.get('plans', []))), 'exchanges': tr.get('llm') or []}


def missions_summary(experiment, models):
    table, rows = {}, []
    for m_id in MISSION_IDS:
        for model in ('rule', *models):
            runs = _load(experiment, f'{m_id}_{slug(model)}', mission_scenarios(m_id),
                         lambda p: read_mission_run(p, m_id))
            if not runs:
                continue
            checked = [r for r in runs.values() if r['success'] is not None]
            e = {'mission': m_id, 'model': model, 'runs': len(runs), 'planned': len(mission_scenarios(m_id)),
                 'success': share(sum(r['success'] is True for r in checked), len(checked)),
                 'unverified': len(runs) - len(checked),
                 'outcomes': dict(Counter(r['outcome'] for r in runs.values())),
                 'score': round(sum(r['score'] for r in runs.values()) / len(runs), 1),
                 'collected': round(sum(r['collected'] for r in runs.values()) / len(runs), 2),
                 'returned': sum(r['returned'] for r in runs.values()),
                 'battery': round(sum(r['battery'] for r in runs.values()) / len(runs), 1),
                 'by_scenario': {k: r['success'] for k, r in runs.items()}}
            if model != 'rule':
                e['llm'] = exchange_stats([ex for r in runs.values() for ex in r['exchanges']])
                e['rule_decisions'] = sum(r['sources'].get('heuristic', 0) + r['sources'].get('fallback', 0)
                                          for r in runs.values())
            table[f'{m_id}/{model}'] = e
            rows += [{'mission': m_id, 'model': model, 'scenario': k, **{f: r[f] for f in r if f != 'exchanges'}}
                     for k, r in runs.items()]
    # все миссии вместе и сравнение моделей с основной
    for model in models:
        cells = [e for e in table.values() if e['model'] == model]
        if cells:
            table[f'all/{model}'] = {'mission': 'all', 'model': model,
                                     'success': share(sum(e['success']['k'] for e in cells),
                                                      sum(e['success']['n'] for e in cells)),
                                     'llm': exchange_stats([])}
    for key, e in list(table.items()):
        base = table.get(f"{e['mission']}/{MAIN}")
        if e['model'] not in ('rule', MAIN) and base:
            a, b = e['success'], base['success']
            e['minus_main'] = share_diff(a['k'], a['n'], b['k'], b['n'])
    return {'experiment': experiment, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'done',
            'models': list(models), 'missions': {k: MISSIONS[k]['text'] for k in MISSION_IDS},
            'arms': table, 'claims': [], 'errors': [], 'runs': rows}


def missions_report(s):
    lines = []
    for key, e in s['arms'].items():
        line = f"{key}: выполнено {fmt_share(e['success'])}"
        if 'runs' in e:
            line += (f", не проверено {e['unverified']}, прогонов {e['runs']} из {e['planned']}; исходы {e['outcomes']}; "
                     f"счёт {e['score']}, собрано {e['collected']}, вернулся {e['returned']}, заряд {e['battery']}")
        if e.get('llm') and e['llm']['requests']:
            x = e['llm']
            line += (f"; запросов {x['requests']}, годных сразу {x['first_ok_share']:.0%}, отказов {x['failed']}, "
                     f"медиана {x['latency_s']['median']} с")
        if e.get('minus_main'):
            line += f"; − основная модель: {e['minus_main']}"
        lines.append(line)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['prompt', 'missions'])
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--out', default=None)
    ap.add_argument('--arms', nargs='+', default=list(PROMPT_ARMS), help='prompt: какие варианты считать')
    ap.add_argument('--models', nargs='+', default=['rule', *MODELS], help='missions: какие модели считать')
    ap.add_argument('--missions', nargs='+', default=list(MISSION_IDS))
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    experiment = args.out or ('L3c' if args.what == 'prompt' else 'L3d')
    if not args.report:
        if not args.cache_only:
            load_env(ENV_FILE)
        if args.what == 'prompt':
            cells = [('prompt', arm, lv, s) for arm in args.arms for lv, s in PROMPT_SCENARIOS]
            pure = set(args.arms) == {'rule'}
        else:
            cells = [('mission', (m, model), lv, s) for model in args.models for m in args.missions
                     for lv, s in mission_scenarios(m)]
            pure = set(args.models) == {'rule'}
        if pure:                               # серия без модели — чистый счёт: не мешать Gazebo
            wait_for_idle()
        random.Random(0).shuffle(cells)        # модели вперемешку: у каждой свой сервер, очередь делится
        t0 = time.time()
        done = run_cells(cells, lambda c: run_cell(c, experiment, args.cache_only), args.jobs,
                         label=lambda c: f'{c[1]} {c[2]}-{c[3]}')
        print(f'Готово за {time.time() - t0:.0f} с, без результата: {sum(v is None for v in done.values())}', flush=True)
    if args.what == 'prompt':
        summary = prompt_summary(experiment)
        print(prompt_report(summary))
    else:
        summary = missions_summary(experiment, MODELS)
        print(missions_report(summary))
    write_json(RUNS / experiment / 'summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
