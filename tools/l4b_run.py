"""L4b: цепочки прогонов с памятью о лаборатории и парное сравнение с тем же агентом без памяти.

    ./px python tools/l4b_run.py --jobs 2                     # итог: сценарии из experiments/L4b.yaml → runs/L4b
    ./px python tools/l4b_run.py --dev --seeds 1-40 --jobs 2  # отладка на сценариях 1–80 → runs/L4b_dev
    ./px python tools/l4b_run.py --report                     # таблицы по готовой сводке runs/L4b/summary.json

Единица счёта — (агент, правила, уровень, лаборатория). Лаборатория — seed сценария, раскладка образцов —
sample_seed (0 — прежний сценарий без параметра). В единице идут прогоны без памяти на нужных раскладках и
цепочки с памятью из описания опыта (chains): память копится от шага к шагу и лежит в runs/<опыт>/lab/.
Шаг цепочки [own, k] — своя лаборатория, раскладка k; [other, k] — соседняя лаборатория (следующий seed набора).
Сравнение — шаг цепочки минус прогон без памяти на том же сценарии; интервал — бутстреп по парным разностям.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.agent import make_config                      # noqa: E402
from did.experiments import build_index, load_spec     # noqa: E402
from did.labmemory import LabMemory, Settings          # noqa: E402
from did.metrics import _diff_stats                    # noqa: E402
from did.recorder import load_trace                    # noqa: E402
from did.runner import RUNS, run_episode               # noqa: E402

METRICS = {'score': 'счёт, очки', 'samples_share': 'образцы, доля', 'returned': 'возврат, доля',
           'hazard_hits': 'штрафы за зоны, шт.', 'hazard_first': 'из них первый въезд в зону, шт.',
           'battery_used': 'расход заряда, ед.', 'false_collects': 'ложные сборы, шт.'}
BASES = {'adaptive_v2_lab': 'adaptive_v2', 'scientist_v2_lab': 'scientist_v2'}


def hazard_first(trace):
    """Сколько разных зон задето: первый въезд в каждую. Повторные штрафы в той же зоне сюда не входят."""
    sc = trace['scenario']
    zones = list(sc['hazards']) + [e['zone'] for e in sc['events'] if e['type'] == 'new_hazard']
    hit = set()
    for e in trace['events']:
        if e.get('type') == 'hazard_hit' and zones:
            hit.add(min(zones, key=lambda z: abs(np.hypot(e['x'] - z['x'], e['y'] - z['y']) - z['r']))['id'])
    return len(hit)


def _episode(exp, agent, folder, level, seed, layout, rules, lab=None, opts=None):
    s = run_episode(level, seed, agent, experiment=exp, arm=folder, rules=rules,
                    scenario_args={} if not layout else {'sample_seed': layout},
                    config={'lab_opts': opts} if opts and agent in BASES else None, lab=lab)
    m = {k: v for k, v in s['metrics'].items() if isinstance(v, (int, float, bool)) or v is None}
    m['hazard_first'] = hazard_first(load_trace(RUNS / s['file']))
    return {'metrics': m, 'file': s['file'], 'id': s['id'], 'lab': s.get('lab')}


def unit(args):
    """Все прогоны одной единицы. Возвращает список строк сводки."""
    exp, arm, cond, level, seed, other, chains, opts, extra = args
    base, rules = BASES[arm], cond.get('rules')
    rows = []
    try:
        repeated = [c['id'] for c in chains if c['id'] in extra.get('chains', [])]
        more = extra.get('layouts', []) if repeated else []
        for layout in sorted({k for c in chains for who, k in c['steps'] if who == 'own'} | set(more)):
            r = _episode(exp, base, f"{base}@{cond['id']}@L{layout}", level, seed, layout, rules)
            rows.append({'arm': base, 'condition': cond['id'], 'level': level, 'seed': seed, 'layout': layout,
                         'chain': None, 'step': None, 'metrics': r['metrics'], 'file': r['file']})
        settings = Settings(**(opts or make_config(arm).lab_opts or {}))
        for c in chains:
            path = RUNS / exp / 'lab' / f"{arm}@{cond['id']}" / c['id'] / f'{level}-{seed}.json'
            if path.exists():
                path.unlink()
            mem = LabMemory(path, settings)
            for step, (who, layout) in enumerate(c['steps'], 1):
                lab_seed = seed if who == 'own' else other
                prior = mem.priors()
                if step == 2 and c['id'] in repeated:
                    # Повторы второго шага: та же память после первого прогона, другие раскладки образцов.
                    # В память они не идут: цепочка продолжается с основной раскладки.
                    for k in more:
                        r = _episode(exp, arm, f"{arm}@{cond['id']}@{c['id']}2r{k}", level, seed, k, rules,
                                     lab=prior, opts=opts)
                        rows.append({'arm': arm, 'condition': cond['id'], 'level': level, 'seed': seed, 'layout': k,
                                     'lab_seed': seed, 'chain': c['id'], 'step': 2, 'repeat': True,
                                     'metrics': r['metrics'], 'file': r['file']})
                r = _episode(exp, arm, f"{arm}@{cond['id']}@{c['id']}{step}", level, lab_seed, layout, rules,
                             lab=prior, opts=opts)
                mem.learn(r['lab'], r['id'])
                mem.save()
                status = [p['status'] for p in r['lab']['priors']]
                rows.append({'arm': arm, 'condition': cond['id'], 'level': level, 'seed': seed, 'layout': layout,
                             'lab_seed': lab_seed, 'chain': c['id'], 'step': step, 'metrics': r['metrics'],
                             'file': r['file'],
                             'memory': {'runs': prior['runs'] if prior else 0,
                                        'hazards': len(prior['hazards']) if prior else 0,
                                        'soil_zones': len(prior['soil_zones']) if prior else 0,
                                        'volatile': bool(prior and prior['volatile']),
                                        **{k: status.count(k) for k in ('untested', 'narrowed', 'confirmed', 'refuted')},
                                        'mismatch': r['lab']['changes']['mismatch'],
                                        'soil_refuted': r['lab']['changes']['soil_refuted']}})
    except Exception as exc:          # noqa: BLE001 — одна упавшая единица не должна ронять серию
        import traceback
        return [{'error': f'{type(exc).__name__}: {exc}', 'trace': traceback.format_exc(), 'arm': arm,
                 'condition': cond['id'], 'level': level, 'seed': seed}]
    return rows


def summarize(spec, rows, wall_s, opts=None):
    runs = [r for r in rows if 'error' not in r]
    errors = [{k: r[k] for k in ('arm', 'condition', 'level', 'seed', 'error')} for r in rows if 'error' in r]
    rng = np.random.default_rng(0)
    base = {(r['arm'], r['condition'], r['level'], r['seed'], r['layout']): r for r in runs if r['chain'] is None}
    pairs, means = [], []
    for arm in BASES:
        for cond in spec['conditions']:
            for level in spec['levels']:
                for layout in sorted({r['layout'] for r in runs if r['chain'] is None}):
                    sel = [r for r in runs if r['chain'] is None and r['arm'] == BASES[arm] and r['layout'] == layout
                           and r['condition'] == cond['id'] and r['level'] == level]
                    if sel:
                        means.append({'arm': BASES[arm], 'condition': cond['id'], 'level': level, 'layout': layout,
                                      'n': len(sel), **{m: round(float(np.mean([r['metrics'][m] for r in sel])), 3)
                                                        for m in METRICS}})
                for c in spec['chains']:
                    for step in range(1, len(c['steps']) + 1):
                        group = [r for r in runs if r['arm'] == arm and r['chain'] == c['id'] and r['step'] == step
                                 and r['condition'] == cond['id'] and r['level'] == level]
                        sel = [r for r in group if not r.get('repeat')]
                        if not sel:
                            continue
                        row = {'arm': arm, 'condition': cond['id'], 'level': level, 'chain': c['id'], 'step': step,
                               'n': len(sel), 'mean': {m: round(float(np.mean([r['metrics'][m] for r in sel])), 3)
                                                       for m in METRICS},
                               'memory': {k: round(float(np.mean([r['memory'][k] for r in sel])), 3)
                                          for k in sel[0]['memory']}}
                        if c['steps'][step - 1][0] == 'own':
                            both = [(r, base.get((BASES[arm], cond['id'], level, r['seed'], r['layout']))) for r in sel]
                            both = [(a, b) for a, b in both if b is not None]
                            row['base_mean'] = {m: round(float(np.mean([b['metrics'][m] for _, b in both])), 3)
                                                for m in METRICS}
                            row['diff'] = {m: _diff_stats([float(a['metrics'][m]) - float(b['metrics'][m])
                                                           for a, b in both], rng) for m in METRICS}
                        if len(group) > len(sel):
                            # Тот же шаг на нескольких раскладках: разность усредняется внутри лаборатории,
                            # интервал — бутстреп по лабораториям.
                            by_lab = {}
                            for r in group:
                                b = base.get((BASES[arm], cond['id'], level, r['seed'], r['layout']))
                                if b is not None:
                                    by_lab.setdefault(r['seed'], []).append((r, b))
                            row['layouts'] = len(group) // len(sel)
                            row['pooled'] = {m: _diff_stats([float(np.mean([float(a['metrics'][m]) - float(b['metrics'][m])
                                                                            for a, b in v])) for v in by_lab.values()],
                                                            rng) for m in METRICS}
                            row['pooled_base'] = {m: round(float(np.mean([b['metrics'][m] for v in by_lab.values()
                                                                          for _, b in v])), 3) for m in METRICS}
                        pairs.append(row)

    def cell(chain, level, metric='score', arm='adaptive_v2_lab', cond='base', step=2, kind='diff'):
        return next((p[kind][metric] for p in pairs if p['arm'] == arm and p['condition'] == cond and p['chain'] == chain
                     and p['level'] == level and p['step'] == step and kind in p), None)

    def claim(text, metric, stat, ok):
        return {'text': text, 'metric': metric, 'pair': stat,
                'status': 'no_data' if stat is None else 'supported' if ok(stat) else 'refuted'}

    claims = [
        claim('(а) hard, базовые правила: счёт 2-го прогона с памятью выше на 2 очка и больше, интервал выше нуля',
              'score', cell('same_lab', 'hard'), lambda d: d['mean'] >= 2.0 and d['ci'][0] > 0),
        claim('(а) hard, базовые правила: возврат на базу не ниже', 'returned',
              cell('same_lab', 'hard', 'returned'), lambda d: d['mean'] >= 0),
        claim('(в) hard, базовые правила: не хуже чем на 1 очко (интервал не целиком ниже −1)', 'score',
              cell('moved', 'hard'), lambda d: d['ci'][1] >= -1.0),
        claim('(а) medium, базовые правила: не хуже чем на 1 очко', 'score',
              cell('same_lab', 'medium'), lambda d: d['mean'] >= -1.0),
        claim('(в) medium, базовые правила: не хуже чем на 1 очко', 'score',
              cell('moved', 'medium'), lambda d: d['mean'] >= -1.0),
        claim('Гипотеза: (а) hard, базовые правила — разность счёта выше нуля, интервал выше нуля', 'score',
              cell('same_lab', 'hard'), lambda d: d['ci'][0] > 0),
        claim('Механизм: (б) hard, базовые правила — первых въездов в зону с памятью меньше', 'hazard_first',
              cell('repeat', 'hard', 'hazard_first'), lambda d: d['mean'] < 0),
        claim('Уточнение (вторичное): (а) hard, базовые правила, 2-й прогон на четырёх раскладках — интервал выше нуля',
              'score', cell('same_lab', 'hard', kind='pooled'), lambda d: d['ci'][0] > 0),
        claim('Уточнение (вторичное): (в) hard, базовые правила, 2-й прогон на четырёх раскладках — интервал выше −1',
              'score', cell('moved', 'hard', kind='pooled'), lambda d: d['ci'][0] > -1.0),
    ]
    statuses = {c['status'] for c in claims[:5]}
    return {'spec': spec, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'seeds': spec['seeds'],
            'wall_s': round(wall_s, 1), 'lab_opts': opts or {},
            'status': 'supported' if statuses == {'supported'} else 'inconclusive' if 'no_data' in statuses else 'partial',
            'runs': runs, 'errors': errors, 'groups': [], 'claims': claims, 'pairs': pairs, 'means': means,
            'metrics': {k: {'label': v} for k, v in METRICS.items()}}


def _fmt(d, digits=2):
    if d is None:
        return '—'
    return (f"{d['mean']:+.{digits}f} [{d['ci'][0]:+.{digits}f}; {d['ci'][1]:+.{digits}f}]").replace('.', ',')


def report(summary, arms=None, metrics=('score', 'samples_share', 'returned', 'hazard_hits', 'hazard_first',
                                        'battery_used', 'false_collects')):
    """Таблицы для отчёта: парные разности «с памятью − без памяти» по условиям, шагам и метрикам."""
    labels = {c['id']: c['label'] for c in summary['spec']['chains']}
    out = []
    for p in summary['pairs']:
        if arms and p['arm'] not in arms:
            continue
        head = f"{p['arm']} | {p['condition']} | {p['level']} | {labels[p['chain']]} | шаг {p['step']} | n={p['n']}"
        if 'diff' not in p:
            out.append(head + f" | счёт {p['mean']['score']:.2f} (чужая лаборатория, пары нет)")
            continue
        cells = ' | '.join(f"{m}: {_fmt(p['diff'][m], 3 if m in ('samples_share', 'returned') else 2)}" for m in metrics)
        out.append(f"{head} | без памяти {p['base_mean']['score']:.2f} → с памятью {p['mean']['score']:.2f} | {cells}"
                   f" | память: зон {p['memory']['hazards']:.2f}, подтв. {p['memory']['confirmed']:.2f}, "
                   f"снято {p['memory']['refuted']:.2f}")
        if 'pooled' in p:
            cells = ' | '.join(f"{m}: {_fmt(p['pooled'][m], 3 if m in ('samples_share', 'returned') else 2)}"
                               for m in metrics)
            out.append(f"{head} | по {p['layouts']} раскладкам на лабораторию | без памяти "
                       f"{p['pooled_base']['score']:.2f} | {cells}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dev', action='store_true', help='отладка: сценарии 1–80, папка runs/L4b_dev')
    ap.add_argument('--seeds', default=None, help='диапазон сценариев, например 1-40 (только с --dev)')
    ap.add_argument('--tag', default='', help='приставка к имени папки отладочной серии')
    ap.add_argument('--arms', default=None, help='варианты с памятью через запятую')
    ap.add_argument('--levels', default=None)
    ap.add_argument('--conditions', default=None)
    ap.add_argument('--chains', default=None)
    ap.add_argument('--opts', default=None, help='настройки памяти (did.labmemory.Settings), JSON; только с --dev')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--report', action='store_true', help='только напечатать таблицы по готовой сводке')
    args = ap.parse_args()
    spec = load_spec('L4b')
    exp = 'L4b_dev' + args.tag if args.dev else 'L4b'
    if args.report:
        s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
        print(report(s, arms=args.arms.split(',') if args.arms else None))
        for c in s['claims']:
            print(f"- {c['text']}: {_fmt(c['pair'], 3 if c['metric'] == 'returned' else 2)} — {c['status']}")
        return
    if not args.dev and (args.seeds or args.opts or args.tag):
        ap.error('итоговая серия идёт только на сценариях и настройках из описания опыта; для отладки — --dev')
    if args.dev:
        lo, hi = map(int, (args.seeds or '1-40').split('-'))
        if not (1 <= lo <= hi <= 80):
            ap.error('отладка — только на сценариях 1–80')
        seeds = list(range(lo, hi + 1))
    else:
        seeds = list(range(spec['seed_start'], spec['seed_start'] + spec['seeds']))
    pick = lambda items, arg, key=lambda q: q['id']: [q for q in items if arg is None or key(q) in arg.split(',')]  # noqa: E731
    arms = pick(list(BASES), args.arms, key=lambda q: q)
    spec['levels'] = pick(spec['levels'], args.levels, key=lambda q: q)
    spec['conditions'] = pick(spec['conditions'], args.conditions)
    spec['chains'] = pick(spec['chains'], args.chains)
    spec['seeds'] = len(seeds)
    opts = json.loads(args.opts) if args.opts else None
    tasks = [(exp, arm, cond, level, seed, seeds[(i + 1) % len(seeds)], spec['chains'], opts,
              spec.get('step2_repeats') or {})
             for arm in arms for cond in spec['conditions'] for level in spec['levels'] for i, seed in enumerate(seeds)]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, res in enumerate(pool.map(unit, tasks), 1):
            rows.extend(res)
            print(f'  {i}/{len(tasks)}', end='\r', flush=True)
    s = summarize(spec, rows, time.perf_counter() - t0, opts)
    (RUNS / exp).mkdir(parents=True, exist_ok=True)
    (RUNS / exp / 'summary.json').write_text(json.dumps(s, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    if not args.dev:
        build_index()
    print(f"{exp}: прогонов {len(s['runs'])}, ошибок {len(s['errors'])}, {s['wall_s']} с")
    for e in s['errors'][:5]:
        print('  ошибка:', e)
    print(report(s))
    for c in s['claims']:
        print(f"- {c['text']}: {_fmt(c['pair'], 3 if c['metric'] == 'returned' else 2)} — {c['status']}")


if __name__ == '__main__':
    main()
