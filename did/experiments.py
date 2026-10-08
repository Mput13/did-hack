"""Серии прогонов: описание опыта в experiments/*.yaml → прогоны → сводка для интерфейса.

    pixi run exp E1            # один опыт
    pixi run exp all           # все опыты
    pixi run exp E1 --seeds 10 # быстрее, для проверки

Опыт — это варианты агента (arms), которые проходят одни и те же сценарии (уровень × seed),
иногда при разных условиях среды (conditions). Вывод по гипотезе делается по разностям на
одинаковых сценариях.
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from . import ROOT
from .config import SCIENCE
from .memory import KnowledgeBase
from .metrics import METRICS, paired, paired_did, summarize, verdict
from .runner import RUNS, run_episode
from .scenario import ROUTE_VERSION

SPECS = ROOT / 'experiments'
BASE_CONDITION = {'id': 'base', 'label': 'Базовые правила'}


def load_spec(exp_id):
    path = SPECS / f'{exp_id}.yaml'
    spec = yaml.safe_load(path.read_text(encoding='utf-8'))
    spec.setdefault('id', exp_id)
    spec.setdefault('levels', ['easy', 'medium', 'hard'])
    spec.setdefault('seeds', 30)
    spec.setdefault('seed_start', 1001)     # агент отлаживался на сценариях 1–80, серии идут на других
    spec.setdefault('conditions', [BASE_CONDITION])
    spec.setdefault('claims', [])
    spec.setdefault('metrics', ['score', 'samples_share', 'returned', 'battery_used'])
    return spec


def list_specs():
    return sorted(p.stem for p in SPECS.glob('*.yaml'))


def _job(args, knowledge=None, soil_probe=False):
    spec_id, arm, cond, level, seed = args
    folder = arm['id'] if cond['id'] == 'base' else f"{arm['id']}@{cond['id']}"
    try:
        s = run_episode(level, seed, arm['agent'], experiment=spec_id, arm=folder,
                        scenario_args={**cond.get('scenario', {})},
                        config=arm.get('config'), rules=cond.get('rules'), llm=arm.get('llm'),
                        sim=cond.get('sim'), knowledge=knowledge, study=arm.get('study'),
                        soil_probe=soil_probe)
        s.pop('study', None)              # отчёт исследования лежит в записи прогона, в сводку идут только метрики
        s['arm'], s['condition'] = arm['id'], cond['id']
        if knowledge is None:
            s.pop('science', None)            # нужно только варианту с памятью, в сводку не идёт
        return s
    except Exception as exc:          # noqa: BLE001 — один упавший прогон не должен ронять серию
        import traceback
        return {'error': f'{type(exc).__name__}: {exc}', 'trace': traceback.format_exc(), 'arm': arm['id'],
                'condition': cond['id'], 'level': level, 'seed': seed}


def run_experiment(exp_id, seeds=None, jobs=8, progress=None):
    spec = load_spec(exp_id)
    n_seeds = seeds or spec['seeds']
    out = RUNS / exp_id
    if out.exists():
        shutil.rmtree(out)
    # Правила среды для всего опыта (например science) плюс поправки отдельных условий.
    base_rules = dict(SCIENCE) if spec.get('rules') == 'science' else dict(spec.get('rules') or {})
    conditions = [{**c, 'rules': {**base_rules, **c.get('rules', {})}} for c in spec['conditions']]
    seeds_range = range(spec['seed_start'], spec['seed_start'] + n_seeds)
    job = partial(_job, soil_probe=bool(spec.get('soil_probe')))     # разбор смены грунта по скрытой правде
    tasks = [(exp_id, arm, cond, level, seed)
             for arm in spec['arms'] if not arm.get('memory') for cond in conditions
             for level in spec['levels'] for seed in seeds_range]
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, res in enumerate(pool.map(job, tasks, chunksize=2), 1):
            results.append(res)
            if progress and (i % 10 == 0 or i == len(tasks)):
                progress(i, len(tasks))
    # Варианты с памятью идут по одному, по порядку: каждый прогон начинает с того, что выяснили предыдущие.
    for arm in (a for a in spec['arms'] if a.get('memory')):
        for cond in conditions:
            kb = KnowledgeBase(out / f"kb_{arm['id']}_{cond['id']}.json")
            order = 0
            for seed in seeds_range:
                for level in spec['levels']:
                    order += 1
                    res = job((exp_id, arm, cond, level, seed), knowledge=kb.priors())
                    res['order'] = order
                    kb.learn(res.pop('science', None), res.get('id'))
                    results.append(res)
            kb.save()
            if spec.get('publish_knowledge'):          # эти знания видны на странице «Знания» и доступны новым прогонам
                (RUNS / '_knowledge').mkdir(parents=True, exist_ok=True)
                shutil.copy(kb.path, RUNS / '_knowledge' / 'kb_state.json')
                shutil.copy(kb.path.parent / 'kb.json', RUNS / '_knowledge' / 'kb.json')
    summary = summarize_experiment(spec, results, n_seeds, time.perf_counter() - t0)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    build_index()
    return summary


def summarize_experiment(spec, results, n_seeds, wall_s):
    runs = [r for r in results if 'error' not in r]
    errors = [{k: r[k] for k in ('arm', 'condition', 'level', 'seed', 'error')} for r in results if 'error' in r]
    rng = np.random.default_rng(0)

    def pick(arm, cond=None, level=None):
        return [r for r in runs if r['arm'] == arm and (cond is None or r['condition'] == cond)
                and (level is None or r['level'] == level)]

    groups = []
    for arm in spec['arms']:
        for cond in spec['conditions']:
            for level in spec['levels'] + ['all']:
                sel = pick(arm['id'], cond['id'], None if level == 'all' else level)
                if not sel:
                    continue
                groups.append({'arm': arm['id'], 'condition': cond['id'], 'level': level, 'n': len(sel),
                               'stats': {m: summarize(sel, m, rng) for m in METRICS}})

    claims = []
    for c in spec['claims']:
        better = c.get('better') or METRICS[c['metric']][2]
        minus = c.get('minus')        # разность разностей: из (a − b) в условии вычесть (a − b) в условии minus
        cells = []
        for cond in spec['conditions']:
            if cond['id'] == minus or cond['id'] not in c.get('conditions', [cond['id']]):
                continue
            for level in c.get('levels', spec['levels']) + (['all'] if len(c.get('levels', spec['levels'])) > 1 else []):
                lv = None if level == 'all' else level
                a, b = pick(c['a'], cond['id'], lv), pick(c['b'], cond['id'], lv)
                pair = (paired_did(a, b, pick(c['a'], minus, lv), pick(c['b'], minus, lv), c['metric'], rng)
                        if minus else paired(a, b, c['metric'], rng))
                cells.append({'condition': cond['id'], 'level': level, 'pair': pair,
                              'verdict': verdict(pair, better, c.get('kind'), c.get('margin'))})
        focus = c.get('focus')        # уровень, по которому судим о гипотезе; иначе — по всем вместе
        headline = [x for x in cells if x['level'] == (focus or ('all' if any(k['level'] == 'all' for k in cells)
                                                                 else cells[0]['level']))]
        verdicts = {x['verdict'] for x in headline}
        status = ('supported' if verdicts == {'supported'} else 'refuted' if verdicts == {'refuted'}
                  else 'no_data' if verdicts == {'no_data'} else 'mixed' if {'supported', 'refuted'} <= verdicts
                  else 'inconclusive' if 'inconclusive' in verdicts and len(verdicts) == 1 else 'partial')
        claims.append({**c, 'better': better, 'cells': cells, 'status': status})

    statuses = {c['status'] for c in claims}
    if not claims:
        status = 'descriptive'
    elif statuses == {'supported'}:
        status = 'supported'
    elif statuses == {'refuted'}:
        status = 'refuted'
    elif statuses <= {'inconclusive', 'no_data'}:
        status = 'inconclusive'
    else:
        status = 'partial'
    extra = {}
    if spec.get('soil_probe'):
        extra['soil_probe'] = _soil_probe_table(spec, pick)
    if any(c.get('scenario', {}).get('soil_change_mode') == 'route' for c in spec['conditions']):
        extra['generator'] = ROUTE_VERSION      # сценарии «на пути» сравнимы только при одной версии построения
    return {'spec': spec, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'seeds': n_seeds,
            'wall_s': round(wall_s, 1), 'status': status, 'runs': runs, 'errors': errors, 'groups': groups,
            'claims': claims, 'metrics': {k: {'label': v[0], 'unit': v[1], 'better': v[2]} for k, v in METRICS.items()},
            **extra}


def _soil_probe_table(spec, pick):
    """Разбор смены грунта по условиям и вариантам (метрики did.metrics.SoilProbe).

    В описании опыта: soil_probe: {control: вариант без механизма, oracle: вариант, знающий грунты}.
    gap_closed — какую долю разрыва в счёте между control и oracle закрывает вариант: отношение средних
    разностей на одних сценариях, интервал — бутстреп по сценариям.
    """
    names = spec['soil_probe'] if isinstance(spec['soil_probe'], dict) else {}
    rng = np.random.default_rng(10)
    rows = []
    for cond in spec['conditions']:
        score = {arm['id']: {(r['level'], r['seed']): r['metrics']['score'] for r in pick(arm['id'], cond['id'])}
                 for arm in spec['arms']}
        low, high = score.get(names.get('control'), {}), score.get(names.get('oracle'), {})
        for arm in spec['arms']:
            sel = pick(arm['id'], cond['id'])
            if not sel:
                continue
            m = [r['metrics'] for r in sel]
            reached = [x for x in m if x['soil_dearer_m'] >= 0.3]
            delays = [x['soil_alarm_delay'] for x in m if x['soil_alarm_delay'] is not None]
            row = {'condition': cond['id'], 'arm': arm['id'], 'n': len(sel),
                   'runs_on_dearer': len(reached),                       # прогонов с ≥0,3 м по подорожавшему полу
                   'runs_on_dearer_alarmed': sum(x['soil_alarms_true'] > 0 for x in reached),
                   'runs_entry_known': sum(bool(x['soil_entry_known']) for x in reached),    # въехал по знакомому полу
                   'runs_reentered': sum(x['soil_dearer_entries'] > 1 for x in reached),     # въезжал больше одного раза
                   'runs_alarm_true': sum(x['soil_alarms_true'] > 0 for x in m),
                   'runs_alarm_false': sum(x['soil_alarms_false'] > 0 for x in m),
                   'alarms_true': sum(x['soil_alarms_true'] for x in m),
                   'alarms_false': sum(x['soil_alarms_false'] for x in m),
                   'alarm_delay_median': float(np.median(delays)) if delays else None,
                   'stats': {k: summarize(sel, k, rng) for k in ('soil_dearer_m', 'soil_changed_m', 'soil_extra_energy')}}
            keys = sorted(set(score[arm['id']]) & set(low) & set(high))
            if keys and arm['id'] not in names.values():
                gain = np.array([score[arm['id']][k] - low[k] for k in keys])
                gap = np.array([high[k] - low[k] for k in keys])
                idx = rng.integers(len(keys), size=(4000, len(keys)))
                gaps = gap[idx].mean(axis=1)
                boot = gain[idx].mean(axis=1)[gaps > 0] / gaps[gaps > 0]
                row['gap'] = round(float(gap.mean()), 3)
                if gap.mean() > 0 and len(boot):
                    row['gap_closed'] = round(float(gain.mean() / gap.mean()), 3)
                    row['gap_closed_ci'] = [round(float(np.percentile(boot, 2.5)), 3),
                                            round(float(np.percentile(boot, 97.5)), 3)]
            rows.append(row)
    return rows


def rebuild_from_traces(exp_id):
    """Сводка опыта по уже лежащим записям (для прогонов в Gazebo, которые запускаются по одному)."""
    from .recorder import load_trace
    spec = load_spec(exp_id)
    results = []
    for path in sorted((RUNS / exp_id).glob('*/*.json.gz')):
        tr = load_trace(path)
        arm, _, cond = path.parent.name.partition('@')
        results.append({'id': tr['id'], 'experiment': exp_id, 'arm': arm, 'condition': cond or 'base',
                        'agent': tr['agent']['name'], 'level': tr['scenario']['level'],
                        'seed': tr['scenario']['seed'], 'backend': tr['backend'], 'metrics': tr['result'],
                        'wall_s': None, 'file': str(path.relative_to(RUNS))})
    seeds = len({r['seed'] for r in results})
    summary = summarize_experiment(spec, results, seeds, 0.0)
    (RUNS / exp_id).mkdir(parents=True, exist_ok=True)
    (RUNS / exp_id / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':')),
                                                 encoding='utf-8')
    build_index()
    return summary


def build_index():
    """runs/index.json: список опытов для главной страницы интерфейса (включая ещё не запущенные)."""
    items = []
    for exp_id in list_specs():
        spec = load_spec(exp_id)
        item = {'id': exp_id, 'title': spec.get('title', exp_id), 'kind': spec.get('kind', 'experiment'),
                'question': spec.get('question'), 'hypothesis': spec.get('hypothesis'),
                'status': 'not_run', 'runs': 0, 'generated': None, 'claims': []}
        path = RUNS / exp_id / 'summary.json'
        if path.exists():
            s = json.loads(path.read_text(encoding='utf-8'))
            item.update(status=s['status'], runs=len(s['runs']), errors=len(s['errors']), generated=s['generated'],
                        claims=[{'text': c.get('text'), 'metric': c['metric'], 'status': c['status']}
                                for c in s['claims']])
        items.append(item)
    RUNS.mkdir(exist_ok=True)
    index = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'experiments': items}
    (RUNS / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', help='идентификатор опыта (E1), all или index')
    ap.add_argument('--seeds', type=int, default=None, help='сколько сценариев на уровень (по умолчанию из описания)')
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    if args.what == 'index':
        build_index()
        return
    if args.what != 'all' and load_spec(args.what).get('manual'):
        s = rebuild_from_traces(args.what)
        print(f'{args.what}: сводка пересобрана по {len(s["runs"])} записям; новые прогоны — {s["spec"]["manual"]}')
        return
    for exp_id in (list_specs() if args.what == 'all' else [args.what]):
        spec = load_spec(exp_id)
        if spec.get('manual'):
            print(f'{exp_id}: запускается вручную ({spec["manual"]})')
            continue
        print(f'{exp_id}: {spec.get("title", "")}')
        s = run_experiment(exp_id, args.seeds, args.jobs,
                           progress=lambda i, n: print(f'  {i}/{n}', end='\r', flush=True))
        print(f'  прогонов {len(s["runs"])}, ошибок {len(s["errors"])}, {s["wall_s"]} с, итог: {s["status"]}')
        for c in s['claims']:
            print(f'   - {c.get("text", c["metric"])}: {c["status"]}')
    build_index()


if __name__ == '__main__':
    main()
