"""R12: подсчёт опыта E27 по расследованиям — чего стоит выбор опыта по ожидаемой пользе.

    ./px python -m tools.r12_table E27_pilot                  # отладка, сценарии 1–80
    ./px python -m tools.r12_table E27 --out research/findings/R12-results.json

Берёт сводку runs/<опыт>/summary.json и записи прогонов (расследования со сверкой и то, как выбирались
опыты — поле choices). Сверка со скрытой правдой сделана после прогона (did.metrics.score_inquiries), агент
её не видел.

Определения (знаменатели названы в каждой строке таблицы):
  закрытое расследование — есть вывод; не закрытое (прогон кончился раньше) считается отдельно;
  с опытом — на расследование ушёл хотя бы один манёвр (пауза, проезд, разворот); остальные — мгновенные:
    сверка у взятого образца, вывод по исключению, расследование датчика, делившее паузу с проверкой батареи;
  с выбором — хотя бы на одном шаге допустимых опытов было два или больше;
  правда есть — сверка нашла хотя бы одну настоящую причину; расследования расхода, в которых нет ни утечки,
    ни дорогого грунта, сверить нечем, они идут отдельной строкой и в доли верных не входят;
  верный вывод — причина названа и она настоящая (в том числе одна из двух настоящих).
Интервалы — 95%, бутстреп по сценариям (уровень × номер), парно: оба варианта берутся на одних и тех же
сценариях в каждой выборке.
"""
import argparse
import json
from collections import Counter

import numpy as np

from did.recorder import load_trace
from did.runner import RUNS

RUN_METRICS = ('score', 'samples_share', 'returned', 'battery_used')
RULES = ('gain', 'bits', 'cheapest', 'fixed', 'worst')


def load(exp):
    """Прогоны опыта: {(условие, вариант): {(уровень, номер): {'m': метрики, 'q': [расследования]}}}."""
    s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
    if s['errors']:
        raise SystemExit(f'в сводке {len(s["errors"])} упавших прогонов: {s["errors"][:3]}')
    out = {}
    for r in s['runs']:
        tr = load_trace(RUNS / r['file'])
        how = {c['id']: c for c in tr.get('choices', [])}
        qs = []
        for q in tr.get('inquiries', []):
            c = how[q['id']]
            steps = [x for x in c['steps'] if x['chosen']]
            closed = bool(q.get('conclusion'))
            qs.append({
                'id': q['id'], 'topic': q['topic'], 'trigger': q['anomaly'].get('trigger'), 'closed': closed,
                'maneuvers': c['maneuvers'], 'energy': sum(x['cost'] for x in q['tests'] if x.get('measured')),
                'time': (q['t_close'] - q['t_open']) if closed else None,
                'verdict': q.get('verdict'), 'truth': q.get('truth'), 'stop': c['stop'] or 'instant',
                'best': (q.get('conclusion') or {}).get('best'),
                'choice': any(len(x['options']) >= 2 for x in c['steps']), 'steps': c['steps'],
                'first': steps[0]['chosen'] if steps else None,
                'order': [x['id'] for x in sorted((x for x in q['tests'] if x.get('measured')),
                                                  key=lambda x: x['measured']['t'])],
            })
        out.setdefault((r['condition'], r['arm']), {})[(r['level'], r['seed'])] = {'m': r['metrics'], 'q': qs}
    return s, out


# --- величины: каждая — (числитель, знаменатель) по одному прогону; итог — сумма числителей / сумма знаменателей


def _sel(run, exp=None, topic=None, truth=None, choice=None):
    for q in run['q']:
        if not q['closed']:
            continue
        if exp is not None and (q['maneuvers'] >= 1) != exp:
            continue
        if topic is not None and q['topic'] != topic:
            continue
        if truth is not None and bool(q['truth']) != truth:
            continue
        if choice is not None and q['choice'] != choice:
            continue
        yield q


def ratio(num, **sel):
    def f(run):
        qs = list(_sel(run, **sel))
        return sum(num(q) for q in qs), len(qs)
    return f


def per_run(num):
    return lambda run: (num(run), 1)


CORRECT = lambda q: q['verdict'] in ('correct', 'partial')       # noqa: E731
STATS = {
    # id: (подпись, функция прогона → (числитель, знаменатель))
    'inq_per_run': ('закрытых расследований на прогон', per_run(lambda r: len(list(_sel(r))))),
    'exp_per_run': ('из них с опытом, на прогон', per_run(lambda r: len(list(_sel(r, exp=True))))),
    'choice_per_run': ('из них с выбором (≥2 допустимых опыта), на прогон',
                       per_run(lambda r: len(list(_sel(r, choice=True))))),
    'energy_per_run': ('заряда на опыты за прогон, ед.', per_run(lambda r: sum(q['energy'] for q in _sel(r)))),
    'maneuvers': ('опытов на расследование с опытом', ratio(lambda q: q['maneuvers'], exp=True)),
    'energy': ('заряда на расследование с опытом, ед.', ratio(lambda q: q['energy'], exp=True)),
    'time': ('времени на расследование с опытом, с', ratio(lambda q: q['time'], exp=True)),
    'budget': ('оборвано по бюджету заряда, доля расследований с опытом или попыткой выбора',
               lambda run: (sum(q['stop'] == 'budget' for q in _sel(run) if q['steps']),
                            sum(1 for q in _sel(run) if q['steps']))),
    'correct': ('верных выводов, доля закрытых расследований с правдой', ratio(CORRECT, truth=True)),
    'wrong': ('ошибочных выводов, та же доля', ratio(lambda q: q['verdict'] == 'wrong', truth=True)),
    'insufficient': ('«данных недостаточно», та же доля', ratio(lambda q: q['verdict'] == 'insufficient', truth=True)),
    'correct_exp': ('верных выводов, доля расследований с опытом и с правдой', ratio(CORRECT, truth=True, exp=True)),
    'insufficient_exp': ('«данных недостаточно», та же доля',
                         ratio(lambda q: q['verdict'] == 'insufficient', truth=True, exp=True)),
    # только расследования расхода заряда: выбор опыта бывает только в них
    'e_maneuvers': ('расход: опытов на расследование с опытом', ratio(lambda q: q['maneuvers'], exp=True, topic='energy')),
    'e_energy': ('расход: заряда на расследование с опытом, ед.', ratio(lambda q: q['energy'], exp=True, topic='energy')),
    'e_time': ('расход: времени на расследование с опытом, с', ratio(lambda q: q['time'], exp=True, topic='energy')),
    'e_correct': ('расход: верных, доля расследований с опытом и с правдой',
                  ratio(CORRECT, truth=True, exp=True, topic='energy')),
    'e_wrong': ('расход: ошибочных, та же доля',
                ratio(lambda q: q['verdict'] == 'wrong', truth=True, exp=True, topic='energy')),
    'e_insufficient': ('расход: «данных недостаточно», та же доля',
                       ratio(lambda q: q['verdict'] == 'insufficient', truth=True, exp=True, topic='energy')),
    # расследования, где выбор был на самом деле
    'c_maneuvers': ('с выбором: опытов на расследование', ratio(lambda q: q['maneuvers'], choice=True)),
    'c_energy': ('с выбором: заряда на расследование, ед.', ratio(lambda q: q['energy'], choice=True)),
    'c_time': ('с выбором: времени на расследование, с', ratio(lambda q: q['time'], choice=True)),
    'c_correct': ('с выбором: верных, доля расследований с правдой', ratio(CORRECT, truth=True, choice=True)),
    'c_insufficient': ('с выбором: «данных недостаточно», та же доля',
                       ratio(lambda q: q['verdict'] == 'insufficient', truth=True, choice=True)),
    **{m: (m, per_run(lambda r, m=m: float(r['m'][m]))) for m in RUN_METRICS},
}


def _table(runs, keys, f):
    return np.array([f(runs[k]) for k in keys], dtype=float)        # (сценарии, 2)


def _value(t):
    return float(t[:, 0].sum() / t[:, 1].sum()) if t[:, 1].sum() > 0 else float('nan')


def stat(runs, keys, f, idx):
    t = _table(runs, keys, f)
    boot = t[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = boot[:, 0] / boot[:, 1]
    return {'value': round(_value(t), 4), 'num': round(float(t[:, 0].sum()), 3), 'den': int(t[:, 1].sum()),
            'ci': [round(float(np.nanpercentile(b, q)), 4) for q in (2.5, 97.5)]}


def diff(runs_a, runs_b, keys, f, idx):
    """Парная разность a − b: в каждой выборке бутстрепа оба варианта — на одних и тех же сценариях."""
    ta, tb = _table(runs_a, keys, f), _table(runs_b, keys, f)
    ba, bb = ta[idx].sum(axis=1), tb[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = ba[:, 0] / ba[:, 1] - bb[:, 0] / bb[:, 1]
    lo, hi = (round(float(np.nanpercentile(d, q)), 4) for q in (2.5, 97.5))
    return {'value': round(_value(ta) - _value(tb), 4), 'ci': [lo, hi], 'zero_inside': bool(lo <= 0.0 <= hi)}


# --- как устроен выбор: сколько допустимых опытов и совпадают ли правила выбора между собой


def _rule_pick(rule, options):
    if rule == 'gain':
        best, score = None, 0.0
        for x in options:
            v = x['gain_bits'] / (x['cost'] + 0.05)
            if v > score:
                best, score = x, v
        return best['id'] if best else None
    if rule == 'bits':
        return max(options, key=lambda x: x['gain_bits'])['id']
    if rule == 'cheapest':
        return min(options, key=lambda x: x['cost'])['id']
    if rule == 'worst':
        return min(options, key=lambda x: x['gain_bits'])['id']
    return options[0]['id']       # fixed


def structure(runs):
    """По шагам выбора одного варианта: сколько было допустимых опытов и что выбрало бы каждое правило."""
    n_opts, by_topic, sets, agree, steps, first = Counter(), {}, Counter(), Counter(), 0, Counter()
    stops, kinds, no_closed = Counter(), Counter(), 0
    for run in runs.values():
        for q in run['q']:
            if not q['closed']:
                no_closed += 1
                continue
            kinds[(q['topic'], q['trigger'] or '-', 'с опытом' if q['maneuvers'] else 'мгновенное')] += 1
            if q['steps']:
                stops[q['stop']] += 1
            for i, s in enumerate(q['steps']):
                if not s['chosen']:
                    continue
                steps += 1
                n = len(s['options'])
                n_opts[n] += 1
                by_topic.setdefault(q['topic'], Counter())[n] += 1
                sets['+'.join(x['id'] for x in s['options'])] += 1
                if i == 0:
                    first[s['chosen']] += 1
                ref = _rule_pick('gain', s['options'])
                for rule in RULES:
                    agree[rule] += _rule_pick(rule, s['options']) == ref
                agree['random'] += 1.0 / n
    return {'choice_steps': steps, 'options_per_step': dict(sorted(n_opts.items())),
            'options_per_step_by_topic': {k: dict(sorted(v.items())) for k, v in by_topic.items()},
            'option_sets': dict(sets.most_common()), 'first_choice': dict(first.most_common()),
            'same_as_gain': {k: round(v / steps, 4) if steps else None for k, v in agree.items()},
            'stops': dict(stops.most_common()), 'not_closed': no_closed,
            'kinds': {' / '.join(k): v for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])}}


def verdicts(runs):
    """Исходы сверки: с правдой и без неё — разными строками, отдельно расследования с опытом."""
    out = {}
    for run in runs.values():
        for q in run['q']:
            if not q['closed']:
                continue
            for group in ('все', 'с опытом' if q['maneuvers'] else 'мгновенные',
                          *((f'расход, с опытом',) if q['topic'] == 'energy' and q['maneuvers'] else ()),
                          *(('с выбором',) if q['choice'] else ())):
                cell = out.setdefault(group, {'правда есть': Counter(), 'правды нет': Counter()})
                cell['правда есть' if q['truth'] else 'правды нет'][q['verdict']] += 1
    return {g: {k: dict(v) for k, v in c.items()} for g, c in out.items()}


def causes(runs):
    """Расследования расхода с опытом: настоящая причина × вывод, первый опыт, число опытов."""
    out = {}
    for run in runs.values():
        for q in _sel(run, exp=True, topic='energy'):
            key = '+'.join(q['truth']) or 'правды нет'
            c = out.setdefault(key, {'n': 0, 'maneuvers': 0, 'energy': 0.0, 'verdict': Counter(), 'order': Counter()})
            c['n'] += 1
            c['maneuvers'] += q['maneuvers']
            c['energy'] += q['energy']
            c['verdict'][q['verdict']] += 1
            c['order']['→'.join(q['order'])] += 1
    return {k: {'n': c['n'], 'maneuvers': round(c['maneuvers'] / c['n'], 3), 'energy': round(c['energy'] / c['n'], 3),
                'verdict': dict(c['verdict']), 'order': dict(c['order'].most_common())} for k, c in out.items()}


def build(exp, ref='random', main='gain'):
    s, data = load(exp)
    conds = [c['id'] for c in s['spec']['conditions']]
    arms = [a['id'] for a in s['spec']['arms']]
    rng = np.random.default_rng(0)
    out = {'experiment': exp, 'generated': s['generated'], 'seed_start': s['spec']['seed_start'], 'seeds': s['seeds'],
           'levels': s['spec']['levels'], 'labels': {k: v[0] for k, v in STATS.items()}, 'conditions': {}}
    for cond in conds + ['all']:
        runs = {}
        for arm in arms:
            if cond == 'all':             # оба набора правил вместе: сценарий — (правила, уровень, номер)
                runs[arm] = {(c, *k): v for c in conds for k, v in data[(c, arm)].items()}
            else:
                runs[arm] = data[(cond, arm)]
        keys = sorted(set.intersection(*(set(r) for r in runs.values())))
        idx = rng.integers(len(keys), size=(4000, len(keys)))
        block = {'scenarios': len(keys), 'arms': {}, 'vs_' + ref: {}, 'vs_' + main: {}}
        for arm in arms:
            block['arms'][arm] = {'stats': {k: stat(runs[arm], keys, f, idx) for k, (_, f) in STATS.items()},
                                  'structure': structure(runs[arm]), 'verdicts': verdicts(runs[arm]),
                                  'causes': causes(runs[arm])}
            if arm != ref:
                block['vs_' + ref][arm] = {k: diff(runs[arm], runs[ref], keys, f, idx) for k, (_, f) in STATS.items()}
            if arm != main:
                block['vs_' + main][arm] = {k: diff(runs[arm], runs[main], keys, f, idx) for k, (_, f) in STATS.items()}
        out['conditions'][cond] = block
    return out


def show(out, ref='random'):
    for cond, block in out['conditions'].items():
        arms = list(block['arms'])
        print(f'\n=== правила: {cond}; сценариев {block["scenarios"]} ===')
        print(f'{"величина":<18}' + ''.join(f'{a:>16}' for a in arms))
        for k in STATS:
            cells = [block['arms'][a]['stats'][k] for a in arms]
            print(f'{k:<18}' + ''.join(f'{c["value"]:>9.3f}/{c["den"]:<6}' for c in cells))
        print(f'--- парные разности: вариант − {ref} [95%] ---')
        for k in STATS:
            row = []
            for a in arms:
                if a == ref:
                    continue
                d = block['vs_' + ref][a][k]
                row.append(f'{a}: {d["value"]:+.3f} [{d["ci"][0]:+.3f}; {d["ci"][1]:+.3f}]{"" if d["zero_inside"] else " *"}')
            print(f'{k:<18}' + '  '.join(row))
        for a in arms:
            st = block['arms'][a]['structure']
            print(f'[{a}] шагов выбора {st["choice_steps"]}, допустимых на шаг {st["options_per_step"]}, '
                  f'первый опыт {st["first_choice"]}, остановки {st["stops"]}')
        st = block['arms']['gain']['structure']
        print('gain: наборы допустимых', st['option_sets'])
        print('gain: правило выбрало бы тот же опыт', st['same_as_gain'])
        print('gain: виды расследований', st['kinds'])
        print('gain: исходы', json.dumps(block['arms']['gain']['verdicts'], ensure_ascii=False))
        for a in arms:
            print(f'[{a}] расход с опытом по причинам:', json.dumps(block['arms'][a]['causes'], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp')
    ap.add_argument('--out', default=None, help='куда записать таблицу (JSON)')
    args = ap.parse_args()
    out = build(args.exp)
    show(out)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f'\nзаписано: {args.out}')


if __name__ == '__main__':
    main()
