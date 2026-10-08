#!/usr/bin/env python3
"""L3a: исследователь на коде против того же исследователя с моделью МАИ в ролях автора и критика.

Варианты (агент scientist, правила science, планировщик — правило):
  code                — без модели;
  roles@<модель>      — модель в ролях в обвязке как есть: автор сужает список объяснений, опыт выбирает расчёт;
  plan@<модель>       — то же и флажок inquiry_follow_plan: опыты идут в порядке плана модели.

    ./px python tools/l3_roles.py                                   # вся серия по сети (ответы — в кэш)
    ./px python tools/l3_roles.py --arms code plan@DeepSeek-V4-Flash
    ./px python tools/l3_roles.py --cache-only --out L3a_replay     # строгий повтор без сети
    ./px python tools/l3_roles.py --report                          # только пересобрать сводку
Записи — runs/<--out>/<вариант>/<сценарий>.json.gz, сводка — runs/<--out>/summary.json.
"""
import argparse
import time
from collections import Counter

from l3_common import (ENV_FILE, MAIN, MODELS, RUNS, exchange_stats, fmt_share, llm_opts, paired_diff, run_cells, share,
                       share_diff, slug, wait_for_idle, write_json)

from did.llm import load_env
from did.recorder import load_trace
from did.runner import run_episode
from did.science import OTHER

EXPERIMENT = 'L3a'
SCENARIOS = [('hard', s) for s in range(1001, 1021)] + [('medium', s) for s in range(1001, 1011)]
ARMS = ['code', f'roles@{MAIN}', f'plan@{MAIN}'] + [f'plan@{m}' for m in MODELS[1:]]


def arm_dir(arm):
    return slug(arm.replace('@', '_'))


def run_cell(cell, experiment, cache_only):
    arm, level, seed = cell
    kind, _, model = arm.partition('@')
    roles = llm_opts(model, cache_only) if model else None
    config = {'inquiry_follow_plan': True} if kind == 'plan' else None
    res = run_episode(level, seed, 'scientist', rules='science', experiment=experiment, arm=arm_dir(arm),
                      config=config, roles=roles)
    return res['file']


# --- разбор записи -------------------------------------------------------------------------------

def read_run(path):
    """Строка сводки по записи: итог судьи, исходы расследований, что решила модель."""
    tr = load_trace(path)
    res = tr['result']
    rows, exchanges = [], list(tr.get('llm') or [])
    for q in tr.get('inquiries') or []:
        if not q.get('conclusion'):
            continue
        info = q.get('llm') or {}
        exchanges += (info.get('explain') or {}).get('exchanges') or []
        row = {'id': q['id'], 'topic': q['topic'], 'trigger': q['anomaly'].get('trigger'), 'truth': q.get('truth'),
               'verdict': q.get('verdict'), 'best': q['conclusion']['best'], 'status': q['conclusion']['status'],
               'tests_run': [x['id'] for x in q['tests'] if x.get('measured')],
               'energy': round(sum(x['cost'] for x in q['tests'] if x.get('measured')), 2)}
        if 'draft' in info:                     # расследование обсуждалось с моделью
            offered = [a for a in info['offered'] if a != OTHER]
            final, draft = info['final'], info['draft']
            dropped = [a for a in offered if a not in final['consider']]
            row['llm'] = {
                'author': draft['source'], 'critic': info['critic']['source'], 'final': final['source'],
                'verdict': info['critic']['verdict'], 'issues': [i['kind'] for i in info['critic']['issues']],
                'revised': info['revised'],
                'changed': info['revised'] and (final['plan'] != draft['plan']
                                                or set(final['consider']) != set(draft['consider'])),
                'dropped': dropped, 'dropped_truth': sorted(set(dropped) & set(q.get('truth') or [])),
                'n_tests_offered': len(info['tests_offered']), 'code_first': info['code_first'],
                'model_first': final['plan'][0] if final['plan'] else None,
                'plan_len': len(final['plan']), 'extra': final['extra'] or draft['extra'],
                'open_issues': info['open_issues'], 'latency_s': round(info['latency_ms'] / 1000, 1),
                'explain': (info.get('explain') or {}).get('source')}
        elif info.get('explain'):
            row['llm'] = {'explain': info['explain']['source']}
        rows.append(row)
    return {'score': res['score'], 'collected': res['samples_collected'], 'total': res['samples_total'],
            'returned': bool(res['returned']), 'inquiries': rows, 'exchanges': exchanges}


def arm_summary(runs):
    """Сводка по варианту: runs — {(уровень, номер): строка read_run}."""
    qs = [q for r in runs.values() for q in r['inquiries']]
    n = len(qs)
    identified = [q for q in qs if q['status'] == 'identified']
    right = [q for q in identified if q['verdict'] in ('correct', 'partial')]
    checkable = [q for q in identified if q['verdict'] != 'unverifiable']
    out = {
        'runs': len(runs), 'score': None, 'inquiries': n,
        'identified': share(len(identified), n),
        'correct_of_identified': share(len(right), len(checkable)),
        'correct_of_all': share(len(right), n),
        'wrong': sum(q['verdict'] == 'wrong' for q in qs),
        'insufficient': sum(q['status'] != 'identified' for q in qs),
        'tests_per_inquiry': round(sum(len(q['tests_run']) for q in qs) / max(1, n), 2),
        'energy_per_inquiry': round(sum(q['energy'] for q in qs) / max(1, n), 2),
        'by_topic': {t: {'n': sum(q['topic'] == t for q in qs),
                         'right': sum(q['topic'] == t and q in right for q in qs),
                         'wrong': sum(q['topic'] == t and q['verdict'] == 'wrong' for q in qs),
                         'insufficient': sum(q['topic'] == t and q['status'] != 'identified' for q in qs)}
                     for t in ('energy', 'fault', 'sensor')},
    }
    talks = [q['llm'] for q in qs if 'author' in (q.get('llm') or {})]
    if talks:
        choice = [t for t in talks if t['n_tests_offered'] >= 2 and t['code_first']]     # выбор опыта был
        exchanges = [ex for r in runs.values() for ex in r['exchanges']]
        by_role = {role: exchange_stats([ex for ex in exchanges if ex.get('role') == role])
                   for role in ('author', 'critic', 'explainer')}
        out['llm'] = {
            'deliberations': len(talks),
            'author_by_model': share(sum(t['author'] == 'llm' for t in talks), len(talks)),
            'critic_by_model': share(sum(t['critic'] == 'llm' for t in talks), len(talks)),
            'critic_revise': share(sum(t['verdict'] == 'revise' for t in talks), len(talks)),
            'critic_issue_kinds': dict(Counter(k for t in talks for k in t['issues'])),
            'plan_changed_by_critic': share(sum(bool(t['changed']) for t in talks), len(talks)),
            'dropped_alternative': share(sum(bool(t['dropped']) for t in talks), len(talks)),
            'dropped_truth': sum(bool(t['dropped_truth']) for t in talks),
            'dropped_kinds': dict(Counter(a for t in talks for a in t['dropped'])),
            'first_test_same_as_code': share(sum(t['model_first'] == t['code_first'] for t in choice), len(choice)),
            'first_test_pairs': dict(Counter(f"{t['code_first']}→{t['model_first']}" for t in choice
                                             if t['model_first'] != t['code_first'])),
            'with_extra': share(sum(bool(t['extra']) for t in talks), len(talks)),
            'extra': [x for t in talks for x in t['extra']],
            'open_issues_after': sum(t['open_issues'] > 0 for t in talks),
            'latency_per_deliberation_s': round(sum(t['latency_s'] for t in talks) / len(talks), 1),
            'explain_by_model': share(sum(q['llm'].get('explain') == 'llm' for q in qs if q.get('llm')),
                                      sum(bool((q.get('llm') or {}).get('explain')) for q in qs)),
            'exchanges': exchange_stats(exchanges), 'by_role': by_role,
        }
    return out


def compile_summary(experiment, arms):
    data = {}
    for arm in arms:
        planned = {f'{level}-{seed}' for level, seed in SCENARIOS}
        files = sorted((RUNS / experiment / arm_dir(arm)).glob('*.json.gz'))
        if experiment == EXPERIMENT:             # в основной серии — только сценарии плана
            files = [p for p in files if p.name[:-len('.json.gz')] in planned]
        runs = {p.name[:-len('.json.gz')]: read_run(p) for p in files}
        if runs:
            data[arm] = runs
    table = {}
    for arm, runs in data.items():
        entry = arm_summary(runs)
        scores = {k: r['score'] for k, r in runs.items()}
        entry['score'] = round(sum(scores.values()) / len(scores), 2)
        entry['returned'] = sum(r['returned'] for r in runs.values())
        entry['collected_share'] = round(sum(r['collected'] for r in runs.values())
                                         / sum(r['total'] for r in runs.values()), 3)
        entry['missing'] = [f'{lv}-{s}' for lv, s in SCENARIOS if f'{lv}-{s}' not in runs]
        if arm != 'code' and 'code' in data:
            base = arm_summary(data['code'])
            entry['score_minus_code'] = paired_diff(scores, {k: r['score'] for k, r in data['code'].items()})
            a, b = entry['correct_of_identified'], base['correct_of_identified']
            entry['correct_minus_code'] = share_diff(a['k'], a['n'], b['k'], b['n'])
            a, b = entry['correct_of_all'], base['correct_of_all']
            entry['correct_of_all_minus_code'] = share_diff(a['k'], a['n'], b['k'], b['n'])
        table[arm] = entry
    summary = {'experiment': experiment, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'done',
               'scenarios': [f'{lv}-{s}' for lv, s in SCENARIOS], 'arms': table, 'claims': [], 'errors': [],
               'runs': [{'arm': arm, 'scenario': k, **{f: r[f] for f in ('score', 'collected', 'total', 'returned')},
                         'inquiries': r['inquiries']} for arm, runs in data.items() for k, r in runs.items()]}
    write_json(RUNS / experiment / 'summary.json', summary)
    return summary


def report(summary):
    lines = []
    for arm, e in summary['arms'].items():
        lines.append(f"{arm}: прогонов {e['runs']}, счёт {e['score']}, вернулся {e['returned']}, расследований "
                     f"{e['inquiries']}; верных среди установленных {fmt_share(e['correct_of_identified'])}; неверных "
                     f"{e['wrong']}, без вывода {e['insufficient']}; опытов на расследование {e['tests_per_inquiry']}, "
                     f"заряда {e['energy_per_inquiry']}")
        if e.get('score_minus_code'):
            d = e['score_minus_code']
            lines.append(f"   счёт − код: {d['mean']:+.2f} [{d['ci'][0]:+.2f}; {d['ci'][1]:+.2f}], выше {d['a_higher']}, "
                         f"ниже {d['b_higher']}; доля верных − код: {e['correct_minus_code']}")
        if e.get('llm'):
            m = e['llm']
            lines.append(f"   обсуждений {m['deliberations']}; критик требует правки {fmt_share(m['critic_revise'])}, план "
                         f"изменён {fmt_share(m['plan_changed_by_critic'])}; автор убрал объяснение "
                         f"{fmt_share(m['dropped_alternative'])} (из них правду — {m['dropped_truth']}); первый опыт как у "
                         f"расчёта {fmt_share(m['first_test_same_as_code'])}; свои объяснения {fmt_share(m['with_extra'])}")
            x = m['exchanges']
            lines.append(f"   запросов {x['requests']} (обменов {x['exchanges']}): годных сразу {x['first_ok_share']:.0%}, "
                         f"отказов {x['failed']}; ответ: медиана {x['latency_s']['median']} с; на обсуждение "
                         f"{m['latency_per_deliberation_s']} с; ошибки {x['errors']}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--jobs', type=int, default=6, help='сколько прогонов одновременно ждут модель')
    ap.add_argument('--out', default=EXPERIMENT)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--levels', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=None, help='номера сценариев вместо плана (отладка)')
    args = ap.parse_args()
    if not args.report:
        scen = SCENARIOS if not args.seeds else [(lv, s) for lv in (args.levels or ['hard']) for s in args.seeds]
        if not args.cache_only:
            load_env(ENV_FILE)
        if set(args.arms) == {'code'}:           # серия без модели — чистый счёт: не мешать Gazebo
            wait_for_idle()
        cells = [(arm, lv, s) for arm in args.arms for lv, s in scen]
        t0 = time.time()
        done = run_cells(cells, lambda c: run_cell(c, args.out, args.cache_only), args.jobs,
                         label=lambda c: f'{c[0]} {c[1]}-{c[2]}')
        print(f'Готово за {time.time() - t0:.0f} с, без результата: {sum(v is None for v in done.values())}', flush=True)
    summary = compile_summary(args.out, ARMS)
    print(report(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
