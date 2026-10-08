"""R16b: режимы «не стоять, пока модель думает» на настоящей модели (qwen3.8-flash-next, сервер МАИ).

Группы прогонов:
  plain  — обычная миссия, hard, сценарии 7001–7010 (первые десять из E24), режимы off / rule / leash;
  m1     — миссия M1 «ровно два образца», шесть сценариев R13, обвязка llm_ask, режимы off / rule / leash;
  m4     — миссия M4 «после штрафа домой», шесть сценариев hard из R13, где правило получает штраф,
           обвязка llm_ask_pen (модель видит счётчик штрафов), режимы off / leash;
  m4main — M4 на шести основных сценариях R13 (medium и hard, 1001–1003), те же режимы.
Время ответа — настоящее, своё у каждого ответа (llm_wait_measured); off — «стоять и ждать» это время.

    ./px python tools/r16b_eval.py run --groups plain m1 m4 --jobs 3     # сеть; готовые записи не пересчитываются
    ./px python tools/r16b_eval.py run --groups plain m1 m4 --cache-only --out R16b_replay --jobs 3
    ./px python tools/r16b_eval.py compare --out R16b_replay --with R16b # сверка повтора с исходными записями
    ./px python tools/r16b_eval.py report                                # таблицы и снимок чисел

Записи — runs/<--out>/<группа>_<режим>/<сценарий>.json.gz, сводка — runs/<--out>/summary.json; report кладёт
снимок чисел в research/findings/R16b-results.json и таблицы в research/findings/R16b-tables.md.
Один и тот же сценарий одной группы одновременно в двух режимах не идёт (замок на файле): иначе одинаковый
первый вопрос ушёл бы на сервер дважды, и в кэше осталось бы время ответа только одного из них.
"""
import argparse
import fcntl
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import CacheMiss, load_env                                  # noqa: E402
from did.llm_cache import ReplyCache                                     # noqa: E402
from did.mission_criteria import MISSIONS                                # noqa: E402
from did.recorder import load_trace                                      # noqa: E402
from did.runner import RUNS, run_episode                                 # noqa: E402
from did.wait_eval import ENV_FILE, LLM, MODEL, mission_row              # noqa: E402
from did.waiting import LATE                                             # noqa: E402

OUT = 'R16b'
FINDINGS = ROOT / 'research' / 'findings'
MODES = {'off': {}, 'rule': {'llm_act_while_waiting': 'rule'}, 'leash': {'llm_act_while_waiting': 'leash'}}
ASK = {'mission_triggers': True, 'llm_min_interval_s': 0.0}              # обвязка llm_ask из R13
R13_MAIN = [(lv, sd) for lv in ('medium', 'hard') for sd in (1001, 1002, 1003)]
GROUPS = {
    'plain': {'mission': None, 'config': {}, 'arms': ('off', 'rule', 'leash'),
              'scenarios': [('hard', sd) for sd in range(7001, 7011)]},
    'm1': {'mission': 'M1', 'config': ASK, 'arms': ('off', 'rule', 'leash'), 'scenarios': R13_MAIN},
    'm4': {'mission': 'M4', 'config': {**ASK, 'state_penalties': True}, 'arms': ('off', 'leash'),
           'scenarios': [('hard', sd) for sd in (1009, 1013, 1015, 1016, 1018, 1024)]},
    'm4main': {'mission': 'M4', 'config': {**ASK, 'state_penalties': True}, 'arms': ('off', 'leash'),
               'scenarios': R13_MAIN},
}
IN_TIME = ('agree', 'same', 'tail', 'switched')        # ответ пришёл, пока вопрос стоял, и что-то решил
RESULT_KEYS = ('t', 'battery', 'samples_collected', 'returned', 'score', 'collisions', 'false_collects',
               'hazard_hits', 'distance', 'reason')
ARM_NAMES = {'off': 'стоять и ждать (`off`)', 'rule': 'ехать по правилу (`rule`)', 'leash': 'привязь (`leash`)'}


def tasks_of(groups):
    return [(g, arm, lv, sd) for g in groups for arm in GROUPS[g]['arms'] for lv, sd in GROUPS[g]['scenarios']]


def record_path(out, task):
    g, arm, lv, sd = task
    return RUNS / out / f'{g}_{arm}' / f'{lv}-{sd}.json.gz'


def row_of(out, task):
    """Строка сводки по записи прогона."""
    g, arm, lv, sd = task
    path = record_path(out, task)
    trace = load_trace(path)
    res = trace['result']
    exchanges = trace.get('llm') or []
    wait = res.get('llm_wait') or {}
    row = {'group': g, 'arm': arm, 'level': lv, 'seed': sd, 'score': res['score'],
           'collected': res['samples_collected'], 'total': res['samples_total'], 'returned': bool(res['returned']),
           'hazard_hits': res.get('hazard_hits', 0), 'time_s': res['time'], 'idle_s': res.get('idle_s') or 0.0,
           'reason': res.get('reason'), 'llm_wait': wait,
           # Запрос — первый обмен и его исправления; время ответа — по обменам (у ответа из кэша — исходное).
           'requests': sum(1 for ex in exchanges if ex.get('attempt', 1) == 1),
           'exchanges': len(exchanges), 'cached': sum(1 for ex in exchanges if ex.get('cached')),
           'failed_exchanges': sum(1 for ex in exchanges if not ex.get('ok')),
           'latency_s': [round((ex.get('latency_ms') or 0) / 1000.0, 1) for ex in exchanges],
           'plans_by_source': {}}
    for p in trace.get('plans') or []:
        row['plans_by_source'][p['source']] = row['plans_by_source'].get(p['source'], 0) + 1
    if wait:
        row.update(asked=wait['asked'], answered=wait['answered'], in_time=sum(wait[k] for k in IN_TIME),
                   late=sum(wait[k] for k in LATE if k != 'timeout'), timeouts=wait['timeout'],
                   failed=wait['failed'], gave_up=wait['gave_up'], unanswered=wait['unanswered'])
    mission = GROUPS[g]['mission']
    if mission:
        m = mission_row(mission, arm, lv, sd, str(path.relative_to(RUNS)))
        row.update({k: m[k] for k in ('success', 'outcome', 'exactly_two', 'return_delay_s', 'had_penalty',
                                      'reaction_s', 'collected_after', 'home_after_penalty') if k in m})
    return row


def run_one(job):
    task, out, cache_only, force = job
    g, arm, lv, sd = task
    spec = GROUPS[g]
    path = record_path(out, task)
    lock = RUNS / out / 'locks' / f'{g}-{lv}-{sd}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)          # тот же сценарий той же группы в другом режиме подождёт
        if path.exists() and not force:
            return {'task': task, 'status': 'kept'}
        config = {**spec['config'], **MODES[arm], 'llm_wait_measured': True}
        if spec['mission']:
            config['mission'] = MISSIONS[spec['mission']]['text']
        t0 = time.time()
        try:
            run_episode(lv, sd, 'adaptive_llm', experiment=out, arm=f'{g}_{arm}', config=config, truth=True,
                        llm={**LLM, 'cache': 'only'} if cache_only else dict(LLM))
        except CacheMiss as e:
            print(f'[{g}][{arm}][{lv}-{sd}] нет ответа в кэше: {e}', flush=True)
            return {'task': task, 'status': 'miss', 'error': str(e)[:300]}
        except Exception as e:                 # noqa: BLE001 — упавший прогон не должен ронять серию
            print(f'[{g}][{arm}][{lv}-{sd}] ОШИБКА: {type(e).__name__}: {e}', flush=True)
            return {'task': task, 'status': 'error', 'error': f'{type(e).__name__}: {e}'[:300]}
    r = row_of(out, task)
    print(f"[{g}][{arm}][{lv}-{sd}] счёт {r['score']:.1f}, вернулся {int(r['returned'])}, простой {r['idle_s']:.0f} с, "
          f"запросов {r['requests']} (непринятых обменов {r['failed_exchanges']}), {time.time() - t0:.0f} с", flush=True)
    return {'task': task, 'status': 'ok'}


def cmd_run(args):
    if not args.cache_only:
        load_env(ENV_FILE)
    tasks = tasks_of(args.groups)
    before = ReplyCache().calls().get(MODEL, 0)
    t0 = time.time()
    print(f"Прогонов: {len(tasks)}, одновременно {args.jobs}, runs/{args.out}"
          f"{', только кэш' if args.cache_only else ''}", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        done = list(pool.map(run_one, [(t, args.out, args.cache_only, args.force) for t in tasks]))
    made = ReplyCache().calls().get(MODEL, 0) - before
    log = RUNS / args.out / 'launches.json'
    old = json.loads(log.read_text(encoding='utf-8')) if log.exists() else []
    old.append({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'groups': args.groups, 'cache_only': args.cache_only,
                'network_calls': made, 'wall_s': round(time.time() - t0),
                'status': {s: sum(d['status'] == s for d in done) for s in ('ok', 'kept', 'miss', 'error')},
                'not_done': [d for d in done if d['status'] in ('miss', 'error')]})
    log.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"Готово за {time.time() - t0:.0f} с: {old[-1]['status']}; настоящих обращений к сети за запуск: {made}")


# --- сводка ---------------------------------------------------------------------------------

def boot(values, n=10000, seed=0):
    """Среднее и 95%-й интервал бутстрепом по сценариям."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [round(float(v.mean()), 2), round(float(np.percentile(means, 2.5)), 2),
            round(float(np.percentile(means, 97.5)), 2)]


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def collect(out):
    rows, missing = [], []
    for g in GROUPS:
        for task in tasks_of([g]):
            if record_path(out, task).exists():
                rows.append(row_of(out, task))
            else:
                missing.append('/'.join(map(str, task)))
    return rows, missing


def arm_summary(sel):
    n = len(sel)
    lat = [x for r in sel for x in r['latency_s']]
    e = {'n': n, 'score': boot([r['score'] for r in sel]), 'returned': sum(r['returned'] for r in sel),
         'hazard_hits_mean': round(sum(r['hazard_hits'] for r in sel) / n, 2),
         'collected_mean': round(sum(r['collected'] for r in sel) / n, 2),
         'idle_s': boot([r['idle_s'] for r in sel]), 'time_s_mean': round(sum(r['time_s'] for r in sel) / n, 1),
         'requests': sum(r['requests'] for r in sel), 'exchanges': sum(r['exchanges'] for r in sel),
         'failed_exchanges': sum(r['failed_exchanges'] for r in sel),
         'latency_s': {'n': len(lat), 'mean': round(float(np.mean(lat)), 1) if lat else None, 'median': pct(lat, 50),
                       'p90': pct(lat, 90), 'max': max(lat) if lat else None,
                       'over_45': sum(x > 45 for x in lat)}}
    if all('asked' in r for r in sel):
        for k in ('asked', 'answered', 'in_time', 'late', 'timeouts', 'failed', 'gave_up', 'unanswered'):
            e[k] = sum(r[k] for r in sel)
        e['outcomes'] = {k: sum(r['llm_wait'].get(k, 0) for r in sel)
                         for k in (*IN_TIME, 'agree_late', 'stale', 'returning', 'failed', 'timeout', 'expired',
                                   'after_replan')}
        e['longest_hold_s'] = max(r['llm_wait'].get('longest_hold_s', 0) for r in sel)
    return e


def summarize(out):
    rows, missing = collect(out)
    summary = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': MODEL, 'llm': LLM, 'out': out,
               'missing': missing, 'groups': {}, 'runs': rows}
    log = RUNS / out / 'launches.json'
    summary['launches'] = json.loads(log.read_text(encoding='utf-8')) if log.exists() else []
    for g, spec in GROUPS.items():
        grp = [r for r in rows if r['group'] == g]
        if not grp:
            continue
        entry = {'mission': spec['mission'], 'scenarios': [f'{lv}-{sd}' for lv, sd in spec['scenarios']], 'arms': {},
                 'diffs': {}}
        by = {arm: {(r['level'], r['seed']): r for r in grp if r['arm'] == arm} for arm in spec['arms']}
        for arm in spec['arms']:
            if by[arm]:
                e = arm_summary(list(by[arm].values()))
                sel = list(by[arm].values())
                if spec['mission'] == 'M1':
                    e.update(strict=sum(r['success'] is True for r in sel), exactly_two=sum(r['exactly_two'] for r in sel),
                             more_than_two=sum(r['collected'] > 2 for r in sel),
                             outcomes_strict={o: sum(r['outcome'] == o for r in sel) for o in {r['outcome'] for r in sel}})
                if spec['mission'] == 'M4':
                    pen = [r for r in sel if r['had_penalty']]
                    e.update(with_penalty=len(pen), strict=sum(r['success'] is True for r in pen),
                             home_after_penalty=sum(r['home_after_penalty'] for r in pen),
                             collected_after=sum(r['collected_after'] or 0 for r in pen),
                             returned_without_penalty=sum(r['returned'] for r in sel if not r['had_penalty']),
                             outcomes_strict={o: sum(r['outcome'] == o for r in sel) for o in {r['outcome'] for r in sel}})
                entry['arms'][arm] = e
        pairs = [(a, b) for a, b in (('rule', 'off'), ('leash', 'off'), ('rule', 'leash')) if by.get(a) and by.get(b)]
        for a, b in pairs:
            keys = sorted(set(by[a]) & set(by[b]))
            d = {'n': len(keys)}
            for metric in ('score', 'hazard_hits', 'idle_s', 'returned', 'collected'):
                diffs = [float(by[a][k][metric]) - float(by[b][k][metric]) for k in keys]
                d[metric] = boot(diffs)
                if metric == 'score':
                    d['score_higher_in'] = sum(x > 0 for x in diffs)
                    d['score_by_scenario'] = {f'{k[0]}-{k[1]}': round(x, 2) for k, x in zip(keys, diffs)}
            entry['diffs'][f'{a}-{b}'] = d
        summary['groups'][g] = entry
    lat = [x for r in rows for x, c in zip(r['latency_s'], [0] * len(r['latency_s']))]
    summary['latency_s_all'] = {'n': len(lat), 'mean': round(float(np.mean(lat)), 1) if lat else None,
                                'median': pct(lat, 50), 'p90': pct(lat, 90), 'max': max(lat) if lat else None,
                                'over_45_share': round(sum(x > 45 for x in lat) / len(lat), 3) if lat else None}
    summary['network_calls'] = sum(x['network_calls'] for x in summary['launches'] if not x['cache_only'])
    summary['cache_counter'] = ReplyCache().calls().get(MODEL, 0)
    return summary


def _ci(b, digits=1, sign=False):
    if b is None:
        return '—'
    f = f'{{:{"+" if sign else ""}.{digits}f}}'
    return f'{f.format(b[0])} [{f.format(b[1])}; {f.format(b[2])}]'.replace('.', ',')


def _num(x, digits=1):
    return '—' if x is None else f'{x:.{digits}f}'.replace('.', ',')


def tables(summary):
    lines = [f"# R16b: таблицы (настоящая модель {summary['model']})", '',
             f"Собрано {summary['created']} по записям runs/{summary['out']}; интервалы — 95%, бутстреп по сценариям.", '']
    for g, entry in summary['groups'].items():
        n = len(entry['scenarios'])
        lines += [f"## {g}: {entry['mission'] or 'обычная миссия'}, сценарии {', '.join(entry['scenarios'])}", '',
                  '| Режим | Прогонов | Счёт | Вернулся | Штрафы за зоны | Собрано | Простой, с | Длина прогона, с '
                  '| Запросов | Непринятых обменов | Ответ: среднее / медиана / 90% / самый долгий, с | Дольше 45 с |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for arm, e in entry['arms'].items():
            la = e['latency_s']
            lines.append(f"| {ARM_NAMES[arm]} | {e['n']} | {_ci(e['score'])} | {e['returned']} из {e['n']} | "
                         f"{_num(e['hazard_hits_mean'], 2)} | {_num(e['collected_mean'], 2)} | {_ci(e['idle_s'], 0)} | "
                         f"{_num(e['time_s_mean'], 0)} | {e['requests']} | {e['failed_exchanges']} | {_num(la['mean'])} / "
                         f"{_num(la['median'])} / {_num(la['p90'])} / {_num(la['max'])} | {la['over_45']} из {la['n']} |")
        waits = {a: e for a, e in entry['arms'].items() if 'asked' in e}
        if waits:
            lines += ['', '| Режим | Вопросов | Ответов | В срок | из них: подтвердил / то же / остаток в очередь / смена '
                      'плана | Поздних (план сменился / устарел / уже домой) | Негодных | Сроков истекло | Просроченных '
                      'ответов | Прогонов с отказом от модели | Пришло после нового решения правила | Самая долгая '
                      'стоянка по вопросу, с |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for arm, e in waits.items():
                o = e['outcomes']
                share = f"{100 * e['in_time'] / e['asked']:.0f}%" if e['asked'] else '—'
                lines.append(f"| {ARM_NAMES[arm]} | {e['asked']} | {e['answered']} | {e['in_time']} ({share} вопросов) | "
                             f"{o['agree']} / {o['same']} / {o['tail']} / {o['switched']} | {e['late']} "
                             f"({o['agree_late']} / {o['stale']} / {o['returning']}) | {e['failed']} | {e['timeouts']} | "
                             f"{o['expired']} | {e['gave_up']} | {o['after_replan']} | {_num(e['longest_hold_s'])} |")
        if entry['mission'] == 'M1':
            lines += ['', '| Режим | Строгий критерий R13 | По существу: ровно два и вернулся | Собрал больше двух | '
                      'Исходы по строгому критерию |', '|---|---|---|---|---|']
            for arm, e in entry['arms'].items():
                lines.append(f"| {ARM_NAMES[arm]} | {e['strict']} из {e['n']} | {e['exactly_two']} из {e['n']} | "
                             f"{e['more_than_two']} | {e['outcomes_strict']} |")
        if entry['mission'] == 'M4':
            lines += ['', '| Режим | Прогонов со штрафом | Строгий критерий R13 (из прогонов со штрафом) | По существу: '
                      'после штрафа домой без сбора | Собрано после штрафа, всего | Исходы по строгому критерию |',
                      '|---|---|---|---|---|---|']
            for arm, e in entry['arms'].items():
                lines.append(f"| {ARM_NAMES[arm]} | {e['with_penalty']} из {e['n']} | {e['strict']} из "
                             f"{e['with_penalty']} | {e['home_after_penalty']} из {e['with_penalty']} | "
                             f"{e['collected_after']} | {e['outcomes_strict']} |")
        if entry['diffs']:
            lines += ['', f'| Парная разность ({n} сценариев) | Счёт | Выше в скольких сценариях | Штрафы за зоны | '
                      'Простой, с | Возврат | Собрано |', '|---|---|---|---|---|---|---|']
            for name, d in entry['diffs'].items():
                a, b = name.split('-')
                lines.append(f"| `{a}` − `{b}` | {_ci(d['score'], sign=True)} | {d['score_higher_in']} из {d['n']} | "
                             f"{_ci(d['hazard_hits'], 2, True)} | {_ci(d['idle_s'], 0, True)} | "
                             f"{_ci(d['returned'], 2, True)} | {_ci(d['collected'], 2, True)} |")
            lines += ['', '| Сценарий | ' + ' | '.join(f"`{a}`" for a in entry['arms']) + ' |',
                      '|---|' + '---|' * len(entry['arms'])]
            for sc in entry['scenarios']:
                cells = []
                for arm in entry['arms']:
                    r = next((r for r in summary['runs'] if r['group'] == g and r['arm'] == arm
                              and f"{r['level']}-{r['seed']}" == sc), None)
                    cells.append('—' if r is None else f"{_num(r['score'])}; собрано {r['collected']}; "
                                 f"{'вернулся' if r['returned'] else 'не вернулся'}; простой {_num(r['idle_s'], 0)} с")
                lines.append(f'| {sc} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    la = summary['latency_s_all']
    lines += [f"Время ответа по всем обменам ({la['n']}): среднее {_num(la['mean'])} с, медиана {_num(la['median'])} с, "
              f"90-й процентиль {_num(la['p90'])} с, самый долгий {_num(la['max'])} с, дольше 45 с — "
              f"{_num(100 * (la['over_45_share'] or 0))}%.",
              f"Настоящих обращений к сети по запускам: {summary['network_calls']}.",
              f"Не посчитано: {', '.join(summary['missing']) or 'нет'}.", '']
    return lines


def cmd_report(args):
    summary = summarize(args.out)
    path = RUNS / args.out / 'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    text = '\n'.join(tables(summary))
    if args.out == OUT:
        (FINDINGS / 'R16b-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
        (FINDINGS / 'R16b-tables.md').write_text(text, encoding='utf-8')
    print(text)


def _comparable(trace):
    track = trace.get('track') or {}
    return {'result': {k: (trace.get('result') or {}).get(k) for k in RESULT_KEYS},
            'idle_s': (trace.get('result') or {}).get('idle_s'), 'llm_wait': (trace.get('result') or {}).get('llm_wait'),
            'events': trace.get('events'), 'plans': trace.get('plans'),
            'llm': [(ex.get('t'), ex.get('attempt'), ex.get('ok'), ex.get('response')) for ex in trace.get('llm') or []],
            'track': [track.get(k) for k in ('t', 'x', 'y', 'th', 'battery', 'mode')]}


def cmd_compare(args):
    same = total = 0
    report = {}
    for task in tasks_of(args.groups):
        a, b = record_path(args.out, task), record_path(args.other, task)
        if not b.exists():
            continue
        total += 1
        name = '/'.join(map(str, task))
        if not a.exists():
            report[name] = 'нет записи повтора'
            continue
        x, y = _comparable(load_trace(a)), _comparable(load_trace(b))
        diff = [k for k in x if x[k] != y[k]]
        same += not diff
        report[name] = 'same' if not diff else 'отличается: ' + ', '.join(diff)
    for name, verdict in report.items():
        if verdict != 'same':
            print(f'{name}: {verdict}')
    print(f'Совпало с runs/{args.other} до цифры: {same} из {total} прогонов')
    path = RUNS / args.out / 'compare.json'
    path.write_text(json.dumps({'same': same, 'total': total, 'cells': report}, ensure_ascii=False, indent=1),
                    encoding='utf-8')
    if args.other == OUT:
        (FINDINGS / 'R16b-replay.json').write_text(path.read_text(encoding='utf-8'), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('run', 'report', 'compare'))
    ap.add_argument('--groups', nargs='+', choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3, help='к модели — не больше трёх одновременно')
    ap.add_argument('--out', default=OUT, help='папка в runs/')
    ap.add_argument('--with', dest='other', default=OUT, help='compare: с какой папкой сверять')
    ap.add_argument('--cache-only', action='store_true', help='ответы только из кэша, сеть не трогается')
    ap.add_argument('--force', action='store_true', help='пересчитать и те прогоны, записи которых уже есть')
    args = ap.parse_args()
    {'run': cmd_run, 'report': cmd_report, 'compare': cmd_compare}[args.what](args)


if __name__ == '__main__':
    main()
