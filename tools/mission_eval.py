"""R13: выполняет ли робот миссию, заданную другими словами. Правило против языковой модели.

Варианты:
  rule         — правило (HeuristicPlanner): текст миссии не читает;
  llm          — модель в обвязке как есть: её спрашивают по обычным поводам и не чаще раза в 4 с,
                 на поводы внутри этих 4 с отвечает правило;
  llm_triggers — то же плюс поводы «столкновение» и «заряд у порога миссии» (флажок mission_triggers);
  llm_ask      — то же, что llm_triggers, и модель спрашивают по каждому поводу (llm_min_interval_s = 0);
  llm_ask_pen  — то же, что llm_ask, и в сводке состояния есть счётчик штрафов (state_penalties); нужен для M4.

План опыта — список EXPECTED: 156 ячеек «миссия × вариант × сценарий». Сводка всегда строится по нему:
ячейка без записи, с упавшим запуском или с записью от другой настройки показывается отдельно и входит
в знаменатель.

Повтор без сети (ответы и отказы модели — только из runs/_llm_cache; чего там нет — ошибка ячейки):
    ./px python tools/mission_eval.py --all --cache-only --out missions_strict --jobs 3
    ./px python tools/mission_eval.py --out missions_strict --compare missions     # сверка с исходными записями
    ./px python tools/mission_eval.py --out missions_strict --report               # только пересобрать сводку
Новый счёт по сети (настройки и ключ — в .env основного каталога; ответы и отказы кладутся в кэш):
    ./px python tools/mission_eval.py --variants rule llm llm_triggers llm_ask --jobs 3
    ./px python tools/mission_eval.py --missions M4 --levels hard --seeds 1004 1005 1007 --variants rule llm llm_ask
Записи — в runs/<--out>/<миссия>_<вариант>/<сценарий>.json.gz, сводка — runs/<--out>/summary.json, исход
последнего запуска каждой ячейки — runs/<--out>/status.json.
"""
import argparse
import json
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from did.llm import CacheMiss, llm_stats, load_env                       # noqa: E402
from did.llm_cache import ReplyCache                                     # noqa: E402
from did.mission_criteria import MISSIONS, verify_mission                # noqa: E402
from did.recorder import load_trace                                      # noqa: E402
from did.runner import RUNS, run_episode                                 # noqa: E402

EXPERIMENT = 'missions'
MODEL = 'qwen3.8-flash-next'
ENV_FILE = '/Users/a/MAI/DID/.env'
LLM = {'kind': 'http', 'model': MODEL, 'use_schema': True, 'min_tokens': 3000, 'cache': True}
VARIANTS = {
    'rule': ('adaptive', None, {}),
    'llm': ('adaptive_llm', LLM, {}),
    'llm_triggers': ('adaptive_llm', LLM, {'mission_triggers': True}),
    'llm_ask': ('adaptive_llm', LLM, {'mission_triggers': True, 'llm_min_interval_s': 0.0}),
    'llm_ask_pen': ('adaptive_llm', LLM, {'mission_triggers': True, 'llm_min_interval_s': 0.0,
                                          'state_penalties': True}),
}
MAIN_SCENARIOS = [f'{lvl}-{seed}' for lvl in ('medium', 'hard') for seed in (1001, 1002, 1003)]
M4_EXTRA_SEEDS = (1009, 1013, 1015, 1016, 1018, 1024)      # hard: правило получает здесь штраф
BASE_VARIANTS = ('rule', 'llm', 'llm_triggers', 'llm_ask')


def _expected():
    main = [(lvl, seed) for lvl in ('medium', 'hard') for seed in (1001, 1002, 1003)]
    cells = [(m, v, lvl, seed) for m in MISSIONS for v in BASE_VARIANTS for lvl, seed in main]
    cells += [('M4', 'llm_ask_pen', lvl, seed) for lvl, seed in main]
    cells += [('M4', v, 'hard', seed) for v in VARIANTS for seed in M4_EXTRA_SEEDS]
    return cells


EXPECTED = _expected()                 # план опыта: (миссия, вариант, уровень, номер сценария)
# Настройки агента, по которым запись узнаётся как запись своей ячейки.
SETTING_KEYS = ('mission', 'planner', 'mission_triggers', 'llm_min_interval_s', 'state_penalties')
# Что должно совпасть у повтора и у исходной записи (итог судьи, события, решения, обмены с моделью, путь).
RESULT_KEYS = ('t', 'battery', 'samples_collected', 'returned', 'score', 'collisions', 'false_collects',
               'hazard_hits', 'distance', 'reason')
_status_lock = threading.Lock()


def cell_id(task):
    m_id, variant, level, seed = task
    return f'{m_id}_{variant}/{level}-{seed}'


def cell_path(experiment, task):
    return RUNS / experiment / f'{cell_id(task)}.json.gz'


def read_status(experiment):
    try:
        data = json.loads((RUNS / experiment / 'status.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def mark_cell(experiment, task, status, **info):
    """Исход последнего запуска ячейки: running — начат, ok — запись сохранена, error — упал.

    По этой отметке сводка не примет запись прошлого запуска за результат нового, упавшего или оборванного.
    """
    with _status_lock:
        data = read_status(experiment)
        data[cell_id(task)] = {'status': status, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **info}
        path = RUNS / experiment / 'status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


def run_single(task, experiment=EXPERIMENT, cache_only=False):
    m_id, variant, level, seed = task
    agent, llm, cfg = VARIANTS[variant]
    if llm and cache_only:
        llm = {**llm, 'cache': 'only'}
    setup = {'model': llm['model'] if llm else None, 'cache': llm['cache'] if llm else None}
    mark_cell(experiment, task, 'running', **setup)
    t0 = time.time()
    try:
        res = run_episode(level, seed, agent, experiment=experiment, arm=f'{m_id}_{variant}',
                          config={'mission': MISSIONS[m_id]['text'], **cfg}, llm=llm, save=True, truth=True)
    except Exception as e:
        if not isinstance(e, CacheMiss):          # промах кэша понятен из сообщения, стек не нужен
            traceback.print_exc()
        print(f'[{m_id}][{variant}][{level}-{seed}] ОШИБКА: {e}', file=sys.stderr, flush=True)
        mark_cell(experiment, task, 'error', error=f'{type(e).__name__}: {e}'[:400], **setup)
        return None
    mark_cell(experiment, task, 'ok', **setup)
    print(f"[{m_id}][{variant}][{level}-{seed}] счёт {res['metrics'].get('score')} ({time.time() - t0:.0f} с)",
          flush=True)
    return res['file']


def settings_mismatch(trace, task):
    """Чем запись отличается от настроек своей ячейки; пустая строка — запись своя."""
    from did.runner import make_agent
    m_id, variant, level, seed = task
    agent, _, cfg = VARIANTS[variant]
    want = make_agent(agent, {'mission': MISSIONS[m_id]['text'], **cfg})[1].to_dict()
    # Настройки, которой в старой записи ещё нет, тогда не существовало: это её значение по умолчанию.
    got = {**make_agent(agent)[1].to_dict(), **((trace.get('agent') or {}).get('config') or {})}
    diff = [f'{k}: в записи {got.get(k)!r}, нужно {want[k]!r}' for k in SETTING_KEYS if got.get(k) != want[k]]
    if (trace.get('agent') or {}).get('name') != want['name']:
        diff.append(f"агент: в записи {(trace.get('agent') or {}).get('name')!r}, нужен {want['name']!r}")
    sc = trace.get('scenario') or {}
    if (sc.get('level'), sc.get('seed')) != (level, seed):
        diff.append(f"сценарий: в записи {sc.get('level')}-{sc.get('seed')}")
    return '; '.join(diff)


def llm_counts(exchanges):
    """Обмены с моделью по видам. Запрос — первый ответ и его исправления; обмен — один вызов модели.

    first_ok — запрос принят с первой попытки, repaired — исправлен повтором, failed_requests — окончательный
    отказ (решение ушло правилу); failed_exchanges — непринятые обмены: rejected_exchanges — ответ пришёл, но
    не прошёл проверку, transport_failures — ответа нет (обрыв связи); cached — обмены, взятые из кэша.
    """
    stats = llm_stats(exchanges)
    bad = [ex for ex in exchanges if not ex.get('ok')]
    lost = sum(1 for ex in bad if ex.get('response') is None)
    return {'requests': stats['requests'], 'exchanges': stats['exchanges'], 'first_ok': stats['first_ok'],
            'repaired': stats['repaired'], 'failed_requests': stats['failed'], 'failed_exchanges': len(bad),
            'rejected_exchanges': len(bad) - lost, 'transport_failures': lost, 'cached': stats['cached']}


def read_run(path, task):
    """Строка сводки по записи прогона runs/<опыт>/<миссия>_<вариант>/<сценарий>.json.gz."""
    m_id, variant, level, seed = task
    trace = load_trace(path)
    result = trace['result']
    plans = trace.get('plans', [])
    check = verify_mission(m_id, trace.get('track', {}), trace.get('events', []), result, plans=plans,
                           modes=trace.get('modes'), truth=trace.get('truth'))
    exchanges = trace.get('llm', [])
    counts = llm_counts(exchanges)
    return {
        'success': check['success'], 'outcome': check['outcome'], 'details': check['details'],
        'score': result['score'], 'samples_collected': result['samples_collected'],
        'samples_total': result['samples_total'], 'returned': result['returned'],
        'battery': round(result['battery'], 1), 't_end': result.get('t'),
        'penalties': result['collisions'] + result['false_collects'] + result['hazard_hits'],
        # Кто принимал решения: llm — модель, heuristic — правило (у варианта с моделью это поводы внутри
        # llm_min_interval_s), fallback — правило вместо негодного ответа модели, rule — возврат, который решил агент.
        'plans_by_source': dict(Counter(p['source'] for p in plans)),
        'llm': counts,
        'llm_latency_s': round(llm_stats(exchanges)['latency_ms']['total'] / 1000.0, 1) if exchanges else 0.0,
        'plans': [{'t': p['t'], 'source': p['source'], 'trigger': p['trigger'], 'reasoning': p['reasoning'],
                   'subgoals': [s.get('target') and f"{s['type']} {s['target']} ({s['x']}; {s['y']})"
                                or (f"goto ({s['x']}; {s['y']})" if s['type'] == 'goto' else s['type'])
                                for s in p['subgoals']]} for p in plans],
    }


def read_cell(experiment, task, status):
    """Ячейка плана опыта: исход прогона или причина, по которой его нет.

    status: ok — есть своя запись; no_record — записи нет; error — последний запуск упал или оборван (старая
    запись, если она лежит на диске, не берётся); other_settings — запись сделана с другими настройками.
    """
    m_id, variant, level, seed = task
    cell = {'mission': m_id, 'variant': variant, 'scenario': f'{level}-{seed}', 'status': 'ok', 'note': '',
            'success': None, 'outcome': None}
    last = status.get(cell_id(task)) or {}
    if last.get('status') in ('error', 'running'):
        note = last.get('error') or 'запуск начат и не завершён'
        return {**cell, 'status': 'error', 'note': f"{note} ({last.get('time')})"}
    path = cell_path(experiment, task)
    if not path.is_file():
        return {**cell, 'status': 'no_record', 'note': 'записи прогона нет'}
    try:
        trace = load_trace(path)
        wrong = settings_mismatch(trace, task)
        if not wrong and VARIANTS[variant][1] and last.get('model') not in (None, MODEL):
            wrong = f"модель: запуск на {last['model']!r}, нужна {MODEL!r}"
        if wrong:
            return {**cell, 'status': 'other_settings', 'note': wrong}
        return {**cell, **read_run(path, task)}
    except Exception as e:                     # испорченный файл — тоже «нет результата», а не пропуск молча
        return {**cell, 'status': 'error', 'note': f'запись не читается: {type(e).__name__}: {e}'[:300]}


def _mean(values):
    return round(sum(values) / len(values), 2) if values else None


def _sum_counts(rows):
    total = Counter()
    for r in rows:
        total.update(r.get('llm') or {})
    return {k: total.get(k, 0) for k in ('requests', 'exchanges', 'first_ok', 'repaired', 'failed_requests',
                                         'failed_exchanges', 'rejected_exchanges', 'transport_failures', 'cached')}


def _tally(cells):
    """Счёт по ячейкам: все ожидаемые — в знаменателе, непосчитанные — отдельно."""
    done = [c for c in cells if c['status'] == 'ok']
    return {'expected': len(cells),
            'success': sum(c['success'] is True for c in done),
            'failed': sum(c['success'] is False for c in done),
            'unverified': sum(c['success'] is None for c in done),
            'missing': len(cells) - len(done),
            'outcomes': dict(Counter(c['outcome'] for c in done)),
            'not_counted': {c['scenario']: c['status'] for c in cells if c['status'] != 'ok'}}


def compile_summary(experiment=EXPERIMENT):
    status = read_status(experiment)
    cells = [read_cell(experiment, task, status) for task in EXPECTED]
    table = []
    for m_id in MISSIONS:
        for variant in VARIANTS:
            arm = [c for c in cells if c['mission'] == m_id and c['variant'] == variant]
            main = [c for c in arm if c['scenario'] in MAIN_SCENARIOS]
            if not main:
                continue
            done = [c for c in main if c['status'] == 'ok']
            entry = {
                'mission': m_id, 'variant': variant, **_tally(main),
                'score_mean': _mean([r['score'] for r in done]),
                'samples_mean': _mean([r['samples_collected'] for r in done]),
                'returned': sum(r['returned'] for r in done),
                'battery_mean': _mean([r['battery'] for r in done]),
                'llm': _sum_counts(done),
                'plans_by_source': dict(sum((Counter(r['plans_by_source']) for r in done), Counter())),
                'by_scenario': {c['scenario']: c['success'] if c['status'] == 'ok' else c['status'] for c in main},
            }
            if m_id == 'M4':            # по существу условие проверяется только там, где штраф был (все 12 сценариев)
                ok = [c for c in arm if c['status'] == 'ok']
                hit = [c for c in ok if c['details']['had_penalty']]
                calm = [c for c in ok if not c['details']['had_penalty']]
                entry['with_penalty'] = {
                    'expected': len(arm), 'missing': len(arm) - len(ok), 'runs': len(hit),
                    'success': sum(c['success'] is True for c in hit),
                    'unverified': sum(c['success'] is None for c in hit),
                    'outcomes': dict(Counter(c['outcome'] for c in hit)),
                    'scenarios': {c['scenario']: c['outcome'] for c in hit},
                    'no_penalty_runs': len(calm),
                    # Возврат по решению модели без единого штрафа при несобранных образцах: условие миссии
                    # для него не наступало.
                    'model_return_without_penalty': sorted(
                        c['scenario'] for c in calm if c['details']['return_source'] == 'llm'
                        and c['samples_collected'] < c['samples_total']),
                }
            table.append(entry)
    expected_files = {cell_path(experiment, task) for task in EXPECTED}
    extra = sorted(f'{p.parent.name}/{p.name[:-len(".json.gz")]}' for p in (RUNS / experiment).glob('M*_*/*.json.gz')
                   if p not in expected_files)
    done = [c for c in cells if c['status'] == 'ok']
    summary = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'experiment': experiment, 'model': MODEL,
        'main_scenarios': MAIN_SCENARIOS,
        'missions': {k: {'name': v['name'], 'text': v['text']} for k, v in MISSIONS.items()},
        'cells': {'expected': len(cells), **{k: sum(c['status'] == k for c in cells)
                                              for k in ('ok', 'no_record', 'error', 'other_settings')}},
        'unexpected_records': extra,           # записи на диске, которых нет в плане опыта: в счёт не идут
        'llm': _sum_counts(done),
        # Счётчик сетевых вызовов в кэше: растёт при каждом запуске с сетью, к этим записям не привязан.
        'cache_calls_counter': ReplyCache().calls().get(MODEL, 0),
        'table': table, 'runs': cells,
    }
    out = RUNS / experiment / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary, out


STATUS_NAMES = {'no_record': 'нет записи', 'error': 'ошибка', 'other_settings': 'запись от другой настройки'}


def table_lines(summary):
    lines = [f"{'миссия':<7}{'вариант':<14}{'выполнено':<11}{'нет':>4}{'не пров.':>10}{'не посчитано':>14}"
             f"{'счёт':>7}{'собрано':>9}{'вернулся':>10}{'заряд':>7}{'запросов':>10}"]
    for e in summary['table']:
        note = ''
        if 'with_penalty' in e:
            w = e['with_penalty']
            note = f"   со штрафом (12 сценариев): {w['success']} из {w['runs']}, без штрафа {w['no_penalty_runs']}"
        lines.append(f"{e['mission']:<7}{e['variant']:<14}{str(e['success']) + ' из ' + str(e['expected']):<11}"
                     f"{e['failed']:>4}{e['unverified']:>10}{e['missing']:>14}"
                     f"{str(e['score_mean']):>7}{str(e['samples_mean']):>9}{e['returned']:>10}"
                     f"{str(e['battery_mean']):>7}{e['llm']['requests']:>10}{note}")
    bad = [c for c in summary['runs'] if c['status'] != 'ok']
    c = summary['cells']
    lines.append(f"Ячеек по плану {c['expected']}: с записью {c['ok']}, нет записи {c['no_record']}, "
                 f"ошибка {c['error']}, запись от другой настройки {c['other_settings']}")
    for cell in bad[:40]:
        lines.append(f"  {cell['mission']}_{cell['variant']}/{cell['scenario']}: {STATUS_NAMES[cell['status']]} — "
                     f"{cell['note']}")
    if len(bad) > 40:
        lines.append(f'  … и ещё {len(bad) - 40}')
    if summary['unexpected_records']:
        lines.append(f"Записей вне плана опыта (в счёт не идут): {len(summary['unexpected_records'])}")
    n = summary['llm']
    lines.append(f"Модель: запросов {n['requests']}, попыток (обменов) {n['exchanges']}; принято с первой попытки "
                 f"{n['first_ok']}, исправлено повтором {n['repaired']}, окончательных отказов {n['failed_requests']}; "
                 f"непринятых обменов {n['failed_exchanges']} (ответ не прошёл проверку {n['rejected_exchanges']}, "
                 f"нет ответа {n['transport_failures']}); попаданий в кэш {n['cached']} из {n['exchanges']}")
    return lines


def _comparable(trace):
    track = trace.get('track') or {}
    return {
        'result': {k: (trace.get('result') or {}).get(k) for k in RESULT_KEYS},
        'events': trace.get('events'),
        'plans': trace.get('plans'),
        'llm': [(ex.get('t'), ex.get('attempt'), ex.get('ok'), ex.get('response'), ex.get('errors'))
                for ex in trace.get('llm') or []],
        'track': [track.get(k) for k in ('t', 'x', 'y', 'th', 'battery', 'mode')],
        'modes': trace.get('modes'),
    }


def compare_runs(experiment, other):
    """Сверка записей двух каталогов по плану опыта: {ячейка: 'same' | 'нет записи…' | 'отличается: поле'}."""
    out = {}
    for task in EXPECTED:
        a, b = cell_path(experiment, task), cell_path(other, task)
        if not a.is_file() or not b.is_file():
            out[cell_id(task)] = f"нет записи в {experiment if not a.is_file() else other}"
            continue
        if (read_status(experiment).get(cell_id(task)) or {}).get('status') in ('error', 'running'):
            out[cell_id(task)] = f'последний запуск в {experiment} не удался'
            continue
        x, y = _comparable(load_trace(a)), _comparable(load_trace(b))
        diff = [k for k in x if x[k] != y[k]]
        out[cell_id(task)] = 'same' if not diff else 'отличается: ' + ', '.join(diff)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=3, help='одновременных прогонов (к модели — не больше трёх)')
    ap.add_argument('--missions', nargs='+', default=list(MISSIONS), choices=list(MISSIONS))
    ap.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    ap.add_argument('--variants', nargs='+', default=['rule', 'llm'], choices=list(VARIANTS))
    ap.add_argument('--all', action='store_true', help='весь план опыта (156 ячеек) вместо выбора по флажкам')
    ap.add_argument('--cache-only', action='store_true',
                    help='только кэш ответов: сеть не трогается, запрос без сохранённого ответа — ошибка ячейки')
    ap.add_argument('--out', default=EXPERIMENT, help='каталог записей и сводки внутри runs/ (по умолчанию missions)')
    ap.add_argument('--report', action='store_true', help='не считать, только пересобрать сводку из записей')
    ap.add_argument('--compare', metavar='КАТАЛОГ', help='не считать: сверить записи --out с записями runs/КАТАЛОГ')
    args = ap.parse_args()

    if args.compare:
        res = compare_runs(args.out, args.compare)
        same = sum(v == 'same' for v in res.values())
        for cell, verdict in res.items():
            if verdict != 'same':
                print(f'{cell}: {verdict}')
        print(f'Совпало с runs/{args.compare}: {same} из {len(res)} ячеек плана')
        return 0 if same == len(res) else 1
    if not args.report:
        if not args.cache_only:
            load_env(ENV_FILE)
        tasks = EXPECTED if args.all else [(m, v, lvl, seed) for v in VARIANTS if v in args.variants
                                           for m in args.missions for lvl in args.levels for seed in args.seeds]
        print(f"Прогонов: {len(tasks)}, одновременно: {args.jobs}, каталог runs/{args.out}"
              f"{', только кэш (сеть не используется)' if args.cache_only else ''}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, min(3, args.jobs))) as pool:
            done = list(pool.map(lambda task: run_single(task, args.out, args.cache_only), tasks))
        print(f'Готово за {time.time() - t0:.0f} с, сбоев: {done.count(None)}', flush=True)
    summary, out = compile_summary(args.out)
    print('\n'.join(table_lines(summary)))
    print(f'Сводка: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
