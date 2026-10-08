#!/usr/bin/env python3
"""L3b: модель выдвигает гипотезы сама — без списка объяснений, вероятностей и предсказаний, которые готовит код.

Берутся расследования исследователя на коде (записи runs/L3a/code). Робот ехал по коду; модель отвечает
«в тени» на те же случаи и на прогон не влияет, поэтому сравнение моделей и сравнение с расчётом — парные.

  шаг 1 — по странности, состоянию робота и перечню доступных манёвров: 2–5 гипотез и первый опыт;
  шаг 2 — в том же разговоре: результаты опытов, которые провёл робот, и нормы; вывод о причине;
  разметка — отдельный запрос (всегда qwen3.8-flash-next): к какой категории относится каждая гипотеза и вывод.

    ./px python tools/l3_open.py                                   # три модели, все расследования
    ./px python tools/l3_open.py --models DeepSeek-V4-Flash
    ./px python tools/l3_open.py --source L3a_dev --out L3b_dev    # отладка подсказок на сценариях 1–80
    ./px python tools/l3_open.py --cache-only                      # повтор без сети
    ./px python tools/l3_open.py --report
Ответы по случаям — runs/<--out>/<модель>/<сценарий>_<расследование>.json, сводка — runs/<--out>/summary.json.
"""
import argparse
import json
import math
import time
from collections import Counter

from l3_common import (ENV_FILE, MAIN, MODELS, RUNS, client, exchange_stats, fmt_share, paired_binary, read_json,
                       run_cells, share, slug, write_json)

from did.llm import LLMError, _obj, extract_json, load_env, request_json
from did.recorder import load_trace

EXPERIMENT = 'L3b'

WORLD = """Робот TurtleBot3 ездит по закрытой арене со столбами, ищет скрытые образцы и должен вернуться на базу
до разрядки батареи. Датчик образцов показывает близость к ближайшему несобранному образцу: число от 0 до 1 с
шумом, направления не даёт. Заряд тратится на путь, на повороты и на простой; гружёный робот тратит больше.
Часть пола покрыта «грунтами», где метр пути стоит дороже обычного. На арене есть опасные зоны: въезд
штрафуется, и после него возможны поломки и отказы. Среда может меняться по ходу прогона, об изменениях
роботу не сообщают: он должен заметить их сам по расходу батареи, штрафам и поведению датчика."""

STAGE1 = f"""# Роль
Ты — исследователь на борту автономного робота. Робот заметил странность: измерения не сходятся с его
моделью мира. Твоя задача — выдвинуть объяснения и выбрать опыт, который их различит.

# Мир
{WORLD}

# Вход (JSON в сообщении пользователя)
- `anomaly` — странность: `text`, место `x`, `y`, значения `observed` и `expected`, `unit`.
- `state` — что известно о роботе в этот момент.
- `norms` — обычные значения по модели робота.
- `maneuvers` — что робот может сделать прямо сейчас, чтобы проверить объяснения: `id`, `name`,
  `cost` (заряд), `unit` (что измеряется). Список может быть пустым.

# Что вернуть
1. `hypotheses` — от 2 до 5 разных объяснений странности, от самого правдоподобного к менее
   правдоподобному. `statement` — в чём причина, одной фразой; `predicts` — что покажут измерения,
   если причина в этом. Готового списка причин нет: думай сам.
2. `first_test` — `id` манёвра из `maneuvers`, который стоит сделать первым, или "none", если
   манёвров нет или ни один не поможет.
3. `why` — почему этот опыт: какие объяснения он разведёт. До 300 символов.
Пиши по-русски. Числа — только из входа.

# Формат ответа
Только один JSON-объект, без Markdown и без текста вне JSON:
{{"hypotheses": [{{"statement": строка, "predicts": строка}}, ...], "first_test": "id" или "none", "why": строка}}"""

STAGE2 = """Робот провёл опыты. Результаты:
{results}
Сделай вывод. Верни один JSON-объект:
{{"verdict": номер твоей гипотезы из прошлого ответа (с 1), которую измерения подтверждают, или 0, если ни одна
не подтверждена или данных мало,
 "cause": причина одной фразой — как ты её понимаешь по измерениям,
 "enough": true, если причина установлена, false — если данных недостаточно,
 "why": на какие числа опираешься, до 300 символов}}"""

# Категории для разметки. Первые — объяснения из списка кода, остальные в коде отсутствуют.
CATEGORIES = {
    'energy': {
        'soil': 'пол в этом месте дороже обычного: грунт, покрытие, зона с повышенным расходом (в том числе '
                'сменившийся или ещё не нанесённый на карту)',
        'leak': 'батарея теряет заряд сама по себе: утечка, поломка или сбой батареи/питания, в том числе после '
                'штрафа или опасной зоны; расход идёт и когда робот стоит',
        'turn': 'заряд ушёл на повороты, развороты, манёвры',
        'load': 'расход вырос из-за груза: робот везёт образцы',
        'model': 'ошибка самой модели или оценки робота: неверный прогноз, калибровка, шум измерения заряда, '
                 'погрешность одометрии',
        'none': 'ничего не случилось: батарея и пол в порядке, случайное отклонение',
        'other': 'другое: не подходит ни к одной категории выше',
    },
    'sensor': {
        'noise': 'датчик образцов стал шуметь сильнее: вырос разброс показаний',
        'stuck': 'датчик залип, завис: повторяет одно и то же значение',
        'bias': 'показания датчика систематически занижены или смещены',
        'ok': 'датчик исправен: показание изменилось по естественной причине (образец далеко или уже собран, '
              'робот двигался, обычный шум)',
        'other': 'другое: не подходит ни к одной категории выше',
    },
}
CATEGORIES['fault'] = CATEGORIES['energy']
IN_CODE = {'energy': ('soil', 'leak', 'turn'), 'fault': ('leak', 'none'), 'sensor': ('noise', 'stuck', 'bias', 'ok')}

JUDGE = """Ты — разметчик. Дан список утверждений о причине странности в поведении робота. Каждому
утверждению назначь ровно одну категорию из списка. Не оценивай, верно ли утверждение: только к какой категории
оно относится по смыслу. Если утверждение называет две причины сразу — выбери ту, что названа главной.

# Категории
{categories}

# Формат ответа
Только один JSON-объект: {{"labels": ["категория", ...]}} — по одной на утверждение, в том же порядке."""

NORMS = {
    'rest': 'обычный расход на месте — около 0.01 ед/с',
    'straight': '1.0 — расход как на обычном полу',
    'spin': 'обычная цена поворота — около 0.12 ед/рад',
    'listen_std': 'обычный разброс показаний исправного датчика — около 0.05; ровно 0 — показания одинаковы '
                  'до последнего знака',
    'listen_shift': '0 — среднее показание не изменилось относительно прежнего; отрицательное — упало',
    'onset': 'положительное число — показания после штрафа лучше объясняет версия «датчик занижает», '
             'отрицательное — версия «датчик исправен»',
    'at_sample': 'исправный датчик рядом со взятым образцом показывает около 0.93',
    'at_sample_std': 'обычный разброс показаний рядом с образцом — около 0.05',
}


# --- случаи из записей ---------------------------------------------------------------------------

def _at(track, t, key):
    ts = track['t']
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t)) if ts else None
    return track[key][i] if i is not None else None


def build_cases(source):
    """Расследования из записей runs/<source>/code: всё, что видел робот, без объяснений и предсказаний кода."""
    cases = []
    for path in sorted((RUNS / source / 'code').glob('*.json.gz')):
        tr = load_trace(path)
        scenario = path.name[:-len('.json.gz')]
        track, events = tr['track'], tr.get('events') or []
        for q in tr.get('inquiries') or []:
            if not q.get('conclusion'):
                continue
            t = q['t_open']
            carried = sum(1 for e in events if e['type'] == 'sample_collected' and e['t'] <= t)
            hits = [e['t'] for e in events if e['type'] != 'sample_collected' and e['t'] <= t]
            window = [(tt, th, x, y) for tt, th, x, y in zip(track['t'], track['th'], track['x'], track['y'])
                      if t - 5.0 <= tt <= t]
            turned = sum(abs((b[1] - a[1] + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(window, window[1:]))
            moved = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(window, window[1:]))
            state = {'time_s': t, 'battery': _at(track, t, 'battery'), 'carried_samples': carried,
                     'last_penalty_s_ago': round(t - hits[-1], 1) if hits else None,
                     'last_5s': {'moved_m': round(moved, 2), 'turned_rad': round(turned, 2)}}
            done = sorted([x for x in q['tests'] if x.get('measured')], key=lambda x: x['measured']['t'])
            cases.append({
                'key': f"{scenario}_{q['id']}", 'scenario': scenario, 'topic': q['topic'],
                'anomaly': {k: v for k, v in q['anomaly'].items() if k != 'trigger'}, 'state': state,
                'norms': {'drain_per_m': round(2.5 * (1 + 0.05 * carried), 2), 'drain_idle_per_s': 0.01,
                          'drain_per_rad': 0.12, 'sensor_noise': 0.05},
                'maneuvers': [{'id': x['id'], 'name': x['name'], 'cost': x['cost'], 'unit': x['unit']}
                              for x in q['tests']],
                'results': [{'id': x['id'], 'name': x['name'], 'value': x['measured']['value'], 'unit': x['unit']}
                            for x in done],
                'truth': q.get('truth') or [], 'code': {'verdict': q.get('verdict'), 'best': q['conclusion']['best'],
                                                       'status': q['conclusion']['status'],
                                                       'first_test': done[0]['id'] if done else None,
                                                       'tests': [x['id'] for x in done]}})
    return cases


# --- запросы ---------------------------------------------------------------------------------------

def _parse(model_cls_check):
    def parse(text):
        try:
            obj = extract_json(text)
        except LLMError as e:
            return None, [str(e)]
        errors = model_cls_check(obj)
        return (None, errors) if errors else (obj, [])
    return parse


def stage1_schema(case):
    ids = [m['id'] for m in case['maneuvers']] + ['none']
    return _obj(hypotheses={'type': 'array', 'items': _obj(statement={'type': 'string'}, predicts={'type': 'string'})},
                first_test={'type': 'string', 'enum': ids}, why={'type': 'string'})


def check_stage1(case):
    ids = {m['id'] for m in case['maneuvers']} | {'none'}

    def check(obj):
        hyps = obj.get('hypotheses')
        errors = []
        if not isinstance(hyps, list) or not 2 <= len(hyps) <= 5:
            errors.append('hypotheses: нужно от 2 до 5 объяснений')
        elif any(not isinstance(h, dict) or not str(h.get('statement') or '').strip() for h in hyps):
            errors.append('hypotheses: у каждого объяснения нужен непустой statement')
        if obj.get('first_test') not in ids:
            errors.append(f"first_test: нужно одно из: {', '.join(sorted(ids))}")
        return errors
    return check


STAGE2_SCHEMA = _obj(verdict={'type': 'integer'}, cause={'type': 'string'}, enough={'type': 'boolean'},
                     why={'type': 'string'})


def check_stage2(n):
    def check(obj):
        errors = []
        if not isinstance(obj.get('verdict'), int) or isinstance(obj.get('verdict'), bool) \
                or not 0 <= obj['verdict'] <= n:
            errors.append(f'verdict: нужно целое от 0 до {n}')
        if not isinstance(obj.get('enough'), bool):
            errors.append('enough: нужно true или false')
        if not str(obj.get('cause') or '').strip():
            errors.append('cause: нужна непустая строка')
        return errors
    return check


def ask_judge(judge, topic, statements):
    """Категории утверждений: список той же длины или None."""
    cats = CATEGORIES[topic]
    system = JUDGE.format(categories='\n'.join(f'- `{k}` — {v}' for k, v in cats.items()))
    user = 'Утверждения:\n' + '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(statements)) + '\nВерни разметку: один JSON-объект.'
    schema = _obj(labels={'type': 'array', 'items': {'type': 'string', 'enum': list(cats)}})

    def check(obj):
        labels = obj.get('labels')
        if not isinstance(labels, list) or len(labels) != len(statements):
            return [f'labels: нужно ровно {len(statements)} категорий, по одной на утверждение']
        bad = [x for x in labels if x not in cats]
        return [f"labels: нет категории {bad[0]!r}; есть: {', '.join(cats)}"] if bad else []
    obj, error, exchanges, _ = request_json(judge, [{'role': 'system', 'content': system},
                                                    {'role': 'user', 'content': user}], _parse(check), schema,
                                            role='judge', noun='ответ')
    return (obj['labels'] if obj else None), error, exchanges


def run_case(case, model, llm, judge, out_dir):
    path = out_dir / slug(model) / f"{case['key']}.json"
    payload = {k: case[k] for k in ('anomaly', 'state', 'norms', 'maneuvers')}
    messages = [{'role': 'system', 'content': STAGE1},
                {'role': 'user', 'content': 'Вход:\n' + json.dumps(payload, ensure_ascii=False)
                 + '\nВерни гипотезы и первый опыт: один JSON-объект.'}]
    rec = {'key': case['key'], 'model': model, 'stage1': None, 'stage2': None, 'labels': None, 'cause_label': None,
           'errors': {}, 'exchanges': []}
    s1, error, exchanges, _ = request_json(llm, messages, _parse(check_stage1(case)), stage1_schema(case),
                                           role='hypotheses', noun='ответ')
    rec['exchanges'] += exchanges
    if s1 is None:
        rec['errors']['stage1'] = error
        write_json(path, rec)
        return rec
    rec['stage1'] = s1
    statements = [str(h['statement']) for h in s1['hypotheses']]
    if case['results']:
        results = '\n'.join(f"- {r['name']}: {r['value']:g} {r['unit']} ({NORMS.get(r['id'], '')})"
                            for r in case['results'])
        talk = messages + [{'role': 'assistant', 'content': json.dumps(s1, ensure_ascii=False)},
                           {'role': 'user', 'content': STAGE2.format(results=results)}]
        s2, error, exchanges, _ = request_json(llm, talk, _parse(check_stage2(len(statements))), STAGE2_SCHEMA,
                                               role='verdict', noun='ответ')
        rec['exchanges'] += exchanges
        if s2 is None:
            rec['errors']['stage2'] = error
        else:
            rec['stage2'] = s2
            statements = statements + [str(s2['cause'])]
    labels, error, exchanges = ask_judge(judge, case['topic'], statements)
    rec['judge_exchanges'] = exchanges
    if labels is None:
        rec['errors']['judge'] = error
    else:
        n = len(s1['hypotheses'])
        rec['labels'] = labels[:n]
        rec['cause_label'] = labels[n] if len(labels) > n else None
    write_json(path, rec)
    return rec


# --- сводка ----------------------------------------------------------------------------------------

def score_case(case, rec):
    """Исходы по одному случаю: None — проверить нечем или ответа нет."""
    truth = set(case['truth'])
    out = {'answered': rec.get('stage1') is not None, 'recall': None, 'top1': None, 'n_hyp': None, 'novel': None,
           'first_same': None, 'verdict': None, 'code_right': None, 'model_right': None}
    labels = rec.get('labels')
    if labels and truth:
        out['recall'] = bool(set(labels) & truth)
        out['top1'] = labels[0] in truth
    if labels:
        out['n_hyp'] = len(labels)
        out['novel'] = sum(lb not in IN_CODE[case['topic']] for lb in labels)
    if rec.get('stage1') and len(case['maneuvers']) >= 2 and case['code']['first_test']:
        out['first_same'] = rec['stage1']['first_test'] == case['code']['first_test']
    if case['results'] and truth:
        out['code_right'] = case['code']['verdict'] in ('correct', 'partial')
        s2 = rec.get('stage2')
        if s2 is None or rec.get('cause_label') is None:
            out['verdict'] = 'no_answer'
            out['model_right'] = False
        elif not s2['enough']:
            out['verdict'] = 'insufficient'
            out['model_right'] = False
        else:
            out['model_right'] = rec['cause_label'] in truth
            out['verdict'] = 'right' if out['model_right'] else 'wrong'
    return out


def compile_summary(out, source, models):
    cases = build_cases(source)
    table, rows = {}, []
    for model in models:
        scored = {}
        recs = {}
        for case in cases:
            rec = read_json(RUNS / out / slug(model) / f"{case['key']}.json")
            if rec:
                recs[case['key']] = rec
                scored[case['key']] = score_case(case, rec)
        if not scored:
            continue
        by = {c['key']: c for c in cases}
        vals = list(scored.values())
        pick = lambda f: [v[f] for v in vals if v[f] is not None]        # noqa: E731
        hyps = [lb for k, r in recs.items() for lb in (r.get('labels') or [])]
        verdicts = Counter(v['verdict'] for v in vals if v['verdict'])
        e = {
            'cases': len(cases), 'done': len(scored), 'answered': share(sum(v['answered'] for v in vals), len(vals)),
            'hypotheses_per_case': round(sum(pick('n_hyp')) / max(1, len(pick('n_hyp'))), 2),
            'truth_among_hypotheses': share(sum(pick('recall')), len(pick('recall'))),
            'truth_first': share(sum(pick('top1')), len(pick('top1'))),
            'labels': dict(Counter(hyps)),
            'not_in_code_list': share(sum(pick('novel')), len(hyps)),
            'first_test_same_as_code': share(sum(pick('first_same')), len(pick('first_same'))),
            'verdict_right': share(verdicts['right'], sum(verdicts.values())),
            'verdict_outcomes': dict(verdicts),
            'code_right_same_cases': share(sum(pick('code_right')), len(pick('code_right'))),
            'code_minus_model': paired_binary({k: v['code_right'] for k, v in scored.items()},
                                              {k: v['model_right'] for k, v in scored.items()}),
            'by_topic': {}, 'by_truth': {},
            'llm': exchange_stats([ex for r in recs.values() for ex in r['exchanges']]),
            'judge': exchange_stats([ex for r in recs.values() for ex in r.get('judge_exchanges') or []]),
        }
        for topic in ('energy', 'fault', 'sensor'):
            sub = [v for k, v in scored.items() if by[k]['topic'] == topic]
            rc = [v['recall'] for v in sub if v['recall'] is not None]
            mr = [v['model_right'] for v in sub if v['model_right'] is not None]
            cr = [v['code_right'] for v in sub if v['code_right'] is not None]
            e['by_topic'][topic] = {'n': len(sub), 'truth_among_hypotheses': share(sum(rc), len(rc)),
                                    'verdict_right': share(sum(mr), len(mr)), 'code_right': share(sum(cr), len(cr))}
        for k, v in scored.items():
            key = '+'.join(by[k]['truth']) or '—'
            cell = e['by_truth'].setdefault(key, {'n': 0, 'recall': 0, 'verdict_cases': 0, 'model_right': 0,
                                                  'code_right': 0})
            cell['n'] += 1
            cell['recall'] += bool(v['recall'])
            if v['model_right'] is not None:
                cell['verdict_cases'] += 1
                cell['model_right'] += v['model_right']
                cell['code_right'] += bool(v['code_right'])
        table[model] = e
        for k, v in scored.items():
            r = recs[k]
            rows.append({'model': model, 'key': k, 'topic': by[k]['topic'], 'truth': by[k]['truth'],
                         'code': by[k]['code'], **v,
                         'hypotheses': [h['statement'] for h in (r.get('stage1') or {}).get('hypotheses', [])],
                         'labels': r.get('labels'), 'first_test': (r.get('stage1') or {}).get('first_test'),
                         'cause': (r.get('stage2') or {}).get('cause'), 'cause_label': r.get('cause_label'),
                         'enough': (r.get('stage2') or {}).get('enough'), 'errors': r.get('errors')})
    # парное сравнение моделей с основной по выводу
    base = {r['key']: r['model_right'] for r in rows if r['model'] == MAIN}
    for model in table:
        if model != MAIN and base:
            table[model]['verdict_minus_main'] = paired_binary(
                {r['key']: r['model_right'] for r in rows if r['model'] == model}, base)
    summary = {'experiment': out, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'done', 'source': source,
               'judge_model': MAIN, 'arms': table, 'claims': [], 'errors': [], 'runs': rows}
    write_json(RUNS / out / 'summary.json', summary)
    return summary


def report(s):
    lines = []
    for model, e in s['arms'].items():
        d = e['code_minus_model']
        lines.append(
            f"{model}: случаев {e['done']} из {e['cases']}, ответ получен {fmt_share(e['answered'])}; гипотез на случай "
            f"{e['hypotheses_per_case']}\n   правда среди гипотез {fmt_share(e['truth_among_hypotheses'])}, первой "
            f"{fmt_share(e['truth_first'])}; вне списка кода {fmt_share(e['not_in_code_list'])}; категории {e['labels']}\n"
            f"   первый опыт как у расчёта {fmt_share(e['first_test_same_as_code'])}\n"
            f"   вывод верен {fmt_share(e['verdict_right'])} {e['verdict_outcomes']}; расчёт на тех же случаях "
            f"{fmt_share(e['code_right_same_cases'])}; код − модель: "
            + (f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}; {d['ci'][1]:+.3f}], только код {d['only_a']}, только модель "
               f"{d['only_b']}, p={d['sign_p']}" if d else '—')
            + f"\n   по темам {json.dumps(e['by_topic'], ensure_ascii=False)}"
            + f"\n   запросов {e['llm']['requests']}: годных сразу {e['llm']['first_ok_share']:.0%}, отказов "
              f"{e['llm']['failed']}, медиана {e['llm']['latency_s']['median']} с; разметчик: запросов "
              f"{e['judge']['requests']}, отказов {e['judge']['failed']}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--source', default='L3a', help='каталог с записями исследователя на коде (runs/<source>/code)')
    ap.add_argument('--out', default=EXPERIMENT)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if not args.report:
        if not args.cache_only:
            load_env(ENV_FILE)
        cases = build_cases(args.source)
        judge = client(MAIN, args.cache_only)
        clients = {m: client(m, args.cache_only) for m in args.models}
        cells = [(m, i) for m in args.models for i in range(len(cases))]
        print(f'Случаев {len(cases)}, моделей {len(args.models)}', flush=True)
        run_cells(cells, lambda c: bool(run_case(cases[c[1]], c[0], clients[c[0]], judge, RUNS / args.out)), args.jobs,
                  label=lambda c: f'{c[0]} {cases[c[1]]["key"]}')
    print(report(compile_summary(args.out, args.source, MODELS)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
