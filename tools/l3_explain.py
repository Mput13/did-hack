#!/usr/bin/env python3
"""L3e: модель объясняет завершённое расследование — не расходится ли её текст с числами расчёта.

Берутся завершённые расследования исследователя на коде (записи runs/L3a/code). На каждое — запрос роли explain
(did/llm_roles.py) и одно исправление, если проверка программы не прошла. Независимо от этой проверки:
  причина — отдельный запрос-разметчик (всегда qwen3.8-flash-next) определяет, какое объяснение текст называет
            итогом; оно сверяется с выводом расчёта (conclusion.best или «данных недостаточно»);
  числа   — каждое число текста ищется во входе; не найденное проверяется как разность, сумма, отношение или
            процент двух входных чисел; остальное — «числа нет во входе».

    ./px python tools/l3_explain.py                       # три модели
    ./px python tools/l3_explain.py --source L3a_dev --out L3e_dev
    ./px python tools/l3_explain.py --cache-only
    ./px python tools/l3_explain.py --report
Ответы — runs/<--out>/<модель>/<сценарий>_<расследование>.json, сводка — runs/<--out>/summary.json.
"""
import argparse
import json
import re
import time
from collections import Counter

from l3_common import (ENV_FILE, MAIN, MODELS, RUNS, client, exchange_stats, fmt_share, read_json, run_cells, share,
                       share_diff, slug, write_json)

from did.llm import LLMError, _obj, error_kind, extract_json, load_env, request_json
from did.llm_roles import explain
from did.recorder import load_trace

EXPERIMENT = 'L3e'
KEEP = ('id', 't_open', 't_close', 'topic', 'anomaly', 'alternatives', 'tests', 'conclusion', 'action', 'source',
        'critique', 'note')

JUDGE = """Ты — разметчик. Дан текст-объяснение расследования робота и список возможных объяснений с
идентификаторами. Определи, какое объяснение текст называет итоговой причиной странности.
- Если текст прямо говорит, что причина не установлена или данных недостаточно, — ответ "insufficient"
  (даже если он называет самое вероятное объяснение).
- Если текст называет причину установленной — её идентификатор из списка.
- Если понять нельзя или названа причина не из списка — "unclear".
Не оценивай, прав ли текст.

# Формат ответа
Только один JSON-объект: {"cause": "идентификатор" | "insufficient" | "unclear"}"""

_NUM = re.compile(r'(?<![\w.,])\d+(?:[.,]\d+)?')


def build_cases(source):
    cases = []
    for path in sorted((RUNS / source / 'code').glob('*.json.gz')):
        scenario = path.name[:-len('.json.gz')]
        for q in load_trace(path).get('inquiries') or []:
            if q.get('conclusion'):
                cases.append({'key': f"{scenario}_{q['id']}", 'inquiry': {k: q[k] for k in KEEP if k in q},
                              'truth': q.get('truth') or [], 'code_verdict': q.get('verdict')})
    return cases


# --- числа -------------------------------------------------------------------------------------------

def _numbers(text):
    return [(raw, float(raw.replace(',', '.'))) for raw in _NUM.findall(text)]


def _close(value, target, raw=None, relative=0.02):
    decimals = len(re.split('[.,]', raw)[1]) if raw and re.search('[.,]', raw) else 0
    tolerance = max(0.5 * 10 ** -decimals, relative * abs(target))
    return abs(value - target) <= tolerance + 1e-9


def _core(inquiry):
    """Числа, из которых текст вправе что-то вывести: странность, измерения и ожидания опытов."""
    anomaly = inquiry.get('anomaly') or {}
    out = [anomaly.get('observed'), anomaly.get('expected')]
    for t in inquiry.get('tests') or []:
        m = t.get('measured')
        out.append(m.get('value') if isinstance(m, dict) else m)
        out += [p.get('mean') for p in (t.get('predictions') or {}).values()]
    return sorted({float(v) for v in out if isinstance(v, (int, float)) and not isinstance(v, bool)})


def check_numbers(text, inquiry):
    """Числа текста: {'given': из входа, 'derived': получаются из двух входных, 'missing': нет во входе}.

    given — число есть во входе (с округлением и допуском 2%), вероятность — и в процентах; целые 1–5 — счёт
    опытов и объяснений. derived — разность, сумма, отношение или процент двух чисел из странности, измерений и
    ожиданий опытов, с точностью до округления.
    """
    pool = sorted({v for _, v in _numbers(json.dumps(inquiry, ensure_ascii=False))})
    pool += [round(v * 100, 1) for v in pool if 0 < v <= 1]            # вероятности в процентах
    core = _core(inquiry)
    out = {'given': [], 'derived': [], 'missing': []}
    for raw, v in _numbers(text):
        if any(_close(v, p, raw) for p in pool) or (v == int(v) and 1 <= v <= 5):
            out['given'].append(raw)
            continue
        derived = any(_close(v, x, raw, relative=0.0) for a in core for b in core if a != b
                      for x in (a - b, a + b, (a / b if b else None), (100 * (a - b) / b if b else None),
                                (100 * a / b if b else None)) if x is not None and x > 0)
        out['derived' if derived else 'missing'].append(raw)
    return out


# --- разметчик ---------------------------------------------------------------------------------------

def ask_cause(judge, text, inquiry):
    alts = {a['id']: a['statement'] for a in inquiry['alternatives']}
    options = [*alts, 'insufficient', 'unclear']
    user = ('Объяснения:\n' + '\n'.join(f'- `{k}` — {v}' for k, v in alts.items()) + f'\n\nТекст:\n{text}\n\n'
            'Верни один JSON-объект.')

    def parse(reply):
        try:
            obj = extract_json(reply)
        except LLMError as e:
            return None, [str(e)]
        return (obj, []) if obj.get('cause') in options else (None, [f"cause: нужно одно из: {', '.join(options)}"])
    obj, error, exchanges, _ = request_json(judge, [{'role': 'system', 'content': JUDGE}, {'role': 'user', 'content': user}],
                                            parse, _obj(cause={'type': 'string', 'enum': options}), role='judge',
                                            noun='ответ')
    return (obj['cause'] if obj else None), error, exchanges


def run_case(case, model, llm, judge, out_dir):
    inquiry = case['inquiry']
    text = explain(llm, inquiry)
    first = text.exchanges[0] if text.exchanges else {}
    rec = {'key': case['key'], 'model': model, 'source': text.source, 'error': text.error, 'text': str(text),
           'first_ok': bool(first.get('ok')), 'first_errors': first.get('errors') or [],
           'first_text': None, 'exchanges': list(text.exchanges)}
    try:
        rec['first_text'] = extract_json(first.get('response') or '').get('text')
    except LLMError:
        pass
    if text.source == 'llm':
        rec['numbers'] = check_numbers(str(text), inquiry)
        rec['cause'], rec['judge_error'], rec['judge_exchanges'] = ask_cause(judge, str(text), inquiry)
    write_json(out_dir / slug(model) / f"{case['key']}.json", rec)
    return rec


# --- сводка ------------------------------------------------------------------------------------------

def expected_cause(inquiry):
    c = inquiry['conclusion']
    return c['best'] if c['status'] == 'identified' else 'insufficient'


def compile_summary(out, source, models):
    cases = build_cases(source)
    table, rows = {}, []
    for model in models:
        recs = {c['key']: r for c in cases if (r := read_json(RUNS / out / slug(model) / f"{c['key']}.json"))}
        if not recs:
            continue
        by = {c['key']: c for c in cases}
        n = len(recs)
        accepted = {k: r for k, r in recs.items() if r['source'] == 'llm'}
        judged = {k: r for k, r in accepted.items() if r.get('cause')}
        same = {k: r['cause'] == expected_cause(by[k]['inquiry']) for k, r in judged.items()}
        missing = {k: r['numbers']['missing'] for k, r in accepted.items() if r['numbers']['missing']}
        bad = set(missing) | {k for k, ok in same.items() if not ok}
        errors = Counter(error_kind(e) if not e.startswith('text:') else e.split(' — ')[0][:60]
                         for r in recs.values() for e in r['first_errors'])
        e = {'cases': len(cases), 'done': n,
             'first_ok': share(sum(r['first_ok'] for r in recs.values()), n),
             'accepted': share(len(accepted), n),
             'fallback': n - len(accepted),
             'first_errors': dict(errors.most_common()),
             'cause_same_as_code': share(sum(same.values()), len(same)),
             'cause_mismatch': {k: {'text_says': judged[k]['cause'], 'code': expected_cause(by[k]['inquiry'])}
                                for k, ok in same.items() if not ok},
             'with_missing_number': share(len(missing), len(accepted)),
             'with_derived_number': share(sum(bool(r['numbers']['derived']) for r in accepted.values()), len(accepted)),
             'contradicts_or_invents': share(len(bad), len(accepted)),
             'missing_numbers': missing,
             'chars': round(sum(len(r['text']) for r in accepted.values()) / max(1, len(accepted))),
             'llm': exchange_stats([ex for r in recs.values() for ex in r['exchanges']]),
             'judge': exchange_stats([ex for r in recs.values() for ex in r.get('judge_exchanges') or []])}
        table[model] = e
        rows += [{'model': model, 'key': k, 'source': r['source'], 'first_ok': r['first_ok'],
                  'first_errors': r['first_errors'], 'text': r['text'], 'first_text': r.get('first_text'),
                  'cause': r.get('cause'), 'code_cause': expected_cause(by[k]['inquiry']),
                  'numbers': r.get('numbers'), 'error': r.get('error')} for k, r in recs.items()]
    for model, e in table.items():
        if model != MAIN and MAIN in table:
            a, b = e['first_ok'], table[MAIN]['first_ok']
            e['first_ok_minus_main'] = share_diff(a['k'], a['n'], b['k'], b['n'])
    summary = {'experiment': out, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'done', 'source': source,
               'judge_model': MAIN, 'arms': table, 'claims': [], 'errors': [], 'runs': rows}
    write_json(RUNS / out / 'summary.json', summary)
    return summary


def report(s):
    lines = []
    for model, e in s['arms'].items():
        lines.append(
            f"{model}: расследований {e['done']} из {e['cases']}; годных сразу {fmt_share(e['first_ok'])}, принято после "
            f"исправления {fmt_share(e['accepted'])}, шаблон вместо модели {e['fallback']}\n"
            f"   ошибки первого ответа: {e['first_errors']}\n"
            f"   причина как у расчёта {fmt_share(e['cause_same_as_code'])}; расхождения: {e['cause_mismatch']}\n"
            f"   текстов с числом не из входа {fmt_share(e['with_missing_number'])}: {e['missing_numbers']}\n"
            f"   с выведенным числом {fmt_share(e['with_derived_number'])}; расходится с расчётом или с числом не из "
            f"входа {fmt_share(e['contradicts_or_invents'])}\n"
            f"   запросов {e['llm']['requests']}, медиана ответа {e['llm']['latency_s']['median']} с, длина текста {e['chars']}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--source', default='L3a')
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
        cells = [(m, i) for i in range(len(cases)) for m in args.models]
        print(f'Расследований {len(cases)}, моделей {len(args.models)}', flush=True)
        run_cells(cells, lambda c: bool(run_case(cases[c[1]], c[0], clients[c[0]], judge, RUNS / args.out)), args.jobs,
                  label=lambda c: f'{c[0]} {cases[c[1]]["key"]}')
    print(report(compile_summary(args.out, args.source, MODELS)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
