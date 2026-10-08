"""Таблицы отчёта R3 из сводок прогонов — чтобы в отчёте не было чисел по памяти.

    ./px python tools/build_r3_report.py

Читает runs/E16/summary.json (имитатор) и runs/R3_real/summary.json (настоящая модель), пишет
research/findings/R3-tables.md (таблицы для отчёта) и research/findings/R3-results.json (снимок чисел:
папка runs/ в репозиторий не входит, а проверить отчёт по чему-то нужно).
"""
import json
from pathlib import Path

from did.metrics import paired, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'findings'
ARMS = ('rule', 'single', 'critic', 'vote', 'scored', 'scored_calc')
WAITS = ('wait15', 'wait24', 'wait29')       # один запрос к имитатору и ожидание 15, 24, 29 с после каждого решения
CHARACTERS = {'normal': 'обычный', 'timid': 'робкий', 'random': 'случайный', 'careless': 'беспечный'}
MODES = {'frozen': '(а) мир стоит', 'charged': '(б) ожидание стоит времени'}


def num(x, digits=2, sign=False):
    return f'{x:{"+" if sign else ""}.{digits}f}'.replace('.', ',')


def ci(d):
    return f"{num(d['mean'], sign=True)} [{num(d['ci'][0], sign=True)}; {num(d['ci'][1], sign=True)}]"


def table(head, rows):
    return ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def mock_tables(e16):
    runs = e16['runs']
    rows = []
    for cond, name in CHARACTERS.items():
        base = [r for r in runs if r['condition'] == cond and r['arm'] == 'single']
        for arm in ARMS + WAITS:
            rs = [r for r in runs if r['condition'] == cond and r['arm'] == arm]
            rows.append([name, arm, num(summarize(rs, 'score')['mean']),
                         '—' if arm == 'single' else ci(paired(rs, base, 'score')),
                         num(summarize(rs, 'samples_share')['mean']), num(summarize(rs, 'returned')['mean']),
                         num(summarize(rs, 'llm_calls')['mean'], 1), num(summarize(rs, 'time')['mean'], 0),
                         num(summarize(rs, 'penalties')['mean'], 1)])
    out = [f"Имитатор, опыт E16: {len(runs)} прогонов, ошибок {len(e16['errors'])}. В строке 40 сценариев "
           '(medium и hard, номера 1001–1020); Δ — парная разность счёта с single того же характера, 95% интервал.', '']
    out += table(['Характер имитатора', 'Способ', 'Счёт', 'Δ к single [95%]', 'Доля образцов', 'Доля возвратов',
                  'Обращений на прогон', 'Длительность прогона, с', 'Штрафов на прогон'], rows)
    by_level = []
    for cond, name in CHARACTERS.items():
        for level in ('medium', 'hard'):
            pick = lambda arm: [r for r in runs if (r['condition'], r['level'], r['arm']) == (cond, level, arm)]  # noqa: E731
            by_level.append([name, level] + [ci(paired(pick(arm), pick('single'), 'score'))
                                             for arm in ('critic', 'vote', 'scored', 'scored_calc')]
                            + [ci(paired(pick('scored'), pick('scored_calc'), 'score'))])
    out += ['', 'То же по уровням (по 20 сценариев): Δ счёта к single и разность scored − scored_calc.', '']
    out += table(['Характер', 'Уровень', 'critic', 'vote', 'scored', 'scored_calc', 'scored − scored_calc'], by_level)
    waits = []
    for level in ('medium', 'hard'):
        pick = lambda arm: [r for r in runs if (r['condition'], r['level'], r['arm']) == ('normal', level, arm)]   # noqa: E731
        for arm in ('single',) + WAITS:
            rs = pick(arm)
            waits.append([level, arm, num(summarize(rs, 'score')['mean']),
                          '—' if arm == 'single' else ci(paired(rs, pick('single'), 'score')),
                          f"{sum(r['metrics']['score'] < 0 for r in rs)}/{len(rs)}",
                          num(summarize(rs, 'samples_share')['mean']), num(summarize(rs, 'returned')['mean']),
                          num(summarize(rs, 'penalties')['mean'], 1), num(summarize(rs, 'time')['mean'], 0)])
    out += ['', 'Цена одного только ожидания (обычный характер: решения те же, что у правила; по 20 сценариев на уровень):', '']
    out += table(['Уровень', 'Вариант', 'Счёт', 'Δ к single [95%]', 'Прогонов со счётом ниже нуля', 'Доля образцов',
                  'Доля возвратов', 'Штрафов на прогон', 'Длительность прогона, с'], waits)
    return out


def real_tables(real):
    runs = real['runs']
    scen = [(s.rsplit('-', 1)[0], int(s.rsplit('-', 1)[1])) for s in real['scenarios']]
    wrap = lambda rs: [{'level': r['level'], 'seed': r['seed'], 'metrics': r} for r in rs]     # noqa: E731
    pick = lambda strategy, mode: sorted((r for r in runs if (r['strategy'], r['mode']) == (strategy, mode)),  # noqa: E731
                                         key=lambda r: scen.index((r['level'], r['seed'])))
    strategies = [s for s in ARMS if s != 'rule' and len(pick(s, 'frozen')) == len(scen)]
    rule = pick('rule', 'frozen')
    out = [f"Настоящая модель {real['model']}: шесть сценариев ({', '.join(real['scenarios'])}), "
           f"сводка от {real['created']}.", '']

    rows = []
    for mode, mode_name in MODES.items():
        base = pick('single', mode)
        if mode == 'frozen':
            rows.append([mode_name, 'правило (без модели)', num(sum(r['score'] for r in rule) / len(rule)),
                         ci(paired(wrap(rule), wrap(base), 'score')), f"{sum(r['collected'] for r in rule)}/{sum(r['total'] for r in rule)}",
                         f"{sum(r['returned'] for r in rule)}/{len(rule)}", '0', '0', '0', '—', '—'])
        for s in strategies:
            rs = pick(s, mode)
            if len(rs) < len(scen):
                continue
            n = sum(r['decisions'] for r in rs)
            rows.append([mode_name, s, num(sum(r['score'] for r in rs) / len(rs)),
                         '—' if s == 'single' else ci(paired(wrap(rs), wrap(base), 'score')),
                         f"{sum(r['collected'] for r in rs)}/{sum(r['total'] for r in rs)}",
                         f"{sum(r['returned'] for r in rs)}/{len(rs)}",
                         num(sum(len(r['exchanges']) for r in rs) / len(rs), 1),
                         num(sum(r['answer_s'] for r in rs) / len(rs), 0),
                         num(sum(r['decisions'] * r['llm_wait_s'] for r in rs) / len(rs), 0),
                         f"{sum(r['rule_matches'] for r in rs)}/{n} ({num(100 * sum(r['rule_matches'] for r in rs) / n, 0)}%)",
                         str(sum(r['fallbacks'] for r in rs))])
    out += ['Сводная таблица (среднее по шести сценариям; Δ — парная разность с single в том же режиме, '
            'бутстреп-интервал по шести парам — описательный):', '']
    out += table(['Режим времени', 'Способ', 'Счёт', 'Δ к single [95%]', 'Собрано', 'Вернулся', 'Обращений на прогон',
                  'Модель отвечала, с на прогон', 'Робот ждал, с на прогон', 'Первая подцель как у правила',
                  'Решило запасное правило'], rows)

    for mode, mode_name in MODES.items():
        cols = (['rule'] if mode == 'frozen' else []) + [s for s in strategies if len(pick(s, mode)) == len(scen)]
        data = {s: (rule if s == 'rule' else pick(s, mode)) for s in cols}
        rows = []
        for i, (level, seed) in enumerate(scen):
            cells = []
            for s in cols:
                r = data[s][i]
                cell = (f"{num(r['score'])} ({r['collected']}/{r['total']}{'' if r['returned'] else ', не вернулся'}"
                        f"{', зона ×' + str(r['hazard_hits']) if r.get('hazard_hits') else ''})")
                if s not in ('rule', 'single'):
                    cell += f" Δ {num(r['score'] - data['single'][i]['score'], sign=True)}"
                cells.append(cell)
            rows.append([f'{level}-{seed}'] + cells)
        out += ['', f'Режим {mode_name}: счёт по сценариям (в скобках собрано образцов и число штрафов за опасную зону; Δ — разность с single на том же сценарии).', '']
        out += table(['Сценарий'] + ['правило' if s == 'rule' else s for s in cols], rows)

    rows = []
    for mode, mode_name in MODES.items():
        for s in strategies:
            rs = pick(s, mode)
            if len(rs) < len(scen):
                continue
            ex = [e for r in rs for e in r['exchanges']]
            ok = sorted(e['latency_ms'] for e in ex if e['ok'])
            rows.append([mode_name, s, str(sum(r['decisions'] for r in rs)), str(len(ex)),
                         str(sum(not e['cached'] for e in ex)), str(sum(not e['ok'] for e in ex)),
                         num(ok[len(ok) // 2] / 1000, 1) if ok else '—', num(rs[0]['llm_wait_s'], 1),
                         num(sum(r['time_s'] for r in rs) / len(rs), 0)])
    out += ['', 'Обращения и время (сумма по шести сценариям; «мимо кэша» — ответ получен от сервера в последнем запуске, '
            'а не из сохранённых):', '']
    out += table(['Режим времени', 'Способ', 'Решений с моделью', 'Обращений', 'Из них мимо кэша', 'Негодных ответов',
                  'Медиана ответа, с', 'llm_wait_s, с на решение', 'Длительность прогона, с'], rows)

    notes = ['', 'Что делали добавленные шаги (сумма по шести сценариям):', '']
    for mode, mode_name in MODES.items():
        for s in strategies:
            rs = pick(s, mode)
            if len(rs) < len(scen):
                continue
            if s == 'single':
                notes.append(f"- {mode_name}, single: решений с моделью {sum(r['decisions'] for r in rs)}.")
            if s == 'critic':
                c = {k: sum(r['critic'][k] for r in rs) for k in rs[0]['critic']}
                notes.append(f"- {mode_name}, critic: отзывов {c['reviews']}, из них «исправить» {c['revise']}, "
                             f"исправлений получено {c['revised']}, первая подцель изменилась {c['changed']} раз.")
            if s == 'vote':
                notes.append(f"- {mode_name}, vote: голоса разошлись в {sum(r['vote']['split'] for r in rs)} решениях "
                             f"из {sum(r['decisions'] for r in rs)}.")
            if s in ('scored', 'scored_calc'):
                c = {k: sum(r['scored'][k] for r in rs) for k in rs[0]['scored']}
                text = (f"- {mode_name}, {s}: таблиц {c['tables']}, вариантов в среднем "
                        f"{num(c['proposals'] / max(1, c['tables']), 1)}; выбор правила был среди вариантов модели "
                        f"{c['rule_offered']} раз из {c['tables']}; расчёт выбрал не первый вариант модели "
                        f"{c['calc_differs_from_first']} раз")
                if s == 'scored':
                    text += (f"; модель выбирала из таблицы {c['asked']} раз, её выбор разошёлся с расчётом "
                             f"{c['model_differs_from_calc']} раз")
                notes.append(text + '.')
            miss = [f"{r['level']}-{r['seed']} t={m['t']}: {m['chosen']} вместо {m['rule']}" for r in rs for m in r['mismatches']]
            if miss:
                notes.append(f"  Расхождения с правилом ({s}, {mode}): " + '; '.join(miss) + '.')
    out += notes
    for a, b in (('scored', 'scored_calc'), ('critic', 'single'), ('vote', 'single')):
        for mode, mode_name in MODES.items():
            ra, rb = pick(a, mode), pick(b, mode)
            if len(ra) == len(rb) == len(scen):
                diffs = ', '.join(num(x['score'] - y['score'], sign=True) for x, y in zip(ra, rb))
                out.append(f'- {mode_name}, {a} − {b} по сценариям: {diffs}; среднее {ci(paired(wrap(ra), wrap(rb), "score"))}.')
    for s in strategies:
        ra, rb = pick(s, 'charged'), pick(s, 'frozen')
        if len(ra) == len(rb) == len(scen):
            diffs = ', '.join(num(x['score'] - y['score'], sign=True) for x, y in zip(ra, rb))
            out.append(f'- Цена ожидания, {s}: (б) − (а) по сценариям: {diffs}; среднее {ci(paired(wrap(ra), wrap(rb), "score"))}.')
    out.append(f"- Счётчик настоящих обращений к модели в кэше (включая запросы предыдущего исполнителя): {real['real_calls_total']}.")
    return out


def main():
    e16 = json.loads((ROOT / 'runs/E16/summary.json').read_text(encoding='utf-8'))
    real = json.loads((ROOT / 'runs/R3_real/summary.json').read_text(encoding='utf-8'))
    lines = ['# R3: таблицы (собраны tools/build_r3_report.py из runs/E16/summary.json и runs/R3_real/summary.json)',
             '', '## Имитатор', ''] + mock_tables(e16) + ['', '## Настоящая модель', ''] + real_tables(real)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'R3-tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    keep = ('score', 'samples_collected', 'samples_total', 'samples_share', 'returned', 'llm_calls', 'time')
    snapshot = {'E16': {'generated': e16['generated'], 'errors': e16['errors'], 'claims': e16['claims'],
                        'runs': [{'arm': r['arm'], 'condition': r['condition'], 'level': r['level'], 'seed': r['seed'],
                                  **{k: r['metrics'].get(k) for k in keep}} for r in e16['runs']]},
                'real': real}
    (OUT / 'R3-results.json').write_text(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
