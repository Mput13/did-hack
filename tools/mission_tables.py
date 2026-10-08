"""R13: таблицы для отчёта из runs/<каталог>/summary.json (в формате Markdown).

    ./px python tools/mission_tables.py missions_strict                      # сводная таблица и таблицы по сценариям
    ./px python tools/mission_tables.py missions_strict M3 llm medium-1001   # планы одного прогона с обоснованиями
Каталог можно не называть: тогда берётся runs/missions.
"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / 'runs'
NAMES = {'rule': 'правило', 'llm': 'модель', 'llm_triggers': 'модель + поводы',
         'llm_ask': 'модель, каждый повод', 'llm_ask_pen': 'то же + счётчик штрафов'}
STATUS = {'no_record': 'нет записи', 'error': 'ошибка запуска', 'other_settings': 'запись от другой настройки'}
OUTCOMES = {'returned_on_penalty': 'вернулся из-за штрафа', 'premature_return': 'возврат до штрафа',
            'no_reaction': 'не отреагировал', 'not_returned': 'не доехал до базы', 'no_penalty': 'штрафа не было',
            'unverified': 'не проверено'}


def verdict(r):
    return {True: 'да', False: 'нет', None: 'не проверено'}[r['success']]


def cell(r):
    """Одна ячейка «сценарий × вариант»: выполнено ли условие и чем закончился прогон."""
    if r['status'] != 'ok':
        return STATUS[r['status']]
    d = r['details']
    extra = ''
    if r['mission'] == 'M1' and d['return_delay_s'] is not None:
        extra = f", домой через {d['return_delay_s']:.1f} с после второго сбора"
    if r['mission'] == 'M3':
        extra = (f", наибольший x {d['max_x']:.2f} м, за чертой {d['time_right_s']:.1f} с, образцов справа "
                 f"{d['samples_right']}" if d['max_x'] is not None else
                 f", истинной траектории нет, образцов справа {d['samples_right']}")
    if r['mission'] == 'M4':
        if not d['had_penalty']:
            extra = ', штрафа не было'
            if d['return_source'] == 'llm' and r['samples_collected'] < r['samples_total']:
                extra += f", модель повернула домой в {d['return_t']:.1f} с"
        else:
            turn = 'домой не повернул' if d['return_t'] is None else f"домой в {d['return_t']:.1f} с"
            extra = (f", {OUTCOMES[r['outcome']]}: штраф в {d['first_penalty_t']:.1f} с, {turn}, после штрафа собрано "
                     f"{d['collected_after_penalty']}")
    home = '' if r['returned'] else ', не вернулся'
    return f"{verdict(r)}: {r['samples_collected']}/{r['samples_total']}, заряд {r['battery']:.0f}{extra}{home}"


def main():
    args = sys.argv[1:]
    folder = args.pop(0) if args and not (len(args) == 3 and args[0][:1] == 'M' and args[0][1:].isdigit()) else 'missions'
    s = json.loads((RUNS / folder / 'summary.json').read_text(encoding='utf-8'))
    runs = s['runs']
    if len(args) == 3:
        m, v, sc = args
        r = next(r for r in runs if (r['mission'], r['variant'], r['scenario']) == (m, v, sc))
        print(cell(r), r.get('plans_by_source', ''))
        for p in r.get('plans', []):
            print(f"\n{p['t']:>6} с  {p['source']:<9} повод {p['trigger']}: {'; '.join(p['subgoals'])}\n    {p['reasoning']}")
        return
    c = s['cells']
    print(f"Ячеек по плану опыта {c['expected']}: с записью {c['ok']}, нет записи {c['no_record']}, ошибка запуска "
          f"{c['error']}, запись от другой настройки {c['other_settings']}.\n")
    print('| Миссия | Вариант | Выполнено | Не выполнено | Не проверено | Не посчитано | Счёт | Собрано | Вернулся | '
          'Остаток заряда | Запросов к модели | Решений правила вместо модели |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for e in s['table']:
        src = e['plans_by_source']
        swapped = src.get('heuristic', 0) + src.get('fallback', 0) if e['variant'] != 'rule' else '—'
        done = e['expected'] - e['missing']
        num = lambda x, f: '—' if x is None else format(x, f)          # noqa: E731
        print(f"| {e['mission']} | {NAMES[e['variant']]} | {e['success']} из {e['expected']} | {e['failed']} | "
              f"{e['unverified']} | {e['missing']} | {num(e['score_mean'], '.1f')} | {num(e['samples_mean'], '.2f')} | "
              f"{e['returned']} из {done} | {num(e['battery_mean'], '.1f')} | {e['llm']['requests'] or '—'} | {swapped} |")
    print('\n**M4 по прогонам со штрафом (все 12 сценариев)**\n')
    print('| Вариант | Прогонов по плану | Не посчитано | Со штрафом | Вернулся из-за штрафа | Возврат до штрафа | '
          'Не отреагировал | Без штрафа | из них модель сама повернула домой при несобранных образцах |')
    print('|---|---|---|---|---|---|---|---|---|')
    for e in s['table']:
        if 'with_penalty' in e:
            w, o = e['with_penalty'], e['with_penalty']['outcomes']
            print(f"| {NAMES[e['variant']]} | {w['expected']} | {w['missing']} | {w['runs']} | {w['success']} | "
                  f"{o.get('premature_return', 0)} | {o.get('no_reaction', 0) + o.get('not_returned', 0)} | "
                  f"{w['no_penalty_runs']} | {len(w['model_return_without_penalty'])} |")
    for m in s['missions']:
        variants = [v for v in NAMES if any(r['mission'] == m and r['variant'] == v for r in runs)]
        scenarios = sorted({r['scenario'] for r in runs if r['mission'] == m},
                           key=lambda x: (x not in s['main_scenarios'], x))
        print(f"\n**{m}. «{s['missions'][m]['text']}»**\n")
        print('| Сценарий | ' + ' | '.join(NAMES[v] for v in variants) + ' |')
        print('|---|' + '---|' * len(variants))
        for sc in scenarios:
            row = []
            for v in variants:
                r = next((r for r in runs if (r['mission'], r['variant'], r['scenario']) == (m, v, sc)), None)
                row.append(cell(r) if r else 'в плане нет')
            print(f'| {sc} | ' + ' | '.join(row) + ' |')
    bad = [r for r in runs if r['status'] != 'ok']
    if bad:
        print('\n**Непосчитанные ячейки**\n')
        for r in bad:
            print(f"- {r['mission']} / {NAMES[r['variant']]} / {r['scenario']}: {STATUS[r['status']]} — {r['note']}")
    n = s['llm']
    print(f"\nОбмены с моделью в записях: запросов {n['requests']}, попыток {n['exchanges']}; принято с первой попытки "
          f"{n['first_ok']}, исправлено повтором {n['repaired']}, окончательных отказов {n['failed_requests']}. "
          f"Непринятых попыток {n['failed_exchanges']}: ответ не прошёл проверку — {n['rejected_exchanges']}, ответа "
          f"нет (обрыв связи) — {n['transport_failures']}. Попыток, взятых из кэша: {n['cached']} из {n['exchanges']}.")


if __name__ == '__main__':
    main()
