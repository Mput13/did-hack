"""R13: таблицы для отчёта из runs/missions/summary.json (в формате Markdown).

    ./px python tools/mission_tables.py            # сводная таблица и таблицы по сценариям
    ./px python tools/mission_tables.py M3 llm medium-1001      # планы одного прогона с обоснованиями
"""
import json
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / 'runs' / 'missions' / 'summary.json'
NAMES = {'rule': 'правило', 'llm': 'модель', 'llm_triggers': 'модель + поводы',
         'llm_ask': 'модель, каждый повод', 'llm_ask_pen': 'то же + счётчик штрафов'}


def cell(r):
    """Одна ячейка «сценарий × вариант»: выполнено ли условие и чем закончился прогон."""
    d = r['details']
    extra = ''
    if r['mission'] == 'M3':
        extra = f", справа {d['fraction_right'] * 100:.0f}% пути, образцов справа {d['samples_right']}"
    if r['mission'] == 'M4':
        extra = (f", штраф в {d['first_penalty_t']:.0f} с, после него собрано {d['collected_after_penalty']}"
                 if d['had_penalty'] else ', штрафа не было')
    home = '' if r['returned'] else ', не вернулся'
    return f"{'да' if r['success'] else 'нет'}: {r['samples_collected']}/{r['samples_total']}, заряд {r['battery']:.0f}{extra}{home}"


def main():
    s = json.loads(SUMMARY.read_text(encoding='utf-8'))
    runs = s['runs']
    if len(sys.argv) == 4:
        m, v, sc = sys.argv[1:]
        r = next(r for r in runs if (r['mission'], r['variant'], r['scenario']) == (m, v, sc))
        print(cell(r), r['plans_by_source'])
        for p in r['plans']:
            print(f"\n{p['t']:>6} с  {p['source']:<9} повод {p['trigger']}: {'; '.join(p['subgoals'])}\n    {p['reasoning']}")
        return
    print('| Миссия | Вариант | Выполнено | Счёт | Собрано | Вернулся | Остаток заряда | Обращений к модели | Решений правила вместо модели |')
    print('|---|---|---|---|---|---|---|---|---|')
    for e in s['table']:
        src = e['plans_by_source']
        swapped = src.get('heuristic', 0) + src.get('fallback', 0) if e['variant'] != 'rule' else '—'
        print(f"| {e['mission']} | {NAMES[e['variant']]} | {e['success']} из {e['runs']} | {e['score_mean']:.1f} | "
              f"{e['samples_mean']:.2f} | {e['returned']} из {e['runs']} | {e['battery_mean']:.1f} | "
              f"{e['llm_requests'] or '—'} | {swapped} |")
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
                row.append(cell(r) if r else '—')
            print(f'| {sc} | ' + ' | '.join(row) + ' |')
    print(f"\nПрогонов {s['total_runs']}; обращений к модели в прогонах {s['llm_requests_in_runs']}, "
          f"настоящих (не из кэша) {s['llm_real_calls']}; ответов модели, не принятых проверкой: "
          f"{sum(r['llm_failed'] for r in runs)}")


if __name__ == '__main__':
    main()
