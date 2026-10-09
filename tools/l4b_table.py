"""L4b: таблицы по сводке серии (runs/<папка>/summary.json): парные разности «с памятью − без памяти».

    ./px python tools/l4b_table.py L4b_dev_d80            # коротко, по строке на шаг цепочки
    ./px python tools/l4b_table.py --md L4b               # таблицы для отчёта (markdown)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.runner import RUNS                             # noqa: E402

COLS = ('score', 'hazard_first', 'samples_share', 'returned', 'battery_used')
MD = (('score', 'счёт, очки', 2), ('samples_share', 'образцы, доля', 3), ('returned', 'возврат, доля', 3),
      ('hazard_hits', 'штрафы за зоны, шт.', 2), ('hazard_first', 'первый въезд, шт.', 2),
      ('battery_used', 'расход, ед.', 2), ('false_collects', 'ложные сборы, шт.', 2))
CHAINS = {'same_lab': '(а) та же лаборатория, другие образцы', 'repeat': '(б) тот же сценарий повторно',
          'moved': '(в) лабораторию переставили', 'repeat_std': 'прежний сценарий повторно (справочно)'}


def ci(d, digits=2):
    return f"{d['mean']:+.{digits}f} [{d['ci'][0]:+.{digits}f}; {d['ci'][1]:+.{digits}f}]".replace('.', ',').replace('-', '−')


def num(v, digits=2):
    return f'{v:.{digits}f}'.replace('.', ',')


def short(s):
    for p in s['pairs']:
        if 'diff' not in p or p['step'] == 1:
            continue
        cells = '  '.join(f"{m} {p['diff'][m]['mean']:+.2f} [{p['diff'][m]['ci'][0]:+.2f}; {p['diff'][m]['ci'][1]:+.2f}]"
                          for m in COLS)
        print(f"{p['arm'][:12]:12} {p['condition']:7} {p['level']:6} {p['chain']:10} шаг {p['step']} n={p['n']:<3} "
              f"без {p['base_mean']['score']:.1f} (зон {p['base_mean']['hazard_first']:.2f})  {cells}  "
              f"| в памяти зон {p['memory']['hazards']:.2f}, снято {p['memory']['refuted']:.2f}, "
              f"подтв. {p['memory']['confirmed']:.2f}, грунт снят {p['memory']['soil_refuted']:.2f}, "
              f"менялась {p['memory']['volatile']:.2f}")
        if 'pooled' in p:
            cells = '  '.join(f"{m} {p['pooled'][m]['mean']:+.2f} [{p['pooled'][m]['ci'][0]:+.2f}; "
                              f"{p['pooled'][m]['ci'][1]:+.2f}]" for m in COLS)
            print(f"{'':12} {'':7} {'':6} {p['chain']:10} шаг 2 по {p['layouts']} раскладкам: без "
                  f"{p['pooled_base']['score']:.1f} (зон {p['pooled_base']['hazard_first']:.2f})  {cells}")


def markdown(s):
    conds = {c['id']: c['label'].lower() for c in s['spec']['conditions']}
    for arm in sorted({p['arm'] for p in s['pairs']}):
        for cond in conds:
            for level in s['spec']['levels']:
                rows = [p for p in s['pairs'] if (p['arm'], p['condition'], p['level']) == (arm, cond, level)]
                if not rows:
                    continue
                print(f"\n**`{arm}`, {conds[cond]}, уровень {level}** (n = {rows[0]['n']} лабораторий; в ячейках — "
                      'среднее без памяти → с памятью и парная разность с 95% интервалом)\n')
                print('| Условие | Прогон | ' + ' | '.join(label for _, label, _ in MD) + ' | В памяти: зон / участков грунта |')
                print('|---|---|' + '---|' * (len(MD) + 1))
                for p in rows:
                    if 'diff' not in p:
                        continue
                    cells = [f"{num(p['base_mean'][m], d)} → {num(p['mean'][m], d)}; {ci(p['diff'][m], d)}"
                             if p['step'] > 1 else num(p['mean'][m], d) for m, _, d in MD]
                    print(f"| {CHAINS[p['chain']]} | {p['step']} | " + ' | '.join(cells)
                          + f" | {num(p['memory']['hazards'])} / {num(p['memory']['soil_zones'])} |")
                    if 'pooled' in p:
                        cells = [f"{num(p['pooled_base'][m], d)} → {num(p['pooled_base'][m] + p['pooled'][m]['mean'], d)}; "
                                 f"{ci(p['pooled'][m], d)}" for m, _, d in MD]
                        print(f"| {CHAINS[p['chain']]} | 2, по {p['layouts']} раскладкам | " + ' | '.join(cells) + ' | |')
    print('\n**Критерии**\n')
    for c in s['claims']:
        word = {'supported': 'выполнено', 'refuted': 'не выполнено', 'no_data': 'нет данных'}[c['status']]
        print(f"- {c['text']}: {ci(c['pair'], 3 if c['metric'] == 'returned' else 2) if c['pair'] else '—'} — {word}")


def main():
    md = '--md' in sys.argv
    for exp in [a for a in sys.argv[1:] if a != '--md']:
        s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
        print(f"== {exp}: настройки {s.get('lab_opts') or 'по умолчанию'}, прогонов {len(s['runs'])}, "
              f"ошибок {len(s['errors'])}")
        (markdown if md else short)(s)


if __name__ == '__main__':
    main()
