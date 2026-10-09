"""L4b: два дополнительных разбора итоговой серии (оба — сверх заранее записанных критериев).

    ./px python tools/l4b_extra.py own      # выигрыш именно от знания своей лаборатории: (а) минус (в)
    ./px python tools/l4b_extra.py loss     # разбор потерь по статьям без памяти и с памятью (tools/loss_breakdown.py)

own — парная разность разностей: на одних и тех же сценариях второго прогона (четыре раскладки на лабораторию) счёт
с памятью о своей лаборатории минус счёт с памятью о чужой. Оба прогона идут «с памятью», поэтому случайное
отклонение прогона без памяти в этой разности сокращается. Интервал — бутстреп по лабораториям.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from did.metrics import _diff_stats                    # noqa: E402
from did.runner import RUNS                             # noqa: E402

METRICS = ('score', 'samples_share', 'returned', 'hazard_first', 'battery_used')


def own(exp):
    s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(0)
    print('| Агент | Правила | Уровень | ' + ' | '.join(METRICS) + ' |')
    print('|---|---|---|' + '---|' * len(METRICS))
    for arm in ('adaptive_v2_lab', 'scientist_v2_lab'):
        for cond in ('base', 'science'):
            for level in ('hard', 'medium'):
                pick = lambda chain: {(r['seed'], r['layout']): r['metrics'] for r in s['runs']      # noqa: E731
                                      if (r['arm'], r['condition'], r['level'], r['chain'], r['step'])
                                      == (arm, cond, level, chain, 2)}
                a, b = pick('same_lab'), pick('moved')
                if not a:
                    continue
                cells = []
                for m in METRICS:
                    by_lab = {}
                    for key in a:
                        by_lab.setdefault(key[0], []).append(float(a[key][m]) - float(b[key][m]))
                    d = _diff_stats([float(np.mean(v)) for v in by_lab.values()], rng)
                    k = 3 if m in ('samples_share', 'returned') else 2
                    cells.append(f"{d['mean']:+.{k}f} [{d['ci'][0]:+.{k}f}; {d['ci'][1]:+.{k}f}]"
                                 .replace('.', ',').replace('-', '−'))
                print(f'| `{arm}` | {cond} | {level} | ' + ' | '.join(cells) + ' |')


def loss(exp):
    import loss_breakdown as lb
    for arm, cond in (('adaptive_v2', 'base'), ('adaptive_v2', 'science')):
        for title, folders in (
                ('без памяти', [f'{arm}@{cond}@L{k}' for k in (2, 4, 5, 6)]),
                ('с памятью, (а) та же лаборатория', [f'{arm}_lab@{cond}@same_lab2'] + [f'{arm}_lab@{cond}@same_lab2r{k}' for k in (4, 5, 6)]),
                ('с памятью, (в) лабораторию переставили', [f'{arm}_lab@{cond}@moved2'] + [f'{arm}_lab@{cond}@moved2r{k}' for k in (4, 5, 6)])):
            rows = [r for f in folders for r in lb.breakdown(RUNS / exp / f, 'hard', jobs=2)]
            print(lb.render(f'{arm}, {cond}, hard, второй прогон на четырёх раскладках: {title}', lb.summarize(rows)))


if __name__ == '__main__':
    {'own': own, 'loss': loss}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else 'L4b')
