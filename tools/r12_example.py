"""R12: одно и то же расследование при двух способах выбора опыта — пример для показа.

    ./px python -m tools.r12_example E27                       # список подходящих сценариев
    ./px python -m tools.r12_example E27 science hard 12007    # журнал расследования в обоих вариантах

До первого выбора опыта прогоны вариантов совпадают такт в такт, поэтому первое расследование с выбором у них
одно и то же: та же странность, те же объяснения, те же допустимые опыты. Дальше — разный опыт и разный исход.
"""
import json
import sys

from did.recorder import load_trace
from did.runner import RUNS


def _trace(exp, arm, cond, level, seed):
    folder = arm if cond == 'base' else f'{arm}@{cond}'
    return load_trace(RUNS / exp / folder / f'{level}-{seed}.json.gz')


def first_fork(a, b):
    """Первое расследование, в котором варианты выбрали разный опыт при одинаковом начале."""
    ca, cb = ({c['id']: c for c in t['choices']} for t in (a, b))
    for qa, qb in zip(a['inquiries'], b['inquiries']):
        if (qa['id'], qa['t_open'], qa['anomaly']) != (qb['id'], qb['t_open'], qb['anomaly']):
            return None
        sa, sb = ca[qa['id']]['steps'], cb[qb['id']]['steps']
        if sa and sb and sa[0]['chosen'] != sb[0]['chosen']:
            return qa, qb, ca[qa['id']], cb[qb['id']]
        if sa != sb:
            return None
    return None


def find(exp, a='gain', b='random'):
    s = json.loads((RUNS / exp / 'summary.json').read_text(encoding='utf-8'))
    spec = s['spec']
    for cond in spec['conditions']:
        for level in spec['levels']:
            for seed in range(spec['seed_start'], spec['seed_start'] + s['seeds']):
                ta, tb = (_trace(exp, arm, cond['id'], level, seed) for arm in (a, b))
                fork = first_fork(ta, tb)
                if fork is None:
                    continue
                qa, qb, ca, cb = fork
                print(f"{cond['id']:8} {level:6} {seed}  {qa['id']} t={qa['t_open']:6.1f}  {qa['anomaly']['text'][:44]:44}"
                      f" | {a}: {'→'.join(x['chosen'] for x in ca['steps'] if x['chosen']):12} {qa.get('verdict'):12}"
                      f" {ca['spent']:.2f} ед."
                      f" | {b}: {'→'.join(x['chosen'] for x in cb['steps'] if x['chosen']):14} {qb.get('verdict'):12}"
                      f" {cb['spent']:.2f} ед. ({cb['stop']})"
                      f" | правда {qa.get('truth')} | счёт {ta['result']['score']:.1f} / {tb['result']['score']:.1f}")


def show(exp, cond, level, seed, a='gain', b='random'):
    ta, tb = (_trace(exp, arm, cond, level, int(seed)) for arm in (a, b))
    fork = first_fork(ta, tb)
    if fork is None:
        raise SystemExit('в этом сценарии варианты не расходятся на первом выборе опыта')
    for arm, t, q, c in ((a, ta, fork[0], fork[2]), (b, tb, fork[1], fork[3])):
        print(f"\n===== {arm}: {exp} {cond} {level}-{seed}, расследование {q['id']} =====")
        for e in t['journal']['entries'] if isinstance(t['journal'], dict) else t['journal']:
            if (e.get('data') or {}).get('inquiry') == q['id']:
                print(f"  {e['t']:6.1f} с  {e['text']}")
        for i, st in enumerate(c['steps'], 1):
            opts = '; '.join(f"{x['id']}: польза {x['gain_bits']:.2f} бит, цена {x['cost']:.2f} ед., "
                             f"на единицу заряда {x['gain_bits'] / (x['cost'] + 0.05):.1f}" for x in st['options'])
            print(f"  шаг {i}: вероятности {st['posterior']}; допустимые — {opts or 'нет'}; "
                  f"не по карману — {[x['id'] for x in st['over_budget']] or 'нет'}; выбран {st['chosen']}")
        print(f"  остановка: {c['stop']}; бюджет {c['budget']} ед., потрачено {c['spent']} ед.; "
              f"время {q['t_close'] - q['t_open']:.1f} с; правда {q.get('truth')}; сверка: {q.get('verdict')}")
        r = t['result']
        print(f"  прогон: счёт {r['score']}, собрано {r['samples_collected']} из {r['samples_total']}, "
              f"вернулся: {r['returned']}, заряда потрачено {r['battery_used']}")


if __name__ == '__main__':
    (show if len(sys.argv) > 2 else find)(*sys.argv[1:])
