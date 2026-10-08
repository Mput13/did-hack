"""R12: прежние варианты агента должны давать прежние числа до цифры.

    ./px python -m tools.r12_regress dump /tmp/r12/before.json     # на коде main
    ./px python -m tools.r12_regress dump /tmp/r12/after.json      # на коде ветки
    ./px python -m tools.r12_regress diff /tmp/r12/before.json /tmp/r12/after.json
    ./px python -m tools.r12_regress summary /Users/a/MAI/DID/runs/E10/summary.json /tmp/r12/after.json

20 прогонов исследователя из опыта E10 («научные» правила, средний и трудный уровень, сценарии 1001–1010)
и 8 прогонов при базовых правилах. Сравниваются все метрики, расследования целиком и журнал.
"""
import json
import sys

import did
from did.runner import run_episode

RUNS = [('science', level, seed) for level in ('medium', 'hard') for seed in range(1001, 1011)] + \
       [(None, level, seed) for level in ('medium', 'hard') for seed in range(1001, 1005)]


def dump(path):
    out = {}
    for rules, level, seed in RUNS:
        s = run_episode(level, seed, 'scientist', rules=rules, save=False, knowledge={})
        s.pop('wall_s')
        out[f'{rules or "base"}/{level}-{seed}'] = s
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'код: {did.__file__}; прогонов {len(out)} → {path}')


def diff(a, b):
    a, b = (json.load(open(p, encoding='utf-8')) for p in (a, b))
    bad = 0
    for k in sorted(a):
        x, y = a[k], b[k]
        for q in y.get('science', {}).get('inquiries', []):
            q.pop('choice', None)
        y.get('science', {}).pop('choices', None)       # новое поле ветки: как выбирался опыт
        if x != y:
            bad += 1
            keys = [m for m in x['metrics'] if x['metrics'][m] != y['metrics'].get(m)]
            print(f'расходится {k}: метрики {keys or "совпали, отличие в расследованиях"}')
    print(f'сравнено {len(a)} прогонов, расходятся {bad}')
    return bad


def summary(path, mine):
    """Сверка со сводкой опыта E10 из основного каталога: те же сценарии, вариант scientist."""
    s, mine = json.load(open(path, encoding='utf-8')), json.load(open(mine, encoding='utf-8'))
    ref = {f"science/{r['level']}-{r['seed']}": r['metrics'] for r in s['runs'] if r['arm'] == 'scientist'}
    bad = n = 0
    for k, r in mine.items():
        if k not in ref:
            continue
        n += 1
        keys = [m for m, v in ref[k].items() if r['metrics'].get(m) != v]
        if keys:
            bad += 1
            print(f'расходится {k}: {keys}')
    print(f'сводка от {s["generated"]}: сравнено {n} прогонов, расходятся {bad}')
    return bad


if __name__ == '__main__':
    sys.exit(1 if {'dump': dump, 'diff': diff, 'summary': summary}[sys.argv[1]](*sys.argv[2:]) else 0)
