"""Проверка главного числа P1 на текущем коде: adaptive против adaptive_v2, базовые правила, hard, 8001–8040.

    pixi run python -m tools.showcase_p1_check     # 80 прогонов быстрого симулятора, около двух минут

Итог (runs/_showcase/p1_check.json) и записи прогонов (runs/_showcase/<агент>/) читает tools/build_showcase.py.
"""
import json
import sys
from pathlib import Path

from did.metrics import paired
from did.runner import run_episode

OUT = Path(__file__).resolve().parents[1] / 'runs' / '_showcase' / 'p1_check.json'
METRICS = ('score', 'returned', 'samples_share', 'time')

out = {}
for arm in ('adaptive', 'adaptive_v2'):
    out[arm] = [run_episode('hard', seed, arm, experiment='_showcase', arm=arm) for seed in range(8001, 8041)]
    print(arm, 'готово', file=sys.stderr, flush=True)
res = {'pair': {m: paired(out['adaptive_v2'], out['adaptive'], m) for m in METRICS},
       'means': {a: {m: sum(r['metrics'][m] for r in rs) / len(rs) for m in METRICS} for a, rs in out.items()},
       'runs': {a: [{'seed': r['seed'], **{m: r['metrics'][m] for m in METRICS}} for r in rs] for a, rs in out.items()}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: res[k] for k in ('pair', 'means')}, ensure_ascii=False, indent=1))
