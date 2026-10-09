"""L4b: сверка прежних агентов до и после правки — итоги прогонов должны совпасть до последнего знака.

    ./px python tools/l4b_regress.py --out /tmp/l4b_before.json      # на коде до правки
    ./px python tools/l4b_regress.py --out /tmp/l4b_after.json --compare /tmp/l4b_before.json

Записи прогонов не сохраняются. Сравниваются все числовые метрики судьи и прогона.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.runner import run_episode                     # noqa: E402

AGENTS = ('fixed', 'adaptive', 'adaptive_v2', 'adaptive_v3', 'adaptive_v4', 'adaptive_fs', 'scientist',
          'scientist_v2', 'oracle_env', 'adaptive_tour')
CASES = [(agent, level, seed, rules) for agent in AGENTS for level, seed in (('hard', 3), ('hard', 17), ('medium', 5))
         for rules in (None, 'science')]


def _run(case):
    agent, level, seed, rules = case
    s = run_episode(level, seed, agent, rules=rules, save=False)
    m = {k: v for k, v in s['metrics'].items() if isinstance(v, (int, float, bool)) or v is None}
    return '/'.join(map(str, case)), m, s['wall_s']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare', default=None)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(_run, CASES))
    out = {key: m for key, m, _ in rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print(f'прогонов {len(out)}, время счёта {sum(w for _, _, w in rows):.0f} с')
    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding='utf-8'))
        bad = [k for k in sorted(set(old) | set(out)) if old.get(k) != json.loads(json.dumps(out.get(k)))]
        print(f'расхождений: {len(bad)} из {len(out)}')
        for k in bad:
            print('  ', k)
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
