"""Пересчёт итога E28 после правки по ревью: заново идут только варианты *_v4, на тех же сценариях, в один поток.

    nice -n 15 ./px python tools/p3_rerun.py

Прежние записи и сводка первого прогона переносятся в runs/E28_first (один раз). Прогоны *_v2 не повторяются:
их код не менялся (tools/p3_regress.py), в новую сводку они идут из первой как есть.
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from did.experiments import RUNS, _job, load_spec, summarize_experiment      # noqa: E402

EXP, FIRST = 'E28', 'E28_first'


def main():
    spec = load_spec(EXP)
    out, keep = RUNS / EXP, RUNS / FIRST
    arms = [a for a in spec['arms'] if a['id'].endswith('_v4')]
    if not keep.exists():
        keep.mkdir()
        shutil.copy(out / 'summary.json', keep / 'summary.json')
        for folder in sorted(out.iterdir()):
            if folder.is_dir() and folder.name.partition('@')[0] in {a['id'] for a in arms}:
                shutil.move(str(folder), str(keep / folder.name))
    first = json.loads((keep / 'summary.json').read_text(encoding='utf-8'))
    results = [r for r in first['runs'] if not r['arm'].endswith('_v4')]
    conditions = [{**c, 'rules': dict(c.get('rules', {}))} for c in spec['conditions']]
    seeds = range(spec['seed_start'], spec['seed_start'] + spec['seeds'])
    tasks = [(EXP, arm, cond, level, seed) for arm in arms for cond in conditions for level in spec['levels']
             for seed in seeds]
    t0 = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        results.append(_job(task))
        if i % 40 == 0:
            print(f'{i}/{len(tasks)}', flush=True)
    summary = summarize_experiment(spec, results, spec['seeds'], time.perf_counter() - t0)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f"прогонов {len(tasks)}, ошибок {len(summary['errors'])}")


if __name__ == '__main__':
    main()
