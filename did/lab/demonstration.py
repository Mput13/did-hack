"""Пары сохранённых прогонов с одинаковыми условиями, без новых вызовов модели."""
import gzip
import json

from ..runner import RUNS


def comparison_catalog():
    root = RUNS / 'llm_real'
    # Один заранее выбранный набор, а не подбор модели по выигрышу на текущей карте.
    model = 'mai-qwen3.8-flash-next'
    pairs = []
    for file in sorted((root / 'rule').glob('*.json.gz')):
        other = root / model / file.name
        if not other.is_file():
            continue
        try:
            a = json.loads(gzip.decompress(file.read_bytes()))
            b = json.loads(gzip.decompress(other.read_bytes()))
            ca = dict(a['agent']['config'])
            cb = dict(b['agent']['config'])
            for cfg in (ca, cb):
                cfg.pop('name', None)
                cfg.pop('planner', None)
            if (a['scenario'] != b['scenario'] or a['rules'] != b['rules'] or ca != cb
                    or a.get('backend') != b.get('backend') or a.get('llm') or not b.get('llm')
                    or a['agent']['config'].get('planner') != 'heuristic'
                    or b['agent']['config'].get('planner') != 'llm'
                    or not any(e.get('ok') for e in b['llm'])):
                continue
            sc = a['scenario']
            pairs.append({'level': sc['level'], 'seed': sc['seed'], 'backend': a.get('backend'),
                          'files': [file.relative_to(RUNS).as_posix(), other.relative_to(RUNS).as_posix()],
                          'model': model, 'labels': ['Без LLM', 'С LLM'],
                          'results': [a.get('result', {}), b.get('result', {})]})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {'pairs': pairs, 'source': 'recordings'}
