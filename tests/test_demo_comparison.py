"""Каталог не сопоставляет прогоны с разными условиями или без ответов модели."""
import copy
import gzip
import json

import pytest

from did.lab import demonstration as demo


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, 'RUNS', tmp_path)
    base = {'scenario': {'level': 'hard', 'seed': 1001}, 'rules': {'battery_start': 60},
            'backend': 'fastsim', 'agent': {'config': {'name': 'adaptive', 'planner': 'heuristic', 'learn_soil': True}},
            'result': {}, 'llm': []}
    model = copy.deepcopy(base)
    model['agent']['config'].update(name='adaptive_llm', planner='llm')
    model['llm'] = [{'ok': True}]
    def write(a=base, b=model):
        for folder, obj in [('rule', a), ('mai-qwen3.8-flash-next', b)]:
            p = tmp_path / 'llm_real' / folder / 'hard-1001.json.gz'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(json.dumps(obj).encode()))
    write()
    return base, model, write


def test_catalog_accepts_same_conditions(records):
    pairs = demo.comparison_catalog()['pairs']
    assert len(pairs) == 1 and pairs[0]['seed'] == 1001
    assert pairs[0]['labels'] == ['Без LLM', 'С LLM']


@pytest.mark.parametrize('difference', ['scenario', 'rules', 'config', 'backend', 'failed_model'])
def test_catalog_refuses_mismatched_pair(records, difference):
    a, b, write = records
    if difference == 'scenario': b['scenario']['seed'] = 1002
    elif difference == 'rules': b['rules']['battery_start'] = 80
    elif difference == 'config': b['agent']['config']['learn_soil'] = False
    elif difference == 'backend': b['backend'] = 'gazebo'
    else: b['llm'] = [{'ok': False}]
    write(a, b)
    assert not demo.comparison_catalog()['pairs']
