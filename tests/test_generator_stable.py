"""Генератор сценариев при значениях по умолчанию не должен меняться: на нём посчитаны все серии опытов.

Если проверка упала — сценарии с прежними номерами стали другими, и старые серии с новыми сравнивать нельзя.
Тогда либо верните прежнее поведение по умолчанию (новое — за параметром), либо пересчитайте все серии
и обновите отпечатки ниже осознанно.
"""
import hashlib
import json

import pytest

from did.arena import load_arena
from did.scenario import generate

FINGERPRINTS = {
    ('easy', 1): '0b81aaf0e24f8efa', ('easy', 1001): '5d85088cb198783c', ('easy', 1040): '6dc36dde79d047ce',
    ('medium', 1): '35add46c0d9a4968', ('medium', 1001): 'a516edab2a04e6b7', ('medium', 1040): '317841c1db1044c2',
    ('hard', 1): 'abda95c45e41476c', ('hard', 1001): '4f55a530d0cf91f7', ('hard', 1040): 'e8821ca0a55b9eba',
}


@pytest.mark.parametrize('level,seed', sorted(FINGERPRINTS))
def test_default_scenarios_do_not_change(level, seed):
    scenario = generate(level, seed, load_arena()).to_dict()
    digest = hashlib.sha256(json.dumps(scenario, sort_keys=True, default=float).encode()).hexdigest()[:16]
    assert digest == FINGERPRINTS[(level, seed)]
