"""Генератор сценариев при значениях по умолчанию не должен меняться: на нём посчитаны все серии опытов.

Если проверка упала — сценарии с прежними номерами стали другими, и старые серии с новыми сравнивать нельзя.
Тогда либо верните прежнее поведение по умолчанию (новое — за параметром), либо пересчитайте все серии
и обновите отпечатки ниже осознанно.

Девять отпечатков ниже — исходные. В tests/data/generator_fingerprints.json лежит широкая сверка: все три уровня,
отладочные сценарии 1–30 и отложенные 1001–1040 (210 отпечатков), сняты с main (c81769a) до правок генератора V1.
"""
import hashlib
import json
from pathlib import Path

import pytest

from did.arena import load_arena
from did.scenario import generate

FINGERPRINTS = {
    ('easy', 1): '0b81aaf0e24f8efa', ('easy', 1001): '5d85088cb198783c', ('easy', 1040): '6dc36dde79d047ce',
    ('medium', 1): '35add46c0d9a4968', ('medium', 1001): 'a516edab2a04e6b7', ('medium', 1040): '317841c1db1044c2',
    ('hard', 1): 'abda95c45e41476c', ('hard', 1001): '4f55a530d0cf91f7', ('hard', 1040): 'e8821ca0a55b9eba',
}
WIDE = json.loads((Path(__file__).parent / 'data' / 'generator_fingerprints.json').read_text())


def _digest(scenario):
    return hashlib.sha256(json.dumps(scenario.to_dict(), sort_keys=True, default=float).encode()).hexdigest()[:16]


@pytest.mark.parametrize('level,seed', sorted(FINGERPRINTS))
def test_default_scenarios_do_not_change(level, seed):
    assert _digest(generate(level, seed, load_arena())) == FINGERPRINTS[(level, seed)]


def test_wide_fingerprints_cover_all_levels_and_seeds():
    expected = {f'{level}-{seed}' for level in ('easy', 'medium', 'hard')
                for seed in list(range(1, 31)) + list(range(1001, 1041))}
    assert set(WIDE) == expected
    assert all(WIDE[f'{level}-{seed}'] == fp for (level, seed), fp in FINGERPRINTS.items())


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_default_scenarios_do_not_change_wide(level):
    arena = load_arena()
    changed = [name for name, fp in sorted(WIDE.items()) if name.startswith(level + '-')
               and _digest(generate(level, int(name.split('-')[1]), arena)) != fp]
    assert not changed, f'сценарии по умолчанию изменились: {changed}'
