"""Тесты масштабирования изменений среды (n_soil_changes, n_new_hazards) и обратной совместимости."""
import pytest
from did.arena import load_arena
from did.scenario import generate
from did.judge import Judge
from did.config import Rules


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def test_scenario_default_backwards_compatibility(arena):
    """По умолчанию (n_soil_changes=1, n_new_hazards=1) сценарии обязаны совпадать побитно."""
    for level in ('easy', 'medium', 'hard'):
        for seed in (1, 2, 3, 7, 10, 42, 100, 2024):
            sc_default = generate(level, seed, arena)
            sc_explicit = generate(level, seed, arena, n_soil_changes=1, n_new_hazards=1)
            assert sc_default.to_dict() == sc_explicit.to_dict()


def test_scenario_multiple_soil_changes(arena):
    """Проверка генерации нескольких смен грунта."""
    sc = generate('hard', seed=42, arena=arena, events=['soil_change'], n_soil_changes=3, n_new_hazards=0)
    soil_events = [e for e in sc.events if e['type'] == 'soil_change']
    assert len(soil_events) == 3
    # Проверяем упорядоченность по времени
    times = [e['t'] for e in soil_events]
    assert times == sorted(times)
    # Все события содержат валидные зоны грунтов
    for ev in soil_events:
        assert len(ev['soils']) == len(sc.soils)


def test_scenario_multiple_new_hazards(arena):
    """Проверка генерации нескольких новых опасных зон."""
    sc = generate('hard', seed=42, arena=arena, events=['new_hazard'], n_soil_changes=0, n_new_hazards=3)
    hazard_events = [e for e in sc.events if e['type'] == 'new_hazard']
    assert len(hazard_events) == 3
    zone_ids = [e['zone'].id for e in hazard_events]
    assert len(set(zone_ids)) == 3  # все id уникальны
    for ev in hazard_events:
        assert ev['zone'].r >= 0.2


def test_scenario_zero_events(arena):
    """При n=0 события не должны создаваться."""
    sc = generate('hard', seed=1, arena=arena, n_soil_changes=0, n_new_hazards=0, events=['soil_change', 'new_hazard'])
    assert len(sc.events) == 0


def test_judge_applies_multiple_events(arena):
    """Судья корректно применяет несколько событий подряд во времени."""
    sc = generate('hard', seed=1, arena=arena, events=['soil_change', 'new_hazard'], n_soil_changes=3, n_new_hazards=3)
    judge = Judge(sc, arena, Rules(), seed=1)

    initial_hazards_count = len(judge.hazards)
    # Выполняем шаги времени, охватывающие все события
    max_t = max(e['t'] for e in sc.events) + 1.0
    t = 0.0
    while t <= max_t:
        judge.step(t, sc.base[0], sc.base[1])
        t += 0.5

    # Все 3 смены грунта и 3 новые опасные зоны должны быть в world_log
    log_types = [w['type'] for w in judge.world_log]
    assert log_types.count('soil_change') == 3
    assert log_types.count('new_hazard') == 3
    assert len(judge.hazards) == initial_hazards_count + 3
