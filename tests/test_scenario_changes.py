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


MANY = [{'n_soil_changes': 2, 'n_new_hazards': 2}, {'n_soil_changes': 3, 'n_new_hazards': 3},
        {'events': ['soil_change'], 'n_soil_changes': 3, 'n_new_hazards': 0},
        {'events': ['new_hazard'], 'n_soil_changes': 0, 'n_new_hazards': 3}]
SEEDS = list(range(1, 31)) + list(range(1001, 1041))


def test_original_new_hazard_keeps_its_fault_when_extra_zone_comes_first(arena):
    """hard-1002, 3 смены и 3 зоны: дополнительная зона X3 появляется раньше исходной X2 — сбой должен быть у обеих."""
    sc = generate('hard', 1002, arena, n_soil_changes=3, n_new_hazards=3)
    zones = [e['zone'] for e in sc.events if e['type'] == 'new_hazard']
    assert [z.id for z in zones][:2] == ['X3', 'X2']
    assert all(z.fault and z.fault_s > 0 for z in zones)


@pytest.mark.parametrize('args', MANY, ids=['c2_2', 'c3_3', 'soil_x3', 'hazard_x3'])
def test_every_new_hazard_gets_exactly_one_fault(arena, args):
    """Исходной новой зоне сбой назначается так же, как в сценарии по умолчанию; дополнительным — свой, один раз."""
    for seed in SEEDS:
        sc = generate('hard', seed, arena, **args)
        zones = {e['zone'].id: e['zone'] for e in sc.events if e['type'] == 'new_hazard'}
        assert all(z.fault and 20.0 <= z.fault_s <= 30.0 for z in list(zones.values()) + sc.hazards), seed
        if 'events' in args:
            continue                # без события sensor_fault исходная зона та же, но сверять удобнее на полном наборе
        base = next(e['zone'] for e in generate('hard', seed, arena).events if e['type'] == 'new_hazard')
        assert (zones[base.id].fault, zones[base.id].fault_s) == (base.fault, base.fault_s), seed


@pytest.mark.parametrize('args', MANY, ids=['c2_2', 'c3_3', 'soil_x3', 'hazard_x3'])
def test_event_times_do_not_coincide(arena, args):
    """При нескольких сменах или зонах все события разнесены во времени не меньше чем на MIN_EVENT_GAP_S."""
    from did.scenario import MIN_EVENT_GAP_S
    for seed in SEEDS:
        sc = generate('hard', seed, arena, **args)
        times = [e['t'] for e in sc.events]
        assert times == sorted(times), seed
        assert all(b - a >= MIN_EVENT_GAP_S - 1e-9 for a, b in zip(times, times[1:])), (seed, times)
        assert all(20.0 <= t <= 70.0 for t in times), (seed, times)


def test_soil_changes_stay_in_chain_order(arena):
    """Смены грунта строятся одна из другой; после разноса времён порядок цепочки не должен нарушиться."""
    for seed in SEEDS:
        direct = generate('hard', seed, arena, events=['soil_change'], n_soil_changes=3, n_new_hazards=0)
        full = generate('hard', seed, arena, n_soil_changes=3, n_new_hazards=3)
        chain = [e['soils'] for e in direct.events]
        assert [e['soils'] for e in full.events if e['type'] == 'soil_change'] == chain, seed
