"""B1: пересчёт карты образцов после сбора, который не теряет соседа и не верит показаниям при сбое датчика
(did/replay.py: suspects, legacy_replay, trusted_replay, NeighborReplay; пресеты *_v5)."""
import copy
import math
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from did.agent import PRESETS, V5, Agent, make_config
from did.arena import load_arena
from did.belief import SampleBelief
from did.replay import (NOISE_LAG, NeighborReplay, exact_replay, legacy_replay, suspects, trusted_replay)
from did.runner import run_episode
from did.scenario import generate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import b1_regress      # noqa: E402

SIGMA = 0.05


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _clone(b, replay=None):
    c = copy.copy(b)
    c.p, c._epoch, c._log, c.replay = b.p.copy(), b._epoch.copy(), list(b._log), replay
    return c


def _drive(arena, kind, seed=3, n_samples=3):
    """Робот едет от образца B к образцу A; на части пути датчик залип ('stuck'), шумит ('noise') или исправен."""
    a, b = (0.9, -0.4), (-1.2, 0.3)
    n = 56
    path = [(-1.2 + 2.1 * i / (n - 1), -0.1 - 0.3 * i / (n - 1) + 0.25 * math.sin(math.pi * i / (n - 1))) for i in range(n)]
    rng = np.random.default_rng(seed)
    m = SampleBelief(arena, n_samples, 2.0)
    held = None
    for i, (x, y) in enumerate(path):
        z = max(0.0, 1.0 - min(math.dist((x, y), a), math.dist((x, y), b)) / 2.0)
        sigma = SIGMA
        if kind == 'stuck' and 3 <= i < 34:
            z = float(np.clip(z + rng.normal(0.0, SIGMA), 0.0, 1.0)) if held is None else held
            held = z
        elif kind == 'noise' and 3 <= i < 34:
            z = float(np.clip(z + rng.normal(0.0, 0.25), 0.0, 1.0))
            sigma = SIGMA if i < 15 else 0.22            # оценка шума поднимает тревогу с опозданием
        else:
            z = float(np.clip(z + rng.normal(0.0, SIGMA), 0.0, 1.0))
        m.update(x, y, z, sigma)
    return m, a, b, path[-1]


def _false(m, truth, min_mass=0.35):
    return [c for c in m.candidates(min_mass=min_mass) if math.dist((c['x'], c['y']), truth) > 0.5]


# --- флажок и пресеты --------------------------------------------------------------------------------

def test_new_replay_is_off_by_default(arena):
    assert make_config('adaptive').neighbor_replay == ''
    for name, cfg in PRESETS.items():
        if not name.endswith('_v5'):
            assert cfg.neighbor_replay == '', name
            assert Agent(arena, cfg, n_samples=3).belief.replay is None or cfg.exact_replay, name
    for old, new in (('adaptive_v2', 'adaptive_v5'), ('scientist_v2', 'scientist_v5')):
        a, b = PRESETS[old], PRESETS[new]
        differ = {f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)}
        assert differ == {'name', 'neighbor_replay'} and V5 == {'neighbor_replay': 'gated_clean'}
        assert isinstance(Agent(arena, b, n_samples=3).belief.replay, NeighborReplay)
    # прежняя проба за своим флажком — та же функция, что и раньше
    assert Agent(arena, make_config('adaptive_v2', exact_replay=True), n_samples=3).belief.replay is exact_replay
    with pytest.raises(ValueError):
        NeighborReplay(SIGMA, mode='other')


def test_previous_agents_give_previous_maps_and_results():
    """34 прогона прежних пресетов: итоги и карта образцов после каждого сбора — как у кода до правки B1."""
    bad, n = b1_regress.compare()
    assert n == 34 and not bad, bad


# --- прежний пересчёт, вынесенный в функцию ----------------------------------------------------------

@pytest.mark.parametrize('kind', ['ok', 'stuck', 'noise'])
def test_legacy_replay_is_the_previous_replay(arena, kind):
    m, a, b, end = _drive(arena, kind)
    old, same = _clone(m), _clone(m, legacy_replay)
    old.collected(*end)
    same.collected(*end)
    assert np.array_equal(old.p, same.p) and np.array_equal(old._epoch, same._epoch)
    assert old.left == same.left == 2 and not same._log


# --- недоверие к показаниям ---------------------------------------------------------------------------

def test_suspects_marks_stuck_runs_and_noise_with_its_lag():
    log = [(0.0, 0.0, z, SIGMA) for z in (0.31, 0.42, 0.42, 0.42, 0.42, 0.37, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    bad = suspects(log, SIGMA)
    # первое показание серии настоящее; упёршиеся в 0 и 1 одинаковы и у исправного датчика
    assert bad.tolist() == [False, False, True, True, True] + [False] * 8
    assert not suspects([(0.0, 0.0, 0.42, SIGMA), (0.0, 0.0, 0.42, SIGMA)], SIGMA).any()      # двух мало
    rng = np.random.default_rng(0)
    log = [(0.0, 0.0, float(z), SIGMA) for z in rng.uniform(0.2, 0.8, 30)]
    log += [(0.0, 0.0, float(z), 0.2) for z in rng.uniform(0.2, 0.8, 5)]
    log += [(0.0, 0.0, float(z), SIGMA) for z in rng.uniform(0.2, 0.8, 5)]
    bad = suspects(log, SIGMA)
    assert bad[30:35].all() and bad[30 - NOISE_LAG:30].all()         # шумные и столько же перед тревогой
    assert not bad[:30 - NOISE_LAG].any() and not bad[35:].any()
    assert len(suspects([], SIGMA)) == 0


@pytest.mark.parametrize('seed', [3, 5])
def test_stuck_readings_do_not_become_a_confident_candidate(arena, seed):
    """Датчик залип у образца B и повторял высокое показание всю дорогу к образцу A."""
    m, a, b, end = _drive(arena, 'stuck', seed)
    assert suspects(m._log, SIGMA).sum() == 30
    exact, new = _clone(m, exact_replay), _clone(m, lambda q, x, y: trusted_replay(q, x, y, SIGMA))
    exact.collected(*end)
    new.collected(*end)
    assert max(c['mass'] for c in _false(exact, b)) >= 0.7          # прежняя проба: уверенный ложный кандидат
    assert not _false(new, b)                                        # новый пересчёт: ни одного кандидата не у B
    assert max((c['mass'] for c in _false(new, b, min_mass=0.2)), default=0.0) < 0.35


@pytest.mark.parametrize('seed', [3, 4, 5])
def test_v5_replay_drops_stuck_readings_even_when_it_keeps_the_previous_map(arena, seed):
    """Известное место не стирается, карта остаётся прежнего пересчёта — но залипшие показания в неё не идут."""
    m, a, b, end = _drive(arena, 'stuck', seed)
    old, new = _clone(m), _clone(m, NeighborReplay(SIGMA))
    old.collected(*end)
    new.collected(*end)
    assert new.replay.stats['switched'] == 0
    assert _false(old, b) and not _false(new, b)                     # прежний пересчёт оставлял ложного кандидата


def test_noisy_readings_do_not_become_a_confident_candidate(arena):
    m, a, b, end = _drive(arena, 'noise', 5)
    exact, new = _clone(m, exact_replay), _clone(m, lambda q, x, y: trusted_replay(q, x, y, SIGMA))
    exact.collected(*end)
    new.collected(*end)
    assert _false(exact, b) and not _false(new, b)


def test_without_faults_trusted_replay_is_the_exact_one(arena):
    m, a, b, end = _drive(arena, 'ok')
    assert not suspects(m._log, SIGMA).any()
    exact, new = _clone(m, exact_replay), _clone(m, lambda q, x, y: trusted_replay(q, x, y, SIGMA))
    exact.collected(*end)
    new.collected(*end)
    assert np.array_equal(exact.p, new.p)


# --- число оставшихся образцов и нормировка -----------------------------------------------------------

@pytest.mark.parametrize('kind', ['ok', 'stuck', 'noise'])
@pytest.mark.parametrize('mode', ['full', 'gated', 'gated_clean'])
def test_count_and_normalisation_agree(arena, kind, mode):
    m, a, b, end = _drive(arena, kind)
    new = _clone(m, NeighborReplay(SIGMA, mode=mode))
    new.collected(*end)
    assert new.left == m.left - 1 == 2
    assert new.total() == pytest.approx(new.left, rel=0.02)          # сумма вероятностей = число оставшихся
    assert new.prob_within(*end, 0.3) < 0.02                         # место собранного образца очищено
    assert not new._log and np.array_equal(new._epoch, new.p)
    assert (new.p >= 1e-7).all() and (new.p <= 0.995).all()
    new.update(0.0, 0.0, 0.3, SIGMA)                                 # дальше карта живёт как обычно
    assert new.total() == pytest.approx(2.0, rel=0.02) and len(new._log) == 1


def test_last_sample_leaves_an_empty_map(arena):
    m, a, b, end = _drive(arena, 'ok', n_samples=1)
    for mode in ('full', 'gated', 'gated_clean'):
        new = _clone(m, NeighborReplay(SIGMA, mode=mode))
        new.collected(*end)
        assert new.left == 0 and new.total() < 0.01 and not new.candidates(min_mass=0.2)


# --- сосед сохраняется ---------------------------------------------------------------------------------

def _first_collect(agent, level, seed):
    """Карта образцов агента прямо перед первым пересчётом после сбора и место сбора."""
    seen = {}
    collected = SampleBelief.collected

    class Stop(Exception):
        pass

    def grab(self, x, y):
        seen['map'], seen['at'] = _clone(self), (x, y)
        raise Stop

    SampleBelief.collected = grab
    try:
        run_episode(level, seed, agent, save=False)
    except Stop:
        pass
    finally:
        SampleBelief.collected = collected
    return seen['map'], seen['at']


def test_hard_11005_neighbour_survives_the_collection(arena):
    """Пример из P3: образец № 2 был кандидатом (0,61), после сбора соседа прежний пересчёт оставлял 0,08."""
    m, at = _first_collect('adaptive_v2', 'hard', 11005)
    spot = tuple(float(v) for v in generate('hard', 11005, arena).samples[2])
    before = m.prob_within(*spot, 0.35)
    assert before == pytest.approx(0.61, abs=0.02)
    old, new = _clone(m), _clone(m, NeighborReplay(SIGMA, 0.35))
    old.collected(*at)
    new.collected(*at)
    assert old.prob_within(*spot, 0.35) < 0.1
    assert not [c for c in old.candidates(min_mass=0.35) if math.dist((c['x'], c['y']), spot) <= 0.5]
    assert new.prob_within(*spot, 0.35) > 0.35
    assert [c for c in new.candidates(min_mass=0.35) if math.dist((c['x'], c['y']), spot) <= 0.5]
    assert new.replay.stats == {'collects': 1, 'switched': 1, 'skipped': 0}
    assert new.left == old.left and new.total() == pytest.approx(old.total(), rel=0.02)


def test_v5_replay_is_the_previous_one_when_nothing_known_is_lost(arena):
    """Известного места на карте нет, сбоев датчика не было — карта после сбора совпадает с прежней до цифры."""
    m, a, b, end = _drive(arena, 'ok')
    assert not [c for c in m.candidates(min_mass=0.35) if math.dist((c['x'], c['y']), end) > 0.5]
    for mode in ('gated', 'gated_clean'):
        old, new = _clone(m), _clone(m, NeighborReplay(SIGMA, 0.35, mode))
        old.collected(*end)
        new.collected(*end)
        assert np.array_equal(old.p, new.p) and new.replay.stats['switched'] == 0
    full = _clone(m, NeighborReplay(SIGMA, 0.35, 'full'))
    full.collected(*end)
    assert not np.array_equal(old.p, full.p)


def test_v5_runs_a_whole_scenario():
    old = run_episode('hard', 33, 'adaptive_v2', save=False)['metrics']
    new = run_episode('hard', 33, 'adaptive_v5', save=False)['metrics']
    assert new['returned'] and new['false_collects'] <= old['false_collects']
    assert new['samples_collected'] >= old['samples_collected']
    sci = run_episode('hard', 7, 'scientist_v5', rules='science', save=False)['metrics']
    assert sci['score'] > 0
