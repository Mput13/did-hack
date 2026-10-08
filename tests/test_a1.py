"""A1: план на остаток прогона (did/tour.py), оракулы-мерила (did/oracle.py)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.judge import Judge
from did.oracle import LEAD_S, ORACLES, KnownSamples, Truth
from did.runner import make_agent, run_episode
from did.scenario import generate
from did.tour import Settings, TourScheme


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _scheme(**opts):
    ts = TourScheme.__new__(TourScheme)
    ts.s = Settings(**opts)
    return ts


def _line(xs, values, base=0.0):
    """Цели на прямой: робот и база в точке base, метр пути стоит 1 ед."""
    nodes = [{'id': f'C{i + 1}', 'kind': 'investigate', 'x': x, 'y': 0.0, 'value': v, 'carry': v,
              'to_m': abs(x - base), 'back_m': abs(x - base)} for i, (x, v) in enumerate(zip(xs, values))]
    metres = np.array([[abs(a - b) for b in xs] for a in xs], dtype=float)
    return nodes, metres


def _solve(ts, nodes, metres, battery, margin=1.0, absolute=0.0, **kw):
    return ts.solve(nodes, metres, battery, (1.0, 0.0), (margin, absolute), 10.0, 0.1, **kw)


# --- перебор --------------------------------------------------------------------------------------

def test_tour_takes_everything_in_the_shortest_order_when_charge_allows():
    nodes, metres = _line([3.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    order, info = _solve(_scheme(dwell=0.0), nodes, metres, battery=10.0)
    assert sorted(order) == [0, 1, 2]
    assert info['cost'] == pytest.approx(6.0)             # 0 → 3 и обратно, остальные по пути
    assert info['value'] == pytest.approx(3.0)


def test_tour_drops_what_does_not_fit_and_keeps_the_most_valuable_subset():
    # Слева одна цель в 4 м, справа две в 1 и 2 м: на всё нужно 12 ед., есть 9.
    nodes, metres = _line([-4.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    order, info = _solve(_scheme(dwell=0.0), nodes, metres, battery=9.0)
    assert sorted(order) == [1, 2]
    assert info['cost'] == pytest.approx(4.0)
    # Если левая цель стоит трёх правых, выбирается она.
    nodes, metres = _line([-4.0, 1.0, 2.0], [3.0, 1.0, 1.0])
    assert _solve(_scheme(dwell=0.0), nodes, metres, battery=9.0)[0] == [0]


def test_tour_respects_reserve_and_returns_nothing_when_no_target_fits():
    nodes, metres = _line([2.0], [1.0])
    assert _solve(_scheme(dwell=0.0), nodes, metres, battery=4.0)[0] == [0]
    assert _solve(_scheme(dwell=0.0), nodes, metres, battery=4.0, absolute=0.5)[0] == []
    assert _solve(_scheme(dwell=0.0), nodes, metres, battery=4.0, margin=1.1)[0] == []
    assert _solve(_scheme(dwell=0.3), nodes, metres, battery=4.0)[0] == []     # остановка тоже стоит заряда


def test_forced_first_target_is_kept():
    nodes, metres = _line([3.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    order, _ = _solve(_scheme(dwell=0.0), nodes, metres, battery=20.0, first=0)
    assert order[0] == 0 and sorted(order) == [0, 1, 2]


@pytest.mark.parametrize('haste', [0.0, 3.0])
def test_haste_prefers_the_near_target_first(haste):
    # База и робот в нуле; цель A в 1 м справа, цель B в 3 м слева. Заряда хватает на обе в любом порядке,
    # длина объезда одинакова. Со «спешкой» сначала берётся ближняя.
    nodes, metres = _line([-3.0, 1.0], [1.0, 1.0])
    order, info = _solve(_scheme(dwell=0.0, haste=haste), nodes, metres, battery=30.0)
    assert sorted(order) == [0, 1] and info['cost'] == pytest.approx(8.0)
    if haste:
        assert order == [1, 0]


def test_haste_search_matches_subset_search_when_haste_is_negligible():
    rng = np.random.default_rng(3)
    for _ in range(20):
        pts = rng.uniform(-3, 3, size=(5, 2))
        nodes = [{'id': f'C{i}', 'kind': 'investigate', 'x': x, 'y': y, 'value': float(rng.uniform(0.3, 1.0)),
                  'carry': 0.0, 'to_m': math.hypot(x, y), 'back_m': math.hypot(x, y)} for i, (x, y) in enumerate(pts)]
        metres = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        battery = float(rng.uniform(6, 16))
        a = _solve(_scheme(dwell=0.1), nodes, metres, battery)[1]
        b = _solve(_scheme(dwell=0.1, haste=1e9), nodes, metres, battery)[1]
        assert a['score'] == pytest.approx(b['score'], abs=1e-6)


def test_load_makes_later_legs_dearer():
    # Одна цель в 2 м: домой робот везёт образец, метр дороже на 50%.
    nodes, metres = _line([2.0], [1.0])
    ts = _scheme(dwell=0.0)
    assert ts.solve(nodes, metres, 5.0, (1.0, 0.5), (1.0, 0.0), 10.0, 0.1)[1]['cost'] == pytest.approx(5.0)
    assert ts.solve(nodes, metres, 4.9, (1.0, 0.5), (1.0, 0.0), 10.0, 0.1)[0] == []


# --- прежние варианты не тронуты, правда доступна только оракулам ----------------------------------

def test_old_presets_do_not_use_the_new_scheme(arena):
    new = {'adaptive_tour', 'adaptive_tour_full', 'adaptive_survey', 'adaptive_pickup', 'adaptive_picky'}
    for name, cfg in PRESETS.items():
        assert (cfg.scheme != 'rule') == (name in new and name != 'adaptive_picky'), name
    bot = Agent(arena, make_config('adaptive'), n_samples=5)
    assert bot.tour is None and bot._oracle is None


def test_oracles_are_not_presets():
    assert not set(ORACLES) & set(PRESETS)
    for name in ORACLES:
        cls, cfg = make_agent(name)
        assert cls is Agent and cfg.name == name


def test_only_oracles_get_the_truth(arena, monkeypatch):
    seen = {}
    real = Agent.__init__

    def spy(self, *args, **kwargs):
        seen[args[1].name] = kwargs.get('truth')
        real(self, *args, **kwargs)

    monkeypatch.setattr(Agent, '__init__', spy)
    for name in ('adaptive', 'adaptive_tour', 'oracle_all'):
        run_episode('medium', 3, name, save=False, rules={'time_limit_s': 2.0})
    assert seen['adaptive'] is None and seen['adaptive_tour'] is None
    assert isinstance(seen['oracle_all'], Truth)


def test_known_samples_follow_the_judge(arena):
    sc = generate('medium', 4, arena)
    judge = Judge(sc, arena)
    known = KnownSamples(arena, lambda: list(judge.remaining.values()))
    assert known.left == 5 and len(known.candidates()) == 5
    x, y = sc.samples[0]
    assert known.prob_within(x + 0.1, y, 0.25) == 1.0 and known.prob_within(x + 0.4, y, 0.25) == 0.0
    judge.collect(x, y)
    assert known.left == 4 and known.prob_within(x, y, 0.25) == 0.0
    assert known.grid().sum() == 4


def test_env_truth_knows_changes_ahead_of_time_and_touch_truth_only_after_entry(arena):
    sc = generate('hard', 2, arena)
    change = next(e for e in sc.events if e['type'] == 'soil_change')
    hazard = next(e for e in sc.events if e['type'] == 'new_hazard')
    judge = Judge(sc, arena)
    judge.step(0.0, *sc.base)
    truth = Truth(judge, sc, arena, env=True)
    start = truth.soil_grid()
    assert truth.soil_grid() is start                       # пока ничего не меняется — тот же массив
    X, Y = arena.cell_centers()
    for z in sc.soils:
        assert (start[z.mask(X, Y)] >= z.mult).all()
    judge.t = change['t'] - LEAD_S + 0.1                    # смена ещё не случилась, но оракул её уже учитывает
    ahead = truth.soil_grid()
    assert ahead is not start
    for z in list(sc.soils) + list(change['soils']):
        assert (ahead[z.mask(X, Y)] >= z.mult).all()
    judge.t = change['t'] + 0.1
    after = truth.soil_grid()
    assert after.sum() <= ahead.sum()

    agent = SimpleNamespace(_risk=None, hazards=[], _request_plan=lambda trigger: None)
    truth.refresh(agent, SimpleNamespace(t=0.0))
    assert len(agent.hazards) == len(sc.hazards)
    truth.refresh(agent, SimpleNamespace(t=hazard['t'] - LEAD_S + 0.1))
    assert len(agent.hazards) == len(sc.hazards) + 1
    z = hazard['zone']
    assert agent._risk[arena.w2g(z.x, z.y)[::-1]] == 1.0

    touch = Truth(Judge(sc, arena), sc, arena, env='touch')
    touch.judge.step(0.0, *sc.base)
    assert touch.touched_grid().max() == 1.0                # ни в одну зону не въезжал — пол «обычный»
    z = sc.soils[0]
    touch.judge._pose = (z.x, z.y)
    grid = touch.touched_grid()
    assert (grid[z.mask(X, Y)] >= z.mult).all()
    other = [o for o in sc.soils[1:] if not o.contains(z.x, z.y)]
    assert all((grid[o.mask(X, Y) & ~z.mask(X, Y)] == 1.0).all() for o in other
               if not any(o.mask(X, Y)[z.mask(X, Y)]))


# --- прогоны целиком --------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['adaptive_tour', 'adaptive_tour_full', 'adaptive_survey', 'adaptive_pickup'])
def test_new_variants_run_and_plan_by_tour(name):
    s = run_episode('medium', 7, name, save=False)
    m = s['metrics']
    assert m['reason'] in ('finish', 'battery', 'timeout')
    assert m['samples_collected'] >= 1
    if name != 'adaptive_pickup':
        assert m['plans'].get('tour', 0) >= 1


def test_oracle_all_collects_everything_and_returns():
    for seed in (1, 2):
        m = run_episode('hard', seed, 'oracle_all', save=False)['metrics']
        assert m['samples_collected'] == m['samples_total'] and m['returned'] and m['hazard_hits'] == 0
