"""R10: смена грунта «на пути», правка забывания после тревоги, оракул грунта, разность разностей."""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.config import Rules
from did.experiments import _soil_probe_table, summarize_experiment
from did.journal import Journal
from did.judge import Judge
from did.metrics import SoilProbe, changed_distance, paired, paired_did, verdict
from did.nav import CostGraph
from did.runner import ORACLE, _soil_truth, make_agent, run_episode
from did.scenario import Scenario, Zone, generate

SEEDS = [1, 2, 3, 20, 80, 1001, 1040]


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _route_zone(scenario):
    event = next(e for e in scenario.events if e['type'] == 'soil_change')
    return event, event['soils'][-1]


# --- генератор: смена грунта на пути ------------------------------------------------------------

@pytest.mark.parametrize('seed', SEEDS)
def test_route_change_keeps_the_rest_of_the_scenario(arena, seed):
    old = generate('hard', seed, arena)
    new = generate('hard', seed, arena, soil_change_mode='route')
    assert new.to_dict() == generate('hard', seed, arena, soil_change_mode='route').to_dict()
    assert (new.samples, new.soils, new.hazards) == (old.samples, old.soils, old.hazards)
    rest = [e for e in new.events if e['type'] != 'soil_change']
    assert rest == [e for e in old.events if e['type'] != 'soil_change']
    event, zone = _route_zone(new)
    assert 20.0 <= event['t'] <= 40.0
    assert event['soils'][:-1] == new.soils              # прежние зоны на месте, добавилась одна
    assert (zone.shape, zone.mult) == ('circle', 4.0) and 0.4 <= zone.r <= 0.6


@pytest.mark.parametrize('seed', SEEDS)
def test_route_zone_lies_on_a_corridor_and_can_be_bypassed(arena, seed):
    sc = generate('hard', seed, arena, soil_change_mode='route')
    _, zone = _route_zone(sc)
    base, pts = tuple(sc.base), [tuple(p) for p in sc.samples]
    assert not zone.contains(*base) and not any(zone.contains(*p) for p in pts)
    graph = CostGraph(arena)
    corridors = [(base, p) for p in pts] + [(p, min((q for q in pts if q != p), key=lambda q: math.dist(p, q)))
                                            for p in pts]
    crossed = [(a, b) for a, b in corridors if any(zone.contains(x, y) for x, y in graph.plan(a, b)[0])]
    assert crossed                                        # зона перекрывает кратчайший путь хотя бы одного коридора
    X, Y = arena.cell_centers()
    labels, _ = ndimage.label(graph.ok & ~zone.mask(X, Y))
    cells = {int(labels[graph.iy[n], graph.ix[n]]) for n in (graph.node(*p) for p in [base] + pts)}
    assert len(cells) == 1 and 0 not in cells             # база и образцы связаны в обход зоны
    graph.set_cost(np.where(zone.mask(X, Y), 4.0, 1.0))
    assert any(not any(zone.contains(x, y) for x, y in graph.plan(a, b)[0]) for a, b in crossed)   # объезд выгоден


@pytest.mark.parametrize('seed', [26, 2, 5, 1001])
def test_route_zone_lies_on_the_chosen_corridor(arena, seed, monkeypatch):
    # Отклонённый кандидат не должен влиять на следующие коридоры: исходный путь выбранного коридора
    # считается по карте без зон-кандидатов и проходит через принятую зону (сценарий 26 это нарушал:
    # коридор от базы строился по графу с чужой зоной ×4, и принятая зона лежала мимо него).
    calls = []
    plan = CostGraph.plan

    def spy(self, a, b, *args, **kwargs):
        calls.append((self, tuple(a), tuple(b), float(self.mult.max())))
        return plan(self, a, b, *args, **kwargs)
    monkeypatch.setattr(CostGraph, 'plan', spy)
    sc = generate('hard', seed, arena, events=['soil_change'], soil_change_mode='route')
    monkeypatch.undo()
    _, zone = _route_zone(sc)
    detour, a, b, _ = calls[-1]                           # последним считается объезд принятой зоны
    assert any(zone.contains(x, y) for x, y in CostGraph(arena).plan(a, b)[0])
    originals = [c for c in calls if c[0] is not detour]
    assert originals and all(c[3] == 1.0 for c in originals)        # исходные коридоры — по неизменённому графу
    assert (a, b) in {(c[1], c[2]) for c in originals}
    X, Y = arena.cell_centers()
    cost = np.ones(X.shape)
    for z in sc.events[0]['soils']:
        cost = np.where(z.mask(X, Y), np.maximum(cost, z.mult), cost)
    graph = CostGraph(arena)
    graph.set_cost(cost)
    assert not any(zone.contains(x, y) for x, y in graph.plan(a, b)[0])     # объезд именно этого коридора выгоден


def test_route_change_uses_its_own_random_stream(arena):
    # Сторона, место и момент не зависят от того, сколько чисел взяла прежняя ветка генератора.
    a = generate('hard', 5, arena, events=['soil_change'], soil_change_mode='route')
    b = generate('hard', 5, arena, soil_change_mode='route')
    assert _route_zone(a) == _route_zone(b)
    with pytest.raises(ValueError):
        generate('hard', 5, arena, soil_change_mode='path')


def test_route_change_respects_event_list_and_count(arena):
    assert generate('hard', 5, arena, events=[], soil_change_mode='route').events == []
    sc = generate('hard', 5, arena, events=['soil_change'], soil_change_mode='route', n_soil_changes=2)
    first, second = sc.events
    assert first['t'] <= second['t'] and len(second['soils']) == len(first['soils']) + 1 == len(sc.soils) + 2


# --- правка забывания --------------------------------------------------------------------------

def _alarm_on_known_floor(arena, name, segments=4, **config):
    """Робот знает участок как обычный пол (2 м данных), затем 0,24 м по нему стоят вчетверо дороже."""
    bot = Agent(arena, make_config(name, **config), 7)
    y = -0.5
    for x in np.arange(-1.9, 0.1, 0.05):
        bot.soil.add(x, y, 0.05, 1.0)
    before = bot.soil.dist.sum()
    for i in range(segments):
        x0 = -1.50 + 0.06 * i
        bot._on_segment(x0, y, x0 + 0.06, y, 4.0 * 2.5 * 0.06, 0.0, 10.0 + i)
    return bot, before


def test_fix_is_off_by_default():
    assert [name for name, cfg in PRESETS.items() if cfg.fresh_soil] == ['adaptive_fresh']
    assert make_config('adaptive_fresh') == replace(make_config('adaptive'), name='adaptive_fresh', fresh_soil=True)


def test_alarm_is_raised_and_old_forgetting_loses_fresh_evidence(arena):
    old, _ = _alarm_on_known_floor(arena, 'adaptive', segments=2)        # тревога на втором отрезке, после 0,12 м
    new, _ = _alarm_on_known_floor(arena, 'adaptive_fresh', segments=2)
    for bot in (old, new):
        assert [e['data']['tag'] for e in bot.journal.entries if e['kind'] == 'alarm'] == ['model_mismatch']
        assert bot._trigger == 'model_mismatch' and bot._cost_dirty
    # Там, где робот только что видел расход ×4: прежнее забывание возвращает оценку к обычному полу
    # (ниже порога 1,45, с которого участок считается дорогим), правка оставляет её дорогой.
    assert old.soil.at(-1.42, -0.5)[0] < 1.45
    assert new.soil.at(-1.42, -0.5)[0] > 2.4


def test_fresh_evidence_is_added_once_and_far_data_only_weakened(arena):
    bot, before = _alarm_on_known_floor(arena, 'adaptive_fresh')
    alarm = next(e for e in bot.journal.entries if e['kind'] == 'alarm')
    ys = bot.soil.y0 + (np.arange(bot.soil.h) + 0.5) * bot.soil.res
    xs = bot.soil.x0 + (np.arange(bot.soil.w) + 0.5) * bot.soil.res
    near = np.hypot(xs[None, :] - alarm['data']['x'], ys[:, None] - alarm['data']['y']) <= 0.7
    segments = round((alarm['t'] - 10.0) + 1)             # отрезков до тревоги включительно
    after_alarm = 4 - segments                            # отрезки после тревоги ложатся в карту как обычно
    window = 0.06 * min(segments, 3)                      # окно 0,15 м — три отрезка по 0,06 м
    assert bot.soil.dist[near].sum() == pytest.approx(window + 0.06 * after_alarm)
    assert bot.soil.drain[near].sum() == pytest.approx(4.0 * (window + 0.06 * after_alarm))
    kept = bot.soil.dist[~near].sum()
    assert 0.0 < kept < 0.6 * before                      # дальние данные ослаблены, как в прежнем забывании
    far, _ = _alarm_on_known_floor(arena, 'adaptive_fresh', fresh_keep_far=1.0)
    assert far.soil.dist[~near].sum() == pytest.approx(kept / 0.6)


def test_window_keeps_only_recent_path(arena):
    bot = Agent(arena, make_config('adaptive_fresh', detect_change=False), 7)
    for i in range(30):
        x0 = -1.9 + 0.06 * i
        bot._on_segment(x0, -0.5, x0 + 0.06, -0.5, 2.5 * 0.06, 0.0, float(i))
    assert 0.15 <= sum(s[2] for s in bot._fresh) < 0.21 + 1e-9
    assert bot._fresh[-1][0] == pytest.approx(-1.9 + 0.06 * 29.5)
    assert not Agent(arena, make_config('adaptive'), 7)._fresh


# --- оракул грунта -----------------------------------------------------------------------------

def test_oracle_exists_only_for_fast_simulator():
    assert ORACLE not in PRESETS
    with pytest.raises(KeyError):
        make_config(ORACLE)
    cls, cfg = make_agent(ORACLE)
    assert cls is Agent and cfg == replace(make_config('no_change'), name=ORACLE)


def test_oracle_truth_is_current_soil_only(arena):
    zone, late = Zone('A', 'circle', 0.0, 0.0, r=0.5, mult=3.0), Zone('B', 'circle', 1.0, 1.0, r=0.4, mult=4.0)
    sc = Scenario('hard', 1, samples=[[1.5, 0.0]], soils=[zone],
                  events=[{'t': 5.0, 'type': 'soil_change', 'soils': [zone, late]}])
    judge = Judge(sc, arena, Rules())
    truth = _soil_truth(judge, arena)
    X, Y = arena.cell_centers()
    first = truth()
    assert first is truth()                                              # пока грунты те же — тот же массив
    assert np.array_equal(first, np.where(zone.mask(X, Y), 3.0, 1.0))    # будущей зоны B в нём нет
    judge.step(6.0, -2.0, -0.5)
    assert truth() is not first and truth()[late.mask(X, Y)].min() == 4.0


def test_oracle_plans_by_truth_and_replans_on_change(arena):
    grids = [np.ones(arena.free.shape), np.full(arena.free.shape, 4.0)]
    state = {'now': grids[0]}
    bot = Agent(arena, make_agent(ORACLE)[1], 7, soil_truth=lambda: state['now'])
    bot._trigger = None
    bot._refresh_costs(SimpleNamespace(t=0.0))
    assert np.all(bot.graph.mult == 1.0) and np.all(bot.home_graph.mult == 1.0) and bot._trigger is None
    assert np.array_equal(bot.home_graph.cost, bot.graph.cost)           # без надбавки за непроверенный пол
    state['now'] = grids[1]
    bot._refresh_costs(SimpleNamespace(t=20.0))
    assert np.all(bot.graph.mult == 4.0) and np.all(bot.home_graph.mult == 4.0)
    assert bot._trigger == 'soil_truth'


def test_oracle_episode_avoids_new_zone_more_than_blind_agent():
    args = {'events': ['soil_change'], 'soil_change_mode': 'route'}
    runs = {a: [run_episode('hard', s, a, scenario_args=args, save=False, soil_probe=True)['metrics']
                for s in (2, 3)] for a in ('no_change', ORACLE)}
    assert all('soil_extra_energy' in m and m['soil_alarms'] == 0 for ms in runs.values() for m in ms)
    assert (sum(m['soil_dearer_m'] for m in runs[ORACLE]) <= sum(m['soil_dearer_m'] for m in runs['no_change']))
    assert 'soil_extra_energy' not in run_episode('hard', 2, 'no_change', save=False)['metrics']


# --- разбор по скрытой правде ------------------------------------------------------------------

def test_soil_probe_counts_cost_of_change_and_sorts_alarms():
    new = Zone('R1', 'circle', 0.0, 0.0, r=0.5, mult=4.0)
    cheap = Zone('A', 'circle', 2.0, 0.0, r=0.5, mult=3.0)
    sc = Scenario('hard', 1, samples=[], soils=[cheap],
                  events=[{'t': 10.0, 'type': 'soil_change', 'soils': [Zone('A', 'circle', 2.0, 0.0, r=0.5, mult=1.5), new]}])
    probe = SoilProbe(sc, Rules())
    probe.step((-0.2, 0.0), (0.2, 0.0), sc.soils, 5.0)             # до события: не считается
    after = sc.events[0]['soils']
    probe.step((-1.0, 0.0), (-0.8, 0.0), after, 12.0)             # обычный пол
    probe.step((-0.2, 0.0), (0.2, 0.0), after, 14.0)              # 0,4 м по новой зоне ×4
    probe.step((1.9, 0.0), (2.1, 0.0), after, 20.0)               # 0,2 м по подешевевшей зоне
    journal = Journal()
    for t, x in ((3.0, 0.0), (15.0, 0.1), (16.0, -1.5), (30.0, 0.7)):
        journal.add(t, 'alarm', 'тревога', tag='model_mismatch', x=x, y=0.0)
    m = probe.metrics(journal)
    assert m['soil_dearer_m'] == pytest.approx(0.4) and m['soil_changed_m'] == pytest.approx(0.6)
    assert m['soil_extra_energy'] == pytest.approx(2.5 * (0.4 * 3.0 - 0.2 * 1.5))
    # До события и вдали от изменившегося пола — ложные; на зоне и в 0,2 м от её края — по делу.
    assert (m['soil_alarms'], m['soil_alarms_true'], m['soil_alarms_false']) == (4, 2, 2)
    assert m['soil_alarm_delay'] == pytest.approx(1.0)


def _probe(start, after, t=1.0):
    sc = Scenario('hard', 1, samples=[], soils=start, events=[{'t': t, 'type': 'soil_change', 'soils': after}])
    return SoilProbe(sc, Rules())


def test_alarm_near_changed_floor_is_not_missed_between_probe_points():
    # Новая зона ×4 в 0,29 м от места тревоги, в стороне от прежних 24 пробных точек: тревога по делу.
    zone = Zone('R1', 'circle', 0.79 * math.cos(math.pi / 12), 0.79 * math.sin(math.pi / 12), r=0.5, mult=4.0)
    probe = _probe([], [zone])
    assert changed_distance([], [zone], 0.0, 0.0) == pytest.approx(0.29)
    assert probe._changed_near(0.0, 0.0, 2.0)
    assert not probe._changed_near(0.0, 0.0, 0.5)                       # до события изменившегося пола нет
    far = Zone('R1', 'circle', 0.81, 0.0, r=0.5, mult=4.0)              # 0,31 м — уже не «рядом»
    assert not _probe([], [far])._changed_near(0.0, 0.0, 2.0)


def test_changed_distance_respects_overlaps_and_rectangles():
    old = Zone('A', 'circle', 0.0, 0.0, r=0.5, mult=4.0)
    new = Zone('R1', 'circle', 0.6, 0.0, r=0.5, mult=4.0)               # часть новой зоны лежит на прежней ×4
    # Слева от прежней зоны: ближайший изменившийся пол — за ней, там, где новая зона выходит из-под прежней.
    assert changed_distance([old], [old, new], -0.6, 0.0) == pytest.approx(math.hypot(0.9, 0.4))
    assert changed_distance([old], [old, new], 0.2, 0.0) == pytest.approx(0.3)      # внутри перекрытия: до края прежней
    assert changed_distance([old], [old, new], 0.8, 0.0) == 0.0
    assert changed_distance([old], [old], 0.8, 0.0) == math.inf                     # ничего не изменилось
    # Прямоугольник сменил цену: расстояние до стороны и до угла.
    rect, cheap = Zone('B', 'rect', 2.0, 0.0, w=1.0, h=0.6, mult=3.0), Zone('B', 'rect', 2.0, 0.0, w=1.0, h=0.6, mult=1.5)
    assert changed_distance([rect], [cheap], 1.3, 0.1) == pytest.approx(0.2)
    assert changed_distance([rect], [cheap], 1.2, 0.7) == pytest.approx(math.hypot(0.3, 0.4))
    # Зона переехала: изменились и старое место (подешевело), и новое (подорожало).
    moved = Zone('A', 'circle', 3.0, 3.0, r=0.4, mult=4.0)
    assert changed_distance([old], [moved], 0.0, 0.7) == pytest.approx(0.2)
    assert changed_distance([old], [moved], 3.0, 2.0) == pytest.approx(0.6)


def test_changed_distance_matches_dense_grid():
    rng = np.random.default_rng(3)

    def zone(i, mult):
        x, y = rng.uniform(-1.0, 1.0, 2)
        if rng.random() < 0.5:
            return Zone(f'Z{i}', 'circle', x, y, r=rng.uniform(0.3, 0.6), mult=mult)
        return Zone(f'Z{i}', 'rect', x, y, w=rng.uniform(0.5, 1.2), h=rng.uniform(0.5, 1.2), mult=mult)
    xs = np.arange(-2.5, 2.5, 0.004)
    X, Y = np.meshgrid(xs, xs)

    def grid(zones):
        m = np.ones(X.shape)
        for z in zones:
            m = np.where(z.mask(X, Y), np.maximum(m, z.mult), m)
        return m
    for _ in range(12):
        start = [zone(i, float(rng.choice([2.0, 3.0, 4.0]))) for i in range(3)]
        after = [replace(start[0], mult=4.0 if start[0].mult < 4.0 else 1.5), start[1], zone(2, start[2].mult),
                 zone(3, 4.0)]
        changed = grid(start) != grid(after)
        for px, py in rng.uniform(-2.0, 2.0, (25, 2)):
            brute = float(np.hypot(X[changed] - px, Y[changed] - py).min())
            assert changed_distance(start, after, px, py) == pytest.approx(brute, abs=0.008)


# --- разность разностей и допуск ---------------------------------------------------------------

def _runs(values, metric='score'):
    return [{'level': 'hard', 'seed': seed, 'metrics': {metric: value}} for seed, value in values]


def test_paired_did_is_difference_of_paired_differences():
    a, b = _runs([(1, 8), (2, 15), (3, 9), (4, 99)]), _runs([(1, 3), (2, 8), (3, 9)])
    base_a, base_b = _runs([(1, 5), (2, 6), (3, 4)]), _runs([(1, 2), (2, 1), (3, 5), (5, 0)])
    did = paired_did(a, b, base_a, base_b, 'score')
    # сценарии 4 и 5 пройдены не всеми четырьмя группами и в расчёт не идут
    assert did['n'] == 3 and did['mean'] == pytest.approx(round(((5 - 3) + (7 - 5) + (0 + 1)) / 3, 3))
    assert (did['a_higher'], did['b_higher'], did['ties']) == (3, 0, 0)
    assert did['ci'][0] <= did['mean'] <= did['ci'][1]
    assert did['mean'] == pytest.approx(paired(a, b, 'score')['mean'] - paired(base_a, base_b, 'score')['mean'], abs=2e-3)
    assert paired_did(a, b, base_a, _runs([(9, 1)]), 'score') is None


def test_did_interval_is_narrower_when_noise_is_shared():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 10, 40)                          # особенность сценария: одна и та же в обоих условиях
    a = _runs([(s, 3 + noise[s] + rng.normal(0, 0.1)) for s in range(40)])
    b = _runs([(s, rng.normal(0, 0.1)) for s in range(40)])
    base_a = _runs([(s, noise[s] + rng.normal(0, 0.1)) for s in range(40)])
    base_b = _runs([(s, rng.normal(0, 0.1)) for s in range(40)])
    did = paired_did(a, b, base_a, base_b, 'score')
    plain = paired(a, b, 'score')
    assert did['ci'][0] > 2.5 and did['ci'][1] < 3.5
    assert did['ci'][1] - did['ci'][0] < 0.1 * (plain['ci'][1] - plain['ci'][0])


def test_verdict_kinds():
    assert verdict({'ci': [0.1, 2.0]}, 'higher') == 'supported'
    assert verdict({'ci': [-0.1, 2.0]}, 'higher') == 'inconclusive'
    assert verdict(None, 'higher', 'noninferiority', 1) == 'no_data'
    # «не хуже» с допуском 1: важна только нижняя граница
    assert verdict({'ci': [-0.5, 2.5]}, 'higher', 'noninferiority', 1) == 'supported'
    assert verdict({'ci': [-2.0, 0.0]}, 'higher', 'noninferiority', 1) == 'inconclusive'
    assert verdict({'ci': [-3.0, -2.0]}, 'higher', 'noninferiority', 1) == 'refuted'
    assert verdict({'ci': [-2.5, 0.5]}, 'lower', 'noninferiority', 1) == 'supported'
    # «не отличается»: интервал целиком внутри допуска
    assert verdict({'ci': [-0.5, 0.9]}, 'higher', 'equivalence', 1) == 'supported'
    assert verdict({'ci': [-0.5, 2.5]}, 'higher', 'equivalence', 1) == 'inconclusive'
    assert verdict({'ci': [1.5, 2.5]}, 'higher', 'equivalence', 1) == 'refuted'


def _toy_results():
    score = {('a', 'none'): 1.5, ('b', 'none'): 1.0, ('a', 'route'): 5.5, ('b', 'route'): 3.0}
    return [{'arm': arm, 'condition': cond, 'level': 'hard', 'seed': seed, 'metrics': {'score': v + 0.1 * seed}}
            for seed in (1, 2, 3) for (arm, cond), v in score.items()]


def test_claims_with_minus_and_margin_in_summary():
    spec = {'arms': [{'id': 'a'}, {'id': 'b'}], 'conditions': [{'id': 'none'}, {'id': 'route'}], 'levels': ['hard'],
            'claims': [{'a': 'a', 'b': 'b', 'metric': 'score'},
                       {'a': 'a', 'b': 'b', 'metric': 'score', 'minus': 'none'},
                       {'a': 'a', 'b': 'b', 'metric': 'score', 'kind': 'noninferiority', 'margin': 1,
                        'conditions': ['none']}]}
    out = summarize_experiment(spec, _toy_results(), 3, 0.0)
    plain, did, not_worse = out['claims']
    assert [(c['condition'], c['pair']['mean']) for c in plain['cells']] == [('none', 0.5), ('route', 2.5)]
    assert [(c['condition'], c['pair']['mean']) for c in did['cells']] == [('route', 2.0)]
    assert did['status'] == 'supported'
    assert [c['condition'] for c in not_worse['cells']] == ['none'] and not_worse['status'] == 'supported'
    assert 'soil_probe' not in out and 'generator' not in out


def test_old_style_summary_is_untouched_by_new_claim_kinds():
    # Утверждения прежнего вида считаются теми же вызовами и тем же потоком случайных чисел, что и раньше.
    spec = {'arms': [{'id': 'a'}, {'id': 'b'}], 'conditions': [{'id': 'none'}, {'id': 'route'}], 'levels': ['hard'],
            'claims': [{'a': 'a', 'b': 'b', 'metric': 'score'}]}
    rng = np.random.default_rng(1)
    results = [{'arm': arm, 'condition': cond, 'level': 'hard', 'seed': seed, 'metrics': {'score': float(rng.normal())}}
               for seed in range(12) for arm in 'ab' for cond in ('none', 'route')]
    out = summarize_experiment(spec, results, 12, 0.0)

    def pick(arm, cond):
        return [r for r in results if r['arm'] == arm and r['condition'] == cond]
    ref = np.random.default_rng(0)
    for arm in 'ab':                                   # сводка по группам берёт числа из того же потока раньше
        for cond in ('none', 'route'):
            for _ in ('hard', 'all'):
                ref.choice(np.zeros(12), size=(2000, 12), replace=True)
    for cell in out['claims'][0]['cells']:
        d = np.array([x['metrics']['score'] - y['metrics']['score']
                      for x, y in zip(pick('a', cell['condition']), pick('b', cell['condition']))])
        boot = ref.choice(d, size=(4000, 12), replace=True).mean(axis=1)
        assert cell['pair']['ci'] == [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


# --- доля закрытого разрыва --------------------------------------------------------------------

def _gap_rows(control, oracle, variant):
    blank = {'soil_dearer_m': 0.0, 'soil_changed_m': 0.0, 'soil_extra_energy': 0.0, 'soil_alarm_delay': None,
             'soil_alarms_true': 0, 'soil_alarms_false': 0, 'soil_entry_known': None, 'soil_dearer_entries': 0}
    runs = [{'arm': arm, 'condition': 'route', 'level': 'hard', 'seed': seed, 'metrics': {**blank, 'score': v}}
            for arm, values in (('control', control), ('oracle', oracle), ('variant', variant))
            for seed, v in enumerate(values)]
    spec = {'arms': [{'id': 'control'}, {'id': 'oracle'}, {'id': 'variant'}], 'conditions': [{'id': 'route'}],
            'soil_probe': {'control': 'control', 'oracle': 'oracle'}}
    rows = _soil_probe_table(spec, lambda arm, cond: [r for r in runs if r['arm'] == arm and r['condition'] == cond])
    return {r['arm']: r for r in rows}


def test_gap_closed_is_not_reported_when_the_gap_itself_is_not_established():
    # Оракул лучше контроля в среднем на 0,5, но в четверти бутстреп-выборок разрыв равен −1:
    # раньше такие выборки молча выбрасывались и получался «95% интервал» доли [0,5; 2,0].
    row = _gap_rows(control=[0, 0], oracle=[-1, 2], variant=[1, 1])['variant']
    assert row['gap'] == 0.5 and row['gap_ci'][0] <= 0.0 and row['gap_stable'] is False
    assert 'gap_closed' not in row and 'gap_closed_ci' not in row
    rows = _gap_rows(control=[0, 0], oracle=[-1, -2], variant=[1, 1])
    assert rows['variant']['gap_stable'] is False and 'gap_closed' not in rows['variant']
    assert 'gap' not in rows['control'] and 'gap' not in rows['oracle']


def test_gap_closed_with_interval_when_the_gap_is_stably_positive():
    rng = np.random.default_rng(5)
    control = rng.normal(70, 5, 40)
    oracle = control + rng.normal(8, 2, 40)
    row = _gap_rows(control, oracle, control + 0.25 * (oracle - control))['variant']
    assert row['gap_stable'] is True and row['gap_ci'][0] > 0 and row['gap_boot_nonpositive'] == 0.0
    assert row['gap_closed'] == pytest.approx(0.25) and row['gap_closed_ci'] == pytest.approx([0.25, 0.25])
    noisy = _gap_rows(control, oracle, control + rng.normal(2, 4, 40))['variant']
    assert noisy['gap_closed_ci'][0] < noisy['gap_closed'] < noisy['gap_closed_ci'][1]
