"""A1: план на остаток прогона (did/tour.py), оракулы-мерила (did/oracle.py)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.judge import Judge
from did.config import Rules
from did.oracle import LEAD_S, ORACLES, EnvTruth, KnownSamples, SampleTruth, TouchTruth, Truth, make_truth
from did.runner import make_agent, run_episode
from did.scenario import Scenario, generate
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
    truth = make_truth(judge, sc, arena, env=True).env
    agent = SimpleNamespace(_risk=None, hazards=[], _request_plan=lambda trigger: None)
    truth.refresh(agent, SimpleNamespace(t=0.0))
    start = truth.soil_grid()
    assert truth.soil_grid() is start                       # пока ничего не меняется — тот же массив
    X, Y = arena.cell_centers()
    for z in sc.soils:
        assert (start[z.mask(X, Y)] >= z.mult).all()
    truth.refresh(agent, SimpleNamespace(t=change['t'] - LEAD_S + 0.1))   # смены ещё нет, но оракул её уже учитывает
    ahead = truth.soil_grid()
    assert ahead is not start
    for z in list(sc.soils) + list(change['soils']):
        assert (ahead[z.mask(X, Y)] >= z.mult).all()
    truth.refresh(agent, SimpleNamespace(t=change['t'] + 0.1))
    after = truth.soil_grid()
    assert after.sum() <= ahead.sum()

    truth = make_truth(judge, sc, arena, env=True).env
    truth.refresh(agent, SimpleNamespace(t=0.0))
    assert len(agent.hazards) == len(sc.hazards)
    truth.refresh(agent, SimpleNamespace(t=hazard['t'] - LEAD_S + 0.1))
    assert len(agent.hazards) == len(sc.hazards) + 1
    z = hazard['zone']
    assert agent._risk[arena.w2g(z.x, z.y)[::-1]] == 1.0

    tj = Judge(sc, arena)
    touch = make_truth(tj, sc, arena, env='touch').env
    tj.step(0.0, *sc.base)
    assert touch.touched_grid().max() == 1.0                # ни в одну зону не въезжал — пол «обычный»
    z = sc.soils[0]
    tj._pose = (z.x, z.y)
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


# --- исправления по ревью ---------------------------------------------------------------------------

def _bot(arena, name, level, seed, t=10.0, battery=60.0, **config):
    """Агент на базе сценария с посчитанной картой стоимостей; повод для плана снят."""
    sc = generate(level, seed, arena)
    bot = Agent(arena, make_config(name, **config), n_samples=len(sc.samples), rules=Rules())
    obs = SimpleNamespace(t=t, x=bot.base[0], y=bot.base[1], battery=battery, events=[])
    bot._refresh_costs(obs)
    bot._trigger = None
    return sc, bot, obs


def test_current_target_is_dropped_as_soon_as_it_stops_fitting_the_charge(arena):
    # Находка 1: робот на базе, цель — второй образец карты hard-6001, заряда 5 ед. Дорога домой из этой
    # точки ничего не стоит, поэтому прежняя проверка (только когда заряд упал до цены возврата) молчала.
    sc, bot, obs = _bot(arena, 'adaptive_tour', 'hard', 6001, battery=5.0)
    x, y = sc.samples[1]
    bot.queue = [{'type': 'investigate', 'target': 'C1', 'x': x, 'y': y}]
    assert bot.tour._trip(obs, bot.queue[0]) * bot.cfg.reserve_margin + bot.cfg.reserve_abs > 25.0
    bot.tour.check(obs)
    assert bot.queue == [] and bot._trigger == 'battery'
    # При полном заряде та же цель остаётся.
    sc, bot, obs = _bot(arena, 'adaptive_tour', 'hard', 6001)
    bot.queue = [{'type': 'investigate', 'target': 'C1', 'x': x, 'y': y}]
    bot.tour.check(obs)
    assert len(bot.queue) == 1 and bot._trigger is None and not bot._returning


def test_target_check_counts_the_stop_and_the_load(arena):
    # Находка 1: в цену текущей цели входят остановка на сбор и подорожавший с грузом метр дороги домой.
    sc, bot, obs = _bot(arena, 'adaptive_tour', 'hard', 6001)
    x, y = sc.samples[1]
    sg = {'type': 'investigate', 'target': 'C1', 'x': x, 'y': y}
    plain = bot.tour._trip(obs, {**sg, 'type': 'goto'})
    assert bot.tour._trip(obs, sg) == pytest.approx(plain + bot.tour.s.dwell)
    bot.rules = Rules(load_drain=0.5)
    bot.tour.s = Settings(**{**bot.cfg.scheme_opts, 'true_rates': True})
    per_m, per_load = bot.tour._rates()
    back = bot._home_cost(x, y) / bot._per_m()
    assert bot.tour._trip(obs, sg) - bot.tour._trip(obs, {**sg, 'type': 'goto'}) == pytest.approx(
        bot.tour.s.dwell + back * per_load)


def test_rule_targets_are_checked_by_the_rule_formula(arena):
    # Цель выбрало правило (adaptive_pickup): проверка та же, что у правила, — запас только на дорогу домой.
    sc, bot, obs = _bot(arena, 'adaptive_pickup', 'hard', 6001)
    x, y = sc.samples[1]
    sg = {'type': 'explore', 'target': 'E1', 'x': x, 'y': y}
    dist, pred = bot.graph.field(obs.x, obs.y)
    to, back = bot.graph.energy(dist, pred, x, y) * bot._per_m(), bot._home_cost(x, y)
    need = to + back * bot.cfg.reserve_margin + bot.cfg.reserve_abs
    for battery, kept in ((need - bot.tour.s.commit_slack + 0.1, True), (need - bot.tour.s.commit_slack - 0.1, False)):
        bot.queue, bot._trigger = [dict(sg)], None
        bot.tour.check(SimpleNamespace(t=10.0, x=obs.x, y=obs.y, battery=battery, events=[]))
        assert (len(bot.queue) == 1) == kept, battery


def test_survey_stops_when_its_budget_is_spent(arena):
    # Находка 2: бюджет объезда проверялся только при построении плана, а план сразу выдавал все точки.
    sc, bot, obs = _bot(arena, 'adaptive_survey', 'hard', 6011)
    ts = bot.tour
    assert ts._survey and len(ts._wps) > 2
    bot.queue = [{'type': 'goto', 'x': x, 'y': y} for x, y in ts._wps]
    ts.check(obs)
    assert ts._survey and len(bot.queue) == len(ts._wps)            # заряд на объезд ещё есть
    spent = SimpleNamespace(t=13.0, x=obs.x, y=obs.y, battery=bot.rules.battery_start - ts.s.survey_budget - 0.5,
                            events=[])
    ts.check(spent)
    assert not ts._survey and bot.queue == [] and bot._trigger == 'survey'
    assert ts.plan(spent, {'samples': {'total': 7, 'collected': 0}, 'candidates': [], 'trigger': 'survey'}) is None


def test_survey_budget_holds_in_a_whole_run():
    # Тот самый прогон из находки 2: на объезд с бюджетом 6 ед. было потрачено 9.
    s = run_episode('hard', 6011, 'adaptive_survey', save=True, experiment='_test')
    import gzip
    import json
    from did.runner import RUNS
    trace = json.loads(gzip.open(RUNS / s['file'], 'rt', encoding='utf-8').read())
    ends = [e for e in trace['journal'] if 'Обзорный объезд закончен' in e.get('text', '')]
    assert len(ends) == 1
    t_end = ends[0]['t']
    queued = [p for p in trace['plans'] if p['t'] > t_end and p['source'] == 'tour']
    assert not queued                                               # после конца объезда обзорных планов нет
    track, budget = trace['track'], PRESETS['adaptive_survey'].scheme_opts['survey_budget']
    crossed = next(t for t, b in zip(track['t'], track['battery']) if 60.0 - b >= budget)
    assert t_end <= crossed + 1.2                                   # и кончается он не позже чем через секунду


@pytest.mark.parametrize('haste', [0.0, 8.0])
def test_tour_prefers_staying_to_a_trip_that_loses_points(haste):
    # Находка 3: одна цель с уверенностью 3% в 20 м: ожидаемые 0,3 очка против 4 очков заряда на дорогу.
    nodes = [{'id': 'C1', 'kind': 'investigate', 'x': 20.0, 'y': 0.0, 'value': 0.03, 'carry': 0.03,
              'to_m': 20.0, 'back_m': 20.0}]
    ts = _scheme(dwell=0.0, haste=haste)
    order, info = ts.solve(nodes, np.zeros((1, 1)), 60.0, (1.0, 0.0), (1.0, 0.0), 10.0, 0.1)
    assert order == [] and info['score'] == 0.0 and info['why'] == 'gain'
    # Если робот и так в 20 м от базы (цель рядом с ним), дорога домой уже неизбежна — цель берётся.
    nodes[0]['to_m'] = 0.5
    order, info = ts.solve(nodes, np.zeros((1, 1)), 60.0, (1.0, 0.0), (1.0, 0.0), 10.0, 0.1, home=20.0)
    assert order == [0]
    # Не по заряду — другой повод вернуться.
    assert ts.solve(nodes, np.zeros((1, 1)), 10.0, (1.0, 0.0), (1.0, 0.0), 10.0, 0.1)[1]['why'] == 'charge'


def test_haste_does_not_send_home_from_a_target_worth_its_charge():
    # «Спешка» — способ выбрать порядок, а не оценка выгоды: дальний уверенный кандидат окупает дорогу,
    # хотя его ценность «со спешкой» меньше цены пути.
    nodes, metres = _line([15.0], [0.9])
    order, info = _solve(_scheme(dwell=0.0, haste=8.0), nodes, metres, battery=40.0)
    assert order == [0] and info['gain'] == pytest.approx(9.0 - 3.0)


def _reachable(obj, stop=(), cells=True, seen=None):
    """Все объекты, до которых можно дойти от obj по полям, содержимому и замыканиям функций. Объекты типов
    stop попадают в список, но внутрь них обход не идёт."""
    seen = {} if seen is None else seen
    if id(obj) in seen or isinstance(obj, (str, bytes, int, float, bool, type(None), np.ndarray, type)):
        return seen
    seen[id(obj)] = obj
    if isinstance(obj, stop):
        return seen
    if isinstance(obj, dict):
        kids = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        kids = list(obj)
    else:
        kids = list(vars(obj).values()) if hasattr(obj, '__dict__') else []
        if cells and getattr(obj, '__closure__', None):
            kids += [c.cell_contents for c in obj.__closure__]
        if getattr(obj, '__self__', None) is not None:
            kids.append(obj.__self__)
    for k in kids:
        _reachable(k, stop, cells, seen)
    return seen


@pytest.mark.parametrize('name', sorted(ORACLES))
def test_truth_holds_only_what_the_oracle_is_declared_to_know(arena, name):
    # Находка 4: раньше любой Truth держал судью и расписание среды целиком.
    sc = generate('hard', 2, arena)
    judge = Judge(sc, arena)
    judge.step(0.0, *sc.base)
    know = ORACLES[name][2]
    truth = make_truth(judge, sc, arena, **know)
    assert type(truth.samples) is (SampleTruth if know.get('samples') else type(None))
    assert type(truth.env) is {True: EnvTruth, 'touch': TouchTruth, None: type(None)}[know.get('env')]
    assert not hasattr(truth, 'judge') and isinstance(truth, Truth)
    bot = Agent(arena, make_agent(name)[1], n_samples=len(sc.samples), truth=truth)
    bot._refresh_costs(SimpleNamespace(t=0.0, x=sc.base[0], y=sc.base[1], battery=60.0, events=[]))

    shared = set(_reachable(arena))                     # арена общая, правды сценария в ней нет
    whole = (Judge, Scenario)
    objs = [o for i, o in _reachable(truth, stop=whole).items() if i not in shared]
    fields = [o for i, o in _reachable(truth, stop=whole, cells=False).items() if i not in shared]
    # Судьи и сценария нет ни в полях, ни в замыканиях. Исключение — «зона после въезда»: две узкие функции
    # (где робот; какие зоны действуют сейчас) замкнуты на судью, но в полях его нет.
    assert not any(isinstance(o, whole) for o in fields), name
    if know.get('env') != 'touch':
        assert not any(isinstance(o, whole) for o in objs), name
    samples = {tuple(p) for p in sc.samples}
    now = {id(z) for z in sc.soils}
    later = {id(z) for ev in sc.events for z in ev.get('soils', [])}
    hazards = {id(z) for z in sc.hazards} | {id(ev['zone']) for ev in sc.events if 'zone' in ev}
    held = {id(o) for o in objs}
    if not know.get('samples'):                         # мест образцов нет
        assert not any(isinstance(o, tuple) and any(o == p for p in samples) for o in objs), name
        assert id(judge.remaining) not in held, name
    else:
        assert id(judge.remaining) in held
    if know.get('env') is True:                         # всё расписание среды
        assert (now | later | hazards) <= held and later and hazards
    elif know.get('env') == 'touch':                    # ни одной зоны, пока робот в неё не въехал
        assert not (now | later | hazards) & held, name
    else:
        assert not (now | later | hazards) & held, name
