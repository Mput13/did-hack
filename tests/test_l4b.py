"""L4b: память о лаборатории между прогонами (did/labmemory.py) и раскладка образцов sample_seed."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.belief import SoilModel
from did.config import BASE
from did.labmemory import EXPENSIVE, LabMemory, LabSoilModel
from did.robot_io import Observation
from did.runner import run_episode
from did.scenario import generate

LABS = ('adaptive_v2_lab', 'scientist_v2_lab', 'adaptive_llm_lab')


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _numbers(summary):
    return {k: v for k, v in summary['metrics'].items() if isinstance(v, (int, float, bool)) or v is None}


def _cloud(x, y, r=0.29):
    """Гипотезы о зоне, как их оставляет штраф: несколько центров вокруг (x, y)."""
    pts = [(x, y), (x + 0.05, y), (x - 0.05, y), (x, y + 0.05), (x, y - 0.05)]
    return [[px, py, r, 1.0 / len(pts)] for px, py in pts]


def _zone(x, y, conf=0.75, t_from=0.0, zid='Z1'):
    return {'id': zid, 'x': x, 'y': y, 'r': 0.34, 'cloud': _cloud(x, y), 'conf': conf, 'hits': 1, 'runs': ['r1'],
            't_from': t_from}


def _bot(arena, hazards=(), soil=(), volatile=False, name='adaptive_v2_lab', **opts):
    bot = Agent(arena, make_config(name, lab_opts=opts), n_samples=5)
    grid = bot.lab.grid()
    bot = Agent(arena, make_config(name, lab_opts=opts), n_samples=5,
                lab={'runs': 1, 'grid': grid, 'hazards': list(hazards), 'soil': list(soil), 'soil_zones': [],
                     'volatile': volatile})
    return bot


# --- флажок -------------------------------------------------------------------------------------------------

def test_flag_is_off_everywhere_except_lab_presets():
    for name, cfg in PRESETS.items():
        assert cfg.lab_memory == (name in LABS), name
    assert PRESETS['adaptive_v2_lab'] == replace(PRESETS['adaptive_v2'], name='adaptive_v2_lab', lab_memory=True)
    assert PRESETS['scientist_v2_lab'] == replace(PRESETS['scientist_v2'], name='scientist_v2_lab', lab_memory=True)
    assert PRESETS['adaptive_llm_lab'] == replace(PRESETS['adaptive_llm'], name='adaptive_llm_lab', lab_memory=True)


def test_agent_without_flag_has_plain_soil_model_and_no_memory(arena):
    bot = Agent(arena, make_config('adaptive_v2'), n_samples=5, lab={'runs': 3})
    assert bot.lab is None and type(bot.soil) is SoilModel


def test_soil_model_without_prior_equals_plain(arena):
    plain, lab = SoilModel(arena, 2.5, 0.01), LabSoilModel(arena, 2.5, 0.01)
    rng = np.random.default_rng(1)
    for _ in range(60):
        x, y = rng.uniform(-1.5, 1.5, 2)
        seg = (x, y, x + 0.06, y, float(rng.uniform(0.1, 0.5)), 0.3)
        assert plain.observe(*seg) == lab.observe(*seg)
    assert np.array_equal(plain.mult_grid(), lab.mult_grid())
    assert np.array_equal(plain.confidence_grid(), lab.confidence_grid())


@pytest.mark.parametrize('agent,rules', [('adaptive_v2', None), ('scientist_v2', 'science')])
def test_first_run_with_empty_memory_equals_agent_without_memory(agent, rules):
    args = dict(rules=rules, save=False, scenario_args={'sample_seed': 1})
    a, b = run_episode('hard', 3, agent, **args), run_episode('hard', 3, agent + '_lab', **args)
    assert _numbers(a) == _numbers(b)
    assert 'lab' not in a and b['lab']['priors'] == []


# --- условие «та же лаборатория, другие образцы» ----------------------------------------------------------------

@pytest.mark.parametrize('level', ['medium', 'hard'])
def test_sample_seed_changes_only_samples(arena, level):
    for seed in (1, 7, 15003):
        ref = generate(level, seed, arena).to_dict()
        again = generate(level, seed, arena, sample_seed=None).to_dict()
        assert ref == again
        one, two = generate(level, seed, arena, sample_seed=1), generate(level, seed, arena, sample_seed=2)
        for sc in (one, two):
            d = sc.to_dict()
            for key in ('soils', 'hazards', 'events', 'base', 'level', 'seed'):
                assert d[key] == ref[key], key
            assert len(sc.samples) == len(ref['samples'])
            zones = sc.hazards + [e['zone'] for e in sc.events if e['type'] == 'new_hazard']
            for i, p in enumerate(sc.samples):
                assert math.dist(p, BASE) > 1.0 and arena.clearance(*p) >= 0.15
                assert all(math.dist(p, q) > 0.9 for q in sc.samples[:i])
                assert all(math.dist(p, (z.x, z.y)) > 0.65 for z in zones)
        assert one.samples != ref['samples'] and one.samples != two.samples
        assert generate(level, seed, arena, sample_seed=1).samples == one.samples


def test_sample_seed_refuses_route_mode(arena):
    with pytest.raises(ValueError):
        generate('hard', 1, arena, sample_seed=1, soil_change_mode='route')


# --- память сохраняется и читается -------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def first_run():
    """Прогон, в котором робот получил штраф за зону и нашёл дорогой грунт."""
    return run_episode('hard', 3, 'adaptive_v2_lab', rules='science', save=False, scenario_args={'sample_seed': 1})


def test_memory_holds_only_what_robot_found(arena, first_run, tmp_path):
    assert first_run['metrics']['hazard_hits'] >= 1
    mem = LabMemory(tmp_path / 'lab.json')
    mem.learn(first_run['lab'], 'runA')
    mem.save()
    again = LabMemory(tmp_path / 'lab.json')
    assert again.data == json.loads(json.dumps(mem.data)) and again.priors() == json.loads(json.dumps(mem.priors()))
    d = again.data
    assert [r['id'] for r in d['runs']] == ['runA']
    # зона — там, где пришёл штраф, с прогоном-источником; мест образцов в памяти нет
    sc = generate('hard', 3, arena, sample_seed=1)
    zones = sc.hazards + [e['zone'] for e in sc.events if e['type'] == 'new_hazard']
    assert d['hazards'] and all(h['runs'] == ['runA'] and h['evidence'] == 'penalty' for h in d['hazards'])
    for h in d['hazards']:
        assert min(math.dist((h['x'], h['y']), (z.x, z.y)) for z in zones) < 0.45
    assert len(d['hazards']) <= first_run['metrics']['hazard_hits']
    text = (tmp_path / 'lab.json').read_text(encoding='utf-8')
    assert 'sample' not in text
    for z in d['soil_zones']:
        assert z['mult'] >= EXPENSIVE and z['evidence_m'] > 0 and z['runs'] == ['runA']
    assert d['soil_zones'] and len(d['soil']) > 20


def test_run_without_penalty_remembers_no_zones():
    s = run_episode('hard', 3, 'adaptive_v2_lab', save=False, scenario_args={'sample_seed': 1})
    assert s['metrics']['hazard_hits'] == 0 and s['lab']['hazards'] == []      # зоны в сценарии есть, робот их не нашёл
    mem = LabMemory()
    mem.learn(s['lab'], 'r')
    assert mem.priors()['hazards'] == []


def test_memory_ages_and_drops_refuted(first_run):
    mem = LabMemory()
    mem.learn(first_run['lab'], 'r1')
    before = mem.priors()['hazards'][0]
    idle = {**first_run['lab'], 'hazards': [], 'soil': [],
            'priors': [{'id': before['id'], 'status': 'untested', 'x': before['x'], 'y': before['y'], 'r': before['r'],
                        'cloud': before['cloud']}]}
    mem.learn(idle, 'r2')
    after = mem.priors()['hazards'][0]
    assert after['conf'] < before['conf'] and mem.data['hazards'][0]['untested'] == ['r2']
    cell = next(iter(mem.data['soil'].values()))
    assert cell['age'] == 1
    gone = {**idle, 'priors': [{**idle['priors'][0], 'status': 'refuted', 'cloud': []}]}
    mem.learn(gone, 'r3')
    assert mem.priors()['hazards'] == [] and mem.data['dropped'][0]['refuted_by'] == 'r3'


def test_second_penalty_in_same_place_raises_confidence(first_run):
    mem = LabMemory()
    mem.learn(first_run['lab'], 'r1')
    zid = mem.data['hazards'][0]['id']
    again = {**first_run['lab'], 'hazards': [{**first_run['lab']['hazards'][0], 'confirms': zid}],
             'priors': [{'id': zid, 'status': 'confirmed', 'x': 0, 'y': 0, 'r': 0.3, 'cloud': []}]}
    c1 = mem.priors()['hazards'][0]['conf']
    mem.learn(again, 'r2')
    assert len(mem.data['hazards']) == len(first_run['lab']['hazards'])
    assert mem.data['hazards'][0]['runs'] == ['r1', 'r2'] and mem.priors()['hazards'][0]['conf'] > c1


# --- предположения в прогоне: объезд, снятие, подтверждение ----------------------------------------------------------

def test_remembered_zone_is_costly_but_not_forbidden(arena):
    bot = _bot(arena, hazards=[_zone(0.0, 0.0)])
    risk = bot.lab.risk(bot._risk)
    ix, iy = arena.w2g(0.0, 0.0)
    assert 0.5 < risk[iy, ix] <= 0.75
    assert not bot._risk.any()             # свой риск — только по штрафам этого прогона: цели в зоне не запрещены
    bot._refresh_costs(Observation(t=5.0, x=BASE[0], y=BASE[1], th=0.0, v=0.0, w=0.0, battery=60.0, sensor=None,
                                   scan=None))
    node = bot.graph.node(0.0, 0.0)
    assert bot.graph.cost[node] > 10.0 and math.isfinite(bot.graph.plan(BASE, (0.0, 0.0))[1])
    # путь мимо зоны её объезжает, заряд на путь считается без надбавки за риск
    pts, _ = bot.graph.plan((-0.8, 0.0), (0.8, 0.0))
    assert min(math.hypot(x, y) for x, y in pts) > 0.3


def test_safe_pass_refutes_remembered_zone(arena):
    bot = _bot(arena, hazards=[_zone(0.0, 0.0)])
    bot._trail_point(0.30, 0.0, 10.0)      # краем: часть гипотез отпала, зона ещё в силе
    z = bot.lab.zones[0]
    assert z['status'] == 'narrowed' and bot.lab.active()
    before = bot.lab._exists(z)
    assert before < 0.75
    bot._trail_point(0.0, 0.0, 12.0)       # через середину без штрафа: зоны нет
    assert z['status'] == 'refuted' and not bot.lab.active()
    assert bot.lab.risk(bot._risk) is bot._risk
    assert any(e.get('data', {}).get('tag') == 'lab_refuted' for e in bot.journal.entries)
    assert bot.lab.export(20.0)['priors'][0]['status'] == 'refuted'


def test_pass_before_zone_appeared_last_time_does_not_refute(arena):
    bot = _bot(arena, hazards=[_zone(0.0, 0.0, t_from=40.0)])
    bot._trail_point(0.0, 0.0, 10.0)
    assert bot.lab.zones[0]['status'] == 'untested'
    bot._trail_point(0.0, 0.0, 41.0)
    assert bot.lab.zones[0]['status'] == 'refuted'


def test_penalty_confirms_remembered_zone(arena):
    bot = _bot(arena, hazards=[_zone(0.0, 0.0)])
    obs = Observation(t=30.0, x=-0.28, y=0.0, th=0.0, v=0.1, w=0.0, battery=40.0, sensor=None, scan=None,
                      events=[{'type': 'hazard_hit', 't': 30.0, 'x': -0.28, 'y': 0.0}])
    bot._on_event(obs.events[0], obs)
    assert bot.lab.zones[0]['status'] == 'confirmed' and not bot.lab.active()
    out = bot.lab.export(60.0)
    assert out['hazards'][0]['confirms'] == 'Z1' and out['hazards'][0]['hits'] == 1


def test_zone_that_appeared_mid_run_is_marked(arena):
    bot = _bot(arena)
    for k in range(12):                    # проехал это место без штрафа
        bot._trail_point(-0.3 + 0.05 * k, 0.0, 5.0 + k)
    bot._trail.clear()
    obs = Observation(t=50.0, x=-0.3, y=0.0, th=0.0, v=0.1, w=0.0, battery=40.0, sensor=None, scan=None,
                      events=[{'type': 'hazard_hit', 't': 50.0, 'x': -0.3, 'y': 0.0}])
    bot._on_event(obs.events[0], obs)
    zone = bot.hazard_map.zones[0]
    keep = np.abs(zone['c'][:, 1]) < 0.02          # оставим гипотезы «зона прямо по курсу»
    zone['c'], zone['r'], zone['w'] = zone['c'][keep], zone['r'][keep], zone['w'][keep]
    out = bot.lab.export(60.0)['hazards'][0]
    assert out['appeared_after'] is not None and out['appeared_after'] < 50.0
    mem = LabMemory()
    mem.learn({**bot.lab.export(60.0), 'hazards': [out]}, 'r1')
    assert mem.priors()['hazards'][0]['t_from'] == 50.0


def test_soil_prior_yields_to_own_measurements(arena):
    soil = LabSoilModel(arena, 2.5, 0.01)
    iy, ix = soil.cell(0.0, 0.0)
    soil.set_prior([[iy, ix + k, 3.0, 0.7] for k in range(-2, 3)], 0.2)
    mult, conf = soil.at(0.0, 0.0)
    assert mult > 2.2 and conf >= 0.6          # память: дорого, и уверенности хватает, чтобы заметить расхождение
    for k in range(-8, 8):                     # проезд с обычным расходом
        x = 0.05 * k
        soil.observe(x, 0.0, x + 0.05, 0.0, 2.5 * 0.05, 0.0)
    assert soil.at(0.0, 0.0)[0] < 1.25
    assert soil.zones() == []                  # в выгрузку идёт только своё


def test_mismatch_alarm_erases_soil_memory_nearby(arena):
    bot = _bot(arena, soil=[[*SoilModel(arena, 2.5, 0.01).cell(0.0, 0.0), 3.0, 0.7],
                            [*SoilModel(arena, 2.5, 0.01).cell(1.5, 1.5), 3.0, 0.7]])
    assert bot.soil.prior_at(0.0, 0.0) == (3.0, pytest.approx(0.14))
    bot._on_model_mismatch('дешевле', 0.0, 0.0, 1.0, 3.0, 20.0)
    assert bot.soil.prior_at(0.0, 0.0)[1] == 0.0
    assert 0.0 < bot.soil.prior_at(1.5, 1.5)[1] < 0.14      # в остальных местах памяти веры меньше
    assert bot.lab.mismatch == 1
    assert any(e.get('data', {}).get('tag') == 'lab_soil_refuted' for e in bot.journal.entries)


def test_confidence_falls_during_run_only_where_changes_were_seen(arena):
    ix, iy = arena.w2g(0.0, 0.0)
    for volatile in (False, True):
        bot = _bot(arena, hazards=[_zone(0.0, 0.0)], volatile=volatile)
        start = bot.lab.risk(bot._risk)[iy, ix]
        bot.lab.tick(240.0)
        late = bot.lab.risk(bot._risk)[iy, ix]
        assert late == pytest.approx(start * (math.exp(-1) if volatile else 1.0))
    bot = _bot(arena, hazards=[_zone(0.0, 0.0)], volatile=True, decay_s=0.0)
    bot.lab.tick(240.0)
    assert bot.lab.risk(bot._risk)[iy, ix] == pytest.approx(start)


def test_memory_can_be_split_by_kind(arena):
    cell = [*SoilModel(arena, 2.5, 0.01).cell(1.0, 1.0), 3.0, 0.7]
    bot = Agent(arena, make_config('adaptive_v2_lab', lab_opts={'hazards': False}), n_samples=5,
                lab={'runs': 1, 'grid': _bot(arena).lab.grid(), 'hazards': [_zone(0.0, 0.0)], 'soil': [cell],
                     'soil_zones': [], 'volatile': False})
    assert bot.lab.zones == [] and bot.soil.prior_at(1.0, 1.0)[1] > 0
    bot = Agent(arena, make_config('adaptive_v2_lab', lab_opts={'soils': False}), n_samples=5,
                lab={'runs': 1, 'grid': _bot(arena).lab.grid(), 'hazards': [_zone(0.0, 0.0)], 'soil': [cell],
                     'soil_zones': [], 'volatile': False})
    assert len(bot.lab.zones) == 1 and bot.soil.prior_at(1.0, 1.0)[1] == 0


def test_planner_state_carries_memory_summary_only_with_memory(arena):
    obs = Observation(t=5.0, x=BASE[0], y=BASE[1], th=0.0, v=0.0, w=0.0, battery=60.0, sensor=None, scan=None)
    bot = _bot(arena, hazards=[_zone(0.0, 0.0)])
    lm = bot._state(obs, 'start')['lab_memory']
    assert lm['past_runs'] == 1 and lm['hazards'][0]['confidence'] == 0.75 and lm['hazards'][0]['id'] == 'M1'
    fresh = Agent(arena, make_config('adaptive_v2_lab'), n_samples=5)
    assert 'lab_memory' not in fresh._state(obs, 'start')
    assert 'lab_memory' not in Agent(arena, make_config('adaptive_v2'), n_samples=5)._state(obs, 'start')


# --- устаревшая память не запирает робота --------------------------------------------------------------------------

def test_wrong_memory_leaves_base_and_samples_reachable(arena):
    """Память от «другой лаборатории»: зоны прямо на образцах и у базы, дорогой грунт вокруг каждого образца."""
    sc = generate('medium', 4, arena, sample_seed=1)
    grid = _bot(arena).lab.grid()
    hazards = [_zone(x, y, conf=0.9, zid=f'Z{i + 1}') for i, (x, y) in enumerate(sc.samples)]
    hazards.append(_zone(BASE[0] + 0.5, BASE[1], conf=0.9, zid='Z9'))
    soil = [[iy, ix, 4.0, 0.9] for iy in range(grid['h']) for ix in range(grid['w'])
            if any(math.dist((grid['x0'] + (ix + 0.5) * grid['res'], grid['y0'] + (iy + 0.5) * grid['res']), p) < 0.5
                   for p in sc.samples)]
    lab = {'runs': 2, 'grid': grid, 'hazards': hazards, 'soil': soil, 'soil_zones': [], 'volatile': False}
    bot = Agent(arena, make_config('adaptive_v2_lab'), n_samples=len(sc.samples), lab=lab)
    bot._refresh_costs(Observation(t=2.0, x=BASE[0], y=BASE[1], th=0.0, v=0.0, w=0.0, battery=60.0, sensor=None,
                                   scan=None))
    for p in sc.samples:                   # до любой клетки есть путь, и по своему риску она не запрещена
        ix, iy = arena.w2g(*p)
        assert math.isfinite(bot.graph.plan(BASE, tuple(p))[1]) and bot._risk[iy, ix] < 0.5
        assert math.isfinite(bot.home_graph.plan(tuple(p), BASE)[1])
    clean = run_episode('medium', 4, 'adaptive_v2', save=False, scenario=sc)
    wrong = run_episode('medium', 4, 'adaptive_v2_lab', save=False, scenario=sc, lab=lab)
    assert wrong['metrics']['returned'] == 1
    assert wrong['metrics']['samples_share'] >= clean['metrics']['samples_share'] - 0.21     # не больше одного образца
    status = [p['status'] for p in wrong['lab']['priors']]
    assert status.count('refuted') >= 3    # ложные зоны сняты проездом, а не остались навсегда
    mem = LabMemory()
    mem.data.update(runs=[{'id': 'old', 'mismatch': 0, 'zones_refuted': 0, 'soil_cells_changed': 0}], grid=grid,
                    hazards=[{'id': h['id'], 'x': h['x'], 'y': h['y'], 'r': h['r'], 'cloud': h['cloud'], 'age': 0,
                              'hits': 1, 'runs': ['old'], 'untested': [], 't_hit': 0.0, 'appeared_after': None,
                              'evidence': 'penalty'} for h in hazards],
                    soil={f'{iy},{ix}': {'k': 4.0, 'm': 0.2, 'n': 1, 'age': 0, 'first': 0, 'last': 0}
                          for iy, ix, _, _ in soil})
    mem.learn(wrong['lab'], 'new')
    assert len(mem.data['hazards']) <= len(hazards) - 3 and mem.priors()['volatile']
    # клетки, по которым робот проехал, в памяти теперь с его собственным замером
    driven = [c for c in mem.data['soil'].values() if c['last'] == 1]
    assert len(driven) > 20 and np.median([c['k'] for c in driven]) < 2.0
