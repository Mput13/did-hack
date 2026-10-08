"""Unit tests for score_hypotheses against hidden ground truth."""
import pytest
from did.metrics import score_hypotheses
from did.scenario import Scenario, Zone


def make_test_scenario(samples=None, soils=None, hazards=None, events=None):
    return Scenario(
        level='medium',
        seed=1,
        samples=samples or [],
        soils=soils or [],
        hazards=hazards or [],
        events=events or [],
        base=(-2.0, -0.5),
    )


# --- SAMPLE HYPOTHESES ---

def test_sample_hypothesis_correct_when_uncollected():
    sc = make_test_scenario(samples=[[1.0, 1.0], [-1.0, 0.5]])
    hyp = [{
        'id': 'H1', 'key': 'sample:1', 't_open': 10.0,
        'statement': 'образец лежит около (1.1; 0.9)',
        'status': 'confirmed', 'data': {'x': 1.1, 'y': 0.9}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'correct'
    assert res['correct'] == 1
    assert res['confirmed_correct'] == 1.0


def test_sample_hypothesis_wrong_too_far():
    sc = make_test_scenario(samples=[[1.0, 1.0]])
    hyp = [{
        'id': 'H1', 'key': 'sample:1', 't_open': 10.0,
        'statement': 'образец лежит около (1.5; 1.5)',
        'status': 'refuted', 'data': {'x': 1.5, 'y': 1.5}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'wrong'
    assert res['wrong'] == 1
    assert res['refuted_correct'] == 0.0


def test_sample_hypothesis_already_collected_is_wrong():
    sc = make_test_scenario(samples=[[1.0, 1.0]])
    events = [{'type': 'sample_collected', 't': 8.0, 'sample': 0, 'x': 1.0, 'y': 1.0}]
    hyp = [{
        'id': 'H1', 'key': 'sample:1', 't_open': 12.0,
        'statement': 'образец лежит около (1.0; 1.0)',
        'status': 'open', 'data': {'x': 1.0, 'y': 1.0}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=events)
    assert hyp[0]['truth_verdict'] == 'wrong'
    assert res['wrong'] == 1


def test_sample_hypothesis_collected_after_open_is_correct():
    sc = make_test_scenario(samples=[[1.0, 1.0]])
    events = [{'type': 'sample_collected', 't': 15.0, 'sample': 0, 'x': 1.0, 'y': 1.0}]
    hyp = [{
        'id': 'H1', 'key': 'sample:1', 't_open': 10.0,
        'statement': 'образец лежит около (1.0; 1.0)',
        'status': 'confirmed', 'data': {'x': 1.0, 'y': 1.0}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=events)
    assert hyp[0]['truth_verdict'] == 'correct'


# --- SOIL HYPOTHESES ---

def test_soil_hypothesis_correct_and_relative_error():
    zone = Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=3.0)
    sc = make_test_scenario(soils=[zone])
    hyp = [{
        'id': 'H1', 'key': 'soil:5:0.1:0.1', 't_open': 5.0,
        'statement': 'около (0.1; 0.1) грунт дороже обычного примерно в 3.0 раза',
        'status': 'confirmed', 'data': {'x': 0.1, 'y': 0.1, 'mult': 3.0}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'correct'
    assert hyp[0]['rel_error'] == pytest.approx(0.0)
    assert res['soil_error'] == pytest.approx(0.0)


def test_soil_hypothesis_wrong_on_normal_ground():
    sc = make_test_scenario(soils=[Zone(id='A', shape='circle', x=-1.0, y=-1.0, r=0.3, mult=2.0)])
    hyp = [{
        'id': 'H1', 'key': 'soil:5:1.5:1.5', 't_open': 5.0,
        'statement': 'около (1.5; 1.5) грунт дороже обычного примерно в 2.5 раза',
        'status': 'refuted', 'data': {'x': 1.5, 'y': 1.5, 'mult': 2.5}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'wrong'
    # true_mult is 1.0, named 2.5 -> rel_error = |2.5 - 1.0| / 1.0 = 1.5
    assert hyp[0]['rel_error'] == pytest.approx(1.5)


def test_soil_changed_before_and_after():
    z_init = Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=1.5)
    z_after = Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=4.0)
    events = [{'t': 20.0, 'type': 'soil_change', 'soils': [z_after]}]
    sc = make_test_scenario(soils=[z_init], events=events)
    world = [{'t': 20.0, 'type': 'soil_change'}]

    # Before soil change: true mult is 1.5
    hyp_before = {
        'id': 'H1', 'key': 'soil:10:0.0:0.0', 't_open': 10.0,
        'statement': 'около (0.0; 0.0) грунт дороже обычного примерно в 1.5 раза',
        'status': 'confirmed', 'data': {'x': 0.0, 'y': 0.0, 'mult': 1.5}
    }
    # After soil change: true mult is 4.0
    hyp_after = {
        'id': 'H2', 'key': 'soil:25:0.0:0.0', 't_open': 25.0,
        'statement': 'около (0.0; 0.0) грунт дороже обычного примерно в 4.0 раза',
        'status': 'confirmed', 'data': {'x': 0.0, 'y': 0.0, 'mult': 4.0}
    }
    score_hypotheses([hyp_before, hyp_after], sc, world=world, events=[])
    assert hyp_before['truth_verdict'] == 'correct'
    assert hyp_before['rel_error'] == pytest.approx(0.0)
    assert hyp_after['truth_verdict'] == 'correct'
    assert hyp_after['rel_error'] == pytest.approx(0.0)


# --- HAZARD HYPOTHESES ---

def test_hazard_hypothesis_correct_intersection():
    hz = Zone(id='X1', shape='circle', x=0.5, y=0.5, r=0.3)
    sc = make_test_scenario(hazards=[hz])
    hyp = [{
        'id': 'H1', 'key': 'hazard:1', 't_open': 10.0,
        'statement': 'около (0.7; 0.6) опасная зона радиусом около 0.3 м',
        'status': 'confirmed', 'data': {'x': 0.7, 'y': 0.6, 'r': 0.3}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'correct'
    assert res['confirmed_correct'] == 1.0


def test_hazard_hypothesis_wrong_far_away():
    hz = Zone(id='X1', shape='circle', x=0.5, y=0.5, r=0.3)
    sc = make_test_scenario(hazards=[hz])
    hyp = [{
        'id': 'H1', 'key': 'hazard:1', 't_open': 10.0,
        'statement': 'около (-1.5; -1.5) опасная зона радиусом около 0.3 м',
        'status': 'refuted', 'data': {'x': -1.5, 'y': -1.5, 'r': 0.3}
    }]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'wrong'


def test_hazard_new_event_timing():
    new_hz = Zone(id='X2', shape='circle', x=1.0, y=1.0, r=0.3)
    events = [{'t': 30.0, 'type': 'new_hazard', 'zone': new_hz}]
    sc = make_test_scenario(hazards=[], events=events)
    world = [{'t': 30.0, 'type': 'new_hazard'}]

    hyp_early = {
        'id': 'H1', 'key': 'hazard:1', 't_open': 20.0,
        'statement': 'около (1.0; 1.0) опасная зона радиусом около 0.3 м',
        'status': 'open', 'data': {'x': 1.0, 'y': 1.0, 'r': 0.3}
    }
    hyp_late = {
        'id': 'H2', 'key': 'hazard:2', 't_open': 35.0,
        'statement': 'около (1.0; 1.0) опасная зона радиусом около 0.3 м',
        'status': 'confirmed', 'data': {'x': 1.0, 'y': 1.0, 'r': 0.3}
    }
    score_hypotheses([hyp_early, hyp_late], sc, world=world, events=[])
    assert hyp_early['truth_verdict'] == 'wrong'
    assert hyp_late['truth_verdict'] == 'correct'


# --- SENSOR HYPOTHESES ---

def test_sensor_hypothesis_active_and_recovered():
    world = [
        {'t': 20.0, 'type': 'fault', 'kind': 'sensor_noise', 'until': 50.0},
        {'t': 50.0, 'type': 'sensor_recovered', 'kind': 'sensor_noise'},
    ]
    sc = make_test_scenario()

    hyp_during = {
        'id': 'H1', 'key': 'sensor', 't_open': 30.0,
        'statement': 'датчик образцов неисправен',
        'status': 'confirmed'
    }
    hyp_after = {
        'id': 'H2', 'key': 'sensor', 't_open': 65.0,
        'statement': 'датчик образцов неисправен',
        'status': 'confirmed'
    }
    score_hypotheses([hyp_during, hyp_after], sc, world=world, events=[])
    assert hyp_during['truth_verdict'] == 'correct'
    assert hyp_after['truth_verdict'] == 'wrong'


# --- AGGREGATES & UNVERIFIABLE ---

def test_unknown_hypothesis_unverifiable():
    sc = make_test_scenario()
    hyp = [{'id': 'H1', 'key': 'law:per_rad', 't_open': 10.0, 'statement': 'повороты стоят заряда', 'status': 'confirmed'}]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert hyp[0]['truth_verdict'] == 'unverifiable'
    assert res['unverifiable'] == 1
    assert res['correct_share'] == 0.0
