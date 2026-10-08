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
    # проверить нечем — значит, доли верных не определены, а охват сверкой нулевой
    assert res['correct_share'] is None
    assert res['confirmed_correct'] is None
    assert res['verified_share'] == 0.0


def test_unverifiable_hypotheses_are_not_in_denominators():
    """Одна верная подтверждённая гипотеза об образце и один подтверждённый закон: точность 100%, охват 50%."""
    sc = make_test_scenario(samples=[[1.0, 1.0]])
    hyp = [
        {'id': 'H1', 'key': 'sample:1', 't_open': 10.0, 'statement': 'образец лежит около (1.0; 1.0)',
         'status': 'confirmed', 'data': {'x': 1.0, 'y': 1.0}},
        {'id': 'H2', 'key': 'law:per_rad', 't_open': 12.0, 'statement': 'повороты стоят заряда', 'status': 'confirmed'},
        {'id': 'H3', 'key': 'law:per_m_load', 't_open': 13.0, 'statement': 'груз удорожает путь', 'status': 'refuted'},
    ]
    res = score_hypotheses(hyp, sc, world=[], events=[])
    assert res['confirmed_correct'] == 1.0
    assert res['correct_share'] == 1.0
    assert res['refuted_correct'] is None            # среди опровергнутых проверяемых нет
    assert res['verified_share'] == pytest.approx(1 / 3, abs=1e-4)
    assert res['by_kind']['law']['verified'] == 0
    assert res['by_kind']['sample']['verified'] == 1


# --- НЕСКОЛЬКО СОБЫТИЙ ЗА ПРОГОН ---

def test_soil_state_follows_every_change():
    """Множитель 2 -> 4 (10 с) -> 1,5 (20 с): на 25 с действует 1,5, на 15 с — 4."""
    def zone(mult):
        return Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=mult)
    events = [{'t': 10.0, 'type': 'soil_change', 'soils': [zone(4.0)]},
              {'t': 20.0, 'type': 'soil_change', 'soils': [zone(1.5)]}]
    sc = make_test_scenario(soils=[zone(2.0)], events=events)
    world = [{'t': 10.0, 'type': 'soil_change'}, {'t': 20.0, 'type': 'soil_change'}]

    def hyp(t, mult):
        return {'id': f'H{t}', 'key': f'soil:{t}:0.0:0.0', 't_open': float(t), 'status': 'confirmed',
                'statement': 'около (0.0; 0.0) грунт дороже обычного', 'data': {'x': 0.0, 'y': 0.0, 'mult': mult}}
    hs = [hyp(5, 2.0), hyp(15, 4.0), hyp(25, 1.5)]
    score_hypotheses(hs, sc, world=world, events=[])
    assert [h['rel_error'] for h in hs] == [pytest.approx(0.0)] * 3
    hs = [hyp(5, 2.0), hyp(15, 4.0), hyp(25, 1.5)]
    score_hypotheses(hs, sc, world=[], events=[])          # без журнала судьи — по временам сценария
    assert [h['rel_error'] for h in hs] == [pytest.approx(0.0)] * 3


def test_soil_change_not_yet_applied_is_ignored():
    """Вторая смена в сценарии есть, но прогон до неё не дошёл позже гипотезы: журнал судьи главнее расписания."""
    def zone(mult):
        return Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=mult)
    events = [{'t': 10.0, 'type': 'soil_change', 'soils': [zone(4.0)]},
              {'t': 20.0, 'type': 'soil_change', 'soils': [zone(1.5)]}]
    sc = make_test_scenario(soils=[zone(2.0)], events=events)
    world = [{'t': 10.1, 'type': 'soil_change'}, {'t': 20.1, 'type': 'soil_change'}]
    h = {'id': 'H1', 'key': 'soil:20:0.0:0.0', 't_open': 20.05, 'status': 'confirmed',
         'statement': 'около (0.0; 0.0) грунт дороже обычного', 'data': {'x': 0.0, 'y': 0.0, 'mult': 4.0}}
    score_hypotheses([h], sc, world=world, events=[])
    assert h['rel_error'] == pytest.approx(0.0)


def test_each_new_hazard_has_its_own_time():
    """Зоны появляются на 10 и 30 с: гипотеза о второй зоне на 20 с неверна, на 35 с — верна."""
    z1 = Zone(id='X1', shape='circle', x=1.0, y=1.0, r=0.3)
    z2 = Zone(id='X2', shape='circle', x=-1.0, y=-1.0, r=0.3)
    events = [{'t': 10.0, 'type': 'new_hazard', 'zone': z1}, {'t': 30.0, 'type': 'new_hazard', 'zone': z2}]
    sc = make_test_scenario(events=events)
    world = [{'t': 10.0, 'type': 'new_hazard'}, {'t': 30.0, 'type': 'new_hazard'}]

    def hyp(t, x):
        return {'id': f'H{t}', 'key': f'hazard:{t}', 't_open': float(t), 'status': 'open',
                'statement': 'опасная зона', 'data': {'x': x, 'y': x, 'r': 0.3}}
    for w in (world, []):
        hs = [hyp(20, -1.0), hyp(35, -1.0), hyp(20, 1.0), hyp(5, 1.0)]
        score_hypotheses(hs, sc, world=w, events=[])
        assert [h['truth_verdict'] for h in hs] == ['wrong', 'correct', 'correct', 'wrong']


def test_events_at_the_same_time_are_matched_in_order():
    """Два события одного вида в один момент (в сценариях по умолчанию такое бывает) сопоставляются по порядку."""
    z1 = Zone(id='X1', shape='circle', x=1.0, y=1.0, r=0.3)
    z2 = Zone(id='X2', shape='circle', x=-1.0, y=-1.0, r=0.3)
    events = [{'t': 10.0, 'type': 'new_hazard', 'zone': z1}, {'t': 10.0, 'type': 'new_hazard', 'zone': z2}]
    sc = make_test_scenario(events=events)
    world = [{'t': 10.0, 'type': 'new_hazard'}, {'t': 10.0, 'type': 'new_hazard'}]
    hs = [{'id': 'H1', 'key': 'hazard:1', 't_open': 12.0, 'status': 'open', 'statement': 'опасная зона',
           'data': {'x': x, 'y': x, 'r': 0.3}} for x in (1.0, -1.0)]
    score_hypotheses(hs, sc, world=world, events=[])
    assert [h['truth_verdict'] for h in hs] == ['correct', 'correct']


# --- ОКРЕСТНОСТЬ ГРУНТА ---

def test_soil_neighbourhood_is_the_whole_disc_circle():
    """Круг грунта r=0,38 м в 0,575 м под углом 22,5°: до грунта 0,195 м — между пробными лучами старой проверки."""
    import math
    a = math.radians(22.5)
    zone = Zone(id='A', shape='circle', x=0.575 * math.cos(a), y=0.575 * math.sin(a), r=0.38, mult=3.0)
    sc = make_test_scenario(soils=[zone])
    h = {'id': 'H1', 'key': 'soil:5:0.0:0.0', 't_open': 5.0, 'status': 'confirmed',
         'statement': 'около (0.0; 0.0) грунт дороже обычного', 'data': {'x': 0.0, 'y': 0.0, 'mult': 3.0}}
    score_hypotheses([h], sc, world=[], events=[])
    assert h['truth_verdict'] == 'correct'
    assert h['rel_error'] == pytest.approx(0.0)
    far = Zone(id='A', shape='circle', x=0.585 * math.cos(a), y=0.585 * math.sin(a), r=0.38, mult=3.0)
    h2 = dict(h)
    score_hypotheses([h2], make_test_scenario(soils=[far]), world=[], events=[])
    assert h2['truth_verdict'] == 'wrong'             # до грунта 0,205 м — уже вне окрестности


def test_soil_neighbourhood_is_the_whole_disc_rect():
    """Угол прямоугольника в 0,19 м по диагонали от точки: ни одна из пробных точек старой проверки в него не попадала."""
    import math
    c = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    # прямоугольник, ближайшая точка которого — угол на луче 22,5°, в 0,19 м от гипотезы
    zone = Zone(id='A', shape='rect', x=0.19 * c[0] + 0.5, y=0.19 * c[1] + 0.5, w=1.0, h=1.0, mult=2.5)
    sc = make_test_scenario(soils=[zone])
    h = {'id': 'H1', 'key': 'soil:5:0.0:0.0', 't_open': 5.0, 'status': 'confirmed',
         'statement': 'около (0.0; 0.0) грунт дороже обычного', 'data': {'x': 0.0, 'y': 0.0, 'mult': 2.5}}
    score_hypotheses([h], sc, world=[], events=[])
    assert h['truth_verdict'] == 'correct'
    assert h['rel_error'] == pytest.approx(0.0)


def test_soil_neighbourhood_takes_the_dearest_zone():
    cheap = Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.3, mult=2.0)
    dear = Zone(id='B', shape='rect', x=0.65, y=0.0, w=1.0, h=1.0, mult=4.0)      # край в 0,15 м
    sc = make_test_scenario(soils=[cheap, dear])
    h = {'id': 'H1', 'key': 'soil:5:0.0:0.0', 't_open': 5.0, 'status': 'confirmed',
         'statement': 'около (0.0; 0.0) грунт дороже обычного', 'data': {'x': 0.0, 'y': 0.0, 'mult': 4.0}}
    score_hypotheses([h], sc, world=[], events=[])
    assert h['rel_error'] == pytest.approx(0.0)


# --- ВИД ГИПОТЕЗЫ ---

def test_kind_comes_from_key_not_from_text():
    """key='sensor' с текстом про образец остаётся гипотезой о датчике; текст решает только для записей без ключа."""
    world = [{'t': 20.0, 'type': 'fault', 'kind': 'sensor_noise', 'until': 50.0}]
    sc = make_test_scenario(samples=[[1.0, 1.0]])
    hyp = [
        {'id': 'H1', 'key': 'sensor', 't_open': 30.0, 'status': 'confirmed', 'statement': 'датчик не видит образец'},
        {'id': 'H2', 'key': 'hazard:1', 't_open': 30.0, 'status': 'open', 'data': {'x': 5.0, 'y': 5.0, 'r': 0.3},
         'statement': 'опасная зона: здесь дорогой грунт и рядом образец'},
        {'id': 'H3', 'key': 'law:per_rad', 't_open': 30.0, 'status': 'confirmed', 'statement': 'датчик и грунт'},
        {'id': 'H4', 't_open': 30.0, 'status': 'confirmed', 'statement': 'образец лежит около (1.0; 1.0)'},
        {'id': 'H5', 'key': 'H5', 't_open': 30.0, 'status': 'confirmed', 'statement': 'датчик образцов неисправен'},
    ]
    score_hypotheses(hyp, sc, world=world, events=[])
    assert [h['kind'] for h in hyp] == ['sensor', 'hazard', 'law', 'sample', 'sensor']
    assert hyp[0]['truth_verdict'] == 'correct'


# --- РАССЛЕДОВАНИЯ: ТО ЖЕ СОСТОЯНИЕ МИРА ---

def test_inquiry_truth_follows_every_soil_change():
    """Дорогой грунт уехал на 10 с и вернулся на 20 с: вывод «грунт» на 25 с верен, на 15 с проверить нечем."""
    from did.metrics import score_inquiries
    here = Zone(id='A', shape='circle', x=0.0, y=0.0, r=0.5, mult=3.0)
    away = Zone(id='A', shape='circle', x=2.0, y=2.0, r=0.5, mult=3.0)
    events = [{'t': 10.0, 'type': 'soil_change', 'soils': [away]}, {'t': 20.0, 'type': 'soil_change', 'soils': [here]}]
    sc = make_test_scenario(soils=[here], events=events)
    world = [{'t': 10.0, 'type': 'soil_change'}, {'t': 20.0, 'type': 'soil_change'}]
    qs = [{'id': f'Q{t}', 'topic': 'energy', 't_open': float(t), 't_close': t + 1.0, 'tests': [],
           'anomaly': {'x': 0.0, 'y': 0.0}, 'conclusion': {'status': 'identified', 'best': 'soil'}} for t in (5, 15, 25)]
    score_inquiries(qs, sc, world)
    assert [q['verdict'] for q in qs] == ['correct', 'unverifiable', 'correct']
