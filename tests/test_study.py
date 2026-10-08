"""Проверки конструктора исследования: задание и его смысловая проверка, исполнитель, отчёт, качество оценок."""
import json
import math

import numpy as np
import pytest

from did import study as S
from did.arena import load_arena
from did.config import SCIENCE, Rules
from did.fastsim import FastSim
from did.nav import CostGraph
from did.recorder import Recorder, load_trace
from did.runner import RUNS, run_episode
from did.scenario import generate, soil_mult
from did.study_agent import STUDY_CONFIG, StudyAgent

SOIL = {'quantity': 'soil_cost', 'region': {'zone': 'A'}, 'allowed': {'straight': {'length_m': 0.5, 'repeats': 2}, 'pause': {}},
        'controls': {'max': 2}, 'stop': {'rel_error': 0.03}, 'budget': {'energy': 45, 'reserve': 3}}
TURN = {'quantity': 'turn_cost', 'allowed': {'spin': {'angle_deg': 360, 'repeats': 2}, 'pause': {'seconds': 3}},
        'stop': {'rel_error': 0.05}, 'budget': {'energy': 10, 'reserve': 1}}


@pytest.fixture(scope='module')
def arena():
    return load_arena()


@pytest.fixture(scope='module')
def rules():
    return Rules(**SCIENCE)


def study(spec, level='medium', seed=3, **kw):
    return S.run_study(spec, level, seed, save=False, **kw)['study']


def texts(plan, level='error'):
    return ' | '.join(p['text'] for p in plan['problems'] if p['level'] == level)


# --- задание: схема ------------------------------------------------------------------------------

def test_spec_reads_json_yaml_and_the_example_from_the_task(tmp_path):
    example = S.preset('soil_b')
    assert example['text'].startswith('Исследовать расход в области B. Сравнивать прямолинейное движение.')
    spec = S.load_spec(example['spec'])
    assert spec.quantity == 'soil_cost' and spec.region.zone == 'B' and spec.controls.max == 2
    assert spec.allowed.kinds() == ['straight'] and spec.budget.reserve > 0          # только пробеги, запас на возврат

    as_json = tmp_path / 'spec.json'
    as_json.write_text(json.dumps(example['spec'], ensure_ascii=False), encoding='utf-8')
    as_yaml = tmp_path / 'spec.yaml'
    as_yaml.write_text('quantity: turn_cost\nallowed: [spin, pause]\nstop: {rel_error: 0.1}\n', encoding='utf-8')
    assert S.load_spec(as_json).to_dict() == spec.to_dict()
    turn = S.load_spec(str(as_yaml))
    assert turn.allowed.spin.angle_deg == 360 and turn.allowed.pause.seconds == 3 and turn.allowed.straight is None
    assert S.load_spec('{"quantity": "idle_cost"}').allowed.kinds() == ['straight', 'pause', 'spin']   # по умолчанию — всё


def test_spec_errors_are_readable():
    with pytest.raises(S.StudyError) as e:
        S.load_spec({'quantity': 'weight', 'budge': 1, 'stop': {'rel_error': -1}, 'allowed': {'straight': {'length_m': 9}}})
    text = ' | '.join(e.value.problems)
    assert 'что исследуем: нужно одно из: soil_cost' in text and 'budge: такого поля в задании нет' in text
    assert 'условия остановки → точность: нужно больше 0' in text
    assert 'разрешённые опыты → прямые пробеги → длина пробега: нужно не больше 2' in text
    with pytest.raises(S.StudyError, match='нужно либо имя области'):
        S.load_spec({'quantity': 'soil_cost', 'region': {'x': 1, 'y': 1}})
    with pytest.raises(S.StudyError, match='разные id'):
        S.load_spec({'quantity': 'soil_cost', 'region': {'zone': 'A'},
                     'hypotheses': [{'id': 'a', 'statement': 'x', 'value': 1}, {'id': 'a', 'statement': 'y', 'value': 2}]})
    with pytest.raises(S.StudyError, match='файл задания не найден'):
        S.load_spec('нет_такого.json')


# --- задание: смысловая проверка ------------------------------------------------------------------

def test_check_explains_what_is_wrong(arena, rules):
    sc = generate('medium', 3, arena)

    def plan(**spec):
        return S.check_study({'quantity': 'soil_cost', **spec}, arena, rules, sc)

    assert 'не задана область' in texts(plan())
    assert 'область целиком вне арены' in texts(plan(region={'shape': 'circle', 'x': 5, 'y': 5, 'r': 0.4}))
    assert 'в стене' in texts(plan(region={'shape': 'circle', 'x': 0, 'y': 0, 'r': 0.1}))             # центральный столб
    assert 'нет области «Z»: есть A, B, C' in texts(plan(region={'zone': 'Z'}))
    assert 'пробег не помещается' in texts(plan(region={'shape': 'circle', 'x': -0.55, 'y': 0.0, 'r': 0.15}))
    assert 'не помещается: робот проедет' in texts(plan(region={'shape': 'circle', 'x': -0.55, 'y': 0.0, 'r': 0.22}), 'warning')
    small = plan(region={'zone': 'B'}, budget={'energy': 8, 'reserve': 3})
    assert 'меньше дороги туда и обратно' in texts(small) and not small['ok']
    assert 'не успеет даже доехать' in texts(plan(region={'zone': 'B'}, stop={'time_s': 25}))
    assert 'нужны прямые пробеги' in texts(plan(region={'zone': 'B'}, allowed=['pause']))
    assert 'внутри исследуемой области' in texts(plan(region={'zone': 'B'}, controls={'max': 1, 'sites': [{'x': -0.06, 'y': 2.1}]}))
    assert 'разрешено участков: 1' in texts(plan(region={'zone': 'B'}, controls={'max': 1, 'sites': [{'x': -2, 'y': 0}, {'x': -2, 'y': 1}]}))
    assert 'больше, чем есть в батарее' in texts(plan(region={'zone': 'B'}, budget={'energy': 100}))
    assert 'без контрольного участка' in texts(plan(region={'zone': 'B'}, controls={'max': 0}), 'warning')
    assert 'не несёт образцов' in texts(S.check_study({'quantity': 'load_effect'}, arena, rules, sc))
    assert S.check_study({'quantity': 'load_effect'}, arena, rules, sc, carried=2)['ok']
    one = plan(region={'zone': 'B'}, hypotheses=[{'id': 'a', 'statement': 'обычный пол'}])
    assert 'нужно значение' in texts(one) and 'объяснение одно' in texts(one)


def test_plan_places_test_and_control_segments_properly(arena, rules):
    graph = CostGraph(arena)
    for seed in range(1, 9):
        sc = generate('medium', seed, arena)
        plan = S.check_study(SOIL, arena, rules, sc, graph=graph)
        if not plan['ok']:
            continue
        zone = sc.soils[0]
        t, c = plan['sites']['test'], plan['sites']['controls'][0]
        for site, inside in ((t, True), (c, False)):
            (ax, ay), (bx, by) = site['a'], site['b']
            assert math.dist(site['a'], site['b']) == pytest.approx(site['length'], abs=1e-3) and site['length'] >= S.MIN_RUN_M
            for k in np.linspace(0, 1, 11):                       # весь отрезок свободен и лежит по нужную сторону границы
                x, y = ax + k * (bx - ax), ay + k * (by - ay)
                assert arena.clearance(x, y) >= S.RUN_CLEAR_M - 0.05
                assert zone.contains(x, y) == inside
        assert len(plan['steps']) >= 4 and plan['costs']['road'] > 0
        assert 'mult' not in json.dumps(plan)                     # цена пола в план не попадает


def test_every_preset_passes_its_own_check():
    assert len(S.PRESETS) >= 4
    for p in S.PRESETS:
        code, plan = S.api_plan({'level': p['level'], 'seed': p['seed'], 'spec': p['spec']})
        assert code == 200 and plan['ok'], (p['id'], plan['problems'])
        assert plan['task'].startswith('Выяснить, ')


# --- исполнитель: цена пола -----------------------------------------------------------------------

def test_soil_study_measures_the_hidden_multiplier(arena):
    summary = S.run_study(SOIL, 'medium', 5, experiment='_test')
    r, m = summary['study'], summary['metrics']
    truth = generate('medium', 5, arena).soils[0].mult
    est = r['estimate']
    assert r['status'] == 'done' and r['stop']['reason'] == 'precision' and m['returned']
    assert est['value'] == pytest.approx(truth, rel=0.03) and est['rel_error'] <= 0.03
    assert est['ci95'][0] < est['value'] < est['ci95'][1] and est['unit'].startswith('×')
    assert r['truth']['value'] == truth and r['truth']['error_pct'] < 3.0
    roles = [(x['role'], x['kind']) for x in r['measurements'] if x['used']]
    assert roles[0] == ('calibration', 'pause')
    assert roles.count(('test', 'straight')) >= 2 and roles.count(('control', 'straight')) >= 2     # область и контроль
    for x in r['measurements']:
        assert {'n', 't', 'kind', 'role', 'x', 'y', 'ds', 'dt', 'spent', 'value', 'sigma', 'unit', 'used'} <= set(x)
    legs = [x for x in r['measurements'] if x['kind'] == 'straight' and x['used']]
    assert all(0.4 < x['ds'] < 0.56 and x['sigma'] > 0 for x in legs)
    assert np.mean([x['value'] for x in legs if x['role'] == 'control']) == pytest.approx(1.0, abs=0.03)
    assert r['control']['vs_nominal'] == pytest.approx(1.0, abs=0.04) and 'контрольном участке' in r['control']['text']
    e = r['energy']
    assert e['spent'] <= e['budget'] and e['spent'] == pytest.approx(e['travel'] + e['test'] + e['control'] + e['home'], abs=0.3)
    assert m['study_error_pct'] == r['truth']['error_pct'] and m['study_reached'] and m['study_energy'] == e['spent']

    # запись прогона открывается проигрывателем: в ней обычные поля и отчёт исследования
    trace = load_trace(RUNS / summary['file'])
    assert trace['study']['estimate'] == est and trace['agent']['name'] == 'study'
    assert len(trace['track']['t']) > 100 and trace['paths'] and trace['plans'] and 'опыт' in trace['modes']
    assert any('Следующий замер' in j['text'] for j in trace['journal'])


def test_tight_budget_ends_with_an_honest_conclusion():
    # бюджета хватает на дорогу и минимум замеров, но не на ±0,5 %
    spec = {**SOIL, 'stop': {'rel_error': 0.005}, 'budget': {'energy': 32, 'reserve': 3}}
    summary = S.run_study(spec, 'medium', 5, save=False)
    r = summary['study']
    assert r['status'] == 'not_reached' and r['stop']['reason'] == 'budget'
    assert 'Точность не достигнута: получено ±' in r['conclusion'] and 'при требуемых ±0.5%' in r['conclusion']
    assert r['estimate']['rel_error'] > 0.005
    assert summary['metrics']['returned'] and r['energy']['spent'] <= 32.5        # запас на возврат сохранён
    assert not summary['metrics']['study_reached']


def test_without_control_the_robot_says_it_knows_less():
    r = study({**SOIL, 'controls': {'max': 0}}, seed=5)
    assert r['status'] == 'not_reached' and r['stop']['reason'] == 'no_gain'
    assert 0.15 < r['estimate']['rel_error'] < 0.25                # номинал известен примерно на 10 %
    assert r['truth']['covered'] and 'номинал' in r['control']['text']
    assert any('без контрольного участка' in f for f in r['failures'])


def test_hypotheses_are_compared_like_an_inquiry():
    p = S.preset('soil_which')
    r = study(p['spec'], p['level'], p['seed'])
    q = r['inquiry']
    assert {'id', 'topic', 'anomaly', 'alternatives', 'tests', 'conclusion', 'action'} <= set(q)      # формат расследования
    assert q['conclusion']['status'] == 'identified' and q['conclusion']['best'] == 'x2'
    assert [a['id'] for a in q['alternatives']] == ['x1', 'x2', 'x3', 'x4', 'other']
    done = [t for t in q['tests'] if t['measured']]
    assert done and all(set(t['predictions']) == {'x1', 'x2', 'x3', 'x4'} for t in done)
    assert 'Из объяснений: пол вдвое дороже обычного' in r['conclusion']


def test_soil_change_during_the_study_is_noticed():
    # hard-1005: грунты меняются на 33-й секунде, пока робот едет мерный пробег
    r = study(SOIL, 'hard', 1005)
    assert any('расход в области изменился' in f for f in r['failures'])
    dropped = [x for x in r['measurements'] if x['note'] == 'до изменения пола']
    assert dropped and dropped[0]['halves']['uneven']
    assert r['estimate']['value'] == pytest.approx(r['truth']['value'], rel=0.04)       # оценка — по замерам после изменения


def test_penalty_and_leak_do_not_get_into_measurements():
    # hard-1003: по дороге опасная зона, после штрафа утечка заряда
    summary = S.run_study(SOIL, 'hard', 1003, save=False)
    r = summary['study']
    assert summary['metrics']['hazard_hits'] >= 1
    notes = {x['note'] for x in r['measurements'] if not x['used']}
    assert 'идёт утечка заряда' in notes
    assert r['estimate']['value'] == pytest.approx(r['truth']['value'], rel=0.04)
    assert 'утечка' in r['control']['text']


# --- исполнитель: поворот, простой, груз ----------------------------------------------------------

def test_turn_study_is_accurate_and_its_intervals_are_honest(rules):
    hits, errors = 0, []
    for seed in range(1, 21):
        r = study(TURN, 'easy', seed)
        est = r['estimate']
        assert r['status'] == 'done' and est['unit'] == 'ед/рад' and est['rel_error'] <= 0.05
        hits += est['ci95'][0] <= rules.drain_per_rad <= est['ci95'][1]
        errors.append(abs(est['value'] - rules.drain_per_rad) / rules.drain_per_rad)
        spins = [x for x in r['measurements'] if x['kind'] == 'spin']
        assert len(spins) >= 2 and all(abs(x['dth'] - 2 * math.pi) < 0.3 for x in spins)
        assert r['energy']['spent'] < 4.0 and r['time_s'] < 80
    assert hits >= 17 and np.median(errors) < 0.03                 # 95% интервал накрывает истину почти всегда


def test_idle_preset_admits_that_precision_is_not_reached():
    p = S.preset('idle')
    r = study(p['spec'], p['level'], p['seed'])
    assert r['status'] == 'not_reached' and r['stop']['reason'] == 'time'
    assert r['conclusion'].startswith('Точность не достигнута: получено ±')
    assert r['estimate']['value'] == pytest.approx(0.01, rel=0.25) and r['time_s'] <= 62


def test_load_effect_is_refused_without_cargo_and_measured_with_it(arena, rules):
    refused = S.run_study({'quantity': 'load_effect'}, 'medium', 3, save=False)
    r = refused['study']
    assert r['status'] == 'refused' and 'не несёт образцов' in r['failures'][0] and r['estimate'] is None
    assert refused['metrics']['returned'] and refused['metrics']['distance'] == 0      # робот остался на базе
    code, out = S.api_run({'level': 'medium', 'seed': 3, 'spec': {'quantity': 'load_effect'}})
    assert code == 400 and 'не несёт образцов' in out['error']

    # тот же вопрос, когда груз есть: два образца уже на борту
    sc = generate('medium', 3, arena)
    world = FastSim(arena, sc, rules, seed=3)
    world.judge.collected = [(0, 0.0), (1, 0.0)]
    spec = {'quantity': 'load_effect', 'allowed': ['straight', 'pause'], 'stop': {'rel_error': 0.5}, 'budget': {'energy': 20}}
    bot = StudyAgent(arena, STUDY_CONFIG, len(sc.samples), rules=rules, recorder=Recorder(),
                     study=S.prepare(spec, arena, rules, sc, carried=2))
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    r = bot.study_report(world.judge.reason)
    est = r['estimate']
    assert world.judge.returned and est['unit'] == 'доля на образец'
    assert est['ci95'][0] <= rules.load_drain <= est['ci95'][1] and abs(est['value'] - rules.load_drain) < 0.08
    assert 'номинал' in r['control']['text']


# --- исполнитель: закон датчика -------------------------------------------------------------------

def test_sensor_law_study_finds_a_sample_and_confirms_the_law(arena):
    p = S.preset('sensor')
    summary = S.run_study(p['spec'], p['level'], p['seed'], save=False)
    r, q = summary['study'], summary['study']['inquiry']
    assert summary['metrics']['samples_collected'] == 0            # образец найден, но не собран: он источник сигнала
    sample = r['plan']['sites']['sample']
    assert min(math.dist(sample, s) for s in generate(p['level'], p['seed'], arena).samples) < 0.12
    assert r['status'] == 'done' and q['conclusion']['best'] == 'linear_2m' and q['conclusion']['confidence'] >= 0.9
    assert r['truth']['verdict'] == 'correct'
    assert len([t for t in q['tests'] if t['measured']]) >= 2      # вывод не по одному расстоянию
    listens = [x for x in r['measurements'] if x['kind'] == 'listen' and x['used']]
    assert [x for x in listens if x['role'] == 'control'] and all(0 <= x['value'] <= 1 for x in listens)
    far = [x for x in listens if x['role'] == 'test' and x['d'] > 0.4]
    assert far and all(abs(x['value'] - (1 - x['d'] / 2)) < 0.12 for x in far)
    assert r['estimate']['unit'] == 'м' and r['estimate']['value'] == pytest.approx(2.0, abs=0.4)
    assert 'за это время он не сбился' in r['control']['text']


def test_sensor_law_never_names_a_wrong_law(arena):
    p = S.preset('sensor')
    verdicts = [study(p['spec'], level, seed)['truth'].get('verdict', 'failed')
                for level, seeds in (('easy', range(1, 9)), ('medium', range(1, 7))) for seed in seeds]
    assert 'wrong' not in verdicts and verdicts.count('correct') >= 10


# --- отчёт и подключение --------------------------------------------------------------------------

def test_markdown_report_is_complete_and_russian():
    p = S.preset('soil_which')
    r = study(p['spec'], p['level'], p['seed'])
    md = r['markdown']
    for part in ('# Отчёт об исследовании: Какой грунт в области A', '**Задание.**', '**Итог.**', '**Вывод.**',
                 '**Сравнение с контролем.**', '**Почему остановились.**', '**Цена.**', '## План', '## Замеры',
                 '## Цепочка рассуждений'):
        assert part in md, part
    assert 'Сверка со сценарием' not in md and 'Сверка со сценарием' in r['markdown_truth']     # правда — только по просьбе
    assert md.count('\n| ') >= len(r['measurements']) + 1
    assert not any(ch.isdigit() and nxt == '.' and after.isdigit() for ch, nxt, after in zip(md, md[1:], md[2:]))   # запятые
    assert S.ru('±5.0% за 1 опыт(ов)') == '±5,0 % за 1 опыта'
    assert S.plural(1, 'замер', 'замера', 'замеров') == 'замер' and S.plural(12, 'замер', 'замера', 'замеров') == 'замеров'


def test_run_episode_and_api_entry_points():
    summary = run_episode('easy', 2, 'study', rules='science', study=TURN, save=False)
    assert summary['agent'] == 'study' and summary['study']['spec']['quantity'] == 'turn_cost'
    assert summary['metrics']['study_status'] == 'done' and 'study_covered' in summary['metrics']

    presets = S.api_presets()
    assert {p['id'] for p in presets['presets']} >= {'soil_b', 'turn', 'sensor'}
    assert [q['id'] for q in presets['quantities']] == list(S.QUANTITIES)
    code, out = S.api_run({'level': 'easy', 'seed': 2, 'spec': TURN})
    assert code == 200 and out['file'].startswith('study/turn_cost-') and out['study']['status'] == 'done'
    json.dumps(out)                                                 # ответ сервера сериализуется без типов numpy
    code, out = S.api_run({'level': 'medium', 'seed': 3, 'spec': {'quantity': 'soil_cost'}})
    assert code == 400 and 'не задана область' in out['error'] and out['plan']['ok'] is False
    code, out = S.api_plan({'level': 'nope', 'seed': 3, 'spec': TURN})
    assert code == 400 and 'нет уровня' in out['error']
    assert S.spec_key(TURN) == S.spec_key(dict(reversed(list(TURN.items())))) != S.spec_key(SOIL)


def test_truth_is_taken_along_the_measured_path(arena, rules):
    # область шире зоны грунта: истина — средний множитель вдоль мерного участка, а не множитель зоны
    sc = generate('medium', 5, arena)
    z = sc.soils[0]
    region = {'shape': 'circle', 'x': float(z.x) + 0.3, 'y': float(z.y), 'r': 0.6}
    r = study({**SOIL, 'region': region, 'controls': {'max': 1}}, 'medium', 5)
    legs = [x for x in r['measurements'] if x['role'] == 'test' and x['kind'] == 'straight' and x['used']]
    if legs:
        along = np.mean([soil_mult(sc.soils, a, b) for x in legs
                         for a, b in zip(np.linspace(x['x0'], x['x1'], 200), np.linspace(x['y0'], x['y1'], 200))])
        assert r['truth']['value'] == pytest.approx(along, rel=0.02)
        assert r['estimate']['value'] == pytest.approx(r['truth']['value'], rel=0.06)
