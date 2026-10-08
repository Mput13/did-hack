"""Проверки сравнения будущих маршрутов: расчёт на игрушечных входах и связка с агентом."""
import time
from dataclasses import replace

import numpy as np
import pytest

from did.agent import PRESETS, Agent, make_config
from did.arena import load_arena
from did.config import SCIENCE, Rules
from did.fastsim import FastSim
from did.foresight import Foresight, Option, Settings, Step, line_leg
from did.recorder import Recorder
from did.scenario import generate

BASE, NEAR, FAR = (-2.0, -0.5), (-1.5, -0.5), (1.0, -0.5)
PER_M = 2.5


@pytest.fixture(scope='module')
def fs():
    return Foresight(rules=Rules(**SCIENCE))


def _options(fs, unknown=0.5):
    """База слева, рядом слабая цель (образец есть с вероятностью 0,3), далеко — ценная (0,9)."""
    w = fs.worlds
    home = (line_leg(w, BASE, BASE, 0.0), 0.0)          # робот стоит на базе: дорога домой нулевая

    def there_and_back(name, xy, p, k):
        back = line_leg(w, xy, BASE, unknown)
        return Option(name, f'{name} → домой', [Step(name, 'candidate', p, line_leg(w, BASE, xy, unknown), k)],
                      [(back, back.metres), home])

    return [Option('home', 'сразу домой', [], [home]), there_and_back('near', NEAR, 0.3, 0),
            there_and_back('far', FAR, 0.9, 1)]


# --- правило выбора ------------------------------------------------------------------------------

def test_big_reserve_goes_far_small_reserve_goes_home(fs):
    rows = fs.compare(_options(fs), battery=55.0, per_m=PER_M)
    best = fs.choose(rows)
    assert best['id'] == 'far' and best['ok']
    assert best['samples'] == pytest.approx(0.9, abs=0.12)             # образец есть в 9 мирах из 10
    assert best['battery_p5'] <= best['battery_p50'] <= best['battery_p95']

    rows = {r['id']: r for r in fs.compare(_options(fs), battery=13.0, per_m=PER_M)}
    assert not rows['far']['ok'] and rows['far']['risk'] > 0.5         # до дальней цели и обратно нужно около 15 ед.
    assert fs.choose(list(rows.values()))['id'] == 'near'              # ближняя ещё по силам

    rows = fs.compare(_options(fs), battery=2.0, per_m=PER_M)
    assert [r['ok'] for r in rows] == [True, False, False]
    assert fs.choose(rows)['id'] == 'home'


def test_no_option_within_risk_means_home():
    fs = Foresight(rules=Rules(**SCIENCE))
    far = line_leg(fs.worlds, FAR, BASE, 1.0)
    rows = fs.compare([Option('home', 'сразу домой', [], [(far, far.metres)])], battery=3.0, per_m=PER_M)
    assert not rows[0]['ok'] and fs.choose(rows)['id'] == 'home'


# --- чего агент не знает -------------------------------------------------------------------------

def test_risk_grows_with_unverified_share(fs):
    """Тот же путь и тот же заряд: чем большая часть пути не проверена колёсами, тем вероятнее не вернуться."""
    risks, worst = [], []
    for share in (0.0, 0.3, 0.6, 1.0):
        leg = line_leg(fs.worlds, FAR, BASE, share)
        row = fs.compare([Option('home', 'сразу домой', [], [(leg, leg.metres)])], battery=9.5, per_m=PER_M)[0]
        risks.append(row['risk'])
        worst.append(row['battery_p5'])
    assert risks == sorted(risks) and risks[-1] > risks[0] + 0.1
    assert worst == sorted(worst, reverse=True)                         # и худший случай всё хуже


def test_known_hazard_and_running_leak_cost_battery(fs):
    leg = line_leg(fs.worlds, FAR, BASE, 0.0)
    calm, = fs.compare([Option('home', 'домой', [], [(leg, leg.metres)])], battery=20.0, per_m=PER_M)
    risky_leg = line_leg(fs.worlds, FAR, BASE, 0.0, risk=0.8)
    risky, = fs.compare([Option('home', 'домой', [], [(risky_leg, risky_leg.metres)])], battery=20.0, per_m=PER_M)
    assert risky['hits'] > calm['hits'] + 0.5 and risky['score'] < calm['score'] - 2.0     # штраф −5 за въезд
    leaking, = fs.compare([Option('home', 'домой', [], [(leg, leg.metres)])], battery=20.0, per_m=PER_M, leak=6.0)
    assert leaking['battery_p50'] < calm['battery_p50'] - 4.0


def test_turning_back_on_the_way_lowers_risk():
    """Дорога к цели не проверена, домой от цели далеко: если неожиданность встретится в начале пути, агент вернётся."""
    out = []
    for below in (-1e9, 2.0):                                           # −∞ — агент не передумывает никогда
        fs = Foresight(settings=replace(Settings(), turn_back_below=below), rules=Rules(**SCIENCE))
        w = fs.worlds
        home = (line_leg(w, BASE, BASE, 0.0), 0.0)
        back = line_leg(w, FAR, BASE, 0.0)
        opt = Option('far', 'far → домой', [Step('far', 'candidate', 0.9, line_leg(w, BASE, FAR, 1.0))],
                     [(back, back.metres), home])
        out.append(fs.compare([opt], battery=20.0, per_m=PER_M)[0]['risk'])
    assert out[1] < out[0]


def test_same_worlds_every_time(fs):
    a = fs.compare(_options(fs), battery=30.0, per_m=PER_M)
    b = Foresight(rules=Rules(**SCIENCE)).compare(_options(fs), battery=30.0, per_m=PER_M)
    assert a == b


def test_decision_is_fast(fs):
    """Полтора десятка вариантов с путями по 3–6 м при 48 мирах — заметно быстрее 30 мс."""
    w = fs.worlds
    rng = np.random.default_rng(1)
    home = (line_leg(w, (0.0, 0.0), BASE, 0.5), 2.1)
    opts = [Option('home', 'сразу домой', [], [home])]
    for k in range(14):
        xy = tuple(rng.uniform(-2.5, 2.5, 2))
        back = line_leg(w, xy, BASE, 0.7)
        opts.append(Option(f'T{k}', f'T{k} → домой', [Step(f'T{k}', 'candidate', 0.5, line_leg(w, (0.0, 0.0), xy, 0.7), k)],
                           [(back, back.metres), home]))
    best = 1e9
    for _ in range(7):
        t0 = time.perf_counter()
        fs.choose(fs.compare(opts, battery=40.0, per_m=PER_M))
        best = min(best, time.perf_counter() - t0)
    assert best < 0.030


# --- вместе с агентом ----------------------------------------------------------------------------

def test_presets_differ_only_by_foresight():
    assert PRESETS['scientist_fs'] == replace(PRESETS['scientist'], name='scientist_fs', foresight=True, risk_limit=0.15)
    assert PRESETS['adaptive_fs'] == replace(PRESETS['adaptive'], name='adaptive_fs', foresight=True, risk_limit=0.15)
    assert not any(c.foresight for name, c in PRESETS.items() if not name.endswith('_fs'))


def _run(name, level='medium', seed=3, **config):
    arena = load_arena()
    rules = Rules(**SCIENCE)
    sc = generate(level, seed, arena)
    world = FastSim(arena, sc, rules, seed=seed)
    rec = Recorder()
    bot = Agent(arena, make_config(name, **config), n_samples=len(sc.samples), rules=rules, recorder=rec)
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    trace = rec.build(run_id='t', experiment='t', arm=name, backend='fastsim', agent={'name': name}, scenario=sc.to_dict(),
                      rules=rules.to_dict(), result=world.judge.score(), world=world.judge.world_log, journal=bot.journal)
    return bot, world, trace


@pytest.mark.parametrize('name', ['scientist_fs', 'adaptive_fs'])
def test_agent_decides_by_comparison_and_records_tables(name):
    bot, world, trace = _run(name)
    assert world.judge.returned and len(world.judge.collected) >= 2
    decisions = trace['foresight']
    assert len(decisions) >= 5 and {p['source'] for p in trace['plans']} <= {'foresight', 'rule'}
    keys = {c['key'] for c in decisions[0]['columns']}
    for d in decisions:
        assert d['worlds'] == 48 and d['risk_limit'] == 0.15
        assert sum(r['chosen'] for r in d['rows']) == 1 and d['rows'][0]['id'] == 'home'
        assert all(keys <= set(r) for r in d['rows'])
        chosen = next(r for r in d['rows'] if r['chosen'])
        assert chosen['id'] == d['chosen'] and (chosen['ok'] or chosen['id'] == 'home')
    assert np.median([d['ms'] for d in decisions]) < 30.0
    # Та же таблица лежит в данных записи журнала «Сравнил N вариантов при M возможных состояниях среды…».
    notes = [e for e in trace['journal'] if e['text'].startswith('Сравнил ')]
    assert len(notes) == len(decisions) and all('rows' in e['data']['foresight'] for e in notes)
    assert 'при 48 возможных состояниях среды' in notes[0]['text']


def test_choice_by_expected_score_also_works():
    bot, world, trace = _run('scientist_fs', foresight_choice='score')
    assert world.judge.returned and trace['foresight']
    for d in trace['foresight']:
        ok = [r for r in d['rows'] if r['ok']]
        if ok and d['trigger'] != 'check':
            assert d['chosen'] == max(ok, key=lambda r: r['score'])['id']


def test_agents_without_foresight_are_untouched():
    bot, world, trace = _run('scientist')
    assert bot.fs is None and 'foresight' not in trace
    assert not any(e['text'].startswith('Сравнил ') for e in trace['journal'])
