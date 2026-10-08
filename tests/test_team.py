"""Два робота на одной арене (исследование M1): общий мир, счёт команды, координация по каналу сообщений."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from did.agent import make_config
from did.arena import load_arena
from did.config import ROBOT_RADIUS, Rules
from did.fastsim import FastSim
from did.fastsim_team import TeamSim
from did.judge_team import CONTACT_M, SLOTS, min_gap, team_score
from did.robot_io import Observation
from did.runner import run_episode
from did.scenario import Scenario, generate
from did.team import TeamAgent, TeamChannel
from did.team_runner import merge_parts, run_team_episode

ARENA = load_arena()
E1_REFERENCE = Path('/Users/a/MAI/DID/runs/E1/summary.json')


def _world(samples=((0.5, 0.5), (0.5, -1.6)), **kw):
    sc = Scenario(level='medium', seed=1, samples=[list(p) for p in samples])
    return TeamSim(ARENA, sc, Rules(), seed=1, **kw)


def _place(bot, x, y, th=0.0):
    bot.x, bot.y, bot.th = x, y, th
    bot.judge._pose = (x, y)


def _obs(t=1.0, x=-2.0, y=-0.5, battery=60.0):
    return Observation(t=t, x=x, y=y, th=0.0, v=0.0, w=0.0, battery=battery, sensor=None, scan=None)


def _agent(name='tb1', n_samples=5, channel=None, **team):
    from did.team import TeamConfig
    bot = TeamAgent(ARENA, make_config('adaptive'), n_samples, name=name, home=SLOTS[0 if name == 'tb1' else 1],
                    channel=channel or TeamChannel(), team=TeamConfig(**team))
    bot._pose = (-2.0, -0.5, 0.0)
    return bot


def _peak(bot, x, y, p=0.5):
    """Поставить на карте вероятностей уверенный пик в (x, y)."""
    near = np.hypot(bot.belief.cx - x, bot.belief.cy - y) < 0.15
    bot.belief.p[:] = 1e-4
    bot.belief.p[near] = p


# --- мир ------------------------------------------------------------------------------------------

def test_slots_are_apart_and_free():
    a, b = SLOTS
    assert math.dist(a, b) >= 2 * CONTACT_M - 0.03          # не мешают друг другу на старте и на стоянке
    assert ARENA.clearance(*b) > ARENA.clearance(*a) > 0.3
    for level in ('medium', 'hard'):
        for seed in (1, 2, 3, 10001, 10002):
            sc = generate(level, seed, ARENA)
            zones = sc.soils + sc.hazards + [e['zone'] for e in sc.events if e['type'] == 'new_hazard'] \
                + [z for e in sc.events if e['type'] == 'soil_change' for z in e['soils']]
            assert not any(z.contains(*b) for z in zones)
            assert all(math.dist(b, s) > 0.5 for s in sc.samples)


def test_one_robot_world_is_plain_fastsim():
    """С одним роботом мир команды — тот же быстрый симулятор: шаг в шаг те же поза, заряд и датчик."""
    sc = generate('hard', 4, ARENA)
    team, plain = TeamSim(ARENA, sc, Rules(), seed=4, n_robots=1), FastSim(ARENA, sc, Rules(), seed=4)
    for k in range(300):
        cmd = (0.2, 0.6 * math.sin(k / 17))
        team.robots[0].command(*cmd)
        plain.command(*cmd)
        team.advance()
        plain.advance()
        a, b = team.robots[0].observe(), plain.observe()
        assert (a.x, a.y, a.th, a.battery, a.sensor) == (b.x, b.y, b.th, b.battery, b.sensor)
        assert (a.scan is None) == (b.scan is None) and (a.scan is None or np.array_equal(a.scan, b.scan))


def test_solo_run_equals_ordinary_run():
    for level, seed in (('medium', 3), ('hard', 5)):
        a = run_episode(level, seed, 'adaptive', save=False)['metrics']
        b = run_team_episode(level, seed, 'solo', save=False)['metrics']
        for k in ('score', 'samples_collected', 'returned', 'battery_left', 'battery_used', 'distance', 'time',
                  'collisions', 'false_collects', 'hazard_hits', 'detect', 'hypotheses', 'plans'):
            assert a[k] == b[k], (level, seed, k)


def test_collected_sample_disappears_for_both():
    world = _world()
    a, b = world.robots
    _place(a, 0.5, 0.4)
    _place(b, 0.5, 0.75)
    assert b.judge.read_sensor(b.x, b.y) > 0.8               # оба слышат один и тот же образец
    ok, _ = a.collect()
    assert ok and len(a.judge.collected) == 1
    assert b.judge.read_sensor(b.x, b.y) < 0.2               # для второго он исчез: слышен только дальний
    ok, _ = b.collect()
    assert not ok and b.judge.counts['false_collect'] == 1   # и собрать его второй раз нельзя
    score = world.score()
    assert score['samples_collected'] == 1 and score['samples_total'] == 2
    assert [r['samples_collected'] for r in score['per_robot']] == [1, 0]


def test_batteries_are_separate():
    world = _world()
    a, b = world.robots
    a.command(0.2, 0.0)
    for _ in range(50):
        world.advance()
    spent_a = 60.0 - a.judge.battery
    spent_b = 60.0 - b.judge.battery
    assert spent_a > 2.0                                     # проехал около метра
    assert spent_b == pytest.approx(0.01 * 5.0, abs=1e-6)    # второй стоял: только расход «включён»
    assert a.judge.distance > 0.9 and b.judge.distance == 0.0


def test_robots_block_and_collide():
    world = _world()
    a, b = world.robots
    _place(a, 0.0, -0.5)
    _place(b, 0.5, -0.5)
    a.command(0.2, 0.0)
    gaps = []
    for _ in range(60):
        world.advance()
        gaps.append(math.hypot(a.x - b.x, a.y - b.y))
    assert min(gaps) >= CONTACT_M - 1e-9                     # сквозь напарника не проехать
    assert a.x < b.x - 2 * ROBOT_RADIUS + 1e-9
    assert world.contacts == 1
    assert a.judge.counts['collision'] == 1 and b.judge.counts['collision'] == 1   # штраф обоим, один раз
    assert [e['type'] for e in b.judge.pop_events()] == ['collision']


def test_finished_robot_is_obstacle_without_penalty():
    world = _world()
    a, b = world.robots
    _place(a, 0.0, -0.5)
    _place(b, 0.5, -0.5)
    b.finish()
    a.command(0.2, 0.0)
    for _ in range(60):
        world.advance()
    assert a.x < b.x - 2 * ROBOT_RADIUS + 1e-9 and a.judge.counts['collision'] == 1
    assert b.judge.counts['collision'] == 0 and not world.done


def test_lidar_sees_the_other_robot():
    world = _world()
    a, b = world.robots
    _place(a, 0.0, -0.5)
    _place(b, 0.8, -0.5)
    a._next_lidar = a.t
    world.advance()
    scan = a.observe().scan
    walls = ARENA.raycast(a.x - 0.032, a.y, 0.0)
    assert walls[0] > 1.5 and 0.7 < scan[0] < 0.85           # башенка лидара напарника, а не стена за ним
    assert abs(scan[180] - walls[180]) < 0.05                # остальные лучи видят то же, что и без напарника


def test_team_score_is_sum_of_robots():
    one = {'score': 51.0, 'returned': True, 'battery': 20.0, 'distance': 5.0, 't': 80.0, 'collisions': 1,
           'false_collects': 0, 'hazard_hits': 0}
    two = {'score': 28.0, 'returned': False, 'battery': 0.0, 'distance': 9.0, 't': 120.0, 'collisions': 0,
           'false_collects': 1, 'hazard_hits': 0}
    s = team_score([one, two], [10.0, 30.0, 50.0], n_samples=5)
    assert s['score'] == 79.0 and s['score_per_robot'] == 39.5
    assert s['samples_share'] == 0.6 and s['returned'] == 0.5 and not s['returned_all']
    assert s['time'] == 120.0 and s['t_last_collect'] == 50.0 and s['penalties'] == 2
    assert s['mean_sample_time'] == pytest.approx((10 + 30 + 50 + 600 * 2) / 5)


# --- координация ----------------------------------------------------------------------------------

def test_channel_delivers_only_others_messages_once():
    ch = TeamChannel()
    ch.post('tb1', 0.0, 'hello', x=0, y=0, home=[0, 0])
    ch.post('tb2', 0.1, 'claim', kind='explore', x=1.0, y=1.0, cost=2.0)
    assert [m['type'] for m in ch.read('tb1')] == ['claim']
    assert ch.read('tb1') == []
    assert [m['from'] for m in ch.read('tb2')] == ['tb1']
    json.dumps(ch.log)                                       # сообщения — простые данные: их можно слать по топику


def test_claimed_target_is_not_taken():
    """«Эту взял я»: кандидат, который объявил напарник, второй робот в план не берёт."""
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    _peak(bot, 0.55, 0.55)
    obs = _obs()
    free = bot._state(obs, 'test')
    assert [c['feasible'] for c in free['candidates']] == [True]
    ch.post('tb2', 1.0, 'hello', x=0.0, y=0.5, home=list(SLOTS[1]))
    ch.post('tb2', 1.0, 'claim', kind='investigate', x=0.5, y=0.5, cost=1.5)
    bot._receive(obs)
    taken = bot._state(obs, 'test')
    assert [(c['feasible'], c.get('claimed_by')) for c in taken['candidates']] == [(False, 'tb2')]
    plan = bot.planner.plan(taken)
    assert plan['subgoals'][0] != {'type': 'investigate', 'target': 'C1'}
    assert all(math.hypot(e['x'] - 0.5, e['y'] - 0.5) >= 0.6 for e in taken['explore_points'])   # и рядом не разведывает


def test_cheaper_robot_may_take_over_claim():
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    _peak(bot, -1.4, -0.5)                                   # в полуметре от меня
    ch.post('tb2', 1.0, 'claim', kind='investigate', x=-1.4, y=-0.5, cost=9.0)   # напарнику до неё далеко
    bot._receive(_obs())
    st = bot._state(_obs(), 'test')
    assert st['candidates'][0]['feasible'] and 'claimed_by' not in st['candidates'][0]


def test_yield_same_target_to_cheaper_partner():
    """Оба выбрали одну цель: уступает тот, кому до неё дороже, — и только он."""
    for cost, yields in ((1.0, True), (30.0, False)):
        ch = TeamChannel()
        bot = _agent('tb1', channel=ch)
        bot.queue = [{'type': 'investigate', 'x': 0.5, 'y': 0.5}]
        bot._claim = {'kind': 'investigate', 'x': 0.5, 'y': 0.5, 'cost': 6.0}
        ch.post('tb2', 1.0, 'claim', kind='investigate', x=0.55, y=0.45, cost=cost)
        bot._receive(_obs())
        assert (bot.queue == []) == yields
        assert (bot._trigger == 'partner_claim') == yields


def test_collected_message_changes_plan_and_map():
    ch = TeamChannel()
    bot = _agent('tb1', n_samples=5, channel=ch)
    _peak(bot, 0.55, 0.55)
    bot.queue = [{'type': 'investigate', 'x': 0.55, 'y': 0.55}]
    bot._trigger = None
    ch.post('tb2', 2.0, 'collected', x=0.5, y=0.5, left=4)
    bot._receive(_obs(t=2.0))
    assert bot.queue == [] and bot._trigger == 'partner_collected'        # ехал туда же — план пересматривается
    assert bot.n_samples == 4 and bot.left == 4 and bot.belief.left == 4   # и образцов на арене теперь меньше
    assert bot.belief.prob_within(0.55, 0.55, 0.25) < 0.05
    assert any(e.get('data', {}).get('tag') == 'partner_collected' for e in bot.journal.entries)


def test_stale_readings_after_collection_are_dropped():
    """Показание, снятое до сбора, а пришедшее после сообщения о сборе, ещё «слышит» собранный образец."""
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    ch.post('tb2', 5.0, 'collected', x=0.5, y=0.5, left=4)
    bot._receive(_obs(t=5.0))
    before = bot.belief.p.copy()
    old = dict(x=0.0, y=0.5, th=0.0, battery=50.0, state='active', claim=None, soil=[], safe=[])
    ch.post('tb2', 5.2, 'obs', readings=[[0.0, 0.5, 0.75, 0.05, 4.8]], **old)
    bot._receive(_obs(t=5.2))
    assert np.array_equal(bot.belief.p, before)
    ch.post('tb2', 6.2, 'obs', readings=[[0.0, 0.5, 0.0, 0.05, 6.0]], **old)
    bot._receive(_obs(t=6.2))
    assert not np.array_equal(bot.belief.p, before)                       # свежее показание в карту идёт


def test_reading_is_sent_with_measurement_time_not_delivery_time():
    """В ROS показание приходит с опозданием (sensor_age). В канал идёт время измерения: иначе показание,
    снятое до сбора, но доставленное после, получатель принял бы за свежее."""
    ch = TeamChannel()
    sender, bot = _agent('tb2', channel=ch), _agent('tb1', channel=ch)
    ch.post('tb2', 5.0, 'collected', x=0.5, y=0.5, left=4)
    bot._receive(_obs(t=5.0))                                              # последний известный сбор — в 5,0
    before = bot.belief.p.copy()
    late = Observation(t=5.2, x=0.0, y=0.5, th=0.0, v=0.0, w=0.0, battery=50.0, sensor=0.75, scan=None, sensor_age=0.4)
    sender._on_reading(0.75, late)                                         # снято в 4,8, доставлено в 5,2
    sender._flush(5.2)
    assert ch.log[-1]['type'] == 'obs' and ch.log[-1]['t'] == 5.2
    assert [r[4] for r in ch.log[-1]['readings']] == [4.8]
    bot._receive(_obs(t=5.2))
    assert np.array_equal(bot.belief.p, before)                            # старое измерение карту не меняет
    fresh = Observation(t=5.6, x=0.0, y=0.5, th=0.0, v=0.0, w=0.0, battery=50.0, sensor=0.0, scan=None, sensor_age=0.4)
    sender._on_reading(0.0, fresh)                                         # снято в 5,2 — уже после сбора
    sender._flush(5.6)
    bot._receive(_obs(t=5.6))
    assert not np.array_equal(bot.belief.p, before)


def test_partner_reading_narrows_my_map():
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    far = np.hypot(bot.belief.cx - 1.0, bot.belief.cy - 1.0) < 0.6
    before = bot.belief.p[far].sum()
    ch.post('tb2', 1.0, 'obs', x=1.0, y=1.0, th=0.0, battery=55.0, state='active', claim=None, soil=[], safe=[],
            readings=[[1.0, 1.0, 0.0, 0.05, 1.0]] * 5)                    # напарник там и ничего не слышит
    bot._receive(_obs())
    assert bot.belief.p[far].sum() < 0.2 * before
    assert bot.partners['tb2']['x'] == 1.0


def test_hazard_and_soil_messages_reach_partner_map():
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    ch.post('tb2', 3.0, 'hazard', x=0.4, y=1.0, heading=0.0, trail=[[0.0, 1.0], [0.2, 1.0]])
    ch.post('tb2', 3.0, 'obs', x=0.4, y=1.0, th=0.0, battery=50.0, state='active', claim=None, readings=[],
            safe=[], soil=[[1.0, -1.0, 0.3, 3.0], [1.05, -1.0, 0.3, 3.0]])
    bot._receive(_obs(t=3.0))
    assert len(bot.hazards) == 1 and bot._trigger == 'partner_hazard'
    ix, iy = ARENA.w2g(0.65, 1.0)
    assert bot._risk[iy, ix] > 0.5                                         # впереди по ходу напарника — опасно
    assert bot.soil.at(1.0, -1.0)[0] > 2.0 and bot._cost_dirty             # дорогой грунт — на моей карте


def test_sectors_are_split_and_revised():
    ch = TeamChannel()
    bot = _agent('tb1', channel=ch)
    ch.post('tb2', 0.0, 'hello', x=SLOTS[1][0], y=SLOTS[1][1], home=list(SLOTS[1]))
    bot._receive(_obs(t=0.0))
    assert bot.sector['owners'] == {'neg': 'tb1', 'pos': 'tb2'}            # старший по имени делит арену
    assert bot._mine(0.0, -1.5) and not bot._mine(0.0, 1.5)                # tb1 стоит южнее — ему юг
    assert [m['type'] for m in ch.log if m['from'] == 'tb1'] == ['sector']
    ch.post('tb2', 40.0, 'status', state='returning', battery=12.0)
    bot._receive(_obs(t=40.0))
    assert bot.sector['all'] == 'tb1' and bot._mine(0.0, 1.5)              # напарник уехал — вся арена моя
    assert ch.log[-1]['type'] == 'sector' and ch.log[-1]['all'] == 'tb1'


def test_team_run_talks_and_does_not_duplicate():
    r = run_team_episode('medium', 3, 'team', save=False)['metrics']
    pair = run_team_episode('medium', 3, 'pair', save=False)['metrics']
    kinds = r['messages_by_type']
    assert kinds['hello'] == 2 and kinds['collected'] == r['samples_collected'] and kinds['sector'] >= 1
    assert kinds['claim'] >= 2 and kinds['obs'] > 50
    assert r['samples_collected'] == 5 and r['returned'] == 1.0 and r['robot_contacts'] == 0
    assert r['same_target_s'] < 3.0 < pair['same_target_s']                # без координации едут к одной цели
    assert pair['messages'] == 0


def test_team_trace_has_both_robots_and_messages(tmp_path, monkeypatch):
    import did.team_runner as tr
    from did.recorder import load_trace
    monkeypatch.setattr(tr, 'RUNS', tmp_path)
    s = run_team_episode('medium', 3, 'team', experiment='t')
    trace = load_trace(tmp_path / s['file'])
    assert [r['name'] for r in trace['robots']] == ['tb1', 'tb2']
    assert all(len(r['track']['t']) > 50 and r['journal'] for r in trace['robots'])
    assert trace['track'] == trace['robots'][0]['track']                   # прежний проигрыватель видит первого
    assert {'hello', 'claim', 'collected', 'sector', 'obs', 'status'} <= {m['type'] for m in trace['team']['messages']}
    assert trace['result']['score'] == pytest.approx(sum(r['result']['score'] for r in trace['robots']))


def test_one_agent_failure_does_not_stop_the_other(monkeypatch):
    """Агент одного робота упал: робот встаёт и молчит, второй замечает молчание, забирает арену и
    возвращается на базу; прогон кончается по общему сроку, а не обрывается исключением."""
    tick = TeamAgent.tick

    def broken(self, obs, io):
        if self.name == 'tb2' and obs.t >= 12.0:
            raise RuntimeError('датчик отвалился')
        return tick(self, obs, io)
    monkeypatch.setattr(TeamAgent, 'tick', broken)
    with pytest.raises(RuntimeError):                                      # в опытах ошибка агента видна
        run_team_episode('medium', 3, 'team', save=False)
    r = run_team_episode('medium', 3, 'team', save=False, tolerate_errors=True)['metrics']
    assert list(r['agent_errors']) == ['tb2'] and 'датчик отвалился' in r['agent_errors']['tb2']
    assert r['reason'] == 'finish+timeout'                                 # tb1 сам закончил на базе, tb2 простоял до срока
    assert r['returned'] == 0.5 and not r['returned_all']                  # первый вернулся, упавший — нет
    assert r['time'] == pytest.approx(Rules().time_limit_s, abs=0.2)       # общий срок прогона
    assert r['samples_collected'] >= 4 and r['robot_contacts'] == 0        # оставшийся доработал один


# --- запись Gazebo: склейка частей двух агентов -----------------------------------------------------

def test_min_gap_uses_common_times_not_indices():
    a = {'t': [0.0, 1.0], 'x': [0.0, 1.0], 'y': [0.0, 0.0]}
    b = {'t': [0.5, 1.5], 'x': [0.5, 1.5], 'y': [0.0, 0.0]}
    assert min_gap(a, b) == pytest.approx(0.0)                             # по номерам точек вышло бы 0,5 м
    # Закончивший робот стоит на месте и остаётся препятствием: второй проезжает мимо уже после конца его пути.
    stopped = {'t': [0.0, 1.0], 'x': [1.0, 1.0], 'y': [0.3, 0.3]}
    passing = {'t': [0.0, 5.0, 10.0], 'x': [-4.0, -4.0, 6.0], 'y': [0.0, 0.0, 0.0]}
    assert min_gap(stopped, passing) == pytest.approx(0.3)                 # и между точками, а не только в них
    assert min_gap({'t': [], 'x': [], 'y': []}, a) is None


def _part(name, home, t, x, collisions=1, judge=None):
    n = len(t)
    part = {'id': 'x', 'scenario': {'base': list(SLOTS[0])}, 'robot': {'name': name, 'home': list(home)},
            'result': {'score': 37.0, 'returned': True, 'battery': 20.0, 'distance': 3.0, 't': t[-1],
                       'collisions': collisions, 'false_collects': 0, 'hazard_hits': 0, 'samples_total': 5},
            'track': {'t': list(t), 'x': list(x), 'y': [0.0] * n, 'th': [0.0] * n, 'battery': [50.0] * n,
                      'sensor': [0.0] * n, 'mode': [0] * n},
            'modes': ['m'], 'events': [{'t': 1.0, 'type': 'sample_collected', 'sample': 0}], 'journal': [],
            'hypotheses': [], 'plans': [], 'paths': [], 'belief': None, 'soil': None, 'hazards': [],
            'pose_fix': [{'t': 0.0, 'dx': 0.01 if name == 'tb1' else 0.07, 'dy': 0.0, 'dth': 0.0}],
            'messages': [{'t': 0.0, 'from': 'tb1', 'type': 'hello'}]}
    if judge:
        part['judge'] = judge
    return part


def test_merged_gazebo_trace_keeps_judge_contacts():
    """Столкновения роботов друг с другом берутся у судьи на двоих и доходят до итога записи."""
    t1, t2 = [0.0, 1.0], [0.5, 1.5]
    parts = [_part('tb1', SLOTS[0], t1, [0.0, 1.0], judge={'t': 1.0, 'robot_contacts': 0, 'min_gap_m': 0.4}),
             _part('tb2', SLOTS[1], t2, [0.5, 1.5], judge={'t': 1.6, 'robot_contacts': 1, 'min_gap_m': 0.2})]
    trace = merge_parts(parts, 'team')
    r = trace['result']
    assert r['collisions'] == 2 and r['robot_contacts'] == 1               # не «сумма пополам»: что насчитал судья
    assert (r['min_gap_m'], r['min_gap_source']) == (0.2, 'judge')         # позже закончивший знает больше
    assert r['min_gap_track_m'] == pytest.approx(0.0)
    assert r['same_target_s'] is None and r['messages'] == 1               # повтор сообщения из двух журналов убран
    assert 'judge' not in trace and 'messages' not in trace and 'judge' in parts[0]
    # У каждого робота своя поправка положения и своё место на базе — для вида «глазами tb2».
    assert [x['pose_fix'][0]['dx'] for x in trace['robots']] == [0.01, 0.07]
    assert [x['home'] for x in trace['robots']] == [list(SLOTS[0]), list(SLOTS[1])]


def test_merged_gazebo_trace_without_judge_total_is_not_zero_contacts():
    """Запись, снятая до появления итога судьи: показатель не измерен, а не равен нулю."""
    parts = [_part('tb1', SLOTS[0], [0.0, 1.0], [0.0, 1.0]), _part('tb2', SLOTS[1], [0.5, 1.5], [0.5, 1.5])]
    r = merge_parts(parts, 'team')['result']
    assert r['collisions'] == 2 and r['robot_contacts'] is None
    assert (r['min_gap_m'], r['min_gap_source']) == (pytest.approx(0.0), 'tracks')


def test_lab_shows_unmeasured_contacts_and_own_base():
    """Интерфейс не подставляет ноль вместо отсутствующего показателя и показывает базу и поправку своего робота."""
    js = (Path(__file__).resolve().parents[1] / 'lab' / 'views' / 'run.js').read_text(encoding='utf-8')
    assert 'robot_contacts ?? 0' not in js and 'не измерено' in js
    assert 'base: r.home' in js and "own('pose_fix')" in js


def test_returned_all_is_in_run_metrics():
    r = run_team_episode('medium', 3, 'team', save=False)['metrics']
    assert r['returned_all'] is True and r['returned'] == 1.0
    from did.metrics import LATE_METRICS, METRICS
    assert 'returned_all' in METRICS and 'returned_all' in LATE_METRICS


# --- обычный прогон одного робота не изменился -----------------------------------------------------

@pytest.mark.skipif(not E1_REFERENCE.exists(), reason='нет сводки E1 основного каталога')
def test_e1_runs_match_main_summary():
    """Прогоны опыта E1 в этом дереве дают те же числа, что записаны в сводке основного каталога."""
    ref = {(r['arm'], r['level'], r['seed']): r['metrics'] for r in json.loads(E1_REFERENCE.read_text())['runs']}
    for arm in ('fixed', 'adaptive'):
        for level in ('easy', 'medium', 'hard'):
            for seed in (1001, 1002, 1003):
                got = run_episode(level, seed, arm, save=False)['metrics']
                want = ref[(arm, level, seed)]
                for k in ('score', 'samples_collected', 'returned', 'battery_left', 'distance', 'time', 'penalties'):
                    assert got[k] == want[k], (arm, level, seed, k)
