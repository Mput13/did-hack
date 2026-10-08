"""R16: робот не стоит, пока модель думает (did/waiting.py)."""
import math
import threading
from dataclasses import replace

import pytest

from did import runner
from did.agent import Agent, AgentConfig, make_config
from did.arena import load_arena
from did.fastsim import FastSim
from did.planner import HeuristicPlanner, resolve_subgoals
from did.recorder import Recorder
from did.runner import make_planner
from did.scenario import generate
from did.waiting import ActWhileWaiting, answer_delay, wait_metrics


@pytest.fixture(autouse=True)
def _runs_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'RUNS', tmp_path)


class Stub:
    """Планировщик-«модель»: отвечает функцией choose(state) -> подцели, считает вызовы."""

    def __init__(self, choose, latency_ms=3000):
        self.choose, self.latency_ms, self.calls = choose, latency_ms, 0

    def plan(self, state):
        self.calls += 1
        return {'reasoning': 'ответ модели', 'hypotheses': [], 'subgoals': self.choose(state), 'source': 'llm',
                'exchanges': [{'attempt': 1, 'ok': True, 'latency_ms': self.latency_ms}], 'error': None}


def as_rule(state):
    return HeuristicPlanner().plan(state)['subgoals']


def other_point(state):
    """Достижимая точка разведки, которую правило НЕ выбрало."""
    rule = as_rule(state)[0].get('target')
    rest = [p['id'] for p in state['explore_points'] if p['feasible'] and p['id'] != rule]
    return [{'type': 'explore', 'target': rest[-1]}] if rest else as_rule(state)


def setup(level='medium', seed=1001, planner=None, **cfg):
    arena = load_arena()
    cfg = make_config('adaptive_llm', **cfg)
    scenario = generate(level, seed, arena)
    world = FastSim(arena, scenario, seed=seed)
    rec = Recorder()
    bot = Agent(arena, cfg, n_samples=len(scenario.samples), recorder=rec,
                planner=planner or make_planner(cfg, {'kind': 'mock'}, seed))
    return world, bot, rec


def drive(world, bot, until, each=None):
    while world.t < until and not world.done:
        obs = world.observe()
        bot.tick(obs, world)
        if each:
            each(obs)
        world.advance()


def test_off_by_default_and_only_for_the_model_planner():
    assert AgentConfig().llm_act_while_waiting == 'off'
    _, bot, _ = setup()
    assert bot.aw is None
    arena = load_arena()
    rule = Agent(arena, make_config('adaptive', llm_act_while_waiting='rule'), n_samples=5)
    assert rule.aw is None                                   # у правила модели нет — ждать некого
    with pytest.raises(ValueError):
        Agent(arena, make_config('adaptive_llm', llm_act_while_waiting='fast'), n_samples=5)


def test_robot_follows_the_rule_plan_while_the_answer_is_pending():
    world, bot, _ = setup(llm_wait_s=15.0, llm_act_while_waiting='rule')
    ref_world, ref, _ = setup()                              # тот же сценарий без ожидания
    start = (world.x, world.y)
    modes = set()
    drive(world, bot, 6.0, each=lambda obs: modes.add(bot.mode))
    drive(ref_world, ref, 6.0)
    assert bot.aw.pending is not None and bot.planner.calls == 1
    assert 'think' not in modes and math.dist((world.x, world.y), start) > 0.3
    assert (world.x, world.y) == (ref_world.x, ref_world.y)  # едет ровно так, как ехал бы без ожидания
    assert bot._future is None and bot._wait_until == 0.0    # прежняя стоянка не включалась


def test_agreeing_answer_changes_nothing():
    world, bot, rec = setup(planner=Stub(as_rule), llm_wait_s=1.0, llm_act_while_waiting='rule')
    drive(world, bot, 0.5)
    goal = bot.queue[0]
    drive(world, bot, 1.3)
    assert bot.aw.stats['agree'] == 1 and bot.aw.pending is None
    assert bot.queue[0] is goal
    assert rec.plans[0]['source'] == 'heuristic' and rec.plans[0]['rule_match'] is True
    assert len(rec.llm) == 1                                 # обмен с моделью попал в запись
    assert any((e.get('data') or {}).get('outcome') == 'agree' for e in bot.journal.entries)


def test_different_answer_that_is_still_feasible_takes_over():
    world, bot, rec = setup(planner=Stub(other_point), llm_wait_s=1.0, llm_act_while_waiting='rule')
    drive(world, bot, 0.5)
    rule_goal = dict(bot.queue[0])
    state = bot.aw.pending['state']
    want = resolve_subgoals(other_point(state), state)[0]
    assert (want['x'], want['y']) != (rule_goal['x'], rule_goal['y'])
    drive(world, bot, 1.3)
    assert bot.aw.stats['switched'] == 1
    assert (bot.queue[0]['x'], bot.queue[0]['y']) == (want['x'], want['y'])
    last = rec.plans[-1]
    assert last['source'] == 'llm' and last['rule_match'] is False and last['asked_t'] == 0.0
    assert last['rule_first']['target'] == rule_goal['target']


def test_answer_about_a_point_already_seen_is_dropped_as_stale():
    world, bot, _ = setup(planner=Stub(other_point), llm_wait_s=1.0, llm_act_while_waiting='rule')
    drive(world, bot, 0.5)
    state = bot.aw.pending['state']
    want = resolve_subgoals(other_point(state), state)[0]
    goal = bot.queue[0]
    bot._visited.append((world.t, want['x'], want['y']))     # за время ожидания точку успели осмотреть
    drive(world, bot, 1.3)
    assert bot.aw.stats['stale'] == 1 and bot.aw.stats['switched'] == 0
    assert bot.queue[0] is goal
    notes = [e for e in bot.journal.entries if (e.get('data') or {}).get('outcome') == 'stale']
    assert notes and 'устарел' in notes[0]['text']


def test_answer_is_checked_against_the_present_state():
    world, bot, _ = setup(planner=Stub(other_point), llm_wait_s=30.0, llm_act_while_waiting='rule')
    drive(world, bot, 1.0)
    obs = world.observe()
    state = bot.aw.pending['state']
    want = resolve_subgoals(other_point(state), state)
    assert bot.aw._still_valid(obs, want)[0]                 # сейчас цель выполнима
    valid, why = bot.aw._still_valid(replace(obs, battery=1.0), want)
    assert valid == [] and 'заряда' in why                   # заряда уже не хватает
    gone = [{'type': 'investigate', 'target': 'C1', 'x': 1.0, 'y': 1.0}]
    valid, why = bot.aw._still_valid(obs, gone)
    assert valid == [] and 'кандидата' in why                # кандидата на месте нет (собран или не подтвердился)
    assert bot.aw._still_valid(obs, [{'type': 'return_base'}])[0] == [{'type': 'return_base'}]


def test_new_trigger_while_pending_goes_to_the_rule_and_requests_do_not_pile_up():
    stub = Stub(as_rule)
    world, bot, _ = setup(planner=stub, llm_wait_s=15.0, llm_act_while_waiting='rule')
    drive(world, bot, 3.0)
    for _ in range(3):                                       # три повода подряд, пока ответа нет
        bot._request_plan('test')
        drive(world, bot, world.t + 1.0)
    assert stub.calls == 1 and bot.aw.stats['rule_while_pending'] >= 3
    assert bot.queue and bot._trigger is None                # каждый повод сразу решён правилом
    drive(world, bot, 15.3)
    # Ответ пришёл: план с тех пор сменился, поэтому модель спрашивают заново — снова один запрос в работе.
    assert bot.aw.stats['answered'] == 1 and bot.aw.stats['agree_late'] == 1
    assert stub.calls == 2 and bot.aw.stats['asked_again'] == 1 and bot.aw.pending is not None


def test_return_to_base_is_final():
    world, bot, _ = setup(planner=Stub(other_point), llm_wait_s=1.0, llm_act_while_waiting='rule')
    drive(world, bot, 0.5)
    bot._go_home(world.t, 'тест')
    drive(world, bot, 1.3)
    assert bot.aw.stats['returning'] == 1 and bot._returning
    assert bot.queue == [{'type': 'return_base'}]


def test_safe_mode_neither_collects_nor_leaves_the_leash_before_the_answer():
    world, bot, _ = setup(level='hard', seed=1, llm_wait_s=15.0, llm_act_while_waiting='safe')
    seen = {'collect': 0, 'far': 0.0, 'stood_in_danger': 0, 'held': 0}
    collect = world.collect

    def counted():
        seen['collect'] += bot.aw.pending is not None
        return collect()
    world.collect = counted

    escaped = set()                                          # вопросы, во время которых робот выезжал из зоны

    def each(obs):
        if bot.aw.pending is None or bot._returning:
            return
        danger = bot.aw.in_danger(obs)
        seen['held'] += bot.mode == 'think'
        seen['stood_in_danger'] += danger and bot.mode == 'think'
        if danger:
            escaped.add(bot.aw.pending['t'])
        elif bot.aw.pending['t'] not in escaped:
            seen['far'] = max(seen['far'], math.dist((obs.x, obs.y), bot.aw.pending['at']))
    drive(world, bot, 600.0, each=each)
    assert seen['collect'] == 0                              # до ответа модели сбора не было
    assert seen['held'] > 0 and seen['stood_in_danger'] == 0
    assert seen['far'] <= bot.cfg.llm_wait_leash_m + 0.05    # привязь (запас — путь за один такт)
    assert world.judge.score()['samples_collected'] >= 5     # а после ответа робот собирает как обычно


def test_waiting_in_a_hazard_zone_no_longer_costs_a_penalty_every_five_seconds():
    def run(**cfg):
        return runner.run_episode('hard', 1, 'adaptive_llm', llm={'kind': 'mock'}, save=False,
                                  config={'llm_wait_s': 15, **cfg})['metrics']
    stand, act, safe = run(), run(llm_act_while_waiting='rule'), run(llm_act_while_waiting='safe')
    rule = runner.run_episode('hard', 1, 'adaptive', save=False)['metrics']
    assert stand['hazard_hits'] >= 5 and stand['idle_s'] > 100          # как сейчас: штраф за штрафом
    assert act['hazard_hits'] <= rule['hazard_hits'] and safe['hazard_hits'] <= rule['hazard_hits']
    assert act['score'] == rule['score'] and act['idle_s'] == rule['idle_s']
    assert safe['score'] > stand['score'] + 50 and safe['idle_s'] < stand['idle_s']
    assert act['llm_wait']['asked'] == act['llm_wait']['answered'] + act['llm_wait']['unanswered']


def test_flag_off_keeps_the_old_waiting():
    old = runner.run_episode('medium', 1001, 'adaptive_llm', llm={'kind': 'mock'}, save=False,
                             config={'llm_wait_s': 15})['metrics']
    assert old['idle_s'] > 100 and 'llm_wait' not in old and 'llm_late' not in old


def test_threaded_answer_robot_drives_meanwhile_and_takes_the_answer_when_it_is_ready():
    """Как в ROS: ответ считает поток агента (async_planner), готовность — по самому потоку."""
    gate = threading.Event()

    class Slow(Stub):
        def plan(self, state):
            assert gate.wait(timeout=20.0)
            return super().plan(state)

    stub = Slow(other_point)
    world, bot, rec = setup(planner=stub, async_planner=True, llm_act_while_waiting='rule')
    try:
        start = (world.x, world.y)
        modes = set()
        drive(world, bot, 2.0, each=lambda obs: modes.add(bot.mode))
        assert math.dist((world.x, world.y), start) > 0.1 and 'think' not in modes   # едет, пока поток занят
        assert bot.aw.pending is not None and bot._future is None
        bot._request_plan('test')
        drive(world, bot, 2.5)
        assert bot.aw.stats['asked'] == 1 and bot.aw.stats['rule_while_pending'] == 1   # второй запрос не ушёл
        state = bot.aw.pending['state']
        want = resolve_subgoals(other_point(state), state)[0]
        future = bot.aw.pending['answer'].future
        gate.set()
        future.result(timeout=20.0)
        drive(world, bot, world.t + 0.2)
        assert bot.aw.stats['answered'] == 1 and bot.aw.stats['switched'] == 1
        assert (bot.queue[0]['x'], bot.queue[0]['y']) == (want['x'], want['y'])
        assert len(rec.llm) == 1
    finally:
        gate.set()
        bot._pool.shutdown(wait=True)


def test_failure_in_the_model_thread_leaves_the_rule_plan():
    class Broken:
        calls = 0

        def plan(self, state):
            raise RuntimeError('сеть упала')

    world, bot, _ = setup(planner=Broken(), async_planner=True, llm_act_while_waiting='rule')
    try:
        drive(world, bot, 0.3)
        if bot.aw.pending is not None:
            bot.aw.pending['answer'].future.exception(timeout=20.0)
        drive(world, bot, 1.0)
        assert bot.aw.stats['failed'] == 1 and bot.queue and bot.mode != 'think'
    finally:
        bot._pool.shutdown(wait=True)


def test_answer_delay_fixed_or_measured():
    plan = {'exchanges': [{'latency_ms': 1200}, {'latency_ms': 800}]}
    assert answer_delay(make_config('adaptive_llm', llm_wait_s=15.0), plan) == 15.0
    assert answer_delay(make_config('adaptive_llm', llm_wait_s=15.0, llm_wait_measured=True), plan) == 2.0
    assert answer_delay(make_config('adaptive_llm', llm_wait_measured=True), {'exchanges': []}) == 0.0


def test_measured_delay_makes_the_old_mode_stand_for_the_real_answer_time():
    world, bot, _ = setup(planner=Stub(as_rule, latency_ms=4000), llm_wait_measured=True)
    start = (world.x, world.y)
    drive(world, bot, 3.5)
    assert (world.x, world.y) == start and bot.mode == 'think'
    drive(world, bot, 6.0)
    assert (world.x, world.y) != start


def test_wait_metrics_count_seconds_in_think_mode():
    rec = Recorder()
    rec.modes = ['explore', 'think']
    rec.track['mode'] = [0, 1, 1, 0, 1]
    assert wait_metrics(rec, object()) == {'idle_s': 0.6}


def test_mock_deviate_names_another_feasible_target():
    import json

    from did.llm import EXAMPLE_STATE, build_messages
    from did.llm_mock import MockResponder, mock_plan
    state = {**EXAMPLE_STATE, 'explore_points': EXAMPLE_STATE['explore_points'] + [
        {'id': 'E2', 'x': -1.0, 'y': 1.0, 'unseen_share': 0.2, 'cost_to': 3.0, 'cost_back': 4.0, 'feasible': True}]}
    messages = build_messages(state)
    rule = mock_plan(state)['subgoals']
    assert json.loads(MockResponder(seed=3)(messages))['subgoals'] == rule
    assert json.loads(MockResponder(seed=3, deviate=0.0)(messages))['subgoals'] == rule
    always = MockResponder(seed=3, deviate=1.0)
    for _ in range(5):
        got = json.loads(always(messages))['subgoals']
        assert len(got) == 1 and got[0] != rule[0] and got[0]['target'] in ('C1', 'E1', 'E2')


def test_mission_imitator_reads_the_mission():
    import json

    from did.llm import EXAMPLE_STATE, build_messages
    from did.wait_eval import MissionResponder
    two = json.loads(MissionResponder('M1')(build_messages(EXAMPLE_STATE)))       # в примере собрано два
    assert two['subgoals'] == [{'type': 'return_base'}]
    one = {**EXAMPLE_STATE, 'samples': {'collected': 1, 'total': 5}}
    assert json.loads(MissionResponder('M1')(build_messages(one)))['subgoals'][0]['type'] == 'investigate'
    left = json.loads(MissionResponder('M3')(build_messages(EXAMPLE_STATE)))      # обе цели примера — справа
    assert left['subgoals'] == [{'type': 'return_base'}]


def test_aw_is_plain_object_with_summary():
    _, bot, _ = setup(llm_wait_s=15.0, llm_act_while_waiting='rule')
    assert isinstance(bot.aw, ActWhileWaiting)
    assert bot.aw.summary()['asked'] == 0 and bot.aw.summary()['unanswered'] == 0


def test_ros_node_takes_the_mode_from_config():
    """В ROS режим включается настройкой агента; ответ там приходит из потока (async_planner), код общий."""
    pytest.importorskip('rclpy')
    import inspect

    from did import ros_agent
    assert 'config' in inspect.signature(ros_agent.run).parameters
    source = inspect.getsource(ros_agent.run)
    assert 'make_config(agent, async_planner=llm_on, **(config or {}))' in source
    cfg = make_config('adaptive_llm', async_planner=True, llm_act_while_waiting='rule')
    bot = Agent(load_arena(), cfg, n_samples=5)
    try:
        assert bot.aw is not None and bot._pool is not None
    finally:
        bot._pool.shutdown(wait=True)


def _late_answer(choose, **cfg):
    """Ответ приходит после того, как правило уже решало заново: вернуть агента и цель перед ответом."""
    world, bot, _ = setup(planner=Stub(choose), llm_wait_s=1.0, llm_act_while_waiting='rule', **cfg)
    drive(world, bot, 0.5)
    bot._request_plan('test')                                # новый повод, пока ответа нет
    drive(world, bot, 0.8)
    assert bot.aw.pending['superseded'] == 1
    goal = bot.queue[0]
    drive(world, bot, 1.3)
    return bot, goal


def test_late_answer_that_differs_applies_or_drops_by_setting():
    bot, goal = _late_answer(other_point)
    assert bot.aw.stats['switched'] == 1 and bot.queue[0] is not goal            # apply: выполним сейчас — переход
    bot, goal = _late_answer(other_point, llm_wait_superseded='drop')
    assert bot.aw.stats['stale'] == 1 and bot.queue[0] is goal                   # drop: решение правила свежее
    home = lambda state: [{'type': 'return_base'}]                               # noqa: E731
    bot, _ = _late_answer(home, llm_wait_superseded='drop')
    assert bot.aw.stats['switched'] == 1 and bot._returning                      # возврат на базу принимается всегда
    with pytest.raises(ValueError):
        setup(llm_act_while_waiting='rule', llm_wait_superseded='maybe')


def _mission(m_id, level, seed, **cfg):
    from did.llm import LocalClient
    from did.mission_criteria import MISSIONS
    from did.wait_eval import ASK, MissionResponder
    trace = {}
    res = runner.run_episode(level, seed, 'adaptive_llm', truth=True, experiment='t', arm='m',
                             config={'mission': MISSIONS[m_id]['text'], **ASK, 'llm_wait_s': 15.0, **cfg},
                             llm={'client': LocalClient(MissionResponder(m_id))})
    from did.recorder import load_trace
    trace = load_trace(runner.RUNS / res['file'])
    return res['metrics'], trace


def test_mission_exactly_two_rule_plan_overshoots_and_safe_mode_does_not():
    """M1: правило миссию не читает и в ожидании собирает третий образец; safe до ответа не собирает."""
    act, _ = _mission('M1', 'medium', 1001, llm_act_while_waiting='rule')
    safe, _ = _mission('M1', 'medium', 1001, llm_act_while_waiting='safe')
    assert act['samples_collected'] > 2
    assert safe['samples_collected'] == 2 and safe['returned']


def test_mission_left_half_only_standing_safe_mode_keeps_out_of_the_right_half():
    """M3: с планом правила робот уезжает в запретную половину; safe без привязи стоит до ответа."""
    act, t_act = _mission('M3', 'medium', 1003, llm_act_while_waiting='rule')
    still, t_still = _mission('M3', 'medium', 1003, llm_act_while_waiting='safe', llm_wait_leash_m=0.0)
    assert max(t_act['truth']['x']) > 0.5
    assert max(t_still['truth']['x']) <= 0.5 and still['returned']
