"""Тесты настраиваемой миссии и критериев M0–M4 (R13)."""
import pytest
from did.agent import Agent, AgentConfig, make_config
from did.arena import load_arena
from did.config import BASE, Rules
from did.fastsim import Observation
from did.llm import LocalClient, request_plan
from did.llm_mock import MockResponder
from did.mission_criteria import MISSIONS, verify_mission
from did.planner import MISSION, HeuristicPlanner, LLMPlanner


def test_mission_default_unchanged():
    cfg = AgentConfig()
    assert cfg.mission == MISSION
    assert cfg.mission_triggers is False


def test_custom_mission_in_state_and_prompt():
    custom_text = 'Собери ровно два образца и сразу возвращайся на базу.'
    seen_states = []

    class CapturingResponder:
        def __call__(self, messages):
            import json
            for m in messages:
                if m['role'] == 'user' and 'Состояние:' in m['content']:
                    raw = m['content'].split('Состояние:\n', 1)[1].rsplit('\nВерни план:', 1)[0]
                    seen_states.append(json.loads(raw))
            return json.dumps({
                'reasoning': 'Тестовый ответ',
                'hypotheses': [],
                'subgoals': [{'type': 'return_base'}]
            })

    client = LocalClient(CapturingResponder())
    planner = LLMPlanner(client, mission=custom_text)
    state = {'candidates': [], 'explore_points': [], 'samples': {'collected': 0, 'total': 3}}
    res = planner.plan(state)
    assert len(seen_states) == 1
    assert seen_states[0]['mission'] == custom_text


def test_agent_passes_custom_mission_to_planner():
    custom_text = 'Особая тестовая миссия агента'
    seen_states = []

    class CapturingResponder:
        def __call__(self, messages):
            import json
            for m in messages:
                if m['role'] == 'user' and 'Состояние:' in m['content']:
                    raw = m['content'].split('Состояние:\n', 1)[1].rsplit('\nВерни план:', 1)[0]
                    seen_states.append(json.loads(raw))
            return json.dumps({
                'reasoning': 'Тест',
                'hypotheses': [],
                'subgoals': [{'type': 'return_base'}]
            })

    client = LocalClient(CapturingResponder())
    planner = LLMPlanner(client)
    cfg = AgentConfig(name='adaptive_llm', planner='llm', mission=custom_text)
    arena = load_arena()
    bot = Agent(arena, cfg, n_samples=3, planner=planner)

    obs = Observation(t=1.0, x=-2.0, y=-0.5, th=0.0, v=0.0, w=0.0, battery=55.0,
                      sensor=0.1, scan=None, scan_pose=None, scan_step=1, events=[], done=False)

    class MockIO:
        def command(self, v, w): pass
        def collect(self): return True, 'ok'
        def finish(self): return True, 'ok'

    bot.tick(obs, MockIO())
    assert len(seen_states) >= 1
    assert seen_states[0]['mission'] == custom_text


def test_verify_mission_m0():
    res_ok = {'returned': True, 'samples_collected': 2, 'battery': 25.0}
    assert verify_mission('M0', {}, [], res_ok)['success'] is True

    res_no_samples = {'returned': True, 'samples_collected': 0, 'battery': 40.0}
    assert verify_mission('M0', {}, [], res_no_samples)['success'] is False

    res_not_returned = {'returned': False, 'samples_collected': 3, 'battery': 0.0}
    assert verify_mission('M0', {}, [], res_not_returned)['success'] is False


def test_verify_mission_m1():
    assert verify_mission('M1', {}, [], {'returned': True, 'samples_collected': 2})['success'] is True
    assert verify_mission('M1', {}, [], {'returned': True, 'samples_collected': 3})['success'] is False
    assert verify_mission('M1', {}, [], {'returned': True, 'samples_collected': 1})['success'] is False
    assert verify_mission('M1', {}, [], {'returned': False, 'samples_collected': 2})['success'] is False


def test_verify_mission_m2():
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 30.0})['success'] is True
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 35.5})['success'] is True
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 29.9})['success'] is False
    assert verify_mission('M2', {}, [], {'returned': False, 'battery': 45.0})['success'] is False


def test_verify_mission_m3():
    good_track = {'x': [-2.0, -1.5, -0.5, 0.2, 0.49, 0.1]}
    bad_track = {'x': [-2.0, -1.5, 0.0, 0.8, 1.2, 0.9]}
    events_left = [{'type': 'sample_collected', 'x': -0.4, 'y': 1.0, 't': 5.0}]
    events_right = [{'type': 'sample_collected', 'x': 0.7, 'y': 1.0, 't': 5.0}]

    assert verify_mission('M3', good_track, events_left, {'returned': True})['success'] is True
    assert verify_mission('M3', bad_track, events_left, {'returned': True})['success'] is False
    assert verify_mission('M3', good_track, events_right, {'returned': True})['success'] is False
    assert verify_mission('M3', good_track, events_left, {'returned': False})['success'] is False


def test_verify_mission_m4():
    no_penalties = [{'type': 'sample_collected', 't': 5.0}]
    assert verify_mission('M4', {}, no_penalties, {'returned': True})['success'] is True

    pen_and_stop = [
        {'type': 'sample_collected', 't': 4.0},
        {'type': 'hazard_hit', 't': 8.0, 'x': 0.0, 'y': 0.0},
    ]
    assert verify_mission('M4', {}, pen_and_stop, {'returned': True})['success'] is True

    pen_and_continue = [
        {'type': 'hazard_hit', 't': 8.0, 'x': 0.0, 'y': 0.0},
        {'type': 'sample_collected', 't': 12.0},
    ]
    assert verify_mission('M4', {}, pen_and_continue, {'returned': True})['success'] is False

    assert verify_mission('M4', {}, pen_and_stop, {'returned': False})['success'] is False


def test_mission_triggers_flag_behavior():
    arena = load_arena()
    obs_coll = Observation(t=2.0, x=0.0, y=0.0, th=0.0, v=0.0, w=0.0, battery=50.0,
                           sensor=0.0, scan=None, scan_pose=None, scan_step=1,
                           events=[{'type': 'collision', 'x': 0.0, 'y': 0.0}], done=False)

    class MockIO:
        def command(self, v, w): pass
        def collect(self): return True, 'ok'
        def finish(self): return True, 'ok'

    # 1. По умолчанию выключено
    bot_default = Agent(arena, AgentConfig(mission_triggers=False), n_samples=3)
    bot_default.tick(obs_coll, MockIO())
    assert bot_default._trigger != 'collision'

    # 2. При mission_triggers=True повод уходит в планировщик
    seen_triggers = []
    planner_triggered = HeuristicPlanner()
    orig_plan = planner_triggered.plan
    def track_plan(state):
        seen_triggers.append(state.get('trigger'))
        return orig_plan(state)
    planner_triggered.plan = track_plan

    bot_triggered = Agent(arena, AgentConfig(mission_triggers=True), n_samples=3, planner=planner_triggered)
    bot_triggered.tick(obs_coll, MockIO())
    assert 'collision' in seen_triggers


def _capture_planner(seen):
    def responder(messages):
        import json
        raw = messages[-1]['content'].split('Состояние:\n', 1)[1].rsplit('\nВерни план:', 1)[0]
        seen.append(json.loads(raw))
        return json.dumps({'reasoning': 'Тест', 'hypotheses': [], 'subgoals': [{'type': 'explore', 'target': 'E1'}]})
    return LLMPlanner(LocalClient(responder))


class _IO:
    def command(self, v, w): pass
    def collect(self): return True, 'ok'
    def finish(self): return True, 'ok'


def _obs(t, battery, events=()):
    return Observation(t=t, x=-2.0, y=-0.5, th=0.0, v=0.0, w=0.0, battery=battery, sensor=0.1, scan=None,
                       scan_pose=None, scan_step=1, events=list(events), done=False)


def test_default_request_keeps_mission_first():
    """Запрос по умолчанию не изменился: миссия прежняя и стоит первым ключом (от этого зависит кэш ответов)."""
    seen = []
    bot = Agent(load_arena(), AgentConfig(name='adaptive_llm', planner='llm'), n_samples=3,
                planner=_capture_planner(seen))
    bot.tick(_obs(1.0, 55.0), _IO())
    assert list(seen[0])[0] == 'mission' and seen[0]['mission'] == MISSION


def test_battery_floor_trigger_waits_for_model():
    """Повод «заряд у порога миссии» не тратится, пока на него ответило бы правило, и срабатывает один раз."""
    seen = []
    cfg = AgentConfig(name='adaptive_llm', planner='llm', mission_triggers=True)
    bot = Agent(load_arena(), cfg, n_samples=3, planner=_capture_planner(seen))
    bot.tick(_obs(1.0, 55.0), _IO())                 # план на старте: модель спрошена в t = 1
    bot.tick(_obs(2.5, 29.0), _IO())                 # заряд уже ниже порога, но модель спрашивали 1,5 с назад
    assert [s['trigger'] for s in seen] == ['start']
    bot.tick(_obs(6.0, 28.0), _IO())
    assert [s['trigger'] for s in seen] == ['start', 'battery_threshold']
    bot.tick(_obs(12.0, 27.0), _IO())
    assert [s['trigger'] for s in seen].count('battery_threshold') == 1

    off = []
    bot = Agent(load_arena(), AgentConfig(name='adaptive_llm', planner='llm'), n_samples=3,
                planner=_capture_planner(off))
    for t, b in ((1.0, 55.0), (6.0, 28.0), (12.0, 27.0)):
        bot.tick(_obs(t, b), _IO())
    assert 'battery_threshold' not in [s['trigger'] for s in off]


def test_verify_mission_details():
    hit = [{'type': 'false_collect', 't': 8.0, 'x': 0.0, 'y': 0.0}]
    d = verify_mission('M4', {}, hit, {'returned': True, 't': 20.0})['details']
    assert d['had_penalty'] and d['after_penalty_s'] == 12.0
    assert verify_mission('M4', {}, [], {'returned': True})['details']['had_penalty'] is False
    # M3: ровно на границе x = 0,5 — ещё левая половина; 1 точка из 50 справа — в пределах допуска 1% нет
    assert verify_mission('M3', {'x': [0.5] * 50}, [], {'returned': True})['success'] is True
    res = verify_mission('M3', {'x': [0.0] * 49 + [0.6]}, [], {'returned': True})
    assert res['success'] is False and res['details']['max_x'] == 0.6


def test_penalties_in_state_only_with_flag():
    hit = [{'type': 'false_collect', 't': 1.0, 'x': -2.0, 'y': -0.5}]
    for flag in (False, True):
        seen = []
        cfg = AgentConfig(name='adaptive_llm', planner='llm', state_penalties=flag)
        bot = Agent(load_arena(), cfg, n_samples=3, planner=_capture_planner(seen))
        bot.tick(_obs(1.0, 55.0, hit), _IO())
        assert ('penalties' in seen[0]) is flag
        if flag:
            assert seen[0]['penalties'] == {'total': 1, 'hazard_hit': 0, 'false_collect': 1, 'collision': 0}
