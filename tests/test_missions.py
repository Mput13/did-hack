"""Тесты настраиваемой миссии и критериев M0–M4 (R13)."""
import pytest
from did.agent import Agent, AgentConfig, make_config
from did.arena import load_arena
from did.config import BASE, Rules
from did.fastsim import Observation
from did.llm import LocalClient, request_plan
from did.llm_mock import MockResponder
from did.mission_criteria import MISSIONS, RETURN_DELAY_S, return_decision, verify_mission
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


def _track(switches, t_end, dt=0.2):
    """Трек с режимами: switches — [(t, режим), ...] по возрастанию времени."""
    modes = []
    for _, m in switches:
        if m not in modes:
            modes.append(m)
    ts, ms = [], []
    for i in range(int(round(t_end / dt)) + 1):
        t = round(i * dt, 2)
        ts.append(t)
        ms.append(modes.index([m for s, m in switches if s <= t + 1e-9][-1]))
    return {'t': ts, 'mode': ms}, modes


def _plan(t, kinds, source='llm', trigger='x'):
    return {'t': t, 'source': source, 'trigger': trigger, 'reasoning': '', 'subgoals': [{'type': k} for k in kinds]}


def _m1(t_return, extra_plans=(), collected=2, returned=True):
    events = [{'type': 'sample_collected', 't': 5.0}, {'type': 'sample_collected', 't': 10.0}][:collected]
    track, modes = _track([(0.0, 'explore'), (t_return, 'return')], t_return + 10.0)
    plans = [_plan(0.0, ['explore']), *extra_plans, _plan(t_return, ['return_base'])]
    return verify_mission('M1', track, events, {'returned': returned, 'samples_collected': collected},
                          plans=plans, modes=modes)


def test_verify_mission_m1():
    ok = _m1(11.0)                                    # пауза исполнителя после сбора — 1 с
    assert ok['success'] is True and ok['outcome'] == 'done' and ok['details']['return_delay_s'] == 1.0
    assert _m1(11.0, collected=1)['outcome'] == 'wrong_count'
    assert _m1(11.0, returned=False)['success'] is False
    # «сразу» проверяется: два образца и возврат через минуту — не выполнено
    late = _m1(70.0)
    assert late['success'] is False and late['outcome'] == 'late_return'
    assert _m1(10.0 + RETURN_DELAY_S)['success'] is True
    assert _m1(10.0 + RETURN_DELAY_S + 0.2)['success'] is False
    # после второго сбора модель назначила ещё один сбор (не удался, счёт остался 2) — не выполнено
    again = _m1(11.8, extra_plans=[_plan(11.0, ['investigate'])])
    assert again['success'] is False and again['outcome'] == 'collect_after_second'
    # без планов и режимов «сразу» проверить нечем: не «выполнено», а «не проверено»
    blind = verify_mission('M1', {}, [], {'returned': True, 'samples_collected': 2})
    assert blind['success'] is None and blind['outcome'] == 'unverified'
    assert verify_mission('M1', {}, [], {'returned': True, 'samples_collected': 3})['success'] is False


def test_verify_mission_m2():
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 30.0})['success'] is True
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 35.5})['success'] is True
    assert verify_mission('M2', {}, [], {'returned': True, 'battery': 29.9})['success'] is False
    assert verify_mission('M2', {}, [], {'returned': False, 'battery': 45.0})['success'] is False


def _truth(xs, dt=0.1):
    return {'dt': dt, 'x': list(xs), 'y': [0.0] * len(xs)}


def test_verify_mission_m3_uses_true_path():
    home = {'returned': True}
    clean = verify_mission('M3', {}, [], home, truth=_truth([-2.0, -1.0, 0.2, 0.5, 0.1]))
    assert clean['success'] is True and clean['outcome'] == 'clean'         # ровно на черте — ещё слева
    assert clean['details']['max_x'] == 0.5 and clean['details']['time_right_s'] == 0.0
    # один шаг симулятора за чертой из шести тысяч — заезд был: допуска в долях нет
    one = verify_mission('M3', {}, [], home, truth=_truth([0.0] * 3000 + [0.5001] + [0.0] * 2999))
    assert one['success'] is False and one['outcome'] == 'entered_right'
    assert one['details']['time_right_s'] == 0.1 and one['details']['first_right_t'] == 300.0
    # оценка позы агентом говорит «слева», истинная траектория — «справа»: решает истинная
    lie = verify_mission('M3', {'x': [0.0] * 100}, [], home, truth=_truth([0.0] * 50 + [2.0] * 50))
    assert lie['success'] is False and lie['details']['max_x'] == 2.0
    assert verify_mission('M3', {}, [], {'returned': False}, truth=_truth([0.0]))['success'] is False


def test_verify_mission_m3_no_truth_is_unverified():
    """Примеры из ревью: без истинной траектории исход — «не проверено», а не успех."""
    home = {'returned': True}
    for track in ({'x': [0.0] * 99 + [2.0]}, {}, {'x': [0.0] * 100}):
        res = verify_mission('M3', track, [], home)
        assert res['success'] is None and res['outcome'] == 'unverified'
    assert verify_mission('M3', {}, [], home, truth={'dt': 0.1, 'x': []})['success'] is None
    # образец, собранный справа (место — из события судьи, оно истинное), — провал и без траектории
    right = [{'type': 'sample_collected', 'x': 0.7, 'y': 1.0, 't': 5.0}]
    assert verify_mission('M3', {}, right, home)['success'] is False


def _m4(t_penalty, t_return, events=(), returned=True, source='llm', t_end=None):
    events = [*events, {'type': 'hazard_hit', 't': t_penalty, 'x': 0.0, 'y': 0.0}]
    switches = [(0.0, 'explore')] + ([(t_return, 'return')] if t_return is not None else [])
    track, modes = _track(switches, t_end or max(t_penalty, t_return or 0.0) + 15.0)
    plans = [_plan(0.0, ['explore'])] + ([_plan(t_return, ['return_base'], source)] if t_return is not None else [])
    return verify_mission('M4', track, sorted(events, key=lambda e: e['t']), {'returned': returned, 't': 60.0},
                          plans=plans, modes=modes)


def test_verify_mission_m4_three_outcomes():
    ok = _m4(8.0, 8.0)
    assert ok['success'] is True and ok['outcome'] == 'returned_on_penalty' and ok['details']['reaction_s'] == 0.0
    assert _m4(8.0, 8.0 + RETURN_DELAY_S)['success'] is True
    # случай из ревью (M4_llm_ask/hard-1009): возврат решён в 33,0 с, первый штраф — в 37,5 с по дороге домой
    early = _m4(37.5, 33.0, events=[{'type': 'sample_collected', 't': 29.8}])
    assert early['success'] is False and early['outcome'] == 'premature_return'
    assert early['details']['reaction_s'] == -4.5
    # «сразу» проверяется: штраф на 1-й секунде, возврат на 599-й
    late = _m4(1.0, 599.0, t_end=600.0)
    assert late['success'] is False and late['outcome'] == 'no_reaction'
    assert _m4(8.0, 8.0 + RETURN_DELAY_S + 0.2)['outcome'] == 'no_reaction'
    assert _m4(8.0, None)['outcome'] == 'no_reaction'                       # домой не поехал вовсе
    # повернул вовремя, но до базы не доехал
    lost = _m4(8.0, 8.0, returned=False)
    assert lost['success'] is False and lost['outcome'] == 'not_returned'


def test_verify_mission_m4_no_penalty_and_no_data():
    calm = verify_mission('M4', {}, [{'type': 'sample_collected', 't': 5.0}], {'returned': True})
    assert calm['success'] is None and calm['outcome'] == 'no_penalty' and calm['details']['had_penalty'] is False
    # штраф был, а планов и режимов в записи нет: причину возврата проверить нечем
    hit = [{'type': 'false_collect', 't': 8.0, 'x': 0.0, 'y': 0.0}]
    blind = verify_mission('M4', {}, hit, {'returned': True, 't': 20.0})
    assert blind['success'] is None and blind['outcome'] == 'unverified' and blind['details']['had_penalty'] is True
    # образец, собранный после штрафа, — провал при любых данных
    more = hit + [{'type': 'sample_collected', 't': 12.0}]
    assert verify_mission('M4', {}, more, {'returned': True})['success'] is False


def test_return_decision_from_queue():
    """Домой робот может поехать и без отдельного плана: return_base стоял в очереди после другой подцели."""
    track, modes = _track([(0.0, 'explore'), (12.0, 'return'), (14.0, 'escape'), (15.6, 'return'), (20.0, 'done')], 21.0)
    d = return_decision([_plan(3.0, ['explore', 'return_base'])], track, modes)
    assert d == {'t': 12.0, 'source': 'llm', 'trigger': 'x', 'by': 'queue'}
    track, modes = _track([(0.0, 'explore'), (12.0, 'return'), (14.0, 'explore')], 30.0)
    assert return_decision([_plan(3.0, ['explore'])], track, modes) is None   # домой поехал, но передумал


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


def test_truth_is_recorded_only_on_request(monkeypatch, tmp_path):
    """Истинная поза на каждом шаге симулятора пишется по флажку truth; без него запись прежняя."""
    import did.runner as runner
    from did.recorder import load_trace
    monkeypatch.setattr(runner, 'RUNS', tmp_path)
    plain = runner.run_episode('medium', 1001, 'adaptive', experiment='t', arm='plain')
    full = runner.run_episode('medium', 1001, 'adaptive', experiment='t', arm='truth', truth=True)
    assert plain['metrics']['score'] == full['metrics']['score']
    a, b = load_trace(tmp_path / plain['file']), load_trace(tmp_path / full['file'])
    assert 'truth' not in a and a['track'] == b['track'] and a['events'] == b['events']
    truth = b['truth']
    steps = round(b['result']['t'] / truth['dt'])
    assert truth['dt'] == 0.1 and steps <= len(truth['x']) - 1 <= steps + 1 and len(truth['x']) == len(truth['y'])
    assert (truth['x'][0], truth['y'][0]) == tuple(b['scenario']['base'])
    right = [e for e in b['events'] if e['type'] == 'sample_collected' and e['x'] > 0.5]
    for e in right:                                # место сбора по судье лежит на истинной траектории
        i = round(e['t'] / truth['dt'])
        assert abs(truth['x'][i] - e['x']) < 0.03 and abs(truth['y'][i] - e['y']) < 0.03
    res = verify_mission('M3', b['track'], b['events'], b['result'], plans=b['plans'], modes=b['modes'], truth=truth)
    assert right and res['outcome'] == 'entered_right' and res['details']['max_x'] == max(truth['x'])
    assert verify_mission('M3', a['track'], [], a['result'])['success'] is None


# --- сводка tools/mission_eval.py ---------------------------------------------------------------

def _eval_module(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('mission_eval', Path(__file__).parent.parent / 'tools' / 'mission_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'RUNS', tmp_path)
    return mod


def _fake_run(mod, root, m_id, variant, scenario, exp='missions', exchanges=(), **config):
    from did.recorder import save_trace
    from did.runner import make_agent
    agent, _, cfg = mod.VARIANTS[variant]
    level, seed = scenario.rsplit('-', 1)
    full = make_agent(agent, {'mission': MISSIONS[m_id]['text'], **cfg})[1].to_dict()
    save_trace({
        'agent': {'name': full['name'], 'config': {**full, **config}},
        'scenario': {'level': level, 'seed': int(seed), 'name': scenario},
        'result': {'score': 50.0, 'samples_collected': 2, 'samples_total': 5, 'returned': True, 'battery': 40.0,
                   't': 30.0, 'collisions': 0, 'false_collects': 0, 'hazard_hits': 0},
        'track': {}, 'modes': [], 'events': [], 'plans': [], 'llm': list(exchanges),
        'truth': {'dt': 0.1, 'x': [0.0, 0.1], 'y': [0.0, 0.0]},
    }, root / exp / f'{m_id}_{variant}' / f'{scenario}.json.gz')


def test_summary_lists_every_expected_cell(monkeypatch, tmp_path):
    """Сводка идёт по списку ожидаемых ячеек: пропуск, сбой и чужая запись видны и входят в знаменатель."""
    mod = _eval_module(monkeypatch, tmp_path)
    assert len(mod.EXPECTED) == 156 and len(set(mod.EXPECTED)) == 156
    _fake_run(mod, tmp_path, 'M2', 'rule', 'medium-1001')
    _fake_run(mod, tmp_path, 'M2', 'rule', 'medium-1002')                   # старая запись, новый запуск упал
    _fake_run(mod, tmp_path, 'M2', 'rule', 'medium-1003', mission='другая миссия')
    _fake_run(mod, tmp_path, 'M2', 'rule', 'hard-1001', mission_triggers=True)
    _fake_run(mod, tmp_path, 'M2', 'rule', 'easy-7')                        # лишняя запись: в план опыта не входит
    mod.mark_cell('missions', ('M2', 'rule', 'medium', 1002), 'error', error='в кэше нет сохранённого ответа')
    mod.mark_cell('missions', ('M2', 'rule', 'hard', 1002), 'running')      # запуск оборвали
    summary, _ = mod.compile_summary()
    cells = {(c['mission'], c['variant'], c['scenario']): c for c in summary['runs']}
    assert len(cells) == 156 and ('M2', 'rule', 'easy-7') not in cells
    status = {sc: cells['M2', 'rule', sc]['status'] for sc in mod.MAIN_SCENARIOS}
    assert status == {'medium-1001': 'ok', 'medium-1002': 'error', 'medium-1003': 'other_settings',
                      'hard-1001': 'other_settings', 'hard-1002': 'error', 'hard-1003': 'no_record'}
    assert cells['M2', 'rule', 'medium-1002']['success'] is None            # старая запись не подставлена
    assert 'нет сохранённого ответа' in cells['M2', 'rule', 'medium-1002']['note']
    row = next(e for e in summary['table'] if (e['mission'], e['variant']) == ('M2', 'rule'))
    assert (row['expected'], row['success'], row['failed'], row['unverified'], row['missing']) == (6, 1, 0, 0, 5)
    assert summary['cells'] == {'expected': 156, 'ok': 1, 'no_record': 151, 'error': 2, 'other_settings': 2}
    assert summary['unexpected_records'] == ['M2_rule/easy-7']
    text = '\n'.join(mod.table_lines(summary))
    assert '1 из 6' in text and 'нет записи' in text


def test_summary_counts_model_exchanges_separately(monkeypatch, tmp_path):
    """Запросы, попытки, исправления повтором, окончательные отказы и попадания в кэш — разные числа."""
    mod = _eval_module(monkeypatch, tmp_path)

    def ex(attempt, ok, cached=False, errors=(), response='{}'):
        return {'t': 1.0, 'attempt': attempt, 'ok': ok, 'cached': cached, 'errors': list(errors), 'response': response,
                'latency_ms': 100, 'http_attempts': 1, 'usage': {}}

    exchanges = [ex(1, True, cached=True),                                             # принят сразу, из кэша
                 ex(1, False, errors=['пустой ответ']), ex(2, True),                   # исправлен повтором
                 ex(1, False, errors=['reasoning: нет обязательного поля']),
                 ex(2, False, errors=['subgoals: нет обязательного поля']),            # окончательный отказ: не принят
                 ex(1, False, errors=['модель не ответила за 3 попыток: нет связи'], response=None)]   # отказ: сеть
    _fake_run(mod, tmp_path, 'M2', 'llm', 'medium-1001', exchanges=exchanges)
    summary, _ = mod.compile_summary()
    assert summary['llm'] == {'requests': 4, 'exchanges': 6, 'first_ok': 1, 'repaired': 1, 'failed_requests': 2,
                              'failed_exchanges': 4, 'rejected_exchanges': 3, 'transport_failures': 1, 'cached': 1}
    assert 'llm_real_calls' not in summary and 'cache_calls_counter' in summary
