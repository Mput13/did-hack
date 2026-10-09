"""L4a: память планировщика — без флажка запросы прежние побайтно, с флажком история и события собираются верно,
повторные отправки и миссии N1–N3 считаются по записи верно."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from did.agent import PRESETS, make_config
from did.journal import Journal
from did.llm import PROMPT_PATH, load_system_prompt, make_client
from did.plan_memory import (HISTORY, MISSIONS, PlanMemory, approaches, empty_places, revisits, verify,
                             without_memory)
from did.recorder import load_trace
from did.runner import RUNS, run_episode

OLD_PROMPT_SHA = 'd300bb1a2f0faafa480a6416b5fd40a5803aa3dc5bffda431db762cb13a61f2b'
OLD_KEYS = ['mission', 'trigger', 'time_s', 'time_limit_s', 'battery', 'battery_start', 'pose', 'base', 'samples',
            'return_cost', 'candidates', 'explore_points', 'soil_zones', 'hazards', 'sensor', 'recent_events', 'alarms',
            'open_hypotheses']
EVERY = {'llm_min_interval_s': 0.0}


class Spy:
    """Имитатор модели, который запоминает запросы дословно."""
    model, use_schema = 'spy', False

    def __init__(self):
        self.inner, self.seen = make_client('mock', seed=0), []

    def chat(self, messages, **kw):
        self.seen.append(json.dumps(messages, ensure_ascii=False))
        return self.inner.chat(messages, **kw)


def _state_of(request):
    text = json.loads(request)[1]['content']
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


@pytest.fixture(scope='module')
def runs():
    out = {}
    for name, agent, extra in (('off', 'adaptive_llm', {}), ('track', 'adaptive_llm', {'planner_memory': 'track'}),
                               ('on', 'adaptive_llm_mem', {})):
        spy = Spy()
        res = run_episode('hard', 3, agent, experiment='_test_l4a', arm=name, config={**EVERY, **extra},
                          llm={'client': spy}, truth=True)
        out[name] = (spy.seen, load_trace(RUNS / res['file']))
    return out


def test_flag_is_off_everywhere_but_the_new_preset():
    for name, cfg in PRESETS.items():
        assert cfg.planner_memory == ('on' if name == 'adaptive_llm_mem' else ''), name
    assert make_config('adaptive_llm_mem').planner == 'llm'
    with pytest.raises(ValueError):
        PlanMemory(SimpleNamespace(cfg=SimpleNamespace(planner_memory='yes')))


def test_old_prompt_file_is_untouched():
    assert hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() == OLD_PROMPT_SHA


def test_requests_without_memory_are_byte_identical(runs):
    off, track = runs['off'], runs['track']
    assert off[0] and off[0] == track[0]                       # память ведётся, а запросы те же до байта
    assert off[1]['track'] == track[1]['track'] and off[1]['result']['score'] == track[1]['result']['score']
    old = load_system_prompt()
    for request in off[0]:
        messages = json.loads(request)
        if len(messages) == 2:                                 # первый запрос решения, не исправление
            assert messages[0]['content'] == old
            state = _state_of(request)
            assert list(state) == OLD_KEYS and state['recent_events'] == []
    assert all('state' not in p and 'memory' not in p for p in off[1]['plans'])
    asked = [p for p in track[1]['plans'] if 'rule_match' in p]
    assert asked and all(list(p['state']) == OLD_KEYS and set(p['memory']) == {
        'recent_events', 'history', 'checked_empty', 'penalty_places'} for p in asked)


def test_memory_request_is_old_request_plus_memory(runs):
    off, on = runs['off'], runs['on']
    # Имитатор выбирает по тем же полям, что правило: прогон идёт той же дорогой, и снимки можно сравнить по одному.
    assert on[1]['track'] == off[1]['track'] and len(on[0]) == len(off[0])
    new = load_system_prompt('planner_system_mem')
    for a, b in zip(on[0], off[0]):
        assert json.loads(a)[0]['content'] == new
        state = _state_of(a)
        assert list(state) == OLD_KEYS + ['history', 'checked_empty', 'penalty_places']
        assert json.dumps(without_memory(state), ensure_ascii=False) == json.dumps(_state_of(b), ensure_ascii=False)
    last = _state_of(on[0][-1])
    assert 1 <= len(last['history']) <= HISTORY
    for e in last['history']:
        assert e['by'] in ('llm', 'heuristic', 'fallback') and e['plan'] and 'battery_used' in e and 'seconds' in e
        assert all(r.split(': ')[1].split(' by ')[0] in ('reached', 'sample_collected', 'empty', 'false_collect',
                                                         'lost', 'interrupted') for r in e['results'])
        assert all(g.startswith(('investigate (', 'explore (', 'goto (', 'return_base')) for g in e['plan'])
    kinds = {e['type'] for request in on[0] for e in _state_of(request)['recent_events']}
    assert 'sample_collected' in kinds and kinds <= {
        'hazard_hit', 'collision', 'false_collect', 'sample_collected', 'sensor_degraded', 'sensor_recovered',
        'model_mismatch', 'soil_confirmed', 'soil_outdated'}


# --- память на искусственном прогоне -----------------------------------------------------------

class FakeAgent:
    def __init__(self):
        self.cfg = SimpleNamespace(planner_memory='on')
        self.queue = []
        self.journal = Journal()


def _obs(t, battery, x=0.0, y=0.0):
    return SimpleNamespace(t=t, battery=battery, x=x, y=y)


def _state(t, trigger, cands=(), points=()):
    return {'time_s': t, 'trigger': trigger,
            'candidates': [{'id': i, 'x': x, 'y': y, 'cost_to': c, 'feasible': True} for i, x, y, c in cands],
            'explore_points': [{'id': i, 'x': x, 'y': y, 'cost_to': c, 'feasible': True} for i, x, y, c in points]}


def _decide(agent, mem, t, trigger, subgoals, state, source='llm'):
    mem.decided(trigger, source, subgoals, state)
    agent.queue = [dict(s) for s in subgoals]


def test_history_events_and_places_on_a_synthetic_run():
    a = FakeAgent()
    mem = PlanMemory(a)
    inv = lambda i, x, y: {'type': 'investigate', 'target': i, 'x': x, 'y': y}          # noqa: E731
    exp = lambda i, x, y: {'type': 'explore', 'target': i, 'x': x, 'y': y}              # noqa: E731

    mem.observe(_obs(0.0, 60.0))
    st = _state(0.0, 'start', points=[('E1', 1.0, 0.0, 2.5), ('E2', 2.0, 0.0, 5.0)])
    assert mem.snapshot(st) == {'recent_events': [], 'history': [], 'checked_empty': [], 'penalty_places': []}
    _decide(a, mem, 0.0, 'start', [exp('E1', 1.0, 0.0), exp('E2', 2.0, 0.0)], st)

    mem.observe(_obs(10.0, 57.0, 1.0, 0.0))                    # доехал до первой точки, очередь идёт дальше
    mem.ended(a.queue[0], 'subgoal_done')
    a.queue.pop(0)
    mem.observe(_obs(14.0, 56.0, 1.4, 0.0))                    # по дороге ко второй — штраф
    mem.note('hazard_hit', 1.4, 0.0)
    st = _state(14.0, 'hazard', cands=[('C1', 1.5, 1.0, 3.0)])
    snap = mem.snapshot(st)                                    # вопрос посреди подцели: она «прервана» поводом
    assert snap['history'][0]['plan'] == ['explore (1; 0)', 'explore (2; 0)']
    assert snap['history'][0]['results'] == ['explore (1; 0): reached', 'explore (2; 0): interrupted by hazard']
    assert snap['history'][0]['battery_used'] == 4.0 and snap['history'][0]['seconds'] == 14.0
    assert snap['history'][0]['penalties'] == ['hazard_hit'] and snap['history'][0]['cost_est'] == 2.5
    assert snap['penalty_places'] == [{'t': 14.0, 'type': 'hazard_hit', 'x': 1.4, 'y': 0.0, 'near': []}]
    _decide(a, mem, 14.0, 'hazard', [inv('C1', 1.5, 1.0)], st)
    assert mem.decisions[0]['results'][-1] == {'goal': {'type': 'explore', 'x': 2.0, 'y': 0.0},
                                               'result': 'interrupted', 'by': 'hazard'}

    mem.observe(_obs(16.0, 55.5, 1.4, 0.2))                    # кандидат пропал по дороге: место не проверено
    mem.ended(a.queue[0], 'candidate_lost')
    st = _state(16.0, 'candidate_lost', cands=[('C1', -1.0, 1.0, 4.0)])
    assert mem.snapshot(st)['checked_empty'] == []
    _decide(a, mem, 16.0, 'candidate_lost', [inv('C1', -1.0, 1.0)], st)
    assert mem.decisions[1]['results'] == [{'goal': {'type': 'investigate', 'x': 1.5, 'y': 1.0}, 'result': 'lost'}]

    mem.observe(_obs(40.0, 50.0, -0.9, 1.0))                   # подъехал вплотную, образца нет
    mem.ended(a.queue[0], 'candidate_lost')
    st = _state(40.0, 'candidate_lost', cands=[('C1', -0.6, 1.2, 1.0), ('C2', 2.0, 2.0, 6.0)])
    snap = mem.snapshot(st)
    assert snap['checked_empty'] == [{'t': 40.0, 'x': -0.9, 'y': 1.0, 'why': 'candidate_refuted', 'near': ['C1']}]
    assert snap['history'][-1]['results'] == ['investigate (-1; 1): empty'] and snap['history'][-1]['battery_used'] == 5.5
    _decide(a, mem, 40.0, 'candidate_lost', [inv('C2', 2.0, 2.0)], st)

    mem.observe(_obs(70.0, 43.0, 2.0, 2.0))                    # ложный сбор: штраф, место пустое
    mem.note('false_collect', 2.0, 2.0)
    mem.ended(a.queue[0], 'false_collect')
    mem.observe(_obs(71.0, 43.0, 2.0, 2.0))
    mem.note('false_collect', 2.1, 2.0)                        # рядом с прежним: одна запись памяти
    a.journal.open(72.0, 'soil:1', 'грунт дороже', 'проверить', x=0.5, y=0.5, mult=2.34)
    a.journal.close(73.0, 'soil:1', 'confirmed', 'расход ×2.3')
    mem.observe(_obs(110.0, 40.0, 2.0, 2.0))
    st = _state(110.0, 'false_collect', cands=[('C1', 2.2, 2.2, 0.5)], points=[('E1', 1.2, 0.6, 3.0)])
    snap = mem.snapshot(st)
    assert [p['why'] for p in snap['checked_empty']] == ['candidate_refuted', 'false_collect']
    assert snap['checked_empty'][1]['near'] == ['C1'] and snap['checked_empty'][1]['x'] == 2.1
    assert [p['type'] for p in snap['penalty_places']] == ['hazard_hit', 'false_collect', 'false_collect']
    assert snap['penalty_places'][0]['near'] == ['E1'] and snap['penalty_places'][1]['near'] == ['C1']
    # события старше 90 с (штраф на 14-й секунде) в recent_events не идут; вывод о грунте — из журнала
    assert [e['type'] for e in snap['recent_events']] == ['false_collect', 'false_collect', 'soil_confirmed']
    assert snap['recent_events'][-1] == {'t': 73.0, 'type': 'soil_confirmed', 'x': 0.5, 'y': 0.5, 'mult': 2.3}
    assert snap['history'][-1]['penalties'] == ['false_collect', 'false_collect']
    assert snap['history'][-1]['results'] == ['investigate (2; 2): false_collect']

    for i in range(HISTORY + 3):                               # в снимке не больше HISTORY последних решений
        _decide(a, mem, 110.0 + i, 'queue_empty', [exp('E1', 1.2, 0.6)], st, source='heuristic')
    snap = mem.snapshot(_state(125.0, 'queue_empty'))
    assert len(snap['history']) == HISTORY and snap['history'][-1]['by'] == 'heuristic'
    mem.went_home('заряд')                                     # агент сам повернул домой: подцель прервана
    assert mem.decisions[-1]['results'][-1]['by'] == 'return_base' and mem.decisions[-1]['closed']


# --- метрики по записи -------------------------------------------------------------------------

def _plan(t, trigger, kind=None, x=0.0, y=0.0, source='llm'):
    sub = [{'type': 'return_base'}] if kind is None else [{'type': kind, 'x': x, 'y': y}]
    return {'t': t, 'source': source, 'trigger': trigger, 'subgoals': sub}


def _trace(plans, events=(), paths=(), track=None, returned=True):
    ts = [i * 0.2 for i in range(1000)]
    track = track or {}
    xs = [next((x for t0, x, _ in reversed(track.get('pos', [])) if t0 <= t + 1e-9), 0.0) for t in ts]
    ys = [next((y for t0, _, y in reversed(track.get('pos', [])) if t0 <= t + 1e-9), 0.0) for t in ts]
    modes = ['approach', 'return']
    home = plans[-1]['t'] if plans[-1]['subgoals'][0]['type'] == 'return_base' else 1e9
    return {'plans': plans, 'events': list(events), 'paths': [{'t': t, 'goal': [x, y]} for t, x, y in paths],
            'track': {'t': ts, 'x': xs, 'y': ys, 'mode': [int(t >= home) for t in ts]}, 'modes': modes,
            'result': {'returned': returned, 'samples_collected': 1}}


def test_revisits_and_approaches_from_a_trace():
    plans = [
        _plan(0.0, 'start', 'investigate', 1.0, 1.0),
        _plan(20.0, 'candidate_lost', 'investigate', 3.0, 3.0),      # был на месте (1.1; 1.0): пусто
        _plan(21.0, 'candidate_lost', 'investigate', 1.2, 1.2),      # кандидат (3; 3) пропал сразу: не подъезд;
                                                                     # новая цель — у проверенного места: повтор
        _plan(40.0, 'sample_collected', 'explore', 1.3, 0.9),        # повтор кончился сбором; ещё повтор — разведка
        _plan(60.0, 'subgoal_done', 'investigate', 2.0, 0.0),
        _plan(80.0, 'false_collect', 'investigate', 2.3, 0.1, source='fallback'),   # к месту ложного сбора
        _plan(90.0, 'candidate_lost', 'investigate', -2.0, -2.0),
        _plan(120.0, 'candidate_lost'),                              # на месте (-2; -2): пусто, домой
    ]
    events = [{'t': 79.9, 'type': 'false_collect', 'x': 2.0, 'y': 0.0},
              {'t': 30.0, 'type': 'hazard_hit', 'x': -1.0, 'y': 0.0}]
    paths = [(0.0, 1.0, 1.0), (5.0, 1.1, 1.0), (21.0, 1.2, 1.2), (60.0, 2.0, 0.0), (80.0, 2.3, 0.1), (89.0, 2.3, 0.1),
             (90.0, -2.0, -2.0)]
    pos = [(0.0, 0.0, 0.0), (19.0, 1.1, 1.0), (39.0, 1.2, 1.2), (79.0, 2.0, 0.0), (89.0, 0.0, 0.0), (119.0, -2.0, -1.9)]
    tr = _trace(plans, events, paths, {'pos': pos})
    assert [(t, k) for t, k, _, _ in approaches(tr)] == [
        (20.0, 'candidate_refuted'), (21.0, 'lost'), (40.0, 'sample_collected'), (80.0, 'false_collect'),
        (90.0, 'lost'), (120.0, 'candidate_refuted')]
    assert [(t, why) for t, _, _, why in empty_places(tr)] == [
        (20.0, 'candidate_refuted'), (79.9, 'false_collect'), (120.0, 'candidate_refuted')]
    r = revisits(tr)
    assert (r['decisions'], r['empty_places'], r['to_empty'], r['to_empty_collected']) == (8, 3, 3, 1)
    assert [(x['t'], x['kind']) for x in r['list']] == [(21.0, 'empty'), (40.0, 'empty'), (80.0, 'empty'),
                                                        (80.0, 'penalty')]
    assert r['penalties'] == 2 and r['to_penalty'] == 1
    assert revisits(tr, sources=('llm',))['to_empty'] == 2     # решение запасного правила можно не считать
    assert verify('N1', tr)['outcome'] == 'repeated' and verify('N1', tr)['details']['repeats'] == 3
    # N3: цель (2.3; 0.1) ближе метра от штрафа в (2; 0); цель (-2; -2) дальше метра от штрафа в (-1; 0)
    v = verify('N3', tr)
    assert v['success'] is False and v['details']['goals_near'] == 1 and v['details']['penalties'] == 2


def test_mission_n2_two_empty_approaches_in_a_row():
    def trace(after):
        plans = [_plan(0.0, 'start', 'investigate', 1.0, 0.0),
                 _plan(10.0, 'false_collect', 'investigate', 2.0, 0.0),
                 _plan(11.0, 'candidate_lost', 'investigate', 2.1, 0.0),     # пропал по дороге: не подъезд
                 _plan(30.0, 'candidate_lost', *after)]                      # второй пустой подъезд подряд
        events = [{'t': 10.0, 'type': 'false_collect', 'x': 1.0, 'y': 0.0}]
        paths = [(0.0, 1.0, 0.0), (11.0, 2.1, 0.0), (20.0, 2.0, 0.1)]
        return _trace(plans, events, paths, {'pos': [(0.0, 0.0, 0.0), (9.0, 1.0, 0.0), (29.0, 2.0, 0.0)]})

    ok = verify('N2', trace(()))                                             # сразу домой
    assert ok['success'] is True and ok['details']['two_empty_t'] == 30.0 and ok['details']['reaction_s'] == 0.0
    bad = verify('N2', trace(('explore', 0.0, 2.0)))                         # поехал дальше
    assert bad['success'] is False and bad['outcome'] == 'no_reaction' and bad['details']['goals_after'] == 1
    none = _trace([_plan(0.0, 'start', 'investigate', 1.0, 0.0), _plan(10.0, 'sample_collected')],
                  paths=[(0.0, 1.0, 0.0)])
    assert verify('N2', none)['success'] is None and verify('N1', none)['success'] is None
    assert verify('N3', none)['outcome'] == 'no_occasion'
    assert set(MISSIONS) == {'N1', 'N2', 'N3'}


# --- инструмент: вторая мера повторов, точный критерий, контроль шума ----------------------------

def _tool():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
    import l4a_memory
    return l4a_memory


def test_fisher_exact_for_small_shares():
    f = _tool().fisher_p
    assert f(5, 5, 0, 4) == round(1 / 126, 4)                  # 5 из 5 против 0 из 4: одна таблица из 126
    assert f(1, 2, 0, 2) == 1.0 and f(3, 5, 3, 5) == 1.0
    assert f(5, 5, 3, 5) == round(2 * 10 / 45, 4)              # 8 успехов на 10: крайние таблицы с двух сторон
    assert f(0, 0, 1, 2) is None


def test_revisits_by_memory_do_not_depend_on_who_saw_the_memory():
    """Вторая мера считает по спискам памяти, а они одинаково лежат и в снимке (on), и рядом с ним (track)."""
    tool = _tool()
    memory = {'checked_empty': [{'x': 1.0, 'y': 1.0, 'near': ['C1', 'E2']}], 'penalty_places': [{'near': ['C3']}]}
    plans = [{'subgoals': [{'type': 'investigate', 'target': 'C1', 'x': 1.1, 'y': 1.0}]},
             {'subgoals': [{'type': 'investigate', 'target': 'C3', 'x': 3.0, 'y': 0.0}]},      # у штрафа — не повтор
             {'subgoals': [{'type': 'explore', 'target': 'E2', 'x': 1.2, 'y': 1.2}]},
             {'subgoals': [{'type': 'return_base'}]}, {'subgoals': []}]
    track = {'plans': [{**p, 'state': {'candidates': []}, 'memory': memory} for p in plans]}
    on = {'plans': [{**p, 'state': {'candidates': [], **memory}} for p in plans]}
    assert tool.revisits_by_memory(track) == tool.revisits_by_memory(on) == 2
    assert tool.revisits_by_memory({'plans': plans}) is None           # память не велась — меры нет, а не ноль


def test_main_metric_ignores_memory_fields(runs):
    """Основная мера повторных отправок берёт только планы, пути, трек и события: поля памяти её не меняют."""
    for name in ('track', 'on'):
        tr = runs[name][1]
        bare = {**tr, 'plans': [{k: v for k, v in p.items() if k not in ('state', 'memory')} for p in tr['plans']]}
        assert revisits(bare) == revisits(tr)
        for m in MISSIONS:
            assert verify(m, bare) == verify(m, tr)


def test_noise_control_asks_the_same_request_under_another_cache_key():
    tool = _tool()
    seen = []

    class Inner:
        model, use_schema = 'm', False

        def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
            seen.append((messages, temperature, max_tokens))
            return 'ok'

    assert tool.Again(Inner()).chat(['q'], max_tokens=800) == 'ok'
    assert seen == [(['q'], 0.2, 801)]                                 # тот же вопрос, лимит ответа на единицу больше
    assert tool.same_goal({'type': 'return_base'}, {'type': 'return_base'}) and tool.same_goal(None, None)
    assert not tool.same_goal(None, {'type': 'return_base'})
    assert tool.same_goal({'type': 'explore', 'x': 1.0, 'y': 1.0}, {'type': 'explore', 'x': 1.03, 'y': 1.0})
    assert not tool.same_goal({'type': 'explore', 'x': 1.0, 'y': 1.0}, {'type': 'investigate', 'x': 1.0, 'y': 1.0})
