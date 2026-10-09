"""Память планировщика внутри прогона (исследование L4a): что он решал и чем это кончилось.

До L4a каждый вызов модели был отдельным запросом: в снимке состояния была только память, которую ведёт код
(карта образцов, грунты, опасные зоны), а собственных прошлых решений и их исходов модель не видела. Здесь
агент ведёт четыре списка и при флажке planner_memory='on' кладёт их в снимок:

  history         последние HISTORY решений планировщика: время, повод, кто решал, подцели, исход каждой
                  исполненной, оценка цены первой подцели и сколько заряда и секунд ушло на деле;
  recent_events   события за последние EVENTS_S секунд: штрафы, сборы, ложные сборы, начало и конец сбоя
                  датчика, расхождение расхода с прогнозом, выводы о грунте;
  checked_empty   места, где робот стоял, а образца не оказалось (кандидат не подтвердился на месте или
                  ложный сбор); кандидат, пропавший ещё по дороге, сюда не идёт — у него исход lost.
                  ВНИМАНИЕ: название и подсказка говорят «место пустое», а это неверно — в прогонах L4a у 26
                  таких мест из 31 ближе 0,5 м лежал несобранный образец (research/findings/L4a.md, «Разбор
                  после ревью»). Код и подсказка оставлены как в измеренной серии; прежде чем включать память
                  где-либо, исход нужно назвать «не подтвердился» и не запрещать цель рядом безусловно;
  penalty_places  места штрафов за весь прогон.

У двух последних списков есть поле near — идентификаторы нынешних целей из candidates и explore_points рядом
с этим местом: идентификаторы целей меняются от снимка к снимку, а место остаётся.

planner_memory='track' — списки ведутся и пишутся в запись прогона (поле memory у решения), но планировщик
получает прежний снимок: так «теневое» сравнение может спросить подсказку с памятью в том же состоянии.

Здесь же метрики по записи прогона, одинаковые для агентов с памятью и без неё: revisits (повторные отправки
в проверенное пустое место и к месту штрафа) и критерии трёх миссий словами MISSIONS / verify.
Врезки в did/agent.py — вызовы observe, decided, ended, went_home, note; вся логика здесь.
"""
import math
from collections import deque

MODES = ('', 'on', 'track')
HISTORY = 6              # сколько последних решений видит модель
EVENTS_S = 90.0          # за сколько секунд показываются события
MAX_EVENTS = 8
MAX_PLACES = 6           # сколько последних пустых мест и мест штрафов показывается
SAME_PLACE_M = 0.5       # цель ближе этого к проверенному месту — «то же место» (как в did/waiting.py)
ON_SITE_M = 0.3          # кандидат пропал, когда робот был к нему ближе этого (радиус сбора), — место проверено
MERGE_M = 0.3            # пустые места ближе этого друг к другу — одна запись памяти
PENALTY_NEAR_M = 1.0     # цель ближе этого к месту штрафа — «рядом со штрафом»
PENALTIES = ('hazard_hit', 'false_collect', 'collision')
EMPTY = ('candidate_lost', 'false_collect')            # поводы, которыми кончается подъезд без образца
PROMPT = 'planner_system_mem'                          # подсказка, которая объясняет новые поля
FIELDS = ('history', 'checked_empty', 'penalty_places')  # поля, которых нет в прежнем снимке

_RESULT = {'subgoal_done': 'reached', 'sample_collected': 'sample_collected', 'candidate_lost': 'empty',
           'false_collect': 'false_collect', 'candidate_found': 'interrupted'}       # и lost — см. ended


def _goal(sg):
    if sg['type'] == 'return_base':
        return {'type': 'return_base'}
    return {'type': sg['type'], 'x': round(sg['x'], 2), 'y': round(sg['y'], 2)}


def _text(goal):
    """Подцель одной строкой — так в снимке она вдвое короче объекта: «investigate (1.55; 0.95)»."""
    return goal['type'] if goal['type'] == 'return_base' else f"{goal['type']} ({goal['x']:g}; {goal['y']:g})"


def _outcome(r):
    return f"{_text(r['goal'])}: {r['result']}" + (f" by {r['by']}" if 'by' in r else '')


class PlanMemory:

    def __init__(self, agent):
        if agent.cfg.planner_memory not in MODES[1:]:
            raise ValueError(f"planner_memory: «{agent.cfg.planner_memory}»; есть: '', on, track")
        self.a = agent
        self.decisions = []                 # все решения прогона; в снимок идут последние HISTORY
        self.events = deque(maxlen=200)     # [{'t', 'type', ...}]
        self.empty = []                     # [{'t', 'x', 'y', 'why'}]
        self.penalties = []                 # [{'t', 'type', 'x', 'y'}]
        self._open = False                  # текущая подцель последнего решения ещё исполняется
        self._now = (0.0, 0.0, 0.0, 0.0)    # последнее наблюдение: t, заряд, x, y
        self._seen = 0                      # сколько записей журнала уже просмотрено

    # --- врезки агента -------------------------------------------------------------------------

    def observe(self, obs):
        self._now = (obs.t, obs.battery, obs.x, obs.y)

    def note(self, kind, x=None, y=None, **data):
        """Событие для recent_events; штрафы и пустые места раскладываются ещё и по своим спискам."""
        t = self._now[0]
        ev = {'t': round(t, 1), 'type': kind}
        if x is not None:
            ev.update(x=round(x, 2), y=round(y, 2))
        ev.update(data)
        self.events.append(ev)
        if kind in PENALTIES:
            self.penalties.append({'t': round(t, 1), 'type': kind, 'x': round(x, 2), 'y': round(y, 2)})
            if self.decisions and not self.decisions[-1]['closed']:
                self.decisions[-1]['penalties'].append(kind)
        if kind == 'false_collect':
            self._add_empty(x, y, 'false_collect')

    def _add_empty(self, x, y, why):
        self.empty = [p for p in self.empty if math.hypot(p['x'] - x, p['y'] - y) > MERGE_M]
        self.empty.append({'t': round(self._now[0], 1), 'x': round(x, 2), 'y': round(y, 2), 'why': why})

    def decided(self, trigger, source, subgoals, state):
        """Принято новое решение. Вызывается до замены очереди: недоделанная подцель прежнего — «прервана»."""
        t, battery = self._now[:2]
        if self._open and self.a.queue:
            self._result(self.a.queue[0], 'interrupted', by=trigger)
        self._close()
        cost = None
        if subgoals and subgoals[0].get('target'):
            cost = next((c['cost_to'] for c in state['candidates'] + state['explore_points']
                         if c['id'] == subgoals[0]['target']), None)
        self.decisions.append({'t': round(t, 1), 'trigger': trigger, 'by': source, 'plan': [_goal(s) for s in subgoals],
                               'results': [], 'penalties': [], 'cost_est': cost, 't0': t, 'b0': battery,
                               'closed': False})
        self._open = bool(subgoals)

    def ended(self, sg, trigger):
        """Подцель закончилась сама: доехал, собрал, место пустое, по дороге нашёлся новый кандидат."""
        if not self.decisions:
            return
        extra = {'by': 'candidate_found'} if trigger == 'candidate_found' else {}
        result = _RESULT.get(trigger, trigger)
        if trigger == 'candidate_lost':
            x, y = self._now[2:]
            if math.hypot(sg['x'] - x, sg['y'] - y) <= ON_SITE_M:
                self._add_empty(x, y, 'candidate_refuted')     # место — где робот стоял, как у ложного сбора
            else:
                result = 'lost'                                # кандидат пропал ещё по дороге: место не проверено
        self._result(sg, result, **extra)
        # После обычной точки очередь идёт дальше без нового решения; в остальных случаях решение исчерпано.
        self._open = trigger == 'subgoal_done' and len(self.a.queue) > 1

    def went_home(self, reason):
        """Агент сам повернул домой (заряд, время, все образцы): прежнее решение прервано."""
        if self._open and self.a.queue:
            self._result(self.a.queue[0], 'interrupted', by='return_base')
        self._close()

    def _result(self, sg, result, **extra):
        self.decisions[-1]['results'].append({'goal': _goal(sg), 'result': result, **extra})

    def _close(self):
        self._open = False
        if self.decisions and not self.decisions[-1]['closed']:
            d = self.decisions[-1]
            t, battery = self._now[:2]
            d.update(closed=True, battery_used=round(d['b0'] - battery, 1), seconds=round(t - d['t0'], 1))

    def _journal(self):
        """Выводы о грунте — из журнала: гипотезы о грунте, которые подтвердились или устарели."""
        j = self.a.journal
        for e in j.entries[self._seen:]:
            data = e.get('data') or {}
            if e['kind'] != 'verdict' or data.get('status') not in ('confirmed', 'outdated'):
                continue
            h = next((h for h in j.hypotheses if h['id'] == data.get('hypothesis')), None)
            if h and str(h.get('key', '')).startswith('soil:'):
                d = h.get('data') or {}
                self.events.append({'t': e['t'], 'type': f"soil_{data['status']}", 'x': d.get('x'), 'y': d.get('y'),
                                    'mult': round(d['mult'], 1) if 'mult' in d else None})
        self._seen = len(j.entries)

    # --- снимок --------------------------------------------------------------------------------

    def snapshot(self, state):
        """{'recent_events', 'history', 'checked_empty', 'penalty_places'} на момент state."""
        self._journal()
        t = state['time_s']
        targets = state['candidates'] + state['explore_points']

        def near(p, radius):
            return [c['id'] for c in targets if math.hypot(c['x'] - p['x'], c['y'] - p['y']) <= radius]

        history = []
        last = self.decisions[-1] if self.decisions else None
        if self._open and self.a.queue and last is not None:
            # Вопрос задан посреди подцели: в истории она «прервана» нынешним поводом, запись решения не меняется.
            last = {**last, 'results': last['results'] + [{'goal': _goal(self.a.queue[0]), 'result': 'interrupted',
                                                           'by': state['trigger']}]}
        for d in self.decisions[-HISTORY:-1] + ([last] if last else []):
            e = {'t': d['t'], 'trigger': d['trigger'], 'by': d['by'], 'plan': [_text(g) for g in d['plan'][:2]],
                 'results': [_outcome(r) for r in d['results']]}
            if d['penalties']:
                e['penalties'] = d['penalties']
            if d['cost_est'] is not None:
                e['cost_est'] = d['cost_est']
            e['battery_used'] = d['battery_used'] if d['closed'] else round(d['b0'] - self._now[1], 1)
            e['seconds'] = d['seconds'] if d['closed'] else round(self._now[0] - d['t0'], 1)
            history.append(e)
        events = sorted((e for e in self.events if t - e['t'] <= EVENTS_S), key=lambda e: e['t'])[-MAX_EVENTS:]
        return {
            'recent_events': [{k: v for k, v in e.items() if v is not None} for e in events],
            'history': history,
            'checked_empty': [{**p, 'near': near(p, SAME_PLACE_M)} for p in self.empty[-MAX_PLACES:]],
            'penalty_places': [{**p, 'near': near(p, PENALTY_NEAR_M)} for p in self.penalties[-MAX_PLACES:]],
        }


def without_memory(state):
    """Снимок, каким его получил бы планировщик без памяти: те же ключи в том же порядке, событий нет."""
    return {k: ([] if k == 'recent_events' else v) for k, v in state.items() if k not in FIELDS}


# --- метрики по записи прогона -----------------------------------------------------------------

def _first_target(plan):
    sg = (plan.get('subgoals') or [None])[0]
    return sg if sg and sg.get('type') != 'return_base' and 'x' in sg else None


def _pos(track, t):
    """Где робот был в момент t по своей оценке (запись раз в 0,2 с); None — трека нет."""
    ts = (track or {}).get('t') or []
    if not ts:
        return None
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
    return float(track['x'][i]), float(track['y'][i])


def approaches(trace):
    """Исходы подъездов к кандидатам по порядку: [(t, исход, x, y)], исход — sample_collected, false_collect,
    candidate_refuted (робот был на месте и образца не оказалось) или lost (кандидат пропал по дороге).

    Берётся по поводу следующего решения: им кончилась первая подцель предыдущего. «На месте» — робот ближе
    ON_SITE_M к цели, к которой ехал: это цель последнего пути, построенного за время подцели (поле paths;
    исполнитель уточняет место кандидата по ходу). Пути не было — робот к цели и не поехал. x, y — где робот
    стоял в этот момент.
    """
    plans, track, paths = trace.get('plans') or [], trace.get('track'), trace.get('paths') or []
    out = []
    for prev, cur in zip(plans, plans[1:]):
        sg = _first_target(prev)
        kind = cur.get('trigger')
        if kind not in ('sample_collected', *EMPTY) or not sg or sg['type'] != 'investigate':
            continue
        t = float(cur['t'])
        at = _pos(track, t) or (float(sg['x']), float(sg['y']))
        if kind == 'candidate_lost':
            # Путь с временем t построен уже для нового решения: старой подцели принадлежат пути раньше t.
            goal = next((p['goal'] for p in reversed(paths) if float(prev['t']) - 1e-9 <= float(p['t']) < t - 0.05), None)
            on_site = goal is not None and math.hypot(at[0] - goal[0], at[1] - goal[1]) <= ON_SITE_M
            kind = 'candidate_refuted' if on_site else 'lost'
        out.append((t, kind, at[0], at[1]))
    return out


def empty_places(trace):
    """Проверенные пустые места по записи: [(t, x, y, почему)] — где робот стоял, когда подъезд кончился ничем.

    Ложный сбор — событие судьи (его координаты); «кандидат не подтвердился на месте» — см. approaches.
    """
    out = [(float(e['t']), float(e['x']), float(e['y']), 'false_collect')
           for e in trace.get('events') or [] if e.get('type') == 'false_collect']
    out += [(t, x, y, kind) for t, kind, x, y in approaches(trace) if kind == 'candidate_refuted']
    return sorted(out)


def nearest_sample(trace, t, x, y):
    """По истине сценария: (расстояние от места до ближайшего образца, не собранного к моменту t; когда его
    собрали позже или None; номер образца). (None, None, None) — несобранных образцов не осталось.

    Нужна, чтобы проверить саму память: «кандидат не подтвердился на месте» не значит, что образца рядом нет
    (карта вероятностей ставит цель с ошибкой в десятки сантиметров, при сбое датчика кандидат пропадает)."""
    got = {e['sample']: float(e['t']) for e in trace.get('events') or [] if e.get('type') == 'sample_collected'}
    left = [(math.hypot(sx - x, sy - y), got.get(i), i) for i, (sx, sy) in enumerate(trace['scenario']['samples'])
            if got.get(i, math.inf) > t + 1e-9]
    return min(left, key=lambda p: p[0]) if left else (None, None, None)


def revisits(trace, sources=None):
    """Повторные отправки: решения, первая подцель которых лежит у места, уже проверенного и пустого
    (ближе SAME_PLACE_M), или у места прежнего штрафа (ближе SAME_PLACE_M). Счёт одинаков для любых агентов:
    берутся только планы, пути, трек и события судьи. sources — считать решения только этих источников (None — все).

    Возвращает {'decisions', 'empty_places', 'to_empty', 'to_empty_collected', 'penalties', 'to_penalty', 'list'}:
    to_empty_collected — сколько повторных отправок в пустое место кончилось сбором образца (то есть место
    пустым не было или рядом лежал другой образец).
    """
    plans, events = trace.get('plans') or [], trace.get('events') or []
    empties = empty_places(trace)
    pens = [(float(e['t']), float(e['x']), float(e['y']), e['type']) for e in events if e.get('type') in PENALTIES]
    out = {'decisions': 0, 'empty_places': len(empties), 'to_empty': 0, 'to_empty_collected': 0,
           'penalties': len(pens), 'to_penalty': 0, 'list': []}
    for i, p in enumerate(plans):
        sg = _first_target(p)
        if sources is not None and p.get('source') not in sources:
            continue
        out['decisions'] += 1
        if sg is None:
            continue
        t = float(p['t'])
        # Место, проверенное в этот же момент, — это повод нынешнего решения, оно тоже «уже проверено».
        hit_e = [e for e in empties if e[0] <= t + 1e-9 and math.hypot(sg['x'] - e[1], sg['y'] - e[2]) <= SAME_PLACE_M]
        hit_p = [e for e in pens if e[0] <= t + 1e-9 and math.hypot(sg['x'] - e[1], sg['y'] - e[2]) <= SAME_PLACE_M]
        if hit_e:
            nxt = plans[i + 1] if i + 1 < len(plans) else None
            got = bool(nxt and nxt.get('trigger') == 'sample_collected')
            out['to_empty'] += 1
            out['to_empty_collected'] += got
            out['list'].append({'t': t, 'kind': 'empty', 'goal': _goal(sg), 'checked_t': hit_e[-1][0],
                                'why': hit_e[-1][3], 'collected': got, 'source': p.get('source')})
        if hit_p:
            out['to_penalty'] += 1
            out['list'].append({'t': t, 'kind': 'penalty', 'goal': _goal(sg), 'penalty_t': hit_p[-1][0],
                                'penalty': hit_p[-1][3], 'source': p.get('source')})
    return out


# --- миссии словами, которым нужна память о прошлом ----------------------------------------------

MISSIONS = {
    'N1': {'name': 'Не проверять одно место дважды',
           'text': 'Собирай образцы, но не проверяй одно и то же место дважды: если робот уже был на месте и образца '
                   'там не оказалось, больше не выбирай цель ближе 0,5 м от этого места.'},
    'N2': {'name': 'Два пустых подъезда подряд — домой',
           'text': 'Собирай образцы, но если два подъезда к кандидатам подряд не дали образца (робот был на месте, '
                   'а образца нет) — немедленно возвращайся на базу.'},
    'N3': {'name': 'После штрафа не ближе метра',
           'text': 'Собирай образцы, но после любого штрафа больше не выбирай цель ближе 1 м от места этого штрафа.'},
}
RETURN_DELAY_S = 2.0     # «немедленно» в N2 — как в did/mission_criteria.py
_EMPTY_ON_SITE = ('candidate_refuted', 'false_collect')


def verify(mission_id, trace):
    """Выполнена ли миссия N1–N3 по записи прогона: {'success': True | False | None, 'outcome', 'details'}.

    None — повода не было (ни одного пустого места, двух пустых подъездов подряд, штрафа): миссию этот прогон
    не проверяет. Критерии считаются по решениям планировщика любого источника: если решение приняло запасное
    правило вместо модели, нарушение всё равно засчитывается роботу.

      N1 — после первого пустого места ни одно решение не назначило первой подцелью цель ближе 0,5 м от уже
           проверенного пустого места;
      N2 — после первого случая «два пустых подъезда подряд» (кандидат, пропавший по дороге, подъездом не
           считается) робот повернул домой не позже RETURN_DELAY_S
           (did.mission_criteria.return_decision) и ни одной подцели на сбор или разведку больше не назначено;
           поворот домой раньше этого случая — не нарушение и не выполнение (повода не было);
      N3 — после первого штрафа ни одно решение не назначило первой подцелью цель ближе 1 м от места любого уже
           полученного штрафа. Дополнительно (details['truth_reentries']) — сколько раз истинная траектория
           снова вошла в круг 1 м вокруг места штрафа, выйдя из него: дорогу выбирает исполнитель, не планировщик.
    """
    from .mission_criteria import return_decision
    plans, events, res = trace.get('plans') or [], trace.get('events') or [], trace.get('result') or {}
    track = trace.get('track')
    returned = bool(res.get('returned'))
    base = {'returned': returned, 'collected': res.get('samples_collected')}
    if mission_id == 'N1':
        r = revisits(trace)
        d = {**base, 'empty_places': r['empty_places'], 'repeats': r['to_empty'],
             'list': [x for x in r['list'] if x['kind'] == 'empty']}
        if not r['empty_places']:
            return {'success': None, 'outcome': 'no_occasion', 'details': d}
        return {'success': r['to_empty'] == 0, 'outcome': 'clean' if r['to_empty'] == 0 else 'repeated', 'details': d}
    if mission_id == 'N2':
        seq = [a for a in approaches(trace) if a[1] != 'lost']      # пропавший по дороге — не подъезд
        t2 = next((b[0] for a, b in zip(seq, seq[1:]) if a[1] in _EMPTY_ON_SITE and b[1] in _EMPTY_ON_SITE), None)
        dec = return_decision(plans, track, trace.get('modes'))
        d = {**base, 'approaches': len(seq), 'two_empty_t': t2, 'return_t': dec and dec['t'], 'reaction_s': None,
             'goals_after': None}
        if t2 is None:
            return {'success': None, 'outcome': 'no_occasion', 'details': d}
        # Решение с поводом «второй пустой подъезд» принято в тот же момент t2: оно и должно быть возвратом.
        d['goals_after'] = sum(1 for p in plans if float(p['t']) >= t2 - 1e-9 and _first_target(p) is not None)
        if dec is not None:
            d['reaction_s'] = round(dec['t'] - t2, 1)
        if d['goals_after'] or dec is None or d['reaction_s'] > RETURN_DELAY_S + 1e-9:
            return {'success': False, 'outcome': 'no_reaction', 'details': d}
        if d['reaction_s'] < -1e-9:
            return {'success': None, 'outcome': 'no_occasion', 'details': d}
        return {'success': True, 'outcome': 'returned_on_two_empty', 'details': d}
    if mission_id == 'N3':
        pens = [(float(e['t']), float(e['x']), float(e['y'])) for e in events if e.get('type') in PENALTIES]
        near = []
        for p in plans:
            sg = _first_target(p)
            hit = [q for q in pens if sg and q[0] <= float(p['t']) + 1e-9
                   and math.hypot(sg['x'] - q[1], sg['y'] - q[2]) < PENALTY_NEAR_M]
            if hit:
                near.append({'t': float(p['t']), 'goal': _goal(sg), 'penalty_t': hit[-1][0], 'source': p.get('source')})
        d = {**base, 'penalties': len(pens), 'goals_near': len(near), 'list': near,
             'truth_reentries': _reentries(trace.get('truth'), pens)}
        if not pens:
            return {'success': None, 'outcome': 'no_occasion', 'details': d}
        return {'success': not near, 'outcome': 'clean' if not near else 'goal_near_penalty', 'details': d}
    raise ValueError(f'Неизвестная миссия: {mission_id}')


def _reentries(truth, pens):
    """Сколько раз истинная траектория вернулась в круг PENALTY_NEAR_M у места штрафа, уже выехав из него."""
    if not truth or not pens:
        return None
    dt, n = float(truth['dt']), 0
    for t0, px, py in pens:
        left = False
        inside = True
        for i in range(int(t0 / dt) + 1, len(truth['x'])):
            now = math.hypot(truth['x'][i] - px, truth['y'][i] - py) < PENALTY_NEAR_M
            left = left or not now
            n += left and now and not inside
            inside = now
    return n
