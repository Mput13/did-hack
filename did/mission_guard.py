"""Сторож миссии на модели решений Jev (исследование J1). По умолчанию выключен: AgentConfig.mission_guard = ''.

Схема одного решения (GuardedPlanner.plan):
  1. правило (HeuristicPlanner) мгновенно предлагает подцель;
  2. Jev одним запросом отвечает на вопросы «да/нет с вероятностью»: «по миссии пора немедленно на базу?» и,
     для каждой достижимой цели из сводки, «нарушит ли эта подцель миссию?» (все вопросы — в одном запросе:
     состояние читается один раз, вопросы оцениваются независимо, это один сетевой вызов на повод);
  3. вероятность ≥ hi — «да» (подцель отбрасывается, правило предлагает следующую; на «пора на базу» — возврат);
     вероятность ≤ lo — «нет» (робот едет); между lo и hi — сомнение:
       режим 'jev'     — осторожный вариант: сомнительная подцель отбрасывается;
       режим 'jev_llm' — решение целиком отдаётся большой языковой модели (LLMPlanner).
  Если отброшены все цели, правило само отвечает return_base («миссия исчерпана»).

Что видит Jev (build_request): текст миссии как есть и несколько чисел, общих для любой миссии: сколько
образцов собрано, сколько штрафов получено, заряд, сколько заряда останется на базе, если ехать домой сейчас
и если сначала выполнить подцель (осторожная оценка, с тем же запасом, что у агента); координаты цели. Какое из этих чисел важно для миссии и с чем его
сравнивать, код не знает — это и есть работа Jev. Арифметика (заряд после подцели) посчитана кодом: по
документации Jev считать не умеет.

Каждое решение сторожа пишется в запись прогона (поле guard плана): факты, вероятности, время ответа, цена.
По ним tools/jev_eval.py считает калибровку.

Запасной режим (fallback): если Jev недоступен — сеть, таймаут, потолок расходов, чужая модель в ответе, ответ не
по схеме (любая JevError) — сторож отключается до конца миссии, дальше решает одно правило, а причина пишется в
журнал решений: «сторож миссии отключён: …; миссия словами дальше не гарантируется». Причина — код из закрытого
списка (DISABLE_REASONS) и постоянное пояснение к нему; текст исключения в журнал решений и в запись прогона не
попадает вовсе (в нём могли бы оказаться строки сервера или транспорта). Для показа режим включён
(make_guarded без указаний), в исследовательских прогонах выключен (tools/jev_eval.py передаёт fallback=False):
там отказ Jev останавливает прогон, и числа не смешиваются с ездой по правилу. Промах кэша (CacheMiss) в режиме
«только кэш» останавливает прогон всегда: это не сбой модели, а неполный повтор.
"""
import json

from .llm import CacheMiss
from .llm_jev import MODEL as JEV_MODEL, JevClient, JevError, request_key
from .planner import MISSION, HeuristicPlanner

MODES = ('jev', 'jev_llm')
LO, HI = 0.35, 0.8                     # пороги уверенности; подобраны на сценариях 1–20 (tools/jev_eval.py --tune)
PROFILE = 'v5'

# Формулировки вопросов. Подбирались на сценариях 1–20; итоговая — PROFILE.
PROFILES = {
    'v1': {
        'done': {
            'instructions': 'According to `mission`, must the robot stop working and return to base right now?',
            'criteria': {'true': 'The mission is already completed or its condition to go home has been met, '
                                 'so any further work would break the mission.',
                         'false': 'The mission still allows or requires the robot to keep working.'},
        },
        'violates': {
            'question': 'Would the robot break the rule stated in `mission` if it performs `proposed_action` now?',
            'criteria': {'true': 'Performing the action breaks a limit or a prohibition stated in the mission.',
                         'false': 'The action is allowed by the mission.'},
        },
    },
    # v2: вопрос буквальнее — «судить только по тому, что миссия прямо ограничивает».
    'v2': {
        'done': {
            'instructions': 'Does `mission` say that, in the current state of `robot`, the robot must stop working '
                            'and go back to base immediately?',
            'criteria': {'true': 'The condition named in the mission for going back to base has already happened.',
                         'false': 'The condition named in the mission for going back has not happened yet, or the '
                                  'mission names no such condition.'},
        },
        'violates': {
            'question': 'Does performing `proposed_action` now contradict `mission`? Judge only by what the mission '
                        'text explicitly limits or forbids; facts that the mission does not mention do not matter.',
            'criteria': {'true': 'The mission text forbids this action in the current state of the robot, or the '
                                 'action makes a limit named in the mission impossible to keep.',
                         'false': 'Nothing in the mission text forbids this action in the current state of the robot.'},
        },
    },
}
# v3: те же вопросы, что в v2, но факты о роботе и подцель — фразами, а не числами в JSON.
PROFILES['v3'] = {**PROFILES['v2'], 'style': 'text'}
# v4: как v3, и Jev получает только те факты, о которых говорит миссия (отбор — отдельным запросом к Jev).
PROFILES['v4'] = {**PROFILES['v3'], 'filter': True}
# v5: как v3, но без положения самого робота: вопрос о цели, а робот, стоящий в запретной области, сбивал ответ.
PROFILES['v5'] = {**PROFILES['v3'], 'own_position': False}
# Запас к оценке заряда — тот же, с которым агент сам решает, достижима ли цель (AgentConfig.reserve_margin и
# reserve_abs): оценка дороги домой ×1,1 и ещё 4 единицы. Без него сторож пропускал подцели «впритык», а робот
# приезжал с меньшим зарядом, чем обещала оценка (отладочные сценарии 1–5, миссия M2).
RESERVE_MARGIN, RESERVE_ABS = 1.1, 4.0
ACTIONS = {'investigate': 'drive to the target and collect a sample there',
           'explore': 'drive to the target to look for samples around it'}
# Причины отключения сторожа: код (JevError.code) -> постоянное пояснение. Других текстов в журнал решений нет.
DISABLE_REASONS = {
    'network': 'нет связи с сервером Jev',
    'timeout': 'Jev не ответил в срок',
    'server': 'сервер Jev отказал',
    'budget': 'потолок расходов на Jev достигнут или работа с Jev остановлена',
    'journal': 'журнала расходов на Jev нет или он негоден',
    'wrong_model': 'на запрос ответила не Jev 1.13',
    'bad_reply': 'ответ Jev не по схеме',
    'key': 'ключ доступа к Jev не задан или не читается',
    'other': 'сбой обращения к Jev',
}


def disable_reason(error):
    """Код причины из закрытого списка по ошибке клиента; всё незнакомое — 'other'."""
    code = getattr(error, 'code', None)
    return code if isinstance(code, str) and code in DISABLE_REASONS else 'other'


def options(state):
    """Достижимые цели из сводки состояния в общем виде: кандидаты на сбор и точки разведки."""
    out = []
    for kind, key in (('investigate', 'candidates'), ('explore', 'explore_points')):
        for c in state.get(key) or []:
            if c.get('feasible'):
                out.append({'id': c['id'], 'kind': kind, 'x': c['x'], 'y': c['y'],
                            'battery_after': round(state['battery'] - c['cost_to'] - c['cost_back'] * RESERVE_MARGIN
                                                   - RESERVE_ABS, 1)})
    return out


def facts(state):
    """Числа о роботе, одинаковые для любой миссии."""
    pen = state.get('penalties') or {}
    return {'samples_collected': state['samples']['collected'], 'samples_total': state['samples']['total'],
            'penalties': int(pen.get('total', 0)), 'battery': state['battery'],
            'battery_if_return_now': round(state['battery'] - state['return_cost'] * RESERVE_MARGIN - RESERVE_ABS, 1),
            'x': state['pose']['x'], 'y': state['pose']['y'], 'trigger': state.get('trigger')}


# Какие группы фактов бывают и вопрос «миссия об этом?». Ответ — один запрос к Jev на текст миссии.
GROUPS = {
    'samples': 'Does the mission text set a limit or a condition on how many samples the robot collects?',
    'penalties': 'Does the mission text say what the robot must do when it receives a penalty?',
    'battery': 'Does the mission text set a limit on how much battery charge must be left?',
    'position': 'Does the mission text forbid the robot to drive into some part of the arena?',
}


def relevance_request(mission):
    return {'mission': mission}, {k: {'type': 'noul', 'instructions': q} for k, q in GROUPS.items()}


def relevant_groups(answers):
    """Группы фактов, о которых миссия говорит (вероятность от 0,5). Ни одной — передаются все."""
    groups = tuple(k for k in GROUPS if float(answers[k]['noul']) >= 0.5)
    return groups or tuple(GROUPS)


def build_request(state, profile=PROFILE, groups=None):
    """(состояние для Jev, вопросы, цели). У цели поле q — имя вопроса о ней; вопрос о возврате — 'done'."""
    opts = options(state)
    return (*request_from(state.get('mission') or MISSION, facts(state), opts, profile, groups), opts)


def request_from(mission, f, opts, profile=PROFILE, groups=None):
    """Запрос по тексту миссии, фактам о роботе и целям (так его можно собрать заново из записи прогона).

    groups — какие группы фактов передавать (None — все). Цели, вопросы о которых совпали дословно (например,
    когда координаты миссии не важны), спрашиваются один раз: в opts[i]['q'] пишется имя общего вопроса.
    """
    p = PROFILES[profile]
    text = p.get('style') == 'text'
    use = set(groups or GROUPS)
    if text:
        robot = ' '.join(filter(None, [
            'samples' in use and f"The robot has collected {f['samples_collected']} samples so far.",
            'penalties' in use and f"The robot has received {f['penalties']} penalties so far.",
            'battery' in use and (f"The battery of the robot now holds {f['battery']} units. If the robot drives back "
                                  f"to base right now, it arrives with {f['battery_if_return_now']} units of battery "
                                  f"left."),
            'position' in use and p.get('own_position', True) and f"The robot is at x = {f['x']} m, y = {f['y']} m."]))
    else:
        robot = {**({'samples_collected_so_far': f['samples_collected']} if 'samples' in use else {}),
                 **({'penalties_received_so_far': f['penalties']} if 'penalties' in use else {}),
                 **({'battery_now': f['battery'],
                     'battery_left_at_base_if_robot_returns_now': f['battery_if_return_now']} if 'battery' in use else {}),
                 **({'position': {'x': f['x'], 'y': f['y']}} if 'position' in use else {})}
    jev_state = {'mission': mission, 'robot': robot}
    questions = {'done': {'type': 'noul', **{k: p['done'][k] for k in ('instructions', 'criteria')}}}
    asked = {}
    for o in opts:
        if text:
            action = ' '.join(filter(None, [
                f"The robot will {ACTIONS[o['kind']]}.",
                'position' in use and f"The target is at x = {o['x']} m, y = {o['y']} m.",
                'battery' in use and (f"After that and the drive home the robot will arrive at base with "
                                      f"{o['battery_after']} units of battery left.")]))
        else:
            action = {'action': ACTIONS[o['kind']],
                      **({'target': {'x': o['x'], 'y': o['y']}} if 'position' in use else {}),
                      **({'battery_left_at_base_after_this_action': o['battery_after']} if 'battery' in use else {})}
        same = json.dumps(action, sort_keys=True)
        o['q'] = asked.setdefault(same, o['id'])
        if o['q'] == o['id']:
            questions[o['id']] = {'type': 'noul',
                                  'instructions': {'proposed_action': action, 'question': p['violates']['question']},
                                  'criteria': p['violates']['criteria']}
    return jev_state, questions


def verdict(p, lo=LO, hi=HI):
    """'yes' | 'no' | 'unsure' по вероятности «да»."""
    return 'yes' if p >= hi else 'no' if p <= lo else 'unsure'


class GuardedPlanner:
    """Правило + сторож Jev (+ большая модель для сомнительных случаев). Интерфейс планировщика: plan(state)."""
    source = 'guard'

    def __init__(self, jev, mode='jev', big=None, rule=None, lo=LO, hi=HI, profile=PROFILE, fallback=False):
        if mode not in MODES:
            raise ValueError(f"неизвестный режим сторожа «{mode}»; есть: {', '.join(MODES)}")
        if mode == 'jev_llm' and big is None:
            raise ValueError('режиму jev_llm нужна большая модель (LLMPlanner)')
        self.jev, self.mode, self.big = jev, mode, big
        self.rule = rule or HeuristicPlanner()
        self.lo, self.hi, self.profile = float(lo), float(hi), profile
        self.client = getattr(big, 'client', None)
        self.calls = self.escalations = 0
        self.fallback = bool(fallback)                 # при отказе Jev — отключить сторожа и ехать по правилу
        self.disabled = None                           # причина отключения сторожа до конца миссии
        self._groups = {}                              # текст миссии -> (группы фактов, время ответа, цена, токены)

    def _relevant(self, mission):
        """Какие факты относятся к миссии: один запрос к Jev на текст миссии (первое решение прогона)."""
        if not PROFILES[self.profile].get('filter'):
            return None, None
        fresh = None
        if mission not in self._groups:
            reply = self.jev.ask(*relevance_request(mission))
            self.calls += 1
            self._groups[mission] = relevant_groups(reply.answers)
            fresh = {'groups': list(self._groups[mission]),
                     'p': {k: float(a['noul']) for k, a in reply.answers.items()}, 'latency_ms': reply.latency_ms,
                     'cached': reply.cached, 'cost_usd': reply.cost_usd,
                     'input_tokens': reply.usage.get('input_tokens', 0)}
        return self._groups[mission], fresh

    def plan(self, state):
        first = self.rule.plan(state)
        if self.disabled:
            return {**first, 'reasoning': first['reasoning'] + ' (Сторож миссии отключён: решает правило.)'}
        if first['subgoals'][0]['type'] == 'return_base':
            return first                               # правило и так едет домой: спрашивать не о чем
        try:
            groups, relevance = self._relevant(state.get('mission') or MISSION)
            jev_state, questions, opts = build_request(state, self.profile, groups)
            reply = self.jev.ask(jev_state, questions)
        except CacheMiss:
            raise                                      # повтор «только кэш» неполон: прогон останавливается
        except JevError as e:
            if not self.fallback:
                raise
            return self._disable(first, e)
        self.calls += 1
        probs = {name: float(a['noul']) for name, a in reply.answers.items()}
        probs.update({o['id']: probs[o['q']] for o in opts})
        log = {'key': request_key(JEV_MODEL, jev_state, questions)[:16], 'mode': self.mode, 'profile': self.profile, 'lo': self.lo, 'hi': self.hi, 'facts': facts(state),
               'done_p': probs['done'], 'options': [{**o, 'p': probs[o['id']]} for o in opts], 'asked': [],
               'latency_ms': reply.latency_ms, 'cached': reply.cached, 'cost_usd': reply.cost_usd,
               'input_tokens': reply.usage.get('input_tokens', 0), 'decision': None, 'escalated': False}
        wait_s = reply.latency_ms / 1000.0
        if relevance:
            log['relevance'] = relevance
            wait_s += relevance['latency_ms'] / 1000.0

        # Неуверенный ответ на «пора ли на базу» решения не меняет: дальше каждую цель сторож проверяет отдельно.
        if verdict(probs['done'], self.lo, self.hi) == 'yes':
            log['decision'] = 'return'
            return self._result(f"Сторож миссии (Jev): по миссии пора на базу (уверенность {probs['done']:.0%}).",
                                [{'type': 'return_base'}], 'guard', log, wait_s)

        rejected, work, plan = [], state, first
        while plan['subgoals'][0]['type'] != 'return_base':
            target = plan['subgoals'][0]['target']
            log['asked'].append(target)
            v = verdict(probs[target], self.lo, self.hi)
            if v == 'no':
                log['decision'] = 'go'
                note = (f" Сторож миссии (Jev): подцель {target} миссию не нарушает ({probs[target]:.0%})"
                        + (f"; отброшены как нарушающие: {', '.join(rejected)}." if rejected else '.'))
                return self._result(plan['reasoning'] + note, plan['subgoals'], plan['source'], log, wait_s)
            if v == 'unsure' and self.mode == 'jev_llm':
                return self._escalate(state, log, wait_s, f'нарушает ли {target} миссию — неясно ({probs[target]:.0%})')
            rejected.append(f'{target} ({probs[target]:.0%})')
            work = {**work, **{key: [{**c, 'feasible': c['feasible'] and c['id'] != target} for c in work[key]]
                               for key in ('candidates', 'explore_points')}}
            plan = self.rule.plan(work)
        log['decision'] = 'exhausted'
        return self._result(f"Сторож миссии (Jev): все предложенные правилом цели нарушают миссию "
                            f"({', '.join(rejected)}) — возвращаюсь на базу.",
                            [{'type': 'return_base'}], 'guard', log, wait_s)

    def _disable(self, first, error):
        """Jev недоступен: сторож выключается до конца миссии, это и следующие решения принимает правило."""
        # В журнал решений и запись прогона идёт только код причины и постоянное пояснение: текст исключения
        # сюда не копируется, даже очищенный.
        code = disable_reason(error)
        self.disabled = f'{code}: {DISABLE_REASONS[code]}'
        note = (f' Сторож миссии отключён: {self.disabled}; миссия словами дальше не гарантируется, до конца '
                f'миссии решает правило.')
        return {**first, 'reasoning': first['reasoning'] + note,
                'guard': {'mode': self.mode, 'profile': self.profile, 'decision': 'disabled', 'reason': code,
                          'disabled': self.disabled}, 'wait_s': 0.0}

    def _escalate(self, state, log, wait_s, why):
        self.escalations += 1
        plan = self.big.plan(state)
        big_ms = sum(ex.get('latency_ms') or 0 for ex in plan.get('exchanges') or [])
        log.update(decision='escalated', escalated=True, llm_latency_ms=big_ms, llm_source=plan['source'])
        plan = dict(plan)
        plan['reasoning'] = f'Сторож миссии (Jev) не уверен: {why}. Решает большая модель. ' + plan['reasoning']
        plan['guard'] = log
        plan['wait_s'] = wait_s + big_ms / 1000.0
        return plan

    @staticmethod
    def _result(reasoning, subgoals, source, log, wait_s):
        return {'reasoning': reasoning, 'hypotheses': [], 'subgoals': subgoals, 'source': source, 'exchanges': [],
                'error': None, 'guard': log, 'wait_s': wait_s}


def make_guarded(cfg, llm, build_llm):
    """Планировщик со сторожем по настройкам агента. llm — словарь как у did.runner.make_planner; ключ 'jev' —
    опции клиента Jev ({'strict': True} — только кэш) или готовый клиент ({'client': ...}); там же 'fallback':
    что делать при отказе Jev — True (по умолчанию, для показа): отключить сторожа и ехать по правилу; False
    (исследовательские прогоны): остановить прогон ошибкой. build_llm() — большая модель (вызывается только в
    режиме jev_llm)."""
    jev_opts = dict((llm or {}).get('jev') or {})
    fallback = bool(jev_opts.pop('fallback', True))
    jev = jev_opts.pop('client', None) or JevClient(**jev_opts)
    big = build_llm() if cfg.mission_guard == 'jev_llm' else None
    return GuardedPlanner(jev, cfg.mission_guard, big, fallback=fallback)

