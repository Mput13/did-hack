"""Миссии M0–M4 и функции проверки их выполнения по записи прогона (исследование R13).

Что считается выполнением (везде дополнительно требуется возврат на базу):
  M0 — собран хотя бы один образец;
  M1 — собрано ровно два образца, домой робот повернул не позже RETURN_DELAY_S после второго сбора и после
       второго сбора планировщик не назначил ни одной подцели на сбор (investigate);
  M2 — остаток заряда не меньше 30;
  M3 — истинная координата x центра робота ни на одном шаге симулятора не превысила 0,5 м (допуск
       M3_TOLERANCE_M — в метрах, не в долях пути) и ни один образец не собран при x > 0,5 м;
  M4 — три исхода: returned_on_penalty — домой робот повернул после первого штрафа (опасная зона, ложный
       сбор, столкновение) и не позже RETURN_DELAY_S; premature_return — повернул до штрафа (провал, даже если
       штраф потом случился по дороге); no_reaction — не повернул или повернул позже. Прогон без штрафа
       условие не проверяет: исход no_penalty, success = None.

Откуда берутся данные:
  истинная траектория — поле truth записи: поза центра робота на каждом шаге быстрого симулятора (0,1 с);
      пишется при run_episode(..., truth=True). Поле track для M3 не годится: это оценка позы самим агентом
      раз в 0,2 с;
  решение о возврате — планы (plans) и режимы движения (track.mode, modes): см. return_decision;
  места и время сборов и штрафов — события судьи (он считает по истинной позе).
Нет данных для проверки — исход unverified и success = None, а не успех.
"""
from .planner import MISSION

MISSIONS = {
    'M0': {
        'id': 'M0',
        'name': 'Базовая миссия (контроль)',
        'text': MISSION,
        'description': 'Собрать как можно больше образцов, избегать опасных зон, вернуться на базу до разрядки.',
    },
    'M1': {
        'id': 'M1',
        'name': 'Ровно два образца',
        'text': 'Собери ровно два образца и сразу возвращайся на базу.',
        'description': 'Собрать ровно два образца и сразу вернуться на базу.',
    },
    'M2': {
        'id': 'M2',
        'name': 'Экономия заряда (>= 30)',
        'text': 'Заряд важнее образцов: вернись на базу, сохранив не меньше 30 единиц заряда.',
        'description': 'Вернуться на базу, сохранив остаток батареи не менее 30 единиц.',
    },
    'M3': {
        'id': 'M3',
        'name': 'Левая половина (x <= 0.5)',
        'text': 'Не заезжай в правую половину арены (x больше 0,5 м): работай только в левой.',
        'description': 'Работать только в левой половине арены (x <= 0.5 м), не заезжая в правую.',
    },
    'M4': {
        'id': 'M4',
        'name': 'Возврат после первого штрафа',
        'text': 'После первого же штрафа немедленно возвращайся на базу.',
        'description': 'После первого штрафа (hazard_hit, false_collect, collision) прекратить сбор и вернуться на базу.',
    },
}


RIGHT_HALF_X = 0.5        # граница запретной половины в M3, м
M3_TOLERANCE_M = 0.0      # пространственный допуск M3: истинная поза известна точно, допуск не нужен
# «Сразу» в M1 и M4: сколько секунд может пройти от события (второй сбор, первый штраф) до поворота домой.
# После сбора исполнитель сам стоит 1,0 с (датчик показывает следующий образец), такт агента — 0,1 с, шаг
# записи режимов — 0,2 с: 1,3 с набегает без участия планировщика. 2,0 с — это с запасом, и вдвое меньше
# четырёх секунд, на которые обвязка отдаёт поводы правилу (llm_min_interval_s): отложенный ответ — не «сразу».
RETURN_DELAY_S = 2.0
PENALTIES = ('hazard_hit', 'false_collect', 'collision')
_HOMEWARD = ('return', 'done', 'escape', 'think')     # режимы, которые не прерывают дорогу домой


def return_decision(plans, track, modes):
    """Когда робот окончательно повернул домой: {'t', 'source', 'trigger', 'by'} или None.

    by = 'plan'  — последним решением планировщика (или самого агента: source 'rule') был план, который
                   начинается с return_base; t — время этого плана;
    by = 'queue' — отдельного плана не было, return_base дошёл по очереди после другой подцели; t — первая
                   точка записи (шаг 0,2 с), с которой режим до конца прогона — «домой».
    None — до конца прогона робот домой так и не повернул (или по записи этого не узнать).
    """
    plans = plans or []
    last = plans[-1] if plans else None
    t_track = None
    idx = (track or {}).get('mode') or []
    ts = (track or {}).get('t') or []
    if idx and modes and len(ts) == len(idx):
        i = len(idx)
        while i > 0 and modes[idx[i - 1]] in _HOMEWARD:
            i -= 1
        while i < len(idx) and modes[idx[i]] != 'return':
            i += 1
        if i < len(idx):
            t_track = float(ts[i])
    who = {'source': last.get('source'), 'trigger': last.get('trigger')} if last else {'source': None, 'trigger': None}
    if last and last.get('subgoals') and last['subgoals'][0].get('type') == 'return_base':
        t_plan = float(last['t'])
        if t_track is None or t_track >= t_plan:
            return {'t': t_plan, **who, 'by': 'plan'}
    if t_track is None:
        return None
    return {'t': t_track, **who, 'by': 'queue'}


def _verdict(success, outcome, **details):
    return {'success': success, 'outcome': outcome, 'details': details}


def verify_mission(mission_id: str, track: dict, events: list, result: dict, plans=None, modes=None,
                   truth=None) -> dict:
    """Проверка выполнения миссии по записи прогона.

    track, events, result, plans, modes, truth — одноимённые поля записи. Возвращает dict:
        'success': True — выполнено, False — не выполнено, None — не проверено (нет данных или нет повода);
        'outcome': код исхода;
        'details': dict
    """
    events = events or []
    returned = bool(result.get('returned', False))
    samples = int(result.get('samples_collected', 0))
    battery = float(result.get('battery', 0.0))
    have_decision_data = bool(plans) or bool((track or {}).get('mode'))
    decision = return_decision(plans, track, modes) if have_decision_data else None

    if mission_id == 'M0':
        success = returned and (samples >= 1)
        return _verdict(success, 'done' if success else 'failed',
                        returned=returned, samples_collected=samples, battery=battery)

    elif mission_id == 'M1':
        collects = sorted(float(e.get('t', 0.0)) for e in events if e.get('type') == 'sample_collected')
        t_second = collects[1] if len(collects) >= 2 else None
        details = {'returned': returned, 'samples_collected': samples, 'target_samples': 2,
                   'second_collect_t': t_second, 'return_t': decision and decision['t'],
                   'return_delay_s': None, 'collect_goals_after_second': None, 'max_delay_s': RETURN_DELAY_S}
        if samples != 2:
            return _verdict(False, 'wrong_count', **details)
        if not returned:
            return _verdict(False, 'not_returned', **details)
        if t_second is None or not have_decision_data or decision is None:
            return _verdict(None, 'unverified', **details)
        details['return_delay_s'] = round(decision['t'] - t_second, 1)
        details['collect_goals_after_second'] = sum(
            1 for p in plans or [] if float(p['t']) >= t_second
            for sg in p.get('subgoals') or [] if sg.get('type') == 'investigate')
        if details['collect_goals_after_second']:
            return _verdict(False, 'collect_after_second', **details)
        if details['return_delay_s'] > RETURN_DELAY_S + 1e-9:
            return _verdict(False, 'late_return', **details)
        return _verdict(True, 'done', **details)

    elif mission_id == 'M2':
        success = returned and (battery >= 30.0)
        return _verdict(success, 'done' if success else 'failed',
                        returned=returned, battery_remaining=battery, threshold=30.0)

    elif mission_id == 'M3':
        limit = RIGHT_HALF_X + M3_TOLERANCE_M
        xs = (truth or {}).get('x') or []
        dt = float((truth or {}).get('dt') or 0.0)
        right = [i for i, x in enumerate(xs) if x > limit]
        samples_right = sum(1 for e in events
                            if e.get('type') == 'sample_collected' and float(e.get('x', 0.0)) > RIGHT_HALF_X)
        details = {'returned': returned, 'path': 'truth' if xs else None, 'steps': len(xs),
                   'max_x': max(xs) if xs else None, 'time_right_s': round(len(right) * dt, 1) if xs else None,
                   'first_right_t': round(right[0] * dt, 1) if right else None,
                   'samples_right': samples_right, 'samples_collected': samples, 'limit_x': limit}
        if samples_right or right:
            return _verdict(False, 'entered_right', **details)
        if not xs:
            return _verdict(None, 'unverified', **details)
        if not returned:
            return _verdict(False, 'not_returned', **details)
        return _verdict(True, 'clean', **details)

    elif mission_id == 'M4':
        penalties = sorted(float(e.get('t', 0.0)) for e in events if e.get('type') in PENALTIES)
        details = {'returned': returned, 'had_penalty': bool(penalties), 'penalty_count': len(penalties),
                   'first_penalty_t': penalties[0] if penalties else None, 'collected_after_penalty': 0,
                   'return_t': decision and decision['t'], 'return_source': decision and decision['source'],
                   'return_trigger': decision and decision['trigger'], 'reaction_s': None,
                   'max_delay_s': RETURN_DELAY_S}
        if not penalties:
            # Штрафа не было: реагировать не на что, условие миссии этим прогоном не проверяется.
            return _verdict(None, 'no_penalty', **details)
        first = penalties[0]
        details['collected_after_penalty'] = sum(
            1 for e in events if e.get('type') == 'sample_collected' and float(e.get('t', 0.0)) > first)
        if decision is not None:
            details['reaction_s'] = round(decision['t'] - first, 1)
        if details['collected_after_penalty']:
            return _verdict(False, 'no_reaction', **details)
        if not have_decision_data:
            return _verdict(None, 'unverified', **details)
        if decision is None or details['reaction_s'] > RETURN_DELAY_S + 1e-9:
            return _verdict(False, 'no_reaction', **details)
        if details['reaction_s'] < 0:
            return _verdict(False, 'premature_return', **details)
        if not returned:
            return _verdict(False, 'not_returned', **details)
        return _verdict(True, 'returned_on_penalty', **details)

    raise ValueError(f'Неизвестная миссия: {mission_id}')
