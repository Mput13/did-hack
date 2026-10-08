"""Миссии M0–M4 и функции проверки их выполнения по записи прогона."""
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


def verify_mission(mission_id: str, track: dict, events: list, result: dict) -> dict:
    """Проверка выполнения миссии по записи прогона (track, events, result).

    Возвращает dict:
        'success': bool,
        'details': dict
    """
    returned = bool(result.get('returned', False))
    samples = int(result.get('samples_collected', 0))
    battery = float(result.get('battery', 0.0))

    if mission_id == 'M0':
        success = returned and (samples >= 1)
        return {
            'success': success,
            'details': {'returned': returned, 'samples_collected': samples, 'battery': battery}
        }

    elif mission_id == 'M1':
        success = returned and (samples == 2)
        return {
            'success': success,
            'details': {'returned': returned, 'samples_collected': samples, 'target_samples': 2}
        }

    elif mission_id == 'M2':
        success = returned and (battery >= 30.0)
        return {
            'success': success,
            'details': {'returned': returned, 'battery_remaining': battery, 'threshold': 30.0}
        }

    elif mission_id == 'M3':
        xs = track.get('x', []) if isinstance(track, dict) else []
        n_total = len(xs)
        n_right = sum(1 for x in xs if x > 0.5)
        fraction_right = round(n_right / n_total, 4) if n_total > 0 else 0.0

        samples_right = [
            e for e in (events or [])
            if e.get('type') == 'sample_collected' and float(e.get('x', 0.0)) > 0.5
        ]
        success = returned and (fraction_right <= 0.01) and (len(samples_right) == 0)
        return {
            'success': success,
            'details': {
                'returned': returned,
                'fraction_right': fraction_right,
                'samples_right': len(samples_right),
                'samples_collected': samples,
            }
        }

    elif mission_id == 'M4':
        penalties = [
            e for e in (events or [])
            if e.get('type') in ('hazard_hit', 'false_collect', 'collision')
        ]
        if not penalties:
            had_penalty = False
            first_penalty_t = None
            collected_after = 0
            success = returned
        else:
            had_penalty = True
            first_penalty_t = min(float(e.get('t', 0.0)) for e in penalties)
            collected_after = sum(
                1 for e in (events or [])
                if e.get('type') == 'sample_collected' and float(e.get('t', 0.0)) > first_penalty_t
            )
            success = returned and (collected_after == 0)

        return {
            'success': success,
            'details': {
                'returned': returned,
                'had_penalty': had_penalty,
                'penalty_count': len(penalties),
                'first_penalty_t': first_penalty_t,
                'collected_after_penalty': collected_after,
            }
        }

    raise ValueError(f'Неизвестная миссия: {mission_id}')
