"""Решение репетиции показа «прошло / не прошло» (tools/demo_check.py::verdict) — без сети и без стенда."""
import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('demo_check', Path(__file__).resolve().parent.parent / 'tools' / 'demo_check.py')
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)


def _route(done, total=4):
    return [{'x': 0.0, 'y': 0.0, 'done': i < done} for i in range(total)]


@pytest.fixture
def good():
    """Репетиция показа, в которой всё хорошо: объезд по четырём точкам и миссия на трудном уровне, сценарий 2."""
    return {
        'ready': True, 'expect': ['hard', '2'], 'scenario': ('hard', 2),
        'tour': {'points': 4, 'coverage_min': 0.6,
                 'state': {'mode': 'idle', 'route': _route(4), 'arrivals': [0.02], 'map': {'coverage': 0.83},
                           'score': {'collisions': 0}, 'note': {'tone': 'ok', 'text': 'Приехал'}}},
        'mission': {'state': 'finished', 'fresh_run': True, 'trace': 'ok', 'lag': 0.004,
                    'result': {'samples_collected': 7, 'samples_total': 7, 'returned': True, 'score': 91.8,
                               'collisions': 0, 'hazard_hits': 0, 'false_collects': 0}},
    }


def _why(facts, **expect):
    return '; '.join(dc.verdict(facts, **expect))


def test_good_rehearsal_passes(good):
    assert dc.verdict(good) == []
    assert dc.verdict(good, min_samples=7, min_score=90.0, clean=True) == []


def test_five_of_seven_with_hazard_hit_fails_only_when_expected(good):
    good['mission']['result'].update(samples_collected=5, hazard_hits=1, score=60.0)
    assert dc.verdict(good) == []                           # без ключей поведение прежнее: возврат и столкновения
    why = dc.verdict(good, min_samples=7, clean=True)
    assert len(why) == 2 and 'собрано 5 из 7' in why[0] and 'не чистый' in why[1]
    assert 'счёт 60.0 меньше' in _why(good, min_score=80)
    good['mission']['result'].update(samples_collected=7, hazard_hits=0, false_collects=1)
    assert 'ложных сборов 1' in _why(good, min_samples=7, clean=True)


def test_tour_cut_after_first_point_fails(good):
    """Пульт стоит, в arrivals есть запись (от прежнего маршрута), карта построена — но пройдена одна точка из четырёх."""
    good['tour']['state'].update(route=_route(1), note={'tone': 'bad', 'text': 'Пути к точке нет: маршрут остановлен'})
    why = _why(good, min_samples=7, clean=True)
    assert 'объезд не пройден целиком: пройдено точек 1 из 4' in why and 'Пути к точке нет' in why
    assert not dc.tour_done({'mode': 'idle', 'route': _route(4, 6), 'arrivals': [0.01]}, 6)
    assert not dc.tour_done({'mode': 'drive', 'route': _route(4)}, 4)       # ещё едет
    assert not dc.tour_done({'mode': 'idle', 'route': [], 'arrivals': [0.01]}, 4)
    assert dc.tour_done({'mode': 'idle', 'route': _route(4)}, 4)


def test_tour_refused_low_coverage_and_collision_fail(good):
    bad = copy.deepcopy(good)
    bad['tour'].update(refused='Точка 2 попала в стену или столб')
    assert 'объезд не начат: Точка 2' in _why(bad)
    bad = copy.deepcopy(good)
    bad['tour']['state']['map']['coverage'] = 0.4
    assert 'карта построена меньше' in _why(bad)
    bad = copy.deepcopy(good)
    bad['tour']['state']['score']['collisions'] = 1
    assert 'столкновение на объезде' in _why(bad)


def test_missing_trace_fails(good):
    good['mission'].update(trace='missing', lag=None)
    assert 'запись прогона не найдена' in _why(good) and 'нагрузку машины проверить нельзя' in _why(good)
    good['mission'].update(trace='OSError: битый файл')
    assert 'не читается' in _why(good)
    good['mission'].update(trace='ok', lag=0.08)
    assert 'опоздавших тактов 8.0%' in _why(good)


def test_trace_of_another_run_fails(good):
    """Файл записи есть и читается, но он от прежнего прогона того же агента на том же сценарии."""
    result = good['mission']['result']
    row = {'score': 91.8, 'collected': 7, 'lag': 0.0}
    assert dc.trace_foreign(row, result, mtime=1300.0, t_mission=1000.0) is None
    assert 'до старта миссии' in dc.trace_foreign(row, result, mtime=400.0, t_mission=1000.0)
    assert 'счёт в записи 88.1' in dc.trace_foreign({**row, 'score': 88.1}, result, 1300.0, 1000.0)
    assert 'образцов в записи 6' in dc.trace_foreign({**row, 'collected': 6}, result, 1300.0, 1000.0)
    assert dc.trace_foreign(row, {}, 1300.0, 1000.0)        # пульт итога не отдал — сверять не с чем
    good['mission'].update(trace='foreign', trace_why='счёт в записи 88.1, а в итоге пульта 91.8', lag=None)
    why = dc.verdict(good, min_samples=7, clean=True)
    assert len(why) == 1 and 'запись не от этого прогона (счёт в записи 88.1' in why[0]


def test_stale_run_and_refused_reset_fail(good):
    stale = copy.deepcopy(good)
    stale['mission']['fresh_run'] = False
    assert 'не на новом прогоне судьи' in _why(stale)
    del stale['mission']['fresh_run']                       # пульт не сообщил — тоже не подтверждено
    assert 'не на новом прогоне судьи' in _why(stale)
    good['mission'] = {'state': 'refused', 'reason': 'судья не ответил на сброс за 10 с'}
    why = dc.verdict(good, min_samples=7, clean=True)
    assert len(why) == 1 and 'миссия не стартовала: судья не начал новый прогон (судья не ответил' in why[0]
    good['mission'] = {'state': 'aborted', 'reason': None}
    assert 'миссия не прошла до конца: состояние «aborted»' in _why(good)


def test_other_steps(good):
    assert 'пульт не готов' in _why({'ready': False})
    good['mission']['result'].update(returned=False, collisions=2)
    why = _why(good)
    assert 'не вернулся на базу' in why and 'столкновение в миссии' in why
    good['scenario'] = ('hard', 9)
    assert 'hard-9 вместо hard-2' in _why(good)
    assert 'не доехал' in _why({'ready': True, 'goal': {'arrived': False, 'miss': None}})
    assert 'дальше 15 см' in _why({'ready': True, 'goal': {'arrived': True, 'miss': 0.3}})
    assert dc.verdict({'ready': True, 'goal': {'arrived': True, 'miss': 0.04}, 'home': {'at_base': True}}) == []
    assert 'не вернулся на базу по команде' in _why({'ready': True, 'home': {'at_base': False}})
