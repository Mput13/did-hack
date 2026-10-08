"""Разовая потеря заряда при штрафе в опасной зоне — не расход на грунт (находка ревью V1, прогон scientist/hard-1006).

Судья при въезде в опасную зону разово отнимает hazard_battery_hit единиц заряда и сообщает о штрафе событием.
Исследователь (did/inquiry.py) копит расход окнами пути; потеря от штрафа не должна попадать ни в тревогу
о расходе, ни в карту стоимости грунта, ни в подбор модели расхода. При этом настоящий дорогой грунт сразу
за местом штрафа должен замечаться, как и раньше.
"""
import math

import pytest

from did.agent import Agent, make_config
from did.arena import load_arena
from did.config import BASE, Rules
from did.robot_io import Observation
from did.runner import run_episode


def _drive(hit, agent_hit=None, steps=90, hit_step=40, lag=0, dear_from=None, every=1, jumps=None):
    """Робот едет прямо по обычному полу; на шаге hit_step судья отнимает hit единиц заряда и шлёт событие.

    lag — на сколько тактов скачок показания батареи отстаёт от события (в ROS это разные темы); меньше нуля —
    показание приходит раньше события, как в стенде: судья публикует батарею до событий;
    dear_from — с какого шага пол становится дороже в 4 раза;
    every — показание батареи обновляется раз в столько тактов: расход копится и приходит одним числом;
    jumps — {шаг: потеря заряда} без события штрафа.
    """
    arena = load_arena()
    rules = Rules() if agent_hit is None else Rules(hazard_battery_hit=agent_hit)
    bot = Agent(arena, make_config('scientist'), n_samples=3, rules=rules)
    th = next(a for a in (k * math.pi / 8 for k in range(16))
              if all(arena.clearance(BASE[0] + d * math.cos(a), BASE[1] + d * math.sin(a)) >= 0.2
                     for d in (0.3, 0.6, 0.9, 1.2, 1.5)))
    world, battery, step, dt = Rules(), 60.0, 0.015, 0.1
    hit_xy, shown = None, battery
    for k in range(steps):
        x, y = BASE[0] + k * step * math.cos(th), BASE[1] + k * step * math.sin(th)
        events = []
        if k:
            mult = 4.0 if dear_from is not None and k >= dear_from else 1.0
            battery -= world.drain_per_m * mult * step + world.drain_idle_per_s * dt
        if k == hit_step + lag:
            battery -= hit
        battery -= (jumps or {}).get(k, 0.0)
        if k == hit_step:
            events, hit_xy = [{'type': 'hazard_hit', 't': k * dt, 'x': x, 'y': y}], (x, y)
        if k % every == 0:
            shown = battery
        bot._perceive(Observation(t=k * dt, x=x, y=y, th=th, v=0.15, w=0.0, battery=shown, sensor=None, scan=None,
                                  events=events))
    return bot, hit_xy, world


def _energy(bot):
    return [q for q in bot.inv.inquiries if q.topic == 'energy']


def _quiet(bot, hx, hy, world):
    """Штраф не оставил следа: тревоги о расходе нет, пол в месте штрафа обычный, модель расхода не сбита."""
    assert [q.anomaly['text'] for q in _energy(bot)] == []
    assert bot.soil.at(hx, hy)[0] < 1.3
    assert max(z['mult'] for z in bot.soil.zones()) < 1.4 if bot.soil.zones() else True
    assert bot.inv.model.per_meter(0) == pytest.approx(world.drain_per_m, rel=0.1)
    assert bot.inv._held == []                                # придержанных окон не осталось


@pytest.mark.parametrize('hit,agent_hit,lag,found', [(3.0, None, 0, 'hit'), (5.0, None, 0, 'bad'), (3.0, 0.0, 0, 'bad'),
                                                     (3.0, None, 3, 'hit'), (0.5, None, 0, 'bad')],
                         ids=['hit3', 'hit5_agent_believes_3', 'agent_expects_no_loss', 'battery_reading_lags', 'hit0.5'])
def test_penalty_loss_is_not_soil(hit, agent_hit, lag, found):
    """found: 'hit' — потеря сошлась с правилами агента и вычтена; 'bad' — не сошлась, отрезок загрязнён."""
    bot, (hx, hy), world = _drive(hit, agent_hit, lag=lag)
    _quiet(bot, hx, hy, world)
    assert bot.inv.penalty.stats == {'hit': 0, 'bad': 0, 'lost': 0, found: 1}


@pytest.mark.parametrize('lag', [-6, -3, -2, -1, 1, 2, 6])
def test_order_of_battery_and_event_does_not_matter(lag):
    """Находка второго круга: в ROS судья публикует батарею раньше события (lag < 0). При lag=-1 прежняя
    правка давала расследование «расход 24,8 ед/м при прогнозе 2,6»."""
    bot, (hx, hy), world = _drive(3.0, lag=lag)
    _quiet(bot, hx, hy, world)
    assert bot.inv.inquiries == []
    assert bot.inv.penalty.stats == {'hit': 1, 'bad': 0, 'lost': 0}


@pytest.mark.parametrize('hit_step', [40, 41, 42, 43])
@pytest.mark.parametrize('lag', [0, -1])
def test_battery_reading_that_piles_up(hit_step, lag):
    """Показание батареи обновляется раз в четыре такта: расход 0,4 с и штраф приходят одним числом,
    позже события или раньше него."""
    bot, (hx, hy), world = _drive(3.0, every=4, hit_step=hit_step, lag=lag)
    _quiet(bot, hx, hy, world)
    assert bot.inv.penalty.stats == {'hit': 1, 'bad': 0, 'lost': 0}


def test_drive_without_penalty_is_quiet():
    """Та же поездка без штрафа: расследований нет — значит, проверка выше ловит именно штраф."""
    for every in (1, 4):
        bot, _, world = _drive(0.0, hit_step=10 ** 6, every=every)
        assert bot.inv.inquiries == []
        assert bot.inv.penalty.stats == {'hit': 0, 'bad': 0, 'lost': 0}


def test_dear_soil_right_after_penalty_is_still_noticed():
    """Сразу за местом штрафа начинается грунт ×4: вычитается только скачок, настоящий расход остаётся в окнах."""
    bot, _, _ = _drive(3.0, hit_step=40, dear_from=42)
    opened = _energy(bot)
    assert len(opened) == 1
    assert 1.8 < opened[0].est['ratio'] < 5.0      # дорогой грунт (первые окна захватывают и обычный пол), но не ×10


@pytest.mark.parametrize('every,hit_step', [(1, 40), (4, 41)], ids=['every_tick', 'piled_up'])
def test_penalty_on_dear_soil_in_the_same_tick_keeps_real_drain(every, hit_step):
    """Находка второго круга: штраф и грунт ×4 в одном отрезке. Вычитается ровно потеря из правил агента,
    настоящий расход остаётся: тревога и оценка грунта те же, что в такой же поездке без штрафа.
    Прежняя правка оставляла в отрезке расход обычного пола (0,154 вместо 0,604 в примере ревью)."""
    with_hit, _, _ = _drive(3.0, hit_step=hit_step, dear_from=40, every=every)
    no_hit, _, _ = _drive(0.0, hit_step=10 ** 6, dear_from=40, every=every)
    a, b = _energy(with_hit), _energy(no_hit)
    assert len(a) == len(b) == 1
    assert a[0].t_open == b[0].t_open
    assert a[0].est['ratio'] == pytest.approx(b[0].est['ratio'], abs=1e-6)
    assert a[0].anomaly['observed'] == b[0].anomaly['observed']
    assert with_hit.inv.penalty.stats == {'hit': 1, 'bad': 0, 'lost': 0}


def test_reviewer_example_exact_split():
    """Пример ревью: за 0,4 с робот проходит 0,06 м по грунту ×4 (расход 0,604) и получает штраф 3 единицы.
    Из окна уходит ровно 3; следующий скачок на 1 единицу без события окно не трогает."""
    from did.penalty import PenaltyLedger
    r = Rules()
    most = lambda ds, dth, dt, t: 7.0 * r.drain_per_m * ds + r.drain_idle_per_s * dt     # noqa: E731
    led, win = PenaltyLedger(3.0, most), {'b0': 60.0}
    at = lambda t, x, b: Observation(t=t, x=x, y=0.0, th=0.0, v=0.15, w=0.0, battery=b, sensor=None, scan=None)  # noqa: E731
    led.reading(at(0.0, 0.0, 60.0), win)
    led.event(0.4)
    led.reading(at(0.4, 0.06, 60.0 - 0.604 - 3.0), win)
    assert win == {'b0': 57.0} and not led.hold
    led.reading(at(0.5, 0.075, 60.0 - 0.604 - 3.0 - 0.151 - 1.0), win)       # второй скачок, события нет
    assert win == {'b0': 57.0}
    assert led.hold                                # скачок без события: окно ждёт, не придёт ли событие
    led.reading(at(1.1, 0.165, 60.0 - 0.604 - 3.0 - 0.151 - 1.0 - 0.9), win)
    assert win == {'b0': 57.0} and not led.hold and led.stats == {'hit': 1, 'bad': 0, 'lost': 0}


def test_second_jump_without_event_is_not_swallowed():
    """Одному событию штрафа — одна поправка. Второй скачок через 0,3 с, уже без события, — странность:
    тревога та же, что от такого же скачка в поездке без штрафа. Прежняя правка вычитала и его."""
    bot, _, _ = _drive(3.0, hit_step=40, jumps={43: 1.0})
    alone, _, _ = _drive(0.0, hit_step=10 ** 6, jumps={43: 1.0})
    assert len(_energy(bot)) == len(_energy(alone)) == 1
    assert _energy(bot)[0].anomaly['observed'] == _energy(alone)[0].anomaly['observed']
    assert bot.inv.penalty.stats == {'hit': 1, 'bad': 0, 'lost': 0}


def test_two_penalties_in_one_reading():
    """Две зоны разом: два события и одно падение на 6 единиц — две поправки, по одной на событие."""
    from did.penalty import PenaltyLedger
    led, win = PenaltyLedger(3.0, lambda ds, dth, dt, t: 17.5 * ds + 0.01 * dt), {'b0': 60.0}
    at = lambda t, x, b: Observation(t=t, x=x, y=0.0, th=0.0, v=0.15, w=0.0, battery=b, sensor=None, scan=None)  # noqa: E731
    led.reading(at(0.0, 0.0, 60.0), win)
    led.event(0.1)
    led.event(0.1)
    led.reading(at(0.1, 0.015, 60.0 - 0.04 - 6.0), win)
    assert win == {'b0': 54.0} and led.stats == {'hit': 2, 'bad': 0, 'lost': 0} and not led.hold


def test_expected_loss_that_never_shows_spoils_the_neighbourhood():
    """Агент ждёт потерю 3 единицы, судья отнял 0,5, и она утонула в показании за четыре такта. Отделить её
    нельзя: отрезки вокруг события в оценки не идут, ложного дорогого грунта нет."""
    bot, (hx, hy), world = _drive(0.5, every=4, hit_step=41)
    _quiet(bot, hx, hy, world)
    assert bot.inv.penalty.stats == {'hit': 0, 'bad': 0, 'lost': 1}


def test_without_penalty_event_a_jump_still_raises_alarm():
    """Скачок заряда без события штрафа — по-прежнему странность: поправка действует только рядом с событием."""
    bot, _, _ = _drive(3.0, hit_step=10 ** 6, lag=40 - 10 ** 6)
    assert [q.topic for q in bot.inv.inquiries] == ['energy']
    assert 4.0 <= bot.inv.inquiries[0].t_open <= 4.0 + 0.6 + 0.8     # не позже ожидания события и одного окна


def test_event_that_comes_too_late_does_not_cancel_the_alarm():
    """Граница памяти: событие через секунду после скачка к нему уже не относится."""
    bot, _, _ = _drive(3.0, lag=-10)
    assert len(_energy(bot)) == 1


def test_hard_1006_penalty_does_not_open_soil_inquiry():
    """Прогон из ревью: штраф на 18,2 с давал расследование «грунт ×10,4» и гипотезу о грунте с ошибкой 600%."""
    s = run_episode('hard', 1006, 'scientist', save=False, knowledge={})
    assert s['metrics']['hazard_hits'] >= 1
    jumps = [q for q in s['science']['inquiries'] if q['topic'] == 'energy'
             and q['anomaly'].get('unit') == 'ед/м' and q['anomaly']['observed'] > 5 * q['anomaly']['expected']]
    assert [(q['t_open'], q['anomaly']['text']) for q in jumps] == []
    assert s['metrics']['hyp_soil_error'] < 1.0


# --- агент для исследований по заданию: своё окно расхода в дороге (did/study_agent.py, _floor) ----------------

TURN = {'quantity': 'turn_cost', 'allowed': {'spin': {'angle_deg': 360, 'repeats': 2}, 'pause': {'seconds': 3}},
        'stop': {'rel_error': 0.05}, 'budget': {'energy': 10, 'reserve': 1}}


def _study_drive(hit, jump_step, event_step, steps=60):
    """Агент для исследований едет прямо; на шаге jump_step показание батареи падает на hit, событие штрафа
    приходит на шаге event_step."""
    from did.study import prepare
    from did.study_agent import STUDY_CONFIG, StudyAgent
    arena, rules = load_arena(), Rules()
    bot = StudyAgent(arena, STUDY_CONFIG, 3, rules=rules, study=prepare(TURN, arena, rules))
    assert bot.phase == 'study'
    th = next(a for a in (k * math.pi / 8 for k in range(16))
              if all(arena.clearance(BASE[0] + d * math.cos(a), BASE[1] + d * math.sin(a)) >= 0.2
                     for d in (0.3, 0.6, 0.9, 1.2, 1.5)))
    battery, step, dt = 60.0, 0.015, 0.1
    for k in range(steps):
        x, y = BASE[0] + k * step * math.cos(th), BASE[1] + k * step * math.sin(th)
        if k:
            battery -= rules.drain_per_m * step + rules.drain_idle_per_s * dt
        if k == jump_step:
            battery -= hit
        events = [{'type': 'hazard_hit', 't': k * dt, 'x': x, 'y': y}] if k == event_step else []
        bot._perceive(Observation(t=k * dt, x=x, y=y, th=th, v=0.15, w=0.0, battery=battery, sensor=None, scan=None,
                                  events=events))
    return bot


@pytest.mark.parametrize('jump_step', range(18, 24))
def test_study_agent_floor_ignores_penalty_that_arrives_before_its_event(jump_step):
    """Показание батареи со штрафом пришло на такт раньше события (порядок стенда), и как раз на этом такте
    закрылось окно в 0,3 м: штраф не должен стать дорогим полом. Когда событие приходит вместе с показанием
    или раньше него, окно и прежде сбрасывалось — робот пятится из зоны."""
    bot = _study_drive(3.0, jump_step, jump_step + 1)
    assert [z['mult'] for z in bot.soil.zones() if z['mult'] >= 1.4] == []
    assert bot._travel[0] / max(bot._travel[1], 1e-9) < 3.0 if bot._travel[1] else True


def test_study_agent_floor_keeps_a_jump_without_event():
    """Без события штрафа скачок остаётся в окне: поправка действует только рядом с событием."""
    bot = _study_drive(3.0, 20, 10 ** 6)
    assert any(z['mult'] >= 1.4 for z in bot.soil.zones())
