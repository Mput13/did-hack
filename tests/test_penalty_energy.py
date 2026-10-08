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


def _drive(hit, agent_hit=None, steps=90, hit_step=40, lag=0, dear_from=None):
    """Робот едет прямо по обычному полу; на шаге hit_step судья отнимает hit единиц заряда и шлёт событие.

    lag — на сколько тактов показание батареи отстаёт от события (в ROS это разные темы);
    dear_from — с какого шага пол становится дороже в 4 раза.
    """
    arena = load_arena()
    rules = Rules() if agent_hit is None else Rules(hazard_battery_hit=agent_hit)
    bot = Agent(arena, make_config('scientist'), n_samples=3, rules=rules)
    th = next(a for a in (k * math.pi / 8 for k in range(16))
              if all(arena.clearance(BASE[0] + d * math.cos(a), BASE[1] + d * math.sin(a)) >= 0.2
                     for d in (0.3, 0.6, 0.9, 1.2, 1.5)))
    world, battery, step, dt = Rules(), 60.0, 0.015, 0.1
    hit_xy = None
    for k in range(steps):
        x, y = BASE[0] + k * step * math.cos(th), BASE[1] + k * step * math.sin(th)
        events = []
        if k:
            mult = 4.0 if dear_from is not None and k >= dear_from else 1.0
            battery -= world.drain_per_m * mult * step + world.drain_idle_per_s * dt
        if k == hit_step + lag:
            battery -= hit
        if k == hit_step:
            events, hit_xy = [{'type': 'hazard_hit', 't': k * dt, 'x': x, 'y': y}], (x, y)
        bot._perceive(Observation(t=k * dt, x=x, y=y, th=th, v=0.15, w=0.0, battery=battery, sensor=None, scan=None,
                                  events=events))
    return bot, hit_xy, world


@pytest.mark.parametrize('hit,agent_hit,lag', [(3.0, None, 0), (5.0, None, 0), (3.0, 0.0, 0), (3.0, None, 3), (0.5, None, 0)],
                         ids=['hit3', 'hit5_agent_believes_3', 'agent_expects_no_loss', 'battery_reading_lags', 'hit0.5'])
def test_penalty_loss_is_not_soil(hit, agent_hit, lag):
    bot, (hx, hy), world = _drive(hit, agent_hit, lag=lag)
    assert [q.anomaly['text'] for q in bot.inv.inquiries if q.topic == 'energy'] == []
    assert bot.soil.at(hx, hy)[0] < 1.3                       # пол в месте штрафа остался обычным
    assert max(z['mult'] for z in bot.soil.zones()) < 1.4 if bot.soil.zones() else True
    assert bot.inv.model.per_meter(0) == pytest.approx(world.drain_per_m, rel=0.1)


def test_drive_without_penalty_is_quiet():
    """Та же поездка без штрафа: расследований нет — значит, проверка выше ловит именно штраф."""
    bot, _, world = _drive(0.0, hit_step=10 ** 6)
    assert bot.inv.inquiries == []


def test_dear_soil_right_after_penalty_is_still_noticed():
    """Сразу за местом штрафа начинается грунт ×4: вычитается только скачок, настоящий расход остаётся в окнах."""
    bot, _, _ = _drive(3.0, hit_step=40, dear_from=42)
    opened = [q for q in bot.inv.inquiries if q.topic == 'energy']
    assert len(opened) == 1
    assert 1.8 < opened[0].est['ratio'] < 5.0      # дорогой грунт (первые окна захватывают и обычный пол), но не ×10


def test_without_penalty_event_a_jump_still_raises_alarm():
    """Скачок заряда без события штрафа — по-прежнему странность: поправка действует только сразу после штрафа."""
    bot, _, _ = _drive(3.0, hit_step=10 ** 6, lag=40 - 10 ** 6)
    assert [q.topic for q in bot.inv.inquiries] == ['energy']


def test_hard_1006_penalty_does_not_open_soil_inquiry():
    """Прогон из ревью: штраф на 18,2 с давал расследование «грунт ×10,4» и гипотезу о грунте с ошибкой 600%."""
    s = run_episode('hard', 1006, 'scientist', save=False, knowledge={})
    assert s['metrics']['hazard_hits'] >= 1
    jumps = [q for q in s['science']['inquiries'] if q['topic'] == 'energy'
             and q['anomaly'].get('unit') == 'ед/м' and q['anomaly']['observed'] > 5 * q['anomaly']['expected']]
    assert [(q['t_open'], q['anomaly']['text']) for q in jumps] == []
    assert s['metrics']['hyp_soil_error'] < 1.0
