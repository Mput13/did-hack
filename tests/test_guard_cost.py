"""Проверки исследования G3: защита G2 не должна вредить там, где её повод ложный.

Что было не так (каждая проверка ниже на коде G2 падает):
- штраф «столкновение» всегда включал 10 секунд осторожной езды, хотя судья даёт его за вход в полосу у
  преграды, без упора; прогон сдвигался на секунды и дальше шёл по другой траектории;
- законная стоянка (ожидание датчика или ответа модели, опыт исследователя) засчитывалась как «робот ехал
  и не сдвинулся»: отъезд назад и та же осторожная езда.
"""
import pytest

from did.agent import Agent, make_config
from did.arena import load_arena
from did.config import SCIENCE, Rules
from did.fastsim import FastSim
from did.runner import run_episode
from did.scenario import generate
from test_bump import _IO, PILLAR, _agent, _obs

OPEN = (0.5, 0.5, 0.4)                      # место вдали от стен и столбов
SAME = ('score', 'time', 'distance', 'samples_collected', 'returned', 'collisions', 'hazard_hits')


@pytest.fixture(scope='module')
def arena():
    return load_arena()


def _driving(arena, **config):
    """Агент, который едет к дальней цели из открытого места; планировщик его не перебивает."""
    bot = Agent(arena, make_config('adaptive', **config), n_samples=3)
    bot.tracker._calib = None
    bot.queue = [{'type': 'goto', 'x': 1.6, 'y': 0.9}]
    bot._trigger = None
    return bot, _IO()


def _speeds(bot, io, arena, t0, t1, events=()):
    """Скорости, которые агент командует на тактах t0…t1, стоя в OPEN (события — на первом такте)."""
    out = []
    for k in range(round((t1 - t0) * 10)):
        bot.tick(_obs(arena, OPEN, t=t0 + 0.1 * k, events=events if k == 0 else (), scan=k % 4 == 0), io)
        if bot.mode != 'escape':
            out.append(io.cmd[0])
    return out


# --- штраф «столкновение»: объяснимый и необъяснимый ------------------------------------------------

NEAR = (PILLAR[0] + 0.28, PILLAR[1], 0.0)   # по карте до столба 10 см: робот сам в полосе штрафа судьи


def _hit(pose, t):
    return [{'type': 'collision', 't': t, 'x': pose[0], 'y': pose[1]}]


def test_explained_collision_penalty_does_not_start_cautious_driving(arena):
    """Робот и по своей позе в полосе у столба: штраф объясним, упора нет. Отъезд по лидару остаётся,
    а осторожной езды нет. Второй штраф подряд — уже повод."""
    assert arena.clearance(*NEAR[:2]) <= Rules().collision_clearance_m + 0.03
    bot = _agent(arena)
    bot._last_cmd = (0.2, 0.0)
    bot._perceive(_obs(arena, NEAR, t=10.0, events=_hit(NEAR, 10.0)))
    assert bot._escape is not None and bot._alert_until < 0.0
    bot._perceive(_obs(arena, NEAR, t=13.0, events=_hit(NEAR, 13.0)))
    assert bot._alert_until == pytest.approx(13.0 + Agent.ALERT_S)


def test_unexpected_collision_penalty_still_starts_cautious_driving(arena):
    """По карте до преград полметра, а штраф пришёл: поза неверна (так бывает в Gazebo и при рывках курса).
    Здесь защита G2 нужна целиком: робот отъезжает и едет не быстрее ALERT_V."""
    bot, io = _driving(arena)
    assert max(_speeds(bot, io, arena, 0.0, 1.0)) > Agent.ALERT_V + 0.01
    after = _speeds(bot, io, arena, 1.0, 4.0, events=_hit(OPEN, 1.0))
    assert bot._alert_until == pytest.approx(1.0 + Agent.ALERT_S)
    assert after and max(after) <= Agent.ALERT_V + 1e-9


def test_run_with_a_collision_penalty_is_the_same_with_and_without_guard():
    """E15, hard 1024: единственный штраф «столкновение» на 31-й секунде, робот срезал у столба, упора нет.
    С защитой G2 прогон после него расходился с прежним (в условии c2_2 это стоило возврата на базу)."""
    old = run_episode('hard', 1024, 'adaptive', save=False, config={'guard': False})['metrics']
    new = run_episode('hard', 1024, 'adaptive', save=False)['metrics']
    assert old['collisions'] == 1
    assert [new[k] for k in SAME] == [old[k] for k in SAME]


# --- законная стоянка — не «робот не движется» -----------------------------------------------------

def _stuck_after(arena, times, guard=True):
    bot = _agent(arena, guard=guard)
    for t in times:
        bot._watch_stuck(_obs(arena, OPEN, t=t, scan=False), 0.2)
    return bot


def test_voluntary_stop_is_not_taken_for_a_stuck_robot(arena):
    """Робот секунду ехал, 11 секунд стоял по своей воле (ждал датчик, ответ модели, ставил опыт) и поехал
    дальше. Прежний счёт прибавлял всю стоянку ко «времени езды» и объявлял застревание."""
    stop_and_go = [0.1 * k for k in range(11)] + [12.0, 12.1]
    bot = _stuck_after(arena, stop_and_go)
    assert bot._escape is None and bot._alert_until < 0.0
    assert not any('не движется' in e['text'] for e in bot.journal.entries)
    assert _stuck_after(arena, stop_and_go, guard=False)._escape is not None     # рука «как раньше» не тронута

    bot = _stuck_after(arena, [0.1 * k for k in range(42)])               # настоящий упор: ехал 4 с и не сдвинулся
    assert bot._escape is not None and bot._alert_until > 4.0


@pytest.mark.parametrize('seed', [1036, 1011])
def test_scientist_standing_for_an_experiment_is_not_stuck(arena, seed):
    """E10, hard 1036 и E14 (залипание датчика), hard 1011: столкновений нет, исследователь стоит, пока
    датчик не оживёт. С защитой G2 после стоянки шли ложное «робот не движется», отъезд, 10 секунд
    осторожной езды — и робот не возвращался на базу."""
    scenario = generate('hard', seed, arena, **({'fault_kinds': ['sensor_stuck'], 'n_hazards': 3} if seed == 1011 else {}))
    rules = Rules(**SCIENCE)
    world = FastSim(arena, scenario, rules, seed=seed)
    bot = Agent(arena, make_config('scientist'), n_samples=len(scenario.samples), rules=rules)
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    score = world.judge.score()
    assert score['collisions'] == 0 and score['returned']
    assert not any('не движется' in e['text'] for e in bot.journal.entries)
    assert bot._alert_until < 0.0                                         # осторожная езда не включалась ни разу
