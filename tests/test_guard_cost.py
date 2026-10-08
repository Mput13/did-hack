"""Проверки исследования G3: защита G2 не должна срабатывать от стоянки по своей воле — и не должна слабеть.

Что было не так в main (G2): законная стоянка (ожидание датчика или ответа модели, опыт исследователя)
засчитывалась как «робот ехал и не сдвинулся»: отъезд назад и 10 секунд осторожной езды.

Что было не так в первом круге G3 (нашло ревью) и что здесь закреплено:
- детектор застревания начинал счёт заново после любого перерыва между тактами дольше 0,5 с — и при
  опоздавших тактах (Gazebo на загруженной машине) не срабатывал вовсе, хотя робот всё это время давил в преграду;
- штраф «столкновение» у преграды считался «объяснимым» и осторожную езду не включал — в том числе при
  настоящем упоре и на неувиденном месте карты SLAM. Теперь любой такой штраф включает её, как в main.
"""
import math

import pytest

from did.agent import Agent, make_config
from did.arena import load_arena
from did.config import SCIENCE, Rules
from did.fastsim import FastSim
from did.runner import run_episode
from did.scenario import generate
from did.slam_map import UNKNOWN, SlamArena, outline
from test_bump import _IO, KICKS, PILLAR, _agent, _obs, _sim_at

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


# --- штраф «столкновение» всегда включает осторожную езду ---------------------------------------------

NEAR = (PILLAR[0] + 0.28, PILLAR[1], 0.0)   # по карте до столба 10 см: робот и по своей позе в полосе штрафа судьи


def _hit(pose, t):
    return [{'type': 'collision', 't': t, 'x': pose[0], 'y': pose[1]}]


def test_collision_penalty_near_a_pillar_starts_cautious_driving(arena):
    """Робот и по своей позе стоит в полосе у столба. Расстояние по карте не доказывает, что упора не
    было (серия G2, сбои лидара, hard 1004: при 11 см по карте упор настоящий) — осторожность включается."""
    assert arena.clearance(*NEAR[:2]) <= Rules().collision_clearance_m
    bot = _agent(arena)
    bot._last_cmd = (0.2, 0.0)
    bot._perceive(_obs(arena, NEAR, t=10.0, events=_hit(NEAR, 10.0)))
    assert bot._escape is not None and bot._alert_until == pytest.approx(10.0 + Agent.ALERT_S)


def test_collision_penalty_on_unseen_slam_map_starts_cautious_driving(arena):
    """Карта SLAM: место под роботом лидар не увидел, зазор там по карте нулевой. Это не «объяснение»
    штрафа: осторожная езда включается после каждого."""
    values = outline(arena)
    ix, iy = SlamArena(values).w2g(*OPEN[:2])
    values[iy - 8:iy + 9, ix - 8:ix + 9] = UNKNOWN
    slam = SlamArena(values, near=(-2.0, -0.5))
    assert slam.clearance(*OPEN[:2]) == 0.0
    bot = Agent(slam, make_config('adaptive', localize=False), n_samples=3)
    for t in (10.0, 21.0, 32.0):
        bot._last_cmd = (0.2, 0.0)
        bot._perceive(_obs(arena, OPEN, t=t, events=_hit(OPEN, t)))
        assert bot._escape is not None and bot._alert_until == pytest.approx(t + Agent.ALERT_S)


def test_collision_penalty_in_the_open_starts_cautious_driving(arena):
    """По карте до преград полметра, а штраф пришёл: поза неверна (так бывает в Gazebo и при рывках курса).
    Робот отъезжает и едет не быстрее ALERT_V."""
    bot, io = _driving(arena)
    assert max(_speeds(bot, io, arena, 0.0, 1.0)) > Agent.ALERT_V + 0.01
    after = _speeds(bot, io, arena, 1.0, 4.0, events=_hit(OPEN, 1.0))
    assert bot._alert_until == pytest.approx(1.0 + Agent.ALERT_S)
    assert after and max(after) <= Agent.ALERT_V + 1e-9


def test_physical_bump_run_keeps_the_protection_of_main():
    """Серия G2, рывки курса до 140° и сбои лидара, hard 1004 (находка ревью): на 55,6 с штраф у столба при
    11 см по карте — упор настоящий. Без осторожной езды робот через 2,5 с сталкивался снова (2 столкновения,
    152,4 с, 58,86). Числа ниже — этот же прогон на коде main."""
    m = run_episode('hard', 1004, 'adaptive', save=False, sim=dict(KICKS, lidar_fault=20.0, lidar_fault_s=4.0))['metrics']
    assert (m['collisions'], m['returned']) == (1, True)
    assert m['time'] == pytest.approx(109.7, abs=0.05) and m['score'] == pytest.approx(59.40, abs=0.005)


# --- детектор застревания: стоянка по своей воле — не езда, опоздавший такт — езда ---------------------

def _press(bot, io, obs, v=0.2, w=0.0):
    """Такт езды к цели, как его делает _drive_to: сторож застревания, сразу за ним команда."""
    bot._tick_n = getattr(bot, '_tick_n', 0) + 1    # счётчик тактов ведёт Agent.tick; здесь такт собран вручную
    bot._waiting = False
    bot._watch_stuck(obs, v)
    bot._command(io, v, w)
    bot._watch_idle(obs)


def _stand(bot, io, obs, v=0.0, watched=False):
    """Такт стоянки по своей воле: ответ модели, опыт, набор показаний (watched — детектор при этом вызван)."""
    bot._tick_n = getattr(bot, '_tick_n', 0) + 1    # счётчик тактов ведёт Agent.tick; здесь такт собран вручную
    bot._waiting = True
    if watched:
        bot._watch_stuck(obs, v)
    bot._command(io, v, 0.0)
    bot._watch_idle(obs)


def _false_alarm(bot):
    return bot._escape is not None or bot._alert_until > 0.0 or any('не движется' in e['text'] for e in bot.journal.entries)


@pytest.mark.parametrize('how', ['unwatched', 'watched', 'last_tick_forward'])
def test_voluntary_stop_is_not_taken_for_a_stuck_robot(arena, how):
    """Робот секунду ехал, 11 секунд стоял по своей воле и поехал дальше. Счёт main прибавлял всю стоянку
    ко «времени езды» и объявлял застревание. Стоянка не засчитывается, как бы она ни выглядела: такты без
    детектора, такты с детектором и нулевой командой, и когда на последнем такте стоянки чужой исполнитель
    (опыт расследования) дал команду «вперёд»."""
    def run(guard):
        bot, io = _agent(arena, guard=guard), _IO()
        obs = lambda t: _obs(arena, OPEN, t=t, scan=False)
        for k in range(11):
            _press(bot, io, obs(0.1 * k))
        for k in range(11, 121):
            _stand(bot, io, obs(0.1 * k), watched=how == 'watched',
                   v=0.15 if how == 'last_tick_forward' and k == 120 else 0.0)
        for k in range(21):
            _press(bot, io, obs(12.1 + 0.1 * k))
        return bot
    assert not _false_alarm(run(guard=True))
    if how == 'unwatched':
        assert run(guard=False)._escape is not None            # рука «как до G2» не тронута


def _pressing(arena, dt, seconds=12.0):
    """Робот упёрся в столб (физика быстрого симулятора с упором) и давит вперёд, подправляя курс; такты
    приходят раз в dt. Возвращает время, когда сторож начал отъезд (None — не начал)."""
    world = _sim_at(arena, (-0.76, -1.1, math.pi), bump=True, dt=dt)
    bot = _agent(arena)
    for _ in range(round(seconds / dt) + 1):
        pose = (world.x, world.y, world.th)                    # поза по лидару: в упоре она стоит на месте
        _press(bot, world, _obs(arena, pose, t=world.t, scan=False),
               w=max(-1.0, min(1.0, 2.0 * math.remainder(math.pi - world.th, 2 * math.pi))))
        if bot._escape is not None:
            assert bot._alert_until == pytest.approx(world.t + Agent.ALERT_S)
            return world.t
        world.advance()
    return None


@pytest.mark.parametrize('dt, t_main', [(0.1, 8.1), (0.6, 4.2), (1.5, 4.5)])
def test_real_bump_is_caught_as_in_main_even_with_delayed_ticks(arena, dt, t_main):
    """Настоящий упор: команда «вперёд» стоит, робот не сдвигается. Между опоздавшими тактами (Gazebo на
    загруженной машине: 0,6–1,5 с) робот продолжает давить, и это время — езда. Отъезд начинается тогда же,
    когда на коде main (t_main, снято с кода main этой же проверкой; при 0,1 с первое окно в 4 с застало
    подъезд к столбу, срабатывает второе). Счёт «заново после перерыва» при 0,6 и 1,5 с не срабатывал вовсе:
    отъезд начинал только сторож простоя, на 10-й секунде."""
    assert _pressing(arena, dt) == pytest.approx(t_main, abs=1e-6)


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
