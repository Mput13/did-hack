"""P3: попутный сбор (did/pickup.py), проба точного пересчёта карты после сбора (did/replay.py), уточнения разбора потерь."""
import math
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from did.agent import PRESETS, V4, Agent, AgentConfig, make_config
from did.arena import load_arena
from did.belief import SampleBelief
from did.replay import exact_replay
from did.robot_io import Observation
from did.runner import run_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import loss_breakdown as lb     # noqa: E402

HERE = (-0.6, 0.2)


@pytest.fixture(scope='module')
def arena():
    return load_arena()


class FakeIO:
    """Судья на столе: сбор удаётся, если так задано; команды скорости запоминаются."""

    def __init__(self, ok=True):
        self.ok, self.collects, self.cmds = ok, 0, []

    def collect(self):
        self.collects += 1
        return self.ok, 'ok' if self.ok else 'miss'

    def command(self, v, w):
        self.cmds.append((v, w))


def _obs(t, x, y, v=0.0, battery=30.0):
    return Observation(t=t, x=x, y=y, th=0.0, v=v, w=0.0, battery=battery, sensor=None, scan=None, events=[],
                       done=False)


def _bot(arena, name='adaptive_v4', sure=True, **config):
    """Агент по дороге на базу; карта образцов уверена (или нет), что образец лежит прямо под роботом."""
    bot = Agent(arena, make_config(name, **config), n_samples=3)
    bot._returning = True
    bot._trigger = None
    bot.queue = [{'type': 'return_base'}]
    if sure:
        k = int(np.hypot(bot.belief.cx - HERE[0], bot.belief.cy - HERE[1]).argmin())
        bot.belief.p[k] = 0.95
    return bot


# --- новое поведение выключено по умолчанию ----------------------------------------------------------

def test_new_behaviour_is_off_by_default(arena):
    cfg = AgentConfig()
    assert not cfg.pickup and not cfg.exact_replay
    for name in ('adaptive', 'adaptive_v2', 'adaptive_v3', 'scientist', 'scientist_v2', 'scientist_v3', 'fixed'):
        assert not PRESETS[name].pickup and not PRESETS[name].exact_replay, name
    old = Agent(arena, make_config('adaptive_v2'), n_samples=3)
    assert old.pickup is None and old.belief.replay is None
    new = Agent(arena, make_config('adaptive_v4'), n_samples=3)
    assert new.pickup is not None and new.belief.replay is None        # точный пересчёт в v4 не входит


def test_v4_is_v2_plus_pickup():
    for old, new in (('adaptive_v2', 'adaptive_v4'), ('scientist_v2', 'scientist_v4')):
        a, b = PRESETS[old], PRESETS[new]
        changed = {f.name for f in fields(AgentConfig) if getattr(a, f.name) != getattr(b, f.name)}
        assert changed == {'name'} | set(V4), changed
    assert V4 == {'pickup': True}


# --- попутный сбор -----------------------------------------------------------------------------------

def test_old_agent_drives_past_a_sample_in_reach(arena):
    bot, io = _bot(arena, 'adaptive_v2'), FakeIO()
    bot._do_return(_obs(50.0, *HERE), io)
    assert io.collects == 0 and bot.collected == 0 and bot.mode == 'return'


def test_pickup_on_the_way_home_keeps_the_goal(arena):
    bot, io = _bot(arena), FakeIO()
    bot._do_return(_obs(50.0, *HERE, v=0.2), io)               # ещё едет: сначала остановиться
    assert io.collects == 0 and io.cmds[-1] == (0.0, 0.0) and bot.queue == [{'type': 'return_base'}]
    bot._do_return(_obs(50.2, *HERE), io)
    assert io.collects == 1 and bot.collected == 1
    assert bot.queue == [{'type': 'return_base'}] and bot._returning and bot._trigger is None
    assert bot.belief.left == 2 and bot.belief.prob_within(*HERE, 0.25) < 0.05
    bot._wait_until = 0.0
    bot._do_return(_obs(51.5, *HERE), io)                      # образец взят — дальше обычный возврат
    assert io.collects == 1 and bot.mode == 'return'


def test_pickup_while_exploring_asks_for_a_new_plan(arena):
    bot, io = _bot(arena), FakeIO()
    bot._returning = False
    bot._trigger = None
    bot.queue = [{'type': 'explore', 'x': 1.0, 'y': 1.0}]
    bot._do_goto(bot.queue[0], _obs(20.0, *HERE), io)
    assert io.collects == 1 and bot.collected == 1 and bot._trigger == 'sample_collected'


def test_pickup_needs_the_same_confidence_as_an_approach(arena):
    bot, io = _bot(arena, sure=False), FakeIO()
    k = int(np.hypot(bot.belief.cx - HERE[0], bot.belief.cy - HERE[1]).argmin())
    bot.belief.p[k] = bot.cfg.collect_confidence - 0.1
    bot._do_return(_obs(50.0, *HERE), io)
    assert io.collects == 0 and bot.mode == 'return'
    bot.belief.p[k] = 0.95
    far = (HERE[0] + bot.cfg.collect_reach + 0.12, HERE[1])    # образец за радиусом сбора агента
    bot._do_return(_obs(50.2, *far), io)
    assert io.collects == 0


def test_pickup_miss_is_not_repeated(arena):
    bot, io = _bot(arena), FakeIO(ok=False)
    bot._do_return(_obs(50.0, *HERE), io)
    assert io.collects == 1 and bot.collected == 0 and bot.queue == [{'type': 'return_base'}]
    k = int(np.hypot(bot.belief.cx - HERE[0], bot.belief.cy - HERE[1]).argmin())
    bot.belief.p[k] = 0.95                                     # карта снова уверена — но здесь только что был промах
    bot._do_return(_obs(50.4, *HERE), io)
    assert io.collects == 1 and bot.mode == 'return'
    bot._no_collect_until = 1e9                                # пауза после двух промахов действует и на попутный сбор
    bot._misses.clear()
    bot._do_return(_obs(90.0, *HERE), io)
    assert io.collects == 1


def test_pickup_gives_up_if_confidence_drops_while_braking(arena):
    bot, io = _bot(arena), FakeIO()
    bot._do_return(_obs(50.0, *HERE, v=0.2), io)
    bot.belief.p[:] = 1e-4
    bot._do_return(_obs(50.2, *HERE), io)
    assert io.collects == 0 and bot.mode == 'return' and bot.queue == [{'type': 'return_base'}]


def test_v4_collects_a_sample_it_passes_on_the_way_home():
    """Отладочный сценарий 33: по дороге на базу v2 проезжает в 18 см от образца при уверенности 88%."""
    old = run_episode('hard', 33, 'adaptive_v2', save=False)['metrics']
    new = run_episode('hard', 33, 'adaptive_v4', save=False)['metrics']
    assert new['samples_collected'] == old['samples_collected'] + 1
    assert new['returned'] and new['false_collects'] == old['false_collects']
    assert new['score'] > old['score'] + 9.0


# --- проба: точный пересчёт карты после сбора --------------------------------------------------------

def _two_sample_maps(arena):
    """Робот едет к образцу A; по дороге ближайшим какое-то время был образец B, лежащий чуть в стороне."""
    a, b = (0.6, -0.3), (-0.5, 0.75)
    path = [(-1.5 + 0.04 * i, -0.3) for i in range(53)]
    rng = np.random.default_rng(2)
    old, new = SampleBelief(arena, 3, 2.0), SampleBelief(arena, 3, 2.0)
    new.replay = exact_replay
    for x, y in path:
        z = max(0.0, 1.0 - min(math.dist((x, y), a), math.dist((x, y), b)) / 2.0)
        z = float(np.clip(z + rng.normal(0.0, 0.05), 0.0, 1.0))
        old.update(x, y, z, 0.05)
        new.update(x, y, z, 0.05)
    assert np.array_equal(old.p, new.p)                        # до сбора флажок ничего не меняет
    return old, new, a, b, path[-1]


def test_exact_replay_keeps_the_other_sample(arena):
    old, new, a, b, end = _two_sample_maps(arena)
    before = old.prob_within(*b, 0.35)
    old.collected(*end)
    new.collected(*end)
    assert new.left == old.left == 2
    assert new.total() == pytest.approx(2.0, rel=0.05)
    assert new.prob_within(*a, 0.3) < 0.02                     # место собранного образца очищено
    assert new.prob_within(*b, 0.35) >= old.prob_within(*b, 0.35)
    assert new.prob_within(*b, 0.35) >= 0.5 * before
    assert not new._log and np.array_equal(new._epoch, new.p)


# --- разбор потерь -----------------------------------------------------------------------------------

def test_approach_is_assigned_by_its_final_target_too():
    samples = [(0.0, 0.0), (3.0, 3.0)]
    drifted = (10.0, 20.0, 'return', 0.9, 0.0, 0.3, 0.0)           # начальная цель в 0,9 м от образца, последняя — в 0,3 м
    assert dict(lb.assign_attempts([drifted], samples, {})) == {0: [drifted]}
    still = (10.0, 20.0, 'return', 0.9, 0.0, 0.9, 0.0)
    assert dict(lb.assign_attempts([still], samples, {})) == {}
    assert 'sample_vague' in dict(lb.CAUSES)


def test_spot_is_a_peak_not_a_ring(arena):
    from did.recorder import Recorder
    b = SampleBelief(arena, 3, 2.0)
    rec = Recorder()
    b.update(-2.0, -0.5, 0.45, 0.05)                           # одно показание: кольцо радиуса 1,1 м
    ring = (-2.0 + 1.1 * math.cos(1.0), -0.5 + 1.1 * math.sin(1.0))
    rec.add_belief(1.0, b)
    k = int(np.hypot(b.cx - ring[0], b.cy - ring[1]).argmin())
    b.p[k] = 0.9                                               # а теперь место выделено
    rec.add_belief(3.0, b)
    tr = {'belief': rec.belief, 'agent': {'config': {'candidate_mass': 0.35}}}
    assert lb._spot_times(tr, ring) == [3.0]
