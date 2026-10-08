"""Один прогон команды роботов в быстром симуляторе (исследование M1).

    ./px python -m did.team_runner --level hard --seed 3 --mode team

Режимы:
  solo — один робот, как в did.runner (точка отсчёта; тем же кодом, чтобы времена сборов считались одинаково);
  pair — два робота без координации: каждый — обычный агент, как будто он на арене один;
  pair_lidar — тоже без связи, но другой робот замечается лидаром и объезжается (помеха справа — уступить);
  team — два робота с координацией по каналу сообщений (did/team.py).
"""
import argparse
import json
import math
import time

from .agent import Agent, make_config
from .arena import load_arena
from .config import SCIENCE, Rules
from .fastsim_team import TeamSim
from .judge_team import ROBOT_NAMES, SLOTS, min_gap, team_score
from .metrics import run_metrics
from .planner import HeuristicPlanner
from .recorder import Recorder, save_trace
from .runner import RUNS
from .scenario import Scenario, generate
from .team import TeamAgent, TeamChannel, TeamConfig

MODES = {'solo': 1, 'pair': 2, 'pair_lidar': 2, 'team': 2}
# Без связи, но с объездом другого робота по лидару: всё, что делает TeamAgent по сообщениям, выключено.
NO_LINK = dict(share_readings=False, claims=False, share_collected=False, share_hazards=False, share_soil=False,
               sectors=False, avoid=False, lidar_avoid=True)
SAME_TARGET_M = 0.6        # первые цели в планах двух роботов ближе — цели совпадают


def run_team_episode(level, seed, mode='team', *, agent='adaptive', experiment='adhoc', arm=None, scenario=None,
                     scenario_args=None, config=None, rules=None, team=None, sim=None, save=True, quiet=True,
                     tolerate_errors=False):
    """Прогон целиком. Возвращает сводку того же вида, что did.runner.run_episode.

    tolerate_errors — исключение в агенте одного робота не обрывает прогон: робот останавливается и
    замолкает (как упавший процесс агента в Gazebo), второй продолжает, прогон идёт до общего срока.
    Упавшие роботы перечислены в метрике agent_errors. В опытах выключено: там ошибка должна быть видна.
    """
    arena = load_arena()
    if rules == 'science':
        rules = dict(SCIENCE)
    rules = Rules(**(rules or {}))
    if scenario is None:
        scenario = generate(level, seed, arena, **(scenario_args or {}))
    elif isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    n = MODES[mode]
    cfg = make_config(agent, **(config or {}))
    arm = arm or mode
    world = TeamSim(arena, scenario, rules, seed=seed, n_robots=n, **(sim or {}))
    channel = TeamChannel()
    team_cfg = TeamConfig(**(NO_LINK if mode == 'pair_lidar' else team or {}))
    bots, recs = [], []
    for i in range(n):
        rec = Recorder()
        common = dict(n_samples=len(scenario.samples), rules=rules, planner=HeuristicPlanner(), recorder=rec)
        if mode in ('team', 'pair_lidar'):
            bot = TeamAgent(arena, cfg, name=ROBOT_NAMES[i], home=SLOTS[i], team=team_cfg,
                            channel=channel if mode == 'team' else None, **common)
        else:
            bot = Agent(arena, cfg, **common)
            bot.base = tuple(SLOTS[i])
        bots.append(bot)
        recs.append(rec)

    wall = time.perf_counter()
    same_target_s = 0.0
    last = [False] * n                 # роботу уже показано последнее наблюдение
    failed = {}                        # номер робота -> текст исключения его агента

    def tick(i):
        bot, io = bots[i], world.robots[i]
        if not tolerate_errors:
            return bot.tick(io.observe(), io)
        try:
            bot.tick(io.observe(), io)
        except Exception as e:         # noqa: BLE001 — агент упал: робот встаёт, второй работает дальше
            last[i] = True
            failed[i] = f'{type(e).__name__}: {e}'
            io.command(0.0, 0.0)
            bot.journal.add(io.t, 'alarm', f'Агент остановлен из-за ошибки: {failed[i]}', tag='agent_error')

    while not world.done:
        for i, io in enumerate(world.robots):
            if not last[i]:
                last[i] = io.done      # закончивший робот получает итоговое наблюдение один раз
                tick(i)
        if n > 1 and not failed and _same_target(bots):
            same_target_s += world.dt
        world.advance()
    for i in range(n):
        if not last[i]:
            tick(i)
    wall = time.perf_counter() - wall

    score = world.score()
    per_robot = score.pop('per_robot')
    world_log = world.judges[0].world_log
    metrics = dict(score)
    metrics['battery_used'] = round(n * rules.battery_start - score['battery_left'], 2)
    metrics['reason'] = '+'.join(s['reason'] or '?' for s in per_robot)
    metrics['same_target_s'] = round(same_target_s, 1)
    metrics['messages'] = len(channel.log)
    metrics['messages_by_type'] = _count(m['type'] for m in channel.log)
    if failed:
        metrics['agent_errors'] = {ROBOT_NAMES[i]: text for i, text in sorted(failed.items())}
    if n == 1:
        # Один робот: те же метрики, что у обычного прогона (журнал, гипотезы, обнаружение событий).
        solo = run_metrics(world.judges[0].score(), rules, bots[0].journal, world_log, recs[0].plans, recs[0].llm)
        metrics = {**solo, **{k: v for k, v in metrics.items() if k not in solo}}
        metrics['returned'] = bool(per_robot[0]['returned'])
    run_id = f'{experiment}/{arm}/{scenario.name}'
    summary = {'id': run_id, 'experiment': experiment, 'arm': arm, 'agent': cfg.name, 'level': scenario.level,
               'seed': scenario.seed, 'backend': 'fastsim', 'metrics': metrics, 'wall_s': round(wall, 2)}
    if save:
        trace = build_team_trace(run_id=run_id, experiment=experiment, arm=arm, backend='fastsim', mode=mode,
                                 agent={'name': cfg.name, 'config': cfg.to_dict()}, scenario=scenario.to_dict(),
                                 rules=rules.to_dict(), result=metrics, world=world_log, recorders=recs,
                                 journals=[b.journal for b in bots], per_robot=per_robot, homes=SLOTS[:n],
                                 messages=channel.log, team_config=team_cfg.to_dict() if mode == 'team' else None)
        path = save_trace(trace, RUNS / experiment / arm / f'{scenario.name}.json.gz')
        summary['file'] = str(path.relative_to(RUNS))
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def build_team_trace(*, run_id, experiment, arm, backend, mode, agent, scenario, rules, result, world, recorders,
                     journals, per_robot, homes, messages, team_config=None):
    """Запись прогона команды. Верхний уровень — обычная запись первого робота (её читает прежний
    проигрыватель), сверх неё два поля:

      robots  [{name, home: [x, y], result, track, modes, events, journal, hypotheses, plans, paths, belief,
                soil, hazards, pose_fix?}] — то же, что в обычной записи, по каждому роботу (первый тоже
                здесь); pose_fix — поправка положения этого робота, есть только в записях с локализацией;
      team    {mode, config, messages: [{t, from, type, ...}]} — всё, что роботы сказали друг другу.
    """
    trace = recorders[0].build(run_id=run_id, experiment=experiment, arm=arm, backend=backend, agent=agent,
                               scenario=scenario, rules=rules, result=result, world=world, journal=journals[0])
    if len(recorders) == 1:
        return trace
    robots = []
    for i, (rec, journal) in enumerate(zip(recorders, journals)):
        robots.append({'name': per_robot[i]['name'], 'home': list(homes[i]), 'result': per_robot[i],
                       'track': rec.track, 'modes': rec.modes, 'events': rec.events, 'journal': journal.entries,
                       'hypotheses': journal.hypotheses, 'plans': rec.plans, 'paths': rec.paths,
                       'belief': rec.belief, 'soil': rec.soil, 'hazards': rec.hazards,
                       **({'pose_fix': rec.pose_fix} if rec.pose_fix else {})})
    trace['robots'] = robots
    trace['team'] = {'mode': mode, 'config': team_config, 'messages': messages}
    return trace


ROBOT_KEYS = ('track', 'modes', 'events', 'journal', 'hypotheses', 'plans', 'paths', 'belief', 'soil', 'hazards')
LAG_S = 0.45               # между точками пути больше — такт записи опоздал (см. tools/gazebo_batch.py)


def merge_parts(parts, mode, battery_start=None):
    """Склеить записи агентов, писавших каждый свою (Gazebo: по процессу на робота), в запись команды.

    parts — обычные записи с добавками: robot {name, home}, messages (что робот сказал и услышал) и
    judge — последний итог судьи на двоих, который этот агент успел получить ({t, robot_contacts,
    min_gap_m}; у записей, снятых до этого поля, его нет).

    Столкновения роботов друг с другом (robot_contacts) берутся только у судьи: он один видит обоих на общих
    часах. Нет его итога — показатель не измерен (None), а не ноль: из суммы столкновений роботов его не
    вывести, туда входят стены и контакты с уже закончившим роботом. Зазор min_gap_m — тоже от судьи; нет —
    по путям агентов, приведённым к общим временам (min_gap_track_m считается всегда). Итог судьи, снятый
    раньше конца прогона команды (judge_t меньше time), не используется вовсе: он измерил не весь прогон.
    """
    parts = [dict(p) for p in parts]
    trace = dict(parts[0])
    robots, scores = [], []
    for p in parts:
        s = p['result']
        scores.append(s)
        robots.append({'name': p['robot']['name'], 'home': p['robot']['home'], 'result': {'name': p['robot']['name'], **s},
                       **{k: p[k] for k in ROBOT_KEYS}, **{k: p[k] for k in ('pose_fix', 'scans') if p.get(k)}})
    times = [e['t'] for r in robots for e in r['events'] if e['type'] == 'sample_collected']
    total = team_score(scores, times, scores[0]['samples_total'])
    start = Rules().battery_start if battery_start is None else battery_start
    total['battery_used'] = round(len(parts) * start - total['battery_left'], 2)
    judged = [p['judge'] for p in parts if p.get('judge')]
    last = max(judged, key=lambda j: j.get('t', 0.0)) if judged else {}
    # Итог судьи годится, только если он снят не раньше конца прогона команды (после него роботы стоят).
    # Итог старше — судья замолчал, и конец прогона им не измерен: это не «ноль контактов».
    total['judge_t'] = last.get('t')
    covered = last.get('t') is not None and last['t'] >= total['time']
    if not covered:
        last = {}
    total['robot_contacts'] = last.get('robot_contacts')
    total['same_target_s'] = None                 # считается только в быстром симуляторе: там видны планы обоих
    track_gap = min_gap(robots[0]['track'], robots[1]['track']) if len(robots) > 1 else None
    total['min_gap_track_m'] = None if track_gap is None else round(track_gap, 3)
    if last.get('min_gap_m') is not None:
        total['min_gap_m'], total['min_gap_source'] = round(last['min_gap_m'], 3), 'judge'
    else:
        total['min_gap_m'], total['min_gap_source'] = total['min_gap_track_m'], 'tracks'
    seen, messages = set(), []
    for m in sorted((m for p in parts for m in p.pop('messages', [])), key=lambda m: m['t']):
        key = json.dumps(m, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            messages.append(m)
    total['messages'] = len(messages)
    # Доля тактов записи, пришедших с опозданием: признак перегруженной машины.
    lag = []
    for r in robots:
        t = r['track']['t']
        lag += [b - a > LAG_S for a, b in zip(t, t[1:])]
    total['lagged_share'] = round(sum(lag) / max(1, len(lag)), 3)
    for k in ('robot', 'messages', 'judge'):
        trace.pop(k, None)
    trace.update(result=total, robots=robots, team={'mode': mode, 'config': None, 'messages': messages})
    return trace


def _same_target(bots):
    """Первые цели в очередях планов двух роботов совпадают (ближе SAME_TARGET_M; возврат на базу не в счёт).

    Это совпадение целей в планах, а не обязательно одновременная езда: робот с такой целью может в этот
    такт стоять, уступать дорогу или мерить."""
    goals = []
    for b in bots:
        sg = b.queue[0] if b.queue and not b._returning and not b.finished else None
        if not sg or 'x' not in sg:
            return False
        goals.append((sg['x'], sg['y']))
    return math.dist(goals[0], goals[1]) < SAME_TARGET_M


def _count(items):
    out = {}
    for k in items:
        out[k] = out.get(k, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='hard', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--mode', default='team', choices=list(MODES))
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='adhoc', help='папка внутри runs/')
    ap.add_argument('--arm', default=None, help='подпапка записи; по умолчанию — режим')
    ap.add_argument('--rules', default=None, choices=['science'])
    args = ap.parse_args()
    run_team_episode(args.level, args.seed, args.mode, agent=args.agent, experiment=args.exp, arm=args.arm,
                     rules=args.rules, quiet=False, tolerate_errors=True)


if __name__ == '__main__':
    main()
