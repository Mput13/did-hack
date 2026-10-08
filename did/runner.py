"""Один прогон в быстром симуляторе: сценарий → агент → запись и метрики.

    pixi run python -m did.runner --level hard --seed 3 --agent adaptive
"""
import argparse
import json
import time
from dataclasses import replace

import numpy as np

from . import ROOT
from .agent import Agent, make_config
from .arena import load_arena
from .config import SCIENCE, Rules
from .fastsim import FastSim
from .metrics import SoilProbe, run_metrics, score_hypotheses, score_inquiries
from .oracle import ORACLES, make_truth
from .planner import HeuristicPlanner, LLMPlanner
from .recorder import Recorder, save_trace
from .scenario import Scenario, generate

RUNS = ROOT / 'runs'
# Мерило, а не участник: агент no_change, которому быстрый симулятор сообщает настоящую карту грунтов.
# В PRESETS его нет — в Gazebo и на роботе такой правды взять неоткуда.
ORACLE = 'soil_oracle'


def make_planner(cfg, llm=None, seed=0):
    """llm: None | {'kind': 'mock' | 'http' | 'ollama' | 'codex', ...опции did.llm.make_client}.

    Для mock — faults; http берёт адрес и ключ из окружения; 'prompt' — имя другого системного
    промпта из did/prompts (без .md), чтобы сравнивать версии; 'client' — готовый клиент с chat() вместо
    make_client (свой кэш, проверка повтора).
    """
    if cfg.planner != 'llm':
        return HeuristicPlanner()
    from .llm import load_system_prompt, make_client
    opts = dict(llm or {'kind': 'mock'})
    prompt = opts.pop('prompt', None)
    client = opts.pop('client', None)
    if client is None:
        if opts.get('kind', 'mock') == 'mock':
            opts.setdefault('seed', seed)
        client = make_client(**opts)
    return LLMPlanner(client, system_prompt=load_system_prompt(prompt) if prompt else None,
                      mission=getattr(cfg, 'mission', None), strategy=getattr(cfg, 'llm_strategy', 'single'))


def make_agent(name, config=None):
    """Агент по имени: вариант из PRESETS или базовая стратегия из did.baselines. Возвращает (класс, настройки)."""
    from .baselines import BASELINES
    if name == ORACLE:
        return Agent, replace(make_config('no_change'), name=ORACLE, **(config or {}))
    if name in ORACLES:                    # мерила A1: часть правды сценария отдельным каналом (did/oracle.py)
        base, own, _ = ORACLES[name]
        return Agent, replace(make_config(base), name=name, **{**own, **(config or {})})
    if name in BASELINES:
        cls, cfg = BASELINES[name]
        return cls, replace(cfg, **(config or {}))
    if name == 'study':                    # исследование по заданию пользователя (did/study.py)
        from .study_agent import STUDY_CONFIG, StudyAgent
        return StudyAgent, replace(STUDY_CONFIG, **(config or {}))
    return Agent, make_config(name, **(config or {}))


def _soil_truth(judge, arena):
    """Правда для оракула — только множитель расхода по клеткам арены на текущий момент.

    Ни будущих изменений, ни образцов, ни опасных зон через эту функцию не узнать. Карта пересчитывается,
    когда судья применил смену грунта; до тех пор возвращается тот же массив.
    """
    X, Y = arena.cell_centers()
    seen = {}

    def current():
        if seen.get('soils') is not judge.soils:
            grid = np.ones(X.shape)
            for z in judge.soils:
                grid = np.where(z.mask(X, Y), np.maximum(grid, z.mult), grid)
            seen.update(soils=judge.soils, grid=grid)
        return seen['grid']
    return current


def run_episode(level, seed, agent='adaptive', *, experiment='adhoc', arm=None, scenario=None,
                scenario_args=None, config=None, rules=None, agent_rules=None, llm=None, sim=None, save=True, quiet=True,
                knowledge=None, study=None, truth=False, soil_probe=False, roles=None):
    """Прогон целиком. Возвращает сводку: идентификаторы, метрики, путь к записи.

    study — задание исследования (словарь did.study.StudySpec) для агента 'study'.
    rules — правила мира; agent_rules=None — те же правила у агента (прежнее поведение),
    agent_rules={} — агент верит в Rules() независимо от правил мира.
    truth=True — писать в запись истинную позу робота на каждом шаге симулятора (поле truth).
    soil_probe — добавить в метрики разбор смены грунта по скрытой правде (did.metrics.SoilProbe).
    roles — модель в ролях автора, критика и рассказчика расследований отдельно от планировщика:
    опции did.llm.make_client или {'client': готовый клиент}. None — как раньше: роли ведёт модель
    планировщика, если он llm, иначе расследования идут без модели.
    """
    arena = load_arena()
    if rules == 'science':                 # набор правил с несколькими причинами расхода и сбоями
        rules = dict(SCIENCE)
    rules = Rules(**(rules or {}))
    if agent_rules is None:
        bot_rules = rules
    elif agent_rules == 'science':
        bot_rules = Rules(**SCIENCE)
    elif isinstance(agent_rules, Rules):
        bot_rules = agent_rules
    elif isinstance(agent_rules, dict):
        bot_rules = Rules(**agent_rules)
    else:
        raise ValueError(f"Unknown agent_rules: {agent_rules}")

    if scenario is None:
        scenario = generate(level, seed, arena, **(scenario_args or {}))
    elif isinstance(scenario, dict):
        scenario = Scenario.from_dict(scenario)
    cls, cfg = make_agent(agent, config)
    arm = arm or cfg.name
    world = FastSim(arena, scenario, rules, seed=seed, **(sim or {}))
    rec = Recorder()
    planner = make_planner(cfg, llm, seed)
    extra = {'knowledge': knowledge} if getattr(cfg, 'science', False) else {}
    if agent == 'study':                            # проверка задания и план; из сценария берётся только контур области
        from .study import prepare
        extra['study'] = prepare(study, arena, bot_rules, scenario)
    if extra and cfg.planner == 'llm':              # та же модель — автор и критик расследований
        extra['roles'] = planner.client
    if roles is not None and getattr(cfg, 'science', False):
        from .llm import make_client
        extra['roles'] = roles['client'] if 'client' in roles else make_client(**roles)
    if agent == ORACLE:
        extra['soil_truth'] = _soil_truth(world.judge, arena)
    if agent in ORACLES:
        extra['truth'] = make_truth(world.judge, scenario, arena, **ORACLES[agent][2])
    bot = cls(arena, cfg, n_samples=len(scenario.samples), rules=bot_rules, planner=planner, recorder=rec, **extra)

    probe = SoilProbe(scenario, rules) if soil_probe else None
    soil, detector = getattr(bot, 'soil', None), getattr(bot, 'change', None)
    known = (lambda x, y: soil.at(x, y)[1] >= detector.min_confidence) if soil and detector else None
    wall = time.perf_counter()
    if truth:
        rec.add_truth(world.dt, world.x, world.y)
    while not world.done:
        bot.tick(world.observe(), world)
        before = (world.x, world.y)
        world.advance()
        if truth:
            rec.add_truth(world.dt, world.x, world.y)
        if probe:
            probe.step(before, (world.x, world.y), world.judge.soils, world.t, known)
    bot.tick(world.observe(), world)       # последнее наблюдение: итоговые события попадают в запись
    wall = time.perf_counter() - wall

    judge = world.judge
    score = judge.score()
    metrics = run_metrics(score, rules, bot.journal, judge.world_log, rec.plans, rec.llm)
    if probe:
        metrics.update(probe.metrics(bot.journal))
    sh = score_hypotheses(bot.journal.hypotheses, scenario, judge.world_log, events=rec.events)
    metrics['hypotheses_truth'] = sh
    metrics['hyp_correct_share'] = sh['correct_share']
    metrics['hyp_confirmed_correct'] = sh['confirmed_correct']
    metrics['hyp_refuted_correct'] = sh['refuted_correct']
    metrics['hyp_soil_error'] = sh['soil_error']
    metrics['hyp_verified_share'] = sh['verified_share']
    science = bot.inv.export() if getattr(bot, 'inv', None) else {}
    report = None
    if agent == 'study':                            # отчёт исследования и сверка со скрытой правдой сценария
        from .study import add_truth, study_metrics
        report = add_truth(bot.study_report(judge.reason), scenario, rules, judge.world_log)
        metrics.update(study_metrics(report))
        science = {'study': report, **bot.export()}
    if science:
        iq = metrics['inquiries'] = score_inquiries(science['inquiries'], scenario, judge.world_log)
        metrics.update(inq_total=iq['total'], inq_tests=iq['tests'], inq_energy=iq['energy'],
                       inq_correct=(iq['correct'] + iq['partial']) / iq['identified'] if iq['identified'] else None,
                       inq_wrong=iq['wrong'], inq_insufficient=iq['insufficient'],
                       faults_found=iq['faults_found'] / iq['faults'] if iq['faults'] else None)
    if rec.llm:                            # доля годных ответов модели и фактическое время её ответов
        from .llm import llm_stats
        metrics['llm_stats'] = llm_stats(rec.llm)
    run_id = f'{experiment}/{arm}/{scenario.name}'
    summary = {'id': run_id, 'experiment': experiment, 'arm': arm, 'agent': cfg.name, 'level': scenario.level,
               'seed': scenario.seed, 'backend': 'fastsim', 'metrics': metrics, 'wall_s': round(wall, 2)}
    if bot_rules != rules:
        summary['agent_rules'] = bot_rules.to_dict()
    if report is not None:
        summary['study'] = report
    if save:
        trace = rec.build(run_id=run_id, experiment=experiment, arm=arm, backend='fastsim',
                          agent={'name': cfg.name, 'config': cfg.to_dict()}, scenario=scenario.to_dict(),
                          rules=rules.to_dict(), result={**score, **metrics}, world=judge.world_log,
                          journal=bot.journal)
        if bot_rules != rules:
            trace['agent_rules'] = bot_rules.to_dict()
        trace.update(science)
        path = save_trace(trace, RUNS / experiment / arm / f'{scenario.name}.json.gz')
        summary['file'] = str(path.relative_to(RUNS))
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    if knowledge is not None:
        summary['science'] = science          # для памяти между прогонами (did/memory.py)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='easy', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--agent', default='adaptive')
    ap.add_argument('--exp', default='adhoc', help='папка внутри runs/')
    ap.add_argument('--llm', default=None, choices=['mock', 'http', 'ollama', 'codex'],
                    help='для агентов с планировщиком llm: имитатор, сервер из .env, локальная Qwen, GPT по подписке')
    ap.add_argument('--llm-model', default=None, help='модель вместо заданной по умолчанию (ollama, codex, http)')
    ap.add_argument('--arm', default=None, help='подпапка записи; по умолчанию — имя агента')
    ap.add_argument('--rules', default=None, choices=['science'], help='science — правила с несколькими причинами расхода')
    args = ap.parse_args()
    llm = {'kind': args.llm, **({'model': args.llm_model} if args.llm_model else {})} if args.llm else None
    run_episode(args.level, args.seed, args.agent, experiment=args.exp, arm=args.arm, llm=llm, rules=args.rules,
                quiet=False)


if __name__ == '__main__':
    main()
