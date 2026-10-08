"""Один прогон в быстром симуляторе: сценарий → агент → запись и метрики.

    pixi run python -m did.runner --level hard --seed 3 --agent adaptive
"""
import argparse
import json
import time
from dataclasses import replace

from . import ROOT
from .agent import Agent, make_config
from .arena import load_arena
from .config import SCIENCE, Rules
from .fastsim import FastSim
from .metrics import run_metrics, score_inquiries
from .planner import HeuristicPlanner, LLMPlanner
from .recorder import Recorder, save_trace
from .scenario import Scenario, generate

RUNS = ROOT / 'runs'


def make_planner(cfg, llm=None, seed=0):
    """llm: None | {'kind': 'mock' | 'http' | 'ollama' | 'codex', ...опции did.llm.make_client}.

    Для mock — faults; http берёт адрес и ключ из окружения; 'prompt' — имя другого системного
    промпта из did/prompts (без .md), чтобы сравнивать версии.
    """
    if cfg.planner != 'llm':
        return HeuristicPlanner()
    from .llm import load_system_prompt, make_client
    opts = dict(llm or {'kind': 'mock'})
    prompt = opts.pop('prompt', None)
    if opts.get('kind', 'mock') == 'mock':
        opts.setdefault('seed', seed)
    return LLMPlanner(make_client(**opts), system_prompt=load_system_prompt(prompt) if prompt else None)


def make_agent(name, config=None):
    """Агент по имени: вариант из PRESETS или базовая стратегия из did.baselines. Возвращает (класс, настройки)."""
    from .baselines import BASELINES
    if name in BASELINES:
        cls, cfg = BASELINES[name]
        return cls, replace(cfg, **(config or {}))
    return Agent, make_config(name, **(config or {}))


def run_episode(level, seed, agent='adaptive', *, experiment='adhoc', arm=None, scenario=None,
                scenario_args=None, config=None, rules=None, llm=None, sim=None, save=True, quiet=True,
                knowledge=None):
    """Прогон целиком. Возвращает сводку: идентификаторы, метрики, путь к записи."""
    arena = load_arena()
    if rules == 'science':                 # набор правил с несколькими причинами расхода и сбоями
        rules = dict(SCIENCE)
    rules = Rules(**(rules or {}))
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
    if extra and cfg.planner == 'llm':              # та же модель — автор и критик расследований
        extra['roles'] = planner.client
    bot = cls(arena, cfg, n_samples=len(scenario.samples), rules=rules, planner=planner, recorder=rec, **extra)

    wall = time.perf_counter()
    while not world.done:
        bot.tick(world.observe(), world)
        world.advance()
    bot.tick(world.observe(), world)       # последнее наблюдение: итоговые события попадают в запись
    wall = time.perf_counter() - wall

    judge = world.judge
    score = judge.score()
    metrics = run_metrics(score, rules, bot.journal, judge.world_log, rec.plans, rec.llm)
    science = bot.inv.export() if getattr(bot, 'inv', None) else {}
    if science:
        iq = metrics['inquiries'] = score_inquiries(science['inquiries'], scenario, judge.world_log)
        metrics.update(inq_total=iq['total'], inq_tests=iq['tests'], inq_energy=iq['energy'],
                       inq_correct=iq['correct'] / iq['identified'] if iq['identified'] else None,
                       inq_wrong=iq['wrong'], inq_insufficient=iq['insufficient'],
                       faults_found=iq['faults_found'] / iq['faults'] if iq['faults'] else None)
    if rec.llm:                            # доля годных ответов модели и фактическое время её ответов
        from .llm import llm_stats
        metrics['llm_stats'] = llm_stats(rec.llm)
    run_id = f'{experiment}/{arm}/{scenario.name}'
    summary = {'id': run_id, 'experiment': experiment, 'arm': arm, 'agent': cfg.name, 'level': scenario.level,
               'seed': scenario.seed, 'backend': 'fastsim', 'metrics': metrics, 'wall_s': round(wall, 2)}
    if save:
        trace = rec.build(run_id=run_id, experiment=experiment, arm=arm, backend='fastsim',
                          agent={'name': cfg.name, 'config': cfg.to_dict()}, scenario=scenario.to_dict(),
                          rules=rules.to_dict(), result={**score, **metrics}, world=judge.world_log,
                          journal=bot.journal)
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
