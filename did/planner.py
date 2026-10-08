"""Верхний уровень агента: по состоянию выбрать подцели. Скоростью робота планировщик не управляет.

Три варианта с одним интерфейсом plan(state) -> dict:
  HeuristicPlanner — простое правило «выгода на единицу заряда»;
  LLMPlanner       — языковая модель; при любой ошибке модели подставляется правило;
  фиксированный маршрут (did.route) планировщика не вызывает вовсе.
Формат состояния и плана описан в did/prompts/planner_system.md.
"""
import math

MISSION = 'Собери как можно больше образцов, избегай дорогих зон, вернись на базу до разрядки.'


class HeuristicPlanner:
    source = 'heuristic'

    def plan(self, state):
        cands = [c for c in state['candidates'] if c['feasible']]
        points = [p for p in state['explore_points'] if p['feasible']]
        left = state['samples']['total'] - state['samples']['collected']
        if left <= 0:
            return self._result('Все образцы собраны — возвращаюсь на базу.', [{'type': 'return_base'}])
        if cands:
            best = max(cands, key=lambda c: c['confidence'] / (c['cost_to'] + 0.5))
            why = (f"Кандидат {best['id']} в ({best['x']:.1f}; {best['y']:.1f}): уверенность "
                   f"{best['confidence']:.0%}, дорога {best['cost_to']:.1f} ед., возврат оттуда "
                   f"{best['cost_back']:.1f} ед. при заряде {state['battery']:.1f}.")
            return self._result(why, [{'type': 'investigate', 'target': best['id']}])
        if points:
            # Если агент посчитал ожидаемую пользу измерений (gain_bits), выбор идёт по ней.
            best = max(points, key=lambda p: p.get('gain_bits', p['unseen_share']) / (p['cost_to'] + 1.0))
            what = (f"измерения по дороге и на месте уберут около {best['gain_bits']:.1f} бит неопределённости"
                    if 'gain_bits' in best else f"там не проверено {best['unseen_share']:.0%} окрестности")
            why = (f"Уверенных кандидатов нет. Еду на разведку в {best['id']} "
                   f"({best['x']:.1f}; {best['y']:.1f}): {what}, дорога {best['cost_to']:.1f} ед.")
            return self._result(why, [{'type': 'explore', 'target': best['id']}])
        return self._result('Достижимых целей с запасом на возврат не осталось — возвращаюсь на базу.',
                            [{'type': 'return_base'}])

    def _result(self, reasoning, subgoals):
        return {'reasoning': reasoning, 'hypotheses': [], 'subgoals': subgoals, 'source': self.source,
                'exchanges': [], 'error': None}


class LLMPlanner:
    source = 'llm'

    def __init__(self, client, fallback=None, system_prompt=None, max_repairs=1, strategy='single'):
        self.client = client
        self.fallback = fallback or HeuristicPlanner()
        self.system_prompt = system_prompt
        self.max_repairs = max_repairs
        from .llm_orchestration import STRATEGIES
        if strategy not in STRATEGIES:
            raise ValueError(f'неизвестный llm_strategy: {strategy}')
        self.strategy = strategy
        self.calls = self.failures = 0

    def plan(self, state):
        self.calls += 1
        if self.strategy != 'single':
            from .llm_orchestration import plan_orchestrated
            res = plan_orchestrated(self.strategy, self.client, state, fallback=self.fallback,
                                    system_prompt=self.system_prompt, max_repairs=self.max_repairs,
                                    mission=MISSION)
            if res.get('source') == 'fallback':
                self.failures += 1
            return res
        from .llm import request_plan
        res = request_plan(self.client, {'mission': MISSION, **state}, system_prompt=self.system_prompt,
                           max_repairs=self.max_repairs)
        if res.plan is None:
            self.failures += 1
            out = self.fallback.plan(state)
            out.update(source='fallback', exchanges=res.exchanges, error=res.error)
            out['reasoning'] = f'Модель не дала пригодного плана ({res.error}). Решение по правилу: ' + out['reasoning']
            return out
        plan = res.plan.model_dump(exclude_none=True)
        return {'reasoning': plan['reasoning'], 'hypotheses': plan.get('hypotheses', []),
                'subgoals': plan['subgoals'], 'source': 'llm', 'exchanges': res.exchanges, 'error': None}


def resolve_subgoals(subgoals, state):
    """Подставить координаты вместо идентификаторов C1/E2 и отбросить то, что исполнить нельзя."""
    by_id = {c['id']: c for c in state['candidates'] + state['explore_points']}
    out = []
    for sg in subgoals:
        kind = sg.get('type')
        if kind in ('investigate', 'explore'):
            target = by_id.get(sg.get('target'))
            if target is None or not target.get('feasible', True):
                continue
            out.append({'type': kind, 'target': target['id'], 'x': target['x'], 'y': target['y']})
        elif kind == 'goto' and math.isfinite(sg.get('x', math.nan)) and math.isfinite(sg.get('y', math.nan)):
            out.append({'type': 'goto', 'x': float(sg['x']), 'y': float(sg['y'])})
        elif kind == 'return_base':
            out.append({'type': 'return_base'})
            break
    return out
