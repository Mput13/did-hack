"""Роли языковой модели в расследовании: автор плана опытов, критик и рассказчик вывода.

Расследование начинается, когда агент заметил странность: расход заряда не сходится с моделью,
датчик ведёт себя не так. Числа считает детерминированный модуль (did/science.py): предсказания
опытов при каждом объяснении, польза опыта, вероятности после измерений. Модель в них не
вмешивается — она выбирает из допустимого, формулирует и критикует:

  propose(client, context)                    -> Proposal   что рассматривать и какие опыты ставить
  criticize(client, context, proposal)        -> Critique   дыры в плане, вердикт accept | revise
  revise(client, context, proposal, critique) -> Proposal   автор исправляет по замечаниям
  deliberate(client, context)                 -> Deliberation   автор → критик → одно исправление
  explain(client, inquiry)                    -> str        вывод по завершённому расследованию

Вероятности и вердикт о причине модель не назначает: таких полей в схемах нет, а числа-вероятности
в тексте, которых не было во входе, — ошибка ответа. Каждый ответ проходит схему и смысловую
проверку (идентификаторы существуют, опыты из списка, бюджет, замечания критика сверяются с
числами); на ошибку модель исправляет ответ один раз, потом работает правило без модели
(rule_proposal, rule_critique, rule_explanation). Исключений функции не бросают; client=None —
сразу правило. У результата есть source ('llm' | 'fallback'), error, exchanges (для Recorder.add_llm)
и latency_ms.

Пример: python -m did.llm_roles --kind mock [--save runs/llm_real/roles]
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, StringConstraints, ValidationError, field_validator

from .llm import LLMError, _brief, _jsonable, _obj, _schema_errors, extract_json, load_system_prompt, request_json

ISSUE_KINDS = ('missing_alternative', 'non_discriminating', 'unmeasurable', 'no_refutation', 'over_budget')
TASKS = ('propose', 'revise', 'criticize', 'explain')
SEPARATION = 1.0     # предсказания различимы, если средние расходятся больше чем на SEPARATION × (σ₁ + σ₂)

# Контекст расследования, который агент передаёт автору и критику.
EXAMPLE_CONTEXT = {
    'topic': 'energy',
    'anomaly': {'text': 'расход вырос: 6,8 ед/м при прогнозе 2,6', 'x': 0.4, 'y': 1.1, 'observed': 6.8,
                'expected': 2.6, 'unit': 'ед/м'},
    'state': {'battery': 31.5, 'return_cost': 7.2, 'carried': 3, 'time_s': 41.2, 'recent_turning_rad': 5.1,
              'on_known_ground': True, 'last_penalty_s_ago': 6.0},
    'alternatives': [{'id': 'soil', 'statement': 'здесь дорогой грунт', 'prior': 0.4},
                     {'id': 'leak', 'statement': 'батарея теряет заряд сама по себе (сбой)', 'prior': 0.3},
                     {'id': 'turn', 'statement': 'заряд ушёл на повороты', 'prior': 0.2},
                     {'id': 'other', 'statement': 'причина не из этого списка', 'prior': 0.1}],
    'tests': [
        {'id': 'rest', 'name': 'постоять 2 секунды', 'cost': 0.05, 'duration_s': 2.0, 'unit': 'ед/с', 'gain_bits': 0.9,
         'predictions': {'soil': {'mean': 0.01, 'sigma': 0.02}, 'leak': {'mean': 0.15, 'sigma': 0.03},
                         'turn': {'mean': 0.01, 'sigma': 0.02}}},
        {'id': 'straight', 'name': 'проехать прямо 0,3 м', 'cost': 0.8, 'duration_s': 2.0, 'unit': 'ед/м',
         'gain_bits': 0.4,
         'predictions': {'soil': {'mean': 6.5, 'sigma': 0.8}, 'leak': {'mean': 3.9, 'sigma': 0.5},
                         'turn': {'mean': 2.5, 'sigma': 0.3}}},
        {'id': 'spin', 'name': 'развернуться на месте на 90°', 'cost': 0.3, 'duration_s': 1.5, 'unit': 'ед/рад',
         'gain_bits': 0.3,
         'predictions': {'soil': {'mean': 0.12, 'sigma': 0.05}, 'leak': {'mean': 0.25, 'sigma': 0.08},
                         'turn': {'mean': 0.5, 'sigma': 0.1}}}],
    'budget': {'energy': 2.0, 'max_tests': 3},
}

# Завершённое расследование для explain — как в did.science.Inquiry.to_dict(): те же объяснения и
# опыты плюс измерения (число или {'value', 'sigma', 't'}) и итог расчёта.
EXAMPLE_INQUIRY = {
    'topic': 'energy',
    'anomaly': EXAMPLE_CONTEXT['anomaly'],
    'alternatives': [{**a, 'posterior': p} for a, p in zip(EXAMPLE_CONTEXT['alternatives'], (0.03, 0.93, 0.02, 0.02))],
    'tests': [{**t, 'chosen': value is not None,
               'measured': None if value is None else {'value': value, 'sigma': sigma, 't': t_done}}
              for t, (value, sigma, t_done) in zip(EXAMPLE_CONTEXT['tests'],
                                                   ((0.14, 0.02, 43.4), (4.1, 0.4, 46.0), (None, None, None)))],
    'conclusion': {'status': 'identified', 'best': 'leak', 'confidence': 0.93,
                   'text': 'батарея теряет заряд сама по себе (сбой) — вероятность 93% после 2 опыт(ов)'},
    'action': 'заложить утечку 0.14 ед/с в запас на возврат и ехать на базу раньше',
}


# --- числа из контекста ------------------------------------------------------------------------

def _num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _items(context, name):
    """{id: запись} из списка context[name]; мусор пропускается."""
    items = context.get(name) if isinstance(context, dict) else None
    return {str(it['id']): it for it in (items if isinstance(items, list) else [])
            if isinstance(it, dict) and it.get('id') not in (None, '')}


def _testable(alts, tests):
    """Объяснения, у которых есть предсказание хотя бы в одном опыте. Остальные («причина не из
    списка») опытами не проверяются: расчёт держит их в уме сам, забыть их в плане нельзя."""
    return [a for a in alts if any(_prediction(t, a) is not None for t in tests.values())]


def _budget(context, tests):
    budget = context.get('budget') if isinstance(context, dict) else None
    budget = budget if isinstance(budget, dict) else {}
    return _num(budget.get('energy'), math.inf), int(_num(budget.get('max_tests'), len(tests)))


def _prediction(test, alt):
    """(mean, sigma) опыта при объяснении alt или None, если опыт его не проверяет."""
    p = (test.get('predictions') or {}).get(alt) if isinstance(test, dict) else None
    if not isinstance(p, dict) or _num(p.get('mean'), None) is None:
        return None
    return _num(p['mean']), abs(_num(p.get('sigma')))


def _measured(test):
    """Результат опыта числом или None: в записи расследования это число либо {'value', 'sigma', 't'}."""
    m = test.get('measured') if isinstance(test, dict) else None
    return _num(m.get('value') if isinstance(m, dict) else m, None)


def separation(test, a, b):
    """Во сколько раз разность предсказаний больше суммы разбросов; None — предсказаний нет."""
    pa, pb = _prediction(test, a), _prediction(test, b)
    if pa is None or pb is None:
        return None
    gap, spread = abs(pa[0] - pb[0]), pa[1] + pb[1]
    return gap / spread if spread > 0 else (math.inf if gap > 0 else 0.0)


def separated(test, a, b):
    z = separation(test, a, b)
    return z is not None and z > SEPARATION * (1 + 1e-9)


def _pairs(test, alts):
    """Пары объяснений, которые опыт различает: [(запас, a, b)], лучшая первой."""
    alts = list(alts)
    out = [(separation(test, a, b), a, b) for i, a in enumerate(alts) for b in alts[i + 1:] if separated(test, a, b)]
    return sorted(out, key=lambda p: -p[0])


def _plain(obj):
    """Proposal, Critique или словарь -> словарь."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj if isinstance(obj, dict) else {}


def _plan_cost(plan, tests):
    return sum(_num(tests[s['test']].get('cost')) for s in plan if s.get('test') in tests)


# --- схемы -------------------------------------------------------------------------------------

Id = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Line = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class _Result(BaseModel):
    """Ответ роли и сведения о том, как он получен (в JSON ответа они не входят)."""
    _source: str = PrivateAttr('llm')
    _error: Optional[str] = PrivateAttr(None)
    _exchanges: list = PrivateAttr(default_factory=list)
    _latency_ms: int = PrivateAttr(0)

    source = property(lambda self: self._source)             # 'llm' | 'fallback'
    error = property(lambda self: self._error)               # почему сработало правило
    exchanges = property(lambda self: self._exchanges)       # обмены с моделью для журнала
    latency_ms = property(lambda self: self._latency_ms)

    def _stamp(self, source, error=None, exchanges=(), latency_ms=0):
        self._source, self._error, self._exchanges, self._latency_ms = source, error, list(exchanges), latency_ms
        return self

    def to_dict(self):
        return {**self.model_dump(), 'source': self._source, 'error': self._error}


class Extra(BaseModel):
    """Объяснение автора не из списка. Предсказаний для него нет: расчёт его не проверяет."""
    statement: Line
    why: Line
    id: str = ''                             # X1, X2 — назначает программа
    modelled: bool = False                   # всегда False, что бы ни написала модель

    @field_validator('modelled', mode='before')
    @classmethod
    def _never_modelled(cls, value):
        return False


class Step(BaseModel):
    test: Id
    distinguishes: list[Id] = Field(min_length=2, max_length=8)
    refutes_if: Line


class Proposal(_Result):
    consider: list[Id] = Field(min_length=2, max_length=12)
    extra: list[Extra] = Field(default_factory=list, max_length=2)
    plan: list[Step] = Field(default_factory=list, max_length=8)
    rationale: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]

    @field_validator('extra', 'plan', mode='before')
    @classmethod
    def _null_is_empty(cls, value):
        return [] if value is None else value

    @field_validator('extra')
    @classmethod
    def _number_extras(cls, value):
        for i, item in enumerate(value):
            item.id = f'X{i + 1}'
        return value


class Issue(BaseModel):
    kind: Literal[ISSUE_KINDS]
    text: Line
    fix: Line
    target: Optional[Id] = None              # id объяснения или опыта, к которому относится замечание

    @field_validator('target', mode='before')
    @classmethod
    def _blank_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


class Critique(_Result):
    verdict: Literal['accept', 'revise']
    issues: list[Issue] = Field(default_factory=list, max_length=8)

    @field_validator('issues', mode='before')
    @classmethod
    def _null_is_empty(cls, value):
        return [] if value is None else value


class ExplanationText(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=900)]


class Explanation(str):
    """Вывод — обычная строка, а при ней source, error, exchanges, latency_ms."""
    source, error, exchanges, latency_ms = 'llm', None, (), 0

    def _stamp(self, source, error=None, exchanges=(), latency_ms=0):
        self.source, self.error, self.exchanges, self.latency_ms = source, error, list(exchanges), latency_ms
        return self


_STRINGS = {'type': 'array', 'items': {'type': 'string'}}
PROPOSAL_SCHEMA = _obj(
    consider=_STRINGS,
    extra={'type': 'array', 'items': _obj(statement={'type': 'string'}, why={'type': 'string'})},
    plan={'type': 'array', 'items': _obj(test={'type': 'string'}, distinguishes=_STRINGS,
                                         refutes_if={'type': 'string'})},
    rationale={'type': 'string'})
CRITIQUE_SCHEMA = _obj(
    verdict={'type': 'string', 'enum': ['accept', 'revise']},
    issues={'type': 'array', 'items': _obj(kind={'type': 'string', 'enum': list(ISSUE_KINDS)},
                                           target={'type': ['string', 'null']},
                                           text={'type': 'string'}, fix={'type': 'string'})})
EXPLANATION_SCHEMA = _obj(text={'type': 'string'})


# --- запрет на вероятности ---------------------------------------------------------------------

_PROB_WORD = r'(?:вероятност\w*|уверенност\w*|шанс\w*|уверен[аы]?\s+на|probability|confidence)'
_PROB = re.compile(rf'{_PROB_WORD}[^.;!?\d]{{0,30}}?(\d+(?:[.,]\d+)?)\s*(%?)|(\d+(?:[.,]\d+)?)\s*(%)\s*{_PROB_WORD}',
                   re.I)


def probability_claims(text, allowed=()):
    """Вероятности в тексте, которых не было во входе: «уверенность 95%», «с вероятностью 0.8».

    allowed — числа, которые посчитал расчёт (prior, posterior, confidence): их можно приводить.
    """
    out = []
    for m in _PROB.finditer(text or ''):
        raw, percent = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        value = float(raw.replace(',', '.'))
        value = value / 100 if percent or value > 1 else value
        if not any(abs(value - _num(a, -1.0)) <= 0.006 for a in allowed):
            out.append(' '.join(m.group(0).split()))
    return out


def _known_probabilities(*lists):
    return [it[key] for items in lists for it in items or [] if isinstance(it, dict)
            for key in ('prior', 'posterior') if it.get(key) is not None]


def _no_probabilities(texts, allowed):
    """Ошибки для текстов {путь: строка}, в которых модель назначила вероятность сама."""
    return [f'{where}: «{claim}» — не назначай вероятности, их считает расчёт по измерениям'
            for where, text in texts.items() for claim in probability_claims(text, allowed)]


# --- проверка плана расследования --------------------------------------------------------------

def validate_proposal(proposal, context):
    """Смысловая проверка плана автора по контексту: список ошибок, пустой — план годен."""
    if not isinstance(proposal, Proposal):
        try:
            proposal = Proposal.model_validate(proposal)
        except ValidationError as e:
            return _schema_errors(e, 'ответ')
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    energy, max_tests = _budget(context, tests)
    errors = []
    for i, a in enumerate(proposal.consider):
        if a not in alts:
            errors.append(f'consider[{i}]: объяснения "{a}" нет в alternatives ({", ".join(alts) or "список пуст"})')
        elif a in proposal.consider[:i]:
            errors.append(f'consider[{i}]: повтор "{a}"')
    if len(set(proposal.consider) & set(alts)) < 2:
        errors.append('consider: нужно не меньше двух разных объяснений из alternatives')
    seen = []
    for i, step in enumerate(proposal.plan):
        where = f'plan[{i}]'
        test = tests.get(step.test)
        if test is None:
            errors.append(f'{where}.test: опыта "{step.test}" нет в tests ({", ".join(tests) or "список пуст"})')
        elif step.test in seen:
            errors.append(f'{where}.test: повтор опыта "{step.test}"')
        seen.append(step.test)
        for j, a in enumerate(step.distinguishes):
            if a not in proposal.consider:
                errors.append(f'{where}.distinguishes[{j}]: "{a}" нет в consider')
            elif test is not None and a in alts and _prediction(test, a) is None:
                errors.append(f'{where}.distinguishes[{j}]: у "{a}" нет предсказания в опыте "{step.test}" — '
                              'этот опыт его не проверяет')
        if len(set(step.distinguishes)) < 2:
            errors.append(f'{where}.distinguishes: нужно не меньше двух разных объяснений')
    cost = _plan_cost(proposal.model_dump()['plan'], tests)
    if len(proposal.plan) > max_tests:
        errors.append(f'plan: опытов {len(proposal.plan)}, а budget.max_tests = {max_tests}')
    if cost > energy + 1e-9:
        errors.append(f'plan: опыты стоят {cost:g} ед., а budget.energy = {energy:g}; убери или замени самый дорогой')
    if not proposal.plan and any(_num(t.get('cost')) <= energy and _pairs(t, proposal.consider) for t in tests.values()):
        errors.append('plan: пусто, хотя в tests есть опыт, который различает объяснения и проходит по бюджету')
    texts = {'rationale': proposal.rationale}
    texts.update({f'plan[{i}].refutes_if': s.refutes_if for i, s in enumerate(proposal.plan)})
    texts.update({f'extra[{i}].why': x.why for i, x in enumerate(proposal.extra)})
    return errors + _no_probabilities(texts, _known_probabilities(alts.values()))


def parse_proposal(text, context):
    """Ответ автора -> (Proposal или None, список ошибок)."""
    try:
        proposal = Proposal.model_validate(extract_json(text))
    except LLMError as e:
        return None, [str(e)]
    except ValidationError as e:
        return None, _schema_errors(e, 'ответ')
    errors = validate_proposal(proposal, context)
    return (None, errors) if errors else (proposal, [])


def _refutes(test, pair, alts):
    """Условие опровержения по лучшей паре: какой результат какое объяснение исключит."""
    _, a, b = pair
    (ma, _), (mb, _) = _prediction(test, a), _prediction(test, b)
    unit = test.get('unit') or ''
    name = lambda x: (alts.get(x) or {}).get('statement') or x          # noqa: E731
    return _brief(f'около {ma:g} {unit} — отпадает «{name(b)}»; около {mb:g} {unit} — отпадает «{name(a)}»', 299)


def rule_proposal(context):
    """План без модели: все объяснения из списка, опыты по убыванию пользы на единицу заряда."""
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    energy, max_tests = _budget(context, tests)
    order = sorted(tests.values(), key=lambda t: -_num(t.get('gain_bits')) / max(_num(t.get('cost')), 1e-6))
    plan, spent, notes = [], 0.0, []
    for test in order:
        pairs, cost = _pairs(test, alts), _num(test.get('cost'))
        if not pairs or _num(test.get('gain_bits')) <= 0 or len(plan) >= max_tests or spent + cost > energy + 1e-9:
            continue
        ids = [a for a in alts if any(a in p[1:] for p in pairs)]
        plan.append({'test': str(test['id']), 'distinguishes': ids, 'refutes_if': _refutes(test, pairs[0], alts)})
        spent += cost
        notes.append(f"{test['id']} ({_num(test.get('gain_bits')):g} бит за {cost:g} ед.)")
    rationale = 'Правило без модели: все объяснения из списка, опыты по убыванию пользы на единицу заряда'
    if notes:
        total = f'Всего {spent:g} ед.' + (f' из {energy:g}.' if math.isfinite(energy) else '')
        rationale += f": {', '.join(notes)}. {total}"
    else:
        rationale += '; ни один опыт не различает объяснения в пределах бюджета.'
    data = {'consider': list(alts), 'extra': [], 'plan': plan, 'rationale': _brief(rationale, 599)}
    try:
        return Proposal.model_validate(data)
    except ValidationError:                  # в контексте меньше двух объяснений: расследовать нечего
        return Proposal.model_construct(consider=list(alts), extra=[], plan=[],
                                        rationale='В контексте меньше двух объяснений: сравнивать нечего.')


# --- проверка критики --------------------------------------------------------------------------

def discrimination_holes(context, proposal):
    """Где план не различает объяснения: [{'test', 'pair', 'why'}].

    why='step' — ни одна пара из distinguishes опыта не различима по predictions;
    why='uncovered' — пару из consider не различает ни один опыт плана, а опыт test из списка различает;
    affordable — проходит ли этот опыт по остатку бюджета.
    """
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    prop = _plain(proposal)
    consider = [a for a in prop.get('consider') or [] if a in alts]
    steps = [s for s in prop.get('plan') or [] if isinstance(s, dict) and s.get('test') in tests]
    holes = [{'test': s['test'], 'pair': None, 'why': 'step'} for s in steps
             if not _pairs(tests[s['test']], [a for a in s.get('distinguishes') or [] if a in alts])]
    used = [tests[s['test']] for s in steps]
    energy, max_tests = _budget(context, tests)
    left = energy - _plan_cost(steps, tests) if len(steps) < max_tests else -1.0
    for i, a in enumerate(consider):
        for b in consider[i + 1:]:
            if any(separated(t, a, b) for t in used):
                continue
            able = [t for t in tests.values() if separated(t, a, b)]
            if able:
                # Сначала тот, что проходит по остатку бюджета, из них — с наибольшей пользой на единицу заряда.
                best = max(able, key=lambda t: (_num(t.get('cost')) <= left + 1e-9,
                                                _num(t.get('gain_bits')) / max(_num(t.get('cost')), 1e-6)))
                holes.append({'test': str(best['id']), 'pair': (a, b), 'why': 'uncovered',
                              'affordable': _num(best.get('cost')) <= left + 1e-9})
    return holes


def validate_critique(critique, context, proposal):
    """Смысловая проверка критики: вердикт согласован с замечаниями, а замечания — с числами."""
    if not isinstance(critique, Critique):
        try:
            critique = Critique.model_validate(critique)
        except ValidationError as e:
            return _schema_errors(e, 'ответ')
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    energy, max_tests = _budget(context, tests)
    prop = _plain(proposal)
    consider, plan = prop.get('consider') or [], [s for s in prop.get('plan') or [] if isinstance(s, dict)]
    known = set(alts) | set(tests) | {x.get('id') for x in prop.get('extra') or [] if isinstance(x, dict)}
    errors = []
    if critique.verdict == 'accept' and critique.issues:
        errors.append('verdict: при замечаниях нужен "revise"; "accept" — только с пустым issues')
    if critique.verdict == 'revise' and not critique.issues:
        errors.append('issues: при "revise" нужно хотя бы одно замечание с kind, text и fix')
    for i, issue in enumerate(critique.issues):
        where = f'issues[{i}]'
        if issue.target is not None and issue.target not in known:
            errors.append(f'{where}.target: "{issue.target}" нет ни в alternatives, ни в tests')
        if issue.kind == 'missing_alternative':
            missing = [a for a in _testable(alts, tests) if a not in consider]
            if not missing:
                errors.append(f'{where}: все объяснения из alternatives, у которых есть predictions, уже есть '
                              'в consider — замечание неверно')
            elif issue.target in alts and issue.target not in missing:
                errors.append(f'{where}.target: "{issue.target}" уже есть в consider; не рассмотрены: {", ".join(missing)}')
        elif issue.kind == 'over_budget':
            cost = _plan_cost(plan, tests)
            if cost <= energy + 1e-9 and len(plan) <= max_tests:
                errors.append(f'{where}: бюджет не превышен — опытов {len(plan)} из {max_tests}, '
                              f'они стоят {cost:g} ед. из {energy:g}; замечание неверно')
        elif issue.kind == 'non_discriminating' and not discrimination_holes(context, prop):
            errors.append(f'{where}: по predictions каждый опыт плана различает свои объяснения, и каждая пара '
                          'из consider различима планом — замечание неверно')
    texts = {f'issues[{i}].{name}': getattr(issue, name) for i, issue in enumerate(critique.issues)
             for name in ('text', 'fix')}
    return errors + _no_probabilities(texts, _known_probabilities(alts.values()))


def parse_critique(text, context, proposal):
    """Ответ критика -> (Critique или None, список ошибок)."""
    try:
        critique = Critique.model_validate(extract_json(text))
    except LLMError as e:
        return None, [str(e)]
    except ValidationError as e:
        return None, _schema_errors(e, 'ответ')
    errors = validate_critique(critique, context, proposal)
    return (None, errors) if errors else (critique, [])


def rule_critique(context, proposal):
    """Критика без модели: то, что проверяется числами (бюджет, забытые объяснения, неразличимость)."""
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    energy, max_tests = _budget(context, tests)
    prop = _plain(proposal)
    consider, plan = prop.get('consider') or [], [s for s in prop.get('plan') or [] if isinstance(s, dict)]
    issues = []
    cost = _plan_cost(plan, tests)
    if cost > energy + 1e-9 or len(plan) > max_tests:
        issues.append({'kind': 'over_budget', 'target': None,
                       'text': f'Опытов {len(plan)} при пределе {max_tests}, они стоят {cost:g} ед. при бюджете {energy:g}.',
                       'fix': 'Убрать опыты с наименьшей пользой на единицу заряда, пока план не уложится в бюджет.'})
    for a in _testable(alts, tests):
        if a not in consider:
            issues.append({'kind': 'missing_alternative', 'target': a,
                           'text': f"Не рассмотрено «{alts[a].get('statement') or a}» (prior {_num(alts[a].get('prior')):g}), "
                                   'хотя оно есть в списке допустимых.',
                           'fix': f'Добавить {a} в consider.'})
    for hole in discrimination_holes(context, prop):
        test = tests[hole['test']]
        if hole['why'] == 'step':
            issues.append({'kind': 'non_discriminating', 'target': hole['test'],
                           'text': f"Опыт {hole['test']} не различает приписанные ему объяснения: их предсказания "
                                   'расходятся не больше суммы разбросов.',
                           'fix': f"Убрать {hole['test']} или назначить ему пару, которую он различает."})
        elif hole['affordable'] and hole['test'] not in [i['target'] for i in issues]:
            a, b = hole['pair']
            (ma, _), (mb, _) = _prediction(test, a), _prediction(test, b)
            issues.append({'kind': 'non_discriminating', 'target': hole['test'],
                           'text': f'Ни один опыт плана не различает {a} и {b}.',
                           'fix': f"Добавить опыт {hole['test']}: {ma:g} против {mb:g} {test.get('unit') or ''}."})
    for s in plan:
        if len(str(s.get('refutes_if') or '').strip()) < 15:
            issues.append({'kind': 'no_refutation', 'target': s.get('test') if s.get('test') in tests else None,
                           'text': f"У опыта {s.get('test')} нет условия опровержения.",
                           'fix': 'Написать, какой результат измерения какое объяснение исключит, с числом.'})
    issues = [Issue.model_validate({**it, 'text': _brief(it['text'], 299), 'fix': _brief(it['fix'], 299)})
              for it in issues[:8]]
    return Critique(verdict='revise' if issues else 'accept', issues=issues)


# --- проверка вывода ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'(?<=[.!?…])["»)]?\s+(?=[«"(]?[A-ZА-ЯЁ0-9])')
_NUMBER = re.compile(r'(?<![\w.,])\d+(?:[.,]\d+)?')
_INSUFFICIENT = ('не хватило', 'не хватает', 'недостаточно', 'не установлен', 'не удалось', 'неясн', 'не определ',
                 'не выяснен', 'нельзя')
_DENIES = ('причина не установлена', 'причину не удалось', 'данных не хватило', 'недостаточно данных')


def _cites(text, value):
    """Есть ли в тексте число value — с точкой или запятой, возможно, округлённое."""
    for raw in _NUMBER.findall(text):
        decimals = len(re.split('[.,]', raw)[1]) if re.search('[.,]', raw) else 0
        tolerance = max(0.5 * 10 ** -decimals if decimals else 0.0, 0.02 * abs(value))
        if abs(float(raw.replace(',', '.')) - value) <= tolerance + 1e-9:
            return True
    return False


def validate_explanation(text, inquiry):
    """Проверка вывода: объём, ссылки на измерения, согласие с итогом расчёта, без своих вероятностей."""
    inquiry = inquiry if isinstance(inquiry, dict) else {}
    alts, tests = _items(inquiry, 'alternatives'), _items(inquiry, 'tests')
    conclusion = inquiry.get('conclusion') if isinstance(inquiry.get('conclusion'), dict) else {}
    text = ' '.join(str(text or '').split())
    errors = []
    n = len([s for s in _SENTENCE_END.split(text) if s.strip()])
    if not 2 <= n <= 4:
        errors.append(f'text: нужно от 2 до 4 предложений, а сейчас {n}')
    if len(text) > 700:
        errors.append(f'text: нужно не больше 600 символов, а сейчас {len(text)}')
    done = [t for t in tests.values() if _measured(t) is not None]
    if done and not any(_cites(text, _measured(t)) for t in done):
        values = ', '.join(f"{t.get('name') or t['id']}: {_measured(t):g} {t.get('unit') or ''}".strip() for t in done)
        errors.append(f'text: нет ни одного измеренного значения ({values})')
    expected = [p[0] for t in done for a in alts if (p := _prediction(t, a)) is not None]
    if expected and not any(_cites(text, m) for m in expected):
        errors.append('text: не сказано, что ожидалось при разных объяснениях (числа mean из predictions)')
    low = text.lower()
    if conclusion.get('status') == 'insufficient' and not any(mark in low for mark in _INSUFFICIENT):
        errors.append('text: conclusion.status = insufficient — скажи прямо, что данных не хватило и причина не установлена')
    if conclusion.get('status') == 'identified' and any(mark in low for mark in _DENIES):
        errors.append('text: conclusion.status = identified, а в тексте сказано, что причина не установлена')
    ids = [i for i in list(alts) + list(tests) if i.isascii() and re.search(rf'(?<![\w-]){re.escape(i)}(?![\w-])', text)]
    if ids:
        errors.append(f'text: идентификаторы ({", ".join(ids)}) читателю не понятны — пиши словами из statement и name')
    allowed = _known_probabilities(alts.values()) + [conclusion.get('confidence')]
    return errors + _no_probabilities({'text': text}, allowed)


def parse_explanation(text, inquiry):
    """Ответ рассказчика -> (строка или None, список ошибок)."""
    try:
        out = ExplanationText.model_validate(extract_json(text)).text
    except LLMError as e:
        return None, [str(e)]
    except ValidationError as e:
        return None, _schema_errors(e, 'ответ')
    out = ' '.join(out.split())
    errors = validate_explanation(out, inquiry)
    return (None, errors) if errors else (out, [])


def rule_explanation(inquiry):
    """Вывод без модели — по шаблону: странность, опыты с ожиданиями и результатом, итог, действие."""
    inquiry = inquiry if isinstance(inquiry, dict) else {}
    alts, tests = _items(inquiry, 'alternatives'), _items(inquiry, 'tests')
    conclusion = inquiry.get('conclusion') if isinstance(inquiry.get('conclusion'), dict) else {}
    anomaly = inquiry.get('anomaly') if isinstance(inquiry.get('anomaly'), dict) else {}
    name = lambda a: (alts.get(a) or {}).get('statement') or str(a)       # noqa: E731
    parts = []
    best = conclusion.get('best')
    for t in [t for t in tests.values() if _measured(t) is not None][:2]:
        unit = t.get('unit') or ''
        # Ожидания — для итогового объяснения и того, от которого опыт отделяет его лучше всего.
        pairs = [p for p in _pairs(t, alts) if best in p[1:]] or _pairs(t, alts)
        shown = pairs[0][1:] if pairs else [a for a in alts if _prediction(t, a) is not None][:2]
        expected = '; '.join(f'«{name(a)}» — {_prediction(t, a)[0]:g}' for a in shown)
        parts.append(f"опыт «{t.get('name') or t['id']}» показал {_measured(t):g} {unit}"
                     + (f' (ожидалось: {expected})' if expected else ''))
    out = [f"Робот заметил странность: {str(anomaly.get('text') or 'расхождение с моделью').rstrip('.')}."]
    out.append(('Проверка: ' + ', '.join(parts) + '.') if parts else 'Поставить опыты не удалось.')
    confidence = _num(conclusion.get('confidence'), None)
    if conclusion.get('status') == 'identified' and best is not None:
        out.append(f'По расчёту причина — «{name(best)}»' + (f' (вероятность {confidence:g}).' if confidence is not None else '.'))
    else:
        rest = sorted(alts, key=lambda a: -_num(alts[a].get('posterior')))[:2]
        out.append('Данных не хватило, причина не установлена'
                   + (': остаются ' + ' и '.join(f'«{name(a)}»' for a in rest) + '.' if rest else '.'))
    if inquiry.get('action'):
        out.append(f"Дальше: {str(inquiry['action']).rstrip('.')}.")
    return ' '.join(out)


# --- вызовы ролей ------------------------------------------------------------------------------

@dataclass
class Deliberation:
    """Итог обсуждения плана: автор → критик → не больше одного исправления."""
    proposal: Proposal                       # итоговый план
    critique: Critique                       # замечания критика к первому плану
    draft: Proposal                          # первый план автора (тот же объект, если правок не было)
    revised: bool = False                    # автор исправлял план по замечаниям
    open_issues: list = field(default_factory=list)   # что в итоговом плане всё ещё не сходится с числами
    exchanges: list = field(default_factory=list)     # все обмены с моделью по порядку
    latency_ms: int = 0

    def to_dict(self):
        return {'proposal': self.proposal.to_dict(), 'critique': self.critique.to_dict(), 'draft': self.draft.to_dict(),
                'revised': self.revised, 'open_issues': [i.model_dump() for i in self.open_issues],
                'exchanges': self.exchanges, 'latency_ms': self.latency_ms}


def build_messages(prompt, task, ask, **parts):
    """Системный промпт роли и вход одним JSON-объектом {task, ...}."""
    payload = json.dumps({'task': task, **parts}, ensure_ascii=False, default=_jsonable)
    return [{'role': 'system', 'content': load_system_prompt(prompt)},
            {'role': 'user', 'content': f'Вход:\n{payload}\n{ask}: один JSON-объект.'}]


def _call(client, prompt, task, ask, parse, schema, rule, max_repairs, role, noun, **parts):
    """Ответ роли или правило: (результат, source, error, обмены, мс). Исключений не бросает."""
    if client is None:
        return rule(), 'fallback', 'модель не подключена', [], 0
    try:
        messages = build_messages(prompt, task, ask, **parts)
    except Exception as e:                   # noqa: BLE001 — вход не превращается в JSON
        return rule(), 'fallback', f'внутренняя ошибка: {type(e).__name__}: {_brief(e)}', [], 0
    out, error, exchanges, ms = request_json(client, messages, parse, schema, max_repairs, role=role, noun=noun)
    for ex in exchanges:
        ex['task'] = task
    if out is None:
        return rule(), 'fallback', error, exchanges, ms
    return out, 'llm', None, exchanges, ms


def _author(client, task, context, max_repairs, **parts):
    out, *meta = _call(client, 'inquiry_author', task, 'Верни план расследования',
                       lambda text: parse_proposal(text, context), PROPOSAL_SCHEMA, lambda: rule_proposal(context),
                       max_repairs, 'author', 'план расследования', context=context, **parts)
    return out._stamp(*meta)


def propose(client, context, max_repairs=1):
    """Автор: какие объяснения рассматривать и какие опыты в каком порядке ставить."""
    return _author(client, 'propose', context, max_repairs)


def revise(client, context, proposal, critique, max_repairs=1):
    """Автор исправляет свой план по замечаниям критика."""
    return _author(client, 'revise', context, max_repairs, proposal=_plain(proposal), critique=_plain(critique))


def criticize(client, context, proposal, max_repairs=1):
    """Критик: дыры в плане автора и вердикт accept | revise."""
    out, *meta = _call(client, 'inquiry_critic', 'criticize', 'Верни замечания к плану',
                       lambda text: parse_critique(text, context, proposal), CRITIQUE_SCHEMA,
                       lambda: rule_critique(context, proposal), max_repairs, 'critic', 'ответ критика',
                       context=context, proposal=_plain(proposal))
    return out._stamp(*meta)


def explain(client, inquiry, max_repairs=1):
    """Вывод по завершённому расследованию: 2–4 предложения со ссылками на измерения."""
    out, *meta = _call(client, 'inquiry_explain', 'explain', 'Верни вывод',
                       lambda text: parse_explanation(text, inquiry), EXPLANATION_SCHEMA,
                       lambda: rule_explanation(inquiry), max_repairs, 'explainer', 'вывод', inquiry=inquiry)
    return Explanation(out)._stamp(*meta)


def deliberate(client, context, max_repairs=1, critic_client=None):
    """Автор → критик → исправление (не больше одного круга). critic_client — другая модель для критика.

    Итоговый план бесплатно сверяется с числами правилом: что не сошлось — в open_issues,
    второго круга нет.
    """
    draft = propose(client, context, max_repairs)
    if draft.source == 'fallback' and str(draft.error).startswith('модель не'):
        client = None                        # модель недоступна: второй раз её не ждём, дальше правило
    critic = critic_client if critic_client is not None else client
    critique = criticize(critic, context, draft, max_repairs)
    final = revise(client, context, draft, critique, max_repairs) if critique.verdict == 'revise' else draft
    exchanges = draft.exchanges + critique.exchanges + (final.exchanges if final is not draft else [])
    ms = draft.latency_ms + critique.latency_ms + (final.latency_ms if final is not draft else 0)
    return Deliberation(final, critique, draft, final is not draft, rule_critique(context, final).issues, exchanges, ms)


# --- имитатор ролей ----------------------------------------------------------------------------

def find_task(messages):
    """Вход роли {task, ...} из первого сообщения пользователя; None — это не запрос роли."""
    for m in messages or []:
        if isinstance(m, dict) and m.get('role') == 'user':
            try:
                obj = extract_json(m.get('content'))
            except LLMError:
                return None
            return obj if obj.get('task') in TASKS else None
    return None


def _naive_proposal(context):
    """Первый план имитатора-автора — с дырами, которые должен найти критик: только самые вероятные
    объяснения и опыты в порядке списка, без проверки, различают ли они что-нибудь."""
    alts, tests = _items(context, 'alternatives'), _items(context, 'tests')
    energy, max_tests = _budget(context, tests)
    ranked = sorted(alts, key=lambda a: -_num(alts[a].get('prior')))
    consider = [a for a in alts if _num(alts[a].get('prior')) >= 0.25 or a in ranked[:2]]
    plan, spent = [], 0.0
    for test in tests.values():
        ids = [a for a in consider if _prediction(test, a) is not None]
        cost = _num(test.get('cost'))
        if len(ids) < 2 or len(plan) >= max_tests or spent + cost > energy + 1e-9:
            continue
        pairs = _pairs(test, ids) or [(0.0, ids[0], ids[1])]
        plan.append({'test': str(test['id']), 'distinguishes': ids, 'refutes_if': _refutes(test, pairs[0], alts)})
        spent += cost
    return {'consider': consider, 'extra': [], 'plan': plan,
            'rationale': f'Беру самые вероятные объяснения ({", ".join(consider)}) и опыты по порядку списка.'}


def mock_reply(kind, payload, repair=False):
    """Ответ имитатора на запрос роли. kind — 'ok' или сбой из did.llm_mock.FAULTS.

    invalid_target — несуществующий идентификатор; out_of_arena — нарушение по существу: план не по
    бюджету, критика против чисел, своя «уверенность» в выводе. На запрос исправления приходит
    ответ по правилу; первый план автора (task=propose) намеренно слабый — его правит критик.
    """
    task = payload.get('task')
    context, proposal = payload.get('context') or {}, payload.get('proposal') or {}
    if kind == 'empty':
        return ''
    if task == 'explain':
        text = rule_explanation(payload.get('inquiry'))
        if kind == 'invalid_target':
            text = 'Причина — leak. Это видно по опыту rest.'
        elif kind == 'out_of_arena':
            text += ' Уверенность 99%.'
        data = {'text': text}
    elif task == 'criticize':
        data = rule_critique(context, proposal).model_dump()
        if kind == 'invalid_target':
            data = {'verdict': 'revise', 'issues': [{'kind': 'unmeasurable', 'target': 'teleport',
                                                     'text': 'Опыт teleport ничего не измеряет.', 'fix': 'Убрать.'}]}
        elif kind == 'out_of_arena':
            data = {'verdict': 'accept', 'issues': [{'kind': 'over_budget', 'target': None,
                                                     'text': 'План слишком дорогой.', 'fix': 'Сократить.'}]}
    else:
        naive = task == 'propose' and not repair and kind in ('ok', 'wrapped', 'malformed')
        data = _naive_proposal(context) if naive else rule_proposal(context).model_dump()
        if kind == 'invalid_target':
            data['plan'] = [{'test': 'teleport', 'distinguishes': data['consider'][:2], 'refutes_if': 'робот окажется на базе'}]
        elif kind == 'out_of_arena':
            tests = _items(context, 'tests')
            data['plan'] = [{'test': t, 'distinguishes': data['consider'][:2], 'refutes_if': 'результат разойдётся с прогнозом'}
                            for t in list(tests) * 2]
            data['rationale'] += ' Уверенность в утечке 95%.'
    text = json.dumps(data, ensure_ascii=False)
    if kind == 'malformed':
        return text[:len(text) * 2 // 3]
    if kind == 'wrapped':
        return f'<think>\nСверю числа из входа.\n</think>\n\nВот ответ:\n\n```json\n{text}\n```\n'
    return text


# --- пример на любой модели --------------------------------------------------------------------

def main(argv=None):
    import argparse
    from pathlib import Path

    from .llm import CLIENT_KINDS, make_client
    ap = argparse.ArgumentParser(description='Роли расследования на примере: автор, критик, исправление, вывод.')
    ap.add_argument('--kind', default='mock', choices=CLIENT_KINDS)
    ap.add_argument('--model', help='модель вместо заданной по умолчанию')
    ap.add_argument('--context', help='файл контекста в JSON; по умолчанию пример из did.llm_roles')
    ap.add_argument('--inquiry', help='файл завершённого расследования в JSON; по умолчанию пример')
    ap.add_argument('--save', help='папка, куда записать ответы: <имя>_deliberate.json и <имя>_explain.json')
    ap.add_argument('--name', help='имя для файлов; по умолчанию — модель')
    args = ap.parse_args(argv)
    try:
        client = make_client(args.kind, **({'model': args.model} if args.model else {}))
    except LLMError as e:
        print(f'ошибка настроек: {e}')
        return 2
    load = lambda path, default: json.loads(Path(path).read_text(encoding='utf-8')) if path else default   # noqa: E731
    context, inquiry = load(args.context, EXAMPLE_CONTEXT), load(args.inquiry, EXAMPLE_INQUIRY)
    result = deliberate(client, context)
    text = explain(client, inquiry)
    out = {'deliberate': {'context': context, **result.to_dict()},
           'explain': {'inquiry': inquiry, 'text': str(text), 'source': text.source, 'error': text.error,
                       'exchanges': text.exchanges, 'latency_ms': text.latency_ms}}
    for ex in result.exchanges + list(text.exchanges):
        print(f"{ex['role']:9} {ex.get('task', ''):9} попытка {ex['attempt']}: ok={ex['ok']}, {ex['latency_ms']} мс, "
              f"ошибки: {ex['errors'] or 'нет'}")
    show = lambda obj: json.dumps(obj.model_dump(), ensure_ascii=False, indent=1)                          # noqa: E731
    print(f'\nпервый план ({result.draft.source}):\n{show(result.draft)}')
    print(f'\nкритика ({result.critique.source}):\n{show(result.critique)}')
    if result.revised:
        print(f'\nисправленный план ({result.proposal.source}):\n{show(result.proposal)}')
    print(f'\nне сошлось с числами в итоговом плане: {[i.model_dump() for i in result.open_issues] or "ничего"}')
    print(f'\nвывод ({text.source}):\n{text}')
    if args.save:
        name = (args.name or args.model or getattr(client, 'model', '') or args.kind).replace(':', '-').replace('/', '-')
        folder = Path(args.save)
        folder.mkdir(parents=True, exist_ok=True)
        for part, data in out.items():
            (folder / f'{name}_{part}.json').write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'\nзаписано: {folder}/{name}_deliberate.json, {folder}/{name}_explain.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
