"""Роли модели в расследовании: схемы, смысловая проверка, круг исправления, запасное правило, мусор."""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from did.llm import LLMError, LocalClient, llm_stats, load_system_prompt, make_client
from did.llm_mock import FAULTS, MockResponder
from did.llm_roles import (CRITIQUE_SCHEMA, EXAMPLE_CONTEXT, EXAMPLE_INQUIRY, EXPLANATION_SCHEMA, ISSUE_KINDS,
                           PROPOSAL_SCHEMA, Critique, Deliberation, Explanation, Proposal, criticize, deliberate,
                           discrimination_holes, explain, find_task, parse_critique, parse_explanation, parse_proposal,
                           probability_claims, propose, revise, rule_critique, rule_explanation, rule_proposal,
                           separated, separation, validate_critique, validate_explanation, validate_proposal)

ROOT = Path(__file__).resolve().parent.parent
TESTS = {t['id']: t for t in EXAMPLE_CONTEXT['tests']}

# Ответ автора в формате из задания.
GOOD = {
    'consider': ['soil', 'leak', 'turn'],
    'extra': [{'statement': 'колесо проскальзывает', 'why': 'расход вырос без смены грунта'}],
    'plan': [{'test': 'rest', 'distinguishes': ['leak', 'soil'], 'refutes_if': 'за 2 с заряд не убыл заметно — утечки нет'},
             {'test': 'spin', 'distinguishes': ['turn', 'soil'], 'refutes_if': 'около 0.12 ед/рад — повороты ни при чём'}],
    'rationale': 'Сначала дешёвый опыт rest (0.05 ед.), он отделяет утечку; потом spin разводит повороты и грунт.',
}
# Слабый план: два объяснения из четырёх и опыт spin, который их не различает.
WEAK = {
    'consider': ['soil', 'leak'], 'extra': [],
    'plan': [{'test': 'rest', 'distinguishes': ['leak', 'soil'], 'refutes_if': 'меньше 0.05 ед/с — утечки нет'},
             {'test': 'spin', 'distinguishes': ['soil', 'leak'], 'refutes_if': 'около 0.12 ед/рад — утечки нет'}],
    'rationale': 'Беру два самых вероятных объяснения.',
}


def context(**changes):
    ctx = copy.deepcopy(EXAMPLE_CONTEXT)
    ctx.update(changes)
    return ctx


def proposal(**changes):
    return {**copy.deepcopy(GOOD), **changes}


def step(test, *alts, refutes_if='результат около предсказанного числа опровергнет одно из объяснений'):
    return {'test': test, 'distinguishes': list(alts), 'refutes_if': refutes_if}


def issue(kind, target=None, text='Дыра в плане: числа не сходятся.', fix='Исправить план.'):
    return {'kind': kind, 'target': target, 'text': text, 'fix': fix}


def mock(seed=0, **options):
    responder = MockResponder(seed=seed, **options)
    return LocalClient(responder), responder


class Scripted:
    """Ответчик по списку: строка — ответ, исключение — сбой."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


# --- схемы -------------------------------------------------------------------------------------

def test_proposal_schema_accepts_task_format():
    p, errors = parse_proposal(json.dumps(GOOD, ensure_ascii=False), EXAMPLE_CONTEXT)
    assert errors == [] and p.consider == ['soil', 'leak', 'turn'] and [s.test for s in p.plan] == ['rest', 'spin']
    assert p.model_dump()['extra'] == [{'statement': 'колесо проскальзывает', 'why': 'расход вырос без смены грунта',
                                        'id': 'X1', 'modelled': False}]
    assert (p.source, p.error, p.exchanges, p.latency_ms) == ('llm', None, [], 0)
    assert set(p.to_dict()) == {'consider', 'extra', 'plan', 'rationale', 'source', 'error'}


def test_model_cannot_assign_probabilities_or_verdict():
    """Полей для вероятностей и вердикта в схеме нет: написанное моделью отбрасывается или отклоняется."""
    data = proposal(confidence=0.95, verdict='leak', posterior={'leak': 0.9}, best='leak',
                    extra=[{'statement': 'сел аккумулятор', 'why': 'заряд 31.5', 'modelled': True, 'prior': 0.5, 'id': 'leak'}])
    p, errors = parse_proposal(json.dumps(data, ensure_ascii=False), EXAMPLE_CONTEXT)
    assert errors == [] and set(p.model_dump()) == {'consider', 'extra', 'plan', 'rationale'}
    assert p.extra[0].model_dump() == {'statement': 'сел аккумулятор', 'why': 'заряд 31.5', 'id': 'X1', 'modelled': False}
    for field, text in [('rationale', 'Это утечка, уверенность 95%.'), ('rationale', 'С вероятностью 0.8 виноват грунт.'),
                        ('rationale', 'Грунт: 70% вероятности.'), ('rationale', 'Шансы утечки — 9 из 10, то есть 90%')]:
        p, errors = parse_proposal(json.dumps(proposal(**{field: text}), ensure_ascii=False), EXAMPLE_CONTEXT)
        assert p is None and len(errors) == 1 and 'не назначай вероятности' in errors[0], text
    quoting = 'У soil исходная вероятность 0.4, у leak — 0.3 (prior); расход вырос на 160%, опыт rest даёт 0.9 бит.'
    assert parse_proposal(json.dumps(proposal(rationale=quoting), ensure_ascii=False), EXAMPLE_CONTEXT)[1] == []


def test_probability_claims():
    assert probability_claims('уверенность 95%') == ['уверенность 95%']
    assert probability_claims('С вероятностью около 0,8 это грунт; шанс утечки 20 %.') == [
        'вероятностью около 0,8', 'шанс утечки 20 %']
    assert probability_claims('это утечка, 90% уверенности') == ['90% уверенности']
    assert probability_claims('вероятность 0.93, уверенность 93%', allowed=[0.93]) == []
    assert probability_claims('расход 0.14 ед/с, вырос на 160%, уверенно различает 0.15 и 0.01') == []
    assert probability_claims('') == [] and probability_claims(None) == []


SCHEMA_ERRORS = [
    (Proposal, proposal(consider=['soil']), 'consider: нужно не меньше 2 элементов'),
    (Proposal, proposal(consider='soil'), 'consider: нужен список'),
    (Proposal, {k: v for k, v in GOOD.items() if k != 'rationale'}, 'rationale: нет обязательного поля'),
    (Proposal, proposal(rationale='я' * 601), 'rationale: строка длиннее 600 символов'),
    (Proposal, proposal(plan=[{'test': 'rest', 'distinguishes': ['leak', 'soil']}]), 'plan[0].refutes_if: нет обязательного поля'),
    (Proposal, proposal(plan=[step('rest', 'leak')]), 'plan[0].distinguishes: нужно не меньше 2 элементов'),
    (Proposal, proposal(extra=[{'statement': 'a', 'why': 'b'}] * 3), 'extra: нужно не больше 2 элементов'),
    (Critique, {'verdict': 'maybe', 'issues': []}, "verdict: нужно одно из: 'accept' or 'revise'"),
    (Critique, {'issues': []}, 'verdict: нет обязательного поля'),
    (Critique, {'verdict': 'revise', 'issues': [issue('style')]}, 'issues[0].kind: нужно одно из'),
    (Critique, {'verdict': 'revise', 'issues': [{'kind': 'over_budget', 'text': 'дорого'}]}, 'issues[0].fix: нет обязательного поля'),
]


@pytest.mark.parametrize('model, data, expected', SCHEMA_ERRORS, ids=[e for *_, e in SCHEMA_ERRORS])
def test_schema_errors_are_readable(model, data, expected):
    text = json.dumps(data, ensure_ascii=False)
    out, errors = (parse_proposal(text, EXAMPLE_CONTEXT) if model is Proposal
                   else parse_critique(text, EXAMPLE_CONTEXT, GOOD))
    assert out is None and any(e.startswith(expected) for e in errors), errors


def test_schema_defaults():
    p = Proposal.model_validate({'consider': ['soil', 'leak'], 'plan': None, 'extra': None, 'rationale': ' ок '})
    assert (p.plan, p.extra, p.rationale) == ([], [], 'ок')
    c = Critique.model_validate({'verdict': 'accept', 'issues': None})
    assert c.issues == [] and c.source == 'llm'
    c = Critique.model_validate({'verdict': 'revise', 'issues': [{'kind': 'over_budget', 'text': 'a', 'fix': 'b', 'target': ' '}]})
    assert c.issues[0].target is None and ISSUE_KINDS[-1] == 'over_budget'


def strict(schema):
    if isinstance(schema, dict):
        if schema.get('type') == 'object':
            assert schema['additionalProperties'] is False and schema['required'] == list(schema['properties'])
        return all(strict(v) for v in schema.values())
    return all(strict(v) for v in schema) if isinstance(schema, list) else True


def test_json_schemas_are_strict_and_match_models():
    assert strict(PROPOSAL_SCHEMA) and strict(CRITIQUE_SCHEMA) and strict(EXPLANATION_SCHEMA)
    assert list(PROPOSAL_SCHEMA['properties']) == ['consider', 'extra', 'plan', 'rationale']
    kinds = CRITIQUE_SCHEMA['properties']['issues']['items']['properties']['kind']['enum']
    assert tuple(kinds) == ISSUE_KINDS
    for name in ('probability', 'confidence', 'posterior', 'best'):                  # вероятности — не дело модели
        assert name not in json.dumps([PROPOSAL_SCHEMA, CRITIQUE_SCHEMA, EXPLANATION_SCHEMA])


# --- промпты -----------------------------------------------------------------------------------

def test_prompts_examples_pass_their_own_checks():
    answer = load_system_prompt('inquiry_author').split('# Пример')[1].split('Ответ:')[1]
    p, errors = parse_proposal(answer, EXAMPLE_CONTEXT)
    assert errors == [] and p.consider == ['soil', 'leak', 'turn', 'other'] and len(p.rationale) <= 400
    assert [s.test for s in p.plan] == ['rest', 'spin', 'straight'] and rule_critique(EXAMPLE_CONTEXT, p).verdict == 'accept'
    answer = load_system_prompt('inquiry_critic').split('# Пример')[1].split('Ответ:')[1]
    c, errors = parse_critique(answer, EXAMPLE_CONTEXT, WEAK)
    assert errors == [] and c.verdict == 'revise' and [(i.kind, i.target) for i in c.issues] == [
        ('missing_alternative', 'turn'), ('non_discriminating', 'spin')]
    answer = load_system_prompt('inquiry_explain').split('# Пример')[1].split('Ответ:')[1]
    text, errors = parse_explanation(answer, EXAMPLE_INQUIRY)
    assert errors == [] and '0.14' in text


@pytest.mark.parametrize('name, keys', [
    ('inquiry_author', ['task', 'context.topic', 'context.anomaly', 'context.state', 'context.alternatives',
                        'context.tests', 'context.budget', 'proposal', 'critique', 'consider', 'plan', 'distinguishes',
                        'refutes_if', 'extra', 'rationale', 'gain_bits', 'predictions', 'max_tests']),
    ('inquiry_critic', ['context', 'proposal', 'verdict', 'issues', 'target', 'predictions', 'budget', *ISSUE_KINDS]),
    ('inquiry_explain', ['anomaly', 'alternatives', 'tests', 'conclusion', 'action', 'measured', 'predictions',
                         'identified', 'insufficient', 'confidence', 'posterior']),
])
def test_prompts_describe_every_field(name, keys):
    prompt = load_system_prompt(name)
    assert len(prompt.splitlines()) <= 70 and '# Запреты' in prompt and 'вероятности' in prompt
    for key in keys:
        assert f'`{key}`' in prompt, key


# --- проверка плана автора ---------------------------------------------------------------------

def test_validate_proposal_ok():
    assert validate_proposal(GOOD, EXAMPLE_CONTEXT) == []
    assert validate_proposal(Proposal.model_validate(GOOD), EXAMPLE_CONTEXT) == []
    full = proposal(consider=['soil', 'leak', 'turn', 'other'],
                    plan=[step('rest', 'leak', 'soil', 'turn'), step('spin', 'turn', 'soil'), step('straight', 'soil', 'turn')])
    assert validate_proposal(full, EXAMPLE_CONTEXT) == []                               # 1.15 ед. из 2.0, три опыта


PROPOSAL_ERRORS = [
    (proposal(consider=['soil', 'magnet', 'leak', 'turn']), 'consider[1]: объяснения "magnet" нет в alternatives (soil, leak, turn, other)'),
    (proposal(consider=['soil', 'leak', 'turn', 'leak']), 'consider[3]: повтор "leak"'),
    (proposal(consider=['soil', 'soil'], plan=[]), 'consider: нужно не меньше двух разных объяснений'),
    (proposal(plan=[step('teleport', 'leak', 'soil')]), 'plan[0].test: опыта "teleport" нет в tests (rest, straight, spin)'),
    (proposal(plan=[step('rest', 'leak', 'soil'), step('rest', 'leak', 'turn')]), 'plan[1].test: повтор опыта "rest"'),
    (proposal(plan=[step('rest', 'leak', 'other')], consider=['soil', 'leak', 'other']),
     'plan[0].distinguishes[1]: у "other" нет предсказания в опыте "rest"'),
    (proposal(plan=[step('rest', 'leak', 'other')]), 'plan[0].distinguishes[1]: "other" нет в consider'),
    (proposal(plan=[step('rest', 'leak', 'leak')]), 'plan[0].distinguishes: нужно не меньше двух разных объяснений'),
    (proposal(plan=[]), 'plan: пусто, хотя в tests есть опыт'),
]


@pytest.mark.parametrize('data, expected', PROPOSAL_ERRORS, ids=[e[:40] for _, e in PROPOSAL_ERRORS])
def test_validate_proposal_errors(data, expected):
    errors = validate_proposal(data, EXAMPLE_CONTEXT)
    assert any(e.startswith(expected) for e in errors), errors


def test_validate_proposal_budget():
    three = [step('rest', 'leak', 'soil'), step('straight', 'soil', 'turn'), step('spin', 'turn', 'soil')]
    assert validate_proposal(proposal(plan=three), EXAMPLE_CONTEXT) == []
    errors = validate_proposal(proposal(plan=three), context(budget={'energy': 1.0, 'max_tests': 2}))
    assert errors == ['plan: опытов 3, а budget.max_tests = 2',
                      'plan: опыты стоят 1.15 ед., а budget.energy = 1; убери или замени самый дорогой']
    assert validate_proposal(proposal(plan=[]), context(budget={'energy': 0.0, 'max_tests': 3})) == []   # опыты не по карману
    assert validate_proposal(proposal(plan=three), context(budget=None)) == []                           # бюджет не задан


def test_validate_proposal_reports_every_error():
    bad = proposal(consider=['soil', 'magnet'], plan=[step('teleport', 'soil', 'leak'), step('rest', 'soil', 'soil')],
                   rationale='Уверенность 99%.')
    assert len(validate_proposal(bad, EXAMPLE_CONTEXT)) == 6


def test_separation_by_predictions():
    assert separated(TESTS['rest'], 'leak', 'soil') and not separated(TESTS['rest'], 'soil', 'turn')
    assert not separated(TESTS['spin'], 'soil', 'leak')                 # 0.13 при сумме разбросов 0.13 — на границе
    assert separated(TESTS['spin'], 'turn', 'soil') and separated(TESTS['straight'], 'leak', 'turn')
    assert separation(TESTS['rest'], 'leak', 'other') is None and not separated(TESTS['rest'], 'leak', 'other')
    assert separation({'predictions': {'a': {'mean': 1, 'sigma': 0}, 'b': {'mean': 2, 'sigma': 0}}}, 'a', 'b') == float('inf')
    assert separation({'predictions': {'a': {'mean': 1}, 'b': {'mean': 'много'}}}, 'a', 'b') is None


# --- правило без модели ------------------------------------------------------------------------

def test_rule_proposal_orders_tests_by_gain_per_cost():
    p = rule_proposal(EXAMPLE_CONTEXT)
    assert p.consider == ['soil', 'leak', 'turn', 'other'] and p.extra == []
    assert [s.test for s in p.plan] == ['rest', 'spin', 'straight']                     # 18, 1.0 и 0.5 бит на единицу
    assert p.plan[0].distinguishes == ['soil', 'leak', 'turn'] and '0.15 ед/с' in p.plan[0].refutes_if
    assert validate_proposal(p, EXAMPLE_CONTEXT) == [] and rule_critique(EXAMPLE_CONTEXT, p).verdict == 'accept'
    assert 'Всего 1.15 ед. из 2' in p.rationale


def test_rule_proposal_respects_budget():
    tight = context(budget={'energy': 0.3, 'max_tests': 3})
    p = rule_proposal(tight)
    assert [s.test for s in p.plan] == ['rest'] and validate_proposal(p, tight) == []
    assert rule_critique(tight, p).verdict == 'accept'                  # soil и turn не развести: spin не по карману
    one = context(budget={'energy': 2.0, 'max_tests': 1})
    assert [s.test for s in rule_proposal(one).plan] == ['rest'] and rule_critique(one, rule_proposal(one)).verdict == 'accept'
    broke = context(budget={'energy': 0.0, 'max_tests': 3})
    p = rule_proposal(broke)
    assert p.plan == [] and validate_proposal(p, broke) == [] and 'ни один опыт' in p.rationale
    useless = context(tests=[{**TESTS['rest'], 'predictions': {a: {'mean': 0.01, 'sigma': 0.02} for a in ('soil', 'leak', 'turn')}}])
    assert rule_proposal(useless).plan == []                            # опыт ничего не различает


def test_rule_critique_finds_every_kind_of_hole():
    c = rule_critique(EXAMPLE_CONTEXT, WEAK)
    assert c.verdict == 'revise' and c.source == 'llm'
    found = [(i.kind, i.target) for i in c.issues]
    assert found == [('missing_alternative', 'turn'), ('non_discriminating', 'spin')]   # other опытами не проверяется
    assert validate_critique(c, EXAMPLE_CONTEXT, WEAK) == []
    uncovered = proposal(plan=[step('rest', 'leak', 'soil')])           # soil и turn планом не различаются
    c = rule_critique(EXAMPLE_CONTEXT, uncovered)
    assert [(i.kind, i.target) for i in c.issues] == [('non_discriminating', 'spin')] and 'Добавить опыт spin' in c.issues[0].fix
    costly = proposal(plan=[step('rest', 'leak', 'soil'), step('straight', 'soil', 'turn'), step('spin', 'turn', 'soil')])
    c = rule_critique(context(budget={'energy': 1.0, 'max_tests': 2}), costly)
    assert c.issues[0].kind == 'over_budget' and '1.15' in c.issues[0].text
    vague = proposal(plan=[step('rest', 'leak', 'soil', refutes_if='посмотрим'), step('spin', 'turn', 'soil')])
    assert [(i.kind, i.target) for i in rule_critique(EXAMPLE_CONTEXT, vague).issues] == [('no_refutation', 'rest')]
    assert rule_critique(EXAMPLE_CONTEXT, GOOD).verdict == 'accept'     # формат из задания: без other, два опыта


def test_discrimination_holes():
    assert discrimination_holes(EXAMPLE_CONTEXT, rule_proposal(EXAMPLE_CONTEXT)) == []
    holes = discrimination_holes(EXAMPLE_CONTEXT, WEAK)
    assert holes == [{'test': 'spin', 'pair': None, 'why': 'step'}]
    holes = discrimination_holes(context(budget={'energy': 0.1, 'max_tests': 3}), proposal(plan=[step('rest', 'leak', 'soil')]))
    assert holes == [{'test': 'spin', 'pair': ('soil', 'turn'), 'why': 'uncovered', 'affordable': False}]


# --- проверка критики --------------------------------------------------------------------------

def test_validate_critique_ok():
    assert validate_critique({'verdict': 'accept', 'issues': []}, EXAMPLE_CONTEXT, GOOD) == []
    good = {'verdict': 'revise', 'issues': [issue('missing_alternative', 'turn'), issue('non_discriminating', 'spin'),
                                            issue('no_refutation', 'rest'), issue('unmeasurable')]}
    assert validate_critique(good, EXAMPLE_CONTEXT, WEAK) == []
    assert validate_critique({'verdict': 'revise', 'issues': [issue('unmeasurable', 'X1')]}, EXAMPLE_CONTEXT,
                             Proposal.model_validate(GOOD)) == []       # замечание к объяснению автора не из списка


CRITIQUE_ERRORS = [
    (GOOD, {'verdict': 'accept', 'issues': [issue('no_refutation', 'rest')]}, 'verdict: при замечаниях нужен "revise"'),
    (GOOD, {'verdict': 'revise', 'issues': []}, 'issues: при "revise" нужно хотя бы одно замечание'),
    (GOOD, {'verdict': 'revise', 'issues': [issue('unmeasurable', 'teleport')]}, 'issues[0].target: "teleport" нет ни в alternatives, ни в tests'),
    (GOOD, {'verdict': 'revise', 'issues': [issue('missing_alternative', 'other')]},
     'issues[0]: все объяснения из alternatives, у которых есть predictions, уже есть в consider'),
    (WEAK, {'verdict': 'revise', 'issues': [issue('missing_alternative', 'leak')]},
     'issues[0].target: "leak" уже есть в consider; не рассмотрены: turn'),
    (GOOD, {'verdict': 'revise', 'issues': [issue('over_budget')]}, 'issues[0]: бюджет не превышен — опытов 2 из 3, они стоят 0.35 ед. из 2'),
    (proposal(consider=['soil', 'leak', 'turn', 'other'],
              plan=[step('rest', 'leak', 'soil'), step('spin', 'turn', 'soil'), step('straight', 'soil', 'turn')]),
     {'verdict': 'revise', 'issues': [issue('non_discriminating', 'rest')]}, 'issues[0]: по predictions каждый опыт плана различает'),
    (WEAK, {'verdict': 'revise', 'issues': [issue('missing_alternative', 'turn', text='Вероятность поворотов 60%, её забыли.')]},
     'issues[0].text: «Вероятность поворотов 60%» — не назначай вероятности'),
]


@pytest.mark.parametrize('prop, data, expected', CRITIQUE_ERRORS, ids=[e[:45] for *_, e in CRITIQUE_ERRORS])
def test_validate_critique_checks_claims_against_numbers(prop, data, expected):
    errors = validate_critique(data, EXAMPLE_CONTEXT, prop)
    assert len(errors) == 1 and errors[0].startswith(expected), errors


# --- вывод -------------------------------------------------------------------------------------

def inquiry(**changes):
    inq = copy.deepcopy(EXAMPLE_INQUIRY)
    inq.update(changes)
    return inq


UNSETTLED = inquiry(conclusion={'status': 'insufficient', 'best': 'soil', 'confidence': 0.55},
                    action='ехать дальше с запасом на оба объяснения')
TEXT = ('Робот тратил 6.8 ед/м вместо 2.6 и постоял 2 секунды: при утечке ожидалось 0.15 ед/с, при дорогом грунте 0.01, '
        'а вышло 0.14 ед/с. Значит, батарея теряет заряд сама по себе. Дальше он закладывает утечку в запас на возврат.')


def test_rule_explanation_cites_measurements():
    text = rule_explanation(EXAMPLE_INQUIRY)
    assert validate_explanation(text, EXAMPLE_INQUIRY) == []
    assert 'показал 0.14 ед/с' in text and '«батарея теряет заряд сама по себе (сбой)» — 0.15' in text
    assert 'По расчёту причина — «батарея теряет заряд сама по себе (сбой)» (вероятность 0.93)' in text
    assert text.endswith('Дальше: заложить утечку 0.14 ед/с в запас на возврат и ехать на базу раньше.')
    text = rule_explanation(UNSETTLED)
    assert validate_explanation(text, UNSETTLED) == [] and 'Данных не хватило, причина не установлена' in text
    plain = inquiry(tests=[{**t, 'measured': (t['measured'] or {}).get('value')} for t in EXAMPLE_INQUIRY['tests']])
    assert rule_explanation(plain) == rule_explanation(EXAMPLE_INQUIRY)             # измерение числом, а не словарём
    untested = inquiry(tests=[{**t, 'chosen': False, 'measured': None} for t in EXAMPLE_INQUIRY['tests']],
                       conclusion={'status': 'insufficient', 'best': 'soil', 'confidence': 0.4})
    assert 'Поставить опыты не удалось' in rule_explanation(untested) and validate_explanation(rule_explanation(untested), untested) == []


EXPLANATION_ERRORS = [
    (EXAMPLE_INQUIRY, 'Причина в утечке: вышло 0.14 ед/с при ожидании 0.15.', 'text: нужно от 2 до 4 предложений, а сейчас 1'),
    (EXAMPLE_INQUIRY, TEXT + ' Это первое. Это второе.', 'text: нужно от 2 до 4 предложений, а сейчас 5'),
    (EXAMPLE_INQUIRY, 'Робот постоял: при утечке ожидалось 0.15 ед/с. Батарея теряет заряд сама по себе.',
     'text: нет ни одного измеренного значения'),
    (EXAMPLE_INQUIRY, 'Робот постоял 2 секунды, вышло 0.14 ед/с. Значит, батарея теряет заряд сама.', 'text: не сказано, что ожидалось'),
    (EXAMPLE_INQUIRY, TEXT.replace('батарея теряет заряд сама по себе', 'верно объяснение leak по опыту rest'),
     'text: идентификаторы (leak, rest) читателю не понятны'),
    (EXAMPLE_INQUIRY, TEXT.replace('Значит,', 'С уверенностью 99%'), 'text: «уверенностью 99%» — не назначай вероятности'),
    (EXAMPLE_INQUIRY, TEXT.replace('Значит, батарея теряет заряд сама по себе.', 'Но причина не установлена.'),
     'text: conclusion.status = identified, а в тексте сказано, что причина не установлена'),
    (UNSETTLED, TEXT, 'text: conclusion.status = insufficient — скажи прямо'),
]


@pytest.mark.parametrize('inq, text, expected', EXPLANATION_ERRORS, ids=[e[:45] for *_, e in EXPLANATION_ERRORS])
def test_validate_explanation_errors(inq, text, expected):
    errors = validate_explanation(text, inq)
    assert len(errors) == 1 and errors[0].startswith(expected), errors


def test_validate_explanation_ok():
    assert validate_explanation(TEXT, EXAMPLE_INQUIRY) == []
    assert validate_explanation(TEXT.replace('Значит,', 'Расчёт даёт вероятность 93%:'), EXAMPLE_INQUIRY) == []   # как в conclusion
    assert validate_explanation(TEXT.replace('0.14', '0,14').replace('0.15', '0,15'), EXAMPLE_INQUIRY) == []
    assert parse_explanation(json.dumps({'text': f'  {TEXT}\n'}), EXAMPLE_INQUIRY) == (TEXT, [])
    assert parse_explanation(TEXT, EXAMPLE_INQUIRY) == (None, ['в ответе нет JSON-объекта'])
    assert parse_explanation('{"text": ""}', EXAMPLE_INQUIRY) == (None, ['text: пустая строка'])


# --- роли на имитаторе -------------------------------------------------------------------------

def test_mock_author_then_critic_then_revision():
    """Полный круг: слабый план автора → замечания критика → исправленный план. Не больше одного круга."""
    client, responder = mock()
    result = deliberate(client, EXAMPLE_CONTEXT)
    assert isinstance(result, Deliberation) and result.revised and result.open_issues == []
    assert result.draft.consider == ['soil', 'leak'] and [s.test for s in result.draft.plan] == ['rest', 'straight', 'spin']
    assert result.critique.verdict == 'revise' and [(i.kind, i.target) for i in result.critique.issues] == [
        ('missing_alternative', 'turn'), ('non_discriminating', 'spin')]
    assert result.proposal.consider == ['soil', 'leak', 'turn', 'other']
    assert [s.test for s in result.proposal.plan] == ['rest', 'spin', 'straight']
    assert validate_proposal(result.proposal, EXAMPLE_CONTEXT) == []
    assert [result.draft.source, result.critique.source, result.proposal.source] == ['llm'] * 3
    assert [(ex['role'], ex['task'], ex['ok']) for ex in result.exchanges] == [
        ('author', 'propose', True), ('critic', 'criticize', True), ('author', 'revise', True)]
    assert responder.stats == {'ok': 3} and result.latency_ms >= 0
    assert llm_stats(result.exchanges)['requests'] == 3
    json.dumps(result.to_dict(), ensure_ascii=False)                    # в запись прогона — как есть


def test_accepted_plan_is_not_revised():
    two = context(alternatives=EXAMPLE_CONTEXT['alternatives'][:2], tests=EXAMPLE_CONTEXT['tests'][:1])
    client, responder = mock()
    result = deliberate(client, two)
    assert result.critique.verdict == 'accept' and not result.revised and result.proposal is result.draft
    assert [ex['task'] for ex in result.exchanges] == ['propose', 'criticize'] and responder.stats == {'ok': 2}


def test_roles_send_prompt_and_one_json_input():
    responder = Scripted(GOOD, {'verdict': 'accept', 'issues': []})
    client = LocalClient(responder)
    p = propose(client, EXAMPLE_CONTEXT)
    criticize(client, EXAMPLE_CONTEXT, p)
    system, user = responder.seen[0]
    assert system == {'role': 'system', 'content': load_system_prompt('inquiry_author')}
    assert find_task(responder.seen[0]) == {'task': 'propose', 'context': EXAMPLE_CONTEXT}
    assert user['content'].startswith('Вход:\n{"task": "propose"') and user['content'].endswith('один JSON-объект.')
    assert responder.seen[1][0]['content'] == load_system_prompt('inquiry_critic')
    sent = find_task(responder.seen[1])
    assert sent['task'] == 'criticize' and sent['proposal'] == p.model_dump() and sent['context'] == EXAMPLE_CONTEXT
    assert find_task([{'role': 'user', 'content': 'Состояние:\n{"battery": 3}'}]) is None and find_task(None) is None


def test_revise_sends_critique_back_to_author():
    critique = rule_critique(EXAMPLE_CONTEXT, WEAK)
    responder = Scripted(rule_proposal(EXAMPLE_CONTEXT).model_dump())
    p = revise(LocalClient(responder), EXAMPLE_CONTEXT, WEAK, critique)
    sent = find_task(responder.seen[0])
    assert sent['task'] == 'revise' and sent['proposal'] == WEAK and sent['critique'] == critique.model_dump()
    assert p.source == 'llm' and p.exchanges[0]['task'] == 'revise' and len(p.plan) == 3


def test_one_repair_round_then_accept():
    bad = proposal(plan=[step('teleport', 'leak', 'soil')], rationale='Уверенность в утечке 95%.')
    responder = Scripted(bad, f'```json\n{json.dumps(GOOD, ensure_ascii=False)}\n```')
    p = propose(LocalClient(responder), EXAMPLE_CONTEXT)
    assert p.source == 'llm' and [ex['ok'] for ex in p.exchanges] == [False, True]
    first = p.exchanges[0]
    assert len(first['errors']) == 2 and 'teleport' in first['errors'][0] and 'не назначай вероятности' in first['errors'][1]
    repair = responder.seen[1]
    assert [m['role'] for m in repair] == ['system', 'user', 'assistant', 'user']
    assert repair[3]['content'].startswith('Ответ не принят') and 'верни план расследования заново' in repair[3]['content']
    assert all(e in repair[3]['content'] for e in first['errors'])


@pytest.mark.parametrize('fault', ['empty', 'malformed', 'invalid_target', 'out_of_arena'])
def test_every_mock_fault_is_repaired_in_every_role(fault):
    client, responder = mock(seed=3, faults={fault: 1.0})
    p = propose(client, EXAMPLE_CONTEXT)
    assert p.source == 'llm' and [ex['ok'] for ex in p.exchanges] == [False, True] and p.exchanges[0]['errors']
    assert validate_proposal(p, EXAMPLE_CONTEXT) == [] and len(p.consider) == 4
    c = criticize(client, EXAMPLE_CONTEXT, WEAK)
    assert c.source == 'llm' and [ex['ok'] for ex in c.exchanges] == [False, True] and c.verdict == 'revise'
    text = explain(client, EXAMPLE_INQUIRY)
    assert text.source == 'llm' and [ex['ok'] for ex in text.exchanges] == [False, True]
    assert text == rule_explanation(EXAMPLE_INQUIRY)
    assert responder.stats == {fault: 3, 'ok': 3}


def test_mock_faults_break_the_meaning_not_only_the_format():
    client, _ = mock(faults={'out_of_arena': 1.0})
    errors = propose(client, EXAMPLE_CONTEXT).exchanges[0]['errors']
    assert any('budget.max_tests' in e for e in errors) and any('budget.energy' in e for e in errors)
    assert any('не назначай вероятности' in e for e in errors)
    errors = criticize(client, EXAMPLE_CONTEXT, GOOD).exchanges[0]['errors']
    assert any('при замечаниях нужен "revise"' in e for e in errors) and any('бюджет не превышен' in e for e in errors)
    errors = explain(client, EXAMPLE_INQUIRY).exchanges[0]['errors']
    assert errors == ['text: нужно от 2 до 4 предложений, а сейчас 5', 'text: «Уверенность 99%» — не назначай вероятности, '
                      'их считает расчёт по измерениям']
    client, _ = mock(faults={'invalid_target': 1.0})
    assert 'teleport' in criticize(client, EXAMPLE_CONTEXT, GOOD).exchanges[0]['errors'][0]
    assert any('идентификаторы (leak, rest)' in e for e in explain(client, EXAMPLE_INQUIRY).exchanges[0]['errors'])


def test_wrapped_answer_is_read_at_once():
    client, _ = mock(faults={'wrapped': 1.0})
    assert len(propose(client, EXAMPLE_CONTEXT).exchanges) == 1 and len(explain(client, EXAMPLE_INQUIRY).exchanges) == 1


# --- запасной вариант --------------------------------------------------------------------------

def test_fallback_when_model_is_down():
    client, responder = mock(script=['http_500'] * 3)
    p = propose(client, EXAMPLE_CONTEXT)
    assert p.source == 'fallback' and p.error.startswith('модель недоступна')
    assert p.model_dump() == rule_proposal(EXAMPLE_CONTEXT).model_dump()
    assert len(p.exchanges) == 1 and not p.exchanges[0]['ok'] and p.exchanges[0]['task'] == 'propose'
    c = criticize(client, EXAMPLE_CONTEXT, WEAK)
    assert c.source == 'fallback' and c.verdict == 'revise'
    assert c.model_dump() == rule_critique(EXAMPLE_CONTEXT, WEAK).model_dump()
    text = explain(client, EXAMPLE_INQUIRY)
    assert isinstance(text, str) and text.source == 'fallback' and text == rule_explanation(EXAMPLE_INQUIRY)
    assert text.error.startswith('модель недоступна') and len(text.exchanges) == 1


def test_fallback_after_failed_repair():
    for max_repairs, calls in ((0, 1), (1, 2), (2, 3)):
        responder = Scripted(*['Не знаю.'] * 3)
        p = propose(LocalClient(responder), EXAMPLE_CONTEXT, max_repairs=max_repairs)
        assert p.source == 'fallback' and len(responder.seen) == len(p.exchanges) == calls
        assert f'план расследования не принят после {calls} попыток' in p.error and 'нет JSON-объекта' in p.error
        assert p.model_dump() == rule_proposal(EXAMPLE_CONTEXT).model_dump()
    c = criticize(LocalClient(Scripted(*[{'verdict': 'revise', 'issues': []}] * 2)), EXAMPLE_CONTEXT, GOOD)
    assert c.source == 'fallback' and 'ответ критика не принят после 2 попыток' in c.error
    text = explain(LocalClient(Scripted(*[{'text': 'Причина — leak.'}] * 2)), EXAMPLE_INQUIRY)
    assert text.source == 'fallback' and 'вывод не принят после 2 попыток' in text.error and '0.14' in text


def test_no_client_means_rule():
    p = propose(None, EXAMPLE_CONTEXT)
    assert (p.source, p.error, p.exchanges) == ('fallback', 'модель не подключена', [])
    assert criticize(None, EXAMPLE_CONTEXT, p).verdict == 'accept'
    assert explain(None, EXAMPLE_INQUIRY) == rule_explanation(EXAMPLE_INQUIRY)
    result = deliberate(None, EXAMPLE_CONTEXT)
    assert not result.revised and result.exchanges == [] and result.open_issues == []
    assert result.proposal.model_dump() == rule_proposal(EXAMPLE_CONTEXT).model_dump()


def test_deliberate_does_not_wait_for_dead_model_twice():
    client, responder = mock(faults={'http_500': 1.0})
    result = deliberate(client, EXAMPLE_CONTEXT)
    assert responder.stats == {'http_500': 1} and len(result.exchanges) == 1
    assert [result.draft.source, result.critique.source] == ['fallback', 'fallback'] and not result.revised


def test_deliberate_with_another_model_as_critic():
    author, a_stats = mock()
    critic = LocalClient(Scripted({'verdict': 'accept', 'issues': []}))
    result = deliberate(author, EXAMPLE_CONTEXT, critic_client=critic)
    assert not result.revised and a_stats.stats == {'ok': 1} and len(critic.responder.seen) == 1
    assert [(i.kind, i.target) for i in result.open_issues][:1] == [('missing_alternative', 'turn')]   # правило видит дыры


def test_explanation_is_a_plain_string_with_details():
    client, _ = mock()
    text = explain(client, EXAMPLE_INQUIRY)
    assert isinstance(text, Explanation) and isinstance(text, str) and json.loads(json.dumps(text)) == str(text)
    assert text.source == 'llm' and text.exchanges[0]['role'] == 'explainer' and text.exchanges[0]['task'] == 'explain'
    assert validate_explanation(text, EXAMPLE_INQUIRY) == []
    unsettled = explain(client, UNSETTLED)
    assert 'причина не установлена' in unsettled and validate_explanation(unsettled, UNSETTLED) == []


# --- устойчивость к мусору ---------------------------------------------------------------------

GARBAGE = [None, {}, [], 'строка', 42, {'alternatives': None, 'tests': 5}, {'alternatives': [None, 3, {}], 'tests': [{}]},
           {'alternatives': [{'id': 'a'}, {'id': 'b'}], 'tests': [{'id': 't', 'predictions': {'a': None, 'b': 'x'}}],
            'budget': 'много'},
           context(budget={'energy': float('nan'), 'max_tests': None}),
           context(tests=[{'id': 'rest', 'cost': 'дорого', 'gain_bits': None, 'predictions': {'soil': {'mean': 1}, 'leak': [1, 2]}}])]


@pytest.mark.parametrize('ctx', GARBAGE, ids=range(len(GARBAGE)))
def test_roles_never_raise_on_garbage(ctx):
    boom = LocalClient(lambda messages: (_ for _ in ()).throw(RuntimeError('ответчик упал')))
    for client in (None, mock()[0], mock(faults=dict.fromkeys(FAULTS, 0.5))[0], boom, object(), LocalClient(lambda m: 42)):
        p = propose(client, ctx)
        assert isinstance(p, Proposal) and p.source in ('llm', 'fallback')
        c = criticize(client, ctx, p)
        assert isinstance(c, Critique) and c.verdict in ('accept', 'revise')
        assert isinstance(revise(client, ctx, p, c), Proposal)
        assert isinstance(deliberate(client, ctx), Deliberation)
        assert isinstance(explain(client, ctx), str)
        assert isinstance(criticize(client, EXAMPLE_CONTEXT, ctx), Critique)       # мусор вместо плана автора
        json.dumps(deliberate(client, ctx).to_dict(), ensure_ascii=False, default=str)
    assert rule_explanation(ctx) and isinstance(validate_explanation(ctx, ctx), list)
    assert isinstance(validate_proposal(ctx, ctx), list) and isinstance(validate_critique(ctx, ctx, ctx), list)


def test_roles_accept_numpy_values():
    ctx = context(budget={'energy': np.float32(2.0), 'max_tests': np.int64(3)})
    ctx['tests'][0]['cost'] = np.float64(0.05)
    ctx['state']['on_known_ground'] = np.True_
    ctx['anomaly']['observed'] = np.array(6.8)
    p = propose(mock()[0], ctx)
    assert p.source == 'llm' and criticize(mock()[0], ctx, p).source == 'llm'
    inq = inquiry()
    inq['tests'][0]['measured']['value'] = np.float32(0.14)
    assert explain(mock()[0], inq).source == 'llm'


def random_context(rng):
    alts = ['soil', 'leak', 'turn', 'drift', 'other'][:rng.randrange(2, 6)]
    tests = [{'id': f't{i}', 'name': f'опыт {i}', 'cost': round(rng.uniform(0.05, 1.2), 2), 'duration_s': 2.0, 'unit': 'ед/с',
              'gain_bits': round(rng.uniform(0.0, 1.0), 2),
              'predictions': {a: {'mean': round(rng.uniform(0, 3), 2), 'sigma': round(rng.uniform(0.05, 0.6), 2)}
                              for a in alts if a != 'other' and rng.random() < 0.9}}
             for i in range(rng.randrange(0, 5))]
    weights = [rng.uniform(0.05, 1.0) for _ in alts]                      # разные prior: имитатор-автор отбросит малые
    return context(alternatives=[{'id': a, 'statement': f'объяснение {a}', 'prior': round(w / sum(weights), 2)}
                                 for a, w in zip(alts, weights)],
                   tests=tests, budget={'energy': round(rng.uniform(0.0, 2.5), 2), 'max_tests': rng.randrange(1, 4)})


def test_200_inquiries_under_faults(capsys):
    """Под 30% каждого сбоя роли не падают, а итоговый план всегда проходит проверку по контексту."""
    client, responder = mock(seed=1, faults=dict.fromkeys(FAULTS, 0.3))
    rng = random.Random(11)
    sources = {'llm': 0, 'fallback': 0}
    revised = 0
    for _ in range(200):
        ctx = random_context(rng)
        result = deliberate(client, ctx)
        assert validate_proposal(result.proposal, ctx) == [], (ctx, result.proposal)
        assert validate_critique(result.critique, ctx, result.draft) == []
        assert len(result.exchanges) <= 6                               # три роли, у каждой не больше одного исправления
        sources[result.proposal.source] += 1
        revised += result.revised
    with capsys.disabled():
        print(f"\n  расследования, 30% каждого сбоя: итоговый план от модели {sources['llm']}, по правилу "
              f"{sources['fallback']}, исправлялся {revised}; ответы {dict(responder.stats)}")
    assert sources['llm'] > 40 and sources['fallback'] > 40 and revised >= 10


# --- настоящие модели --------------------------------------------------------------------------

REAL = [k for k in os.environ.get('DID_LLM_REAL', '').replace(',', ' ').split() if k]


@pytest.mark.skipif(not REAL, reason='настоящие модели: DID_LLM_REAL=codex,ollama (тратит квоту подписки, ответы кэшируются)')
@pytest.mark.parametrize('kind', REAL or ['—'])
def test_roles_on_real_model(kind, capsys):
    """Каждая роль на настоящей модели; ответы — в runs/llm_real/roles/*.json как примеры для презентации."""
    try:
        client = make_client(kind, **({'cache': True} if kind in ('ollama', 'http') else {}))
    except LLMError as e:
        pytest.skip(str(e))
    name = str(getattr(client, 'model', kind)).replace(':', '-').replace('/', '-')
    result = deliberate(client, EXAMPLE_CONTEXT)
    text = explain(client, EXAMPLE_INQUIRY)
    folder = ROOT / 'runs' / 'llm_real' / 'roles'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}_deliberate.json').write_text(
        json.dumps({'context': EXAMPLE_CONTEXT, **result.to_dict()}, ensure_ascii=False, indent=1), encoding='utf-8')
    (folder / f'{name}_explain.json').write_text(
        json.dumps({'inquiry': EXAMPLE_INQUIRY, 'text': str(text), 'source': text.source, 'error': text.error,
                    'exchanges': text.exchanges, 'latency_ms': text.latency_ms}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    with capsys.disabled():
        print(f'\n  {name}: автор {result.draft.source}, критик {result.critique.source} ({result.critique.verdict}), '
              f'исправление {"было, " + result.proposal.source if result.revised else "не понадобилось"}, '
              f'вывод {text.source}; обменов {len(result.exchanges) + len(text.exchanges)}')
    assert validate_proposal(result.proposal, EXAMPLE_CONTEXT) == []
    assert validate_critique(result.critique, EXAMPLE_CONTEXT, result.draft) == []
    assert validate_explanation(text, EXAMPLE_INQUIRY) == []
    assert all(ex['response'] for ex in result.exchanges + list(text.exchanges)), 'модель не ответила'
