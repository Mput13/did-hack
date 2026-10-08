"""L3: модель в ролях отдельно от планировщика, опыты в порядке плана модели, подсказка без формулы, счётные функции."""
import sys
from pathlib import Path

import pytest

from did.agent import make_config
from did.llm import load_system_prompt, make_client
from did.runner import run_episode
from did.science import Alternative, Inquiry
from did.science import TestOption as Option

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


def inquiry():
    alts = [Alternative('leak', 'утечка', 0.4), Alternative('soil', 'грунт', 0.5), Alternative('other', 'иное', 0.1)]
    tests = [Option('rest', 'постоять', cost=0.05, duration_s=2.0, unit='ед/с',
                        predictions={'leak': (0.15, 0.03), 'soil': (0.01, 0.012)}, sigma=0.015),
             Option('straight', 'проехать', cost=1.0, duration_s=2.0, unit='×',
                        predictions={'leak': (1.5, 0.35), 'soil': (2.5, 0.5)}, sigma=0.05)]
    return Inquiry('Q1', 0.0, 'energy', {'text': 'расход вырос'}, alts, tests)


def test_flags_are_off_by_default():
    for name in ('adaptive', 'scientist', 'scientist_llm'):
        assert make_config(name).inquiry_follow_plan is False
    q = inquiry()
    assert q.follow is None and q.llm is None and 'llm' not in q.to_dict()


def test_choose_by_gain_unless_plan_is_followed():
    q = inquiry()
    assert q.choose().id == 'rest'                      # расчёт: больше пользы на единицу заряда
    q.follow = ['straight', 'rest']
    assert q.choose().id == 'straight'                  # порядок автора, польза на выбор не влияет
    q.record('straight', 2.0, 0.05, 1.0)
    assert q.settled is False and q.choose().id == 'rest'
    q.record('rest', 0.01, 0.015, 2.0)
    assert q.choose() is None


def test_followed_plan_ends_when_exhausted_and_skips_unaffordable():
    q = inquiry()
    q.follow = ['straight']
    q.budget = 0.5                                      # straight стоит 1.0 — не по бюджету
    assert q.choose() is None
    q = inquiry()
    q.follow = []                                       # автор не предложил ни одного опыта
    assert q.choose() is None
    q = inquiry()
    q.follow = ['teleport', 'rest']                     # незнакомый опыт пропускается
    assert q.choose().id == 'rest'


def _run(**kw):
    return run_episode('hard', 3, 'scientist', rules='science', experiment='_test_l3', save=False,
                       knowledge={}, **kw)


def test_roles_run_with_rule_planner_and_are_recorded():
    base = _run()
    assert all('llm' not in q for q in base['science']['inquiries'])
    talk = _run(roles={'client': make_client('mock', seed=0)})
    talked = [q['llm'] for q in talk['science']['inquiries'] if 'draft' in (q.get('llm') or {})]
    assert talked, 'расследования должны обсуждаться с моделью'
    for info in talked:
        assert info['final']['source'] in ('llm', 'fallback') and info['critic']['verdict'] in ('accept', 'revise')
        assert set(info['final']['consider']) <= set(info['offered']) and info['follow'] is False
        assert info['explain']['source'] in ('llm', 'fallback')
    assert talk['metrics']['plans'] == base['metrics']['plans'] or talk['metrics']['score'] > 0   # планировщик — правило


def test_follow_plan_flag_reaches_the_inquiry():
    run = _run(roles={'client': make_client('mock', seed=0)}, config={'inquiry_follow_plan': True})
    talked = [q['llm'] for q in run['science']['inquiries'] if 'draft' in (q.get('llm') or {})]
    assert talked and all(info['follow'] is (info['final']['source'] == 'llm') for info in talked)
    for q in run['science']['inquiries']:
        info = q.get('llm') or {}
        if info.get('follow'):                          # проведены только опыты из плана автора
            done = [x['id'] for x in q['tests'] if x['measured'] and x['cost'] > 0]
            assert set(done) <= set(info['final']['plan'])


def test_goal_prompt_has_no_rule_formula():
    old, new = load_system_prompt(), load_system_prompt('planner_system_goal')
    assert 'confidence / (cost_to + 0.5)' in old and 'unseen_share / (cost_to + 1)' in old
    assert 'cost_to + 0.5' not in new and 'cost_to + 1' not in new and '# Как выбирать' not in new
    assert '# Цель' in new and '+10 за каждый собранный образец' in new and '# Примеры' not in new
    for part in ('# Состояние', '# Запреты', '# Формат ответа'):       # описание входа и формат — те же
        assert old[old.index(part):old.index(part) + 200] == new[new.index(part):new.index(part) + 200]


def test_share_and_paired_helpers():
    from l3_common import paired_binary, paired_diff, share, share_diff
    s = share(8, 10)
    assert s['share'] == 0.8 and s['ci'][0] < 0.8 < s['ci'][1] and share(0, 0)['share'] is None
    assert share(0, 12)['ci'] == [0.0, pytest.approx(0.243, abs=0.002)]
    d = share_diff(9, 10, 5, 10)
    assert d['diff'] == pytest.approx(0.4) and d['ci'][0] < 0.4 < d['ci'][1]
    p = paired_diff({'a': 2.0, 'b': 3.0, 'c': 1.0}, {'a': 1.0, 'b': 3.0, 'd': 0.0})
    assert p['n'] == 2 and p['mean'] == 0.5 and (p['a_higher'], p['ties']) == (1, 1)
    b = paired_binary({1: True, 2: True, 3: False, 4: None}, {1: False, 2: True, 3: False, 4: True})
    assert b['n'] == 3 and (b['only_a'], b['only_b']) == (1, 0) and b['diff'] == pytest.approx(0.333, abs=0.001)


def test_explanation_numbers_are_traced_to_input():
    from l3_explain import check_numbers
    from did.llm_roles import EXAMPLE_INQUIRY
    text = ('Расход вырос до 6,8 ед/м при прогнозе 2,6, то есть на 4,2. Пауза показала 0.14 ед/с, вероятность утечки '
            '93%. На самом деле расход 77.7.')
    out = check_numbers(text, EXAMPLE_INQUIRY)
    assert out['missing'] == ['77.7'] and out['derived'] == ['4,2']
    assert {'6,8', '2,6', '0.14', '93'} <= set(out['given'])


def test_peek_does_not_touch_random_choice_or_step_log():
    """Ревью: справочный выбор опыта не меняет генератор случайного выбора и запись шагов."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('r12_helpers', Path(__file__).with_name('test_r12.py'))
    r12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r12)
    for seed in range(1, 6):
        a, b = r12._inquiry('random', seed=seed), r12._inquiry('random', seed=seed)
        assert a.peek() is not None and a.steps == [] and a.stop is None
        assert a.choose().id == b.choose().id            # после справки случайный выбор тот же, что без неё
