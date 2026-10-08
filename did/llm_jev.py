"""Клиент модели решений Jev 1.13 (TypeSafe, «System One») через OpenRouter — исследование J1.

Jev — не чат-модель. Запрос: состояние (state — строка или JSON) и именованные вопросы трёх видов:
noul (да/нет → вероятность «да»), choice (один вариант из набора → вероятности вариантов и уверенность),
score (оценка по шкале). Текста модель не пишет. Адрес — POST https://openrouter.ai/api/v1/systemone
(«System One API» OpenRouter, формат тот же, что у TypeSafe: model, state, questions → model, answers, usage).

Модель платная, поэтому клиент жёстко ограничен:
  1. идентификатор модели — только MODEL; любой другой (в том числе маршрутизаторы и псевдонимы) отвергается
     до кэша и до сети;
  2. журнал расходов один на все рабочие деревья и лежит вне репозитория: ~/.did/jev_budget.json (другой путь —
     переменная DID_JEV_BUDGET_FILE). Пока журнала нет, платный вызов запрещён: его создаёт человек командой
     --init-budget, называя уже потраченную сумму. Перед КАЖДОЙ попыткой отправки клиент под блокировкой
     резервирует в журнале наибольшую возможную стоимость запроса (max_cost_usd); если «учтено + неизвестно +
     зарезервировано» с новым резервом превысило бы BUDGET_STOP_USD, запрос не отправляется. После ответа резерв
     заменяется стоимостью из ответа; попытка, которая могла дойти до сервера, а стоимости не сообщила (таймаут,
     обрыв, ответ без поля cost), остаётся в журнале по оценке сверху в отдельном поле unknown_cost_usd.
     Журнал перед резервом проверяется по всей схеме (_check_journal): неполный, с неверным типом или
     отрицательным числом — отказ до отправки. Ошибки файловой системы при работе с журналом — тоже ошибка
     клиента (JevJournalError), а не сбой процесса.
     ПОТОЛОК УСЛОВНЫЙ: он держится, пока цена ответа не больше резерва. Резерв — оценка сверху, а не обещание
     сервера: ответ дороже резерва оплачен раньше, чем клиент об этом узнал. Такой ответ клиент не отдаёт
     (JevCostOverrun) и останавливает работу отметкой halted, а между порогом остановки и потолком оставлен
     запас, но денег это не возвращает. Строгий потолок даёт только предел оплаты на ключ у поставщика;
  3. в ответе проверяется, какая модель ответила; не Jev 1.13 — ошибка и отметка halted в журнале: все
     следующие обращения отвергаются, пока человек не разберётся;
  4. ключ читается из файла только на время одного запроса и только после удачного резерва (запрос, которому
     журнал отказал, ключа не читает); в объект клиента, кэш, журнал, записи прогонов и сообщения об ошибках он
     не попадает. Ответ сервера (и успешный тоже) очищается от ключа, пока ключ ещё доступен, и наружу отдаётся
     только проверенная схема: известные поля известных типов. Любое исключение транспорта заменяется ошибкой
     клиента с вымаранным текстом и без цепочки исключений (__cause__ и __context__ пусты). Адрес сервера задан
     константой и из окружения не берётся.

Ответы лежат в runs/_jev_cache/<хэш>.json (ключ — хэш от модели, состояния и вопросов). strict=True — режим
«только кэш»: сеть, ключ и журнал расходов не трогаются, запроса нет в кэше — CacheMiss. Переменная
DID_JEV_CACHE_ONLY=1 делает такими все клиенты процесса.

    ./px python -m did.llm_jev                                               # журнал расходов и размер кэша
    ./px python -m did.llm_jev --init-budget --spent 0.226192974 --calls 3964   # создать журнал (один раз)
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import ROOT
from .llm import CacheMiss, LLMError

MODEL = 'typesafe/jev-1.13'                        # единственная разрешённая модель
ENDPOINT = 'https://openrouter.ai/api/v1/systemone'
# Где лежит ключ, знает только окружение: путь к файлу строк KEY=VALUE задаётся переменной DID_JEV_KEY_FILE.
# В репозитории нет ни ключа, ни пути к нему.
KEY_FILE_ENV = 'DID_JEV_KEY_FILE'
KEY_NAME = 'OPENROUTER_API_KEY'
PRICE_PER_TOKEN = 0.042e-6                         # доллара за входной токен; выходные бесплатны
BUDGET_CAP_USD = 1.00                              # потолок владельца проекта
# «Учтено + зарезервировано» выше этой суммы не поднимается. Запас до потолка — 0,40: резерв есть оценка сверху,
# и ответ дороже резерва оплачивается раньше, чем клиент это увидит. На 3 963 записанных ответах настоящая цена
# не превышала 14 % резерва; запас покрывает ошибку оценки, но не заменяет лимит на ключ у поставщика.
BUDGET_STOP_USD = 0.60
BUDGET_FILE_ENV = 'DID_JEV_BUDGET_FILE'
BUDGET_DEFAULT = ('.did', 'jev_budget.json')       # относительно домашнего каталога
CACHE_ONLY_ENV = 'DID_JEV_CACHE_ONLY'
# Оценка стоимости запроса сверху: токен не короче байта, значит токенов не больше, чем байтов в теле запроса,
# плюс служебная обвязка сервера (пустой запрос с одним вопросом — 283 токена); цена — с двойным запасом.
RESERVE_OVERHEAD_TOKENS = 1024
RESERVE_PRICE_FACTOR = 2.0
CACHE_DIR = ROOT / 'runs' / '_jev_cache'
QUESTION_TYPES = ('noul', 'choice', 'score')
IDENT_CHARS = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/:-*')
# Закрытый список причин отказа (JevError.code): по нему сторож пишет причину отключения, не трогая текст ошибки.
REASON_CODES = ('network', 'timeout', 'server', 'budget', 'journal', 'wrong_model', 'bad_reply', 'key', 'other')
JOURNAL_COUNTERS = ('calls', 'failed_calls', 'input_tokens', 'output_tokens', 'cost_from_response', 'unknown_attempts')


class JevError(LLMError):
    """Сбой обращения к Jev: сеть, сервер, формат ответа. code — причина из REASON_CODES."""
    code = 'other'

    def __init__(self, message, attempts=0, code=None):
        super().__init__(message, attempts)
        if code is not None:
            self.code = code


class JevModelRefused(JevError):
    """Запрошена не та модель: запрос не отправлен."""


class JevBudgetExceeded(JevError):
    """Потолок расходов достигнут, журнала расходов нет или он остановлен отметкой halted: запрос не отправлен."""
    code = 'budget'


class JevJournalError(JevBudgetExceeded):
    """Журнала расходов нет, он не по схеме или файловая система отказала. До резерва — запрос не отправлен;
    после ответа — резерв остаётся в журнале (занятым), ответ наружу не отдаётся."""
    code = 'journal'


class JevCostOverrun(JevError):
    """Ответ стоил больше резерва: он оплачен, но не возвращается; работа с платной моделью остановлена."""
    code = 'budget'


class JevWrongModel(JevError):
    """Ответила не Jev 1.13: работа с платной моделью остановлена."""
    code = 'wrong_model'


@dataclass
class JevReply:
    answers: dict                # {имя вопроса: {'type': 'noul', 'noul': 0.93} | {... 'choice', 'probabilities', 'confidence'}}
    latency_ms: int              # время настоящего вызова (у ответа из кэша — того, первого)
    usage: dict                  # input_tokens, output_tokens
    cost_usd: float              # сколько записано в журнал за удачную попытку
    model: str                   # какая модель ответила (по ответу сервера)
    cached: bool = False
    cost_known: bool = True      # False — стоимости в ответе не было, в журнале оценка сверху


def key_file_path():
    return os.environ.get(KEY_FILE_ENV) or None


def budget_path():
    """Журнал расходов: DID_JEV_BUDGET_FILE или ~/.did/jev_budget.json — один на все рабочие деревья."""
    return Path(os.environ.get(BUDGET_FILE_ENV) or Path.home().joinpath(*BUDGET_DEFAULT))


def _read_key(path):
    """Ключ из файла строк KEY=VALUE. Вызывается только на время одного запроса."""
    if not path:
        raise JevError(f'ключ не задан: укажите путь к файлу с {KEY_NAME} в переменной {KEY_FILE_ENV}', code='key')
    lines = None
    try:
        lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    except (OSError, ValueError):                      # ValueError — не UTF-8: в исключении лежит содержимое файла
        pass
    if lines is None:                                  # вне обработчика: исходное исключение в цепочку не попадает
        raise JevError(f'файл с ключом, заданный в {KEY_FILE_ENV}, не читается', code='key')
    for line in lines:
        name, _, value = line.strip().partition('=')
        if name.strip().removeprefix('export ').strip() == KEY_NAME:
            value = value.strip().strip('"\'')
            if value:
                return value
    raise JevError(f'в файле, заданном в {KEY_FILE_ENV}, нет строки {KEY_NAME}=…', code='key')


def _scrub(text, key):
    text = str(text)
    return text.replace(key, '***') if key else text


def _short(text, key, limit=200):
    """Описание для диагностики: сначала вымарывается ключ, потом текст обрезается (не наоборот: обрезка могла
    бы оставить кусок ключа, который вымарывание уже не узнает)."""
    return _scrub(text, key)[:limit]


def _ident(value, key):
    """Идентификатор модели или поставщика из ответа: без ключа, только буквы, цифры и . _ / : -, до 80 знаков."""
    if not isinstance(value, str):
        return ''
    return ''.join(ch if ch in IDENT_CHARS else '?' for ch in _scrub(value, key))[:80]


def _number(value, lo=None, hi=None):
    """Конечное число (не bool) в пределах [lo, hi] или None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return None
    return float(value)


def _count(value):
    n = _number(value, 0)
    return int(n) if n is not None else 0


def _clean_answer(kind, answer, key=None):
    """Ответ на один вопрос по известной схеме или None. Неизвестные поля отбрасываются, строки очищаются от ключа."""
    if not isinstance(answer, dict) or answer.get('type') != kind:
        return None
    out = {'type': kind}
    if kind == 'noul':
        out['noul'] = _number(answer.get('noul'), 0.0, 1.0)
        return out if out['noul'] is not None else None
    if kind == 'choice':
        if not isinstance(answer.get('choice'), str):
            return None
        out['choice'] = _short(answer['choice'], key)
        probs = answer.get('probabilities')
        if isinstance(probs, dict):
            out['probabilities'] = {_short(k, key): _number(v, 0.0, 1.0) for k, v in probs.items()
                                    if isinstance(k, str) and _number(v, 0.0, 1.0) is not None}
    else:
        out['score'] = _number(answer.get('score'))
        if out['score'] is None:
            return None
    conf = _number(answer.get('confidence'), 0.0, 1.0)
    if conf is not None:
        out['confidence'] = conf
    return out


def _clean_answers(questions, answers, key=None):
    """Ответы на все вопросы запроса по проверенной схеме или None, если чего-то нет или что-то не по схеме."""
    if not isinstance(answers, dict) or set(answers) != set(questions):
        return None
    out = {name: _clean_answer(q['type'], answers[name], key) for name, q in questions.items()}
    return None if any(a is None for a in out.values()) else out


def _clean_response(data, questions, key):
    """Из разобранного ответа сервера — только известные поля известных типов, без ключа."""
    usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
    return {'model': _ident(data.get('model'), key),
            'provider': None if data.get('provider') is None else _ident(data.get('provider'), key) or '?',
            'input_tokens': _count(usage.get('input_tokens')), 'output_tokens': _count(usage.get('output_tokens')),
            'cost': _number(usage.get('cost'), 0.0),
            'answers': _clean_answers(questions, data.get('answers'), key)}


def max_cost_usd(body):
    """Наибольшая возможная стоимость одной попытки отправить body (байты) — см. RESERVE_*."""
    return round((len(body) + RESERVE_OVERHEAD_TOKENS) * PRICE_PER_TOKEN * RESERVE_PRICE_FACTOR, 9)


def _check_journal(d):
    """Журнал по всей схеме, которой пользуется клиент: поля, типы, неотрицательные числа, модель, потолок."""
    def whole(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0
    if not isinstance(d, dict) or d.get('model') != MODEL:
        return False
    if not isinstance(d.get('initialized'), str) or not d['initialized']:
        return False
    cap, stop = _number(d.get('cap_usd'), 0.0, BUDGET_CAP_USD), _number(d.get('stop_usd'), 0.0, BUDGET_CAP_USD)
    if cap is None or stop is None or cap <= 0 or stop > cap:
        return False
    if any(_number(d.get(k), 0.0) is None for k in ('cost_usd', 'unknown_cost_usd')):
        return False
    if not all(whole(d.get(k)) for k in JOURNAL_COUNTERS):
        return False
    if 'halted' not in d or not (d['halted'] is None or isinstance(d['halted'], str)):
        return False
    reserved = d.get('reserved')
    return isinstance(reserved, dict) and all(
        isinstance(rid, str) and isinstance(r, dict) and _number(r.get('usd'), 0.0) is not None
        for rid, r in reserved.items())


class Budget:
    """Журнал расходов на Jev: один файл на все рабочие деревья, процессы и потоки.

    Изменения идут под блокировкой отдельного файла <журнал>.lock и записываются заменой файла (временный файл,
    fsync, rename): сбой посреди записи оставляет прежний журнал целым. Чтение журнал не меняет.
    Потрачено = cost_usd (стоимость из ответов сервера) + unknown_cost_usd (оценка сверху для попыток, стоимость
    которых неизвестна); занято = потрачено + reserved (резервы запросов, которые сейчас в пути).
    """

    def __init__(self, path=None, stop_usd=BUDGET_STOP_USD):
        self.path = Path(path) if path else budget_path()
        self.stop_usd = min(float(stop_usd), BUDGET_CAP_USD)

    @staticmethod
    def spent(d):
        return round(float(d['cost_usd']) + float(d['unknown_cost_usd']), 9)

    @staticmethod
    def committed(d):
        return round(Budget.spent(d) + sum(float(r['usd']) for r in d['reserved'].values()), 9)

    def _not_initialized(self):
        return JevJournalError(
            f'журнала расходов на Jev нет ({self.path}): платный вызов запрещён. Журнал создаёт человек, называя уже '
            f'потраченную сумму: python -m did.llm_jev --init-budget --spent <долларов> --calls <обращений>')

    def _load(self):
        """Журнал как словарь; None — файла нет. Нечитаемый, неполный или чужой файл — отказ: это не «нулевой
        расход». Схема проверяется целиком здесь, то есть до резерва и до отправки."""
        try:
            raw = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            raise JevJournalError(f'журнал расходов {self.path} не читается: платный вызов запрещён') from None
        try:
            d = json.loads(raw)
        except (ValueError, RecursionError):
            d = None
        if not _check_journal(d):
            raise JevJournalError(f'журнал расходов {self.path} пуст, неполон или испорчен: платный вызов '
                                  f'запрещён, исправьте файл вручную')
        return d

    def _write(self, d):
        d['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False, indent=1))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        dir_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _lock(self):
        fd = os.open(self.path.with_name(self.path.name + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _update(self, change):
        """Изменить журнал под блокировкой. Отказ файловой системы (нет прав, диск, блокировка) — JevJournalError:
        журнал при этом остаётся прежним, то есть несостоявшийся резерв не занят, а неснятый — занят."""
        try:
            if not self.path.is_file():
                raise self._not_initialized()          # ни каталога, ни файла блокировки не создаём
            fd = self._lock()
            try:
                d = self._load()
                if d is None:
                    raise self._not_initialized()
                out = change(d)
                self._write(d)
                return out
            finally:
                os.close(fd)
        except OSError as e:
            failed = type(e).__name__
        raise JevJournalError(f'журнал расходов {self.path} недоступен ({failed}): платный вызов запрещён')

    def read(self):
        """Журнал (словарь) или None, если его нет. Ничего не пишет."""
        return self._load()

    def init(self, spent_usd=0.0, calls=0, input_tokens=0, output_tokens=0, failed_calls=0):
        """Создать журнал с уже потраченной суммой. Существующий журнал не трогает: обнулить счёт нельзя."""
        spent_usd = _number(spent_usd, 0.0)
        if spent_usd is None:
            raise ValueError('уже потраченная сумма должна быть неотрицательным числом')
        if min(int(calls), int(input_tokens), int(output_tokens), int(failed_calls)) < 0:
            raise ValueError('счётчики обращений и токенов должны быть неотрицательными')
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = self._lock()
        try:
            if self.path.exists():
                raise FileExistsError(f'журнал расходов {self.path} уже есть; обнулять или пересоздавать его нельзя')
            d = {'model': MODEL, 'cap_usd': BUDGET_CAP_USD, 'stop_usd': min(BUDGET_STOP_USD, BUDGET_CAP_USD),
                 'initialized': time.strftime('%Y-%m-%dT%H:%M:%S'),
                 'carried': {'cost_usd': spent_usd, 'calls': int(calls)},      # перенесено из прежнего учёта
                 'calls': int(calls), 'failed_calls': int(failed_calls), 'input_tokens': int(input_tokens),
                 'output_tokens': int(output_tokens), 'cost_usd': round(spent_usd, 9), 'cost_from_response': 0,
                 'unknown_cost_usd': 0.0, 'unknown_attempts': 0, 'reserved': {}, 'halted': None}
            self._write(d)
            return d
        finally:
            os.close(fd)

    def reserve(self, max_usd):
        """Занять в журнале наибольшую возможную стоимость одной попытки; вернуть номер резерва.

        JevBudgetExceeded — журнала нет, работа остановлена или резерв не помещается под потолок. Проверка и
        запись идут под одной блокировкой: два процесса не могут оба занять последние деньги.
        """
        max_usd = float(max_usd)

        def add(d):
            if d.get('halted'):
                raise JevBudgetExceeded(f"работа с Jev остановлена: {d['halted']}")
            busy = self.committed(d)
            allowed = min(self.stop_usd, float(d['stop_usd']))     # порог из журнала тоже обязателен
            if busy + max_usd > allowed + 1e-12:
                raise JevBudgetExceeded(f'потолок расходов на Jev: занято {busy:.6f} долл. (потрачено '
                                        f'{self.spent(d):.6f}), запрос может стоить до {max_usd:.6f}, разрешено '
                                        f'{allowed:.2f}')
            rid = uuid.uuid4().hex
            d['reserved'][rid] = {'usd': round(max_usd, 9), 'pid': os.getpid(), 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
            return rid
        return self._update(add)

    def release(self, rid):
        """Попытка до сервера заведомо не дошла (связь не установилась): резерв снимается, денег не стоило."""
        self._update(lambda d: d['reserved'].pop(rid, None))

    def settle(self, rid, input_tokens=0, output_tokens=0, cost=None, failed=False):
        """Заменить резерв итогом попытки, которая могла дойти до сервера. Возвращает записанную сумму.

        cost — стоимость из ответа; None — стоимость неизвестна: в unknown_cost_usd остаётся оценка сверху
        (сам резерв). Стоимость выше резерва означает, что оценка сверху неверна: работа останавливается.
        """
        return self._settle(rid, input_tokens, output_tokens, cost, failed)[0]

    def _settle(self, rid, input_tokens=0, output_tokens=0, cost=None, failed=False):
        """(записанная сумма, стоимость оказалась выше резерва)."""
        def done(d):
            reserved = float((d['reserved'].pop(rid, None) or {'usd': 0.0})['usd'])
            d['calls'] += 1
            d['failed_calls'] += bool(failed)
            d['input_tokens'] += int(input_tokens)
            d['output_tokens'] += int(output_tokens)
            if cost is None:
                d['unknown_attempts'] += 1
                d['unknown_cost_usd'] = round(float(d['unknown_cost_usd']) + reserved, 9)
                return reserved, False
            d['cost_from_response'] += 1
            d['cost_usd'] = round(float(d['cost_usd']) + float(cost), 9)
            over = float(cost) > reserved + 1e-12
            if over and not d.get('halted'):
                d['halted'] = 'стоимость ответа выше оценки сверху: проверьте цену модели и RESERVE_* в did/llm_jev.py'
            return float(cost), over
        return self._update(done)

    def halt(self, reason):
        def stop(d):
            d['halted'] = str(reason)[:200]
        self._update(stop)


def request_key(model, state, questions):
    raw = json.dumps([model, state, questions], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class JevClient:
    """ask(state, questions) -> JevReply. Потокобезопасен: общего изменяемого состояния, кроме файлов, нет."""

    def __init__(self, cache_dir=None, budget=None, strict=False, key_file=None, timeout_s=20.0, max_retries=4,
                 transport=None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        # только кэш: ни сети, ни ключа, ни расходов
        self.strict = bool(strict) or os.environ.get(CACHE_ONLY_ENV) == '1'
        self.budget = budget or Budget()
        key_file = None if self.strict else (key_file or key_file_path())
        self.key_file = Path(key_file) if key_file else None
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        # Соединение держится открытым между вызовами (иначе в каждое время ответа входила бы установка связи).
        # Ключа в нём нет: заголовок с ключом передаётся отдельно в каждом запросе.
        self._http = httpx.Client(timeout=self.timeout_s, transport=transport, follow_redirects=False)

    def __repr__(self):
        return f'JevClient({MODEL}, кэш {self.cache_dir}' + (', только кэш)' if self.strict else ')')

    def close(self):
        self._http.close()

    # --- кэш -----------------------------------------------------------------------------------

    def _cache_path(self, key):
        return self.cache_dir / f'{key}.json'

    def _cache_get(self, key, questions):
        try:
            rec = json.loads(self._cache_path(key).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        answers = _clean_answers(questions, rec.get('answers')) if isinstance(rec, dict) else None
        return {**rec, 'answers': answers} if answers else None

    def _cache_put(self, key, record):
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.{os.getpid()}.{id(record)}.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path)

    # --- запрос --------------------------------------------------------------------------------

    @staticmethod
    def _check_request(model, questions):
        if model != MODEL:
            raise JevModelRefused(f'запрошенная модель не разрешена: клиент работает только с {MODEL}; запрос не '
                                  f'отправлен')
        if not isinstance(questions, dict) or not questions:
            raise JevError('нужен непустой словарь вопросов')
        for name, q in questions.items():
            if not isinstance(q, dict) or q.get('type') not in QUESTION_TYPES or not q.get('instructions'):
                raise JevError(f'вопрос «{name}»: нужен type из {QUESTION_TYPES} и instructions')

    @staticmethod
    def _answered_by_jev(model):
        """OpenRouter возвращает идентификатор с датой выпуска: typesafe/jev-1.13-20260917."""
        model = str(model or '')
        return model == MODEL or (model.startswith(MODEL + '-') and model[len(MODEL) + 1:].isdigit())

    def ask(self, state, questions, model=MODEL):
        self._check_request(model, questions)
        key = request_key(model, state, questions)
        hit = self._cache_get(key, questions)
        if hit:
            return JevReply(hit['answers'], int(hit.get('latency_ms') or 0), dict(hit.get('usage') or {}),
                            float(hit.get('cost_usd') or 0.0), _ident(hit.get('answered_by'), None) or MODEL, cached=True)
        if self.strict:
            raise CacheMiss(f'только кэш: нет сохранённого ответа Jev на запрос {key[:12]}; сеть не трогаю')
        body = json.dumps({'model': model, 'state': state, 'questions': questions}, ensure_ascii=False).encode('utf-8')
        data, latency_ms, spent = self._send(body, questions)      # data — уже проверенная схема без ключа
        if not self._answered_by_jev(data['model']) or data['provider'] not in (None, 'TypeSafe'):
            # В журнал — только постоянный текст и очищенные идентификаторы, не произвольные строки сервера.
            reason = f"на запрос к {MODEL} ответила «{data['model'][:60]}» (поставщик «{(data['provider'] or '')[:40]}»)"
            self.budget.halt(reason)
            raise JevWrongModel(reason + '; работа с Jev остановлена')
        if data['answers'] is None:
            raise JevError('ответ Jev не по схеме: нет ответов на все вопросы или типы полей не те', code='bad_reply')
        usage = {'input_tokens': data['input_tokens'] or len(body) // 2,     # нет счётчика — оценка с запасом
                 'output_tokens': data['output_tokens']}
        self._cache_put(key, {'model': model, 'answered_by': data['model'], 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                              'latency_ms': latency_ms, 'usage': usage, 'cost_usd': spent,
                              'cost_known': data['cost'] is not None, 'state': state, 'questions': questions,
                              'answers': data['answers']})
        return JevReply(data['answers'], latency_ms, usage, spent, data['model'], cost_known=data['cost'] is not None)

    def _send(self, body, questions):
        """(проверенный ответ, время в мс вместе с повторами, записанная стоимость удачной попытки).

        Ключ живёт только в этой функции; всё, что сервер прислал, очищается от ключа здесь же. Каждая попытка
        сначала резервирует в журнале свою наибольшую стоимость, а после — заменяет резерв итогом: связь не
        установилась или сервер отказал (4xx) — денег не стоило; ответ со стоимостью — стоимость из ответа;
        всё остальное (таймаут чтения, обрыв, 5xx, ответ без стоимости) — оценка сверху.

        Ключ читается после первого удачного резерва: запрос, которому журнал отказал, ключа не трогает; ключ не
        прочитался — резерв снимается. Исключение транспорта (любое, кроме прерывания процесса) наружу не
        выходит: в обработчике из него берётся только вымаранный текст, а учёт в журнале и ошибка клиента — уже
        после обработчика, чтобы исходное исключение (в нём может быть заголовок с ключом) не осталось ни в
        __cause__, ни в __context__.
        """
        if self.strict:
            raise CacheMiss('только кэш: сеть не трогаю')
        limit = max_cost_usd(body)
        key = headers = r = None
        spent_ms, failures, last, code = 0.0, 0, '', 'other'
        try:
            while True:
                pause = min(0.5 * 2 ** failures, 8.0)
                rid = self.budget.reserve(limit)                   # JevBudgetExceeded — попытка не отправляется
                if key is None:
                    failed = None
                    try:
                        key = _read_key(self.key_file)
                    except JevError as e:
                        failed = e
                    if failed is not None:
                        self.budget.release(rid)                   # ключа нет: отправки не будет
                        raise failed
                    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                t0 = time.monotonic()
                r, outcome, fatal = None, None, None
                try:
                    r = self._http.post(ENDPOINT, content=body, headers=headers)
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    outcome, code = 'not_sent', 'network'          # связь не установилась: запрос не дошёл
                    last = _short(f'{type(e).__name__}: {e}', key)
                except httpx.HTTPError as e:
                    outcome = 'unknown'                            # мог дойти: стоимость неизвестна
                    code = 'timeout' if isinstance(e, httpx.TimeoutException) else 'network'
                    last = _short(f'{type(e).__name__}: {e}', key)
                except Exception as e:                             # транспорт бросил что-то своё: не повторяем
                    outcome = 'unknown'
                    fatal = 'сбой транспорта: ' + _short(f'{type(e).__name__}: {e}', key)
                except BaseException:                              # прерывание процесса не перехватывается
                    try:
                        self.budget.settle(rid, failed=True)
                    except JevError:
                        pass                                       # резерв остался в журнале занятым
                    raise
                if outcome == 'not_sent':
                    self.budget.release(rid)
                elif outcome == 'unknown':
                    self.budget.settle(rid, failed=True)
                    if fatal:
                        raise JevError(fatal, code='other')
                else:
                    spent_ms += (time.monotonic() - t0) * 1000
                    if 200 <= r.status_code < 300:
                        try:
                            data = json.loads(_scrub(r.text, key))
                        except ValueError:
                            data = None
                        if isinstance(data, dict):
                            data = _clean_response(data, questions, key)
                            spent, over = self.budget._settle(rid, data['input_tokens'] or len(body) // 2,
                                                              data['output_tokens'], data['cost'])
                            if over:
                                # Деньги уже списаны и записаны; ответом, цена которого вышла за оценку сверху,
                                # пользоваться нельзя, а остановка должна быть видна сразу.
                                raise JevCostOverrun(f'ответ Jev стоил {spent:.6f} долл. — больше резерва '
                                                     f'{limit:.6f}: ответ отброшен, работа с Jev остановлена')
                            return data, int(round(spent_ms)), spent
                        self.budget.settle(rid, failed=True)
                        last, code = 'ответ не JSON-объект', 'bad_reply'
                    elif r.status_code == 408 or r.status_code >= 500:
                        self.budget.settle(rid, failed=True)
                        last, code = f'HTTP {r.status_code}: ' + _short(r.text, key), 'server'
                    else:
                        self.budget.settle(rid, cost=0.0, failed=True)     # сервер отказал: не оплачивается
                        last, code = f'HTTP {r.status_code}: ' + _short(r.text, key), 'server'
                        if r.status_code != 429:
                            raise JevError(last, code=code)
                    try:
                        pause = max(pause, min(float(r.headers.get('retry-after', 0)), 8.0))
                    except ValueError:
                        pass
                failures += 1
                if failures > self.max_retries:
                    raise JevError(f'Jev не ответил за {failures} попыток: {last}', code=code)
                time.sleep(pause)
                spent_ms += pause * 1000
        finally:
            # В кадре не остаётся ни ключа, ни ответа с запросом (в заголовках запроса — ключ): кадр виден из
            # traceback любого исключения, вышедшего отсюда.
            del key, headers, r


def main(argv=None):
    ap = argparse.ArgumentParser(description='Журнал расходов на Jev и размер кэша ответов.')
    ap.add_argument('--init-budget', action='store_true',
                    help='создать журнал расходов (один раз); существующий журнал не меняется')
    ap.add_argument('--spent', type=float, help='уже потрачено, долларов (из прежнего учёта)')
    ap.add_argument('--calls', type=int, default=0, help='уже сделано обращений')
    ap.add_argument('--input-tokens', type=int, default=0)
    ap.add_argument('--output-tokens', type=int, default=0)
    ap.add_argument('--failed-calls', type=int, default=0)
    args = ap.parse_args(argv)
    budget = Budget()
    if args.init_budget:
        if args.spent is None:
            ap.error('--init-budget требует --spent: сколько долларов уже потрачено (0, если ничего)')
        try:
            budget.init(args.spent, args.calls, args.input_tokens, args.output_tokens, args.failed_calls)
        except (FileExistsError, ValueError) as e:
            print(f'ОТКАЗ: {e}')
            return 2
        print(f'журнал расходов создан: {budget.path}')
    try:
        b = budget.read()
    except JevBudgetExceeded as e:
        print(f'ОТКАЗ: {e}')
        return 2
    print(f'журнал расходов: {budget.path}')
    if b is None:
        print('  журнала нет: платные вызовы запрещены. Создать: python -m did.llm_jev --init-budget --spent '
              '<долларов> --calls <обращений>')
    else:
        held = Budget.committed(b) - Budget.spent(b)
        print(f"  обращений {b['calls']} (неудачных {b['failed_calls']}), входных токенов {b['input_tokens']}")
        print(f"  потрачено {Budget.spent(b):.6f} долл.: по ответам сервера {b['cost_usd']:.6f}, по оценке сверху "
              f"{b['unknown_cost_usd']:.6f} ({b['unknown_attempts']} попыток с неизвестной стоимостью)")
        print(f"  в резерве {held:.6f} ({len(b['reserved'])} запросов в пути или оборванных); отказ с "
              f"{b['stop_usd']:.2f}, потолок {b['cap_usd']:.2f}")
        if b.get('halted'):
            print(f"  ОСТАНОВЛЕНО: {b['halted']}")
    n = len(list(CACHE_DIR.glob('*.json'))) if CACHE_DIR.is_dir() else 0
    print(f'кэш ответов: {CACHE_DIR}, ответов {n}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
