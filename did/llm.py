"""Планировщик верхнего уровня: транспорт к модели, разбор ответа, схема и проверка плана.

Модель выдаёт только список подцелей, скоростью робота она не управляет. Сервер считается
совместимым с OpenAI /chat/completions; для QWEN и DeepSeek на ai.mai.ru это допущение, поэтому
адрес, модель, таймаут и добавки к запросу задаются переменными DID_LLM_* (см. ChatClient.from_env).
Сбой сети или модели — всегда LLMError, а request_plan не бросает исключений вовсе.

Клиенты с одним интерфейсом chat(messages, temperature, max_tokens, schema=None) -> ChatReply
собирает make_client(kind): mock (имитатор), http (.env), ollama (локальная Qwen), codex (GPT по
подписке через Codex CLI, did/llm_codex.py).

Проверка связи: python -m did.llm [--kind http|ollama|codex|mock] [--state state.json]
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union

import httpx
from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator

from . import ROOT

log = logging.getLogger('did.llm')

ARENA_LIMIT = 2.6            # допустимые |x| и |y| точки goto, м
MAX_PAUSE_S = 10.0           # потолок паузы между повторами, в том числе по Retry-After
PROMPTS = Path(__file__).resolve().parent / 'prompts'
PROMPT_PATH = PROMPTS / 'planner_system.md'
SUBGOAL_TYPES = ('investigate', 'explore', 'goto', 'return_base')
CLIENT_KINDS = ('mock', 'http', 'ollama', 'codex')
OLLAMA_URL = 'http://127.0.0.1:11434/v1'
OLLAMA_MODEL = 'qwen2.5:3b'

# Пример состояния, которое агент передаёт планировщику. Любой список может быть пустым.
EXAMPLE_STATE = {
    'trigger': 'sample_collected',
    'time_s': 123.4, 'time_limit_s': 600,
    'battery': 41.2, 'battery_start': 60,
    'pose': {'x': 0.42, 'y': 1.10}, 'base': {'x': -2.0, 'y': -0.5},
    'samples': {'collected': 2, 'total': 5},
    'return_cost': 5.1,
    'candidates': [
        {'id': 'C1', 'x': 0.5, 'y': 1.2, 'confidence': 0.74, 'cost_to': 3.2, 'cost_back': 5.0, 'feasible': True},
    ],
    'explore_points': [
        {'id': 'E1', 'x': 1.4, 'y': -1.2, 'unseen_share': 0.31, 'cost_to': 4.0, 'cost_back': 6.2,
         'feasible': True},
    ],
    'soil_zones': [
        {'id': 'S1', 'x': 0.3, 'y': 0.9, 'radius': 0.4, 'mult': 2.9, 'evidence_m': 0.8, 'status': 'confirmed'},
    ],
    'hazards': [{'id': 'Z1', 'x': 1.0, 'y': 0.2, 'radius': 0.35}],
    'sensor': {'reading': 0.42, 'noise': 0.05, 'status': 'ok'},
    'recent_events': [{'t': 118.0, 'type': 'sample_collected'}],
    'alarms': ['расход батареи на 80% выше прогноза около (0.4; 1.0)'],
    'open_hypotheses': [{'id': 'H3', 'statement': 'в районе (0.3; 0.9) грунт дороже примерно в 3 раза'}],
}


class LLMError(Exception):
    """Любой сбой на пути к плану: сеть, сервер, формат ответа, настройки."""

    def __init__(self, message, attempts=0):
        super().__init__(message)
        self.attempts = attempts     # сколько запросов ушло на сервер до отказа


# --- окружение ---------------------------------------------------------------------------------

_QUOTED = re.compile(r'''^(["'])(.*?)\1\s*(?:#.*)?$''')


def load_env(path=ROOT / '.env'):
    """Читает строки KEY=VALUE в os.environ. Уже заданные переменные не трогает.

    Возвращает имена добавленных переменных (без значений: среди них бывает ключ).
    """
    path = Path(path)
    if not path.is_file():
        return []
    added = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        quoted = _QUOTED.match(value)
        value = quoted.group(2) if quoted else value.split(' #', 1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added


def _env_number(name, default, cast):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise LLMError(f'{name}: нужно число, а задано «{raw}»') from None


# --- транспорт ---------------------------------------------------------------------------------

@dataclass
class ChatReply:
    text: str
    latency_ms: int              # всё время вызова вместе с повторами
    usage: dict
    attempts: int                # сколько запросов ушло на сервер
    finish_reason: str = None    # 'length' — ответ обрезан по max_tokens
    cached: bool = False         # ответ взят с диска; latency_ms — время того, настоящего вызова


def _ms(t0):
    return int(round((time.monotonic() - t0) * 1000))


def _read_reply(body):
    """(текст, usage, finish_reason) из тела ответа /chat/completions. ValueError — формат не тот."""
    try:
        data = json.loads(body)
    except ValueError:
        raise ValueError(f'ответ сервера — не JSON: {body[:200]}') from None
    try:
        choice = data['choices'][0]
        content = choice['message'].get('content')
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError(f'в ответе сервера нет choices[0].message: {body[:200]}') from None
    if isinstance(content, list):            # содержимое частями: [{"type": "text", "text": "..."}]
        content = ''.join(p.get('text') or '' for p in content if isinstance(p, dict))
    usage = data.get('usage')
    return str(content or ''), usage if isinstance(usage, dict) else {}, choice.get('finish_reason')


def _retry_after(headers):
    try:
        return min(max(float(headers.get('retry-after') or 0.0), 0.0), MAX_PAUSE_S)
    except ValueError:                       # дата вместо секунд — не разбираем
        return 0.0


class ChatClient:
    """Клиент OpenAI-совместимого /chat/completions с повторами. Ключ наружу не выдаёт."""

    def __init__(self, base_url, api_key, model, timeout_s=30, max_retries=2, json_mode=True,
                 retry_pause_s=0.5, extra_body=None, use_schema=False, min_tokens=0):
        base = str(base_url or '').strip().rstrip('/')
        if base.endswith('/chat/completions'):
            base = base[:-len('/chat/completions')]
        if not base.startswith(('http://', 'https://')):
            raise LLMError(f'адрес модели должен начинаться с http:// или https://, а задан «{base}»')
        self.base_url = base
        self.model = model or ''
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.json_mode = bool(json_mode)             # сбрасывается, если сервер не знает response_format
        self.use_schema = bool(use_schema)           # слать JSON-схему ответа, если вызывающий её дал
        self.retry_pause_s = float(retry_pause_s)    # пауза перед первым повтором, дальше удваивается
        self.extra_body = dict(extra_body or {})     # добавки к телу запроса, например enable_thinking
        # Нижняя граница лимита ответа: у рассуждающих моделей рассуждение тратит тот же лимит, и при 800
        # токенах ответ обрывается на середине JSON.
        self.min_tokens = int(min_tokens or 0)
        self._api_key = str(api_key or '')
        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
        # Системный прокси нужен для внешнего адреса, а до локального сервера он запрос не довезёт.
        local = httpx.URL(base).host in ('127.0.0.1', 'localhost', '::1')
        try:
            self._http = httpx.Client(timeout=self.timeout_s, headers=headers, trust_env=not local)
        except (UnicodeError, ValueError):
            raise LLMError('ключ доступа содержит недопустимые знаки: в нём могут быть только латиница, цифры и '
                           'знаки ASCII') from None

    @classmethod
    def from_env(cls, env_path=ROOT / '.env', **overrides):
        """Клиент по переменным окружения; None, если DID_LLM_BASE_URL не задан.

        Основные: DID_LLM_BASE_URL, DID_LLM_API_KEY, DID_LLM_MODEL, DID_LLM_TIMEOUT_S (30).
        Дополнительные: DID_LLM_MAX_RETRIES (2), DID_LLM_JSON_MODE (1), DID_LLM_JSON_SCHEMA (0), DID_LLM_MIN_TOKENS (0),
        DID_LLM_EXTRA_BODY (JSON-объект).
        Сначала читается файл env_path; env_path=None — только то, что уже в окружении.
        overrides — параметры конструктора поверх окружения, например model или timeout_s.
        """
        if env_path:
            load_env(env_path)
        env = os.environ
        base = str(overrides.pop('base_url', None) or env.get('DID_LLM_BASE_URL', '')).strip()
        if not base:
            return None
        try:
            extra = json.loads(env.get('DID_LLM_EXTRA_BODY') or '{}')
        except ValueError:
            extra = None
        if not isinstance(extra, dict):
            raise LLMError('DID_LLM_EXTRA_BODY: нужен JSON-объект, например {"enable_thinking": false}')
        off = ('0', 'false', 'no', 'off', '')
        opts = dict(api_key=env.get('DID_LLM_API_KEY', '').strip(), model=env.get('DID_LLM_MODEL', '').strip(),
                    timeout_s=_env_number('DID_LLM_TIMEOUT_S', 30.0, float),
                    max_retries=_env_number('DID_LLM_MAX_RETRIES', 2, int),
                    min_tokens=_env_number('DID_LLM_MIN_TOKENS', 0, int),
                    json_mode=env.get('DID_LLM_JSON_MODE', '1').strip().lower() not in off,
                    use_schema=env.get('DID_LLM_JSON_SCHEMA', '0').strip().lower() not in off, extra_body=extra)
        return cls(base, **{**opts, **overrides})

    def __repr__(self):
        return (f'ChatClient({self.base_url}, model={self.model or "—"}, timeout_s={self.timeout_s:g}, '
                f'key={"задан" if self._api_key else "нет"})')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._http.close()

    def _safe(self, text, limit=300):
        """Текст для исключений и логов: одна строка, без ключа, не длиннее limit."""
        text = ' '.join(str(text).split())
        if self._api_key:
            text = text.replace(self._api_key, '***')
        return text if len(text) <= limit else text[:limit] + '…'

    def _body(self, messages, temperature, max_tokens, use_json, schema=None):
        body = {'messages': messages}
        if self.model:
            body['model'] = self.model
        if temperature is not None:
            body['temperature'] = temperature
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        if use_json == 'schema':
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': 'reply', 'strict': True, 'schema': schema}}
        elif use_json:
            body['response_format'] = {'type': 'json_object'}
        body.update(self.extra_body)
        try:
            return json.dumps(body, ensure_ascii=False).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise LLMError(f'запрос не превращается в JSON: {e}') from None

    def _post(self, content):
        """(код, заголовки, тело ответа). timeout_s ограничивает весь ответ, а не только паузы в нём:
        иначе сервер, который в ожидании очереди шлёт пустые строки, держит вызов минутами."""
        deadline = time.monotonic() + self.timeout_s
        with self._http.stream('POST', self.base_url + '/chat/completions', content=content,
                               headers={'Content-Type': 'application/json'}) as r:
            chunks = []
            for chunk in r.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout('ответ не уложился в срок')
            return r.status_code, r.headers, b''.join(chunks).decode('utf-8', 'replace')

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        """Один ответ модели. Повторяет запрос при таймауте, обрыве связи, 429 и 5xx.

        Сервер, который отвечает 400 на response_format, получает тот же запрос без него (один раз,
        попытка не тратится), и дальше клиент этот параметр не шлёт. schema — JSON-схема ответа; она
        уходит на сервер только при use_schema, а отказ сервера снижает запрос до json_object.
        """
        t0 = time.monotonic()
        if self.min_tokens:
            max_tokens = max(int(max_tokens or 0), self.min_tokens)
        want_schema = bool(schema) and self.use_schema and self.json_mode
        use_json = 'schema' if want_schema else self.json_mode
        attempts = failures = 0
        while True:
            attempts += 1
            pause = min(self.retry_pause_s * 2 ** failures, MAX_PAUSE_S)
            content = self._body(messages, temperature, max_tokens, use_json, schema)
            try:
                code, headers, body = self._post(content)
            except httpx.TimeoutException:
                last = f'таймаут {self.timeout_s:g} с'
            except httpx.TransportError as e:
                last = self._safe(f'нет связи с {self.base_url}: {type(e).__name__}: {e}')
            except Exception as e:
                raise LLMError(self._safe(f'запрос не отправлен: {type(e).__name__}: {e}'), attempts) from None
            else:
                if 200 <= code < 300:
                    try:
                        text, usage, finish = _read_reply(body)
                    except ValueError as e:
                        last = self._safe(e)
                    else:
                        if self.json_mode and not use_json:
                            self.json_mode = False
                            log.info('LLM: сервер не принимает response_format, дальше без него')
                        if want_schema and use_json != 'schema':
                            self.use_schema = False
                            log.info('LLM: сервер не принимает JSON-схему ответа, дальше без неё')
                        return ChatReply(text, _ms(t0), usage, attempts, finish)
                elif code in (400, 422) and use_json:
                    use_json = use_json == 'schema'      # схема → json_object → без формата
                    continue
                elif code in (408, 429) or code >= 500:
                    last = self._safe(f'HTTP {code}: {body}')
                    pause = max(pause, _retry_after(headers))
                elif 300 <= code < 400:
                    where = headers.get('location', '?')
                    raise LLMError(self._safe(f'HTTP {code}: сервер перенаправляет на {where}, исправьте адрес'),
                                   attempts)
                else:
                    raise LLMError(self._safe(f'HTTP {code}: {body}'), attempts)
            failures += 1
            if failures > self.max_retries:
                raise LLMError(f'модель не ответила за {failures} попыток: {last}', attempts)
            log.info('LLM: попытка %d не удалась (%s), повтор через %.1f с', failures, last, pause)
            time.sleep(pause)

    def models(self):
        """Идентификаторы моделей из GET {base_url}/models — быстрая проверка адреса и ключа."""
        try:
            r = self._http.get(self.base_url + '/models')
        except Exception as e:
            raise LLMError(self._safe(f'нет связи с {self.base_url}: {type(e).__name__}: {e}')) from None
        if r.status_code != 200:
            raise LLMError(self._safe(f'HTTP {r.status_code}: {r.text}'))
        try:
            return [str(m['id']) for m in r.json()['data'] or []]     # у Ollama без моделей data — null
        except (ValueError, KeyError, TypeError):
            raise LLMError(self._safe(f'ответ /models не похож на OpenAI: {r.text}')) from None


class LocalClient:
    """Тот же chat(), но отвечает функция responder(messages) -> str в процессе, без HTTP."""

    def __init__(self, responder):
        self.responder = responder

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        t0 = time.monotonic()
        try:
            text = self.responder(messages)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f'локальный ответчик: {type(e).__name__}: {e}') from e
        if not isinstance(text, str):
            raise LLMError(f'локальный ответчик вернул {type(text).__name__}, а нужна строка')
        return ChatReply(text, _ms(t0), {}, 1)


# --- разбор ответа -----------------------------------------------------------------------------

_THINK = re.compile(r'<think>.*?</think>', re.S | re.I)
_TRAILING_COMMA = re.compile(r',(\s*[}\]])')
# Невидимые знаки, которые модели иногда вставляют в строки (GPT дописал такой к идентификатору
# цели: "C1 "): область частного использования и знаки нулевой ширины.
_INVISIBLE = re.compile('[-​-‍⁠﻿]')


def strip_think(text):
    """Убирает рассуждения модели: пары <think>…</think>, текст до одинокого </think>, незакрытый <think>."""
    text = _THINK.sub('', text)
    close = text.lower().rfind('</think>')
    if close >= 0:
        text = text[close + len('</think>'):]
    start = text.lower().find('<think>')
    if start >= 0:
        text = text[:start]
    return text.strip()


def _object_end(text, start):
    """Индекс скобки, закрывающей объект с началом в start, или -1. Скобки внутри строк не считаются."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_json(text):
    """Первый сбалансированный JSON-объект из ответа модели.

    Рассуждения <think>, обёртка ```json и текст вокруг отбрасываются; лишняя запятая перед
    закрывающей скобкой прощается, невидимые знаки удаляются.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMError('пустой ответ')
    body = _INVISIBLE.sub('', strip_think(text))
    start = body.find('{')
    if start < 0:
        raise LLMError('в ответе нет JSON-объекта')
    for _ in range(50):                      # потолок работы на ответе из одних скобок
        if start < 0:
            break
        end = _object_end(body, start)
        if end >= 0:
            chunk = body[start:end + 1]
            for variant in (chunk, _TRAILING_COMMA.sub(r'\1', chunk)):
                try:
                    obj = json.loads(variant)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    return obj
        start = body.find('{', start + 1)
    raise LLMError('JSON-объект в ответе не закрыт или испорчен (ответ обрезан?)')


# --- схема плана -------------------------------------------------------------------------------

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Coord = Annotated[float, Field(allow_inf_nan=False)]


class Investigate(BaseModel):
    """Подъехать к кандидату и попытаться собрать образец."""
    type: Literal['investigate']
    target: Text


class Explore(BaseModel):
    """Доехать до точки разведки."""
    type: Literal['explore']
    target: Text


class Goto(BaseModel):
    """Произвольная точка арены."""
    type: Literal['goto']
    x: Coord
    y: Coord


class ReturnBase(BaseModel):
    """Вернуться на базу и завершить прогон."""
    type: Literal['return_base']


Subgoal = Annotated[Union[Investigate, Explore, Goto, ReturnBase], Field(discriminator='type')]


class Hypothesis(BaseModel):
    statement: Text
    test: Text


class Plan(BaseModel):
    reasoning: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=3)
    subgoals: list[Subgoal] = Field(min_length=1, max_length=6)

    @field_validator('hypotheses', mode='before')
    @classmethod
    def _null_is_empty(cls, value):
        return [] if value is None else value


def _obj(**props):
    """Объект JSON-схемы в строгом виде: все поля обязательны, лишних нет (так требуют OpenAI и Codex)."""
    return {'type': 'object', 'additionalProperties': False, 'required': list(props), 'properties': props}


def _feasible_ids(state, name):
    items = state.get(name) if isinstance(state, dict) else None
    return [str(it['id']) for it in items or []
            if isinstance(it, dict) and it.get('id') and (it.get('feasible') is None or it['feasible'])]


def plan_schema(state=None):
    """JSON-схема плана под состояние — для клиентов, которые заставляют модель отвечать по схеме.

    Цель подцели — перечисление достижимых id из состояния, отдельно для кандидатов и точек
    разведки: назвать несуществующую, недостижимую или испорченную цель модель не может. На первой
    проверке GPT в свободной строке дописывал к идентификатору мусор ("E2خ", "C1}]}").
    """
    def one(kind, **fields):
        return _obj(type={'type': 'string', 'enum': [kind]}, **fields)

    variants = [one(kind, target={'type': 'string', 'enum': ids})
                for kind, ids in (('investigate', _feasible_ids(state, 'candidates')),
                                  ('explore', _feasible_ids(state, 'explore_points'))) if ids]
    variants += [one('goto', x={'type': 'number'}, y={'type': 'number'}), one('return_base')]
    return _obj(
        reasoning={'type': 'string'},
        hypotheses={'type': 'array', 'items': _obj(statement={'type': 'string'}, test={'type': 'string'})},
        subgoals={'type': 'array', 'items': {'anyOf': variants}})


# Схема под пример состояния: показывает форму, в запросах берётся plan_schema(state).
PLAN_SCHEMA = plan_schema(EXAMPLE_STATE)


_SCHEMA_TEXT = {
    'missing': 'нет обязательного поля',
    'string_too_short': 'пустая строка',
    'string_too_long': 'строка длиннее {max_length} символов (сейчас {length})',
    'too_short': 'нужно не меньше {min_length} элементов (сейчас {actual_length})',
    'too_long': 'нужно не больше {max_length} элементов (сейчас {actual_length})',
    'union_tag_not_found': 'у подцели нет поля "type"',
    'union_tag_invalid': 'неизвестный тип "{tag}"; допустимы: ' + ', '.join(SUBGOAL_TYPES),
    'string_type': 'нужна строка',
    'float_type': 'нужно число',
    'float_parsing': 'нужно число',
    'finite_number': 'нужно конечное число',
    'list_type': 'нужен список',
    'model_type': 'нужен JSON-объект',
    'model_attributes_type': 'нужен JSON-объект',
    'dict_type': 'нужен JSON-объект',
    'literal_error': 'нужно одно из: {expected}',
}


def _schema_errors(exc, root='план'):
    """Ошибки pydantic по-русски, с путём к полю: «subgoals[1].x: нужно число»."""
    out = []
    for e in exc.errors(include_url=False):
        where = ''
        for part in e['loc']:
            if isinstance(part, int):
                where += f'[{part}]'
            elif not (part in SUBGOAL_TYPES and where.endswith(']')):   # метка варианта подцели — не поле
                where += ('.' if where else '') + str(part)
        ctx = dict(e.get('ctx') or {})
        if e['type'] == 'string_too_long':
            ctx['length'] = len(e['input'])
        try:
            what = _SCHEMA_TEXT[e['type']].format(**ctx)
        except (KeyError, IndexError):
            what = e['msg']
        out.append(f'{where or root}: {what}')
    return out


# --- смысловые проверки ------------------------------------------------------------------------

_LISTS = {'investigate': ('candidates', 'explore_points', 'explore'),
          'explore': ('explore_points', 'candidates', 'investigate')}


def validate_plan(plan, state):
    """Проверка плана по состоянию: список сообщений об ошибках, пустой — план можно исполнять."""
    if not isinstance(plan, Plan):
        try:
            plan = Plan.model_validate(plan)
        except ValidationError as e:
            return _schema_errors(e)
    state = state if isinstance(state, dict) else {}
    known = {name: {str(item.get('id')): item for item in state.get(name) or [] if isinstance(item, dict)}
             for name in ('candidates', 'explore_points')}
    errors = []
    prev = None
    for i, sg in enumerate(plan.subgoals):
        where = f'subgoals[{i}]'
        if sg.type == 'goto':
            key = ('goto', round(sg.x, 2), round(sg.y, 2))
            if abs(sg.x) > ARENA_LIMIT or abs(sg.y) > ARENA_LIMIT:
                errors.append(f'{where}: точка ({sg.x:g}; {sg.y:g}) вне арены, нужно |x| ≤ {ARENA_LIMIT} '
                              f'и |y| ≤ {ARENA_LIMIT}')
        elif sg.type == 'return_base':
            key = ('return_base',)
            if i < len(plan.subgoals) - 1:
                errors.append(f'{where}: return_base может быть только последней подцелью')
        else:
            key = (sg.type, sg.target)
            own, other, other_type = _LISTS[sg.type]
            item = known[own].get(sg.target)
            if item is None and sg.target in known[other]:
                errors.append(f'{where}: "{sg.target}" есть в {other}, а не в {own}; для такой цели нужен '
                              f'тип "{other_type}"')
            elif item is None:
                ids = ', '.join(known[own]) or 'список пуст'
                errors.append(f'{where}: цели "{sg.target}" нет в {own} ({ids})')
            elif item.get('feasible') is not None and not item['feasible']:
                errors.append(f'{where}: цель "{sg.target}" недостижима по заряду (feasible=false); '
                              f'выбери другую или return_base')
        if key == prev:
            errors.append(f'{where}: повтор предыдущей подцели, одна цель не идёт два раза подряд')
        prev = key
    return errors


_ID_HEAD = re.compile(r'[A-Za-z]{1,3}\d{1,3}')


def _trim_targets(data, state):
    """Цель с мусором после идентификатора ("E1ъ", "C1}]} <>") -> сам идентификатор, если он есть в состоянии.

    Так ошибается GPT в свободной строке; намерение при этом однозначно, и тратить запрос на
    исправление незачем. Незнакомая цель остаётся как есть и не пройдёт проверку.
    """
    state = state if isinstance(state, dict) else {}
    known = {str(it.get('id')) for name in ('candidates', 'explore_points')
             for it in state.get(name) or [] if isinstance(it, dict)}
    subgoals = data.get('subgoals')
    for sg in subgoals if isinstance(subgoals, list) else []:
        target = sg.get('target') if isinstance(sg, dict) else None
        if isinstance(target, str) and target.strip() not in known:
            head = _ID_HEAD.match(target.strip())
            if head and head.group(0) in known:
                sg['target'] = head.group(0)


def parse_plan(text, state):
    """Ответ модели -> (Plan или None, список ошибок): разбор JSON, схема, смысловые проверки."""
    try:
        data = extract_json(text)
    except LLMError as e:
        return None, [str(e)]
    _trim_targets(data, state)
    try:
        plan = Plan.model_validate(data)
    except ValidationError as e:
        return None, _schema_errors(e)
    errors = validate_plan(plan, state)
    return (None, errors) if errors else (plan, [])


# --- запрос плана ------------------------------------------------------------------------------

@dataclass
class PlanResult:
    plan: Plan = None                    # None — плана нет, причина в error
    error: str = None
    exchanges: list = field(default_factory=list)   # обмены с моделью для журнала
    latency_ms: int = 0                  # суммарно, вместе с исправлениями

    @property
    def ok(self):
        return self.plan is not None

    def to_dict(self):
        return {'plan': self.plan.model_dump() if self.plan else None, 'error': self.error,
                'exchanges': self.exchanges, 'latency_ms': self.latency_ms}


@lru_cache(maxsize=16)
def load_system_prompt(name=None):
    """Системный промпт планировщика; name — другой файл из did/prompts без «.md» (роли, прежние версии)."""
    path = PROMPTS / f'{name}.md' if name else PROMPT_PATH
    return path.read_text(encoding='utf-8').strip()


def _jsonable(value):
    """Для json.dumps: скаляры и массивы numpy, множества, всё прочее — строкой."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _brief(text, limit=300):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit] + '…'


def build_messages(state, system_prompt=None):
    """Системный промпт и состояние в JSON — первый запрос плана."""
    dump = json.dumps(state, ensure_ascii=False, default=_jsonable)
    return [{'role': 'system', 'content': system_prompt or load_system_prompt()},
            {'role': 'user', 'content': f'Состояние:\n{dump}\nВерни план: один JSON-объект.'}]


def repair_message(errors, noun='план'):
    return ('Ответ не принят. Ошибки:\n' + '\n'.join(f'- {e}' for e in errors)
            + f'\nИсправь и верни {noun} заново: один JSON-объект по схеме, без пояснений и без Markdown.')


def request_json(client, messages, parse, schema=None, max_repairs=1, role='user', noun='план'):
    """Общий цикл планировщика и ролей: запрос → разбор и проверка → исправление моделью.

    parse(text) -> (объект или None, список ошибок). Возвращает (объект или None, ошибка или None,
    обмены, время в мс) и исключений не бросает. Запись обмена:
    {attempt, role, request, response, ok, errors, latency_ms, cached, http_attempts, usage}.
    latency_ms — фактическое время ответа модели; у ответа из кэша это время исходного вызова.
    schema уходит клиенту, только если он умеет заставить модель отвечать по схеме (use_schema).
    """
    t0 = time.monotonic()
    exchanges = []
    saved = 0                                # время ответов, взятых из кэша: в часах этого вызова его нет
    try:
        tries = 1 + max(0, int(max_repairs))
        extra = {'schema': schema} if schema is not None and getattr(client, 'use_schema', False) else {}
        for attempt in range(1, tries + 1):
            ex = {'attempt': attempt, 'role': role, 'request': _brief(messages[-1]['content']),
                  'response': None, 'ok': False, 'errors': [], 'latency_ms': 0, 'cached': False,
                  'http_attempts': 0, 'usage': {}}
            exchanges.append(ex)
            t1 = time.monotonic()
            try:
                reply = client.chat(messages, **extra)
            except Exception as e:           # транспорт уже сделал свои повторы — исправлять нечего
                ex.update(latency_ms=_ms(t1), errors=[_brief(e)], http_attempts=getattr(e, 'attempts', 0))
                return None, f'модель недоступна: {_brief(e)}', exchanges, _ms(t0) + saved
            cached = bool(getattr(reply, 'cached', False))
            saved += reply.latency_ms if cached else 0
            ex.update(response=reply.text, latency_ms=int(getattr(reply, 'latency_ms', None) or _ms(t1)),
                      cached=cached, http_attempts=reply.attempts, usage=dict(reply.usage or {}))
            obj, errors = parse(reply.text)
            if obj is not None:
                ex['ok'] = True
                return obj, None, exchanges, _ms(t0) + saved
            if getattr(reply, 'finish_reason', None) == 'length':
                errors.append('ответ обрезан по лимиту токенов, отвечай короче')
            ex['errors'] = errors
            messages = messages + [
                {'role': 'assistant', 'content': strip_think(reply.text)[:2000] or '(пустой ответ)'},
                {'role': 'user', 'content': repair_message(errors, noun)}]
        return None, f'{noun} не принят после {tries} попыток: ' + '; '.join(errors), exchanges, _ms(t0) + saved
    except Exception as e:
        return None, f'внутренняя ошибка: {type(e).__name__}: {_brief(e)}', exchanges, _ms(t0) + saved


def request_plan(client, state, system_prompt=None, max_repairs=1):
    """План от модели с проверкой; на ошибку разбора или проверки модель исправляет ответ.

    Исключений не бросает: при любой неудаче PlanResult(plan=None, error=...). Обмены — как в request_json.
    """
    t0 = time.monotonic()
    try:
        messages = build_messages(state, system_prompt)
    except Exception as e:
        return PlanResult(None, f'внутренняя ошибка планировщика: {type(e).__name__}: {_brief(e)}', [], _ms(t0))
    return PlanResult(*request_json(client, messages, lambda text: parse_plan(text, state), plan_schema(state),
                                    max_repairs))


# --- сводка по обменам -------------------------------------------------------------------------

_ERROR_KINDS = (
    ('квота вызовов исчерпана', ('квота вызовов',)),
    ('нет ответа модели', ('модель не ответила', 'таймаут', 'нет связи', 'HTTP ', 'codex', 'ответчик',
                           'сбой транспорта')),
    ('ответ не JSON', ('пустой ответ', 'нет JSON-объекта', 'не закрыт или испорчен')),
    ('ответ обрезан', ('обрезан по лимиту',)),
    ('цель недостижима по заряду', ('недостижима по заряду',)),
    ('цель не из списка', ('нет в candidates', 'нет в explore_points', 'а не в ')),
    ('точка вне арены', ('вне арены',)),
    ('return_base не последняя', ('только последней',)),
    ('повтор цели подряд', ('повтор предыдущей',)),
)


def error_kind(message):
    """Вид ошибки по её тексту — для подсчёта типичных ошибок модели."""
    for kind, marks in _ERROR_KINDS:
        if any(m in message for m in marks):
            return kind
    return 'нарушена схема'


def llm_stats(exchanges):
    """Сводка по обменам с моделью за прогон (записи request_plan и ролей, см. Recorder.llm).

    Запрос — это первый ответ и его исправления. first_ok — принят сразу, repaired — после
    исправления, failed — не принят вовсе (решение ушло запасному правилу); из них quota — модель
    не спрашивали, потому что кончилась квота вызовов. Время — по обменам, где модель отвечала;
    transport_retries — сколько раз транспорт повторял запрос (таймаут, обрыв, 5xx); tokens — средний
    размер запроса и ответа в токенах, если сервер их сообщает.
    """
    requests = []
    for ex in exchanges or []:
        if ex.get('attempt', 1) == 1 or not requests:
            requests.append([])
        requests[-1].append(ex)
    first = sum(1 for r in requests if r[0].get('ok'))
    repaired = sum(1 for r in requests if not r[0].get('ok') and r[-1].get('ok'))
    errors = {}
    for ex in exchanges or []:
        for kind in {error_kind(str(e)) for e in ex.get('errors') or []}:
            errors[kind] = errors.get(kind, 0) + 1
    def blocked(ex):
        return any(error_kind(str(e)) == 'квота вызовов исчерпана' for e in ex.get('errors') or [])

    ms = sorted(int(ex.get('latency_ms') or 0) for ex in exchanges or [] if not blocked(ex))
    n = len(requests)
    usage = [ex['usage'] for ex in exchanges or [] if (ex.get('usage') or {}).get('prompt_tokens')]

    def tokens(key):
        return round(sum(u.get(key) or 0 for u in usage) / len(usage)) if usage else 0

    return {'requests': n, 'exchanges': len(exchanges or []), 'first_ok': first, 'repaired': repaired,
            'failed': n - first - repaired, 'quota': sum(1 for r in requests if blocked(r[-1])),
            'cached': sum(1 for ex in exchanges or [] if ex.get('cached')),
            'transport_retries': sum(max(0, int(ex.get('http_attempts') or 0) - 1) for ex in exchanges or []),
            'tokens': {'prompt': tokens('prompt_tokens'), 'completion': tokens('completion_tokens')},
            'errors': dict(sorted(errors.items(), key=lambda kv: -kv[1])),
            'latency_ms': {'mean': round(sum(ms) / len(ms)) if ms else 0, 'median': ms[len(ms) // 2] if ms else 0,
                           'max': ms[-1] if ms else 0, 'total': sum(ms)}}


# --- выбор клиента -----------------------------------------------------------------------------

def make_client(kind='mock', **opts):
    """Клиент модели с интерфейсом chat(messages, temperature, max_tokens, schema=None) -> ChatReply.

    mock   — имитатор в процессе: seed, faults, script (см. did.llm_mock.MockResponder);
    http   — адрес, ключ и модель из DID_LLM_* (окружение или .env); opts — параметры ChatClient
             поверх окружения, например model или timeout_s;
    ollama — локальный сервер Ollama: model ('qwen2.5:3b'), base_url, timeout_s (120), use_schema (True).
             Окно контекста задаёт сервер (на 16 ГБ памяти — 4096 токенов), а запрос планировщика —
             3–4 тыс. токенов: если ответы портятся, запустите сервер с OLLAMA_CONTEXT_LENGTH=8192;
    codex  — GPT по подписке через Codex CLI: model ('gpt-6-luna'), effort ('low'), timeout_s (120),
             cache_dir, max_calls (см. did.llm_codex.CodexCliClient); ответы всегда кэшируются на диск.
    Для http и ollama cache=True кладёт ответы в тот же кэш (повторяемость прогонов).
    Ошибка настроек — LLMError.
    """
    opts = dict(opts)
    if kind == 'mock':
        from .llm_mock import MockResponder
        return LocalClient(MockResponder(seed=opts.get('seed', 0), faults=opts.get('faults'),
                                         script=opts.get('script')))
    if kind == 'codex':
        from .llm_codex import CodexCliClient
        return CodexCliClient(**opts)
    if kind not in CLIENT_KINDS:
        raise LLMError(f"неизвестный вид клиента «{kind}»; есть: {', '.join(CLIENT_KINDS)}")
    cache = opts.pop('cache', False)
    if kind == 'ollama':
        client = ChatClient(**{'base_url': os.environ.get('DID_OLLAMA_URL') or OLLAMA_URL, 'api_key': 'ollama',
                               'model': os.environ.get('DID_OLLAMA_MODEL') or OLLAMA_MODEL, 'timeout_s': 120,
                               'max_retries': 1, 'use_schema': True, **opts})
    else:
        client = ChatClient.from_env(**opts)
        if client is None:
            raise LLMError('не задан DID_LLM_BASE_URL: укажите адрес модели в .env или возьмите kind=mock')
    if cache:
        from .llm_cache import CachedClient
        client = CachedClient(client)
    return client


# --- проверка связи ----------------------------------------------------------------------------

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Проверка связи с моделью: план для одного состояния.')
    ap.add_argument('--state', help='файл состояния в JSON; по умолчанию пример из did.llm')
    ap.add_argument('--kind', default='http', choices=CLIENT_KINDS, help='какой клиент проверять (http — из .env)')
    ap.add_argument('--model', help='модель вместо заданной по умолчанию')
    args = ap.parse_args(argv)
    try:
        client = make_client(args.kind, **({'model': args.model} if args.model else {}))
    except LLMError as e:
        print(f'ошибка настроек: {e}')
        if args.kind == 'http':
            print(f'нужны DID_LLM_BASE_URL, DID_LLM_API_KEY, DID_LLM_MODEL в окружении или в {ROOT / ".env"}')
        return 2
    print(client)
    if hasattr(client, 'models'):
        try:
            print('модели:', ', '.join(client.models()))
        except LLMError as e:
            print(f'список моделей недоступен: {e}')
    state = json.loads(Path(args.state).read_text(encoding='utf-8')) if args.state else EXAMPLE_STATE
    result = request_plan(client, state)
    if hasattr(client, 'close'):
        client.close()
    for ex in result.exchanges:
        print(f"обмен {ex['attempt']}: ok={ex['ok']}, {ex['latency_ms']} мс, запросов {ex['http_attempts']}, "
              f"ошибки: {ex['errors'] or 'нет'}\n  ответ: {_brief(ex['response'], 400)}")
    if result.plan is None:
        print(f'плана нет: {result.error}')
        return 1
    print(json.dumps(result.plan.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
