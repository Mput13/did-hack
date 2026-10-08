"""GPT по подписке через Codex CLI: тот же chat(), ответ по JSON-схеме, кэш на диске.

Codex — агент для работы с кодом, поэтому каждый вызов — отдельный процесс `codex exec`: пустая
временная папка, песочница «только чтение», без записи сессии и без настроек пользователя, а в
запросе сказано ничего не запускать. Накладные расходы оболочки — около 15 тыс. токенов и 8–10 с
на вызов; температуру и лимит токенов Codex CLI задать не даёт.

Квота подписки конечна. Каждый ответ кладётся в runs/_llm_cache (см. did/llm_cache.py), настоящие
вызовы считаются там же; потолок — max_calls или DID_CODEX_MAX_CALLS, по умолчанию его нет.
Настройки: DID_CODEX_MODEL (gpt-6-luna), DID_CODEX_EFFORT (low), DID_CODEX_TIMEOUT_S (120),
DID_CODEX_BIN (codex).

Проверка связи: python -m did.llm --kind codex
"""
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .llm import ChatReply, LLMError, _env_number, _ms
from .llm_cache import ReplyCache

log = logging.getLogger('did.llm')

MODEL = 'gpt-6-luna'
PREFACE = ('Ты работаешь как компонент программы, а не как помощник в чате. Не запускай команды, не читай '
           'и не создавай файлы, не задавай вопросов. Твой ответ — одно итоговое сообщение: ')
# Ошибки, при которых повтор бесполезен: неверный запрос, нет входа в подписку, кончилась квота.
_FATAL = ('invalid_request', 'unsupported', 'not supported', 'usage limit', 'unauthorized', 'not logged in',
          'please login', '"status":400', '"status":401', '"status":403', '"status":404')
_SECRET = re.compile(r'sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_.-]+|(?i:bearer)\s+\S+')
_NOISE = ('Reading additional input from stdin', 'failed to refresh available models')


def _safe(text, limit=300):
    """Текст для исключений и логов: одна строка, без похожего на ключи, не длиннее limit."""
    text = _SECRET.sub('***', ' '.join(str(text).split()))
    return text if len(text) <= limit else text[:limit] + '…'


def render_prompt(messages, structured=True):
    """Переписка одним текстом: у `codex exec` один входной запрос."""
    tags = {'system': 'инструкция', 'user': 'запрос', 'assistant': 'твой_прежний_ответ'}
    parts = [PREFACE + ('один JSON-объект по заданной схеме.' if structured else 'только текст ответа.')]
    try:
        for m in messages:
            tag = tags.get(m['role'], 'запрос')
            if not isinstance(m['content'], str):
                raise TypeError
            parts.append(f"<{tag}>\n{m['content'].strip()}\n</{tag}>")
    except (TypeError, KeyError):
        raise LLMError('codex: нужен список сообщений {role, content} с текстом') from None
    return '\n\n'.join(parts)


def _kill(proc):
    """Остановить процесс вместе с потомками: codex запускает помощников."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except Exception:                        # noqa: BLE001 — процесс уже убит, вывод не нужен
        pass


def _events(stdout):
    out = []
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


class CodexCliClient:
    """chat() через `codex exec`. Схема ответа — параметр chat() или schema в конструкторе."""

    use_schema = True                        # request_plan и роли передают сюда схему ответа

    def __init__(self, model=None, effort=None, timeout_s=None, schema=None, cache_dir=None, use_cache=True,
                 max_calls=None, max_retries=1, retry_pause_s=2.0, binary=None, isolated=True, extra_args=()):
        env = os.environ
        self.model = model or env.get('DID_CODEX_MODEL', '').strip() or MODEL
        self.effort = effort or env.get('DID_CODEX_EFFORT', '').strip() or 'low'
        self.timeout_s = float(timeout_s if timeout_s is not None else _env_number('DID_CODEX_TIMEOUT_S', 120.0, float))
        self.schema = schema
        self.cache = ReplyCache(cache_dir)
        self.use_cache = bool(use_cache)
        self.max_calls = max_calls if max_calls is not None else _env_number('DID_CODEX_MAX_CALLS', None, int)
        self.max_retries = max(0, int(max_retries))
        self.retry_pause_s = float(retry_pause_s)
        self.binary = binary or env.get('DID_CODEX_BIN', '').strip() or 'codex'
        self.isolated = bool(isolated)       # не читать config.toml и правила пользователя: быстрее на ~4 с
        self.extra_args = [str(a) for a in extra_args]
        self.calls = self.hits = 0           # за жизнь клиента: настоящие вызовы и ответы из кэша

    def __repr__(self):
        limit = f', потолок вызовов {self.max_calls}' if self.max_calls is not None else ''
        return (f'CodexCliClient(model={self.model}, effort={self.effort}, timeout_s={self.timeout_s:g}, '
                f'кэш {self.cache.root if self.use_cache else "выключен"}{limit})')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        """Один ответ модели; temperature и max_tokens не используются (Codex CLI их не принимает).

        Ответ из кэша помечен cached=True и несёт время исходного вызова. Сбой повторяется
        max_retries раз, кроме бесполезных случаев (неверный запрос, нет входа, квота).
        """
        t0 = time.monotonic()
        schema = schema or self.schema
        prompt = render_prompt(messages, schema is not None)
        key = self.cache.key(self.model, self.effort, prompt, schema)
        hit = self.cache.get(self.model, key) if self.use_cache else None
        if hit:
            self.hits += 1
            return ChatReply(hit['text'], int(hit.get('latency_ms') or 0), dict(hit.get('usage') or {}), 0,
                             cached=True)
        attempts = 0
        while True:
            try:
                self.cache.charge(self.model, self.max_calls)
                attempts += 1
                self.calls += 1
                text, usage = self._run(prompt, schema)
            except LLMError as e:
                e.attempts = attempts
                if attempts > self.max_retries or getattr(e, 'fatal', False) or 'квота' in str(e):
                    raise
                log.info('LLM: codex, попытка %d не удалась (%s), повтор через %.1f с', attempts, e,
                         self.retry_pause_s)
                time.sleep(self.retry_pause_s)
                continue
            reply = ChatReply(text, _ms(t0), usage, attempts)
            if self.use_cache:
                self.cache.put(self.model, key, {
                    'model': self.model, 'effort': self.effort, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'latency_ms': reply.latency_ms, 'usage': usage, 'schema': schema, 'prompt': prompt, 'text': text})
            return reply

    def _command(self, prompt, tmp, schema):
        work = tmp / 'work'                  # пустая папка: модели нечего читать и негде писать
        work.mkdir()
        cmd = [self.binary, 'exec', '-m', self.model, '-c', f'model_reasoning_effort={self.effort}',
               '-s', 'read-only', '--skip-git-repo-check', '--ephemeral', '--json', '--color', 'never',
               '-C', str(work), '-o', str(tmp / 'reply.txt')]
        if self.isolated:
            cmd += ['--ignore-user-config', '--ignore-rules']
        if schema is not None:
            (tmp / 'schema.json').write_text(json.dumps(schema, ensure_ascii=False), encoding='utf-8')
            cmd += ['--output-schema', str(tmp / 'schema.json')]
        return cmd + self.extra_args + [prompt], work

    def _run(self, prompt, schema):
        """(текст, usage) одного запуска codex. Любой сбой — LLMError; e.fatal — повторять бесполезно."""
        with tempfile.TemporaryDirectory(prefix='did-codex-') as tmp:
            tmp = Path(tmp)
            cmd, work = self._command(prompt, tmp, schema)
            try:
                # Без stdin codex ждёт ввод; своя группа процессов — чтобы по таймауту убить всех потомков.
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=work, start_new_session=True)
            except OSError as e:
                err = LLMError(f'codex: программа «{self.binary}» не запускается ({type(e).__name__}); '
                               'установите Codex CLI или задайте DID_CODEX_BIN')
                err.fatal = True
                raise err from None
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                _kill(proc)
                raise LLMError(f'codex: таймаут {self.timeout_s:g} с, процесс остановлен') from None
            except BaseException:            # Ctrl+C и прочее: процесс за собой не оставляем
                _kill(proc)
                raise
            events = _events(stdout.decode('utf-8', 'replace'))
            try:
                text = (tmp / 'reply.txt').read_text(encoding='utf-8')
            except OSError:
                text = ''
        usage, failure, said = {}, None, ''
        for ev in events:
            item = ev.get('item') if isinstance(ev.get('item'), dict) else {}
            if ev.get('type') == 'turn.completed' and isinstance(ev.get('usage'), dict):
                usage = {k: v for k, v in ev['usage'].items() if isinstance(v, (int, float))}
            elif ev.get('type') in ('error', 'turn.failed'):
                error = ev.get('error')
                failure = ev.get('message') or (error.get('message') if isinstance(error, dict) else error) or failure
            elif item.get('type') == 'agent_message':
                said = str(item.get('text') or '')
        text = text if text.strip() else said    # файла ответа нет — берём последнее сообщение модели
        if proc.returncode != 0 or not text.strip():
            tail = [ln for ln in stderr.decode('utf-8', 'replace').splitlines()
                    if ln.strip() and not any(n in ln for n in _NOISE)]
            reason = failure or (tail[-1] if tail else 'пустой ответ')
            err = LLMError(_safe(f'codex: код {proc.returncode}: {reason}'))
            err.fatal = any(mark in str(reason) for mark in _FATAL)
            raise err
        if usage:
            usage = {'prompt_tokens': usage.get('input_tokens', 0), 'completion_tokens': usage.get('output_tokens', 0),
                     'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0), **usage}
        return text.strip(), usage
