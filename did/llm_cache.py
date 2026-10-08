"""Ответы модели на диске: повторный прогон не тратит квоту и повторяет те же ответы.

Папка runs/_llm_cache/<модель>/<хэш>.json — по файлу на ответ; ключ — хэш от модели и запроса.
Там же calls.json — счётчик настоящих (не из кэша) вызовов по моделям: он только растёт и считает все
запуски подряд, поэтому числа попаданий в кэш из него не получить.
Неудачный вызов (пустой ответ, обрыв связи) лежит рядом в <хэш>.fail.json. Обычный режим такие файлы не
читает и спрашивает модель заново; строгий режим «только кэш» (CachedClient(strict=True)) повторяет по ним
тот же отказ, чтобы прогон шёл той же дорогой, и в сеть не ходит вовсе.

    python -m did.llm_cache              # счётчик вызовов и число сохранённых ответов
"""
import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import ROOT
from .llm import CacheMiss, ChatReply, LLMError

CACHE_DIR = ROOT / 'runs' / '_llm_cache'


class ReplyCache:
    """Файлы ответов и счётчик вызовов. Несколько процессов могут работать с одной папкой."""

    def __init__(self, root=None):
        self.root = Path(root or CACHE_DIR)

    @staticmethod
    def key(*parts):
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    @staticmethod
    def folder(model):
        """Имя папки модели: «qwen2.5:3b» -> «qwen2.5-3b»."""
        return re.sub(r'[^\w.-]+', '-', str(model) or 'model')

    def _path(self, model, key):
        return self.root / self.folder(model) / f'{key}.json'

    def get(self, model, key):
        try:
            record = json.loads(self._path(model, key).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) and isinstance(record.get('text'), str) else None

    def put(self, model, key, record):
        path = self._path(model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path)                # другой процесс не увидит недописанный файл

    def _fail_path(self, model, key):
        return self.root / self.folder(model) / f'{key}.fail.json'

    def get_failure(self, model, key):
        """Сохранённый отказ: {'text': ''} — пустой ответ, {'error': ...} — обрыв; None — отказа не было."""
        try:
            record = json.loads(self._fail_path(model, key).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) and ('error' in record or record.get('text') == '') else None

    def put_failure(self, model, key, record):
        path = self._fail_path(model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, path)

    def calls(self):
        """{модель: сколько настоящих вызовов сделано}."""
        try:
            data = json.loads((self.root / 'calls.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def charge(self, model, limit=None):
        """Учесть ещё один настоящий вызов. LLMError, если их уже limit: квоту нельзя превысить молча."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / 'calls.json', 'a+', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                data = json.loads(f.read() or '{}')
            except ValueError:
                data = {}
            n = int(data.get(model, 0))
            if limit is not None and n >= limit:
                raise LLMError(f'квота вызовов {model} исчерпана: сделано {n} из {limit}')
            data[model] = n + 1
            f.seek(0)
            f.truncate()
            f.write(json.dumps(data, ensure_ascii=False, indent=1))
            return n + 1

    def saved(self):
        """{модель: сколько ответов лежит в кэше}."""
        if not self.root.is_dir():
            return {}
        return {d.name: len([p for p in d.glob('*.json') if not p.name.endswith('.fail.json')])
                for d in sorted(self.root.iterdir()) if d.is_dir()}

    def saved_failures(self):
        """{модель: сколько отказов лежит в кэше}."""
        if not self.root.is_dir():
            return {}
        return {d.name: len(list(d.glob('*.fail.json'))) for d in sorted(self.root.iterdir()) if d.is_dir()}


class OfflineClient:
    """Клиент без сети для строгого режима: от него нужны только имя модели и use_schema (они входят в ключ)."""

    def __init__(self, model, use_schema=False):
        self.model = model
        self.use_schema = bool(use_schema)

    def __repr__(self):
        return f'OfflineClient(model={self.model}, сеть запрещена)'

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        raise LLMError('режим «только кэш»: обращение к модели запрещено')


class CachedClient:
    """Кэш поверх любого клиента с chat(): для локальной модели — повторяемость прогонов.

    strict=True — режим «только кэш»: настоящий клиент не вызывается никогда. Сохранённый ответ возвращается,
    сохранённый отказ повторяется, а запрос, которого в кэше нет, — CacheMiss.
    """

    def __init__(self, client, cache_dir=None, strict=False):
        self.client = client
        self.cache = ReplyCache(cache_dir)
        self.model = getattr(client, 'model', '') or type(client).__name__
        self.use_schema = getattr(client, 'use_schema', False)
        self.strict = bool(strict)

    def __repr__(self):
        return f'{self.client!r} + кэш {self.cache.root}' + (' (только кэш)' if self.strict else '')

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        key = self.cache.key(self.model, messages, temperature, max_tokens, schema if self.use_schema else None)
        hit = self.cache.get(self.model, key)
        if hit:
            return ChatReply(hit['text'], int(hit.get('latency_ms') or 0), dict(hit.get('usage') or {}), 0,
                             hit.get('finish_reason'), cached=True)
        if self.strict:
            return self._replay_failure(key, messages)
        try:
            reply = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, schema=schema)
        except LLMError as e:                # обрыв — тоже исход прогона: строгий повтор его воспроизведёт
            self._keep_failure(key, messages, error=str(e), attempts=getattr(e, 'attempts', 0))
            raise
        self.cache.charge(self.model)
        if reply.text.strip():               # пустой ответ не запоминаем: в следующий раз спросим снова
            self.cache.put(self.model, key, {
                'model': self.model, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'latency_ms': reply.latency_ms,
                'usage': reply.usage, 'finish_reason': reply.finish_reason, 'messages': messages, 'text': reply.text})
        else:                                # ...но как исход храним отдельно, для строгого повтора
            self._keep_failure(key, messages, text='', latency_ms=reply.latency_ms, usage=reply.usage,
                               finish_reason=reply.finish_reason, attempts=reply.attempts)
        return reply

    def _keep_failure(self, key, messages, **outcome):
        self.cache.put_failure(self.model, key, {'model': self.model, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                 'messages': messages, **outcome})

    def _replay_failure(self, key, messages):
        fail = self.cache.get_failure(self.model, key)
        if fail is None:
            ask = ' '.join(str(messages[-1].get('content', '')).split())[:160] if messages else ''
            raise CacheMiss(f'только кэш: нет сохранённого ответа {self.model} на запрос {key[:12]} («{ask}…»); '
                            f'сеть не трогаю')
        if 'error' in fail:
            raise LLMError(fail['error'], int(fail.get('attempts') or 0))
        return ChatReply('', int(fail.get('latency_ms') or 0), dict(fail.get('usage') or {}),
                         0, fail.get('finish_reason'), cached=True)

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()


def main():
    cache = ReplyCache()
    calls, saved, failed = cache.calls(), cache.saved(), cache.saved_failures()
    print(f'кэш ответов: {cache.root}')
    for model in sorted(set(calls) | (set(saved) - {cache.folder(m) for m in calls})):
        print(f'  {model}: настоящих вызовов {calls.get(model, 0)}, ответов в кэше {saved.get(cache.folder(model), 0)}, '
              f'отказов в кэше {failed.get(cache.folder(model), 0)}')
    if not calls and not saved:
        print('  пусто')


if __name__ == '__main__':
    main()
