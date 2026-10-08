"""Ответы модели на диске: повторный прогон не тратит квоту и повторяет те же ответы.

Папка runs/_llm_cache/<модель>/<хэш>.json — по файлу на ответ; ключ — хэш от модели и запроса.
Там же calls.json — счётчик настоящих (не из кэша) вызовов по моделям.

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
from .llm import ChatReply, LLMError

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
        return {d.name: len(list(d.glob('*.json'))) for d in sorted(self.root.iterdir()) if d.is_dir()}


class CachedClient:
    """Кэш поверх любого клиента с chat(): для локальной модели — повторяемость прогонов."""

    def __init__(self, client, cache_dir=None):
        self.client = client
        self.cache = ReplyCache(cache_dir)
        self.model = getattr(client, 'model', '') or type(client).__name__
        self.use_schema = getattr(client, 'use_schema', False)

    def __repr__(self):
        return f'{self.client!r} + кэш {self.cache.root}'

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        key = self.cache.key(self.model, messages, temperature, max_tokens, schema if self.use_schema else None)
        hit = self.cache.get(self.model, key)
        if hit:
            return ChatReply(hit['text'], int(hit.get('latency_ms') or 0), dict(hit.get('usage') or {}), 0,
                             hit.get('finish_reason'), cached=True)
        reply = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, schema=schema)
        self.cache.charge(self.model)
        if reply.text.strip():               # пустой ответ не запоминаем: в следующий раз спросим снова
            self.cache.put(self.model, key, {
                'model': self.model, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'latency_ms': reply.latency_ms,
                'usage': reply.usage, 'finish_reason': reply.finish_reason, 'messages': messages, 'text': reply.text})
        return reply

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()


def main():
    cache = ReplyCache()
    calls, saved = cache.calls(), cache.saved()
    print(f'кэш ответов: {cache.root}')
    for model in sorted(set(calls) | (set(saved) - {cache.folder(m) for m in calls})):
        print(f'  {model}: настоящих вызовов {calls.get(model, 0)}, ответов в кэше {saved.get(cache.folder(model), 0)}')
    if not calls and not saved:
        print('  пусто')


if __name__ == '__main__':
    main()
