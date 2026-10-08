"""Jev Router (TypeSafe) через OpenRouter — модель, которая сама выбирает исполнителя под каждый запрос.

Это единственная платная модель проекта, поэтому клиент нарочно узкий:
- обращается только к MODEL; любую другую модель отвергает;
- ведёт учёт расходов в runs/_llm_budget/openrouter.json и перестаёт отвечать на потолке;
- ключ берёт из окружения DID_OPENROUTER_API_KEY или из файла, путь к которому задан в
  DID_OPENROUTER_KEY_FILE (строка OPENROUTER_API_KEY=...). В репозиторий ключ не попадает и не печатается.

    DID_OPENROUTER_KEY_FILE=/путь/к/файлу pixi run python -m did.llm_openrouter           # проверка связи, один запрос
    DID_OPENROUTER_KEY_FILE=... pixi run python -m did.llm_eval --kind openrouter --cache   # шесть сценариев
    pixi run python -m did.llm_openrouter --budget                                          # сколько потрачено
"""
import fcntl
import json
import os
import time
from pathlib import Path

from . import ROOT
from .llm import ChatClient, LLMError

BASE_URL = 'https://openrouter.ai/api/v1'
MODEL = 'typesafe/jev-router'
LEDGER = ROOT / 'runs' / '_llm_budget' / 'openrouter.json'
DEFAULT_BUDGET_USD = 5.0


def _key():
    key = os.environ.get('DID_OPENROUTER_API_KEY', '').strip()
    path = os.environ.get('DID_OPENROUTER_KEY_FILE', '').strip()
    if not key and path:
        try:
            for line in Path(path).read_text(encoding='utf-8').splitlines():
                name, _, value = line.partition('=')
                if name.strip() == 'OPENROUTER_API_KEY':
                    key = value.strip().strip('"').strip("'")
        except OSError as e:
            raise LLMError(f'файл с ключом OpenRouter не читается: {type(e).__name__}') from None
    if not key:
        raise LLMError('нет ключа OpenRouter: задайте DID_OPENROUTER_KEY_FILE (путь к файлу со строкой '
                       'OPENROUTER_API_KEY=...) или DID_OPENROUTER_API_KEY')
    return key


class Ledger:
    """Учёт расходов. Несколько процессов могут писать одновременно."""

    def __init__(self, path=LEDGER):
        self.path = Path(path)

    def read(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'spent_usd': 0.0, 'calls': 0, 'routed': {}, 'tokens': {'prompt': 0, 'completion': 0}}

    def add(self, cost, routed, usage):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix('.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self.read()
            data['spent_usd'] = round(data['spent_usd'] + float(cost or 0.0), 6)
            data['calls'] += 1
            data['routed'][routed or '?'] = data['routed'].get(routed or '?', 0) + 1
            for src, dst in (('prompt_tokens', 'prompt'), ('completion_tokens', 'completion')):
                data['tokens'][dst] += int(usage.get(src) or 0)
            data['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
            return data


class JevClient(ChatClient):
    """ChatClient, привязанный к одной модели и к потолку расходов."""

    def __init__(self, budget_usd=None, timeout_s=90, max_retries=1, ledger=None, **opts):
        model = opts.pop('model', MODEL)
        if model != MODEL:
            raise LLMError(f'этот клиент обращается только к {MODEL}; модель «{model}» не разрешена')
        opts.pop('base_url', None)
        opts.pop('api_key', None)
        super().__init__(BASE_URL, _key(), MODEL, timeout_s=timeout_s, max_retries=max_retries,
                         extra_body={'usage': {'include': True}, **(opts.pop('extra_body', None) or {})}, **opts)
        self.budget_usd = float(budget_usd if budget_usd is not None
                                else os.environ.get('DID_OPENROUTER_BUDGET_USD', DEFAULT_BUDGET_USD))
        self.ledger = ledger or Ledger()
        self._raw = None

    def _post(self, content):
        code, headers, body = super()._post(content)
        self._raw = body if 200 <= code < 300 else None
        return code, headers, body

    def chat(self, messages, temperature=0.2, max_tokens=800, schema=None):
        spent = self.ledger.read()['spent_usd']
        if spent >= self.budget_usd:
            raise LLMError(f'потолок расходов на Jev исчерпан: потрачено {spent:.2f} из {self.budget_usd:.2f} долл.')
        self._raw = None
        reply = super().chat(messages, temperature=temperature, max_tokens=max_tokens, schema=schema)
        routed, cost = None, 0.0
        try:
            data = json.loads(self._raw or '{}')
            routed = data.get('model')
            cost = float((data.get('usage') or {}).get('cost') or 0.0)
        except (ValueError, TypeError):
            pass
        reply.usage = {**(reply.usage or {}), 'routed_model': routed, 'cost_usd': cost}
        self.ledger.add(cost, routed, reply.usage)
        return reply


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--budget', action='store_true', help='показать учёт расходов и выйти')
    args = ap.parse_args(argv)
    if args.budget:
        print(json.dumps(Ledger().read(), ensure_ascii=False, indent=1))
        return 0
    client = JevClient()
    reply = client.chat([{'role': 'system', 'content': 'Отвечай одним JSON-объектом.'},
                         {'role': 'user', 'content': 'Верни {"ok": true, "word": "привет"}'}], max_tokens=60)
    print(f"ответ: {reply.text.strip()[:120]}")
    print(f"исполнитель: {reply.usage.get('routed_model')}, цена запроса {reply.usage.get('cost_usd'):.5f} долл., "
          f"время {reply.latency_ms / 1000:.1f} с")
    print('всего потрачено:', Ledger().read()['spent_usd'], 'долл.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
