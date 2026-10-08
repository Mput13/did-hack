"""Журнал эксперимента: что агент заметил, что предположил, как проверил и что решил."""

KINDS = ('observe', 'hypothesis', 'verdict', 'decision', 'action', 'alarm', 'llm')


class Journal:

    def __init__(self):
        self.entries = []        # [{'t', 'kind', 'text', ...}]
        self.hypotheses = []     # [{'id', 'key', 't_open', 'statement', 'test', 'status', 't_close', 'verdict'}]
        self._by_key = {}

    def add(self, t, kind, text, **data):
        entry = {'t': round(float(t), 1), 'kind': kind, 'text': text}
        if data:
            entry['data'] = data
        self.entries.append(entry)
        return entry

    def open(self, t, key, statement, test, **data):
        """Новая гипотеза. key не даёт завести вторую про то же самое."""
        if key in self._by_key and self._by_key[key]['status'] == 'open':
            return self._by_key[key]
        h = {'id': f'H{len(self.hypotheses) + 1}', 'key': key, 't_open': round(float(t), 1),
             'statement': statement, 'test': test, 'status': 'open', 't_close': None, 'verdict': None}
        if data:
            h['data'] = data
        self.hypotheses.append(h)
        self._by_key[key] = h
        self.add(t, 'hypothesis', f"{h['id']}: {statement}. Проверка: {test}", hypothesis=h['id'])
        return h

    def get(self, key):
        return self._by_key.get(key)

    def close(self, t, key, status, verdict):
        """status: confirmed | refuted | outdated."""
        h = self._by_key.get(key)
        if not h or h['status'] != 'open':
            return None
        h['status'], h['t_close'], h['verdict'] = status, round(float(t), 1), verdict
        word = {'confirmed': 'подтверждена', 'refuted': 'опровергнута', 'outdated': 'устарела'}[status]
        self.add(t, 'verdict', f"{h['id']} {word}: {verdict}", hypothesis=h['id'], status=status)
        return h

    def open_list(self):
        return [h for h in self.hypotheses if h['status'] == 'open']
