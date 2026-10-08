#!/usr/bin/env python3
"""Текст условия задачи с вопросами на полях: выделил место — задал вопрос — ответ появляется рядом.

    python3 tools/tz_notes.py            # http://127.0.0.1:8792

Текст берётся из research/task_statement.txt как есть (у каждой строки свой номер). Вопросы и ответы лежат в
research/tz_questions.json: страница пишет туда вопросы, ответы дописываются в тот же файл и подхватываются страницей.
Слушает только этот компьютер.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / 'research' / 'task_statement.txt'
NOTES = ROOT / 'research' / 'tz_questions.json'
PORT = 8792

# Таблицы исходного документа в текстовом файле разложены по строкам: (первая строка, последняя, столбцов, первая ячейка).
TABLES = [(11, 25, 3, 'Что'), (28, 57, 3, 'Топик / сервис'), (60, 75, 4, 'Сценарий'), (77, 94, 3, 'Уровень'),
          (112, 134, 3, 'Критерий'), (139, 150, 2, 'Активность'), (152, 157, 2, 'Активность')]
HEADINGS = {'1. Постановка задачи', 'Легенда', 'Среда и исходные данные', 'Интерфейс агента (ROS 2) стандартный для бота',
            'Сценарии для реализации', 'Уровни решения задачи', 'Тайминг', 'Бонусные треки', '2. Критерии оценивания',
            '3. Связь с мастер-классами DID2026 и дисциплинам учебных планов ТОП ИТ', 'Инфраструктура разработки', 'Ссылки',
            'День 1', 'День 2', 'День 1 Рекомендуемые активности DID2026', 'День 2 Рекомендуемые активности DID2026',
            'Робот, мир и карта', 'Прямые ссылки на файлы, которые использует задача', 'Gazebo, ROS 2, навигация',
            'Рекомендуемый состав команд:'}


def lines():
    return TEXT.read_text(encoding='utf-8').split('\n')


def notes():
    try:
        return json.loads(NOTES.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


def blocks():
    """Строки условия, сгруппированные для показа: заголовки, абзацы и таблицы. Номера строк — как в файле, с единицы."""
    src = lines()
    tables = {a: (b, cols) for a, b, cols, head in TABLES if a <= len(src) and src[a - 1].strip() == head}
    out, n = [], 1
    while n <= len(src):
        text = src[n - 1].strip()
        if n in tables:
            end, cols = tables[n]
            cells = [{'n': k, 't': src[k - 1].strip()} for k in range(n, min(end, len(src)) + 1)]
            out.append({'kind': 'table', 'cols': cols, 'cells': cells})
            n = end + 1
            continue
        if text:
            kind = 'title' if n <= 2 else 'head' if text in HEADINGS else 'p'
            out.append({'kind': kind, 'n': n, 't': text})
        n += 1
    return out


class Handler(BaseHTTPRequestHandler):

    def _send(self, body, ctype='application/json; charset=utf-8', code=200):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith('/api/state'):
            return self._send(json.dumps({'blocks': blocks(), 'questions': notes()}, ensure_ascii=False))
        if self.path.startswith('/check'):          # сверка с требованиями — отдельная страница рядом с текстом условия
            return self._send((ROOT / 'docs' / 'requirements_check.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path.split('?')[0] in ('/', '/index.html'):
            return self._send(PAGE, 'text/html; charset=utf-8')
        self._send('{}', code=404)

    def do_POST(self):
        if not self.path.startswith('/api/questions'):
            return self._send('{}', code=404)
        try:
            items = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)).decode('utf-8'))
            assert isinstance(items, list)
        except (ValueError, AssertionError):
            return self._send('{"ok": false}', code=400)
        answers = {q.get('id'): q.get('answer') for q in notes() if q.get('answer')}   # ответ, дописанный в файл, не затирать
        for q in items:
            if not q.get('answer') and answers.get(q.get('id')):
                q['answer'] = answers[q['id']]
        NOTES.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
        self._send('{"ok": true}')

    def log_message(self, *args):
        pass


PAGE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Условие задачи DID Hack — с вопросами на полях</title>
<style>
 :root { --fg: #14181c; --bg: #fff; --mut: #5b6570; --line: rgba(127,127,127,.28); --card: rgba(127,127,127,.09); --acc: #eb6834; --ans: #1baf7a; }
 @media (prefers-color-scheme: dark) { :root { --fg: #e9ecef; --bg: #15171a; --mut: #9aa3ad; --card: rgba(255,255,255,.06); } }
 body { margin: 0; font: 15.5px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); }
 .bar { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); padding: 8px 16px; font-size: 13.5px; color: var(--mut); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
 .bar b { color: var(--fg); } .bar button { margin-left: auto; }
 main { max-width: 860px; margin: 0 auto; padding: 10px 16px 120px; }
 .hint { background: var(--card); border-radius: 10px; padding: 10px 14px; font-size: 14px; margin: 10px 0 18px; }
 .ln { position: relative; padding-left: 34px; margin: 7px 0; }
 .ln::before { content: attr(data-n); position: absolute; left: 0; top: 2px; width: 26px; text-align: right; font: 11px ui-monospace, Menlo, monospace; color: var(--mut); opacity: .7; user-select: none; }
 .title { font-size: 20px; font-weight: 700; margin: 14px 0; } .head { font-size: 17px; font-weight: 700; margin: 26px 0 8px; }
 table { width: 100%; border-collapse: collapse; margin: 10px 0 14px; font-size: 14.5px; table-layout: fixed; }
 th, td { text-align: left; vertical-align: top; padding: 7px 8px 7px 30px; border: 1px solid var(--line); position: relative; word-break: break-word; }
 th { background: var(--card); font-weight: 650; }
 td::before, th::before { content: attr(data-n); position: absolute; left: 3px; top: 8px; width: 22px; text-align: right; font: 10.5px ui-monospace, Menlo, monospace; color: var(--mut); opacity: .7; user-select: none; }
 .has-q { background: rgba(235,104,52,.10); box-shadow: inset 3px 0 0 var(--acc); }
 .q { margin: 8px 0 14px 34px; border: 1px solid var(--acc); border-radius: 10px; padding: 10px 12px; font-size: 14.5px; background: var(--bg); }
 .q .quote { color: var(--mut); border-left: 3px solid var(--line); padding-left: 8px; margin: 0 0 6px; font-size: 13.5px; }
 .q .who { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--acc); margin-top: 6px; }
 .q .who.a { color: var(--ans); } .q .wait { color: var(--mut); font-style: italic; }
 .q .tools { margin-top: 8px; display: flex; gap: 8px; } .q p { margin: 4px 0; white-space: pre-wrap; }
 button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 7px; border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }
 button.main { background: var(--acc); border-color: var(--acc); color: #fff; font-weight: 650; }
 #ask { position: absolute; z-index: 9; display: none; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
 textarea { width: 100%; box-sizing: border-box; min-height: 70px; font: inherit; font-size: 14.5px; padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
</style></head><body>
<div class="bar"><span><b>Условие задачи DID Hack</b> · вопросов: <b id="count">0</b>, с ответом: <b id="answered">0</b></span><a href="/check" style="color:var(--acc)">Сверка с требованиями →</a><button id="copy">Скопировать все вопросы</button></div>
<main>
<div class="hint"><b>Как пользоваться.</b> Выделите мышью любое место текста — рядом появится кнопка «Вопрос к этому месту». Напишите вопрос и сохраните: он останется под выделенной строкой. Когда вопросы готовы, напишите в чате любое слово (например, «вопросы») — ассистент прочитает их отсюда и ответит здесь же, под каждым вопросом. Цифры слева — номера строк текста: на них удобно ссылаться.</div>
<div id="text"></div>
</main>
<button id="ask" class="main">Вопрос к этому месту</button>
<script>
let S = { blocks: [], questions: [] }, draft = null, shown = '';
const $ = id => document.getElementById(id);
const esc = t => String(t ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
const save = () => fetch('/api/questions', { method: 'POST', body: JSON.stringify(S.questions) });

function render() {
  const marked = new Set(); S.questions.forEach(q => { for (let n = q.from; n <= q.to; n++) marked.add(n); });
  const cls = n => marked.has(n) ? ' has-q' : '';
  const after = n => S.questions.filter(q => q.to === n).map(card).join('') + (draft && draft.to === n ? form() : '');
  let h = '';
  for (const b of S.blocks) {
    if (b.kind === 'table') {
      const rows = []; for (let i = 0; i < b.cells.length; i += b.cols) rows.push(b.cells.slice(i, i + b.cols));
      h += '<table>' + rows.map((r, i) => '<tr>' + r.map(c => `<${i ? 'td' : 'th'} class="cell${cls(c.n)}" data-n="${c.n}">${esc(c.t)}</${i ? 'td' : 'th'}>`).join('') + '</tr>').join('') + '</table>';
      h += b.cells.map(c => after(c.n)).join('');
    } else h += `<div class="ln ${b.kind}${cls(b.n)}" data-n="${b.n}">${esc(b.t)}</div>` + after(b.n);
  }
  $('text').innerHTML = h;
  $('count').textContent = S.questions.length; $('answered').textContent = S.questions.filter(q => q.answer).length;
  const ta = document.querySelector('#draft'); if (ta) { ta.value = draft.text || ''; ta.focus(); ta.oninput = () => draft.text = ta.value; }
}
const where = q => q.from === q.to ? `строка ${q.from}` : `строки ${q.from}–${q.to}`;
function card(q) {
  return `<div class="q" data-id="${q.id}"><div class="quote">${esc(where(q))}: «${esc(q.quote)}»</div>
   <div class="who">Вопрос</div><p>${esc(q.question)}</p>
   <div class="who a">Ответ</div>${q.answer ? `<p>${esc(q.answer)}</p>` : '<p class="wait">ответа пока нет — напишите в чате, что вопросы готовы</p>'}
   <div class="tools"><button onclick="copyOne('${q.id}')">Скопировать для чата</button><button onclick="drop('${q.id}')">Удалить</button></div></div>`;
}
const form = () => `<div class="q"><div class="quote">${esc(where(draft))}: «${esc(draft.quote)}»</div><textarea id="draft" placeholder="Что здесь непонятно или как вы это понимаете?"></textarea>
  <div class="tools"><button class="main" onclick="commit()">Сохранить вопрос</button><button onclick="draft = null; render()">Отмена</button></div></div>`;
function commit() {
  const text = (draft.text || '').trim(); if (!text) return;
  S.questions.push({ id: 'q' + Date.now().toString(36), from: draft.from, to: draft.to, quote: draft.quote, question: text, answer: '', created: new Date().toISOString() });
  draft = null; render(); save();
}
function drop(id) { S.questions = S.questions.filter(q => q.id !== id); render(); save(); }
const asText = q => `Вопрос по условию, ${where(q)}: «${q.quote}»\n${q.question}`;
function copyOne(id) { navigator.clipboard.writeText(asText(S.questions.find(q => q.id === id))); }
$('copy').onclick = () => navigator.clipboard.writeText(S.questions.map(asText).join('\n\n'));

const lineOf = node => { const el = (node.nodeType === 1 ? node : node.parentElement).closest('[data-n]'); return el ? +el.dataset.n : null; };
document.addEventListener('mouseup', e => {
  if (e.target.closest('#ask, .q')) return;
  setTimeout(() => {
    const sel = window.getSelection(), btn = $('ask'), text = sel.toString().trim();
    if (!text || sel.rangeCount === 0 || !$('text').contains(sel.anchorNode)) { btn.style.display = 'none'; return; }
    const a = lineOf(sel.anchorNode), b = lineOf(sel.focusNode); if (a == null || b == null) { btn.style.display = 'none'; return; }
    const r = sel.getRangeAt(0).getBoundingClientRect();
    btn.style.display = 'block'; btn.style.top = (window.scrollY + r.bottom + 6) + 'px'; btn.style.left = Math.max(8, Math.min(window.scrollX + r.left, window.innerWidth - 220)) + 'px';
    btn.onclick = () => { draft = { from: Math.min(a, b), to: Math.max(a, b), quote: text.replace(/\s+/g, ' ').slice(0, 600), text: '' }; btn.style.display = 'none'; sel.removeAllRanges(); render(); };
  }, 0);
});
async function load() {
  const r = await (await fetch('/api/state')).json();
  const key = JSON.stringify(r.questions);
  if (!S.blocks.length) { S = r; shown = key; render(); return; }
  if (key !== shown && !draft && !window.getSelection().toString()) { S.questions = r.questions; shown = key; render(); }   // пришёл ответ
}
load(); setInterval(load, 4000);
</script></body></html>
'''


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Условие задачи с вопросами: http://127.0.0.1:{port}  (Ctrl+C — остановить)')
    server.serve_forever()


if __name__ == '__main__':
    main()
