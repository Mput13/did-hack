#!/usr/bin/env python3
"""Исследовательский контур: рабочие деревья для исследований, проверка сданного, пауза.

    python3 tools/research.py start R1        # ветка research/R1 и дерево ../DID-research/R1 от текущего main
    python3 tools/research.py check R1        # что сдано: отчёт, разница с main, автоматические проверки
    python3 tools/research.py list            # все исследования и их состояние
    python3 tools/research.py pause | resume  # остановить и запретить расчёты исследователей / разрешить снова

Правила для исследователей — research/PROTOCOL.md, план — research/agenda.yaml.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREES = ROOT.parent / 'DID-research'
PAUSE = ROOT / 'research' / 'PAUSE'
SECTIONS = ('## Вопрос', '## Гипотеза', '## Что сделано', '## Результат', '## Вывод', '## Ограничения', '## Как повторить')


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


def start(rid):
    tree = TREES / rid
    if tree.exists():
        print(f'{rid}: дерево уже есть — {tree}')
        return 0
    TREES.mkdir(exist_ok=True)
    git('worktree', 'add', str(tree), '-b', f'research/{rid}', 'main')
    print(f'{rid}: ветка research/{rid}, дерево {tree}')
    return 0


def check(rid):
    tree = TREES / rid
    if not tree.exists():
        print(f'{rid}: дерева нет')
        return 1
    bad = []
    dirty = git('status', '--short', cwd=tree)
    if dirty:
        bad.append('есть незакоммиченные изменения:\n' + dirty)
    ahead = git('rev-list', '--count', f'main..research/{rid}')
    print(f'{rid}: коммитов поверх main — {ahead}')
    print(git('diff', '--stat', f'main...research/{rid}') or '  (изменений нет)')
    report = tree / 'research' / 'findings' / f'{rid}.md'
    if not report.exists():
        bad.append(f'нет отчёта research/findings/{rid}.md')
    else:
        text = report.read_text(encoding='utf-8')
        missing = [s for s in SECTIONS if s not in text]
        if missing:
            bad.append('в отчёте нет разделов: ' + ', '.join(missing))
    protected = git('diff', '--name-only', f'main...research/{rid}', '--', 'runs', 'explain.html', 'presentation', 'demo')
    if protected:
        bad.append('тронуты файлы, которые исследователь менять не должен:\n' + protected)
    tests = subprocess.run([str(tree / 'px'), 'python', '-m', 'pytest', 'tests', '-q'], cwd=tree, capture_output=True, text=True)
    last = [line for line in tests.stdout.strip().splitlines() if line.strip()][-1:] or ['нет вывода']
    print('проверки:', last[0])
    if tests.returncode != 0:
        bad.append('автоматические проверки не проходят')
    for item in bad:
        print('НЕ ПРИНЯТО:', item)
    print('итог:', 'можно принимать' if not bad else f'замечаний {len(bad)}')
    return 1 if bad else 0


def listing():
    out = git('worktree', 'list')
    print(out)
    print('пауза:', 'ДА' if PAUSE.exists() else 'нет')
    return 0


def pause():
    PAUSE.write_text('Исследования приостановлены: идёт показ или репетиция.\n', encoding='utf-8')
    subprocess.run(['pkill', '-f', str(TREES)], capture_output=True)
    subprocess.run(['pkill', '-f', 'tools/gazebo_batch.py'], capture_output=True)
    print('пауза включена; расчёты в деревьях исследований остановлены')
    return 0


def resume():
    PAUSE.unlink(missing_ok=True)
    print('пауза снята')
    return 0


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == 'start' and rest:
        return start(rest[0])
    if cmd == 'check' and rest:
        return check(rest[0])
    if cmd == 'list':
        return listing()
    if cmd == 'pause':
        return pause()
    if cmd == 'resume':
        return resume()
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
