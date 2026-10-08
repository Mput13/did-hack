"""Память между прогонами: проверенные знания о среде с условиями и ссылками на прогоны-доказательства.

После каждого прогона исследователь оставляет то, что выяснил: коэффициенты модели расхода, какие
сбои следуют за штрафом и сколько они длятся, с какой скоростью идёт утечка. Следующий прогон
начинает не с нуля: эти оценки становятся его исходными предположениями (но не истиной — опыт
прогона может их опровергнуть, и тогда правило помечается как спорное).
"""
import json
import math
import time
from pathlib import Path

from . import ROOT

KB_PATH = ROOT / 'runs' / '_knowledge' / 'kb_state.json'     # рядом пишется kb.json — вид для интерфейса
SENSOR = ('noise', 'stuck', 'bias', 'ok')
TEXT = {
    'energy.per_m': ('расход', 'метр по обычному полу стоит', 'ед/м'),
    'energy.per_m_load': ('расход', 'каждый несомый образец удорожает метр на', 'ед/м'),
    'energy.per_rad': ('расход', 'поворот на радиан стоит', 'ед/рад'),
    'energy.per_s': ('расход', 'секунда простоя стоит', 'ед/с'),
    'leak.rate': ('сбой', 'при утечке батарея теряет', 'ед/с'),
    'fault.duration': ('сбой', 'сбой после штрафа длится', 'с'),
}


class KnowledgeBase:

    def __init__(self, path=KB_PATH):
        self.path = Path(path)
        self.data = {'runs': 0, 'updated': None, 'stats': {},
                     # проверки после штрафа: батарея (утечка / в порядке / не выяснено) и отдельно датчик
                     'battery': {'leak': 0, 'none': 0, 'unknown': 0}, 'sensor': {k: 0 for k in SENSOR + ('unknown',)}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))

    # --- накопление --------------------------------------------------------------------------

    def _add(self, key, value, weight, run_id):
        """Ещё одно наблюдение величины: взвешенное среднее, разброс между прогонами, ссылка на прогон."""
        s = self.data['stats'].setdefault(key, {'w': 0.0, 'wx': 0.0, 'wxx': 0.0, 'n': 0, 'support': [],
                                                'contradictions': [], 'first': run_id})
        if s['n'] >= 3:
            mean, sd = s['wx'] / s['w'], self._sd(s)
            if abs(value - mean) > 3.0 * max(sd, 0.05 * abs(mean) + 0.01):
                s['contradictions'] = (s['contradictions'] + [run_id])[-20:]
        s['w'] += weight
        s['wx'] += weight * value
        s['wxx'] += weight * value * value
        s['n'] += 1
        s['support'] = (s['support'] + [run_id])[-30:]
        s['last'] = run_id

    @staticmethod
    def _retired(s):
        """Правило под сомнением: слишком много прогонов ему противоречат. В предположения оно не идёт."""
        return len(s['contradictions']) > max(2, s['n'] // 3)

    @staticmethod
    def _sd(s):
        mean = s['wx'] / s['w']
        return math.sqrt(max(s['wxx'] / s['w'] - mean * mean, 0.0))

    def learn(self, science, run_id):
        """science — то, что исследователь выгрузил в запись прогона (did.inquiry.Investigator.export)."""
        if not science:
            return
        model = science.get('energy_model', {})
        for name, v in model.items():
            # У всех прогонов равный вес: оценка прогона уже опирается на память, и взвешивать её по
            # собственной заявленной точности значило бы учитывать прежние знания повторно.
            if v['sigma'] < 0.06 or name == 'per_s':
                self._add(f'energy.{name}', v['value'], 1.0, run_id)
        per_s = model.get('per_s', {}).get('value', 0.01)
        for kind, seconds in science.get('fault_durations', []):
            self._add('fault.duration', seconds, 1.0, run_id)
        for q in science.get('inquiries', []):
            c = q.get('conclusion') or {}
            known = c.get('status') == 'identified'
            if q['topic'] == 'fault':                         # каждая проверка батареи после штрафа — в счёт
                self.data['battery'][c['best'] if known and c.get('best') in ('leak', 'none') else 'unknown'] += 1
            elif q['topic'] == 'sensor' and q['anomaly'].get('trigger') == 'checkup':
                self.data['sensor'][c['best'] if known and c.get('best') in SENSOR else 'unknown'] += 1
            if known and c['best'] == 'leak':
                rest = next((x['measured'] for x in q['tests'] if x['id'] == 'rest' and x.get('measured')), None)
                if rest:
                    self._add('leak.rate', rest['value'] - per_s, 1.0, run_id)   # без расхода за простой
        self.data['runs'] += 1
        self.data['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    # --- использование -----------------------------------------------------------------------

    def priors(self):
        """Исходные предположения для следующего прогона (did.inquiry.Investigator)."""
        st, out = self.data['stats'], {}
        energy = {}
        for name in ('per_m', 'per_m_load', 'per_rad', 'per_s'):
            s = st.get(f'energy.{name}')
            if s and s['n'] >= 2 and not self._retired(s):
                # не увереннее, чем позволяет разброс между прогонами: знание — предположение, а не истина
                energy[name] = {'value': s['wx'] / s['w'], 'sigma': max(self._sd(s), 0.02 if name != 'per_s' else 0.003)}
        if energy:
            out['energy'] = energy
        s = st.get('fault.duration')
        if s and s['n'] >= 3:
            out['fault_s'] = s['wx'] / s['w']
        s = st.get('leak.rate')
        if s and s['n'] >= 2:
            out['leak_rate'] = {'value': s['wx'] / s['w'], 'sigma': max(self._sd(s), 0.03)}
        # Частоты сбоев: с осторожным исходным предположением и только по проверкам с установленным исходом.
        b = self.data['battery']
        if b['leak'] + b['none'] >= 3:
            out['p_leak_after_penalty'] = (b['leak'] + 1.0) / (b['leak'] + b['none'] + 4.0)
        sn = self.data['sensor']
        total = sum(sn[k] for k in SENSOR)
        if total >= 4:
            out['sensor_priors'] = {k: (sn[k] + 1.0) / (total + len(SENSOR)) * 0.9 for k in SENSOR}
        return out

    # --- для интерфейса ----------------------------------------------------------------------

    def to_dict(self):
        rules = []
        for key, s in sorted(self.data['stats'].items()):
            kind, text, unit = TEXT.get(key, ('прочее', key, ''))
            mean, sd = s['wx'] / s['w'], self._sd(s)
            status = 'retired' if self._retired(s) else 'confirmed' if s['n'] >= 3 else 'tentative'
            rules.append({'id': key, 'kind': kind, 'statement': f'{text} {mean:.2f} {unit}', 'value': round(mean, 4),
                          'sigma': round(sd, 4), 'unit': unit, 'support': s['support'],
                          'contradictions': s['contradictions'], 'status': status, 'n': s['n'],
                          'first_seen': s.get('first'), 'last_seen': s.get('last')})
        b, sn = self.data['battery'], self.data['sensor']
        if sum(b.values()):
            done = b['leak'] + b['none']
            rules.append({'id': 'fault.battery', 'kind': 'сбой',
                          'statement': (f"после штрафа батарея начинает терять заряд примерно в "
                                        f"{b['leak'] / done:.0%} случаев" if done else
                                        'после штрафа батарею проверить пока не удалось'),
                          'value': round(b['leak'] / done, 3) if done else None, 'sigma': 0.0,
                          'unit': f"проверок с ясным исходом: {done}, без ясного: {b['unknown']}",
                          'support': [], 'contradictions': [], 'status': 'confirmed' if done >= 5 else 'tentative',
                          'n': sum(b.values()), 'first_seen': None, 'last_seen': None})
        if sum(sn.values()):
            names = {'noise': 'шум', 'stuck': 'залипание', 'bias': 'занижение показаний', 'ok': 'датчик исправен'}
            done = sum(sn[k] for k in SENSOR)
            share = ', '.join(f'{names[k]} — {sn[k] / done:.0%}' for k in SENSOR if sn[k]) if done else '—'
            rules.append({'id': 'fault.sensor', 'kind': 'сбой',
                          'statement': f'после штрафа с датчиком образцов бывает: {share}',
                          'value': None, 'sigma': 0.0,
                          'unit': f"проверок с ясным исходом: {done}, без ясного: {sn['unknown']}",
                          'support': [], 'contradictions': [], 'status': 'confirmed' if done >= 5 else 'tentative',
                          'n': sum(sn.values()), 'first_seen': None, 'last_seen': None})
        return {'updated': self.data['updated'], 'runs': self.data['runs'], 'rules': rules}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding='utf-8')
        (self.path.parent / 'kb.json').write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding='utf-8')
