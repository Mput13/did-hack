// Примеры новых данных для отладки вида страницы: подмешиваются в любую запись прогона флагом в адресе,
//   #/run?file=…&demo=inquiry        расследования (три исхода) и сводка по ним в итоге прогона
//   #/run?file=…&demo=pose           поправка положения по лидару
//   #/run?file=…&demo=inquiry,pose   всё сразу
// Настоящие данные флаг не портит: меняется только копия записи на странице.

const round = (v, d = 2) => Number(v.toFixed(d));

function at(track, t) {
  const T = (track && track.t) || [];
  let i = 0;
  while (i < T.length - 1 && T[i + 1] <= t) i += 1;
  return T.length ? { x: track.x[i], y: track.y[i] } : { x: 0, y: 0 };
}

function inquiries(trace) {
  const T = (trace.track && trace.track.t) || [];
  const t0 = T.length ? T[0] : 0;
  const dur = T.length ? T[T.length - 1] : 100;
  const span = Math.max(dur - t0, 30);
  const when = (f) => round(t0 + span * f, 1);
  const a = when(0.32);
  const b = when(0.55);
  const c = when(0.78);
  const pa = at(trace.track, a);
  const pb = at(trace.track, b);
  const pc = at(trace.track, c);
  return [
    {
      id: 'Q1', t_open: a, t_close: round(a + 6.7, 1), topic: 'energy',
      anomaly: { text: 'расход вырос: 6,8 ед/м при прогнозе 2,6', x: pa.x, y: pa.y, observed: 6.8, expected: 2.6, unit: 'ед/м' },
      alternatives: [
        { id: 'soil', statement: 'здесь дорогой грунт', prior: 0.5, posterior: 0.03 },
        { id: 'leak', statement: 'батарея теряет заряд сама по себе (сбой)', prior: 0.3, posterior: 0.95 },
        { id: 'turn', statement: 'заряд ушёл на повороты', prior: 0.2, posterior: 0.02 },
      ],
      tests: [
        { id: 'rest', name: 'постоять 2 секунды', cost: 0.05, duration_s: 2.0, gain_bits: 0.9, unit: 'ед/с',
          predictions: { soil: { mean: 0.01, sigma: 0.02 }, leak: { mean: 0.15, sigma: 0.03 }, turn: { mean: 0.01, sigma: 0.02 } },
          chosen: true, measured: { value: 0.14, sigma: 0.02, t: round(a + 2.8, 1) } },
        { id: 'straight', name: 'проехать прямо 0,3 м', cost: 0.8, duration_s: 2.0, gain_bits: 0.4, unit: 'ед/м',
          predictions: { soil: { mean: 6.5, sigma: 0.8 }, leak: { mean: 3.9, sigma: 0.5 }, turn: { mean: 2.5, sigma: 0.3 } }, chosen: false },
      ],
      conclusion: { status: 'identified', best: 'leak', confidence: 0.95, text: 'батарея теряет около 0,14 ед/с независимо от движения' },
      action: 'заложил в запас на возврат 2,1 ед.; маршрут не меняю',
      source: 'rule', critique: [{ issue: 'не проверено, что утечка не связана с местом', resolved: true }],
      truth: ['leak'], verdict: 'correct',
    },
    {
      id: 'Q2', t_open: b, t_close: round(b + 5.2, 1), topic: 'sensor',
      anomaly: { text: 'датчик образцов десять раз подряд показал одно и то же значение, хотя робот едет', x: pb.x, y: pb.y },
      alternatives: [
        { id: 'stuck', statement: 'датчик залип и повторяет одно значение (сбой)', prior: 0.45, posterior: 0.52 },
        { id: 'ok', statement: 'датчик исправен: рядом просто нет образцов', prior: 0.35, posterior: 0.41 },
        { id: 'noise', statement: 'датчик стал шуметь сильнее (сбой)', prior: 0.1, posterior: 0.02 },
        { id: 'other', statement: 'причина не из этого списка', prior: 0.1, posterior: 0.05 },
      ],
      tests: [
        { id: 'listen_std', name: 'постоять 2 секунды и измерить разброс показаний', cost: 0.02, duration_s: 2.0, gain_bits: 0.31, unit: 'разброс',
          predictions: { stuck: { mean: 0.0, sigma: 0.02 }, ok: { mean: 0.03, sigma: 0.03 }, noise: { mean: 0.14, sigma: 0.04 } },
          chosen: true, measured: { value: 0.012, sigma: 0.012, t: round(b + 2.4, 1) } },
        { id: 'move', name: 'отъехать на 0,5 м и сравнить показание', cost: 1.4, duration_s: 4.0, gain_bits: 0.62, unit: 'сдвиг',
          predictions: { stuck: { mean: 0.0, sigma: 0.01 }, ok: { mean: 0.06, sigma: 0.05 }, noise: { mean: 0.06, sigma: 0.15 } }, chosen: false },
      ],
      conclusion: { status: 'insufficient', best: 'stuck', confidence: 0.52,
        text: 'недостаточно данных: вероятнее всего «датчик залип» (52%), но не исключено «датчик исправен» (41%); на второй опыт не хватает запаса заряда' },
      action: 'причина не названа: показаниям датчика пока доверяю вполовину и не собираю образцы вслепую',
      source: 'llm', critique: [{ issue: 'второй опыт различил бы объяснения, но он стоит 1,4 ед. заряда', resolved: false }],
      truth: ['stuck'], verdict: 'insufficient',
    },
    {
      id: 'Q3', t_open: c, t_close: round(c + 4.1, 1), topic: 'fault',
      anomaly: { text: 'после штрафа в опасной зоне проверяю, не начался ли сбой', x: pc.x, y: pc.y },
      alternatives: [
        { id: 'none', statement: 'штраф прошёл без последствий', prior: 0.4, posterior: 0.93 },
        { id: 'leak', statement: 'после штрафа батарея начала терять заряд', prior: 0.3, posterior: 0.02 },
        { id: 'noise', statement: 'после штрафа датчик образцов стал шуметь', prior: 0.3, posterior: 0.05 },
      ],
      tests: [
        { id: 'rest', name: 'постоять 2 секунды и измерить расход на месте', cost: 0.06, duration_s: 2.0, gain_bits: 0.71, unit: 'ед/с',
          predictions: { none: { mean: 0.01, sigma: 0.012 }, leak: { mean: 0.16, sigma: 0.05 }, noise: { mean: 0.01, sigma: 0.012 } },
          chosen: true, measured: { value: 0.008, sigma: 0.015, t: round(c + 2.2, 1) } },
        { id: 'listen_std', name: 'за ту же паузу измерить разброс показаний датчика', cost: 0.0, duration_s: 2.0, gain_bits: 0.66, unit: 'разброс',
          predictions: { none: { mean: 0.05, sigma: 0.02 }, leak: { mean: 0.05, sigma: 0.02 }, noise: { mean: 0.22, sigma: 0.07 } },
          chosen: true, measured: { value: 0.046, sigma: 0.012, t: round(c + 2.2, 1) } },
      ],
      conclusion: { status: 'identified', best: 'none', confidence: 0.93, text: 'штраф прошёл без последствий для батареи и датчика' },
      action: 'запас на возврат не увеличиваю; зону запомнил и объезжаю',
      source: 'rule', critique: [], truth: ['none'], verdict: 'correct',
    },
  ];
}

function poseFix(trace) {
  const T = (trace.track && trace.track.t) || [];
  if (T.length < 2) return [];
  const t0 = T[0];
  const dur = T[T.length - 1];
  const out = [];
  let dx = 0;
  let dy = 0;
  let dth = 0;
  for (let t = t0, k = 0; t <= dur; t += 1.1, k += 1) {
    // плавный уход одометрии с редкими скачками на разворотах
    dx += 0.0011 * Math.sin(k * 0.21) + (k % 23 === 11 ? 0.03 : 0);
    dy += 0.0016 * Math.cos(k * 0.13) + (k % 31 === 17 ? -0.04 : 0);
    dth += (k % 23 === 11 ? 0.035 : 0.0004 * Math.sin(k * 0.3));
    out.push({ t: round(t, 2), dx: round(dx, 4), dy: round(dy, 4), dth: round(dth, 4) });
  }
  return out;
}

/** Копия записи с примерами. kinds — список из адреса: 'inquiry', 'pose'. */
export function applyDemo(trace, kinds) {
  const out = { ...trace, result: { ...(trace.result || {}) } };
  if (kinds.includes('inquiry')) {
    out.inquiries = inquiries(trace);
    out.result.inquiries = { total: 3, identified: 2, insufficient: 1, correct: 2, wrong: 0, tests: 4, energy: 0.13 };
    out.result.knowledge = { used: 4, confirmed: 3, learned: 1 };
    out.energy_model = {
      per_m: { value: 2.49, sigma: 0.05, label: 'заряд на метр обычного пола' },
      per_rad: { value: 0.128, sigma: 0.009, label: 'заряд на радиан поворота' },
      per_s: { value: 0.011, sigma: 0.004, label: 'заряд за секунду простоя' },
    };
  }
  if (kinds.includes('pose')) out.pose_fix = poseFix(trace);
  return out;
}

/** Какие примеры просит адрес страницы. */
export function demoKinds(query) {
  return String((query && query.demo) || '').split(',').map((x) => x.trim()).filter((x) => x === 'inquiry' || x === 'pose');
}
