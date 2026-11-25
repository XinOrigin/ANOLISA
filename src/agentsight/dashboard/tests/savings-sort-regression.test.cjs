const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { dirname, join } = require('node:path');
const { runInNewContext } = require('node:vm');
const test = require('node:test');
const { nextSavingsSort, sortSavingsSessions } = require(process.env.AGENTSIGHT_SAVINGS_SORT_BUILD);
const { serializeSavingsCsv } = require(process.env.AGENTSIGHT_SAVINGS_CSV_BUILD);
const { fillTokenBuckets, fillModelBuckets } = require(process.env.AGENTSIGHT_TIMESERIES_BUCKETS_BUILD);

const session = (id, input, output, saved, rate) => ({
  session_id: id, agent_name: 'Agent', request_count: 1,
  total_input_tokens: input, total_output_tokens: output, total_tokens: input + output,
  baseline_tokens: input + output + saved, saved_tokens: 0, compounded_saved: saved,
  savings_rate: 0, compounded_savings_rate: rate, tool_saved: 0,
  mcp_saved: 0, optimization_items: [],
});
const rows = [session('a', 10, 300, 60, 0.2), session('b', 30, 100, 20, 0.4), session('c', 20, 200, 40, 0.3)];
const ids = (sessions) => sessions.map((row) => row.session_id);

test('each sortable metric uses its displayed numeric field in both directions', () => {
  for (const [key, descending] of [
    ['total_input_tokens', ['b', 'c', 'a']],
    ['total_output_tokens', ['a', 'c', 'b']],
    ['compounded_saved', ['a', 'c', 'b']],
    ['compounded_savings_rate', ['a', 'c', 'b']],
  ]) {
    assert.deepEqual(ids(sortSavingsSessions(rows, { key, direction: 'descending' })), descending, key);
    assert.deepEqual(ids(sortSavingsSessions(rows, { key, direction: 'ascending' })), [...descending].reverse(), key);
  }
});

test('savings rate follows the displayed ratio when server rates clamp or diverge', () => {
  const clamped = [
    session('clamped-low', 10, 0, 20, 1),
    session('zero-actual', 0, 0, 5, 0),
    session('clamped-high', 10, 0, 40, 1),
  ];
  const descending = ids(sortSavingsSessions(clamped, { key: 'compounded_savings_rate', direction: 'descending' }));
  assert.deepEqual(descending, ['zero-actual', 'clamped-high', 'clamped-low']);
  const ascending = ids(sortSavingsSessions(clamped, { key: 'compounded_savings_rate', direction: 'ascending' }));
  assert.deepEqual(ascending, ['clamped-low', 'clamped-high', 'zero-actual']);
});

test('sort cycles highest, lowest and query order, and changing metric starts highest', () => {
  const descending = nextSavingsSort(null, 'compounded_saved');
  assert.deepEqual(descending, { key: 'compounded_saved', direction: 'descending' });
  const ascending = nextSavingsSort(descending, 'compounded_saved');
  assert.deepEqual(ascending, { key: 'compounded_saved', direction: 'ascending' });
  assert.equal(nextSavingsSort(ascending, 'compounded_saved'), null);
  assert.deepEqual(nextSavingsSort(ascending, 'total_input_tokens'), { key: 'total_input_tokens', direction: 'descending' });
});

test('equal metrics keep query order without mutating frozen snapshots or hits', () => {
  const input = Object.freeze([
    Object.freeze(session('first', 3, 1, 3, 0.2)),
    Object.freeze(session('second', 3, 1, 3, 0.2)),
    Object.freeze(session('small', 1, 1, 1, 0.1)),
  ]);
  const result = sortSavingsSessions(input, { key: 'compounded_saved', direction: 'descending' });
  assert.deepEqual(ids(result), ['first', 'second', 'small']);
  assert.equal(result[0], input[0]);
  assert.notEqual(result, input);
  result.reverse();
  assert.deepEqual(ids(sortSavingsSessions(input, null)), ['first', 'second', 'small']);
  assert.deepEqual(sortSavingsSessions([], null), []);
});

// Exercise the compiled page's actual controls and query/export callbacks.
// Child component elements keep their props, including each SessionRow's hit.
function savingsPage() {
  const states = [];
  const refs = [];
  let stateIndex = 0;
  let refIndex = 0;
  let response = rows;
  let exported;
  let requests = 0;
  const exports = {};
  const t = (key) => key;
  runInNewContext(readFileSync(process.env.AGENTSIGHT_SAVINGS_PAGE_BUILD, 'utf8'), {
    exports,
    URLSearchParams,
    require(name) {
      if (name === 'react') return {
        useState(initial) {
          const index = stateIndex++;
          if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial;
          return [states[index], (value) => { states[index] = typeof value === 'function' ? value(states[index]) : value; }];
        },
        useRef(initial) {
          const index = refIndex++;
          return refs[index] ??= { current: initial };
        },
        useMemo: (callback) => callback(),
        useCallback: (callback) => callback,
        useEffect() {},
      };
      if (name === 'react/jsx-runtime') return {
        jsx: (type, props) => ({ type, props }),
        jsxs: (type, props) => ({ type, props }),
      };
      if (name === 'react-router-dom') return { useSearchParams: () => [new URLSearchParams()] };
      if (name === 'recharts') return {};
      if (name === '../i18n') return { useI18n: () => ({ t }), useLocaleTag: () => 'en-US' };
      if (name === '../components/DateTimePicker') return { DateTimePicker: () => null };
      if (name === '../components/CopyButton') return { CopyButton: () => null };
      if (name === '../components/SessionIdHelp') return { SessionIdHelp: () => null };
      if (name === '../utils/savingsSort') return { nextSavingsSort, sortSavingsSessions };
      if (name === '../utils/savings') return require(join(dirname(process.env.AGENTSIGHT_SAVINGS_SORT_BUILD), 'savings.js'));
      if (name === '../utils/optimizationStrategies') return require(process.env.AGENTSIGHT_STRATEGIES_BUILD);
      if (name === '../utils/savingsCsv') return { downloadSavingsCsv: (snapshot) => { exported = snapshot; } };
      if (name === '../utils/apiClient') return {
        fetchAgentNames: async () => [],
        fetchTokenSavings: async () => {
          requests++;
          return { sessions: response, summary: {}, stats_available: true };
        },
      };
      throw new Error(`Unexpected TokenSavingsPage dependency: ${name}`);
    },
  });
  function render() {
    stateIndex = 0;
    refIndex = 0;
    return exports.TokenSavingsPage();
  }
  function nodes(node, predicate) {
    if (!node || typeof node !== 'object') return [];
    return [...(predicate(node) ? [node] : []),
      ...[node.props?.children].flat(Infinity).flatMap((child) => nodes(child, predicate))];
  }
  const buttons = () => nodes(render(), (node) => node.type === 'button');
  return {
    query: () => buttons().find((node) => node.props.children === 'common.query').props.onClick(),
    sort(label) { buttons().find((node) => node.props['aria-label']?.startsWith(`${label}:`)).props.onClick(); },
    order: () => nodes(render(), (node) => !!node.props?.session).map((node) => node.props.session.session_id),
    aria: () => nodes(render(), (node) => node.type === 'th' && node.props['aria-sort']).map((node) => node.props['aria-sort']),
    export() { buttons().find((node) => node.props.children === 'ts.exportCsv').props.onClick(); return exported; },
    select(id) { nodes(render(), (node) => node.props?.session?.session_id === id)[0].props.onToggleSelected(id); },
    exportSelected() { buttons().find((node) => node.props.children === 'ts.exportSelectedCsv').props.onClick(); return exported; },
    response(value) { response = value; },
    requests: () => requests,
  };
}

test('page header controls reorder rows and export the same snapshot without refetching', async () => {
  const page = savingsPage();
  await page.query();
  assert.deepEqual(page.order(), ['a', 'b', 'c']);
  page.sort('ts.savedCol');
  assert.deepEqual(page.order(), ['a', 'c', 'b']);
  assert.deepEqual(page.aria(), ['none', 'none', 'descending', 'none']);
  assert.deepEqual(ids(page.export()), ['a', 'c', 'b']);
  assert.ok(serializeSavingsCsv(page.export()).indexOf('"c"') < serializeSavingsCsv(page.export()).indexOf('"b"'));
  page.sort('ts.savedCol');
  assert.deepEqual(page.order(), ['b', 'c', 'a']);
  page.sort('ts.savedCol');
  assert.deepEqual(page.order(), ['a', 'b', 'c']);
  page.sort('ts.inputTokens');
  assert.deepEqual(page.order(), ['b', 'c', 'a']);
  assert.equal(page.requests(), 1);
});

test('a fresh successful query preserves the active sort and exports its new records', async () => {
  const page = savingsPage();
  await page.query();
  page.sort('ts.savingsRateCol');
  page.response([session('new-low', 5, 5, 1, 0.1), session('new-high', 5, 5, 5, 0.5)]);
  await page.query();
  assert.deepEqual(page.order(), ['new-high', 'new-low']);
  assert.deepEqual(ids(page.export()), ['new-high', 'new-low']);
  assert.equal(page.requests(), 2);
});

test('selected CSV keeps the displayed sort order without fetching new evidence', async () => {
  const page = savingsPage();
  await page.query();
  page.sort('ts.inputTokens');
  page.select('a');
  page.select('b');
  assert.deepEqual(page.order(), ['b', 'c', 'a']);
  assert.deepEqual(ids(page.exportSelected()), ['b', 'a']);
  assert.equal(page.requests(), 1);
});

test('fillTokenBuckets accumulates colliding rows in the same bucket slot without mutating frozen inputs', () => {
  const startNs = 1_000_000;
  const endNs = 4_000_000; // 3 buckets of 1_000_000 ns each
  const firstSlot0 = Object.freeze({
    bucket_start_ns: 1_000_000,
    input_tokens: 120,
    output_tokens: 30,
    total_tokens: 150,
  });
  const secondSlot0 = Object.freeze({
    // Within rounding wobble of slot 0
    bucket_start_ns: 1_000_100,
    input_tokens: 80,
    output_tokens: 20,
    total_tokens: 100,
  });
  const thirdSlot0 = Object.freeze({
    bucket_start_ns: 1_000_000,
    input_tokens: 5,
    output_tokens: 15,
    total_tokens: 20,
  });
  const slot2 = Object.freeze({
    bucket_start_ns: 3_000_000,
    input_tokens: 40,
    output_tokens: 10,
    total_tokens: 50,
  });

  const filled = fillTokenBuckets(
    [firstSlot0, secondSlot0, thirdSlot0, slot2],
    startNs,
    endNs,
    3,
  );

  assert.equal(filled.length, 3);
  assert.deepEqual(filled[0], {
    bucket_start_ns: 1_000_000,
    input_tokens: 205,
    output_tokens: 65,
    total_tokens: 270,
  });
  assert.deepEqual(filled[1], {
    bucket_start_ns: 2_000_000,
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
  });
  assert.deepEqual(filled[2], {
    bucket_start_ns: 3_000_000,
    input_tokens: 40,
    output_tokens: 10,
    total_tokens: 50,
  });
  assert.notEqual(filled[0], firstSlot0);
  assert.notEqual(filled[2], slot2);
  assert.equal(firstSlot0.total_tokens, 150);
  assert.equal(secondSlot0.total_tokens, 100);
});

test('fillModelBuckets accumulates colliding rows for the same bucket index and model', () => {
  const startNs = 10_000;
  const endNs = 30_000; // 2 buckets of 10_000 ns each
  const rows = [
    // Exact collision on (idx=0, model='GPT-4o'), e.g. after case-normalization in ModelTimeseriesChart
    Object.freeze({ bucket_start_ns: 10_000, model: 'GPT-4o', total_tokens: 37 }),
    Object.freeze({ bucket_start_ns: 10_000, model: 'GPT-4o', total_tokens: 12 }),
    // Separate model in slot 0 remains distinct
    Object.freeze({ bucket_start_ns: 10_000, model: 'claude-3-5-sonnet', total_tokens: 25 }),
    // Rounding collision on (idx=1, model='GPT-4o')
    Object.freeze({ bucket_start_ns: 20_000, model: 'GPT-4o', total_tokens: 100 }),
    Object.freeze({ bucket_start_ns: 20_050, model: 'GPT-4o', total_tokens: 45 }),
  ];

  const filled = fillModelBuckets(rows, startNs, endNs, 2, ['GPT-4o', 'claude-3-5-sonnet']);

  assert.deepEqual(filled, [
    { bucket_start_ns: 10_000, model: 'GPT-4o', total_tokens: 49 },
    { bucket_start_ns: 10_000, model: 'claude-3-5-sonnet', total_tokens: 25 },
    { bucket_start_ns: 20_000, model: 'GPT-4o', total_tokens: 145 },
    { bucket_start_ns: 20_000, model: 'claude-3-5-sonnet', total_tokens: 0 },
  ]);
  assert.equal(rows[0].total_tokens, 37);
  assert.equal(rows[1].total_tokens, 12);
});
