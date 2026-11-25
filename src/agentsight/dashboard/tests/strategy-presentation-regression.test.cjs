const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const test = require('node:test');
const { OPTIMIZATION_STRATEGIES } = require(process.env.AGENTSIGHT_STRATEGIES_BUILD);

function component(file, name, locale = 'en') {
  const exports = {};
  const t = (key, params) => `${locale}:${key}${params?.strategy ? `:${params.strategy}` : ''}`;
  runInNewContext(readFileSync(file, 'utf8') + `\nexports.fixture = ${name};`, {
    exports, setTimeout, clearTimeout,
    require(id) {
      if (id === 'react') return { useState: (value) => [value, () => {}] };
      if (id === 'react/jsx-runtime') return { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) };
      if (id === '../utils/optimizationStrategies') return { OPTIMIZATION_STRATEGIES };
      if (id === '../i18n') return { useI18n: () => ({ t }), useLocaleTag: () => locale === 'zh' ? 'zh-CN' : 'en-US' };
      if (id === 'recharts' || id === 'react-router-dom' || id.startsWith('../components/') || id.startsWith('../utils/')) return {};
      throw new Error(`Unexpected component dependency: ${id}`);
    },
  });
  return exports.fixture;
}

function nodes(node) {
  if (!node || typeof node !== 'object') return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(nodes)];
}

function badges(strategy, locale) {
  const item = { id: 'call', category: 'tool_output', strategy, strategy_label: 'server label', title: 'output', before_tokens: 100, after_tokens: 60, compounded_saved: 40 };
  const atif = component(process.env.AGENTSIGHT_ATIF_PAGE_BUILD, 'ToolCallItem', locale)({ tc: { tool_call_id: 'call', function_name: 'fixture', arguments: {} }, savingsMap: new Map([['call', item]]) });
  const savings = component(process.env.AGENTSIGHT_SAVINGS_PAGE_BUILD, 'OptimizationTableRow', locale)({ item });
  return { atif: nodes(atif), savings: nodes(savings) };
}

for (const strategy of ['compress-schema', 'compress-response', 'rewrite-command', 'compress-toon']) {
  test(`${strategy} shares translated labels and badge styles across both production views`, () => {
    const config = OPTIMIZATION_STRATEGIES[strategy];
    for (const locale of ['en', 'zh']) {
      const result = badges(strategy, locale);
      const label = `${locale}:${config.labelKey}`;
      assert.ok(result.atif.some((node) => node.props?.children === `${locale}:atif.optimizedTokens:${label}` && node.props.className.includes(config.color) && node.props.className.includes(config.bg)));
      assert.ok(result.savings.some((node) => [node.props?.children].flat().includes(label) && node.props.className.includes(config.color) && node.props.className.includes(config.bg)));
      assert.ok(result.savings.some((node) => [node.props?.children].flat().includes(`${locale}:${config.tooltipKey}`)));
    }
  });
}

test('all existing chart colors and translated tooltip keys remain available', () => {
  assert.deepEqual(Object.values(OPTIMIZATION_STRATEGIES).map((config) => config.pie), ['#3b82f6', '#8b5cf6', '#f59e0b', '#14b8a6']);
  assert.ok(Object.values(OPTIMIZATION_STRATEGIES).every((config) => config.tooltipKey.startsWith('ts.')));
});

test('unknown strategies retain each view’s server-label and neutral-style fallback', () => {
  const result = badges('future-strategy', 'en');
  assert.equal(OPTIMIZATION_STRATEGIES['future-strategy'], undefined);
  assert.ok(result.atif.some((node) => node.props?.children === 'en:atif.optimizedTokens:server label' && node.props.className.includes('bg-gray-100') && node.props.className.includes('text-gray-700')));
  assert.ok(result.savings.some((node) => [node.props?.children].flat().includes('server label') && node.props.className.includes('bg-gray-100') && node.props.className.includes('text-gray-700')));
});

const { resolve } = require('node:path');
const ts = require('typescript');
const OPT_PAGE_CODE = ts.transpileModule(readFileSync(resolve(__dirname, '../src/pages/OptimizationPage.tsx'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText;
const settle = () => new Promise((r) => setTimeout(r, 0));
const textOf = (node) => [node].flat(Infinity).map((n) => (n == null || typeof n === 'boolean' ? '' : typeof n === 'object' ? textOf(n.props?.children) : String(n))).join('');
const historyRow = (session_id, dimensions) => ({ session_id, dimensions, created_at_ns: 1_700_000_000_000_000_000, updated_at_ns: 1_700_000_000_000_000_000 });
const deferred = () => { let res, rej; return { promise: new Promise((a, b) => { res = a; rej = b; }), resolve: res, reject: rej }; };

function buildOptPage(overrides = {}) {
  const slots = [], effects = [], callbacks = [];
  let cursor = 0, cbCursor = 0, fxCursor = 0, pending = [], writes = 0;
  let translate = (k, p) => (p ? `en:${k}:${JSON.stringify(p)}` : `en:${k}`);
  const hooks = {
    Fragment: ({ children }) => children,
    useState(init) {
      const s = (slots[cursor++] ??= { v: typeof init === 'function' ? init() : init, set: (u) => { writes += 1; s.v = typeof u === 'function' ? u(s.v) : u; } });
      return [s.v, s.set];
    },
    useRef: (init) => (slots[cursor++] ??= { current: init }),
    useCallback(fn, deps = []) {
      const i = cbCursor++, prev = callbacks[i];
      if (!prev || prev.deps.length !== deps.length || deps.some((d, j) => !Object.is(d, prev.deps[j]))) callbacks[i] = { fn, deps };
      return callbacks[i].fn;
    },
    useEffect(fn, deps) {
      const i = fxCursor++, prev = effects[i];
      const same = prev && Array.isArray(deps) && Array.isArray(prev.deps) && deps.length === prev.deps.length && deps.every((d, j) => Object.is(d, prev.deps[j]));
      effects[i] = { deps, cleanup: prev?.cleanup };
      if (!same) pending.push({ i, fn });
    },
  };
  const renderJsx = (type, props) => (typeof type === 'function' ? type(props ?? {}) : { type, props: props ?? {} });
  const noop = () => null, mod = { exports: {} };
  const stubs = {
    react: { __esModule: true, default: hooks, ...hooks },
    'react/jsx-runtime': { Fragment: hooks.Fragment, jsx: renderJsx, jsxs: renderJsx },
    'react-router-dom': { useNavigate: () => noop, useParams: () => overrides.params ?? { sessionId: 'sess-1' } },
    recharts: { Cell: noop, Pie: noop, PieChart: noop, ResponsiveContainer: noop, Tooltip: noop },
    '../utils/apiClient': { fetchOptimizeResults: async () => ({}), fetchOptimizeHistory: async () => [], runOptimizeDimension: async () => ({}), ApiRequestError: class extends Error {}, ...overrides.apiClient },
    '../i18n': { useI18n: () => ({ t: translate }), useLocaleTag: () => 'en-US' },
    '../utils/clipboard': { copyText: noop }, '../utils/formatDuration': { formatDurationSecs: String },
    '../utils/accuracyAttribution': { fixLocusDiverges: () => false, fixLocusLabel: (v) => v },
    '../utils/richText': { RichText: hooks.Fragment }, '../components/TokenFlameChart': { __esModule: true, default: noop },
  };
  new Function('require', 'module', 'exports', OPT_PAGE_CODE)((id) => stubs[id] ?? assert.fail(`Unexpected OptimizationPage dependency: ${id}`), mod, mod.exports);
  return {
    writes: () => writes,
    setLocale: (loc) => { translate = (k, p) => (p ? `${loc}:${k}:${JSON.stringify(p)}` : `${loc}:${k}`); },
    unmount: () => { for (const e of effects) if (typeof e?.cleanup === 'function') { e.cleanup(); e.cleanup = undefined; } },
    render: () => {
      cursor = 0; cbCursor = 0; fxCursor = 0; pending = [];
      const tree = mod.exports.OptimizationPage({});
      for (const { i, fn } of pending) { effects[i]?.cleanup?.(); effects[i].cleanup = fn(); }
      return tree;
    },
  };
}

test('SessionAnalysisView: manual runAnalysis() clears stale resultsUnavailable banner', async () => {
  const history = deferred(), dimDeferreds = new Map(), dimensionCalls = [];
  const { render } = buildOptPage({
    params: { sessionId: 'sess-1' },
    apiClient: { fetchOptimizeResults: () => history.promise, runOptimizeDimension: (_id, dim) => { dimensionCalls.push(dim); const d = deferred(); dimDeferreds.set(dim, d); return d.promise; } },
  });
  render();
  history.reject(new Error('503 service unavailable'));
  await settle(); await settle();
  const failedTree = render();
  assert.equal(dimensionCalls.length, 0);
  assert.ok(textOf(failedTree).includes('opt.session.resultsUnavailable'));
  const startBtn = nodes(failedTree).find((n) => n.type === 'button' && textOf(n).includes('opt.session.action.start'));
  assert.ok(startBtn && startBtn.props.disabled === false);
  startBtn.props.onClick();
  assert.equal(textOf(render()).includes('opt.session.resultsUnavailable'), false);
  assert.deepEqual([...dimensionCalls].sort(), ['accuracy', 'cost', 'cost-waste', 'perf', 'perf-issues', 'summary']);
  dimDeferreds.get('summary').resolve({ goal: 'Test goal', process: ['Step 1'], outcome: 'Done' });
  dimDeferreds.get('perf').resolve({ wall_secs: 10, model_secs: 6, tool_secs: 4, idle_secs: 0, tool_count: 2, top_slow: [] });
  dimDeferreds.get('perf-issues').resolve({ items: [] }); dimDeferreds.get('cost-waste').resolve({ items: [] });
  dimDeferreds.get('cost').resolve({ total_events: 5, total_chars: 100 });
  dimDeferreds.get('accuracy').resolve({ extraction: { final_answer: 'ok' }, failures: [], issues: [] });
  await settle(); await settle();
  const completedText = textOf(render());
  assert.equal(completedText.includes('opt.session.resultsUnavailable'), false);
  assert.equal(dimensionCalls.length, 6);
  assert.ok(completedText.includes('Test goal'));
});

test('SessionEntryView: re-fetch clears stale error and renders history rows on retry success', async () => {
  let attempt = 0;
  const retry = deferred();
  const { render, setLocale } = buildOptPage({
    params: {},
    apiClient: { fetchOptimizeHistory: () => (++attempt === 1 ? Promise.reject(new Error('initial history timeout')) : retry.promise) },
  });
  render(); await settle(); await settle();
  const failedTree = render();
  assert.ok(textOf(failedTree).includes('opt.history.loadFailed') && textOf(failedTree).includes('initial history timeout'));
  assert.equal(nodes(failedTree).filter((n) => n.type === 'tbody').length, 0);
  setLocale('zh'); render();
  const loadingText = textOf(render());
  assert.equal(loadingText.includes('opt.history.loadFailed'), false);
  assert.ok(loadingText.includes('opt.history.loading'));
  retry.resolve([historyRow('sess-recovered-1', ['summary', 'perf'])]);
  await settle(); await settle();
  const recoveredText = textOf(render());
  assert.equal(recoveredText.includes('opt.history.loadFailed'), false);
  assert.equal(recoveredText.includes('opt.history.loading'), false);
  assert.ok(recoveredText.includes('sess-recovered-1'));
});

test('SessionEntryView: cancellation guard drops stale responses across locale switch and unmount', async () => {
  const req1 = deferred(), req2 = deferred(), req3 = deferred();
  let calls = 0;
  const page1 = buildOptPage({ params: {}, apiClient: { fetchOptimizeHistory: () => (++calls === 1 ? req1.promise : req2.promise) } });
  page1.render(); assert.equal(calls, 1);
  page1.setLocale('zh'); page1.render(); assert.equal(calls, 2);
  req2.resolve([historyRow('sess-latest-2', ['cost'])]);
  await settle(); await settle();
  req1.reject(new Error('stale req1 failure'));
  await settle(); await settle();
  const latestText = textOf(page1.render());
  assert.equal(latestText.includes('opt.history.loadFailed'), false);
  assert.ok(latestText.includes('sess-latest-2'));
  const page2 = buildOptPage({ params: {}, apiClient: { fetchOptimizeHistory: () => req3.promise } });
  page2.render(); page2.unmount();
  const writesAtUnmount = page2.writes();
  req3.resolve([historyRow('sess-unmounted', [])]);
  await settle(); await settle();
  assert.equal(page2.writes(), writesAtUnmount);
  page1.unmount();
});
