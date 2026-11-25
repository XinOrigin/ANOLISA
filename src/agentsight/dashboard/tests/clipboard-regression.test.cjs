const assert = require('node:assert/strict');
const { join } = require('node:path');
const test = require('node:test');
const babel = require('@babel/core');

const { copyText } = require(process.env.AGENTSIGHT_CLIPBOARD_BUILD);

function browser(t, { secure = true, writeText, execCommand = () => true } = {}) {
  const descriptors = new Map(
    ['navigator', 'window', 'document'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]),
  );
  const children = [];
  const copies = [];
  let focused = false;
  let selected = false;
  const document = {
    createElement(tag) {
      assert.equal(tag, 'textarea');
      return {
        value: '',
        style: {},
        focus() { focused = true; },
        select() { selected = true; },
      };
    },
    body: {
      appendChild(node) { children.push(node); },
      removeChild(node) { children.splice(children.indexOf(node), 1); },
    },
    execCommand(command) {
      assert.equal(command, 'copy');
      assert.ok(focused && selected);
      assert.equal(children.length, 1);
      copies.push(children[0].value);
      return execCommand();
    },
  };
  const navigator = { clipboard: writeText ? { writeText } : undefined };
  for (const [key, value] of Object.entries({ navigator, window: { isSecureContext: secure }, document })) {
    Object.defineProperty(globalThis, key, { configurable: true, value });
  }
  t.after(() => {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  return { children, copies };
}

const cacheDiagnosisCode = babel.transformFileSync(join(__dirname, '../src/components/CacheDiagnosisPanel.tsx'), {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
    ['@babel/preset-typescript', { isTSX: true, allExtensions: true }],
    ['@babel/preset-react', { runtime: 'classic' }],
  ],
}).code;
const settle = () => new Promise(setImmediate);
const cacheSnap = (agent_name, truncated = false, status = 'healthy') => ({
  start_ns: 0, end_ns: 100, truncated, groups: [{ agent_name, provider: 'anthropic', model: 'claude-sonnet', status }],
});

function createCacheDiagnosisHarness(locale = 'en-US') {
  const slots = [], callbacks = [], effects = [], calls = [];
  let stateWrites = 0, unmounted = false, rangeMs = 7 * 86_400_000;
  let cursor = 0, cbCursor = 0, effCursor = 0, pending = [];
  let t = (key) => `${locale}:${key}`;
  const sameDeps = (a, b) => a && b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const reactStub = {
    __esModule: true,
    createElement: (type, props, ...children) => ({ type, props: { ...(props || {}), children } }),
    useState(init) {
      let s = slots[cursor];
      if (!s) s = slots[cursor] = { value: init, set: (u) => { stateWrites += 1; s.value = typeof u === 'function' ? u(s.value) : u; } };
      cursor += 1;
      return [s.value, s.set];
    },
    useRef: (current) => (slots[cursor++] ??= { value: { current } }).value,
    useCallback(fn, deps) {
      const i = cbCursor++, prev = callbacks[i];
      return prev && sameDeps(prev.deps, deps) ? prev.fn : (callbacks[i] = { fn, deps }, fn);
    },
    useEffect: (fn, deps) => { pending.push({ i: effCursor++, fn, deps }); },
  };
  reactStub.default = reactStub;
  const mod = { exports: {} };
  const deps = {
    react: reactStub, '../i18n': { useI18n: () => ({ locale, t }) },
    '../utils/apiClient': { fetchCacheDiagnosis: (s, e) => new Promise((resolve, reject) => { calls.push({ s, e, resolve, reject }); }) },
  };
  new Function('require', 'module', 'exports', cacheDiagnosisCode)((id) => {
    assert.ok(id in deps, `Unexpected CacheDiagnosisPanel dependency: ${id}`);
    return deps[id];
  }, mod, mod.exports);

  function render(nextRangeMs = rangeMs) {
    assert.equal(unmounted, false);
    rangeMs = nextRangeMs;
    cursor = cbCursor = effCursor = 0;
    pending = [];
    const before = stateWrites;
    const tree = mod.exports.CacheDiagnosisPanel({ rangeMs });
    for (const p of pending) {
      const prev = effects[p.i];
      if (prev && p.deps && sameDeps(prev.deps, p.deps)) continue;
      if (prev && typeof prev.cleanup === 'function') prev.cleanup();
      effects[p.i] = { deps: p.deps, cleanup: p.fn() };
    }
    return stateWrites !== before ? render(rangeMs) : JSON.stringify(tree);
  }
  return {
    calls, render, slots: () => slots.map((s) => s.value), writes: () => stateWrites,
    setLocale: (next) => { locale = next; t = (key) => `${locale}:${key}`; },
    unmount: () => { unmounted = true; effects.forEach((e) => { e?.cleanup?.(); if (e) e.cleanup = undefined; }); },
  };
}

test('CacheDiagnosisPanel clears stale rows, truncated notice, and errors on range reload and failure', async () => {
  const h = createCacheDiagnosisHarness('en-US');
  assert.match(h.render(), /en-US:common\.loading/);
  h.calls[0].resolve(cacheSnap('claude-7d', true));
  await settle();
  assert.match(h.render(), /en-US:cache\.truncated.*claude-7d/);
  const pending1h = h.render(3_600_000);
  assert.match(pending1h, /en-US:common\.loading/);
  assert.doesNotMatch(pending1h, /claude-7d|cache\.truncated/);
  h.calls[1].reject(new Error('GenAI store unavailable'));
  await settle();
  const failed1h = h.render();
  assert.match(failed1h, /GenAI store unavailable/);
  assert.doesNotMatch(failed1h, /cache\.truncated/);
  assert.equal(h.slots()[0], null);
  assert.doesNotMatch(h.render(21_600_000), /GenAI store unavailable/);
  h.calls[2].resolve(cacheSnap('codex-6h', false, 'needs_review'));
  await settle();
  const loaded6h = h.render();
  assert.match(loaded6h, /codex-6h/);
  assert.doesNotMatch(loaded6h, /GenAI store unavailable|cache\.truncated/);
});

test('CacheDiagnosisPanel invalidates in-flight loads on unmount and drops out-of-order responses', async () => {
  for (const outcome of ['resolve', 'reject']) {
    const h = createCacheDiagnosisHarness('en-US');
    h.render(3_600_000);
    h.unmount();
    const writesAtUnmount = h.writes();
    if (outcome === 'resolve') h.calls[0].resolve(cacheSnap('late', true));
    else h.calls[0].reject(new Error('late failure'));
    await settle();
    assert.equal(h.writes(), writesAtUnmount);
    assert.deepEqual(h.slots().slice(0, 2), [null, null]);
  }
  const h = createCacheDiagnosisHarness('en-US');
  h.render(7 * 86_400_000);
  h.render(3_600_000);
  h.calls[1].resolve(cacheSnap('fresh-1h'));
  await settle();
  h.calls[0].resolve(cacheSnap('stale-7d', true));
  await settle();
  let out = h.render();
  assert.match(out, /fresh-1h/);
  assert.doesNotMatch(out, /stale-7d|cache\.truncated/);
  h.render(86_400_000);
  h.render(21_600_000);
  h.calls[3].resolve(cacheSnap('fresh-6h'));
  await settle();
  h.calls[2].reject(new Error('obsolete 24h failure'));
  await settle();
  out = h.render();
  assert.match(out, /fresh-6h/);
  assert.doesNotMatch(out, /obsolete 24h failure/);
});

test('CacheDiagnosisPanel falls back to localized cache.error on nullish or blank rejections without refetching', async () => {
  const h = createCacheDiagnosisHarness('en-US');
  h.render(3_600_000);
  h.calls[0].reject(null);
  await settle();
  assert.match(h.render(), /en-US:cache\.error/);
  h.setLocale('zh-CN');
  assert.match(h.render(), /zh-CN:cache\.error/);
  assert.equal(h.calls.length, 1);
  h.render(21_600_000);
  h.calls[1].reject(new Error('   '));
  await settle();
  assert.match(h.render(), /zh-CN:cache\.error/);
});

test('secure clipboard waits for completion before showing feedback', async (t) => {
  let resolveWrite;
  let written;
  const state = browser(t, {
    writeText(text) {
      written = text;
      return new Promise((resolve) => { resolveWrite = resolve; });
    },
  });
  let completions = 0;
  copyText('session-1', () => { completions += 1; });
  assert.equal(written, 'session-1');
  assert.equal(completions, 0);
  resolveWrite();
  await new Promise(setImmediate);
  assert.equal(completions, 1);
  assert.deepEqual(state.copies, []);
  assert.deepEqual(state.children, []);
});

test('an insecure origin uses the fallback even when clipboard is present', (t) => {
  const state = browser(t, {
    secure: false,
    writeText() { throw new Error('clipboard must not be used over HTTP'); },
  });
  let completions = 0;
  copyText('会话\nsecond line', () => { completions += 1; });
  assert.deepEqual(state.copies, ['会话\nsecond line']);
  assert.deepEqual(state.children, []);
  assert.equal(completions, 1);
});

test('a browser without the Clipboard API uses the fallback', (t) => {
  const state = browser(t);
  let completed = false;
  copyText('', () => { completed = true; });
  assert.deepEqual(state.copies, ['']);
  assert.deepEqual(state.children, []);
  assert.equal(completed, true);
});

test('a rejected clipboard write falls back once and removes its textarea', async (t) => {
  const state = browser(t, { writeText: () => Promise.reject(new Error('permission denied')) });
  let completions = 0;
  copyText('optimization details', () => { completions += 1; });
  await new Promise(setImmediate);
  assert.deepEqual(state.copies, ['optimization details']);
  assert.deepEqual(state.children, []);
  assert.equal(completions, 1);
});

test('the fallback retains legacy feedback when execCommand reports failure', (t) => {
  const state = browser(t, { execCommand: () => false });
  let completions = 0;
  copyText('session-1', () => { completions += 1; });
  assert.equal(completions, 1);
  assert.deepEqual(state.children, []);
});

test('the fallback retains legacy cleanup and feedback if execCommand throws', (t) => {
  const state = browser(t, { execCommand: () => { throw new Error('copy unsupported'); } });
  let completions = 0;
  copyText('session-1', () => { completions += 1; });
  assert.equal(completions, 1);
  assert.deepEqual(state.children, []);
});
