const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const test = require('node:test');

// Exercise the compiled cards through the handles used by the page's Save
// button. Delay only configuration I/O; keep state and effect dependencies.
async function card(kind, { refreshStatus, persistLimit } = {}) {
  const states = [], effects = [], callbacks = [], pending = [], submitted = [];
  let stateIndex, effectIndex, callbackIndex, renderCard;
  let resolveLoad, rejectLoad;
  let reads = 0;
  const loaded = new Promise((resolve, reject) => { resolveLoad = resolve; rejectLoad = reject; });
  const ref = { current: null };
  const t = (key, params) => key + (params ? JSON.stringify(params) : '');
  const react = {
    forwardRef(render) { renderCard = () => render({}, ref); return renderCard; },
    useImperativeHandle(handle, create) { handle.current = create(); },
    useState(initial) {
      const index = stateIndex++;
      if (!(index in states)) states[index] = initial;
      return [states[index], (value) => { states[index] = typeof value === 'function' ? value(states[index]) : value; }];
    },
    useEffect(callback, deps) {
      const index = effectIndex++;
      if (!effects[index] || deps.some((dep, i) => !Object.is(dep, effects[index][i]))) pending.push(callback);
      effects[index] = deps;
    },
    useCallback(callback, deps) {
      const index = callbackIndex++;
      if (!callbacks[index] || deps.some((dep, i) => !Object.is(dep, callbacks[index].deps[i]))) {
        callbacks[index] = { callback, deps };
      }
      return callbacks[index].callback;
    },
  };
  runInNewContext(readFileSync(kind === 'storage'
    ? process.env.AGENTSIGHT_SETTINGS_PAGE_BUILD : process.env.AGENTSIGHT_LLM_CONFIG_BUILD, 'utf8'), {
    exports: {},
    require(name) {
      if (name === 'react') return react;
      if (name === 'react/jsx-runtime') return {
        jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }),
      };
      if (name === '../i18n') return { useI18n: () => ({ t }) };
      if (name === '../components/OptimizationSettings') return { LlmConfigForm: () => null };
      if (name === '../utils/apiClient') return {
        fetchStorageStatus: () => reads++ === 0 ? loaded
          : refreshStatus ? refreshStatus() : Promise.resolve({ total_physical_bytes: 1024 }),
        fetchOptimizeConfig: () => loaded,
        async saveStorageLimit(value) { submitted.push(value); if (persistLimit) await persistLimit(value); },
        async saveOptimizeConfig(value) { submitted.push(value); return value; },
      };
      throw new Error(`Unexpected card dependency: ${name}`);
    },
  });
  function render() {
    stateIndex = effectIndex = callbackIndex = 0;
    return renderCard();
  }
  async function settle() {
    render();
    while (pending.length) {
      for (const effect of pending.splice(0)) effect();
      await new Promise(setImmediate);
      render();
    }
    await new Promise(setImmediate);
    return render();
  }
  function nodes(node) {
    if (!node || typeof node !== 'object') return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(nodes)];
  }
  await settle();
  return {
    submitted,
    async load(value) { resolveLoad(value); await settle(); },
    async fail() { rejectLoad('configuration unavailable'); await settle(); },
    async save() { render(); const result = await ref.current.save(); await settle(); return result; },
    input(value) {
      nodes(render()).find((node) => node.props?.id === 'storage-limit-mb')
        .props.onChange({ target: { value } });
    },
    text: () => JSON.stringify(render()),
  };
}

for (const kind of ['storage', 'llm']) {
  test(`${kind}: saving while the initial read is pending does not write defaults`, async () => {
    const form = await card(kind);
    assert.equal(await form.save(), false);
    assert.deepEqual(form.submitted, []);
    await form.load(kind === 'storage'
      ? { max_total_size_mb: 512, total_physical_bytes: 1024 }
      : { base_url: 'https://example.test/v1', model: 'existing', search_timeout_secs: 45 });
    assert.equal(await form.save(), true);
    if (kind === 'storage') assert.deepEqual(form.submitted, [512]);
    else {
      assert.equal(form.submitted[0].search_timeout_secs, 45);
      assert.equal(form.submitted[0].model, 'existing');
    }
  });

  test(`${kind}: a failed initial read remains visible and cannot be saved`, async () => {
    const form = await card(kind);
    await form.fail();
    assert.equal(await form.save(), false);
    assert.deepEqual(form.submitted, []);
    assert.match(form.text(), /loadFailed/);
  });
}

test('storage: clearing the limit is invalid, but explicitly entering zero disables it', async () => {
  const form = await card('storage');
  await form.load({ max_total_size_mb: 512, total_physical_bytes: 1024 });
  for (const invalid of ['', ' ', '8', '-1', '9.5']) {
    form.input(invalid);
    assert.equal(await form.save(), false, `invalid limit: ${JSON.stringify(invalid)}`);
  }
  assert.deepEqual(form.submitted, []);
  form.input('0');
  assert.equal(await form.save(), true);
  form.input('9');
  assert.equal(await form.save(), true);
  assert.deepEqual(form.submitted, [0, 9]);
});

test('storage: a failed usage refresh does not undo a confirmed limit save', async () => {
  let unavailable = true;
  const form = await card('storage', {
    refreshStatus: async () => {
      if (unavailable) throw new Error('usage unavailable');
      return { total_physical_bytes: 2048 };
    },
  });
  await form.load({ max_total_size_mb: 512, total_physical_bytes: 1024 });
  assert.match(form.text(), /1\.00 KiB/, 'initial usage is visible');
  form.input('128');
  assert.equal(await form.save(), true, 'the configuration POST succeeded');
  assert.deepEqual(form.submitted, [128]);
  assert.match(form.text(), /128.0 MiB/, 'the confirmed limit remains visible');
  assert.match(form.text(), /usage unavailable/);
  assert.doesNotMatch(form.text(), /1\.00 KiB/, 'stale usage is not current usage');
  assert.match(form.text(), /—/, 'failed refresh displays unknown usage');
  unavailable = false;
  assert.equal(await form.save(), true);
  assert.doesNotMatch(form.text(), /usage unavailable/);
  assert.match(form.text(), /2\.00 KiB/, 'retry displays fresh usage');
});

test('storage: a rejected configuration write remains a failed save', async () => {
  let reads = 0;
  const form = await card('storage', {
    persistLimit: async () => { throw new Error('configuration read-only'); },
    refreshStatus: async () => { reads += 1; return { total_physical_bytes: 2048 }; },
  });
  await form.load({ max_total_size_mb: 512, total_physical_bytes: 1024 });
  form.input('128');
  assert.equal(await form.save(), false);
  assert.equal(reads, 0);
  assert.match(form.text(), /configuration read-only/);
});

test('security utils: verdictTone, isPassVerdict, and badgeClasses classify negated/failure/pass/warning values', () => {
  const s = require(process.env.AGENTSIGHT_SECURITY_UTILS_BUILD);
  for (const v of [
    'unsuccessful', 'disallowed', 'not_allowed', 'not-allowed', 'not allowed', 'policy_disallowed',
    'block_succeeded', 'deny_success', 'access_denied', 'deny', 'blocked', 'failed', 'reject',
    'rejected', 'non_pass', 'tampered', 'unsafe', 'success_with_errors',
  ]) {
    assert.deepEqual([s.verdictTone(v), s.isPassVerdict(v), s.verdictBadgeClasses(v)], ['risk', false, 'bg-red-100 text-red-700'], v);
  }
  for (const v of ['swallow', 'bypass', 'book', 'not_found', 'unknown', '-', '', null, undefined]) {
    assert.deepEqual([s.verdictTone(v), s.isPassVerdict(v)], ['unknown', false], String(v));
  }
  for (const v of ['pass', 'passed', 'allow', 'allowed', 'success', 'succeeded', 'successful', 'safe', 'ok', 'clean', '  PASS  ', 'policy_allowed', 'check_succeeded']) {
    assert.deepEqual([s.verdictTone(v), s.isPassVerdict(v), s.verdictBadgeClasses(v)], ['pass', true, 'bg-green-100 text-green-700'], v);
  }
  for (const v of ['drifted', 'unmanaged', 'warn', 'review', 'caution', 'suspicious', 'warn_succeeded', 'passed_with_warnings']) {
    assert.deepEqual([s.verdictTone(v), s.isPassVerdict(v)], ['warning', false], v);
  }
  for (const v of ['unsuccessful', 'disallowed', 'not_allowed', 'not allowed', 'block_succeeded', 'deny_success', 'failed', 'rejected']) {
    assert.equal(s.badgeClasses(v, 'result'), 'bg-red-100 text-red-700', v);
  }
  for (const v of ['pass', 'passed', 'allow', 'allowed', 'success', 'succeeded', 'safe', 'ok', 'clean']) {
    assert.equal(s.badgeClasses(v, 'result'), 'bg-green-100 text-green-700', v);
  }
  for (const v of ['swallow', 'bypass', 'warn_succeeded', 'unknown', null]) {
    assert.equal(s.badgeClasses(v, 'result'), 'bg-gray-100 text-gray-700', String(v));
  }
});

test('security utils: findDetailValue, securityDetailRows, and securityEventVerdict honor priority and skip blanks', () => {
  const s = require(process.env.AGENTSIGHT_SECURITY_UTILS_BUILD);
  assert.equal(s.findDetailValue({ error_message: null, error: 'policy denied' }, ['error_message', 'error', 'message']), 'policy denied');
  assert.equal(s.findDetailValue({ error_message: '   ', error: 'policy denied' }, ['error_message', 'error', 'message']), 'policy denied');
  assert.equal(s.findDetailValue({ message: 'low', error: 'mid', error_message: 'high' }, ['error_message', 'error', 'message']), 'high');
  assert.equal(s.findDetailValue({ verdict: null, result: { verdict: 'block' } }, ['verdict']), 'block');
  assert.equal(s.findDetailValue({ verdict: '', result: { verdict: 'block' } }, ['verdict']), 'block');
  assert.equal(s.findDetailValue({ verdict: false }, ['verdict']), false);
  assert.equal(s.findDetailValue({ code: 0 }, ['code']), 0);
  const rows = s.securityDetailRows({
    verdict: null, error_message: null, error: 'policy denied', reason: '   ', policy_reason: 'disallowed_tool', result: { verdict: 'block' },
  });
  assert.deepEqual(rows.map((r) => [r.id, r.value]), [['verdict', 'block'], ['error', 'policy denied'], ['reason', 'disallowed_tool']]);
  assert.equal(s.securityEventVerdict({ event_id: 'evt-1', verdict: null, details: { verdict: null, result: { verdict: 'block' } } }), 'block');
  assert.equal(s.securityEventVerdict({ event_id: 'evt-2', verdict: '  ', details: { verdict: 'deny' } }), 'deny');
});

test('security utils: timestampToMs and fmtTime reject non-finite or out-of-range numbers and fall back cleanly', () => {
  const s = require(process.env.AGENTSIGHT_SECURITY_UTILS_BUILD);
  const { formatMsCompact } = require(process.env.AGENTSIGHT_DATETIME_BUILD);
  for (const input of [{ timestamp_ns: NaN }, { timestamp_ns: Infinity }, { timestamp_epoch: -Infinity }, { started_at_ns: NaN, first_seen_epoch: Infinity }, { timestamp_epoch: 1e308 }, { timestamp_epoch: 1e15 }]) {
    assert.equal(s.timestampToMs(input), null);
  }
  const iso = '2026-01-01T00:00:00Z';
  assert.equal(s.timestampToMs({ timestamp_ns: NaN, timestamp_epoch: Infinity, timestamp: iso }), Date.parse(iso));
  assert.equal(s.timestampToMs({ timestamp_ns: NaN, timestamp_epoch: 1_700_000_000 }), 1_700_000_000_000);
  assert.equal(s.timestampToMs({ timestamp_epoch: 0 }), 0);
  assert.equal(s.timestampToMs({ timestamp_ns: 1_700_000_000_000_000_000 }), 1_700_000_000_000);
  assert.equal(s.fmtTime({ timestamp_ns: NaN }, 'en-US'), '-');
  assert.equal(s.fmtTime({ timestamp_epoch: Infinity }, 'en-US'), '-');
  assert.equal(s.fmtTime({ timestamp_epoch: 1e308 }, 'en-US'), '-');
  assert.equal(s.fmtTime({ timestamp_ns: NaN, timestamp: iso }, 'en-US'), formatMsCompact(Date.parse(iso), 'en-US'));
});
