const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const test = require('node:test');
const apiClient = require(process.env.AGENTSIGHT_API_CLIENT_BUILD);

// Exercise the compiled page's event handlers with a persistent hook state.
// The element objects expose the same props React passes to the form/input.
function loginPage() {
  const states = [];
  let stateIndex = 0;
  let authenticated = 0;
  const exports = {};
  runInNewContext(readFileSync(process.env.AGENTSIGHT_LOGIN_PAGE_BUILD, 'utf8'), {
    exports,
    require(name) {
      if (name === 'react') return {
        useState(initial) {
          const index = stateIndex++;
          if (!(index in states)) states[index] = initial;
          return [states[index], (value) => { states[index] = value; }];
        },
      };
      if (name === 'react/jsx-runtime') return {
        jsx: (type, props) => ({ type, props }),
        jsxs: (type, props) => ({ type, props }),
      };
      if (name === '../i18n') return {
        useI18n: () => ({ t: (key) => key }),
        LanguageSwitcher: () => null,
      };
      if (name === '../utils/apiClient') return apiClient;
      throw new Error(`Unexpected LoginPage dependency: ${name}`);
    },
  });
  function render() {
    stateIndex = 0;
    return exports.LoginPage({ onAuthenticated: () => { authenticated++; } });
  }
  function find(node, predicate) {
    if (!node || typeof node !== 'object') return undefined;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat()) {
      const result = find(child, predicate);
      if (result) return result;
    }
  }
  return {
    input(value) {
      find(render(), (node) => node.type === 'input').props.onChange({ target: { value } });
    },
    async submit() {
      let prevented = false;
      await find(render(), (node) => node.type === 'form').props.onSubmit({
        preventDefault() { prevented = true; },
      });
      assert.equal(prevented, true);
    },
    error: () => find(render(), (node) => node.props?.className?.startsWith('text-red-600'))?.props.children,
    disabled: () => find(render(), (node) => node.type === 'button').props.disabled,
    authenticated: () => authenticated,
  };
}

test('login accepts a successful response and preserves the cookie request', async (t) => {
  let request;
  t.mock.method(global, 'fetch', async (url, init) => {
    request = { url, init };
    return new Response(null, { status: 200 });
  });
  assert.equal(await apiClient.login('valid-token'), true);
  assert.equal(new URL(request.url, 'http://localhost').pathname, '/api/auth/login');
  assert.deepEqual(request.init, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ token: 'valid-token' }),
  });
});

test('login returns false only for the backend invalid-token response', async (t) => {
  t.mock.method(global, 'fetch', async () => new Response(null, { status: 401 }));
  assert.equal(await apiClient.login('invalid-token'), false);
});

test('login rejects other HTTP failures with the response status', async (t) => {
  let status;
  t.mock.method(global, 'fetch', async () => new Response(null, { status }));
  for (status of [403, 429, 500, 503]) {
    await assert.rejects(apiClient.login('valid-token'), new RegExp(`POST /api/auth/login -> ${status}`));
  }
});

test('login page distinguishes invalid tokens from HTTP and network failures', async (t) => {
  let response;
  t.mock.method(global, 'fetch', async () => {
    if (response instanceof Error) throw response;
    return new Response(null, { status: response });
  });
  const page = loginPage();
  page.input('valid-token');
  for (const [result, expected] of [
    [401, 'login.error.invalid'],
    [500, 'login.error.connection'],
    [503, 'login.error.connection'],
    [new TypeError('fetch failed'), 'login.error.connection'],
  ]) {
    response = result;
    await page.submit();
    assert.equal(page.error(), expected);
    assert.equal(page.authenticated(), 0);
    assert.equal(page.disabled(), false);
  }
});

test('login page validates empty input and trims a successful login', async (t) => {
  const requests = [];
  t.mock.method(global, 'fetch', async (_url, init) => {
    requests.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  });
  const page = loginPage();
  page.input('  ');
  await page.submit();
  assert.equal(page.error(), 'login.error.required');
  assert.equal(requests.length, 0);
  page.input('  valid-token  ');
  await page.submit();
  assert.deepEqual(requests, [{ token: 'valid-token' }]);
  assert.equal(page.authenticated(), 1);
  assert.equal(page.error(), undefined);
  assert.equal(page.disabled(), false);
});

const { join } = require('node:path');
const babel = require('@babel/core');

const transpileTsx = (rel) => babel.transformFileSync(join(__dirname, '..', rel), {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }],
    ['@babel/preset-typescript', { isTSX: true, allExtensions: true }],
    ['@babel/preset-react', { runtime: 'classic' }],
  ],
}).code;
const navMod = { exports: {} };
new Function('require', 'module', 'exports', transpileTsx('src/utils/navigation.ts'))(() => { throw new Error('unexpected dep'); }, navMod, navMod.exports);
const appCode = transpileTsx('src/App.tsx');
const settle = () => new Promise(setImmediate);
const deferred = () => { let resolve, reject; return { promise: new Promise((res, rej) => { resolve = res; reject = rej; }), resolve, reject }; };
async function withWindow(win, fn) {
  const prev = global.window; global.window = win;
  try { return await fn(); } finally { if (prev === undefined) delete global.window; else global.window = prev; }
}
function findTreeNode(node, pred) {
  if (!node || typeof node !== 'object') return undefined;
  if (pred(node)) return node;
  for (const c of [...(node.children || []), ...[node.props?.children].flat()]) { const m = findTreeNode(c, pred); if (m) return m; }
}

function loadAppHarness({ initialLocation = '/', initialWindowUrl = 'http://localhost/#/', apiMocks = {} } = {}) {
  const slots = [], cleanups = [], replaceCalls = [], parsed = new URL(initialWindowUrl, 'http://localhost');
  let cursor = 0, effects = [];
  const win = {
    location: { pathname: parsed.pathname, search: parsed.search, hash: parsed.hash },
    history: {
      replaceState(state, title, url) {
        replaceCalls.push({ state, title, url });
        const u = new URL(url, 'http://localhost');
        win.location = { pathname: u.pathname, search: u.search, hash: u.hash };
      },
    },
    replaceCalls,
  };
  const routerState = { pathname: initialLocation, search: '', navigatedTo: [] };
  const markers = { HashRouter: Symbol('HashRouter'), Routes: Symbol('Routes'), Route: Symbol('Route'), Navigate: Symbol('Navigate'), NavBar: Symbol('NavBar'), LoginPage: Symbol('LoginPage'), AgentHealthNotifier: Symbol('AgentHealthNotifier') };
  const createElement = (type, props, ...children) => ({ type, props: props || {}, children });
  const reactStub = {
    __esModule: true, createElement, Fragment: Symbol('Fragment'),
    useState(init) {
      const s = (slots[cursor++] ??= { value: typeof init === 'function' ? init() : init, set: (u) => { s.value = typeof u === 'function' ? u(s.value) : u; } });
      return [s.value, s.set];
    },
    useEffect(fn) { effects.push(fn); },
  };
  reactStub.default = reactStub;
  const mod = { exports: {} };
  const deps = {
    react: reactStub, './i18n': { useI18n: () => ({ t: (k) => k }) }, './utils/navigation': navMod.exports,
    './components/NavBar': { NavBar: markers.NavBar }, './pages/LoginPage': { LoginPage: markers.LoginPage },
    './components/AgentHealthNotifier': { AgentHealthNotifier: markers.AgentHealthNotifier },
    'react-router-dom': { ...markers, useNavigate: () => (to) => { routerState.navigatedTo.push(to); }, useLocation: () => ({ pathname: routerState.pathname, search: routerState.search }) },
    './utils/apiClient': { login: apiMocks.login || (async () => false), fetchAuthStatus: apiMocks.fetchAuthStatus || (async () => ({ auth_enabled: true })), fetchAuthVerify: apiMocks.fetchAuthVerify || (async () => ({ authenticated: false })) },
  };
  new Function('require', 'module', 'exports', appCode)((name) => {
    if (name in deps) return deps[name];
    if (name.startsWith('./pages/')) return { [name.split('/').pop()]: Symbol(name) };
    throw new Error(`Unexpected App.tsx dependency: ${name}`);
  }, mod, mod.exports);
  const wild = mod.exports.default({}).children[0].children.find((r) => r.props.path === '/*');
  const AuthGate = wild.props.element.type, renderAppShell = wild.props.element.children[0];
  const unmount = () => { for (const fn of cleanups.splice(0)) if (typeof fn === 'function') fn(); };
  return {
    win, routerState, markers, renderAppShell, unmount,
    renderAuthGate(children = renderAppShell) { cursor = 0; effects = []; return { element: AuthGate({ children }), effects }; },
    runEffects(effs) { unmount(); for (const fn of effs) { const c = fn(); if (typeof c === 'function') cleanups.push(c); } },
  };
}

test('AuthGate scrubs ?token= from window.location even when login(urlToken) throws or unmounts', async () => {
  const loginCalls = [], pending = deferred();
  const h = loadAppHarness({ initialLocation: '/health', initialWindowUrl: 'http://localhost/app?foo=1&token=super-secret&bar=2#/health', apiMocks: { login: async (t) => { loginCalls.push(t); throw new Error('500'); } } });
  await withWindow(h.win, async () => {
    h.runEffects(h.renderAuthGate().effects); await settle();
    assert.deepEqual([loginCalls, h.win.location.search, h.win.replaceCalls[0]?.url, h.renderAuthGate().element.type], [['super-secret'], '?foo=1&bar=2', '/app?foo=1&bar=2#/health', h.markers.LoginPage]);
  });
  const h2 = loadAppHarness({ initialLocation: '/health', initialWindowUrl: 'http://localhost/?token=ephemeral#/health', apiMocks: { login: () => pending.promise } });
  await withWindow(h2.win, async () => {
    h2.runEffects(h2.renderAuthGate().effects); h2.unmount(); pending.resolve(true); await settle();
    assert.deepEqual([h2.win.location.search, h2.renderAuthGate().element.type], ['', 'div']);
  });
});

test('AuthGate syncs fetchAuthStatus() after URL-token login and falls back when unreachable', async () => {
  const restricted = { auth_enabled: true, mode: 'local', capabilities: ['sessions'] };
  let statusCalls = 0, passed, fb = 'UNSET';
  const h = loadAppHarness({ initialLocation: '/health', initialWindowUrl: 'http://localhost/?token=cli-secret#/health', apiMocks: { login: async () => true, fetchAuthStatus: async () => (statusCalls++, restricted) } });
  await withWindow(h.win, async () => {
    h.runEffects(h.renderAuthGate().effects); await settle();
    const shell = h.renderAuthGate((s) => ((passed = s), h.renderAppShell(s))).element.children[0];
    assert.deepEqual([statusCalls, h.win.location.search, passed, shell.type(shell.props).props.to], [1, '', restricted, '/sessions']);
  });
  const h2 = loadAppHarness({ initialLocation: '/health', initialWindowUrl: 'http://localhost/?token=cli-secret#/health', apiMocks: { login: async () => true, fetchAuthStatus: async () => { throw new Error('503'); } } });
  await withWindow(h2.win, async () => {
    h2.runEffects(h2.renderAuthGate().effects); await settle();
    const shell = h2.renderAuthGate((s) => ((fb = s), h2.renderAppShell(s))).element.children[0];
    assert.equal(fb, null);
    assert.ok(findTreeNode(shell.type(shell.props), (n) => n.type === h2.markers.NavBar)?.props.capabilities.includes('agent_health'));
  });
});

test('AuthGate cancels stale auth checks across location.pathname changes', async () => {
  const statusQ = [], verifyQ = [];
  const h = loadAppHarness({ initialLocation: '/health', initialWindowUrl: 'http://localhost/#/health', apiMocks: { fetchAuthStatus: () => { const d = deferred(); statusQ.push(d); return d.promise; }, fetchAuthVerify: () => { const d = deferred(); verifyQ.push(d); return d.promise; } } });
  await withWindow(h.win, async () => {
    h.runEffects(h.renderAuthGate().effects);
    statusQ[0].resolve({ auth_enabled: true, capabilities: ['agent_health', 'sessions'] }); await settle();
    h.routerState.pathname = '/sessions'; h.runEffects(h.renderAuthGate().effects);
    statusQ[1].resolve({ auth_enabled: true, capabilities: ['sessions'] }); await settle();
    verifyQ[1].resolve({ authenticated: true }); await settle();
    verifyQ[0].resolve({ authenticated: false }); await settle();
    let observed;
    assert.equal(h.renderAuthGate((s) => ((observed = s), { type: 'ok', s })).element.children[0].type, 'ok');
    assert.deepEqual(observed, { auth_enabled: true, capabilities: ['sessions'] });
    for (const p of ['/audit', '/login']) { h.routerState.pathname = p; h.runEffects(h.renderAuthGate().effects); }
    statusQ[2].reject(new Error('network error')); await settle();
    assert.equal(h.renderAuthGate().element.type, h.markers.LoginPage);
  });
});
