import assert from "node:assert/strict";
import test from "node:test";
import { TaskScope } from "../src/task-scope.mjs";

const REV_A = "a".repeat(64);
const REV_B = "b".repeat(64);

function fakeInstance({
  binding = { instance_id: "inst-1" }, policyRevision = REV_A, watchRevision = REV_A, onCall, onTask, onClose,
} = {}) {
  const tasks = [];
  const instance = {
    state: "ready", closed: false,
    options: { availability: "required", capabilities: { events: { "tool.before": ["observe"] } } },
    async task(opts) {
      if (onTask) return onTask(opts, tasks);
      const task = {
        binding, routes: { policy_revision: policyRevision, hooks: [] }, acknowledged: 0, closed: 0,
        async acknowledge() { this.acknowledged++; },
        async call(op) {
          if (onCall) return onCall(op);
          if (op.method === "enable_invocations") return { invocations_per_binding: 4 };
          return op.method === "watch_scope" ? { policy_revision: watchRevision, interval_ms: 2000 } : {};
        },
        async close() { this.closed++; if (onClose) await onClose(); },
      };
      tasks.push(task);
      return task;
    },
  };
  return { instance, tasks };
}

test("TaskScope.open rejects reuse, closed/failed scopes, and closes unbound tasks", async () => {
  const { instance, tasks } = fakeInstance();
  const scope = new TaskScope(instance, { timeoutMs: 1000 });
  await scope.open(process.cwd(), {});
  await assert.rejects(() => scope.open(process.cwd(), {}), /cannot be reopened/);
  await scope.close();
  assert.equal(tasks[0].closed, 1);
  await assert.rejects(() => scope.open(process.cwd(), {}), /cannot be reopened/);
  const failed = new TaskScope(instance, { timeoutMs: 1000 });
  failed.invalidate(new Error("boom"));
  await assert.rejects(() => failed.open(process.cwd(), {}), /cannot be reopened/);
  assert.equal(tasks.length, 1);
  const unbound = fakeInstance({ binding: null });
  const leaky = new TaskScope(unbound.instance, { timeoutMs: 1000 });
  await assert.rejects(() => leaky.open(process.cwd(), {}), /closed, unbound/);
  assert.equal(leaky.closed, true);
  assert.equal(unbound.tasks[0].closed, 1);
});

test("TaskScope.acknowledge checks state, revision drift, duplicate ack, and closes on error", async (t) => {
  const ok = fakeInstance();
  const scope = new TaskScope(ok.instance, { timeoutMs: 1000 });
  const drift = fakeInstance({ policyRevision: REV_A, watchRevision: REV_B });
  const driftScope = new TaskScope(drift.instance, { timeoutMs: 1000 });
  let firstTimer;
  t.after(() => { clearTimeout(firstTimer); clearTimeout(scope.monitorTimer); clearTimeout(driftScope.monitorTimer); });
  await scope.open(process.cwd(), {});
  await scope.acknowledge();
  firstTimer = scope.monitorTimer;
  assert.equal(scope.invocationLimit, 4);
  await assert.rejects(() => scope.acknowledge(), /already acknowledged/);
  await scope.close();
  await assert.rejects(() => scope.acknowledge(), /closed, unbound/);
  assert.equal(ok.tasks[0].acknowledged, 1);
  await driftScope.open(process.cwd(), {});
  await assert.rejects(() => driftScope.acknowledge(), /revision changed during admission/);
  assert.equal(driftScope.closed, true);
  assert.equal(drift.tasks[0].closed, 1);
  const fail = fakeInstance({ onCall(op) { if (op.method === "enable_invocations") throw new Error("enable failed"); } });
  const failScope = new TaskScope(fail.instance, { timeoutMs: 1000 });
  await failScope.open(process.cwd(), {});
  await assert.rejects(() => failScope.acknowledge(), /enable failed/);
  assert.equal(failScope.closed, true);
  assert.equal(fail.tasks[0].closed, 1);
});

test("TaskScope.close guards cross-fork inheritance, drains concurrent open, and keeps releaseError", async () => {
  const fork = fakeInstance();
  const inherited = new TaskScope(fork.instance, { timeoutMs: 1000 });
  await inherited.open(process.cwd(), {});
  inherited.pid = process.pid + 1;
  await assert.rejects(() => inherited.close(), /Cannot close an inherited AW task/);
  assert.equal(inherited.closed, false);
  assert.equal(fork.tasks[0].closed, 0);
  const deferred = Promise.withResolvers();
  const lateTask = { binding: { instance_id: "late" }, routes: { policy_revision: REV_A, hooks: [] }, closed: 0, async close() { this.closed++; } };
  const racing = new TaskScope(fakeInstance({ onTask: () => deferred.promise }).instance, { timeoutMs: 1000 });
  const opening = racing.open(process.cwd(), {});
  const closing = racing.close();
  deferred.resolve(lateTask);
  await assert.rejects(() => opening, /closed, unbound/);
  await closing;
  assert.equal(lateTask.closed, 1);
  const failingClose = new TaskScope(fakeInstance({ onClose: () => { throw new Error("release failed"); } }).instance, { timeoutMs: 1000 });
  await failingClose.open(process.cwd(), {});
  await assert.rejects(() => failingClose.close(), /release failed/);
  assert.match(String(failingClose.releaseError), /release failed/);
});
