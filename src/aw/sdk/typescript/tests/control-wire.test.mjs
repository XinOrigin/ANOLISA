import test from "node:test";
import assert from "node:assert/strict";
import net from "node:net";
import { mkdtemp, chmod, rm, readFile } from "node:fs/promises";
import { once } from "node:events";
import { Instance } from "../src/client.mjs";

const contract = JSON.parse(
  await readFile(
    new URL("../../tests/fixtures/contract.json", import.meta.url),
  ),
);
const CONTROL_VERSION = "aw-sdk-control/v1";

async function until(check) {
  const deadline = Date.now() + 5000;
  while (!(await check())) {
    assert(Date.now() < deadline, "control wire deadline");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

async function controlServer(handler) {
  const directory = await mkdtemp("/tmp/aw-sdk-control-wire-");
  const endpoint = `${directory}/aw.sock`;
  const sockets = new Set();
  const listener = net.createServer((socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
    socket.on("error", () => {});
    let data = Buffer.alloc(0);
    socket.on("data", (chunk) => {
      data = Buffer.concat([data, chunk]);
      if (data.length < 4 || data.length < data.readUInt32BE(0) + 4) return;
      const request = JSON.parse(data.subarray(4));
      const reply = (identity, result, error = null) => {
        const body = Buffer.from(
          JSON.stringify({
            api_version: contract.service_protocol,
            identity,
            result,
            error,
          }),
        );
        const size = Buffer.alloc(4);
        size.writeUInt32BE(body.length);
        if (!socket.destroyed) socket.end(Buffer.concat([size, body]));
      };
      handler(request.operation, reply, socket, request);
    });
  });
  listener.listen(endpoint);
  await once(listener, "listening");
  await chmod(endpoint, 0o600);
  return {
    endpoint,
    async close() {
      for (const socket of sockets) socket.destroy();
      listener.close();
      await once(listener, "close");
      await rm(directory, { recursive: true, force: true });
    },
  };
}

test("service restart during incomplete attach resets detached state and pending key", async () => {
  const firstIdentity = { generation: "gen-1", config_revision: "rev-1" };
  const secondIdentity = { generation: "gen-2", config_revision: "rev-1" };
  let currentIdentity = firstIdentity;
  let attachCompleted = false;
  const registrationActions = [];
  const handle1 = { binding: { instance_id: "instance-1" } };
  const handle2 = { binding: { instance_id: "instance-2" } };
  let secondAttachRequested = false;

  const peer = await controlServer((operation, reply) => {
    const id = currentIdentity;
    if (operation.method === "status") {
      return reply(id, {
        sdk_protocol: contract.sdk_protocol,
        sdk_control_protocol: CONTROL_VERSION,
      });
    }
    if (operation.method === "sdk_control") {
      const { command } = operation;
      if (command.action === "poll") {
        if (id === firstIdentity) {
          return reply(id, {
            application_id: "app-1",
            request: {
              request_id: "attach-1",
              intent: "attach",
              state: "pending",
            },
          });
        }
        return reply(id, {
          application_id: "app-2",
          request:
            secondAttachRequested && !attachCompleted
              ? {
                  request_id: "attach-2",
                  intent: "attach",
                  state: "pending",
                }
              : null,
        });
      }
      if (command.action === "acknowledge") {
        attachCompleted = true;
        return reply(id, { state: "completed" });
      }
      return reply(id, { withdrawn: true });
    }
    if (operation.method === "registration") {
      const { command } = operation;
      registrationActions.push({ generation: id.generation, action: command.action });
      if (id !== firstIdentity && command.action === "ready" && command.handle?.binding?.instance_id === "instance-1") {
        return reply(id, null, "stale_service");
      }
      if (command.action === "register_sdk") {
        return reply(id, {
          state: "pending_ready",
          handle: id === firstIdentity ? handle1 : handle2,
          policy_revision: "r1",
          routes: { events: {}, hooks: [] },
        });
      }
      return reply(id, {
        state: command.action === "ready" ? "ready" : "released",
      });
    }
    reply(id, {});
  });

  let entered, release;
  let gateFirstInstall = true;
  const installing = new Promise((resolve) => {
    entered = resolve;
  });
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const instance = new Instance({
    endpoint: peer.endpoint,
    target: "app",
    capabilities: {},
    availability: "native_fallback",
    timeoutMs: 1000,
    reconnectMs: 100,
    install: async () => {
      if (gateFirstInstall) {
        gateFirstInstall = false;
        entered();
        await gate;
      }
      return true;
    },
  });
  try {
    instance.listen();
    await installing;
    currentIdentity = secondIdentity;
    release();
    await until(
      () =>
        instance.applicationId === "app-2" && !instance.controller.active,
    );
    assert.equal(instance.state, "detached");
    assert.equal(instance.pending, undefined);
    assert.equal(instance.lastError, undefined);
    assert.equal(instance.connection, undefined);
    assert.equal(
      registrationActions.some(
        (entry) => entry.generation === "gen-2" && entry.action === "release",
      ),
      false,
    );
    secondAttachRequested = true;
    await until(() => instance.state === "ready");
    assert.equal(instance.connection.handle.binding.instance_id, "instance-2");
  } finally {
    release();
    await instance.close();
    await peer.close();
  }
});

test("stopping listener during incomplete attach releases unconfirmed registration", async () => {
  const identity = { generation: "gen-1", config_revision: "rev-1" };
  const handle = { binding: { instance_id: "unconfirmed-instance" } };
  const events = [];
  let dropReadyOnce = true;

  const peer = await controlServer((operation, reply, socket) => {
    if (operation.method === "status") {
      return reply(identity, {
        sdk_protocol: contract.sdk_protocol,
        sdk_control_protocol: CONTROL_VERSION,
      });
    }
    if (operation.method === "sdk_control") {
      const { command } = operation;
      events.push(`control:${command.action}`);
      if (command.action === "poll") {
        return reply(identity, {
          application_id: "app-1",
          request: {
            request_id: "attach-1",
            intent: "attach",
            state: "pending",
          },
        });
      }
      return reply(identity, { withdrawn: true });
    }
    if (operation.method === "registration") {
      const { command } = operation;
      events.push(`registration:${command.action}`);
      if (command.action === "register_sdk") {
        return reply(identity, {
          state: "pending_ready",
          handle,
          policy_revision: "r1",
          routes: { events: {}, hooks: [] },
        });
      }
      if (command.action === "ready" && dropReadyOnce) {
        dropReadyOnce = false;
        return socket.destroy();
      }
      return reply(identity, {
        state: command.action === "ready" ? "ready" : "released",
      });
    }
    reply(identity, {});
  });

  const instance = new Instance({
    endpoint: peer.endpoint,
    target: "app",
    capabilities: {},
    availability: "native_fallback",
    timeoutMs: 1000,
    reconnectMs: 1000,
    install: () => true,
  });
  try {
    instance.listen();
    await until(() => instance.state === "unavailable");
    await instance.stop();
    assert.equal(instance.state, "detached");
    assert.equal(instance.pending, undefined);
    assert.equal(instance.lastError, undefined);
    assert.equal(instance.controller.active, undefined);
    assert.deepEqual(events.slice(-2), [
      "registration:release",
      "control:withdraw",
    ]);
  } finally {
    await instance.close();
    await peer.close();
  }
});

test("stopping listener while service is unavailable still resets incomplete attach state", async () => {
  const identity = { generation: "gen-1", config_revision: "rev-1" };
  const handle = { binding: { instance_id: "offline-instance" } };

  let entered, release;
  const installing = new Promise((resolve) => {
    entered = resolve;
  });
  const gate = new Promise((resolve) => {
    release = resolve;
  });

  const peer = await controlServer((operation, reply) => {
    if (operation.method === "status") {
      return reply(identity, {
        sdk_protocol: contract.sdk_protocol,
        sdk_control_protocol: CONTROL_VERSION,
      });
    }
    if (operation.method === "sdk_control") {
      return reply(identity, {
        application_id: "app-1",
        request: {
          request_id: "attach-1",
          intent: "attach",
          state: "pending",
        },
      });
    }
    if (operation.method === "registration") {
      return reply(identity, {
        state: "pending_ready",
        handle,
        policy_revision: "r1",
        routes: { events: {}, hooks: [] },
      });
    }
    reply(identity, {});
  });

  const instance = new Instance({
    endpoint: peer.endpoint,
    target: "app",
    capabilities: {},
    availability: "native_fallback",
    timeoutMs: 1000,
    reconnectMs: 1000,
    install: async () => {
      entered();
      await gate;
      return true;
    },
  });
  let closedPeer = false;
  try {
    instance.listen();
    await installing;
    await peer.close();
    closedPeer = true;
    release();
    await until(() => instance.state === "unavailable");
    await instance.stop();
    assert.equal(instance.state, "detached");
    assert.equal(instance.pending, undefined);
    assert.equal(instance.lastError, undefined);
    assert.equal(instance.controller.active, undefined);
  } finally {
    release();
    await instance.close();
    if (!closedPeer) await peer.close();
  }
});
