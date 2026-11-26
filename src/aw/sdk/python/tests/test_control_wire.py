"""Cooperative SDK control wire tests for aborted attach rollback."""

import asyncio
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from aw_sdk import Instance
from aw_sdk.control import VERSION as CONTROL_VERSION

CONTRACT = json.loads(Path(os.environ["AW_CONTRACT_FIXTURE"]).read_text())


class ControlWire(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="aw-sdk-control-wire-")
        self.endpoint = str(Path(self.directory.name) / "aw.sock")
        self.writers = set()
        self.operations = set()
        self.handler = None
        self.server = await asyncio.start_unix_server(self.accept, self.endpoint)
        Path(self.endpoint).chmod(0o600)

    async def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        self.operations.add(task)
        self.writers.add(writer)
        try:
            (length,) = struct.unpack("!I", await reader.readexactly(4))
            request = json.loads(await reader.readexactly(length))
            identity, result, error = await self.handler(request["operation"], writer)
            body = json.dumps(
                {
                    "api_version": CONTRACT["service_protocol"],
                    "identity": identity,
                    "result": result,
                    "error": error,
                }
            ).encode()
            writer.write(struct.pack("!I", len(body)) + body)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            self.writers.discard(writer)
            self.operations.discard(task)

    async def close_server(self) -> None:
        self.server.close()
        for task in list(self.operations):
            task.cancel()
        await asyncio.gather(*self.operations, return_exceptions=True)
        for writer in list(self.writers):
            writer.close()
        await self.server.wait_closed()

    async def asyncTearDown(self) -> None:
        await self.close_server()
        self.directory.cleanup()

    async def until(self, check) -> None:
        async with asyncio.timeout(5):
            while not check():
                await asyncio.sleep(0.01)

    async def test_service_restart_during_incomplete_attach_resets_detached_state(self) -> None:
        first_id = {"generation": "gen-1", "config_revision": "rev-1"}
        second_id = {"generation": "gen-2", "config_revision": "rev-1"}
        current_id = first_id
        attach_completed = False
        second_attach_requested = False
        registration_actions = []
        handle1 = {"binding": {"instance_id": "instance-1"}}
        handle2 = {"binding": {"instance_id": "instance-2"}}

        async def handler(operation: dict, _: asyncio.StreamWriter) -> tuple:
            nonlocal attach_completed
            identity = current_id
            if operation["method"] == "status":
                return (
                    identity,
                    {
                        "sdk_protocol": CONTRACT["sdk_protocol"],
                        "sdk_control_protocol": CONTROL_VERSION,
                    },
                    None,
                )
            if operation["method"] == "sdk_control":
                command = operation["command"]
                if command["action"] == "poll":
                    if identity is first_id:
                        return (
                            identity,
                            {
                                "application_id": "app-1",
                                "request": {
                                    "request_id": "attach-1",
                                    "intent": "attach",
                                    "state": "pending",
                                },
                            },
                            None,
                        )
                    return (
                        identity,
                        {
                            "application_id": "app-2",
                            "request": (
                                {
                                    "request_id": "attach-2",
                                    "intent": "attach",
                                    "state": "pending",
                                }
                                if second_attach_requested and not attach_completed
                                else None
                            ),
                        },
                        None,
                    )
                if command["action"] == "acknowledge":
                    attach_completed = True
                    return identity, {"state": "completed"}, None
                return identity, {"withdrawn": True}, None
            if operation["method"] == "registration":
                command = operation["command"]
                registration_actions.append((identity["generation"], command["action"]))
                if command["action"] == "register_sdk":
                    return (
                        identity,
                        {
                            "state": "pending_ready",
                            "handle": handle1 if identity is first_id else handle2,
                            "policy_revision": "r1",
                            "routes": {"events": {}, "hooks": []},
                        },
                        None,
                    )
                state = "ready" if command["action"] == "ready" else "released"
                return identity, {"state": state}, None
            return identity, {}, None

        self.handler = handler
        entered, release = asyncio.Event(), asyncio.Event()
        gate_first = True

        async def install(_: dict) -> bool:
            nonlocal gate_first
            if gate_first:
                gate_first = False
                entered.set()
                await release.wait()
            return True

        instance = Instance(
            endpoint=self.endpoint,
            target="app",
            capabilities={},
            availability="native_fallback",
            timeout_ms=1000,
            reconnect_ms=100,
            install=install,
        )
        try:
            instance.listen()
            async with asyncio.timeout(5):
                await entered.wait()
            current_id = second_id
            release.set()
            await self.until(
                lambda: instance.application_id == "app-2"
                and instance.controller.active is None
            )
            self.assertEqual(instance.state, "detached")
            self.assertIsNone(instance.pending)
            self.assertIsNone(instance.last_error)
            self.assertIsNone(instance.connection)
            self.assertNotIn(("gen-2", "release"), registration_actions)
            second_attach_requested = True
            await self.until(lambda: instance.state == "ready")
            self.assertEqual(
                instance.connection["handle"]["binding"]["instance_id"], "instance-2"
            )
        finally:
            release.set()
            await instance.close()

    async def test_stop_during_incomplete_attach_releases_unconfirmed_registration(self) -> None:
        identity = {"generation": "gen-1", "config_revision": "rev-1"}
        handle = {"binding": {"instance_id": "unconfirmed-instance"}}
        events = []
        drop_ready_once = True

        async def handler(operation: dict, _: asyncio.StreamWriter) -> tuple:
            nonlocal drop_ready_once
            if operation["method"] == "status":
                return (
                    identity,
                    {
                        "sdk_protocol": CONTRACT["sdk_protocol"],
                        "sdk_control_protocol": CONTROL_VERSION,
                    },
                    None,
                )
            if operation["method"] == "sdk_control":
                command = operation["command"]
                events.append(f"control:{command['action']}")
                if command["action"] == "poll":
                    return (
                        identity,
                        {
                            "application_id": "app-1",
                            "request": {
                                "request_id": "attach-1",
                                "intent": "attach",
                                "state": "pending",
                            },
                        },
                        None,
                    )
                return identity, {"withdrawn": True}, None
            if operation["method"] == "registration":
                command = operation["command"]
                events.append(f"registration:{command['action']}")
                if command["action"] == "register_sdk":
                    return (
                        identity,
                        {
                            "state": "pending_ready",
                            "handle": handle,
                            "policy_revision": "r1",
                            "routes": {"events": {}, "hooks": []},
                        },
                        None,
                    )
                if command["action"] == "ready" and drop_ready_once:
                    drop_ready_once = False
                    raise ConnectionResetError()
                state = "ready" if command["action"] == "ready" else "released"
                return identity, {"state": state}, None
            return identity, {}, None

        self.handler = handler
        instance = Instance(
            endpoint=self.endpoint,
            target="app",
            capabilities={},
            availability="native_fallback",
            timeout_ms=1000,
            reconnect_ms=1000,
            install=lambda _: True,
        )
        try:
            instance.listen()
            await self.until(lambda: instance.state == "unavailable")
            await instance.stop()
            self.assertEqual(instance.state, "detached")
            self.assertIsNone(instance.pending)
            self.assertIsNone(instance.last_error)
            self.assertIsNone(instance.controller.active)
            self.assertEqual(events[-2:], ["registration:release", "control:withdraw"])
        finally:
            await instance.close()

    async def test_stop_while_service_is_unavailable_resets_incomplete_attach_state(self) -> None:
        identity = {"generation": "gen-1", "config_revision": "rev-1"}
        handle = {"binding": {"instance_id": "offline-instance"}}
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(operation: dict, _: asyncio.StreamWriter) -> tuple:
            if operation["method"] == "status":
                return (
                    identity,
                    {
                        "sdk_protocol": CONTRACT["sdk_protocol"],
                        "sdk_control_protocol": CONTROL_VERSION,
                    },
                    None,
                )
            if operation["method"] == "sdk_control":
                return (
                    identity,
                    {
                        "application_id": "app-1",
                        "request": {
                            "request_id": "attach-1",
                            "intent": "attach",
                            "state": "pending",
                        },
                    },
                    None,
                )
            if operation["method"] == "registration":
                return (
                    identity,
                    {
                        "state": "pending_ready",
                        "handle": handle,
                        "policy_revision": "r1",
                        "routes": {"events": {}, "hooks": []},
                    },
                    None,
                )
            return identity, {}, None

        async def install(_: dict) -> bool:
            entered.set()
            await release.wait()
            return True

        self.handler = handler
        instance = Instance(
            endpoint=self.endpoint,
            target="app",
            capabilities={},
            availability="native_fallback",
            timeout_ms=1000,
            reconnect_ms=1000,
            install=install,
        )
        try:
            instance.listen()
            async with asyncio.timeout(5):
                await entered.wait()
            await self.close_server()
            Path(self.endpoint).unlink(missing_ok=True)
            release.set()
            await self.until(lambda: instance.state == "unavailable")
            await instance.stop()
            self.assertEqual(instance.state, "detached")
            self.assertIsNone(instance.pending)
            self.assertIsNone(instance.last_error)
            self.assertIsNone(instance.controller.active)
        finally:
            release.set()
            await instance.close()


if __name__ == "__main__":
    unittest.main()
