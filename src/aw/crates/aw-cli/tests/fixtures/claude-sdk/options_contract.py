"""Exercise public option composition without an installed SDK or native CLI."""

from dataclasses import dataclass, fields
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

native = (
    Path(sys.argv.pop(1))
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir()
    else Path(__file__).resolve().parents[4] / "aw-adapter/native/claude-sdk"
)
sys.path.insert(0, str(native))


@dataclass
class Options:
    setting_sources: object = None
    settings: object = None
    plugins: object = None
    mcp_servers: object = None
    hooks: object = None
    can_use_tool: object = None
    permission_prompt_tool_name: object = None
    extra_args: object = None
    cli_path: object = None
    cwd: object = None
    env: object = None


@dataclass
class Matcher:
    hooks: object
    timeout: object = None


@dataclass
class PermissionResultAllow:
    updated_input: object = None


@dataclass
class PermissionResultDeny:
    message: str = ""


sdk = ModuleType("claude_agent_sdk")
sdk.ClaudeAgentOptions, sdk.HookMatcher = Options, Matcher
sdk.ClaudeSDKClient = object
sdk.PermissionResultAllow, sdk.PermissionResultDeny = PermissionResultAllow, PermissionResultDeny
sys.modules["claude_agent_sdk"] = sdk

from aw_claude import Bridge


class Contracts(unittest.IsolatedAsyncioTestCase):
    def test_native_sources_settings_plugins_and_mcp_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk.__file__ = str(Path(directory) / "__init__.py")
            bundled = Path(directory) / "_bundled/claude"
            bundled.parent.mkdir()
            bundled.touch()
            for sources, settings in ((None, None), (["user", "project", "local"],
                                      '{"env":{"REFERENCE":"${TOKEN}"},"future":[1]}'),
                                     (["project"], "./relative-settings.json")):
                with self.subTest(sources=sources, settings=settings):
                    native = object()
                    hooks = {"SessionStart": [Matcher(hooks=[native])]}
                    options = Options(
                        setting_sources=sources, settings=settings,
                        plugins=[{"type": "local", "path": "./plugin"}],
                        mcp_servers={"remote": {"type": "http", "url": "https://fixture.invalid/mcp",
                                               "headers": {"X-Context": "${TOKEN}"}}},
                        hooks=hooks, extra_args={}, cwd="./business",
                        env={"BUSINESS_CONFIG": "./config.json"},
                    )
                    bridge = object.__new__(Bridge)
                    bridge.final = None
                    bridge.config = {"hooks": [{"event": "tool.before", "budget_ms": 1000,
                                                "replace_result": False}]}
                    composed = bridge.options(options)
                    for field in fields(options):
                        if field.name not in {"hooks", "cli_path"}:
                            self.assertIs(getattr(composed, field.name), getattr(options, field.name))
                    self.assertIs(composed.hooks["SessionStart"][0], hooks["SessionStart"][0])
                    self.assertEqual(composed.hooks["PreToolUse"][0].hooks, [bridge.callback])
                    self.assertEqual(list(hooks), ["SessionStart"])
                    self.assertIsNone(options.cli_path)
                    self.assertEqual(composed.cli_path, str(bundled))

    async def test_after_callbacks_isolate_payload_mutations_without_result_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk.__file__ = str(Path(directory) / "__init__.py")
            bundled = Path(directory) / "_bundled/claude"
            bundled.parent.mkdir()
            bundled.touch()
            for event in ("PostToolUse", "PostToolUseFailure"):
                seen = []
                rewrite = {"hookSpecificOutput": {"updatedToolOutput": {"stdout": "native"}}}

                async def caller_hook(payload, call_id, context):
                    payload["tool_input"]["command"] = "mutated"
                    payload["tool_use_id"] = "mutated-id"
                    return rewrite

                bridge = object.__new__(Bridge)
                bridge.final = None
                bridge.config = {"hooks": [{"event": "tool.after", "step": "observe",
                    "on_error": "report", "budget_ms": 1000, "replace_result": False}]}
                bridge._invoke = lambda hook, payload, seen=seen: (
                    seen.append((payload["tool_use_id"], dict(payload["tool_input"]))) or self._empty()
                )
                composed = bridge.options(Options(extra_args={}, hooks={event: [Matcher([caller_hook])]}))
                payload = {"hook_event_name": event, "tool_use_id": "call-1",
                           "tool_name": "Bash", "tool_input": {"command": "original"}}
                self.assertEqual(await composed.hooks[event][0].hooks[0](payload, "call-1", {}), rewrite)
                await composed.hooks[event][1].hooks[0](payload, "call-1", {})
                self.assertEqual(seen, [("call-1", {"command": "original"})])

    async def _empty(self):
        return {}

    async def test_resident_bridge_without_before_route_guards_input_rewrites_and_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk.__file__ = str(Path(directory) / "__init__.py")
            bundled = Path(directory) / "_bundled/claude"
            bundled.parent.mkdir()
            bundled.touch()
            bridge = object.__new__(Bridge)
            bridge.final = None
            bridge.invocations = SimpleNamespace()
            bridge.config = {"hooks": [{"event": "tool.after", "step": "observe",
                "on_error": "report", "budget_ms": 1000, "replace_result": False}]}
            with self.assertRaisesRegex(ValueError, "permission_prompt_tool_name"):
                bridge.options(Options(extra_args={}, permission_prompt_tool_name="stdio"))
            with self.assertRaisesRegex(ValueError, "PermissionRequest"):
                bridge.options(Options(extra_args={}, hooks={"PermissionRequest": [Matcher([object()])]}))

            async def rewrite_hook(payload, call_id, context):
                payload["tool_input"]["command"] = "mutated"
                return {"hookSpecificOutput": {"updatedInput": {"command": "rewritten"}}}

            async def rewrite_permission(name, tool_input, context):
                tool_input["command"] = "rewritten"
                return PermissionResultAllow(updated_input=tool_input)

            composed = bridge.options(Options(
                extra_args={}, hooks={"PreToolUse": [Matcher([rewrite_hook])]},
                can_use_tool=rewrite_permission,
            ))
            payload = {"hook_event_name": "PreToolUse", "tool_input": {"command": "original"}}
            hook_reply = await composed.hooks["PreToolUse"][0].hooks[0](payload, "call-1", {})
            self.assertEqual(hook_reply["hookSpecificOutput"]["permissionDecision"], "deny")
            self.assertEqual(payload["tool_input"], {"command": "original"})
            perm_reply = await composed.can_use_tool("Bash", payload["tool_input"], None)
            self.assertIsInstance(perm_reply, PermissionResultDeny)
            self.assertEqual(payload["tool_input"], {"command": "original"})


if __name__ == "__main__":
    unittest.main()
