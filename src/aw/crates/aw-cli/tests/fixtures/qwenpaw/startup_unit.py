"""Native startup must fence early App requests and preserve plugin search paths."""

import asyncio
import builtins
import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "aw_qwenpaw_startup", ROOT / "crates/aw-adapter/native/qwenpaw/startup.py",
)
STARTUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STARTUP)


class StartupTests(unittest.TestCase):
    def test_plugin_directory_and_ready_receipt_gate_native_build(self):
        class Loader:
            def __init__(self, plugin_dirs):
                self.directories = plugin_dirs

        class Builder:
            calls = 0

            async def build(self):
                self.calls += 1
                return "native-agent"

        modules = {}
        for name, values in {
            "qwenpaw": {}, "qwenpaw.plugins": {}, "qwenpaw.runtime": {},
            "qwenpaw.plugins.loader": {"PluginLoader": Loader},
            "qwenpaw.runtime.builder": {"AgentBuilder": Builder},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "plugins/aw-native-fixture"
            plugin.mkdir(parents=True, mode=0o700)
            ready = root / "ready.json"
            config = root / "config.json"
            config.write_text(json.dumps({
                "resident": True, "resident_policy_reload": True,
                "plugin_id": plugin.name, "ready_path": str(ready), "ready_token": "owned-token",
            }))
            config.chmod(0o600)
            reply = {"environment": {"AW_NATIVE_CONFIG": str(config), "AW_QWENPAW_PLUGIN": str(plugin)}}
            with patch.dict(sys.modules, modules), patch.dict(os.environ):
                STARTUP._wire(reply)
                self.assertEqual(Loader([root / "user-plugins"]).directories,
                                 [root / "user-plugins", plugin.parent])
                builder = Builder()
                with self.assertRaises(FileNotFoundError):
                    asyncio.run(builder.build())
                self.assertEqual(builder.calls, 0)
                ready.write_text(json.dumps({"adapter": "qwenpaw", "pid": os.getpid(), "token": "other"}))
                ready.chmod(0o600)
                with self.assertRaisesRegex(RuntimeError, "readiness"):
                    asyncio.run(builder.build())
                self.assertEqual(builder.calls, 0)
                ready.write_text(json.dumps({"adapter": "qwenpaw", "pid": os.getpid(), "token": "owned-token"}))
                self.assertEqual(asyncio.run(builder.build()), "native-agent")
                self.assertEqual(builder.calls, 1)
                ready.unlink()
                with self.assertRaises(FileNotFoundError):
                    asyncio.run(builder.build())
                self.assertEqual(builder.calls, 1)

    def test_native_arguments_must_match_the_admitted_target(self):
        native = types.ModuleType("qwenpaw.cli.main")
        native.cli = lambda: self.fail("unadmitted native CLI was called")
        package = types.ModuleType("qwenpaw.cli")
        package.main = native
        reply = {"binding": "/binding", "command": {"program": "/venv/bin/qwenpaw", "args": ["app"]}}
        with patch.dict(sys.modules, {"qwenpaw.cli": package, "qwenpaw.cli.main": native}), \
                patch.object(sys, "argv", ["/venv/bin/qwenpaw", "app", "--workers", "2"]), \
                patch.object(STARTUP, "_prepare", return_value=({"binary": "/aw"}, reply)), \
                patch.object(STARTUP.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as release:
            STARTUP.install()
            installed = native.cli
            STARTUP.install()
            self.assertIs(native.cli, installed)
            with self.assertRaisesRegex(RuntimeError, "admitted target"):
                native.cli()
            self.assertEqual(release.call_args.args[0], ["/aw", "instance", "release", "--binding", "/binding"])

    def test_targeted_import_failure_cannot_be_swallowed_by_python_site(self):
        original = builtins.__import__

        def load(name, *args, **kwargs):
            if name == "qwenpaw.cli":
                raise ImportError("native entrypoint unavailable")
            return original(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=load), \
                patch.object(sys, "argv", ["/venv/bin/qwenpaw", "app"]):
            with self.assertRaisesRegex(SystemExit, "native entrypoint"):
                STARTUP.install()
        with patch.object(builtins, "__import__", side_effect=load), \
                patch.object(sys, "argv", ["/venv/bin/python", "unrelated.py"]):
            STARTUP.install()

    def test_native_sigterm_unwinds_binding_release(self):
        native = types.ModuleType("qwenpaw.cli.main")
        native.cli = lambda: signal.raise_signal(signal.SIGTERM)
        package = types.ModuleType("qwenpaw.cli")
        package.main = native
        reply = {"binding": "/binding", "command": {"program": "/venv/bin/qwenpaw", "args": ["app"]}}
        previous = signal.getsignal(signal.SIGTERM)
        with patch.dict(sys.modules, {"qwenpaw.cli": package, "qwenpaw.cli.main": native}), \
                patch.object(sys, "argv", ["/venv/bin/qwenpaw", "app"]), \
                patch.object(STARTUP, "_prepare", return_value=({"binary": "/aw"}, reply)), \
                patch.object(STARTUP, "_wire"), \
                patch.object(STARTUP.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as release:
            STARTUP.install()
            with self.assertRaises(SystemExit) as stopped:
                native.cli()
            self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
            self.assertEqual(release.call_args.args[0], ["/aw", "instance", "release", "--binding", "/binding"])

    def test_descriptor_rejects_public_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptor.json"
            path.write_text("{}")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                STARTUP._private_json(str(path))
            path.chmod(0o600)
            link = path.with_name("link")
            link.symlink_to(path)
            with self.assertRaises(OSError):
                STARTUP._private_json(str(link))
            self.assertEqual(STARTUP._private_json(str(path)), {})


LAUNCH_SPEC = importlib.util.spec_from_file_location(
    "aw_qwenpaw_launch",
    ROOT / "crates/aw-adapter/native/qwenpaw/launch.py",
)
LAUNCH = importlib.util.module_from_spec(LAUNCH_SPEC)
LAUNCH_SPEC.loader.exec_module(LAUNCH)


class PathTests(unittest.TestCase):
    def normalized(self):
        original = lambda path: os.path.normcase(os.path.realpath(str(path)))
        isolation = types.ModuleType("qwenpaw.plugins.module_isolation")
        isolation._norm = original
        plugins = types.ModuleType("qwenpaw.plugins")
        plugins.module_isolation = isolation
        return original, isolation, {
            "qwenpaw": types.ModuleType("qwenpaw"),
            "qwenpaw.plugins": plugins,
            "qwenpaw.plugins.module_isolation": isolation,
        }

    def test_owned_helper_loads_with_isolated_python(self):
        subprocess.run(
            [sys.executable, "-I", "-c",
             f"import runpy; runtime=runpy.run_path({str(LAUNCH_SPEC.origin)!r}); "
             "assert callable(runtime['accelerate_paths'])"],
            check=True, capture_output=True, timeout=5,
        )

    def test_only_the_verified_version_uses_native_path_resolution(self):
        original, isolation, modules = self.normalized()
        with patch.dict(sys.modules, modules):
            for version in (None, "2.2.2.post1", "2.1.1b3+local"):
                LAUNCH.accelerate_paths(version)
                self.assertIs(isolation._norm, original)
            LAUNCH.accelerate_paths("2.1.1b3")
            self.assertIsNot(isolation._norm, original)

    def test_repeated_path_acceleration_keeps_one_wrapper(self):
        _, isolation, modules = self.normalized()
        with patch.dict(sys.modules, modules):
            first = LAUNCH.accelerate_paths("2.1.1b3")
            self.assertIs(LAUNCH.accelerate_paths("2.1.1b3"), first)
            self.assertIs(isolation._norm, first)

    def test_unavailable_acceleration_preserves_native_resolution(self):
        import ctypes
        original, isolation, modules = self.normalized()
        with patch.dict(sys.modules, modules):
            with patch.dict(sys.modules, {"ctypes": None}):
                self.assertIsNone(LAUNCH.accelerate_paths("2.1.1b3"))
            self.assertIs(isolation._norm, original)
            with patch.object(ctypes, "CDLL", side_effect=OSError("unavailable")):
                self.assertIsNone(LAUNCH.accelerate_paths("2.1.1b3"))
            self.assertIs(isolation._norm, original)
            with patch.object(ctypes, "CDLL", return_value=types.SimpleNamespace()):
                self.assertIsNone(LAUNCH.accelerate_paths("2.1.1b3"))
            self.assertIs(isolation._norm, original)
            isolation._norm = None
            self.assertIsNone(LAUNCH.accelerate_paths("2.1.1b3"))
            self.assertIsNone(isolation._norm)

    def test_native_resolution_errors_remain_visible(self):
        _, isolation, modules = self.normalized()
        def fail(path):
            raise RuntimeError("native resolution failed")
        isolation._norm = fail
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            LAUNCH.accelerate_paths("2.1.1b3")
            with self.assertRaisesRegex(RuntimeError, "native resolution failed"):
                isolation._norm(Path(directory) / "missing")

    def test_existing_missing_and_invalid_paths_keep_native_semantics(self):
        original, isolation, modules = self.normalized()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            (root / "folder").mkdir()
            (root / "folder/file").touch()
            (root / "link").symlink_to("folder")
            (root / "broken").symlink_to("absent")
            (root / "loop").symlink_to("loop")
            undecodable = root / os.fsdecode(b"name-\xff")
            undecodable.touch()
            LAUNCH.accelerate_paths("2.1.1b3")
            paths = [root, root / "link/file", root / "link/../folder/file",
                     root / "absent/child", root / "broken/child", root / "loop",
                     root / "folder/file/child", root / "中文", undecodable,
                     "", ".", "..", "//" + str(root).lstrip("/"), "bad\0path"]
            for path in paths:
                with self.subTest(path=path):
                    try:
                        expected = original(path)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            isolation._norm(path)
                    else:
                        self.assertEqual(isolation._norm(path), expected)

    def test_symlink_changes_are_observed_without_a_cache(self):
        original, isolation, modules = self.normalized()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            (root / "first").mkdir()
            (root / "second").mkdir()
            link = root / "link"
            link.symlink_to("first")
            LAUNCH.accelerate_paths("2.1.1b3")
            self.assertEqual(isolation._norm(link), str(root / "first"))
            link.unlink()
            link.symlink_to("second")
            self.assertEqual(isolation._norm(link), original(link))
            self.assertEqual(isolation._norm(link), str(root / "second"))


class ManagedTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, root, failure=False):
        from contextlib import asynccontextmanager

        class Registry:
            def __init__(self):
                self.manifests = {"user": {"id": "user"}}
                self.middleware = ["user"]
                self.startup = []

            def get_plugin_manifest(self, identity):
                return self.manifests.get(identity)

            def register_plugin_manifest(self, identity, manifest):
                self.manifests[identity] = manifest

            def unregister_plugin(self, identity):
                self.manifests.pop(identity, None)
                self.middleware[:] = [x for x in self.middleware if x != identity]
                self.startup[:] = [x for x in self.startup if x != identity]

        class Api:
            def __init__(self, identity, config, manifest):
                self.identity, self.config, self.manifest = identity, config, manifest

            def set_registry(self, registry):
                self.registry = registry

        registry = Registry()

        class Loader:
            def __init__(self):
                self.registry = registry
                self.native_calls = []
                self.lock = asyncio.Lock()

            async def load_all_plugins(self, configs=None, types=None):
                self.native_calls.append((configs, types))
                return {"user": "original-record"}

            @asynccontextmanager
            async def plugin_lifecycle(self, identity):
                async with self.lock:
                    yield

        class Builder:
            calls = 0

            async def build(self):
                self.calls += 1
                return "native-agent"

        modules = {}
        for name, values in {
            "qwenpaw.plugins.loader": {"PluginLoader": Loader},
            "qwenpaw.plugins.api": {"PluginApi": Api},
            "qwenpaw.runtime.builder": {"AgentBuilder": Builder},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        source = root / "aw-owned"
        source.mkdir(mode=0o700)
        config = {
            "resident": True,
            "resident_policy_reload": True,
            "plugin_id": source.name,
            "ready_token": "fixture-token",
            "ready_path": str(root / "ready.json"),
            "hooks": [],
        }
        (source / "plugin.py").write_text("""class Plugin:
    def register(self, api):
        api.registry.middleware.append(api.identity)
        api.registry.startup.append(api.identity)
        if api.config.get('failure'):raise RuntimeError('registration rejected')
plugin = Plugin()
""")
        (source / "aw_native.py").write_text("""from qwenpaw.runtime.builder import AgentBuilder
def install_guard(*, build=None):
    if getattr(AgentBuilder, '_guarded', False):return
    original = AgentBuilder.build if build is None else build
    async def guarded(self, *args, **kwargs):
        return 'guarded:' + await original(self, *args, **kwargs)
    if build is None:AgentBuilder.build = guarded
    AgentBuilder._guarded = True
    return guarded
""")
        manifest = source / "plugin.json"
        manifest.write_text(json.dumps({"id": source.name}))
        manifest.chmod(0o600)
        return source, config, modules, Loader, Builder

    def ready(self, config, **changes):
        value = {
            "version": 1,
            "adapter": "qwenpaw",
            "pid": os.getpid(),
            "token": config["ready_token"],
            "hooks": len(config["hooks"]),
        }
        value.update(changes)
        path = Path(config["ready_path"])
        path.write_text(json.dumps(value))
        path.chmod(0o600)

    async def test_descriptor_limits_preserve_large_native_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = {"configuration": "x" * 70000}
            path.write_text(json.dumps(value))
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                LAUNCH.private_json(path)
            self.assertEqual(LAUNCH.private_json(path, limit=1024 * 1024), value)
            with path.open("wb") as stream:
                stream.write(b"x" * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                LAUNCH.private_json(path, limit=1024 * 1024)

    async def test_runtime_import_waits_for_the_native_plugin_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules), patch.object(
                LAUNCH, "load_runtime", wraps=LAUNCH.load_runtime
            ) as load:
                LAUNCH.install(source, config)
                load.assert_not_called()
                await Loader().load_all_plugins(types=["channel"])
                load.assert_called_once_with(source, config)

    async def test_native_loads_precede_idempotent_managed_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                loader = Loader()
                settings = {"user": {"enabled": True}}
                result = await loader.load_all_plugins(configs=settings, types=["channel"])
                self.assertEqual(loader.registry.middleware, ["user"])
                self.assertEqual(result, {"user": "original-record"})
                await asyncio.gather(
                    loader.load_all_plugins(configs=settings),
                    loader.load_all_plugins(configs=settings),
                )
                self.assertEqual(loader.native_calls[0], (settings, ["channel"]))
                self.assertEqual(loader.registry.middleware, ["user", source.name])
                self.assertEqual(loader.registry.startup, [source.name])
                self.assertEqual(loader.registry.manifests["user"], {"id": "user"})
                self.assertNotIn("aw_native", sys.modules)

    async def test_loaders_share_registration_and_preserve_unrelated_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                first, second = Loader(), Loader()
                self.assertIs(first.registry, second.registry)
                first.registry.startup.append("user")
                await first.load_all_plugins()
                await second.load_all_plugins()
                await asyncio.gather(first.load_all_plugins(), second.load_all_plugins())
                self.assertEqual(first.registry.middleware, ["user", source.name])
                self.assertEqual(first.registry.startup, ["user", source.name])
                first.registry.unregister_plugin(source.name)
                await asyncio.gather(second.load_all_plugins(), first.load_all_plugins())
                self.assertEqual(first.registry.middleware, ["user", source.name])
                self.assertEqual(first.registry.startup, ["user", source.name])

    async def test_failed_registration_can_retry_through_another_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                first, second = Loader(), Loader()
                with self.assertRaisesRegex(RuntimeError, "registration rejected"):
                    await first.load_all_plugins(configs={source.name: {"failure": True}})
                self.assertEqual(first.registry.manifests, {"user": {"id": "user"}})
                self.assertEqual(first.registry.middleware, ["user"])
                self.assertEqual(first.registry.startup, [])
                await second.load_all_plugins()
                await first.load_all_plugins()
                self.assertEqual(first.registry.middleware, ["user", source.name])
                self.assertEqual(first.registry.startup, [source.name])

    async def test_registration_follows_the_actual_registry_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                first, second = Loader(), Loader()
                second.registry = type(first.registry)()
                await asyncio.gather(first.load_all_plugins(), second.load_all_plugins())
                for loader in (first, second):
                    self.assertEqual(loader.registry.middleware, ["user", source.name])
                    self.assertEqual(loader.registry.startup, [source.name])

    async def test_early_build_waits_and_enters_the_native_assembly_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, Builder = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                builder = Builder()
                task = asyncio.create_task(builder.build())
                await asyncio.sleep(0.01)
                self.assertEqual(builder.calls, 0)
                self.assertFalse(task.done())
                self.ready(config)
                await asyncio.sleep(0.03)
                self.assertEqual(builder.calls, 0)
                self.assertFalse(task.done())
                await Loader().load_all_plugins(types=["channel"])
                await Loader().load_all_plugins()
                self.assertEqual(await task, "guarded:native-agent")
                self.assertEqual(builder.calls, 1)

    async def test_ready_identity_and_permissions_gate_every_build(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, Builder = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                await Loader().load_all_plugins(types=["channel"])
                builder = Builder()
                for changed in (
                    {"pid": os.getpid() + 1},
                    {"token": "other"},
                    {"hooks": 1},
                    {"adapter": "other"},
                    {"version": 2},
                ):
                    self.ready(config, **changed)
                    with self.assertRaisesRegex(RuntimeError, "readiness"):
                        await builder.build()
                self.ready(config)
                Path(config["ready_path"]).chmod(0o644)
                with self.assertRaises(ValueError):
                    await builder.build()
                self.assertEqual(builder.calls, 0)

    async def test_missing_ready_has_a_deadline_and_remains_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, Builder = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules), patch.object(LAUNCH, "READY_TIMEOUT", 0.001):
                LAUNCH.install(source, config)
                builder = Builder()
                with self.assertRaisesRegex(RuntimeError, "deadline"):
                    await builder.build()
                self.assertEqual(builder.calls, 0)

    async def test_registration_failure_rolls_back_only_owned_registry_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                LAUNCH.install(source, config)
                loader = Loader()
                with self.assertRaisesRegex(RuntimeError, "registration rejected"):
                    await loader.load_all_plugins(configs={source.name: {"failure": True}})
                self.assertEqual(loader.registry.manifests, {"user": {"id": "user"}})
                self.assertEqual(loader.registry.middleware, ["user"])
                self.assertIsNone(loader.registry.get_plugin_manifest(source.name))

    async def test_bad_module_removes_its_private_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, Loader, _ = self.fixture(Path(directory))
            (source / "plugin.py").write_text("raise RuntimeError('bad module')")
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(SystemExit, "runtime initialization") as error:
                    LAUNCH.install(source, config)
                    await Loader().load_all_plugins(types=["channel"])
                self.assertIsInstance(error.exception.__cause__, RuntimeError)
                self.assertEqual(str(error.exception.__cause__), "bad module")
                self.assertNotIn("aw_managed_fixture_token", sys.modules)

    async def test_managed_mode_requires_matching_private_owned_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, modules, _, _ = self.fixture(Path(directory))
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "resident"):
                    LAUNCH.install(source, {**config, "resident": False})
                source.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "private"):
                    LAUNCH.install(source, config)
                source.chmod(0o700)
                with self.assertRaisesRegex(ValueError, "owned"):
                    LAUNCH.install(source, {**config, "plugin_id": "other"})
                (source / "plugin.json").write_text('{"id":"other"}')
                with self.assertRaisesRegex(ValueError, "manifest"):
                    LAUNCH.install(source, config)

    async def test_bootstrap_runs_original_console_with_original_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}")
            config.chmod(0o600)
            program = root / "venv with spaces/bin/qwenpaw"
            program.parent.mkdir(parents=True)
            output = root / "context.json"
            program.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                f"Path({str(output)!r}).write_text(json.dumps([sys.argv, sys.path[0]]))\n"
            )
            link = root / "linked-console"
            link.symlink_to(program)
            old_path = sys.path.copy()
            expected_path = old_path[0] if sys.flags.safe_path else str(program.parent)
            try:
                for console in (program, link):
                    arguments = ["launch.py", str(console), "/owned", "app", "--port", "8096"]
                    with patch.dict(os.environ, AW_NATIVE_CONFIG=str(config)), patch.object(
                        sys, "argv", arguments
                    ), patch.object(LAUNCH, "install") as install, patch.object(
                        LAUNCH.runpy, "run_path", wraps=LAUNCH.runpy.run_path
                    ) as run:
                        def check_native_context(*_):
                            self.assertEqual(sys.argv, [str(console), "app", "--port", "8096"])
                            self.assertEqual(sys.path[0], expected_path)
                        install.side_effect = check_native_context
                        LAUNCH.main()
                        install.assert_called_once_with(Path("/owned"), {})
                        run.assert_called_once_with(str(console), run_name="__main__")
                        self.assertEqual(json.loads(output.read_text()), [
                            [str(console), "app", "--port", "8096"], expected_path,
                        ])
            finally:
                sys.path[:] = old_path


class ResidentInputTests(unittest.IsolatedAsyncioTestCase):
    def load_resident(self):
        class Model:
            def __init__(self, **v): self.__dict__.update(v)
            def model_dump(self, mode="json"): return json.loads(json.dumps(self.__dict__, default=lambda o: o.__dict__))
            @classmethod
            def model_validate(cls, v): return cls(**json.loads(json.dumps(v, default=lambda o: o.__dict__)))
            def model_copy(self, deep=False): return Model.model_validate(self.model_dump())
        def repair(raw, schema=None):
            data = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
            props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
            for k, spec in (props.items() if isinstance(data, dict) else ()):
                if spec.get("type") == "integer" and isinstance(data.get(k), str) and data[k].isdigit():
                    data[k] = int(data[k])
            return data
        tk_mod, st = types.ModuleType("agentscope.tool._toolkit"), types.SimpleNamespace(DENIED="denied", ERROR="error", INTERRUPTED="interrupted")
        tk_mod._json_loads_with_repair = repair
        ev = {"ReplyEndEvent": Model, "UserConfirmResultEvent": Model, "UserInterruptEvent": Model, "ExternalExecutionResultEvent": Model}
        specs = {"agentscope": {}, "agentscope._utils": {}, "agentscope._utils._common": {"_json_loads_with_repair": repair},
                 "agentscope.middleware": {"MiddlewareBase": object}, "agentscope.message": {"Msg": Model, "TextBlock": Model, "ToolResultState": st},
                 "agentscope.event": ev, "agentscope.tool": {"ToolChunk": Model, "ToolResponse": Model, "_toolkit": tk_mod}, "agentscope.tool._toolkit": tk_mod.__dict__}
        typed = {k: (tk_mod if k == "agentscope.tool._toolkit" else types.ModuleType(k)) for k in specs}
        for k, v in specs.items(): typed[k].__dict__.update(v)
        self.addCleanup(patch.dict(sys.modules, typed).stop)
        patch.dict(sys.modules, typed).start()
        pkg, qwen_dir = "aw_qwenpaw_resident_unit", ROOT / "crates/aw-adapter/native/qwenpaw"
        spec = importlib.util.spec_from_file_location(pkg, qwen_dir / "plugin.py", submodule_search_locations=[str(qwen_dir)])
        sys.modules[pkg] = plugin = importlib.util.module_from_spec(spec)
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")])
        spec.loader.exec_module(plugin)
        class Scope:
            def __init__(self, cfg): self.invocation_limit, self.callbacks, self.probe, self.on_failure = 4, [], None, None
            def check(self): self.probe and self.probe()
            async def open(self, *a, **k): pass
            async def validate(self): self.check()
            async def register(self, *a): self.check()
            async def finish(self, *a): pass
            async def close(self): pass
            async def invoke_native(self, hook, payload):
                self.callbacks.append((hook["event"], hook["step"], json.loads(json.dumps(payload))))
                return await self.handler(hook, payload)
        scope_mod = types.ModuleType(pkg + ".aw_scope")
        scope_mod.Scope, scope_mod.ScopeTransportFailure = Scope, type("ScopeTransportFailure", (RuntimeError,), {})
        sys.modules[pkg + ".aw_scope"] = scope_mod
        return (plugin, importlib.import_module(pkg + ".aw_resident"), importlib.import_module(pkg + ".aw_native"),
                importlib.import_module(pkg + ".aw_private"), Model, tk_mod)

    async def execute_call(self, resident, native, Model, tk_mod, mw, call, schema, executed, mcp_seen=None):
        class Toolkit:
            async def check_tool_available(self, name, groups): return types.SimpleNamespace(input_schema=schema)
            async def call_tool(self, tc, state):
                inv = resident.current_call.get()
                if mcp_seen is not None: mcp_seen.append(inv.mcp_dispatch.arguments if getattr(inv, "mcp_dispatch", None) else None)
                executed.append(tk_mod._json_loads_with_repair(tc.input, schema))
                res = await inv.owner.hooks(inv, "tool.after", Model(state="success", content=[Model(text="ok")]))
                inv.native_done = True
                yield res
        async def noop(*a, **k): pass
        agent = types.SimpleNamespace(toolkit=Toolkit(), _reply_middlewares=[mw], _acting_middlewares=[mw], reply_stream=noop, reply=noop, close=noop,
                                      state=types.SimpleNamespace(session_id="s", tool_context=types.SimpleNamespace(activated_groups=[])))
        agent._acting_impl = lambda tc: agent.toolkit.call_tool(tc, agent.state)
        native.bind_agent(agent)
        return [x async for x in mw.on_reply(agent, {}, lambda: mw.on_acting(agent, {"tool_call": call}, lambda **kw: agent._acting_impl(kw["tool_call"])))]

    async def test_updated_input_propagates_across_before_steps_and_post_denial(self):
        plugin, resident, native, _, Model, tk_mod = self.load_resident()
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
        hooks = [{"event": e, "step": s, "on_error": a, "budget_ms": 1000} for e, s, a in (("tool.before", "rewrite", "block"), ("tool.before", "gate", "block"), ("tool.after", "audit", "report"))]
        for deny_second in (False, True):
            with self.subTest(deny_second=deny_second):
                executed, mw = [], resident.ResidentMiddleware({"hooks": hooks}, plugin.AwToolMiddleware, types.SimpleNamespace())
                mw.scope.handler = lambda h, p, d=deny_second: asyncio.sleep(0, (0, b'{"aw_updated_input":{"value":"42"}}', b"") if h["step"] == "rewrite" else ((2, b"", b"") if h["step"] == "gate" and d else (0, b"", b"")))
                out = await self.execute_call(resident, native, Model, tk_mod, mw, Model(id="c1", name="tool", input='{"value":1}'), schema, executed)
                self.assertEqual([(e, s, json.loads(p["tool_call"]["input"])) for e, s, p in mw.scope.callbacks],
                                 [("tool.before", "rewrite", {"value": 1}), ("tool.before", "gate", {"value": 42}), ("tool.after", "audit", {"value": 42})])
                self.assertEqual((executed, out[-1].state), (([] if deny_second else [{"value": 42}]), ("denied" if deny_second else "success")))

    async def test_updated_input_enforces_mutation_mcp_and_credential_guards(self):
        from contextlib import asynccontextmanager
        from dataclasses import dataclass
        plugin, resident, native, private, Model, tk_mod = self.load_resident()
        FakeMcp = dataclass(frozen=True)(type("FakeMcp", (), {"__annotations__": {"arguments": str}, "payload": lambda s: {"arguments": s.arguments}}))
        @asynccontextmanager
        async def lease_factory():
            yield private.PrivateExecution({}, {}, lambda v: json.loads(json.dumps(v).replace("secret-token", "[redacted]")), {"SECRET": "secret-token"})
        hooks = [{"event": "tool.before", "step": "rewrite", "on_error": "block", "budget_ms": 1000}]
        for case in ("mutated_call", "secret_input", "mcp_sync"):
            with self.subTest(case=case):
                executed, mcp_seen, ctx = [], [], types.SimpleNamespace(aw_execution_factory=lease_factory, aw_environment_version=1)
                mw, call = resident.ResidentMiddleware({"hooks": hooks}, plugin.AwToolMiddleware, ctx), Model(id="c1", name="tool", input='{"value":"clean"}')
                async def handler(hook, payload):
                    inv = next(iter(mw.active.values()))
                    inv.mcp_dispatch = FakeMcp(inv.arguments)
                    if case == "mutated_call": call.input = '{"value":"tampered"}'
                    val = "secret-token" if case == "secret_input" else "approved"
                    return 0, f'{{"aw_updated_input":{{"value":"{val}"}}}}'.encode(), b""
                mw.scope.handler = handler
                if case in ("mutated_call", "secret_input"):
                    with self.assertRaises(resident.ResidentFailure):
                        await self.execute_call(resident, native, Model, tk_mod, mw, call, {}, executed, mcp_seen)
                    self.assertEqual((executed, call.input if case == "secret_input" else None), ([], '{"value": "clean"}' if case == "secret_input" else None))
                else:
                    await self.execute_call(resident, native, Model, tk_mod, mw, call, {}, executed, mcp_seen)
                    self.assertEqual((executed, mcp_seen), ([{"value": "approved"}], ['{"value": "approved"}']))


if __name__ == "__main__":
    unittest.main()
