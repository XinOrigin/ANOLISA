#!/usr/bin/env python3
"""Offline argument-validation tests for tests/run-all-tests.sh."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE_SCRIPT = Path(__file__).resolve().parent / "run-all-tests.sh"


class RunAllTestsArgValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls_log = self.root / "calls.log"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()

        # Mirror repo layout expected by run-all-tests.sh
        tests_dir = self.root / "tests"
        tests_dir.mkdir()
        self.script = tests_dir / "run-all-tests.sh"
        self.script.write_text(SOURCE_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        self.script.chmod(0o755)

        for rel in (
            "deprecated/copilot-shell",
            "src/agent-sec-core",
            "src/agentsight",
            "src/tokenless",
            "src/agent-memory",
        ):
            (self.root / rel).mkdir(parents=True)

        for tool in ("npm", "uv", "make", "cargo"):
            shim = self.bin_dir / tool
            shim.write_text(
                f'#!/bin/bash\necho "{tool}:$*" >> "{self.calls_log}"\nexit 0\n',
                encoding="utf-8",
            )
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR)

        self.env = {
            "PATH": f"{self.bin_dir}:/bin:/usr/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), *args],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )

    def _recorded_calls(self) -> list[str]:
        if not self.calls_log.exists():
            return []
        return self.calls_log.read_text(encoding="utf-8").splitlines()

    def test_help_exits_zero_without_running_tests(self) -> None:
        for flag in ("-h", "--help"):
            res = self._run(flag)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("--filter", res.stdout)
            self.assertEqual(self._recorded_calls(), [])

    def test_malformed_invocations_fail_without_dispatch(self) -> None:
        cases = [
            ("--filter",),
            ("--filter", ""),
            ("--unknown",),
            ("sight",),
            ("--filter", "sight", "extra"),
            ("--filter", "shell", "--help"),
        ]
        for args in cases:
            with self.subTest(args=args):
                res = self._run(*args)
                self.assertNotEqual(res.returncode, 0, f"expected non-zero for {args}")
                self.assertEqual(self._recorded_calls(), [], f"dispatched for {args}")

    def test_unknown_filter_keeps_exit_1_without_dispatch(self) -> None:
        res = self._run("--filter", "bogus")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Unknown filter: bogus", res.stdout + res.stderr)
        self.assertEqual(self._recorded_calls(), [])

    def test_no_args_runs_aggregate_suite(self) -> None:
        res = self._run()
        self.assertEqual(res.returncode, 0, res.stderr)
        calls = self._recorded_calls()
        self.assertTrue(any(c.startswith("npm:") for c in calls))
        self.assertTrue(any(c.startswith("cargo:") for c in calls))

    def test_valid_single_filter_runs_only_target(self) -> None:
        res = self._run("--filter", "sight")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self._recorded_calls(), ["cargo:test"])


if __name__ == "__main__":
    unittest.main()
