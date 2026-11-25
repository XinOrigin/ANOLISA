#!/usr/bin/env python3
"""Tests for pinned source validation in capture_ctest.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
COMPONENT_ROOT = TESTS_DIR.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import capture_ctest  # noqa: E402


class TestCaptureCtestTimeout(unittest.TestCase):
    def test_capture_times_out_after_pipe_eof_and_cleans_up_container(self) -> None:
        script = (
            "printf 'before-eof\\n'; "
            "printf 'err-before-eof\\n' >&2; "
            "exec 1>&- 2>&-; "
            "sleep 0.8"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            started = time.monotonic()
            with (
                mock.patch.object(capture_ctest.subprocess, "run") as mock_run,
                self.assertRaises(TimeoutError) as ctx,
            ):
                capture_ctest.capture(
                    ["/bin/sh", "-c", script],
                    ["docker", "run", "demo"],
                    out_dir,
                    container_name="tok-test-eof",
                    timeout_seconds=0.15,
                )
            elapsed = time.monotonic() - started
            self.assertIn("exceeded", str(ctx.exception))
            self.assertLess(elapsed, 0.5)
            mock_run.assert_called_once_with(
                ["docker", "rm", "-f", "tok-test-eof"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            for artifact in (
                "capture.json",
                "raw_bytes.json",
                "stdout.txt",
                "stderr.txt",
                "observed.txt",
            ):
                self.assertFalse((out_dir / artifact).exists())

    def test_capture_times_out_with_open_pipes_without_select_overshoot(self) -> None:
        script = "sleep 0.8"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            started = time.monotonic()
            with (
                mock.patch.object(capture_ctest.subprocess, "run") as mock_run,
                self.assertRaises(TimeoutError),
            ):
                capture_ctest.capture(
                    ["/bin/sh", "-c", script],
                    ["docker", "run", "demo"],
                    out_dir,
                    container_name="tok-test-open",
                    timeout_seconds=0.15,
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.5)
            mock_run.assert_called_once_with(
                ["docker", "rm", "-f", "tok-test-open"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.assertFalse((out_dir / "capture.json").exists())
            self.assertFalse((out_dir / "raw_bytes.json").exists())

    def test_capture_timeout_kills_process_even_when_docker_rm_raises_oserror(self) -> None:
        script = "exec 1>&- 2>&-; sleep 0.8"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            started = time.monotonic()
            with (
                mock.patch.object(
                    capture_ctest.subprocess, "run", side_effect=FileNotFoundError("docker")
                ),
                self.assertRaises(TimeoutError),
            ):
                capture_ctest.capture(
                    ["/bin/sh", "-c", script],
                    ["docker", "run", "demo"],
                    out_dir,
                    container_name="tok-test-missing-docker",
                    timeout_seconds=0.15,
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.5)
            self.assertFalse((out_dir / "capture.json").exists())

    def test_capture_records_bytes_and_order_when_completed_before_deadline(self) -> None:
        script = (
            "import sys, time; "
            "sys.stdout.buffer.write(b'first\\r\\n'); sys.stdout.buffer.flush(); "
            "time.sleep(0.03); "
            "sys.stderr.buffer.write(b'second\\x00\\n'); sys.stderr.buffer.flush(); "
            "sys.exit(3)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            with mock.patch.object(capture_ctest.subprocess, "run") as mock_run:
                record = capture_ctest.capture(
                    [sys.executable, "-c", script],
                    ["docker", "run", "demo"],
                    out_dir,
                    container_name="tok-test-ok",
                    timeout_seconds=5.0,
                )
            mock_run.assert_not_called()
            self.assertEqual(record["exit_code"], 3)
            self.assertEqual(record["stdout_cr"], 1)
            self.assertEqual(record["stdout_lf"], 1)
            self.assertEqual(record["stderr_nul"], 1)
            self.assertEqual(record["observed_bytes"], 15)
            self.assertEqual(
                record["observed_chunks"],
                [{"stream": "stdout", "bytes": 7}, {"stream": "stderr", "bytes": 8}],
            )
            self.assertEqual((out_dir / "stdout.txt").read_bytes(), b"first\r\n")
            self.assertEqual((out_dir / "stderr.txt").read_bytes(), b"second\x00\n")
            self.assertEqual((out_dir / "observed.txt").read_bytes(), b"first\r\nsecond\x00\n")
            capture_meta = json.loads((out_dir / "capture.json").read_text(encoding="utf-8"))
            raw_meta = json.loads((out_dir / "raw_bytes.json").read_text(encoding="utf-8"))
            self.assertEqual(capture_meta["exit_code"], 3)
            self.assertEqual(raw_meta, {"command": ["docker", "run", "demo"], "exit_code": 3})


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.STDOUT).decode()


def init_repo(repo: Path, filename: str, content: str) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "--quiet", str(repo)])
    git(repo, "config", "user.name", "Tokenless Test")
    git(repo, "config", "user.email", "tokenless-test@example.com")
    (repo / filename).write_text(content, encoding="utf-8")
    git(repo, "add", filename)
    git(repo, "commit", "--quiet", "-m", f"add {filename}")
    return git(repo, "rev-parse", "HEAD").strip()


class TestCaptureCtestSourcePinning(unittest.TestCase):
    def test_rejects_unrelated_cached_repo(self) -> None:
        _, tag, expected_commit, _, _ = capture_ctest.PROJECTS["brotli"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root / "sources" / "brotli", "README.md", "unrelated\n")
            with self.assertRaises(SystemExit) as ctx:
                capture_ctest.resolve_source(
                    "brotli",
                    "google/brotli",
                    tag,
                    expected_commit,
                    root / "dest",
                    root / "sources",
                )
            self.assertIn(expected_commit, str(ctx.exception))

    def test_rejects_missing_cached_project_without_network(self) -> None:
        repo, tag, expected_commit, _, _ = capture_ctest.PROJECTS["brotli"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            with self.assertRaises(SystemExit) as ctx:
                capture_ctest.resolve_source(
                    "brotli",
                    repo,
                    tag,
                    expected_commit,
                    root / "dest",
                    root / "sources",
                )
            self.assertIn("missing cached repository", str(ctx.exception))

    def test_accepts_exact_head_and_ancestor_commit_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "sources" / "demo"
            pinned_commit = init_repo(cached, "src.txt", "pinned\n")
            git(cached, "tag", "-a", "-m", "v1.0.0", "v1.0.0")

            dest_exact = root / "dest_exact"
            resolved_exact, clone_exact = capture_ctest.resolve_source(
                "demo",
                "example/demo",
                "v1.0.0",
                pinned_commit,
                dest_exact,
                root / "sources",
            )
            self.assertEqual(resolved_exact, pinned_commit)
            self.assertEqual(
                clone_exact,
                "git clone --depth 1 file://<cached> (pinned tag+commit, unmodified)",
            )
            self.assertEqual((dest_exact / "src.txt").read_text(encoding="utf-8"), "pinned\n")

            (cached / "src.txt").write_text("newer\n", encoding="utf-8")
            git(cached, "add", "src.txt")
            git(cached, "commit", "--quiet", "-m", "advance head")

            dest_ancestor = root / "dest_ancestor"
            resolved_ancestor, clone_ancestor = capture_ctest.resolve_source(
                "demo",
                "example/demo",
                "v1.0.0",
                pinned_commit,
                dest_ancestor,
                root / "sources",
            )
            self.assertEqual(resolved_ancestor, pinned_commit)
            self.assertIn("detached checkout", clone_ancestor)
            self.assertEqual(
                (dest_ancestor / "src.txt").read_text(encoding="utf-8"), "pinned\n"
            )

    def test_rejects_missing_or_mismatched_cached_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "sources" / "demo"
            pinned_commit = init_repo(cached, "src.txt", "pinned\n")

            with self.assertRaises(SystemExit) as missing_ctx:
                capture_ctest.resolve_source(
                    "demo",
                    "example/demo",
                    "v1.0.0",
                    pinned_commit,
                    root / "dest_missing",
                    root / "sources",
                )
            self.assertIn("missing tag v1.0.0", str(missing_ctx.exception))

            (cached / "src.txt").write_text("newer\n", encoding="utf-8")
            git(cached, "add", "src.txt")
            git(cached, "commit", "--quiet", "-m", "advance head")
            git(cached, "tag", "v1.0.0")

            with self.assertRaises(SystemExit) as mismatch_ctx:
                capture_ctest.resolve_source(
                    "demo",
                    "example/demo",
                    "v1.0.0",
                    pinned_commit,
                    root / "dest_mismatch",
                    root / "sources",
                )
            self.assertIn("cached tag v1.0.0 resolves to", str(mismatch_ctx.exception))

    def test_accepts_relative_sources_dir_for_shallow_and_detached_clones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "cache" / "demo"
            pinned_commit = init_repo(cached, "src.txt", "pinned\n")
            git(cached, "tag", "-a", "-m", "v1.0.0", "v1.0.0")

            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                resolved_exact, clone_exact = capture_ctest.resolve_source(
                    "demo",
                    "example/demo",
                    "v1.0.0",
                    pinned_commit,
                    Path("out_exact"),
                    Path("cache"),
                )
                self.assertEqual(resolved_exact, pinned_commit)
                self.assertEqual(
                    clone_exact,
                    "git clone --depth 1 file://<cached> (pinned tag+commit, unmodified)",
                )
                self.assertEqual((root / "out_exact" / "src.txt").read_text(encoding="utf-8"), "pinned\n")

                (cached / "src.txt").write_text("newer\n", encoding="utf-8")
                git(cached, "add", "src.txt")
                git(cached, "commit", "--quiet", "-m", "advance head")

                resolved_detached, clone_detached = capture_ctest.resolve_source(
                    "demo",
                    "example/demo",
                    "v1.0.0",
                    pinned_commit,
                    Path("out_detached"),
                    Path("cache"),
                )
                self.assertEqual(resolved_detached, pinned_commit)
                self.assertIn("detached checkout", clone_detached)
                self.assertEqual(
                    (root / "out_detached" / "src.txt").read_text(encoding="utf-8"), "pinned\n"
                )
            finally:
                os.chdir(old_cwd)

    def test_projects_match_committed_provenance(self) -> None:
        for rel_path, expected_count in (
            ("crates/tokenless-compressors/tests/fixtures/build_logs/ctest/provenance.json", 6),
            ("crates/tokenless-compressors/tests/fixtures/build_logs/cmake_build/provenance.json", 8),
        ):
            entries = json.loads((COMPONENT_ROOT / rel_path).read_text(encoding="utf-8"))
            self.assertEqual(len(entries), expected_count)
            by_repo = {
                entry["environment"]["repository"]: (
                    entry["environment"]["tag"],
                    entry["environment"]["commit"],
                    entry["environment"]["cmake_options"],
                )
                for entry in entries
            }
            for repo, (tag, commit, cmake_opts) in by_repo.items():
                matching = [
                    spec for spec in capture_ctest.PROJECTS.values() if spec[0] == repo
                ]
                self.assertEqual(len(matching), 1)
                self.assertEqual((tag, commit, cmake_opts), matching[0][1:4])

        ctest_entries = json.loads(
            (
                COMPONENT_ROOT
                / "crates/tokenless-compressors/tests/fixtures/build_logs/ctest/provenance.json"
            ).read_text(encoding="utf-8")
        )
        ctest_repos = {entry["environment"]["repository"] for entry in ctest_entries}
        for repo, *_ in capture_ctest.PROJECTS.values():
            self.assertIn(repo, ctest_repos)

    def test_suite_flag_defaults_to_ctest_and_accepts_build_suites(self) -> None:
        parser = capture_ctest.build_parser()
        default_args = parser.parse_args(["--project", "brotli", "--output", "/tmp/out"])
        self.assertEqual(default_args.suite, "ctest")
        for suite in ("ctest", "build", "all"):
            parsed = parser.parse_args(
                ["--project", "brotli", "--suite", suite, "--output", "/tmp/out"]
            )
            self.assertEqual(parsed.suite, suite)


if __name__ == "__main__":
    unittest.main()
