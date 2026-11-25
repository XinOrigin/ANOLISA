# Copyright 2026 Alibaba Cloud
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "assets" / "scripts" / "analyze_solve_rate.py"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_fixture(evals: Path, runs: Path) -> None:
    repeat = runs / "host1" / "out" / "repeat-1"
    _write_json(repeat / "arm-a" / "selection.json", {"instance_ids": ["scored"]})
    preds = repeat / "arm-a" / "run" / "preds.json"
    _write_json(preds, [{"instance_id": "scored"}])
    _write_json(repeat / "arm-b" / "selection.json", {"instance_ids": ["waiting"]})

    cell = evals / "host1" / "repeat-1-arm-a"
    _write_json(
        cell / "evaluate" / "model.eval-id.json",
        {"resolved_ids": ["scored"], "submitted_instances": 1, "completed_instances": 1},
    )
    digest = hashlib.sha256(preds.read_bytes()).hexdigest()
    (cell / ".eval-complete").write_text(f"{digest} eval-id\n", encoding="utf-8")


def _run_cli(
    evals: str, runs: str | None, out: Path, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-X", "utf8", str(SCRIPT), "--evals", evals]
    if runs is not None:
        cmd.extend(["--runs", runs])
    cmd.extend(["--out", str(out)])
    return subprocess.run(
        cmd,
        check=False,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )


class SolveRatePathTests(unittest.TestCase):
    def test_bracketed_roots_match_plain_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            plain_evals, plain_runs = work / "plain-evals", work / "plain-runs"
            _build_fixture(plain_evals, plain_runs)
            plain_out = work / "plain.json"
            plain_res = _run_cli(str(plain_evals), str(plain_runs), plain_out)
            self.assertEqual(plain_res.returncode, 0, plain_res.stderr)
            plain = json.loads(plain_out.read_text(encoding="utf-8"))
            self.assertEqual(sorted(plain["per_repeat_arm"]), ["repeat1-arma", "repeat1-armb"])
            self.assertEqual(plain["prediction_verification"], "incomplete")
            self.assertEqual(plain["per_repeat_arm"]["repeat1-arma"]["solve_rate_pct"], 100.0)
            self.assertEqual(plain["per_repeat_arm"]["repeat1-armb"]["unscored_ids"], ["waiting"])

            for name, eval_name, runs_name in (
                ("both-bracketed", "evals[ab]", "runs[ab]"),
                ("runs-bracketed", "other-evals", "other-runs[ab]"),
                ("evals-bracketed", "only-evals[ab]", "only-runs"),
            ):
                with self.subTest(case=name):
                    evals, runs = work / eval_name, work / runs_name
                    _build_fixture(evals, runs)
                    out = work / f"{name}.json"
                    res = _run_cli(str(evals), str(runs), out)
                    self.assertEqual(res.returncode, 0, res.stderr)
                    self.assertEqual(json.loads(out.read_text(encoding="utf-8")), plain)

    def test_empty_directory_does_not_scan_cwd(self) -> None:
        spec = importlib.util.spec_from_file_location("l4_solve_rate_paths", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _build_fixture(work, work)
            old_cwd = os.getcwd()
            try:
                os.chdir(work)
                self.assertEqual(
                    module.glob_under("", "**/repeat-*-arm-*/evaluate/*.json", recursive=True),
                    [],
                )
                self.assertEqual(list(module.unscored_cells(str(work), "", set())), [])
            finally:
                os.chdir(old_cwd)
            res = _run_cli("", None, work / "empty.json", cwd=work)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("no evaluation reports found", res.stderr)


if __name__ == "__main__":
    unittest.main()
