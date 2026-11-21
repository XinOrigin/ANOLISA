"""Execute merge-group change detection against real merge commits."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS_WORKFLOW = ROOT / ".github/workflows/ci.yaml"
RUNNER_WORKFLOW = ROOT / ".github/workflows/agentsight-runner-test.yml"
GOVERNANCE_WORKFLOW = ROOT / ".github/workflows/repository-metadata.yml"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def run_script(script: str, repo: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def extract_run(workflow: str, marker: str, next_marker: str) -> str:
    block = workflow.split(marker, 1)[1]
    return textwrap.dedent(
        block.split("        run: |\n", 1)[1].split(next_marker, 1)[0]
    )


class MergeQueueWorkflowTests(unittest.TestCase):
    def make_merge(self, root: Path, changed_path: str) -> tuple[str, str]:
        subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
        git(root, "config", "user.email", "ci@example.invalid")
        git(root, "config", "user.name", "CI Test")
        target = root / changed_path
        target.parent.mkdir(parents=True)
        target.write_text("base\n")
        git(root, "add", changed_path)
        git(root, "commit", "-qm", "base")
        base = git(root, "rev-parse", "HEAD")
        git(root, "checkout", "-qb", "feature")
        target.write_text("changed\n")
        git(root, "add", changed_path)
        git(root, "commit", "-qm", "change")
        git(root, "checkout", "-q", "main")
        git(root, "merge", "--no-ff", "--no-edit", "feature")
        head = git(root, "rev-parse", "HEAD")
        return base, head

    def run_runner(self, repo: Path, base: str, head: str, checkout: str) -> str:
        workflow = RUNNER_WORKFLOW.read_text()
        script = extract_run(
            workflow, "      - name: Select AgentSight validation\n", "  standard:\n"
        )
        output = repo / "github-output"
        environment = {
            **os.environ,
            "EVENT_NAME": "merge_group",
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "CHECKOUT_SHA": checkout,
            "RUNNER_TEMP": str(repo),
            "GITHUB_OUTPUT": str(output),
        }
        run_script(script, repo, environment)
        return output.read_text().strip()

    def run_components(self, repo: Path, base: str, head: str) -> str:
        workflow = COMPONENTS_WORKFLOW.read_text()
        script = extract_run(
            workflow,
            "      - name: Check which components changed\n",
            "  # =========================================================================\n  # Step 2:",
        )
        script = re.sub(r"\$\{\{\s*inputs\.[A-Za-z0-9_]+\s*\}\}", "false", script)
        output = repo / "github-output"
        summary = repo / "github-summary"
        environment = {
            **os.environ,
            "EVENT_NAME": "merge_group",
            "MERGE_GROUP_BASE_SHA": base,
            "MERGE_GROUP_HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        }
        run_script(script, repo, environment)
        return output.read_text()

    def test_workflows_declare_merge_group_trigger(self) -> None:
        for workflow in (COMPONENTS_WORKFLOW, RUNNER_WORKFLOW, GOVERNANCE_WORKFLOW):
            with self.subTest(workflow=workflow.name):
                self.assertRegex(workflow.read_text(), r"(?m)^  merge_group:\s*$")
        detection = COMPONENTS_WORKFLOW.read_text().split("  detect-changes:\n", 1)[1]
        self.assertIn(
            "fetch-depth: 0", detection.split("  check-component-versions:\n", 1)[0]
        )

    def test_runner_selects_only_agentsight_changes(self) -> None:
        for changed_path, expected in (
            ("src/agentsight/core.txt", "run_ebpf=true"),
            ("docs/unrelated.txt", "run_ebpf=false"),
        ):
            with self.subTest(changed_path=changed_path):
                with tempfile.TemporaryDirectory() as temporary:
                    repo = Path(temporary)
                    base, head = self.make_merge(repo, changed_path)
                    self.assertEqual(self.run_runner(repo, base, head, head), expected)

    def test_runner_rejects_a_mismatched_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            base, head = self.make_merge(repo, "src/agentsight/core.txt")
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_runner(repo, base, head, base)

    def test_components_use_the_merge_group_range(self) -> None:
        for changed_path, expected in (
            ("src/agentsight/core.txt", "agentsight=true"),
            ("docs/unrelated.txt", "agentsight=false"),
        ):
            with self.subTest(changed_path=changed_path):
                with tempfile.TemporaryDirectory() as temporary:
                    repo = Path(temporary)
                    base, head = self.make_merge(repo, changed_path)
                    output = self.run_components(repo, base, head)
                    self.assertIn(expected, output.splitlines())

    def test_components_fail_closed_when_merge_group_objects_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            _, head = self.make_merge(repo, "src/agentsight/core.txt")
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_components(repo, "0" * 40, head)


if __name__ == "__main__":
    unittest.main()
