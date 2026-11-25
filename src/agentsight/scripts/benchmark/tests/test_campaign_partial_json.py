"""An interrupted artifact write must not erase the rest of a campaign report."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_benchmark import aggregate_report, campaign_data, campaign_evidence, complete_summary


@pytest.mark.parametrize(
    "artifact",
    ["runs/broken/run-result.json", "capacity-baseline.json", "regression.json", "manifest.json"],
)
@pytest.mark.parametrize("content", [b'{"incomplete":', b"\xff\xfe"])
def test_cli_keeps_reports_after_an_interrupted_artifact(
    tmp_path: Path, artifact: str, content: bytes
) -> None:
    configuration = campaign_data(tmp_path)
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(configuration))
    (tmp_path / "manifest.json").write_text("{}")
    good = tmp_path / "runs/good/run-result.json"
    good.parent.mkdir(parents=True)
    good.write_text(
        json.dumps(
            {
                "scenario": "matrix",
                "version": "baseline",
                "qps": 100,
                "repetition": 1,
                "summary": complete_summary(),
                "evaluation": {"verdict": "PASS"},
            }
        )
    )
    broken = tmp_path / artifact
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(content)
    result = subprocess.run(
        [
            sys.executable,
            aggregate_report.__file__,
            "--campaign",
            str(config_path),
            "--results",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert str(broken) in result.stderr
    summary = json.loads((tmp_path / "final-summary.json").read_text())
    assert summary["verdict"] == "INCONCLUSIVE"
    assert "runs/good/run-result.json" in (tmp_path / "run-inventory.csv").read_text()
    assert "100" in (tmp_path / "performance-comparison.csv").read_text()
    assert broken.read_bytes() == content


def test_invalid_campaign_configuration_is_still_an_error(tmp_path: Path) -> None:
    config = tmp_path / "campaign.json"
    config.write_text("{")
    result = subprocess.run(
        [
            sys.executable,
            aggregate_report.__file__,
            "--campaign",
            str(config),
            "--results",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("content", [b'{"incomplete":', b"\xff\xfe"])
def test_corrupt_fault_evidence_keeps_reports_and_healthy_runs(
    tmp_path: Path, content: bytes
) -> None:
    configuration = campaign_data(tmp_path)
    config = tmp_path / "campaign.json"
    config.write_text(json.dumps(configuration))
    (tmp_path / "manifest.json").write_text("{}")
    for label, scenario in (("good", "matrix"), ("fault", "fault")):
        run = tmp_path / "runs" / label / "run-result.json"
        run.parent.mkdir(parents=True)
        run.write_text(
            json.dumps(
                {
                    "scenario": scenario,
                    "version": "baseline",
                    "qps": 100,
                    "repetition": 1,
                    "summary": complete_summary(),
                    "evaluation": {"verdict": "PASS"},
                }
            )
        )
    fault_run = tmp_path / "runs/fault/run-result.json"
    broken = fault_run.parent / "measurement/fault-results.json"
    broken.parent.mkdir()
    broken.write_bytes(content)
    result = subprocess.run(
        [
            sys.executable,
            aggregate_report.__file__,
            "--campaign",
            str(config),
            "--results",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert str(broken) in result.stderr
    for name in ("fault-report.md", "final-report.md", "final-summary.json"):
        assert (tmp_path / name).is_file(), name
    summary = json.loads((tmp_path / "final-summary.json").read_text())
    assert summary["verdict"] == "INCONCLUSIVE"
    inventory = (tmp_path / "run-inventory.csv").read_text()
    assert "runs/good/run-result.json" in inventory
    assert "runs/fault/run-result.json" in inventory
    assert "100" in (tmp_path / "performance-comparison.csv").read_text()
    outcome = campaign_evidence.fault_outcome(
        fault_run,
        json.loads(fault_run.read_text()),
        configuration["fault"],
        configuration["thresholds"],
    )
    assert outcome == {
        "verdict": "INCONCLUSIVE",
        "missing": ["fault-results.json"],
        "failed": [],
    }
    assert broken.read_bytes() == content


@pytest.mark.parametrize("bad_value", [None, ["bad"], 3, "bad", True])
def test_write_capacity_tolerates_non_object_subfields(
    tmp_path: Path, bad_value: object
) -> None:
    (tmp_path / "capacity-baseline.json").write_text(
        json.dumps(
            {
                "pretest": bad_value,
                "probes": bad_value,
                "confirmation": bad_value,
                "search_start_qps": 100,
                "maximum_sustainable_qps": 100,
                "first_failed_qps": 150,
                "boundary_confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capacity-optimized.json").write_bytes(b'{"incomplete":')

    capacities = aggregate_report.write_capacity(tmp_path)

    assert "_artifact_error" in capacities["optimized"]
    report = (tmp_path / "capacity-report.md").read_text(encoding="utf-8")
    assert "| baseline | — | 100 | — | 100 | — | 150 | True | — |" in report
    assert "| optimized | — | — | — | — | — | — | False | — |" in report


@pytest.mark.parametrize("bad_value", [None, ["bad"], 3, "bad", True])
@pytest.mark.parametrize("shape", ["top_level", "frozen_versions", "version_entry"])
def test_write_final_tolerates_non_object_manifest_and_capacity_subfields(
    tmp_path: Path, bad_value: object, shape: str
) -> None:
    configuration = campaign_data(tmp_path)
    if shape == "top_level":
        manifest = {"host": bad_value, "frozen": bad_value, "tools": bad_value}
    elif shape == "frozen_versions":
        manifest = {
            "host": {},
            "frozen": {"versions": bad_value, "comparison_mode": "ab_comparison"},
            "tools": bad_value,
        }
    else:
        manifest = {
            "host": {},
            "frozen": {
                "versions": {"baseline": bad_value, "optimized": bad_value},
                "comparison_mode": "ab_comparison",
            },
            "tools": {"k6": None, "openssl": None, "h2load": None},
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    capacities = {
        "baseline": {
            "pretest": bad_value,
            "search_start_qps": 100,
            "maximum_sustainable_qps": 100,
            "first_failed_qps": 150,
            "boundary_confirmed": True,
        },
        "optimized": bad_value,
    }

    summary = aggregate_report.write_final(
        tmp_path, configuration, [], capacities, [], {}, {}, {}
    )

    assert summary["verdict"] == "INCONCLUSIVE"
    report = (tmp_path / "final-report.md").read_text(encoding="utf-8")
    assert "| baseline | `baseline` | `—` | `—` |" in report
    assert "| baseline | — | 100 | 100 | 150 | True | — |" in report
    assert "| k6 | — |" in report
    assert "| OpenSSL | — |" in report
    assert "未安装 h2load" in report


def test_cli_tolerates_non_object_subfields_in_capacity_and_manifest(
    tmp_path: Path,
) -> None:
    configuration = campaign_data(tmp_path)
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "host": ["x86_64"],
                "frozen": {
                    "versions": {"baseline": "bad", "optimized": 42},
                    "comparison_mode": "ab_comparison",
                },
                "tools": "missing",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capacity-baseline.json").write_text(
        json.dumps(
            {
                "pretest": ["bad"],
                "probes": 5,
                "confirmation": "PASS",
                "maximum_sustainable_qps": 100,
                "first_failed_qps": 150,
                "boundary_confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capacity-optimized.json").write_bytes(b'{"incomplete":')
    good = tmp_path / "runs/good/run-result.json"
    good.parent.mkdir(parents=True)
    good.write_text(
        json.dumps(
            {
                "scenario": "matrix",
                "version": "baseline",
                "qps": 100,
                "repetition": 1,
                "summary": complete_summary(),
                "evaluation": {"verdict": "PASS"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            aggregate_report.__file__,
            "--campaign",
            str(config_path),
            "--results",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    capacity_report = (tmp_path / "capacity-report.md").read_text(encoding="utf-8")
    assert "| baseline | — | — | — | 100 | — | 150 | True | — |" in capacity_report
    assert "| optimized | — | — | — | — | — | — | False | — |" in capacity_report
    final_summary = json.loads((tmp_path / "final-summary.json").read_text())
    assert final_summary["verdict"] == "INCONCLUSIVE"
    assert "runs/good/run-result.json" in (tmp_path / "run-inventory.csv").read_text()


def test_comparison_summary_tolerates_non_dict_capacity_entries() -> None:
    comparison = aggregate_report.comparison_summary(
        [], {"baseline": None, "optimized": ["bad"]}, []
    )
    assert comparison["metrics"]["maximum_sustainable_qps"] == {
        "baseline": None,
        "optimized": None,
        "delta_pct": None,
    }
