"""Overflow tolerance for campaign aggregation over metrics artifacts.

Campaign artifacts are read with plain ``json.loads`` (via ``read_json``),
which happily produces arbitrary-precision integers. A corrupted counter
such as ``10**1000`` makes ``float()`` and ``math.isfinite()`` raise
``OverflowError``, which crashed report aggregation (``nested``), the quick
overhead comparison (``quick_value``) and the measurement gates
(``numeric_measurement``), while the single-run ``benchmark_stats.numeric``
and ``campaign_config.finite_number`` already tolerate the same outliers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARK_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(BENCHMARK_DIR / "campaign"))

import aggregate_report
import campaign_evaluation
import run_campaign

HUGE = json.loads("1" + "0" * 1000)


def _write_quick_fixture(results: Path) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for version, scenario in (
        ("baseline", "smoke"),
        ("optimized", "metrics_off"),
        ("optimized", "smoke"),
    ):
        path = results / f"runs/{scenario}/{version}/qps-10/rep-1/run-result.json"
        run_campaign.write_json(
            path,
            {
                "version": version,
                "metrics_enabled": scenario == "smoke",
                "evaluation": {"verdict": "PASS"},
                "summary": {
                    "http_success_rate": 1.0,
                    "trace_completeness": 1.0,
                    "capture_loss_rate": 0.0,
                    "process_survived": True,
                    "effective_qps": 10.0,
                    "latency_ms": {"p99": 20.0},
                    "resources": {"cpu_pct": {"avg": 10.0}, "rss_mb": {"avg": 100.0}},
                },
            },
        )
        paths[(version, scenario)] = path
    snapshot = results / "runtime/optimized/internal-metrics.txt"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text('agentsight_stage_calls_total{stage="parser"} 1\n', encoding="utf-8")
    return paths


def test_nested_ignores_overflowing_integers() -> None:
    assert aggregate_report.nested({"rss_mb": {"max": HUGE}}, "rss_mb", "max") is None


def test_nested_ignores_overflowing_json_outliers() -> None:
    # json.dumps/json.loads round-trips arbitrary-precision integers,
    # exactly like a metrics artifact written by a corrupted counter.
    artifact = json.dumps({'resources': {'rss_mb': {'max': HUGE}}})
    summary = json.loads(artifact)
    assert aggregate_report.nested(summary, 'resources', 'rss_mb', 'max') is None


def test_nested_keeps_ordinary_values_and_booleans() -> None:
    assert aggregate_report.nested({"rss_mb": {"max": 5.5}}, "rss_mb", "max") == 5.5
    assert aggregate_report.nested({"gate": True}, "gate") is True
    assert aggregate_report.nested({"name": "agent"}, "name") is None


def test_numeric_measurement_ignores_overflowing_integers() -> None:
    assert campaign_evaluation.numeric_measurement(HUGE) is None


def test_numeric_measurement_keeps_usable_values() -> None:
    assert campaign_evaluation.numeric_measurement(7) == 7.0
    assert campaign_evaluation.numeric_measurement(0.5) == 0.5
    assert campaign_evaluation.numeric_measurement(True) is None


def test_quick_value_ignores_overflowing_integers() -> None:
    assert run_campaign.quick_value({"summary": {"x": HUGE}}, ("summary", "x")) is None


def test_quick_value_keeps_ordinary_values() -> None:
    assert run_campaign.quick_value({"summary": {"x": 3.5}}, ("summary", "x")) == 3.5
    assert run_campaign.quick_value({"summary": {}}, ("summary", "x")) is None


@pytest.mark.parametrize(
    ("value", "suffix", "expected"),
    [
        (None, "", "—"),
        (10, "", "10.000"),
        (12.5, "%", "12.500%"),
        (True, "", "—"),
        (False, "%", "—"),
        ("bad", "", "—"),
        ("1.5", "", "—"),
        (float("nan"), "", "—"),
        (float("inf"), "%", "—"),
        (float("-inf"), "", "—"),
        (HUGE, "", "—"),
        (-HUGE, "%", "—"),
    ],
    ids=[
        "none",
        "int",
        "float-pct",
        "bool-true",
        "bool-false",
        "str-bad",
        "str-numeric",
        "nan",
        "pos-inf",
        "neg-inf",
        "huge-pos",
        "huge-neg",
    ],
)
def test_quick_number_guards_non_numeric_bool_non_finite_and_overflow(
    value: Any, suffix: str, expected: str
) -> None:
    assert run_campaign.quick_number(value, suffix) == expected


@pytest.mark.parametrize("kind", ["json_error", "unicode_error", "non_dict", "os_error"])
def test_write_quick_report_handles_unreadable_run_json(
    tmp_path: Path, kind: str
) -> None:
    paths = _write_quick_fixture(tmp_path)
    target = paths[("baseline", "smoke")]
    if kind == "json_error":
        target.write_text("{truncated", encoding="utf-8")
    elif kind == "unicode_error":
        target.write_bytes(b"\xff\xfe\x00")
    elif kind == "non_dict":
        target.write_text("[]", encoding="utf-8")
    else:
        target.unlink()
        target.mkdir()

    report_path, verdict = run_campaign.write_quick_report(tmp_path)
    assert verdict == "INCONCLUSIVE"
    report = run_campaign.read_json(report_path)
    issues = report["observability_overhead"]["issues"]
    assert report["verdict"] == "INCONCLUSIVE"
    assert any("unreadable or malformed quick run" in item for item in issues)
    assert any("missing quick run: baseline:smoke" in item for item in issues)
    assert "Overall smoke verdict: **INCONCLUSIVE**" in (
        tmp_path / "quick-report.md"
    ).read_text(encoding="utf-8")


def test_write_quick_report_handles_non_dict_summary_evaluation_and_version(
    tmp_path: Path,
) -> None:
    paths = _write_quick_fixture(tmp_path)
    baseline = run_campaign.read_json(paths[("baseline", "smoke")])
    baseline["summary"] = None
    run_campaign.write_json(paths[("baseline", "smoke")], baseline)

    optimized = run_campaign.read_json(paths[("optimized", "smoke")])
    optimized["evaluation"] = ["PASS"]
    run_campaign.write_json(paths[("optimized", "smoke")], optimized)

    metrics_off = run_campaign.read_json(paths[("optimized", "metrics_off")])
    metrics_off["evaluation"] = {"verdict": ["PASS"]}
    run_campaign.write_json(paths[("optimized", "metrics_off")], metrics_off)

    extra = tmp_path / "runs/smoke/extra/run-result.json"
    run_campaign.write_json(extra, {"version": ["baseline"], "summary": {}, "evaluation": {}})

    report_path, verdict = run_campaign.write_quick_report(tmp_path)
    assert verdict == "INCONCLUSIVE"
    report = run_campaign.read_json(report_path)
    issues = report["observability_overhead"]["issues"]
    assert "malformed quick run summary: baseline:smoke" in issues
    assert "malformed quick run evaluation: optimized:smoke" in issues
    assert "malformed quick run verdict: optimized:metrics_off" in issues
    assert any("unexpected or duplicate quick run" in item for item in issues)
    markdown = (tmp_path / "quick-report.md").read_text(encoding="utf-8")
    assert "| baseline | on | PASS | — |" in markdown
    assert "| optimized | on | — | 1.000 |" in markdown


def test_write_quick_report_handles_malformed_and_overflowing_metrics(
    tmp_path: Path,
) -> None:
    paths = _write_quick_fixture(tmp_path)
    baseline = run_campaign.read_json(paths[("baseline", "smoke")])
    baseline["summary"]["trace_completeness"] = HUGE
    run_campaign.write_json(paths[("baseline", "smoke")], baseline)

    off = run_campaign.read_json(paths[("optimized", "metrics_off")])
    off["summary"]["http_success_rate"] = True
    off["summary"]["capture_loss_rate"] = float("nan")
    off["summary"]["process_survived"] = "yes"
    off["summary"]["resources"]["cpu_pct"]["avg"] = -1e308
    run_campaign.write_json(paths[("optimized", "metrics_off")], off)

    on = run_campaign.read_json(paths[("optimized", "smoke")])
    on["summary"]["trace_completeness"] = "N/A"
    on["summary"]["resources"]["cpu_pct"]["avg"] = 1e308
    run_campaign.write_json(paths[("optimized", "smoke")], on)

    report_path, verdict = run_campaign.write_quick_report(tmp_path)
    assert verdict == "INCONCLUSIVE"
    # Strict JSON check ensures no NaN/Infinity leaked into quick-report.json.
    report = json.loads(
        report_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    cpu = report["observability_overhead"]["measurements"]["cpu_avg_pct"]
    assert cpu["delta_enabled_minus_disabled"] is None
    assert "| baseline | on | PASS | — |" in (
        tmp_path / "quick-report.md"
    ).read_text(encoding="utf-8")


def test_write_quick_report_valid_pass_and_fail_precedence(tmp_path: Path) -> None:
    paths = _write_quick_fixture(tmp_path)
    report_path, verdict = run_campaign.write_quick_report(tmp_path)
    assert verdict == "PASS"
    assert run_campaign.read_json(report_path)["observability_overhead"]["issues"] == []

    paths[("baseline", "smoke")].write_text("{corrupt", encoding="utf-8")
    on = run_campaign.read_json(paths[("optimized", "smoke")])
    on["evaluation"]["verdict"] = "FAIL"
    run_campaign.write_json(paths[("optimized", "smoke")], on)
    _, fail_verdict = run_campaign.write_quick_report(tmp_path)
    assert fail_verdict == "FAIL"
