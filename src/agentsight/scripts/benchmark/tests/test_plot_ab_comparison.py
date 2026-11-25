"""Regression tests for metric hardening in plot_ab_comparison.py."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

SINGLE_RUN = Path(__file__).parents[1] / "single_run"
if str(SINGLE_RUN) not in sys.path:
    sys.path.insert(0, str(SINGLE_RUN))

import plot_ab_comparison as ab


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        pytest.param(10**1000, id="huge-int"),
        pytest.param(-(10**1000), id="neg-huge-int"),
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "123",
        [1],
        {"a": 1},
    ],
)
def test_number_rejects_booleans_overflow_and_non_finite(value: Any) -> None:
    assert ab.number(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (-0.0, 0.0),
        (42, 42.0),
        (-3.25, -3.25),
    ],
)
def test_number_accepts_finite_ints_and_floats(value: float, expected: float) -> None:
    assert ab.number(value) == pytest.approx(expected)


@pytest.mark.parametrize("bad_sub", [None, "bad", 123, [1, 2], True])
def test_run_value_handles_non_dict_resource_subfields(bad_sub: Any) -> None:
    run = {
        "summary": {
            "latency_ms": bad_sub,
            "tokens_per_second": bad_sub,
        },
        "report": {},
        "resources": {
            "rss_mb": bad_sub,
            "cpu_pct": bad_sub,
            "file_descriptors": bad_sub,
            "threads": bad_sub,
        },
    }
    for key in (
        "rss_peak",
        "rss_avg",
        "cpu_avg",
        "cpu_p95",
        "p99",
        "tps",
        "fd_slope",
        "thread_slope",
    ):
        assert ab.run_value(run, key) is None


def test_run_value_handles_non_dict_top_level_sections() -> None:
    run = {"summary": None, "report": "bad", "resources": [1]}
    assert ab.run_value(run, "rss_peak") is None
    assert ab.run_value(run, "qps") is None
    assert ab.run_value(run, "completeness") is None
    assert ab.run_value(run, "unknown_key") is None


def test_run_value_rejects_invalid_numbers_and_falls_back_for_completeness() -> None:
    for bad_admitted in (True, False, float("nan"), float("inf"), 10**1000, "bad", None):
        run = {
            "summary": {
                "effective_qps": True,
                "drop_rate": 1e308,
                "latency_ms": {"p99": float("nan")},
                "tokens_per_second": {"generated": 10**1000},
            },
            "report": {
                "admitted_completeness": bad_admitted,
                "completeness_ratio": 0.95,
                "token_accuracy": False,
            },
            "resources": {},
            "authoritative": {
                "rss_peak_mib": True,
                "rss_average_mib": float("nan"),
                "cpu_average_pct": 10**1000,
                "cpu_p95_pct": float("inf"),
            },
        }
        assert ab.run_value(run, "completeness") == pytest.approx(0.95)
        assert ab.run_value(run, "qps") is None
        assert ab.run_value(run, "drop_rate") is None
        assert ab.run_value(run, "p99") is None
        assert ab.run_value(run, "tps") is None
        assert ab.run_value(run, "token_accuracy") is None
        assert ab.run_value(run, "rss_peak") is None
        assert ab.run_value(run, "rss_avg") is None
        assert ab.run_value(run, "cpu_avg") is None
        assert ab.run_value(run, "cpu_p95") is None


def test_run_value_scales_valid_p99_and_drop_rate() -> None:
    run = {
        "summary": {
            "latency_ms": {"p99": 2500.0},
            "drop_rate": 0.025,
        },
        "report": {"admitted_completeness": 0.99, "completeness_ratio": 0.5},
        "resources": {},
    }
    assert ab.run_value(run, "p99") == pytest.approx(2.5)
    assert ab.run_value(run, "drop_rate") == pytest.approx(2.5)
    assert ab.run_value(run, "completeness") == pytest.approx(0.99)


@pytest.mark.parametrize("bad_rss", [None, "bad", 123, [1], True])
def test_run_rss_slope_handles_non_dict_rss_and_invalid_values(bad_rss: Any) -> None:
    run = {"metrics": {}, "resources": {"rss_mb": bad_rss}}
    assert ab.run_rss_slope(run, 0.0, 10.0) is None


@pytest.mark.parametrize(
    "bad_slope",
    [True, float("nan"), float("inf"), pytest.param(10**1000, id="huge-int")],
)
def test_run_rss_slope_rejects_invalid_fallback_slope(bad_slope: Any) -> None:
    run = {"metrics": {}, "resources": {"rss_mb": {"slope": bad_slope}}}
    assert ab.run_rss_slope(run, 0.0, 10.0) is None


def test_run_rss_slope_handles_non_finite_warmup_slope_and_valid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ab, "warmup_slope", lambda *_args: (float("inf"), [], []))
    run = {
        "metrics": {"times": [0.0, 1.0], "columns": {"rss_mb": [10.0, 20.0]}},
        "resources": {"rss_mb": {"slope": 1.5}},
    }
    assert ab.run_rss_slope(run, 0.0, 1.0) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "bad",
    [None, True, False, float("nan"), float("inf"), pytest.param(10**1000, id="huge-int")],
)
def test_format_metric_rejects_invalid_values(bad: Any) -> None:
    assert ab.format_metric("rss_peak", bad) == "—"
    assert ab.format_metric("completeness", bad) == "—"


def test_format_metric_handles_percentage_overflow_and_valid_values() -> None:
    assert ab.format_metric("completeness", 1e308) == "—"
    assert ab.format_metric("completeness", 0.985) == "98.5%"
    assert ab.format_metric("rss_peak", 128.0) == "128.0 MiB"
    assert ab.format_metric("rss_slope", 1.2) == "+1.2 MiB/h"
    assert ab.format_metric("cpu_avg", 12.3) == "12.3 %"
    assert ab.format_metric("drop_rate", 0.25) == "0.25%"
    assert ab.format_metric("qps", 42.5) == "42.5"
    assert ab.format_metric("tps", 1234.0) == "1,234 tok/s"
    assert ab.format_metric("p99", 0.75) == "0.75 s"
    assert ab.format_metric("fd_slope", -0.5) == "-0.5/h"
    assert ab.format_metric("other", 3.5) == "3.5"


@pytest.mark.parametrize(
    ("base", "opt"),
    [
        (float("nan"), 100.0),
        (100.0, float("nan")),
        (float("inf"), 100.0),
        (100.0, float("-inf")),
        (True, 100.0),
        (100.0, False),
        pytest.param(10**1000, 100.0, id="huge-base"),
        pytest.param(100.0, 10**1000, id="huge-opt"),
    ],
)
def test_delta_and_verdict_rejects_invalid_numbers(base: Any, opt: Any) -> None:
    assert ab.delta_and_verdict("rss_peak", base, opt) == ("—", "N/A")
    assert ab.delta_and_verdict("qps", base, opt) == ("—", "N/A")


@pytest.mark.parametrize(
    ("key", "base", "opt", "expected"),
    [
        ("rss_peak", 1e-308, 1e10, ("—", "轻微退化")),
        ("rss_peak", 1e-308, -1e10, ("—", "改善")),
        ("qps", 1e-308, 1e10, ("—", "改善")),
        ("qps", -1e308, 1e308, ("—", "改善")),
        ("qps", 1e308, -1e308, ("—", "轻微退化")),
        ("rss_peak", 100.0, 101.0, ("+1.0%", "持平")),
        ("rss_peak", 100.0, 80.0, ("-20.0%", "改善")),
        ("rss_peak", 100.0, 120.0, ("+20.0%", "轻微退化")),
        ("qps", 100.0, 120.0, ("+20.0%", "改善")),
        ("qps", 100.0, 80.0, ("-20.0%", "轻微退化")),
    ],
)
def test_delta_and_verdict_handles_overflow_and_normal_changes(
    key: str, base: float, opt: float, expected: tuple[str, str]
) -> None:
    assert ab.delta_and_verdict(key, base, opt) == expected


def test_align_series_drops_non_finite_and_bool_entries() -> None:
    grid, base, opt = ab.align_series(
        [0.0, True, float("nan"), 2.0, 10**1000],
        [10.0, 999.0, 50.0, 30.0, 40.0],
        [0.0, 1.0, float("inf"), 2.0],
        [100.0, float("nan"), 200.0, 300.0],
    )
    assert grid == [0.0, 1.0, 2.0]
    assert base == [10.0, 20.0, 30.0]
    assert opt == [100.0, 200.0, 300.0]
    assert all(v is None or math.isfinite(v) for v in (*base, *opt))


def test_render_survives_malformed_artifacts_without_nan_or_inf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for label, p99_val, admitted in (
        ("before", -1e308, float("nan")),
        ("after", 1e308, True),
    ):
        directory = tmp_path / label
        directory.mkdir()
        summary_text = (
            '{"resources": {"rss_mb": "corrupt", "cpu_pct": [1, 2], '
            '"file_descriptors": true, "threads": 42}, '
            '"effective_qps": true, "drop_rate": NaN, '
            f'"latency_ms": {{"p99": {p99_val!r}}}, '
            '"tokens_per_second": {"generated": ' + ("1" + "0" * 1000) + "}}"
        )
        (directory / "run-summary.json").write_text(summary_text, encoding="utf-8")
        report_text = (
            f'{{"admitted_completeness": {"NaN" if isinstance(admitted, float) else "true"}, '
            '"completeness_ratio": 0.98, "token_accuracy": Infinity}'
        )
        (directory / "report.json").write_text(report_text, encoding="utf-8")
        (directory / "worker-result.json").write_text("[]", encoding="utf-8")

    resource_summary = tmp_path / "resource-summary.json"
    resource_summary.write_text(
        json.dumps(
            {
                "valid": True,
                "comparison_order": ["before", "after"],
                "runs": {
                    "before": {
                        "rss_peak_mib": 1e-308,
                        "rss_average_mib": True,
                        "cpu_average_pct": float("nan"),
                        "cpu_p95_pct": float("inf"),
                    },
                    "after": {
                        "rss_peak_mib": 1e10,
                        "rss_average_mib": 50.0,
                        "cpu_average_pct": 10.0,
                        "cpu_p95_pct": 12.0,
                    },
                },
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    output = tmp_path / "ab-comparison.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ab_comparison.py",
            "--baseline-dir",
            str(tmp_path / "before"),
            "--optimized-dir",
            str(tmp_path / "after"),
            "--resource-summary",
            str(resource_summary),
            "--output",
            str(output),
        ],
    )
    ab.render(ab.parse_args())
    rendered = output.read_text(encoding="utf-8")

    assert 'data-validity="valid"' in rendered
    assert "98.0%" in rendered
    assert "+nan%" not in rendered.lower()
    assert "nan%" not in rendered.lower()
    assert "inf%" not in rendered.lower()
    assert "+inf" not in rendered.lower()
    assert "-inf" not in rendered.lower()
    assert "nan " not in rendered.lower()
