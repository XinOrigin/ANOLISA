"""Hardening regression tests for single_run/plot_metrics.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SINGLE_RUN = Path(__file__).parents[1] / "single_run"
if str(SINGLE_RUN) not in sys.path:
    sys.path.insert(0, str(SINGLE_RUN))


def test_number_rejects_non_numeric_and_non_finite_values() -> None:
    from plot_metrics import number

    row = {
        "ok": "12.5",
        "empty": "",
        "missing": None,
        "extra": ["1", "2"],
        "mapping": {"x": 1},
        "flag_true": True,
        "flag_false": False,
        "nan": "nan",
        "inf": "inf",
        "ninf": "-inf",
        "text": "abc",
    }
    assert number(row, "ok") == 12.5
    for key in (
        "empty",
        "missing",
        "absent",
        "extra",
        "mapping",
        "flag_true",
        "flag_false",
        "nan",
        "inf",
        "ninf",
        "text",
    ):
        assert number(row, key) is None


@pytest.mark.parametrize(
    "csv_text",
    [
        "monotonic_timestamp,rss_mb\n0,100,extra1,extra2\n1,101\n",
        "monotonic_timestamp,rss_mb\n0,100\n1,101,extra1,extra2\n",
    ],
)
def test_read_metrics_ignores_extra_csv_columns(tmp_path: Path, csv_text: str) -> None:
    from plot_metrics import read_metrics

    metrics = tmp_path / "metrics.csv"
    metrics.write_text(csv_text, encoding="utf-8")

    rows, times, columns = read_metrics(metrics)

    assert len(rows) == 2
    assert times == [0.0, 1.0]
    assert None not in columns
    assert columns["rss_mb"] == [100.0, 101.0]


def test_render_succeeds_with_extra_csv_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plot_metrics

    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "monotonic_timestamp,rss_mb\n0,100,extra_col\n1,102\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics-report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_metrics.py", str(metrics), "--output", str(output), "--warmup-seconds", "0"],
    )
    plot_metrics.render(plot_metrics.parse_args())
    assert "<html" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("runtime_limits", "expected_snippets", "forbidden_snippets"),
    [
        (
            {
                "ring_buffer_mb": None,
                "event_channel_max_bytes_mb": "invalid",
                "connection_max_bytes_mb": [1, 2],
                "max_connection_body_mb": True,
                "max_sse_retained_mb": False,
                "connection_idle_timeout_secs": None,
                "connection_hard_timeout_secs": "30s",
                "cpu_budget_percent": None,
                "connection_capacity": True,
                "pid_cache_size": "auto",
            },
            [
                "<td><code>ring_buffer_mb</code></td><td>N/A</td>",
                "<td><code>event_channel_max_bytes_mb</code></td><td>invalid</td>",
                "<td><code>connection_max_bytes_mb</code></td><td>[1, 2]</td>",
                "<td><code>max_connection_body_mb</code></td><td>true</td>",
                "<td><code>max_sse_retained_mb</code></td><td>false</td>",
                "<td><code>connection_idle_timeout_secs</code></td><td>N/A</td>",
                "<td><code>connection_hard_timeout_secs</code></td><td>30s</td>",
                "<td><code>cpu_budget_percent</code></td><td>N/A</td>",
                "<td><code>connection_capacity</code></td><td>true</td>",
                "<td><code>pid_cache_size</code></td><td>auto</td>",
            ],
            ["1 MiB", "0 MiB", "None %", "True %"],
        ),
        (
            {
                "ring_buffer_mb": 1024,
                "max_connection_body_mb": 1.5,
                "connection_idle_timeout_secs": 300,
                "cpu_budget_percent": 25,
                "retain_raw_sse": False,
                "connection_capacity": 2048,
                "event_channel_policy": "backpressure",
                "ffi_channel_max_bytes_mb": float("nan"),
                "pending_genai_max_bytes_mb": "<script>x</script>",
            },
            [
                "<td><code>ring_buffer_mb</code></td><td>1,024 MiB</td>",
                "<td><code>max_connection_body_mb</code></td><td>1.5 MiB</td>",
                "<td><code>connection_idle_timeout_secs</code></td><td>300 s</td>",
                "<td><code>cpu_budget_percent</code></td><td>25 %</td>",
                "<td><code>retain_raw_sse</code></td><td>false</td>",
                "<td><code>connection_capacity</code></td><td>2,048</td>",
                "<td><code>event_channel_policy</code></td><td>backpressure</td>",
                "<td><code>ffi_channel_max_bytes_mb</code></td><td>nan</td>",
                "<td><code>pending_genai_max_bytes_mb</code></td><td>&lt;script&gt;x&lt;/script&gt;</td>",
            ],
            [],
        ),
    ],
)
def test_runtime_limits_section_guards_null_bool_and_non_numeric(
    tmp_path: Path,
    runtime_limits: dict,
    expected_snippets: list[str],
    forbidden_snippets: list[str],
) -> None:
    from plot_metrics import runtime_limits_section

    (tmp_path / "config.json").write_text(
        json.dumps({"runtime_limits": runtime_limits}),
        encoding="utf-8",
    )
    rendered = runtime_limits_section(tmp_path)
    for snippet in expected_snippets:
        assert snippet in rendered
    for snippet in forbidden_snippets:
        assert snippet not in rendered


@pytest.mark.parametrize(
    ("step_per_sec", "expected_ref", "expected_card"),
    [
        (-0.01, "warmup 后 -36 MiB/h", "warmup 后 -36.0 MiB/h"),
        (0.01, "warmup 后 +36 MiB/h", "warmup 后 +36.0 MiB/h"),
        (0.0, "warmup 后 +0 MiB/h", "warmup 后 +0.0 MiB/h"),
    ],
)
def test_rss_slope_signed_formatting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_per_sec: float,
    expected_ref: str,
    expected_card: str,
) -> None:
    import plot_metrics

    metrics = tmp_path / "metrics.csv"
    rows = ["monotonic_timestamp,rss_mb"]
    for t in range(10):
        rows.append(f"{t},{200.0 + step_per_sec * t:.4f}")
    metrics.write_text("\n".join(rows) + "\n", encoding="utf-8")

    output = tmp_path / "metrics-report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_metrics.py",
            str(metrics),
            "--output",
            str(output),
            "--warmup-seconds",
            "0",
            "--slope-window-seconds",
            "2",
        ],
    )
    plot_metrics.render(plot_metrics.parse_args())
    rendered = output.read_text(encoding="utf-8")

    assert "+-" not in rendered
    assert expected_ref in rendered
    assert expected_card in rendered
    assert f"斜率 {expected_card.removeprefix('warmup 后 ')}" in rendered


def test_rss_slope_none_renders_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plot_metrics

    metrics = tmp_path / "metrics.csv"
    metrics.write_text("monotonic_timestamp,rss_mb\n0,100\n1,101\n", encoding="utf-8")
    output = tmp_path / "metrics-report.html"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_metrics.py", str(metrics), "--output", str(output), "--warmup-seconds", "10"],
    )
    plot_metrics.render(plot_metrics.parse_args())
    rendered = output.read_text(encoding="utf-8")
    assert "斜率 —" in rendered
    assert "样本不足，无法计算滚动斜率" in rendered
