"""Verify CPU units, interval weighting, phase boundaries and A/B validity."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "single_run"))

import collect_metrics
import resource_report


def row(
    timestamp: float, interval: float | str, cpu: float | str, rss: float | str
) -> dict[str, str]:
    return {
        "monotonic_timestamp": str(timestamp),
        "cpu_interval_seconds": str(interval),
        "cpu_pct": str(cpu),
        "rss_mb": str(rss),
        "process_alive": "1",
    }


def test_cpu_uses_actual_elapsed_time_and_allows_multiple_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collect_metrics, "read_status", lambda _: {"VmRSS": 256000})
    monkeypatch.setattr(collect_metrics, "read_cpu_ticks", lambda _: 400)
    monkeypatch.setattr(collect_metrics.time, "monotonic", lambda: 12.0)
    monkeypatch.setattr(collect_metrics.os, "sysconf", lambda _: 100)
    measured, ticks, timestamp = collect_metrics.sample(123, 100, 10.0, 0, None)
    assert float(measured["cpu_pct"]) == 150
    assert measured["rss_mb"] == "250.000"
    assert measured["cpu_interval_seconds"] == 2
    assert measured["monotonic_timestamp"] == timestamp == 12
    assert ticks == 400


def test_load_and_drain_statistics_use_separate_windows_and_weight_cpu() -> None:
    rows = [row(9, 1, 999, 999), row(10, "", "", 200), row(11, 1, 30, 240)]
    rows += [row(14, 3, 100, 260), row(15, 1, 0, 900)]
    load = resource_report.resource_stats(rows, 10, 14)
    assert load["rss_average_mib"] == 220
    assert load["rss_peak_mib"] == 240
    assert load["cpu_average_pct"] == pytest.approx(82.5)
    assert load["cpu_seconds"] == pytest.approx(3.3)
    assert load["cpu_measurement_seconds"] == 4
    assert load["cpu_p95_pct"] == load["cpu_peak_pct"] == 100
    tail = resource_report.resource_stats(rows, 14, float("inf"))
    assert tail["rss_peak_mib"] == 900
    assert tail["cpu_average_pct"] == tail["cpu_peak_pct"] == 0
    assert tail["cpu_measurement_seconds"] == 1


def test_boundary_intervals_are_split_by_overlap() -> None:
    rows = [row(13, 2, 150, 250)]
    load = resource_report.resource_stats(rows, 10, 12)
    tail = resource_report.resource_stats(rows, 12, float("inf"))
    assert load["cpu_seconds"] == tail["cpu_seconds"] == 1.5
    assert load["cpu_average_pct"] == tail["cpu_average_pct"] == 150


def test_invalid_cpu_values_are_rejected_and_unrelated_intervals_are_ignored() -> None:
    valid = row(11, 1, 25, 250)
    for invalid in [
        row(float("nan"), 1, 25, 250),
        row(12, 1, "nan", 250),
        row(12, 1, -1, 250),
    ]:
        stats = resource_report.resource_stats([valid, invalid], 10, 13)
        assert not stats["cpu_samples_valid"]
        assert stats["cpu_average_pct"] == 25
    stats = resource_report.resource_stats([row(9, 1, 999, 999), valid], 10, 12)
    assert stats["cpu_samples_valid"]
    assert stats["cpu_average_pct"] == stats["cpu_peak_pct"] == 25


@pytest.mark.parametrize("interval", ["", "nan", "0", "-1"])
def test_missing_or_invalid_interval_is_not_a_second_baseline(interval: str) -> None:
    rows = [row(10, "", "", 250), row(11, interval, "", 250), row(12, 1, 30, 250)]
    assert not resource_report.resource_stats(rows, 10, 12)["cpu_samples_valid"]


def artifacts(output: Path, label: str, cpu: float, rss: float) -> Path:
    directory = output / label
    (directory / "agents").mkdir(parents=True)
    records = {
        "worker-result.json": {
            "status": "complete",
            "returncode": 0,
            "trace_survived": True,
            "host_memory_mib": 1000,
            "logical_cpus": 8,
        },
        "agents/start.json": {"start": 10, "end": 12},
        "agent-summary.json": {"passed": True},
        "report.json": {
            "sent": 1,
            "completeness_ratio": 1,
            "token_accuracy": 1,
            "conflicts": 0,
        },
    }
    for name, record in records.items():
        (directory / name).write_text(json.dumps(record))
    rows = [row(10, "", "", rss), row(11, 1, cpu, rss), row(12, 1, cpu, rss)]
    with (directory / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return directory


def test_ab_reports_memory_reduction_and_cpu_regression(tmp_path: Path) -> None:
    artifacts(tmp_path, "before", 30, 400)
    artifacts(tmp_path, "after", 100, 250)
    assert resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["comparison"]["rss_average_mib"] == {"mib": 150, "percent": 37.5}
    change = summary["comparison"]["cpu_average_pct"]
    assert change["percentage_points"] == -70
    assert change["percent"] == pytest.approx(-233.33333333333334)
    assert "100% = one logical CPU" in summary["cpu_unit"]
    report = (tmp_path / "resource-report.txt").read_text()
    assert "-70.00 pp" in report and "Average CPU" in report and "Peak CPU" in report


@pytest.mark.parametrize("cpu", ["", "nan", "inf", "-1", "broken"])
def test_invalid_cpu_suppresses_all_ab_improvement(tmp_path: Path, cpu: str) -> None:
    artifacts(tmp_path, "before", 30, 400)
    directory = artifacts(tmp_path, "after", 100, 250)
    path = directory / "metrics.csv"
    path.write_text(path.read_text().replace(",100,", f",{cpu},"))
    assert not resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert not summary["valid"] and not summary["comparison"]
    assert summary["runs"]["after"]["resource_error"]


def test_missing_window_and_partial_cpu_coverage_are_invalid(tmp_path: Path) -> None:
    directory = artifacts(tmp_path, "run", 0, 250)
    window = directory / "agents/start.json"
    window.write_text(json.dumps({"start": 10, "end": 20}))
    measured = resource_report.summarize(directory, {})
    assert not measured["valid"] and "95%" in measured["resource_error"]
    window.unlink()
    measured = resource_report.summarize(directory, {})
    assert not measured["valid"] and "window" in measured["resource_error"]


def test_zero_cpu_baseline_has_no_relative_percentage(tmp_path: Path) -> None:
    artifacts(tmp_path, "before", 0, 250)
    artifacts(tmp_path, "after", 150, 250)
    assert resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["comparison"]["cpu_peak_pct"] == {
        "percentage_points": -150,
        "percent": None,
    }


def test_collector_row_without_vmrss_cannot_certify_a_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts(tmp_path, "before", 10, 400)
    directory = artifacts(tmp_path, "after", 10, 250)
    # Procfs can expose numeric status fields without VmRSS (for example
    # during process exit). sample() emits a live row with an empty RSS field.
    monkeypatch.setattr(collect_metrics, "read_status", lambda _: {"Threads": 1})
    monkeypatch.setattr(collect_metrics, "read_cpu_ticks", lambda _: 110)
    monkeypatch.setattr(collect_metrics.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(collect_metrics.os, "sysconf", lambda _: 100)
    measured, _, _ = collect_metrics.sample(123, 100, 10.0, 0, None)
    assert measured["process_alive"] == 1
    assert measured["rss_mb"] == ""
    assert measured["cpu_pct"] == "10.000"
    with (directory / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collect_metrics.FIELDS)
        writer.writeheader()
        writer.writerows([row(10, "", "", 250), measured, row(12, 1, 10, 250)])
    assert not resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["runs"]["after"]["cpu_samples_valid"]
    assert not summary["comparison"]


@pytest.mark.parametrize("rss", ["", "nan", "inf", "-1", "broken"])
def test_invalid_load_rss_suppresses_improvement(tmp_path: Path, rss: str) -> None:
    artifacts(tmp_path, "before", 10, 400)
    directory = artifacts(tmp_path, "after", 10, 250)
    path = directory / "metrics.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Keep one usable RSS point and complete CPU coverage. Dropping only the
    # bad point used to certify this run and advertise a 37.5% improvement.
    rows[1]["rss_mb"] = rss
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert not resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert not summary["valid"] and not summary["comparison"]
    after = summary["runs"]["after"]
    assert after["rss_average_mib"] == after["rss_peak_mib"] == 250
    assert after["cpu_samples_valid"]
    assert not after["rss_samples_valid"]
    assert "RSS" in after["resource_error"]


def test_rss_validity_follows_the_half_open_measurement_window() -> None:
    rows = [row(9, "", "", ""), row(10, 1, 10, 0), row(11, 1, 10, 250)]
    rows += [row(12, 1, 10, ""), row(13, 1, 10, 300)]
    load = resource_report.resource_stats(rows, 10, 12)
    assert load["rss_samples_valid"]
    assert load["rss_average_mib"] == 125
    assert not resource_report.resource_stats(rows, 12, 14)["rss_samples_valid"]
    assert not resource_report.resource_stats([], 10, 12)["rss_samples_valid"]


def write_counters(directory: Path, counters: list[dict[str, str]]) -> None:
    path = directory / "metrics.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for measured, values in zip(rows, counters, strict=True):
        measured.update(values)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "counter", ["eviction_count", "ring_buffer_dropped", "channel_dropped"]
)
@pytest.mark.parametrize("value", ["broken", "NaN", "inf", "-1", "1e999"])
def test_invalid_counter_preserves_reports_and_suppresses_comparison(
    tmp_path: Path, counter: str, value: str
) -> None:
    artifacts(tmp_path, "before", 30, 400)
    directory = artifacts(tmp_path, "after", 20, 250)
    write_counters(directory, [{counter: "0"}, {counter: value}, {counter: "0"}])
    assert not resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert not summary["valid"] and not summary["comparison"]
    run = summary["runs"]["after"]
    assert not run["valid"]
    assert counter in run["resource_error"]
    if counter != "eviction_count":
        assert run["capture"] == "FAIL"
    text = (tmp_path / "resource-report.txt").read_text()
    assert counter in text and "Result: INVALID" in text


@pytest.mark.parametrize("counter", ["ring_buffer_dropped", "channel_dropped"])
def test_valid_drop_evidence_survives_an_invalid_sample(
    tmp_path: Path, counter: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    write_counters(directory, [{counter: "0"}, {counter: "broken"}, {counter: "3"}])
    run = resource_report.summarize(directory, {})
    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drops"][counter] == 3
    assert counter in run["resource_error"]


@pytest.mark.parametrize("value", ["", "0"])
def test_optional_blank_and_zero_counters_preserve_validity(
    tmp_path: Path, value: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    counters = {
        key: value
        for key in ("eviction_count", "ring_buffer_dropped", "channel_dropped")
    }
    (directory / "metrics-baseline.json").write_text(
        json.dumps({"ring_buffer_dropped": "", "channel_dropped": ""})
    )
    write_counters(directory, [counters, counters, counters])
    run = resource_report.summarize(directory, {})
    assert run["valid"] and run["capture"] == "PASS"
    assert run["resource_error"] is None
    expected = 0 if value else None
    assert run["connection_evictions"] == expected
    assert all(observed == expected for observed in run["probe_drops"].values())


@pytest.mark.parametrize(
    "counter", ["eviction_count", "ring_buffer_dropped", "channel_dropped"]
)
def test_valid_counter_peaks_retain_existing_capture_rules(
    tmp_path: Path, counter: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    write_counters(directory, [{counter: "0"}, {counter: "3"}, {counter: "2"}])
    run = resource_report.summarize(directory, {})
    assert run["resource_error"] is None
    if counter == "eviction_count":
        assert run["connection_evictions"] == 3
        assert run["valid"] and run["capture"] == "PASS"
    else:
        assert run["probe_drops"][counter] == 3
        assert not run["valid"] and run["capture"] == "FAIL"


@pytest.mark.parametrize("counter", ["ring_buffer_dropped", "channel_dropped"])
def test_preload_drops_are_attributed_without_failing_load_or_drain(
    tmp_path: Path, counter: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    (directory / "agents/start.json").write_text(json.dumps({"start": 11, "end": 12}))
    write_counters(directory, [{counter: "5"}, {counter: "5"}, {counter: "5"}])

    run = resource_report.summarize(directory, {})

    assert run["valid"] and run["capture"] == "PASS"
    assert run["probe_drops"][counter] == 5
    assert run["probe_drop_phases"][counter] == {
        "before_load": 5,
        "load": 0,
        "tail": 0,
        "unattributed": 0,
    }
    assert resource_report.report(tmp_path, {"run": {}})
    assert "confined to pre-load" in (tmp_path / "resource-report.txt").read_text()


@pytest.mark.parametrize("counter", ["ring_buffer_dropped", "channel_dropped"])
@pytest.mark.parametrize(
    ("values", "expected_phase"),
    [(["0", "1", "1"], "load"), (["0", "0", "1"], "tail")],
)
def test_load_and_tail_drops_remain_fatal(
    tmp_path: Path, counter: str, values: list[str], expected_phase: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    write_counters(directory, [{counter: value} for value in values])

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"][counter][expected_phase] == 1


def test_counter_reset_is_unattributed_and_fatal(tmp_path: Path) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    write_counters(
        directory,
        [
            {"ring_buffer_dropped": "0"},
            {"ring_buffer_dropped": "3"},
            {"ring_buffer_dropped": "2"},
        ],
    )

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["load"] == 3
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["unattributed"] == 2


@pytest.mark.parametrize("counter", ["ring_buffer_dropped", "channel_dropped"])
def test_collector_baseline_separates_preflight_from_first_load_sample(
    tmp_path: Path, counter: str
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    (directory / "metrics-baseline.json").write_text(
        json.dumps({counter: 5, "monotonic_timestamp": 9, "counter_timestamp": 9})
    )
    write_counters(directory, [{counter: "5"}, {counter: "5"}, {counter: "5"}])

    run = resource_report.summarize(directory, {})

    assert run["valid"] and run["capture"] == "PASS"
    assert run["probe_drop_phases"][counter] == {
        "before_load": 5,
        "load": 0,
        "tail": 0,
        "unattributed": 0,
    }


def test_zero_counter_reset_is_unattributed_and_fatal(tmp_path: Path) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    (directory / "agents/start.json").write_text(json.dumps({"start": 11, "end": 12}))
    write_counters(
        directory,
        [
            {"ring_buffer_dropped": "5"},
            {"ring_buffer_dropped": "0"},
            {"ring_buffer_dropped": "0"},
        ],
    )

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["before_load"] == 5
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["unattributed"] == 1


def test_final_counter_reset_to_zero_is_unattributed_and_fatal(tmp_path: Path) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    (directory / "agents/start.json").write_text(json.dumps({"start": 11, "end": 12}))
    write_counters(
        directory,
        [
            {"ring_buffer_dropped": "5"},
            {"ring_buffer_dropped": "5"},
            {"ring_buffer_dropped": "5"},
        ],
    )
    (directory / "internal-metrics.prom").write_text(
        "agentsight_ring_buffer_dropped_total 0\nagentsight_channel_dropped_total 0\n"
    )

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["before_load"] == 5
    assert run["probe_drop_phases"]["ring_buffer_dropped"]["unattributed"] == 1


@pytest.mark.parametrize("timestamp", [None, 10])
def test_missing_or_late_baseline_timestamp_is_unattributed_and_fatal(
    tmp_path: Path, timestamp: float | None
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    baseline = {"ring_buffer_dropped": 1}
    if timestamp is not None:
        baseline["counter_timestamp"] = timestamp
    (directory / "metrics-baseline.json").write_text(json.dumps(baseline))
    write_counters(
        directory,
        [
            {"ring_buffer_dropped": "1"},
            {"ring_buffer_dropped": "1"},
            {"ring_buffer_dropped": "1"},
        ],
    )

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"]["ring_buffer_dropped"] == {
        "before_load": 0,
        "load": 0,
        "tail": 0,
        "unattributed": 1,
    }


def test_baseline_counter_read_after_load_start_is_unattributed_and_fatal(
    tmp_path: Path,
) -> None:
    directory = artifacts(tmp_path, "run", 30, 250)
    (directory / "metrics-baseline.json").write_text(
        json.dumps(
            {
                "monotonic_timestamp": 9.477,
                "counter_timestamp": 10.026,
                "ring_buffer_dropped": 1,
            }
        )
    )
    write_counters(
        directory,
        [
            {"ring_buffer_dropped": "1"},
            {"ring_buffer_dropped": "1"},
            {"ring_buffer_dropped": "1"},
        ],
    )

    run = resource_report.summarize(directory, {})

    assert not run["valid"] and run["capture"] == "FAIL"
    assert run["probe_drop_phases"]["ring_buffer_dropped"] == {
        "before_load": 0,
        "load": 0,
        "tail": 0,
        "unattributed": 1,
    }


@pytest.mark.parametrize(
    ("before_mem", "after_mem"),
    [(None, None), (0, 0), (1000, None), (None, 1000)],
)
def test_ab_valid_runs_with_missing_or_zero_host_memory_emit_none_host_pct(
    tmp_path: Path, before_mem: int | None, after_mem: int | None
) -> None:
    for label, cpu, rss, mem in (
        ("before", 30, 400, before_mem),
        ("after", 20, 250, after_mem),
    ):
        directory = artifacts(tmp_path, label, cpu, rss)
        worker_path = directory / "worker-result.json"
        worker = json.loads(worker_path.read_text())
        if mem is None:
            worker.pop("host_memory_mib", None)
        else:
            worker["host_memory_mib"] = mem
        worker_path.write_text(json.dumps(worker))

    assert resource_report.report(tmp_path, {"before": {}, "after": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["valid"]
    assert summary["comparison"]["rss_average_mib"] == {"mib": 150, "percent": 37.5}
    for key in ("rss_average_host_pct", "rss_peak_host_pct"):
        assert summary["comparison"][key] == {"percentage_points": None}
    assert "N/A pp" in (tmp_path / "resource-report.txt").read_text()


def _ffi_run_artifacts(tmp_path: Path, label: str = "run") -> Path:
    directory = artifacts(tmp_path, label, 25, 40)
    worker_path = directory / "worker-result.json"
    worker = json.loads(worker_path.read_text())
    worker["trace_returncode"] = 0
    worker_path.write_text(json.dumps(worker))
    (directory / "clients-ready.json").write_text('{"request_ids":["preflight-0"]}')
    (directory / "ffi-consumer.json").write_text('{"completed":true,"llm_events":2}')
    (directory / "internal-metrics.prom").write_text(
        "agentsight_ring_buffer_dropped_total 0\n"
        "agentsight_channel_dropped_total 0\n"
        'agentsight_ffi_dropped_total{reason="count"} 0\n'
    )
    return directory


@pytest.mark.parametrize(
    "relative",
    [
        "worker-result.json",
        "agent-summary.json",
        "report.json",
        "safety-stop.json",
        "metrics-baseline.json",
        "agents/start.json",
        "ffi-consumer.json",
    ],
)
@pytest.mark.parametrize("content", [b'{"incomplete":', b"\xff\xfe"])
def test_malformed_run_json_artifacts_fail_closed_without_crashing(
    tmp_path: Path, relative: str, content: bytes
) -> None:
    directory = _ffi_run_artifacts(tmp_path, "run")
    (directory / relative).write_bytes(content)

    assert not resource_report.report(tmp_path, {"run": {"mode": "ffi"}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    run = summary["runs"]["run"]
    assert not summary["valid"] and not run["valid"]
    if relative == "report.json":
        assert run["capture"] == "FAIL"
    elif relative == "agent-summary.json":
        assert run["load"] == "FAIL"
    elif relative == "safety-stop.json":
        assert run["safety_stop"] is not None
    elif relative == "metrics-baseline.json":
        assert run["resource_error"]
    elif relative == "ffi-consumer.json":
        assert run["ffi_error"]


@pytest.mark.parametrize("payload", [None, [], "not-a-dict", 1, True])
@pytest.mark.parametrize(
    "relative",
    [
        "worker-result.json",
        "agent-summary.json",
        "safety-stop.json",
        "metrics-baseline.json",
        "agents/start.json",
        "ffi-consumer.json",
    ],
)
def test_non_dict_auxiliary_artifacts_fail_closed_without_crashing(
    tmp_path: Path, relative: str, payload: object
) -> None:
    directory = _ffi_run_artifacts(tmp_path, "run")
    (directory / relative).write_text(json.dumps(payload))

    assert not resource_report.report(tmp_path, {"run": {"mode": "ffi"}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert not summary["valid"] and not summary["runs"]["run"]["valid"]


@pytest.mark.parametrize(
    "content",
    [b'{"incomplete":', b"\xff\xfe", b"null", b"[]", b'"bad"', b"1"],
)
def test_malformed_or_non_dict_settings_json_does_not_crash_report(
    tmp_path: Path, content: bytes
) -> None:
    artifacts(tmp_path, "run", 25, 40)
    (tmp_path / "settings.json").write_bytes(content)

    assert resource_report.report(tmp_path, {"run": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["valid"] and summary["settings"] == {}


@pytest.mark.parametrize(
    "content",
    [b'{"incomplete":', b"\xff\xfe", b"null", b"[]", b'{"request_ids":null}', b'{"request_ids":"abc"}'],
)
def test_malformed_or_non_dict_clients_ready_does_not_crash_ffi_report(
    tmp_path: Path, content: bytes
) -> None:
    directory = _ffi_run_artifacts(tmp_path, "run")
    (directory / "clients-ready.json").write_bytes(content)

    assert not resource_report.report(tmp_path, {"run": {"mode": "ffi"}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert not summary["valid"] and not summary["runs"]["run"]["valid"]
    assert summary["runs"]["run"]["ffi_error"] == "invalid clients-ready artifact"


def test_absent_clients_ready_defaults_preflight_count_to_zero_on_ffi_report(
    tmp_path: Path,
) -> None:
    directory = _ffi_run_artifacts(tmp_path, "run")
    (directory / "clients-ready.json").unlink()
    (directory / "ffi-consumer.json").write_text('{"completed":true,"llm_events":1}')

    assert resource_report.report(tmp_path, {"run": {"mode": "ffi"}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["valid"] and summary["runs"]["run"]["valid"]
    assert summary["runs"]["run"]["ffi_error"] is None


@pytest.mark.parametrize("llm_events", [None, True, False, "2", 1.5, [], {}])
def test_ffi_non_int_llm_events_fail_closed_without_crashing(
    tmp_path: Path, llm_events: object
) -> None:
    directory = _ffi_run_artifacts(tmp_path, "run")
    (directory / "clients-ready.json").unlink()
    (directory / "ffi-consumer.json").write_text(
        json.dumps({"completed": True, "llm_events": llm_events})
    )

    assert not resource_report.report(tmp_path, {"run": {"mode": "ffi"}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    run = summary["runs"]["run"]
    assert not summary["valid"] and not run["valid"]
    assert run["ffi_error"]


def test_partial_settings_and_non_string_worker_fields_do_not_crash_report(
    tmp_path: Path,
) -> None:
    directory = artifacts(tmp_path, "run", 25, 40)
    worker_path = directory / "worker-result.json"
    worker = json.loads(worker_path.read_text())
    worker.update({"status": None, "error": 123, "host_memory_mib": True})
    worker_path.write_text(json.dumps(worker))
    (tmp_path / "settings.json").write_text(
        json.dumps({"mode": "ffi", "agents": 4, "tokens_per_second": None})
    )

    assert not resource_report.report(tmp_path, {"run": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["runs"]["run"]["status"] == "not-run"
    assert summary["runs"]["run"]["host_memory_mib"] is None
    assert "123" in (tmp_path / "resource-report.txt").read_text()
    assert resource_report.safety_message(["unexpected"]) == "['unexpected']"


def test_working_tree_source_without_tree_and_binary_run_error_are_tolerated(
    tmp_path: Path,
) -> None:
    directory = artifacts(tmp_path, "run", 25, 40)
    worker_path = directory / "worker-result.json"
    worker = json.loads(worker_path.read_text())
    worker["error"] = "see worker.log"
    worker_path.write_text(json.dumps(worker))
    (directory / "worker.log").write_text("worker failed\n")
    (tmp_path / "run-error.txt").write_bytes(b"fatal: \xff\xfe\n")

    assert not resource_report.report(tmp_path, {"run": {"ref": "working-tree"}})
    text = (tmp_path / "resource-report.txt").read_text()
    assert "working-tree (tree unknown)" in text
    assert "worker failed" in text


def test_complete_settings_and_missing_ids_fallback_render_cleanly(
    tmp_path: Path,
) -> None:
    directory = artifacts(tmp_path, "run", 25, 40)
    capture_path = directory / "report.json"
    capture = json.loads(capture_path.read_text())
    capture.pop("missing_count", None)
    capture["missing_ids"] = ["req-1", "req-2"]
    capture_path.write_text(json.dumps(capture))
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "agents": 4,
                "tokens_per_second": 20,
                "duration": 30,
                "tokens_per_response": 128,
                "payload_kb": 4,
            }
        )
    )

    assert resource_report.report(tmp_path, {"run": {}})
    summary = json.loads((tmp_path / "resource-summary.json").read_text())
    assert summary["runs"]["run"]["capture_counts"]["missing_count"] == 2
    assert (
        "Load target: 4 agents x 20 output tok/s, 30s; 128 tokens/response, 4 KiB request padding"
        in (tmp_path / "resource-report.txt").read_text()
    )
