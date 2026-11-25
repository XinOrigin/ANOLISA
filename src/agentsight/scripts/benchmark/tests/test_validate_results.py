from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "single_run"))

import validate_results


@pytest.mark.parametrize(
    ("observations", "complete", "correct", "captured_total"),
    [
        ([("pending", 37)], 0, 0, None),
        ([("error", 37)], 0, 0, None),
        ([("pending", 37), ("complete", 99)], 1, 0, 99),
        ([("complete", 99), ("pending", 37)], 1, 0, 99),
        ([("pending", 99), ("complete", 37)], 1, 1, 37),
        ([("pending", None), ("complete", 37)], 1, 1, 37),
        ([("pending", 37), ("complete", None)], 1, 0, None),
    ],
)
def test_token_accuracy_requires_completed_observations(
    observations: list[tuple[str, int | None]],
    complete: int,
    correct: int,
    captured_total: int | None,
) -> None:
    expected = {"bench-1"}
    report = validate_results.make_report(
        expected, expected, {"bench-1": observations}, 37
    )

    assert report["complete"] == complete
    assert report["captured_total_tokens"] == captured_total
    assert report["token_correct"] == correct
    assert report["token_accuracy"] == float(correct)


def test_streaming_pending_count_cannot_validate_incorrect_completion(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agentsight.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript("""
            CREATE TABLE genai_events (
                id INTEGER PRIMARY KEY,
                call_id TEXT,
                status TEXT,
                total_tokens INTEGER
            );
            INSERT INTO genai_events VALUES (1, 'bench-1', 'pending', 37);
            """)
    captured, genai_id, token_rowid, pending = (
        validate_results.load_captured_incremental(db_path, "bench-", None, 0, 0, set())
    )
    assert pending == {1}

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE genai_events SET status = 'complete', total_tokens = 99 WHERE id = 1"
        )
    incremental, _, _, pending = validate_results.load_captured_incremental(
        db_path, "bench-", None, genai_id, token_rowid, pending
    )
    validate_results.merge_captured(captured, incremental)
    assert pending == set()
    assert captured == {"bench-1": [("pending", 37), ("complete", 99)]}

    report = validate_results.make_report({"bench-1"}, {"bench-1"}, captured, 37)
    assert report["completeness_ratio"] == 1.0
    assert report["captured_total_tokens"] == 99
    assert report["token_correct"] == 0
    assert report["token_accuracy"] == 0.0


def test_capture_status_marks_capacity_rejection_and_incomplete_reasons() -> None:
    rejected = json.dumps(
        {"metadata": {"incomplete": "true", "abandon_reason": "capacity_rejected"}}
    )
    incomplete = json.dumps({"metadata": {"abandon_reason": "over_budget"}})

    assert (
        validate_results.capture_status("complete", rejected)
        == "rejected:capacity_rejected"
    )
    assert validate_results.capture_status("complete", incomplete) == "incomplete:over_budget"
    assert validate_results.capture_status("complete", None) == "complete"


def test_nested_user_content_does_not_mark_capture_incomplete() -> None:
    event = json.dumps(
        {
            "metadata": {"request_id": "req-1"},
            "request": {
                "messages": [
                    {
                        "content": {
                            "metadata": {
                                "abandon_reason": "capacity_rejected",
                                "incomplete": True,
                            }
                        }
                    }
                ]
            },
        }
    )
    assert validate_results.capture_status("complete", event) == "complete"


def test_make_report_separates_rejections_conflicts_and_unexplained_loss() -> None:
    captured = {
        "bench-1": [("complete", 20)],
        "bench-2": [("rejected:capacity_rejected", None)],
        "bench-3": [("complete", 20), ("incomplete:over_budget", None)],
    }
    expected = {"bench-1", "bench-2", "bench-3", "bench-4"}

    report = validate_results.make_report(expected, expected - {"bench-4"}, captured, 20)

    assert report["complete"] == 2
    assert report["rejected"] == 1
    assert report["conflicts"] == 1
    assert report["admitted_sent"] == 3
    assert report["admitted_completeness"] == pytest.approx(2 / 3)
    assert report["unexplained_loss"] == 1
    assert report["completeness_ratio"] == 0.5


def test_capacity_rejected_metrics_sum_states_and_ignore_other_reasons(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.prom"
    baseline.write_text(
        "\n".join(
            [
                'agentsight_http_conn_removed_total{reason="capacity_rejected",state="admission"} 1',
                'agentsight_http_conn_removed_total{state="request_body_pending",reason="capacity_rejected"} 2',
                'agentsight_http_conn_removed_total{reason="other",state="admission"} 99',
            ]
        )
        + "\n"
    )
    metrics = tmp_path / "internal-metrics.prom"
    metrics.write_text(
        "\n".join(
            [
                'agentsight_http_conn_removed_total{reason="capacity_rejected",state="admission"} 2',
                'agentsight_http_conn_removed_total{state="request_body_pending",reason="capacity_rejected"} 4',
                'agentsight_http_conn_removed_total{reason="other",state="admission"} 99',
            ]
        )
        + "\n"
    )
    assert validate_results.load_capacity_rejected_events(metrics, baseline) == 3


def test_capacity_rejected_metrics_require_a_baseline(tmp_path: Path) -> None:
    metrics = tmp_path / "internal-metrics.prom"
    metrics.write_text(
        'agentsight_http_conn_removed_total{reason="capacity_rejected",state="admission"} 6\n'
    )
    assert validate_results.load_capacity_rejected_events(metrics) is None


def test_capacity_rejected_metrics_reject_counter_reset(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.prom"
    baseline.write_text(
        'agentsight_http_conn_removed_total{reason="capacity_rejected",state="admission"} 7\n'
    )
    metrics = tmp_path / "internal-metrics.prom"
    metrics.write_text(
        'agentsight_http_conn_removed_total{reason="capacity_rejected",state="admission"} 6\n'
    )
    assert validate_results.load_capacity_rejected_events(metrics, baseline) is None


@pytest.mark.parametrize(
    "contents",
    [
        "",
        'agentsight_http_conn_removed_total{reason="capacity_rejected"} nope\n',
    ],
)
def test_capacity_rejected_metrics_invalid_or_missing_falls_back(
    tmp_path: Path, contents: str
) -> None:
    metrics = tmp_path / "internal-metrics.prom"
    baseline = tmp_path / "baseline.prom"
    baseline.write_text(
        'agentsight_http_conn_removed_total{reason="capacity_rejected"} 0\n'
    )
    if contents:
        metrics.write_text(contents)
    assert validate_results.load_capacity_rejected_events(metrics, baseline) is None


@pytest.mark.parametrize(
    ("count", "rejected_ids", "expected_rejected", "expected_loss", "expected_completeness"),
    [
        (None, 2, 2, 0, 1.0),
        (6, 2, 2, 0, 1.0),
        (20, 2, 2, 0, 1.0),
        (None, 0, 0, 2, 0.75),
        (1, 0, 1, 1, pytest.approx(6 / 7)),
        (2, 3, 2, 0, 1.0),
    ],
)
def test_apply_capacity_rejected_metrics_clamps_aggregate(
    count: int | None,
    rejected_ids: int,
    expected_rejected: int,
    expected_loss: int,
    expected_completeness: float,
) -> None:
    report = {"sent": 8, "complete": 6, "rejected_request_count": rejected_ids}
    validate_results.apply_capacity_rejected_metrics(report, count)
    assert report["rejected"] == expected_rejected
    assert report["unexplained_loss"] == expected_loss
    assert report["admitted_completeness"] == expected_completeness


def test_validation_rejects_conflicting_capture_evidence() -> None:
    report = validate_results.make_report(
        {"bench-1"},
        {"bench-1"},
        {"bench-1": [("complete", 20), ("incomplete:over_budget", None)]},
        20,
    )
    assert report["conflicts"] == 1
    assert report["completeness_ratio"] == 1.0
    assert not validate_results._validation_passed(report, 0.999)


def test_apply_capacity_rejected_metrics_defaults_missing_request_id_count() -> None:
    report = {"sent": 8, "complete": 6, "rejected": 0}

    validate_results.apply_capacity_rejected_metrics(report, 2)

    assert report["rejected_request_count"] == 0
    assert report["rejected"] == 2
    assert report["admitted_sent"] == 6
    assert report["admitted_completeness"] == 1.0
    assert report["unexplained_loss"] == 0


def test_resolve_metrics_path_prefers_cli_then_env_then_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_results = tmp_path / "agents.jsonl.gz"
    load_results.write_text("")
    explicit = tmp_path / "explicit.prom"
    explicit.write_text("")
    from_env = tmp_path / "env.prom"
    from_env.write_text("")
    sibling = tmp_path / "internal-metrics.prom"
    sibling.write_text("")
    monkeypatch.setenv("AGENTSIGHT_METRICS_FILE", str(from_env))

    assert validate_results.resolve_metrics_path(load_results, explicit) == explicit
    assert validate_results.resolve_metrics_path(load_results, None) == from_env

    monkeypatch.delenv("AGENTSIGHT_METRICS_FILE")
    assert validate_results.resolve_metrics_path(load_results, None) == sibling

    sibling.unlink()
    assert validate_results.resolve_metrics_path(load_results, None) is None


def test_resolve_metrics_baseline_prefers_cli_then_sibling(tmp_path: Path) -> None:
    load_results = tmp_path / "agents.jsonl.gz"
    load_results.write_text("")
    explicit = tmp_path / "explicit-baseline.prom"
    explicit.write_text("")
    sibling = tmp_path / "internal-metrics-baseline.prom"
    sibling.write_text("")

    assert validate_results.resolve_metrics_baseline(load_results, explicit) == explicit
    assert validate_results.resolve_metrics_baseline(load_results, None) == sibling

    sibling.unlink()
    assert validate_results.resolve_metrics_baseline(load_results, None) is None


def test_find_request_ids_handles_deep_nesting_cycles_big_ints_and_scalars() -> None:
    deep_dict: dict[str, object] = {"leaf": "ok"}
    for _ in range(1500):
        deep_dict = {"child": deep_dict}
    deep_with_sibling = {"deep": deep_dict, "meta": {"request_id": "bench-sibling"}}
    assert validate_results.find_request_ids(deep_with_sibling) == {"bench-sibling"}

    cyclic: dict[str, object] = {"request_id": "bench-cyclic"}
    cyclic["self"] = cyclic
    assert validate_results.find_request_ids(cyclic) == {"bench-cyclic"}

    deep_json = "[" * 1500 + "]" * 1500
    big_int_json = '{"n": ' + ("9" * 4500) + "}"
    assert validate_results.find_request_ids(deep_json) == set()
    assert validate_results.find_request_ids(big_int_json) == set()
    assert validate_results.find_request_ids({"request": {"raw_body": big_int_json}}) == set()
    assert validate_results.find_request_ids([{"request_id": "bench-list"}]) == {"bench-list"}
    for non_container in (None, 123, True, "plain", '"scalar"'):
        assert validate_results.find_request_ids(non_container) == set()


@pytest.mark.parametrize("mode", ["full", "incremental"])
def test_load_captured_handles_pathological_and_non_dict_event_json(
    tmp_path: Path, mode: str
) -> None:
    db_path = tmp_path / "capture.db"
    deep_json = "[" * 1500 + "]" * 1500
    big_int_json = '{"n": ' + ("1" * 4500) + "}"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE genai_events (
                id INTEGER PRIMARY KEY,
                call_id TEXT,
                status TEXT,
                total_tokens INTEGER,
                event_json TEXT
            );
            """
        )
        rows = [
            (1, "bench-deep", "complete", 10, deep_json),
            (2, "bench-bigint", "complete", 11, big_int_json),
            (3, "bench-scalar-null", "complete", 12, "null"),
            (4, "bench-scalar-int", "complete", 13, "123"),
            (5, "bench-scalar-str", "complete", 14, '"hello"'),
            (6, None, "complete", 15, '[{"request_id": "bench-from-list"}]'),
        ]
        connection.executemany("INSERT INTO genai_events VALUES (?, ?, ?, ?, ?)", rows)

    if mode == "full":
        captured = validate_results.load_captured(db_path, "bench-")
    else:
        captured, _, _, _ = validate_results.load_captured_incremental(
            db_path, "bench-", None, 0, 0, set()
        )

    assert captured == {
        "bench-deep": [("complete", 10)],
        "bench-bigint": [("complete", 11)],
        "bench-scalar-null": [("complete", 12)],
        "bench-scalar-int": [("complete", 13)],
        "bench-scalar-str": [("complete", 14)],
        "bench-from-list": [("complete", 15)],
    }


def test_capture_status_and_json_lines_handle_pathological_json(tmp_path: Path) -> None:
    deep_json = "[" * 1500 + "]" * 1500
    big_int_json = '{"n": ' + ("1" * 4500) + "}"
    assert validate_results.capture_status("complete", deep_json) == "complete"
    assert validate_results.capture_status("complete", big_int_json) == "complete"
    assert validate_results.capture_status("complete", "null") == "complete"

    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                deep_json,
                big_int_json,
                '{"request_id": "bench-valid", "http_status": 200}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(validate_results.json_lines(jsonl)) == [
        {"request_id": "bench-valid", "http_status": 200}
    ]


def test_load_ffi_captured_handles_bad_utf8_and_non_dict_payloads(tmp_path: Path) -> None:
    path = tmp_path / "ffi-events.log"
    deep_json = "[" * 1500 + "]" * 1500
    big_int_json = '{"n": ' + ("1" * 4500) + "}"
    lines = [
        b"\n",
        b'J\t{"request":{"text":"bad\xff\xfe"},"metadata":{"request_id":"bench-utf8","status_code":200}}\n',
        b'J\t{"broken":\xff}\n',
        b"J\tnull\n",
        b"J\t[1, 2, 3]\n",
        b"J\t123\n",
        f"J\t{deep_json}\n".encode("utf-8"),
        f"J\t{big_int_json}\n".encode("utf-8"),
        b'J\t{"request":{"messages":[{"content":"[agentsight-benchmark:bench-bad-meta]"}]},"metadata":["bad"],"token_usage":["bad"]}\n',
        b'J\t{"metadata":{"request_id":"bench-abandoned","status_code":null,"abandon_reason":"over_budget"}}\n',
        b"T\tmalformed-only-two-cols\t200\n",
        b"T\tbench-bad-typed\tnot-int\tnot-int\t1\n",
        b"T\tbench-neg-tokens\t200\t-5\t1\n",
        b'J\t{"metadata":{"request_id":"bench-ok","status_code":"200"},"token_usage":{"total_tokens":42}}\n',
    ]
    path.write_bytes(b"".join(lines))

    captured, offset = validate_results.load_ffi_captured(path)
    assert offset == path.stat().st_size
    assert captured == {
        "bench-utf8": [("complete", None)],
        "bench-bad-meta": [("error", None)],
        "bench-abandoned": [("incomplete:over_budget", None)],
        "bench-bad-typed": [("error", None)],
        "bench-neg-tokens": [("complete", None)],
        "bench-ok": [("complete", 42)],
    }


@pytest.mark.parametrize(
    "status_code",
    ["__missing__", None, True, False, "not-a-number", "", "9" * 4500, ["200"]],
)
def test_load_ffi_captured_handles_invalid_status_code_without_crashing(
    tmp_path: Path, status_code: object
) -> None:
    path = tmp_path / "ffi-events.log"
    metadata: dict[str, object] = {"request_id": "bench-status"}
    if status_code != "__missing__":
        metadata["status_code"] = status_code
    path.write_text("J\t" + json.dumps({"metadata": metadata}) + "\n", encoding="utf-8")

    captured, _ = validate_results.load_ffi_captured(path)
    assert captured == {"bench-status": [("error", None)]}


def test_load_ffi_captured_rejects_unknown_receipt_kind(tmp_path: Path) -> None:
    path = tmp_path / "ffi-events.log"
    path.write_text("X\tpayload\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown FFI callback receipt type"):
        validate_results.load_ffi_captured(path)
