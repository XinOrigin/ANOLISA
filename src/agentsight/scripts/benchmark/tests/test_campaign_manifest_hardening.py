from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

BENCHMARK_DIR = Path(__file__).parents[1]
CAMPAIGN_DIR = BENCHMARK_DIR / "campaign"
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import campaign_config
import campaign_manifest


def _valid_campaign(tmp_path: Path) -> dict[str, Any]:
    binary = tmp_path / "agentsight"
    config = tmp_path / "agentsight.json"
    binary.write_bytes(b"binary")
    config.write_text(
        json.dumps(
            {
                "cmdline": {
                    "allow": [
                        {
                            "rule": ["*python*", "*mock_llm_server.py*"],
                            "agent_name": "BenchmarkHarness",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "campaign_id": "hardening-test",
        "versions": {
            name: {
                "commit": name,
                "binary": str(binary),
                "config": str(config),
                "pid": os.getpid(),
                "db": str(tmp_path / f"{name}.db"),
            }
            for name in ("baseline", "optimized")
        },
        "thresholds": {
            "min_throughput_ratio": 0.9,
            "min_http_success_rate": 0.99,
            "min_trace_completeness": 0.999,
            "min_token_accuracy": 1.0,
            "max_p99_ms": 100,
            "max_drop_rate": 0.01,
            "max_rss_mb": 100,
            "max_rss_slope_mb_per_hour": 10,
            "max_fd_slope_per_hour": 1,
            "max_thread_slope_per_hour": 1,
            "max_socket_slope_per_hour": 1,
            "max_recovery_seconds": 10,
        },
        "load": {"protocol": "sse", "payload_kb": 1, "chunks": 2, "chunk_bytes": 4},
        "smoke": {"qps": 10, "duration_seconds": 1},
        "capacity": {
            "qps_start": 100,
            "qps_resolution": 50,
            "qps_safety_max": 400,
            "pretest_warmup_seconds": 0,
            "pretest_duration_seconds": 1,
            "search_start_ratio": 0.8,
            "probe_warmup_seconds": 0,
            "probe_duration_seconds": 1,
            "confirm_warmup_seconds": 0,
            "confirm_duration_seconds": 1,
            "confirm_repetitions": 3,
        },
        "matrix": {
            "qps": [],
            "warmup_seconds": 0,
            "duration_seconds": 1,
            "repetitions": 3,
        },
        "soak": {"warmup_seconds": 0, "duration_seconds": 1},
        "recovery": {
            "stable_seconds": 1,
            "overload_seconds": 1,
            "recover_seconds": 1,
            "repetitions": 3,
            "tolerance_ratio": 0.1,
            "recovery_window_seconds": 1,
        },
        "fault": {
            "warmup_seconds": 0,
            "duration_seconds": 1,
            "repetitions_per_case": 1,
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"cmdline": None},
        {"cmdline": []},
        {"cmdline": "mock_llm_server.py"},
        {"cmdline": 123},
        {"cmdline": {"allow": None}},
        {"cmdline": {"allow": "mock_llm_server.py"}},
        {"cmdline": {"allow": {"mock_llm_server.py": 1}}},
        {"cmdline": {"allow": 123}},
        {"cmdline": {"allow": [None, "mock_llm_server.py", 42]}},
        {"cmdline": {"allow": [{"rule": None}]}},
        {"cmdline": {"allow": [{"rule": 123}]}},
        {"cmdline": {"allow": [{"rule": "*mock_llm_server.py*"}]}},
        {"cmdline": {"allow": [{"rule": {"mock_llm_server.py": 1}}]}},
        {"cmdline": {"allow": [{"rule": ["*python*", {"mock_llm_server.py": 1}]}]}},
        {"cmdline": {"allow": [{"rule": [123, "*mock_llm_server.py*"]}]}},
    ],
)
def test_observes_mock_server_handles_malformed_shapes_without_crashing(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    config = tmp_path / "agentsight.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    assert campaign_manifest.observes_mock_server(config) is False


def test_observes_mock_server_skips_bad_entries_and_matches_valid_list_rule(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    assert campaign_manifest.observes_mock_server(missing) is False

    config = tmp_path / "agentsight.json"
    config.write_text(
        json.dumps(
            {
                "cmdline": {
                    "allow": [
                        None,
                        "ignored",
                        {"rule": None},
                        {"rule": "*mock_llm_server.py*"},
                        {"rule": [1, "*mock_llm_server.py*"]},
                        {"rule": ["*python*", "*mock_llm_server.py*"]},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert campaign_manifest.observes_mock_server(config) is True


def test_frozen_inputs_raises_value_error_on_malformed_cmdline_config(
    tmp_path: Path,
) -> None:
    data = _valid_campaign(tmp_path)
    bad_config = tmp_path / "bad_agentsight.json"
    bad_config.write_text(json.dumps({"cmdline": None}), encoding="utf-8")
    data["versions"]["baseline"]["config"] = str(bad_config)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="must discover"):
        campaign_manifest.frozen_inputs(campaign_path, data)


@pytest.mark.parametrize(
    "field", ["qps_start", "qps_resolution", "qps_safety_max"]
)
@pytest.mark.parametrize(
    "bad_value", [None, "100", [], {}, True, False, 1.5, 0, -10]
)
def test_validate_campaign_capacity_qps_fields_raise_value_error(
    tmp_path: Path, field: str, bad_value: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data["capacity"][field] = bad_value
    with pytest.raises(ValueError, match=f"capacity\\.{field}"):
        campaign_config.validate_campaign(data)


def test_validate_campaign_capacity_bool_pair_raises_positive_integer_error(
    tmp_path: Path,
) -> None:
    data = _valid_campaign(tmp_path)
    data["capacity"]["qps_start"] = True
    data["capacity"]["qps_safety_max"] = False
    with pytest.raises(ValueError, match="capacity\\.qps_start"):
        campaign_config.validate_campaign(data)


def test_validate_campaign_capacity_ordering_checks_preserved(tmp_path: Path) -> None:
    low_max = _valid_campaign(tmp_path)
    low_max["capacity"]["qps_safety_max"] = 50
    with pytest.raises(ValueError, match="qps_safety_max must be at least qps_start"):
        campaign_config.validate_campaign(low_max)

    high_res = _valid_campaign(tmp_path)
    high_res["capacity"]["qps_resolution"] = 500
    with pytest.raises(
        ValueError, match="qps_safety_max must be at least qps_resolution"
    ):
        campaign_config.validate_campaign(high_res)


@pytest.mark.parametrize("bad_version", [True, False, 1.0, "1", None, 2])
def test_validate_campaign_rejects_non_int_schema_version(
    tmp_path: Path, bad_version: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data["schema_version"] = bad_version
    with pytest.raises(ValueError, match="schema_version"):
        campaign_config.validate_campaign(data)


@pytest.mark.parametrize("bad_mode", [[], {}, None, True, 1, "unknown"])
def test_validate_campaign_rejects_unhashable_or_invalid_comparison_mode(
    tmp_path: Path, bad_mode: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data["comparison_mode"] = bad_mode
    with pytest.raises(ValueError, match="comparison_mode"):
        campaign_config.validate_campaign(data)


@pytest.mark.parametrize("bad_protocol", [[], {}, None, True, 1, "grpc"])
def test_validate_campaign_rejects_unhashable_or_invalid_load_protocol(
    tmp_path: Path, bad_protocol: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data["load"]["protocol"] = bad_protocol
    with pytest.raises(ValueError, match="load\\.protocol"):
        campaign_config.validate_campaign(data)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("smoke", "qps", True),
        ("smoke", "duration_seconds", None),
        ("capacity", "pretest_duration_seconds", True),
        ("capacity", "pretest_warmup_seconds", False),
        ("capacity", "search_start_ratio", None),
        ("matrix", "duration_seconds", False),
        ("matrix", "warmup_seconds", True),
        ("soak", "duration_seconds", True),
        ("soak", "warmup_seconds", None),
        ("recovery", "stable_seconds", True),
        ("recovery", "tolerance_ratio", None),
        ("recovery", "tolerance_ratio", float("nan")),
        ("recovery", "tolerance_ratio", float("inf")),
        ("recovery", "tolerance_ratio", True),
        ("fault", "duration_seconds", True),
        ("fault", "warmup_seconds", False),
        ("load", "payload_kb", True),
        ("load", "chunks", None),
        ("thresholds", "max_rss_mb", True),
        ("thresholds", "max_rss_mb", None),
    ],
)
def test_validate_campaign_rejects_booleans_and_nulls_across_numeric_fields(
    tmp_path: Path, section: str, field: str, bad_value: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data[section][field] = bad_value
    with pytest.raises(ValueError):
        campaign_config.validate_campaign(data)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_results_gb", True),
        ("max_results_gb", None),
        ("max_agentsight_rss_mb", False),
        ("max_agentsight_rss_mb", None),
    ],
)
def test_validate_campaign_rejects_booleans_and_nulls_in_safety_fields(
    tmp_path: Path, field: str, bad_value: Any
) -> None:
    data = _valid_campaign(tmp_path)
    data["safety"] = {field: bad_value}
    with pytest.raises(ValueError, match=f"safety\\.{field}"):
        campaign_config.validate_campaign(data)


def test_validate_campaign_rejects_booleans_in_matrix_qps(tmp_path: Path) -> None:
    data = _valid_campaign(tmp_path)
    data["matrix"]["qps"] = [True, 2, 3, 4, 5]
    with pytest.raises(
        ValueError, match="matrix\\.qps values must be positive integers"
    ):
        campaign_config.validate_campaign(data)

    valid = deepcopy(_valid_campaign(tmp_path))
    campaign_config.validate_campaign(valid)
