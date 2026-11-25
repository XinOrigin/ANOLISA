"""Validate the frozen configuration used by formal benchmark campaigns."""

from __future__ import annotations

import math
from typing import Any

REQUIRED_THRESHOLDS = {
    "min_throughput_ratio",
    "min_http_success_rate",
    "min_trace_completeness",
    "min_token_accuracy",
    "max_p99_ms",
    "max_drop_rate",
    "max_rss_mb",
    "max_rss_slope_mb_per_hour",
    "max_fd_slope_per_hour",
    "max_thread_slope_per_hour",
    "max_socket_slope_per_hour",
    "max_recovery_seconds",
}

DEFAULT_SAFETY = {
    "max_results_gb": 30,
    "min_free_disk_gb": 5,
    "min_available_memory_mb": 2048,
    "max_agentsight_rss_mb": 1536,
    "max_k6_vus": 256,
}


def finite_number(value: Any) -> bool:
    """Whether value is a real number whose float conversion stays finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_positive_int(value: Any) -> bool:
    """Whether value is a strictly positive integer (excluding booleans)."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def safety_settings(campaign: dict[str, Any]) -> dict[str, int]:
    """Return explicit safety limits, retaining compatibility with old campaigns."""
    configured = campaign.get("safety", {})
    if not isinstance(configured, dict):
        raise TypeError("safety must be an object")
    return {
        name: configured.get(name, default) for name, default in DEFAULT_SAFETY.items()
    }


def validate_campaign(campaign: dict[str, Any]) -> None:
    """Reject incomplete, unsafe, or internally inconsistent campaign inputs."""
    schema_version = campaign.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("campaign schema_version must be 1")
    comparison_mode = campaign.get("comparison_mode", "ab_comparison")
    if not isinstance(comparison_mode, str) or comparison_mode not in {
        "ab_comparison",
        "aa_calibration",
    }:
        raise ValueError("comparison_mode must be ab_comparison or aa_calibration")
    versions = campaign.get("versions")
    if not isinstance(versions, dict) or set(versions) != {"baseline", "optimized"}:
        raise ValueError("versions must contain exactly baseline and optimized")
    for name, version in versions.items():
        if not isinstance(version, dict):
            raise TypeError(f"versions.{name} must be an object")
        for field in ("commit", "binary", "config", "db"):
            if not isinstance(version.get(field), str) or not version[field].strip():
                raise ValueError(
                    f"versions.{name}.{field} must be a non-empty path/value"
                )
        for field in ("metrics_file", "log_file"):
            if field in version and (
                not isinstance(version[field], str) or not version[field].strip()
            ):
                raise ValueError(
                    f"versions.{name}.{field} must be a non-empty path when set"
                )
    thresholds = campaign.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise TypeError("thresholds must be an object")
    missing = REQUIRED_THRESHOLDS - set(thresholds)
    if missing:
        raise ValueError(f"missing frozen thresholds: {', '.join(sorted(missing))}")
    for name in REQUIRED_THRESHOLDS:
        value = thresholds[name]
        if not finite_number(value) or value < 0:
            raise ValueError(f"thresholds.{name} must be a finite non-negative number")
    for name in (
        "min_throughput_ratio",
        "min_http_success_rate",
        "min_trace_completeness",
        "min_token_accuracy",
        "max_drop_rate",
    ):
        if thresholds[name] > 1:
            raise ValueError(f"thresholds.{name} must be a ratio between 0 and 1")
    capacity = campaign.get("capacity", {})
    if not isinstance(capacity, dict):
        raise TypeError("capacity must be an object")
    for field in ("qps_start", "qps_resolution", "qps_safety_max"):
        if not _is_positive_int(capacity.get(field)):
            raise ValueError(f"capacity.{field} must be a positive integer")
    if capacity["qps_safety_max"] < capacity["qps_start"]:
        raise ValueError("capacity qps_safety_max must be at least qps_start")
    if capacity["qps_safety_max"] < capacity["qps_resolution"]:
        raise ValueError("capacity qps_safety_max must be at least qps_resolution")
    search_start_ratio = capacity.get("search_start_ratio")
    if not finite_number(search_start_ratio) or search_start_ratio != 0.8:
        raise ValueError("capacity.search_start_ratio must be 0.8")
    matrix = campaign.get("matrix", {})
    if not isinstance(matrix, dict):
        raise TypeError("matrix must be an object")
    matrix_qps = matrix.get("qps", [])
    if not isinstance(matrix_qps, list):
        raise ValueError("matrix.qps must be a list")
    if matrix_qps and any(not _is_positive_int(value) for value in matrix_qps):
        raise ValueError("matrix.qps values must be positive integers")
    if matrix_qps and (len(matrix_qps) != 5 or len(set(matrix_qps)) != 5):
        raise ValueError("matrix.qps must be empty or contain five unique values")
    positive_fields = {
        "smoke": ("qps", "duration_seconds"),
        "capacity": (
            "pretest_duration_seconds",
            "probe_duration_seconds",
            "confirm_duration_seconds",
            "confirm_repetitions",
        ),
        "matrix": ("duration_seconds", "repetitions"),
        "soak": ("duration_seconds",),
        "recovery": (
            "stable_seconds",
            "overload_seconds",
            "recover_seconds",
            "repetitions",
            "recovery_window_seconds",
        ),
        "fault": ("duration_seconds", "repetitions_per_case"),
    }
    for section, fields in positive_fields.items():
        settings = campaign.get(section)
        if not isinstance(settings, dict):
            raise TypeError(f"{section} must be an object")
        for field in fields:
            if not _is_positive_int(settings.get(field)):
                raise ValueError(f"{section}.{field} must be a positive integer")
    for section, field in (
        ("capacity", "pretest_warmup_seconds"),
        ("capacity", "probe_warmup_seconds"),
        ("capacity", "confirm_warmup_seconds"),
        ("matrix", "warmup_seconds"),
        ("soak", "warmup_seconds"),
        ("fault", "warmup_seconds"),
    ):
        value = campaign[section].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{section}.{field} must be a non-negative integer")
    load = campaign.get("load")
    protocol = load.get("protocol") if isinstance(load, dict) else None
    if not isinstance(protocol, str) or protocol not in {"sse", "json"}:
        raise ValueError("load.protocol must be sse or json for a formal campaign")
    for field in ("payload_kb", "chunks", "chunk_bytes"):
        if not _is_positive_int(load.get(field)):
            raise ValueError(f"load.{field} must be a positive integer")
    safety = safety_settings(campaign)
    for field, value in safety.items():
        if not _is_positive_int(value):
            raise ValueError(f"safety.{field} must be a positive integer")
    if safety["max_agentsight_rss_mb"] < thresholds["max_rss_mb"]:
        raise ValueError(
            "safety.max_agentsight_rss_mb must be at least thresholds.max_rss_mb"
        )
    recovery = campaign["recovery"]
    tolerance = recovery.get("tolerance_ratio")
    if not finite_number(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("recovery.tolerance_ratio must be in [0, 1)")
    if recovery["recovery_window_seconds"] > recovery["recover_seconds"]:
        raise ValueError("recovery window cannot exceed the recovery phase")
