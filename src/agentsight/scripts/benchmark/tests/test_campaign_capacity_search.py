"""Regressions for capacity search grid alignment, soak QPS floor, and bool bounds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "campaign"))

import campaign
import campaign_evidence
from test_campaign_matrix_evidence import complete_formal_evidence
from test_campaign_runner import campaign_data


@pytest.mark.parametrize("resume", [False, True])
def test_capacity_unaligned_qps_start_converges_without_repeating_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    data = campaign_data(tmp_path)
    data["capacity"]["qps_start"] = 55
    data["capacity"]["qps_resolution"] = 4
    data["capacity"]["qps_safety_max"] = 1000
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    calls: list[tuple[str, int, int]] = []

    def fake_once(*args: object, **kwargs: object) -> dict[str, Any]:
        label = str(args[4])
        qps = int(args[5])
        repetition = int(args[8])
        key = (label, repetition)
        calls.append((label, qps, repetition))
        if len(calls) > 40:
            pytest.fail(f"capacity search did not terminate: {calls[-6:]}")
        if key in seen:
            if not kwargs.get("resume", False):
                raise FileExistsError(f"duplicate probe: {label} rep-{repetition}")
            return seen[key]
        result = {"evaluation": {"verdict": "PASS" if qps <= 59 else "FAIL"}}
        seen[key] = result
        return result

    monkeypatch.setattr(campaign, "run_once", fake_once)
    result = campaign.capacity(
        data, tmp_path / "results", "baseline", os.getpid(), resume=resume
    )

    assert result["pretest"]["last_passed_qps"] == 55
    assert result["maximum_sustainable_qps"] == 56
    assert result["first_failed_qps"] == 60
    assert result["boundary_confirmed"] is True
    assert all(int(level) % 4 == 0 for level in result["probes"])
    assert all(int(level) % 4 == 0 for level in result["confirmation"])
    assert len(calls) == len(set(calls))


def test_capacity_unaligned_qps_start_below_resolution_fails_at_minimum_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = campaign_data(tmp_path)
    data["capacity"]["qps_start"] = 2
    data["capacity"]["qps_resolution"] = 4
    data["capacity"]["qps_safety_max"] = 100
    formal_probes: list[int] = []

    def fake_once(*args: object, **kwargs: object) -> dict[str, Any]:
        label = str(args[4])
        qps = int(args[5])
        if label.startswith("search-"):
            formal_probes.append(qps)
        return {"evaluation": {"verdict": "PASS" if qps <= 2 else "FAIL"}}

    monkeypatch.setattr(campaign, "run_once", fake_once)
    with pytest.raises(RuntimeError, match="minimum capacity search level.*4 QPS"):
        campaign.capacity(data, tmp_path / "results", "baseline", os.getpid())
    assert formal_probes == [4]


def test_grid_midpoint_never_repeats_lower_bound() -> None:
    assert campaign.grid_midpoint(52, 64, 4) == 56
    assert campaign.grid_midpoint(56, 64, 4) == 60
    assert campaign.grid_midpoint(56, 60, 4) is None
    assert campaign.grid_midpoint(59, 64, 4) is None


def test_resolved_qps_preserves_minimum_resolution_for_soak(tmp_path: Path) -> None:
    data = campaign_data(tmp_path)
    data["capacity"]["qps_resolution"] = 50
    data["matrix"]["qps"] = [10, 20, 30, 40, 50]
    for version in data["versions"]:
        (tmp_path / f"capacity-{version}.json").write_text(
            json.dumps({"maximum_sustainable_qps": 50, "first_failed_qps": 100}),
            encoding="utf-8",
        )

    matrix, soak, overload = campaign.resolved_qps(data, tmp_path)

    assert matrix == [10, 20, 30, 40, 50]
    assert soak == 50
    assert overload == 100
    saved = json.loads((tmp_path / "campaign-resolution.json").read_text(encoding="utf-8"))
    assert saved["soak_qps"] == 50


@pytest.mark.parametrize(
    ("maximum", "failed", "match"),
    [
        (True, 2, "confirmed capacity results"),
        (1, True, "adjacent failed QPS"),
    ],
)
def test_resolved_qps_rejects_boolean_capacity_bounds(
    tmp_path: Path, maximum: object, failed: object, match: str
) -> None:
    data = campaign_data(tmp_path)
    data["capacity"]["qps_resolution"] = 1
    data["matrix"]["qps"] = [1, 2, 3, 4, 5]
    for version in data["versions"]:
        (tmp_path / f"capacity-{version}.json").write_text(
            json.dumps({"maximum_sustainable_qps": maximum, "first_failed_qps": failed}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=match):
        campaign.resolved_qps(data, tmp_path)


def test_audit_campaign_preserves_minimum_resolution_for_soak() -> None:
    campaign_data_obj, items, capacities, recovery, faults, regression = (
        complete_formal_evidence(1)
    )
    campaign_data_obj["matrix"]["qps"] = [10, 20, 30, 40, 50]
    for version in campaign_evidence.VERSIONS:
        capacities[version].update(
            {
                "maximum_sustainable_qps": 50,
                "first_failed_qps": 100,
                "confirmation": {"50": ["PASS"], "100": ["FAIL"]},
            }
        )
    updated_items = []
    for path, run in items:
        updated = dict(run)
        if updated["scenario"] == "matrix":
            updated["qps"] //= 10
            updated["label"] = f"qps-{updated['qps']}"
        elif updated["scenario"] == "soak":
            updated["qps"] = 50
            updated["label"] = "qps-50"
        elif updated["scenario"] == "recovery":
            updated["qps"] = 100 if updated["label"] == "overload" else 50
        elif updated["scenario"] == "fault":
            updated["qps"] = 50
        updated_items.append((path, updated))

    assert (
        campaign_evidence.audit_campaign(
            campaign_data_obj, updated_items, capacities, recovery, faults, regression
        )
        == []
    )


def test_audit_campaign_rejects_boolean_capacity_bounds() -> None:
    campaign_data_obj, items, capacities, recovery, faults, regression = (
        complete_formal_evidence(1)
    )
    for version in campaign_evidence.VERSIONS:
        capacities[version].update(
            {
                "maximum_sustainable_qps": True,
                "first_failed_qps": 51,
                "confirmation": {"True": ["PASS"], "51": ["FAIL"]},
            }
        )

    issues = campaign_evidence.audit_campaign(
        campaign_data_obj, items, capacities, recovery, faults, regression
    )

    assert "baseline capacity lacks an adjacent failed QPS" in issues
    assert "optimized capacity lacks an adjacent failed QPS" in issues
    assert "five unique common matrix QPS levels are unavailable" in issues

    campaign_data_obj, items, capacities, recovery, faults, regression = (
        complete_formal_evidence(1)
    )
    campaign_data_obj["capacity"]["qps_resolution"] = 1
    for version in campaign_evidence.VERSIONS:
        capacities[version].update(
            {
                "maximum_sustainable_qps": 1,
                "first_failed_qps": True,
                "confirmation": {"1": ["PASS"], "True": ["FAIL"]},
            }
        )
    bool_fail_issues = campaign_evidence.audit_campaign(
        campaign_data_obj, items, capacities, recovery, faults, regression
    )
    assert "baseline capacity lacks an adjacent failed QPS" in bool_fail_issues
    assert "baseline recovery overload rep 1 has drifted inputs" in bool_fail_issues
