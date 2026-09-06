"""Fail-closed gate contract for Workbook-77 metric versioning."""

from __future__ import annotations

import math

import pytest

from evaluation.review_gates import GateSpec, evaluate_gates


SPECS = (
    GateSpec(name="efficiency", metric="complete_track_efficiency", op=">=", bound=0.8),
    GateSpec(name="complete_fake", metric="complete_fake_rate", op="<=", bound=0.05),
    GateSpec(name="all_route_purity", metric="all_route_purity", op=">=", bound=0.9),
)


def test_all_finite_metrics_can_pass():
    report = evaluate_gates(
        {
            "metric_version": "route_accounting_v2",
            "complete_track_efficiency": 0.89,
            "complete_fake_rate": 0.02,
            "all_route_purity": 0.95,
        },
        SPECS,
        required_metric_version="route_accounting_v2",
    )
    assert report["gate_pass"] is True
    assert report["n_passed"] == 3


def test_missing_metric_never_passes():
    report = evaluate_gates(
        {"metric_version": "route_accounting_v2", "complete_track_efficiency": 0.99},
        SPECS,
        required_metric_version="route_accounting_v2",
    )
    assert report["gate_pass"] is False
    assert report["results"][1]["reason"] == "missing metric"


def test_none_and_nan_never_pass():
    report = evaluate_gates(
        {
            "metric_version": "route_accounting_v2",
            "complete_track_efficiency": 0.99,
            "complete_fake_rate": None,
            "all_route_purity": math.nan,
        },
        SPECS,
        required_metric_version="route_accounting_v2",
    )
    assert report["gate_pass"] is False
    assert all(not item["passed"] for item in report["results"][1:])


def test_empty_specification_is_a_failure():
    report = evaluate_gates({"metric_version": "route_accounting_v2"}, ())
    assert report["gate_pass"] is False
    assert "empty gate" in report["reason"]


def test_wrong_metric_version_is_a_failure():
    report = evaluate_gates(
        {
            "metric_version": "legacy",
            "complete_track_efficiency": 1.0,
            "complete_fake_rate": 0.0,
            "all_route_purity": 1.0,
        },
        SPECS,
        required_metric_version="route_accounting_v2",
    )
    assert report["gate_pass"] is False
    assert "metric_version" in report["reason"]


def test_non_finite_bound_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        GateSpec(name="bad", metric="x", op="<=", bound=math.inf)
