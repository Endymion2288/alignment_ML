"""Frozen real-data candidate-graph DQ gate and campaign decision.

Real data has no truth.  This module never inspects efficiency, purity,
fake rate, AP, or AUC.  The gate does not retune V2 or occupancy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.real_data_operating_protocol import (
    BLOCK_CANDIDATE_GRAPH_DQ_FAILED,
    BLOCK_CROSS_LEVEL_CONTAMINATED,
    BLOCK_DQ_FAILED,
    ROLE_CALIBRATION,
)

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-candidate-graph-dq-gate"
DEFAULT_GATE_RELATIVE = Path("configs") / "operating_protocol_v1_real_data_candidate_graph_dq_gate.yaml"

CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED = "candidate_graph_dq_failed"
CAMPAIGN_STATION_FIT_FAILED = "station_fit_failed"
CAMPAIGN_CROSS_LEVEL_CONTAMINATED = "cross_level_contaminated"
CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED = "cross_level_precondition_unverified"
CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE = "station_mode_dryrun_passed_but_not_writable"
CAMPAIGN_DECISIONS = (
    CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED,
    CAMPAIGN_STATION_FIT_FAILED,
    CAMPAIGN_CROSS_LEVEL_CONTAMINATED,
    CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED,
    CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE,
)

PERCENTILE_CHANNELS = {
    "residual_x_mm": ("residual_percentiles", "x_mm"),
    "residual_y_mm": ("residual_percentiles", "y_mm"),
    "residual_tx": ("residual_percentiles", "tx"),
    "residual_ty": ("residual_percentiles", "ty"),
    "pull_x": ("pull_percentiles", "x"),
    "pull_y": ("pull_percentiles", "y"),
    "chi2": ("chi2_percentiles", None),
    "dz": ("dz_percentiles", None),
}


def load_candidate_graph_dq_gate(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_GATE_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate-graph DQ gate must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected candidate-graph DQ gate schema: {source}")
    if payload.get("frozen") is not True:
        raise ValueError("candidate-graph DQ gate must be frozen")
    if payload.get("do_not_retune_v2") is not True or payload.get("do_not_change_thresholds") is not True:
        raise ValueError("candidate-graph DQ gate must forbid V2/threshold retuning")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("candidate-graph DQ gate lacks requirements")
    return {
        "path": str(source),
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "requirements": dict(requirements),
        "truth_metrics_forbidden": list(payload.get("truth_metrics_forbidden") or ()),
    }


def _share(block: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    cursor: Any = block
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return default
        cursor = cursor[key]
    if cursor is None:
        return default
    return float(cursor)


def _percentile_interval(block: Mapping[str, Any], channel: str) -> tuple[float, float] | None:
    spec = PERCENTILE_CHANNELS.get(channel)
    if spec is None:
        return None
    group, inner = spec
    payload = block.get(group)
    if not isinstance(payload, Mapping):
        return None
    if inner is not None:
        payload = payload.get(inner)
        if not isinstance(payload, Mapping):
            return None
    low = payload.get("p16")
    high = payload.get("p84")
    if low is None or high is None:
        return None
    return float(low), float(high)


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) >= max(left[0], right[0])


def evaluate_block_candidate_graph_dq(
    block: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one run against the frozen candidate-graph DQ gate."""
    requirements = gate["requirements"]
    reasons: list[str] = []
    if requirements.get("physical_all_pairs_nonempty") and not block.get(
        "physical_all_pairs_candidate_graph_nonempty"
    ):
        reasons.append("empty_physical_all_pairs_graph")
    if requirements.get("selected_edges_nonempty") and int(block.get("selected_edges") or 0) <= 0:
        reasons.append("empty_selected_edges")
    if requirements.get("selected_routes_nonempty") and int(block.get("selected_routes") or 0) <= 0:
        reasons.append("empty_selected_routes")
    if requirements.get("values_finite") and block.get("values_finite") is not True:
        reasons.append("non_finite_values")
    physical_share = _share(block, "physical_event_concentration", "max_event_share")
    physical_top2 = _share(block, "physical_event_concentration", "top2_event_share")
    max_physical = float(requirements.get("max_event_share_of_physical_candidates", 1.0))
    max_top2 = float(requirements.get("max_top2_event_share_of_physical_candidates", 1.0))
    if physical_share is not None and physical_share > max_physical:
        reasons.append("physical_single_event_dominated")
    if physical_top2 is not None and physical_top2 > max_top2:
        reasons.append("physical_high_multiplicity_dominated")
    selected_share = _share(block, "per_event_route_distribution", "max_event_share_of_selected_routes")
    max_selected = float(requirements.get("max_event_share_of_selected_routes", 1.0))
    if int(block.get("selected_routes") or 0) > 0 and selected_share is not None and selected_share > max_selected:
        reasons.append("selected_single_event_dominated")
    return {
        "source_id": block.get("source_id"),
        "run": block.get("run"),
        "role": block.get("role"),
        "passed": not reasons,
        "reasons": reasons,
        "do_not_retune": True,
    }


def evaluate_calibration_compatibility(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Require overlapping central intervals on the frozen feature channels."""
    reasons: list[str] = []
    overlap: dict[str, bool | None] = {}
    channels = list(gate["requirements"].get("calibration_percentile_overlap") or ())
    for channel in channels:
        left_interval = _percentile_interval(left, str(channel))
        right_interval = _percentile_interval(right, str(channel))
        if left_interval is None or right_interval is None:
            overlap[str(channel)] = None
            reasons.append(f"missing_percentile:{channel}")
            continue
        ok = _intervals_overlap(left_interval, right_interval)
        overlap[str(channel)] = ok
        if not ok:
            reasons.append(f"percentile_iqr_disjoint:{channel}")
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "percentile_iqr_overlap": overlap,
        "do_not_retune_from_this_check": True,
    }


def evaluate_candidate_graph_dq_gate(
    blocks: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibration-only Station Mode admission from the frozen DQ gate."""
    per_block = [evaluate_block_candidate_graph_dq(block, gate) for block in blocks]
    calibration = [block for block in blocks if block.get("role") == ROLE_CALIBRATION]
    if len(calibration) != 2:
        raise ValueError("candidate-graph DQ gate expects exactly two calibration blocks")
    compatibility = evaluate_calibration_compatibility(calibration[0], calibration[1], gate)
    calibration_ids = {str(block.get("source_id")) for block in calibration}
    calibration_block_pass = all(
        item["passed"] for item in per_block if str(item.get("source_id")) in calibration_ids
    )
    passed = bool(calibration_block_pass and compatibility["compatible"])
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": str(gate.get("path")),
        "passed": passed,
        "per_block": per_block,
        "calibration_compatibility": compatibility,
        "do_not_lower_gate": True,
        "do_not_retune_v2": True,
        "note": (
            "Station Mode may start only if both calibration selected graphs are "
            "non-empty, finite, not single-event dominated, and mutually compatible."
        ),
    }


def assign_campaign_decision(
    *,
    candidate_graph_dq_failed: bool,
    station_fit_failed: bool,
    implied_cdx_exceeds_operating_band: bool,
    independent_cdx_evidence: bool,
    station_and_holdout_dq_ok: bool,
) -> str:
    """Return exactly one Operating Protocol V1 real-data campaign decision."""
    if candidate_graph_dq_failed:
        return CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED
    if station_fit_failed:
        return CAMPAIGN_STATION_FIT_FAILED
    if implied_cdx_exceeds_operating_band:
        return CAMPAIGN_CROSS_LEVEL_CONTAMINATED
    if not independent_cdx_evidence:
        return CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED
    if station_and_holdout_dq_ok:
        return CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE
    return CAMPAIGN_STATION_FIT_FAILED


def campaign_allows_cdx_mode(decision: str) -> bool:
    """Dedicated C_dx Mode starts only after a fully clean Station Mode contract."""
    return decision == CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE
