"""Full-segment Station Mode self-nulling dry-run helpers.

Route DQ, residual DQ, and frozen-A leakage are observables.  Residual
reduction is never alignment success.  This module never writes official
conditions, never starts C_dx Mode, and never floats extra DoF.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.calibration_modes import (
    A_STABILITY_MAX_REL_DEVIATION,
    load_mode_validity_contract,
)
from alignment.real_data_candidate_graph_dq import (
    CAMPAIGN_CROSS_LEVEL_CONTAMINATED,
    CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED,
    CAMPAIGN_STATION_FIT_FAILED,
    CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE,
    evaluate_candidate_graph_dq_gate,
    load_candidate_graph_dq_gate,
)
from alignment.real_data_operating_protocol import (
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    ROLE_HOLDOUT,
    assign_block_status,
    cross_level_contamination_diagnostic,
)

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-station-mode-self-nulling-fullscale"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1.yaml"

DECISION_ROUTE_DQ_INSUFFICIENT = "fullscale_route_dq_insufficient"
DECISION_SELF_NULLING_PENDING = "station_mode_self_nulling_pending"
DECISION_SELF_NULLING_FD_SUBMITTED = "station_mode_self_nulling_fd_submitted"
DECISION_STATION_FIT_FAILED = CAMPAIGN_STATION_FIT_FAILED
DECISION_CROSS_LEVEL_CONTAMINATED = CAMPAIGN_CROSS_LEVEL_CONTAMINATED
DECISION_CROSS_LEVEL_PRECONDITION_UNVERIFIED = CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED
DECISION_DRYRUN_PASSED_NOT_WRITABLE = CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE
DECISIONS = (
    DECISION_ROUTE_DQ_INSUFFICIENT,
    DECISION_SELF_NULLING_PENDING,
    DECISION_SELF_NULLING_FD_SUBMITTED,
    DECISION_STATION_FIT_FAILED,
    DECISION_CROSS_LEVEL_CONTAMINATED,
    DECISION_CROSS_LEVEL_PRECONDITION_UNVERIFIED,
    DECISION_DRYRUN_PASSED_NOT_WRITABLE,
)


def load_fullscale_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"fullscale config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected fullscale config schema: {source}")
    if payload.get("frozen") is not True:
        raise ValueError("fullscale config must be frozen")
    if payload.get("geometry_write_allowed") is not False:
        raise ValueError("fullscale config must keep geometry_write_allowed false until all gates pass")
    if payload.get("official_conditions_db_write") is not False:
        raise ValueError("fullscale config must forbid official conditions writes")
    if payload.get("do_not_enter_cdx_mode") is not True or payload.get("cdx_mode_blocked") is not True:
        raise ValueError("fullscale config must block C_dx Mode")
    if payload.get("joint_station_cdx_newton") is not False:
        raise ValueError("fullscale config must forbid joint Newton")
    if payload.get("new_layer_or_module_dof") is not False:
        raise ValueError("fullscale config must forbid new DoF")
    if payload.get("residual_reduction_is_not_alignment_success") is not True:
        raise ValueError("fullscale config must treat residual reduction as DQ only")
    if payload.get("do_not_change_thresholds") is not True or payload.get("do_not_retrain_v2") is not True:
        raise ValueError("fullscale config must freeze V2 and thresholds")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    names = ("p05", "p16", "p50", "p84", "p95")
    if values.size == 0:
        return {name: None for name in names}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in names}
    return {name: float(np.percentile(finite, float(name[1:]))) for name in names}


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) >= max(left[0], right[0])


def summarize_selected_routes(routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Truth-free selected-route audit.  No MC completeness labels."""
    n_selected = int(len(routes))
    n_complete = int(sum(1 for row in routes if bool(row.get("is_complete_four_station_route"))))
    lengths = [int(len(row.get("endpoint_stations") or row.get("endpoint_indices") or ())) for row in routes]
    length_hist = {str(length): int(count) for length, count in sorted(Counter(lengths).items())}
    per_event: dict[tuple[int, int], int] = defaultdict(int)
    endpoint_uses: Counter[tuple[int, int, int, int]] = Counter()
    utilities: list[float] = []
    for row in routes:
        per_event[(int(row["run_id"]), int(row["event_id"]))] += 1
        if row.get("utility") is not None:
            utilities.append(float(row["utility"]))
        for item in row.get("endpoint_provenance") or []:
            endpoint_uses[
                (
                    int(item["origin_run_id"]),
                    int(item["origin_event_id"]),
                    int(item["station_id"]),
                    int(item["origin_tracklet_id"]),
                )
            ] += 1
    selected_total = float(sum(per_event.values()))
    max_share = max(per_event.values()) / selected_total if selected_total else 0.0
    reused = int(sum(1 for count in endpoint_uses.values() if count > 1))
    return {
        "selected_routes": n_selected,
        "complete_four_station_routes": n_complete,
        "truth_free_complete_route_fraction": (
            None if n_selected <= 0 else float(n_complete) / float(n_selected)
        ),
        "route_length_histogram": length_hist,
        "selected_utility_distribution": _percentiles(np.asarray(utilities, dtype=np.float64)),
        "edge_reuse": {
            "unique_endpoints": int(len(endpoint_uses)),
            "reused_endpoints": reused,
            "max_uses": 0 if not endpoint_uses else int(max(endpoint_uses.values())),
        },
        "event_concentration": {
            "n_events_with_selected_routes": int(len(per_event)),
            "max_event_share_of_selected_routes": float(max_share),
        },
        "route_multiplicity": {
            "mean": None if not per_event else float(np.mean(list(per_event.values()))),
            "max": None if not per_event else int(max(per_event.values())),
        },
    }


def summarize_residual_observables(
    field_edges: Sequence[Mapping[str, Any]],
    leave_one_out: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Selected-route residual DQ.  Never an alignment-success score."""
    def column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
        values: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
        return np.asarray(values, dtype=np.float64)

    field = {
        "n_edges": int(len(field_edges)),
        "residual_x_mm": _percentiles(column(field_edges, "residual_x_mm")),
        "residual_y_mm": _percentiles(column(field_edges, "residual_y_mm")),
        "residual_tx": _percentiles(column(field_edges, "residual_tx")),
        "residual_ty": _percentiles(column(field_edges, "residual_ty")),
        "pull_x": _percentiles(column(field_edges, "pull_x_mm")),
        "pull_y": _percentiles(column(field_edges, "pull_y_mm")),
        "chi2": _percentiles(column(field_edges, "chi2")),
        "values_finite": bool(
            all(
                math.isfinite(float(row[key]))
                for row in field_edges
                for key in ("residual_x_mm", "residual_y_mm", "chi2")
                if row.get(key) not in (None, "")
            )
        ),
    }
    closure = {
        "n_leave_one_out": int(len(leave_one_out)),
        "residual_x_mm": _percentiles(column(leave_one_out, "residual_x_mm")),
        "residual_y_mm": _percentiles(column(leave_one_out, "residual_y_mm")),
        "residual_tx": _percentiles(column(leave_one_out, "residual_tx")),
        "residual_ty": _percentiles(column(leave_one_out, "residual_ty")),
        "chi2": _percentiles(column(leave_one_out, "chi2")),
    }
    return {
        "selected_field_edge": field,
        "unbiased_leave_one_out": closure,
        "residual_reduction_is_not_alignment_success": True,
        "dq_observable_only": True,
    }


def run_to_run_stability(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Calibration-only selected-route stability.  Overlap is DQ, not correctness."""
    reasons: list[str] = []
    overlap: dict[str, bool | None] = {}
    channels = (
        ("residual_x_mm", ("residual_observables", "selected_field_edge", "residual_x_mm")),
        ("pull_x", ("residual_observables", "selected_field_edge", "pull_x")),
        ("chi2", ("residual_observables", "selected_field_edge", "chi2")),
    )
    for name, path in channels:
        cursor_left: Any = left
        cursor_right: Any = right
        for key in path:
            cursor_left = None if not isinstance(cursor_left, Mapping) else cursor_left.get(key)
            cursor_right = None if not isinstance(cursor_right, Mapping) else cursor_right.get(key)
        if not isinstance(cursor_left, Mapping) or not isinstance(cursor_right, Mapping):
            overlap[name] = None
            reasons.append(f"missing_percentile:{name}")
            continue
        if cursor_left.get("p16") is None or cursor_right.get("p16") is None:
            overlap[name] = None
            reasons.append(f"missing_percentile:{name}")
            continue
        ok = _intervals_overlap(
            (float(cursor_left["p16"]), float(cursor_left["p84"])),
            (float(cursor_right["p16"]), float(cursor_right["p84"])),
        )
        overlap[name] = ok
        if not ok:
            reasons.append(f"percentile_iqr_disjoint:{name}")
    left_complete = left.get("truth_free_complete_route_fraction")
    right_complete = right.get("truth_free_complete_route_fraction")
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "percentile_iqr_overlap": overlap,
        "truth_free_complete_route_fraction": {
            "left": left_complete,
            "right": right_complete,
        },
        "do_not_treat_as_alignment_correctness": True,
    }


def evaluate_fullscale_route_dq(
    blocks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibration-only selected-graph admission for Station Mode."""
    if any(block.get("role") in {ROLE_HOLDOUT, ROLE_HELD_OUT_DQ} and block.get("used_for_verdict") for block in blocks):
        raise ValueError("holdout and held-out DQ must not be used for the route-DQ verdict")
    calibration = [block for block in blocks if block.get("role") == ROLE_CALIBRATION]
    if len(calibration) != 2:
        raise ValueError("fullscale route DQ expects exactly two calibration blocks")
    min_routes = int(config.get("min_selected_routes_calibration", 1))
    min_pairs = int(config.get("min_all_pairs_candidates", 50))
    max_share = float(config.get("max_event_share_of_selected_routes", 0.5))
    reasons: list[str] = []
    per_block: list[dict[str, Any]] = []
    for block in calibration:
        block_reasons: list[str] = []
        if int(block.get("n_all_pairs_candidates") or 0) < min_pairs:
            block_reasons.append("too_few_all_pairs_candidates")
        if int(block.get("selected_routes") or 0) < min_routes:
            block_reasons.append("too_few_selected_routes")
        if not block.get("candidate_graph_nonempty"):
            block_reasons.append("empty_all_pairs_graph")
        share = float(((block.get("event_concentration") or {}).get("max_event_share_of_selected_routes")) or 0.0)
        if int(block.get("selected_routes") or 0) > 0 and share > max_share:
            block_reasons.append("selected_single_event_dominated")
        if block.get("values_finite") is False:
            block_reasons.append("non_finite_values")
        per_block.append(
            {
                "run": block.get("run"),
                "source_id": block.get("source_id"),
                "passed": not block_reasons,
                "reasons": block_reasons,
            }
        )
        reasons.extend(block_reasons)
    stability = run_to_run_stability(calibration[0], calibration[1])
    if not stability["compatible"]:
        reasons.extend(str(item) for item in stability["reasons"])
    passed = all(item["passed"] for item in per_block) and bool(stability["compatible"])
    return {
        "passed": passed,
        "per_block": per_block,
        "run_to_run_stability": stability,
        "verdict_roles": [ROLE_CALIBRATION],
        "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
        "do_not_retune_v2": True,
        "residual_reduction_is_not_alignment_success": True,
        "reasons": list(dict.fromkeys(reasons)),
    }


def frozen_a_stability_status(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Report the already-frozen MC A 10% gate.  Do not remeasure A on real data."""
    stability = dict(contract.get("A_stability") or {})
    max_rel = float(stability.get("transfer_max_rel_deviation", A_STABILITY_MAX_REL_DEVIATION))
    spread_dx = stability.get("observed_A_dx_rel_spread")
    spread_ry = stability.get("observed_A_ry_rel_spread")
    frozen_passed = bool(
        spread_dx is not None
        and spread_ry is not None
        and float(spread_dx) <= max_rel
        and float(spread_ry) <= max_rel
    )
    return {
        "frozen_mc_A_stability_passed": frozen_passed,
        "real_data_A_remeasurement": "not_available_without_dedicated_C_dx_variation",
        "do_not_retune": True,
        "transfer_max_rel_deviation": max_rel,
        "observed_A_dx_rel_spread": None if spread_dx is None else float(spread_dx),
        "observed_A_ry_rel_spread": None if spread_ry is None else float(spread_ry),
        "gate": "frozen_mc_registration_only",
    }


def assign_next_decision(
    *,
    route_dq_passed: bool,
    fd_submitted: bool,
    newton_complete: bool,
    capture_passed: bool | None,
    implied_cdx_exceeds_operating_band: bool,
    independent_cdx_evidence: bool,
    holdout_dq_worsened: bool,
    residual_used_as_success: bool = False,
) -> dict[str, Any]:
    """Unique next-stage label.  Write stays false unless every contract gate passes."""
    if residual_used_as_success:
        raise ValueError("residual reduction must not be used as alignment success")
    reasons: list[str] = []
    if not route_dq_passed:
        decision = DECISION_ROUTE_DQ_INSUFFICIENT
        reasons.append("fullscale_selected_graph_failed_route_dq")
    elif holdout_dq_worsened:
        decision = DECISION_STATION_FIT_FAILED
        reasons.append("holdout_dq_worsened")
    elif not newton_complete:
        decision = DECISION_SELF_NULLING_FD_SUBMITTED if fd_submitted else DECISION_SELF_NULLING_PENDING
        reasons.append("self_nulling_newton_not_complete")
        if fd_submitted:
            reasons.append("calibration_fd_athena_submitted")
    elif capture_passed is False:
        decision = DECISION_STATION_FIT_FAILED
        reasons.append("self_nulling_station_capture_not_passed")
    elif implied_cdx_exceeds_operating_band:
        decision = DECISION_CROSS_LEVEL_CONTAMINATED
        reasons.append("implied_C_dx_exceeds_operating_band")
    elif not independent_cdx_evidence:
        decision = DECISION_CROSS_LEVEL_PRECONDITION_UNVERIFIED
        reasons.append("true_C_dx_not_independently_proven")
    else:
        decision = DECISION_DRYRUN_PASSED_NOT_WRITABLE
        reasons.append("station_mode_dryrun_observables_ok_write_still_blocked")
    write_allowed = False
    return {
        "decision": decision,
        "geometry_write_allowed": write_allowed,
        "cdx_mode_allowed": False,
        "official_conditions_db_write": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "residual_reduction_is_not_alignment_success": True,
        "reasons": reasons,
    }


def leakage_report(
    contract: Mapping[str, Any],
    *,
    station_dx_by_run: Mapping[int, float | None],
    station_ry_by_run: Mapping[int, float | None],
    independent_cdx_evidence: bool = False,
    independent_abs_C_dx_um: float | None = None,
) -> dict[str, Any]:
    """Frozen-A implied |C_dx|.  Not a C_dx measurement."""
    diagnostics: dict[str, Any] = {}
    dx_values = [float(value) for value in station_dx_by_run.values() if value is not None]
    spread = None if len(dx_values) < 2 else float(max(dx_values) - min(dx_values))
    for run, dx in station_dx_by_run.items():
        diagnostics[str(run)] = cross_level_contamination_diagnostic(
            contract,
            station_dx_mm=dx,
            station_ry_mrad=station_ry_by_run.get(run),
            run_to_run_dx_spread_mm=spread,
            independent_cdx_evidence=independent_cdx_evidence,
            independent_abs_C_dx_um=independent_abs_C_dx_um,
        )
    exceeds = any(bool(item.get("exceeds_operating_band")) for item in diagnostics.values())
    write = any(bool((item.get("mode_validity") or {}).get("geometry_write_allowed")) for item in diagnostics.values())
    return {
        "A_stability": frozen_a_stability_status(contract),
        "per_run": diagnostics,
        "run_to_run_dx_spread_mm": spread,
        "exceeds_operating_band": exceeds,
        "geometry_write_allowed": False if not independent_cdx_evidence else bool(write and not exceeds),
        "not_a_C_dx_measurement": True,
        "cdx_mode_started": False,
        "residual_reduction_is_not_alignment_success": True,
    }


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False


def campaign_allows_geometry_write(decision: str) -> bool:
    return False


def load_frozen_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return load_mode_validity_contract(root / str(config["mode_validity_contract"]))


def load_frozen_candidate_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return load_candidate_graph_dq_gate(root / str(config["candidate_graph_dq_gate"]))


def evaluate_frozen_selected_graph_gate(
    blocks: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-frozen candidate-graph DQ gate to full-segment selected graphs."""
    gate = load_frozen_candidate_gate(config)
    return evaluate_candidate_graph_dq_gate(blocks, gate)


def block_status_for_role(
    *,
    role: str,
    route_dq_passed: bool,
    newton_complete: bool,
    cross_level_contaminated: bool,
) -> str:
    return assign_block_status(
        role=role,
        dq_failed=not route_dq_passed and role == ROLE_CALIBRATION,
        cross_level_contaminated=cross_level_contaminated,
        station_mode_valid=False,
        cdx_mode_valid=False,
        holdout_dq_worsened=False,
        anomalous_run_drift=False,
        candidate_graph_dq_failed=not route_dq_passed and role == ROLE_CALIBRATION,
        candidate_graph_dq_passed=bool(route_dq_passed and not newton_complete),
    )
