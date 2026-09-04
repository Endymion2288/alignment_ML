"""Cluster-local observable cross-run identifiability and transfer V1.

Workbooks 68/69 remain frozen tracklet-level failures.  Workbook 70 restored
exact join only.  Entry 58 remains a negative-control metric until this
campaign classifies it under unified exact-join wiring.

Official rank uses ``A = W^{1/2} J S`` with native millimetre / milliradian
columns.  The entry-58 naked mixed-unit SVD is reproduced only as a
classification control.  First stage is r14973 vs r14974; three-arm and
Frozen-V2 alignment stay closed.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.cluster_local_jacobian_transfer_repair import (
    EXACT_JOIN_KEY,
    INHERITED_ENTRY_58,
    exact_join_audit,
)
from alignment.identifiable_subspace import (
    CLUSTER_LOCAL_NATIVE_NAMES,
    FROZEN_RANK_TOLERANCE,
    MixedUnitNakedJacobianError,
    core_contained_in_projector,
    flatten_diagonal_jacobian,
    frozen_scales_for,
    frozen_units_for,
    identifiable_svd,
    principal_angles_deg,
    refuse_fuzzy_or_nearest_neighbour_join,
    refuse_rank_threshold_from_spectrum,
    subspace_as_json,
    subspace_distance,
    svd_naked_jacobian,
)
from alignment.module_level_residual_poc import cosine, load_selected_routes, select_smoke_modules
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.true_cluster_local_residual import (
    MEASUREMENT_SOURCE,
    RESIDUAL_KIND,
    UNBIASED_METHOD,
    assert_no_alignment_payload,
    common_operating_state,
    parameter_columns,
    run_fd_smoke,
    validate_residuals,
)
from alignment.true_cluster_local_stability_transfer import (
    DECISION_NOT_TRANSFERABLE,
    attach_route_multiplicity,
    bootstrap_event_summaries,
    build_jacobian,
    coverage_matched_draws,
    coverage_tables,
    decide_next_stage,
    event_groups,
    fd_steps,
    group_indices,
    half_split_summaries,
    load_run_measurements,
    parameter_subspace_transfer,
    route_metadata,
    summarize_distribution,
    summarize_jacobian,
    track_slope,
)


SCHEMA_VERSION = "faser-cluster-local-observable-cross-run-identifiability-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "cluster_local_observable_cross_run_identifiability_v1.yaml"
INHERITED_ENTRY_68 = "tracker_only_identifiable_basis_unstable_solve_stopped"
INHERITED_ENTRY_69 = "cross_source_stable_core_independent_validation_fail"
INHERITED_ENTRY_70 = "cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign"
DECISION_TRANSFERABLE = "cluster_local_observable_cross_run_subspace_transferable_three_arm_not_opened"
DECISION_NOT_PORTABLE = "cluster_local_observable_not_cross_run_portable"
DECISION_JACOBIAN_INVALID = "cluster_local_jacobian_validity_failed"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "survey_is_alignment_input",
    "retune_scales_from_singular_values",
    "unknown_association_frozen_v2_closure_opened",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_run_full_parameter_newton",
    "do_not_solve_alignment_correction_on_real_data",
    "do_not_write_official_conditions",
    "do_not_write_geometry_pool_or_cool",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_invent_new_cosine_cut",
    "do_not_select_events_from_residual_or_cosine",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_open_sealed_test",
    "do_not_svd_naked_mixed_unit_jacobian",
    "do_not_retune_scales_from_singular_values",
    "do_not_retune_rank_threshold_from_spectrum",
    "do_not_rescue_tracklet_level_observable",
    "do_not_redefine_tracklet_stable_core_criterion",
    "do_not_fuzzy_match_join",
    "do_not_nearest_neighbour_join",
    "do_not_reselect_population_to_force_nonzero_join",
    "do_not_run_three_arm_this_stage",
    "do_not_open_frozen_v2_alignment_loop",
    "do_not_expand_beyond_r14973_r14974_this_stage",
    "entry_70_join_repair_is_not_identifiability_success",
    "entry_58_decision_not_silently_overwritten",
    "software_fd_sensitivity_only",
    "survey_is_external_cross_check_only",
)


def load_campaign_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"cluster-local identifiability config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected cluster-local identifiability schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"cluster-local identifiability config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"cluster-local identifiability config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("inherited_entry_58_decision") != INHERITED_ENTRY_58:
        raise ValueError("campaign must inherit the entry-58 identifiability freeze")
    if payload.get("inherited_entry_68_decision") != INHERITED_ENTRY_68:
        raise ValueError("campaign must inherit the workbook-68 freeze")
    if payload.get("inherited_entry_69_decision") != INHERITED_ENTRY_69:
        raise ValueError("campaign must inherit the workbook-69 freeze")
    if payload.get("inherited_entry_70_decision") != INHERITED_ENTRY_70:
        raise ValueError("campaign must inherit the workbook-70 join-repair freeze")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("exact_join_key") != EXACT_JOIN_KEY:
        raise ValueError("exact join key must remain run+event+cluster_identifier")
    if float(payload["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        refuse_rank_threshold_from_spectrum()
    if tuple(payload["parameter_names"]) != CLUSTER_LOCAL_NATIVE_NAMES:
        raise ValueError("cluster-local parameter names must remain the frozen 3-DoF set")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def _operating_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "survey_is_alignment_input": False,
            "survey_is_external_cross_check_only": True,
            "geometry_write_allowed": False,
            "unknown_association_frozen_v2_closure_opened": False,
            "real_data_alignment_correction_solved": False,
            "inherited_entry_58_decision": INHERITED_ENTRY_58,
            "inherited_entry_68_decision": INHERITED_ENTRY_68,
            "inherited_entry_69_decision": INHERITED_ENTRY_69,
            "inherited_entry_70_decision": INHERITED_ENTRY_70,
            "modes_are_reconstruction_observable_linear_combinations": True,
            "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
            "unbiased_method": UNBIASED_METHOD,
            "measurement_source": MEASUREMENT_SOURCE,
            "tracklet_level_rescue_forbidden": True,
        }
    )
    return state


def native_jacobian_mm_mrad(
    measurements: Sequence[Any],
    *,
    station_id: int,
    steps: Mapping[str, float],
) -> np.ndarray:
    """Central FD Jacobian with native columns ``[1/mm, 1/mrad, 1/mm]``.

    ``parameter_columns`` differentiates ``ry`` per radian.  Convert to
    milliradian before forming ``A = W^{1/2} J S``.
    """
    _names, jacobian = parameter_columns(
        measurements,
        station_id=int(station_id),
        translation_step_mm=float(steps["translation_step_mm"]),
        rotation_step_rad=float(steps["rotation_step_rad"]),
        c_dx_step_mm=float(steps["c_dx_step_mm"]),
    )
    native = np.asarray(jacobian, dtype=np.float64)
    native = np.array(native, copy=True)
    native[:, 1] = native[:, 1] * 1.0e-3
    return native


def official_subspace_from_native(
    native_jacobian: object,
    residual_variance: object,
    *,
    rank_tolerance: float = FROZEN_RANK_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    names = CLUSTER_LOCAL_NATIVE_NAMES
    scales = frozen_scales_for(names)
    units = frozen_units_for(names)
    weighted, flat, weight_sqrt = flatten_diagonal_jacobian(
        native_jacobian, residual_variance, scales
    )
    subspace = identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=units,
        parameter_scales=scales,
        rank_tolerance=rank_tolerance,
    )
    extras = {
        "weighted_matrix": weighted,
        "flat_jacobian": flat,
        "weight_sqrt": weight_sqrt,
        "n_observations": int(weighted.shape[0]),
        "normal_matrix": weighted.T @ weighted,
    }
    extras["normal_matrix_condition"] = float(
        np.linalg.cond(extras["normal_matrix"])
    ) if extras["n_observations"] else float("inf")
    extras["column_norms_native"] = [
        float(np.linalg.norm(flat[:, index])) for index in range(flat.shape[1])
    ]
    extras["column_norms_weighted"] = [
        float(np.linalg.norm(weighted[:, index])) for index in range(weighted.shape[1])
    ]
    extras["column_cosines_native"] = _column_cosines(flat)
    extras["column_cosines_weighted"] = _column_cosines(weighted)
    extras["not_relabeled_as_mechanical_ry_or_C_dx"] = True
    return subspace, extras


def _column_cosines(matrix: np.ndarray) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        return {}
    return {
        "station_dx_vs_C_dx": abs(cosine(matrix[:, 0], matrix[:, 2])),
        "station_ry_vs_C_dx": abs(cosine(matrix[:, 1], matrix[:, 2])),
        "station_dx_vs_station_ry": abs(cosine(matrix[:, 0], matrix[:, 1])),
    }


def jacobian_validity_audit(
    *,
    join: Mapping[str, Any],
    measurements: Sequence[Any],
    native_jacobian: np.ndarray,
    fd_smoke: Mapping[str, Any],
    validity_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    residual_report = validate_residuals(measurements) if measurements else {
        "n_measurements": 0,
        "n_routes": 0,
        "finite": False,
    }
    finite_jac = bool(np.isfinite(native_jacobian).all()) if native_jacobian.size else False
    missing_rate = 0.0
    if int(join.get("n_wanted_cluster_keys", 0)):
        missing_rate = float(join.get("n_missing_cluster_keys", 0)) / float(join["n_wanted_cluster_keys"])
    column_norms = [
        float(np.linalg.norm(native_jacobian[:, index]))
        for index in range(native_jacobian.shape[1])
    ] if native_jacobian.ndim == 2 and native_jacobian.size else []
    max_norm = max(column_norms) if column_norms else 0.0
    near_zero = [
        bool(norm <= float(validity_cfg["min_column_norm_relative"]) * max_norm)
        for norm in column_norms
    ] if max_norm > 0.0 else [True] * len(column_norms)
    reasons = []
    if validity_cfg.get("require_exact_join_complete") and not join.get("join_complete"):
        reasons.append("exact_join_incomplete")
    if int(residual_report.get("n_measurements", 0)) < int(validity_cfg["min_measurements"]):
        reasons.append("too_few_measurements")
    if int(residual_report.get("n_routes", 0)) < int(validity_cfg["min_routes"]):
        reasons.append("too_few_routes")
    if validity_cfg.get("require_finite_residuals") and not residual_report.get("finite"):
        reasons.append("non_finite_residuals")
    if validity_cfg.get("require_finite_jacobian") and not finite_jac:
        reasons.append("non_finite_jacobian")
    if validity_cfg.get("require_fd_smoke_pass") and not fd_smoke.get("passed"):
        reasons.append("fd_smoke_failed")
    if any(near_zero):
        reasons.append("near_zero_jacobian_column")
    if join.get("fuzzy_join_used") or join.get("nearest_neighbour_join_used"):
        reasons.append("non_exact_join")
        refuse_fuzzy_or_nearest_neighbour_join()
    return {
        "valid": not reasons,
        "failure_reasons": reasons,
        "join_complete": bool(join.get("join_complete")),
        "n_measurements": int(residual_report.get("n_measurements", 0)),
        "n_routes": int(residual_report.get("n_routes", 0)),
        "missing_cluster_key_rate": missing_rate,
        "finite_residuals": bool(residual_report.get("finite")),
        "finite_jacobian": finite_jac,
        "plus_minus_probes_present": True,
        "plus_minus_probes_source": "software_central_finite_difference_both_signs",
        "population_exact_matched_across_plus_minus": True,
        "athena_plus_minus_probes_not_used": True,
        "fd_smoke_passed": bool(fd_smoke.get("passed")),
        "fd_linearity_ok": bool(fd_smoke.get("jacobian_linear_region", fd_smoke.get("passed"))),
        "column_norms_native": column_norms,
        "near_zero_columns": near_zero,
        "residual_summary": residual_report,
        "fd_smoke": fd_smoke,
        "software_fd_only": True,
        "athena_fd_probes": False,
    }


def slope_tertile_official(
    native_jacobian: np.ndarray,
    measurements: Sequence[Any],
    residual_variance: np.ndarray,
    *,
    n_bins: int,
    min_routes: int,
    rank_tolerance: float,
) -> dict[str, Any]:
    groups = group_indices(measurements)
    keys = list(groups)
    slopes = []
    for key in keys:
        values = [track_slope(measurements[index]) for index in groups[key]]
        finite = [value for value in values if np.isfinite(value)]
        slopes.append(float(np.mean(finite)) if finite else float("nan"))
    slope_arr = np.asarray(slopes, dtype=np.float64)
    finite_mask = np.isfinite(slope_arr)
    reports = []
    if int(np.sum(finite_mask)) < int(n_bins) * int(min_routes):
        return {"too_small": True, "bins": [], "rank_drop": False}
    edges = np.quantile(slope_arr[finite_mask], np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[-1] = edges[-1] + 1.0e-12
    for index in range(int(n_bins)):
        selected = [
            key
            for key, slope in zip(keys, slope_arr)
            if np.isfinite(float(slope)) and edges[index] <= float(slope) < edges[index + 1]
        ]
        rows = [row for key in selected for row in groups[key]]
        if len(selected) < int(min_routes):
            reports.append(
                {
                    "label": f"slope_bin_{index}",
                    "n_routes": int(len(selected)),
                    "too_small": True,
                    "identifiable_rank": None,
                }
            )
            continue
        subspace, _extras = official_subspace_from_native(
            native_jacobian[rows],
            residual_variance[rows],
            rank_tolerance=rank_tolerance,
        )
        reports.append(
            {
                "label": f"slope_bin_{index}",
                "n_routes": int(len(selected)),
                "too_small": False,
                "identifiable_rank": int(subspace.identifiable_rank),
                "n_observations": int(len(rows)),
                "slope_interval": [float(edges[index]), float(edges[index + 1])],
            }
        )
    usable = [row for row in reports if not row.get("too_small") and row.get("identifiable_rank") is not None]
    ranks = [int(row["identifiable_rank"]) for row in usable]
    return {
        "too_small": False,
        "slope_bin_edges": [float(value) for value in edges],
        "bins": reports,
        "min_rank": min(ranks) if ranks else None,
        "max_rank": max(ranks) if ranks else None,
        "rank_drop": bool(ranks) and min(ranks) < max(ranks),
    }


def compare_official_subspaces(
    left: Any,
    right: Any,
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    distance = subspace_distance(left, right)
    persistence = core_contained_in_projector(left.v_id, right.projector_id) if left.identifiable_rank else {
        "min_mode_persistence": None,
        "max_principal_angle_deg": 0.0,
        "missing_projector_frobenius": 0.0,
        "mode_persistence": [],
    }
    reasons = []
    same_rank = bool(left.identifiable_rank == right.identifiable_rank)
    if gates.get("require_same_identifiable_rank") and not same_rank:
        reasons.append("identifiable_rank_inconsistent")
    max_angle = distance.get("max_identifiable_principal_angle_deg")
    if max_angle is not None and float(max_angle) > float(gates["max_principal_angle_deg"]):
        reasons.append("principal_angle_exceeds_predeclared_gate")
    frobenius = float(distance["projector_frobenius_distance"])
    if frobenius > float(gates["max_projector_frobenius"]):
        reasons.append("projector_frobenius_exceeds_predeclared_gate")
    persist = persistence.get("min_mode_persistence")
    if persist is not None and float(persist) < float(gates["min_mode_persistence"]):
        reasons.append("mode_persistence_below_predeclared_gate")
    weak_angle = None
    if (
        same_rank
        and int(left.identifiable_rank) >= 2
        and int(right.identifiable_rank) >= 2
    ):
        n_weak = min(2, int(left.identifiable_rank))
        weak_angles = principal_angles_deg(left.v_id[:, -n_weak:], right.v_id[:, -n_weak:])
        weak_angle = float(np.max(weak_angles)) if weak_angles.size else 0.0
        if weak_angle > float(gates["max_weak_plane_principal_angle_deg"]):
            reasons.append("weak_plane_principal_angle_exceeds_predeclared_gate")
    return {
        "same_identifiable_rank": same_rank,
        "left_rank": int(left.identifiable_rank),
        "right_rank": int(right.identifiable_rank),
        "max_principal_angle_deg": max_angle,
        "weak_plane_principal_angle_deg": weak_angle,
        "projector_frobenius_distance": frobenius,
        "min_mode_persistence": persist,
        "mode_persistence": persistence.get("mode_persistence"),
        "pass": not reasons,
        "failure_reasons": reasons,
        "compares_subspace_not_signed_vector_elements": True,
        "runs_are_statistical_unit": True,
        "not_pooled_before_rank": True,
        "not_relabeled_as_mechanical_ry_or_C_dx": True,
    }


def _bootstrap_pack(jacobian, measurements, *, n_event, n_half, seed, min_routes, reference_ry=None):
    event_rows = bootstrap_event_summaries(
        jacobian,
        measurements,
        n_replicates=n_event,
        seed=seed,
        min_routes=min_routes,
    )
    halves = half_split_summaries(
        jacobian,
        measurements,
        n_splits=n_half,
        seed=seed,
        min_routes=min_routes,
    )
    half_ry = [abs(left["station_ry_vs_C_dx"] - right["station_ry_vs_C_dx"]) for left, right in halves]
    return {
        "n_events": int(len(event_groups(measurements))),
        "n_routes": int(len(group_indices(measurements))),
        "n_measurements": int(len(measurements)),
        "event_bootstrap": {
            "station_ry_vs_C_dx": summarize_distribution(
                [row["station_ry_vs_C_dx"] for row in event_rows],
                reference=reference_ry,
            ),
            "station_dx_vs_C_dx": summarize_distribution(
                [row["station_dx_vs_C_dx"] for row in event_rows]
            ),
            "rank": summarize_distribution([row["rank"] for row in event_rows]),
        },
        "random_half_split": {
            "station_ry_vs_C_dx": summarize_distribution(half_ry),
        },
        "n_event_replicates": int(n_event),
        "n_half_splits": int(n_half),
        "naked_mixed_unit_svd": True,
        "official_campaign_metric": False,
    }


def reproduce_entry_58(
    *,
    config: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    transfer_bundle: Mapping[str, Any],
    station_id: int,
    steps: Mapping[str, float],
) -> dict[str, Any]:
    reproduction = config["entry_58_reproduction"]
    ref_meas = reference_bundle["measurements"]
    xfer_meas = transfer_bundle["measurements"]
    ref_j = build_jacobian(ref_meas, station_id=station_id, steps=steps)
    xfer_j = build_jacobian(xfer_meas, station_id=station_id, steps=steps)
    ref_point = summarize_jacobian(ref_j)
    xfer_point = summarize_jacobian(xfer_j)
    ref_meta = route_metadata(ref_meas)
    xfer_meta = route_metadata(xfer_meas)
    root = project_root()
    ref_routes = load_selected_routes(resolve_under_root(root, str(config["runs"]["reference"]["selected_routes"])))
    xfer_routes = load_selected_routes(resolve_under_root(root, str(config["runs"]["transfer"]["selected_routes"])))
    attach_route_multiplicity(ref_meta, ref_routes)
    attach_route_multiplicity(xfer_meta, xfer_routes)
    coverage = coverage_tables(
        ref_j,
        ref_meas,
        ref_meta,
        group_indices(ref_meas),
        n_slope_bins=int(config["gates"]["n_slope_bins"]),
        min_subset_routes=int(reproduction["min_routes"]),
    )
    bootstrap = _bootstrap_pack(
        ref_j,
        ref_meas,
        n_event=int(reproduction["n_event_replicates"]),
        n_half=int(reproduction["n_half_splits"]),
        seed=int(reproduction["seed"]),
        min_routes=int(reproduction["min_routes"]),
        reference_ry=float(ref_point["station_ry_vs_C_dx"]),
    )
    transfer_bootstrap = _bootstrap_pack(
        xfer_j,
        xfer_meas,
        n_event=int(reproduction["n_event_replicates"]),
        n_half=int(reproduction["n_half_splits"]),
        seed=int(reproduction["seed"]) + 1,
        min_routes=int(reproduction["min_routes"]),
        reference_ry=float(xfer_point["station_ry_vs_C_dx"]),
    )
    matched = coverage_matched_draws(
        ref_meta,
        xfer_j,
        xfer_meta,
        group_indices(xfer_meas),
        n_draws=int(reproduction["n_coverage_matched_draws"]),
        seed=int(reproduction["seed"]),
        n_slope_bins=int(config["gates"]["n_slope_bins"]),
        min_subset_routes=int(reproduction["min_routes"]),
    )
    decision = decide_next_stage(
        reference_point=ref_point,
        bootstrap=bootstrap,
        coverage=coverage,
        transfer_point=xfer_point,
        transfer_bootstrap=transfer_bootstrap,
        coverage_matched=matched,
        proxy_ry=float(reproduction["module_proxy_ry_cdx_abs_cosine"]),
    )
    decision["reproduced_on_unified_exact_join"] = True
    decision["keep_as_negative_control"] = True
    decision["official_campaign_does_not_use_this_rank"] = True
    decision["naked_mixed_unit_svd"] = True
    return {
        "reference_point": ref_point,
        "transfer_point": xfer_point,
        "parameter_space_principal_angles": parameter_subspace_transfer(ref_point, xfer_point),
        "coverage": {
            "slope_bins": coverage.get("slope_bins"),
            "largest_ry_cdx_shifts_when_leaving_a_group_out": coverage.get(
                "largest_ry_cdx_shifts_when_leaving_a_group_out"
            ),
            "topology_route_counts": coverage.get("topology_route_counts"),
        },
        "coverage_matched": matched,
        "bootstrap_summary": {
            "reference_ry_cdx": bootstrap["event_bootstrap"]["station_ry_vs_C_dx"],
            "transfer_ry_cdx": transfer_bootstrap["event_bootstrap"]["station_ry_vs_C_dx"],
        },
        "decision": decision,
    }


def classify_failure_cause(
    *,
    official_compare: Mapping[str, Any],
    reference_validity: Mapping[str, Any],
    transfer_validity: Mapping[str, Any],
    reference_tertiles: Mapping[str, Any],
    transfer_tertiles: Mapping[str, Any],
    entry_58: Mapping[str, Any],
    wrong_dump: Mapping[str, Any],
) -> dict[str, Any]:
    entry_decision = str((entry_58.get("decision") or {}).get("decision", ""))
    old_metric_reproduced = entry_decision == DECISION_NOT_TRANSFERABLE
    construction_diff = bool(
        official_compare.get("left_rank")
        != (entry_58.get("reference_point") or {}).get("rank")
        or official_compare.get("right_rank")
        != (entry_58.get("transfer_point") or {}).get("rank")
    )
    coverage = bool(reference_tertiles.get("rank_drop") or transfer_tertiles.get("rank_drop"))
    provenance = bool(
        not wrong_dump.get("exact_join_nonzero")
        and wrong_dump.get("mismatch_reason") == "run_id_disjoint_dump_does_not_contain_route_run"
    )
    physics = bool(
        old_metric_reproduced
        and (coverage or not official_compare.get("pass"))
        and reference_validity.get("valid")
        and transfer_validity.get("valid")
    )
    labels = []
    if provenance:
        labels.append("old_provenance_data_wiring_reproduced_as_negative_control")
    if construction_diff:
        labels.append("observable_or_jacobian_construction_difference_naked_vs_A")
    if coverage:
        labels.append("source_or_topology_coverage_difference_slope_tertile_rank_drop")
    if physics:
        labels.append("true_physics_or_coverage_non_transferability")
    if not labels:
        labels.append("unclassified")
    return {
        "labels": labels,
        "old_entry_58_metric_reproduced": old_metric_reproduced,
        "workbook_70_wrong_dump_still_zero": provenance,
        "official_rank_differs_from_naked_rank": construction_diff,
        "slope_tertile_rank_drop": coverage,
        "true_non_transferability_under_unified_exact_join": physics,
        "entry_70_does_not_automatically_overturn_entry_58": True,
    }


def decide_campaign(
    *,
    reference_valid: bool,
    transfer_valid: bool,
    rank_consistent: bool,
    subspace_transferable: bool,
    entry_58_decision: str,
    failure_cause: Mapping[str, Any],
) -> dict[str, Any]:
    four = {
        "cluster_local_reference_jacobian_valid": bool(reference_valid),
        "cluster_local_transfer_jacobian_valid": bool(transfer_valid),
        "cross_run_rank_consistent": bool(rank_consistent),
        "cross_run_subspace_transferable": bool(subspace_transferable),
    }
    all_pass = all(four.values())
    if not reference_valid or not transfer_valid:
        decision = DECISION_JACOBIAN_INVALID
        reasons = ["jacobian_validity_failed", "do_not_chase_by_retuning_or_reselecting"]
    elif not all_pass:
        decision = DECISION_NOT_PORTABLE
        reasons = list(failure_cause.get("labels") or [])
        reasons.append("do_not_chase_by_retuning_rank_S_selection_or_population")
    else:
        decision = DECISION_TRANSFERABLE
        reasons = [
            "four_predeclared_gates_passed",
            "three_arm_still_not_opened_this_stage",
            "frozen_v2_alignment_loop_still_not_opened",
        ]
    return {
        "decision": decision,
        "gates": four,
        "all_four_gates_passed": all_pass,
        "three_arm_authorized": False,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
        "geometry_write_allowed": False,
        "inherited_entry_58_decision": INHERITED_ENTRY_58,
        "entry_58_reproduction_decision": entry_58_decision,
        "entry_58_not_silently_overwritten": True,
        "authorize_more_runs": False,
        "reasons": reasons,
        "failure_cause": failure_cause,
        "if_failed_do_not_chase_by_retuning": True,
    }


def official_event_bootstrap(
    native_jacobian: np.ndarray,
    measurements: Sequence[Any],
    residual_variance: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
    min_routes: int,
    rank_tolerance: float,
    reference: Any | None = None,
) -> dict[str, Any]:
    events = list(event_groups(measurements).items())
    if len(events) < int(min_routes):
        return {"too_small": True, "n_events": int(len(events)), "n_valid": 0}
    rng = np.random.default_rng(int(seed))
    ranks = []
    angles = []
    persistences = []
    n_valid = 0
    for _ in range(int(n_replicates)):
        choice = rng.integers(0, len(events), size=len(events))
        rows = [row for index in choice for row in events[int(index)][1]]
        if len(rows) < 3:
            continue
        subspace, _extras = official_subspace_from_native(
            native_jacobian[rows],
            residual_variance[rows],
            rank_tolerance=rank_tolerance,
        )
        n_valid += 1
        ranks.append(int(subspace.identifiable_rank))
        if reference is not None and subspace.identifiable_rank == reference.identifiable_rank:
            distance = subspace_distance(reference, subspace)
            angle = distance.get("max_identifiable_principal_angle_deg")
            if angle is not None:
                angles.append(float(angle))
            contained = core_contained_in_projector(reference.v_id, subspace.projector_id)
            persist = contained.get("min_mode_persistence")
            if persist is not None:
                persistences.append(float(persist))
    return {
        "too_small": False,
        "kind": "event_with_replacement_official_subspace",
        "n_events": int(len(events)),
        "n_replicates_requested": int(n_replicates),
        "n_valid": int(n_valid),
        "rank": summarize_distribution(ranks),
        "max_principal_angle_vs_full_sample_deg": summarize_distribution(angles),
        "min_mode_persistence_vs_full_sample": summarize_distribution(persistences),
        "official_campaign_metric": True,
        "runs_remain_statistical_unit": True,
    }


def _fd_smoke_for(measurements: Sequence[Any], config: Mapping[str, Any]) -> dict[str, Any]:
    fd = config["finite_difference"]
    counts = Counter(item.module_id for item in measurements)
    try:
        smoke_modules = select_smoke_modules(
            {"hits_by_module": dict(counts)},
            station_id=int(config["representative"]["station_id"]),
            layer_id=int(config["representative"]["layer_id"]),
            n_modules=min(4, max(1, len(counts))),
        )
    except ValueError:
        smoke_modules = []
    return run_fd_smoke(
        measurements,
        smoke_modules,
        station_id=int(config["representative"]["station_id"]),
        translation_steps_mm=list(fd["translation_steps_mm"]),
        rotation_steps_mrad=list(fd["rotation_steps_mrad"]),
        c_dx_steps_mm=list(fd["c_dx_steps_mm"]),
        linearity_max_relative_deviation=float(fd["linearity_max_relative_deviation"]),
        targeting_max_offmodule_fraction=float(fd["targeting_max_offmodule_fraction"]),
        l1_locality_max_fraction=float(fd["l1_locality_max_fraction"]),
    )


def _run_pack(
    spec: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    cluster_dump: str,
) -> dict[str, Any]:
    root = project_root()
    station_id = int(config["representative"]["station_id"])
    join = exact_join_audit(spec, cluster_dump=cluster_dump, representative_station=station_id)
    bundle = load_run_measurements(
        spec,
        root=root,
        cluster_dump=cluster_dump,
        representative_station=station_id,
    )
    measurements = bundle["measurements"]
    steps = fd_steps({"finite_difference": {
        "translation_step_mm": config["finite_difference"]["translation_step_mm"],
        "rotation_step_mrad": config["finite_difference"]["rotation_step_mrad"],
        "c_dx_step_mm": config["finite_difference"]["c_dx_step_mm"],
    }})
    native = native_jacobian_mm_mrad(measurements, station_id=station_id, steps=steps) if measurements else np.zeros((0, 3))
    variance = np.asarray([float(item.local_u_var_mm2) for item in measurements], dtype=np.float64)
    smoke = _fd_smoke_for(measurements, config) if measurements else {"passed": False, "probes": []}
    validity = jacobian_validity_audit(
        join=join,
        measurements=measurements,
        native_jacobian=native,
        fd_smoke=smoke,
        validity_cfg=config["jacobian_validity"],
    )
    subspace = None
    extras: dict[str, Any] = {}
    tertiles: dict[str, Any] = {"too_small": True, "bins": [], "rank_drop": False}
    bootstrap: dict[str, Any] = {"too_small": True, "n_valid": 0}
    if measurements and native.size:
        subspace, extras = official_subspace_from_native(
            native,
            variance,
            rank_tolerance=float(config["rank_tolerance"]),
        )
        tertiles = slope_tertile_official(
            native,
            measurements,
            variance,
            n_bins=int(config["gates"]["n_slope_bins"]),
            min_routes=int(config["gates"]["min_tertile_routes"]),
            rank_tolerance=float(config["rank_tolerance"]),
        )
        bootstrap_cfg = config.get("bootstrap") or {}
        bootstrap = official_event_bootstrap(
            native,
            measurements,
            variance,
            n_replicates=int(bootstrap_cfg.get("n_event_replicates", 40)),
            seed=int(bootstrap_cfg.get("seed", 20260903)),
            min_routes=int(bootstrap_cfg.get("min_routes", 8)),
            rank_tolerance=float(config["rank_tolerance"]),
            reference=subspace,
        )
    return {
        "join": join,
        "bundle": bundle,
        "measurements": measurements,
        "native_jacobian": native,
        "variance": variance,
        "validity": validity,
        "subspace": subspace,
        "extras": extras,
        "tertiles": tertiles,
        "event_bootstrap": bootstrap,
        "steps": steps,
    }


def build_all_reports(
    config: Mapping[str, Any],
    *,
    packs: Mapping[str, Mapping[str, Any]] | None = None,
    wrong_dump: Mapping[str, Any] | None = None,
    entry_58_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat()
    root = project_root()
    reference_spec = config["runs"]["reference"]
    transfer_spec = config["runs"]["transfer"]
    if packs is None:
        reference = _run_pack(reference_spec, config=config, cluster_dump=str(reference_spec["cluster_dump"]))
        transfer = _run_pack(transfer_spec, config=config, cluster_dump=str(transfer_spec["cluster_dump"]))
        wrong = exact_join_audit(transfer_spec, cluster_dump=str(reference_spec["cluster_dump"]))
    else:
        reference = dict(packs["reference"])
        transfer = dict(packs["transfer"])
        wrong = dict(wrong_dump or packs.get("wrong_dump") or {})
    compare = {"pass": False, "same_identifiable_rank": False, "failure_reasons": ["missing_subspace"]}
    if reference["subspace"] is not None and transfer["subspace"] is not None:
        compare = compare_official_subspaces(
            reference["subspace"],
            transfer["subspace"],
            gates=config["gates"],
        )
        if config["gates"].get("require_no_slope_tertile_rank_drop") and (
            reference["tertiles"].get("rank_drop") or transfer["tertiles"].get("rank_drop")
        ):
            compare = dict(compare)
            compare["pass"] = False
            reasons = list(compare.get("failure_reasons") or [])
            reasons.append("slope_tertile_rank_drop")
            compare["failure_reasons"] = reasons
            compare["coverage_conditioned_extra_dimension"] = True
    if entry_58_override is not None:
        entry_58 = dict(entry_58_override)
        entry_58.setdefault("keep_as_negative_control", True)
        entry_58.setdefault("official_campaign_does_not_use_this_rank", True)
    else:
        entry_58 = reproduce_entry_58(
            config=config,
            reference_bundle=reference["bundle"],
            transfer_bundle=transfer["bundle"],
            station_id=int(config["representative"]["station_id"]),
            steps=reference["steps"],
        )
    failure_cause = classify_failure_cause(
        official_compare=compare,
        reference_validity=reference["validity"],
        transfer_validity=transfer["validity"],
        reference_tertiles=reference["tertiles"],
        transfer_tertiles=transfer["tertiles"],
        entry_58=entry_58,
        wrong_dump=wrong,
    )
    decision = decide_campaign(
        reference_valid=bool(reference["validity"]["valid"]),
        transfer_valid=bool(transfer["validity"]["valid"]),
        rank_consistent=bool(compare.get("same_identifiable_rank")),
        subspace_transferable=bool(compare.get("pass")),
        entry_58_decision=str((entry_58.get("decision") or {}).get("decision", "")),
        failure_cause=failure_cause,
    )

    def _run_json(pack: Mapping[str, Any]) -> dict[str, Any]:
        subspace = pack["subspace"]
        extras = pack["extras"]
        return {
            "join": pack["join"],
            "validity": pack["validity"],
            "n_measurements": int(len(pack["measurements"])),
            "identifiable_rank": None if subspace is None else int(subspace.identifiable_rank),
            "null_dimension": None if subspace is None else int(subspace.null_dimension),
            "singular_values": None if subspace is None else [float(value) for value in subspace.singular_values],
            "column_norms_native": extras.get("column_norms_native"),
            "column_norms_weighted": extras.get("column_norms_weighted"),
            "column_cosines_native": extras.get("column_cosines_native"),
            "column_cosines_weighted": extras.get("column_cosines_weighted"),
            "normal_matrix_condition": extras.get("normal_matrix_condition"),
            "subspace": None if subspace is None else subspace_as_json(subspace),
            "slope_tertiles": pack["tertiles"],
            "event_bootstrap": pack.get("event_bootstrap"),
            "not_relabeled_as_mechanical_ry_or_C_dx": True,
        }

    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": _operating_state(),
            "exact_join_key": EXACT_JOIN_KEY,
            "parameter_names": list(CLUSTER_LOCAL_NATIVE_NAMES),
            "parameter_units": list(frozen_units_for(CLUSTER_LOCAL_NATIVE_NAMES)),
            "scale_matrix_S": [float(value) for value in frozen_scales_for(CLUSTER_LOCAL_NATIVE_NAMES)],
            "weighted_matrix": "A = W^{1/2} J S",
            "rank_tolerance": float(config["rank_tolerance"]),
            "jacobian_native_ry_unit": "mrad",
            "residual_weight": "diagonal_inverse_cluster_local_u_variance",
            "assumptions": list(config.get("assumptions", [])),
            "forbidden": {
                "naked_mixed_unit_svd_as_official_rank": True,
                "tracklet_level_rescue": True,
                "fuzzy_join": True,
                "three_arm_this_stage": True,
                "frozen_v2_alignment_loop": True,
                "geometry_write": True,
            },
        },
        "reference_run": _run_json(reference),
        "transfer_run": _run_json(transfer),
        "wrong_dump_control": wrong,
        "cross_run_subspace": compare,
        "entry_58_reproduction": entry_58,
        "next_stage_decision": decision,
    }
    for payload in reports.values():
        if isinstance(payload, Mapping):
            assert_no_alignment_payload(payload)
    return reports


def refuse_forbidden_operations() -> dict[str, Any]:
    try:
        svd_naked_jacobian(np.eye(3), parameter_units=("mm", "mrad", "mm"))
        naked = False
    except MixedUnitNakedJacobianError:
        naked = True
    try:
        refuse_fuzzy_or_nearest_neighbour_join()
        fuzzy = False
    except Exception:
        fuzzy = True
    return {
        "refused_naked_mixed_unit_svd": naked,
        "refused_fuzzy_or_nearest_neighbour_join": fuzzy,
        "geometry_write_allowed": False,
        "full_parameter_newton": False,
        "real_data_alignment_correction": False,
        "opened_sealed_test": False,
        "rescued_tracklet_level_observable": False,
        "three_arm_opened": False,
    }
