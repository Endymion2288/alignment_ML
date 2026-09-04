"""Rigid-station-only tracker alignment identifiability V1.

Workbooks 68-72 remain frozen.  This campaign does not rescue the 7D
tracklet-level observable, the cluster-local observable, or the
cross-source stable core.  Rank tolerance and S are inherited and never
retuned from the 5DoF spectrum.

Tracker fit may float only station rigid-body ``dx, dy, rx, ry, rz``.
``dz`` is not given to tracks.  Internal plane/module geometry and
``C_dx`` stay fixed at the current reconstruction geometry / metrology
state.  That is a model boundary, not a measurement of ``C_dx=0``.

Native five-column ``A = W^{1/2} J S`` is reconstructed from hierarchical
V1 physical FD banks via ``only_parameters``.  Deleting ``dz`` / ``C_dx``
from a previous 7D SVD object is refused.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.five_dof_sampling import FREE_PARAMETERS, LINEAR_SEVERITY_MAX
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    IdentifiableSubspace,
    MixedUnitNakedJacobianError,
    frozen_scales_for,
    frozen_units_for,
    refuse_rank_threshold_from_spectrum,
    refuse_scale_or_threshold_retune,
    subspace_as_json,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.tracker_only_identifiable_subspace import (
    _pool_campaign_banks,
    bootstrap_subspaces,
    evaluate_three_arms,
    half_split_subspaces,
    load_physical_banks,
    subspace_from_physical_bank,
    summarize_stability,
    three_arm_payloads,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, assert_no_alignment_payload, common_operating_state
from datasets.root_loader import load_events
from scripts.audit_6dof_identifiability import _complete_route_mask, _source_entries
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-rigid-station-only-tracker-alignment-identifiability-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "rigid_station_only_tracker_alignment_identifiability_v1.yaml"
STATION_FIVE_NAMES: tuple[str, ...] = FREE_PARAMETERS
FORBIDDEN_TRACKER_PARAMETERS: tuple[str, ...] = ("ift_dz_mm", "C_dx")
INHERITED_ENTRY_68 = "tracker_only_identifiable_basis_unstable_solve_stopped"
INHERITED_ENTRY_69 = "cross_source_stable_core_independent_validation_fail"
INHERITED_ENTRY_70 = "cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign"
INHERITED_ENTRY_71 = "cluster_local_observable_not_cross_run_portable"
INHERITED_ENTRY_72 = "tracklet_independent_failure_provenance_audit_complete_sources_not_dropped"

DECISION_PASS = "rigid_station_five_dof_identifiable_and_portable_three_arm_not_opened"
DECISION_CLOSURE_PASS = "rigid_station_five_dof_identifiable_and_truth_selected_closure_pass"
DECISION_IDENTIFIABILITY_FAIL = "rigid_station_five_dof_not_source_or_coverage_portable"
DECISION_JACOBIAN_INVALID = "rigid_station_five_dof_jacobian_validity_failed"
DECISION_THREE_ARM_FAIL = "rigid_station_five_dof_identifiable_but_truth_selected_closure_fail"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_identifiable_modes",
    "survey_is_alignment_input",
    "three_arm_authorized",
    "frozen_v2_alignment_loop_authorized",
    "real_data_correction_authorized",
    "retune_scales_from_singular_values",
    "unknown_association_frozen_v2_closure_opened",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_construct_station_calibration_mode",
    "do_not_reopen_workbook_51_reduced_station_mode",
    "do_not_reopen_workbook_36_as_this_campaign",
    "do_not_enter_cdx_mode",
    "do_not_run_full_parameter_newton",
    "do_not_solve_alignment_correction_on_real_data",
    "do_not_write_official_conditions",
    "do_not_write_geometry_pool_or_cool",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_enter_full_module_identifiability_map",
    "do_not_invent_new_cosine_cut",
    "do_not_select_events_from_residual_or_cosine",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_open_sealed_test",
    "do_not_svd_naked_mixed_unit_jacobian",
    "do_not_retune_scales_from_singular_values",
    "do_not_retune_rank_threshold_from_spectrum",
    "do_not_treat_null_zero_as_a_measurement",
    "do_not_relabel_modes_as_mechanical_parameters",
    "do_not_rescue_seven_d_observable",
    "do_not_rescue_cluster_local_observable",
    "do_not_redefine_tracklet_stable_core_criterion",
    "do_not_drop_entry_69_failure_sources",
    "do_not_drop_failed_sources_to_recover_rank_five",
    "do_not_delete_dz_or_cdx_columns_from_seven_d_svd",
    "do_not_force_write_old_survey_central_values",
    "do_not_open_three_arm_unless_identifiability_passes",
    "do_not_open_frozen_v2_alignment_loop",
    "solver_restricted_to_identifiable_subspace",
    "truth_selected_association_is_solver_control",
    "survey_is_external_cross_check_only",
    "residual_reduction_is_not_alignment_success",
    "implied_cdx_is_not_a_measurement",
    "cdx_fixed_is_not_a_measurement_of_zero",
    "dz_is_not_a_tracker_free_parameter",
    "pooled_rank_is_not_portability",
    "this_is_not_seven_d_column_deletion",
    "this_is_not_workbook_36_survey_dz_curriculum",
)


class SevenDColumnDeletionError(ValueError):
    """Raised when a caller tries to form 5DoF SVD by deleting 7D columns."""


def refuse_seven_d_column_deletion() -> None:
    raise SevenDColumnDeletionError(
        "refusing 5DoF identifiability from a previous 7D SVD with dz/C_dx "
        "columns deleted; reconstruct native five-column A = W^{1/2} J S "
        "via only_parameters"
    )


def load_campaign_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"rigid-station config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected rigid-station schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"rigid-station config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"rigid-station config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("inherited_entry_68_decision") != INHERITED_ENTRY_68:
        raise ValueError("campaign must inherit workbook-68 freeze")
    if payload.get("inherited_entry_69_decision") != INHERITED_ENTRY_69:
        raise ValueError("campaign must inherit workbook-69 freeze")
    if payload.get("inherited_entry_70_decision") != INHERITED_ENTRY_70:
        raise ValueError("campaign must inherit workbook-70 freeze")
    if payload.get("inherited_entry_71_decision") != INHERITED_ENTRY_71:
        raise ValueError("campaign must inherit workbook-71 freeze")
    if payload.get("inherited_entry_72_decision") != INHERITED_ENTRY_72:
        raise ValueError("campaign must inherit workbook-72 freeze")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the residual label")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if float(payload["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        refuse_rank_threshold_from_spectrum()
    if payload.get("retune_scales_from_singular_values") is True:
        refuse_scale_or_threshold_retune(from_singular_values=True)
    names = tuple(str(name) for name in payload["parameter_names"])
    if names != STATION_FIVE_NAMES:
        raise ValueError(
            "rigid-station parameter_names must be the frozen station 5DoF order "
            f"{STATION_FIVE_NAMES}; got {names}"
        )
    corpus_names = tuple(str(name) for name in payload["jacobian_corpus"]["parameter_names"])
    if corpus_names != names:
        raise ValueError("jacobian_corpus.parameter_names must match parameter_names")
    if payload["jacobian_corpus"].get("forbidden_reconstruction") != "drop_dz_cdx_from_previous_seven_d_svd":
        raise ValueError("config must forbid 7D column-deletion reconstruction")
    if payload["jacobian_corpus"].get("reconstruction_method") != "native_five_column_only_parameters":
        raise ValueError("5DoF Jacobian must be reconstructed via only_parameters")
    forbidden = set(FORBIDDEN_TRACKER_PARAMETERS).intersection(names)
    if forbidden:
        raise ValueError(f"tracker fit must not float {sorted(forbidden)}")
    scales = payload["scale_matrix_S"]
    expected = frozen_scales_for(names)
    declared = np.asarray([float(scales[name]) for name in names], dtype=np.float64)
    if not np.allclose(declared, expected, rtol=0.0, atol=0.0):
        raise ValueError(
            "scale_matrix_S must match the frozen station-free severity scales; "
            f"declared={declared.tolist()} frozen={expected.tolist()}"
        )
    units = tuple(str(unit) for unit in payload["parameter_units"])
    if units != frozen_units_for(names):
        raise ValueError("parameter_units must match frozen station-free units")
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
            "official_conditions_write_allowed": False,
            "three_arm_authorized": False,
            "frozen_v2_alignment_loop_authorized": False,
            "real_data_correction_authorized": False,
            "solver_restricted_to_identifiable_subspace": True,
            "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
            "modes_are_reconstruction_observable_linear_combinations": True,
            "cdx_fixed_is_not_a_measurement_of_zero": True,
            "dz_is_not_a_tracker_free_parameter": True,
            "this_is_not_seven_d_column_deletion": True,
            "this_is_not_workbook_36_survey_dz_curriculum": True,
            "pooled_rank_is_not_portability": True,
        }
    )
    return state


def _predeclared_source_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    corpus = config["jacobian_corpus"]
    train = tuple(str(item) for item in corpus["train_source_ids"])
    validation = tuple(str(item) for item in corpus["validation_source_ids"])
    if set(train) & set(validation):
        raise ValueError("train and validation source ids overlap")
    return train + validation


def _loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    corpus = dict(config["jacobian_corpus"])
    corpus["source_ids"] = list(_predeclared_source_ids(config))
    payload["jacobian_corpus"] = corpus
    return payload


def attach_source_tracklet_slopes(
    bank: Mapping[str, Any],
    *,
    iteration_manifest: Mapping[str, Any],
    anchor_point: str,
) -> dict[str, Any]:
    """Attach hypot(source_tx, source_ty) from the FD-anchor tracklet ROOT.

    The slope is local SegmentFit state at the source station, not leftover
    residual ``(rtx, rty)`` and not spectrometer ``Δx/Δz``.
    """
    work = dict(bank)
    if "source_tx" in work and "source_ty" in work and "source_slope" in work:
        return work
    source_id = str(bank["source_id"])
    entries = {str(entry["source_id"]): entry for entry in _source_entries(iteration_manifest)}
    if source_id not in entries:
        raise ValueError(f"iteration manifest has no source {source_id}")
    root = Path(str(entries[source_id]["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    points = {str(point["name"]): point for point in plan["points"]}
    if anchor_point not in points:
        raise ValueError(f"scan plan lacks anchor {anchor_point}")
    relative = points[anchor_point].get("relative_point_dir")
    tracklets_path = root / str(relative) / "refit" / "tracklets.root"
    events = load_events(tracklets_path, require_mc_labels=True)
    by_tracklet: dict[tuple[int, int, int, int], tuple[float, float]] = {}
    by_truth: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
    for event in events:
        truth = event.truth_particle_id
        for row in range(event.size):
            tx = float(event.state[row, 2])
            ty = float(event.state[row, 3])
            by_tracklet[
                (
                    int(event.run_id),
                    int(event.event_id),
                    int(event.station_id[row]),
                    int(event.tracklet_id[row]),
                )
            ] = (tx, ty)
            if truth is None:
                continue
            by_truth.setdefault(
                (
                    int(event.run_id),
                    int(event.event_id),
                    int(event.station_id[row]),
                    int(truth[row]),
                ),
                [],
            ).append((tx, ty))
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    source_tx = np.full(n_pairs, np.nan, dtype=np.float64)
    source_ty = np.full(n_pairs, np.nan, dtype=np.float64)
    missing = 0
    tracklet_ids = bank.get("source_tracklet_id")
    for index in range(n_pairs):
        values = None
        if tracklet_ids is not None:
            values = by_tracklet.get(
                (
                    int(bank["run_id"][index]),
                    int(bank["event_id"][index]),
                    int(bank["source_station_id"][index]),
                    int(tracklet_ids[index]),
                )
            )
        if values is None:
            truth_key = (
                int(bank["run_id"][index]),
                int(bank["event_id"][index]),
                int(bank["source_station_id"][index]),
                int(bank["truth_particle_id"][index]),
            )
            candidates = by_truth.get(truth_key, [])
            if len(candidates) == 1:
                values = candidates[0]
        if values is None:
            missing += 1
            continue
        source_tx[index], source_ty[index] = values
    work["source_tx"] = source_tx
    work["source_ty"] = source_ty
    work["source_slope"] = np.hypot(source_tx, source_ty)
    work["source_slope_definition"] = "hypot_source_tracklet_tx_ty"
    work["source_slope_missing_pairs"] = int(missing)
    work["source_slope_is_not_leftover_residual_rtx_rty"] = True
    return work


def load_five_dof_banks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = tuple(str(name) for name in config["parameter_names"])
    if names != STATION_FIVE_NAMES:
        raise ValueError("five-DoF banks require the frozen station-free names")
    if any(name in FORBIDDEN_TRACKER_PARAMETERS for name in names):
        raise ValueError("five-DoF banks must not load dz or C_dx probes")
    banks = load_physical_banks(_loader_config(config))
    manifest = _read_json(
        resolve_under_root(project_root(), str(config["jacobian_corpus"]["iteration_manifest"]))
    )
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    expected = list(_predeclared_source_ids(config))
    loaded = [str(bank["source_id"]) for bank in banks]
    if loaded != expected:
        raise ValueError(
            "loaded 5DoF banks must match the pre-declared source order; "
            f"loaded={loaded} expected={expected}"
        )
    splits = {str(bank["source_id"]): str(bank["split"]) for bank in banks}
    for source_id in config["jacobian_corpus"]["train_source_ids"]:
        if splits[str(source_id)] != "train":
            raise ValueError(f"pre-declared train source {source_id} is not train")
    for source_id in config["jacobian_corpus"]["validation_source_ids"]:
        if splits[str(source_id)] != "validation":
            raise ValueError(f"pre-declared validation source {source_id} is not validation")
    attached = [
        attach_source_tracklet_slopes(
            bank,
            iteration_manifest=manifest,
            anchor_point=str(config["jacobian_corpus"]["anchor_point"]),
        )
        for bank in banks
    ]
    for bank in attached:
        if tuple(bank["names"]) != STATION_FIVE_NAMES:
            raise ValueError(
                f"native bank {bank['source_id']} has names {tuple(bank['names'])}; "
                "expected frozen station 5DoF"
            )
        if any(name in tuple(bank["names"]) for name in FORBIDDEN_TRACKER_PARAMETERS):
            raise ValueError(f"native bank {bank['source_id']} loaded a forbidden tracker parameter")
        if bank.get("held_out_physical_points_loaded"):
            raise ValueError("held-out physical points must not enter the 5DoF SVD")
        if bank.get("test_data_accessed"):
            raise ValueError("sealed test was accessed")
    return attached


def drop_seven_d_columns(subspace: IdentifiableSubspace) -> None:
    del subspace
    refuse_seven_d_column_deletion()


def _finite_fraction(array: object) -> float:
    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(values)))


def jacobian_validity_audit(
    bank: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    names = tuple(str(name) for name in bank["names"])
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    plus = np.asarray(bank["positive_residual"])
    minus = np.asarray(bank["negative_residual"])
    plus_values = np.asarray(bank["positive_values"], dtype=np.float64)
    minus_values = np.asarray(bank["negative_values"], dtype=np.float64)
    missing = []
    for index, name in enumerate(names):
        if plus.shape[0] <= index or minus.shape[0] <= index:
            missing.append(name)
            continue
        if plus[index].shape[0] != n_pairs or minus[index].shape[0] != n_pairs:
            missing.append(name)
    steps = plus_values - minus_values
    declared_steps = config["finite_difference_steps"]
    observed_steps = 0.5 * np.abs(steps)
    step_ok = True
    step_report = {}
    for index, name in enumerate(names):
        expected = float(declared_steps[name])
        observed = float(observed_steps[index])
        step_report[name] = {"declared": expected, "observed_half_plus_minus": observed}
        if not math.isfinite(observed) or abs(observed - expected) > 1.0e-9:
            step_ok = False
    derivative = np.asarray(extras["fit"].derivative_native, dtype=np.float64)
    column_norms = [
        float(np.linalg.norm(derivative[:, :, index]))
        for index in range(derivative.shape[2])
    ]
    maximum = max(column_norms) if column_norms else 0.0
    relative = float(config["jacobian_validity"]["min_column_norm_relative"])
    near_zero = [
        bool(norm <= relative * maximum) if maximum > 0.0 else True
        for norm in column_norms
    ]
    finite_ok = (
        _finite_fraction(bank["anchor_residual"]) == 1.0
        and _finite_fraction(plus) == 1.0
        and _finite_fraction(minus) == 1.0
        and _finite_fraction(bank["covariance"]) == 1.0
        and np.isfinite(derivative).all()
    )
    reasons = []
    if names != STATION_FIVE_NAMES:
        reasons.append("parameter_names_are_not_frozen_station_five")
    if missing:
        reasons.append("missing_fd_probes")
    if plus.shape[1] != n_pairs or minus.shape[1] != n_pairs:
        reasons.append("plus_minus_population_misaligned")
    if not bool(np.all(np.abs(steps) > 0.0)):
        reasons.append("plus_minus_steps_not_distinct")
    if not step_ok:
        reasons.append("finite_difference_step_mismatch")
    if not finite_ok:
        reasons.append("non_finite_residuals_or_jacobian")
    if any(near_zero):
        reasons.append("near_zero_jacobian_column")
    if n_pairs < int(config["jacobian_validity"]["min_pairs"]):
        reasons.append("too_few_pairs")
    if bank.get("held_out_physical_points_loaded"):
        reasons.append("held_out_points_mixed_into_svd")
    if bank.get("test_data_accessed"):
        reasons.append("sealed_test_accessed")
    payload = {
        "source_id": str(bank["source_id"]),
        "split": str(bank["split"]),
        "n_pairs": n_pairs,
        "parameter_names": list(names),
        "native_five_parameters": names == STATION_FIVE_NAMES,
        "plus_minus_probes_present": not missing and plus.shape[0] == len(names),
        "missing_probes": missing,
        "plus_minus_population_aligned": bool(plus.shape[1] == n_pairs and minus.shape[1] == n_pairs),
        "finite_difference_steps": step_report,
        "finite_difference_steps_match_declared": step_ok,
        "finite_residuals_and_jacobian": finite_ok,
        "column_norms": column_norms,
        "near_zero_columns": near_zero,
        "identifiable_rank": int(subspace.identifiable_rank),
        "held_out_physical_points_loaded": bool(bank.get("held_out_physical_points_loaded")),
        "test_data_accessed": bool(bank.get("test_data_accessed")),
        "reconstructed_via_only_parameters": True,
        "not_seven_d_column_deletion": True,
        "valid": not reasons,
        "failure_reasons": reasons,
    }
    return payload


def pair_event_groups(bank: Mapping[str, Any]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    runs = np.asarray(bank["run_id"])
    events = np.asarray(bank["event_id"])
    for index, (run, event) in enumerate(zip(runs.tolist(), events.tolist())):
        groups.setdefault((int(run), int(event)), []).append(int(index))
    return groups


def slope_tertile_audit(
    bank: Mapping[str, Any],
    *,
    rank_tolerance: float,
    rcond: float,
    n_bins: int,
    min_events: int,
) -> dict[str, Any]:
    slopes = np.asarray(bank["source_slope"], dtype=np.float64)
    groups = pair_event_groups(bank)
    keys = list(groups)
    event_slopes = []
    for key in keys:
        values = [float(slopes[index]) for index in groups[key] if np.isfinite(slopes[index])]
        event_slopes.append(float(np.mean(values)) if values else float("nan"))
    slope_arr = np.asarray(event_slopes, dtype=np.float64)
    finite_mask = np.isfinite(slope_arr)
    if int(np.sum(finite_mask)) < int(n_bins) * int(min_events):
        return {
            "too_small": True,
            "bins": [],
            "rank_drop": False,
            "n_finite_events": int(np.sum(finite_mask)),
            "source_slope_missing_pairs": int(bank.get("source_slope_missing_pairs", 0)),
            "slope_definition": "hypot_source_tracklet_tx_ty",
        }
    edges = np.quantile(slope_arr[finite_mask], np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[-1] = float(edges[-1]) + 1.0e-12
    reports = []
    n_pairs = int(slopes.shape[0])
    for index in range(int(n_bins)):
        selected = [
            key
            for key, slope in zip(keys, slope_arr)
            if np.isfinite(float(slope)) and float(edges[index]) <= float(slope) < float(edges[index + 1])
        ]
        mask = np.zeros(n_pairs, dtype=bool)
        for key in selected:
            mask[groups[key]] = True
        if len(selected) < int(min_events):
            reports.append(
                {
                    "label": f"slope_bin_{index}",
                    "n_events": int(len(selected)),
                    "too_small": True,
                    "identifiable_rank": None,
                    "slope_interval": [float(edges[index]), float(edges[index + 1])],
                }
            )
            continue
        subspace, extras = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=mask
        )
        reports.append(
            {
                "label": f"slope_bin_{index}",
                "n_events": int(len(selected)),
                "n_pairs": int(extras["n_pairs"]),
                "too_small": False,
                "identifiable_rank": int(subspace.identifiable_rank),
                "singular_values": [float(value) for value in subspace.singular_values],
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
        "n_finite_events": int(np.sum(finite_mask)),
        "source_slope_missing_pairs": int(bank.get("source_slope_missing_pairs", 0)),
        "slope_definition": "hypot_source_tracklet_tx_ty",
        "usable_bins": int(len(usable)),
    }


def topology_audit(
    bank: Mapping[str, Any],
    *,
    rank_tolerance: float,
    rcond: float,
    required_rank: int,
) -> dict[str, Any]:
    mask = _complete_route_mask(bank)
    n_complete = int(np.sum(mask))
    if n_complete < 1:
        return {
            "n_complete_pairs": 0,
            "too_small": True,
            "identifiable_rank": None,
            "same_rank": False,
        }
    subspace, extras = subspace_from_physical_bank(
        bank, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=mask
    )
    rank = int(subspace.identifiable_rank)
    return {
        "n_complete_pairs": n_complete,
        "n_pairs": int(extras["n_pairs"]),
        "too_small": False,
        "identifiable_rank": rank,
        "singular_values": [float(value) for value in subspace.singular_values],
        "same_rank": rank == int(required_rank),
        "topology_definition": "complete_four_station_truth_routes",
        "subspace": subspace_as_json(subspace),
        "_subspace": subspace,
    }


def diagnose_rank_loss(
    *,
    validity: Mapping[str, Any],
    rank: int,
    required_rank: int,
    tertiles: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(rank) >= int(required_rank) and not (tertiles or {}).get("rank_drop"):
        return {
            "dropped": False,
            "classification": None,
            "reason": "native_rank_meets_required_five",
        }
    if not validity.get("valid"):
        return {
            "dropped": True,
            "classification": "explicit_artifact",
            "reason": "jacobian_validity_failed",
            "failure_reasons": list(validity.get("failure_reasons") or []),
        }
    labels = []
    if int(rank) < int(required_rank):
        labels.append("native_source_rank_below_five")
    if (tertiles or {}).get("rank_drop"):
        labels.append("slope_tertile_rank_drop")
    if topology is not None and not topology.get("too_small") and not topology.get("same_rank"):
        labels.append("complete_route_topology_rank_drop")
    return {
        "dropped": True,
        "classification": "normal_physics_or_coverage_loss",
        "reason": ";".join(labels) if labels else "rank_below_five_without_artifact_flags",
        "do_not_drop_source": True,
        "do_not_retune_s_or_rank_tolerance": True,
        "do_not_reselect_population": True,
    }


def decide_campaign(
    *,
    jacobian_valid: bool,
    identifiability_pass: bool,
    three_arm: Mapping[str, Any] | None,
    pooled_rank: int,
    reasons: Sequence[str],
) -> dict[str, Any]:
    if not jacobian_valid:
        decision = DECISION_JACOBIAN_INVALID
    elif not identifiability_pass:
        decision = DECISION_IDENTIFIABILITY_FAIL
    elif three_arm is None:
        decision = DECISION_PASS
    elif not bool(three_arm.get("pass")):
        decision = DECISION_THREE_ARM_FAIL
    else:
        decision = DECISION_CLOSURE_PASS
    return {
        "decision": decision,
        "jacobian_valid": bool(jacobian_valid),
        "five_dof_identifiable_and_portable": bool(identifiability_pass),
        "pooled_identifiable_rank": int(pooled_rank),
        "pooled_rank_is_not_portability": True,
        "null_injection_leakage_gate": None if three_arm is None else bool(three_arm.get("null_injection_leakage_gate")),
        "mixed_injection_projected_closure": None
        if three_arm is None
        else bool(three_arm.get("mixed_injection_projected_closure")),
        "three_arm_authorized": False,
        "truth_selected_joint_closure_opened": three_arm is not None,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "survey_is_external_cross_check_only": True,
        "authorize_frozen_v2_unknown_association_closure": False,
        "reasons": list(reasons),
        "if_failed_do_not_chase_by_retuning_or_dropping_sources": True,
        "next_if_failed": (
            "physically_different_track_coverage_or_explicit_gauge_or_external_constraint"
        ),
        "inherited_entry_68_decision": INHERITED_ENTRY_68,
        "inherited_entry_69_decision": INHERITED_ENTRY_69,
        "inherited_entry_70_decision": INHERITED_ENTRY_70,
        "inherited_entry_71_decision": INHERITED_ENTRY_71,
        "inherited_entry_72_decision": INHERITED_ENTRY_72,
    }


def build_all_reports(
    config: Mapping[str, Any],
    *,
    banks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = project_root()
    created = datetime.now(timezone.utc).isoformat()
    if banks is None:
        banks = load_five_dof_banks(config)
    rank_tolerance = float(config["rank_tolerance"])
    rcond = float(config["normal_matrix_rcond"])
    gates = config["gates"]
    required_rank = int(gates["required_identifiable_rank"])
    coverage_cfg = config["coverage"]

    per_source = []
    source_subspaces: list[tuple[str, IdentifiableSubspace]] = []
    validity_rows = []
    tertile_rows = []
    topology_rows = []
    topology_pairs: list[tuple[str, IdentifiableSubspace]] = []
    diagnoses = []
    for bank in banks:
        subspace, extras = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond
        )
        label = f"{bank['split']}:{bank['source_id']}"
        source_subspaces.append((label, subspace))
        validity = jacobian_validity_audit(bank, config=config, subspace=subspace, extras=extras)
        tertiles = slope_tertile_audit(
            bank,
            rank_tolerance=rank_tolerance,
            rcond=rcond,
            n_bins=int(coverage_cfg["n_slope_bins"]),
            min_events=int(coverage_cfg["min_tertile_events"]),
        )
        topology = topology_audit(
            bank,
            rank_tolerance=rank_tolerance,
            rcond=rcond,
            required_rank=required_rank,
        )
        diagnosis = diagnose_rank_loss(
            validity=validity,
            rank=int(subspace.identifiable_rank),
            required_rank=required_rank,
            tertiles=tertiles,
            topology=topology,
        )
        topology_space = topology.pop("_subspace", None)
        if topology_space is not None:
            topology_pairs.append((f"{label}:complete_routes", topology_space))
        per_source.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "n_pairs": int(extras["n_pairs"]),
                "identifiable_rank": int(subspace.identifiable_rank),
                "null_dimension": int(subspace.null_dimension),
                "singular_values": [float(value) for value in subspace.singular_values],
                "subspace": subspace_as_json(subspace),
                "source_slope_missing_pairs": int(bank.get("source_slope_missing_pairs", 0)),
            }
        )
        validity_rows.append(validity)
        tertile_rows.append({"source_id": str(bank["source_id"]), "split": str(bank["split"]), **tertiles})
        topology_rows.append({"source_id": str(bank["source_id"]), "split": str(bank["split"]), **topology})
        diagnoses.append({"source_id": str(bank["source_id"]), "split": str(bank["split"]), **diagnosis})

    pooled = _pool_campaign_banks(list(banks))
    pooled["source_id"] = "pooled_train_validation"
    pooled_subspace, pooled_extras = subspace_from_physical_bank(
        pooled, rank_tolerance=rank_tolerance, rcond=rcond
    )
    source_stability = summarize_stability(
        pooled_subspace,
        source_subspaces,
        max_principal_angle_deg=float(gates["max_source_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_source_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    train_banks = [bank for bank in banks if str(bank["split"]) == "train"]
    validation_banks = [bank for bank in banks if str(bank["split"]) == "validation"]
    train_subspace, _ = subspace_from_physical_bank(
        _pool_campaign_banks(train_banks), rank_tolerance=rank_tolerance, rcond=rcond
    )
    validation_subspace, _ = subspace_from_physical_bank(
        _pool_campaign_banks(validation_banks), rank_tolerance=rank_tolerance, rcond=rcond
    )
    split_stability = summarize_stability(
        train_subspace,
        [("train_vs_validation", validation_subspace)],
        max_principal_angle_deg=float(gates["max_train_validation_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_train_validation_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    source_disjoint_rows = []
    for index, (label, subspace) in enumerate(source_subspaces):
        others = [item for item in source_subspaces if item[0] != label]
        if not others:
            continue
        remaining = _pool_campaign_banks([bank for bank in banks if f"{bank['split']}:{bank['source_id']}" != label])
        remaining_subspace, _ = subspace_from_physical_bank(
            remaining, rank_tolerance=rank_tolerance, rcond=rcond
        )
        report = summarize_stability(
            remaining_subspace,
            [(label, subspace)],
            max_principal_angle_deg=float(gates["max_source_disjoint_principal_angle_deg"]),
            max_projector_frobenius=float(gates["max_source_disjoint_projector_frobenius"]),
            require_same_rank=bool(gates["require_same_identifiable_rank"]),
        )
        source_disjoint_rows.append({"left_out": label, **report})
    source_disjoint_stable = bool(source_disjoint_rows) and all(
        bool(row["stable"]) for row in source_disjoint_rows
    )
    topology_stability = summarize_stability(
        pooled_subspace,
        topology_pairs,
        max_principal_angle_deg=float(gates["max_topology_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_topology_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    bootstrap_cfg = config["bootstrap"]
    bootstrap_spaces = bootstrap_subspaces(
        pooled,
        n_replicates=int(bootstrap_cfg["n_event_replicates"]),
        seed=int(bootstrap_cfg["seed"]),
        rank_tolerance=rank_tolerance,
        rcond=rcond,
        min_pairs=int(bootstrap_cfg["min_pairs"]),
    )
    bootstrap_stability = summarize_stability(
        pooled_subspace,
        [(f"bootstrap_{index}", space) for index, space in enumerate(bootstrap_spaces)],
        max_principal_angle_deg=float(gates["max_bootstrap_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_bootstrap_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    half_spaces = half_split_subspaces(
        pooled,
        n_splits=int(bootstrap_cfg["n_half_splits"]),
        seed=int(bootstrap_cfg["seed"]),
        rank_tolerance=rank_tolerance,
        rcond=rcond,
        min_pairs=int(bootstrap_cfg["min_pairs"]),
    )
    half_stability = summarize_stability(
        pooled_subspace,
        [(f"half_split_{index}", left) for index, (left, _right) in enumerate(half_spaces)]
        + [(f"half_split_{index}_b", right) for index, (_left, right) in enumerate(half_spaces)],
        max_principal_angle_deg=float(gates["max_bootstrap_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_bootstrap_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )

    jacobian_valid = all(bool(row["valid"]) for row in validity_rows)
    every_source_rank = all(int(row["identifiable_rank"]) == required_rank for row in per_source)
    usable_tertiles = [
        bin_row
        for row in tertile_rows
        for bin_row in row.get("bins") or []
        if not bin_row.get("too_small") and bin_row.get("identifiable_rank") is not None
    ]
    tertile_rank_ok = bool(usable_tertiles) and all(
        int(bin_row["identifiable_rank"]) == required_rank for bin_row in usable_tertiles
    )
    tertile_drop = any(bool(row.get("rank_drop")) for row in tertile_rows)
    topology_rank_ok = all(
        bool(row.get("too_small")) or int(row.get("identifiable_rank") or -1) == required_rank
        for row in topology_rows
    )
    reasons = []
    if not jacobian_valid:
        reasons.append("jacobian_validity_failed")
    if int(pooled_subspace.identifiable_rank) != required_rank:
        reasons.append("pooled_identifiable_rank_is_not_five")
    if not every_source_rank:
        reasons.append("predeclared_source_rank_is_not_five")
    if not source_stability["stable"]:
        reasons.append("source_subspace_unstable")
    if not split_stability["stable"]:
        reasons.append("train_validation_subspace_unstable")
    if not source_disjoint_stable:
        reasons.append("source_disjoint_subspace_unstable")
    if not topology_rank_ok or not topology_stability["stable"]:
        reasons.append("topology_coverage_rank_or_subspace_failed")
    if tertile_drop or not tertile_rank_ok:
        reasons.append("slope_tertile_rank_drop_or_not_five")
    if not bootstrap_stability["stable"]:
        reasons.append("event_bootstrap_unstable")
    if not half_stability["stable"]:
        reasons.append("event_half_split_unstable")
    identifiability_pass = not reasons
    three_arm = None
    if identifiability_pass and jacobian_valid:
        # Identifiability passed.  First-stage truth-selected joint closure
        # is authorized by this gate, but remains a separate payload and is
        # still not Frozen-V2 unknown-association.
        injections = three_arm_payloads(
            train_subspace,
            identifiable_scaled_amplitude=float(config["injections"]["identifiable_scaled_amplitude"]),
            null_scaled_amplitude=float(config["injections"]["null_scaled_amplitude"]),
            seed=int(config["injections"]["seed"]),
            n_replicates=int(config["injections"]["n_replicates"]),
        )
        _val_subspace, val_extras = subspace_from_physical_bank(
            _pool_campaign_banks(validation_banks), rank_tolerance=rank_tolerance, rcond=rcond
        )
        three_arm = evaluate_three_arms(
            train_subspace,
            val_extras["weighted_matrix"],
            injections,
            gates=gates,
        )
        three_arm["injections_are_linear_on_validated_jacobian"] = True
        three_arm["athena_not_rerun"] = True
        three_arm["association_control"] = "truth_selected_physical_edge"
        three_arm["definition_jacobian"] = "train_truth_selected_five_dof_jacobian"
        three_arm["generating_jacobian"] = "validation_truth_selected_five_dof_jacobian"
        three_arm["frozen_v2_unknown_association_not_opened"] = True
        three_arm["truth_used_only_as_mc_control"] = True
        if not bool(three_arm.get("pass")):
            reasons.append("truth_selected_joint_closure_failed")
    decision = decide_campaign(
        jacobian_valid=jacobian_valid,
        identifiability_pass=identifiability_pass,
        three_arm=three_arm,
        pooled_rank=int(pooled_subspace.identifiable_rank),
        reasons=reasons,
    )
    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": _operating_state(),
            "parameter_names": list(STATION_FIVE_NAMES),
            "parameter_units": list(frozen_units_for(STATION_FIVE_NAMES)),
            "scale_matrix_S": [float(value) for value in frozen_scales_for(STATION_FIVE_NAMES)],
            "excluded_from_tracker_fit": config["excluded_from_tracker_fit"],
            "fixed_internal_geometry": config["fixed_internal_geometry"],
            "finite_difference_steps": config["finite_difference_steps"],
            "scale_convention": "frozen_severity_scale_u_equals_theta_over_S",
            "residual_weight": "nominal_WLS_inverse_4x4_pair_covariance",
            "weighted_matrix": "A = W^{1/2} J S",
            "rank_tolerance": float(rank_tolerance),
            "rank_tolerance_source": "frozen LEAKAGE_RANK_RELATIVE_TOLERANCE = 1e-2",
            "normal_matrix_rcond": float(rcond),
            "reconstruction_method": "native_five_column_only_parameters",
            "not_seven_d_column_deletion": True,
            "cdx_fixed_is_not_a_measurement_of_zero": True,
            "dz_is_not_a_tracker_free_parameter": True,
            "jacobian_source": {
                "kind": "physical_central_finite_difference",
                "chain": "Tracker/Align -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper -> Acts mode 0",
                "iteration_manifest": str(
                    resolve_under_root(root, str(config["jacobian_corpus"]["iteration_manifest"]))
                ),
                "observation_semantics": "truth_selected_physical_edge",
                "q_over_p_mode": 0,
                "n_sources": int(len(banks)),
                "n_pairs_pooled": int(pooled_extras["n_pairs"]),
                "train_source_ids": list(config["jacobian_corpus"]["train_source_ids"]),
                "validation_source_ids": list(config["jacobian_corpus"]["validation_source_ids"]),
            },
            "forbidden": {
                "naked_mixed_unit_svd": True,
                "scale_retune_from_singular_values": True,
                "rank_threshold_retune_from_spectrum": True,
                "seven_d_column_deletion": True,
                "drop_failed_sources": True,
                "full_parameter_newton": True,
                "real_data_alignment_correction": True,
                "geometry_write": True,
            },
            "assumptions": list(config.get("assumptions", [])),
        },
        "jacobian_validity": {
            "all_valid": jacobian_valid,
            "per_source": validity_rows,
        },
        "identifiable_basis": {
            "pooled": subspace_as_json(pooled_subspace),
            "identifiable_rank": int(pooled_subspace.identifiable_rank),
            "null_dimension": int(pooled_subspace.null_dimension),
            "fit_normal_matrix_rank": int(pooled_extras["fit_normal_matrix_rank"]),
            "per_source": per_source,
            "modes_are_reconstruction_observable_linear_combinations": True,
            "not_mechanical_ry": True,
            "pooled_rank_is_not_portability": True,
        },
        "coverage": {
            "slope_definition": "hypot_source_tracklet_tx_ty",
            "topology_definition": "complete_four_station_truth_routes",
            "per_source_slope_tertiles": tertile_rows,
            "per_source_topology": topology_rows,
            "usable_tertile_bins": int(len(usable_tertiles)),
            "slope_tertile_rank_drop": tertile_drop,
            "every_usable_tertile_rank_five": tertile_rank_ok,
            "every_topology_rank_five_or_too_small": topology_rank_ok,
        },
        "subspace_stability": {
            "source_disjoint_vs_pooled": source_stability,
            "leave_one_source_out": source_disjoint_rows,
            "leave_one_source_out_stable": source_disjoint_stable,
            "train_validation": split_stability,
            "complete_truth_route_topology": topology_stability,
            "event_bootstrap": {
                **bootstrap_stability,
                "n_replicates_requested": int(bootstrap_cfg["n_event_replicates"]),
                "n_replicates_kept": int(len(bootstrap_spaces)),
                "resamples_events_not_jacobian_entries_independently": True,
            },
            "event_half_split": {
                **half_stability,
                "n_splits_kept": int(len(half_spaces)),
            },
        },
        "rank_loss_diagnosis": {
            "per_source": diagnoses,
            "sources_dropped": False,
            "rank_tolerance_retuned": False,
            "scales_retuned": False,
            "population_reselected": False,
        },
        "three_arm_closure": three_arm,
        "next_stage_decision": decision,
        "linear_severity_envelope": float(LINEAR_SEVERITY_MAX),
    }
    for payload in reports.values():
        if isinstance(payload, Mapping):
            assert_no_alignment_payload(payload)
    return reports


def refuse_forbidden_operations() -> dict[str, Any]:
    try:
        svd_naked_jacobian(np.eye(5), parameter_units=("mm", "mm", "mrad", "mrad", "mrad"))
        naked = False
        naked_error = None
    except MixedUnitNakedJacobianError as error:
        naked = True
        naked_error = str(error)
    try:
        refuse_seven_d_column_deletion()
        deletion = False
        deletion_error = None
    except SevenDColumnDeletionError as error:
        deletion = True
        deletion_error = str(error)
    return {
        "refused_naked_mixed_unit_svd": naked,
        "naked_mixed_unit_error": naked_error,
        "refused_seven_d_column_deletion": deletion,
        "seven_d_column_deletion_error": deletion_error,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "three_arm_authorized": False,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
    }
