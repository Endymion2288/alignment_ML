"""Physically-distinct track-coverage identifiability feasibility V1.

Stage 1 is a residual-blind coverage inventory.  It does not construct
``A = W^{1/2} J S``, does not SVD, does not inspect rank, and does not
reverse-pick samples from residual / cosine / Jacobian singular values.

Workbooks 59-73 remain frozen.  Rank-rescue of the 7D, cluster-local,
stable-core, and rigid-station 5DoF observables is refused.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE
from alignment.noncollision_crossyear_topology import (
    inventory_eos_layout,
    parse_job_log,
    probe_xaod,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, assert_no_alignment_payload, common_operating_state
from datasets.root_loader import EventTracklets, load_events


SCHEMA_VERSION = "faser-physically-distinct-track-coverage-identifiability-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = (
    Path("configs") / "physically_distinct_track_coverage_identifiability_feasibility_v1.yaml"
)
INHERITED_ENTRY_59 = "no_portable_alternative_topology_in_current_r0022"
INHERITED_ENTRY_60 = "real_track_topology_insufficient_for_ry_cdx_separation"
INHERITED_ENTRY_61 = "external_survey_or_metrology_required_and_year_iov_parameterization_required"
INHERITED_ENTRY_68 = "tracker_only_identifiable_basis_unstable_solve_stopped"
INHERITED_ENTRY_69 = "cross_source_stable_core_independent_validation_fail"
INHERITED_ENTRY_70 = "cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign"
INHERITED_ENTRY_71 = "cluster_local_observable_not_cross_run_portable"
INHERITED_ENTRY_72 = "tracklet_independent_failure_provenance_audit_complete_sources_not_dropped"
INHERITED_ENTRY_73 = "rigid_station_five_dof_not_source_or_coverage_portable"

DECISION_OPEN_FD = "physically_distinct_population_admits_separate_fd_campaign"
DECISION_EXPORT = "residual_blind_export_authorized_fd_not_opened"
DECISION_INSUFFICIENT = "current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof"
GO_NO_GO_QUESTION = (
    "Does a physically different track-angle / origin / topology population "
    "supply the independent alignment information that current collision-like "
    "and canonical 5 mrad FLUKA-E tracks lack, so that rigid-station 5DoF "
    "becomes stably identifiable under source-disjoint and coverage-disjoint "
    "conditions?"
)

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
    "do_not_restack_canonical_5mrad_unused_files",
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
    "do_not_delete_rz_or_further_dof",
    "do_not_force_write_old_survey_central_values",
    "do_not_open_three_arm_unless_identifiability_passes",
    "do_not_open_frozen_v2_alignment_loop",
    "do_not_construct_fd_identifiability_this_stage",
    "do_not_compute_svd_or_rank_this_stage",
    "do_not_guess_branch_meaning_from_filename",
    "do_not_lower_statistical_gates",
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "do_not_open_gauge_branch_until_inventory_closes",
    "filename_5mrad_is_not_phase_space_proof",
    "this_is_not_seven_d_column_deletion",
    "this_is_not_workbook_36_survey_dz_curriculum",
    "this_is_not_workbook_59_or_60_rank_rescue",
)

REQUIRED_STATIONS = (0, 1, 2, 3)
COLLISION_LIKE_CLASSES = {
    "real_collision_like",
    "real_r0022_collision_like",
    "canonical_5mrad_flukaE_particle_gun",
}


class ResidualBlindStageError(RuntimeError):
    """Stage 1 forbids Jacobian, SVD, rank, and residual-driven selection."""


class SealedTestAccessError(RuntimeError):
    """The historical 100116/100117 sealed test must stay closed."""


class FilenameGuessError(RuntimeError):
    """Branch meaning must come from logs/ROOT/docs, not from filenames."""


def load_inventory_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"coverage inventory config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected coverage inventory schema: {source}")
    if payload.get("stage") != "residual_blind_coverage_inventory":
        raise ValueError("this campaign stage must be residual_blind_coverage_inventory")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"coverage inventory config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"coverage inventory config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("inherited_entry_73_decision") != INHERITED_ENTRY_73:
        raise ValueError("workbook 73 rigid-station decision must stay frozen")
    contract = payload.get("rigid_station_five_dof_contract") or {}
    if tuple(contract.get("parameter_names") or ()) != FREE_PARAMETERS:
        raise ValueError("rigid-station 5DoF names must stay frozen")
    if float(contract.get("rank_tolerance")) != float(FROZEN_RANK_TOLERANCE):
        raise ValueError("rank_tolerance=0.01 must stay frozen")
    scales = contract.get("scale_matrix_S") or {}
    if float(scales.get("ift_dx_mm")) != 5.0 or float(scales.get("ift_ry_mrad")) != 60.0:
        raise ValueError("frozen S = (5, 5, 60, 60, 60) must not be retuned")
    if "ift_rz_mrad" not in scales:
        raise ValueError("rz remains in the frozen 5DoF model; it is not deleted")
    if contract.get("do_not_reconstruct_this_stage") is not True:
        raise ValueError("stage 1 must not reconstruct A = W^{1/2} J S")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def common_inventory_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "residual_blind_coverage_inventory",
            "go_no_go_question": GO_NO_GO_QUESTION,
            "do_not_construct_fd_identifiability_this_stage": True,
            "do_not_compute_svd_or_rank_this_stage": True,
            "do_not_select_events_from_residual_or_cosine": True,
            "do_not_restack_2024_r0022_collision_like": True,
            "do_not_restack_canonical_5mrad_unused_files": True,
            "do_not_open_sealed_test": True,
            "do_not_rescue_seven_d_observable": True,
            "do_not_rescue_cluster_local_observable": True,
            "do_not_drop_failed_sources_to_recover_rank_five": True,
            "do_not_delete_rz_or_further_dof": True,
            "filename_5mrad_is_not_phase_space_proof": True,
            "three_arm_authorized": False,
            "frozen_v2_alignment_loop_authorized": False,
            "real_data_correction_authorized": False,
            "geometry_write_allowed": False,
            "official_conditions_write_allowed": False,
            "inherited_entry_59_decision": INHERITED_ENTRY_59,
            "inherited_entry_60_decision": INHERITED_ENTRY_60,
            "inherited_entry_68_decision": INHERITED_ENTRY_68,
            "inherited_entry_69_decision": INHERITED_ENTRY_69,
            "inherited_entry_70_decision": INHERITED_ENTRY_70,
            "inherited_entry_71_decision": INHERITED_ENTRY_71,
            "inherited_entry_72_decision": INHERITED_ENTRY_72,
            "inherited_entry_73_decision": INHERITED_ENTRY_73,
        }
    )
    return state


def refuse_fd_identifiability_this_stage() -> None:
    raise ResidualBlindStageError(
        "Stage 1 is residual-blind coverage inventory.  Native 5DoF "
        "A = W^{1/2} J S, SVD, rank, cosine, and residual selection are forbidden."
    )


def refuse_filename_guess(token: str) -> None:
    raise FilenameGuessError(
        f"do not infer phase space or topology from filename token '{token}'"
    )


def refuse_sealed_source(source_id: str, forbidden: Sequence[str]) -> None:
    if source_id in set(forbidden):
        raise SealedTestAccessError(f"sealed test source must stay closed: {source_id}")


def _distribution(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "p01": None,
        "p50": None,
        "p99": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "p01": float(np.quantile(finite, 0.01)),
                "p50": float(np.quantile(finite, 0.50)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
            }
        )
    return result


def _truth_muon_mask(event: EventTracklets, min_truth_match_fraction: float) -> np.ndarray:
    n = int(event.size)
    if event.truth_pdg is None or event.truth_match_fraction is None or event.truth_particle_id is None:
        return np.ones(n, dtype=bool)
    return (
        (np.abs(np.asarray(event.truth_pdg)) == 13)
        & (np.asarray(event.truth_match_fraction) >= float(min_truth_match_fraction))
        & (np.asarray(event.truth_particle_id) >= 0)
    )


def summarize_events(
    events: Sequence[EventTracklets],
    *,
    source_id: str,
    physics: Mapping[str, Any],
    phase_space: Mapping[str, Any],
) -> dict[str, Any]:
    """Residual-blind station and (tx, ty) coverage for one tracklet file."""
    preferred = int(phase_space.get("preferred_station") or 0)
    min_frac = float(phase_space.get("min_truth_match_fraction") or 0.99)
    wide_cut = float(physics["wide_angle_min_slope"])
    ip_cut = float(physics["ip_like_max_slope"])
    n_events = int(len(events))
    station_event_counts = {str(station): 0 for station in REQUIRED_STATIONS}
    complete = 0
    ift_events = 0
    tx_values: list[float] = []
    ty_values: list[float] = []
    slope_values: list[float] = []
    azimuth_values: list[float] = []
    n_tracklets = 0
    n_selected_tracklets = 0
    for event in events:
        n_tracklets += int(event.size)
        mask = _truth_muon_mask(event, min_frac)
        stations = set(int(value) for value in np.unique(event.station_id))
        for station in REQUIRED_STATIONS:
            if station in stations:
                station_event_counts[str(station)] += 1
        if 0 in stations:
            ift_events += 1
        if set(REQUIRED_STATIONS).issubset(stations):
            complete += 1
        selected = mask
        if preferred in stations:
            selected = selected & (event.station_id == preferred)
        elif np.any(selected):
            first = int(event.station_id[np.flatnonzero(selected)[0]])
            selected = selected & (event.station_id == first)
        chosen = np.flatnonzero(selected)
        if chosen.size == 0:
            continue
        index = int(chosen[0])
        tx = float(event.state[index, 2])
        ty = float(event.state[index, 3])
        if not math.isfinite(tx) or not math.isfinite(ty):
            continue
        n_selected_tracklets += 1
        tx_values.append(tx)
        ty_values.append(ty)
        slope = float(math.hypot(tx, ty))
        slope_values.append(slope)
        azimuth_values.append(float(math.atan2(ty, tx)))
    tx_arr = np.asarray(tx_values, dtype=np.float64)
    ty_arr = np.asarray(ty_values, dtype=np.float64)
    slope_arr = np.asarray(slope_values, dtype=np.float64)
    n_slope = int(slope_arr.size)
    wide_n = int(np.count_nonzero(slope_arr >= wide_cut)) if n_slope else 0
    ip_n = int(np.count_nonzero(slope_arr < ip_cut)) if n_slope else 0
    return {
        "source_id": source_id,
        "n_events": n_events,
        "n_tracklets": int(n_tracklets),
        "n_angular_tracklets": n_selected_tracklets,
        "station_event_counts": station_event_counts,
        "n_ift_events": int(ift_events),
        "n_complete_four_station_events": int(complete),
        "stations_present": [
            station for station in REQUIRED_STATIONS if station_event_counts[str(station)] > 0
        ],
        "tx": _distribution(tx_arr),
        "ty": _distribution(ty_arr),
        "slope": _distribution(slope_arr),
        "azimuth": _distribution(np.asarray(azimuth_values, dtype=np.float64)),
        "wide_local_slope_fraction": (wide_n / n_slope) if n_slope else None,
        "ip_like_local_slope_fraction": (ip_n / n_slope) if n_slope else None,
        "slope_definition": "hypot_source_tracklet_tx_ty",
        "slope_source": "canonical_tracklet_local_state",
        "slope_is_not_leftover_residual_rtx_rty": True,
        "slope_is_not_spectrometer_delta_x_over_delta_z": True,
        "residual_blind": True,
        "tx_values": tx_arr,
        "ty_values": ty_arr,
    }


def _empty_source_summary(source_id: str, reason: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "n_events": 0,
        "n_tracklets": 0,
        "n_angular_tracklets": 0,
        "station_event_counts": {str(station): 0 for station in REQUIRED_STATIONS},
        "n_ift_events": 0,
        "n_complete_four_station_events": 0,
        "stations_present": [],
        "tx": _distribution(np.asarray([], dtype=np.float64)),
        "ty": _distribution(np.asarray([], dtype=np.float64)),
        "slope": _distribution(np.asarray([], dtype=np.float64)),
        "azimuth": _distribution(np.asarray([], dtype=np.float64)),
        "wide_local_slope_fraction": None,
        "ip_like_local_slope_fraction": None,
        "residual_blind": True,
        "tx_values": np.asarray([], dtype=np.float64),
        "ty_values": np.asarray([], dtype=np.float64),
        "missing_reason": reason,
    }


def load_source_coverage(
    source: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    sealed = (config.get("sealed_test") or {}).get("forbidden_source_ids") or []
    source_id = str(source["id"])
    refuse_sealed_source(source_id, [str(item) for item in sealed])
    relative = source.get("tracklets")
    if not relative:
        summary = _empty_source_summary(source_id, "no_tracklets_declared")
        summary["split"] = source.get("split")
        return summary
    path = resolve_under_root(root, str(relative))
    if not path.is_file():
        summary = _empty_source_summary(source_id, "tracklets_file_missing")
        summary["split"] = source.get("split")
        summary["tracklets"] = str(path)
        return summary
    events = load_events(path)
    summary = summarize_events(
        events,
        source_id=source_id,
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
    )
    summary["split"] = source.get("split")
    summary["tracklets"] = str(path)
    return summary


def _concat_xy(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    tx = np.concatenate(
        [np.asarray(row.get("tx_values"), dtype=np.float64).reshape(-1) for row in rows]
        or [np.asarray([], dtype=np.float64)]
    )
    ty = np.concatenate(
        [np.asarray(row.get("ty_values"), dtype=np.float64).reshape(-1) for row in rows]
        or [np.asarray([], dtype=np.float64)]
    )
    return tx, ty


def _quantile_box(tx: np.ndarray, ty: np.ndarray, lo: float, hi: float) -> dict[str, float | None]:
    finite = np.isfinite(tx) & np.isfinite(ty)
    if int(np.count_nonzero(finite)) == 0:
        return {"tx_lo": None, "tx_hi": None, "ty_lo": None, "ty_hi": None, "n": 0}
    return {
        "tx_lo": float(np.quantile(tx[finite], lo)),
        "tx_hi": float(np.quantile(tx[finite], hi)),
        "ty_lo": float(np.quantile(ty[finite], lo)),
        "ty_hi": float(np.quantile(ty[finite], hi)),
        "n": int(np.count_nonzero(finite)),
    }


def _histogram_2d(
    tx: np.ndarray,
    ty: np.ndarray,
    *,
    range_tx: Sequence[float],
    range_ty: Sequence[float],
    bins: int,
) -> np.ndarray:
    finite = np.isfinite(tx) & np.isfinite(ty)
    if int(np.count_nonzero(finite)) == 0:
        return np.zeros((int(bins), int(bins)), dtype=np.float64)
    hist, _, _ = np.histogram2d(
        tx[finite],
        ty[finite],
        bins=int(bins),
        range=[(float(range_tx[0]), float(range_tx[1])), (float(range_ty[0]), float(range_ty[1]))],
    )
    total = float(np.sum(hist))
    if total <= 0.0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist.astype(np.float64) / total


def histogram_intersection(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    if float(np.sum(left)) <= 0.0 or float(np.sum(right)) <= 0.0:
        return None
    return float(np.sum(np.minimum(left, right)))


def outside_envelope_fraction(
    tx: np.ndarray,
    ty: np.ndarray,
    box: Mapping[str, Any],
) -> float | None:
    if box.get("tx_lo") is None or box.get("n") in {None, 0}:
        return None
    finite = np.isfinite(tx) & np.isfinite(ty)
    if int(np.count_nonzero(finite)) == 0:
        return None
    inside = (
        (tx[finite] >= float(box["tx_lo"]))
        & (tx[finite] <= float(box["tx_hi"]))
        & (ty[finite] >= float(box["ty_lo"]))
        & (ty[finite] <= float(box["ty_hi"]))
    )
    return float(1.0 - np.mean(inside))


def pool_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    phase = config["phase_space"]
    physics = config["physics_scales"]
    tx, ty = _concat_xy(rows)
    n_events = int(sum(int(row.get("n_events") or 0) for row in rows))
    n_ift = int(sum(int(row.get("n_ift_events") or 0) for row in rows))
    n_complete = int(sum(int(row.get("n_complete_four_station_events") or 0) for row in rows))
    n_tracklets = int(sum(int(row.get("n_tracklets") or 0) for row in rows))
    station_event_counts = {str(station): 0 for station in REQUIRED_STATIONS}
    for row in rows:
        counts = row.get("station_event_counts") or {}
        for station in REQUIRED_STATIONS:
            station_event_counts[str(station)] += int(counts.get(str(station)) or 0)
    hist = _histogram_2d(
        tx,
        ty,
        range_tx=phase["histogram_range_tx"],
        range_ty=phase["histogram_range_ty"],
        bins=int(phase["histogram_bins"]),
    )
    quantiles = list(phase.get("envelope_quantiles") or [0.01, 0.99])
    box = _quantile_box(tx, ty, float(quantiles[0]), float(quantiles[1]))
    slope = np.hypot(tx, ty) if tx.size else np.asarray([], dtype=np.float64)
    azimuth = np.arctan2(ty, tx) if tx.size else np.asarray([], dtype=np.float64)
    n_slope = int(slope.size)
    wide_cut = float(physics["wide_angle_min_slope"])
    ip_cut = float(physics["ip_like_max_slope"])
    payload = {
        "id": population_id,
        "n_sources": int(len(rows)),
        "source_ids": [row.get("source_id") for row in rows],
        "n_events": n_events,
        "n_tracklets": n_tracklets,
        "n_angular_tracklets": int(tx.size),
        "n_ift_events": n_ift,
        "n_complete_four_station_events": n_complete,
        "station_event_counts": station_event_counts,
        "stations_present": [
            station for station in REQUIRED_STATIONS if station_event_counts[str(station)] > 0
        ],
        "tx": _distribution(tx),
        "ty": _distribution(ty),
        "slope": _distribution(slope),
        "azimuth": _distribution(azimuth),
        "wide_local_slope_fraction": (
            float(np.count_nonzero(slope >= wide_cut) / n_slope) if n_slope else None
        ),
        "ip_like_local_slope_fraction": (
            float(np.count_nonzero(slope < ip_cut) / n_slope) if n_slope else None
        ),
        "quantile_box": box,
        "histogram_intersection_self": 1.0 if n_slope else None,
        "residual_blind": True,
        "coverage_measured": bool(n_events > 0 and n_tracklets > 0),
        "_histogram": hist,
        "_tx": tx,
        "_ty": ty,
    }
    return payload


def overlap_with_canonical(
    candidate: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> dict[str, Any]:
    hist_c = candidate.get("_histogram")
    hist_k = canonical.get("_histogram")
    intersection = None
    if isinstance(hist_c, np.ndarray) and isinstance(hist_k, np.ndarray):
        intersection = histogram_intersection(hist_c, hist_k)
    outside = outside_envelope_fraction(
        np.asarray(candidate.get("_tx"), dtype=np.float64),
        np.asarray(candidate.get("_ty"), dtype=np.float64),
        canonical.get("quantile_box") or {},
    )
    return {
        "histogram_intersection": intersection,
        "outside_canonical_quantile_box_fraction": outside,
        "canonical_quantile_box": dict(canonical.get("quantile_box") or {}),
        "coordinate_system": "canonical_tracklet_local_tx_ty",
        "filename_5mrad_is_not_phase_space_proof": True,
        "residual_blind": True,
    }


def _public_population(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def inherited_real_data_inventory(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inherited = config.get("inherited_inventories") or {}
    rows = {}
    for key, spec in inherited.items():
        block = {"id": key, "decision": spec.get("decision")}
        for field in ("inventory", "next_stage", "data_source"):
            relative = spec.get(field)
            if not relative:
                continue
            path = resolve_under_root(root, str(relative))
            payload = load_json_if_present(path)
            block[field] = {
                "path": str(path),
                "exists": path.is_file(),
                "decision": None if payload is None else payload.get("decision"),
                "answer": None if payload is None else payload.get("answer"),
            }
            if payload is not None and field == "inventory" and key == "workbook_59":
                block["n_selected_routes"] = payload.get("n_selected_routes")
                block["n_complete_four_station_selected_routes"] = payload.get(
                    "n_complete_four_station_selected_routes"
                )
            if payload is not None and field == "next_stage":
                block["inherited_decision"] = payload.get("decision")
        rows[key] = block
    return {
        **common_inventory_state(),
        "note": (
            "Workbook 59/60 inventories are inherited residual-blind evidence. "
            "They are not reopened as rank rescue of ry↔C_dx or rigid-station 5DoF."
        ),
        "workbook_59_decision": INHERITED_ENTRY_59,
        "workbook_60_decision": INHERITED_ENTRY_60,
        "do_not_restack_2024_r0022_collision_like": True,
        "entries": rows,
    }


def admit_candidate(
    candidate: Mapping[str, Any],
    *,
    pooled: Mapping[str, Any],
    overlap: Mapping[str, Any] | None,
    gates: Mapping[str, Any],
    xaod: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    metadata_class = str(candidate.get("metadata_class") or "")
    n_events = int(pooled.get("n_events") or 0)
    n_ift = int(pooled.get("n_ift_events") or 0)
    n_complete = int(pooled.get("n_complete_four_station_events") or 0)
    n_sources = int(pooled.get("n_sources") or 0)
    coverage_measured = bool(pooled.get("coverage_measured"))
    stations = set(int(value) for value in (pooled.get("stations_present") or []))
    min_events = int(gates["min_events"])
    min_ift = int(gates["min_ift_events"])
    min_complete = int(gates["min_complete_four_station_events"])
    min_sources = int(gates["min_independent_sources_or_runs"])
    min_outside = float(gates["min_outside_canonical_envelope_fraction"])
    max_intersection = float(gates["max_histogram_intersection_for_distinct"])

    if candidate.get("same_production_as_canonical") is True:
        reasons.append("same production as the canonical 5 mrad FLUKA-E hierarchical V1 corpus")
    if metadata_class in COLLISION_LIKE_CLASSES and bool(gates.get("collision_like_metadata_cannot_admit")):
        reasons.append("collision-like metadata cannot admit a new-observable FD campaign")
    if candidate.get("sealed") is True:
        reasons.append("sealed test population")
    if candidate.get("do_not_restack") is True:
        reasons.append("predeclared do_not_restack; workbook 59/60 already froze this topology")
    if candidate.get("inherit_workbook") in {59, 60} and not coverage_measured:
        reasons.append(
            f"workbook {candidate.get('inherit_workbook')} already classified this real-data "
            "topology as insufficient; not restacked"
        )

    hypothesis = bool(candidate.get("physically_distinct_hypothesis"))
    outside = None if overlap is None else overlap.get("outside_canonical_quantile_box_fraction")
    intersection = None if overlap is None else overlap.get("histogram_intersection")
    wide_frac = pooled.get("wide_local_slope_fraction")
    # Local SegmentFit hypot(tx, ty) is a station-internal angle.  It is not
    # the workbook-59 spectrometer Δx/Δz wide-angle cut, so an absolute
    # wide-fraction gate cannot declare a new population.  Distinctness is
    # the (tx, ty) overlap with the canonical envelope.
    observed_distinct = False
    if outside is not None and float(outside) >= min_outside:
        observed_distinct = True
    if intersection is not None and float(intersection) <= max_intersection:
        observed_distinct = True
    if not hypothesis and not observed_distinct:
        reasons.append("neither metadata nor observed (tx, ty) establishes a distinct population")
    if hypothesis and coverage_measured and not observed_distinct:
        reasons.append(
            "metadata hypothesis is distinct, but observed tracklet (tx, ty) still overlaps "
            "the canonical envelope"
        )

    if not coverage_measured:
        reasons.append("coverage unmeasured: no IFT+1+2+3 tracklets")
    if n_events < min_events:
        reasons.append(f"n_events {n_events} < {min_events}")
    if n_ift < min_ift:
        reasons.append(f"n_ift_events {n_ift} < {min_ift}")
    if n_complete < min_complete:
        reasons.append(f"n_complete_four_station_events {n_complete} < {min_complete}")
    if n_sources < min_sources:
        reasons.append(f"n_independent_sources {n_sources} < {min_sources}")
    if bool(gates.get("require_stations_0_1_2_3_present")) and coverage_measured:
        missing = [station for station in REQUIRED_STATIONS if station not in stations]
        if missing:
            reasons.append(f"missing stations {missing}")
    if bool(gates.get("require_ift_station_0")) and coverage_measured and 0 not in stations:
        reasons.append("IFT station 0 absent")

    stats_ok = (
        coverage_measured
        and n_events >= min_events
        and n_ift >= min_ift
        and n_complete >= min_complete
        and n_sources >= min_sources
        and set(REQUIRED_STATIONS).issubset(stations)
    )
    metadata_forbid = bool(
        candidate.get("same_production_as_canonical") is True
        or (
            metadata_class in COLLISION_LIKE_CLASSES
            and bool(gates.get("collision_like_metadata_cannot_admit"))
        )
        or candidate.get("sealed") is True
        or candidate.get("do_not_restack") is True
        or candidate.get("known_empty_segmentfit") is True
        or (candidate.get("inherit_workbook") in {59, 60} and not coverage_measured)
    )
    if candidate.get("known_empty_segmentfit") is True:
        reasons.append("workbook/data audit already found empty SegmentFit/Segments")
    admitted = bool(hypothesis and observed_distinct and stats_ok and not metadata_forbid)

    export_reasons: list[str] = []
    xaod_ok = False
    if xaod and xaod.get("exists"):
        collections = xaod.get("collection_keys_of_interest") or {}
        n_xaod_events = int(xaod.get("n_events") or 0)
        if not collections.get("SegmentFit"):
            export_reasons.append("representative xAOD lacks SegmentFit")
        if n_xaod_events < int(gates.get("min_xaod_events_per_file_for_export") or 1000):
            export_reasons.append(
                f"representative xAOD events {n_xaod_events} < "
                f"{gates.get('min_xaod_events_per_file_for_export')}"
            )
        xaod_ok = not export_reasons
    authorize_export = bool(
        hypothesis
        and not coverage_measured
        and not metadata_forbid
        and xaod_ok
        and candidate.get("inherit_workbook") not in {59, 60}
        and candidate.get("do_not_restack") is not True
        and candidate.get("known_empty_segmentfit") is not True
    )
    if authorize_export:
        export_reasons.append("metadata-distinct xAOD may be exported residual-blind on HTCondor")
    elif not coverage_measured and not authorize_export:
        if "coverage unmeasured: no IFT+1+2+3 tracklets" not in export_reasons:
            export_reasons.append("coverage unmeasured: no IFT+1+2+3 tracklets")

    verdict = "admitted_to_separate_fd_campaign" if admitted else "insufficient"
    if authorize_export and not admitted:
        verdict = "insufficient_authorize_residual_blind_export"
    return {
        "id": candidate.get("id"),
        "metadata_class": metadata_class,
        "verdict": verdict,
        "admitted_to_separate_fd_campaign": admitted,
        "authorize_residual_blind_export": authorize_export,
        "physically_distinct_hypothesis": hypothesis,
        "observed_phase_space_distinct": observed_distinct,
        "coverage_measured": coverage_measured,
        "n_events": n_events,
        "n_ift_events": n_ift,
        "n_complete_four_station_events": n_complete,
        "n_independent_sources": n_sources,
        "outside_canonical_quantile_box_fraction": outside,
        "histogram_intersection": intersection,
        "wide_local_slope_fraction": wide_frac,
        "gates_not_lowered": True,
        "reason": "; ".join(reasons) if reasons else "admitted",
        "export_reason": "; ".join(export_reasons) if export_reasons else None,
    }


def decide_next_stage(
    admissions: Sequence[Mapping[str, Any]],
    *,
    fd_executed: bool = False,
) -> dict[str, Any]:
    if fd_executed:
        refuse_fd_identifiability_this_stage()
    admitted = [row for row in admissions if row.get("admitted_to_separate_fd_campaign")]
    exportable = [row for row in admissions if row.get("authorize_residual_blind_export")]
    if admitted:
        decision = DECISION_OPEN_FD
        answer = "Coverage-sufficient physically distinct population found; FD not executed in this stage."
        next_step = "separate_native_five_dof_fd_identifiability_campaign_source_disjoint"
        freeze_insufficient = False
        open_gauge = False
    elif exportable:
        decision = DECISION_EXPORT
        answer = (
            "No already-exported population meets the frozen statistical gates. "
            "A metadata-distinct xAOD may be exported residual-blind; that is not rank rescue."
        )
        next_step = "htcondor_residual_blind_tracklet_export_then_repeat_coverage_inventory"
        freeze_insufficient = False
        open_gauge = False
    else:
        decision = DECISION_INSUFFICIENT
        answer = "No"
        next_step = "gauge_constrained_or_external_constraint_alignment_feasibility"
        freeze_insufficient = True
        open_gauge = True
    return {
        **common_inventory_state(),
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": answer if decision != DECISION_INSUFFICIENT else "No",
        "decision": decision,
        "admitted_populations": [row.get("id") for row in admitted],
        "export_authorized_populations": [row.get("id") for row in exportable],
        "fd_identifiability_executed": False,
        "svd_or_rank_computed": False,
        "freeze_current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof": freeze_insufficient,
        "open_gauge_constrained_or_external_constraint_branch": open_gauge,
        "gauge_convention_is_not_a_physical_measurement": True,
        "survey_may_enter_only_as_independent_external_constraint": True,
        "do_not_treat_population_spread_as_prior": True,
        "three_arm_authorized": False,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "next_allowed_step": next_step,
        "reason": (
            "A later native 5DoF FD campaign is authorized only for populations that "
            "are metadata-distinct, observed-(tx,ty)-distinct, and above frozen "
            "statistical gates.  Mixed pooled rank is not complementarity.  If no "
            "such population exists in available reconstructions, tracker-only "
            "unconstrained rigid-station 5DoF is frozen and the next branch is "
            "gauge / external constraint."
        ),
    }


def _strip_arrays(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("tx_values", None)
    payload.pop("ty_values", None)
    return payload


def build_all_reports(
    config: Mapping[str, Any],
    *,
    source_summaries: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    probe_unmeasured_xaod: bool = True,
) -> dict[str, Any]:
    root = project_root()
    created = datetime.now(timezone.utc).isoformat()
    sealed = [str(item) for item in ((config.get("sealed_test") or {}).get("forbidden_source_ids") or [])]
    declared_source_ids = [
        str(source["id"]) for source in config["canonical_population"]["sources"]
    ]
    for candidate in config["predeclared_candidates"]:
        declared_source_ids.extend(str(source["id"]) for source in (candidate.get("sources") or []))
    for source_id in declared_source_ids:
        refuse_sealed_source(source_id, sealed)

    if source_summaries is None:
        canonical_rows = [
            load_source_coverage(source, config=config, root=root)
            for source in config["canonical_population"]["sources"]
        ]
        candidate_rows: dict[str, list[dict[str, Any]]] = {}
        for candidate in config["predeclared_candidates"]:
            candidate_rows[str(candidate["id"])] = [
                load_source_coverage(source, config=config, root=root)
                for source in (candidate.get("sources") or [])
            ]
    else:
        canonical_rows = list(source_summaries["canonical"])
        candidate_rows = {
            key: list(value) for key, value in source_summaries.items() if key != "canonical"
        }

    canonical_pooled = pool_population(
        canonical_rows, population_id=str(config["canonical_population"]["id"]), config=config
    )
    xaod_probes = []
    candidate_reports = []
    admissions = []
    for candidate in config["predeclared_candidates"]:
        cid = str(candidate["id"])
        rows = candidate_rows.get(cid) or []
        pooled = pool_population(rows, population_id=cid, config=config)
        overlap = overlap_with_canonical(pooled, canonical_pooled)
        xaod_report = None
        xaod_path = candidate.get("representative_xaod")
        need_xaod = (
            probe_unmeasured_xaod
            and (not bool(pooled.get("coverage_measured")))
            and bool(xaod_path)
        )
        if need_xaod:
            xaod_report = probe_xaod(xaod_path)
            xaod_report["id"] = cid
            xaod_probes.append(
                {
                    "id": cid,
                    "path": xaod_path,
                    "exists": xaod_report.get("exists"),
                    "n_events": xaod_report.get("n_events"),
                    "collection_keys_of_interest": xaod_report.get("collection_keys_of_interest"),
                    "usable_as_cluster_local_jacobian_input": xaod_report.get(
                        "usable_as_cluster_local_jacobian_input"
                    ),
                }
            )
        admission = admit_candidate(
            candidate,
            pooled=pooled,
            overlap=overlap,
            gates=config["coverage_gates"],
            xaod=xaod_report,
        )
        admissions.append(admission)
        log_path = candidate.get("representative_log")
        log_report = parse_job_log(log_path) if log_path else None
        candidate_reports.append(
            {
                "id": cid,
                "metadata_class": candidate.get("metadata_class"),
                "declared_from": candidate.get("declared_from"),
                "expected_difference": candidate.get("expected_difference"),
                "physically_distinct_hypothesis": candidate.get("physically_distinct_hypothesis"),
                "same_production_as_canonical": candidate.get("same_production_as_canonical"),
                "inherit_workbook": candidate.get("inherit_workbook"),
                "sources": [_strip_arrays(row) for row in rows],
                "pooled": _public_population(pooled),
                "overlap_with_canonical": overlap,
                "admission": admission,
                "representative_log": None if log_report is None else {
                    "path": log_report.get("path"),
                    "exists": log_report.get("exists"),
                    "geometry_tag": log_report.get("geometry_tag"),
                    "conditions_tag": log_report.get("conditions_tag"),
                    "geom_flag": log_report.get("geom_flag"),
                    "cosmics_only": log_report.get("cosmics_only"),
                },
                "residual_blind": True,
            }
        )

    selection_contract = {
        **common_inventory_state(),
        "created_utc": created,
        "git_head": git_head_sha(root),
        "frozen_before_residual_or_rank": True,
        "selection_forbidden": [
            "alignment_residual",
            "jacobian_singular_value",
            "cosine",
            "alignment_response",
            "final_rank",
            "filename_5mrad_as_phase_space",
        ],
        "allowed_selection": [
            "data_provenance",
            "run_type",
            "generator_process",
            "detector_configuration",
            "track_angular_phase_space_from_canonical_tracklets",
            "station_coverage",
        ],
        "coverage_gates": dict(config["coverage_gates"]),
        "rigid_station_five_dof_contract": dict(config["rigid_station_five_dof_contract"]),
        "do_not_lower_statistical_gates": True,
        "filename_5mrad_is_not_phase_space_proof": True,
        "predeclared_candidate_ids": [row["id"] for row in config["predeclared_candidates"]],
    }
    inherited = inherited_real_data_inventory(config, root)
    layout = inventory_eos_layout(
        {
            "eos": {
                "rec_root": config["eos"]["rec_root"],
                "data0_rec_root": config["eos"]["data0_rec_root"],
                "phys_root": config["eos"]["phys_root"],
                "years": config["eos"]["years"],
                "extra_rec_trees": config["eos"]["extra_rec_trees"],
            }
        }
    )
    decision = decide_next_stage(admissions, fd_executed=False)
    reports = {
        "selection_contract": selection_contract,
        "canonical_coverage": {
            **common_inventory_state(),
            "created_utc": created,
            "population": dict(config["canonical_population"]),
            "sources": [_strip_arrays(row) for row in canonical_rows],
            "pooled": _public_population(canonical_pooled),
            "unused_same_production_files_are_not_a_new_population": True,
        },
        "candidate_coverage": {
            **common_inventory_state(),
            "created_utc": created,
            "candidates": candidate_reports,
        },
        "inherited_real_data": inherited,
        "eos_layout": {
            **common_inventory_state(),
            "created_utc": created,
            "layout": layout,
            "xaod_probes": xaod_probes,
        },
        "admission": {
            **common_inventory_state(),
            "created_utc": created,
            "gates": dict(config["coverage_gates"]),
            "admissions": admissions,
            "n_admitted": int(sum(1 for row in admissions if row.get("admitted_to_separate_fd_campaign"))),
            "n_export_authorized": int(
                sum(1 for row in admissions if row.get("authorize_residual_blind_export"))
            ),
        },
        "next_stage_decision": {**decision, "created_utc": created, "git_head": git_head_sha(root)},
    }
    for payload in reports.values():
        if isinstance(payload, Mapping):
            assert_no_alignment_payload(payload)
    return reports


def refuse_forbidden_operations() -> dict[str, Any]:
    fd_refused = False
    fd_error = None
    try:
        refuse_fd_identifiability_this_stage()
    except ResidualBlindStageError as error:
        fd_refused = True
        fd_error = str(error)
    sealed_refused = False
    sealed_error = None
    try:
        refuse_sealed_source("mc24_100116_00030_00039", ["mc24_100116_00030_00039"])
    except SealedTestAccessError as error:
        sealed_refused = True
        sealed_error = str(error)
    filename_refused = False
    filename_error = None
    try:
        refuse_filename_guess("5mrad")
    except FilenameGuessError as error:
        filename_refused = True
        filename_error = str(error)
    return {
        "refused_fd_identifiability_this_stage": fd_refused,
        "fd_identifiability_error": fd_error,
        "refused_sealed_test": sealed_refused,
        "sealed_test_error": sealed_error,
        "refused_filename_guess": filename_refused,
        "filename_guess_error": filename_error,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "three_arm_authorized": False,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
    }
