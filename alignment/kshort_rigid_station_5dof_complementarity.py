"""Physically-Distinct K-short Rigid-Station 5DoF FD Complementarity V1.

Workbook 76.  The campaign asks whether the physically-distinct
angular/topological support of the K-short charged-pion daughter population
provides stable, source-disjoint and coverage-disjoint NEW alignment
information complementary to the canonical population, resolving the
workbook-73 rigid-station 5DoF portability failure.  It is NOT "is K-short
alone rank 5" and NOT "is pooled canonical+K-short rank 5"; pooled rank 5
alone is never a success criterion.

Everything numerical is inherited from the frozen workbook 68/69/73
contracts: central FD steps (0.5, 0.5, 10, 10, 10), S = (5, 5, 60, 60, 60),
rank_tolerance = 0.01, W = nominal WLS inverse 4x4 pair covariance,
A = W^{1/2} J S, the truth-selected mode-0 pair residual observable, the
slope-tertile/topology coverage protocol, the event bootstrap, and the
workbook-69 stable-core independent-validation gate with the pre-registered
role assignment hypothesis = canonical, independent = K-short.

Merged-rec physical-event identity is a hard contract: the K-short FD bank
is loaded through the ntuple-bridged occurrence-augmented identity in
``datasets.physical_event_identity``; sorted (run,event) grouping, fuzzy
joins, nearest-neighbour joins, and residual-proximity joins are forbidden,
and duplicate physical-event identity is a hard failure.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.cross_source_stable_core import (
    build_core_from_rows,
    independent_core_gate,
)
from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    frozen_scales_for,
    frozen_units_for,
    principal_angles_deg,
    subspace_as_json,
    subspace_distance,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.rigid_station_only_identifiability import (
    attach_source_tracklet_slopes,
    jacobian_validity_audit,
    slope_tertile_audit,
    topology_audit,
)
from alignment.rigid_station_only_identifiability import (
    build_all_reports as build_canonical_reports,
)
from alignment.rigid_station_only_identifiability import (
    load_campaign_config as load_canonical_config,
)
from alignment.rigid_station_only_identifiability import (
    load_five_dof_banks as load_canonical_five_dof_banks,
)
from alignment.tracker_only_identifiable_subspace import (
    _pool_campaign_banks,
    bootstrap_subspaces,
    half_split_subspaces,
    load_fd_only_bank,
    subspace_from_physical_bank,
    summarize_stability,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state
from datasets.physical_event_identity import (
    assert_no_duplicate_physical_identity,
    load_events_ntuple_identity,
    load_propagation_records_ntuple_identity,
    read_ntuple_event_index,
)
from evaluation.field_propagation import evaluate_field_propagation
from scripts.audit_6dof_identifiability import _source_entries
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-kshort-rigid-station-5dof-fd-complementarity-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "kshort_rigid_station_5dof_fd_complementarity_feasibility_v1.yaml"
STATION_FIVE_NAMES: tuple[str, ...] = FREE_PARAMETERS
FORBIDDEN_TRACKER_PARAMETERS: tuple[str, ...] = ("ift_dz_mm", "C_dx")

INHERITED_ENTRY_73 = "rigid_station_five_dof_not_source_or_coverage_portable"

DECISION_COMPLEMENTARITY_PASS = "kshort_rigid_station_five_dof_complementarity_pass"
DECISION_COMPLEMENTARITY_FAIL = "kshort_rigid_station_five_dof_complementarity_fail"
DECISION_PHYSICAL_FAILURE = "kshort_rigid_station_five_dof_physical_or_provenance_failure"
DECISION_CANONICAL_REGRESSION_FAILED = "kshort_campaign_canonical_regression_failed"

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
    "do_not_change_pion_track_or_event_selection_after_fd",
    "do_not_pool_kshort_with_muon_floor_100120",
    "do_not_use_fuzzy_or_nearest_neighbour_join",
    "do_not_use_residual_proximity_join",
    "do_not_use_sorted_run_event_grouping_for_merged_rec",
    "do_not_merge_kshort_sources_into_one",
    "do_not_rerun_physics_sources_to_improve_rank",
    "joint_pooled_rank_five_alone_is_not_success",
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

FROZEN_FINITE_DIFFERENCE_STEPS = {
    "ift_dx_mm": 0.5,
    "ift_dy_mm": 0.5,
    "ift_rx_mrad": 10.0,
    "ift_ry_mrad": 10.0,
    "ift_rz_mrad": 10.0,
}

INHERITANCE_FILES = {
    "workbook_73_config": "configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml",
    "workbook_74_config": "configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml",
    "workbook_75_config": "configs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml",
    "workbook_75_input_manifest": "outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/input_manifest.json",
    "workbook_75_reinventory": "outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/reinventory.json",
    "canonical_iteration_manifest": "outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/iteration_manifest.json",
}


def load_config(source: str | Path | None = None) -> dict[str, Any]:
    path = (
        Path(source).expanduser().resolve()
        if source is not None
        else resolve_under_root(project_root(), str(DEFAULT_CONFIG_RELATIVE))
    )
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"complementarity config must be a mapping: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected complementarity schema: {path}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"complementarity config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"complementarity config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the true cluster-local residual label")
    if float(payload["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        raise ValueError("rank_tolerance is frozen at 0.01 and must not be retuned")
    names = tuple(str(name) for name in payload["parameter_names"])
    if names != STATION_FIVE_NAMES:
        raise ValueError("parameter vector must be the frozen station 5DoF set")
    if any(name in FORBIDDEN_TRACKER_PARAMETERS for name in names):
        raise ValueError("dz and C_dx must stay out of the tracker fit")
    declared_scales = {str(k): float(v) for k, v in payload["scale_matrix_S"].items()}
    frozen = frozen_scales_for(names)
    if not np.allclose(
        [declared_scales[name] for name in names], frozen, rtol=0.0, atol=0.0
    ):
        raise ValueError("scale matrix S must match the frozen station-free scales")
    steps = {str(k): float(v) for k, v in payload["finite_difference_steps"].items()}
    if steps != FROZEN_FINITE_DIFFERENCE_STEPS:
        raise ValueError("finite-difference steps are frozen from the workbook-44/73 bank")
    identity = payload.get("physical_event_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("physical_event_identity contract is required")
    if identity.get("contract") != "occurrence_augmented_file_order":
        raise ValueError("physical-event identity must be occurrence-augmented file order")
    for forbidden in (
        "fuzzy_join_allowed",
        "nearest_neighbour_join_allowed",
        "residual_proximity_join_allowed",
        "sorted_run_event_grouping_allowed",
    ):
        if identity.get(forbidden) is not False:
            raise ValueError(f"physical_event_identity must set {forbidden}=false")
    if identity.get("duplicate_physical_event_identity") != "hard_failure":
        raise ValueError("duplicate physical-event identity must be a hard failure")
    contract = payload.get("complementarity_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("complementarity_contract is required")
    if contract.get("joint_source_stability_loso_mode") != "pooled_remainder":
        raise ValueError("joint source stability must use the pooled_remainder LOSO mode")
    if int(contract.get("require_joint_pooled_rank", -1)) != 5:
        raise ValueError("joint pooled rank criterion is frozen at 5")
    if int(contract.get("require_canonical_hypothesis_core_dimension", -1)) != 5:
        raise ValueError("canonical hypothesis core dimension is frozen at 5")
    _verify_inheritance(payload)
    config = dict(payload)
    config["config_path"] = str(path)
    return config


def _verify_inheritance(payload: Mapping[str, Any]) -> None:
    declared = payload.get("inheritance_sha256")
    if not isinstance(declared, Mapping):
        raise ValueError("inheritance_sha256 block is required")
    for key, relative in INHERITANCE_FILES.items():
        expected = str(declared.get(key, ""))
        if len(expected) != 64:
            raise ValueError(f"inheritance_sha256 lacks {key}")
        path = resolve_under_root(project_root(), relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"inheritance file {relative} SHA256 mismatch: "
                f"declared={expected} actual={actual}"
            )
    if payload.get("inherited_entry_73_decision") != INHERITED_ENTRY_73:
        raise ValueError("workbook-73 frozen decision string mismatch")


def _predeclared_source_ids(corpus: Mapping[str, Any]) -> tuple[str, ...]:
    train = tuple(str(item) for item in corpus["train_source_ids"])
    validation = tuple(str(item) for item in corpus["validation_source_ids"])
    if set(train) & set(validation):
        raise ValueError("train and validation source ids overlap")
    return train + validation


# ---------------------------------------------------------------------------
# Physical-event-identity evaluation (merged-rec hard contract)
# ---------------------------------------------------------------------------


def physical_identity_evaluation(tracklets: Path, propagations: Path, fraction: float):
    """Frozen FD evaluation with ntuple-bridged occurrence-augmented identity.

    Same ``evaluate_field_propagation`` call as the canonical chain
    (truth-matched, mode 0, min truth fraction), but both inputs carry the
    occurrence-augmented physical-event uid bridged through the sibling
    enhanced ntuple, so reused generator-job event numbers can never merge
    or misjoin distinct physical events across FD geometries.
    """
    tracklets = Path(tracklets)
    propagations = Path(propagations)
    enhanced = tracklets.parent / "enhanced_tracklets.root"
    index = read_ntuple_event_index(enhanced)
    events = load_events_ntuple_identity(tracklets, index, require_mc_labels=True)
    assert_no_duplicate_physical_identity(events)
    records = load_propagation_records_ntuple_identity(propagations, index)
    return evaluate_field_propagation(
        events,
        records,
        require_truth_match=True,
        q_over_p_mode=0,
        min_truth_match_fraction=fraction,
    )


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------


def load_kshort_five_dof_banks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    corpus = config["kshort_corpus"]
    names = tuple(str(name) for name in corpus["parameter_names"])
    if names != STATION_FIVE_NAMES:
        raise ValueError("K-short banks require the frozen station-free names")
    if corpus.get("physical_event_identity") != "occurrence_augmented_file_order":
        raise ValueError("K-short banks require the occurrence-augmented identity contract")
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    manifest = _read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("K-short iteration manifest has an invalid test-access declaration")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("K-short bank requires q_over_p_mode=0")
    if int(manifest.get("nevents_per_source", -1)) != int(corpus["nevents_per_source"]):
        raise ValueError("K-short bank nevents differs from the pre-registered full-file count")
    entries = _source_entries(manifest)
    expected = list(_predeclared_source_ids(corpus))
    by_id = {str(entry["source_id"]): entry for entry in entries}
    if sorted(by_id) != sorted(expected):
        raise ValueError("K-short iteration manifest sources differ from the pre-declared set")
    banks = []
    for source_id in expected:
        entry = by_id[source_id]
        bank = load_fd_only_bank(
            entry,
            anchor_point=str(corpus["anchor_point"]),
            min_truth_match_fraction=float(corpus["min_truth_match_fraction"]),
            only_parameters=names,
            evaluation_fn=physical_identity_evaluation,
        )
        banks.append(bank)
    loaded = [str(bank["source_id"]) for bank in banks]
    if loaded != expected:
        raise ValueError(
            "loaded K-short banks must match the pre-declared source order; "
            f"loaded={loaded} expected={expected}"
        )
    splits = {str(bank["source_id"]): str(bank["split"]) for bank in banks}
    for source_id in corpus["train_source_ids"]:
        if splits[str(source_id)] != "train":
            raise ValueError(f"pre-declared train source {source_id} is not train")
    for source_id in corpus["validation_source_ids"]:
        if splits[str(source_id)] != "validation":
            raise ValueError(f"pre-declared validation source {source_id} is not validation")
    attached = [
        attach_source_tracklet_slopes(
            bank,
            iteration_manifest=manifest,
            anchor_point=str(corpus["anchor_point"]),
            physical_event_identity=True,
        )
        for bank in banks
    ]
    for bank in attached:
        if tuple(bank["names"]) != STATION_FIVE_NAMES:
            raise ValueError(f"K-short bank {bank['source_id']} is not native five-column")
        if bank.get("held_out_physical_points_loaded"):
            raise ValueError("held-out physical points must not enter the 5DoF SVD")
        if bank.get("test_data_accessed"):
            raise ValueError("sealed test was accessed")
    return attached


def load_canonical_banks(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the canonical banks through the frozen workbook-73 loader."""
    corpus = config["canonical_corpus"]
    if corpus.get("physical_event_identity") != "raw_unique_run_event":
        raise ValueError("canonical bank must keep the raw unique (run,event) identity")
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    if sha256_file(manifest_path) != str(corpus["iteration_manifest_sha256"]):
        raise ValueError("canonical iteration manifest SHA256 mismatch; refusing to mix banks")
    canonical_config = load_canonical_config(
        resolve_under_root(
            project_root(),
            INHERITANCE_FILES["workbook_73_config"],
        )
    )
    banks = load_canonical_five_dof_banks(canonical_config)
    return canonical_config, banks


# ---------------------------------------------------------------------------
# Canonical-only regression (negative control)
# ---------------------------------------------------------------------------


def canonical_regression(
    config: Mapping[str, Any],
    *,
    canonical_config: Mapping[str, Any] | None = None,
    banks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    corpus = config["canonical_corpus"]
    regression = corpus["regression_artifacts"]
    if canonical_config is None or banks is None:
        canonical_config, banks = load_canonical_banks(config)
    frozen_basis = _read_json(
        resolve_under_root(project_root(), str(regression["identifiable_basis"]))
    )
    frozen_decision = _read_json(
        resolve_under_root(project_root(), str(regression["next_stage_decision"]))
    )
    if sha256_file(
        resolve_under_root(project_root(), str(regression["identifiable_basis"]))
    ) != str(regression["identifiable_basis_sha256"]):
        raise ValueError("frozen workbook-73 identifiable_basis SHA256 mismatch")
    if sha256_file(
        resolve_under_root(project_root(), str(regression["next_stage_decision"]))
    ) != str(regression["next_stage_decision_sha256"]):
        raise ValueError("frozen workbook-73 next_stage_decision SHA256 mismatch")

    reports = build_canonical_reports(canonical_config, banks=list(banks))
    rebuilt_basis = reports["identifiable_basis"]
    rebuilt_decision = reports["next_stage_decision"]

    frozen_sources = {
        str(row["source_id"]): row for row in frozen_basis.get("per_source", [])
    }
    rebuilt_sources = {
        str(row["source_id"]): row for row in rebuilt_basis.get("per_source", [])
    }
    reasons = []
    if set(frozen_sources) != set(rebuilt_sources):
        reasons.append("canonical_source_set_changed")
    rtol = float(regression["singular_value_rtol"])
    for source_id, frozen_row in frozen_sources.items():
        rebuilt_row = rebuilt_sources.get(source_id)
        if rebuilt_row is None:
            continue
        if int(rebuilt_row["identifiable_rank"]) != int(frozen_row["identifiable_rank"]):
            reasons.append(f"canonical_source_rank_changed:{source_id}")
        frozen_sv = np.asarray(frozen_row["singular_values"], dtype=np.float64)
        rebuilt_sv = np.asarray(rebuilt_row["singular_values"], dtype=np.float64)
        if frozen_sv.shape != rebuilt_sv.shape or not np.allclose(
            rebuilt_sv, frozen_sv, rtol=rtol, atol=0.0
        ):
            reasons.append(f"canonical_source_singular_values_changed:{source_id}")
    frozen_pooled = frozen_basis.get("pooled", {})
    rebuilt_pooled = rebuilt_basis.get("pooled", {})
    if int(rebuilt_pooled.get("identifiable_rank", -1)) != int(
        frozen_pooled.get("identifiable_rank", -2)
    ):
        reasons.append("canonical_pooled_rank_changed")
    frozen_decision_string = str(frozen_decision.get("decision", ""))
    rebuilt_decision_string = str(rebuilt_decision.get("decision", ""))
    if rebuilt_decision_string != frozen_decision_string:
        reasons.append("canonical_decision_changed")
    if rebuilt_decision_string != str(regression["expected_decision"]):
        reasons.append("canonical_decision_no_longer_matches_frozen_expectation")
    return {
        "kind": "canonical_only_regression_negative_control",
        "frozen_identifiable_basis_sha256": str(regression["identifiable_basis_sha256"]),
        "frozen_next_stage_decision_sha256": str(regression["next_stage_decision_sha256"]),
        "frozen_decision": frozen_decision_string,
        "rebuilt_decision": rebuilt_decision_string,
        "n_sources": int(len(rebuilt_sources)),
        "pooled_identifiable_rank": int(rebuilt_pooled.get("identifiable_rank", -1)),
        "singular_value_rtol": rtol,
        "pass": not reasons,
        "failure_reasons": reasons,
        "not_a_retune": True,
        "reports": reports,
    }


# ---------------------------------------------------------------------------
# Population report (K-short-only and joint), built from frozen primitives
# ---------------------------------------------------------------------------


def population_report(
    banks: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    label: str,
    required_rank: int | None,
    loso_mode: str = "left_out_source",
) -> dict[str, Any]:
    """Workbook-73 audit primitives applied to one population.

    ``required_rank`` is the frozen gate rank (5) when the population must
    meet the workbook-73 criterion, or ``None`` for the K-short-only report,
    where stability is referenced to the population's own pooled native rank
    (K-short-only is not required to reach rank 5).  The three-arm closure
    is never opened here.

    ``loso_mode`` selects the pre-registered leave-one-source-out semantics:
    ``left_out_source`` (workbook-73: remaining pool vs the left-out source,
    requires the left-out source to reach the pooled rank) or
    ``pooled_remainder`` (joint-campaign gate: the pooled subspace rebuilt
    without each source vs the full pooled subspace, both at the pooled
    rank).  The latter is the well-defined joint source-stability question
    when low-rank canonical sources stay in the pool; both use the frozen
    15 deg / 1.0 Frobenius / same-rank thresholds.
    """
    if loso_mode not in ("left_out_source", "pooled_remainder"):
        raise ValueError(f"unknown loso_mode '{loso_mode}'")
    if not banks:
        raise ValueError(f"{label}: no banks supplied")
    rank_tolerance = float(config["rank_tolerance"])
    rcond = float(config["normal_matrix_rcond"])
    gates = config["gates"]
    coverage_cfg = config["coverage"]

    per_source = []
    source_subspaces = []
    validity_rows = []
    tertile_rows = []
    topology_rows = []
    for bank in banks:
        subspace, extras = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond
        )
        source_subspaces.append((f"{bank['split']}:{bank['source_id']}", subspace))
        validity_rows.append(
            jacobian_validity_audit(bank, config=config, subspace=subspace, extras=extras)
        )
        tertile_rows.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                **slope_tertile_audit(
                    bank,
                    rank_tolerance=rank_tolerance,
                    rcond=rcond,
                    n_bins=int(coverage_cfg["n_slope_bins"]),
                    min_events=int(coverage_cfg["min_tertile_events"]),
                ),
            }
        )
        topology_row = topology_audit(
            bank,
            rank_tolerance=rank_tolerance,
            rcond=rcond,
            required_rank=int(gates["required_identifiable_rank"]),
        )
        topology_row.pop("_subspace", None)
        topology_rows.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                **topology_row,
            }
        )
        per_source.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "n_pairs": int(extras["n_pairs"]),
                "identifiable_rank": int(subspace.identifiable_rank),
                "singular_values": [float(value) for value in subspace.singular_values],
                "subspace": subspace_as_json(subspace),
                "source_slope_missing_pairs": int(bank.get("source_slope_missing_pairs", 0)),
            }
        )

    pooled = _pool_campaign_banks(list(banks))
    pooled["source_id"] = f"pooled_{label}"
    pooled_subspace, pooled_extras = subspace_from_physical_bank(
        pooled, rank_tolerance=rank_tolerance, rcond=rcond
    )
    pooled_rank = int(pooled_subspace.identifiable_rank)
    reference_rank = pooled_rank if required_rank is None else int(required_rank)

    source_stability = summarize_stability(
        pooled_subspace,
        source_subspaces,
        max_principal_angle_deg=float(gates["max_source_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_source_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    train_banks = [bank for bank in banks if str(bank["split"]) == "train"]
    validation_banks = [bank for bank in banks if str(bank["split"]) == "validation"]
    split_stability = None
    if train_banks and validation_banks:
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
    for label_i, subspace in source_subspaces:
        remaining = _pool_campaign_banks(
            [
                bank
                for bank in banks
                if f"{bank['split']}:{bank['source_id']}" != label_i
            ]
        )
        remaining_subspace, _ = subspace_from_physical_bank(
            remaining, rank_tolerance=rank_tolerance, rcond=rcond
        )
        if loso_mode == "left_out_source":
            report = summarize_stability(
                remaining_subspace,
                [(label_i, subspace)],
                max_principal_angle_deg=float(gates["max_source_disjoint_principal_angle_deg"]),
                max_projector_frobenius=float(gates["max_source_disjoint_projector_frobenius"]),
                require_same_rank=bool(gates["require_same_identifiable_rank"]),
            )
        else:
            report = summarize_stability(
                pooled_subspace,
                [(f"without_{label_i}", remaining_subspace)],
                max_principal_angle_deg=float(gates["max_source_disjoint_principal_angle_deg"]),
                max_projector_frobenius=float(gates["max_source_disjoint_projector_frobenius"]),
                require_same_rank=bool(gates["require_same_identifiable_rank"]),
            )
        source_disjoint_rows.append({"left_out": label_i, "loso_mode": loso_mode, **report})

    pooled_tertiles = slope_tertile_audit(
        pooled,
        rank_tolerance=rank_tolerance,
        rcond=rcond,
        n_bins=int(coverage_cfg["n_slope_bins"]),
        min_events=int(coverage_cfg["min_tertile_events"]),
    )
    pooled_topology = topology_audit(
        pooled,
        rank_tolerance=rank_tolerance,
        rcond=rcond,
        required_rank=reference_rank,
    )
    topology_space = pooled_topology.pop("_subspace", None)
    topology_stability = None
    if topology_space is not None:
        topology_stability = summarize_stability(
            pooled_subspace,
            [(f"{label}:complete_routes", topology_space)],
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

    usable_tertiles = [
        bin_row
        for row in tertile_rows
        for bin_row in row.get("bins") or []
        if not bin_row.get("too_small") and bin_row.get("identifiable_rank") is not None
    ]
    pooled_usable_tertiles = [
        bin_row
        for bin_row in pooled_tertiles.get("bins") or []
        if not bin_row.get("too_small") and bin_row.get("identifiable_rank") is not None
    ]
    return {
        "kind": f"{label}_population_report",
        "label": label,
        "n_sources": int(len(banks)),
        "required_rank_gate": None if required_rank is None else int(required_rank),
        "stability_reference_rank": int(reference_rank),
        "stability_reference": (
            "population_pooled_native_rank" if required_rank is None else "frozen_required_rank"
        ),
        "per_source": per_source,
        "jacobian_validity": validity_rows,
        "jacobian_valid": all(bool(row["valid"]) for row in validity_rows),
        "pooled": {
            "identifiable_rank": pooled_rank,
            "null_dimension": int(pooled_subspace.null_dimension),
            "singular_values": [float(value) for value in pooled_subspace.singular_values],
            "n_pairs": int(pooled_extras["n_pairs"]),
            "subspace": subspace_as_json(pooled_subspace),
        },
        "source_stability": source_stability,
        "train_validation_stability": split_stability,
        "source_disjoint_stability": {
            "stable": bool(source_disjoint_rows)
            and all(bool(row["stable"]) for row in source_disjoint_rows),
            "comparisons": source_disjoint_rows,
        },
        "slope_tertiles": {
            "per_source": tertile_rows,
            "pooled": pooled_tertiles,
            "any_rank_drop": any(bool(row.get("rank_drop")) for row in tertile_rows)
            or bool(pooled_tertiles.get("rank_drop")),
            "usable_tertiles_at_reference_rank": bool(pooled_usable_tertiles)
            and all(
                int(bin_row["identifiable_rank"]) == int(reference_rank)
                for bin_row in pooled_usable_tertiles
            ),
        },
        "topology": {
            "per_source": topology_rows,
            "pooled": pooled_topology,
            "stability": topology_stability,
        },
        "bootstrap_stability": bootstrap_stability,
        "half_split_stability": half_stability,
        "three_arm_opened": False,
        "_pooled_subspace": pooled_subspace,
        "_pooled_weighted_matrix": pooled_extras["weighted_matrix"],
        "_source_subspaces": source_subspaces,
    }


# ---------------------------------------------------------------------------
# Joint complementarity
# ---------------------------------------------------------------------------


def _stable_core_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    per_source = {
        str(row["source_id"]): row for row in report["per_source"]
    }
    rows = []
    for label, subspace in report["_source_subspaces"]:
        split, _, source_id = str(label).partition(":")
        summary = per_source[source_id]
        rows.append(
            {
                "source_id": source_id,
                "split": split,
                "subspace": subspace,
                "projector": np.asarray(subspace.projector_id, dtype=np.float64),
                "identifiable_rank": int(subspace.identifiable_rank),
                "n_pairs": int(summary["n_pairs"]),
                "singular_values": list(summary["singular_values"]),
            }
        )
    return rows


def joint_complementarity_report(
    config: Mapping[str, Any],
    *,
    canonical_report: Mapping[str, Any],
    kshort_report: Mapping[str, Any],
    joint_report: Mapping[str, Any],
) -> dict[str, Any]:
    core_cfg = config["stable_core"]
    contract = config["complementarity_contract"]

    canonical_rows = _stable_core_rows(canonical_report)
    kshort_rows = _stable_core_rows(kshort_report)

    hypothesis_core = build_core_from_rows(
        canonical_rows,
        min_consensus_eigenvalue=float(core_cfg["min_consensus_eigenvalue"]),
        min_mode_persistence=float(core_cfg["min_mode_persistence"]),
        min_source_support_fraction=float(core_cfg["min_hypothesis_source_support_fraction"]),
    )
    independent = independent_core_gate(
        kshort_rows,
        hypothesis_core,
        min_mode_persistence=float(core_cfg["independent_min_mode_persistence"]),
        min_source_support_fraction=float(core_cfg["independent_min_source_support_fraction"]),
        max_principal_angle_deg=float(core_cfg["independent_max_principal_angle_deg"]),
        max_missing_frobenius=float(core_cfg["independent_max_missing_frobenius"]),
    )

    canonical_pooled = canonical_report["_pooled_subspace"]
    kshort_pooled = kshort_report["_pooled_subspace"]
    relation = subspace_distance(canonical_pooled, kshort_pooled)
    angles = principal_angles_deg(canonical_pooled.v_id, kshort_pooled.v_id)

    # Weakest canonical direction information report (no gate; no frozen
    # threshold exists and inventing one is forbidden).
    canonical_weighted = np.asarray(canonical_report["_pooled_weighted_matrix"], dtype=np.float64)
    kshort_weighted = np.asarray(kshort_report["_pooled_weighted_matrix"], dtype=np.float64)
    singular = np.asarray(canonical_pooled.singular_values, dtype=np.float64)
    v_matrix = np.asarray(canonical_pooled.v, dtype=np.float64)
    weakest_index = int(np.argmin(singular)) if singular.size else 0
    weakest_direction = v_matrix[:, weakest_index] if singular.size else np.zeros(5)
    weakest_norm = float(np.linalg.norm(weakest_direction))
    if weakest_norm > 0.0:
        weakest_direction = weakest_direction / weakest_norm
    canonical_strength = float(
        np.linalg.norm(canonical_weighted @ weakest_direction)
    )
    kshort_strength = float(np.linalg.norm(kshort_weighted @ weakest_direction))

    hypothesis_dimension = int(hypothesis_core["core_dimension"])
    required_core_dimension = int(contract["require_canonical_hypothesis_core_dimension"])

    checks = {
        "kshort_pooled_rank_at_least": int(kshort_report["pooled"]["identifiable_rank"])
        >= int(contract["require_kshort_pooled_rank_at_least"]),
        "kshort_source_stability_at_native_rank": bool(
            kshort_report["source_stability"]["stable"]
        ),
        "kshort_coverage_stability_at_native_rank": bool(
            not kshort_report["slope_tertiles"]["any_rank_drop"]
            and kshort_report["slope_tertiles"]["usable_tertiles_at_reference_rank"]
            and kshort_report["topology"]["pooled"].get("same_rank", False)
            and (kshort_report["topology"]["stability"] or {}).get("stable", False)
        ),
        "joint_pooled_rank_five": int(joint_report["pooled"]["identifiable_rank"])
        == int(contract["require_joint_pooled_rank"]),
        "joint_source_stability": bool(joint_report["source_disjoint_stability"]["stable"]),
        "joint_coverage_stability": bool(
            not joint_report["slope_tertiles"]["any_rank_drop"]
            and joint_report["slope_tertiles"]["usable_tertiles_at_reference_rank"]
            and joint_report["topology"]["pooled"].get("same_rank", False)
            and (joint_report["topology"]["stability"] or {}).get("stable", False)
        ),
        "joint_bootstrap_stability": bool(
            joint_report["bootstrap_stability"]["stable"]
            and joint_report["half_split_stability"]["stable"]
        ),
        "canonical_hypothesis_core_dimension_five": hypothesis_dimension
        == required_core_dimension,
        "kshort_independent_validation_of_canonical_core": bool(independent["pass"]),
    }
    return {
        "kind": "joint_complementarity",
        "hypothesis_set": "canonical_18_sources",
        "independent_set": "kshort_10_sources",
        "stable_core": {
            key: value
            for key, value in hypothesis_core.items()
            if key not in {"v_core", "v_orthogonal", "projector_core", "core_space"}
        },
        "canonical_hypothesis_core_dimension": hypothesis_dimension,
        "independent_validation": independent,
        "canonical_kshort_pooled_subspace_relation": {
            **{key: value for key, value in relation.items()},
            "identifiable_principal_angles_deg": [float(value) for value in angles],
        },
        "weakest_canonical_direction_report": {
            "weakest_canonical_singular_value": float(singular[weakest_index])
            if singular.size
            else None,
            "canonical_information_along_weakest": canonical_strength,
            "kshort_information_along_weakest": kshort_strength,
            "kshort_to_canonical_ratio": (
                None
                if canonical_strength <= 0.0
                else float(kshort_strength / canonical_strength)
            ),
            "report_only_no_frozen_threshold": True,
        },
        "equal_per_source_consensus_prevents_track_count_reweighting": True,
        "checks": checks,
        "joint_pooled_rank_five_alone_is_not_success": True,
    }


# ---------------------------------------------------------------------------
# Physical closure and campaign decision
# ---------------------------------------------------------------------------


def physical_closure_report(
    config: Mapping[str, Any],
    *,
    kshort_banks: Sequence[Mapping[str, Any]],
    kshort_report: Mapping[str, Any],
) -> dict[str, Any]:
    corpus = config["kshort_corpus"]
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    manifest = _read_json(manifest_path)
    entries = {str(entry["source_id"]): entry for entry in _source_entries(manifest)}
    per_source = []
    reasons = []
    for bank in kshort_banks:
        source_id = str(bank["source_id"])
        entry = entries.get(source_id)
        if entry is None:
            reasons.append(f"missing_manifest_entry:{source_id}")
            continue
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        points = plan.get("points", [])
        missing_points = []
        for point in points:
            relative = point.get("relative_point_dir")
            point_root = root / str(relative)
            for artifact in (
                point_root / "payload" / "alignment_payload.json",
                point_root / "refit" / "tracklets.root",
                point_root / "refit" / "propagations.root",
                point_root / "refit" / "enhanced_tracklets.root",
                point_root / "refit" / "content_audit.json",
            ):
                if not artifact.is_file():
                    missing_points.append(f"{point.get('name')}:{artifact.name}")
        overlap = dict(bank.get("overlap", {}))
        validity = next(
            (
                row
                for row in kshort_report["jacobian_validity"]
                if str(row["source_id"]) == source_id
            ),
            {},
        )
        per_source.append(
            {
                "source_id": source_id,
                "n_plan_points": int(len(points)),
                "missing_point_artifacts": missing_points,
                "join_overlap": overlap,
                "n_pairs": int(validity.get("n_pairs", 0)),
                "jacobian_valid": bool(validity.get("valid", False)),
            }
        )
        if missing_points:
            reasons.append(f"incomplete_fd_points:{source_id}")
        if not validity.get("valid", False):
            reasons.append(f"jacobian_validity_failed:{source_id}")
    return {
        "kind": "physical_closure",
        "iteration_manifest": str(manifest_path),
        "iteration_manifest_sha256": sha256_file(manifest_path),
        "n_sources": int(len(per_source)),
        "per_source": per_source,
        "exact_join_contract": "occurrence_augmented_file_order",
        "duplicate_physical_event_identity_failures": 0,
        "no_geometry_write": True,
        "no_official_conditions_write": True,
        "pass": not reasons,
        "failure_reasons": reasons,
    }


def decide_campaign(
    *,
    physical_closure: Mapping[str, Any],
    canonical_regression_report: Mapping[str, Any],
    complementarity: Mapping[str, Any],
) -> dict[str, Any]:
    if not physical_closure.get("pass", False):
        return {
            "decision": DECISION_PHYSICAL_FAILURE,
            "complementarity_pass": False,
            "reason": "physical_closure_or_provenance_failed",
            "physical_failure_reasons": list(physical_closure.get("failure_reasons", [])),
            "no_rank_claimed": True,
        }
    if not canonical_regression_report.get("pass", False):
        return {
            "decision": DECISION_CANONICAL_REGRESSION_FAILED,
            "complementarity_pass": False,
            "reason": "canonical_regression_no_longer_reproduces_workbook_73",
            "regression_failure_reasons": list(
                canonical_regression_report.get("failure_reasons", [])
            ),
            "no_rank_claimed": True,
        }
    checks = dict(complementarity["checks"])
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return {
            "decision": DECISION_COMPLEMENTARITY_FAIL,
            "complementarity_pass": False,
            "failed_checks": failed,
            "enables": "gauge_constrained_external_constraint_alignment_feasibility",
            "forbidden_after_failure": [
                "rank_tolerance_retune",
                "scale_matrix_retune",
                "source_dropping",
                "pion_selection_change",
                "residual_change",
                "nearby_mc_trial",
                "seven_d_or_cluster_local_or_stable_core_rescue",
            ],
        }
    return {
        "decision": DECISION_COMPLEMENTARITY_PASS,
        "complementarity_pass": True,
        "failed_checks": [],
        "next_steps_require_separate_preregistration": [
            "full_tracker_only_five_dof_controlled_closure",
            "three_arm_synthetic_injected_closure",
            "frozen_v2_unknown_association_integration",
        ],
        "real_data_correction_remains_closed": True,
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat()
    canonical_config, canonical_banks = load_canonical_banks(config)
    regression = canonical_regression(
        config, canonical_config=canonical_config, banks=canonical_banks
    )
    canonical_population = population_report(
        canonical_banks,
        config=config,
        label="canonical_only",
        required_rank=int(config["gates"]["required_identifiable_rank"]),
    )
    kshort_banks = load_kshort_five_dof_banks(config)
    kshort_report = population_report(
        kshort_banks, config=config, label="kshort_only", required_rank=None
    )
    joint_banks = list(canonical_banks) + list(kshort_banks)
    joint_report = population_report(
        joint_banks,
        config=config,
        label="joint_canonical_kshort",
        required_rank=int(config["gates"]["required_identifiable_rank"]),
        loso_mode=str(
            config["complementarity_contract"]["joint_source_stability_loso_mode"]
        ),
    )
    closure = physical_closure_report(
        config, kshort_banks=kshort_banks, kshort_report=kshort_report
    )
    complementarity = joint_complementarity_report(
        config,
        canonical_report=canonical_population,
        kshort_report=kshort_report,
        joint_report=joint_report,
    )
    decision = decide_campaign(
        physical_closure=closure,
        canonical_regression_report=regression,
        complementarity=complementarity,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "git_head_sha": git_head_sha(),
        "config_path": str(config.get("config_path", "")),
        "operating_state": common_operating_state(),
        "physical_closure": closure,
        "canonical_regression": {
            key: value for key, value in regression.items() if key != "reports"
        },
        "canonical_only": canonical_population,
        "kshort_only": kshort_report,
        "joint": joint_report,
        "complementarity": complementarity,
        "next_stage_decision": decision,
    }
