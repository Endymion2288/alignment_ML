"""Cross-source stable-core identifiable subspace, independent validation V1.

Workbook 68 remains frozen.  Rank tolerance and S are inherited.  Rank-6
source projectors are never truncated.  The seven V1 sources construct a
hypothesis only; confirmatory evidence is the already-produced disjoint
FD banks that were not inspected for the rank-flip.  Three-arm uses frozen
``V_core``, not workbook-68 pooled ``V_id``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    MixedUnitNakedJacobianError,
    consensus_operator,
    core_contained_in_projector,
    identifiable_subspace_from_core,
    principal_angles_deg,
    projector_frobenius_distance,
    refuse_forced_identifiable_rank,
    refuse_rank_threshold_from_spectrum,
    select_stable_core,
    signed_eigh_descending,
    subspace_as_json,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    sha256_file,
)
from alignment.tracker_only_identifiable_subspace import (
    DECISION_BASIS_UNSTABLE,
    bootstrap_subspaces,
    evaluate_three_arms,
    load_physical_banks,
    subspace_from_physical_bank,
    three_arm_payloads,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, assert_no_alignment_payload, common_operating_state


SCHEMA_VERSION = "faser-cross-source-stable-core-identifiable-subspace-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "cross_source_stable_core_identifiable_subspace_v1.yaml"
INHERITED_V1_DECISION = DECISION_BASIS_UNSTABLE
DECISION_INDEPENDENT_PASS = "cross_source_stable_core_independent_validation_pass"
DECISION_HYPOTHESIS_FAIL = "cross_source_stable_core_hypothesis_unstable_independent_validation_not_opened"
DECISION_INDEPENDENT_FAIL = "cross_source_stable_core_independent_validation_fail"
DECISION_THREE_ARM_FAIL = "cross_source_stable_core_independent_pass_but_three_arm_fail"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_identifiable_modes",
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
    "do_not_open_sealed_test",
    "do_not_svd_naked_mixed_unit_jacobian",
    "do_not_retune_scales_from_singular_values",
    "do_not_retune_rank_threshold_from_spectrum",
    "do_not_force_identifiable_rank_five",
    "do_not_truncate_rank_six_source",
    "do_not_drop_v1_failure_source",
    "do_not_open_three_arm_unless_independent_core_passes",
    "do_not_use_v1_pooled_v_id_for_three_arm",
    "do_not_merge_cluster_local_repair_into_stable_core_gate",
    "do_not_remove_dz_to_improve_this_svd",
    "hypothesis_sources_are_not_confirmatory_evidence",
    "dimension_determined_by_eigenvalue_and_persistence",
    "survey_is_external_cross_check_only",
)


def load_campaign_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"stable-core config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected stable-core schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"stable-core config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"stable-core config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("inherited_v1_decision") != INHERITED_V1_DECISION:
        raise ValueError("campaign must inherit workbook-68 frozen decision")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the true cluster-local residual label")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if float(payload["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        refuse_rank_threshold_from_spectrum()
    if payload.get("algorithm", {}).get("forced_core_dimension") not in (None, False):
        raise ValueError("core dimension must remain automatic; forced_core_dimension is forbidden")
    hypothesis = tuple(str(item) for item in payload["jacobian_corpus"]["hypothesis_source_ids"])
    independent = tuple(str(item) for item in payload["jacobian_corpus"]["independent_validation_source_ids"])
    overlap = set(hypothesis) & set(independent)
    if overlap:
        raise ValueError(f"independent sources overlap hypothesis sources: {sorted(overlap)}")
    failure_source = "mc24_100047_00150_00199"
    if failure_source not in hypothesis:
        raise ValueError("rank-6 V1 failure source must remain in the hypothesis list")
    if failure_source in independent:
        raise ValueError("rank-6 V1 failure source cannot be reused as confirmatory evidence")
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
            "solver_restricted_to_identifiable_subspace": True,
            "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
            "modes_are_reconstruction_observable_linear_combinations": True,
            "truth_selected_association_is_solver_control": True,
            "unknown_association_frozen_v2_closure_opened": False,
            "real_data_alignment_correction_solved": False,
            "inherited_v1_decision": INHERITED_V1_DECISION,
            "hypothesis_sources_are_not_confirmatory_evidence": True,
            "cluster_local_repair_is_not_a_stable_core_gate": True,
            "dz_not_removed_to_improve_svd": True,
        }
    )
    return state


def _loader_config(config: Mapping[str, Any], source_ids: Sequence[str]) -> dict[str, Any]:
    payload = dict(config)
    corpus = dict(config["jacobian_corpus"])
    corpus["source_ids"] = list(source_ids)
    payload["jacobian_corpus"] = corpus
    return payload


def load_role_banks(config: Mapping[str, Any], *, role: str) -> list[dict[str, Any]]:
    key = "hypothesis_source_ids" if role == "hypothesis" else "independent_validation_source_ids"
    source_ids = tuple(str(item) for item in config["jacobian_corpus"][key])
    banks = load_physical_banks(_loader_config(config, source_ids))
    loaded = {str(bank["source_id"]) for bank in banks}
    missing = [item for item in source_ids if item not in loaded]
    if missing:
        raise ValueError(f"{role} Jacobian banks missing: {missing}")
    extra = sorted(loaded - set(source_ids))
    if extra:
        raise ValueError(f"{role} loader returned unexpected sources: {extra}")
    if any(bool(bank.get("test_data_accessed")) for bank in banks):
        raise ValueError("sealed test was accessed")
    return banks


def source_projectors(
    banks: Sequence[Mapping[str, Any]],
    *,
    rank_tolerance: float,
    rcond: float,
) -> list[dict[str, Any]]:
    rows = []
    for bank in banks:
        subspace, extras = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond
        )
        refuse_forced_identifiable_rank(
            native_rank=subspace.identifiable_rank, requested_rank=None
        )
        rows.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "n_pairs": int(extras["n_pairs"]),
                "identifiable_rank": int(subspace.identifiable_rank),
                "null_dimension": int(subspace.null_dimension),
                "singular_values": [float(value) for value in subspace.singular_values],
                "truncated_to_five": False,
                "subspace": subspace,
                "extras": extras,
                "projector": np.asarray(subspace.projector_id, dtype=np.float64),
            }
        )
    return rows


def build_core_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_consensus_eigenvalue: float,
    min_mode_persistence: float,
    min_source_support_fraction: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("stable-core construction requires at least one source")
    names = rows[0]["subspace"].parameter_names
    units = rows[0]["subspace"].parameter_units
    scales = rows[0]["subspace"].parameter_scales
    for row in rows:
        if row["subspace"].parameter_names != names:
            raise ValueError("source parameter names disagree")
        if not np.allclose(row["subspace"].parameter_scales, scales):
            raise ValueError("source scale matrices disagree; S must stay frozen")
    projectors = [row["projector"] for row in rows]
    consensus = consensus_operator(projectors)
    eigenvalues, eigenvectors, signs = signed_eigh_descending(consensus)
    selected = select_stable_core(
        eigenvalues,
        eigenvectors,
        projectors,
        min_consensus_eigenvalue=min_consensus_eigenvalue,
        min_mode_persistence=min_mode_persistence,
        min_source_support_fraction=min_source_support_fraction,
    )
    core_space = identifiable_subspace_from_core(
        parameter_names=names,
        parameter_units=units,
        parameter_scales=scales,
        v_core=selected["v_core"],
        v_orthogonal=selected["v_orthogonal"],
    )
    per_source = []
    for row, projector in zip(rows, projectors):
        contained = core_contained_in_projector(selected["v_core"], projector)
        per_source.append(
            {
                "source_id": row["source_id"],
                "split": row["split"],
                "n_pairs": row["n_pairs"],
                "native_identifiable_rank": row["identifiable_rank"],
                "truncated_to_five": False,
                "singular_values": row["singular_values"],
                "core_containment": contained,
            }
        )
    return {
        "parameter_names": list(names),
        "parameter_units": list(units),
        "parameter_scales": [float(value) for value in scales],
        "n_sources": int(len(rows)),
        "native_ranks": [int(row["identifiable_rank"]) for row in rows],
        "includes_rank_six_source": bool(any(int(row["identifiable_rank"]) == 6 for row in rows)),
        "consensus_eigenvalues": [float(value) for value in eigenvalues],
        "consensus_eigenvectors": [[float(item) for item in column] for column in eigenvectors.T],
        "mode_signs": list(signs),
        "sign_convention": "largest_abs_right_vector_entry_positive",
        "core_dimension": int(selected["core_dimension"]),
        "mode_labels": list(selected["labels"]),
        "mode_persistence_by_mode": selected["mode_persistence_by_mode"],
        "source_support_fraction_by_mode": selected["source_support_fraction_by_mode"],
        "v_core": selected["v_core"],
        "v_orthogonal": selected["v_orthogonal"],
        "projector_core": selected["projector_core"],
        "core_space": core_space,
        "per_source": per_source,
        "dimension_determined_by_eigenvalue_and_persistence": True,
        "forced_core_dimension": None,
        "not_forced_to_rank_five": True,
        "not_mechanical_ry_or_C_dx": True,
    }


def loso_core_stability(
    rows: Sequence[Mapping[str, Any]],
    hypothesis_core: Mapping[str, Any],
    *,
    min_consensus_eigenvalue: float,
    min_mode_persistence: float,
    min_source_support_fraction: float,
    max_principal_angle_deg: float,
    max_projector_frobenius: float,
    require_same_dimension: bool,
) -> dict[str, Any]:
    comparisons = []
    for leave_index, left_out in enumerate(rows):
        kept = [row for index, row in enumerate(rows) if index != leave_index]
        rebuilt = build_core_from_rows(
            kept,
            min_consensus_eigenvalue=min_consensus_eigenvalue,
            min_mode_persistence=min_mode_persistence,
            min_source_support_fraction=min_source_support_fraction,
        )
        same_dim = int(rebuilt["core_dimension"]) == int(hypothesis_core["core_dimension"])
        if same_dim and hypothesis_core["core_dimension"]:
            angles = [
                float(value)
                for value in principal_angles_deg(hypothesis_core["v_core"], rebuilt["v_core"])
            ]
            max_angle = float(max(angles)) if angles else 0.0
        else:
            angles = None
            max_angle = None
        distance = projector_frobenius_distance(
            hypothesis_core["projector_core"], rebuilt["projector_core"]
        )
        reasons = []
        if require_same_dimension and not same_dim:
            reasons.append("core_dimension_changed")
        if max_angle is not None and max_angle > float(max_principal_angle_deg):
            reasons.append("principal_angle_exceeds_predeclared_gate")
        if distance > float(max_projector_frobenius):
            reasons.append("projector_frobenius_exceeds_predeclared_gate")
        comparisons.append(
            {
                "left_out_source_id": left_out["source_id"],
                "left_out_native_rank": left_out["identifiable_rank"],
                "core_dimension": rebuilt["core_dimension"],
                "same_core_dimension": same_dim,
                "principal_angles_deg": angles,
                "max_principal_angle_deg": max_angle,
                "projector_frobenius_distance": float(distance),
                "stable": not reasons,
                "failure_reasons": reasons,
                "compares_subspace_not_signed_vector_elements": True,
            }
        )
    unstable = [row["left_out_source_id"] for row in comparisons if not row["stable"]]
    same_rank_angles = [
        row["max_principal_angle_deg"]
        for row in comparisons
        if row["max_principal_angle_deg"] is not None
    ]
    return {
        "n_comparisons": int(len(comparisons)),
        "stable": not unstable,
        "unstable_labels": unstable,
        "max_principal_angle_deg": max(same_rank_angles) if same_rank_angles else None,
        "max_projector_frobenius_distance": max(
            float(row["projector_frobenius_distance"]) for row in comparisons
        )
        if comparisons
        else None,
        "comparisons": comparisons,
    }


def _compare_core_to_reference(
    rebuilt: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    max_principal_angle_deg: float,
    max_projector_frobenius: float,
    require_same_dimension: bool,
) -> dict[str, Any]:
    same_dim = int(rebuilt["core_dimension"]) == int(reference["core_dimension"])
    contained = core_contained_in_projector(reference["v_core"], rebuilt["projector_core"])
    if same_dim and reference["core_dimension"]:
        angles = [
            float(value) for value in principal_angles_deg(reference["v_core"], rebuilt["v_core"])
        ]
        max_angle = float(max(angles)) if angles else 0.0
    else:
        angles = [float(value) for value in contained["principal_angles_deg"]]
        max_angle = float(contained["max_principal_angle_deg"])
    distance = projector_frobenius_distance(reference["projector_core"], rebuilt["projector_core"])
    reasons = []
    if require_same_dimension and not same_dim:
        reasons.append("core_dimension_changed")
    if max_angle > float(max_principal_angle_deg):
        reasons.append("principal_angle_exceeds_predeclared_gate")
    if distance > float(max_projector_frobenius):
        reasons.append("projector_frobenius_exceeds_predeclared_gate")
    return {
        "core_dimension": rebuilt["core_dimension"],
        "same_core_dimension": same_dim,
        "principal_angles_deg": angles,
        "max_principal_angle_deg": max_angle,
        "projector_frobenius_distance": float(distance),
        "min_mode_persistence": contained["min_mode_persistence"],
        "stable": not reasons,
        "failure_reasons": reasons,
        "compares_subspace_not_signed_vector_elements": True,
    }


def source_bootstrap_core_stability(
    rows: Sequence[Mapping[str, Any]],
    hypothesis_core: Mapping[str, Any],
    *,
    n_replicates: int,
    seed: int,
    min_consensus_eigenvalue: float,
    min_mode_persistence: float,
    min_source_support_fraction: float,
    max_principal_angle_deg: float,
    max_projector_frobenius: float,
) -> dict[str, Any]:
    """Resample hypothesis sources with replacement and rebuild the consensus core."""
    rng = np.random.default_rng(int(seed))
    n_sources = len(rows)
    comparisons = []
    for replicate in range(int(n_replicates)):
        choice = rng.integers(0, n_sources, size=n_sources)
        resampled = [rows[int(index)] for index in choice]
        rebuilt = build_core_from_rows(
            resampled,
            min_consensus_eigenvalue=min_consensus_eigenvalue,
            min_mode_persistence=min_mode_persistence,
            min_source_support_fraction=min_source_support_fraction,
        )
        compared = _compare_core_to_reference(
            rebuilt,
            hypothesis_core,
            max_principal_angle_deg=max_principal_angle_deg,
            max_projector_frobenius=max_projector_frobenius,
            require_same_dimension=False,
        )
        compared["replicate"] = int(replicate)
        compared["resampled_source_ids"] = [row["source_id"] for row in resampled]
        comparisons.append(compared)
    unstable = [row["replicate"] for row in comparisons if not row["stable"]]
    angles = [float(row["max_principal_angle_deg"]) for row in comparisons]
    distances = [float(row["projector_frobenius_distance"]) for row in comparisons]
    return {
        "kind": "source_with_replacement_consensus",
        "n_replicates_requested": int(n_replicates),
        "n_replicates_kept": int(len(comparisons)),
        "stable": not unstable,
        "n_unstable": int(len(unstable)),
        "max_principal_angle_deg": max(angles) if angles else None,
        "max_projector_frobenius_distance": max(distances) if distances else None,
        "comparisons": comparisons,
    }


def event_bootstrap_core_persistence(
    banks: Sequence[Mapping[str, Any]],
    hypothesis_core: Mapping[str, Any],
    *,
    n_replicates: int,
    seed: int,
    rank_tolerance: float,
    rcond: float,
    min_pairs: int,
    min_mode_persistence: float,
    max_principal_angle_deg: float,
    max_missing_frobenius: float,
) -> dict[str, Any]:
    """Event-bootstrap each hypothesis source projector and measure V_core persistence."""
    per_source = []
    n_total = 0
    n_passed = 0
    for source_index, bank in enumerate(banks):
        subspaces = bootstrap_subspaces(
            bank,
            n_replicates=n_replicates,
            seed=int(seed) + int(source_index),
            rank_tolerance=rank_tolerance,
            rcond=rcond,
            min_pairs=min_pairs,
        )
        replicates = []
        for replicate, subspace in enumerate(subspaces):
            contained = core_contained_in_projector(hypothesis_core["v_core"], subspace.projector_id)
            persistences = [float(value) for value in contained["mode_persistence"]]
            reasons = []
            if persistences and min(persistences) < float(min_mode_persistence):
                reasons.append("mode_persistence_below_predeclared_gate")
            if float(contained["max_principal_angle_deg"]) > float(max_principal_angle_deg):
                reasons.append("principal_angle_exceeds_predeclared_gate")
            if float(contained["missing_projector_frobenius"]) > float(max_missing_frobenius):
                reasons.append("missing_projector_frobenius_exceeds_predeclared_gate")
            passed = not reasons
            n_total += 1
            n_passed += int(passed)
            replicates.append(
                {
                    "replicate": int(replicate),
                    "native_identifiable_rank": int(subspace.identifiable_rank),
                    "truncated_to_five": False,
                    "mode_persistence": persistences,
                    "min_mode_persistence": contained["min_mode_persistence"],
                    "max_principal_angle_deg": contained["max_principal_angle_deg"],
                    "missing_projector_frobenius": contained["missing_projector_frobenius"],
                    "pass": passed,
                    "failure_reasons": reasons,
                }
            )
        per_source.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "n_replicates_kept": int(len(replicates)),
                "n_passed": int(sum(1 for row in replicates if row["pass"])),
                "replicates": replicates,
            }
        )
    return {
        "kind": "event_bootstrap_core_persistence",
        "n_sources": int(len(banks)),
        "n_replicates_per_source_requested": int(n_replicates),
        "n_replicates_total": int(n_total),
        "n_passed": int(n_passed),
        "pass_fraction": float(n_passed) / float(n_total) if n_total else 0.0,
        "per_source": per_source,
    }


def independent_core_gate(
    independent_rows: Sequence[Mapping[str, Any]],
    hypothesis_core: Mapping[str, Any],
    *,
    min_mode_persistence: float,
    min_source_support_fraction: float,
    max_principal_angle_deg: float,
    max_missing_frobenius: float,
) -> dict[str, Any]:
    comparisons = []
    hits = 0
    for row in independent_rows:
        contained = core_contained_in_projector(hypothesis_core["v_core"], row["projector"])
        persistences = [float(value) for value in contained["mode_persistence"]]
        support = (
            1.0
            if persistences and min(persistences) >= float(min_mode_persistence)
            else 0.0
        )
        reasons = []
        if persistences and min(persistences) < float(min_mode_persistence):
            reasons.append("mode_persistence_below_predeclared_gate")
        if float(contained["max_principal_angle_deg"]) > float(max_principal_angle_deg):
            reasons.append("principal_angle_exceeds_predeclared_gate")
        if float(contained["missing_projector_frobenius"]) > float(max_missing_frobenius):
            reasons.append("missing_projector_frobenius_exceeds_predeclared_gate")
        passed = not reasons
        hits += int(passed)
        comparisons.append(
            {
                "source_id": row["source_id"],
                "split": row["split"],
                "native_identifiable_rank": row["identifiable_rank"],
                "truncated_to_five": False,
                "mode_persistence": persistences,
                "min_mode_persistence": contained["min_mode_persistence"],
                "max_principal_angle_deg": contained["max_principal_angle_deg"],
                "missing_projector_frobenius": contained["missing_projector_frobenius"],
                "pass": passed,
                "failure_reasons": reasons,
                "hypothesis_sources_are_not_used_here": True,
            }
        )
    fraction = float(hits) / float(len(independent_rows)) if independent_rows else 0.0
    return {
        "n_sources": int(len(independent_rows)),
        "n_passed": int(hits),
        "source_support_fraction": fraction,
        "min_source_support_fraction": float(min_source_support_fraction),
        "pass": bool(independent_rows) and fraction >= float(min_source_support_fraction),
        "comparisons": comparisons,
        "hypothesis_sources_are_not_confirmatory_evidence": True,
        "sealed_test_not_opened": True,
    }


def decide_campaign(
    *,
    hypothesis_core_dimension: int,
    hypothesis_stable: bool,
    independent: Mapping[str, Any] | None,
    three_arm: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if int(hypothesis_core_dimension) < 1 or not hypothesis_stable:
        decision = DECISION_HYPOTHESIS_FAIL
        authorize = False
        reasons = [
            "hypothesis_stable_core_failed",
            "independent_validation_cannot_override_unstable_hypothesis",
            "three_arm_not_opened",
            "do_not_retune_scale_rank_selection_or_drop_source",
        ]
        independent_pass = None if independent is None else bool(independent.get("pass"))
    elif independent is None or not bool(independent.get("pass")):
        decision = DECISION_INDEPENDENT_FAIL
        authorize = False
        independent_pass = False if independent is not None else None
        reasons = [
            "hypothesis_stable_core_constructed",
            "independent_validation_failed_or_missing",
            "three_arm_not_opened",
            "do_not_retune_scale_rank_selection_or_drop_source",
        ]
    elif three_arm is None or not bool(three_arm.get("pass")):
        decision = DECISION_THREE_ARM_FAIL
        authorize = False
        independent_pass = True
        reasons = [
            "independent_stable_core_pass",
            "three_arm_on_v_core_failed_or_missing",
            "do_not_retune_scale_rank_selection_or_drop_source",
        ]
    else:
        decision = DECISION_INDEPENDENT_PASS
        authorize = True
        independent_pass = True
        reasons = [
            "independent_stable_core_pass",
            "three_arm_used_frozen_v_core_not_v1_pooled_v_id",
        ]
    return {
        "decision": decision,
        "inherited_v1_decision": INHERITED_V1_DECISION,
        "core_dimension": int(hypothesis_core_dimension),
        "hypothesis_core_stable": bool(hypothesis_stable),
        "independent_validation_pass": independent_pass,
        "null_injection_leakage_gate": None if three_arm is None else bool(three_arm.get("null_injection_leakage_gate")),
        "mixed_injection_projected_closure": None
        if three_arm is None
        else bool(three_arm.get("mixed_injection_projected_closure")),
        "authorize_frozen_v2_unknown_association_closure": False,
        "real_data_alignment_correction_authorized": False,
        "geometry_write_allowed": False,
        "three_arm_authorized": bool(authorize) and decision == DECISION_INDEPENDENT_PASS,
        "used_v1_pooled_v_id": False,
        "cluster_local_repair_is_not_a_gate": True,
        "survey_is_external_cross_check_only": True,
        "reasons": reasons,
        "if_failed_do_not_chase_by_retuning": True,
    }


def _core_json(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(core)
    payload["v_core"] = [[float(item) for item in row] for row in np.asarray(core["v_core"]).T]
    payload["v_orthogonal"] = [[float(item) for item in row] for row in np.asarray(core["v_orthogonal"]).T]
    payload["projector_core"] = [[float(item) for item in row] for row in core["projector_core"]]
    payload["core_space"] = subspace_as_json(core["core_space"])
    payload["mode_persistence_by_mode"] = [
        [float(value) for value in row] for row in core["mode_persistence_by_mode"]
    ]
    payload["source_support_fraction_by_mode"] = [
        float(value) for value in core["source_support_fraction_by_mode"]
    ]
    return payload


def build_all_reports(config: Mapping[str, Any], *, banks_by_role: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    root = project_root()
    created = datetime.now(timezone.utc).isoformat()
    rank_tolerance = float(config["rank_tolerance"])
    rcond = float(config["normal_matrix_rcond"])
    core_cfg = config["core_definition"]
    gates = config["gates"]
    if banks_by_role is None:
        hypothesis_banks = load_role_banks(config, role="hypothesis")
        independent_banks = load_role_banks(config, role="independent")
    else:
        hypothesis_banks = list(banks_by_role["hypothesis"])
        independent_banks = list(banks_by_role.get("independent", []))
    hypothesis_rows = source_projectors(hypothesis_banks, rank_tolerance=rank_tolerance, rcond=rcond)
    hypothesis_core = build_core_from_rows(
        hypothesis_rows,
        min_consensus_eigenvalue=float(core_cfg["min_consensus_eigenvalue"]),
        min_mode_persistence=float(core_cfg["min_mode_persistence"]),
        min_source_support_fraction=float(core_cfg["min_hypothesis_source_support_fraction"]),
    )
    loso = loso_core_stability(
        hypothesis_rows,
        hypothesis_core,
        min_consensus_eigenvalue=float(core_cfg["min_consensus_eigenvalue"]),
        min_mode_persistence=float(core_cfg["min_mode_persistence"]),
        min_source_support_fraction=float(core_cfg["min_hypothesis_source_support_fraction"]),
        max_principal_angle_deg=float(gates["max_loso_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_loso_projector_frobenius"]),
        require_same_dimension=bool(gates["require_loso_same_core_dimension"]),
    )
    bootstrap_cfg = config["bootstrap"]
    source_bootstrap = source_bootstrap_core_stability(
        hypothesis_rows,
        hypothesis_core,
        n_replicates=int(bootstrap_cfg["n_source_replicates"]),
        seed=int(bootstrap_cfg["seed"]),
        min_consensus_eigenvalue=float(core_cfg["min_consensus_eigenvalue"]),
        min_mode_persistence=float(core_cfg["min_mode_persistence"]),
        min_source_support_fraction=float(core_cfg["min_hypothesis_source_support_fraction"]),
        max_principal_angle_deg=float(gates["max_hypothesis_bootstrap_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_hypothesis_bootstrap_projector_frobenius"]),
    )
    event_bootstrap = event_bootstrap_core_persistence(
        hypothesis_banks,
        hypothesis_core,
        n_replicates=int(bootstrap_cfg["n_event_replicates_per_source"]),
        seed=int(bootstrap_cfg["seed"]),
        rank_tolerance=rank_tolerance,
        rcond=rcond,
        min_pairs=int(bootstrap_cfg["min_pairs"]),
        min_mode_persistence=float(core_cfg["min_mode_persistence"]),
        max_principal_angle_deg=float(gates["max_hypothesis_bootstrap_principal_angle_deg"]),
        max_missing_frobenius=float(gates["max_hypothesis_bootstrap_projector_frobenius"]),
    )
    hypothesis_stable = bool(
        int(hypothesis_core["core_dimension"]) >= int(gates["require_hypothesis_core_dimension_at_least"])
        and loso["stable"]
        and source_bootstrap["stable"]
    )
    independent_rows = source_projectors(
        independent_banks, rank_tolerance=rank_tolerance, rcond=rcond
    )
    independent = independent_core_gate(
        independent_rows,
        hypothesis_core,
        min_mode_persistence=float(gates["min_independent_mode_persistence"]),
        min_source_support_fraction=float(gates["min_independent_source_support_fraction"]),
        max_principal_angle_deg=float(gates["max_independent_core_principal_angle_deg"]),
        max_missing_frobenius=float(gates["max_independent_core_missing_frobenius"]),
    )
    independent["hypothesis_loso_stable"] = bool(loso["stable"])
    independent["hypothesis_source_bootstrap_stable"] = bool(source_bootstrap["stable"])
    independent["hypothesis_core_stable"] = bool(hypothesis_stable)
    independent["cannot_override_unstable_hypothesis_core"] = True
    three_arm = None
    if hypothesis_stable and independent["pass"] and independent_rows:
            generating = independent_rows[0]["extras"]["weighted_matrix"]
            generating_label = f"independent:{independent_rows[0]['source_id']}"
            injections = three_arm_payloads(
                hypothesis_core["core_space"],
                identifiable_scaled_amplitude=float(config["injections"]["identifiable_scaled_amplitude"]),
                null_scaled_amplitude=float(config["injections"]["null_scaled_amplitude"]),
                seed=int(config["injections"]["seed"]),
                n_replicates=int(config["injections"]["n_replicates"]),
            )
            three_arm = evaluate_three_arms(
                hypothesis_core["core_space"],
                generating,
                injections,
                gates=gates,
            )
            three_arm["solver_basis"] = "frozen_v_core"
            three_arm["used_v1_pooled_v_id"] = False
            three_arm["athena_not_rerun"] = True
            three_arm["association_control"] = "truth_selected_physical_edge"
            three_arm["generating_jacobian"] = generating_label
            three_arm["arm_b_injects_core_orthogonal_not_v1_null"] = True
            three_arm["frozen_v2_unknown_association_not_opened"] = True
    decision = decide_campaign(
        hypothesis_core_dimension=int(hypothesis_core["core_dimension"]),
        hypothesis_stable=hypothesis_stable,
        independent=independent,
        three_arm=three_arm,
    )
    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": _operating_state(),
            "inherited_v1_decision": INHERITED_V1_DECISION,
            "parameter_names": list(hypothesis_core["parameter_names"]),
            "parameter_units": list(hypothesis_core["parameter_units"]),
            "scale_matrix_S": list(hypothesis_core["parameter_scales"]),
            "scale_convention": "frozen_severity_scale_u_equals_theta_over_S",
            "residual_weight": "nominal_WLS_inverse_4x4_pair_covariance",
            "weighted_matrix": "A = W^{1/2} J S",
            "rank_tolerance": float(rank_tolerance),
            "rank_tolerance_source": "frozen LEAKAGE_RANK_RELATIVE_TOLERANCE = 1e-2",
            "normal_matrix_rcond": float(rcond),
            "algorithm": dict(config["algorithm"]),
            "core_definition": dict(core_cfg),
            "hypothesis_source_ids": list(config["jacobian_corpus"]["hypothesis_source_ids"]),
            "independent_validation_source_ids": list(
                config["jacobian_corpus"]["independent_validation_source_ids"]
            ),
            "hypothesis_sources_are_not_confirmatory_evidence": True,
            "forbidden": {
                "naked_mixed_unit_svd": True,
                "scale_retune_from_singular_values": True,
                "rank_threshold_retune_from_spectrum": True,
                "force_identifiable_rank_five": True,
                "truncate_rank_six_source": True,
                "drop_v1_failure_source": True,
                "full_parameter_newton": True,
                "real_data_alignment_correction": True,
                "geometry_write": True,
                "sealed_test": True,
                "remove_dz_to_improve_this_svd": True,
            },
            "assumptions": list(config.get("assumptions", [])),
        },
        "hypothesis_core": _core_json(hypothesis_core),
        "loso_stability": loso,
        "source_bootstrap": source_bootstrap,
        "event_bootstrap": event_bootstrap,
        "independent_validation": independent,
        "three_arm_closure": three_arm,
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
    return {
        "refused_naked_mixed_unit_svd": naked,
        "geometry_write_allowed": False,
        "full_parameter_newton": False,
        "real_data_alignment_correction": False,
        "truncated_rank_six_source": False,
        "used_v1_pooled_v_id": False,
        "opened_sealed_test": False,
    }
