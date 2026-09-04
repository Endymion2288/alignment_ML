"""Tracker-only identifiable-subspace definition and three-arm closure V1.

Survey/metrology is an external cross-check.  Frozen V2 / association policy
stay unchanged.  Real data stays residual-DQ-only.  Geometry write is false.

The first-round solver control is truth-selected association on the already
validated physical finite-difference Jacobian.  Unknown-association Frozen-V2
closure is not opened here.  Linear three-arm injections are formed in the
identifiable / null coordinates of ``A = W^{1/2} J S``; Athena is not rerun
and no alignment payload is written.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.five_dof_sampling import LINEAR_SEVERITY_MAX
from alignment.identifiable_subspace import (
    CLUSTER_LOCAL_NATIVE_NAMES,
    CLUSTER_LOCAL_SCALE_MAP,
    FROZEN_RANK_TOLERANCE,
    IdentifiableSubspace,
    MixedUnitNakedJacobianError,
    closure_metrics,
    flatten_diagonal_jacobian,
    flatten_physical_jacobian,
    frozen_scales_for,
    frozen_units_for,
    identifiable_svd,
    inject_identifiable,
    inject_null,
    native_to_scaled,
    refuse_rank_threshold_from_spectrum,
    refuse_scale_or_threshold_retune,
    solve_identifiable_amplitudes,
    subspace_as_json,
    subspace_distance,
    svd_naked_jacobian,
)
from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.physical_jacobian import solve_physical_finite_difference
from alignment.true_cluster_local_residual import (
    RESIDUAL_KIND,
    assert_no_alignment_payload,
    common_operating_state,
)
from alignment.true_cluster_local_stability_transfer import (
    build_jacobian,
    fd_steps,
    load_audit_config as load_cluster_local_config,
    load_run_measurements,
)
from alignment.physical_jacobian import parameter_values_from_payload
from alignment.layer_hierarchy import IFT_LAYER_IDS
from scripts.audit_6dof_identifiability import _complete_route_mask, _masked_bank, _source_entries
from scripts.audit_layer_identifiability import (
    _aligned_rows,
    _evaluation,
    _layer_weights_from_hit_pattern,
    _ordered_fd_points,
    _payload_for_point,
    _pool_layer_banks,
)
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-tracker-only-identifiable-subspace-three-arm-closure-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "tracker_only_identifiable_subspace_three_arm_closure_v1.yaml"
DECISION_PASS = "tracker_only_identifiable_subspace_stable_and_three_arm_closure_pass"
DECISION_BASIS_UNSTABLE = "tracker_only_identifiable_basis_unstable_solve_stopped"
DECISION_ARM_FAIL = "tracker_only_identifiable_basis_stable_but_three_arm_closure_fail"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_identifiable_modes",
    "survey_is_alignment_input",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
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
    "solver_restricted_to_identifiable_subspace",
    "truth_selected_association_is_solver_control",
    "survey_is_external_cross_check_only",
    "residual_reduction_is_not_alignment_success",
    "implied_cdx_is_not_a_measurement",
)


def load_campaign_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"identifiable-subspace config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected identifiable-subspace schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"identifiable-subspace config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"identifiable-subspace config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the true cluster-local residual label")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    rank_tolerance = float(payload["rank_tolerance"])
    if rank_tolerance != float(FROZEN_RANK_TOLERANCE):
        refuse_rank_threshold_from_spectrum()
    if payload.get("retune_scales_from_singular_values") is True:
        refuse_scale_or_threshold_retune(from_singular_values=True)
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
        }
    )
    return state


def _fit_bank(bank: Mapping[str, Any], *, rcond: float) -> Any:
    return solve_physical_finite_difference(
        bank["anchor_residual"],
        bank["positive_residual"],
        bank["negative_residual"],
        bank["reference_residual"],
        bank["covariance"],
        parameter_names=tuple(bank["names"]),
        positive_values=bank["positive_values"],
        negative_values=bank["negative_values"],
        parameter_scales=np.asarray(bank["scales"], dtype=np.float64),
        rcond=float(rcond),
    )


def subspace_from_physical_bank(
    bank: Mapping[str, Any],
    *,
    rank_tolerance: float = FROZEN_RANK_TOLERANCE,
    rcond: float = 1.0e-10,
    pair_mask: np.ndarray | None = None,
) -> tuple[IdentifiableSubspace, dict[str, Any]]:
    work = bank if pair_mask is None else _masked_bank(bank, np.asarray(pair_mask, dtype=bool))
    names = tuple(str(name) for name in work["names"])
    declared = np.asarray(work["scales"], dtype=np.float64)
    frozen = frozen_scales_for(names)
    if not np.allclose(declared, frozen, rtol=0.0, atol=0.0):
        raise ValueError(
            "physical-scan severity_scale must match the frozen scale matrix S; "
            f"declared={declared.tolist()} frozen={frozen.tolist()}"
        )
    units = frozen_units_for(names)
    fit = _fit_bank(work, rcond=rcond)
    weighted, flat, blocks = flatten_physical_jacobian(
        fit.derivative_native, work["covariance"], frozen
    )
    subspace = identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=units,
        parameter_scales=frozen,
        rank_tolerance=rank_tolerance,
    )
    extras = {
        "fit_normal_matrix_rank": int(fit.normal_matrix_rank),
        "n_pairs": int(fit.used_pairs),
        "source_id": str(work.get("source_id", "")),
        "split": str(work.get("split", "")),
        "weighted_matrix": weighted,
        "flat_jacobian": flat,
        "weight_blocks": blocks,
        "fit": fit,
    }
    return subspace, extras


def load_fd_only_bank(
    entry: Mapping[str, object],
    *,
    anchor_point: str,
    min_truth_match_fraction: float,
    only_parameters: Sequence[str],
    evaluation_fn: Any = None,
) -> dict[str, Any]:
    """Load reference + axial FD probes.  Held-out physical points are not needed.

    Three-arm closure is linear on the validated Jacobian, so the unused
    ``reference_residual`` is the nominal residual.  Truth is used only to
    keep pair identities fixed.

    ``evaluation_fn`` defaults to the frozen ``_evaluation`` chain.  The
    workbook-76 merged-rec K-short campaign injects its occurrence-augmented
    physical-event-identity evaluation through this hook; canonical banks
    must keep the default so their behaviour stays bit-identical.
    """
    source_id = str(entry["source_id"])
    root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    specs, names, scales, movable, ordered, _targets = _ordered_fd_points(
        plan,
        anchor_point=anchor_point,
        target_points=(),
        only_parameters=tuple(str(name) for name in only_parameters),
        require_targets=False,
    )
    evaluate = _evaluation if evaluation_fn is None else evaluation_fn
    payloads = []
    evaluations = []
    for point in ordered:
        payload, tracklets, propagations = _payload_for_point(root, point)
        payloads.append(payload)
        evaluations.append(evaluate(tracklets, propagations, min_truth_match_fraction))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    nested_layers = [
        {
            str(station): {
                str(layer): list(payload.transform_for_layer(int(station), int(layer)))
                for layer in IFT_LAYER_IDS
            }
            for station in (0,)
        }
        for payload in payloads
    ]
    anchor_values = parameter_values_from_payload(specs, payloads[0].transforms, nested_layers[0])
    parameters = len(names)
    positive_values = np.asarray(
        [
            parameter_values_from_payload(
                specs, payloads[1 + 2 * index].transforms, nested_layers[1 + 2 * index]
            )[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    negative_values = np.asarray(
        [
            parameter_values_from_payload(
                specs, payloads[2 + 2 * index].transforms, nested_layers[2 + 2 * index]
            )[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    anchor_evaluation = evaluations[0]
    anchor_rows = rows[0]
    anchor_residual = np.asarray(anchor_evaluation.residual[anchor_rows], dtype=np.float64)
    return {
        "source_id": source_id,
        "split": str(entry.get("split", "")),
        "specs": specs,
        "names": names,
        "scales": scales,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "reference_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": positive_values,
        "negative_values": negative_values,
        "anchor_residual": anchor_residual,
        "positive_residual": np.asarray(
            [evaluations[1 + 2 * index].residual[rows[1 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "negative_residual": np.asarray(
            [evaluations[2 + 2 * index].residual[rows[2 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "reference_residual": np.array(anchor_residual, copy=True),
        "covariance": np.asarray(anchor_evaluation.combined_covariance[anchor_rows], dtype=np.float64),
        "run_id": np.asarray(anchor_evaluation.run_id[anchor_rows], dtype=np.int64),
        "event_id": np.asarray(anchor_evaluation.event_id[anchor_rows], dtype=np.int64),
        "source_tracklet_id": np.asarray(anchor_evaluation.source_tracklet_id[anchor_rows], dtype=np.int32),
        "target_tracklet_id": np.asarray(anchor_evaluation.target_tracklet_id[anchor_rows], dtype=np.int32),
        "truth_particle_id": np.asarray(anchor_evaluation.truth_particle_id[anchor_rows], dtype=np.int64),
        "source_station_id": np.asarray(anchor_evaluation.source_station_id[anchor_rows], dtype=np.int64),
        "target_station_id": np.asarray(anchor_evaluation.target_station_id[anchor_rows], dtype=np.int64),
        "overlap": overlap,
        "layer_weights": _layer_weights_from_hit_pattern(anchor_evaluation, anchor_rows),
        "held_out_physical_points_loaded": False,
        "truth_used_only_to_fix_pair_identities": True,
        "test_data_accessed": False,
    }


def load_physical_banks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    corpus = config["jacobian_corpus"]
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    manifest = _read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("campaign requires q_over_p_mode=0")
    entries = _source_entries(manifest)
    allowed = tuple(str(item) for item in corpus.get("splits", ("train", "validation")))
    selected_ids = corpus.get("source_ids")
    banks = []
    for entry in entries:
        if str(entry.get("split", "")) not in allowed:
            continue
        if selected_ids is not None and str(entry["source_id"]) not in set(selected_ids):
            continue
        banks.append(
            load_fd_only_bank(
                entry,
                anchor_point=str(corpus["anchor_point"]),
                min_truth_match_fraction=float(corpus["min_truth_match_fraction"]),
                only_parameters=tuple(str(name) for name in corpus["parameter_names"]),
            )
        )
    if not banks:
        raise ValueError("no physical Jacobian banks matched the campaign config")
    return banks


def _pool_campaign_banks(banks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prepared = []
    for bank in banks:
        item = dict(bank)
        item.setdefault("specs", ())
        item.setdefault("target_names", ())
        item.setdefault("target_values", {})
        item.setdefault("target_residuals", {})
        if "layer_weights" not in item:
            item["layer_weights"] = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        prepared.append(item)
    return _pool_layer_banks(prepared)


def _pair_event_groups(bank: Mapping[str, Any]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    runs = np.asarray(bank["run_id"])
    events = np.asarray(bank["event_id"])
    for index, (run, event) in enumerate(zip(runs.tolist(), events.tolist())):
        groups.setdefault((int(run), int(event)), []).append(int(index))
    return groups


def bootstrap_subspaces(
    bank: Mapping[str, Any],
    *,
    n_replicates: int,
    seed: int,
    rank_tolerance: float,
    rcond: float,
    min_pairs: int,
) -> list[IdentifiableSubspace]:
    groups = list(_pair_event_groups(bank).values())
    if len(groups) < 2:
        raise ValueError("event bootstrap needs at least two events")
    rng = np.random.default_rng(int(seed))
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    rows: list[IdentifiableSubspace] = []
    for _ in range(int(n_replicates)):
        choice = rng.integers(0, len(groups), size=len(groups))
        selected = np.zeros(n_pairs, dtype=bool)
        for index in choice:
            selected[groups[int(index)]] = True
        if int(selected.sum()) < int(min_pairs):
            continue
        subspace, _extras = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=selected
        )
        rows.append(subspace)
    if not rows:
        raise ValueError("bootstrap produced no valid subspaces")
    return rows


def half_split_subspaces(
    bank: Mapping[str, Any],
    *,
    n_splits: int,
    seed: int,
    rank_tolerance: float,
    rcond: float,
    min_pairs: int,
) -> list[tuple[IdentifiableSubspace, IdentifiableSubspace]]:
    groups = list(_pair_event_groups(bank).values())
    if len(groups) < 4:
        raise ValueError("half-splits need at least four events")
    rng = np.random.default_rng(int(seed) + 17)
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    rows: list[tuple[IdentifiableSubspace, IdentifiableSubspace]] = []
    half = len(groups) // 2
    for _ in range(int(n_splits)):
        order = rng.permutation(len(groups))
        left_idx = order[:half]
        right_idx = order[half:]
        left_mask = np.zeros(n_pairs, dtype=bool)
        right_mask = np.zeros(n_pairs, dtype=bool)
        for index in left_idx:
            left_mask[groups[int(index)]] = True
        for index in right_idx:
            right_mask[groups[int(index)]] = True
        if int(left_mask.sum()) < int(min_pairs) or int(right_mask.sum()) < int(min_pairs):
            continue
        left, _ = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=left_mask
        )
        right, _ = subspace_from_physical_bank(
            bank, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=right_mask
        )
        rows.append((left, right))
    if not rows:
        raise ValueError("half-splits produced no valid subspaces")
    return rows


def summarize_stability(
    reference: IdentifiableSubspace,
    others: Sequence[tuple[str, IdentifiableSubspace]],
    *,
    max_principal_angle_deg: float,
    max_projector_frobenius: float,
    require_same_rank: bool,
) -> dict[str, Any]:
    comparisons = []
    unstable = []
    for label, other in others:
        distance = subspace_distance(reference, other)
        distance["label"] = label
        same_rank = bool(distance["same_rank"])
        angle = distance["max_identifiable_principal_angle_deg"]
        projector = float(distance["projector_frobenius_distance"])
        ok = True
        reasons = []
        if require_same_rank and not same_rank:
            ok = False
            reasons.append("identifiable_rank_changed")
        if same_rank and angle is not None and float(angle) > float(max_principal_angle_deg):
            ok = False
            reasons.append("principal_angle_exceeds_predeclared_gate")
        if projector > float(max_projector_frobenius):
            ok = False
            reasons.append("projector_frobenius_exceeds_predeclared_gate")
        distance["stable"] = ok
        distance["failure_reasons"] = reasons
        comparisons.append(distance)
        if not ok:
            unstable.append(label)
    max_angle = None
    finite_angles = [
        float(row["max_identifiable_principal_angle_deg"])
        for row in comparisons
        if row["max_identifiable_principal_angle_deg"] is not None
    ]
    if finite_angles:
        max_angle = max(finite_angles)
    return {
        "n_comparisons": int(len(comparisons)),
        "unstable_labels": unstable,
        "stable": not unstable,
        "max_principal_angle_deg": max_angle,
        "max_projector_frobenius_distance": max(
            float(row["projector_frobenius_distance"]) for row in comparisons
        )
        if comparisons
        else None,
        "compares_subspace_not_signed_vector_elements": True,
        "comparisons": comparisons,
        "gates": {
            "max_principal_angle_deg": float(max_principal_angle_deg),
            "max_projector_frobenius": float(max_projector_frobenius),
            "require_same_identifiable_rank": bool(require_same_rank),
        },
    }


def three_arm_payloads(
    subspace: IdentifiableSubspace,
    *,
    identifiable_scaled_amplitude: float,
    null_scaled_amplitude: float,
    seed: int,
    n_replicates: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rank = subspace.identifiable_rank
    null_dim = subspace.null_dimension
    rows = []
    for index in range(int(n_replicates)):
        if rank:
            identifiable_dir = rng.normal(size=rank)
            identifiable_dir /= float(np.linalg.norm(identifiable_dir)) or 1.0
            identifiable = identifiable_dir * float(identifiable_scaled_amplitude)
        else:
            identifiable = np.zeros(0, dtype=np.float64)
        if null_dim:
            null_dir = rng.normal(size=null_dim)
            null_dir /= float(np.linalg.norm(null_dir)) or 1.0
            null = null_dir * float(null_scaled_amplitude)
        else:
            null = np.zeros(0, dtype=np.float64)
        theta_a = inject_identifiable(subspace, identifiable)
        theta_b = inject_null(subspace, null)
        theta_c = theta_a + theta_b
        rows.append(
            {
                "replicate": int(index),
                "arm_a_identifiable_amplitudes": identifiable,
                "arm_b_null_amplitudes": null,
                "theta_a_native": theta_a,
                "theta_b_native": theta_b,
                "theta_c_native": theta_c,
            }
        )
    return rows


def _solve_injection(
    subspace: IdentifiableSubspace,
    generating_weighted: np.ndarray,
    theta_native: np.ndarray,
) -> dict[str, Any]:
    scaled = native_to_scaled(theta_native, subspace.parameter_scales)
    weighted_residual = np.asarray(generating_weighted, dtype=np.float64) @ scaled
    solved = solve_identifiable_amplitudes(
        subspace,
        weighted_residual,
        generating_weighted=generating_weighted,
    )
    metrics = closure_metrics(
        subspace,
        q_hat_native=solved["native"],
        q_truth_native=theta_native,
        weighted_residual=weighted_residual,
        predicted_weighted_residual=solved["predicted_weighted_residual"],
    )
    scaled_hat = native_to_scaled(solved["native"], subspace.parameter_scales)
    null_of_hat = (
        subspace.v_null.T @ scaled_hat if subspace.null_dimension else np.zeros(0, dtype=np.float64)
    )
    if subspace.null_dimension and float(np.linalg.norm(null_of_hat)) > 1.0e-10:
        raise ValueError("identifiable solver leaked a null component; V_null^T u_hat must be 0 by construction")
    return {
        "q_hat_native": solved["native"],
        "amplitudes": solved["amplitudes"],
        "metrics": metrics,
        "null_component_of_hat_scaled_norm": float(np.linalg.norm(null_of_hat)),
        "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
    }


def evaluate_three_arms(
    subspace: IdentifiableSubspace,
    generating_weighted: np.ndarray,
    payloads: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    arm_a_rows = []
    arm_b_rows = []
    arm_c_rows = []
    for payload in payloads:
        a = _solve_injection(subspace, generating_weighted, np.asarray(payload["theta_a_native"]))
        b = _solve_injection(subspace, generating_weighted, np.asarray(payload["theta_b_native"]))
        c = _solve_injection(subspace, generating_weighted, np.asarray(payload["theta_c_native"]))
        if subspace.identifiable_rank:
            amplitude_error = float(
                np.linalg.norm(
                    np.asarray(a["amplitudes"]) - np.asarray(payload["arm_a_identifiable_amplitudes"])
                )
            )
        else:
            amplitude_error = 0.0
        a["amplitude_error_norm"] = amplitude_error
        a["projected_truth_recovered"] = bool(
            (a["metrics"]["projected_identifiable_error_relative_to_projected_truth"] or 0.0)
            <= float(gates["arm_a_max_projected_error_relative"])
            and amplitude_error <= float(gates["arm_a_max_amplitude_error"])
        )
        injected_null = float(np.linalg.norm(native_to_scaled(payload["theta_b_native"], subspace.parameter_scales)))
        fake = float(a_to_identifiable_norm(b, subspace))
        b["identifiable_fake_norm"] = fake
        b["identifiable_fake_relative_to_null_injection"] = (
            None if injected_null <= 0.0 else fake / injected_null
        )
        b["no_significant_identifiable_fake_correction"] = bool(
            (b["identifiable_fake_relative_to_null_injection"] or 0.0)
            <= float(gates["arm_b_max_identifiable_fake_relative"])
        )
        c["projected_closure_ok"] = bool(
            (c["metrics"]["projector_closure_relative_to_projected_truth"] or 0.0)
            <= float(gates["arm_c_max_projector_closure_relative"])
        )
        arm_a_rows.append(a)
        arm_b_rows.append(b)
        arm_c_rows.append(c)

    def _all(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
        return bool(rows) and all(bool(row[key]) for row in rows)

    arm_a_pass = _all(arm_a_rows, "projected_truth_recovered")
    arm_b_pass = _all(arm_b_rows, "no_significant_identifiable_fake_correction")
    arm_c_pass = _all(arm_c_rows, "projected_closure_ok")
    observable = all(
        bool(row["metrics"].get("postfit_observable_improved"))
        for row in arm_a_rows + arm_c_rows
    )
    return {
        "n_replicates": int(len(payloads)),
        "arm_a_identifiable_injection": {
            "pass": arm_a_pass,
            "max_projected_error_relative": _max_metric(
                arm_a_rows, "projected_identifiable_error_relative_to_projected_truth"
            ),
            "max_amplitude_error_norm": max(float(row["amplitude_error_norm"]) for row in arm_a_rows)
            if arm_a_rows
            else None,
            "replicates": [_arm_row_json(row) for row in arm_a_rows],
        },
        "arm_b_null_injection": {
            "pass": arm_b_pass,
            "max_identifiable_fake_relative": _max_metric(
                arm_b_rows, "identifiable_fake_relative_to_null_injection"
            ),
            "replicates": [_arm_row_json(row) for row in arm_b_rows],
            "tracker_only_solver_must_not_invent_identifiable_fake_correction": True,
        },
        "arm_c_mixed_injection": {
            "pass": arm_c_pass,
            "max_projector_closure_relative": _max_metric(
                arm_c_rows, "projector_closure_relative_to_projected_truth"
            ),
            "does_not_require_full_physical_truth_recovery": True,
            "requires_q_hat_approx_P_id_q_truth": True,
            "replicates": [_arm_row_json(row) for row in arm_c_rows],
        },
        "postfit_observable_improved_on_identifiable_and_mixed_arms": observable,
        "null_injection_leakage_gate": arm_b_pass,
        "mixed_injection_projected_closure": arm_c_pass,
        "linear_severity_envelope": float(LINEAR_SEVERITY_MAX),
        "per_physical_parameter_truth_error_is_not_the_gate": True,
        "pass": bool(arm_a_pass and arm_b_pass and arm_c_pass and observable),
    }


def a_to_identifiable_norm(row: Mapping[str, Any], subspace: IdentifiableSubspace) -> float:
    hat = native_to_scaled(row["q_hat_native"], subspace.parameter_scales)
    if subspace.identifiable_rank == 0:
        return 0.0
    return float(np.linalg.norm(subspace.v_id.T @ hat))


def _max_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        metrics = row.get("metrics", row)
        value = metrics.get(key, row.get(key))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return max(values) if values else None


def _arm_row_json(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "q_hat_native": [float(value) for value in np.asarray(row["q_hat_native"])],
        "amplitudes": [float(value) for value in np.asarray(row["amplitudes"])],
        "metrics": dict(row["metrics"]),
        "null_component_of_hat_scaled_norm": float(row["null_component_of_hat_scaled_norm"]),
        "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
    }
    for key in (
        "amplitude_error_norm",
        "projected_truth_recovered",
        "identifiable_fake_norm",
        "identifiable_fake_relative_to_null_injection",
        "no_significant_identifiable_fake_correction",
        "projected_closure_ok",
    ):
        if key in row:
            value = row[key]
            payload[key] = value if not hasattr(value, "tolist") else json_ready(value)
    return payload


def cluster_local_subspace(
    measurements: Sequence[Any],
    *,
    station_id: int,
    steps: Mapping[str, float],
    rank_tolerance: float,
) -> IdentifiableSubspace:
    names = CLUSTER_LOCAL_NATIVE_NAMES
    jacobian = np.asarray(
        build_jacobian(measurements, station_id=int(station_id), steps=steps),
        dtype=np.float64,
    )
    n_measurements = int(len(measurements))
    if jacobian.ndim != 2 or jacobian.shape != (n_measurements, 3) or n_measurements < 1:
        raise ValueError(
            "cluster-local Jacobian must have shape [n_measurements, 3] with "
            f"n_measurements>=1; got shape={getattr(jacobian, 'shape', None)} "
            f"n_measurements={n_measurements}"
        )
    # parameter_columns differentiates ry with a radian step, so convert to /mrad.
    jacobian = np.array(jacobian, copy=True)
    jacobian[:, 1] *= 0.001
    units = frozen_units_for(names)
    scales = frozen_scales_for(names, CLUSTER_LOCAL_SCALE_MAP)
    variance = np.asarray([float(item.local_u_var_mm2) for item in measurements], dtype=np.float64)
    weighted, _flat, _weights = flatten_diagonal_jacobian(jacobian, variance, scales)
    return identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=units,
        parameter_scales=scales,
        rank_tolerance=rank_tolerance,
    )


def load_cluster_local_subspaces(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    cluster_cfg = load_cluster_local_config(
        resolve_under_root(root, str(config["cluster_local"]["stability_config"]))
    )
    steps = fd_steps(cluster_cfg)
    station_id = int(cluster_cfg["representative"]["station_id"])
    dump = cluster_cfg["entry_57"]["cluster_dump"]
    loaded: dict[str, Any] = {}
    role_notes: dict[str, Any] = {}
    for role, spec in (
        ("reference", cluster_cfg["runs"]["reference"]),
        ("transfer", cluster_cfg["runs"]["transfer"]),
    ):
        try:
            bundle = load_run_measurements(
                spec,
                root=root,
                cluster_dump=spec.get("cluster_dump", dump),
                representative_station=station_id,
            )
            n_measurements = int(len(bundle["measurements"]))
            if n_measurements < 1:
                role_notes[role] = {
                    "run": int(spec["run"]),
                    "n_measurements": 0,
                    "reason": "no_cluster_local_measurements_after_join",
                }
                continue
            subspace = cluster_local_subspace(
                bundle["measurements"],
                station_id=station_id,
                steps=steps,
                rank_tolerance=float(config["rank_tolerance"]),
            )
            loaded[role] = {
                "run": int(spec["run"]),
                "n_measurements": n_measurements,
                "subspace": subspace_as_json(subspace),
                "_subspace": subspace,
                "join_complete": bool(bundle["join_complete"]),
            }
        except (FileNotFoundError, OSError, ValueError) as error:
            role_notes[role] = {"run": int(spec["run"]), "reason": str(error)}
    if "reference" not in loaded or "transfer" not in loaded:
        return {
            "skipped": True,
            "reason": "cluster_local_reference_or_transfer_unavailable",
            "roles": role_notes,
            "loaded_roles": sorted(loaded),
            "not_used_as_real_data_alignment_solve": True,
            "not_a_campaign_gate": True,
        }
    distance = subspace_distance(loaded["reference"]["_subspace"], loaded["transfer"]["_subspace"])
    for row in loaded.values():
        row.pop("_subspace", None)
    return {
        "residual_kind": RESIDUAL_KIND,
        "parameter_names": list(CLUSTER_LOCAL_NATIVE_NAMES),
        "ry_column_converted_from_per_radian_to_per_mrad": True,
        "software_fd_not_athena_payload": True,
        "not_used_as_real_data_alignment_solve": True,
        "not_a_campaign_gate": True,
        "reference": loaded["reference"],
        "transfer": loaded["transfer"],
        "source_disjoint_subspace": distance,
        "role_notes": role_notes,
    }


def decide_campaign(
    *,
    identifiable_rank: int,
    basis_stable: bool,
    three_arm: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not basis_stable:
        decision = DECISION_BASIS_UNSTABLE
        authorize = False
        reasons = [
            "identifiable_basis_unstable_across_source_or_bootstrap",
            "three_arm_solve_not_opened",
            "do_not_retune_scale_rank_selection_or_route_policy",
        ]
    elif three_arm is None or not bool(three_arm.get("pass")):
        decision = DECISION_ARM_FAIL
        authorize = False
        reasons = [
            "identifiable_basis_stable",
            "three_arm_closure_failed_or_missing",
            "do_not_retune_scale_rank_selection_or_route_policy",
        ]
    else:
        decision = DECISION_PASS
        authorize = True
        reasons = [
            "identifiable_basis_stable",
            "null_injection_does_not_create_identifiable_fake_correction",
            "mixed_injection_recovers_P_id_q_truth_only",
        ]
    return {
        "decision": decision,
        "identifiable_rank": int(identifiable_rank),
        "identifiable_basis_stable": bool(basis_stable),
        "null_injection_leakage_gate": None if three_arm is None else bool(three_arm.get("null_injection_leakage_gate")),
        "mixed_injection_projected_closure": None
        if three_arm is None
        else bool(three_arm.get("mixed_injection_projected_closure")),
        "authorize_frozen_v2_unknown_association_closure": bool(authorize) and decision == DECISION_PASS,
        "real_data_alignment_correction_authorized": False,
        "geometry_write_allowed": False,
        "survey_is_external_cross_check_only": True,
        "reasons": reasons,
        "if_failed_do_not_chase_by_retuning": True,
    }


def build_all_reports(config: Mapping[str, Any], *, banks: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    root = project_root()
    created = datetime.now(timezone.utc).isoformat()
    if banks is None:
        banks = load_physical_banks(config)
    pooled = _pool_campaign_banks(list(banks))
    pooled["source_id"] = "pooled_train_validation"
    rank_tolerance = float(config["rank_tolerance"])
    rcond = float(config["normal_matrix_rcond"])
    gates = config["gates"]
    pooled_subspace, pooled_extras = subspace_from_physical_bank(
        pooled, rank_tolerance=rank_tolerance, rcond=rcond
    )
    per_source = []
    source_subspaces: list[tuple[str, IdentifiableSubspace]] = []
    for bank in banks:
        subspace, extras = subspace_from_physical_bank(bank, rank_tolerance=rank_tolerance, rcond=rcond)
        label = f"{bank['split']}:{bank['source_id']}"
        source_subspaces.append((label, subspace))
        per_source.append(
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "n_pairs": int(extras["n_pairs"]),
                "identifiable_rank": int(subspace.identifiable_rank),
                "null_dimension": int(subspace.null_dimension),
                "singular_values": [float(value) for value in subspace.singular_values],
                "subspace": subspace_as_json(subspace),
            }
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
    if train_banks and validation_banks:
        train_subspace, _ = subspace_from_physical_bank(
            _pool_campaign_banks(train_banks), rank_tolerance=rank_tolerance, rcond=rcond
        )
        validation_subspace, _ = subspace_from_physical_bank(
            _pool_campaign_banks(validation_banks), rank_tolerance=rank_tolerance, rcond=rcond
        )
        split_reference = train_subspace
        split_rows: list[tuple[str, IdentifiableSubspace]] = [
            ("train_vs_validation", validation_subspace)
        ]
    else:
        split_reference = pooled_subspace
        split_rows = []
    split_stability = summarize_stability(
        split_reference,
        split_rows,
        max_principal_angle_deg=float(gates["max_train_validation_principal_angle_deg"]),
        max_projector_frobenius=float(gates["max_train_validation_projector_frobenius"]),
        require_same_rank=bool(gates["require_same_identifiable_rank"]),
    )
    complete_mask = _complete_route_mask(pooled)
    topology_subspace, _ = subspace_from_physical_bank(
        pooled, rank_tolerance=rank_tolerance, rcond=rcond, pair_mask=complete_mask
    )
    topology_stability = summarize_stability(
        pooled_subspace,
        [("complete_truth_routes", topology_subspace)],
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
    half_cfg = config.get("half_split") or {}
    half_spaces = half_split_subspaces(
        pooled,
        n_splits=int(half_cfg.get("n_splits", 0) or 0) or int(bootstrap_cfg.get("n_half_splits", 8)),
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
    basis_stable = bool(
        source_stability["stable"]
        and split_stability["stable"]
        and topology_stability["stable"]
        and bootstrap_stability["stable"]
        and half_stability["stable"]
    )
    three_arm = None
    if basis_stable:
        if train_banks and validation_banks:
            definition_subspace = split_reference
            _val_subspace, val_extras = subspace_from_physical_bank(
                _pool_campaign_banks(validation_banks), rank_tolerance=rank_tolerance, rcond=rcond
            )
            generating = val_extras["weighted_matrix"]
            generating_label = "validation_truth_selected_jacobian"
            definition_label = "train_truth_selected_jacobian"
        else:
            definition_subspace = pooled_subspace
            generating = pooled_extras["weighted_matrix"]
            generating_label = "pooled_truth_selected_jacobian"
            definition_label = "pooled_truth_selected_jacobian"
        injections = three_arm_payloads(
            definition_subspace,
            identifiable_scaled_amplitude=float(config["injections"]["identifiable_scaled_amplitude"]),
            null_scaled_amplitude=float(config["injections"]["null_scaled_amplitude"]),
            seed=int(config["injections"]["seed"]),
            n_replicates=int(config["injections"]["n_replicates"]),
        )
        three_arm = evaluate_three_arms(
            definition_subspace,
            generating,
            injections,
            gates=gates,
        )
        three_arm["injections_are_linear_on_validated_jacobian"] = True
        three_arm["athena_not_rerun"] = True
        three_arm["association_control"] = "truth_selected_physical_edge"
        three_arm["definition_jacobian"] = definition_label
        three_arm["generating_jacobian"] = generating_label
        three_arm["frozen_v2_unknown_association_not_opened"] = True
    decision = decide_campaign(
        identifiable_rank=pooled_subspace.identifiable_rank,
        basis_stable=basis_stable,
        three_arm=three_arm,
    )
    cluster_local = None
    if bool(config.get("cluster_local", {}).get("enabled", False)):
        try:
            cluster_local = load_cluster_local_subspaces(config)
        except (FileNotFoundError, OSError, ValueError) as error:
            cluster_local = {
                "skipped": True,
                "reason": str(error),
                "not_used_as_real_data_alignment_solve": True,
                "not_a_campaign_gate": True,
            }
    assumptions = list(config.get("assumptions", []))
    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": _operating_state(),
            "parameter_names": list(pooled_subspace.parameter_names),
            "parameter_units": list(pooled_subspace.parameter_units),
            "scale_matrix_S": [float(value) for value in pooled_subspace.parameter_scales],
            "scale_convention": "frozen_severity_scale_u_equals_theta_over_S",
            "residual_weight": "nominal_WLS_inverse_4x4_pair_covariance",
            "weighted_matrix": "A = W^{1/2} J S",
            "rank_tolerance": float(rank_tolerance),
            "rank_tolerance_source": "frozen LEAKAGE_RANK_RELATIVE_TOLERANCE = 1e-2",
            "normal_matrix_rcond": float(rcond),
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
            },
            "forbidden": {
                "naked_mixed_unit_svd": True,
                "scale_retune_from_singular_values": True,
                "rank_threshold_retune_from_spectrum": True,
                "full_parameter_newton": True,
                "real_data_alignment_correction": True,
                "geometry_write": True,
            },
            "assumptions": assumptions,
        },
        "identifiable_basis": {
            "pooled": subspace_as_json(pooled_subspace),
            "identifiable_rank": int(pooled_subspace.identifiable_rank),
            "null_dimension": int(pooled_subspace.null_dimension),
            "fit_normal_matrix_rank": int(pooled_extras["fit_normal_matrix_rank"]),
            "per_source": per_source,
            "modes_are_reconstruction_observable_linear_combinations": True,
            "not_mechanical_ry_or_C_dx": True,
        },
        "subspace_stability": {
            "source_disjoint": source_stability,
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
            "identifiable_basis_stable": basis_stable,
            "stopped_solve_because_basis_unstable": not basis_stable,
        },
        "three_arm_closure": three_arm,
        "cluster_local_diagnostic": cluster_local,
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
    }
