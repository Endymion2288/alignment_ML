"""Workbook-77 gauge-constrained / external-constraint feasibility V1.

Constraint-space design / feasibility only.  Two concepts are kept strictly
separate:

A) software/reconstruction gauge constraint - a linear ``G theta = 0`` that
   picks one definite mathematical representative inside the
   tracker-unidentifiable (null) directions of the frozen workbook-68 7D
   tracker information.  It is NOT a physical measurement; gauge-fixed
   parameters are never reported as "measured to be 0".

B) external physical constraint - only a truly independent survey/metrology
   quantity with validated Calypso frame mapping, known measurement
   covariance, and identified IOV/mechanical-stability provenance.  The
   eligibility audit is evaluated mechanically against the frozen
   workbook 61-67 artifacts.

The tracker information is the pooled workbook-68 7D normal matrix rebuilt
from the frozen hierarchical V1 bank through the frozen loader and regressed
against the frozen workbook-68 identifiable-basis artifact (negative
control).  No new reconstruction, no real data, no Newton, no geometry or
conditions writes.  The workbook-76 K-short FD spectrum is never consulted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    IdentifiableSubspace,
    frozen_scales_for,
    principal_angles_deg,
)
from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.tracker_only_identifiable_subspace import (
    _pool_campaign_banks,
    load_physical_banks,
    subspace_from_physical_bank,
)

SCHEMA_VERSION = "faser-gauge-constraint-external-constraint-feasibility-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs/gauge_constraint_external_constraint_feasibility_v1.yaml")

PARAMETER_NAMES = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_dz_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
    "C_dx",
)

DECISION_TRACKER_REGRESSION_FAILED = "tracker_information_regression_failed"
DECISION_GAUGE_CONTRACT_INVALID = "gauge_constraint_contract_invalid"
DECISION_SOLVER_NOT_WELL_POSED = "gauge_fixed_solver_not_well_posed"
DECISION_CLOSURE_FAILED = "gauge_invariant_observable_closure_failed"
DECISION_EXTERNAL_CANDIDATE_FOUND = (
    "external_physical_constraint_candidate_found_separate_subcampaign_required"
)
DECISION_GAUGE_FEASIBILITY_CLOSED = (
    "gauge_feasibility_closed_no_eligible_external_physical_constraint"
)

CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_run_full_parameter_newton",
    "do_not_solve_alignment_correction_on_real_data",
    "do_not_write_geometry_pool_or_cool",
    "do_not_write_official_conditions",
    "do_not_open_sealed_test",
    "do_not_svd_naked_mixed_unit_jacobian",
    "do_not_retune_scales_from_singular_values",
    "do_not_retune_rank_threshold_from_spectrum",
    "do_not_treat_null_zero_as_a_measurement",
    "do_not_relabel_modes_as_mechanical_parameters",
    "do_not_lower_min_pairs_post_hoc",
    "do_not_use_kshort_wb76_spectrum_to_design_gauges",
    "do_not_use_population_spread_as_measurement_sigma",
    "do_not_use_conditions_constants_as_physical_prior",
    "do_not_use_kabsch_or_dz_vs_x_tilt_as_station_ry",
    "do_not_compare_gauge_dependent_absolute_parameters_across_gauges",
    "do_not_use_full_truth_recovery_as_success_gate",
    "do_not_reopen_7d_cluster_local_stable_core_or_rigid_5dof_rescue",
    "do_not_add_more_tracker_data_to_chase_rank",
    "solver_restricted_to_constrained_kkt_or_nullspace_projection",
    "solver_restricted_to_identifiable_subspace",
    "gauge_fixed_parameters_are_not_measurements",
    "null_zero_is_minimum_norm_gauge_not_a_measurement",
    "survey_is_external_cross_check_only",
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "real_data_candidate_alignment_authorized",
    "survey_is_alignment_input",
)

_INHERITANCE_PATHS = {
    "workbook_68_config": "configs/tracker_only_identifiable_subspace_three_arm_closure_v1.yaml",
    "workbook_68_identifiable_basis": "outputs/tracker_only_identifiable_subspace_three_arm_closure_v1/identifiable_basis.json",
    "workbook_68_next_stage_decision": "outputs/tracker_only_identifiable_subspace_three_arm_closure_v1/next_stage_decision.json",
    "workbook_66_config": "configs/cad_survey_nov22_frame_covariance_audit_v1.yaml",
    "workbook_66_official_constraint_slots": "outputs/cad_survey_nov22_frame_covariance_audit_v1/official_constraint_slots.json",
    "workbook_66_next_stage_decision": "outputs/cad_survey_nov22_frame_covariance_audit_v1/next_stage_decision.json",
    "workbook_67_config": "configs/nov22_metrology_provenance_station_ry_contract_v1.yaml",
    "workbook_67_official_constraint_slots": "outputs/nov22_metrology_provenance_station_ry_contract_v1/official_constraint_slots.json",
    "workbook_67_station_ry_contract": "outputs/nov22_metrology_provenance_station_ry_contract_v1/calypso_station_ry_transform_contract.json",
    "workbook_73_config": "configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml",
    "workbook_76_config": "configs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1.yaml",
    "workbook_76_next_stage_decision": "outputs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1/next_stage_decision.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(
        project_root(), str(path or DEFAULT_CONFIG_RELATIVE)
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in CONFIG_MUST_BE_TRUE:
        if config.get(key) is not True:
            raise ValueError(f"config must set {key}: true")
    for key in CONFIG_MUST_BE_FALSE:
        if config.get(key) is not False:
            raise ValueError(f"config must set {key}: false")

    space = config["parameter_space"]
    names = tuple(str(name) for name in space["parameter_names"])
    if names != PARAMETER_NAMES:
        raise ValueError("parameter space must be the frozen hierarchical V1 7D order")
    scales = frozen_scales_for(names)
    declared = np.asarray(
        [float(space["scale_matrix_S"][name]) for name in names], dtype=np.float64
    )
    if not np.array_equal(declared, scales):
        raise ValueError("scale_matrix_S must equal the frozen severity scales")
    if float(space["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        raise ValueError("rank_tolerance must remain the frozen 0.01")
    units = tuple(str(unit) for unit in space["parameter_units"])
    if units != ("mm", "mm", "mm", "mrad", "mrad", "mrad", "mm"):
        raise ValueError("parameter units must stay mm/mm/mm/mrad/mrad/mrad/mm")

    corpus = config["tracker_information"]
    if tuple(str(n) for n in corpus["parameter_names"]) != PARAMETER_NAMES:
        raise ValueError("tracker_information parameter_names must match the 7D space")
    if float(corpus["min_truth_match_fraction"]) != 0.99:
        raise ValueError("min_truth_match_fraction must stay 0.99")
    if int(corpus["q_over_p_mode"]) != 0:
        raise ValueError("q_over_p_mode must stay 0")

    for key, expected_sha in config["inheritance_sha256"].items():
        artifact = resolve_under_root(project_root(), _INHERITANCE_PATHS[key])
        actual = sha256_file(artifact)
        if actual != str(expected_sha):
            raise ValueError(f"inheritance SHA256 mismatch for {key}: {actual}")

    closure = config["closure"]
    if closure.get("full_truth_recovery_is_not_a_gate") is not True:
        raise ValueError("closure must declare full_truth_recovery_is_not_a_gate")
    if closure.get("gauge_dependent_absolute_components_expected_to_differ") is not True:
        raise ValueError("closure must expect gauge-dependent components to differ")

    decision = config["decision_contract"]
    for key in (
        "external_constraint_ingest_authorized",
        "real_data_candidate_alignment_authorized",
        "geometry_write_allowed",
        "official_conditions_write_allowed",
    ):
        if decision.get(key) is not False:
            raise ValueError(f"decision_contract must pre-register {key}: false")

    config["config_path"] = str(config_path)
    return config


# ---------------------------------------------------------------------------
# Tracker information (frozen workbook-68 contract) + regression
# ---------------------------------------------------------------------------


def regression_against_frozen_basis(
    subspace: IdentifiableSubspace,
    frozen_pooled: Mapping[str, Any],
    *,
    singular_value_rtol: float,
    max_projector_frobenius: float,
    expected_identifiable_rank: int,
    expected_null_dimension: int,
) -> dict[str, Any]:
    """Basis-independent regression of a rebuilt subspace against WB68.

    The gated metric is the projector Frobenius distance (the frozen
    workbook 68-69 ``subspace_distance`` convention), which has full
    float64 resolution.  Principal angles are reported as diagnostics only:
    arccos(1 - eps) quantizes angles below ~1.2e-6 deg, so they cannot
    represent exact subspace agreement (workbook-77 amendment).
    """
    reasons = []
    frozen_sv = np.asarray(frozen_pooled["singular_values"], dtype=np.float64)
    rebuilt_sv = np.asarray(subspace.singular_values, dtype=np.float64)
    if frozen_sv.shape != rebuilt_sv.shape or not np.allclose(
        rebuilt_sv, frozen_sv, rtol=float(singular_value_rtol), atol=0.0
    ):
        reasons.append("pooled_singular_values_changed")
    if int(subspace.identifiable_rank) != int(expected_identifiable_rank):
        reasons.append("pooled_identifiable_rank_changed")
    if int(subspace.null_dimension) != int(expected_null_dimension):
        reasons.append("pooled_null_dimension_changed")
    # The frozen artifact stores basis matrices as (modes x parameters);
    # the IdentifiableSubspace dataclass uses (parameters x modes).
    frozen_vid = np.asarray(frozen_pooled["v_id_scaled"], dtype=np.float64).T
    frozen_vnull = np.asarray(frozen_pooled["v_null_scaled"], dtype=np.float64).T
    projector_id_distance = float(
        np.linalg.norm(subspace.projector_id - frozen_vid @ frozen_vid.T)
    )
    projector_null_distance = float(
        np.linalg.norm(subspace.v_null @ subspace.v_null.T - frozen_vnull @ frozen_vnull.T)
    )
    if projector_id_distance > float(max_projector_frobenius):
        reasons.append("pooled_identifiable_subspace_changed")
    if projector_null_distance > float(max_projector_frobenius):
        reasons.append("pooled_null_subspace_changed")
    angle_id = float(np.max(principal_angles_deg(subspace.v_id, frozen_vid)))
    angle_null = float(np.max(principal_angles_deg(subspace.v_null, frozen_vnull)))
    return {
        "rebuilt_singular_values": [float(v) for v in rebuilt_sv],
        "frozen_singular_values": [float(v) for v in frozen_sv],
        "singular_value_rtol": float(singular_value_rtol),
        "identifiable_projector_frobenius_distance": projector_id_distance,
        "null_projector_frobenius_distance": projector_null_distance,
        "max_projector_frobenius": float(max_projector_frobenius),
        "identifiable_basis_max_principal_angle_deg_diagnostic": angle_id,
        "null_basis_max_principal_angle_deg_diagnostic": angle_null,
        "principal_angles_are_arccos_quantized_diagnostics_only": True,
        "identifiable_rank": int(subspace.identifiable_rank),
        "null_dimension": int(subspace.null_dimension),
        "pass": not reasons,
        "failure_reasons": reasons,
    }


def load_tracker_information(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], IdentifiableSubspace, dict[str, Any], dict[str, Any]]:
    """Rebuild the pooled 7D tracker information and regress it against WB68.

    Returns (banks, pooled_subspace, pooled_extras, regression_report).  The
    regression report compares the rebuilt pooled spectrum and subspaces
    against the frozen workbook-68 identifiable-basis artifact; any mismatch
    fails the campaign before any gauge use.
    """
    corpus = dict(config["tracker_information"])
    banks = load_physical_banks({"jacobian_corpus": corpus})
    pooled = _pool_campaign_banks(list(banks))
    pooled["source_id"] = "pooled_workbook68_corpus"
    space = config["parameter_space"]
    subspace, extras = subspace_from_physical_bank(
        pooled,
        rank_tolerance=float(space["rank_tolerance"]),
        rcond=float(space["normal_matrix_rcond"]),
    )

    frozen = _read_json(
        resolve_under_root(project_root(), str(corpus["regression_artifact"]))
    )["pooled"]
    regression = regression_against_frozen_basis(
        subspace,
        frozen,
        singular_value_rtol=float(corpus["regression_singular_value_rtol"]),
        max_projector_frobenius=float(corpus["regression_max_projector_frobenius"]),
        expected_identifiable_rank=int(corpus["expected_pooled_identifiable_rank"]),
        expected_null_dimension=int(corpus["expected_pooled_null_dimension"]),
    )
    regression.update(
        {
            "kind": "tracker_information_regression",
            "regression_artifact": str(corpus["regression_artifact"]),
            "n_sources": len(banks),
            "n_pairs": int(extras["n_pairs"]),
        }
    )
    return banks, subspace, extras, regression


# ---------------------------------------------------------------------------
# Gauge candidates
# ---------------------------------------------------------------------------


def build_gauge_candidates(
    config: Mapping[str, Any], subspace: IdentifiableSubspace
) -> list[dict[str, Any]]:
    """Build the explicit native-space ``G theta = 0`` matrix per candidate.

    Every candidate must have rank 2 and complement the frozen null space:
    ``G . S . V_null`` (2x2) must be invertible so the constraint intersects
    each gauge orbit exactly once.
    """
    names = tuple(subspace.parameter_names)
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)
    v_null = np.asarray(subspace.v_null, dtype=np.float64)
    null_native = s_mat @ v_null  # native-space null basis (7 x 2)

    candidates = []
    for spec in config["gauge_candidates"]:
        kind = str(spec["kind"])
        if kind == "fixed_reference_rigid_mode":
            g_native = np.zeros((len(spec["constraint_rows_native"]), len(names)))
            values = []
            for row_index, row in enumerate(spec["constraint_rows_native"]):
                parameter = str(row["parameter"])
                if parameter not in names:
                    raise ValueError(f"unknown gauge parameter {parameter}")
                g_native[row_index, names.index(parameter)] = 1.0
                values.append(float(row["value"]))
            rhs = np.asarray(values, dtype=np.float64)
            if not np.all(rhs == 0.0):
                raise ValueError("this campaign pre-registers zero-valued gauge rows only")
        elif kind == "minimum_norm_representative":
            metric = str(spec["metric"])
            if metric == "severity_scaled":
                # V_null^T u = 0 with u = S^{-1} theta  ->  G = V_null^T S^{-1}
                g_native = v_null.T @ np.linalg.inv(s_mat)
            elif metric == "native":
                # theta orthogonal to native null directions: (S V_null)^T theta = 0
                g_native = null_native.T
            else:
                raise ValueError(f"unknown minimum-norm metric {metric}")
        else:
            raise ValueError(f"unknown gauge kind {kind}")

        rank_g = int(np.linalg.matrix_rank(g_native, tol=1.0e-12))
        intersection = g_native @ null_native  # 2 x 2
        det = float(np.linalg.det(intersection))
        cond = float(np.linalg.cond(intersection))
        valid = (
            rank_g == int(subspace.null_dimension)
            and math.isfinite(det)
            and abs(det) > 1.0e-12
        )
        candidates.append(
            {
                "id": str(spec["id"]),
                "kind": kind,
                "g_native": [[float(v) for v in row] for row in g_native],
                "constraint_summary": str(
                    spec.get("constraint", "")
                )
                or "theta_%s = 0 rows" % ",".join(
                    str(r["parameter"]) for r in spec.get("constraint_rows_native", [])
                ),
                "remaining_dof": int(subspace.n_parameters) - rank_g,
                "expected_remaining_dof": int(spec["remaining_dof"]),
                "rank_g": rank_g,
                "null_intersection_determinant": det,
                "null_intersection_condition": cond,
                "valid": valid
                and int(subspace.n_parameters) - rank_g == int(spec["remaining_dof"]),
                "changes_tracker_observable_prediction": bool(
                    spec["changes_tracker_observable_prediction"]
                ),
                "calypso_convention_relation": str(spec["calypso_convention_relation"]),
                "parameters_without_independent_interpretation": [
                    str(p) for p in spec["parameters_without_independent_interpretation"]
                ],
                "interpretation_note": str(spec["interpretation_note"]),
                "gauge_fixed_parameters_are_not_measurements": True,
                "_g_native": g_native,
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Constrained WLS solve (KKT).  Full-parameter unconstrained Newton forbidden.
# ---------------------------------------------------------------------------


def solve_gauge_fixed_kkt(
    weighted_matrix: np.ndarray, weighted_residual: np.ndarray, g_scaled: np.ndarray
) -> dict[str, Any]:
    """Solve ``min ||A u - r||^2`` subject to ``G u = 0`` via the KKT system.

    ``A`` is the frozen ``W^{1/2} J S``, ``u = S^{-1} theta`` the scaled
    parameter vector and ``G`` the gauge constraint in scaled coordinates.
    """
    a = np.asarray(weighted_matrix, dtype=np.float64)
    r = np.asarray(weighted_residual, dtype=np.float64)
    g = np.asarray(g_scaled, dtype=np.float64)
    n = a.shape[1]
    m = g.shape[0]
    normal = a.T @ a
    rhs = np.concatenate([a.T @ r, np.zeros(m)])
    kkt = np.zeros((n + m, n + m))
    kkt[:n, :n] = normal
    kkt[:n, n:] = g.T
    kkt[n:, :n] = g
    condition = float(np.linalg.cond(kkt))
    rank = int(np.linalg.matrix_rank(kkt, tol=1.0e-12))
    solution = np.linalg.solve(kkt, rhs)
    u_hat = solution[:n]
    multipliers = solution[n:]
    solve_residual = float(np.linalg.norm(kkt @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return {
        "u_hat": u_hat,
        "lagrange_multipliers": multipliers,
        "kkt_condition": condition,
        "kkt_rank": rank,
        "kkt_full_rank": rank == n + m,
        "solve_residual_relative": solve_residual / max(rhs_norm, 1.0e-300),
        "gauge_residual_max_abs": float(np.max(np.abs(g @ u_hat))) if m else 0.0,
    }


# ---------------------------------------------------------------------------
# Gauge-invariant observable closure (synthetic measurements on frozen A)
# ---------------------------------------------------------------------------


def _unit(rng: np.random.Generator, size: int) -> np.ndarray:
    vector = rng.normal(size=size)
    norm = float(np.linalg.norm(vector))
    return vector / (norm if norm > 0.0 else 1.0)


def gauge_invariant_closure(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    extras: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pre-registered pure-gauge closure on the frozen tracker information.

    For each replicate a truth ``u*`` with identifiable and null content is
    drawn; additional observable-equivalent representatives ``u*_k`` differ
    from ``u*`` only along the null directions and therefore produce the
    identical weighted measurement.  Each gauge must (i) satisfy its own
    constraint exactly, (ii) return one unique representative per gauge, and
    (iii) agree with every other gauge on the observable prediction, the
    identifiable projection, and held-out tracker observables.  Recovering
    the injected ``u*`` is impossible by construction and is NOT a gate.

    Following the frozen ``solver_restricted_to_identifiable_subspace``
    doctrine (workbook 68), the solve uses the identifiable restriction
    ``A_eff = A P_id``: the two near-null directions below the frozen
    rank_tolerance cut are excluded from the solve, so the gauge space is
    exactly ``null(A_eff)`` and gauge invariance is an exact linear-algebra
    statement, not an approximation up to near-null leakage.  The discarded
    near-null information is reported as a diagnostic, never used.
    """
    closure = config["closure"]
    gates = closure["gates"]
    a_full = np.asarray(extras["weighted_matrix"], dtype=np.float64)
    n_pairs = int(extras["n_pairs"])
    if a_full.shape[0] != 4 * n_pairs:
        raise ValueError("weighted matrix must carry exactly 4 rows per pair")
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    v_null = np.asarray(subspace.v_null, dtype=np.float64)
    rank = int(subspace.identifiable_rank)
    null_dim = int(subspace.null_dimension)
    if null_dim < 1:
        raise ValueError("gauge closure requires a non-trivial null space")
    projector_id = v_id @ v_id.T
    projector_null = v_null @ v_null.T
    a = a_full @ projector_id  # identifiable restriction: null(A_eff) is exact

    rng = np.random.default_rng(int(closure["seed"]))
    sigma = float(closure["measurement_noise_sigma_weighted"])
    amp_id = float(closure["identifiable_scaled_amplitude"])
    amp_null = float(closure["null_scaled_amplitude"])
    amp_off = float(closure["observable_equivalent_null_offset_scaled_amplitude"])
    n_replicates = int(closure["n_replicates"])
    n_offsets = int(closure["n_null_offsets_per_replicate"])

    gauge_rows = [
        (str(candidate["id"]), np.asarray(candidate["_g_native"], dtype=np.float64))
        for candidate in candidates
    ]
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)

    max_gauge_residual = 0.0
    max_uniqueness = 0.0
    max_observable_diff = 0.0
    max_identifiable_diff = 0.0
    max_solve_residual = 0.0
    max_condition = 0.0
    max_near_null_leakage = 0.0
    all_full_rank = True
    null_coordinate_spread: dict[str, list[float]] = {gid: [] for gid, _ in gauge_rows}

    for _replicate in range(n_replicates):
        u_star = v_id @ (_unit(rng, rank) * amp_id) + v_null @ (
            _unit(rng, null_dim) * amp_null
        )
        full_norm = float(np.linalg.norm(a_full @ u_star))
        max_near_null_leakage = max(
            max_near_null_leakage,
            float(np.linalg.norm(a_full @ projector_null @ u_star))
            / max(full_norm, 1.0e-300),
        )
        noise = rng.normal(size=a.shape[0]) * sigma
        r_w = a @ u_star + noise
        # Observable-equivalent representatives: identical A u, hence the
        # identical measurement; the noise realization is shared because the
        # measurement is of the observable content only.
        equivalents = [u_star] + [
            u_star + v_null @ (_unit(rng, null_dim) * amp_off)
            for _ in range(n_offsets)
        ]

        solved: dict[str, list[np.ndarray]] = {}
        for gauge_id, g_native in gauge_rows:
            g_scaled = g_native @ s_mat
            intersection = g_scaled @ v_null
            if abs(float(np.linalg.det(intersection))) <= 1.0e-12:
                # A gauge that does not complement the null space yields a
                # singular KKT system: record the instability instead of
                # raising, so the closure reports it as a gate failure.
                all_full_rank = False
                max_condition = math.inf
                solved[gauge_id] = None
                continue
            solutions = [
                solve_gauge_fixed_kkt(a, r_w, g_scaled) for _u in equivalents
            ]
            solved[gauge_id] = [item["u_hat"] for item in solutions]
            for item in solutions:
                max_gauge_residual = max(
                    max_gauge_residual, float(item["gauge_residual_max_abs"])
                )
                max_solve_residual = max(
                    max_solve_residual, float(item["solve_residual_relative"])
                )
                max_condition = max(max_condition, float(item["kkt_condition"]))
                all_full_rank = all_full_rank and bool(item["kkt_full_rank"])
            reference = solved[gauge_id][0]
            for other in solved[gauge_id][1:]:
                max_uniqueness = max(
                    max_uniqueness, float(np.max(np.abs(other - reference)))
                )

        solved_ids = [gid for gid, _ in gauge_rows if solved.get(gid) is not None]
        for left in range(len(solved_ids)):
            for right in range(left + 1, len(solved_ids)):
                gid_l = solved_ids[left]
                gid_r = solved_ids[right]
                u_l = solved[gid_l][0]
                u_r = solved[gid_r][0]
                observable_norm = float(np.linalg.norm(a @ u_l))
                max_observable_diff = max(
                    max_observable_diff,
                    float(np.linalg.norm(a @ (u_l - u_r)))
                    / max(observable_norm, 1.0e-300),
                )
                max_identifiable_diff = max(
                    max_identifiable_diff,
                    float(np.max(np.abs(v_id.T @ (u_l - u_r)))),
                )
        for gauge_id in solved_ids:
            null_coords = v_null.T @ solved[gauge_id][0]
            null_coordinate_spread[gauge_id].append(
                [float(value) for value in null_coords]
            )

    # Held-out tracker observables: split pairs into halves, solve each
    # gauge on half 1, and require identical predictions on half 2.
    held_out_rng = np.random.default_rng(int(closure["held_out_seed"]))
    permutation = held_out_rng.permutation(n_pairs)
    half = int(round(n_pairs * float(closure["held_out_pair_fraction"])))
    first = np.sort(permutation[:half])
    second = np.sort(permutation[half:])
    rows_1 = np.concatenate([np.arange(4 * p, 4 * p + 4) for p in first])
    rows_2 = np.concatenate([np.arange(4 * p, 4 * p + 4) for p in second])
    a1 = a[rows_1]
    a2 = a[rows_2]
    max_held_out_diff = 0.0
    for _replicate in range(n_replicates):
        u_star = v_id @ (_unit(rng, rank) * amp_id) + v_null @ (
            _unit(rng, null_dim) * amp_null
        )
        r1 = a1 @ u_star + rng.normal(size=rows_1.size) * sigma
        predictions = []
        for gauge_id, g_native in gauge_rows:
            g_scaled = g_native @ s_mat
            if abs(float(np.linalg.det(g_scaled @ v_null))) <= 1.0e-12:
                all_full_rank = False
                max_condition = math.inf
                continue
            result = solve_gauge_fixed_kkt(a1, r1, g_scaled)
            predictions.append(a2 @ result["u_hat"])
        for left in range(len(predictions)):
            for right in range(left + 1, len(predictions)):
                norm = float(np.linalg.norm(predictions[left]))
                max_held_out_diff = max(
                    max_held_out_diff,
                    float(np.linalg.norm(predictions[left] - predictions[right]))
                    / max(norm, 1.0e-300),
                )

    # Report-only: gauge-dependent null coordinates differ across gauges.
    null_means = {
        gauge_id: (
            [
                float(value)
                for value in np.mean(np.asarray(history, dtype=np.float64), axis=0)
            ]
            if history
            else None
        )
        for gauge_id, history in null_coordinate_spread.items()
    }
    cross_gauge_null_spread = 0.0
    gauge_ids = [gid for gid, means in null_means.items() if means is not None]
    for left in range(len(gauge_ids)):
        for right in range(left + 1, len(gauge_ids)):
            cross_gauge_null_spread = max(
                cross_gauge_null_spread,
                float(
                    np.max(
                        np.abs(
                            np.asarray(null_means[gauge_ids[left]])
                            - np.asarray(null_means[gauge_ids[right]])
                        )
                    )
                ),
            )

    checks = {
        "gauge_constraint_exactly_satisfied": max_gauge_residual
        <= float(gates["gauge_residual_max_abs"]),
        "representative_unique_per_gauge": max_uniqueness
        <= float(gates["representative_uniqueness_max_abs"]),
        "observable_prediction_gauge_invariant": max_observable_diff
        <= float(gates["observable_prediction_max_relative_diff"]),
        "identifiable_projection_gauge_invariant": max_identifiable_diff
        <= float(gates["identifiable_projection_max_abs_diff"]),
        "held_out_observables_gauge_invariant": max_held_out_diff
        <= float(gates["held_out_prediction_max_relative_diff"]),
        "kkt_solve_residual_within_gate": max_solve_residual
        <= float(gates["kkt_solve_residual_max_relative"]),
        "kkt_numerically_stable": all_full_rank
        and max_condition <= float(gates["kkt_condition_max"]),
    }
    return {
        "kind": "gauge_invariant_observable_closure",
        "n_replicates": n_replicates,
        "n_null_offsets_per_replicate": n_offsets,
        "held_out_pairs": int(second.size),
        "calibration_pairs": int(first.size),
        "max_gauge_residual_abs": max_gauge_residual,
        "max_representative_uniqueness_abs": max_uniqueness,
        "max_observable_prediction_relative_diff": max_observable_diff,
        "max_identifiable_projection_abs_diff": max_identifiable_diff,
        "max_held_out_prediction_relative_diff": max_held_out_diff,
        "max_kkt_solve_residual_relative": max_solve_residual,
        "max_kkt_condition": max_condition,
        "kkt_full_rank_everywhere": all_full_rank,
        "gauge_dependent_null_coordinate_means": null_means,
        "cross_gauge_null_coordinate_spread": cross_gauge_null_spread,
        "near_null_information_discarded_by_frozen_rank_cut_max_relative": max_near_null_leakage,
        "identifiable_restriction_follows_frozen_solver_doctrine": True,
        "gauge_dependent_components_differ_as_expected": cross_gauge_null_spread
        > float(gates["identifiable_projection_max_abs_diff"]),
        "full_truth_recovery_is_not_a_gate": True,
        "checks": checks,
        "pass": all(bool(value) for value in checks.values()),
    }


# ---------------------------------------------------------------------------
# External-constraint eligibility table
# ---------------------------------------------------------------------------


def external_eligibility_table(config: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically evaluate the pre-registered eligibility rule.

    A row is an eligible physical constraint only if ALL four gates pass:
    validated Calypso parameter mapping, independent measurement covariance,
    identified measurement-year/conditions IOV with mechanical-stability
    provenance, and independence from tracks and conditions.  The three
    official slots are cross-checked against the frozen workbook 66/67 slot
    artifacts; diagnostic rows restate frozen workbook 61-67 conclusions.
    """
    audit = config["external_constraint_eligibility"]
    slots66 = _read_json(
        resolve_under_root(project_root(), str(audit["slot_artifacts"]["workbook_66_slots"]))
    )
    slots67 = _read_json(
        resolve_under_root(project_root(), str(audit["slot_artifacts"]["workbook_67_slots"]))
    )
    frozen_slots = {}
    for artifact in (slots66, slots67):
        for slot in artifact["slots"]:
            frozen_slots[str(slot["constraint_id"])] = slot

    rows = []
    for spec in audit["audit_rows"]:
        mapping_text = str(spec["calypso_parameter_mapping"])
        mapping_ok = mapping_text.startswith("validated_as")
        cov_ok = spec["real_measurement_sigma_or_covariance"] is not None
        iov_text = str(spec["iov_mechanical_stability_provenance"])
        iov_ok = iov_text.startswith("valid_iov:")
        independence = spec["independent_of_tracks_and_conditions"]
        independence_ok = independence is True
        gates = {
            "parameter_mapping_validated": mapping_ok,
            "measurement_covariance_has_independent_provenance": cov_ok,
            "measurement_year_conditions_iov_identified": iov_ok,
            "independent_of_tracks_and_conditions": independence_ok,
        }
        eligible = all(gates.values())
        row = {
            "id": str(spec["id"]),
            "value": spec["value"],
            "unit": str(spec["unit"]),
            "coordinate_frame": str(spec["coordinate_frame"]),
            "pivot_rotation_convention": str(spec["pivot_rotation_convention"]),
            "calypso_parameter_mapping": mapping_text,
            "measurement_date": spec["measurement_date"],
            "iov_mechanical_stability_provenance": spec[
                "iov_mechanical_stability_provenance"
            ],
            "evidence_source": str(spec["evidence_source"]),
            "gates": gates,
            "eligible_physical_constraint": eligible,
        }

        slot = frozen_slots.get(str(spec["id"]))
        if slot is not None:
            availability = str(slot.get("availability", ""))
            slot_value = slot.get("value")
            if isinstance(slot_value, list) and len(slot_value) == 1:
                slot_value = slot_value[0]
            consistent = availability != "measured" and slot.get("sigma") is None
            if spec["value"] is not None and slot_value is not None:
                consistent = consistent and math.isclose(
                    float(slot_value), float(spec["value"]), rel_tol=1.0e-6, abs_tol=0.0
                )
            if spec["value"] is None:
                consistent = consistent and slot_value is None
            row["frozen_slot_availability"] = availability
            row["consistent_with_frozen_slot_artifact"] = bool(consistent)
        rows.append(row)

    eligible_ids = [row["id"] for row in rows if row["eligible_physical_constraint"]]
    expected = [str(item) for item in audit["expected_eligible"]]
    consistency_ok = all(
        row.get("consistent_with_frozen_slot_artifact", True) for row in rows
    )
    return {
        "kind": "external_constraint_eligibility",
        "n_candidates": len(rows),
        "rows": rows,
        "eligible_external_physical_constraints": eligible_ids,
        "expected_eligible": expected,
        "matches_preregistered_expectation": eligible_ids == expected,
        "consistent_with_frozen_slot_artifacts": bool(consistency_ok),
        "no_ingestable_external_physical_constraint_available": not eligible_ids,
        "fisher_or_posterior_ingest_ran": False,
        "reason_if_none": (
            None
            if eligible_ids
            else (
                "Every existing survey/metrology candidate fails at least one "
                "pre-registered gate: no independent measurement covariance "
                "exists anywhere, no candidate carries a valid 2024/2025 "
                "conditions IOV with mechanical-stability provenance, the "
                "Stations ry survey mapping is unresolved, and the conditions "
                "constants are reconstruction state, not independent survey."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Campaign decision
# ---------------------------------------------------------------------------


def decide_campaign(
    *,
    regression: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    closure_report: Mapping[str, Any],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    contract_valid = all(bool(candidate["valid"]) for candidate in candidates)
    solver_well_posed = bool(
        closure_report["checks"]["kkt_numerically_stable"]
        and closure_report["checks"]["kkt_solve_residual_within_gate"]
        and closure_report["checks"]["gauge_constraint_exactly_satisfied"]
        and closure_report["checks"]["representative_unique_per_gauge"]
    )
    closure_pass = bool(closure_report["pass"])
    eligible = list(eligibility["eligible_external_physical_constraints"])

    if not bool(regression["pass"]):
        decision = DECISION_TRACKER_REGRESSION_FAILED
    elif not contract_valid:
        decision = DECISION_GAUGE_CONTRACT_INVALID
    elif not solver_well_posed:
        decision = DECISION_SOLVER_NOT_WELL_POSED
    elif not closure_pass:
        decision = DECISION_CLOSURE_FAILED
    elif eligible:
        decision = DECISION_EXTERNAL_CANDIDATE_FOUND
    else:
        decision = DECISION_GAUGE_FEASIBILITY_CLOSED

    return {
        "kind": "next_stage_decision",
        "decision": decision,
        "gauge_constraint_contract_valid": bool(contract_valid),
        "gauge_fixed_solver_well_posed": bool(solver_well_posed),
        "gauge_invariant_observable_closure": bool(closure_pass),
        "eligible_external_physical_constraints": eligible,
        "external_constraint_ingest_authorized": False,
        "no_ingestable_external_physical_constraint_available": not eligible,
        "real_data_candidate_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "gauge_fixed_parameters_are_not_measurements": True,
        "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
        "full_truth_recovery_was_not_a_gate": True,
        "kshort_wb76_spectrum_never_consulted": True,
        "next_stage_if_gauge_closure_passes": (
            "gauge_fixed_real_data_alignment_diagnostic_v1_requires_separate_preregistration"
        ),
        "official_cool_pool_write_remains_closed": True,
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    from datetime import datetime, timezone

    banks, subspace, extras, regression = load_tracker_information(config)
    candidates = build_gauge_candidates(config, subspace)
    closure_report = gauge_invariant_closure(config, subspace, extras, candidates)
    eligibility = external_eligibility_table(config)
    decision = decide_campaign(
        regression=regression,
        candidates=candidates,
        closure_report=closure_report,
        eligibility=eligibility,
    )
    gauge_public = [
        {key: value for key, value in candidate.items() if not key.startswith("_")}
        for candidate in candidates
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": str(config.get("config_path", "")),
        "tracker_information_regression": regression,
        "gauge_candidates": gauge_public,
        "gauge_invariant_closure": closure_report,
        "external_constraint_eligibility": eligibility,
        "next_stage_decision": decision,
        "_banks": banks,
    }
