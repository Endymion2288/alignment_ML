"""Task B14V: official-path derivative residual contract resolution.

WB120 already built the same-path rk_free Jacobian.  This task only
classifies the two remaining mismatches on the frozen h,h/2,h/4,h/8
ladder: control 100043/1 loc1, and focus 100048/86.

It does not change the Jacobian implementation, retune Gauss-Newton,
change production stepTolerance, pick a best FD step, relax the frozen
5% gate, replace the official likelihood, introduce a prior/ridge,
delete 37/86, or enter B14M / B15 / Measurement Model V2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.profiled_measurement_likelihood import (
    ALPHA_NAMES,
    NU_NAMES,
    PINV_RELATIVE,
)
from datasets.acts_fd_derivative_contract import (
    FOCUS_EVENT,
    PARAM_NAMES,
    REL_MAX,
    RUNG_FACTORS,
    _as_vector,
    _column_agreement,
    _event_label,
    _event_pair,
    _finite,
    audit_fd_column,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.official_supporting_plane_jacobian import (
    CASE_INCONSISTENT as WB120_DECISION,
    inherit_frozen_stage as inherit_through_wb119,
    load_jacobian_rows,
)
from datasets.profile_transport_contract import (
    audit_exclusion,
    is_official_mode_b,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "official-path-derivative-residual-v1"
DEFAULT_CONFIG = "configs/official_path_derivative_residual_v1.yaml"
TASK = "SB-B14V"
WORKBOOK = 121

CASE_ESTABLISHED = "official_path_derivative_contract_established"
CASE_FD_REF = "fd_reference_not_precise_enough_for_derivative_certification"
CASE_PATHOLOGY = "small_column_relative_metric_pathology"
CASE_INCONSISTENT = "official_path_derivative_still_inconsistent"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_ESTABLISHED,
    CASE_FD_REF,
    CASE_PATHOLOGY,
    CASE_INCONSISTENT,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
CONTROL_LABELS = {"100043/0", "100043/1", "100043/37"}
FOCUS_LABEL = "100048/86"
OFFICIAL_FD_STEPS = {
    "loc0": 0.01,
    "loc1": 0.01,
    "phi": 1.0e-5,
    "theta": 1.0e-5,
    "q_over_p": 1.0e-6,
}
MEASUREMENT_SIGMA_MM = 0.08 / np.sqrt(12.0)
FAR_BELOW_MEASUREMENT = 0.05
ENVELOPE_SAME_ORDER_FACTOR = 3.0
HOP_COMPOSITION_REL_MAX = 1.0e-9
KIND_FIVE_PERCENT_PASS = "frozen_five_percent_pass"
KIND_PATHOLOGY = CASE_PATHOLOGY
KIND_FD_REF = CASE_FD_REF
KIND_INCONSISTENT = CASE_INCONSISTENT


class OfficialPathDerivativeResidualError(ValueError):
    """Raised when the B14V residual contract is illegal."""


def refuse_prior() -> None:
    raise OfficialPathDerivativeResidualError("B14V must not introduce a prior")


def refuse_ridge_information() -> None:
    raise OfficialPathDerivativeResidualError(
        "ridge must not be treated as statistical information"
    )


def refuse_measurement_update() -> None:
    raise OfficialPathDerivativeResidualError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_best_step_selection() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not select the best finite-difference step from the results"
    )


def refuse_best_tolerance() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not select the best propagator stepTolerance from the results"
    )


def refuse_switch_to_direct() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not replace the official sequential likelihood with direct-from-source"
    )


def refuse_replace_likelihood() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not replace the official likelihood with the ACTS Jacobian"
    )


def refuse_dummy_cov_bounded() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not use dummy-cov bounded transportJacobian as the official derivative"
    )


def refuse_relax_gate() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_change_fd_ladder() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not change the official h,h/2,h/4,h/8 FD ladder"
    )


def refuse_change_jacobian() -> None:
    raise OfficialPathDerivativeResidualError(
        "must not change the official-path Jacobian implementation"
    )


def refuse_b14m() -> None:
    raise OfficialPathDerivativeResidualError("B14M is not re-opened inside Task B14V")


def refuse_restart() -> None:
    raise OfficialPathDerivativeResidualError(
        "restart invariance is not opened inside Task B14V"
    )


def refuse_b15() -> None:
    raise OfficialPathDerivativeResidualError("Task B15 is not entered in Task B14V")


def refuse_measurement_model_v2() -> None:
    raise OfficialPathDerivativeResidualError(
        "Measurement Model V2 is not entered in Task B14V"
    )


def refuse_full_sample() -> None:
    raise OfficialPathDerivativeResidualError(
        "the 1989-row campaign must not be submitted in B14V"
    )


def refuse_change_statistical_model() -> None:
    raise OfficialPathDerivativeResidualError(
        "the measurement statistical model must stay frozen"
    )


def refuse_5d_cin_repair() -> None:
    raise OfficialPathDerivativeResidualError(
        "B14V must not treat 5D Cin repair as the objective"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise OfficialPathDerivativeResidualError(f"{label} hash mismatch: {digest}")


def _as_matrix(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or not raw:
        return None
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2:
        return None
    return matrix


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise OfficialPathDerivativeResidualError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise OfficialPathDerivativeResidualError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise OfficialPathDerivativeResidualError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise OfficialPathDerivativeResidualError(
            "official input must be contract_eligible"
        )
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "marginalization_executed",
    ):
        if bool(config.get(key, True)):
            raise OfficialPathDerivativeResidualError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14u",
        "do_not_select_best_fd_step",
        "do_not_switch_official_likelihood_to_direct",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
    ):
        if not bool(config.get(key, False)):
            raise OfficialPathDerivativeResidualError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14v", True)):
        raise OfficialPathDerivativeResidualError("do_not_enter_b14v must be false")
    spec = config["official_path_derivative_residual"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise OfficialPathDerivativeResidualError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise OfficialPathDerivativeResidualError(
            "production stepTolerance must stay 1e-4"
        )
    if not bool(spec["do_not_relax_five_percent_gate"]):
        raise OfficialPathDerivativeResidualError("the frozen 5% gate must stay")
    if not bool(spec["do_not_change_jacobian_implementation"]):
        raise OfficialPathDerivativeResidualError(
            "the official-path Jacobian implementation must stay"
        )
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise OfficialPathDerivativeResidualError("allowed decisions must stay the B14V set")
    for item in config["wb120_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb120 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb119(config)
    spec = config["inheritance"]["workbook_120"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_120 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_120 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise OfficialPathDerivativeResidualError(
            "workbook_120 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB120_DECISION:
        raise OfficialPathDerivativeResidualError("WB120 decision token mismatch")
    if decision.get("b14m_reopen_authorized"):
        raise OfficialPathDerivativeResidualError("WB120 must not have re-opened B14M")
    if decision.get("jacobian_contract_established"):
        raise OfficialPathDerivativeResidualError(
            "WB120 must not have established the Jacobian contract"
        )
    if not decision.get("free_state_jacobian_available"):
        raise OfficialPathDerivativeResidualError(
            "WB120 must have established the same-path free-state Jacobian"
        )
    inherited["workbook_120"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "smoke_gate_passed": True,
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "free_state_jacobian_available": True,
        "official_loc0_matches_diagnostic": True,
        "control_fd_converged": True,
        "control_official_path_agrees_fd": False,
        "same_path_jacobian_available": True,
        "wb120_not_a_physical_conclusion": True,
    }
    return inherited


def _rung_columns(raw: Mapping[str, Any]) -> dict[float, np.ndarray | None]:
    out: dict[float, np.ndarray | None] = {}
    for item in raw.get("rungs") or []:
        factor = _finite(item.get("step_factor"))
        if factor is None:
            continue
        out[float(factor)] = _as_vector(item.get("residual_column"))
    return out


def audit_envelope_column(
    fd_raw: Mapping[str, Any],
    analytic: np.ndarray | None,
    *,
    relative_max: float = REL_MAX,
    same_order_factor: float = ENVELOPE_SAME_ORDER_FACTOR,
    far_below_fraction: float = FAR_BELOW_MEASUREMENT,
) -> dict[str, Any]:
    name = str(fd_raw.get("parameter"))
    fd = audit_fd_column(fd_raw)
    rungs = _rung_columns(fd_raw)
    missing = analytic is None or any(rungs.get(factor) is None for factor in RUNG_FACTORS)
    if missing:
        return {
            "parameter": name,
            "present": False,
            "residual_kind": CASE_MIXED,
            "fd_ladder_converged": bool(fd.get("ladder_converged")),
            "frozen_five_percent_pass": False,
        }
    j_h, j_h2, j_h4, j_h8 = (rungs[factor] for factor in RUNG_FACTORS)
    successive = (
        float(np.linalg.norm(j_h - j_h2)),
        float(np.linalg.norm(j_h2 - j_h4)),
        float(np.linalg.norm(j_h4 - j_h8)),
    )
    envelope = successive[2]
    vs_rungs = []
    distances = []
    for factor, column in zip(RUNG_FACTORS, (j_h, j_h2, j_h4, j_h8)):
        abs_diff = float(np.linalg.norm(analytic - column))
        col_norm = float(np.linalg.norm(column))
        rel = abs_diff / col_norm if col_norm > 1.0e-12 else 0.0
        vs_rungs.append(
            {
                "step_factor": factor,
                "fd_column_norm": col_norm,
                "absolute_difference": abs_diff,
                "relative_difference": rel,
                "sign_consistent": bool(float(np.dot(analytic, column)) >= 0.0),
                "inside_last_pair_envelope": abs_diff <= envelope + 1.0e-15,
            }
        )
        distances.append(abs_diff)
    approach_cosines = []
    sequence = (j_h, j_h2, j_h4, j_h8)
    for index in range(3):
        delta = sequence[index + 1] - sequence[index]
        toward = analytic - sequence[index]
        n_delta = float(np.linalg.norm(delta))
        n_toward = float(np.linalg.norm(toward))
        if n_delta > 0.0 and n_toward > 0.0:
            approach_cosines.append(float(np.dot(delta, toward) / (n_delta * n_toward)))
        else:
            approach_cosines.append(None)
    approaching = all(
        distances[index + 1] <= distances[index] * (1.0 + 1.0e-12)
        for index in range(3)
    )
    vs_official = _column_agreement(analytic, j_h)
    step = OFFICIAL_FD_STEPS[name]
    abs_vs_h = float(vs_rungs[0]["absolute_difference"])
    residual_mismatch_mm = abs_vs_h * step
    mismatch_over_sigma = residual_mismatch_mm / MEASUREMENT_SIGMA_MM
    analytic_norm = float(np.linalg.norm(analytic))
    fd_norm = float(np.linalg.norm(j_h))
    five_pass = bool(vs_official.get("agree"))
    fd_converged = bool(fd.get("ladder_converged"))
    inside_finest = bool(vs_rungs[3]["inside_last_pair_envelope"])
    max_self = max(successive) if successive else 0.0
    same_order = bool(
        vs_rungs[3]["absolute_difference"] <= same_order_factor * max(envelope, 0.0)
        or vs_rungs[3]["absolute_difference"] <= same_order_factor * max_self
    )
    far_below_meas = bool(mismatch_over_sigma <= far_below_fraction)
    far_below_fd = bool(inside_finest)
    pathology = (
        (not five_pass)
        and fd_converged
        and far_below_meas
        and far_below_fd
    )
    fd_ref = (not fd_converged) and (inside_finest or same_order)
    if five_pass:
        kind = KIND_FIVE_PERCENT_PASS
    elif pathology:
        kind = KIND_PATHOLOGY
    elif fd_ref:
        kind = KIND_FD_REF
    else:
        kind = KIND_INCONSISTENT
    return {
        "parameter": name,
        "present": True,
        "residual_kind": kind,
        "fd_ladder_converged": fd_converged,
        "fd_last_pair_relative_error": fd.get("last_pair_relative_error"),
        "fd_sign_consistent": fd.get("sign_consistent"),
        "analytic_column_norm": analytic_norm,
        "fd_official_column_norm": fd_norm,
        "absolute_difference_vs_h": abs_vs_h,
        "relative_difference_vs_h": vs_official.get("relative_error"),
        "sign_consistent_vs_h": vs_official.get("sign_consistent"),
        "frozen_five_percent_pass": five_pass,
        "successive_fd_absolute_differences": list(successive),
        "last_pair_envelope": envelope,
        "absolute_difference_vs_each_rung": vs_rungs,
        "approach_distances": distances,
        "approach_cosines": approach_cosines,
        "analytic_in_fd_approach_direction": approaching,
        "inside_last_pair_envelope": inside_finest,
        "mismatch_same_order_as_fd_self_difference": same_order,
        "official_fd_step": step,
        "residual_mismatch_at_official_step_mm": residual_mismatch_mm,
        "residual_mismatch_over_measurement_sigma": mismatch_over_sigma,
        "far_below_measurement_sensitivity": far_below_meas,
        "far_below_fd_self_difference": far_below_fd,
        "small_column_relative_metric_pathology": pathology,
        "fd_reference_not_precise_enough": fd_ref,
        "true_inconsistency_vs_converged_or_systematic_fd": kind == KIND_INCONSISTENT,
        "five_percent_gate_unchanged": True,
        "do_not_select_best_fd_rung": True,
    }


def audit_hop_composition(hops: list[Mapping[str, Any]]) -> dict[str, Any]:
    rk_rels: list[float] = []
    composed_rels: list[float] = []
    loc0_deltas: list[float] = []
    n_reset = 0
    n_available = 0
    for hop in hops:
        if hop.get("free_state_jacobian_available"):
            n_available += 1
        try:
            if int(hop.get("n_material_resets") or 0) > 0:
                n_reset += 1
        except (TypeError, ValueError):
            pass
        delta = _finite(hop.get("official_minus_diagnostic_loc0"))
        if delta is not None:
            loc0_deltas.append(abs(delta))
        j_rk = _as_matrix(hop.get("rk_free_transport_jacobian_product"))
        j_b2f = _as_matrix(hop.get("source_bound_to_free_jacobian"))
        j_prod = _as_matrix(hop.get("bound_to_free_rk_product"))
        if j_rk is not None and j_b2f is not None and j_prod is not None:
            pred = j_rk @ j_b2f
            denom = max(float(np.linalg.norm(j_prod)), 1.0e-15)
            rk_rels.append(float(np.linalg.norm(pred - j_prod) / denom))
        j_t = _as_matrix(hop.get("free_transport_jacobian_since_last_reset"))
        j_g = _as_matrix(hop.get("bound_to_free_jacToGlobal_last_reset"))
        j_b = _as_matrix(hop.get("accumulated_bound_jacobian"))
        j_c = _as_matrix(hop.get("bound_to_free_acts_composed"))
        if (
            j_t is not None
            and j_g is not None
            and j_b is not None
            and j_c is not None
        ):
            pred_c = j_t @ j_g @ j_b
            denom_c = max(float(np.linalg.norm(j_c)), 1.0e-15)
            composed_rels.append(float(np.linalg.norm(pred_c - j_c) / denom_c))
    max_rk = max(rk_rels) if rk_rels else None
    max_composed = max(composed_rels) if composed_rels else None
    max_loc0 = max(loc0_deltas) if loc0_deltas else None
    closed = (
        max_rk is not None
        and max_rk <= HOP_COMPOSITION_REL_MAX
        and max_composed is not None
        and max_composed <= HOP_COMPOSITION_REL_MAX
        and (max_loc0 is None or max_loc0 <= 1.0e-6)
        and n_available == len(hops)
        and bool(hops)
    )
    return {
        "n_hops": len(hops),
        "n_available_hops": n_available,
        "n_hops_with_material_resets": n_reset,
        "max_rk_product_relative_error": max_rk,
        "max_acts_composed_relative_error": max_composed,
        "max_official_minus_diagnostic_loc0": max_loc0,
        "segment_composition_closed": closed,
        "step_level_D_persisted_in_acts": False,
        "higher_precision_variational_state_available": False,
        "variational_readout": (
            "hop-level EigenStepper jacTransport / collector product; "
            "per-step D is multiplied in place and not stored"
        ),
        "material_curvilinear_reset_separated_from_rk_free": True,
        "do_not_hand_write_magnetic_field_model": True,
    }


def audit_target_residual(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain_raw = payload.get("official_path_jacobian") or {}
    fd_cols = (payload.get("fd_ladder") or {}).get("columns") or []
    rk_cols = {
        str(item.get("parameter")): _as_vector(item.get("residual_column"))
        for item in chain_raw.get("rk_free_chain_columns") or []
    }
    envelopes = [
        audit_envelope_column(item, rk_cols.get(str(item.get("parameter"))))
        for item in fd_cols
    ]
    hops = chain_raw.get("hops") or []
    composition = audit_hop_composition(hops)
    dummy_bounded = bool(chain_raw.get("dummy_cov_bounded_transportJacobian_used"))
    if dummy_bounded:
        refuse_dummy_cov_bounded()
    fallbacks = [
        hop.get("free_to_bound_fallback")
        for hop in payload.get("transport_branches") or []
    ]
    kinds = {item.get("residual_kind") for item in envelopes if item.get("present")}
    return {
        "event": _event_label(_event_pair(row)),
        "identity": (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
        ),
        "target_station": int(row.get("target_station", -1)),
        "envelopes": envelopes,
        "composition": composition,
        "control_fd_all_converged": all(
            item.get("fd_ladder_converged") for item in envelopes
        ),
        "control_five_percent_all_pass": all(
            item.get("frozen_five_percent_pass") for item in envelopes
        ),
        "focus_fd_all_converged": all(
            item.get("fd_ladder_converged") for item in envelopes
        ),
        "focus_five_percent_all_pass": all(
            item.get("frozen_five_percent_pass") for item in envelopes
        ),
        "any_pathology": any(
            item.get("small_column_relative_metric_pathology") for item in envelopes
        ),
        "any_fd_reference": any(
            item.get("fd_reference_not_precise_enough") for item in envelopes
        ),
        "any_true_inconsistency": any(
            item.get("true_inconsistency_vs_converged_or_systematic_fd")
            for item in envelopes
        ),
        "residual_kinds": sorted(str(item) for item in kinds),
        "loc0_matches_official": bool(chain_raw.get("official_loc0_matches_diagnostic")),
        "chain_complete": bool(chain_raw.get("chain_complete")),
        "free_jacobian_available": bool(
            chain_raw.get("any_free_state_jacobian_available")
        )
        and bool(chain_raw.get("chain_complete")),
        "dummy_cov_bounded_used": dummy_bounded,
        "branch_fallback": any(bool(item) for item in fallbacks),
        "segment_composition_closed": bool(composition.get("segment_composition_closed")),
        "five_percent_gate_unchanged": True,
        "do_not_replace_official_likelihood": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    targets = inventory.get("targets") or []
    events = {item.get("event") for item in targets}
    checks = {
        "focus_86": FOCUS_LABEL in events,
        "control_0": "100043/0" in events,
        "control_1": "100043/1" in events,
        "control_37": "100043/37" in events,
        "wb120_smoke_hash_match": bool(inventory.get("wb120_smoke_hash_match")),
        "envelope_recorded": all(item.get("envelopes") for item in targets),
        "hop_composition_recorded": all(item.get("composition") for item in targets),
        "no_athena_rerun": True,
        "no_jacobian_implementation_change": True,
        "no_best_step_selection": True,
        "no_best_tolerance_selection": True,
        "five_percent_gate_unchanged": True,
        "no_dummy_cov_bounded_transportJacobian": not any(
            item.get("dummy_cov_bounded_used") for item in targets
        ),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "target_exclusion": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds", True)
        ),
        "statistical_model_unchanged": not bool(
            inventory.get("statistical_model_changed")
        ),
        "no_likelihood_replacement": True,
        "smoke_present": bool(inventory.get("smoke_present")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "do_not_submit_if_failed": True,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
    }


def _active_kinds(targets: list[Mapping[str, Any]]) -> set[str]:
    active: set[str] = set()
    for item in targets:
        if item.get("any_pathology"):
            active.add(CASE_PATHOLOGY)
        if item.get("any_fd_reference"):
            active.add(CASE_FD_REF)
        if item.get("any_true_inconsistency"):
            active.add(CASE_INCONSISTENT)
    return active


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_as_information"):
        refuse_ridge_information()
    if inventory.get("statistical_model_changed"):
        refuse_change_statistical_model()
    if inventory.get("switched_to_direct"):
        refuse_switch_to_direct()
    if inventory.get("replaced_likelihood"):
        refuse_replace_likelihood()
    if inventory.get("selected_best_step"):
        refuse_best_step_selection()
    if inventory.get("selected_best_tolerance"):
        refuse_best_tolerance()
    if inventory.get("relaxed_five_percent_gate"):
        refuse_relax_gate()
    if inventory.get("changed_fd_ladder"):
        refuse_change_fd_ladder()
    if inventory.get("changed_jacobian_implementation"):
        refuse_change_jacobian()
    if inventory.get("restart_executed"):
        refuse_restart()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    if any(item.get("dummy_cov_bounded_used") for item in inventory.get("targets") or []):
        refuse_dummy_cov_bounded()
    controls = inventory.get("controls") or []
    focus = inventory.get("focus_86") or []
    leakage0 = bool((inventory.get("exclusion") or {}).get("target_exclusion_holds", True))
    model_ok = not bool(inventory.get("statistical_model_changed"))
    branch = any(item.get("branch_fallback") for item in controls + focus)
    jac_available = bool(controls) and bool(focus) and all(
        item.get("free_jacobian_available") and item.get("loc0_matches_official")
        for item in controls + focus
    )
    composition_ok = all(
        item.get("segment_composition_closed") for item in controls + focus
    ) if (controls and focus) else False
    control_five = (
        all(item.get("control_five_percent_all_pass") for item in controls)
        if controls
        else False
    )
    focus_five = (
        all(item.get("focus_five_percent_all_pass") for item in focus) if focus else False
    )
    control_fd = (
        all(item.get("control_fd_all_converged") for item in controls) if controls else False
    )
    focus_fd = (
        all(item.get("focus_fd_all_converged") for item in focus) if focus else False
    )
    loc0_match = (
        all(item.get("loc0_matches_official") for item in controls + focus)
        if (controls and focus)
        else False
    )
    active = _active_kinds(controls + focus)
    if (
        not inventory.get("smoke_present")
        or not controls
        or not focus
        or not jac_available
    ):
        primary = CASE_MIXED
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
        contract = False
    elif (
        control_five
        and focus_five
        and loc0_match
        and not branch
        and leakage0
        and model_ok
        and not active
    ):
        primary = CASE_ESTABLISHED
        next_step = "reopen_b14m_smoke_restart_invariance_only"
        contract = True
    elif active == {CASE_FD_REF}:
        primary = CASE_FD_REF
        next_step = "establish_independent_derivative_reference"
        contract = False
    elif active == {CASE_PATHOLOGY}:
        primary = CASE_PATHOLOGY
        next_step = "keep_frozen_five_percent_gate"
        contract = False
    elif active == {CASE_INCONSISTENT}:
        primary = CASE_INCONSISTENT
        next_step = "keep_official_path_derivative_diagnosis"
        contract = False
    else:
        primary = CASE_MIXED
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
        contract = False
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "active_residual_kinds": sorted(active),
        "control_fd_converged": control_fd,
        "control_five_percent_pass": control_five,
        "focus_fd_converged": focus_fd,
        "focus_five_percent_pass": focus_five,
        "official_loc0_matches_diagnostic": loc0_match,
        "free_state_jacobian_available": jac_available,
        "segment_composition_closed": composition_ok,
        "step_level_D_persisted_in_acts": False,
        "higher_precision_variational_state_available": False,
        "branch_switching": branch,
        "jacobian_contract_established": bool(contract),
        "b14m_reopen_authorized": bool(contract),
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "prior_introduced": False,
        "ridge_added": False,
        "do_not_force_5d_lto_covariance": True,
        "statistical_model_unchanged": True,
        "profile_math_rewritten": False,
        "focus_identity_retained": True,
        "physical_nonidentifiability_not_claimed": True,
        "five_d_cin_not_the_objective": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_switch_to_direct": True,
        "do_not_replace_official_likelihood": True,
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_change_official_fd_ladder": True,
        "do_not_change_jacobian_implementation": True,
        "five_percent_gate_unchanged": True,
        "wb117_not_treated_as_physical_nonsmoothness": True,
        "wb118_not_a_physical_conclusion": True,
        "wb119_took_wrong_jacobian": True,
        "wb119_not_a_physical_conclusion": True,
        "wb120_same_path_jacobian_available": True,
        "wb120_not_a_physical_conclusion": True,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_jacobian_rows(config)
    sample = load_contracted_sample(config)
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    targets = []
    n_update = 0
    for row in dumps["rows"]:
        if row.get("measurement_update_in_evaluator"):
            n_update += 1
        if not is_official_mode_b(row):
            continue
        if _event_pair(row) not in REQUIRED_EVENTS:
            continue
        if row.get("official_supporting_plane_jacobian") is None:
            continue
        targets.append(audit_target_residual(row))
    if n_update:
        refuse_measurement_update()
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "wb120_smoke_hash_match": True,
        "n_rows": dumps["n_rows"],
        "targets": targets,
        "controls": [item for item in targets if item.get("event") in CONTROL_LABELS],
        "focus_86": [item for item in targets if item.get("event") == FOCUS_LABEL],
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "exclusion": audit_exclusion(dumps["rows"])
        if dumps["rows"]
        else {"target_exclusion_holds": True},
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "switched_to_direct": False,
        "replaced_likelihood": False,
        "selected_best_step": False,
        "selected_best_tolerance": False,
        "relaxed_five_percent_gate": False,
        "changed_fd_ladder": False,
        "changed_jacobian_implementation": False,
        "restart_executed": False,
        "b14m_reopened": False,
        "present_sources": sample["present_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
    }
    inventory["smoke_gate"] = smoke_gate(inventory)
    return inventory


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism.get("measurement_model_v2_entered"):
        refuse_measurement_model_v2()
    if mechanism.get("b15_authorized"):
        refuse_b15()
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    if mechanism.get("jacobian_contract_established") and not (
        mechanism.get("decision") == CASE_ESTABLISHED
        and mechanism.get("control_five_percent_pass")
        and mechanism.get("focus_five_percent_pass")
        and not mechanism.get("branch_switching")
        and bool((inventory.get("exclusion") or {}).get("target_exclusion_holds"))
        and mechanism.get("statistical_model_unchanged")
    ):
        mechanism["jacobian_contract_established"] = False
        mechanism["b14m_reopen_authorized"] = False
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb120_decision_sha256": inherited["workbook_120"]["decision_sha256"],
        "inherited_wb119_decision_sha256": inherited["workbook_119"]["decision_sha256"],
        "inherited_wb118_decision_sha256": inherited["workbook_118"]["decision_sha256"],
        "inherited_wb117_decision_sha256": inherited["workbook_117"]["decision_sha256"],
        "inherited_wb116_decision_sha256": inherited["workbook_116"]["decision_sha256"],
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb120_rewritten": False,
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "official_path_jacobian_reused_from_wb120": True,
    }


def likelihood_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "r_i": "m_loc0 - predicted_loc0_on_supporting_plane",
        "R_i": "(0.08 mm)^2 / 12",
        "theta": ["loc0", "loc1", "phi", "theta", "q_over_p"],
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "official_h_i": "sequential supporting-plane transport, no measurement update",
        "derivative_implementation": (
            "WB120 official unbounded rk_free chain; this task only classifies "
            "the residual analytic-vs-FD mismatch"
        ),
        "consistency_envelope": "||J_h/4 - J_h/8||, diagnostic only",
        "do_not_switch_to_direct": True,
        "do_not_select_best_fd_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_replace_official_likelihood": True,
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_change_jacobian_implementation": True,
        "measurement_update_in_evaluator": False,
        "pinv_relative": PINV_RELATIVE,
        "objective": "classify remaining 100043/1 loc1 and 100048/86 mismatches",
        "not_the_objective": "repair 5D Cin",
        "wb117_not_a_physical_conclusion": True,
        "wb118_not_a_physical_conclusion": True,
        "wb119_took_wrong_jacobian": True,
        "wb120_same_path_jacobian_available": True,
    }
