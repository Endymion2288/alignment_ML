"""Task B14M-T: profile globalization repair and stationarity recontract.

Repairs only the numerical optimizer: explicit nuisance profiling plus
range-space trust-region globalization.  The statistical model, Jacobian,
and mean transport stay frozen.  LM lambda is a globalization multiplier
only: not prior, not ridge, not information.  Does not submit 1989.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping

import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.b14m_profile_basin_diagnosis import (
    CASE_A as WB130_DECISION,
    inherit_frozen_stage as inherit_through_wb130,
)
from datasets.b14m_restart_invariance import (
    RESTARTS,
    _branch,
    _chi2,
    _native,
    _pred_meas,
    _pred_target,
    _rel_chi2,
    recover_optimizer_contract,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.shadow_mean_transport_contract import FROZEN_FIELD_GRADIENT_SHA
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS

SCHEMA_VERSION = "b14m-profile-globalization-repair-v1"
DEFAULT_CONFIG = "configs/b14m_profile_globalization_repair_v1.yaml"
TASK = "SB-B14MT"
WORKBOOK = 131

CASE_A = "profile_globalization_and_restart_contract_established"
CASE_B = "trust_region_globalization_still_fails"
CASE_C = "inner_nuisance_profile_not_converged"
CASE_D = "profile_outer_optimizer_restart_sensitive"
CASE_E = "profile_nuisance_nonidentifiability_prediction_stable"
CASE_F = "optimizer_repair_regression"
CASE_G = "mixed_or_inconclusive"
CASE_INHERIT = "inherited_contract_mismatch"

ALLOWED_DECISIONS = (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    CASE_F,
    CASE_G,
    CASE_INHERIT,
)

SMOKE_EVENTS = ((100043, 0), (100043, 1), (100043, 37), (100048, 86))
TARGETS = (1, 2, 3)
REQUIRED_IDENTITIES = tuple(
    (run, event, target)
    for run, event in SMOKE_EVENTS
    for target in TARGETS
)
GATE_IDENTITIES = ((100043, 37, 1), (100043, 37, 2), (100048, 86, 1))
REGRESSION_CONTROLS = ((100043, 0, 1), (100043, 1, 1), (100048, 86, 2))
STALL_CHI2_CEILING = {
    (100043, 37, 1): 1.0e4,
    (100043, 37, 2): 1.0e4,
}
GRADIENT_TOL = 1.0e-6
VALID_TERMINATIONS = {
    "supported_stationary",
    "step_converged",
}
FORBIDDEN_SUCCESS_TERMINATIONS = {"flat_direction"}
WB130_FROZEN = WB130_DECISION


class B14MProfileGlobalizationError(ValueError):
    """Raised when the B14M-T contract is illegal."""


def refuse_prior() -> None:
    raise B14MProfileGlobalizationError("B14M-T must not introduce a prior")


def refuse_ridge() -> None:
    raise B14MProfileGlobalizationError(
        "B14M-T must not add ridge as statistical information"
    )


def refuse_smaller_lambda() -> None:
    raise B14MProfileGlobalizationError(
        "B14M-T must not extend the frozen line search with smaller lambda"
    )


def refuse_min_restarts() -> None:
    raise B14MProfileGlobalizationError("must not take min of four restarts as a fix")


def refuse_held_out() -> None:
    raise B14MProfileGlobalizationError(
        "held-out target must not enter globalization or stationarity"
    )


def refuse_full_sample() -> None:
    raise B14MProfileGlobalizationError(
        "the 1989-row campaign must not be submitted in B14M-T"
    )


def refuse_b15() -> None:
    raise B14MProfileGlobalizationError("Task B15 is not entered in Task B14M-T")


def refuse_jacobian_reopen() -> None:
    raise B14MProfileGlobalizationError("WB128 Jacobian diagnosis must stay closed")


def refuse_retune_tr() -> None:
    raise B14MProfileGlobalizationError(
        "trust-region constants are preregistered and must not be retuned"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise B14MProfileGlobalizationError(f"{label} hash mismatch: {digest}")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _identity_label(item: tuple[int, int, int]) -> str:
    return f"{item[0]}/{item[1]} T{item[2]}"


def _identity_of(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("run_id", -1)),
        int(row.get("event_id", -1)),
        int(row.get("target_station", -1)),
    )


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise B14MProfileGlobalizationError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise B14MProfileGlobalizationError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise B14MProfileGlobalizationError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_submit_1989",
        "do_not_change_statistical_model",
        "do_not_change_parameter_scaling",
        "do_not_change_pinv_relative",
        "do_not_add_random_restarts",
        "do_not_take_min_of_four_restarts_as_fix",
        "do_not_drop_event_37",
        "do_not_drop_event_86",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise B14MProfileGlobalizationError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14mt", True)):
        raise B14MProfileGlobalizationError("do_not_enter_b14mt must be false")
    if bool(config.get("do_not_change_optimizer", True)):
        raise B14MProfileGlobalizationError(
            "do_not_change_optimizer must be false: WB131 repairs globalization only"
        )
    spec = config["b14m_profile_globalization_repair"]
    events = [tuple(item) for item in spec["smoke_events"]]
    gates = [tuple(item) for item in spec["gate_identities"]]
    controls = [tuple(item) for item in spec["regression_controls"]]
    if events != list(SMOKE_EVENTS):
        raise B14MProfileGlobalizationError("smoke events must stay 0/1/37/86")
    if gates != list(GATE_IDENTITIES):
        raise B14MProfileGlobalizationError("gate identities must stay 37/T1, 37/T2, 86/T1")
    if controls != list(REGRESSION_CONTROLS):
        raise B14MProfileGlobalizationError("regression controls must stay 0/T1, 1/T1, 86/T2")
    if 44 in {item[1] for item in events}:
        raise B14MProfileGlobalizationError("must not replace the smoke set with 100043/44")
    if spec.get("reused_existing_project_trust_region") is True:
        raise B14MProfileGlobalizationError(
            "must not silently reuse the alignment 0.15 trust-region contract"
        )
    if float(spec["pinv_relative"]) != PINV_RELATIVE:
        raise B14MProfileGlobalizationError("pinv_relative must stay 1e-8")
    if float(spec["chi2_rel_tolerance"]) != 0.01:
        raise B14MProfileGlobalizationError("chi2_rel_tolerance is frozen at 0.01")
    if float(spec["prediction_abs_tolerance_mm"]) != 0.1:
        raise B14MProfileGlobalizationError("prediction_abs_tolerance_mm is frozen at 0.1")
    loc0 = float(spec["supported_abs_tolerance"]["loc0_mm"])
    theta = float(spec["supported_abs_tolerance"]["theta"])
    if loc0 != 0.05 or theta != 1.0e-4:
        raise B14MProfileGlobalizationError("supported_abs_tolerance is frozen")
    smoke_root = str(config.get("profile_smoke_root", ""))
    if "b14m_reopen_smoke" in smoke_root or smoke_root.endswith("b14m_smoke"):
        raise B14MProfileGlobalizationError("must not overwrite frozen smoke roots")
    if "b14ms_basin_smoke" in smoke_root:
        raise B14MProfileGlobalizationError("must not overwrite WB130 basin smoke")
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb130(config)
    spec = config["inheritance"]["workbook_130"]
    decision_path = resolve_under_root(project_root(), spec["decision_path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_130 config",
    )
    _expect_sha(decision_path, spec["decision_sha256"], "workbook_130 decision")
    _expect_sha(
        resolve_under_root(project_root(), spec["dump_100043_path"]),
        spec["dump_100043_sha256"],
        "workbook_130 dump 100043",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["dump_100048_path"]),
        spec["dump_100048_sha256"],
        "workbook_130 dump 100048",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise B14MProfileGlobalizationError("workbook_130 decision must stay frozen")
    if spec["frozen_decision"] != WB130_FROZEN:
        raise B14MProfileGlobalizationError("WB130 decision token mismatch")
    if decision.get("full_sample_authorized"):
        refuse_full_sample()
    if not decision.get("jacobian_contract_established"):
        raise B14MProfileGlobalizationError("WB130 lost the Jacobian contract")
    repair_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        )
    )
    if repair_sha != FROZEN_FIELD_GRADIENT_SHA:
        raise B14MProfileGlobalizationError(
            "inherited_contract_mismatch: FieldGradient SHA"
        )
    inherited["workbook_130"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "verdict": spec["frozen_verdict"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "nuisance_profile_multibasin": spec["frozen_nuisance_profile_multibasin"],
        "profile_objective_multimodality": spec["frozen_profile_objective_multimodality"],
        "profile_hysteresis": spec["frozen_profile_hysteresis"],
        "b14m_smoke_passed": False,
        "restart_invariance_established": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "jacobian_contract_established": True,
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
    }
    inherited["workbook_103"] = {
        "contracted_denominator": FROZEN_N_CONTRACTED,
        "n_raw": FROZEN_N_RAW,
        "n_ineligible": FROZEN_N_INELIGIBLE,
        "n_official_pairs": FROZEN_N_OFFICIAL_PAIRS,
    }
    inherited["current"] = {
        "full_sample_authorized": False,
        "restart_invariance_authorized": False,
        "restart_invariance_established": False,
        "b14m_smoke_passed": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "field_gradient_extension_sha256": FROZEN_FIELD_GRADIENT_SHA,
    }
    return inherited


def likelihood_contract() -> dict[str, Any]:
    return {
        "theta": ["loc0", "loc1", "phi", "theta", "q_over_p"],
        "alpha": ["loc0", "theta"],
        "nu": ["loc1", "phi", "q_over_p"],
        "prior": None,
        "ridge": None,
        "truth_q_over_p": None,
        "fixed_q_over_p": False,
        "deleted_q_over_p": False,
        "target_measurement_in_optimizer": False,
        "q_over_p_is_explicit_nuisance": True,
        "R_mm2": 0.08 * 0.08 / 12.0,
        "chi2_prof_definition": "min_nu chi2(alpha, nu)",
        "lm_lambda_is_not_prior": True,
        "lm_lambda_is_not_ridge": True,
        "lm_lambda_is_not_information": True,
        "trust_region_is_algorithmic_globalization_only": True,
        "do_not_submit_1989": True,
        "statistical_model_unchanged": True,
    }


def preregistered_trust_region(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["b14m_profile_globalization_repair"]
    return {
        "coordinate_system": spec["coordinate_system"],
        "provenance": spec["provenance"],
        "reused_existing_project_trust_region": False,
        "pinv_relative": float(spec["pinv_relative"]),
        "gradient_norm_z": float(spec["gradient_norm_z"]),
        "step_norm_z": float(spec["step_norm_z"]),
        "relative_chi2_decrease": float(spec["relative_chi2_decrease"]),
        "max_outer_iterations": int(spec["max_outer_iterations"]),
        "max_inner_iterations": int(spec["max_inner_iterations"]),
        "max_trust_region_trials": int(spec["max_trust_region_trials"]),
        "Delta0_rule": spec["Delta0_rule"],
        "Delta_max": float(spec["Delta_max"]),
        "Delta_min": float(spec["Delta_min"]),
        "rho_accept": float(spec["rho_accept"]),
        "rho_expand": float(spec["rho_expand"]),
        "boundary_frac": float(spec["boundary_frac"]),
        "shrink_factor": float(spec["shrink_factor"]),
        "expand_factor": float(spec["expand_factor"]),
        "lambda_meaning": "numerical globalization multiplier only",
    }


def load_repair_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    smoke_root = resolve_under_root(project_root(), str(config["profile_smoke_root"]))
    name = str(config.get("profile_smoke_filename"))
    if smoke_root.is_dir():
        matches = sorted(smoke_root.rglob(name))
        if not matches:
            matches = sorted(smoke_root.rglob("*.jsonl"))
        for path in matches:
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    optimize = [
        row
        for row in rows
        if row.get("row_role") == "profile_optimize"
        and row.get("solver") == "explicit_profile_range_space_trust_region"
        and not bool(row.get("evaluate_only"))
    ]
    return {
        "rows": rows,
        "optimize_rows": optimize,
        "n_rows": len(rows),
        "n_optimize": len(optimize),
        "smoke_present": bool(optimize),
        "paths": paths,
    }


def _row_stationary(row: Mapping[str, Any], gradient_tol: float) -> bool:
    gn = _finite(row.get("norm_g_n_R"))
    ga = _finite(row.get("norm_g_alpha_prof_z"))
    inner = bool(row.get("inner_nuisance_stationary"))
    outer = bool(row.get("profile_alpha_stationary"))
    flagged = bool(row.get("valid_stationary"))
    if gn is None or ga is None:
        return False
    return bool(
        flagged
        and inner
        and outer
        and gn <= gradient_tol
        and ga <= gradient_tol
    )


def _row_valid(row: Mapping[str, Any], gradient_tol: float) -> bool:
    chi2 = _chi2(row)
    term = str(row.get("termination_reason") or "")
    return bool(
        chi2 is not None
        and math.isfinite(chi2)
        and bool(row.get("ok", False))
        and bool(row.get("propagation_success", False))
        and bool(row.get("valid_solution", False))
        and _row_stationary(row, gradient_tol)
        and term in VALID_TERMINATIONS
        and term not in FORBIDDEN_SUCCESS_TERMINATIONS
        and int(row.get("target_station_measurements_used") or 0) == 0
        and bool(row.get("held_out_used_for_solver")) is False
        and bool(row.get("ridge_added")) is False
        and bool(row.get("prior_term_present")) is False
    )


def audit_identity(
    identity: tuple[int, int, int],
    rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
    tr: Mapping[str, Any],
) -> dict[str, Any]:
    by_restart = {row.get("profile_init_variant"): row for row in rows}
    missing = [name for name in RESTARTS if name not in by_restart]
    chi2_rel = float(contract["chi2_rel_tolerance"])
    pred_abs = float(contract["prediction_abs_tolerance_mm"])
    loc0_abs = float(contract["supported_abs_tolerance"]["loc0_mm"])
    theta_abs = float(contract["supported_abs_tolerance"]["theta"])
    gradient_tol = float(tr.get("gradient_norm_z", GRADIENT_TOL))
    if missing:
        return {
            "identity": _identity_label(identity),
            "present": False,
            "missing_restarts": missing,
            "all_stationary": False,
            "optimization_valid": False,
            "objective_invariance": False,
            "prediction_invariance": False,
            "transport_branch_invariance": False,
            "case": CASE_G,
        }
    restarts = [by_restart[name] for name in RESTARTS]
    stationaries = [_row_stationary(row, gradient_tol) for row in restarts]
    valids = [_row_valid(row, gradient_tol) for row in restarts]
    chi2s = [_chi2(row) for row in restarts]
    preds = [_pred_target(row) for row in restarts]
    meas = [_pred_meas(row) for row in restarts]
    natives = [_native(row) for row in restarts]
    branches = [_branch(row) for row in restarts]
    terms = [str(row.get("termination_reason") or "") for row in restarts]
    inners = [bool(row.get("inner_nuisance_stationary")) for row in restarts]
    gn = [_finite(row.get("norm_g_n_R")) for row in restarts]
    ga = [_finite(row.get("norm_g_alpha_prof_z")) for row in restarts]
    pairwise_chi2 = []
    pairwise_pred = []
    pairwise_meas = []
    for i in range(4):
        for j in range(i + 1, 4):
            if chi2s[i] is None or chi2s[j] is None:
                pairwise_chi2.append(None)
            else:
                pairwise_chi2.append(_rel_chi2(chi2s[i], chi2s[j]))
            if preds[i] is None or preds[j] is None:
                pairwise_pred.append(None)
            else:
                pairwise_pred.append(abs(preds[i] - preds[j]))
            keys = set(meas[i]) | set(meas[j])
            diffs = []
            for key in keys:
                if key not in meas[i] or key not in meas[j]:
                    diffs.append(None)
                    continue
                diffs.append(abs(meas[i][key] - meas[j][key]))
            pairwise_meas.append(
                None if any(item is None for item in diffs) else max(diffs or [0.0])
            )
    finite_chi2 = [item for item in pairwise_chi2 if item is not None]
    finite_pred = [item for item in pairwise_pred if item is not None]
    finite_meas = [item for item in pairwise_meas if item is not None]
    objective_ok = bool(finite_chi2) and all(item <= chi2_rel for item in finite_chi2)
    pred_ok = (
        bool(finite_pred)
        and all(item <= pred_abs for item in finite_pred)
        and bool(finite_meas)
        and all(item <= pred_abs for item in finite_meas)
        and all(set(meas[0]) == set(item) for item in meas)
    )
    branch_ok = all(item == branches[0] for item in branches) and bool(branches[0])
    alpha_ok = all(
        natives[i]["loc0"] is not None
        and natives[j]["loc0"] is not None
        and natives[i]["theta"] is not None
        and natives[j]["theta"] is not None
        and abs(natives[i]["loc0"] - natives[j]["loc0"]) <= loc0_abs
        and abs(natives[i]["theta"] - natives[j]["theta"]) <= theta_abs
        for i in range(4)
        for j in range(i + 1, 4)
    )
    nu_deltas = []
    for i in range(4):
        for j in range(i + 1, 4):
            for name in ("loc1", "phi", "q_over_p_per_mev"):
                a = natives[i][name]
                b = natives[j][name]
                if a is None or b is None:
                    nu_deltas.append(None)
                else:
                    nu_deltas.append(abs(a - b))
    nu_unique = all(item is not None and item <= 1.0e-12 for item in nu_deltas)
    leaked = any(
        int(row.get("target_station_measurements_used") or 0) != 0
        or bool(row.get("held_out_used_for_solver"))
        for row in restarts
    )
    flat = any(term == "flat_direction" for term in terms)
    inner_fail = any(
        (not inner)
        or str(row.get("inner_termination") or "")
        in {"inner_nuisance_profile_not_converged", "globalization_failure"}
        or str(row.get("termination_reason") or "")
        == "inner_nuisance_profile_not_converged"
        for inner, row in zip(inners, restarts)
    )
    glob_fail = any(
        str(row.get("termination_reason") or "") == "globalization_failure"
        or str(row.get("outer_termination") or "") == "globalization_failure"
        for row in restarts
    )
    ceiling = STALL_CHI2_CEILING.get(identity)
    finite_chi2_vals = [item for item in chi2s if item is not None]
    stall = bool(
        ceiling is not None
        and finite_chi2_vals
        and min(finite_chi2_vals) >= ceiling
    )
    all_stat = all(stationaries) and all(valids) and not flat and not leaked
    if leaked or flat:
        case = CASE_G
    elif inner_fail and not all_stat:
        case = CASE_C
    elif (glob_fail or stall) and not all_stat:
        case = CASE_B
    elif not all_stat:
        case = CASE_G
    elif not objective_ok or not pred_ok or not branch_ok:
        case = CASE_D
    elif not nu_unique:
        case = CASE_E
    else:
        case = CASE_A
    return {
        "identity": _identity_label(identity),
        "present": True,
        "missing_restarts": [],
        "all_stationary": all_stat,
        "optimization_valid": all(valids),
        "valid_restarts": {name: valids[i] for i, name in enumerate(RESTARTS)},
        "stationary_restarts": {
            name: stationaries[i] for i, name in enumerate(RESTARTS)
        },
        "chi2_prof": {name: chi2s[i] for i, name in enumerate(RESTARTS)},
        "predicted_target_loc0": {name: preds[i] for i, name in enumerate(RESTARTS)},
        "max_pairwise_delta_chi2_prof": max(finite_chi2) if finite_chi2 else None,
        "max_pairwise_delta_target_loc0": max(finite_pred) if finite_pred else None,
        "max_pairwise_delta_surviving_loc0": max(finite_meas) if finite_meas else None,
        "objective_invariance": bool(all_stat and objective_ok),
        "prediction_invariance": bool(all_stat and pred_ok),
        "parameter_invariance_alpha": bool(all_stat and alpha_ok),
        "nuisance_nonunique": bool(all_stat and not nu_unique),
        "transport_branch_invariance": bool(all_stat and branch_ok),
        "inner_nuisance_profile_valid": all(inners) and all(
            item is not None and item <= gradient_tol for item in gn
        ),
        "profile_alpha_stationary": all(
            item is not None and item <= gradient_tol for item in ga
        ),
        "target_leakage": leaked,
        "used_flat_direction_success": flat,
        "stall_not_escaped": stall,
        "inner_not_converged": inner_fail and not all_stat,
        "globalization_failed": glob_fail and not all_stat,
        "terminations": {name: terms[i] for i, name in enumerate(RESTARTS)},
        "norm_g_n_R": {name: gn[i] for i, name in enumerate(RESTARTS)},
        "norm_g_alpha_prof_z": {name: ga[i] for i, name in enumerate(RESTARTS)},
        "hessian_ranks": {
            name: restarts[i].get("hessian_rank") for i, name in enumerate(RESTARTS)
        },
        "inner_iterations": {
            name: restarts[i].get("inner_iterations_total")
            for i, name in enumerate(RESTARTS)
        },
        "outer_iterations": {
            name: restarts[i].get("profile_n_iterations")
            for i, name in enumerate(RESTARTS)
        },
        "accepted_trials": {
            name: restarts[i].get("trust_region_accepted_trials")
            for i, name in enumerate(RESTARTS)
        },
        "rejected_trials": {
            name: restarts[i].get("trust_region_rejected_trials")
            for i, name in enumerate(RESTARTS)
        },
        "final_Delta": {
            name: restarts[i].get("final_Delta_outer") for i, name in enumerate(RESTARTS)
        },
        "natives": natives,
        "case": case,
        "restarts": {
            name: {
                "termination": terms[i],
                "chi2": chi2s[i],
                "valid_stationary": stationaries[i],
                "valid_solution": valids[i],
                "norm_g_n_R": gn[i],
                "norm_g_alpha_prof_z": ga[i],
                "norm_g_R": _finite(restarts[i].get("norm_g_R")),
                "inner_iterations": restarts[i].get("inner_iterations_total"),
                "outer_iterations": restarts[i].get("profile_n_iterations"),
                "accepted_trials": restarts[i].get("trust_region_accepted_trials"),
                "rejected_trials": restarts[i].get("trust_region_rejected_trials"),
                "final_Delta": restarts[i].get("final_Delta_outer"),
                "rank": restarts[i].get("hessian_rank"),
                "predicted_target_loc0": preds[i],
            }
            for i, name in enumerate(RESTARTS)
        },
    }


def classify_overall(
    per: Mapping[str, Mapping[str, Any]],
    *,
    complete: bool,
    leaked: bool,
) -> dict[str, Any]:
    if leaked:
        return {
            "decision": CASE_G,
            "primary_case": CASE_G,
            "verdict": "FAIL",
            "reason": "target_leakage",
        }
    if not complete:
        return {
            "decision": CASE_G,
            "primary_case": CASE_G,
            "verdict": "FAIL",
            "reason": "smoke_incomplete",
        }
    cases = {label: item.get("case") for label, item in per.items()}
    control_cases = {
        _identity_label(item): cases.get(_identity_label(item))
        for item in REGRESSION_CONTROLS
    }
    gate_cases = {
        _identity_label(item): cases.get(_identity_label(item))
        for item in GATE_IDENTITIES
    }
    identity_cases = list(cases.values())
    control_ok = all(case in {CASE_A, CASE_E} for case in control_cases.values())
    gate_ok = all(case in {CASE_A, CASE_E} for case in gate_cases.values())
    fail_set = {
        case
        for case in identity_cases
        if case not in {CASE_A, CASE_E}
    }
    if gate_ok and not control_ok:
        return {
            "decision": CASE_F,
            "primary_case": CASE_F,
            "verdict": "FAIL",
            "reason": "optimizer_repair_regression",
            "control_cases": control_cases,
            "gate_cases": gate_cases,
        }
    if not fail_set:
        return {
            "decision": CASE_A,
            "primary_case": CASE_A,
            "verdict": "PASS",
            "reason": "stationarity_then_restart_invariance",
            "control_cases": control_cases,
            "gate_cases": gate_cases,
            "pass_compatible_nonidentifiability": CASE_E in identity_cases,
        }
    if len(fail_set) == 1:
        only = next(iter(fail_set))
        return {
            "decision": only,
            "primary_case": only,
            "verdict": "FAIL",
            "reason": "uniform_failure_after_stationarity_gate",
            "control_cases": control_cases,
            "gate_cases": gate_cases,
        }
    return {
        "decision": CASE_G,
        "primary_case": CASE_G,
        "verdict": "FAIL",
        "reason": "mixed_failure_mechanisms",
        "control_cases": control_cases,
        "gate_cases": gate_cases,
        "failure_cases": {
            label: case
            for label, case in cases.items()
            if case not in {CASE_A, CASE_E}
        },
    }


def inventory_and_audit(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    tr = preregistered_trust_region(config)
    loaded = load_repair_rows(config)
    by_identity: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in loaded["optimize_rows"]:
        by_identity[_identity_of(row)].append(row)
    per: dict[str, Any] = {}
    for identity in REQUIRED_IDENTITIES:
        role = (
            "gate"
            if identity in GATE_IDENTITIES
            else "regression_control"
            if identity in REGRESSION_CONTROLS
            else "recontract"
        )
        audit = audit_identity(identity, by_identity.get(identity, []), contract, tr)
        audit["role"] = role
        per[_identity_label(identity)] = audit
    leaked = any(item.get("target_leakage") for item in per.values())
    complete = all(item.get("present") for item in per.values()) and loaded[
        "n_optimize"
    ] >= 48
    traces = {
        "inner": [
            {
                "identity": _identity_label(_identity_of(row)),
                "variant": row.get("profile_init_variant"),
                "inner_nuisance_trace": row.get("inner_nuisance_trace"),
            }
            for row in loaded["optimize_rows"]
        ],
        "outer": [
            {
                "identity": _identity_label(_identity_of(row)),
                "variant": row.get("profile_init_variant"),
                "optimizer_trace": row.get("optimizer_trace"),
            }
            for row in loaded["optimize_rows"]
        ],
        "trials": [
            {
                "identity": _identity_label(_identity_of(row)),
                "variant": row.get("profile_init_variant"),
                "trust_region_trial_trace": row.get("trust_region_trial_trace"),
            }
            for row in loaded["optimize_rows"]
        ],
    }
    return {
        "loaded": {
            "n_rows": loaded["n_rows"],
            "n_optimize": loaded["n_optimize"],
            "smoke_present": loaded["smoke_present"],
            "paths": loaded["paths"],
        },
        "per_identity": per,
        "complete_48": complete,
        "target_leakage": leaked,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "lm_lambda_is_not_prior": True,
        "traces": traces,
        "trust_region": tr,
    }


def decide(
    inventory: Mapping[str, Any],
    inherited: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("optimizer_contract_not_recoverable"):
        return {
            "decision": CASE_INHERIT,
            "primary_case": CASE_INHERIT,
            "verdict": "FAIL",
            "restart_invariance_established": False,
            "restart_invariance_authorized": False,
            "b14m_smoke_passed": False,
            "full_sample_authorized": False,
            "reason": "optimizer_contract_not_recoverable",
        }
    mechanism = classify_overall(
        inventory.get("per_identity") or {},
        complete=bool(inventory.get("complete_48")),
        leaked=bool(inventory.get("target_leakage")),
    )
    passed = mechanism.get("decision") == CASE_A
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb130_decision_sha256": inherited["workbook_130"]["decision_sha256"],
        "inherited_wb129_decision_sha256": inherited["workbook_129"]["decision_sha256"],
        "inherited_wb128_decision_sha256": inherited["workbook_128"]["decision_sha256"],
        "inherited_wb127_decision_sha256": inherited["workbook_127"]["decision_sha256"],
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "lm_lambda_is_not_prior": True,
        "lm_lambda_is_not_ridge": True,
        "lm_lambda_is_not_information": True,
        "trust_region_is_algorithmic_globalization_only": True,
        "do_not_submit_1989": True,
        "restart_invariance_established": passed,
        "restart_invariance_authorized": passed,
        "b14m_smoke_passed": passed,
        "full_sample_authorized": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "jacobian_contract_established": True,
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": True,
        "next_step": "WB132_full_sample_preflight" if passed else "remain_on_B14MT",
        "denominator": {
            "n_contracted": inherited["workbook_103"]["contracted_denominator"],
            "n_raw": inherited["workbook_103"]["n_raw"],
            "n_ineligible": inherited["workbook_103"]["n_ineligible"],
            "n_official_pairs": inherited["workbook_103"]["n_official_pairs"],
            "frozen_denominator_holds": True,
        },
    }
