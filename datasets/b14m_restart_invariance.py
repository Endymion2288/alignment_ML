"""Task B14M-R: profile-likelihood smoke reopen and restart invariance.

Reopens the frozen WB114 measurement-level profile on the login-scale
smoke set only.  The numerical Jacobian is the inherited WB123 repaired
tangent plus WB127 certified mean semantics.  That is a derivative
implementation, not a new statistical model.  Does not submit 1989,
enter B15 / WB130 / V4 C/D / Measurement Model V2 / alignment / ML,
retune Gauss-Newton, drop 37 or 86, or treat rank<5 as automatic FAIL.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.profiled_measurement_likelihood import (
    ALPHA_INDICES,
    ALPHA_NAMES,
    NU_INDICES,
    NU_NAMES,
    PINV_RELATIVE,
    explicit_nuisance_minimum,
    joint_nls_step,
    profile_linear_solution,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.certified_mean_common_grid_fd import (
    CASE_ESTABLISHED as WB128_DECISION,
    inherit_frozen_stage as inherit_through_wb128,
)
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profile_likelihood_numerics import audit_exclusion
from datasets.shadow_mean_transport_contract import FROZEN_FIELD_GRADIENT_SHA
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "b14m-restart-invariance-v2"
DEFAULT_CONFIG = "configs/b14m_restart_invariance_v2.yaml"
TASK = "SB-B14MR"
WORKBOOK = 129

CASE_A = "b14m_restart_invariance_established"
CASE_B = "profile_optimizer_restart_sensitive"
CASE_C = "profile_nuisance_nonidentifiability_prediction_stable"
CASE_D = "profile_nuisance_nonidentifiability_affects_prediction"
CASE_E = "restart_transport_branch_instability"
CASE_F = "profile_schur_implementation_inconsistent"
CASE_G = "mixed_or_inconclusive"
CASE_INHERIT = "inherited_contract_mismatch"
CASE_OPT = "optimizer_contract_not_recoverable"

ALLOWED_DECISIONS = (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    CASE_F,
    CASE_G,
    CASE_INHERIT,
    CASE_OPT,
)

REQUIRED_EVENTS = ((100043, 0), (100043, 1), (100043, 37), (100048, 86))
SCHUR_EVENTS = ((100043, 0), (100048, 86))
RESTARTS = ("nominal", "loc1_plus_1mm", "phi_plus_1e-3", "qoverp_times_1p1")
TARGETS = (1, 2, 3)
VALID_TERMINATIONS = {
    "converged",
    "flat_direction",
    "max_iterations",
}
FORBIDDEN_JACOBIANS = (
    "wb119_dummy_cov_bounded_transportJacobian",
    "wb120_incomplete_rk_free",
    "wb124_adaptive_dopri5",
    "wb125_classical_rk4_shadow",
)


class B14MRestartInvarianceError(ValueError):
    """Raised when the B14M-R contract is illegal."""


def refuse_prior() -> None:
    raise B14MRestartInvarianceError("B14M-R must not introduce a prior")


def refuse_ridge() -> None:
    raise B14MRestartInvarianceError("B14M-R must not add ridge as information")


def refuse_truth_qoverp() -> None:
    raise B14MRestartInvarianceError("truth q/p is forbidden")


def refuse_qoverp_fix() -> None:
    raise B14MRestartInvarianceError("q/p must remain an explicit nuisance")


def refuse_b15() -> None:
    raise B14MRestartInvarianceError("Task B15 is not entered in Task B14M-R")


def refuse_full_sample() -> None:
    raise B14MRestartInvarianceError("the 1989-row campaign must not be submitted in B14M-R")


def refuse_wb130() -> None:
    raise B14MRestartInvarianceError("WB130 preflight is not entered in Task B14M-R")


def refuse_drop_37() -> None:
    raise B14MRestartInvarianceError("focus identity 100043/37 must be retained")


def refuse_drop_86() -> None:
    raise B14MRestartInvarianceError("focus identity 100048/86 must be retained")


def refuse_replace_44() -> None:
    raise B14MRestartInvarianceError("must not replace the smoke set with 100043/44")


def refuse_hop6() -> None:
    raise B14MRestartInvarianceError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def refuse_best_seed() -> None:
    raise B14MRestartInvarianceError("must not select the best restart")


def refuse_retune() -> None:
    raise B14MRestartInvarianceError("must not retune Gauss-Newton after seeing WB129")


def refuse_blame_86_fd() -> None:
    raise B14MRestartInvarianceError(
        "86 historical FD instability is closed by WB128 and is not a B14M-R excuse"
    )


def refuse_overwrite_legacy() -> None:
    raise B14MRestartInvarianceError("must not overwrite WB114-WB128 dumps")


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise B14MRestartInvarianceError(f"{label} hash mismatch: {digest}")


def _event_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _event_label(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise B14MRestartInvarianceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise B14MRestartInvarianceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise B14MRestartInvarianceError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14zc",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
        "do_not_submit_1989",
        "do_not_drop_event_37",
        "do_not_drop_event_86",
        "do_not_invent_optimizer_tolerance",
        "do_not_retune_gauss_newton",
        "do_not_treat_rank_lt_5_as_fail",
        "do_not_blame_86_historical_fd",
    ):
        if not bool(config.get(key, False)):
            raise B14MRestartInvarianceError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14m", True)):
        raise B14MRestartInvarianceError("do_not_enter_b14m must be false")
    if bool(config.get("do_not_enter_b14mr", True)):
        raise B14MRestartInvarianceError("do_not_enter_b14mr must be false")
    spec = config["b14m_restart_invariance"]
    events = [tuple(item) for item in spec["smoke_events"]]
    if events != list(REQUIRED_EVENTS):
        raise B14MRestartInvarianceError("smoke events must stay 0/1/37/86")
    if 44 in {event[1] for event in events}:
        refuse_replace_44()
    if list(spec["target_stations"]) != list(TARGETS):
        raise B14MRestartInvarianceError("target stations must stay 1/2/3")
    names = [item["name"] for item in spec["restarts"]]
    if names != list(RESTARTS):
        raise B14MRestartInvarianceError("restarts must stay R0-R3 pre-registered names")
    smoke_root = str(config.get("profile_smoke_root", ""))
    if "b14m_smoke" in smoke_root and "b14m_reopen_smoke" not in smoke_root:
        refuse_overwrite_legacy()
    return config


def recover_optimizer_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read frozen WB114/WB115 numbers.  Do not invent tolerances."""
    root = project_root()
    wb114_path = resolve_under_root(
        root, config["inheritance"]["workbook_114"]["config_path"]
    )
    wb115_path = resolve_under_root(
        root, config["inheritance"]["workbook_115"]["config_path"]
    )
    _expect_sha(
        wb114_path,
        config["inheritance"]["workbook_114"]["config_sha256"],
        "workbook_114 config",
    )
    _expect_sha(
        wb115_path,
        config["inheritance"]["workbook_115"]["config_sha256"],
        "workbook_115 config",
    )
    wb114 = yaml.safe_load(wb114_path.read_text(encoding="utf-8"))
    wb115 = yaml.safe_load(wb115_path.read_text(encoding="utf-8"))
    num114 = wb114.get("profile_numerical") or {}
    seed114 = wb114.get("profile_seed_invariance") or {}
    opt115 = wb115.get("profile_optimizer") or {}
    seed115 = wb115.get("profile_restart_v2") or {}
    required = {
        "pinv_relative": num114.get("pinv_relative"),
        "wb114_max_iterations": num114.get("max_iterations"),
        "wb114_gn_step_damping": num114.get("gn_step_damping"),
        "wb114_fd_steps": num114.get("fd_steps"),
        "wb114_joint_profile_schur_abs_tol": num114.get("joint_profile_schur_abs_tol"),
        "wb114_joint_profile_schur_rel_tol": num114.get("joint_profile_schur_rel_tol"),
        "chi2_rel_tolerance": seed114.get("chi2_rel_tolerance"),
        "prediction_abs_tolerance_mm": seed114.get("prediction_abs_tolerance_mm"),
        "supported_abs_tolerance": seed114.get("supported_abs_tolerance"),
        "wb115_max_iterations": opt115.get("max_iterations"),
        "wb115_line_search": opt115.get("line_search"),
        "wb115_relative_chi2_decrease": opt115.get("relative_chi2_decrease"),
        "wb115_gradient_norm_z": opt115.get("gradient_norm_z"),
        "wb115_step_norm_z": opt115.get("step_norm_z"),
    }
    missing = [key for key, value in required.items() if value in (None, {}, [])]
    if missing:
        return {
            "optimizer_contract_not_recoverable": True,
            "missing": missing,
            "recovered": False,
        }
    if float(required["pinv_relative"]) != PINV_RELATIVE:
        return {
            "optimizer_contract_not_recoverable": True,
            "missing": ["pinv_relative_mismatch"],
            "recovered": False,
        }
    if bool(seed114.get("do_not_select_best_seed")) is not True:
        return {
            "optimizer_contract_not_recoverable": True,
            "missing": ["do_not_select_best_seed"],
            "recovered": False,
        }
    return {
        "optimizer_contract_not_recoverable": False,
        "recovered": True,
        "pinv_relative": float(required["pinv_relative"]),
        "max_iterations": int(required["wb115_max_iterations"]),
        "line_search": list(required["wb115_line_search"]),
        "relative_chi2_decrease": float(required["wb115_relative_chi2_decrease"]),
        "gradient_norm_z": float(required["wb115_gradient_norm_z"]),
        "step_norm_z": float(required["wb115_step_norm_z"]),
        "chi2_rel_tolerance": float(required["chi2_rel_tolerance"]),
        "prediction_abs_tolerance_mm": float(required["prediction_abs_tolerance_mm"]),
        "supported_abs_tolerance": {
            "loc0_mm": float(required["supported_abs_tolerance"]["loc0_mm"]),
            "theta": float(required["supported_abs_tolerance"]["theta"]),
        },
        "schur_abs_tol": float(required["wb114_joint_profile_schur_abs_tol"]),
        "schur_rel_tol": float(required["wb114_joint_profile_schur_rel_tol"]),
        "wb114_max_iterations": int(required["wb114_max_iterations"]),
        "wb114_gn_step_damping": list(required["wb114_gn_step_damping"]),
        "wb115_nuisance_may_differ": bool(seed115.get("nuisance_may_differ", True)),
        "do_not_select_best_seed": True,
        "damping_is_not_prior": True,
        "damping_is_not_statistical_information": True,
        "source": {
            "workbook_114_config_sha256": config["inheritance"]["workbook_114"][
                "config_sha256"
            ],
            "workbook_115_config_sha256": config["inheritance"]["workbook_115"][
                "config_sha256"
            ],
        },
    }


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb128(config)
    spec = config["inheritance"]["workbook_128"]
    decision_path = resolve_under_root(project_root(), spec["decision_path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_128 config",
    )
    _expect_sha(decision_path, spec["decision_sha256"], "workbook_128 decision")
    _expect_sha(
        resolve_under_root(project_root(), spec["dump_path"]),
        spec["dump_sha256"],
        "workbook_128 dump",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise B14MRestartInvarianceError("workbook_128 decision must stay frozen")
    if spec["frozen_decision"] != WB128_DECISION:
        raise B14MRestartInvarianceError("WB128 decision token mismatch")
    if not decision.get("jacobian_contract_established"):
        raise B14MRestartInvarianceError("WB128 Jacobian contract is not established")
    if not decision.get("b14m_reopen_authorized"):
        raise B14MRestartInvarianceError("WB128 did not authorize B14M reopen")
    if decision.get("full_sample_authorized"):
        refuse_full_sample()
    repair_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        )
    )
    if repair_sha != FROZEN_FIELD_GRADIENT_SHA:
        raise B14MRestartInvarianceError("inherited_contract_mismatch: FieldGradient SHA")
    inherited["workbook_128"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "dump_sha256": spec["dump_sha256"],
        "official_run_id": spec["official_run_id"],
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": True,
        "jacobian_contract_established": True,
        "b14m_reopen_authorized": True,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
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
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "field_gradient_extension_sha256": FROZEN_FIELD_GRADIENT_SHA,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    smoke_root = resolve_under_root(project_root(), str(config["profile_smoke_root"]))
    name = str(config.get("profile_smoke_filename"))
    if smoke_root.is_dir():
        for path in sorted(smoke_root.rglob(name)):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    return {
        "rows": rows,
        "n_rows": len(rows),
        "smoke_present": bool(rows),
        "paths": paths,
    }


def _optimize_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = []
    for row in rows:
        if row.get("transport_task") != "B14MR":
            continue
        if bool(row.get("evaluate_only")):
            continue
        if row.get("row_role") == "evaluate_only_nominal_seed":
            continue
        selected.append(row)
    return selected


def _identity_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (*_event_pair(row), int(row.get("target_station", -1)))


def _chi2(row: Mapping[str, Any]) -> float | None:
    for key in ("chi2_prof", "profile_chi2", "chi2"):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def _native(row: Mapping[str, Any]) -> dict[str, float | None]:
    bound = row.get("profiled_native_bound") or {}
    raw = row.get("profiled_native_state")
    loc0 = _finite(bound.get("loc0"))
    loc1 = _finite(bound.get("loc1"))
    phi = _finite(bound.get("phi"))
    theta = _finite(bound.get("theta"))
    qop = _finite(bound.get("q_over_p_per_mev"))
    if isinstance(raw, list) and len(raw) >= 5:
        loc0 = loc0 if loc0 is not None else _finite(raw[0])
        loc1 = loc1 if loc1 is not None else _finite(raw[1])
        phi = phi if phi is not None else _finite(raw[2])
        theta = theta if theta is not None else _finite(raw[3])
        qop = qop if qop is not None else _finite(raw[4])
    return {
        "loc0": loc0,
        "loc1": loc1,
        "phi": phi,
        "theta": theta,
        "q_over_p_per_mev": qop,
    }


def _pred_target(row: Mapping[str, Any]) -> float | None:
    return _finite(row.get("predicted_target_loc0"))


def _pred_meas(row: Mapping[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in row.get("surviving_predicted_measurement_loc0") or []:
        idx = item.get("measurement_index")
        value = _finite(item.get("predicted_loc0"))
        if idx is None or value is None:
            continue
        out[int(idx)] = value
    return out


def _branch(row: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    hops = row.get("transport_branch_identity") or []
    keys = []
    for hop in hops:
        keys.append(
            (
                hop.get("measurement_index"),
                hop.get("identifier"),
                hop.get("station"),
                hop.get("layer"),
                hop.get("side"),
                hop.get("projection_kind"),
                hop.get("continuation_state_construction"),
                hop.get("ok"),
                hop.get("measurement_z_mm"),
            )
        )
    return keys


def _rel_chi2(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _valid_opt(row: Mapping[str, Any]) -> bool:
    chi2 = _chi2(row)
    term = str(row.get("termination_reason") or "")
    return bool(
        chi2 is not None
        and math.isfinite(chi2)
        and bool(row.get("propagation_success", row.get("all_surfaces_reached")))
        and term in VALID_TERMINATIONS
        and int(row.get("target_station_measurements_used") or 0) == 0
        and bool(row.get("target_exclusion_proven", True))
    )


def audit_identity(
    rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    by_restart = {row.get("profile_init_variant"): row for row in rows}
    missing = [name for name in RESTARTS if name not in by_restart]
    chi2_rel = float(contract["chi2_rel_tolerance"])
    pred_abs = float(contract["prediction_abs_tolerance_mm"])
    loc0_abs = float(contract["supported_abs_tolerance"]["loc0_mm"])
    theta_abs = float(contract["supported_abs_tolerance"]["theta"])
    if missing:
        return {
            "present": False,
            "missing_restarts": missing,
            "optimization_valid": False,
            "objective_invariance": False,
            "prediction_invariance": False,
            "parameter_invariance": False,
            "transport_branch_invariance": False,
            "nuisance_nonunique": False,
        }
    valids = [_valid_opt(by_restart[name]) for name in RESTARTS]
    chi2s = [_chi2(by_restart[name]) for name in RESTARTS]
    preds = [_pred_target(by_restart[name]) for name in RESTARTS]
    meas = [_pred_meas(by_restart[name]) for name in RESTARTS]
    natives = [_native(by_restart[name]) for name in RESTARTS]
    branches = [_branch(by_restart[name]) for name in RESTARTS]
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
    return {
        "present": True,
        "missing_restarts": [],
        "optimization_valid": all(valids),
        "valid_restarts": {
            name: valids[i] for i, name in enumerate(RESTARTS)
        },
        "chi2_prof": {name: chi2s[i] for i, name in enumerate(RESTARTS)},
        "predicted_target_loc0": {name: preds[i] for i, name in enumerate(RESTARTS)},
        "max_pairwise_delta_chi2_prof": max(finite_chi2) if finite_chi2 else None,
        "max_pairwise_delta_target_loc0": max(finite_pred) if finite_pred else None,
        "max_pairwise_delta_surviving_loc0": max(finite_meas) if finite_meas else None,
        "objective_invariance": objective_ok and all(valids),
        "prediction_invariance": pred_ok and all(valids),
        "parameter_invariance": alpha_ok and nu_unique and all(valids),
        "parameter_invariance_alpha": alpha_ok,
        "nuisance_nonunique": (not nu_unique) and all(valids),
        "transport_branch_invariance": branch_ok,
        "natives": natives,
        "terminations": {
            name: by_restart[name].get("termination_reason") for name in RESTARTS
        },
        "hessian_ranks": {
            name: by_restart[name].get("hessian_rank") for name in RESTARTS
        },
        "jacobian_implementation": {
            name: by_restart[name].get("jacobian_implementation") for name in RESTARTS
        },
    }


def audit_schur(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    residual = row.get("residual_vector")
    jacobian = row.get("measurement_jacobian_dh_dtheta")
    weights = row.get("measurement_weights")
    if not residual or not jacobian or not weights:
        return {"present": False, "passed": False, "reason": "missing_linearization"}
    jac = np.asarray(jacobian, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64).reshape(-1)
    wdiag = np.asarray(weights, dtype=np.float64).reshape(-1)
    if jac.shape[0] != res.size or wdiag.size != res.size or jac.shape[1] != 5:
        return {"present": True, "passed": False, "reason": "shape_mismatch"}
    if not (np.isfinite(jac).all() and np.isfinite(res).all() and np.isfinite(wdiag).all()):
        return {"present": True, "passed": False, "reason": "non_finite"}
    weight = np.diag(wdiag)
    ja = jac[:, list(ALPHA_INDICES)]
    jn = jac[:, list(NU_INDICES)]
    relative = float(contract["pinv_relative"])
    abs_tol = float(contract["schur_abs_tol"])
    rel_tol = float(contract["schur_rel_tol"])
    step_floor = float(contract["step_norm_z"])
    joint = joint_nls_step(jac, res, weight, relative=relative)
    profiled = profile_linear_solution(ja, jn, res, weight, relative=relative)
    explicit = explicit_nuisance_minimum(
        ja, jn, res, weight, profiled["delta_alpha"], relative=relative
    )
    joint_delta = np.asarray(joint["delta"], dtype=np.float64)
    schur_delta = np.concatenate(
        [
            np.asarray(profiled["delta_alpha"], dtype=np.float64).reshape(-1),
            np.asarray(profiled["delta_nu"], dtype=np.float64).reshape(-1),
        ]
    )
    # Schur solution is ordered (alpha, nu) = (loc0, theta, loc1, phi, q/p).
    # Joint NLS is native (loc0, loc1, phi, theta, q/p).  Reorder Schur.
    schur_native = np.array(
        [
            schur_delta[0],
            schur_delta[2],
            schur_delta[3],
            schur_delta[1],
            schur_delta[4],
        ]
    )
    both_tiny = (
        float(np.linalg.norm(joint_delta)) <= step_floor
        and float(np.linalg.norm(schur_native)) <= step_floor
    )
    joint_alpha = joint_delta[list(ALPHA_INDICES)]
    schur_alpha = np.asarray(profiled["delta_alpha"], dtype=np.float64).reshape(-1)
    loc0_abs = float(contract["supported_abs_tolerance"]["loc0_mm"])
    theta_abs = float(contract["supported_abs_tolerance"]["theta"])
    chi2_rel = float(contract["chi2_rel_tolerance"])
    alpha_ok = both_tiny or bool(
        np.allclose(joint_alpha, schur_alpha, atol=abs_tol, rtol=rel_tol)
        or (
            abs(float(joint_alpha[0] - schur_alpha[0])) <= loc0_abs
            and abs(float(joint_alpha[1] - schur_alpha[1])) <= theta_abs
        )
    )
    pred_joint = float(joint["gradient"] @ joint_delta)
    pred_schur = float(
        np.asarray(joint["gradient"], dtype=np.float64) @ schur_native
    )
    pred_ok = both_tiny or bool(
        abs(pred_joint - pred_schur)
        <= abs_tol + rel_tol * max(abs(pred_joint), abs(pred_schur), 1.0)
        or _rel_chi2(pred_joint, pred_schur) <= chi2_rel
    )
    step_ok = both_tiny or alpha_ok
    projector = np.asarray(profiled["projector"], dtype=np.float64)
    g_prof = ja.T @ projector @ res
    explicit_nu_ok = bool(
        np.allclose(
            np.asarray(profiled["delta_nu"], dtype=np.float64),
            np.asarray(explicit["delta_nu"], dtype=np.float64),
            atol=abs_tol,
            rtol=rel_tol,
        )
        or (
            float(np.linalg.norm(profiled["delta_nu"])) <= step_floor
            and float(np.linalg.norm(explicit["delta_nu"])) <= step_floor
        )
    )
    ridge_free = joint.get("ridge_added") is not True and profiled.get("ridge_added") is not True
    return {
        "present": True,
        "passed": bool(alpha_ok and ridge_free),
        "step_direction_agreement": step_ok,
        "predicted_decrease_agreement": pred_ok,
        "profiled_gradient_agreement": explicit_nu_ok,
        "profiled_alpha_agreement": alpha_ok,
        "rank_handling": {
            "joint_rank": int(joint["rank"]),
            "profile_rank": int(profiled.get("profile_rank") or -1),
            "nuisance_rank": int(profiled.get("nuisance_rank") or -1),
            "ridge_added": False,
            "pinv_relative": relative,
        },
        "joint_step_norm": float(np.linalg.norm(joint_delta)),
        "schur_step_norm": float(np.linalg.norm(schur_native)),
        "predicted_decrease_joint": pred_joint,
        "predicted_decrease_schur": pred_schur,
    }


def held_out_diagnostic(row: Mapping[str, Any]) -> dict[str, Any]:
    pred = _pred_target(row)
    items = []
    sigma = math.sqrt((0.08 * 0.08) / 12.0)
    for meas in row.get("held_out_target_measurements") or []:
        loc0 = _finite(meas.get("loc0"))
        residual = None if loc0 is None or pred is None else loc0 - pred
        items.append(
            {
                "station": meas.get("station"),
                "layer": meas.get("layer"),
                "id": meas.get("id"),
                "measured_loc0": loc0,
                "predicted_target_loc0": pred,
                "held_out_residual": residual,
                "normalized_held_out_residual": None
                if residual is None
                else residual / sigma,
                "not_used_in_fit": True,
                "not_used_for_restart_selection": True,
                "not_used_for_gate": True,
            }
        )
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": row.get("target_station"),
        "restart": row.get("profile_init_variant"),
        "items": items,
        "stage": "B",
    }


def inventory_and_audit(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    dumps = load_merged_rows(config)
    sample = load_contracted_sample(config)
    frozen_ok = (
        sample["n_raw"] == FROZEN_N_RAW
        and sample["n_ineligible"] == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    optimize = _optimize_rows(dumps["rows"])
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    events_seen = set()
    forbidden_j = False
    for row in optimize:
        pair = _event_pair(row)
        events_seen.add(pair)
        grouped[_identity_key(row)].append(row)
        impl = str(row.get("jacobian_implementation") or "")
        if impl in FORBIDDEN_JACOBIANS:
            forbidden_j = True
        if pair[1] == 44:
            refuse_replace_44()
    identities = []
    for event in REQUIRED_EVENTS:
        for target in TARGETS:
            key = (*event, target)
            identities.append(
                {
                    "event": _event_label(event),
                    "run_id": event[0],
                    "event_id": event[1],
                    "target_station": target,
                    **audit_identity(grouped.get(key, []), contract),
                }
            )
    schur = []
    for row in optimize:
        pair = _event_pair(row)
        if pair not in SCHUR_EVENTS:
            continue
        if row.get("profile_init_variant") != "nominal":
            continue
        schur.append(
            {
                "event": _event_label(pair),
                "target_station": row.get("target_station"),
                **audit_schur(row, contract),
            }
        )
    held = []
    for item in identities:
        if not (
            item.get("objective_invariance") and item.get("prediction_invariance")
        ):
            continue
        key = (item["run_id"], item["event_id"], item["target_station"])
        for row in grouped.get(key, []):
            if row.get("profile_init_variant") == "nominal":
                held.append(held_out_diagnostic(row))
    exclusion = (
        audit_exclusion(dumps["rows"])
        if dumps["rows"]
        else {"target_exclusion_holds": False, "n_compared": 0, "n_leaked": 0}
    )
    if (100043, 37) not in events_seen and dumps["smoke_present"]:
        refuse_drop_37()
    if (100048, 86) not in events_seen and dumps["smoke_present"]:
        refuse_drop_86()
    return {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "n_rows": dumps["n_rows"],
        "n_optimize_rows": len(optimize),
        "events_seen": sorted(events_seen),
        "identities": identities,
        "schur": schur,
        "held_out": held,
        "exclusion": exclusion,
        "forbidden_jacobian_used": forbidden_j,
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "full_sample_submitted": False,
    }


def decide_case(inventory: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("optimizer_contract_not_recoverable"):
        return {
            "decision": CASE_OPT,
            "primary_case": CASE_OPT,
            "verdict": "FAIL",
            "restart_invariance_established": False,
            "restart_invariance_authorized": False,
            "b14m_smoke_passed": False,
            "full_sample_authorized": False,
            "jacobian_contract_established": True,
        }
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_added"):
        refuse_ridge()
    if inventory.get("full_sample_submitted"):
        refuse_full_sample()
    if inventory.get("forbidden_jacobian_used"):
        raise B14MRestartInvarianceError("forbidden Jacobian implementation used")
    identities = list(inventory.get("identities") or [])
    schur = list(inventory.get("schur") or [])
    smoke = bool(inventory.get("smoke_present"))
    if not smoke or len(identities) != 12:
        return {
            "decision": CASE_G,
            "primary_case": CASE_G,
            "verdict": "FAIL",
            "restart_invariance_established": False,
            "restart_invariance_authorized": False,
            "b14m_smoke_passed": False,
            "full_sample_authorized": False,
            "jacobian_contract_established": True,
            "reason": "smoke_incomplete",
        }
    branch_fail = any(not item.get("transport_branch_invariance") for item in identities)
    schur_fail = any(item.get("present") and not item.get("passed") for item in schur)
    present_schur_events = {
        (int(item["event"].split("/")[0]), int(item["event"].split("/")[1]))
        for item in schur
        if item.get("present")
    }
    schur_missing = any(event not in present_schur_events for event in SCHUR_EVENTS)
    opt_fail = any(not item.get("optimization_valid") for item in identities)
    obj_fail = any(not item.get("objective_invariance") for item in identities)
    pred_fail = any(not item.get("prediction_invariance") for item in identities)
    nu_nonunique = any(item.get("nuisance_nonunique") for item in identities)
    param_ok = all(item.get("parameter_invariance") for item in identities)
    leakage = not bool((inventory.get("exclusion") or {}).get("target_exclusion_holds"))
    if schur_fail or schur_missing:
        decision = CASE_F
    elif branch_fail:
        decision = CASE_E
    elif pred_fail and not obj_fail:
        decision = CASE_D
    elif obj_fail:
        decision = CASE_B
    elif opt_fail or leakage:
        decision = CASE_G
    elif nu_nonunique and not pred_fail and not obj_fail:
        decision = CASE_C
    elif param_ok and not pred_fail and not obj_fail:
        decision = CASE_A
    else:
        decision = CASE_C if (not pred_fail and not obj_fail) else CASE_G
    passed = decision in {CASE_A, CASE_C}
    return {
        "decision": decision,
        "primary_case": decision,
        "verdict": "PASS" if passed else "FAIL",
        "restart_invariance_established": passed,
        "restart_invariance_authorized": passed,
        "b14m_smoke_passed": passed,
        "full_sample_authorized": False,
        "jacobian_contract_established": True,
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": True,
        "b14m_reopen_authorized": True,
        "nuisance_nonunique_observed": nu_nonunique,
        "target_exclusion_holds": not leakage,
        "next_step": "wb130_full_sample_profile_preflight" if passed else "remain_on_b14m_r",
    }


def decide(
    inventory: Mapping[str, Any],
    inherited: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    mechanism = decide_case(inventory, contract)
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb128_decision_sha256": inherited["workbook_128"]["decision_sha256"],
        "inherited_wb127_decision_sha256": inherited["workbook_127"]["decision_sha256"],
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_submit_1989": True,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "denominator": inventory.get("denominator"),
        "identity_summary": [
            {
                "event": item.get("event"),
                "target_station": item.get("target_station"),
                "optimization_valid": item.get("optimization_valid"),
                "objective_invariance": item.get("objective_invariance"),
                "prediction_invariance": item.get("prediction_invariance"),
                "parameter_invariance": item.get("parameter_invariance"),
                "nuisance_nonunique": item.get("nuisance_nonunique"),
                "transport_branch_invariance": item.get("transport_branch_invariance"),
                "max_pairwise_delta_chi2_prof": item.get("max_pairwise_delta_chi2_prof"),
                "max_pairwise_delta_target_loc0": item.get(
                    "max_pairwise_delta_target_loc0"
                ),
            }
            for item in inventory.get("identities") or []
        ],
    }


def likelihood_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "chi2_prof": "min_nu chi2(alpha, nu)",
        "R_i": "(0.08 mm)^2 / 12",
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "prior": None,
        "ridge": None,
        "q_over_p_is_explicit_nuisance": True,
        "validated_jacobian": "wb123_repaired_production_tangent",
        "validated_jacobian_is_not_a_new_statistical_model": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "do_not_submit_1989": True,
        "do_not_enter_b15": True,
        "do_not_enter_wb130": True,
    }
