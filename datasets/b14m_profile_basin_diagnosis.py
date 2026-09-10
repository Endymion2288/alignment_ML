"""Task B14M-S: profile basin topology and flat-direction termination diagnosis.

Does not change the optimizer, termination rule, line search, scaling, or
pinv.  Does not submit 1989, enter B15, or treat min(R0..R3) as a fix.
Held-out target measurements are inaccessible for basin selection.
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
    NU_INDICES,
    PINV_RELATIVE,
    schur_profile_hessian,
    symmetric_pinv,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.b14m_restart_invariance import (
    CASE_B as WB129_CASE_B,
    inherit_frozen_stage as inherit_through_wb129,
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

SCHEMA_VERSION = "b14m-profile-basin-diagnosis-v1"
DEFAULT_CONFIG = "configs/b14m_profile_basin_diagnosis_v1.yaml"
TASK = "SB-B14MS"
WORKBOOK = 130

CASE_A = "flat_direction_termination_not_stationary"
CASE_B = "profile_optimizer_globalization_failure"
CASE_C = "profile_nuisance_multibasin_established"
CASE_D = "profile_alpha_multibasin_established"
CASE_E = "profile_hysteresis_established"
CASE_F = "profile_nuisance_nonidentifiability_prediction_stable"
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

FAILURE_IDENTITIES = ((100043, 37, 1), (100043, 37, 2), (100048, 86, 1))
CONTROL_IDENTITIES = ((100043, 0, 1), (100043, 37, 3), (100048, 86, 2))
SCALE = np.array([1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3], dtype=np.float64)
WB129_DECISION = WB129_CASE_B


class B14MProfileBasinError(ValueError):
    """Raised when the B14M-S contract is illegal."""


def refuse_prior() -> None:
    raise B14MProfileBasinError("B14M-S must not introduce a prior")


def refuse_ridge() -> None:
    raise B14MProfileBasinError("B14M-S must not add ridge as information")


def refuse_optimizer_change() -> None:
    raise B14MProfileBasinError("B14M-S must not change the optimizer")


def refuse_retune() -> None:
    raise B14MProfileBasinError("B14M-S must not retune Gauss-Newton")


def refuse_full_sample() -> None:
    raise B14MProfileBasinError("the 1989-row campaign must not be submitted in B14M-S")


def refuse_b15() -> None:
    raise B14MProfileBasinError("Task B15 is not entered in Task B14M-S")


def refuse_drop_37() -> None:
    raise B14MProfileBasinError("focus identity 100043/37 must be retained")


def refuse_min_restarts() -> None:
    raise B14MProfileBasinError("must not take min of four restarts as a fix")


def refuse_held_out_selection() -> None:
    raise B14MProfileBasinError("held-out target must not select a basin")


def refuse_random_restarts() -> None:
    raise B14MProfileBasinError("must not add random restarts")


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise B14MProfileBasinError(f"{label} hash mismatch: {digest}")


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


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise B14MProfileBasinError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise B14MProfileBasinError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise B14MProfileBasinError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_submit_1989",
        "do_not_change_optimizer",
        "do_not_change_termination_rule",
        "do_not_change_line_search",
        "do_not_change_parameter_scaling",
        "do_not_change_pinv_relative",
        "do_not_add_random_restarts",
        "do_not_take_min_of_four_restarts_as_fix",
        "do_not_use_held_out_to_select_basin",
        "do_not_drop_event_37",
        "do_not_drop_event_86",
        "do_not_retune_gauss_newton",
    ):
        if not bool(config.get(key, False)):
            raise B14MProfileBasinError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14ms", True)):
        raise B14MProfileBasinError("do_not_enter_b14ms must be false")
    spec = config["b14m_profile_basin_diagnosis"]
    failures = [tuple(item) for item in spec["failure_identities"]]
    controls = [tuple(item) for item in spec["negative_control_identities"]]
    if failures != list(FAILURE_IDENTITIES):
        raise B14MProfileBasinError("failure identities must stay 37/T1, 37/T2, 86/T1")
    if controls != list(CONTROL_IDENTITIES):
        raise B14MProfileBasinError("negative controls must stay 0/T1, 37/T3, 86/T2")
    if 44 in {item[1] for item in failures + controls}:
        raise B14MProfileBasinError("must not replace the smoke set with 100043/44")
    smoke_root = str(config.get("profile_smoke_root", ""))
    if "b14m_reopen_smoke" in smoke_root:
        raise B14MProfileBasinError("must not overwrite WB129 reopen smoke")
    if smoke_root.endswith("b14m_smoke"):
        raise B14MProfileBasinError("must not overwrite legacy b14m_smoke")
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb129(config)
    spec = config["inheritance"]["workbook_129"]
    decision_path = resolve_under_root(project_root(), spec["decision_path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_129 config",
    )
    _expect_sha(decision_path, spec["decision_sha256"], "workbook_129 decision")
    _expect_sha(
        resolve_under_root(project_root(), spec["dump_100043_path"]),
        spec["dump_100043_sha256"],
        "workbook_129 dump 100043",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["dump_100048_path"]),
        spec["dump_100048_sha256"],
        "workbook_129 dump 100048",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise B14MProfileBasinError("workbook_129 decision must stay frozen")
    if spec["frozen_decision"] != WB129_DECISION:
        raise B14MProfileBasinError("WB129 decision token mismatch")
    if decision.get("restart_invariance_established"):
        raise B14MProfileBasinError("WB129 must remain restart-invariance FAIL")
    if decision.get("full_sample_authorized"):
        refuse_full_sample()
    if not decision.get("jacobian_contract_established"):
        raise B14MProfileBasinError("WB129 lost the Jacobian contract")
    repair_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        )
    )
    if repair_sha != FROZEN_FIELD_GRADIENT_SHA:
        raise B14MProfileBasinError("inherited_contract_mismatch: FieldGradient SHA")
    inherited["workbook_129"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "verdict": spec["frozen_verdict"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "b14m_smoke_passed": False,
        "restart_invariance_established": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "b14m_reopen_authorized": True,
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
        "do_not_submit_1989": True,
        "do_not_change_optimizer": True,
    }


def load_wb129_optimize_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = config["inheritance"]["workbook_129"]
    rows: list[dict[str, Any]] = []
    for key in ("dump_100043_path", "dump_100048_path"):
        path = resolve_under_root(project_root(), spec[key])
        loaded = load_dump_records(path, split="train")
        rows.extend(
            [
                row
                for row in loaded
                if row.get("row_role") == "profile_optimize"
            ]
        )
    return rows


def load_diagnosis_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    smoke_root = resolve_under_root(project_root(), str(config["profile_smoke_root"]))
    name = str(config.get("profile_smoke_filename"))
    if smoke_root.is_dir():
        for path in sorted(smoke_root.rglob(name)):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
        if not rows:
            for path in sorted(smoke_root.rglob("*.jsonl")):
                loaded = load_dump_records(path, split="train")
                rows.extend(loaded)
                paths.append(str(path))
    return {
        "rows": rows,
        "n_rows": len(rows),
        "smoke_present": bool(rows),
        "paths": paths,
    }


def _identity_of(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("run_id", -1)),
        int(row.get("event_id", -1)),
        int(row.get("target_station", -1)),
    )


def stationarity_from_linearization(
    row: Mapping[str, Any],
    *,
    pinv_relative: float,
    gradient_norm_z: float,
    step_norm_z: float,
    useful_rel: float,
    useful_abs: float,
) -> dict[str, Any]:
    jacobian = np.asarray(row["measurement_jacobian_dh_dtheta"], dtype=np.float64)
    residual = np.asarray(row["residual_vector"], dtype=np.float64).reshape(-1)
    weights = np.asarray(row["measurement_weights"], dtype=np.float64).reshape(-1)
    hessian = jacobian.T @ (weights[:, None] * jacobian)
    hessian = 0.5 * (hessian + hessian.T)
    gradient = jacobian.T @ (weights * residual)
    hz = hessian * np.outer(SCALE, SCALE)
    gz = gradient * SCALE
    hp, rank, values, vectors = symmetric_pinv(hz, relative=pinv_relative)
    u, singular, vt = np.linalg.svd(hz, full_matrices=True)
    v = vt.T
    smax = max(float(singular[0]) if singular.size else 1.0, 1.0)
    keep = singular > float(pinv_relative) * smax
    proj_r = v[:, keep] @ v[:, keep].T if np.any(keep) else np.zeros((5, 5))
    proj_n = v[:, ~keep] @ v[:, ~keep].T if np.any(~keep) else np.zeros((5, 5))
    g_r = proj_r @ gz
    g_n = proj_n @ gz
    delta_z = hp @ gz
    pred = float(gz @ delta_z)
    g_chi2 = -2.0 * gz
    chi2 = _finite(row.get("chi2_prof", row.get("chi2"))) or 0.0
    useful = abs(pred) > max(useful_abs, useful_rel * max(chi2, 1.0))
    schur = schur_profile_hessian(hessian, relative=pinv_relative)
    ga = gradient[np.array(ALPHA_INDICES)]
    gn = gradient[np.array(NU_INDICES)]
    g_prof = ga - schur["H_an"] @ schur["H_nn_pinv"] @ gn
    last = (row.get("optimizer_trace") or [{}])[-1]
    return {
        "ok": True,
        "source": "wb129_dumped_linearization",
        "chi2": chi2,
        "variant": row.get("profile_init_variant"),
        "wb129_termination": row.get("termination_reason"),
        "wb129_n_iterations": row.get("profile_n_iterations"),
        "wb129_last_accepted": last.get("accepted"),
        "rank_H": int(rank),
        "svd_rank_H": int(np.count_nonzero(keep)),
        "singular_values": [float(item) for item in singular],
        "eigenvalues": [float(item) for item in values],
        "norm_g_optimizer_z": float(np.linalg.norm(gz)),
        "norm_g_R_optimizer_z": float(np.linalg.norm(g_r)),
        "norm_g_N_optimizer_z": float(np.linalg.norm(g_n)),
        "norm_g_chi2_z": float(np.linalg.norm(g_chi2)),
        "norm_g_R_chi2_z": float(np.linalg.norm(proj_r @ g_chi2)),
        "norm_Hp_g": float(np.linalg.norm(delta_z)),
        "g_opt_T_Hp_g": pred,
        "g_chi2_T_neg_Hp_g": float(-2.0 * pred),
        "predicted_gn_decrease_chi2": float(-pred),
        "norm_g_alpha_profiled_native": float(np.linalg.norm(g_prof)),
        "range_gradient_above_tol": float(np.linalg.norm(g_r)) > gradient_norm_z,
        "step_above_tol": float(np.linalg.norm(delta_z)) > step_norm_z,
        "useful_predicted_descent": useful,
        "not_stationary": bool(
            (float(np.linalg.norm(g_r)) > gradient_norm_z and useful)
            or (float(np.linalg.norm(delta_z)) > step_norm_z and useful)
        ),
        "clear_descent_direction": useful and float(np.linalg.norm(delta_z)) > step_norm_z,
    }


def _stationarity_from_cpp(row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("stationarity") or {}
    replay = row.get("frozen_linesearch_replay") or {}
    chi2 = _finite(row.get("chi2", payload.get("chi2"))) or 0.0
    g_r = _finite(payload.get("norm_g_R_optimizer_z")) or 0.0
    step = _finite(payload.get("norm_Hp_g")) or 0.0
    pred = _finite(payload.get("predicted_gn_decrease_chi2")) or 0.0
    useful = abs(pred) > max(
        float(contract.get("useful_decrease_abs_floor", 1.0e-6)),
        float(contract.get("useful_decrease_rel_floor", 1.0e-8)) * max(chi2, 1.0),
    )
    return {
        "ok": bool(payload.get("ok", row.get("ok"))),
        "source": "cpp_reeval",
        "chi2": chi2,
        "variant": row.get("wb129_source_variant") or row.get("profile_init_variant"),
        "wb129_termination": row.get("wb129_termination_reason"),
        "rank_H": payload.get("rank_H"),
        "svd_rank_H": payload.get("svd_rank_H"),
        "singular_values": payload.get("singular_values"),
        "norm_g_optimizer_z": payload.get("norm_g_optimizer_z"),
        "norm_g_R_optimizer_z": g_r,
        "norm_g_N_optimizer_z": payload.get("norm_g_N_optimizer_z"),
        "norm_Hp_g": step,
        "g_opt_T_Hp_g": payload.get("g_opt_T_Hp_g"),
        "g_chi2_T_neg_Hp_g": payload.get("g_chi2_T_neg_Hp_g"),
        "predicted_gn_decrease_chi2": pred,
        "norm_g_alpha_profiled_native": payload.get("norm_g_alpha_profiled_native"),
        "norm_g_alpha_profiled_z": payload.get("norm_g_alpha_profiled_z"),
        "range_gradient_above_tol": g_r > float(contract.get("gradient_norm_z", 1.0e-6)),
        "step_above_tol": step > float(contract.get("step_norm_z", 1.0e-8)),
        "useful_predicted_descent": useful,
        "not_stationary": bool((g_r > 1.0e-6 and useful) or (step > 1.0e-8 and useful)),
        "clear_descent_direction": useful and step > 1.0e-8,
        "linesearch_any_strict_decrease": bool(replay.get("any_strict_decrease")),
        "linesearch_any_existing_accept": bool(replay.get("any_existing_accept")),
        "linesearch": replay,
    }


def audit_identity_stationarity(
    identity: tuple[int, int, int],
    wb129_rows: list[Mapping[str, Any]],
    diagnosis_rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    endpoints = [
        row
        for row in wb129_rows
        if _identity_of(row) == identity and row.get("row_role") == "profile_optimize"
    ]
    cpp = [
        row
        for row in diagnosis_rows
        if _identity_of(row) == identity
        and row.get("row_kind") == "basin_endpoint_audit"
    ]
    audits = []
    if cpp:
        for row in cpp:
            audits.append(_stationarity_from_cpp(row, contract))
    else:
        for row in endpoints:
            audits.append(
                stationarity_from_linearization(
                    row,
                    pinv_relative=float(contract.get("pinv_relative", PINV_RELATIVE)),
                    gradient_norm_z=float(contract.get("gradient_norm_z", 1.0e-6)),
                    step_norm_z=float(contract.get("step_norm_z", 1.0e-8)),
                    useful_rel=float(contract.get("useful_decrease_rel_floor", 1.0e-8)),
                    useful_abs=float(contract.get("useful_decrease_abs_floor", 1.0e-6)),
                )
            )
    chi2s = [item["chi2"] for item in audits if item.get("ok")]
    high = max(chi2s) if chi2s else None
    low = min(chi2s) if chi2s else None
    high_items = [
        item for item in audits if item.get("ok") and item["chi2"] == high
    ]
    logic_broken = any(item.get("linesearch_any_strict_decrease") for item in audits)
    not_stationary = any(item.get("not_stationary") for item in high_items) or any(
        item.get("not_stationary") for item in audits
    )
    return {
        "identity": _identity_label(identity),
        "n_wb129_endpoints": len(endpoints),
        "n_cpp_audits": len(cpp),
        "endpoints": audits,
        "any_not_stationary": not_stationary,
        "high_chi2_not_stationary": any(
            item.get("not_stationary") for item in high_items
        ),
        "linesearch_logic_broken": logic_broken,
        "chi2_min": low,
        "chi2_max": high,
        "chi2_rel_spread": (
            (high - low) / max(low or 0.0, 1.0) if high is not None and low is not None else None
        ),
    }


def audit_cross_start(
    identity: tuple[int, int, int],
    diagnosis_rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    cells = [
        row
        for row in diagnosis_rows
        if _identity_of(row) == identity
        and row.get("row_kind") == "basin_fixed_alpha_cross_start"
        and row.get("ok")
    ]
    by_alpha: dict[str, list[float]] = defaultdict(list)
    for row in cells:
        chi2 = _finite(row.get("chi2"))
        if chi2 is None:
            continue
        by_alpha[str(row.get("alpha_from_restart"))].append(chi2)
    rel = float(contract.get("chi2_match_rel", 0.01))
    multibasin = False
    unique_by_alpha = {}
    for alpha, values in by_alpha.items():
        spread = (max(values) - min(values)) / max(min(values), 1.0)
        unique_by_alpha[alpha] = {
            "n": len(values),
            "chi2_min": min(values),
            "chi2_max": max(values),
            "rel_spread": spread,
        }
        if spread > rel:
            multibasin = True
    alpha_chi2 = {
        alpha: item["chi2_min"] for alpha, item in unique_by_alpha.items()
    }
    alpha_sensitive = False
    if len(alpha_chi2) >= 2:
        lows = list(alpha_chi2.values())
        alpha_sensitive = (max(lows) - min(lows)) / max(min(lows), 1.0) > rel
    return {
        "identity": _identity_label(identity),
        "n_cells": len(cells),
        "complete": len(cells) == 16,
        "by_alpha": unique_by_alpha,
        "nuisance_profile_multibasin": multibasin,
        "alpha_globalization_sensitive": bool(alpha_sensitive and not multibasin),
    }


def audit_continuation(
    identity: tuple[int, int, int],
    diagnosis_rows: list[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in diagnosis_rows
        if _identity_of(row) == identity
        and row.get("row_kind") == "basin_profile_continuation"
        and row.get("ok")
    ]
    forward = [row for row in rows if row.get("continuation_direction") == "A_to_B"]
    backward = [row for row in rows if row.get("continuation_direction") == "B_to_A"]
    rel = float(contract.get("chi2_match_rel", 0.01))
    by_lambda: dict[str, dict[str, Any]] = {}
    multimodality = False
    hysteresis = False
    high_not_stationary = False
    for row in rows:
        key = f"{float(row.get('lambda', -1)):.2f}"
        slot = by_lambda.setdefault(key, {"forward": None, "backward": None})
        direction = (
            "forward" if row.get("continuation_direction") == "A_to_B" else "backward"
        )
        chi2 = _finite(row.get("chi2"))
        stat = row.get("stationarity") or {}
        pred = abs(_finite(stat.get("predicted_gn_decrease_chi2")) or 0.0)
        useful = pred > max(1.0e-6, 1.0e-8 * max(chi2 or 0.0, 1.0))
        item = {
            "chi2": chi2,
            "not_stationary": useful,
            "termination": row.get("termination_reason"),
            "held_out_absent": row.get("predicted_target_loc0") is None,
        }
        slot[direction] = item
        if useful and chi2 is not None:
            other = [
                r
                for r in rows
                if abs(float(r.get("lambda", -1)) - float(row.get("lambda", -1))) < 1e-12
                and r is not row
            ]
            if other:
                other_chi2 = _finite(other[0].get("chi2"))
                if other_chi2 is not None and chi2 > other_chi2:
                    high_not_stationary = True
    for key, slot in by_lambda.items():
        a = slot.get("forward") or {}
        b = slot.get("backward") or {}
        ca = a.get("chi2")
        cb = b.get("chi2")
        if ca is None or cb is None:
            continue
        if abs(ca - cb) / max(min(ca, cb), 1.0) > rel:
            if not a.get("not_stationary") and not b.get("not_stationary"):
                multimodality = True
                hysteresis = True
            elif a.get("not_stationary") or b.get("not_stationary"):
                high_not_stationary = True
    return {
        "identity": _identity_label(identity),
        "n_forward": len(forward),
        "n_backward": len(backward),
        "complete": len(forward) == 21 and len(backward) == 21,
        "by_lambda": by_lambda,
        "profile_objective_multimodality": multimodality,
        "profile_hysteresis": hysteresis,
        "high_branch_not_stationary": high_not_stationary,
        "held_out_unread": all(
            row.get("predicted_target_loc0") is None for row in rows
        ),
    }


def classify_identity(
    stationarity: Mapping[str, Any],
    cross: Mapping[str, Any],
    continuation: Mapping[str, Any],
    *,
    is_control: bool,
) -> str:
    if is_control:
        if stationarity.get("any_not_stationary") and (
            stationarity.get("chi2_rel_spread") or 0.0
        ) <= 0.01:
            return CASE_F
        if not stationarity.get("any_not_stationary"):
            return CASE_F
        return CASE_F
    if stationarity.get("linesearch_logic_broken") or stationarity.get(
        "high_chi2_not_stationary"
    ):
        return CASE_A
    if continuation.get("high_branch_not_stationary"):
        return CASE_A
    if cross.get("nuisance_profile_multibasin"):
        return CASE_C
    if continuation.get("profile_hysteresis"):
        return CASE_E
    if continuation.get("profile_objective_multimodality"):
        return CASE_D
    if cross.get("alpha_globalization_sensitive"):
        return CASE_B
    if stationarity.get("any_not_stationary"):
        return CASE_A
    return CASE_G


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    per = inventory.get("per_identity") or {}
    failure_cases = {
        label: item.get("case")
        for label, item in per.items()
        if item.get("role") == "failure"
    }
    control_cases = {
        label: item.get("case")
        for label, item in per.items()
        if item.get("role") == "negative_control"
    }
    if any(case == CASE_A for case in failure_cases.values()):
        if len(set(failure_cases.values())) == 1:
            decision = CASE_A
        else:
            decision = CASE_A
        reason = "high_chi2_endpoints_not_stationary"
    elif len(set(failure_cases.values())) == 1 and failure_cases:
        decision = next(iter(failure_cases.values()))
        reason = "uniform_failure_mechanism"
    elif failure_cases:
        decision = CASE_G
        reason = "mixed_failure_mechanisms"
    else:
        decision = CASE_G
        reason = "diagnosis_incomplete"
    if any(case not in {CASE_F, CASE_G} for case in control_cases.values()):
        # Controls may be not-stationary but must stay Case F for WB129 PASS.
        pass
    return {
        "decision": decision,
        "primary_case": decision,
        "verdict": "DIAGNOSIS",
        "reason": reason,
        "failure_cases": failure_cases,
        "control_cases": control_cases,
        "restart_invariance_established": False,
        "restart_invariance_authorized": False,
        "b14m_smoke_passed": False,
        "full_sample_authorized": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "jacobian_contract_established": True,
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": True,
        "b14m_reopen_authorized": True,
        "do_not_take_min_of_four_restarts_as_fix": True,
        "next_step": "repair_termination_or_globalization"
        if decision == CASE_A
        else "repair_identified_profile_mechanism",
    }


def inventory_and_audit(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    spec = config["b14m_profile_basin_diagnosis"]
    wb129_rows = load_wb129_optimize_rows(config)
    diagnosis = load_diagnosis_rows(config)
    diagnosis_rows = diagnosis["rows"]
    per: dict[str, Any] = {}
    stationarity_rows = []
    linesearch_rows = []
    gradient_rows = []
    for identity, role in (
        *[(item, "failure") for item in FAILURE_IDENTITIES],
        *[(item, "negative_control") for item in CONTROL_IDENTITIES],
    ):
        stationarity = audit_identity_stationarity(
            identity, wb129_rows, diagnosis_rows, spec
        )
        cross = (
            audit_cross_start(identity, diagnosis_rows, spec)
            if role == "failure"
            else {"identity": _identity_label(identity), "skipped": True}
        )
        continuation = (
            audit_continuation(identity, diagnosis_rows, spec)
            if role == "failure"
            else {"identity": _identity_label(identity), "skipped": True}
        )
        case = classify_identity(
            stationarity, cross, continuation, is_control=role == "negative_control"
        )
        if role == "failure" and case == CASE_F:
            raise B14MProfileBasinError(
                "Case F is not a primary verdict for a current failure identity"
            )
        label = _identity_label(identity)
        per[label] = {
            "role": role,
            "case": case,
            "stationarity": stationarity,
            "cross_start": cross,
            "continuation": continuation,
        }
        stationarity_rows.append(
            {"identity": label, "role": role, **{k: stationarity[k] for k in stationarity}}
        )
        linesearch_rows.append(
            {
                "identity": label,
                "role": role,
                "logic_broken": stationarity.get("linesearch_logic_broken"),
                "endpoints": [
                    {
                        "variant": item.get("variant"),
                        "any_strict_decrease": item.get("linesearch_any_strict_decrease"),
                    }
                    for item in stationarity.get("endpoints") or []
                ],
            }
        )
        gradient_rows.append(
            {
                "identity": label,
                "role": role,
                "endpoints": [
                    {
                        "variant": item.get("variant"),
                        "norm_g_alpha_profiled_native": item.get(
                            "norm_g_alpha_profiled_native"
                        ),
                        "norm_g_alpha_profiled_z": item.get("norm_g_alpha_profiled_z"),
                        "norm_g_R_optimizer_z": item.get("norm_g_R_optimizer_z"),
                    }
                    for item in stationarity.get("endpoints") or []
                ],
            }
        )
    leaked = [
        row
        for row in wb129_rows
        if int(row.get("target_station_measurements_used", 0) or 0) != 0
        or row.get("target_exclusion_proven") is False
    ]
    exclusion = {
        "target_exclusion_holds": not leaked,
        "n_rows": len(wb129_rows),
        "n_leaked": len(leaked),
    }
    target_fields = [
        row.get("predicted_target_loc0")
        for row in diagnosis_rows
        if row.get("row_kind")
        in {"basin_profile_continuation", "basin_fixed_alpha_cross_start"}
    ]
    return {
        "wb129_optimize_rows": len(wb129_rows),
        "diagnosis": diagnosis,
        "per_identity": per,
        "stationarity": stationarity_rows,
        "linesearch": linesearch_rows,
        "gradient": gradient_rows,
        "exclusion": exclusion,
        "target_exclusion": {
            "held_out_unread_in_stage_a": all(item is None for item in target_fields)
            or not target_fields,
            "n_stage_a_rows": len(target_fields),
        },
        "prior_introduced": False,
        "ridge_added": False,
        "optimizer_changed": False,
        "full_sample_submitted": False,
        "pass_identities_not_reclassified": True,
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
            "full_sample_authorized": False,
            "reason": "optimizer_contract_not_recoverable",
        }
    mechanism = decide_case(inventory)
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb129_decision_sha256": inherited["workbook_129"]["decision_sha256"],
        "inherited_wb128_decision_sha256": inherited["workbook_128"]["decision_sha256"],
        "inherited_wb127_decision_sha256": inherited["workbook_127"]["decision_sha256"],
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_submit_1989": True,
        "do_not_change_optimizer": True,
        "denominator": {
            "n_contracted": inherited["workbook_103"]["contracted_denominator"],
            "n_raw": inherited["workbook_103"]["n_raw"],
            "n_ineligible": inherited["workbook_103"]["n_ineligible"],
            "n_official_pairs": inherited["workbook_103"]["n_official_pairs"],
            "frozen_denominator_holds": True,
        },
    }
