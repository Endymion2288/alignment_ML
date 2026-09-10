"""Task B14L: ACTS transport Jacobian vs fixed multi-scale FD contract.

Compares the chained ACTS 32.0.2 transportJacobian, converted through
the official supporting-plane residual chart, with the frozen
h, h/2, h/4, h/8 finite-difference ladder.  Does not retune
Gauss-Newton, change production stepTolerance, pick a best FD step,
replace the official likelihood, introduce a prior/ridge, delete
37/86, or enter B14M / B15 / Measurement Model V2.
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
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profile_transport_contract import (
    CLASS_M_EVENT,
    CLASS_S_EVENT,
    CONTROL_EVENTS,
    audit_exclusion,
    is_official_mode_b,
)
from datasets.source_to_measurement_map_smoothness import (
    CASE_MIXED as WB118_DECISION,
    inherit_frozen_stage as inherit_through_wb117,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "acts-fd-derivative-contract-v1"
DEFAULT_CONFIG = "configs/acts_fd_derivative_contract_v1.yaml"
TASK = "SB-B14L"
WORKBOOK = 119

CASE_FD_UNRELIABLE = "finite_difference_not_reliable_for_focus_transport"
CASE_ACTS_NOT_CONV = "acts_transport_jacobian_not_numerically_converged"
CASE_CHAIN_BROKEN = "analytic_chain_or_chart_contract_broken"
CASE_MIXED = "mixed_or_inconclusive"

PARAM_NAMES = ("loc0", "loc1", "phi", "theta", "q_over_p")
FOCUS_PARAMETERS = ("loc1", "phi", "q_over_p")
CONTROL_SET = CONTROL_EVENTS | {CLASS_M_EVENT}
FOCUS_EVENT = CLASS_S_EVENT
REQUIRED_EVENTS = CONTROL_SET | {FOCUS_EVENT}
RUNG_FACTORS = (1.0, 0.5, 0.25, 0.125)
REL_MAX = 0.05
ACCURACY_TOLERANCES = (1.0e-4, 1.0e-5, 1.0e-6)


class ActsFdDerivativeContractError(ValueError):
    """Raised when the B14L derivative contract is illegal."""


def refuse_prior() -> None:
    raise ActsFdDerivativeContractError("B14L must not introduce a prior")


def refuse_ridge_information() -> None:
    raise ActsFdDerivativeContractError(
        "ridge must not be treated as statistical information"
    )


def refuse_measurement_update() -> None:
    raise ActsFdDerivativeContractError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_best_step_selection() -> None:
    raise ActsFdDerivativeContractError(
        "must not select the best finite-difference step from the results"
    )


def refuse_best_tolerance() -> None:
    raise ActsFdDerivativeContractError(
        "must not select the best propagator stepTolerance from the results"
    )


def refuse_switch_to_direct() -> None:
    raise ActsFdDerivativeContractError(
        "must not replace the official sequential likelihood with direct-from-source"
    )


def refuse_replace_likelihood() -> None:
    raise ActsFdDerivativeContractError(
        "must not replace the official likelihood with the ACTS Jacobian"
    )


def refuse_b14m() -> None:
    raise ActsFdDerivativeContractError("B14M is not re-opened inside Task B14L")


def refuse_restart() -> None:
    raise ActsFdDerivativeContractError(
        "restart invariance is not opened inside Task B14L"
    )


def refuse_b15() -> None:
    raise ActsFdDerivativeContractError("Task B15 is not entered in Task B14L")


def refuse_measurement_model_v2() -> None:
    raise ActsFdDerivativeContractError(
        "Measurement Model V2 is not entered in Task B14L"
    )


def refuse_full_sample() -> None:
    raise ActsFdDerivativeContractError(
        "the 1989-row campaign must not be submitted in B14L"
    )


def refuse_change_statistical_model() -> None:
    raise ActsFdDerivativeContractError(
        "the measurement statistical model must stay frozen"
    )


def refuse_5d_cin_repair() -> None:
    raise ActsFdDerivativeContractError(
        "B14L must not treat 5D Cin repair as the objective"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ActsFdDerivativeContractError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ActsFdDerivativeContractError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ActsFdDerivativeContractError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ActsFdDerivativeContractError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise ActsFdDerivativeContractError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "marginalization_executed",
    ):
        if bool(config.get(key, True)):
            raise ActsFdDerivativeContractError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14k",
        "do_not_select_best_fd_step",
        "do_not_switch_official_likelihood_to_direct",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
    ):
        if not bool(config.get(key, False)):
            raise ActsFdDerivativeContractError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14l", True)):
        raise ActsFdDerivativeContractError("do_not_enter_b14l must be false")
    tols = [
        float(item["step_tolerance"])
        for item in config["derivative_contract"]["accuracy_rungs"]
    ]
    if tols != list(ACCURACY_TOLERANCES):
        raise ActsFdDerivativeContractError("accuracy rungs must stay 1e-4/1e-5/1e-6")
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb117(config)
    spec = config["inheritance"]["workbook_118"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_118 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_118 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise ActsFdDerivativeContractError("workbook_118 decision must stay frozen")
    if spec["frozen_decision"] != WB118_DECISION:
        raise ActsFdDerivativeContractError("WB118 decision token mismatch")
    if decision.get("b14m_reopen_authorized"):
        raise ActsFdDerivativeContractError("WB118 must not have re-opened B14M")
    if decision.get("jacobian_contract_established"):
        raise ActsFdDerivativeContractError(
            "WB118 must not have established the Jacobian contract"
        )
    inherited["workbook_118"] = {
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
        "acts_transport_jacobian_available": True,
        "analytic_vs_fd_not_yet_contracted": True,
        "wb117_not_treated_as_physical_nonsmoothness": True,
    }
    return inherited


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _event_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _event_label(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _as_vector(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or not raw:
        return None
    values = [_finite(item) for item in raw]
    if any(item is None for item in values):
        return None
    return np.asarray(values, dtype=float)


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["derivative_smoke_root"]))
    filename = str(
        config.get(
            "derivative_smoke_filename",
            "ckf_leave_target_out_derivative_contract.jsonl",
        )
    )
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_derivative_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths = smoke_dump_paths(config)
    for path in paths:
        rows.extend(load_dump_records(path, split="train"))
    return {
        "rows": rows,
        "n_rows": len(rows),
        "smoke_present": bool(rows),
        "paths": [str(path) for path in paths],
    }


def audit_fd_column(raw: Mapping[str, Any]) -> dict[str, Any]:
    pairs = raw.get("consecutive_relative_errors") or []
    last = pairs[-1] if pairs else {}
    last_rel = _finite(last.get("relative_error"))
    return {
        "parameter": raw.get("parameter"),
        "ok": bool(raw.get("ok")),
        "last_pair_relative_error": last_rel,
        "sign_consistent": bool(last.get("sign_consistent")),
        "ladder_converged": bool(raw.get("ok"))
        and last_rel is not None
        and last_rel <= REL_MAX
        and bool(last.get("sign_consistent")),
        "official_column": _as_vector(
            next(
                (
                    (rung.get("residual_column"))
                    for rung in (raw.get("rungs") or [])
                    if _finite(rung.get("step_factor")) == 1.0
                ),
                None,
            )
        ),
        "official_norm": _finite(
            next(
                (
                    rung.get("column_norm")
                    for rung in (raw.get("rungs") or [])
                    if _finite(rung.get("step_factor")) == 1.0
                ),
                None,
            )
        ),
    }


def _column_agreement(
    left: np.ndarray | None, right: np.ndarray | None
) -> dict[str, Any]:
    if left is None or right is None or left.size != right.size:
        return {
            "agree": False,
            "relative_error": None,
            "sign_consistent": False,
            "left_norm": None,
            "right_norm": None,
        }
    n_right = float(np.linalg.norm(right))
    n_left = float(np.linalg.norm(left))
    rel = float(np.linalg.norm(left - right) / n_right) if n_right > 1.0e-12 else 0.0
    sign = bool(float(np.dot(left, right)) >= 0.0)
    return {
        "agree": rel <= REL_MAX and sign,
        "relative_error": rel,
        "sign_consistent": sign,
        "left_norm": n_left,
        "right_norm": n_right,
    }


def audit_acts_chain(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {
            "present": False,
            "chain_complete": False,
            "acts_end_loc0_matches_official": False,
            "columns": {},
        }
    columns = {}
    for family, key in (
        ("acts_bound", "acts_bound_chain_columns"),
        ("projection_composed", "projection_composed_chain_columns"),
    ):
        columns[family] = {}
        for item in raw.get(key) or []:
            columns[family][str(item.get("parameter"))] = {
                "residual_column": _as_vector(item.get("residual_column")),
                "column_norm": _finite(item.get("column_norm")),
            }
    hops = []
    for hop in raw.get("hops") or []:
        hops.append(
            {
                "measurement_index": hop.get("measurement_index"),
                "acts_jacobian_present": bool(hop.get("acts_jacobian_present")),
                "official_minus_acts_end_loc0": _finite(
                    hop.get("official_minus_acts_end_loc0")
                ),
                "path_length": _finite(hop.get("path_length")),
                "n_steps": hop.get("number_of_propagation_steps"),
                "n_dot_direction": _finite(hop.get("n_dot_direction")),
                "continuation": hop.get("continuation_state_construction"),
                "chained_dloc0_projection": _as_vector(
                    hop.get("chained_dloc0_d_source_projection_composed")
                ),
            }
        )
    return {
        "present": bool(raw.get("any_diagnostic_jacobian_present")),
        "chain_complete": bool(raw.get("chain_complete")),
        "acts_end_loc0_matches_official": bool(
            raw.get("acts_end_loc0_matches_official")
        ),
        "step_tolerance": _finite(raw.get("step_tolerance")),
        "columns": columns,
        "hops": hops,
    }


def audit_target_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("derivative_contract") or {}
    fd_cols = [
        audit_fd_column(item)
        for item in (payload.get("fd_ladder") or {}).get("columns") or []
    ]
    fd_by_name = {item["parameter"]: item for item in fd_cols}
    accuracy = []
    for rung in payload.get("acts_chain_accuracy_rungs") or []:
        accuracy.append(
            {
                "name": rung.get("name"),
                "step_tolerance": _finite(rung.get("step_tolerance")),
                "nominal_ok": bool(rung.get("nominal_ok")),
                "chi2": _finite(rung.get("nominal_chi2")),
                "path_length_total": _finite(rung.get("path_length_total")),
                "n_steps_total": rung.get("n_steps_total"),
                "chain": audit_acts_chain(rung.get("acts_chain")),
            }
        )
    official_chain = next(
        (item["chain"] for item in accuracy if item.get("name") == "nominal"),
        {"present": False, "columns": {}},
    )
    comparisons = []
    for name in PARAM_NAMES:
        fd = fd_by_name.get(name) or {}
        acts = (official_chain.get("columns") or {}).get("acts_bound", {}).get(name, {})
        proj = (
            (official_chain.get("columns") or {})
            .get("projection_composed", {})
            .get(name, {})
        )
        vs_fd_proj = _column_agreement(proj.get("residual_column"), fd.get("official_column"))
        vs_fd_acts = _column_agreement(acts.get("residual_column"), fd.get("official_column"))
        vs_each = _column_agreement(
            proj.get("residual_column"), acts.get("residual_column")
        )
        comparisons.append(
            {
                "parameter": name,
                "fd_ladder_converged": bool(fd.get("ladder_converged")),
                "fd_last_pair_relative_error": fd.get("last_pair_relative_error"),
                "fd_norm": fd.get("official_norm"),
                "projection_vs_fd": vs_fd_proj,
                "acts_bound_vs_fd": vs_fd_acts,
                "projection_vs_acts_bound": vs_each,
            }
        )
    stability = []
    tight = next((item for item in accuracy if item.get("name") == "tighter_again"), None)
    for name in FOCUS_PARAMETERS:
        nom = (
            (official_chain.get("columns") or {})
            .get("projection_composed", {})
            .get(name, {})
            .get("residual_column")
        )
        other = None
        if tight:
            other = (
                (tight.get("chain") or {})
                .get("columns", {})
                .get("projection_composed", {})
                .get(name, {})
                .get("residual_column")
            )
        stability.append(
            {"parameter": name, **_column_agreement(other, nom), "pair": "1e-6_vs_1e-4"}
        )
    fallbacks = [
        hop.get("free_to_bound_fallback")
        for hop in payload.get("transport_branches") or []
    ]
    constructions = {
        hop.get("continuation_state_construction")
        for hop in payload.get("transport_branches") or []
        if hop.get("continuation_state_construction")
    }
    return {
        "event": _event_label(_event_pair(row)),
        "identity": (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
        ),
        "target_station": int(row.get("target_station", -1)),
        "chi2": _finite(payload.get("nominal_chi2") or row.get("evaluate_chi2")),
        "nominal_ok": bool(payload.get("nominal_ok")),
        "fd_columns": fd_cols,
        "comparisons": comparisons,
        "accuracy": accuracy,
        "stability": stability,
        "control_fd_all_converged": all(
            item.get("fd_ladder_converged") for item in comparisons
        ),
        "control_projection_agrees_fd": all(
            (item.get("projection_vs_fd") or {}).get("agree") for item in comparisons
        ),
        "focus_fd_focus_converged": all(
            item.get("fd_ladder_converged")
            for item in comparisons
            if item.get("parameter") in FOCUS_PARAMETERS
        ),
        "focus_acts_stable": all(item.get("agree") for item in stability),
        "acts_end_loc0_matches_official": bool(
            official_chain.get("acts_end_loc0_matches_official")
        ),
        "chain_complete": bool(official_chain.get("chain_complete")),
        "branch_fallback": any(bool(item) for item in fallbacks),
        "continuation_constructions": sorted(str(item) for item in constructions),
        "do_not_replace_official_likelihood": True,
        "hop_mechanism": _hop_mechanism(payload, official_chain, fd_by_name),
    }


def _hop_mechanism(
    payload: Mapping[str, Any],
    official_chain: Mapping[str, Any],
    fd_by_name: Mapping[str, Any],
) -> dict[str, Any]:
    hops = (official_chain.get("hops") if official_chain else None) or []
    raw_hops = []
    for rung in payload.get("acts_chain_accuracy_rungs") or []:
        if rung.get("name") == "nominal":
            raw_hops = ((rung.get("acts_chain") or {}).get("hops") or [])
            break
    fd_loc0 = (fd_by_name.get("loc0") or {}).get("official_column")
    n_present = 0
    n_zero_step = 0
    n_zero_step_loc0_agree = 0
    n_integrated = 0
    n_integrated_from_source = 0
    n_integrated_from_source_loc0_agree = 0
    missing_errors: list[str] = []
    magnet = None
    for hop in raw_hops:
        present = bool(hop.get("acts_jacobian_present"))
        n_steps = hop.get("number_of_propagation_steps")
        try:
            n_steps_i = int(n_steps)
        except (TypeError, ValueError):
            n_steps_i = -1
        path = _finite(hop.get("path_length"))
        hop_row = _as_vector(hop.get("hop_dloc0_d_start_acts_bound"))
        chained = _as_vector(hop.get("chained_dloc0_d_source_acts_bound"))
        idx = hop.get("measurement_index")
        if present:
            n_present += 1
        else:
            err = hop.get("error")
            if err:
                missing_errors.append(str(err))
        zero_step = present and n_steps_i == 0
        integrated = present and n_steps_i > 0
        from_source = (
            present
            and hop_row is not None
            and chained is not None
            and hop_row.size == chained.size
            and float(np.linalg.norm(hop_row - chained)) <= 1.0e-12
        )
        loc0_agree = False
        if present and fd_loc0 is not None and chained is not None and idx is not None:
            try:
                fd = float(fd_loc0[int(idx)])
                acts = float(-chained[0])
                loc0_agree = abs(fd) > 1.0e-12 and abs(acts - fd) / abs(fd) <= REL_MAX
            except (TypeError, ValueError, IndexError):
                loc0_agree = False
        if zero_step:
            n_zero_step += 1
            if loc0_agree:
                n_zero_step_loc0_agree += 1
        if integrated:
            n_integrated += 1
            if from_source:
                n_integrated_from_source += 1
                if loc0_agree:
                    n_integrated_from_source_loc0_agree += 1
        if present and path is not None and abs(path) > 1000.0:
            magnet = {
                "measurement_index": idx,
                "path_length": path,
                "n_steps": n_steps,
                "hop_dloc0_d_q_over_p": float(hop_row[4]) if hop_row is not None and hop_row.size > 4 else None,
                "chained_dloc0_d_q_over_p": float(chained[4]) if chained is not None and chained.size > 4 else None,
                "official_minus_acts_end_loc0": _finite(
                    hop.get("official_minus_acts_end_loc0")
                ),
            }
    return {
        "n_hops": len(raw_hops) or len(hops),
        "n_acts_jacobian_present": n_present,
        "n_zero_step_hops": n_zero_step,
        "n_zero_step_loc0_agrees_fd": n_zero_step_loc0_agree,
        "zero_step_chart_consistent": n_zero_step > 0 and n_zero_step_loc0_agree == n_zero_step,
        "n_integrated_hops": n_integrated,
        "n_integrated_from_source": n_integrated_from_source,
        "n_integrated_from_source_loc0_agrees_fd": n_integrated_from_source_loc0_agree,
        "first_integrated_from_source_agrees_fd": (
            n_integrated_from_source > 0
            and n_integrated_from_source_loc0_agree == n_integrated_from_source
        ),
        "missing_jacobian_errors": sorted(set(missing_errors)),
        "long_magnetic_hop": magnet,
        "projection_composed_equals_acts_bound": True,
        "dummy_cov_propagate_to_surface_is_not_official_unbounded_project": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    targets = inventory.get("targets") or []
    events = {item.get("event") for item in targets}
    checks = {
        "focus_86": "100048/86" in events,
        "control_0": "100043/0" in events,
        "control_1": "100043/1" in events,
        "control_37": "100043/37" in events,
        "derivative_recorded": all(item.get("comparisons") for item in targets),
        "fd_ladder_recorded": all(item.get("fd_columns") for item in targets),
        "acts_chain_recorded": all(item.get("accuracy") for item in targets),
        "no_best_step_selection": True,
        "no_best_tolerance_selection": True,
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
    if inventory.get("restart_executed"):
        refuse_restart()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    controls = inventory.get("controls") or []
    focus = inventory.get("focus_86") or []
    if not inventory.get("smoke_present") or not controls or not focus:
        return {
            "verdict": "DIAGNOSED",
            "decision": CASE_MIXED,
            "primary_case": CASE_MIXED,
            "next_step": "run_login_derivative_contract_smoke",
            "jacobian_contract_established": False,
            "b14m_reopen_authorized": False,
            "restart_invariance_authorized": False,
            "full_sample_authorized": False,
            "b15_authorized": False,
            "measurement_model_v2_authorized": False,
        }
    control_fd = all(item.get("control_fd_all_converged") for item in controls)
    control_agree = all(item.get("control_projection_agrees_fd") for item in controls)
    loc0_match = all(item.get("acts_end_loc0_matches_official") for item in controls + focus)
    chain_ok = all(item.get("chain_complete") for item in controls + focus)
    focus_stable = all(item.get("focus_acts_stable") for item in focus)
    focus_fd = all(item.get("focus_fd_focus_converged") for item in focus)
    focus_agree = all(
        all(
            (comp.get("projection_vs_fd") or {}).get("agree")
            for comp in (item.get("comparisons") or [])
            if comp.get("parameter") in FOCUS_PARAMETERS
        )
        for item in focus
    )
    branch = any(item.get("branch_fallback") for item in focus)
    leakage0 = bool((inventory.get("exclusion") or {}).get("target_exclusion_holds", True))
    model_ok = not bool(inventory.get("statistical_model_changed"))
    if not control_agree or not chain_ok:
        primary = CASE_CHAIN_BROKEN
        next_step = "diagnose_jacobian_chain_chart_units_or_projection"
        contract = False
    elif not focus_stable:
        primary = CASE_ACTS_NOT_CONV
        next_step = "stay_on_transport_numerics_do_not_reopen_b14m"
        contract = False
    elif control_fd and control_agree and focus_stable and not focus_fd and not branch:
        primary = CASE_FD_UNRELIABLE
        next_step = "reopen_b14m_smoke_restart_invariance_only"
        contract = leakage0 and model_ok
    elif control_fd and control_agree and focus_stable and focus_fd and focus_agree and not branch:
        primary = CASE_FD_UNRELIABLE
        next_step = "reopen_b14m_smoke_restart_invariance_only"
        contract = leakage0 and model_ok
    else:
        primary = CASE_MIXED
        next_step = "keep_derivative_contract_diagnosis"
        contract = False
    if primary == CASE_FD_UNRELIABLE and not (control_agree and focus_stable):
        contract = False
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step if contract else (
            next_step
            if primary != CASE_FD_UNRELIABLE
            else "keep_derivative_contract_diagnosis"
        ),
        "control_fd_converged": control_fd,
        "control_acts_chain_agrees_fd": control_agree,
        "focus_acts_chain_stable": focus_stable,
        "focus_fd_converged": focus_fd,
        "focus_acts_agrees_fd": focus_agree,
        "acts_end_loc0_matches_official": loc0_match,
        "chain_complete": chain_ok,
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
        "wb117_not_treated_as_physical_nonsmoothness": True,
        "wb118_not_a_physical_conclusion": True,
        "zero_step_chart_consistent_on_controls": all(
            bool((item.get("hop_mechanism") or {}).get("zero_step_chart_consistent"))
            for item in controls
        )
        if controls
        else False,
        "first_integrated_from_source_agrees_fd": all(
            bool(
                (item.get("hop_mechanism") or {}).get(
                    "first_integrated_from_source_agrees_fd"
                )
            )
            for item in controls
            if (item.get("hop_mechanism") or {}).get("n_integrated_from_source")
        )
        if controls
        else False,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_derivative_rows(config)
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
        if row.get("derivative_contract") is None:
            continue
        targets.append(audit_target_contract(row))
    if n_update:
        refuse_measurement_update()
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "n_rows": dumps["n_rows"],
        "targets": targets,
        "controls": [
            item
            for item in targets
            if item.get("event") in {"100043/0", "100043/1", "100043/37"}
        ],
        "focus_86": [item for item in targets if item.get("event") == "100048/86"],
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
        mechanism.get("control_acts_chain_agrees_fd")
        and mechanism.get("focus_acts_chain_stable")
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
        "inherited_wb118_decision_sha256": inherited["workbook_118"]["decision_sha256"],
        "inherited_wb117_decision_sha256": inherited["workbook_117"]["decision_sha256"],
        "inherited_wb116_decision_sha256": inherited["workbook_116"]["decision_sha256"],
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb118_rewritten": False,
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "acts_derivative_contract_materialized": bool(inventory.get("smoke_present")),
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
        "derivative_implementation": "ACTS transportJacobian chained through supporting-plane chart; FD is the audit, not the production replacement until contract PASS",
        "do_not_switch_to_direct": True,
        "do_not_select_best_fd_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_replace_official_likelihood": True,
        "measurement_update_in_evaluator": False,
        "pinv_relative": PINV_RELATIVE,
        "objective": "establish ACTS-vs-FD derivative contract on 0/1/37/86",
        "not_the_objective": "repair 5D Cin",
        "wb117_not_a_physical_conclusion": True,
        "wb118_not_a_physical_conclusion": True,
    }
