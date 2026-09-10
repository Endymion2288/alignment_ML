"""Task B14U: official supporting-plane transport Jacobian contract.

Builds the derivative of official Mode-B h_i on the same path:
source bound -> unbounded navigator/stepper -> final free state ->
supporting-plane intersection -> local loc0 -> residual.

Does not reuse dummy-cov bound-to-surface
Propagator::Result::transportJacobian as the official derivative.
Does not retune Gauss-Newton, change production stepTolerance, pick a
best FD step, replace the official likelihood, introduce a prior/ridge,
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
    inherit_frozen_stage as inherit_through_wb118,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profile_transport_contract import (
    audit_exclusion,
    is_official_mode_b,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "official-supporting-plane-jacobian-v1"
DEFAULT_CONFIG = "configs/official_supporting_plane_jacobian_v1.yaml"
TASK = "SB-B14U"
WORKBOOK = 120

CASE_ESTABLISHED = "official_supporting_plane_jacobian_established"
CASE_UNAVAILABLE = "acts_free_state_jacobian_unavailable"
CASE_INCONSISTENT = "official_path_jacobian_inconsistent_with_fd"

WB119_DECISION = "analytic_chain_or_chart_contract_broken"
REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
CONTROL_LABELS = {"100043/0", "100043/1", "100043/37"}
FOCUS_LABEL = "100048/86"


class OfficialSupportingPlaneJacobianError(ValueError):
    """Raised when the B14U official-path Jacobian contract is illegal."""


def refuse_prior() -> None:
    raise OfficialSupportingPlaneJacobianError("B14U must not introduce a prior")


def refuse_ridge_information() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "ridge must not be treated as statistical information"
    )


def refuse_measurement_update() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_best_step_selection() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "must not select the best finite-difference step from the results"
    )


def refuse_best_tolerance() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "must not select the best propagator stepTolerance from the results"
    )


def refuse_switch_to_direct() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "must not replace the official sequential likelihood with direct-from-source"
    )


def refuse_replace_likelihood() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "must not replace the official likelihood with the ACTS Jacobian"
    )


def refuse_dummy_cov_bounded() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "must not use dummy-cov bounded transportJacobian as the official derivative"
    )


def refuse_b14m() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "B14M is not re-opened inside Task B14U"
    )


def refuse_restart() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "restart invariance is not opened inside Task B14U"
    )


def refuse_b15() -> None:
    raise OfficialSupportingPlaneJacobianError("Task B15 is not entered in Task B14U")


def refuse_measurement_model_v2() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "Measurement Model V2 is not entered in Task B14U"
    )


def refuse_full_sample() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "the 1989-row campaign must not be submitted in B14U"
    )


def refuse_change_statistical_model() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "the measurement statistical model must stay frozen"
    )


def refuse_5d_cin_repair() -> None:
    raise OfficialSupportingPlaneJacobianError(
        "B14U must not treat 5D Cin repair as the objective"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise OfficialSupportingPlaneJacobianError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise OfficialSupportingPlaneJacobianError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise OfficialSupportingPlaneJacobianError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise OfficialSupportingPlaneJacobianError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise OfficialSupportingPlaneJacobianError(
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
            raise OfficialSupportingPlaneJacobianError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14l",
        "do_not_select_best_fd_step",
        "do_not_switch_official_likelihood_to_direct",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
    ):
        if not bool(config.get(key, False)):
            raise OfficialSupportingPlaneJacobianError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14u", True)):
        raise OfficialSupportingPlaneJacobianError("do_not_enter_b14u must be false")
    spec = config["official_supporting_plane_jacobian"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise OfficialSupportingPlaneJacobianError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise OfficialSupportingPlaneJacobianError(
            "production stepTolerance must stay 1e-4"
        )
    if not bool(spec["do_not_use_dummy_cov_bounded_transportJacobian"]):
        raise OfficialSupportingPlaneJacobianError(
            "dummy-cov bounded transportJacobian is forbidden as the official derivative"
        )
    allowed = list(spec["allowed_decisions"])
    if allowed != [CASE_ESTABLISHED, CASE_UNAVAILABLE, CASE_INCONSISTENT]:
        raise OfficialSupportingPlaneJacobianError("allowed decisions must stay the B14U set")
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb118(config)
    spec = config["inheritance"]["workbook_119"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_119 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_119 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise OfficialSupportingPlaneJacobianError(
            "workbook_119 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB119_DECISION:
        raise OfficialSupportingPlaneJacobianError("WB119 decision token mismatch")
    if decision.get("b14m_reopen_authorized"):
        raise OfficialSupportingPlaneJacobianError("WB119 must not have re-opened B14M")
    if decision.get("jacobian_contract_established"):
        raise OfficialSupportingPlaneJacobianError(
            "WB119 must not have established the Jacobian contract"
        )
    inherited["workbook_119"] = {
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
        "control_fd_converged": True,
        "control_acts_chain_agrees_fd": False,
        "took_wrong_jacobian": True,
        "wb119_not_a_physical_conclusion": True,
    }
    return inherited


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(
        project_root(), str(config["official_jacobian_smoke_root"])
    )
    filename = str(
        config.get(
            "official_jacobian_smoke_filename",
            "ckf_leave_target_out_official_jacobian.jsonl",
        )
    )
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_jacobian_rows(config: Mapping[str, Any]) -> dict[str, Any]:
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


def audit_official_chain(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {
            "present": False,
            "chain_complete": False,
            "loc0_matches_official": False,
            "dummy_cov_bounded_used": False,
            "official_jac_transport_is_identity": True,
            "columns": {},
            "hops": [],
        }
    columns: dict[str, dict[str, Any]] = {}
    for family, key in (
        ("rk_free", "rk_free_chain_columns"),
        ("acts_composed", "acts_composed_chain_columns"),
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
                "free_state_jacobian_available": bool(
                    hop.get("free_state_jacobian_available")
                ),
                "official_cov_transport": bool(hop.get("official_cov_transport")),
                "official_jac_transport_is_identity": bool(
                    hop.get("official_jac_transport_is_identity", True)
                ),
                "diagnostic_cov_transport": bool(hop.get("diagnostic_cov_transport")),
                "dummy_cov_bounded_used": bool(
                    hop.get("dummy_cov_bounded_transportJacobian_used")
                ),
                "diagnostic_uses_bounded_transportJacobian": bool(
                    hop.get("diagnostic_uses_bounded_transportJacobian")
                ),
                "loc0_matches_official": bool(hop.get("loc0_matches_official")),
                "official_minus_diagnostic_loc0": _finite(
                    hop.get("official_minus_diagnostic_loc0")
                ),
                "n_steps": hop.get("number_of_propagation_steps"),
                "n_material_resets": hop.get("n_material_resets"),
                "path_length": _finite(hop.get("path_length")),
                "continuation": hop.get("continuation_state_construction"),
                "chained_dloc0_rk_free": _as_vector(
                    hop.get("chained_dloc0_d_source_rk_free")
                ),
                "source_chart": hop.get("source_chart"),
            }
        )
    return {
        "present": bool(raw.get("any_free_state_jacobian_available")),
        "chain_complete": bool(raw.get("chain_complete")),
        "loc0_matches_official": bool(raw.get("official_loc0_matches_diagnostic")),
        "dummy_cov_bounded_used": bool(
            raw.get("dummy_cov_bounded_transportJacobian_used")
        ),
        "official_jac_transport_is_identity": bool(
            raw.get("official_path_jacTransport_is_identity")
        ),
        "primary_family": raw.get("primary_family", "rk_free_chain"),
        "columns": columns,
        "hops": hops,
    }


def audit_target_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    fd_cols = [
        audit_fd_column(item)
        for item in (payload.get("fd_ladder") or {}).get("columns") or []
    ]
    fd_by_name = {item["parameter"]: item for item in fd_cols}
    chain = audit_official_chain(payload.get("official_path_jacobian"))
    comparisons = []
    for name in PARAM_NAMES:
        fd = fd_by_name.get(name) or {}
        rk = (chain.get("columns") or {}).get("rk_free", {}).get(name, {})
        composed = (chain.get("columns") or {}).get("acts_composed", {}).get(name, {})
        vs_fd_rk = _column_agreement(rk.get("residual_column"), fd.get("official_column"))
        vs_fd_composed = _column_agreement(
            composed.get("residual_column"), fd.get("official_column")
        )
        comparisons.append(
            {
                "parameter": name,
                "fd_ladder_converged": bool(fd.get("ladder_converged")),
                "fd_last_pair_relative_error": fd.get("last_pair_relative_error"),
                "fd_norm": fd.get("official_norm"),
                "rk_free_vs_fd": vs_fd_rk,
                "acts_composed_vs_fd": vs_fd_composed,
            }
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
    hops = chain.get("hops") or []
    n_ok = sum(1 for hop in hops if hop.get("free_state_jacobian_available"))
    n_stepped = 0
    n_stepped_available = 0
    for hop in hops:
        try:
            n_steps = int(hop.get("n_steps"))
        except (TypeError, ValueError):
            n_steps = -1
        if n_steps > 0:
            n_stepped += 1
            if hop.get("free_state_jacobian_available"):
                n_stepped_available += 1
    dummy_bounded = bool(chain.get("dummy_cov_bounded_used")) or any(
        hop.get("dummy_cov_bounded_used")
        or hop.get("diagnostic_uses_bounded_transportJacobian")
        for hop in hops
    )
    if dummy_bounded:
        refuse_dummy_cov_bounded()
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
        "chain": chain,
        "control_fd_all_converged": all(
            item.get("fd_ladder_converged") for item in comparisons
        ),
        "control_rk_agrees_fd": all(
            (item.get("rk_free_vs_fd") or {}).get("agree") for item in comparisons
        ),
        "focus_fd_all_converged": all(
            item.get("fd_ladder_converged") for item in comparisons
        ),
        "focus_rk_agrees_fd": all(
            (item.get("rk_free_vs_fd") or {}).get("agree") for item in comparisons
        ),
        "loc0_matches_official": bool(chain.get("loc0_matches_official")),
        "chain_complete": bool(chain.get("chain_complete")),
        "free_jacobian_available": bool(chain.get("present"))
        and bool(chain.get("chain_complete"))
        and n_ok == len(hops)
        and (n_stepped == 0 or n_stepped_available == n_stepped),
        "official_jac_transport_is_identity": bool(
            chain.get("official_jac_transport_is_identity")
        ),
        "dummy_cov_bounded_used": dummy_bounded,
        "n_hops": len(hops),
        "n_available_hops": n_ok,
        "n_stepped_hops": n_stepped,
        "n_stepped_available": n_stepped_available,
        "branch_fallback": any(bool(item) for item in fallbacks),
        "continuation_constructions": sorted(str(item) for item in constructions),
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
        "official_path_jacobian_recorded": all(
            item.get("comparisons") and item.get("chain") for item in targets
        ),
        "fd_ladder_recorded": all(item.get("fd_columns") for item in targets),
        "no_dummy_cov_bounded_transportJacobian": not any(
            item.get("dummy_cov_bounded_used") for item in targets
        ),
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
    control_fd = all(item.get("control_fd_all_converged") for item in controls) if controls else False
    control_agree = all(item.get("control_rk_agrees_fd") for item in controls) if controls else False
    focus_fd = all(item.get("focus_fd_all_converged") for item in focus) if focus else False
    focus_agree = all(item.get("focus_rk_agrees_fd") for item in focus) if focus else False
    loc0_match = all(item.get("loc0_matches_official") for item in controls + focus) if (controls and focus) else False
    if not inventory.get("smoke_present") or not controls or not focus or not jac_available:
        primary = CASE_UNAVAILABLE
        next_step = "keep_official_path_jacobian_diagnosis"
        contract = False
    elif (
        control_fd
        and control_agree
        and focus_fd
        and focus_agree
        and loc0_match
        and not branch
        and leakage0
        and model_ok
    ):
        primary = CASE_ESTABLISHED
        next_step = "reopen_b14m_smoke_restart_invariance_only"
        contract = True
    else:
        primary = CASE_INCONSISTENT
        next_step = "keep_official_path_jacobian_diagnosis"
        contract = False
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "control_fd_converged": control_fd,
        "control_official_path_agrees_fd": control_agree,
        "focus_fd_converged": focus_fd,
        "focus_official_path_agrees_fd": focus_agree,
        "official_loc0_matches_diagnostic": loc0_match,
        "free_state_jacobian_available": jac_available,
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
        "wb117_not_treated_as_physical_nonsmoothness": True,
        "wb118_not_a_physical_conclusion": True,
        "wb119_took_wrong_jacobian": True,
        "wb119_not_a_physical_conclusion": True,
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
        targets.append(audit_target_contract(row))
    if n_update:
        refuse_measurement_update()
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
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
        and mechanism.get("control_official_path_agrees_fd")
        and mechanism.get("focus_official_path_agrees_fd")
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
        "inherited_wb119_decision_sha256": inherited["workbook_119"]["decision_sha256"],
        "inherited_wb118_decision_sha256": inherited["workbook_118"]["decision_sha256"],
        "inherited_wb117_decision_sha256": inherited["workbook_117"]["decision_sha256"],
        "inherited_wb116_decision_sha256": inherited["workbook_116"]["decision_sha256"],
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb119_rewritten": False,
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "official_supporting_plane_jacobian_materialized": bool(
            inventory.get("smoke_present")
        ),
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
            "official unbounded free-state Jacobian chained through "
            "supporting-plane intersection and loc0; FD is the audit"
        ),
        "do_not_switch_to_direct": True,
        "do_not_select_best_fd_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_replace_official_likelihood": True,
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "measurement_update_in_evaluator": False,
        "pinv_relative": PINV_RELATIVE,
        "objective": "establish official supporting-plane path Jacobian on 0/1/37/86",
        "not_the_objective": "repair 5D Cin",
        "wb117_not_a_physical_conclusion": True,
        "wb118_not_a_physical_conclusion": True,
        "wb119_took_wrong_jacobian": True,
    }
