"""Task B14J: supporting-plane Jacobian continuity contract.

Proves whether the frozen measurement map h_i(theta) is locally
continuous and has a certifiable Jacobian on the pre-registered smoke
0/1/37/86.  Does not retune Gauss-Newton, pick a best FD step, switch
the official sequential likelihood to direct-from-source, introduce a
prior/ridge, delete 37/86, replace q/p, repair 5D Cin, or enter B14M /
B15 / Measurement Model V2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.profiled_measurement_likelihood import (
    ALPHA_NAMES,
    NU_NAMES,
    PINV_RELATIVE,
    refuse_alignment_rank_tolerance,
    refuse_ridge,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profile_transport_contract import (
    CASE_MIXED as WB116_DECISION,
    CLASS_M_EVENT,
    CLASS_S_EVENT,
    CONTROL_EVENTS,
    audit_exclusion,
    inherit_frozen_stage as inherit_through_wb115,
    is_official_mode_b,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "supporting-plane-jacobian-continuity-v1"
DEFAULT_CONFIG = "configs/supporting_plane_jacobian_continuity_v1.yaml"
TASK = "SB-B14J"
WORKBOOK = 117

CASE_ESTABLISHED = "supporting_plane_jacobian_continuity_established"
CASE_FD_STEP = "finite_difference_step_not_in_asymptotic_region"
CASE_BRANCH = "acts_navigation_material_branch_switching"
CASE_CONTINUATION = "supporting_plane_continuation_state_discontinuous"
CASE_TRANSPORT = "source_to_measurement_transport_not_smooth"
CASE_MIXED = "mixed_or_inconclusive"

PARAM_NAMES = ("loc0", "loc1", "phi", "theta", "q_over_p")
FOCUS_PARAMETERS = ("loc1", "phi", "q_over_p")
REQUIRED_EVENTS = CONTROL_EVENTS | {CLASS_M_EVENT, CLASS_S_EVENT}
RUNG_FACTORS = (1.0, 0.5, 0.25, 0.125)
REL_MAX = 0.05


class SupportingPlaneJacobianContinuityError(ValueError):
    """Raised when the B14J Jacobian continuity contract is illegal."""


def refuse_prior() -> None:
    raise SupportingPlaneJacobianContinuityError("B14J must not introduce a prior")


def refuse_ridge_information() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "ridge must not be treated as statistical information"
    )


def refuse_measurement_update() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_best_step_selection() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "must not select the best finite-difference step from the results"
    )


def refuse_switch_to_direct() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "must not replace the official sequential likelihood with direct-from-source"
    )


def refuse_delete_focus() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "focus identities 100043/37 and 100048/86 must be retained"
    )


def refuse_b14m() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "B14M is not re-opened inside Task B14J before the Jacobian contract"
    )


def refuse_b15() -> None:
    raise SupportingPlaneJacobianContinuityError("Task B15 is not entered in Task B14J")


def refuse_measurement_model_v2() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "Measurement Model V2 is not entered in Task B14J"
    )


def refuse_full_sample() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "the 1989-row campaign must not be submitted in B14J"
    )


def refuse_change_statistical_model() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "the measurement statistical model must stay frozen"
    )


def refuse_rewrite_profile_math() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "WB114 profile mathematics must not be rewritten"
    )


def refuse_5d_cin_repair() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "B14J must not treat 5D Cin repair as the objective"
    )


def refuse_restart_before_contract() -> None:
    raise SupportingPlaneJacobianContinuityError(
        "restart invariance is not re-opened before jacobian_contract_established"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise SupportingPlaneJacobianContinuityError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SupportingPlaneJacobianContinuityError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise SupportingPlaneJacobianContinuityError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise SupportingPlaneJacobianContinuityError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise SupportingPlaneJacobianContinuityError(
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
            raise SupportingPlaneJacobianContinuityError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14n",
        "do_not_enter_b14t",
        "do_not_rewrite_profile_math",
        "do_not_change_statistical_model",
        "do_not_submit_full_sample_without_smoke_gate",
        "do_not_introduce_target_independent_prior",
        "do_not_delete_qoverp",
        "do_not_fix_qoverp",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
        "do_not_select_best_seed",
        "do_not_add_measurement_update_in_evaluator",
        "do_not_select_best_fd_step",
        "do_not_switch_official_likelihood_to_direct",
        "do_not_reopen_b14m_before_jacobian_contract",
        "do_not_run_restart_before_jacobian_contract",
        "do_not_claim_5d_cin_repaired",
    ):
        if bool(config.get(key, False)) is not True:
            raise SupportingPlaneJacobianContinuityError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14j", True)):
        raise SupportingPlaneJacobianContinuityError(
            "B14J config must allow entering B14J"
        )
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise SupportingPlaneJacobianContinuityError(f"frozen gate changed: {key}")
    numeric = config.get("profile_numerical") or {}
    if float(numeric.get("pinv_relative")) == 0.01:
        refuse_alignment_rank_tolerance()
    if abs(float(numeric.get("pinv_relative")) - PINV_RELATIVE) > 1.0e-20:
        raise SupportingPlaneJacobianContinuityError(
            "pinv_relative must stay pre-registered 1e-8"
        )
    if bool(numeric.get("do_not_add_ridge")) is not True:
        refuse_ridge()
    continuity = config.get("jacobian_continuity") or {}
    if list(continuity.get("rung_factors") or []) != list(RUNG_FACTORS):
        raise SupportingPlaneJacobianContinuityError(
            "Jacobian ladder must stay h, h/2, h/4, h/8"
        )
    if bool(continuity.get("do_not_select_best_step")) is not True:
        refuse_best_step_selection()
    if bool(continuity.get("do_not_switch_to_direct")) is not True:
        refuse_switch_to_direct()
    if bool(continuity.get("official_fd_step_unchanged")) is not True:
        raise SupportingPlaneJacobianContinuityError(
            "official FD step must stay the pre-registered h"
        )
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise SupportingPlaneJacobianContinuityError(
            "B14J eligibility must stay identical to WB103"
        )
    partition = config["profile_partition"]
    if list(partition["alpha_native"]) != list(ALPHA_NAMES):
        raise SupportingPlaneJacobianContinuityError("alpha must stay loc0, theta")
    if list(partition["nu_native"]) != list(NU_NAMES):
        raise SupportingPlaneJacobianContinuityError("nu must stay loc1, phi, q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb115(config)
    spec = config["inheritance"]["workbook_116"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_116 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_116 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise SupportingPlaneJacobianContinuityError(
            "workbook_116 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB116_DECISION:
        raise SupportingPlaneJacobianContinuityError("WB116 decision token mismatch")
    if decision.get("b15_authorized"):
        raise SupportingPlaneJacobianContinuityError("WB116 must not have authorized B15")
    if decision.get("prior_introduced"):
        raise SupportingPlaneJacobianContinuityError(
            "WB116 must not have introduced a prior"
        )
    if decision.get("b14m_reopen_authorized"):
        raise SupportingPlaneJacobianContinuityError(
            "WB116 must not have re-opened B14M"
        )
    inherited["workbook_116"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "smoke_gate_passed": False,
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "jacobian_contract_established": False,
        "target_exclusion_holds": True,
        "statistical_model_unchanged": True,
        "prior_introduced": False,
        "b15_authorized": False,
        "do_not_force_5d_lto_covariance": True,
    }
    return inherited


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("source_id")),
        int(row.get("run_id", -1)),
        int(row.get("event_id", -1)),
    )


def _event_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["jacobian_smoke_root"]))
    filename = str(
        config.get(
            "jacobian_smoke_filename", "ckf_leave_target_out_jacobian_continuity.jsonl"
        )
    )
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_jacobian_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in smoke_dump_paths(config):
        for row in load_dump_records(path, split="train"):
            tagged = dict(row)
            tagged["split"] = "smoke"
            tagged["dump_path"] = str(path)
            rows.append(tagged)
    return {
        "rows": rows,
        "smoke_present": bool(rows),
        "n_rows": len(rows),
        "paths": [str(path) for path in smoke_dump_paths(config)],
    }


def _same_sequence(left: Any, right: Any) -> bool:
    return list(left or []) == list(right or [])


def fingerprint_branch(eval_json: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = eval_json or {}
    return {
        "ok": bool(payload.get("ok")),
        "destination_geometry_ids": list(payload.get("destination_geometry_ids") or []),
        "continuation_state_construction": list(
            payload.get("continuation_state_construction") or []
        ),
        "free_to_bound_fallback": [bool(v) for v in (payload.get("free_to_bound_fallback") or [])],
        "inside_active_bounds": [bool(v) for v in (payload.get("inside_active_bounds") or [])],
        "n_steps_total": payload.get("n_steps_total"),
        "path_length_total": payload.get("path_length_total"),
        "n_free_to_bound_fallback": payload.get("n_free_to_bound_fallback"),
        "geometry_sequences": [
            list((hop or {}).get("geometry_surface_sequence") or [])
            for hop in (payload.get("hops") or [])
        ],
    }


def branch_changed(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("destination_geometry_ids") != b.get("destination_geometry_ids"):
        return True
    if a.get("continuation_state_construction") != b.get(
        "continuation_state_construction"
    ):
        return True
    if a.get("free_to_bound_fallback") != b.get("free_to_bound_fallback"):
        return True
    if a.get("geometry_sequences") != b.get("geometry_sequences"):
        return True
    return False


def loc_jump(eval_a: Mapping[str, Any] | None, eval_b: Mapping[str, Any] | None) -> float | None:
    if not eval_a or not eval_b:
        return None
    left = eval_a.get("predicted_loc0") or []
    right = eval_b.get("predicted_loc0") or []
    deltas = []
    for a, b in zip(left, right):
        fa = _finite(a)
        fb = _finite(b)
        if fa is not None and fb is not None:
            deltas.append(abs(fa - fb))
    return max(deltas) if deltas else None


def audit_column(column: Mapping[str, Any], rel_max: float = REL_MAX) -> dict[str, Any]:
    if bool(column.get("selected_best_step")):
        refuse_best_step_selection()
    name = str(column.get("parameter"))
    pairs = list(column.get("consecutive_relative_errors") or [])
    rels = [_finite(item.get("relative_error")) for item in pairs]
    signs = [bool(item.get("sign_consistent")) for item in pairs]
    rungs = list(column.get("rungs") or [])
    if len(rungs) != 4 or list(RUNG_FACTORS) != [
        float((rung.get("step_factor") if rung.get("step_factor") is not None else -1))
        for rung in rungs
    ]:
        raise SupportingPlaneJacobianContinuityError(
            "every column must record the fixed h, h/2, h/4, h/8 ladder"
        )
    branch = False
    fingerprints = []
    for rung in rungs:
        plus = fingerprint_branch(rung.get("plus"))
        minus = fingerprint_branch(rung.get("minus"))
        fingerprints.append({"plus": plus, "minus": minus})
        if branch_changed(plus, minus):
            branch = True
    for left, right in zip(fingerprints, fingerprints[1:]):
        if branch_changed(left["plus"], right["plus"]) or branch_changed(
            left["minus"], right["minus"]
        ):
            branch = True
    last_rel = rels[-1] if rels else None
    first_rel = rels[0] if rels else None
    finite_rels = [value for value in rels if value is not None]
    decreasing = len(finite_rels) >= 2 and all(
        finite_rels[i] + 1.0e-9 >= finite_rels[i + 1] for i in range(len(finite_rels) - 1)
    )
    all_evaluated = all(bool(rung.get("ok")) for rung in rungs)
    sign_ok = bool(column.get("sign_consistency")) and all(signs) if signs else False
    last_pair_ok = last_rel is not None and last_rel <= rel_max
    official_asymptotic = first_rel is not None and first_rel <= rel_max
    converged = all_evaluated and sign_ok and last_pair_ok and not branch
    return {
        "parameter": name,
        "relative_errors": rels,
        "sign_consistency": sign_ok,
        "all_rungs_evaluated": all_evaluated,
        "last_pair_relative_error": last_rel,
        "first_pair_relative_error": first_rel,
        "relative_errors_monotone_decreasing": decreasing,
        "official_fd_in_asymptotic_region": official_asymptotic,
        "surface_path_branch_consistent": not branch,
        "ladder_converged": converged,
        "selected_best_step": False,
        "failed_finite_difference_evaluations": column.get(
            "failed_finite_difference_evaluations"
        ),
    }


def classify_column_mechanism(
    sequential: Mapping[str, Any],
    direct: Mapping[str, Any],
) -> str:
    seq_ok = bool(sequential.get("ladder_converged"))
    dir_ok = bool(direct.get("ladder_converged"))
    if seq_ok:
        return "pass"
    if not sequential.get("surface_path_branch_consistent"):
        return CASE_BRANCH
    if dir_ok and not seq_ok:
        return CASE_CONTINUATION
    if sequential.get("relative_errors_monotone_decreasing") and not sequential.get(
        "official_fd_in_asymptotic_region"
    ):
        return CASE_FD_STEP
    if (not dir_ok) and sequential.get("surface_path_branch_consistent") and direct.get(
        "surface_path_branch_consistent"
    ):
        if sequential.get("relative_errors_monotone_decreasing") and not sequential.get(
            "ladder_converged"
        ):
            return CASE_FD_STEP
        return CASE_TRANSPORT
    return CASE_MIXED


def audit_mode(mode: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = mode or {}
    columns = []
    by_name = {}
    for raw in payload.get("columns") or []:
        report = audit_column(raw)
        columns.append(report)
        by_name[report["parameter"]] = report
    established = bool(columns) and all(item["ladder_converged"] for item in columns)
    return {
        "hop_mode": payload.get("hop_mode"),
        "n_columns": len(columns),
        "columns": columns,
        "by_parameter": by_name,
        "ladder_converged": established,
        "do_not_select_best_step": True,
        "selected_best_step": False,
    }


def audit_target_continuity(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = row.get("jacobian_continuity") or {}
    if bool(audit.get("selected_best_step")):
        refuse_best_step_selection()
    if bool(audit.get("switched_to_direct")):
        refuse_switch_to_direct()
    sequential = audit_mode(audit.get("sequential"))
    direct = audit_mode(audit.get("direct"))
    mechanisms = {}
    for name in PARAM_NAMES:
        seq_col = sequential["by_parameter"].get(name) or {}
        dir_col = direct["by_parameter"].get(name) or {}
        if seq_col or dir_col:
            mechanisms[name] = classify_column_mechanism(seq_col, dir_col)
    focus_fail = [
        name
        for name in FOCUS_PARAMETERS
        if mechanisms.get(name) not in {None, "pass"}
        and not (sequential["by_parameter"].get(name) or {}).get("ladder_converged")
    ]
    if sequential["ladder_converged"]:
        mechanism = "pass"
    elif any(value == CASE_BRANCH for value in mechanisms.values()):
        mechanism = CASE_BRANCH
    elif any(value == CASE_CONTINUATION for value in mechanisms.values()):
        mechanism = CASE_CONTINUATION
    elif any(value == CASE_FD_STEP for value in mechanisms.values()):
        mechanism = CASE_FD_STEP
    elif any(value == CASE_TRANSPORT for value in mechanisms.values()):
        mechanism = CASE_TRANSPORT
    else:
        mechanism = CASE_MIXED
    return {
        "identity": list(_identity(row)),
        "target_station": row.get("target_station"),
        "event": f"{row.get('run_id')}/{row.get('event_id')}",
        "official_is_sequential": True,
        "direct_from_source_is_comparison_only": True,
        "do_not_switch_to_direct": True,
        "do_not_select_best_step": True,
        "all_surfaces_reached": bool(row.get("all_surfaces_reached")),
        "evaluate_chi2": row.get("evaluate_chi2"),
        "sequential": sequential,
        "direct": direct,
        "parameter_mechanisms": mechanisms,
        "focus_parameter_failures": focus_fail,
        "sequential_converged": sequential["ladder_converged"],
        "direct_converged": direct["ladder_converged"],
        "mechanism": mechanism,
        "contract_pass": sequential["ladder_converged"]
        and bool(row.get("all_surfaces_reached")),
    }


def audit_jacobian_continuity(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    targets = []
    by_event: dict[tuple[int, int], list[dict[str, Any]]] = {
        key: [] for key in REQUIRED_EVENTS
    }
    selected_best = 0
    switched = 0
    for row in rows:
        if not is_official_mode_b(row):
            continue
        if _event_pair(row) not in REQUIRED_EVENTS:
            continue
        if row.get("jacobian_continuity") is None:
            continue
        report = audit_target_continuity(row)
        targets.append(report)
        by_event[_event_pair(row)].append(report)
        if report.get("do_not_select_best_step") is False:
            selected_best += 1
    if selected_best:
        refuse_best_step_selection()
    if switched:
        refuse_switch_to_direct()
    event_ok = {}
    for key in REQUIRED_EVENTS:
        items = by_event[key]
        event_ok[f"{key[0]}/{key[1]}"] = bool(items) and all(
            item["contract_pass"] for item in items
        )
    established = bool(targets) and all(event_ok.values())
    focus_86 = [item for item in targets if _event_pair({"run_id": item["identity"][1], "event_id": item["identity"][2]}) == CLASS_S_EVENT]
    # identity is (source, run, event)
    focus_86 = [item for item in targets if tuple(item["identity"][1:]) == CLASS_S_EVENT]
    return {
        "n_targets": len(targets),
        "targets": targets,
        "events": event_ok,
        "jacobian_0_1_established": event_ok.get("100043/0") and event_ok.get("100043/1"),
        "jacobian_37_established": event_ok.get("100043/37"),
        "jacobian_86_established": event_ok.get("100048/86"),
        "jacobian_contract_established": established,
        "focus_86": focus_86,
        "do_not_select_best_step": True,
        "official_is_sequential": True,
        "direct_from_source_is_comparison_only": True,
        "do_not_switch_to_direct": True,
        "note": (
            "jacobian_contract_established requires convergence, sign "
            "consistency, and surface/path branch consistency on every "
            "official sequential column for 0/1/37/86.  Direct-from-source "
            "is comparison only."
        ),
    }


def audit_sequential_vs_direct(continuity: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in continuity.get("targets") or []:
        seq = bool(item.get("sequential_converged"))
        direct = bool(item.get("direct_converged"))
        if direct and not seq:
            priority = CASE_CONTINUATION
        elif (not direct) and (not seq):
            priority = CASE_TRANSPORT if item.get("mechanism") == CASE_TRANSPORT else item.get("mechanism")
        elif seq:
            priority = "pass"
        else:
            priority = item.get("mechanism")
        rows.append(
            {
                "identity": item.get("identity"),
                "target_station": item.get("target_station"),
                "sequential_converged": seq,
                "direct_converged": direct,
                "priority_root_cause": priority,
                "mechanism": item.get("mechanism"),
                "focus_parameter_failures": item.get("focus_parameter_failures"),
            }
        )
    return {
        "official_is_sequential": True,
        "direct_from_source_is_comparison_only": True,
        "do_not_switch_only_because_numerically_better": True,
        "switched_to_direct": False,
        "rows": rows,
        "n_direct_pass_sequential_fail": sum(
            1
            for row in rows
            if row["direct_converged"] and not row["sequential_converged"]
        ),
        "n_both_fail": sum(
            1
            for row in rows
            if (not row["direct_converged"]) and (not row["sequential_converged"])
        ),
    }


def audit_nominal_evaluations(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_event = {
        key: {"run_id": key[0], "event_id": key[1], "n_targets": 0, "n_reached": 0}
        for key in REQUIRED_EVENTS
    }
    n_update = 0
    for row in rows:
        if row.get("measurement_update_in_evaluator"):
            n_update += 1
        if not is_official_mode_b(row):
            continue
        key = _event_pair(row)
        if key not in by_event:
            continue
        by_event[key]["n_targets"] += 1
        if row.get("all_surfaces_reached") or row.get("nominal_chi2_evaluable"):
            by_event[key]["n_reached"] += 1
    if n_update:
        refuse_measurement_update()
    return {
        "events": {f"{k[0]}/{k[1]}": v for k, v in by_event.items()},
        "all_required_evaluable": all(
            item["n_targets"] > 0 and item["n_reached"] == item["n_targets"]
            for item in by_event.values()
        ),
        "measurement_update_in_evaluator": False,
        "focus_retained": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    evals = inventory.get("evaluations") or {}
    jacobian = inventory.get("jacobian") or {}
    exclusion = inventory.get("exclusion") or {}
    comparison = inventory.get("sequential_vs_direct") or {}
    checks = {
        "events_0_1_37_86_nominal": bool(evals.get("all_required_evaluable")),
        "jacobian_ladder_recorded": int(jacobian.get("n_targets") or 0) > 0,
        "no_best_step_selection": bool(jacobian.get("do_not_select_best_step", True)),
        "sequential_vs_direct_recorded": bool(comparison.get("rows")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "target_exclusion": bool(exclusion.get("target_exclusion_holds", True)),
        "statistical_model_unchanged": not bool(inventory.get("statistical_model_changed")),
        "no_measurement_update": evals.get("measurement_update_in_evaluator") is False,
        "focus_retained": bool(evals.get("focus_retained", True)),
        "smoke_present": bool(inventory.get("smoke_present")),
        "jacobian_contract": bool(jacobian.get("jacobian_contract_established")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "b14m_reopen_authorized": passed,
        "full_sample_authorized": False,
        "restart_invariance_authorized": passed,
        "do_not_submit_if_failed": True,
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
    if inventory.get("selected_best_step"):
        refuse_best_step_selection()
    if inventory.get("restart_executed") and not (
        (inventory.get("jacobian") or {}).get("jacobian_contract_established")
    ):
        refuse_restart_before_contract()
    gate = inventory.get("smoke_gate") or {}
    jacobian = inventory.get("jacobian") or {}
    comparison = inventory.get("sequential_vs_direct") or {}
    smoke = bool(inventory.get("smoke_present"))
    established = bool(jacobian.get("jacobian_contract_established"))
    if not smoke:
        primary = CASE_MIXED
        verdict = "DIAGNOSED"
        next_step = "run_login_jacobian_continuity_smoke"
    elif bool(gate.get("passed")) and established:
        primary = CASE_ESTABLISHED
        verdict = "PASS"
        next_step = "reopen_b14m_smoke_restart_invariance_only"
    else:
        verdict = "FAIL"
        n_cont = int(comparison.get("n_direct_pass_sequential_fail") or 0)
        n_both = int(comparison.get("n_both_fail") or 0)
        focus = jacobian.get("focus_86") or []
        focus_mechs = {item.get("mechanism") for item in focus}
        if CASE_BRANCH in focus_mechs:
            primary = CASE_BRANCH
            next_step = "diagnose_acts_navigation_material_branch_on_86"
        elif n_cont > 0 and CASE_CONTINUATION in focus_mechs:
            primary = CASE_CONTINUATION
            next_step = "diagnose_supporting_plane_continuation_state_on_86"
        elif CASE_FD_STEP in focus_mechs and n_both >= 0:
            primary = CASE_FD_STEP
            next_step = "keep_fixed_ladder_do_not_pick_best_step"
        elif CASE_TRANSPORT in focus_mechs or n_both > 0:
            primary = CASE_TRANSPORT
            next_step = "keep_source_to_measurement_map_diagnosis"
        else:
            primary = CASE_MIXED
            next_step = "keep_86_jacobian_continuity_diagnosis"
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "b14m_reopen_authorized": primary == CASE_ESTABLISHED,
        "restart_invariance_authorized": primary == CASE_ESTABLISHED,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "full_sample_authorized": False,
        "prior_introduced": False,
        "ridge_added": False,
        "do_not_force_5d_lto_covariance": True,
        "statistical_model_unchanged": True,
        "profile_math_rewritten": False,
        "focus_identity_retained": True,
        "physical_nonidentifiability_not_claimed": True,
        "five_d_cin_not_the_objective": True,
        "do_not_select_best_step": True,
        "do_not_switch_to_direct": True,
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
    rows = dumps["rows"]
    evaluations = audit_nominal_evaluations(rows)
    jacobian = audit_jacobian_continuity(rows)
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "n_rows": dumps["n_rows"],
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "jacobian": jacobian,
        "sequential_vs_direct": audit_sequential_vs_direct(jacobian),
        "evaluations": evaluations,
        "exclusion": audit_exclusion(rows) if rows else {"target_exclusion_holds": True},
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "switched_to_direct": False,
        "selected_best_step": False,
        "restart_executed": False,
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
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    if mechanism["b15_authorized"]:
        refuse_b15()
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    if mechanism["prior_introduced"]:
        refuse_prior()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb116_decision_sha256": inherited["workbook_116"]["decision_sha256"],
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb113_decision_sha256": inherited["workbook_113"]["decision_sha256"],
        "inherited_wb112_decision_sha256": inherited["workbook_112"]["decision_sha256"],
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb116_rewritten": False,
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "jacobian_contract_established": bool(
            (inventory.get("jacobian") or {}).get("jacobian_contract_established")
        ),
        "acts_jacobian_continuity_materialized": bool(inventory.get("smoke_present")),
    }


def likelihood_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "r_i": "m_loc0 - predicted_loc0_on_supporting_plane",
        "R_i": "(0.08 mm)^2 / 12",
        "theta": list(PARAM_NAMES),
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "official_h_i": "sequential supporting-plane transport, no measurement update",
        "direct_from_source": "comparison only",
        "do_not_switch_to_direct": True,
        "do_not_select_best_fd_step": True,
        "measurement_update_in_evaluator": False,
        "objective": "certify local continuity and Jacobian of h_i(theta) near 86",
        "not_the_objective": "repair 5D Cin",
    }
