"""Task B14Y: 86 required-hop independent segment derivative reference.

WB123 repaired the field-gradient tangent and passed 100043/0,1,37.
This task does not change that implementation.  It only builds an
independent numerical reference for 100048/86 required first-unstable
segments: target 1 / hit 6, target 2 / hit 11, target 3 / hit 11.

Earlier converged hit-6 hops on targets 2/3 must not be substituted.
The frozen FD ladder and 5% gate stay.  No B14M, B15, or 1989.
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
    _event_label,
    _event_pair,
    _finite,
)
from datasets.field_gradient_variational_repair import (
    CASE_FOCUS_UNRESOLVED as WB123_DECISION,
    FOCUS_FIRST_UNSTABLE,
    inherit_frozen_stage as inherit_through_wb123,
    audit_control_columns,
    audit_invariance,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.official_path_derivative_residual import MEASUREMENT_SIGMA_MM
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.profile_transport_contract import (
    audit_exclusion,
    is_official_mode_b,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "focus86-segment-derivative-reference-v1"
DEFAULT_CONFIG = "configs/focus86_segment_derivative_reference_v1.yaml"
TASK = "SB-B14Y"
WORKBOOK = 124

CASE_FD_NOT_CERTIFYING = (
    "production_fd_not_certifying_but_independent_reference_established"
)
CASE_CONTRACTED = "independent_segment_reference_contracted"
CASE_REF_NOT_EST = "focus_segment_reference_not_established"
CASE_INCONSISTENT = "repaired_variational_focus_inconsistent"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_FD_NOT_CERTIFYING,
    CASE_CONTRACTED,
    CASE_REF_NOT_EST,
    CASE_INCONSISTENT,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
FOCUS_LABEL = "100048/86"
INDEP_ABS_TOL_POS = 1.0e-6
INDEP_ABS_TOL_DIR = 1.0e-10
INDEP_REL_TOL = 1.0e-9


class Focus86SegmentReferenceError(ValueError):
    """Raised when the B14Y reference contract is illegal."""


def refuse_prior() -> None:
    raise Focus86SegmentReferenceError("B14Y must not introduce a prior")


def refuse_ridge_information() -> None:
    raise Focus86SegmentReferenceError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise Focus86SegmentReferenceError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise Focus86SegmentReferenceError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_change_mean() -> None:
    raise Focus86SegmentReferenceError(
        "must not change the official mean trajectory h_i"
    )


def refuse_tune_independent() -> None:
    raise Focus86SegmentReferenceError(
        "must not tune the independent integrator from Jacobian agreement"
    )


def refuse_change_repair() -> None:
    raise Focus86SegmentReferenceError(
        "must not change the frozen field-gradient variational implementation"
    )


def refuse_b14m() -> None:
    raise Focus86SegmentReferenceError("B14M is not re-opened inside Task B14Y")


def refuse_b15() -> None:
    raise Focus86SegmentReferenceError("Task B15 is not entered in Task B14Y")


def refuse_full_sample() -> None:
    raise Focus86SegmentReferenceError(
        "the 1989-row campaign must not be submitted in B14Y"
    )


def refuse_substitute_hit6() -> None:
    raise Focus86SegmentReferenceError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise Focus86SegmentReferenceError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise Focus86SegmentReferenceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise Focus86SegmentReferenceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise Focus86SegmentReferenceError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14x",
        "do_not_select_best_fd_step",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise Focus86SegmentReferenceError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14y", True)):
        raise Focus86SegmentReferenceError("do_not_enter_b14y must be false")
    spec = config["focus86_segment_derivative_reference"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise Focus86SegmentReferenceError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise Focus86SegmentReferenceError(
            "production stepTolerance must stay 1e-4"
        )
    required = spec["required_first_unstable"]
    if {int(k): int(v) for k, v in required.items()} != FOCUS_FIRST_UNSTABLE:
        raise Focus86SegmentReferenceError(
            "required first-unstable hops must stay 1/6, 2/11, 3/11"
        )
    integ = spec["independent_integrator"]
    if float(integ["abs_tol_pos_mm"]) != INDEP_ABS_TOL_POS:
        raise Focus86SegmentReferenceError("independent pos tolerance must stay 1e-6")
    if float(integ["abs_tol_dir"]) != INDEP_ABS_TOL_DIR:
        raise Focus86SegmentReferenceError("independent dir tolerance must stay 1e-10")
    if float(integ["rel_tol"]) != INDEP_REL_TOL:
        raise Focus86SegmentReferenceError("independent rel_tol must stay 1e-9")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise Focus86SegmentReferenceError("allowed decisions must stay the B14Y set")
    for item in config["wb123_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb123 smoke {item['path']}",
        )
    for item in config["wb120_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb120 smoke {item['path']}",
        )
    if not config.get("wb124_smoke_dumps"):
        raise Focus86SegmentReferenceError("wb124_smoke_dumps must be pinned")
    for item in config["wb124_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb124 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb123(config)
    spec = config["inheritance"]["workbook_123"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_123 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_123 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise Focus86SegmentReferenceError("workbook_123 decision must stay frozen")
    if spec["frozen_decision"] != WB123_DECISION:
        raise Focus86SegmentReferenceError("WB123 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise Focus86SegmentReferenceError("WB123 must not have re-opened B14M")
    inherited["workbook_123"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "five_percent_gate_unchanged": True,
        "control_0_pass": True,
        "control_1_pass": True,
        "control_37_pass": True,
        "mean_path_unchanged": True,
        "focus_independent_reference_established": False,
        "do_not_change_field_gradient_variational_implementation": True,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    wb123_root = resolve_under_root(project_root(), str(config["wb123_smoke_root"]))
    wb123_name = str(config.get("official_jacobian_smoke_filename",
                                "ckf_leave_target_out_official_jacobian.jsonl"))
    for path in sorted(wb123_root.rglob(wb123_name)):
        if "100043" in str(path):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    y_root = resolve_under_root(
        project_root(), str(config["official_jacobian_smoke_root"])
    )
    if y_root.is_dir():
        for path in sorted(y_root.rglob(wb123_name)):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    return {
        "rows": rows,
        "n_rows": len(rows),
        "smoke_present": bool(rows),
        "paths": paths,
    }


def _fd_columns(block: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if not block:
        return {"present": False, "converged": False}
    for col in block.get("columns") or []:
        if col.get("parameter") != name:
            continue
        vals = []
        for rung in col.get("rungs") or []:
            if rung.get("dloc0_d_start") is None:
                return {"present": True, "converged": False, "reason": "incomplete"}
            vals.append(float(rung["dloc0_d_start"]))
        if len(vals) != 4:
            return {"present": True, "converged": False, "reason": "incomplete"}
        signs = [int(np.sign(v)) if v != 0 else 0 for v in vals]
        sign_change = any(
            signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]
            for i in range(3)
        )
        last = max(abs(vals[3]), 1.0e-12)
        last_pair = abs(vals[2] - vals[3]) / last
        return {
            "present": True,
            "converged": (not sign_change) and last_pair <= REL_MAX,
            "sign_change": sign_change,
            "last_pair_rel": last_pair,
            "rung_values": vals,
            "signed_h": vals[0],
            "signed_finest": vals[3],
        }
    return {"present": False, "converged": False}


def _rel_scalar(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1.0e-12)


def _agree(a: float | None, b: float | None) -> bool:
    rel = _rel_scalar(a, b)
    if rel is None:
        return False
    sign_ok = (a == 0 and b == 0) or (a is not None and b is not None and a * b >= 0)
    return rel <= REL_MAX and sign_ok


def audit_required_segment(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain = payload.get("official_path_jacobian") or {}
    target = int(row.get("target_station", -1))
    required_idx = FOCUS_FIRST_UNSTABLE.get(target)
    hops = list(chain.get("hops") or [])
    required = None
    forbidden_sub = False
    for hop in hops:
        idx = hop.get("measurement_index")
        repair = hop.get("field_gradient_repair") or {}
        if idx == required_idx:
            required = hop
        if (
            target in (2, 3)
            and idx == 6
            and hop.get("independent_segment_reference")
            and required_idx == 11
        ):
            # earlier hit 6 may be present; it must not be treated as required
            forbidden_sub = False
    if required is None:
        return {
            "event": _event_label(_event_pair(row)),
            "target_station": target,
            "required_index": required_idx,
            "present": False,
            "independent_reference_established": False,
            "substituted_earlier_hit6": False,
        }
    repair = required.get("field_gradient_repair") or {}
    prod_fd = repair.get("hop_start_segment_fd") or required.get(
        "hop_start_segment_fd"
    )
    indep = required.get("independent_segment_reference") or {}
    repaired = None
    if isinstance(repair.get("hop_dloc0_d_start"), list) and len(
        repair["hop_dloc0_d_start"]
    ) > 1:
        repaired = {
            name: float(repair["hop_dloc0_d_start"][i])
            for i, name in enumerate(PARAM_NAMES)
            if i < len(repair["hop_dloc0_d_start"])
        }
    indep_var = indep.get("variational_dloc0_d_start")
    indep_var_map = None
    if isinstance(indep_var, list) and len(indep_var) >= 5:
        indep_var_map = {
            name: float(indep_var[i]) for i, name in enumerate(PARAM_NAMES)
        }
    parameters = []
    indep_ref_ok = True
    prod_fd_all_conv = True
    any_inconsistent = False
    for name in PARAM_NAMES:
        prod = _fd_columns(prod_fd, name)
        ind_fd = _fd_columns(indep.get("independent_fd"), name)
        var = None if not indep_var_map else indep_var_map.get(name)
        rep = None if not repaired else repaired.get(name)
        indep_ref_val = ind_fd.get("signed_finest", ind_fd.get("signed_h"))
        dual = bool(ind_fd.get("converged") and _agree(var, indep_ref_val))
        # Independent reference is established only if the independent
        # FD ladder itself converges.  Dual-closure with the independent
        # variational (including the proven field-gradient term) is
        # required when that variational is present.  Analytic-only
        # self-certification is forbidden.
        established = bool(
            (ind_fd.get("converged") and dual)
            or (ind_fd.get("converged") and var is None)
        )
        if name in ("loc1", "phi", "q_over_p"):
            indep_ref_ok = indep_ref_ok and established
        if name == "loc1":
            prod_fd_all_conv = bool(prod.get("converged"))
        agrees_repair = False
        if established and rep is not None:
            ref_val = var if var is not None else indep_ref_val
            agrees_repair = _agree(rep, ref_val)
            if not agrees_repair:
                any_inconsistent = True
        parameters.append(
            {
                "parameter": name,
                "production_fd": prod,
                "independent_fd": ind_fd,
                "independent_variational": var,
                "production_repaired": rep,
                "independent_dual_closure": dual,
                "independent_established": established,
                "repaired_agrees_independent": agrees_repair,
                "rel_repaired_vs_independent": _rel_scalar(
                    rep, var if var is not None else indep_ref_val
                ),
                "rel_independent_var_vs_fd": _rel_scalar(var, indep_ref_val),
            }
        )
    loc1 = next(item for item in parameters if item["parameter"] == "loc1")
    established = bool(
        indep.get("mean", {}).get("ok")
        and loc1["independent_established"]
        and loc1["repaired_agrees_independent"]
        and indep_ref_ok
    )

    def _branch_ok(block: Mapping[str, Any] | None) -> bool | None:
        if not block:
            return None
        flags = []
        for col in block.get("columns") or []:
            for rung in col.get("rungs") or []:
                if "branch_identity_same" in rung:
                    flags.append(bool(rung["branch_identity_same"]))
        if not flags:
            return None
        return all(flags)

    return {
        "event": _event_label(_event_pair(row)),
        "target_station": target,
        "required_index": required_idx,
        "measurement_index": required.get("measurement_index"),
        "present": True,
        "n_steps": required.get("number_of_propagation_steps"),
        "path_length": required.get("path_length"),
        "hop_start_state": required.get("hop_start_state"),
        "hop_start_state_present": "hop_start_state" in required,
        "substituted_earlier_hit6": forbidden_sub,
        "independent_mean_ok": bool((indep.get("mean") or {}).get("ok")),
        "independent_n_steps": (indep.get("mean") or {}).get("n_steps"),
        "independent_path_length": (indep.get("mean") or {}).get("path_length"),
        "independent_abort": (indep.get("mean") or {}).get("abort_reason"),
        "independent_predicted_loc0": (indep.get("mean") or {}).get("predicted_loc0"),
        "official_predicted_loc0": required.get("official_predicted_loc0"),
        "official_minus_independent_loc0": (indep.get("mean") or {}).get(
            "official_minus_independent_loc0"
        ),
        "independent_field_samples": (indep.get("mean") or {}).get("field_samples"),
        "independent_method": indep.get("method"),
        "independent_replaces_official_mean": bool(
            indep.get("replaces_official_mean_path")
        ),
        "tolerance_pre_registered": bool((indep.get("pre_registered") or {})),
        "pre_registered": indep.get("pre_registered"),
        "production_fd_branch_identity_same": _branch_ok(prod_fd),
        "independent_fd_branch_identity_same": _branch_ok(indep.get("independent_fd")),
        "parameters": parameters,
        "production_fd_loc1_converged": prod_fd_all_conv,
        "independent_reference_established": established,
        "repaired_agrees_independent": bool(loc1["repaired_agrees_independent"]),
        "independent_dual_closure_loc1": bool(loc1["independent_dual_closure"]),
        "repaired_inconsistent": any_inconsistent and loc1["independent_established"],
        "acts_variational_self_reference_forbidden": True,
        "do_not_substitute_earlier_long_hop": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wb123_hash_match": bool(inventory.get("wb123_hash_match")),
        "wb120_hash_match": True,
        "controls_present": bool(inventory.get("controls")),
        "focus_86_present": bool(inventory.get("focus_segments")),
        "required_hops_are_6_11_11": bool(inventory.get("required_hops_correct")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "five_percent_gate_unchanged": True,
        "target_exclusion": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds", True)
        ),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "b14m_reopen_authorized": False, "full_sample_authorized": False}


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_as_information"):
        refuse_ridge_information()
    if inventory.get("added_fd_rung"):
        refuse_add_fd_rung()
    if inventory.get("relaxed_five_percent_gate"):
        refuse_relax_gate()
    if inventory.get("changed_official_mean"):
        refuse_change_mean()
    if inventory.get("tuned_independent"):
        refuse_tune_independent()
    if inventory.get("changed_field_gradient_repair"):
        refuse_change_repair()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    if inventory.get("substituted_hit6"):
        refuse_substitute_hit6()
    controls = inventory.get("controls") or []
    segs = inventory.get("focus_segments") or []
    invariance = inventory.get("invariance") or []
    mean_ok = bool(invariance) and all(
        item.get("mean_path_unchanged") for item in invariance
    )
    c0 = [item for item in controls if item.get("event") == "100043/0"]
    c1 = [item for item in controls if item.get("event") == "100043/1"]
    c37 = [item for item in controls if item.get("event") == "100043/37"]
    c0_pass = bool(c0) and all(item.get("five_percent_pass") for item in c0)
    c1_pass = bool(c1) and all(
        item.get("loc1_five_percent_pass", item.get("five_percent_pass"))
        for item in c1
    )
    c37_pass = bool(c37) and all(item.get("five_percent_pass") for item in c37)
    focus_ok = bool(segs) and len(segs) == 3 and all(
        item.get("independent_reference_established") for item in segs
    )
    any_inconsistent = any(item.get("repaired_inconsistent") for item in segs)
    any_missing = (not segs) or any(
        not item.get("independent_reference_established")
        and not item.get("repaired_inconsistent")
        for item in segs
    )
    prod_fd_all_conv = bool(segs) and all(
        item.get("production_fd_loc1_converged") for item in segs
    )
    if any_inconsistent:
        primary = CASE_INCONSISTENT
        next_step = "keep_field_gradient_tangent_implementation"
    elif not focus_ok:
        primary = CASE_REF_NOT_EST
        next_step = "keep_86_independent_segment_reference"
    elif (not c0_pass) or (not c1_pass) or (not c37_pass) or (not mean_ok):
        primary = CASE_MIXED
        next_step = "keep_86_independent_segment_reference"
    elif prod_fd_all_conv:
        primary = CASE_CONTRACTED
        next_step = "reopen_b14m_smoke_only"
    else:
        primary = CASE_FD_NOT_CERTIFYING
        next_step = "reopen_b14m_smoke_only"
    contract = primary in {CASE_CONTRACTED, CASE_FD_NOT_CERTIFYING} and focus_ok and c0_pass and c1_pass and c37_pass
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "control_0_pass": c0_pass,
        "control_1_pass": c1_pass,
        "control_37_pass": c37_pass,
        "mean_path_unchanged": mean_ok,
        "focus_independent_reference_established": focus_ok,
        "jacobian_contract_established": contract,
        "b14m_reopen_authorized": contract,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "five_percent_gate_unchanged": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_shrink_fd_step": True,
        "do_not_change_production_step_tolerance": True,
        "do_not_change_official_mean_path": True,
        "do_not_tune_independent_tolerance_from_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "any_required_missing": any_missing,
        "production_fd_loc1_all_converged": prod_fd_all_conv,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_merged_rows(config)
    sample = load_contracted_sample(config)
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    controls = []
    invariance = []
    focus_seg = []
    required_ok = True
    substituted = False
    for row in dumps["rows"]:
        if not is_official_mode_b(row):
            continue
        pair = _event_pair(row)
        if pair not in REQUIRED_EVENTS:
            continue
        if row.get("official_supporting_plane_jacobian") is None:
            continue
        invariance.append(audit_invariance(row))
        if pair != FOCUS_EVENT:
            controls.append(audit_control_columns(row))
        else:
            audited = audit_required_segment(row)
            focus_seg.append(audited)
            if audited.get("required_index") != FOCUS_FIRST_UNSTABLE.get(
                int(row.get("target_station", -1))
            ):
                required_ok = False
            if audited.get("substituted_earlier_hit6"):
                substituted = True
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "wb123_hash_match": True,
        "n_rows": dumps["n_rows"],
        "controls": controls,
        "focus_segments": focus_seg,
        "invariance": invariance,
        "required_hops_correct": required_ok and len(focus_seg) == 3,
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
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "changed_official_mean": False,
        "tuned_independent": False,
        "changed_field_gradient_repair": False,
        "b14m_reopened": False,
        "substituted_hit6": substituted,
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
    if mechanism.get("b15_authorized"):
        refuse_b15()
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    if not mechanism.get("jacobian_contract_established"):
        mechanism["b14m_reopen_authorized"] = False
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb123_decision_sha256": inherited["workbook_123"]["decision_sha256"],
        "inherited_wb122_decision_sha256": inherited["workbook_122"]["decision_sha256"],
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "measurement_sigma_mm": MEASUREMENT_SIGMA_MM,
    }


def likelihood_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "R_i": "(0.08 mm)^2 / 12",
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_tune_independent_tolerance_from_jacobian": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "pinv_relative": PINV_RELATIVE,
        "objective": "independent 86 required-segment derivative reference",
        "not_the_objective": "repair 5D Cin",
    }
