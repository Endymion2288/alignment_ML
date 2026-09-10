"""Task B14ZC: certified-mean common-grid independent derivative contract.

WB127 closed the production surface energy-loss mean
(computeEnergyLossBethe + evaluateMaterialSlab gating + updateState).
This task reuses that certified shadow mean function and builds a
common-grid independent FD on the frozen nominal accepted-step
partition.  Mean semantics are not changed.  WB124 DOPRI5 is not a
reference.
"""

from __future__ import annotations

import json
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
    _event_label,
    _event_pair,
)
from datasets.field_gradient_variational_repair import (
    FOCUS_FIRST_UNSTABLE,
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
from datasets.surface_energy_loss_mean_semantics import (
    CASE_ESTABLISHED as WB127_DECISION,
    FROZEN_FIELD_GRADIENT_SHA,
    PRODUCTION_ELOSS,
    inherit_frozen_stage as inherit_through_wb127,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "certified-mean-common-grid-fd-v1"
DEFAULT_CONFIG = "configs/certified_mean_common_grid_fd_v1.yaml"
TASK = "SB-B14ZC"
WORKBOOK = 128

CASE_ESTABLISHED = "certified_mean_independent_reference_established"
CASE_FIELD = "field_map_derivative_reference_nonsmooth"
CASE_MATERIAL = "material_update_derivative_discontinuity"
CASE_PLANE = "supporting_plane_event_derivative_sensitivity"
CASE_NUMERICAL = "numerical_segment_fd_not_contracted"
CASE_INCONSISTENT = "repaired_variational_focus_inconsistent"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_ESTABLISHED,
    CASE_FIELD,
    CASE_MATERIAL,
    CASE_PLANE,
    CASE_NUMERICAL,
    CASE_INCONSISTENT,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
FOCUS_PARAMS = ("loc1", "phi", "q_over_p")
RECORDED_NOT_SUBSTITUTE = ("loc0", "theta")


class CertifiedMeanCommonGridFdError(ValueError):
    """Raised when the B14ZC certified-mean FD contract is illegal."""


def refuse_prior() -> None:
    raise CertifiedMeanCommonGridFdError("B14ZC must not introduce a prior")


def refuse_ridge_information() -> None:
    raise CertifiedMeanCommonGridFdError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise CertifiedMeanCommonGridFdError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_tune_grid() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not tune the common grid from Jacobian agreement"
    )


def refuse_tune_dopri5() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not retune or reuse the frozen WB124 DOPRI5"
    )


def refuse_change_repair() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not change the frozen field-gradient variational implementation"
    )


def refuse_change_mean() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not change the certified WB127 mean semantics"
    )


def refuse_b14m() -> None:
    raise CertifiedMeanCommonGridFdError("B14M is not re-opened inside Task B14ZC")


def refuse_b15() -> None:
    raise CertifiedMeanCommonGridFdError("Task B15 is not entered in Task B14ZC")


def refuse_full_sample() -> None:
    raise CertifiedMeanCommonGridFdError(
        "the 1989-row campaign must not be submitted in B14ZC"
    )


def refuse_substitute_hit6() -> None:
    raise CertifiedMeanCommonGridFdError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise CertifiedMeanCommonGridFdError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CertifiedMeanCommonGridFdError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise CertifiedMeanCommonGridFdError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise CertifiedMeanCommonGridFdError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14zb",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise CertifiedMeanCommonGridFdError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14zc", True)):
        raise CertifiedMeanCommonGridFdError("do_not_enter_b14zc must be false")
    spec = config["certified_mean_common_grid_fd"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise CertifiedMeanCommonGridFdError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise CertifiedMeanCommonGridFdError(
            "production stepTolerance must stay 1e-4"
        )
    if spec.get("production_eloss_quantity") != PRODUCTION_ELOSS:
        raise CertifiedMeanCommonGridFdError(
            "production Eloss quantity must stay computeEnergyLossBethe"
        )
    if {int(k): int(v) for k, v in spec["required_first_unstable"].items()} != (
        FOCUS_FIRST_UNSTABLE
    ):
        raise CertifiedMeanCommonGridFdError(
            "required first-unstable hops must stay 1/6, 2/11, 3/11"
        )
    if list(spec["required_independent_columns"]) != list(FOCUS_PARAMS):
        raise CertifiedMeanCommonGridFdError(
            "required independent columns must stay loc1/phi/q_over_p"
        )
    for flag in (
        "do_not_change_mean_semantics",
        "do_not_add_fd_rung",
        "do_not_shrink_fd_step",
        "do_not_tune_grid_from_jacobian",
        "do_not_reuse_wb124_adaptive_dopri5",
        "do_not_reuse_wb125_20_10_5",
        "do_not_submit_1989",
    ):
        if not bool(spec.get(flag, False)):
            raise CertifiedMeanCommonGridFdError(f"{flag} must be true")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise CertifiedMeanCommonGridFdError(
            "allowed decisions must stay the B14ZC set"
        )
    _expect_sha(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        ),
        FROZEN_FIELD_GRADIENT_SHA,
        "FieldGradientDefaultExtension",
    )
    for item in config.get("wb127_smoke_dumps") or []:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb127 smoke {item['path']}",
        )
    for item in config["wb123_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb123 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb127(config)
    spec = config["inheritance"]["workbook_127"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_127 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_127 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise CertifiedMeanCommonGridFdError("workbook_127 decision must stay frozen")
    if spec["frozen_decision"] != WB127_DECISION:
        raise CertifiedMeanCommonGridFdError("WB127 decision token mismatch")
    if not decision.get("shadow_mean_contract_established"):
        raise CertifiedMeanCommonGridFdError(
            "WB127 must have established the shadow mean contract"
        )
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise CertifiedMeanCommonGridFdError("WB127 must not have re-opened B14M")
    inherited["workbook_127"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "shadow_mean_contract_established": True,
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "five_percent_gate_unchanged": True,
        "control_0_pass": True,
        "control_1_pass": True,
        "control_37_pass": True,
        "mean_path_unchanged": True,
        "focus_independent_reference_established": False,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "do_not_change_mean_semantics": True,
        "do_not_change_field_gradient_variational_implementation": True,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    wb123_root = resolve_under_root(project_root(), str(config["wb123_smoke_root"]))
    ctrl_name = "ckf_leave_target_out_official_jacobian.jsonl"
    for path in sorted(wb123_root.rglob(ctrl_name)):
        if "100043" in str(path):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    z_root = resolve_under_root(
        project_root(), str(config["official_jacobian_smoke_root"])
    )
    name = str(config.get("official_jacobian_smoke_filename"))
    if z_root.is_dir():
        for path in sorted(z_root.rglob(name)):
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
        if col.get("last_pair_rel") is not None and col.get("signed_h8") is not None:
            return {
                "present": True,
                "converged": bool(col.get("converged")),
                "sign_change": not bool(col.get("sign_consistent", True)),
                "sign_consistent": bool(col.get("sign_consistent")),
                "last_pair_rel": col.get("last_pair_rel"),
                "rung_values": [
                    col.get("signed_h"),
                    col.get("signed_h2"),
                    col.get("signed_h4"),
                    col.get("signed_h8"),
                ],
                "signed_h": col.get("signed_h"),
                "signed_finest": col.get("signed_h8"),
            }
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
            "sign_consistent": not sign_change,
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


def _hop_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        row.get("certified_mean_common_grid_fd")
        or row.get("shadow_mean_transport_contract")
        or row.get("surface_energy_loss_mean_semantics")
        or {}
    )


def _required_hop(payload: Mapping[str, Any], required_idx: int | None):
    for hop in payload.get("hops") or []:
        if hop.get("measurement_index") == required_idx:
            return hop
    return None


def _repaired_tangent(row: Mapping[str, Any], required_idx: int | None):
    chain = (row.get("official_supporting_plane_jacobian") or {}).get(
        "official_path_jacobian"
    ) or {}
    for hop in chain.get("hops") or []:
        if hop.get("measurement_index") != required_idx:
            continue
        repair = hop.get("field_gradient_repair") or {}
        vals = repair.get("hop_dloc0_d_start")
        if isinstance(vals, list) and len(vals) >= 5:
            return {
                name: float(vals[i])
                for i, name in enumerate(PARAM_NAMES)
                if i < len(vals)
            }
    return None


def _classify_segment(seg: Mapping[str, Any]) -> str:
    fd = seg.get("certified_fd") or {}
    arms_ok = bool(fd.get("all_arms_same_mean_contract"))
    branch_ok = bool(fd.get("all_branch_identity_same"))
    material_ok = bool(fd.get("all_material_node_identity_same"))
    proj_fail = bool(fd.get("any_projection_fail"))
    n_dot = fd.get("n_dot_direction")
    plane = proj_fail or (n_dot is not None and abs(float(n_dot)) < 1.0e-3)
    c_ok = bool(seg.get("c_required_converged"))
    a_ok = bool(seg.get("repaired_agrees_independent"))
    if not arms_ok or not branch_ok:
        if plane:
            return CASE_PLANE
        if not material_ok:
            return CASE_MATERIAL
        return CASE_FIELD
    if not c_ok:
        if plane:
            return CASE_PLANE
        if any(
            bool((_param_block(seg, name).get("shadow_fd") or {}).get("sign_change"))
            for name in FOCUS_PARAMS
        ) and int(seg.get("n_material") or 0) > 0:
            return CASE_MATERIAL
        if any(
            not bool((_param_block(seg, name).get("shadow_fd") or {}).get("present"))
            for name in FOCUS_PARAMS
        ):
            return CASE_NUMERICAL
        if any(
            (_param_block(seg, name).get("shadow_fd") or {}).get("last_pair_rel")
            is None
            for name in FOCUS_PARAMS
        ):
            return CASE_NUMERICAL
        return CASE_FIELD
    if c_ok and not a_ok:
        return CASE_INCONSISTENT
    if c_ok and a_ok:
        return CASE_ESTABLISHED
    return CASE_MIXED


def _param_block(seg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for item in seg.get("parameters") or []:
        if item.get("parameter") == name:
            return item
    return {}


def audit_required_segment(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _hop_payload(row)
    target = int(row.get("target_station", -1))
    required_idx = FOCUS_FIRST_UNSTABLE.get(target)
    required = _required_hop(payload, required_idx)
    if required is None:
        return {
            "event": _event_label(_event_pair(row)),
            "target_station": target,
            "required_index": required_idx,
            "present": False,
            "independent_reference_established": False,
            "mean_contract_closed": False,
            "c_required_converged": False,
            "taxonomy": CASE_MIXED,
        }
    residual = required.get("shadow_p_residual") or {}
    mean_closed = bool(residual.get("closed"))
    fd = required.get("certified_mean_common_grid_fd") or {}
    shadow_fd = fd.get("shadow_fd") or {}
    repaired = _repaired_tangent(row, required_idx)
    parameters = []
    c_required = True
    any_inconsistent = False
    for name in PARAM_NAMES:
        c_col = _fd_columns(shadow_fd, name)
        rep = None if not repaired else repaired.get(name)
        established = bool(
            mean_closed
            and fd.get("all_arms_same_mean_contract")
            and fd.get("all_branch_identity_same")
            and c_col.get("converged")
        )
        agrees = False
        if established and rep is not None:
            agrees = _agree(rep, c_col.get("signed_finest"))
            if name in FOCUS_PARAMS and not agrees:
                any_inconsistent = True
        if name in FOCUS_PARAMS:
            c_required = c_required and established
        parameters.append(
            {
                "parameter": name,
                "shadow_fd": c_col,
                "production_repaired": rep,
                "independent_established": established,
                "repaired_agrees_shadow": agrees,
                "rel_repaired_vs_shadow": _rel_scalar(rep, c_col.get("signed_finest")),
                "required_for_reference": name in FOCUS_PARAMS,
                "recorded_not_substitute": name in RECORDED_NOT_SUBSTITUTE,
            }
        )
    focus_agrees = all(
        next(item for item in parameters if item["parameter"] == name)[
            "repaired_agrees_shadow"
        ]
        for name in FOCUS_PARAMS
    )
    established = bool(mean_closed and c_required and focus_agrees)
    out = {
        "event": _event_label(_event_pair(row)),
        "target_station": target,
        "required_index": required_idx,
        "measurement_index": required.get("measurement_index"),
        "present": True,
        "hop_start_state_present": "hop_start_state" in required,
        "mean_contract_closed": mean_closed,
        "shadow_p_residual": residual,
        "certified_fd": fd,
        "n_material": fd.get("nominal_n_material"),
        "parameters": parameters,
        "c_required_converged": c_required,
        "independent_reference_established": established,
        "repaired_agrees_independent": focus_agrees,
        "repaired_inconsistent": any_inconsistent and c_required,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_reuse_wb124_adaptive_dopri5": True,
        "do_not_change_mean_semantics": True,
        "certification_order": fd.get("certification_order"),
        "production_eloss_quantity": required.get(
            "production_eloss_quantity", PRODUCTION_ELOSS
        ),
    }
    out["taxonomy"] = _classify_segment(out)
    return out


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wb123_hash_match": bool(inventory.get("wb123_hash_match")),
        "wb127_hash_match": bool(inventory.get("wb127_hash_match")),
        "controls_present": bool(inventory.get("controls")),
        "focus_86_present": bool(inventory.get("focus_segments")),
        "required_hops_are_6_11_11": bool(inventory.get("required_hops_correct")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "five_percent_gate_unchanged": True,
        "mean_semantics_unchanged": True,
        "target_exclusion": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds", True)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_as_information"):
        refuse_ridge_information()
    if inventory.get("added_fd_rung"):
        refuse_add_fd_rung()
    if inventory.get("relaxed_five_percent_gate"):
        refuse_relax_gate()
    if inventory.get("tuned_grid"):
        refuse_tune_grid()
    if inventory.get("tuned_dopri5"):
        refuse_tune_dopri5()
    if inventory.get("changed_field_gradient_repair"):
        refuse_change_repair()
    if inventory.get("changed_mean_semantics"):
        refuse_change_mean()
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
    taxes = [item.get("taxonomy") for item in segs]
    unique = {item for item in taxes if item}
    any_inconsistent = any(item.get("repaired_inconsistent") for item in segs)
    if not segs:
        primary = CASE_MIXED
        next_step = "keep_certified_mean_common_grid_fd"
    elif any_inconsistent and not focus_ok:
        primary = CASE_INCONSISTENT
        next_step = "keep_field_gradient_tangent_implementation"
    elif focus_ok and unique == {CASE_ESTABLISHED}:
        if (not c0_pass) or (not c1_pass) or (not c37_pass) or (not mean_ok):
            primary = CASE_MIXED
            next_step = "keep_certified_mean_common_grid_fd"
        else:
            primary = CASE_ESTABLISHED
            next_step = "reopen_b14m_smoke_and_restart_invariance"
    elif len(unique) == 1:
        primary = next(iter(unique))
        next_step = "keep_certified_mean_common_grid_fd"
    else:
        primary = CASE_MIXED
        next_step = "keep_certified_mean_common_grid_fd"
    contract = (
        primary == CASE_ESTABLISHED
        and focus_ok
        and c0_pass
        and c1_pass
        and c37_pass
        and mean_ok
    )
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "control_0_pass": c0_pass,
        "control_1_pass": c1_pass,
        "control_37_pass": c37_pass,
        "mean_path_unchanged": mean_ok,
        "shadow_mean_contract_established": True,
        "focus_independent_reference_established": bool(focus_ok and contract),
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
        "do_not_change_mean_semantics": True,
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_reuse_wb124_adaptive_dopri5": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_submit_1989": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
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
        if pair != FOCUS_EVENT:
            if row.get("official_supporting_plane_jacobian") is None:
                continue
            invariance.append(audit_invariance(row))
            controls.append(audit_control_columns(row))
            continue
        if row.get("certified_mean_common_grid_fd") is None and row.get(
            "shadow_mean_transport_contract"
        ) is None:
            continue
        if row.get("official_supporting_plane_jacobian") is not None:
            invariance.append(audit_invariance(row))
        audited = audit_required_segment(row)
        focus_seg.append(audited)
        if audited.get("required_index") != FOCUS_FIRST_UNSTABLE.get(
            int(row.get("target_station", -1))
        ):
            required_ok = False
        if audited.get("measurement_index") == 6 and int(
            row.get("target_station", -1)
        ) in (2, 3):
            substituted = True
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "wb123_hash_match": True,
        "wb127_hash_match": True,
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
        "tuned_grid": False,
        "tuned_dopri5": False,
        "changed_field_gradient_repair": False,
        "changed_mean_semantics": False,
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
        mechanism["focus_independent_reference_established"] = False
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb127_decision_sha256": inherited["workbook_127"][
            "decision_sha256"
        ],
        "inherited_wb126_decision_sha256": inherited["workbook_126"][
            "decision_sha256"
        ],
        "inherited_wb125_decision_sha256": inherited["workbook_125"][
            "decision_sha256"
        ],
        "inherited_wb123_decision_sha256": inherited["workbook_123"][
            "decision_sha256"
        ],
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "measurement_sigma_mm": MEASUREMENT_SIGMA_MM,
        "segment_taxonomy": [
            {
                "target_station": item.get("target_station"),
                "required_index": item.get("required_index"),
                "taxonomy": item.get("taxonomy"),
                "mean_contract_closed": item.get("mean_contract_closed"),
                "c_required_converged": item.get("c_required_converged"),
                "independent_reference_established": item.get(
                    "independent_reference_established"
                ),
                "repaired_agrees_independent": item.get(
                    "repaired_agrees_independent"
                ),
            }
            for item in inventory.get("focus_segments") or []
        ],
    }


def likelihood_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "R_i": "(0.08 mm)^2 / 12",
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "do_not_change_official_mean_path": True,
        "do_not_change_mean_semantics": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_reuse_wb124_adaptive_dopri5": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_submit_1989": True,
        "pinv_relative": PINV_RELATIVE,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "objective": "certified-mean common-grid independent segment FD",
        "not_the_objective": "repair 5D Cin or submit 1989",
    }
