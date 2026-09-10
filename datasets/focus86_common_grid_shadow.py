"""Task B14Z: 86 common-grid shadow segment derivative reference.

WB124 adaptive DOPRI5 failed because ±δ used different accepted-step
sequences.  This task does not retune that integrator and does not
change the field-gradient repair.  It builds a frozen common-grid
shadow of the production mean map for 100048/86 required hops
1/6, 2/11, 3/11.
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
from datasets.focus86_segment_derivative_reference import (
    CASE_REF_NOT_EST as WB124_DECISION,
    inherit_frozen_stage as inherit_through_wb124,
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

SCHEMA_VERSION = "focus86-common-grid-shadow-v1"
DEFAULT_CONFIG = "configs/focus86_common_grid_shadow_v1.yaml"
TASK = "SB-B14Z"
WORKBOOK = 125

CASE_ESTABLISHED = "focus_common_grid_reference_established"
CASE_MEAN_NOT_EST = "shadow_mean_contract_not_established"
CASE_FIELD = "field_map_interpolation_nonsmoothness"
CASE_MATERIAL = "material_map_discontinuity"
CASE_PLANE = "supporting_plane_terminal_event_sensitivity"
CASE_RESOLVE = "shadow_integrator_resolution_unresolved"
CASE_INCONSISTENT = "repaired_variational_focus_inconsistent"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_ESTABLISHED,
    CASE_MEAN_NOT_EST,
    CASE_FIELD,
    CASE_MATERIAL,
    CASE_PLANE,
    CASE_RESOLVE,
    CASE_INCONSISTENT,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
SHADOW_NOM_STEP = 10.0
SHADOW_COARSE_STEP = 20.0
SHADOW_FINE_STEP = 5.0
MEAN_LOC0 = 1.0e-3
MEAN_PATH = 1.0e-2
GRID_DERIV = 0.05
FROZEN_FIELD_GRADIENT_SHA = (
    "ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8"
)
FOCUS_PARAMS = ("loc1", "phi", "q_over_p")


class Focus86CommonGridShadowError(ValueError):
    """Raised when the B14Z common-grid contract is illegal."""


def refuse_prior() -> None:
    raise Focus86CommonGridShadowError("B14Z must not introduce a prior")


def refuse_ridge_information() -> None:
    raise Focus86CommonGridShadowError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise Focus86CommonGridShadowError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise Focus86CommonGridShadowError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_tune_grid() -> None:
    raise Focus86CommonGridShadowError(
        "must not tune the common grid from Jacobian agreement"
    )


def refuse_tune_dopri5() -> None:
    raise Focus86CommonGridShadowError(
        "must not retune the frozen WB124 DOPRI5 tolerances"
    )


def refuse_change_repair() -> None:
    raise Focus86CommonGridShadowError(
        "must not change the frozen field-gradient variational implementation"
    )


def refuse_b14m() -> None:
    raise Focus86CommonGridShadowError("B14M is not re-opened inside Task B14Z")


def refuse_b15() -> None:
    raise Focus86CommonGridShadowError("Task B15 is not entered in Task B14Z")


def refuse_full_sample() -> None:
    raise Focus86CommonGridShadowError(
        "the 1989-row campaign must not be submitted in B14Z"
    )


def refuse_substitute_hit6() -> None:
    raise Focus86CommonGridShadowError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise Focus86CommonGridShadowError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise Focus86CommonGridShadowError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise Focus86CommonGridShadowError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise Focus86CommonGridShadowError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14x",
        "do_not_enter_b14y",
        "do_not_select_best_fd_step",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise Focus86CommonGridShadowError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14z", True)):
        raise Focus86CommonGridShadowError("do_not_enter_b14z must be false")
    spec = config["focus86_common_grid_shadow"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise Focus86CommonGridShadowError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise Focus86CommonGridShadowError(
            "production stepTolerance must stay 1e-4"
        )
    if {int(k): int(v) for k, v in spec["required_first_unstable"].items()} != (
        FOCUS_FIRST_UNSTABLE
    ):
        raise Focus86CommonGridShadowError(
            "required first-unstable hops must stay 1/6, 2/11, 3/11"
        )
    integ = spec["shadow_integrator"]
    if float(integ["nominal_max_step_mm"]) != SHADOW_NOM_STEP:
        raise Focus86CommonGridShadowError("nominal grid step must stay 10 mm")
    if float(integ["coarse_max_step_mm"]) != SHADOW_COARSE_STEP:
        raise Focus86CommonGridShadowError("coarse grid step must stay 20 mm")
    if float(integ["fine_max_step_mm"]) != SHADOW_FINE_STEP:
        raise Focus86CommonGridShadowError("fine grid step must stay 5 mm")
    if float(integ["mean_loc0_abs_mm"]) != MEAN_LOC0:
        raise Focus86CommonGridShadowError("mean loc0 gate must stay 1e-3 mm")
    if float(integ["mean_path_abs_mm"]) != MEAN_PATH:
        raise Focus86CommonGridShadowError("mean path gate must stay 1e-2 mm")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise Focus86CommonGridShadowError("allowed decisions must stay the B14Z set")
    _expect_sha(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        ),
        FROZEN_FIELD_GRADIENT_SHA,
        "FieldGradientDefaultExtension",
    )
    for item in config.get("wb125_smoke_dumps") or []:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb125 smoke {item['path']}",
        )
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
    for item in config["wb124_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb124 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb124(config)
    spec = config["inheritance"]["workbook_124"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_124 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_124 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise Focus86CommonGridShadowError("workbook_124 decision must stay frozen")
    if spec["frozen_decision"] != WB124_DECISION:
        raise Focus86CommonGridShadowError("WB124 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise Focus86CommonGridShadowError("WB124 must not have re-opened B14M")
    inherited["workbook_124"] = {
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
        "do_not_tune_independent_tolerance_from_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    wb123_root = resolve_under_root(project_root(), str(config["wb123_smoke_root"]))
    name = str(config.get("official_jacobian_smoke_filename",
                          "ckf_leave_target_out_official_jacobian.jsonl"))
    for path in sorted(wb123_root.rglob(name)):
        if "100043" in str(path):
            loaded = load_dump_records(path, split="train")
            rows.extend(loaded)
            paths.append(str(path))
    z_root = resolve_under_root(
        project_root(), str(config["official_jacobian_smoke_root"])
    )
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
    for hop in hops:
        if hop.get("measurement_index") == required_idx:
            required = hop
    if required is None:
        return {
            "event": _event_label(_event_pair(row)),
            "target_station": target,
            "required_index": required_idx,
            "present": False,
            "independent_reference_established": False,
            "mean_contract_closed": False,
        }
    repair = required.get("field_gradient_repair") or {}
    prod_fd = repair.get("hop_start_segment_fd") or required.get(
        "hop_start_segment_fd"
    )
    shadow = required.get("common_grid_shadow_reference") or {}
    repaired = None
    if isinstance(repair.get("hop_dloc0_d_start"), list) and len(
        repair["hop_dloc0_d_start"]
    ) > 1:
        repaired = {
            name: float(repair["hop_dloc0_d_start"][i])
            for i, name in enumerate(PARAM_NAMES)
            if i < len(repair["hop_dloc0_d_start"])
        }
    mean_contract = shadow.get("mean_contract") or {}
    mean_closed = bool(mean_contract.get("closed"))
    effects = shadow.get("production_mean_effects") or {}
    grid_mean = shadow.get("grid_mean_refinement") or {}
    parameters = []
    c_ok = True
    any_inconsistent = False
    prod_fd_loc1 = False
    for name in PARAM_NAMES:
        prod = _fd_columns(prod_fd, name)
        c_nom = _fd_columns(shadow.get("shadow_fd_nominal") or shadow.get("shadow_fd"), name)
        c_fine = _fd_columns(shadow.get("shadow_fd_fine"), name)
        c_coarse = _fd_columns(shadow.get("shadow_fd_coarse"), name)
        if name == "loc1":
            prod_fd_loc1 = bool(prod.get("converged"))
        grid_rel = _rel_scalar(c_nom.get("signed_finest"), c_fine.get("signed_finest"))
        grid_stable = bool(
            c_nom.get("converged")
            and c_fine.get("converged")
            and grid_rel is not None
            and grid_rel <= GRID_DERIV
            and _agree(c_nom.get("signed_finest"), c_fine.get("signed_finest"))
        )
        established = bool(mean_closed and c_nom.get("converged") and grid_stable)
        if name in FOCUS_PARAMS:
            c_ok = c_ok and established
        rep = None if not repaired else repaired.get(name)
        agrees = False
        if established and rep is not None:
            agrees = _agree(rep, c_nom.get("signed_finest"))
            if name in FOCUS_PARAMS and not agrees:
                any_inconsistent = True
        parameters.append(
            {
                "parameter": name,
                "production_fd": prod,
                "shadow_fd_nominal": c_nom,
                "shadow_fd_fine": c_fine,
                "shadow_fd_coarse": c_coarse,
                "production_repaired": rep,
                "grid_refinement_rel": grid_rel,
                "grid_stable": grid_stable,
                "independent_established": established,
                "repaired_agrees_shadow": agrees,
                "rel_repaired_vs_shadow": _rel_scalar(rep, c_nom.get("signed_finest")),
            }
        )
    loc1 = next(item for item in parameters if item["parameter"] == "loc1")
    focus_agrees = all(
        next(item for item in parameters if item["parameter"] == name)[
            "repaired_agrees_shadow"
        ]
        for name in FOCUS_PARAMS
    )
    established = bool(mean_closed and c_ok and focus_agrees)
    n_dot = (shadow.get("mean_nominal") or {}).get("n_dot_direction")
    n_mat = int(effects.get("n_surface_material_crossings") or 0)
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": target,
        "required_index": required_idx,
        "measurement_index": required.get("measurement_index"),
        "present": True,
        "n_steps": required.get("number_of_propagation_steps"),
        "hop_start_state_present": "hop_start_state" in required,
        "mean_contract_closed": mean_closed,
        "mean_contract": mean_contract,
        "grid_mean_stable": bool(grid_mean.get("stable")),
        "production_mean_effects": effects,
        "n_surface_material_crossings": n_mat,
        "n_dot_direction": n_dot,
        "shadow_method": shadow.get("method"),
        "tolerance_pre_registered": bool(shadow.get("pre_registered")),
        "pre_registered": shadow.get("pre_registered"),
        "meshes": shadow.get("meshes"),
        "mean_nominal": shadow.get("mean_nominal"),
        "mean_coarse": shadow.get("mean_coarse"),
        "mean_fine": shadow.get("mean_fine"),
        "grid_mean_refinement": grid_mean,
        "production_partition_path": shadow.get("production_partition_path"),
        "production_partition_n_steps": shadow.get("production_partition_n_steps"),
        "parameters": parameters,
        "production_fd_loc1_converged": prod_fd_loc1,
        "independent_reference_established": established,
        "repaired_agrees_independent": bool(loc1["repaired_agrees_shadow"]),
        "repaired_inconsistent": any_inconsistent and loc1["independent_established"],
        "c_track_fd_loc1_converged": bool(loc1["shadow_fd_nominal"].get("converged")),
        "c_grid_stable_loc1": bool(loc1["grid_stable"]),
        "acts_variational_self_reference_forbidden": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_pass_on_target3_production_loc1_alone": True,
    }


def _param_block(seg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for item in seg.get("parameters") or []:
        if item.get("parameter") == name:
            return item
    return {}


def _classify_unresolved(segs: list[Mapping[str, Any]]) -> str:
    if not segs or any(not item.get("mean_contract_closed") for item in segs):
        return CASE_MEAN_NOT_EST
    n_dot = [item.get("n_dot_direction") for item in segs]
    if any(v is not None and abs(float(v)) < 1.0e-3 for v in n_dot):
        return CASE_PLANE
    if any(int(item.get("n_surface_material_crossings") or 0) > 0 for item in segs):
        oscillating = [
            item
            for item in segs
            if not item.get("independent_reference_established")
        ]
        if oscillating and any(
            bool(
                (_param_block(item, "loc1").get("shadow_fd_nominal") or {}).get(
                    "sign_change"
                )
            )
            for item in oscillating
        ):
            return CASE_MATERIAL
        return CASE_FIELD
    if any(not item.get("grid_mean_stable") for item in segs):
        return CASE_RESOLVE
    return CASE_FIELD


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wb123_hash_match": bool(inventory.get("wb123_hash_match")),
        "wb124_hash_match": True,
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
    if inventory.get("tuned_grid"):
        refuse_tune_grid()
    if inventory.get("tuned_dopri5"):
        refuse_tune_dopri5()
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
    any_mean_fail = (not segs) or any(
        not item.get("mean_contract_closed") for item in segs
    )
    if any_mean_fail and not focus_ok:
        primary = CASE_MEAN_NOT_EST
        next_step = "keep_86_common_grid_shadow_mean_contract"
    elif any_inconsistent:
        primary = CASE_INCONSISTENT
        next_step = "keep_field_gradient_tangent_implementation"
    elif not focus_ok:
        primary = _classify_unresolved(segs)
        next_step = "keep_86_common_grid_shadow_reference"
    elif (not c0_pass) or (not c1_pass) or (not c37_pass) or (not mean_ok):
        primary = CASE_MIXED
        next_step = "keep_86_common_grid_shadow_reference"
    else:
        primary = CASE_ESTABLISHED
        next_step = "reopen_b14m_smoke_only"
    contract = primary == CASE_ESTABLISHED and focus_ok and c0_pass and c1_pass and c37_pass
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
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_call_vacuum_a_production_map_reference": True,
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
        "tuned_grid": False,
        "tuned_dopri5": False,
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
        "inherited_wb124_decision_sha256": inherited["workbook_124"]["decision_sha256"],
        "inherited_wb123_decision_sha256": inherited["workbook_123"]["decision_sha256"],
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
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_call_vacuum_a_production_map_reference": True,
        "pinv_relative": PINV_RELATIVE,
        "objective": "common-grid 86 required-segment derivative reference",
        "not_the_objective": "repair 5D Cin",
    }
