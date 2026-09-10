"""Task B14ZB: ACTS surface energy-loss mean semantics contract.

WB126 showed production ΔE sits between computeEnergyLossMean and
computeEnergyLossMode.  This task traces the compiled ACTS 32.0.2
MaterialInteractor path and copies the real deterministic update:
whether updateState runs, which Eloss it uses, and how p/E maps back
to q/p.  Derivatives stay closed.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

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
from datasets.acts_fd_derivative_contract import FOCUS_EVENT, _event_label, _event_pair
from datasets.field_gradient_variational_repair import FOCUS_FIRST_UNSTABLE
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
from datasets.shadow_mean_transport_contract import (
    CASE_ELOSS as WB126_DECISION,
    FROZEN_FIELD_GRADIENT_SHA,
    inherit_frozen_stage as inherit_through_wb126,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "surface-energy-loss-mean-semantics-v1"
DEFAULT_CONFIG = "configs/surface_energy_loss_mean_semantics_v1.yaml"
TASK = "SB-B14ZB"
WORKBOOK = 127

CASE_ESTABLISHED = "surface_energy_loss_semantics_established"
CASE_GATING = "surface_material_update_gating_mismatch"
CASE_FORMULA = "energy_loss_formula_or_particle_hypothesis_mismatch"
CASE_QOP = "qop_update_conversion_mismatch"
CASE_PATH = "path_bookkeeping_mismatch_after_material_closure"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_ESTABLISHED,
    CASE_GATING,
    CASE_FORMULA,
    CASE_QOP,
    CASE_PATH,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT}
MEAN_LOC0 = 1.0e-3
MEAN_PATH = 1.0e-2
MEAN_POS = 1.0e-2
MEAN_DIR = 1.0e-6
MEAN_QOP = 1.0e-4
NODE_QOP_ABS = 1.0e-9
NODE_QOP_REL = 1.0e-6
PRODUCTION_ELOSS = "computeEnergyLossBethe"


class SurfaceEnergyLossSemanticsError(ValueError):
    """Raised when the B14ZB energy-loss semantics contract is illegal."""


def refuse_prior() -> None:
    raise SurfaceEnergyLossSemanticsError("B14ZB must not introduce a prior")


def refuse_ridge_information() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise SurfaceEnergyLossSemanticsError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_change_repair() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "must not change the frozen field-gradient variational implementation"
    )


def refuse_evaluate_jacobian() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "B14ZB must not evaluate or read Jacobian agreement"
    )


def refuse_fit_eloss() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "must not fit an Eloss to chase the endpoint"
    )


def refuse_infer_loc0() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "must not infer material loss from final loc0"
    )


def refuse_b14m() -> None:
    raise SurfaceEnergyLossSemanticsError("B14M is not re-opened inside Task B14ZB")


def refuse_b15() -> None:
    raise SurfaceEnergyLossSemanticsError("Task B15 is not entered in Task B14ZB")


def refuse_full_sample() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "the 1989-row campaign must not be submitted in B14ZB"
    )


def refuse_substitute_hit6() -> None:
    raise SurfaceEnergyLossSemanticsError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise SurfaceEnergyLossSemanticsError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SurfaceEnergyLossSemanticsError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise SurfaceEnergyLossSemanticsError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise SurfaceEnergyLossSemanticsError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14za",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise SurfaceEnergyLossSemanticsError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14zb", True)):
        raise SurfaceEnergyLossSemanticsError("do_not_enter_b14zb must be false")
    spec = config["surface_energy_loss_mean_semantics"]
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise SurfaceEnergyLossSemanticsError(
            "production stepTolerance must stay 1e-4"
        )
    if spec.get("production_eloss_quantity") != PRODUCTION_ELOSS:
        raise SurfaceEnergyLossSemanticsError(
            "production Eloss quantity must stay computeEnergyLossBethe"
        )
    if {int(k): int(v) for k, v in spec["required_first_unstable"].items()} != (
        FOCUS_FIRST_UNSTABLE
    ):
        raise SurfaceEnergyLossSemanticsError(
            "required first-unstable hops must stay 1/6, 2/11, 3/11"
        )
    for flag in (
        "derivative_not_evaluated",
        "jacobian_agreement_not_read",
        "do_not_evaluate_fd",
        "do_not_read_jacobian",
        "do_not_fit_eloss_to_endpoint",
        "do_not_infer_eloss_from_loc0",
        "do_not_reopen_b14m",
    ):
        if not bool(spec.get(flag, False)):
            raise SurfaceEnergyLossSemanticsError(f"{flag} must be true")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise SurfaceEnergyLossSemanticsError("allowed decisions must stay the B14ZB set")
    _expect_sha(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        ),
        FROZEN_FIELD_GRADIENT_SHA,
        "FieldGradientDefaultExtension",
    )
    for item in config.get("wb126_smoke_dumps") or []:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb126 smoke {item['path']}",
        )
    for item in config["wb123_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb123 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb126(config)
    spec = config["inheritance"]["workbook_126"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_126 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_126 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise SurfaceEnergyLossSemanticsError("workbook_126 decision must stay frozen")
    if spec["frozen_decision"] != WB126_DECISION:
        raise SurfaceEnergyLossSemanticsError("WB126 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise SurfaceEnergyLossSemanticsError("WB126 must not have re-opened B14M")
    inherited["workbook_126"] = {
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
        "do_not_evaluate_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    name = str(
        config.get(
            "official_jacobian_smoke_filename",
            "ckf_leave_target_out_surface_eloss.jsonl",
        )
    )
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


def _residual(block: Mapping[str, Any] | None) -> dict[str, Any]:
    if not block:
        return {"present": False, "closed": False}
    return {
        "present": True,
        "closed": bool(block.get("closed")),
        "delta_loc0": block.get("delta_loc0"),
        "delta_path": block.get("delta_path"),
        "delta_position": block.get("delta_position"),
        "delta_direction": block.get("delta_direction"),
        "delta_qop_rel": block.get("delta_qop_rel"),
    }


def _surfaces(seg: Mapping[str, Any]) -> list[dict[str, Any]]:
    eloss = seg.get("surface_energy_loss_mean_contract") or {}
    return list(eloss.get("surfaces") or [])


def _node_qop_closed(item: Mapping[str, Any]) -> bool:
    qp = item.get("qop_after_production")
    qs = item.get("qop_after_shadow")
    if qp is None or qs is None:
        return bool(item.get("node_qop_closed"))
    den = max(abs(float(qp)), 1.0e-12)
    abs_err = abs(float(qs) - float(qp))
    return abs_err <= NODE_QOP_ABS or abs_err / den <= NODE_QOP_REL


def _classify_segment(seg: Mapping[str, Any]) -> str:
    surfaces = _surfaces(seg)
    node = seg.get("node_qop_contract") or {}
    nodes_closed = bool(surfaces) and all(_node_qop_closed(item) for item in surfaces)
    if not nodes_closed:
        nodes_closed = bool(node.get("all_closed")) or (
            bool(surfaces)
            and all(bool(item.get("node_qop_closed")) for item in surfaces)
        )
    endpoint_closed = bool(seg.get("mean_contract_closed"))
    if not surfaces:
        if endpoint_closed:
            return CASE_MIXED
        return CASE_MIXED

    gating_fail = False
    formula_fail = False
    conversion_fail = False
    for item in surfaces:
        updated = bool(item.get("update_state_called"))
        pointer = bool(item.get("has_surface_material_pointer", True))
        dq = abs(float(item.get("delta_qop_production") or 0.0))
        if pointer and not updated and dq > 1.0e-18:
            gating_fail = True
        if not updated:
            continue
        bethe = bool(item.get("production_matches_bethe_formula"))
        evmatch = bool(item.get("production_matches_evaluatePointwise"))
        mean = bool(item.get("production_matches_mean_formula"))
        mode = bool(item.get("production_matches_mode_formula"))
        if not bethe and not evmatch:
            if mean or mode:
                formula_fail = True
            else:
                formula_fail = True
        if evmatch is False and bethe:
            conversion_fail = True

    if gating_fail:
        return CASE_GATING
    if formula_fail:
        return CASE_FORMULA
    if conversion_fail:
        return CASE_QOP
    if nodes_closed and endpoint_closed:
        return CASE_ESTABLISHED
    if nodes_closed and not endpoint_closed:
        return CASE_PATH
    if not nodes_closed:
        first = seg.get("first_divergence") or {}
        if first.get("found") and first.get("event_kind") == "surface_material":
            return CASE_FORMULA
        return CASE_MIXED
    return CASE_MIXED


def audit_required_segment(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("surface_energy_loss_mean_semantics") or row.get(
        "shadow_mean_transport_contract"
    ) or {}
    target = int(row.get("target_station", -1))
    required_idx = FOCUS_FIRST_UNSTABLE.get(target)
    hops = list(payload.get("hops") or [])
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
            "mean_contract_closed": False,
            "node_qop_all_closed": False,
            "taxonomy": CASE_MIXED,
        }
    res = _residual(required.get("shadow_p_residual"))
    out = {
        "event": _event_label(_event_pair(row)),
        "target_station": target,
        "required_index": required_idx,
        "measurement_index": required.get("measurement_index"),
        "present": True,
        "hop_start_state_present": "hop_start_state" in required,
        "mean_contract_closed": bool(res.get("closed")),
        "node_qop_all_closed": bool(required.get("node_qop_all_closed")),
        "shadow_p_residual": res,
        "first_divergence": required.get("first_divergence") or {},
        "surface_energy_loss_mean_contract": required.get(
            "surface_energy_loss_mean_contract"
        )
        or required.get("surface_energy_loss_mean_semantics")
        or {},
        "node_qop_contract": required.get("node_qop_contract") or {},
        "material_update_order_contract": required.get(
            "material_update_order_contract"
        )
        or {},
        "qop_energy_loss_unit_contract": required.get(
            "qop_energy_loss_unit_contract"
        )
        or {},
        "production_ledger": required.get("production_ledger") or {},
        "shadow_p_ledger": required.get("shadow_p_ledger") or {},
        "production_eloss_quantity": required.get(
            "production_eloss_quantity", PRODUCTION_ELOSS
        ),
        "derivative_not_evaluated": bool(
            required.get("derivative_not_evaluated", True)
        ),
        "do_not_fit_eloss_to_endpoint": bool(
            required.get("do_not_fit_eloss_to_endpoint", True)
        ),
        "do_not_infer_eloss_from_loc0": bool(
            required.get("do_not_infer_eloss_from_loc0", True)
        ),
        "do_not_substitute_earlier_long_hop": True,
    }
    surfaces = _surfaces(out)
    if surfaces:
        out["node_qop_all_closed"] = all(_node_qop_closed(item) for item in surfaces)
    elif not out["node_qop_all_closed"]:
        node = out["node_qop_contract"]
        out["node_qop_all_closed"] = bool(node.get("all_closed"))
    out["taxonomy"] = _classify_segment(out)
    return out


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wb123_hash_match": bool(inventory.get("wb123_hash_match")),
        "wb126_hash_match": bool(inventory.get("wb126_hash_match")),
        "focus_86_present": bool(inventory.get("focus_segments")),
        "required_hops_are_6_11_11": bool(inventory.get("required_hops_correct")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "no_jacobian_evaluated": not bool(inventory.get("jacobian_evaluated")),
        "no_fitted_eloss": not bool(inventory.get("fitted_eloss")),
        "five_percent_gate_unchanged": True,
        "target_exclusion": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds", True)
        ),
        "controls_inherited_pass": True,
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
    if inventory.get("changed_field_gradient_repair"):
        refuse_change_repair()
    if inventory.get("jacobian_evaluated"):
        refuse_evaluate_jacobian()
    if inventory.get("fitted_eloss"):
        refuse_fit_eloss()
    if inventory.get("inferred_from_loc0"):
        refuse_infer_loc0()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    if inventory.get("substituted_hit6"):
        refuse_substitute_hit6()
    segs = inventory.get("focus_segments") or []
    all_closed = bool(segs) and len(segs) == 3 and all(
        item.get("mean_contract_closed") and item.get("node_qop_all_closed")
        for item in segs
    )
    taxes = [item.get("taxonomy") for item in segs]
    unique = {item for item in taxes if item}
    if all_closed and unique == {CASE_ESTABLISHED}:
        primary = CASE_ESTABLISHED
        next_step = "reuse_certified_mean_shadow_for_common_grid_independent_fd"
    elif not segs:
        primary = CASE_MIXED
        next_step = "keep_surface_energy_loss_mean_semantics"
    elif len(unique) == 1:
        primary = next(iter(unique))
        next_step = "keep_surface_energy_loss_mean_semantics"
    else:
        primary = CASE_MIXED
        next_step = "keep_surface_energy_loss_mean_semantics"
    established = primary == CASE_ESTABLISHED and all_closed
    return {
        "verdict": "PASS" if established else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "shadow_mean_contract_established": established,
        "control_0_pass": True,
        "control_1_pass": True,
        "control_37_pass": True,
        "mean_path_unchanged": bool(inventory.get("mean_path_unchanged", True)),
        "focus_independent_reference_established": False,
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
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
        "do_not_change_production_step_tolerance": True,
        "do_not_change_official_mean_path": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_evaluate_jacobian": True,
        "do_not_fit_eloss_to_endpoint": True,
        "do_not_infer_eloss_from_loc0": True,
        "derivative_not_evaluated": True,
        "jacobian_agreement_not_read": True,
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
    focus_seg = []
    required_ok = True
    jacobian_evaluated = False
    substituted = False
    for row in dumps["rows"]:
        if not is_official_mode_b(row):
            continue
        pair = _event_pair(row)
        if pair not in REQUIRED_EVENTS:
            continue
        if row.get("jacobian_validation") is not None:
            jacobian_evaluated = True
        if row.get("official_supporting_plane_jacobian") is not None:
            jacobian_evaluated = True
        if row.get("surface_energy_loss_mean_semantics") is None and row.get(
            "shadow_mean_transport_contract"
        ) is None:
            continue
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
        "wb126_hash_match": True,
        "n_rows": dumps["n_rows"],
        "controls": [
            {"event": "100043/0", "five_percent_pass": True, "inherited": True},
            {"event": "100043/1", "five_percent_pass": True, "inherited": True},
            {"event": "100043/37", "five_percent_pass": True, "inherited": True},
        ],
        "focus_segments": focus_seg,
        "mean_path_unchanged": True,
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
        "changed_field_gradient_repair": False,
        "jacobian_evaluated": jacobian_evaluated,
        "fitted_eloss": False,
        "inferred_from_loc0": False,
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
    mechanism["focus_independent_reference_established"] = False
    mechanism["jacobian_contract_established"] = False
    mechanism["b14m_reopen_authorized"] = False
    if not mechanism.get("shadow_mean_contract_established"):
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb126_decision_sha256": inherited["workbook_126"][
            "decision_sha256"
        ],
        "inherited_wb125_decision_sha256": inherited["workbook_125"][
            "decision_sha256"
        ],
        "inherited_wb124_decision_sha256": inherited["workbook_124"][
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
                "node_qop_all_closed": item.get("node_qop_all_closed"),
                "residual": item.get("shadow_p_residual"),
                "first_divergence": item.get("first_divergence"),
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
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_evaluate_jacobian": True,
        "do_not_fit_eloss_to_endpoint": True,
        "do_not_infer_eloss_from_loc0": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "pinv_relative": PINV_RELATIVE,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "objective": "copy production surface energy-loss mean update",
        "not_the_objective": "compare derivatives or repair 5D Cin",
    }
