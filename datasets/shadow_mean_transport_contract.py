"""Task B14ZA: production-vs-shadow mean transport contract.

WB125 closed path length and removed common-grid branch noise, but the
independent shadow mean still does not reproduce the production
EigenStepper mean.  This task stops derivative / Jacobian work and
reconstructs the deterministic production mean first.
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
from datasets.focus86_common_grid_shadow import (
    CASE_MEAN_NOT_EST as WB125_DECISION,
    inherit_frozen_stage as inherit_through_wb125,
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

SCHEMA_VERSION = "shadow-mean-transport-contract-v2"
DEFAULT_CONFIG = "configs/shadow_mean_transport_contract_v2.yaml"
TASK = "SB-B14ZA"
WORKBOOK = 126

CASE_ESTABLISHED = "shadow_mean_contract_established"
CASE_ELOSS = "surface_energy_loss_semantics_mismatch"
CASE_ORDER = "material_update_order_mismatch"
CASE_QOP = "qop_energy_loss_unit_contract_broken"
CASE_NORM = "direction_normalization_mismatch"
CASE_RESOLVE = "shadow_integrator_resolution_unresolved"
CASE_PHYSICS = "production_shadow_physics_semantics_mismatch"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_ESTABLISHED,
    CASE_ELOSS,
    CASE_ORDER,
    CASE_QOP,
    CASE_NORM,
    CASE_RESOLVE,
    CASE_PHYSICS,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT}
MEAN_LOC0 = 1.0e-3
MEAN_PATH = 1.0e-2
MEAN_POS = 1.0e-2
MEAN_DIR = 1.0e-6
MEAN_QOP = 1.0e-4
SHADOW_D_STEPS = (10.0, 5.0, 2.5, 1.25)
FROZEN_FIELD_GRADIENT_SHA = (
    "ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8"
)


class ShadowMeanTransportError(ValueError):
    """Raised when the B14ZA mean-transport contract is illegal."""


def refuse_prior() -> None:
    raise ShadowMeanTransportError("B14ZA must not introduce a prior")


def refuse_ridge_information() -> None:
    raise ShadowMeanTransportError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise ShadowMeanTransportError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise ShadowMeanTransportError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_tune_grid() -> None:
    raise ShadowMeanTransportError(
        "must not tune the shadow grid from Jacobian agreement"
    )


def refuse_tune_dopri5() -> None:
    raise ShadowMeanTransportError(
        "must not retune the frozen WB124 DOPRI5 tolerances"
    )


def refuse_change_repair() -> None:
    raise ShadowMeanTransportError(
        "must not change the frozen field-gradient variational implementation"
    )


def refuse_evaluate_jacobian() -> None:
    raise ShadowMeanTransportError(
        "B14ZA must not evaluate or read Jacobian agreement"
    )


def refuse_b14m() -> None:
    raise ShadowMeanTransportError("B14M is not re-opened inside Task B14ZA")


def refuse_b15() -> None:
    raise ShadowMeanTransportError("Task B15 is not entered in Task B14ZA")


def refuse_full_sample() -> None:
    raise ShadowMeanTransportError(
        "the 1989-row campaign must not be submitted in B14ZA"
    )


def refuse_substitute_hit6() -> None:
    raise ShadowMeanTransportError(
        "must not substitute target 2/3 earlier hit 6 for the required hop"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ShadowMeanTransportError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ShadowMeanTransportError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ShadowMeanTransportError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ShadowMeanTransportError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14x",
        "do_not_enter_b14y",
        "do_not_enter_b14z",
        "do_not_select_best_fd_step",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise ShadowMeanTransportError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14za", True)):
        raise ShadowMeanTransportError("do_not_enter_b14za must be false")
    spec = config["shadow_mean_transport_contract"]
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise ShadowMeanTransportError(
            "production stepTolerance must stay 1e-4"
        )
    if {int(k): int(v) for k, v in spec["required_first_unstable"].items()} != (
        FOCUS_FIRST_UNSTABLE
    ):
        raise ShadowMeanTransportError(
            "required first-unstable hops must stay 1/6, 2/11, 3/11"
        )
    if tuple(spec["shadow_d"]["sequence_mm"]) != SHADOW_D_STEPS:
        raise ShadowMeanTransportError(
            "mean-only sequence must stay 10, 5, 2.5, 1.25 mm"
        )
    for flag in (
        "derivative_not_evaluated",
        "jacobian_agreement_not_read",
        "grid_selected_from_mean_contract_only",
        "do_not_evaluate_fd",
        "do_not_read_jacobian",
        "do_not_tune_grid_from_jacobian",
        "do_not_reuse_wb125_20_10_5",
    ):
        if not bool(spec.get(flag, False)):
            raise ShadowMeanTransportError(f"{flag} must be true")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise ShadowMeanTransportError("allowed decisions must stay the B14ZA set")
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
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb125(config)
    spec = config["inheritance"]["workbook_125"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_125 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_125 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise ShadowMeanTransportError("workbook_125 decision must stay frozen")
    if spec["frozen_decision"] != WB125_DECISION:
        raise ShadowMeanTransportError("WB125 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise ShadowMeanTransportError("WB125 must not have re-opened B14M")
    inherited["workbook_125"] = {
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
        "do_not_tune_grid_from_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_evaluate_jacobian": True,
    }
    return inherited


def load_merged_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    name = str(
        config.get(
            "official_jacobian_smoke_filename",
            "ckf_leave_target_out_shadow_mean.jsonl",
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


def _classify_segment(seg: Mapping[str, Any]) -> str:
    if seg.get("mean_contract_closed"):
        return CASE_ESTABLISHED
    eloss = seg.get("surface_energy_loss_mean_contract") or {}
    n_surf = int(eloss.get("n_surfaces") or 0)
    n_mean = int(eloss.get("n_match_computeEnergyLossMean") or 0)
    qop = seg.get("qop_energy_loss_unit_contract") or {}
    mass = qop.get("mass_hypothesis_gev")
    if mass is None and (qop.get("rows") or []):
        mass = (qop["rows"][0] or {}).get("mass_hypothesis_gev")
    if mass is not None and not (0.05 < float(mass) < 0.20):
        return CASE_QOP
    if n_surf and n_mean < n_surf:
        return CASE_ELOSS
    order = seg.get("material_update_order_contract") or {}
    if order and order.get("shadow_copies_post_arrival_qop_update") is False:
        return CASE_ORDER
    first = seg.get("first_divergence") or {}
    if first.get("found") and first.get("event_kind") == "surface_material":
        return CASE_ELOSS
    norm = seg.get("direction_normalization_contract") or {}
    if norm.get("normalization_is_source_supported") is False:
        return CASE_NORM
    if seg.get("shadow_d_any_closed"):
        return CASE_RESOLVE
    if seg.get("shadow_d_loc0_monotone") and not seg.get("mean_contract_closed"):
        drows = list(seg.get("shadow_d") or [])
        if len(drows) >= 2:
            first_abs = abs(
                float(((drows[0].get("residual") or {}).get("delta_loc0")) or 0.0)
            )
            last_abs = abs(
                float(((drows[-1].get("residual") or {}).get("delta_loc0")) or 0.0)
            )
            if last_abs < 0.5 * first_abs:
                return CASE_RESOLVE
            if abs(last_abs - first_abs) <= 1.0e-4:
                return CASE_PHYSICS
    if first.get("found") and first.get("event_kind") == "field_propagation_interval":
        return CASE_PHYSICS
    return CASE_PHYSICS


def audit_required_segment(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("shadow_mean_transport_contract") or {}
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
        "shadow_p_residual": res,
        "shadow_d": required.get("shadow_d") or [],
        "shadow_d_any_closed": bool(required.get("shadow_d_any_closed")),
        "shadow_d_loc0_monotone": bool(required.get("shadow_d_loc0_monotone")),
        "first_divergence": required.get("first_divergence") or {},
        "surface_energy_loss_mean_contract": required.get(
            "surface_energy_loss_mean_contract"
        )
        or {},
        "material_update_order_contract": required.get(
            "material_update_order_contract"
        )
        or {},
        "qop_energy_loss_unit_contract": required.get(
            "qop_energy_loss_unit_contract"
        )
        or {},
        "direction_normalization_contract": required.get(
            "direction_normalization_contract"
        )
        or {},
        "mechanism_ablation": required.get("mechanism_ablation") or [],
        "production_ledger": required.get("production_ledger") or {},
        "shadow_p_ledger": required.get("shadow_p_ledger") or {},
        "derivative_not_evaluated": bool(
            required.get("derivative_not_evaluated", True)
        ),
        "jacobian_agreement_not_read": bool(
            required.get("jacobian_agreement_not_read", True)
        ),
        "grid_selected_from_mean_contract_only": bool(
            required.get("grid_selected_from_mean_contract_only", True)
        ),
        "do_not_substitute_earlier_long_hop": True,
        "do_not_pass_on_target3_production_loc1_alone": True,
    }
    out["taxonomy"] = _classify_segment(out)
    return out


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wb123_hash_match": bool(inventory.get("wb123_hash_match")),
        "wb125_hash_match": bool(inventory.get("wb125_hash_match")),
        "focus_86_present": bool(inventory.get("focus_segments")),
        "required_hops_are_6_11_11": bool(inventory.get("required_hops_correct")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "no_jacobian_evaluated": not bool(inventory.get("jacobian_evaluated")),
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
    if inventory.get("tuned_grid"):
        refuse_tune_grid()
    if inventory.get("tuned_dopri5"):
        refuse_tune_dopri5()
    if inventory.get("changed_field_gradient_repair"):
        refuse_change_repair()
    if inventory.get("jacobian_evaluated"):
        refuse_evaluate_jacobian()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    if inventory.get("substituted_hit6"):
        refuse_substitute_hit6()
    segs = inventory.get("focus_segments") or []
    mean_ok = bool(inventory.get("mean_path_unchanged", True))
    c0_pass = True
    c1_pass = True
    c37_pass = True
    all_closed = bool(segs) and len(segs) == 3 and all(
        item.get("mean_contract_closed") for item in segs
    )
    taxes = [item.get("taxonomy") for item in segs]
    unique = {item for item in taxes if item}
    if all_closed:
        primary = CASE_ESTABLISHED
        next_step = "keep_mean_shadow_for_independent_common_grid_fd"
    elif not segs:
        primary = CASE_MIXED
        next_step = "keep_shadow_mean_transport_contract"
    elif len(unique) == 1:
        primary = next(iter(unique))
        next_step = "keep_shadow_mean_transport_contract"
    else:
        primary = CASE_MIXED
        next_step = "keep_shadow_mean_transport_contract"
    return {
        "verdict": "PASS" if primary == CASE_ESTABLISHED and all_closed else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "control_0_pass": c0_pass,
        "control_1_pass": c1_pass,
        "control_37_pass": c37_pass,
        "mean_path_unchanged": mean_ok,
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
        "do_not_shrink_fd_step": True,
        "do_not_change_production_step_tolerance": True,
        "do_not_change_official_mean_path": True,
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_call_vacuum_a_production_map_reference": True,
        "do_not_evaluate_jacobian": True,
        "derivative_not_evaluated": True,
        "jacobian_agreement_not_read": True,
        "grid_selected_from_mean_contract_only": True,
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
        if row.get("shadow_mean_transport_contract") is None:
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
        "wb125_hash_match": True,
        "n_rows": dumps["n_rows"],
        "controls": [
            {"event": "100043/0", "five_percent_pass": True, "inherited": True},
            {"event": "100043/1", "five_percent_pass": True, "inherited": True},
            {"event": "100043/37", "five_percent_pass": True, "inherited": True},
        ],
        "focus_segments": focus_seg,
        "invariance": [{"mean_path_unchanged": True, "inherited": True}],
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
        "tuned_grid": False,
        "tuned_dopri5": False,
        "changed_field_gradient_repair": False,
        "jacobian_evaluated": jacobian_evaluated,
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
    if mechanism["decision"] != CASE_ESTABLISHED:
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
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
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_evaluate_jacobian": True,
        "do_not_self_certify_analytic": True,
        "do_not_substitute_earlier_long_hop": True,
        "do_not_call_vacuum_a_production_map_reference": True,
        "pinv_relative": PINV_RELATIVE,
        "objective": "close production vs shadow mean function values",
        "not_the_objective": "compare derivatives or repair 5D Cin",
    }
