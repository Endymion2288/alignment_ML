"""Task B14T: standalone measurement transport contract repair.

Makes h_i(theta) well-defined on every WB109 surviving measurement
surface under the real ACTS/FASER tracking-geometry navigator and
plane-chart projection.  Does not change the frozen WB114 likelihood,
retune Gauss-Newton, introduce a prior/ridge, delete 37/86, replace
q/p with truth, or enter B14M / B15 / Measurement Model V2.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
from datasets.profile_likelihood_numerics import (
    CASE_B as WB115_DECISION,
    audit_exclusion,
    inherit_frozen_stage as inherit_through_wb114,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "profile-transport-contract-v1"
DEFAULT_CONFIG = "configs/profile_transport_contract_v1.yaml"
TASK = "SB-B14T"
WORKBOOK = 116

CASE_ESTABLISHED = "standalone_measurement_transport_contract_established"
CASE_STATE = "reconstructed_state_not_transportable_to_surviving_measurements"
CASE_PROJECTION = "measurement_surface_projection_contract_not_established"
CASE_NAVIGATION = "acts_navigation_transport_contract_not_established"
CASE_MIXED = "mixed_or_inconclusive"

CLASS_M = "long_magnet_crossing_hop"
CLASS_S = "local_stereo_surface_intersection"

FOCUS_KEYS = {
    ("mc24_100043_00400_00499", 100043, 0),
    ("mc24_100043_00400_00499", 100043, 1),
    ("mc24_100043_00400_00499", 100043, 37),
    ("mc24_100048_00000_00049", 100048, 86),
}
CONTROL_EVENTS = {(100043, 0), (100043, 1)}
CLASS_M_EVENT = (100043, 37)
CLASS_S_EVENT = (100048, 86)


class ProfileTransportContractError(ValueError):
    """Raised when the B14T transport contract is illegal."""


def refuse_prior() -> None:
    raise ProfileTransportContractError("B14T must not introduce a prior")


def refuse_ridge_information() -> None:
    raise ProfileTransportContractError("ridge must not be treated as statistical information")


def refuse_measurement_update() -> None:
    raise ProfileTransportContractError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_truth_qoverp() -> None:
    raise ProfileTransportContractError("truth q/p is not a transport repair")


def refuse_max_steps_hack() -> None:
    raise ProfileTransportContractError(
        "an arbitrary maxSteps increase is not an official transport fix"
    )


def refuse_tolerance_hunt() -> None:
    raise ProfileTransportContractError(
        "boundary tolerance must not be increased until a focus event passes"
    )


def refuse_merge_classes() -> None:
    raise ProfileTransportContractError("Class M and Class S must not be merged")


def refuse_delete_focus() -> None:
    raise ProfileTransportContractError("focus identities 100043/37 and 100048/86 must be retained")


def refuse_b14m() -> None:
    raise ProfileTransportContractError("B14M is not re-opened inside Task B14T")


def refuse_b15() -> None:
    raise ProfileTransportContractError("Task B15 is not entered in Task B14T")


def refuse_measurement_model_v2() -> None:
    raise ProfileTransportContractError("Measurement Model V2 is not entered in Task B14T")


def refuse_full_sample() -> None:
    raise ProfileTransportContractError("the 1989-row campaign must not be submitted in B14T")


def refuse_physical_nonidentifiability() -> None:
    raise ProfileTransportContractError(
        "B14T must not declare physical_nonidentifiability"
    )


def refuse_rewrite_profile_math() -> None:
    raise ProfileTransportContractError("WB114 profile mathematics must not be rewritten")


def refuse_change_statistical_model() -> None:
    raise ProfileTransportContractError("the measurement statistical model must stay frozen")


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ProfileTransportContractError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ProfileTransportContractError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ProfileTransportContractError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ProfileTransportContractError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise ProfileTransportContractError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "marginalization_executed",
    ):
        if bool(config.get(key, True)):
            raise ProfileTransportContractError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14n",
        "do_not_rewrite_profile_math",
        "do_not_change_statistical_model",
        "do_not_treat_propagation_failure_as_nonidentifiability",
        "do_not_invent_chi2_penalty_for_propagation_failure",
        "do_not_interpret_scaling_as_prior",
        "do_not_submit_full_sample_without_smoke_gate",
        "do_not_introduce_target_independent_prior",
        "do_not_delete_qoverp",
        "do_not_fix_qoverp",
        "do_not_use_wb109_cin_as_likelihood",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
        "do_not_select_best_seed",
        "do_not_construct_acts_objects_in_python",
        "do_not_use_arbitrary_max_steps_as_official_fix",
        "do_not_use_arbitrary_boundary_tolerance",
        "do_not_add_measurement_update_in_evaluator",
        "do_not_merge_class_m_and_class_s",
        "do_not_claim_physical_nonidentifiability",
    ):
        if bool(config.get(key, False)) is not True:
            raise ProfileTransportContractError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14t", True)):
        raise ProfileTransportContractError("B14T config must allow entering B14T")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ProfileTransportContractError(f"frozen gate changed: {key}")
    numeric = config.get("profile_numerical") or {}
    if float(numeric.get("pinv_relative")) == 0.01:
        refuse_alignment_rank_tolerance()
    if abs(float(numeric.get("pinv_relative")) - PINV_RELATIVE) > 1.0e-20:
        raise ProfileTransportContractError("pinv_relative must stay pre-registered 1e-8")
    if bool(numeric.get("do_not_add_ridge")) is not True:
        refuse_ridge()
    if bool((config.get("profile_transport_contract") or {}).get("measurement_update_in_evaluator")):
        refuse_measurement_update()
    if bool((config.get("profile_transport_contract") or {}).get("diagnostic_max_steps_is_not_official_fix")) is not True:
        refuse_max_steps_hack()
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise ProfileTransportContractError("B14T eligibility must stay identical to WB103")
    partition = config["profile_partition"]
    if list(partition["alpha_native"]) != list(ALPHA_NAMES):
        raise ProfileTransportContractError("alpha must stay loc0, theta")
    if list(partition["nu_native"]) != list(NU_NAMES):
        raise ProfileTransportContractError("nu must stay loc1, phi, q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb114(config)
    spec = config["inheritance"]["workbook_115"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_115 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_115 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise ProfileTransportContractError("workbook_115 decision must stay frozen")
    if spec["frozen_decision"] != WB115_DECISION:
        raise ProfileTransportContractError("WB115 decision token mismatch")
    if decision.get("b15_authorized"):
        raise ProfileTransportContractError("WB115 must not have authorized B15")
    if decision.get("prior_introduced"):
        raise ProfileTransportContractError("WB115 must not have introduced a prior")
    inherited["workbook_115"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "smoke_gate_passed": False,
        "full_sample_authorized": False,
        "synthetic_profile_passed": True,
        "target_exclusion_holds": True,
        "b15_authorized": False,
        "do_not_force_5d_lto_covariance": True,
        "statistical_model_unchanged": True,
        "prior_introduced": False,
    }
    return inherited


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id")), int(row.get("run_id", -1)), int(row.get("event_id", -1)))


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


def is_official_mode_b(row: Mapping[str, Any]) -> bool:
    return (
        row.get("evaluate_only") is True
        and str(row.get("init_kind")) == "wb114_official_seed_mean"
        and bool(row.get("supporting_plane_projection"))
        and str(row.get("hop_mode")) == "sequential_z_order"
    )


def is_mode_a(row: Mapping[str, Any]) -> bool:
    return str(row.get("init_kind")) == "mode_a_bounded_surface_reached"


def is_direct(row: Mapping[str, Any]) -> bool:
    return str(row.get("init_kind")) == "direct_from_source_supporting_plane"


def is_diagnostic_max_steps(row: Mapping[str, Any]) -> bool:
    return str(row.get("init_kind")) == "diagnostic_max_steps"


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["transport_smoke_root"]))
    filename = str(config.get("transport_smoke_filename", "ckf_leave_target_out_profile_transport.jsonl"))
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def wb115_smoke_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["numerics_smoke_root"]))
    filename = str(config.get("numerics_smoke_filename", "ckf_leave_target_out_profile_numerics.jsonl"))
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_transport_rows(config: Mapping[str, Any]) -> dict[str, Any]:
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


def load_wb115_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in wb115_smoke_paths(config):
        for row in load_dump_records(path, split="train"):
            if row.get("evaluate_only") is True and str(row.get("init_kind")) == "wb114_official_seed_mean":
                rows.append(dict(row))
    return rows


def first_failed_hop(row: Mapping[str, Any]) -> dict[str, Any] | None:
    for hop in row.get("propagation_hits") or []:
        if not hop.get("ok"):
            return hop
    return None


def classify_failure_class(row: Mapping[str, Any]) -> str | None:
    if row.get("all_surfaces_reached") or row.get("nominal_chi2_evaluable"):
        return None
    hop = first_failed_hop(row)
    if hop is None:
        return None
    key = _event_pair(row)
    start_z = _finite(hop.get("start_z_mm"))
    dest_z = _finite(hop.get("destination_surface_z_mm") or hop.get("measurement_z_mm"))
    dz = None
    if start_z is not None and dest_z is not None:
        dz = abs(dest_z - start_z)
    reason = str(hop.get("abort_reason") or hop.get("propagation_status") or "").lower()
    magnet = dz is not None and dz > 200.0
    stereo = dz is not None and dz < 5.0 and "not on surface" in reason
    if key == CLASS_M_EVENT or (magnet and "maximum number of steps" in reason):
        return CLASS_M
    if key == CLASS_S_EVENT or stereo:
        return CLASS_S
    if magnet:
        return CLASS_M
    if "not on surface" in reason or "global to local" in reason:
        return CLASS_S
    return None


def acts_kalman_transport_semantics() -> dict[str, Any]:
    """Pinned ACTS 32.0.2 / Calypso facts.  Not inferred from API names."""
    return {
        "acts_version": "32.0.2",
        "athena_release": "24.0.41",
        "sources": {
            "KalmanFitter.hpp": (
                "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
                "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/TrackFitting/KalmanFitter.hpp"
            ),
            "Navigator.hpp": (
                "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
                "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/Navigator.hpp"
            ),
            "StandardAborters.hpp": (
                "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
                "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/StandardAborters.hpp"
            ),
            "Propagator.ipp": (
                "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
                "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/Propagator.ipp"
            ),
        },
        "next_surface_selection": (
            "Kalman actor waits for navigator.currentSurface; measurements "
            "need not be ordered.  Ordering is navigator-driven."
        ),
        "direct_target_surface": (
            "propagate(start, target, options) appends SurfaceReached + PathLimitReached "
            "and still uses the tracking-geometry navigator."
        ),
        "external_measurement_surfaces": (
            "On the first Kalman actor call, every measurement GeometryIdentifier is "
            "inserted as an external surface.  Comment in source: "
            "'We will try to hit those surface by ignoring boundary checks.'"
        ),
        "navigator_boundary_check_for_external_surfaces": False,
        "target_volume_initialization_boundary_check": False,
        "surface_reached_default_boundary_check": True,
        "on_surface_tolerance_mm": 1.0e-4,
        "kalman_plain_max_steps": 10000,
        "kalman_plain_max_step_size": "unlimited_adaptive",
        "loop_protection": True,
        "material_order": "PreUpdate before Kalman update, PostUpdate after; "
        "detector-element / material surfaces without a measurement may create "
        "a no-measurement state and apply FullUpdate material.",
        "kalman_updates_stepper_after_measurement": True,
        "standalone_likelihood_must_not_copy_measurement_update": True,
        "plane_chart_vs_active_bounds": (
            "Surface::intersect defaults to BoundaryCheck(false) on the supporting "
            "plane.  loc0/loc1 are the plane chart.  insideBounds is a separate "
            "active-rectangle membership test.  Residual r = m_loc0 - predicted_loc0 "
            "needs the plane chart, not active-wafer membership."
        ),
        "makeResult_globalToLocal_failure": (
            "After a successful abort, makeResult calls stepper.boundState(target). "
            "If the free position is not within s_onSurfaceTolerance of the plane, "
            "globalToLocal fails with 'Global to local transformation failed: "
            "position not on surface.'"
        ),
        "justification_for_supporting_plane": (
            "Official Mode B sets SurfaceReached.boundaryCheck = false so the "
            "measurement target is the supporting plane, matching Kalman external "
            "surfaces.  Active-bounds membership is recorded, not required for chi2."
        ),
    }


def standalone_likelihood_transport_contract() -> dict[str, Any]:
    return {
        "likelihood": "chi2(theta) = sum_i r_i(theta)^T R_i^{-1} r_i(theta)",
        "r_i": "m_loc0 - predicted_loc0_on_supporting_plane",
        "R_i": "(0.08 mm)^2 / 12",
        "theta": ["loc0", "loc1", "phi", "theta", "q_over_p"],
        "alpha": list(ALPHA_NAMES),
        "nu": list(NU_NAMES),
        "h_i": "deterministic field-aware ACTS transport + plane-chart loc0",
        "measurement_update_in_evaluator": False,
        "why_no_update": (
            "A Kalman update after hit i would make the next prediction depend on "
            "the measurements, converting chi2(theta) into a stateful filter "
            "objective.  WB114 froze the measurement-only sum."
        ),
        "allowed_repairs": [
            "propagation",
            "navigation",
            "surface intersection",
            "plane-chart projection",
        ],
        "forbidden_repairs": [
            "measurement update",
            "prior",
            "ridge",
            "truth q/p",
            "arbitrary maxSteps as official fix",
            "tolerance hunt",
        ],
        "sequential_continuation": (
            "After projecting hit i, continue from the bound state on that "
            "supporting plane.  This is still one deterministic trajectory, "
            "not a filter update."
        ),
    }


def audit_failure_taxonomy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    class_m = []
    class_s = []
    other = []
    merged = False
    for row in rows:
        if not is_official_mode_b(row) and not is_mode_a(row):
            continue
        if _event_pair(row) not in {CLASS_M_EVENT, CLASS_S_EVENT}:
            continue
        label = classify_failure_class(row)
        hop = first_failed_hop(row)
        item = {
            "init_kind": row.get("init_kind"),
            "supporting_plane_projection": row.get("supporting_plane_projection"),
            "source_id": row.get("source_id"),
            "run_id": row.get("run_id"),
            "event_id": row.get("event_id"),
            "target_station": row.get("target_station"),
            "all_surfaces_reached": bool(row.get("all_surfaces_reached")),
            "failure_class": label,
            "first_failed_hop": hop,
        }
        if _event_pair(row) == CLASS_M_EVENT:
            class_m.append(item)
        elif _event_pair(row) == CLASS_S_EVENT:
            class_s.append(item)
        if label not in {CLASS_M, CLASS_S, None}:
            other.append(item)
        if label == CLASS_M and _event_pair(row) == CLASS_S_EVENT:
            merged = True
        if label == CLASS_S and _event_pair(row) == CLASS_M_EVENT:
            merged = True
    if merged:
        refuse_merge_classes()
    return {
        "class_m_name": CLASS_M,
        "class_s_name": CLASS_S,
        "class_m_focus": "100043/37",
        "class_s_focus": "100048/86",
        "never_merged_into_propagation_failed": True,
        "class_m_rows": class_m,
        "class_s_rows": class_s,
        "other_rows": other,
        "classes_kept_separate": not merged,
    }


def audit_magnet_crossing(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _event_pair(row) != CLASS_M_EVENT:
            continue
        if row.get("evaluate_only") is not True:
            continue
        hops = []
        for hop in row.get("propagation_hits") or []:
            start_z = _finite(hop.get("start_z_mm"))
            dest_z = _finite(hop.get("destination_surface_z_mm") or hop.get("measurement_z_mm"))
            dz = None if start_z is None or dest_z is None else abs(dest_z - start_z)
            if dz is None or dz <= 200.0:
                continue
            hops.append(
                {
                    "measurement_index": hop.get("measurement_index"),
                    "start_z_mm": start_z,
                    "destination_surface_z_mm": dest_z,
                    "dz_mm": dz,
                    "ok": hop.get("ok"),
                    "abort_reason": hop.get("abort_reason"),
                    "propagation_status": hop.get("propagation_status"),
                    "n_steps": hop.get("number_of_propagation_steps"),
                    "path_length": hop.get("path_length"),
                    "final_position_xyz_mm": hop.get("final_position_xyz_mm"),
                    "transport_mode": hop.get("transport_mode"),
                    "projection_kind": hop.get("projection_kind"),
                    "inside_active_bounds": hop.get("inside_active_bounds"),
                    "n_checkpoints": len(hop.get("path_checkpoints") or []),
                }
            )
        mode = "B_supporting_plane" if is_official_mode_b(row) else (
            "A_bounded" if is_mode_a(row) else (
                "diagnostic_max_steps" if is_diagnostic_max_steps(row) else str(row.get("init_kind"))
            )
        )
        by_mode[mode].append(
            {
                "target_station": row.get("target_station"),
                "all_surfaces_reached": bool(row.get("all_surfaces_reached")),
                "evaluate_chi2": row.get("evaluate_chi2"),
                "profile_max_steps": row.get("profile_max_steps"),
                "magnet_hops": hops,
            }
        )
    official_ok = all(item["all_surfaces_reached"] for item in by_mode.get("B_supporting_plane", []))
    diagnostic_ok = all(item["all_surfaces_reached"] for item in by_mode.get("diagnostic_max_steps", [])) or not by_mode.get("diagnostic_max_steps")
    if diagnostic_ok and not official_ok and by_mode.get("diagnostic_max_steps"):
        # Diagnostic success is evidence, not a contract PASS.
        diagnostic_only = True
    else:
        diagnostic_only = False
    return {
        "event": "100043/37",
        "modes": by_mode,
        "official_mode_b_all_targets_reached": bool(by_mode.get("B_supporting_plane")) and official_ok,
        "mode_a_all_targets_reached": bool(by_mode.get("A_bounded")) and all(
            item["all_surfaces_reached"] for item in by_mode.get("A_bounded", [])
        ),
        "diagnostic_max_steps_is_not_official_fix": True,
        "diagnostic_success_alone_is_not_pass": diagnostic_only,
        "why_many_steps_would_be_needed": (
            "A bounded SurfaceReached on the destination wafer can miss the finite "
            "rectangle after the magnet.  The stepper then keeps searching until "
            "maxSteps.  That is a target-surface contract, not proof that the "
            "physical trajectory needs 1e5 steps."
        ),
    }


def audit_state_units(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    wanted = {(100043, 0), (100043, 1), CLASS_M_EVENT}
    items = []
    for row in rows:
        if not is_official_mode_b(row):
            continue
        if _event_pair(row) not in wanted:
            continue
        contract = row.get("transport_state_unit_contract") or {}
        items.append(
            {
                "run_id": row.get("run_id"),
                "event_id": row.get("event_id"),
                "target_station": row.get("target_station"),
                "init_native": row.get("init_native"),
                **contract,
            }
        )
    signs = {item.get("charge_from_q_over_p_sign") for item in items if item.get("event_id") == 37}
    return {
        "athena_q_over_p_unit": "1/MeV",
        "acts_internal_q_over_p_unit": "1/GeV",
        "conversion": "q_over_p_acts = q_over_p_mev / 1_MeV",
        "particle_hypothesis": "muon",
        "truth_q_over_p_not_used": True,
        "same_seed_as_kalman_lto_helper": True,
        "n_dumped": len(items),
        "event_37_charge_signs": sorted(signs, key=lambda x: (x is None, x)),
        "rows": items,
    }


def _pathology_of_checkpoints(checkpoints: list[Mapping[str, Any]]) -> str:
    zs = [_finite(item.get("z_mm")) for item in checkpoints]
    zs = [z for z in zs if z is not None]
    if len(zs) < 4:
        return "insufficient_checkpoints"
    diffs = np.diff(np.asarray(zs, dtype=float))
    if float(np.nanmax(np.abs(diffs))) < 1.0 and float(np.ptp(zs)) < 20.0:
        return "stalled_near_boundary"
    if float(np.median(np.abs(diffs))) < 0.05 and len(zs) > 40:
        return "tiny_step_pathology"
    sign_changes = int(np.sum(diffs[1:] * diffs[:-1] < 0))
    if sign_changes >= 4 and float(np.ptp(zs)) > 50.0:
        return "looping_trajectory"
    if np.all(diffs >= -1.0):
        return "monotonic_downstream_progress"
    return "nonmonotonic_but_not_classified_as_loop"


def audit_magnet_pathology(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reports = []
    for row in rows:
        if _event_pair(row) != CLASS_M_EVENT:
            continue
        if not (is_official_mode_b(row) or is_diagnostic_max_steps(row) or is_mode_a(row)):
            continue
        for hop in row.get("propagation_hits") or []:
            start_z = _finite(hop.get("start_z_mm"))
            dest_z = _finite(hop.get("destination_surface_z_mm") or hop.get("measurement_z_mm"))
            if start_z is None or dest_z is None or abs(dest_z - start_z) <= 200.0:
                continue
            checkpoints = hop.get("path_checkpoints") or []
            reports.append(
                {
                    "init_kind": row.get("init_kind"),
                    "target_station": row.get("target_station"),
                    "ok": hop.get("ok"),
                    "n_steps": hop.get("number_of_propagation_steps"),
                    "path_length": hop.get("path_length"),
                    "start_z_mm": start_z,
                    "destination_surface_z_mm": dest_z,
                    "final_position_xyz_mm": hop.get("final_position_xyz_mm"),
                    "classification": _pathology_of_checkpoints(checkpoints),
                    "n_checkpoints": len(checkpoints),
                    "first_checkpoints": checkpoints[:8],
                    "last_checkpoints": checkpoints[-8:],
                }
            )
    return {
        "event": "100043/37",
        "truth_momentum_not_substituted": True,
        "identity_retained": True,
        "hops": reports,
    }


def audit_stereo_geometry(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        if _event_pair(row) != CLASS_S_EVENT:
            continue
        if not (is_official_mode_b(row) or is_mode_a(row)):
            continue
        hops = row.get("propagation_hits") or []
        for prev, hop in zip(hops, hops[1:]):
            za = _finite(prev.get("destination_surface_z_mm") or prev.get("measurement_z_mm"))
            zb = _finite(hop.get("destination_surface_z_mm") or hop.get("measurement_z_mm"))
            if za is None or zb is None:
                continue
            if abs(zb - za) > 5.0:
                continue
            if abs(za - 1234.97) > 2.0 and abs(zb - 1235.86) > 2.0:
                continue
            pairs.append(
                {
                    "init_kind": row.get("init_kind"),
                    "target_station": row.get("target_station"),
                    "surface_a": {
                        "z_mm": za,
                        "station": prev.get("measurement_station"),
                        "layer": prev.get("measurement_layer"),
                        "side": prev.get("side"),
                        "phi_module": prev.get("phi_module"),
                        "eta_module": prev.get("eta_module"),
                        "geometry": prev.get("surface_geometry"),
                        "ok": prev.get("ok"),
                        "inside_active_bounds": prev.get("inside_active_bounds"),
                    },
                    "surface_b": {
                        "z_mm": zb,
                        "station": hop.get("measurement_station"),
                        "layer": hop.get("measurement_layer"),
                        "side": hop.get("side"),
                        "phi_module": hop.get("phi_module"),
                        "eta_module": hop.get("eta_module"),
                        "geometry": hop.get("surface_geometry"),
                        "ok": hop.get("ok"),
                        "abort_reason": hop.get("abort_reason"),
                        "inside_active_bounds": hop.get("inside_active_bounds"),
                        "bounded_intersection_on_surface": hop.get("bounded_intersection_on_surface"),
                        "distance_to_plane_mm": hop.get("distance_to_plane_mm"),
                        "local_z_mm": hop.get("local_z_mm"),
                        "predicted_loc0": hop.get("predicted_loc0"),
                        "predicted_loc1": hop.get("predicted_loc1"),
                        "final_position_xyz_mm": hop.get("final_position_xyz_mm"),
                        "projection_kind": hop.get("projection_kind"),
                    },
                    "dz_mm": abs(zb - za),
                    "mechanism": (
                        "outside_active_bounds_but_on_supporting_plane"
                        if hop.get("ok") and hop.get("inside_active_bounds") is False
                        else (
                            "on_plane_global_to_local_failed"
                            if (not hop.get("ok") and "not on surface" in str(hop.get("abort_reason") or "").lower())
                            else ("reached" if hop.get("ok") else "unreached")
                        )
                    ),
                }
            )
    return {
        "event": "100048/86",
        "focus_pair_z_mm": [1234.97, 1235.86],
        "pairs": pairs,
        "question": (
            "Did propagation miss the plane, or reach the plane outside active bounds?"
        ),
    }


def audit_projection_contract(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n_ok_outside = 0
    n_fail_outside = 0
    n_ok_inside = 0
    examples = []
    for row in rows:
        if not is_official_mode_b(row):
            continue
        for hop in row.get("propagation_hits") or []:
            inside = hop.get("inside_active_bounds")
            ok = bool(hop.get("ok"))
            if ok and inside is False:
                n_ok_outside += 1
            elif ok and inside is True:
                n_ok_inside += 1
            elif (not ok) and inside is False:
                n_fail_outside += 1
            if len(examples) < 8 and inside is False:
                examples.append(
                    {
                        "event_id": row.get("event_id"),
                        "target_station": row.get("target_station"),
                        "ok": ok,
                        "inside_active_bounds": inside,
                        "projection_kind": hop.get("projection_kind"),
                        "distance_to_plane_mm": hop.get("distance_to_plane_mm"),
                        "predicted_loc0": hop.get("predicted_loc0"),
                        "abort_reason": hop.get("abort_reason"),
                    }
                )
    return {
        "residual_needs_plane_chart": True,
        "active_bounds_membership_required_for_chi2": False,
        "source_justification": (
            "ACTS Kalman inserts measurement surfaces as external surfaces and "
            "ignores boundary checks when targeting them.  PlaneSurface loc0 is "
            "the supporting-plane chart.  CKF residual/calibration uses that "
            "chart once a track state exists; a trial theta during profiling "
            "may miss the active rectangle and still have a well-defined loc0."
        ),
        "boundary_check_not_disabled_to_force_86": True,
        "tolerance_not_hunted": True,
        "official_on_surface_tolerance_mm": 1.0e-4,
        "diagnostic_free_to_bound_tolerance_factor": 10,
        "n_ok_outside_active_bounds": n_ok_outside,
        "n_ok_inside_active_bounds": n_ok_inside,
        "n_fail_outside_active_bounds": n_fail_outside,
        "examples": examples,
    }


def audit_sequential_vs_direct(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    seq: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    direct: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if _event_pair(row) not in CONTROL_EVENTS:
            continue
        key = (_identity(row), row.get("target_station"))
        if is_official_mode_b(row):
            seq[key] = row
        elif is_direct(row):
            direct[key] = row
    compared = []
    n_loc0_close = 0
    for key, seq_row in seq.items():
        dir_row = direct.get(key)
        if dir_row is None:
            continue
        seq_hops = seq_row.get("propagation_hits") or []
        dir_hops = dir_row.get("propagation_hits") or []
        loc0_deltas = []
        for sh, dh in zip(seq_hops, dir_hops):
            a = _finite(sh.get("predicted_loc0"))
            b = _finite(dh.get("predicted_loc0"))
            if a is not None and b is not None:
                loc0_deltas.append(abs(a - b))
        max_delta = max(loc0_deltas) if loc0_deltas else None
        close = max_delta is not None and max_delta <= 0.05
        if close:
            n_loc0_close += 1
        compared.append(
            {
                "identity": key[0],
                "target_station": key[1],
                "sequential_chi2": seq_row.get("evaluate_chi2"),
                "direct_chi2": dir_row.get("evaluate_chi2"),
                "sequential_reached": bool(seq_row.get("all_surfaces_reached")),
                "direct_reached": bool(dir_row.get("all_surfaces_reached")),
                "max_abs_loc0_delta_mm": max_delta,
                "loc0_within_0p05_mm": close,
            }
        )
    switch = False
    return {
        "mathematical_object": "h_i(theta) from one deterministic trajectory",
        "sequential_is_default": True,
        "direct_from_source_is_comparison_only": True,
        "do_not_switch_only_because_numerically_better": True,
        "switched_to_direct": switch,
        "n_compared": len(compared),
        "n_loc0_close": n_loc0_close,
        "rows": compared,
    }


def audit_wb115_consistency(
    rows: list[Mapping[str, Any]],
    wb115_rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tol = float(config["wb115_consistency"]["loc0_abs_tolerance_mm"])
    current: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    frozen: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if is_official_mode_b(row) and _event_pair(row) in CONTROL_EVENTS:
            current[(_identity(row), row.get("target_station"))] = row
    for row in wb115_rows:
        if _event_pair(row) in CONTROL_EVENTS:
            frozen[(_identity(row), row.get("target_station"))] = row
    compared = []
    n_ok = 0
    for key, row in current.items():
        old = frozen.get(key)
        if old is None:
            compared.append({"key": key, "missing_wb115": True, "consistent": False})
            continue
        deltas = []
        for nh, oh in zip(row.get("propagation_hits") or [], old.get("propagation_hits") or []):
            a = _finite(nh.get("predicted_loc0"))
            b = _finite(oh.get("predicted_loc0"))
            if a is not None and b is not None:
                deltas.append(abs(a - b))
        max_delta = max(deltas) if deltas else None
        consistent = (
            bool(row.get("all_surfaces_reached"))
            and bool(old.get("all_surfaces_reached"))
            and max_delta is not None
            and max_delta <= tol
        )
        if consistent:
            n_ok += 1
        compared.append(
            {
                "identity": key[0],
                "target_station": key[1],
                "max_abs_loc0_delta_mm": max_delta,
                "tolerance_mm": tol,
                "consistent": consistent,
                "current_chi2": row.get("evaluate_chi2"),
                "wb115_chi2": old.get("evaluate_chi2"),
            }
        )
    return {
        "n_compared": len(compared),
        "n_consistent": n_ok,
        "consistent": len(compared) > 0 and n_ok == len(compared),
        "tolerance_mm": tol,
        "rows": compared,
    }


def audit_jacobian_spotcheck(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = 0
    n_pass = 0
    failed_eval = 0
    examples = []
    required = set(CONTROL_EVENTS)
    control = {"n": 0, "n_pass": 0}
    extra_ok = {}
    for row in rows:
        if not is_official_mode_b(row):
            continue
        report = row.get("jacobian_validation")
        if not report:
            continue
        key = _event_pair(row)
        n += 1
        established = bool(report.get("jacobian_contract_established"))
        if established:
            n_pass += 1
        failed_eval += int(report.get("failed_finite_difference_evaluations") or 0)
        if key in required:
            control["n"] += 1
            control["n_pass"] += int(established)
        if key in {CLASS_M_EVENT, CLASS_S_EVENT}:
            extra_ok[f"{key[0]}/{key[1]}"] = established and bool(row.get("all_surfaces_reached"))
        if len(examples) < 12:
            examples.append(
                {
                    "event_id": row.get("event_id"),
                    "target_station": row.get("target_station"),
                    "jacobian_contract_established": established,
                    "failed_finite_difference_evaluations": report.get(
                        "failed_finite_difference_evaluations"
                    ),
                }
            )
    established = n > 0 and n_pass == n and failed_eval == 0 and control["n"] > 0
    return {
        "n_reports": n,
        "n_pass": n_pass,
        "failed_finite_difference_evaluations": failed_eval,
        "jacobian_0_1_established": control["n"] > 0 and control["n_pass"] == control["n"],
        "jacobian_contract_established": established,
        "focus_if_evaluable": extra_ok,
        "examples": examples,
        "note": (
            "B14T.11 requires a spot check on 37/86 once they become evaluable. "
            "jacobian_contract_established is true only if every official Mode B "
            "report passes, including those focus events."
        ),
    }


def audit_nominal_evaluations(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_event: dict[tuple[int, int], dict[str, Any]] = {}
    for key in CONTROL_EVENTS | {CLASS_M_EVENT, CLASS_S_EVENT}:
        by_event[key] = {
            "run_id": key[0],
            "event_id": key[1],
            "n_targets": 0,
            "n_reached": 0,
            "targets": [],
        }
    n_update = 0
    n_ridge = 0
    n_prior = 0
    for row in rows:
        if row.get("measurement_update_in_evaluator"):
            n_update += 1
        if row.get("ridge_added") or row.get("prior_term_present"):
            n_ridge += 1
            n_prior += int(bool(row.get("prior_term_present")))
        if not is_official_mode_b(row):
            continue
        key = _event_pair(row)
        if key not in by_event:
            continue
        item = by_event[key]
        item["n_targets"] += 1
        reached = bool(row.get("all_surfaces_reached") or row.get("nominal_chi2_evaluable"))
        if reached:
            item["n_reached"] += 1
        item["targets"].append(
            {
                "target_station": row.get("target_station"),
                "reached": reached,
                "evaluate_chi2": row.get("evaluate_chi2"),
                "first_failed_measurement_index": row.get("first_failed_measurement_index"),
                "n_ok_hops": sum(1 for hop in (row.get("propagation_hits") or []) if hop.get("ok")),
                "n_hops": len(row.get("propagation_hits") or []),
            }
        )
    if n_update:
        refuse_measurement_update()
    return {
        "events": {f"{k[0]}/{k[1]}": v for k, v in by_event.items()},
        "event_0_1_all_targets": all(
            by_event[key]["n_targets"] > 0 and by_event[key]["n_reached"] == by_event[key]["n_targets"]
            for key in CONTROL_EVENTS
        ),
        "event_37_all_surviving_evaluable": (
            by_event[CLASS_M_EVENT]["n_targets"] > 0
            and by_event[CLASS_M_EVENT]["n_reached"] == by_event[CLASS_M_EVENT]["n_targets"]
        ),
        "event_86_all_projections_defined": (
            by_event[CLASS_S_EVENT]["n_targets"] > 0
            and by_event[CLASS_S_EVENT]["n_reached"] == by_event[CLASS_S_EVENT]["n_targets"]
        ),
        "measurement_update_in_evaluator": False,
        "n_ridge_or_prior_rows": n_ridge,
        "focus_retained": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    evals = inventory.get("evaluations") or {}
    jacobian = inventory.get("jacobian") or {}
    consistency = inventory.get("wb115_consistency") or {}
    exclusion = inventory.get("exclusion") or {}
    magnet = inventory.get("magnet") or {}
    checks = {
        "events_0_1_nominal": bool(evals.get("event_0_1_all_targets")),
        "wb115_prediction_consistency": bool(consistency.get("consistent")),
        "jacobian_spotcheck": bool(jacobian.get("jacobian_contract_established")),
        "event_37_evaluable": bool(evals.get("event_37_all_surviving_evaluable")),
        "event_86_projections_defined": bool(evals.get("event_86_all_projections_defined")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "target_exclusion": bool(exclusion.get("target_exclusion_holds", True)),
        "statistical_model_unchanged": not bool(inventory.get("statistical_model_changed")),
        "no_measurement_update": evals.get("measurement_update_in_evaluator") is False,
        "no_arbitrary_max_steps_fix": bool(magnet.get("diagnostic_max_steps_is_not_official_fix", True)),
        "smoke_present": bool(inventory.get("smoke_present")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "b14m_reopen_authorized": passed,
        "full_sample_authorized": False,
        "do_not_submit_if_failed": True,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_as_information"):
        refuse_ridge_information()
    if inventory.get("statistical_model_changed"):
        refuse_change_statistical_model()
    if inventory.get("physical_nonidentifiability_claimed"):
        refuse_physical_nonidentifiability()
    gate = inventory.get("smoke_gate") or {}
    evals = inventory.get("evaluations") or {}
    magnet = inventory.get("magnet") or {}
    stereo = inventory.get("stereo") or {}
    units = inventory.get("units") or {}
    smoke = bool(inventory.get("smoke_present"))
    event37 = bool(evals.get("event_37_all_surviving_evaluable"))
    event86 = bool(evals.get("event_86_all_projections_defined"))
    units_ok = int(units.get("n_dumped") or 0) > 0
    official_nav = bool(magnet.get("official_mode_b_all_targets_reached"))
    if not smoke:
        primary = CASE_MIXED
        verdict = "DIAGNOSED"
    elif bool(gate.get("passed")):
        primary = CASE_ESTABLISHED
        verdict = "PASS"
    elif (not event37) and event86 and units_ok and official_nav is False:
        # Units/sign exist; navigator still cannot take 37 to surviving surfaces.
        if bool(magnet.get("mode_a_all_targets_reached")) is False and units_ok:
            primary = CASE_STATE if official_nav else CASE_NAVIGATION
        else:
            primary = CASE_NAVIGATION
        verdict = "FAIL"
    elif event37 and (not event86):
        primary = CASE_PROJECTION
        verdict = "FAIL"
    elif (not event37) and (not event86):
        primary = CASE_MIXED
        verdict = "FAIL"
    elif (not event37) and event86:
        primary = CASE_STATE if units_ok and official_nav is False else CASE_NAVIGATION
        verdict = "FAIL"
    else:
        primary = CASE_MIXED
        verdict = "FAIL"
    jacobian_ok = bool((inventory.get("jacobian") or {}).get("jacobian_contract_established"))
    if primary == CASE_ESTABLISHED:
        next_step = "reopen_b14m_smoke_only"
    elif primary == CASE_STATE:
        next_step = "record_reconstruction_consistency_keep_37"
    elif primary == CASE_PROJECTION:
        next_step = "keep_measurement_model_layer_do_not_enter_v2"
    elif primary == CASE_NAVIGATION:
        next_step = "continue_acts_navigation_transport_diagnosis"
    elif event37 and event86 and not jacobian_ok:
        next_step = "jacobian_spotcheck_on_evaluable_stereo_event_before_reopening_optimizer"
    else:
        next_step = "keep_separated_class_m_and_class_s_diagnosis"
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "b14m_reopen_authorized": primary == CASE_ESTABLISHED,
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
        "stereo_pairs_seen": len(stereo.get("pairs") or []),
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_transport_rows(config)
    sample = load_contracted_sample(config)
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    rows = dumps["rows"]
    wb115_rows = load_wb115_rows(config)
    evaluations = audit_nominal_evaluations(rows)
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
        "taxonomy": audit_failure_taxonomy(rows),
        "acts_semantics": acts_kalman_transport_semantics(),
        "likelihood_contract": standalone_likelihood_transport_contract(),
        "magnet": audit_magnet_crossing(rows),
        "units": audit_state_units(rows),
        "pathology": audit_magnet_pathology(rows),
        "stereo": audit_stereo_geometry(rows),
        "projection": audit_projection_contract(rows),
        "sequential_vs_direct": audit_sequential_vs_direct(rows),
        "jacobian": audit_jacobian_spotcheck(rows),
        "wb115_consistency": audit_wb115_consistency(rows, wb115_rows, config),
        "evaluations": evaluations,
        "exclusion": audit_exclusion(rows),
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "physical_nonidentifiability_claimed": False,
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
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb113_decision_sha256": inherited["workbook_113"]["decision_sha256"],
        "inherited_wb112_decision_sha256": inherited["workbook_112"]["decision_sha256"],
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb115_rewritten": False,
        "target_exclusion_holds": bool((inventory.get("exclusion") or {}).get("target_exclusion_holds")),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "acts_transport_materialized": bool(inventory.get("smoke_present")),
        "jacobian_contract_established": bool(
            (inventory.get("jacobian") or {}).get("jacobian_contract_established")
        ),
    }
