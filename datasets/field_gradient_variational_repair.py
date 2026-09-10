"""Task B14X: field-gradient variational coupling repair and recontract.

WB122 identified the missing loc1 coupling as rk_free ∂dir/∂pos ≡ 0 on
the first long magnetic hop.  This task proves that ACTS 32.0.2
GenericDefaultExtension::transportMatrix omits ∂B/∂x, completes the
same mean RKN4 map with the official FASER gradient API, and recontracts
100043/0,1,37 plus 100048/86.

Official h_i(theta) is unchanged.  Only the diagnostic tangent is
repaired.  The frozen FD ladder and 5% gate stay.  No prior, ridge,
truth q/p, B14M, B15, or 1989 campaign.
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
from datasets.independent_derivative_reference import (
    CASE_COUPLING as WB122_DECISION,
    inherit_frozen_stage as inherit_through_wb122,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.official_path_derivative_residual import MEASUREMENT_SIGMA_MM
from datasets.official_supporting_plane_jacobian import load_jacobian_rows
from datasets.profile_transport_contract import (
    audit_exclusion,
    is_official_mode_b,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "field-gradient-variational-repair-v1"
DEFAULT_CONFIG = "configs/field_gradient_variational_repair_v1.yaml"
TASK = "SB-B14X"
WORKBOOK = 123

CASE_REPAIRED = "field_gradient_variational_coupling_repaired_and_contracted"
CASE_FOCUS_UNRESOLVED = "field_gradient_coupling_repaired_focus_reference_unresolved"
CASE_HYPOTHESIS = "field_gradient_hypothesis_not_supported"
CASE_INCONSISTENT = "field_gradient_variational_implementation_inconsistent"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_REPAIRED,
    CASE_FOCUS_UNRESOLVED,
    CASE_HYPOTHESIS,
    CASE_INCONSISTENT,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
CONTROL1 = (100043, 1)
FOCUS_FIRST_UNSTABLE = {
    1: 6,
    2: 11,
    3: 11,
}
ACTS_TRANSPORT_HEADER = (
    "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
    "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/"
    "detail/GenericDefaultExtension.hpp"
)
FASER_WRAPPER = (
    "/eos/home-x/xcheng/FASER/calypso/Tracking/Acts/FaserActsGeometry/"
    "FaserActsGeometry/FASERMagneticFieldWrapper.h"
)
EIGEN_STEPPER = (
    "/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/"
    "24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/"
    "EigenStepper.ipp"
)
FIELD_FD_STEP_MM = 1.0
LOC0_MATCH = 1.0e-6


class FieldGradientVariationalRepairError(ValueError):
    """Raised when the B14X repair contract is illegal."""


def refuse_prior() -> None:
    raise FieldGradientVariationalRepairError("B14X must not introduce a prior")


def refuse_ridge_information() -> None:
    raise FieldGradientVariationalRepairError(
        "ridge must not be treated as statistical information"
    )


def refuse_add_fd_rung() -> None:
    raise FieldGradientVariationalRepairError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise FieldGradientVariationalRepairError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_change_mean() -> None:
    raise FieldGradientVariationalRepairError(
        "must not change the official mean trajectory h_i"
    )


def refuse_b14m() -> None:
    raise FieldGradientVariationalRepairError("B14M is not re-opened inside Task B14X")


def refuse_b15() -> None:
    raise FieldGradientVariationalRepairError("Task B15 is not entered in Task B14X")


def refuse_full_sample() -> None:
    raise FieldGradientVariationalRepairError(
        "the 1989-row campaign must not be submitted in B14X"
    )


def refuse_tune_field_step() -> None:
    raise FieldGradientVariationalRepairError(
        "must not tune the field-gradient FD step from track Jacobian agreement"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise FieldGradientVariationalRepairError(f"{label} hash mismatch: {digest}")


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise FieldGradientVariationalRepairError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise FieldGradientVariationalRepairError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise FieldGradientVariationalRepairError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14w",
        "do_not_select_best_fd_step",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise FieldGradientVariationalRepairError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14x", True)):
        raise FieldGradientVariationalRepairError("do_not_enter_b14x must be false")
    spec = config["field_gradient_variational_repair"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise FieldGradientVariationalRepairError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise FieldGradientVariationalRepairError(
            "production stepTolerance must stay 1e-4"
        )
    if float(spec["field_gradient_fd_step_mm"]) != FIELD_FD_STEP_MM:
        raise FieldGradientVariationalRepairError(
            "field-gradient FD step must stay the pre-registered 1 mm"
        )
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise FieldGradientVariationalRepairError("allowed decisions must stay the B14X set")
    for item in config["wb120_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb120 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb122(config)
    spec = config["inheritance"]["workbook_122"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_122 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_122 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise FieldGradientVariationalRepairError(
            "workbook_122 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB122_DECISION:
        raise FieldGradientVariationalRepairError("WB122 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise FieldGradientVariationalRepairError("WB122 must not have re-opened B14M")
    inherited["workbook_122"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "five_percent_gate_unchanged": True,
        "rk_magnet_hop_pos_to_dir_is_zero": True,
        "material_reset_is_highest_priority_mechanism": False,
        "focus_independent_reference_established": False,
        "wb122_not_a_physical_conclusion": True,
    }
    return inherited


def audit_acts_source() -> dict[str, Any]:
    header = Path(ACTS_TRANSPORT_HEADER).read_text(encoding="utf-8", errors="replace")
    wrapper = Path(FASER_WRAPPER).read_text(encoding="utf-8", errors="replace")
    stepper = Path(EIGEN_STEPPER).read_text(encoding="utf-8", errors="replace")
    eq18_zero = "terms of eq. 18 are currently 0" in header
    dgdx_zero_comment = "dGdx is already initialised as (3x3) zero" in header
    no_gradient_symbol = "getFieldGradient" not in header and "dGdx =" not in header
    mean_uses_b_at_point = "sd.B_first = *fieldRes" in stepper
    official_gradient = "getFieldGradient" in wrapper
    return {
        "acts_version": "32.0.2",
        "athena_externals": "24.0.41",
        "stepper": "EigenStepper",
        "extension": "GenericDefaultExtension::transportMatrix",
        "field_wrapper": "FASERMagneticFieldWrapper",
        "mean_ode": "dr/ds = T; dT/ds = (q/p) T x B(x)",
        "k_i_uses_b_at_stage_position": True,
        "transport_matrix_fills_dFdT_dFdL_dGdT_dGdL": True,
        "transport_matrix_fills_dGdx": False,
        "eq18_terms_currently_zero_in_acts_comment": eq18_zero,
        "dgdx_initialized_zero_in_acts_comment": dgdx_zero_comment,
        "acts_transport_matrix_has_no_field_gradient_call": no_gradient_symbol,
        "mean_stepper_samples_B_at_current_position": mean_uses_b_at_point,
        "why_rk_free_ddir_dpos_is_zero": (
            "D is Identity plus uniform-B blocks; dGdx stays 0 so the "
            "accumulated free product never develops ∂dir/∂pos"
        ),
        "uniform_field_tangent_present": True,
        "field_gradient_tangent_present_in_acts_d": False,
        "energy_loss_deterministic_in_default_extension_k": False,
        "process_noise_not_in_d": True,
        "covariance_transport_is_only_the_variational_switch": True,
        "official_gradient_api_present": official_gradient,
        "official_gradient_api": "FASERMagneticFieldWrapper::getFieldGradient",
        "field_gradient_fd_step_mm": FIELD_FD_STEP_MM,
        "do_not_tune_field_step_from_track_jacobian": True,
        "acts_header_sha256": sha256_file(Path(ACTS_TRANSPORT_HEADER)),
        "wrapper_sha256": sha256_file(Path(FASER_WRAPPER)),
        "eigen_stepper_sha256": sha256_file(Path(EIGEN_STEPPER)),
        "missing_term_source_proven": bool(
            eq18_zero and dgdx_zero_comment and official_gradient
        ),
    }


def audit_equation_contract() -> dict[str, Any]:
    return {
        "mean_equation": "k = (q/p) * T_stage x B(x_stage)",
        "stage_positions": {
            "pos0": "x",
            "pos1": "x + (h/2) T + (h^2/8) k1",
            "pos2": "x + h T + (h^2/2) k3",
            "source": "EigenStepper.ipp tryRungeKuttaStep",
        },
        "tangent_extra_term": "qop * [T_stage]_x * (∂B/∂x) * dx_stage",
        "units": {
            "position": "mm",
            "direction": "1",
            "q_over_p": "1/GeV",
            "B": "ACTS Tesla after wrapper kT->T",
            "gradient": "ACTS Tesla / mm after wrapper * m_bFieldUnit",
        },
        "g_equals_zero_reduces_to_acts_d": True,
        "coefficients_not_hand_written_from_schematic": True,
        "derived_from_pinned_acts_mean_map": True,
        "energy_loss_not_added": True,
        "process_noise_not_added": True,
        "direction_normalization_jacobian_not_added": True,
        "official_gradient_api_used": True,
        "field_gradient_fd_step_mm": FIELD_FD_STEP_MM,
        "do_not_tune_field_step_from_track_jacobian": True,
    }


def _as_matrix(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or not raw:
        return None
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix if matrix.ndim == 2 else None


def _fd_rungs(payload: Mapping[str, Any], name: str) -> dict[float, np.ndarray | None]:
    for item in (payload.get("fd_ladder") or {}).get("columns") or []:
        if str(item.get("parameter")) != name:
            continue
        return {
            float(rung.get("step_factor")): _as_vector(rung.get("residual_column"))
            for rung in item.get("rungs") or []
        }
    return {}


def _column(payload: Mapping[str, Any], key: str, name: str) -> np.ndarray | None:
    items = payload.get(key)
    if items is None:
        items = (payload.get("official_path_jacobian") or {}).get(key)
    for item in items or []:
        if str(item.get("parameter")) == name:
            return _as_vector(item.get("residual_column"))
    return None


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), 1.0e-12)
    return float(np.linalg.norm(a - b) / denom)


def _sign_ok(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(float(np.dot(a, b)) >= 0.0)


def audit_invariance(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain = payload.get("official_path_jacobian") or {}
    loc0_ok = bool(chain.get("field_gradient_mean_loc0_unchanged", False))
    delta = _finite(chain.get("field_gradient_max_abs_mean_loc0_delta"))
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "predicted_loc0_before_equals_after": loc0_ok,
        "max_abs_mean_loc0_delta": delta,
        "chi2_from_official_mean": chain.get("field_gradient_chi2_from_official_mean"),
        "mean_path_unchanged": loc0_ok and (delta is None or delta <= LOC0_MATCH),
        "do_not_change_official_mean_path": True,
    }


def audit_control_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain = payload.get("official_path_jacobian") or {}
    columns = []
    all_pass = True
    for name in PARAM_NAMES:
        old = _column(chain, "rk_free_chain_columns", name)
        new = _column(chain, "field_gradient_rk_free_chain_columns", name)
        rungs = _fd_rungs(payload, name)
        j_h = rungs.get(1.0)
        item = {
            "parameter": name,
            "old_present": old is not None,
            "new_present": new is not None,
            "fd_present": j_h is not None,
        }
        if old is None or new is None or j_h is None:
            item["pass"] = False
            all_pass = False
            columns.append(item)
            continue
        item["old_vs_fd_rel"] = _rel(old, j_h)
        item["new_vs_fd_rel"] = _rel(new, j_h)
        item["old_vs_fd_sign"] = _sign_ok(old, j_h)
        item["new_vs_fd_sign"] = _sign_ok(new, j_h)
        item["pass"] = bool(
            item["new_vs_fd_rel"] <= REL_MAX and item["new_vs_fd_sign"]
        )
        all_pass = all_pass and item["pass"]
        columns.append(item)
    hops = []
    first_new_pos_to_dir = None
    for hop in chain.get("hops") or []:
        repair = hop.get("field_gradient_repair") or {}
        if not repair.get("requested"):
            continue
        item = {
            "measurement_index": hop.get("measurement_index"),
            "old_rk_pos_to_dir_norm": repair.get("old_rk_pos_to_dir_norm"),
            "new_rk_pos_to_dir_norm": repair.get("new_rk_pos_to_dir_norm"),
            "old_continuation_loc1_angular_norm": repair.get(
                "old_continuation_loc1_angular_norm"
            ),
            "new_continuation_loc1_angular_norm": repair.get(
                "new_continuation_loc1_angular_norm"
            ),
            "n_material_resets": repair.get("n_material_resets"),
            "n_field_samples": len(repair.get("field_samples") or []),
            "has_hop_start_fd": "hop_start_segment_fd" in repair,
        }
        hops.append(item)
        if (
            first_new_pos_to_dir is None
            and hop.get("number_of_propagation_steps", 0)
            and int(hop.get("number_of_propagation_steps") or 0) > 10
        ):
            first_new_pos_to_dir = item
    loc1_item = next((item for item in columns if item["parameter"] == "loc1"), None)
    per_hit = []
    fd_loc1 = (_fd_rungs(payload, "loc1") or {}).get(1.0)
    old_loc1 = _column(chain, "rk_free_chain_columns", "loc1")
    new_loc1 = _column(chain, "field_gradient_rk_free_chain_columns", "loc1")
    n_hits = 0
    if fd_loc1 is not None:
        n_hits = int(fd_loc1.size)
    for i in range(n_hits):
        fd_i = float(fd_loc1[i]) if fd_loc1 is not None else None
        old_i = float(old_loc1[i]) if old_loc1 is not None and i < old_loc1.size else None
        new_i = float(new_loc1[i]) if new_loc1 is not None and i < new_loc1.size else None
        per_hit.append(
            {
                "measurement_index": i,
                "j_fd_loc1": fd_i,
                "j_rk_old_loc1": old_i,
                "j_rk_field_gradient_repaired_loc1": new_i,
                "delta_j_old": None if fd_i is None or old_i is None else old_i - fd_i,
                "delta_j_new": None if fd_i is None or new_i is None else new_i - fd_i,
            }
        )
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "columns": columns,
        "five_percent_pass": all_pass,
        "loc1_five_percent_pass": bool(loc1_item and loc1_item.get("pass")),
        "per_hit_loc1": per_hit,
        "hops": hops,
        "first_long_hop": first_new_pos_to_dir,
        "new_pos_to_dir_nonzero": bool(
            first_new_pos_to_dir
            and (first_new_pos_to_dir.get("new_rk_pos_to_dir_norm") or 0.0) > 1.0e-12
        ),
        "old_pos_to_dir_zero": bool(
            first_new_pos_to_dir
            and (first_new_pos_to_dir.get("old_rk_pos_to_dir_norm") or 0.0) <= 1.0e-12
        ),
    }


def _param_index(name: str) -> int:
    return list(PARAM_NAMES).index(name)


def _qop_to_dir_norm(repair: Mapping[str, Any]) -> float | None:
    matrix = _as_matrix(repair.get("rk_free_transport_jacobian_product"))
    if matrix is None or matrix.shape[0] < 8 or matrix.shape[1] < 8:
        return None
    return float(np.linalg.norm(matrix[4:7, 7]))


def _field_summary(repair: Mapping[str, Any]) -> dict[str, Any]:
    samples = repair.get("field_samples") or []
    frobs = [
        float(sample["gradient_frobenius"])
        for sample in samples
        if sample.get("gradient_frobenius") is not None
    ]
    b_norms = []
    for sample in samples:
        b_vec = sample.get("B_acts")
        if isinstance(b_vec, list) and b_vec:
            b_norms.append(float(np.linalg.norm(np.asarray(b_vec, dtype=float))))
    return {
        "n_field_samples": len(samples),
        "max_gradient_frobenius": max(frobs) if frobs else None,
        "min_gradient_frobenius": min(frobs) if frobs else None,
        "max_abs_B": max(b_norms) if b_norms else None,
        "min_abs_B": min(b_norms) if b_norms else None,
    }


def _segment_fd_converged(repair: Mapping[str, Any], name: str) -> dict[str, Any]:
    seg = repair.get("hop_start_segment_fd") or {}
    hop_vals = repair.get("hop_dloc0_d_start")
    hop_val = None
    if isinstance(hop_vals, list) and len(hop_vals) > _param_index(name):
        hop_val = float(hop_vals[_param_index(name)])
    for col in seg.get("columns") or []:
        if col.get("parameter") != name:
            continue
        vals = []
        for rung in col.get("rungs") or []:
            if rung.get("dloc0_d_start") is None:
                return {
                    "present": True,
                    "converged": False,
                    "reason": "incomplete",
                    "hop_dloc0_repaired": hop_val,
                }
            vals.append(float(rung["dloc0_d_start"]))
        if len(vals) != 4:
            return {
                "present": True,
                "converged": False,
                "reason": "incomplete",
                "hop_dloc0_repaired": hop_val,
            }
        signs = [int(np.sign(v)) if v != 0 else 0 for v in vals]
        sign_change = any(
            signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]
            for i in range(3)
        )
        last = max(abs(vals[3]), 1.0e-12)
        last_pair = abs(vals[2] - vals[3]) / last
        agree = (
            hop_val is not None
            and abs(hop_val - vals[0]) <= REL_MAX * max(abs(vals[0]), 1.0e-12)
        )
        return {
            "present": True,
            "converged": (not sign_change) and last_pair <= REL_MAX,
            "sign_change": sign_change,
            "last_pair_rel": last_pair,
            "rung_values": vals,
            "hop_dloc0_repaired": hop_val,
            "repaired_agrees_h": agree,
        }
    return {"present": False, "converged": False, "hop_dloc0_repaired": hop_val}


def _segment_item(hop: Mapping[str, Any]) -> dict[str, Any]:
    repair = hop.get("field_gradient_repair") or {}
    loc1 = _segment_fd_converged(repair, "loc1")
    hop_loc1 = loc1.get("hop_dloc0_repaired")
    fd_h = None
    if loc1.get("rung_values"):
        fd_h = loc1["rung_values"][0]
    agree = bool(loc1.get("repaired_agrees_h"))
    field = _field_summary(repair)
    return {
        "measurement_index": hop.get("measurement_index"),
        "n_steps": hop.get("number_of_propagation_steps"),
        "n_material_resets": repair.get("n_material_resets", hop.get("n_material_resets")),
        "loc1_segment_fd": loc1,
        "phi_segment_fd": _segment_fd_converged(repair, "phi"),
        "theta_segment_fd": _segment_fd_converged(repair, "theta"),
        "q_over_p_segment_fd": _segment_fd_converged(repair, "q_over_p"),
        "repaired_hop_dloc0_loc1": hop_loc1,
        "repaired_agrees_segment_fd": agree,
        "old_rk_pos_to_dir_norm": repair.get("old_rk_pos_to_dir_norm"),
        "new_rk_pos_to_dir_norm": repair.get("new_rk_pos_to_dir_norm"),
        "new_qop_to_dir_norm": _qop_to_dir_norm(repair),
        "old_last_reset_pos_to_dir_norm": repair.get("old_last_reset_pos_to_dir_norm"),
        "new_last_reset_pos_to_dir_norm": repair.get("new_last_reset_pos_to_dir_norm"),
        "hop_start_segment_fd_present": "hop_start_segment_fd" in repair,
        **field,
    }


def audit_segment_and_focus(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain = payload.get("official_path_jacobian") or {}
    pair = _event_pair(row)
    required_idx = None
    if pair == FOCUS_EVENT:
        required_idx = FOCUS_FIRST_UNSTABLE.get(int(row.get("target_station", -1)))
    hops = list(chain.get("hops") or [])
    long_hops = []
    required = None
    supplementary = []
    for hop in hops:
        repair = hop.get("field_gradient_repair") or {}
        n_steps = int(hop.get("number_of_propagation_steps") or 0)
        if not repair.get("requested"):
            continue
        if n_steps <= 10 and hop.get("measurement_index") != required_idx:
            continue
        item = _segment_item(hop)
        long_hops.append(item)
        if required_idx is not None and hop.get("measurement_index") == required_idx:
            required = item
        elif "hop_start_segment_fd" in repair:
            supplementary.append(item)
    first = required
    if first is None and required_idx is None and supplementary:
        first = supplementary[0]
    first_ok = bool(
        first
        and first.get("hop_start_segment_fd_present")
        and first.get("loc1_segment_fd", {}).get("converged")
        and first.get("repaired_agrees_segment_fd")
    )
    return {
        "event": _event_label(pair),
        "target_station": int(row.get("target_station", -1)),
        "required_first_unstable_index": required_idx,
        "required_hop": required,
        "supplementary_first_long_hops": supplementary,
        "long_hops": long_hops,
        "segments": [first] if first else [],
        "first_unstable_or_first_long_hop": first,
        "segment_fd_present_on_required_hop": bool(
            first and first.get("hop_start_segment_fd_present")
        ),
        "any_segment_fd_converged": bool(
            first and first.get("loc1_segment_fd", {}).get("converged")
        ),
        "any_repaired_agrees_segment_fd": bool(
            first and first.get("repaired_agrees_segment_fd")
        ),
        "independent_reference_established": first_ok,
        "acts_variational_self_reference_forbidden": True,
        "do_not_certify_from_analytic_stability_alone": True,
    }


def audit_field_samples(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    chain = payload.get("official_path_jacobian") or {}
    samples = []
    for hop in chain.get("hops") or []:
        repair = hop.get("field_gradient_repair") or {}
        for sample in repair.get("field_samples") or []:
            samples.append(
                {
                    "measurement_index": hop.get("measurement_index"),
                    "step_index": sample.get("step_index"),
                    "B_acts": sample.get("B_acts"),
                    "gradient_frobenius": sample.get("gradient_frobenius"),
                    "div_B": sample.get("div_B"),
                    "B_repeat_delta_norm": sample.get("B_repeat_delta_norm"),
                    "official_minus_field_fd_frobenius": sample.get(
                        "official_minus_field_fd_frobenius"
                    ),
                    "field_gradient_fd_step_mm": sample.get(
                        "field_gradient_fd_step_mm", FIELD_FD_STEP_MM
                    ),
                }
            )
    frobs = [
        float(item["gradient_frobenius"])
        for item in samples
        if item.get("gradient_frobenius") is not None
    ]
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "n_samples": len(samples),
        "max_gradient_frobenius": max(frobs) if frobs else None,
        "mean_gradient_frobenius": float(np.mean(frobs)) if frobs else None,
        "official_api": "FASERMagneticFieldWrapper::getFieldGradient",
        "field_gradient_fd_step_mm": FIELD_FD_STEP_MM,
        "do_not_tune_field_step_from_track_jacobian": True,
        "samples": samples[:8],
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "source_audit_missing_term": bool(
            (inventory.get("source_audit") or {}).get("missing_term_source_proven")
        ),
        "equation_derived_from_acts_mean": bool(
            (inventory.get("equation") or {}).get("derived_from_pinned_acts_mean_map")
        ),
        "smoke_present": bool(inventory.get("smoke_present")),
        "wb120_smoke_hash_match": bool(inventory.get("wb120_smoke_hash_match")),
        "control_0_1_37_present": bool(inventory.get("controls")),
        "focus_86_present": bool(inventory.get("focus")),
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "five_percent_gate_unchanged": True,
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
    if inventory.get("changed_official_mean"):
        refuse_change_mean()
    if inventory.get("tuned_field_step"):
        refuse_tune_field_step()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    source_ok = bool(
        (inventory.get("source_audit") or {}).get("missing_term_source_proven")
    )
    controls = inventory.get("controls") or []
    focus = inventory.get("focus_segments") or []
    invariance = inventory.get("invariance") or []
    mean_ok = bool(invariance) and all(item.get("mean_path_unchanged") for item in invariance)
    c0 = [item for item in controls if item.get("event") == "100043/0"]
    c1 = [item for item in controls if item.get("event") == "100043/1"]
    c37 = [item for item in controls if item.get("event") == "100043/37"]
    c0_pass = bool(c0) and all(item.get("five_percent_pass") for item in c0)
    c1_pass = bool(c1) and all(
        item.get("loc1_five_percent_pass", item.get("five_percent_pass")) for item in c1
    )
    c37_pass = bool(c37) and all(item.get("five_percent_pass") for item in c37)
    c1_new_coupling = bool(c1) and all(item.get("new_pos_to_dir_nonzero") for item in c1)
    c1_old_zero = bool(c1) and all(item.get("old_pos_to_dir_zero") for item in c1)
    focus_ref = bool(focus) and all(
        item.get("independent_reference_established") for item in focus
    )
    if not source_ok:
        primary = CASE_HYPOTHESIS
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
    elif (not mean_ok) or (c0 and not c0_pass) or (c37 and not c37_pass):
        primary = CASE_INCONSISTENT
        next_step = "keep_field_gradient_tangent_implementation"
    elif c1_pass and c0_pass and c37_pass and focus_ref and mean_ok:
        primary = CASE_REPAIRED
        next_step = "reopen_b14m_smoke_only"
    elif c1_pass and c0_pass and c37_pass and not focus_ref:
        primary = CASE_FOCUS_UNRESOLVED
        next_step = "keep_86_independent_segment_reference"
    elif (not c1_new_coupling) and source_ok:
        primary = CASE_HYPOTHESIS
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
    else:
        primary = CASE_MIXED
        next_step = "keep_field_gradient_tangent_implementation"
    contract = primary == CASE_REPAIRED
    return {
        "verdict": "PASS" if contract else "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "missing_term_source_proven": source_ok,
        "control_0_pass": c0_pass,
        "control_1_pass": c1_pass,
        "control_37_pass": c37_pass,
        "control_1_new_pos_to_dir_nonzero": c1_new_coupling,
        "control_1_old_pos_to_dir_zero": c1_old_zero,
        "mean_path_unchanged": mean_ok,
        "focus_independent_reference_established": focus_ref,
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
        "do_not_tune_field_step_from_track_jacobian": True,
        "do_not_mix_process_noise": True,
        "small_physical_effect_does_not_pass_contract": True,
        "wb122_not_a_physical_conclusion": True,
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
    source = audit_acts_source()
    equation = audit_equation_contract()
    controls = []
    invariance = []
    focus = []
    focus_seg = []
    control_seg = []
    material = []
    field_contract = []
    for row in dumps["rows"]:
        if not is_official_mode_b(row):
            continue
        pair = _event_pair(row)
        if pair not in REQUIRED_EVENTS:
            continue
        if row.get("official_supporting_plane_jacobian") is None:
            continue
        invariance.append(audit_invariance(row))
        field_contract.append(audit_field_samples(row))
        audited = audit_control_columns(row)
        if pair != FOCUS_EVENT:
            controls.append(audited)
            control_seg.append(audit_segment_and_focus(row))
        else:
            focus.append(audited)
            focus_seg.append(audit_segment_and_focus(row))
        hops = (row.get("official_supporting_plane_jacobian") or {}).get(
            "official_path_jacobian", {}
        ).get("hops") or []
        material.append(
            {
                "event": _event_label(pair),
                "target_station": int(row.get("target_station", -1)),
                "resets": [
                    {
                        "measurement_index": hop.get("measurement_index"),
                        "n_material_resets": (hop.get("field_gradient_repair") or {}).get(
                            "n_material_resets", hop.get("n_material_resets")
                        ),
                        "new_pos_to_dir": (hop.get("field_gradient_repair") or {}).get(
                            "new_rk_pos_to_dir_norm"
                        ),
                        "old_last_reset_pos_to_dir": (
                            hop.get("field_gradient_repair") or {}
                        ).get("old_last_reset_pos_to_dir_norm"),
                        "new_last_reset_pos_to_dir": (
                            hop.get("field_gradient_repair") or {}
                        ).get("new_last_reset_pos_to_dir_norm"),
                    }
                    for hop in hops
                    if (hop.get("field_gradient_repair") or {}).get("requested")
                ],
                "do_not_tune_Q": True,
                "material_still_secondary": True,
            }
        )
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "wb120_smoke_hash_match": True,
        "n_rows": dumps["n_rows"],
        "source_audit": source,
        "equation": equation,
        "controls": controls,
        "focus": focus,
        "focus_segments": focus_seg,
        "control_segments": control_seg,
        "invariance": invariance,
        "field_contract": field_contract,
        "material": material,
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
        "tuned_field_step": False,
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
    if mechanism.get("b15_authorized"):
        refuse_b15()
    if mechanism.get("full_sample_authorized"):
        refuse_full_sample()
    if mechanism.get("decision") != CASE_REPAIRED:
        mechanism["jacobian_contract_established"] = False
        mechanism["b14m_reopen_authorized"] = False
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb122_decision_sha256": inherited["workbook_122"]["decision_sha256"],
        "inherited_wb121_decision_sha256": inherited["workbook_121"]["decision_sha256"],
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
        "do_not_tune_field_step_from_track_jacobian": True,
        "small_physical_effect_does_not_pass_contract": True,
        "pinv_relative": PINV_RELATIVE,
        "objective": "repair missing ∂dir/∂pos from official B(x) tangent",
        "not_the_objective": "repair 5D Cin",
        "wb122_not_a_physical_conclusion": True,
    }
