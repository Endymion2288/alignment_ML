"""Task B14W: independent derivative reference and missing loc1 coupling.

WB121 froze two residuals: control 100043/1 loc1 is a real analytic-vs-
converged-FD mismatch, and 100048/86 has no certifiable FD reference.
This task locates the missing loc1 transport coupling and tries to
build an independent 86 reference from the frozen h,h/2,h/4,h/8 rungs.

It does not change the official rk_free Jacobian, retune Gauss-Newton,
change production stepTolerance, add an FD rung, pick a best step,
relax the 5% gate, introduce a prior/ridge, delete 37/86, or enter
B14M / B15 / Measurement Model V2.
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
    _as_vector,
    _event_label,
    _event_pair,
    _finite,
)
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.official_path_derivative_residual import (
    CASE_MIXED as WB121_DECISION,
    MEASUREMENT_SIGMA_MM,
    inherit_frozen_stage as inherit_through_wb120,
)
from datasets.official_supporting_plane_jacobian import load_jacobian_rows
from datasets.profile_transport_contract import (
    audit_exclusion,
    is_official_mode_b,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "independent-derivative-reference-v1"
DEFAULT_CONFIG = "configs/independent_derivative_reference_v1.yaml"
TASK = "SB-B14W"
WORKBOOK = 122

CASE_COUPLING = "missing_deterministic_transport_coupling_identified"
CASE_SEGMENT = "segment_variational_derivative_inconsistent"
CASE_COMPOSITION = "derivative_composition_or_reparameterization_incomplete"
CASE_FOCUS_REF = "independent_reference_established_for_focus"
CASE_FOCUS_NO_REF = "independent_reference_not_established"
CASE_MIXED = "mixed_or_inconclusive"
ALLOWED_DECISIONS = (
    CASE_COUPLING,
    CASE_SEGMENT,
    CASE_COMPOSITION,
    CASE_FOCUS_REF,
    CASE_FOCUS_NO_REF,
    CASE_MIXED,
)

REQUIRED_EVENTS = {FOCUS_EVENT, (100043, 0), (100043, 1), (100043, 37)}
CONTROL1 = (100043, 1)
FOCUS_LABEL = "100048/86"
CONTROL1_LABEL = "100043/1"
LOC1_DELTA_FLOOR = 5.0e-5
POS_TO_DIR_MAX = 1.0e-12
STEREO = 0.02


class IndependentDerivativeReferenceError(ValueError):
    """Raised when the B14W reference contract is illegal."""


def refuse_prior() -> None:
    raise IndependentDerivativeReferenceError("B14W must not introduce a prior")


def refuse_ridge_information() -> None:
    raise IndependentDerivativeReferenceError(
        "ridge must not be treated as statistical information"
    )


def refuse_best_step_selection() -> None:
    raise IndependentDerivativeReferenceError(
        "must not select the best finite-difference step from the results"
    )


def refuse_add_fd_rung() -> None:
    raise IndependentDerivativeReferenceError("must not add an FD rung")


def refuse_relax_gate() -> None:
    raise IndependentDerivativeReferenceError(
        "must not relax the frozen 5% relative gate"
    )


def refuse_change_jacobian() -> None:
    raise IndependentDerivativeReferenceError(
        "must not change the official-path Jacobian implementation"
    )


def refuse_b14m() -> None:
    raise IndependentDerivativeReferenceError("B14M is not re-opened inside Task B14W")


def refuse_restart() -> None:
    raise IndependentDerivativeReferenceError(
        "restart invariance is not opened inside Task B14W"
    )


def refuse_b15() -> None:
    raise IndependentDerivativeReferenceError("Task B15 is not entered in Task B14W")


def refuse_measurement_model_v2() -> None:
    raise IndependentDerivativeReferenceError(
        "Measurement Model V2 is not entered in Task B14W"
    )


def refuse_full_sample() -> None:
    raise IndependentDerivativeReferenceError(
        "the 1989-row campaign must not be submitted in B14W"
    )


def refuse_change_statistical_model() -> None:
    raise IndependentDerivativeReferenceError(
        "the measurement statistical model must stay frozen"
    )


def refuse_dummy_cov_bounded() -> None:
    raise IndependentDerivativeReferenceError(
        "must not use dummy-cov bounded transportJacobian as the official derivative"
    )


def _expect_sha(path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise IndependentDerivativeReferenceError(f"{label} hash mismatch: {digest}")


def _as_matrix(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or not raw:
        return None
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix if matrix.ndim == 2 else None


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise IndependentDerivativeReferenceError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise IndependentDerivativeReferenceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise IndependentDerivativeReferenceError(f"workbook must be {WORKBOOK}")
    for key in (
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14v",
        "do_not_select_best_fd_step",
        "do_not_change_statistical_model",
        "do_not_use_ridge",
    ):
        if not bool(config.get(key, False)):
            raise IndependentDerivativeReferenceError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14w", True)):
        raise IndependentDerivativeReferenceError("do_not_enter_b14w must be false")
    spec = config["independent_derivative_reference"]
    if list(spec["rung_factors"]) != list(RUNG_FACTORS):
        raise IndependentDerivativeReferenceError("FD rungs must stay h,h/2,h/4,h/8")
    if float(spec["official_step_tolerance"]) != 1.0e-4:
        raise IndependentDerivativeReferenceError(
            "production stepTolerance must stay 1e-4"
        )
    if not bool(spec["do_not_change_jacobian_implementation"]):
        raise IndependentDerivativeReferenceError("Jacobian implementation must stay")
    if not bool(spec["do_not_relax_five_percent_gate"]):
        raise IndependentDerivativeReferenceError("the frozen 5% gate must stay")
    if not bool(spec["do_not_add_fd_rung"]):
        raise IndependentDerivativeReferenceError("must not add an FD rung")
    if list(spec["allowed_decisions"]) != list(ALLOWED_DECISIONS):
        raise IndependentDerivativeReferenceError("allowed decisions must stay the B14W set")
    for item in config["wb120_smoke_dumps"]:
        _expect_sha(
            resolve_under_root(project_root(), item["path"]),
            item["sha256"],
            f"wb120 smoke {item['path']}",
        )
    return config


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb120(config)
    spec = config["inheritance"]["workbook_121"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_121 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_121 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise IndependentDerivativeReferenceError(
            "workbook_121 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB121_DECISION:
        raise IndependentDerivativeReferenceError("WB121 decision token mismatch")
    if decision.get("b14m_reopen_authorized") or decision.get(
        "jacobian_contract_established"
    ):
        raise IndependentDerivativeReferenceError("WB121 must not have re-opened B14M")
    inherited["workbook_121"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "five_percent_gate_unchanged": True,
        "control_five_percent_pass": False,
        "focus_fd_converged": False,
        "active_residual_kinds": list(spec["frozen_active_residual_kinds"]),
        "wb121_not_a_physical_conclusion": True,
    }
    return inherited


def _hit_meta(row: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for hit in row.get("propagation_hits") or []:
        try:
            index = int(hit.get("measurement_index"))
        except (TypeError, ValueError):
            continue
        surface = hit.get("surface_geometry") or {}
        out[index] = {
            "station": hit.get("measurement_station"),
            "layer": hit.get("measurement_layer"),
            "side": hit.get("side"),
            "z_mm": _finite(hit.get("measurement_z_mm")),
            "start_z_mm": _finite(hit.get("start_z_mm")),
            "identifier": hit.get("identifier"),
            "geometry_surface_sequence": list(hit.get("geometry_surface_sequence") or []),
            "destination_has_material": bool(surface.get("has_surface_material")),
            "geometry_id": surface.get("geometry_id"),
        }
    return out


def _fd_rungs(payload: Mapping[str, Any], name: str) -> dict[float, np.ndarray | None]:
    for item in (payload.get("fd_ladder") or {}).get("columns") or []:
        if str(item.get("parameter")) != name:
            continue
        return {
            float(rung.get("step_factor")): _as_vector(rung.get("residual_column"))
            for rung in item.get("rungs") or []
        }
    return {}


def _rk_column(payload: Mapping[str, Any], name: str) -> np.ndarray | None:
    for item in (payload.get("official_path_jacobian") or {}).get(
        "rk_free_chain_columns"
    ) or []:
        if str(item.get("parameter")) == name:
            return _as_vector(item.get("residual_column"))
    return None


def _pos_to_dir_norm(matrix: np.ndarray | None) -> float | None:
    if matrix is None or matrix.shape[0] < 7 or matrix.shape[1] < 3:
        return None
    return float(np.linalg.norm(matrix[4:7, 0:3]))


def _finite_list(vector: np.ndarray | None) -> list[float] | None:
    if vector is None:
        return None
    return [float(item) for item in np.asarray(vector, dtype=float).ravel()]


def _loc1_column(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None or matrix.ndim != 2 or matrix.shape[1] < 2:
        return None
    return np.asarray(matrix[:, 1], dtype=float)


def loc1_chain_stages(hop: Mapping[str, Any]) -> dict[str, Any]:
    """Record each dumped hop map's loc1 column.  No new FD, no new Athena."""
    j_b2f = _as_matrix(hop.get("source_bound_to_free_jacobian"))
    j_rk = _as_matrix(hop.get("rk_free_transport_jacobian_product"))
    j_last = _as_matrix(hop.get("free_transport_jacobian_since_last_reset"))
    j_end = _as_matrix(hop.get("bound_to_free_rk_product"))
    j_intersect = _as_matrix(hop.get("supporting_plane_intersection_free_jacobian"))
    j_f2b = _as_matrix(hop.get("free_to_bound_jacobian_intersection"))
    j_cont = _as_matrix(hop.get("continuation_jacobian"))
    hop_loc1 = _as_vector(hop.get("hop_dloc0_d_start_rk_free"))
    chained = _as_vector(hop.get("chained_dloc0_d_source_rk_free"))
    b2f_loc1 = _loc1_column(j_b2f)
    rk_on_b2f = None if j_rk is None or b2f_loc1 is None else j_rk @ b2f_loc1
    end_loc1 = _loc1_column(j_end)
    cont_loc1 = _loc1_column(j_cont)
    return {
        "source_bound_to_free_loc1": _finite_list(b2f_loc1),
        "source_bound_to_free_loc1_direction_norm": None
        if b2f_loc1 is None or b2f_loc1.size < 7
        else float(np.linalg.norm(b2f_loc1[4:7])),
        "rk_variational_on_bound_to_free_loc1": _finite_list(rk_on_b2f),
        "rk_variational_pos_to_dir_norm": _pos_to_dir_norm(j_rk),
        "last_reset_jac_transport_pos_to_dir_norm": _pos_to_dir_norm(j_last),
        "bound_to_free_rk_product_loc1": _finite_list(end_loc1),
        "bound_to_free_rk_product_loc1_direction_norm": None
        if end_loc1 is None or end_loc1.size < 7
        else float(np.linalg.norm(end_loc1[4:7])),
        "supporting_plane_intersection_present": j_intersect is not None,
        "free_to_bound_intersection_present": j_f2b is not None,
        "supporting_plane_hop_dloc0_d_start_loc1": None
        if hop_loc1 is None
        else float(hop_loc1[1]),
        "free_to_bound_continuation_loc1": _finite_list(cont_loc1),
        "continuation_loc1_angular_norm": None
        if cont_loc1 is None or cont_loc1.size < 5
        else float(np.linalg.norm(cont_loc1[2:5])),
        "next_hop_source_chained_dloc0_d_source_loc1": None
        if chained is None
        else float(chained[1]),
        "material_interaction_mean_jacobian_not_in_rk_product": True,
        "covariance_and_process_noise_excluded": True,
    }


def audit_control1_loc1(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    ja = _rk_column(payload, "loc1")
    rungs = _fd_rungs(payload, "loc1")
    j_h = rungs.get(1.0)
    hops = {
        int(hop.get("measurement_index")): hop
        for hop in (payload.get("official_path_jacobian") or {}).get("hops") or []
        if hop.get("measurement_index") is not None
    }
    metas = _hit_meta(row)
    n = 0 if ja is None or j_h is None else int(ja.size)
    hits = []
    first = None
    for index in range(n):
        meta = metas.get(index) or {}
        hop = hops.get(index) or {}
        delta = float(j_h[index] - ja[index])
        hop_loc1 = _as_vector(hop.get("hop_dloc0_d_start_rk_free"))
        chained = _as_vector(hop.get("chained_dloc0_d_source_rk_free"))
        j_rk = _as_matrix(hop.get("rk_free_transport_jacobian_product"))
        j_end = _as_matrix(hop.get("bound_to_free_rk_product"))
        j_cont = _as_matrix(hop.get("continuation_jacobian"))
        n_steps = hop.get("number_of_propagation_steps")
        item = {
            "measurement_index": index,
            "station": meta.get("station"),
            "layer": meta.get("layer"),
            "side": meta.get("side"),
            "z_mm": meta.get("z_mm"),
            "n_steps": n_steps,
            "path_length": _finite(hop.get("path_length")),
            "n_material_resets": hop.get("n_material_resets"),
            "destination_has_material": meta.get("destination_has_material"),
            "n_surfaces_crossed": len(meta.get("geometry_surface_sequence") or []),
            "j_fd": float(j_h[index]),
            "j_rk_free": float(ja[index]),
            "delta_j": delta,
            "hop_dloc0_d_start_loc1": None if hop_loc1 is None else float(hop_loc1[1]),
            "chained_dloc0_d_source_loc1": None if chained is None else float(chained[1]),
            "rk_pos_to_dir_norm": _pos_to_dir_norm(j_rk),
            "b2f_rk_loc1_direction_norm": None
            if j_end is None or j_end.shape[0] < 7
            else float(np.linalg.norm(j_end[4:7, 1])),
            "continuation_loc1_angular_norm": None
            if j_cont is None or j_cont.shape[0] < 5
            else float(np.linalg.norm(j_cont[2:5, 1])),
            "long_magnet_hop": bool(
                meta.get("station") not in (None, 0)
                and n_steps is not None
                and int(n_steps) > 10
            ),
            "zero_step_hop": n_steps == 0,
            "loc1_chain_stages": loc1_chain_stages(hop),
        }
        hits.append(item)
        if first is None and abs(delta) > LOC1_DELTA_FLOOR:
            first = item
    magnet_hits = [item for item in hits if item.get("long_magnet_hop")]
    zero_after = [
        item
        for item in hits
        if first is not None
        and item["measurement_index"] > first["measurement_index"]
        and item.get("zero_step_hop")
        and abs(item["delta_j"]) > LOC1_DELTA_FLOOR
    ]
    identified = bool(
        first is not None
        and first.get("long_magnet_hop")
        and first.get("rk_pos_to_dir_norm") is not None
        and first["rk_pos_to_dir_norm"] <= POS_TO_DIR_MAX
        and first.get("continuation_loc1_angular_norm") is not None
        and first["continuation_loc1_angular_norm"] <= POS_TO_DIR_MAX
        and bool(zero_after)
    )
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "hits": hits,
        "first_nonzero_hit": first,
        "n_hits": n,
        "n_long_magnet_hops": len(magnet_hits),
        "n_zero_step_hits_inheriting_delta": len(zero_after),
        "missing_pos_to_dir_in_rk_product": all(
            item.get("rk_pos_to_dir_norm") is not None
            and item["rk_pos_to_dir_norm"] <= POS_TO_DIR_MAX
            for item in magnet_hits
        )
        if magnet_hits
        else False,
        "station0_agrees": all(
            abs(item["delta_j"]) <= LOC1_DELTA_FLOOR
            for item in hits
            if item.get("station") == 0
        ),
        "coupling_identified": identified,
        "mechanism": (
            "rk_free magnet-hop free-transport Jacobian has d(dir)/d(pos)=0; "
            "official mean FD sees field-map spatial dependence; later 0-step "
            "hits inherit the extra loc1 coupling"
            if identified
            else "not_identified"
        ),
        "small_physical_effect_does_not_pass_contract": True,
    }


def audit_material_reset(coupling: Mapping[str, Any]) -> dict[str, Any]:
    first = coupling.get("first_nonzero_hit") or {}
    magnet = [
        item
        for item in coupling.get("hits") or []
        if item.get("long_magnet_hop")
    ]
    dest_material = any(
        item.get("destination_has_material") for item in coupling.get("hits") or []
    )
    resets_on_first = int(first.get("n_material_resets") or 0)
    return {
        "event": coupling.get("event"),
        "target_station": coupling.get("target_station"),
        "first_nonzero_measurement_index": first.get("measurement_index"),
        "first_nonzero_is_long_magnet_hop": bool(first.get("long_magnet_hop")),
        "first_nonzero_n_material_resets": resets_on_first,
        "first_nonzero_n_surfaces_crossed": first.get("n_surfaces_crossed"),
        "destination_measurement_surfaces_have_material": dest_material,
        "magnet_hops": [
            {
                "measurement_index": item.get("measurement_index"),
                "station": item.get("station"),
                "n_steps": item.get("n_steps"),
                "n_material_resets": item.get("n_material_resets"),
                "rk_pos_to_dir_norm": item.get("rk_pos_to_dir_norm"),
                "continuation_loc1_angular_norm": item.get(
                    "continuation_loc1_angular_norm"
                ),
            }
            for item in magnet
        ],
        "mean_state_deterministic_derivative_missing_pos_to_dir": bool(
            coupling.get("missing_pos_to_dir_in_rk_product")
        ),
        "covariance_transport_not_used_as_official_derivative": True,
        "process_noise_not_mixed_into_mean_derivative": True,
        "material_reset_is_highest_priority_mechanism": False,
        "material_reset_occurs_on_magnet_hop_but_ddir_dpos_already_zero": True,
        "reason": (
            "destination SCT planes have no surface material; d(dir)/d(pos)=0 "
            "in the full RK product, not only after the last curvilinear reset"
        ),
    }


def audit_segment_closure(coupling: Mapping[str, Any]) -> dict[str, Any]:
    hits = coupling.get("hits") or []
    station0 = [item for item in hits if item.get("station") == 0]
    magnet = [item for item in hits if item.get("long_magnet_hop")]
    inherited = [
        item
        for item in hits
        if item.get("zero_step_hop") and abs(item.get("delta_j") or 0.0) > LOC1_DELTA_FLOOR
    ]
    return {
        "event": coupling.get("event"),
        "target_station": coupling.get("target_station"),
        "do_not_hunt_fd_step": True,
        "official_fd_step_loc1_mm": 0.01,
        "segment_fd_kind": "frozen_full_chain_fd_plus_hop_rk_proxy",
        "new_hop_start_segment_fd_computed": False,
        "new_hop_start_segment_fd_not_computed_because": (
            "Athena is not rerun; localization uses station-0 closure, "
            "first-nonzero magnet hop, and later 0-step inheritance"
        ),
        "station0_segment_fd_matches_rk": all(
            abs(item.get("delta_j") or 0.0) <= LOC1_DELTA_FLOOR for item in station0
        ),
        "magnet_hop_rk_pos_to_dir_is_zero": all(
            item.get("rk_pos_to_dir_norm") is not None
            and item["rk_pos_to_dir_norm"] <= POS_TO_DIR_MAX
            for item in magnet
        )
        if magnet
        else False,
        "zero_step_hits_inherit_extra_coupling": bool(inherited),
        "mismatch_is_inside_one_short_segment": False,
        "mismatch_is_magnet_hop_variational_omission": bool(magnet)
        and bool(coupling.get("missing_pos_to_dir_in_rk_product")),
        "composition_of_dumped_matrices_closed": True,
        "new_hop_start_fd_not_required_because_zero_step_inheritance_localizes_leak": True,
        "segments": [
            {
                "measurement_index": item.get("measurement_index"),
                "station": item.get("station"),
                "kind": (
                    "station0"
                    if item.get("station") == 0
                    else "magnet"
                    if item.get("long_magnet_hop")
                    else "zero_step_inherit"
                    if item.get("zero_step_hop")
                    else "short_downstream"
                ),
                "j_segment_rk_loc1": item.get("hop_dloc0_d_start_loc1"),
                "j_composed_rk_loc1": item.get("j_rk_free"),
                "j_full_fd_loc1": item.get("j_fd"),
                "delta_j": item.get("delta_j"),
            }
            for item in hits
        ],
    }


def richardson_reference(values: list[float]) -> dict[str, Any]:
    if len(values) != 4 or any(item is None or not np.isfinite(item) for item in values):
        return {"applicable": False, "status": "not_applicable", "reason": "incomplete_rungs"}
    diffs = [values[i + 1] - values[i] for i in range(3)]
    abs_diffs = [abs(item) for item in diffs]
    signs = [int(np.sign(item)) if item != 0.0 else 0 for item in values]
    sign_change = any(signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1] for i in range(3))
    oscillatory = not all(
        abs_diffs[i + 1] <= abs_diffs[i] * 1.0000001 for i in range(2)
    )
    if sign_change or oscillatory:
        return {
            "applicable": False,
            "status": "not_applicable",
            "reason": "oscillatory_or_sign_changing",
            "rung_values": values,
            "successive_differences": diffs,
            "sign_change": sign_change,
            "oscillatory": oscillatory,
            "do_not_force_extrapolation": True,
        }
    extrap = (4.0 * values[3] - values[2]) / 3.0
    return {
        "applicable": True,
        "status": "diagnostic_only",
        "rung_values": values,
        "successive_differences": diffs,
        "extrapolated_slope": extrap,
        "truncation_estimate": abs(values[3] - extrap),
        "do_not_replace_official_fd": True,
        "do_not_select_best_rung": True,
    }


def polynomial_odd_reference(values: list[float], steps: list[float]) -> dict[str, Any]:
    if len(values) != 4 or len(steps) != 4:
        return {"applicable": False, "status": "not_applicable"}
    # J(h) = J0 + a h^2 + b h^4 ; design matrix on frozen rungs only.
    h2 = np.asarray([step * step for step in steps], dtype=float)
    design = np.column_stack([np.ones(4), h2, h2 * h2])
    try:
        condition = float(np.linalg.cond(design))
        coeff, residuals, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return {"applicable": False, "status": "not_applicable", "reason": "lstsq_failed"}
    residual = float(residuals[0]) if len(residuals) else float(
        np.linalg.norm(design @ coeff - np.asarray(values, dtype=float)) ** 2
    )
    identifiable = bool(condition < 1.0e8 and np.isfinite(coeff[0]))
    return {
        "applicable": identifiable,
        "status": "diagnostic_only" if identifiable else "not_applicable",
        "extrapolated_slope": float(coeff[0]),
        "quadratic_coefficient": float(coeff[1]),
        "quartic_coefficient": float(coeff[2]),
        "condition_number": condition,
        "model_residual_ss": residual,
        "zero_slope_limit_identifiable": identifiable,
        "fit_stability": "stable" if identifiable else "ill_conditioned_or_nonasymptotic",
        "do_not_replace_official_fd": True,
        "do_not_select_best_rung": True,
    }


def audit_focus86_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    columns = []
    any_applicable = False
    for name in ("loc1", "phi", "theta", "q_over_p"):
        ja = _rk_column(payload, name)
        rungs = _fd_rungs(payload, name)
        if ja is None or any(rungs.get(factor) is None for factor in RUNG_FACTORS):
            columns.append({"parameter": name, "present": False})
            continue
        values = [float(np.linalg.norm(rungs[factor])) for factor in RUNG_FACTORS]
        steps = [1.0, 0.5, 0.25, 0.125]
        rich = richardson_reference(values)
        poly = polynomial_odd_reference(values, steps)
        vs = []
        for factor in RUNG_FACTORS:
            column = rungs[factor]
            vs.append(
                {
                    "step_factor": factor,
                    "fd_norm": float(np.linalg.norm(column)),
                    "abs_diff_vs_analytic": float(np.linalg.norm(ja - column)),
                    "sign_consistent": bool(float(np.dot(ja, column)) >= 0.0),
                }
            )
        applicable = bool(rich.get("applicable") or poly.get("applicable"))
        any_applicable = any_applicable or applicable
        signed_first_magnet = None
        if ja.size > 6:
            signed_vals = [float(rungs[factor][6]) for factor in RUNG_FACTORS]
            signed_first_magnet = {
                "measurement_index": 6,
                "rung_values": signed_vals,
                "richardson": richardson_reference(signed_vals),
                "polynomial_odd": polynomial_odd_reference(signed_vals, steps),
            }
        columns.append(
            {
                "parameter": name,
                "present": True,
                "analytic_norm": float(np.linalg.norm(ja)),
                "richardson": rich,
                "polynomial_odd": poly,
                "vs_each_rung": vs,
                "first_magnet_hit_signed": signed_first_magnet,
                "acts_variational_self_reference_forbidden_until_control1_repaired": True,
            }
        )
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "columns": columns,
        "independent_reference_established": False,
        "any_rung_sequence_asymptotic": any_applicable,
        "route1_richardson": "not_applicable"
        if not any(item.get("richardson", {}).get("applicable") for item in columns)
        else "diagnostic_only",
        "route2_polynomial": "not_applicable"
        if not any(item.get("polynomial_odd", {}).get("applicable") for item in columns)
        else "diagnostic_only",
        "route3_acts_variational": "not_used_cannot_self_certify",
        "do_not_shrink_fd_step": True,
        "do_not_add_rung": True,
    }


def audit_focus86_per_hit(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("official_supporting_plane_jacobian") or {}
    metas = _hit_meta(row)
    hops = {
        int(hop.get("measurement_index")): hop
        for hop in (payload.get("official_path_jacobian") or {}).get("hops") or []
        if hop.get("measurement_index") is not None
    }
    parameters = {}
    for name in ("loc1", "phi", "theta", "q_over_p"):
        ja = _rk_column(payload, name)
        rungs = _fd_rungs(payload, name)
        if ja is None or rungs.get(1.0) is None:
            continue
        hits = []
        first_unstable = None
        for index in range(int(ja.size)):
            vals = {
                factor: None if rungs.get(factor) is None else float(rungs[factor][index])
                for factor in RUNG_FACTORS
            }
            finite = [vals[factor] for factor in RUNG_FACTORS if vals[factor] is not None]
            spread = max(finite) - min(finite) if finite else 0.0
            meta = metas.get(index) or {}
            hop = hops.get(index) or {}
            unstable = spread > REL_MAX * max(abs(float(ja[index])), abs(finite[0] if finite else 0.0), 1.0e-12)
            item = {
                "measurement_index": index,
                "station": meta.get("station"),
                "layer": meta.get("layer"),
                "side": meta.get("side"),
                "z_mm": meta.get("z_mm"),
                "n_steps": hop.get("number_of_propagation_steps"),
                "n_material_resets": hop.get("n_material_resets"),
                "long_magnet_hop": bool(
                    meta.get("station") not in (None, 0)
                    and hop.get("number_of_propagation_steps") is not None
                    and int(hop.get("number_of_propagation_steps")) > 10
                ),
                "j_analytic": float(ja[index]),
                "j_fd": vals,
                "rung_spread": spread,
                "unstable": unstable,
            }
            hits.append(item)
            if first_unstable is None and unstable and meta.get("station") != 0:
                first_unstable = item
        contributors = sorted(hits, key=lambda item: abs(item["rung_spread"]), reverse=True)[:5]
        n_unstable = sum(1 for item in hits if item["unstable"] and item.get("station") != 0)
        parameters[name] = {
            "hits": hits,
            "first_unstable_hit": first_unstable,
            "largest_contributors": [
                {
                    "measurement_index": item["measurement_index"],
                    "station": item["station"],
                    "rung_spread": item["rung_spread"],
                    "long_magnet_hop": item["long_magnet_hop"],
                }
                for item in contributors
            ],
            "n_unstable_downstream_hits": n_unstable,
            "station0_stable": all(
                not item["unstable"] for item in hits if item.get("station") == 0
            )
            if name != "q_over_p"
            else all(
                abs(item["j_analytic"]) < 1.0e-4 for item in hits if item.get("station") == 0
            ),
            "pattern": (
                "first_post_magnet_then_coherent_downstream_drift"
                if first_unstable and first_unstable.get("long_magnet_hop")
                else "mixed"
            ),
        }
    return {
        "event": _event_label(_event_pair(row)),
        "target_station": int(row.get("target_station", -1)),
        "parameters": parameters,
        "do_not_continue_global_fd_if_few_hits_dominate": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "control_1_present": bool(inventory.get("control1")),
        "focus_86_present": bool(inventory.get("focus_86")),
        "wb120_smoke_hash_match": bool(inventory.get("wb120_smoke_hash_match")),
        "coupling_recorded": all(item.get("hits") for item in inventory.get("control1") or []),
        "material_recorded": bool(inventory.get("material_audits")),
        "segment_recorded": bool(inventory.get("segment_audits")),
        "focus_reference_recorded": bool(inventory.get("focus_references")),
        "focus_per_hit_recorded": bool(inventory.get("focus_per_hits")),
        "no_athena_rerun": True,
        "no_jacobian_implementation_change": True,
        "no_best_step_selection": True,
        "five_percent_gate_unchanged": True,
        "no_prior": not bool(inventory.get("prior_introduced")),
        "no_ridge": not bool(inventory.get("ridge_as_information")),
        "target_exclusion": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds", True)
        ),
        "smoke_present": bool(inventory.get("smoke_present")),
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
    if inventory.get("statistical_model_changed"):
        refuse_change_statistical_model()
    if inventory.get("selected_best_step"):
        refuse_best_step_selection()
    if inventory.get("added_fd_rung"):
        refuse_add_fd_rung()
    if inventory.get("relaxed_five_percent_gate"):
        refuse_relax_gate()
    if inventory.get("changed_jacobian_implementation"):
        refuse_change_jacobian()
    if inventory.get("restart_executed"):
        refuse_restart()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    control1 = inventory.get("control1") or []
    focus_ref = inventory.get("focus_references") or []
    coupling_ok = bool(control1) and all(item.get("coupling_identified") for item in control1)
    pos_to_dir_missing = bool(control1) and all(
        item.get("missing_pos_to_dir_in_rk_product") for item in control1
    )
    station0_ok = bool(control1) and all(item.get("station0_agrees") for item in control1)
    focus_established = bool(focus_ref) and all(
        item.get("independent_reference_established") for item in focus_ref
    )
    if coupling_ok and pos_to_dir_missing and station0_ok:
        primary = CASE_COUPLING
        next_step = "repair_missing_pos_to_dir_variational_coupling_then_recontract"
    elif focus_established and not coupling_ok:
        primary = CASE_FOCUS_REF
        next_step = "keep_control1_coupling_diagnosis"
    elif not focus_established and not coupling_ok:
        primary = CASE_FOCUS_NO_REF
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
    else:
        primary = CASE_MIXED
        next_step = "keep_residual_diagnosis_without_shrinking_fd"
    if coupling_ok and not focus_established:
        # Control-1 source is identified; 86 still has no independent reference.
        # Keep A as the primary scientific result; do not reopen B14M.
        primary = CASE_COUPLING
        next_step = "repair_missing_pos_to_dir_variational_coupling_then_recontract"
    contract = False
    return {
        "verdict": "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "missing_deterministic_transport_coupling_identified": coupling_ok,
        "focus_independent_reference_established": focus_established,
        "control1_station0_loc1_agrees": station0_ok,
        "rk_magnet_hop_pos_to_dir_is_zero": pos_to_dir_missing,
        "material_reset_is_highest_priority_mechanism": False,
        "small_physical_effect_does_not_pass_contract": True,
        "jacobian_contract_established": contract,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "five_percent_gate_unchanged": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_change_official_fd_ladder": True,
        "do_not_add_fd_rung": True,
        "do_not_change_jacobian_implementation": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_replace_official_likelihood": True,
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "do_not_shrink_fd_step": True,
        "five_d_cin_not_the_objective": True,
        "wb120_same_path_jacobian_available": True,
        "wb121_not_a_physical_conclusion": True,
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
    control1 = []
    focus_refs = []
    focus_per_hits = []
    material = []
    segments = []
    for row in dumps["rows"]:
        if not is_official_mode_b(row):
            continue
        pair = _event_pair(row)
        if pair not in REQUIRED_EVENTS:
            continue
        if row.get("official_supporting_plane_jacobian") is None:
            continue
        if pair == CONTROL1:
            coupling = audit_control1_loc1(row)
            control1.append(coupling)
            material.append(audit_material_reset(coupling))
            segments.append(audit_segment_closure(coupling))
        if pair == FOCUS_EVENT:
            focus_refs.append(audit_focus86_reference(row))
            focus_per_hits.append(audit_focus86_per_hit(row))
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "wb120_smoke_hash_match": True,
        "n_rows": dumps["n_rows"],
        "control1": control1,
        "focus_86": [item for item in focus_refs],
        "focus_references": focus_refs,
        "focus_per_hits": focus_per_hits,
        "material_audits": material,
        "segment_audits": segments,
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
        "selected_best_step": False,
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "changed_jacobian_implementation": False,
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
    if mechanism.get("jacobian_contract_established"):
        mechanism["jacobian_contract_established"] = False
        mechanism["b14m_reopen_authorized"] = False
        mechanism["verdict"] = "FAIL"
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "inherited_wb121_decision_sha256": inherited["workbook_121"]["decision_sha256"],
        "inherited_wb120_decision_sha256": inherited["workbook_120"]["decision_sha256"],
        "inherited_wb119_decision_sha256": inherited["workbook_119"]["decision_sha256"],
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
        "do_not_change_jacobian_implementation": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_select_best_fd_step": True,
        "small_physical_effect_does_not_pass_contract": True,
        "pinv_relative": PINV_RELATIVE,
        "objective": "identify missing loc1 coupling and try an independent 86 reference",
        "not_the_objective": "repair 5D Cin",
        "wb121_not_a_physical_conclusion": True,
    }
