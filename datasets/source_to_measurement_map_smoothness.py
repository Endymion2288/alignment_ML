"""Task B14K: source-to-measurement map smoothness root-cause audit.

Diagnoses why h_i(theta) near 100048/86 has no certifiable Jacobian on
loc1 / phi / q/p.  Does not retune Gauss-Newton, pick a best FD step or
stepTolerance, switch to direct-from-source, introduce a prior/ridge,
delete 37/86, replace q/p, repair 5D Cin, or enter B14M / B15 /
Measurement Model V2.
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
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profile_transport_contract import (
    CLASS_S_EVENT,
    audit_exclusion,
    is_official_mode_b,
)
from datasets.supporting_plane_jacobian_continuity import (
    CASE_TRANSPORT as WB117_DECISION,
    inherit_frozen_stage as inherit_through_wb116,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "source-to-measurement-map-smoothness-v1"
DEFAULT_CONFIG = "configs/source_to_measurement_map_smoothness_v1.yaml"
TASK = "SB-B14K"
WORKBOOK = 118

CASE_INTEGRATION = "transport_integration_resolution_insufficient"
CASE_PROJECTION = "supporting_plane_projection_ill_conditioned"
CASE_SCALE = "parameterization_scale_not_resolved"
CASE_ACTS_JAC = "acts_transport_jacobian_inconsistent_with_fd"
CASE_GENUINE = "source_to_measurement_map_genuinely_nonsmooth"
CASE_MIXED = "mixed_or_inconclusive"

FOCUS_PARAMETERS = ("loc1", "phi", "q_over_p")
STAGES = (
    "source_bound",
    "bound_to_free",
    "free_near_plane",
    "plane_intersection",
    "local_chart",
    "predicted_loc0",
)
RUNG_FACTORS = (1.0, 0.5, 0.25, 0.125)
REL_MAX = 0.05
FOCUS_EVENT = (100048, 86)
CONTROL_EVENT = (100048, 44)
ACCURACY_TOLERANCES = (1.0e-4, 1.0e-5, 1.0e-6)


class SourceToMeasurementMapSmoothnessError(ValueError):
    """Raised when the B14K smoothness contract is illegal."""


def refuse_prior() -> None:
    raise SourceToMeasurementMapSmoothnessError("B14K must not introduce a prior")


def refuse_ridge_information() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "ridge must not be treated as statistical information"
    )


def refuse_measurement_update() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "likelihood evaluator must not apply a Kalman measurement update"
    )


def refuse_best_step_selection() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "must not select the best finite-difference step from the results"
    )


def refuse_best_tolerance() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "must not select the best propagator stepTolerance from the results"
    )


def refuse_switch_to_direct() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "must not replace the official sequential likelihood with direct-from-source"
    )


def refuse_b14m() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "B14M is not re-opened inside Task B14K"
    )


def refuse_restart() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "restart invariance is not opened inside Task B14K"
    )


def refuse_b15() -> None:
    raise SourceToMeasurementMapSmoothnessError("Task B15 is not entered in Task B14K")


def refuse_measurement_model_v2() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "Measurement Model V2 is not entered in Task B14K"
    )


def refuse_full_sample() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "the 1989-row campaign must not be submitted in B14K"
    )


def refuse_change_statistical_model() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "the measurement statistical model must stay frozen"
    )


def refuse_5d_cin_repair() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "B14K must not treat 5D Cin repair as the objective"
    )


def refuse_genuine_without_exclusions() -> None:
    raise SourceToMeasurementMapSmoothnessError(
        "genuine nonsmoothness requires determinism, projection, noise, "
        "scaling, integration, and branch exclusions"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise SourceToMeasurementMapSmoothnessError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SourceToMeasurementMapSmoothnessError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if config.get("task") != TASK:
        raise SourceToMeasurementMapSmoothnessError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise SourceToMeasurementMapSmoothnessError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise SourceToMeasurementMapSmoothnessError(
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
            raise SourceToMeasurementMapSmoothnessError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
        "do_not_enter_b14j",
        "do_not_rewrite_profile_math",
        "do_not_change_statistical_model",
        "do_not_submit_full_sample_without_smoke_gate",
        "do_not_introduce_target_independent_prior",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
        "do_not_add_measurement_update_in_evaluator",
        "do_not_select_best_fd_step",
        "do_not_switch_official_likelihood_to_direct",
        "do_not_claim_5d_cin_repaired",
    ):
        if bool(config.get(key, False)) is not True:
            raise SourceToMeasurementMapSmoothnessError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14k", True)):
        raise SourceToMeasurementMapSmoothnessError(
            "B14K config must allow entering B14K"
        )
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise SourceToMeasurementMapSmoothnessError(f"frozen gate changed: {key}")
    numeric = config.get("profile_numerical") or {}
    if float(numeric.get("pinv_relative")) == 0.01:
        refuse_alignment_rank_tolerance()
    if abs(float(numeric.get("pinv_relative")) - PINV_RELATIVE) > 1.0e-20:
        raise SourceToMeasurementMapSmoothnessError(
            "pinv_relative must stay pre-registered 1e-8"
        )
    smoothness = config.get("map_smoothness") or {}
    if list(smoothness.get("rung_factors") or []) != list(RUNG_FACTORS):
        raise SourceToMeasurementMapSmoothnessError(
            "Jacobian ladder must stay h, h/2, h/4, h/8"
        )
    if bool(smoothness.get("do_not_select_best_step")) is not True:
        refuse_best_step_selection()
    if bool(smoothness.get("do_not_select_best_tolerance")) is not True:
        refuse_best_tolerance()
    tols = [
        float(item["step_tolerance"])
        for item in (smoothness.get("accuracy_rungs") or [])
    ]
    if tols != list(ACCURACY_TOLERANCES):
        raise SourceToMeasurementMapSmoothnessError(
            "accuracy rungs must stay ACTS default 1e-4, 1e-5, 1e-6"
        )
    control = config.get("control_selection") or {}
    if int(control.get("selected_event_id", -1)) != CONTROL_EVENT[1]:
        raise SourceToMeasurementMapSmoothnessError(
            "control event must stay the pre-registered 100048/44"
        )
    if bool(control.get("jacobian_not_used")) is not True:
        raise SourceToMeasurementMapSmoothnessError(
            "control must be selected without looking at Jacobian"
        )
    if bool(control.get("truth_not_used")) is not True:
        raise SourceToMeasurementMapSmoothnessError("control must not use truth")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise SourceToMeasurementMapSmoothnessError(
            "B14K eligibility must stay identical to WB103"
        )
    partition = config["profile_partition"]
    if list(partition["alpha_native"]) != list(ALPHA_NAMES):
        raise SourceToMeasurementMapSmoothnessError("alpha must stay loc0, theta")
    if list(partition["nu_native"]) != list(NU_NAMES):
        raise SourceToMeasurementMapSmoothnessError("nu must stay loc1, phi, q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb116(config)
    spec = config["inheritance"]["workbook_117"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_117 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_117 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise SourceToMeasurementMapSmoothnessError(
            "workbook_117 decision must stay frozen"
        )
    if spec["frozen_decision"] != WB117_DECISION:
        raise SourceToMeasurementMapSmoothnessError("WB117 decision token mismatch")
    if decision.get("b14m_reopen_authorized"):
        raise SourceToMeasurementMapSmoothnessError(
            "WB117 must not have re-opened B14M"
        )
    if decision.get("jacobian_contract_established"):
        raise SourceToMeasurementMapSmoothnessError(
            "WB117 must not have established the Jacobian contract"
        )
    inherited["workbook_117"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "official_run_id": spec["official_run_id"],
        "helper_sha256": spec["helper_sha256"],
        "smoke_gate_passed": False,
        "jacobian_contract_established": False,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "target_exclusion_holds": True,
        "statistical_model_unchanged": True,
        "prior_introduced": False,
        "do_not_select_best_step": True,
        "do_not_switch_to_direct": True,
        "five_d_cin_not_the_objective": True,
    }
    return inherited


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _event_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["smoothness_smoke_root"]))
    filename = str(
        config.get("smoothness_smoke_filename", "ckf_leave_target_out_map_smoothness.jsonl")
    )
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_smoothness_rows(config: Mapping[str, Any]) -> dict[str, Any]:
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


def _vec(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, (int, float)):
        number = _finite(values)
        return None if number is None else np.asarray([number], dtype=np.float64)
    array = []
    for item in values:
        number = _finite(item)
        if number is None:
            return None
        array.append(number)
    return np.asarray(array, dtype=np.float64) if array else None


def hop_stage_vector(hop: Mapping[str, Any], stage: str) -> np.ndarray | None:
    if stage == "source_bound":
        return _vec(hop.get("source_bound"))
    if stage == "bound_to_free":
        pos = _vec(hop.get("initial_free_position_xyz_mm"))
        direction = _vec(hop.get("initial_free_direction"))
        qop = _finite(hop.get("initial_free_q_over_p"))
        if pos is None or direction is None or qop is None:
            return None
        return np.concatenate([pos, direction, [qop]])
    if stage == "free_near_plane":
        return _vec(hop.get("free_state_before_projection"))
    if stage == "plane_intersection":
        path = _finite(hop.get("intersection_path_length"))
        xyz = _vec(hop.get("intersection_global_xyz_mm"))
        if path is None or xyz is None:
            return None
        return np.concatenate([[path], xyz])
    if stage == "local_chart":
        return _vec(hop.get("local_loc0_loc1"))
    if stage == "predicted_loc0":
        return _vec(hop.get("predicted_loc0"))
    return None


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    norm = float(np.linalg.norm(right))
    if norm <= 1.0e-12:
        return 0.0
    return float(np.linalg.norm(left - right) / norm)


def audit_stage_column(column: Mapping[str, Any]) -> dict[str, Any]:
    if bool(column.get("selected_best_step")):
        refuse_best_step_selection()
    rungs = list(column.get("rungs") or [])
    if len(rungs) != 4:
        raise SourceToMeasurementMapSmoothnessError(
            "every stagewise column must record h, h/2, h/4, h/8"
        )
    hop_reports = []
    n_hops = 0
    if rungs and (rungs[0].get("plus") or {}).get("hops"):
        n_hops = len(rungs[0]["plus"]["hops"])
    first_failing = {stage: None for stage in STAGES}
    stage_ok = {stage: True for stage in STAGES}
    first_failing_hop = None
    first_failing_stage = None
    for hop_index in range(n_hops):
        hop_item = {"measurement_index": hop_index, "stages": {}}
        previous_ok = True
        for stage in STAGES:
            jac = []
            ok = True
            for rung in rungs:
                plus_hop = ((rung.get("plus") or {}).get("hops") or [None] * n_hops)[
                    hop_index
                ]
                minus_hop = ((rung.get("minus") or {}).get("hops") or [None] * n_hops)[
                    hop_index
                ]
                step = _finite(rung.get("fd_step"))
                left = hop_stage_vector(plus_hop or {}, stage)
                right = hop_stage_vector(minus_hop or {}, stage)
                if left is None or right is None or step is None or step == 0.0:
                    ok = False
                    jac.append(None)
                    continue
                jac.append((left - right) / (2.0 * step))
            rels = []
            signs = []
            for index in range(len(jac) - 1):
                if jac[index] is None or jac[index + 1] is None:
                    rels.append(None)
                    signs.append(False)
                    ok = False
                    continue
                rels.append(_relative_error(jac[index], jac[index + 1]))
                signs.append(float(np.dot(jac[index], jac[index + 1])) >= 0.0)
            last = rels[-1] if rels else None
            converged = (
                ok
                and last is not None
                and last <= REL_MAX
                and all(signs)
            )
            hop_item["stages"][stage] = {
                "relative_errors": rels,
                "sign_consistency": all(signs) if signs else False,
                "last_pair_relative_error": last,
                "converged": converged,
            }
            if not converged:
                stage_ok[stage] = False
                if first_failing[stage] is None:
                    first_failing[stage] = hop_index
                if first_failing_hop is None:
                    first_failing_hop = hop_index
                    first_failing_stage = stage
                previous_ok = False
            elif not previous_ok:
                previous_ok = False
        hop_reports.append(hop_item)
    return {
        "parameter": column.get("parameter"),
        "n_hops": n_hops,
        "hops": hop_reports,
        "stage_converged": stage_ok,
        "first_failing_stage": first_failing_stage,
        "first_failing_hop": first_failing_hop,
        "first_failing_by_stage": first_failing,
        "do_not_select_best_step": True,
    }


def audit_repeatability(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("repeatability") or {}
    nondet = bool(block.get("transport_nondeterministic"))
    return {
        "n_repeats": block.get("n_repeats"),
        "max_predicted_loc0_l2_delta": block.get("max_predicted_loc0_l2_delta"),
        "max_chi2_abs_delta": block.get("max_chi2_abs_delta"),
        "steps_identical": block.get("steps_identical"),
        "transport_nondeterministic": nondet,
        "deterministic": (not nondet) and bool(block.get("steps_identical", True)),
    }


def audit_resolution(payload: Mapping[str, Any], noise_multiplier: float = 10.0) -> dict[str, Any]:
    block = payload.get("resolution_vs_fd") or {}
    rows = []
    below = 0
    above = 0
    for raw in block.get("rows") or []:
        signal = _finite(raw.get("fd_signal_loc0_l2"))
        noise = _finite(raw.get("repeat_noise_loc0_l2"))
        ratio = None
        if signal is not None and noise is not None and noise > 0.0:
            ratio = signal / noise
        is_below = ratio is not None and ratio <= noise_multiplier
        above_zero_noise = (
            signal is not None
            and signal > 0.0
            and noise is not None
            and noise == 0.0
        )
        if is_below:
            below += 1
        elif (ratio is not None and ratio > noise_multiplier) or above_zero_noise:
            above += 1
        rows.append(
            {
                "parameter": raw.get("parameter"),
                "step_factor": raw.get("step_factor"),
                "fd_signal_loc0_l2": signal,
                "repeat_noise_loc0_l2": noise,
                "signal_over_noise": ratio,
                "repeat_noise_is_zero": noise == 0.0 if noise is not None else None,
                "fd_signal_at_or_below_noise": is_below,
                "fd_signal_above_zero_noise": above_zero_noise,
            }
        )
    official = [
        row
        for row in rows
        if _finite(row.get("step_factor")) == 1.0
    ]
    official_below = bool(official) and all(
        row["fd_signal_at_or_below_noise"] for row in official
    )
    return {
        "rows": rows,
        "n_below_noise": below,
        "n_above_noise": above,
        "official_h_signal_at_or_below_noise": official_below,
        "do_not_decrease_step_to_search": True,
    }


def audit_accuracy(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("propagator_accuracy") or {}
    if bool(block.get("selected_best_tolerance")):
        refuse_best_tolerance()
    rungs = []
    for raw in block.get("rungs") or []:
        if bool(raw.get("selected_as_production")):
            refuse_best_tolerance()
        columns = []
        all_conv = True
        last_rels = []
        for column in raw.get("columns") or []:
            pairs = list(column.get("consecutive_relative_errors") or [])
            last = _finite(pairs[-1]["relative_error"]) if pairs else None
            signs = [bool(item.get("sign_consistent")) for item in pairs]
            conv = last is not None and last <= REL_MAX and all(signs)
            all_conv = all_conv and conv
            last_rels.append(last)
            columns.append(
                {
                    "parameter": column.get("parameter"),
                    "last_pair_relative_error": last,
                    "sign_consistency": all(signs) if signs else False,
                    "converged": conv,
                }
            )
        rungs.append(
            {
                "name": raw.get("name"),
                "step_tolerance": raw.get("step_tolerance"),
                "nominal_ok": raw.get("nominal_ok"),
                "columns": columns,
                "ladder_converged": all_conv and bool(columns),
                "mean_last_pair_relative_error": (
                    float(np.mean([v for v in last_rels if v is not None]))
                    if any(v is not None for v in last_rels)
                    else None
                ),
            }
        )
    means = [
        rung["mean_last_pair_relative_error"]
        for rung in rungs
        if rung["mean_last_pair_relative_error"] is not None
    ]
    improving = len(means) >= 2 and all(
        means[i] + 1.0e-12 >= means[i + 1] for i in range(len(means) - 1)
    )
    last_converged = bool(rungs) and bool(rungs[-1]["ladder_converged"])
    first_converged = bool(rungs) and bool(rungs[0]["ladder_converged"])
    return {
        "rungs": rungs,
        "relative_error_improves_with_tighter_tolerance": improving,
        "tighter_again_converged": last_converged,
        "nominal_converged": first_converged,
        "integration_resolution_insufficient": improving
        and last_converged
        and not first_converged,
        "integration_improves_but_not_converged": improving
        and (not last_converged)
        and not first_converged,
        "do_not_select_best_tolerance": True,
    }


def audit_projection(payload: Mapping[str, Any], near_parallel: float = 1.0e-6) -> dict[str, Any]:
    block = payload.get("supporting_plane_projection") or {}
    hops = []
    ill = False
    all_smooth = True
    any_eval = False
    for raw in block.get("hops") or []:
        nd = _finite(raw.get("n_dot_direction"))
        parallel = bool(raw.get("near_parallel")) or (
            nd is not None and abs(nd) < near_parallel
        )
        pos_ok = True
        for column in raw.get("position_fd") or []:
            pairs = list(column.get("consecutive_relative_errors") or [])
            last = _finite(pairs[-1]["relative_error"]) if pairs else None
            signs = [bool(item.get("sign_consistent")) for item in pairs]
            if last is None or last > REL_MAX or not all(signs):
                pos_ok = False
        evaluable = bool(raw.get("projection_map_evaluable"))
        any_eval = any_eval or evaluable
        if parallel:
            ill = True
        if evaluable and not pos_ok:
            all_smooth = False
        hops.append(
            {
                "measurement_index": raw.get("measurement_index"),
                "n_dot_direction": nd,
                "incidence_angle": raw.get("incidence_angle"),
                "near_parallel": parallel,
                "position_fd_converged": pos_ok,
                "projection_map_evaluable": evaluable,
            }
        )
    return {
        "hops": hops,
        "projection_map_smooth": any_eval and all_smooth and not ill,
        "transport_map_problem": any_eval and all_smooth and not ill,
        "ill_conditioned": ill,
        "decoupled_from_transport": True,
    }


def audit_fd_scale(payload: Mapping[str, Any], resolution: Mapping[str, Any]) -> dict[str, Any]:
    scale = payload.get("fd_scale") or {}
    q_rel = _finite(scale.get("q_over_p_h_over_abs_nominal"))
    regime = "unclassified"
    if q_rel is not None and q_rel > 0.1:
        regime = "perturbation_may_be_nonlinear"
    if bool(resolution.get("official_h_signal_at_or_below_noise")):
        regime = "perturbation_at_or_below_transport_noise"
    return {
        "q_over_p_nominal": scale.get("q_over_p_nominal"),
        "q_over_p_h": scale.get("q_over_p_h"),
        "q_over_p_h_over_abs_nominal": q_rel,
        "loc1_h_over_1mm": scale.get("loc1_h_over_1mm"),
        "phi_h_over_one_radian": scale.get("phi_h_over_one_radian"),
        "regime": regime,
        "do_not_select_production_step": True,
    }


def audit_acts_derivative(payload: Mapping[str, Any]) -> dict[str, Any]:
    block = payload.get("acts_derivative") or {}
    present = bool(block.get("any_diagnostic_jacobian_present"))
    matches = []
    n_match = 0
    n_present = 0
    for hop in block.get("hops") or []:
        if not hop.get("acts_jacobian_present"):
            continue
        n_present += 1
        acts_loc0 = _finite(hop.get("acts_end_loc0"))
        official = _finite(hop.get("official_predicted_loc0"))
        if acts_loc0 is None or official is None:
            matches.append(False)
            continue
        ok = abs(acts_loc0 - official) <= 1.0e-6
        matches.append(ok)
        n_match += int(ok)
    return {
        "acts_version": block.get("acts_version"),
        "propagator_result_has_transportJacobian": True,
        "official_path_transports_covariance": False,
        "complex_step_requires_acts_rewrite": True,
        "autodiff_requires_acts_rewrite": True,
        "surface_intersect_is_analytic": True,
        "any_diagnostic_jacobian_present": present,
        "n_diagnostic_jacobians": n_present,
        "n_acts_end_loc0_matches_official": n_match,
        "acts_end_loc0_matches_official": bool(n_present) and n_match == n_present,
        "analytic_vs_fd_not_yet_contracted": True,
        "acts_inconsistent_with_fd": False,
        "do_not_replace_official_likelihood": True,
        "n_hops": len(block.get("hops") or []),
        "next_if_usable": "WB119_analytic_vs_fd_derivative_contract",
    }


def audit_target_smoothness(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("map_smoothness") or {}
    if not payload:
        raise SourceToMeasurementMapSmoothnessError(
            "official Mode B row must carry map_smoothness"
        )
    if bool(payload.get("selected_best_step")):
        refuse_best_step_selection()
    stage_columns = []
    first_stages = []
    first_hops = []
    first_ops = []
    for raw in (payload.get("stagewise") or {}).get("columns") or []:
        report = audit_stage_column(raw)
        stage_columns.append(report)
        first_stages.append(report.get("first_failing_stage"))
        first_hops.append(report.get("first_failing_hop"))
        if report.get("first_failing_stage") is not None:
            first_ops.append(
                {
                    "parameter": report.get("parameter"),
                    "hop": report.get("first_failing_hop"),
                    "stage": report.get("first_failing_stage"),
                }
            )
    repeatability = audit_repeatability(payload)
    resolution = audit_resolution(payload)
    accuracy = audit_accuracy(payload)
    projection = audit_projection(payload)
    fd_scale = audit_fd_scale(payload, resolution)
    acts = audit_acts_derivative(payload)
    return {
        "event": f"{row.get('run_id')}/{row.get('event_id')}",
        "target_station": row.get("target_station"),
        "all_surfaces_reached": bool(row.get("all_surfaces_reached")),
        "evaluate_chi2": row.get("evaluate_chi2"),
        "source_bound": (payload.get("source_bound")),
        "stagewise": stage_columns,
        "first_failing_stages": first_stages,
        "first_failing_hops": first_hops,
        "first_failing_operations": first_ops,
        "repeatability": repeatability,
        "resolution": resolution,
        "accuracy": accuracy,
        "projection": projection,
        "fd_scale": fd_scale,
        "acts_derivative": acts,
        "official_is_sequential": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
    }


def compare_focus_control(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    focus = [item for item in targets if item.get("event") == "100048/86"]
    control = [item for item in targets if item.get("event") == "100048/44"]
    rows = []
    for left, right in zip(focus, control):
        loc_f = (left.get("source_bound") or [None] * 5)
        loc_c = (right.get("source_bound") or [None] * 5)
        rows.append(
            {
                "target_station": left.get("target_station"),
                "focus_q_over_p": loc_f[4] if len(loc_f) > 4 else None,
                "control_q_over_p": loc_c[4] if len(loc_c) > 4 else None,
                "focus_theta": loc_f[3] if len(loc_f) > 3 else None,
                "control_theta": loc_c[3] if len(loc_c) > 3 else None,
                "focus_chi2": left.get("evaluate_chi2"),
                "control_chi2": right.get("evaluate_chi2"),
                "focus_first_failing_stages": left.get("first_failing_stages"),
                "focus_first_failing_hops": left.get("first_failing_hops"),
                "focus_first_failing_operations": left.get("first_failing_operations"),
                "control_first_failing_stages": right.get("first_failing_stages"),
                "control_first_failing_hops": right.get("first_failing_hops"),
                "focus_deterministic": (left.get("repeatability") or {}).get(
                    "deterministic"
                ),
                "control_deterministic": (right.get("repeatability") or {}).get(
                    "deterministic"
                ),
                "focus_projection_smooth": (left.get("projection") or {}).get(
                    "projection_map_smooth"
                ),
                "control_projection_smooth": (right.get("projection") or {}).get(
                    "projection_map_smooth"
                ),
                "focus_official_below_noise": (left.get("resolution") or {}).get(
                    "official_h_signal_at_or_below_noise"
                ),
                "control_official_below_noise": (right.get("resolution") or {}).get(
                    "official_h_signal_at_or_below_noise"
                ),
                "selection_used_jacobian": False,
                "selection_used_truth": False,
            }
        )
    return {
        "focus": "100048/86",
        "control": "100048/44",
        "defined_before_jacobian": True,
        "n_focus_targets": len(focus),
        "n_control_targets": len(control),
        "rows": rows,
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
    if inventory.get("selected_best_tolerance"):
        refuse_best_tolerance()
    if inventory.get("restart_executed"):
        refuse_restart()
    if inventory.get("b14m_reopened"):
        refuse_b14m()
    smoke = bool(inventory.get("smoke_present"))
    focus = inventory.get("focus_86") or []
    if not smoke or not focus:
        return {
            "verdict": "DIAGNOSED",
            "decision": CASE_MIXED,
            "primary_case": CASE_MIXED,
            "next_step": "run_login_map_smoothness_smoke",
            "b14m_reopen_authorized": False,
            "restart_invariance_authorized": False,
            "jacobian_contract_established": False,
            "full_sample_authorized": False,
            "b15_authorized": False,
            "measurement_model_v2_authorized": False,
        }
    det = all((item.get("repeatability") or {}).get("deterministic") for item in focus)
    proj = [item.get("projection") or {} for item in focus]
    ill = any(item.get("ill_conditioned") for item in proj)
    proj_smooth = all(item.get("projection_map_smooth") for item in proj)
    acc = [item.get("accuracy") or {} for item in focus]
    integration = any(item.get("integration_resolution_insufficient") for item in acc)
    improves_not_converged = any(
        item.get("integration_improves_but_not_converged") for item in acc
    )
    res = [item.get("resolution") or {} for item in focus]
    below = any(item.get("official_h_signal_at_or_below_noise") for item in res)
    scales = [item.get("fd_scale") or {} for item in focus]
    nonlinear = any(item.get("regime") == "perturbation_may_be_nonlinear" for item in scales)
    first_stages = [
        stage
        for item in focus
        for stage in (item.get("first_failing_stages") or [])
        if stage is not None
    ]
    acts = [item.get("acts_derivative") or {} for item in focus]
    acts_present = bool(acts) and all(
        item.get("any_diagnostic_jacobian_present") for item in acts
    )
    acts_inconsistent = any(item.get("acts_inconsistent_with_fd") for item in acts)
    if not det:
        primary = CASE_MIXED
        next_step = "fix_numerical_propagation_repeatability_before_jacobian"
    elif ill:
        primary = CASE_PROJECTION
        next_step = "diagnose_near_parallel_supporting_plane_intersection"
    elif integration:
        primary = CASE_INTEGRATION
        next_step = "establish_fixed_numerical_accuracy_contract_do_not_pick_best"
    elif below or nonlinear:
        primary = CASE_SCALE
        next_step = "keep_fixed_fd_ladder_do_not_search_step"
    elif acts_inconsistent:
        primary = CASE_ACTS_JAC
        next_step = "open_wb119_analytic_vs_fd_derivative_contract"
    elif (
        first_stages
        and proj_smooth
        and det
        and not integration
        and not improves_not_converged
        and not below
        and not acts_present
    ):
        # Genuine nonsmoothness is allowed only after determinism, integration
        # non-improvement, projection, noise, scaling, and branch exclusions.
        # Presence of an unused ACTS Jacobian is not an inconsistency and
        # blocks this case until WB119 contracts analytic vs FD.
        exclusions = {
            "determinism_pass": det,
            "integration_failed_to_improve": not integration
            and not improves_not_converged,
            "projection_checked": True,
            "projection_smooth": proj_smooth,
            "fd_signal_above_noise": not below,
            "parameter_scaling_checked": True,
            "branch_not_claimed": True,
        }
        if all(exclusions.values()):
            primary = CASE_GENUINE
            next_step = "keep_source_to_measurement_map_as_object"
        else:
            primary = CASE_MIXED
            next_step = "keep_stagewise_root_cause_diagnosis"
    else:
        primary = CASE_MIXED
        next_step = "keep_stagewise_root_cause_diagnosis"
    if acts_present and primary == CASE_MIXED and det:
        next_step = "open_wb119_analytic_vs_fd_derivative_contract"
    if primary == CASE_GENUINE and (
        not det or improves_not_converged or integration or not proj_smooth
    ):
        refuse_genuine_without_exclusions()
    return {
        "verdict": "FAIL",
        "decision": primary,
        "primary_case": primary,
        "next_step": next_step,
        "acts_transport_jacobian_available": acts_present,
        "analytic_vs_fd_not_yet_contracted": True,
        "acts_inconsistent_with_fd": acts_inconsistent,
        "integration_improves_but_not_converged": improves_not_converged,
        "transport_deterministic": det,
        "projection_map_smooth": proj_smooth,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "jacobian_contract_established": False,
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
        "wb117_not_treated_as_physical_nonsmoothness": True,
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    targets = inventory.get("targets") or []
    events = {item.get("event") for item in targets}
    checks = {
        "focus_86": "100048/86" in events,
        "control_44": "100048/44" in events,
        "stagewise_recorded": all(item.get("stagewise") is not None for item in targets),
        "repeatability_recorded": all(item.get("repeatability") is not None for item in targets),
        "resolution_recorded": all(item.get("resolution") is not None for item in targets),
        "accuracy_recorded": all(item.get("accuracy") is not None for item in targets),
        "projection_recorded": all(item.get("projection") is not None for item in targets),
        "fd_scale_recorded": all(item.get("fd_scale") is not None for item in targets),
        "acts_inventory_recorded": all(item.get("acts_derivative") is not None for item in targets),
        "control_comparison_recorded": bool(
            (inventory.get("focus_vs_control") or {}).get("rows")
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
        "smoke_present": bool(inventory.get("smoke_present")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "do_not_submit_if_failed": True,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_smoothness_rows(config)
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
        if _event_pair(row) not in {FOCUS_EVENT, CONTROL_EVENT}:
            continue
        if row.get("map_smoothness") is None:
            continue
        targets.append(audit_target_smoothness(row))
    if n_update:
        refuse_measurement_update()
    inventory = {
        "dumps": dumps,
        "smoke_present": dumps["smoke_present"],
        "n_rows": dumps["n_rows"],
        "targets": targets,
        "focus_86": [item for item in targets if item.get("event") == "100048/86"],
        "control_44": [item for item in targets if item.get("event") == "100048/44"],
        "focus_vs_control": compare_focus_control(targets),
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
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb117_decision_sha256": inherited["workbook_117"]["decision_sha256"],
        "inherited_wb116_decision_sha256": inherited["workbook_116"]["decision_sha256"],
        "inherited_wb115_decision_sha256": inherited["workbook_115"]["decision_sha256"],
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb117_rewritten": False,
        "target_exclusion_holds": bool(
            (inventory.get("exclusion") or {}).get("target_exclusion_holds")
        ),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "jacobian_contract_established": False,
        "acts_map_smoothness_materialized": bool(inventory.get("smoke_present")),
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
        "direct_from_source": "not used in B14K",
        "do_not_switch_to_direct": True,
        "do_not_select_best_fd_step": True,
        "do_not_select_best_tolerance": True,
        "measurement_update_in_evaluator": False,
        "objective": "locate the first non-convergent operation in h_i(theta) near 86",
        "not_the_objective": "repair 5D Cin",
        "wb117_not_a_physical_conclusion": True,
    }
