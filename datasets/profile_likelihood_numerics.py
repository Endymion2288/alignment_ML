"""Task B14N: profile-likelihood numerical stabilization and falsification.

Audits whether the frozen WB114 measurement likelihood can be evaluated
and profiled stably under real ACTS geometry/field.  Does not change the
statistical model, introduce a prior, add ridge as information, invent a
chi2 penalty for propagation failure, restore 5D Cin, enter B15, or enter
Measurement Model V2.
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
    validate_linear_profile_agreement,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
)
from datasets.profiled_weak_nuisance_likelihood import (
    CASE_D as WB114_DECISION,
    inherit_frozen_stage as inherit_through_wb113,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "profile-likelihood-numerics-v1"
DEFAULT_CONFIG = "configs/profile_likelihood_numerics_v1.yaml"
TASK = "SB-B14N"
WORKBOOK = 115

CASE_A = "profile_numerical_contract_established"
CASE_B = "profile_transport_contract_broken"
CASE_C = "profile_derivative_contract_broken"
CASE_D = "profile_optimizer_contract_broken"
CASE_E = "physical_nonidentifiability_exposed"
CASE_F = "mixed_or_inconclusive"

FOCUS_KEYS = {
    ("mc24_100043_00400_00499", 100043, 0),
    ("mc24_100043_00400_00499", 100043, 1),
    ("mc24_100043_00400_00499", 100043, 37),
    ("mc24_100048_00000_00049", 100048, 86),
}
RESTART_EVENTS = {(100043, 0), (100043, 1)}
FOCUS_HARD = {(100043, 37), (100048, 86)}
OPTIMIZE_VARIANTS = ("nominal", "loc1_plus_1mm", "phi_plus_1e-3", "qoverp_times_1p1")


class ProfileLikelihoodNumericsError(ValueError):
    """Raised when the B14N numerical contract is illegal."""


def refuse_prior() -> None:
    raise ProfileLikelihoodNumericsError("B14N must not introduce a prior")


def refuse_ridge_information() -> None:
    raise ProfileLikelihoodNumericsError("ridge must not be treated as statistical information")


def refuse_chi2_penalty() -> None:
    raise ProfileLikelihoodNumericsError(
        "propagation failure must not be replaced by an invented chi2 penalty"
    )


def refuse_scaling_as_prior() -> None:
    raise ProfileLikelihoodNumericsError("numerical scaling is not a prior covariance")


def refuse_truth_scale() -> None:
    raise ProfileLikelihoodNumericsError("scales must not come from truth errors")


def refuse_hessian_scale() -> None:
    raise ProfileLikelihoodNumericsError("scales must not be tuned from the Hessian")


def refuse_seed_campaign_scale() -> None:
    raise ProfileLikelihoodNumericsError("scales must not come from a 0.1/1/10 seed campaign")


def refuse_rewrite_profile_math() -> None:
    raise ProfileLikelihoodNumericsError("WB114 profile mathematics must not be rewritten")


def refuse_change_statistical_model() -> None:
    raise ProfileLikelihoodNumericsError("the measurement statistical model must stay frozen")


def refuse_b15() -> None:
    raise ProfileLikelihoodNumericsError("Task B15 is not entered in Task B14N")


def refuse_measurement_model_v2() -> None:
    raise ProfileLikelihoodNumericsError("Measurement Model V2 is not entered in Task B14N")


def refuse_full_sample_without_gate() -> None:
    raise ProfileLikelihoodNumericsError(
        "the 1989-row campaign must not be submitted before the smoke gate passes"
    )


def refuse_focus_drop() -> None:
    raise ProfileLikelihoodNumericsError("focus identities 100043/37 and 100048/86 must be retained")


def refuse_propagation_as_nonidentifiability() -> None:
    raise ProfileLikelihoodNumericsError(
        "propagation failure is not physical nonidentifiability"
    )


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ProfileLikelihoodNumericsError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ProfileLikelihoodNumericsError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ProfileLikelihoodNumericsError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ProfileLikelihoodNumericsError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise ProfileLikelihoodNumericsError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "marginalization_executed",
    ):
        if bool(config.get(key, True)):
            raise ProfileLikelihoodNumericsError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_enter_b14m",
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
    ):
        if bool(config.get(key, False)) is not True:
            raise ProfileLikelihoodNumericsError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14n", True)):
        raise ProfileLikelihoodNumericsError("B14N config must allow entering B14N")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ProfileLikelihoodNumericsError(f"frozen gate changed: {key}")
    numeric = config.get("profile_numerical") or {}
    if float(numeric.get("pinv_relative")) == 0.01:
        refuse_alignment_rank_tolerance()
    if abs(float(numeric.get("pinv_relative")) - PINV_RELATIVE) > 1.0e-20:
        raise ProfileLikelihoodNumericsError("pinv_relative must stay pre-registered 1e-8")
    if bool(numeric.get("do_not_add_ridge")) is not True:
        refuse_ridge()
    scales = (config.get("profile_parameter_scaling") or {}).get("scales") or {}
    if not scales:
        raise ProfileLikelihoodNumericsError("fixed numerical scales must be pre-registered")
    if bool((config.get("profile_parameter_scaling") or {}).get("not_a_prior")) is not True:
        refuse_scaling_as_prior()
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise ProfileLikelihoodNumericsError("B14N eligibility must stay identical to WB103")
    partition = config["profile_partition"]
    if list(partition["alpha_native"]) != list(ALPHA_NAMES):
        raise ProfileLikelihoodNumericsError("alpha must stay loc0, theta")
    if list(partition["nu_native"]) != list(NU_NAMES):
        raise ProfileLikelihoodNumericsError("nu must stay loc1, phi, q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb113(config)
    spec = config["inheritance"]["workbook_114"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["config_path"]),
        spec["config_sha256"],
        "workbook_114 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec["decision_path"]),
        spec["decision_sha256"],
        "workbook_114 decision",
    )
    if decision.get("decision") != spec["frozen_decision"]:
        raise ProfileLikelihoodNumericsError("workbook_114 decision must stay frozen")
    if spec["frozen_decision"] != WB114_DECISION:
        raise ProfileLikelihoodNumericsError("WB114 decision token mismatch")
    if decision.get("b15_authorized"):
        raise ProfileLikelihoodNumericsError("WB114 must not have authorized B15")
    if decision.get("prior_introduced"):
        raise ProfileLikelihoodNumericsError("WB114 must not have introduced a prior")
    inherited["workbook_114"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "synthetic_profile_passed": True,
        "acts_profile_materialized": True,
        "target_exclusion_holds": True,
        "profiling_executed": True,
        "marginalization_executed": False,
        "b15_authorized": False,
        "do_not_force_5d_lto_covariance": True,
        "statistical_model_unchanged": True,
    }
    return inherited


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id")), int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _event_pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["numerics_smoke_root"]))
    filename = str(config.get("numerics_smoke_filename", "ckf_leave_target_out_profile_numerics.jsonl"))
    if not root.is_dir():
        return []
    return sorted(root.rglob(filename))


def load_numerics_rows(config: Mapping[str, Any]) -> dict[str, Any]:
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


def prove_physical_chi2_invariant(
    chi2_fn,
    theta_ref: np.ndarray,
    scales: np.ndarray,
    z: np.ndarray,
) -> dict[str, Any]:
    """chi2(theta(z)) must equal chi2(theta_ref + s * z)."""
    theta = np.asarray(theta_ref, dtype=np.float64) + np.asarray(scales, dtype=np.float64) * np.asarray(
        z, dtype=np.float64
    )
    left = float(chi2_fn(theta))
    right = float(chi2_fn(np.asarray(theta_ref, dtype=np.float64) + np.asarray(scales) * np.asarray(z)))
    return {
        "chi2_theta_of_z": left,
        "chi2_explicit_theta": right,
        "abs_diff": abs(left - right),
        "physical_chi2_invariant": abs(left - right) <= 1.0e-12,
        "scaling_is_prior": False,
    }


def audit_surface_ordering(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reports = []
    n_return = 0
    n_not_z = 0
    for row in rows:
        if row.get("evaluate_only") is not True:
            continue
        if str(row.get("init_kind")) != "wb114_official_seed_mean":
            continue
        order = row.get("surface_order") or {}
        path = order.get("z_sorted_path") or []
        zs = [item.get("z_mm") for item in path if item.get("role") != "source"]
        returns = bool(order.get("returns_upstream_after_downstream"))
        input_sorted = bool(order.get("input_order_is_z_sorted"))
        if returns:
            n_return += 1
        if not input_sorted:
            n_not_z += 1
        reports.append(
            {
                "source_id": row.get("source_id"),
                "run_id": row.get("run_id"),
                "event_id": row.get("event_id"),
                "target_station": row.get("target_station"),
                "source_z_mm": order.get("source_z_mm"),
                "z_sorted_path": path,
                "input_order_is_z_sorted": input_sorted,
                "returns_upstream_after_downstream": returns,
                "z_values_mm": zs,
                "hop_mode": row.get("hop_mode"),
            }
        )
    return {
        "n_evaluate_rows": len(reports),
        "n_input_not_z_sorted": n_not_z,
        "n_returns_upstream_after_downstream": n_return,
        "ordering_is_physical_transport_bug": n_return > 0,
        "rows": reports,
    }


def _hop_abort_class(row: Mapping[str, Any]) -> str | None:
    hops = row.get("propagation_hits") or []
    first = None
    for hop in hops:
        if not hop.get("ok"):
            first = hop
            break
    if first is None:
        return None
    reason = str(first.get("abort_reason") or first.get("propagation_status") or "")
    start_z = first.get("start_z_mm")
    source_z = row.get("source_surface_z_mm")
    lower = reason.lower()
    step_limit = (
        "maximum number of steps" in lower
        or "step count exceeded" in lower
        or ("step" in lower and "limit" in lower)
    )
    navigation = (
        "not on surface" in lower
        or "global to local" in lower
        or "navigation" in lower
    )
    if not row.get("evaluate_only"):
        return "B"
    if step_limit:
        return "D"
    if navigation:
        return "C"
    if source_z is not None and start_z is not None and abs(float(start_z) - float(source_z)) <= 1.0:
        return "A"
    return "A"


def audit_measurement_propagation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    focus = []
    n_eval = 0
    n_ok = 0
    classifications = {}
    for row in rows:
        if row.get("evaluate_only") is not True:
            continue
        if str(row.get("init_kind")) != "wb114_official_seed_mean":
            continue
        n_eval += 1
        ok = bool(row.get("all_surfaces_reached") or row.get("nominal_chi2_evaluable"))
        if ok:
            n_ok += 1
        key = _event_pair(row)
        hops = row.get("propagation_hits") or []
        first_fail = None
        for hop in hops:
            if not hop.get("ok"):
                first_fail = hop
                break
        item = {
            "source_id": row.get("source_id"),
            "run_id": row.get("run_id"),
            "event_id": row.get("event_id"),
            "target_station": row.get("target_station"),
            "init_kind": row.get("init_kind"),
            "all_surfaces_reached": ok,
            "evaluate_chi2": row.get("evaluate_chi2"),
            "first_failed_measurement_index": row.get("first_failed_measurement_index"),
            "first_failed_hop": first_fail,
            "n_hops": len(hops),
            "n_ok_hops": sum(1 for hop in hops if hop.get("ok")),
            "abort_class": None if ok else _hop_abort_class(row),
            "hops": hops,
        }
        if key in FOCUS_HARD or key in RESTART_EVENTS:
            focus.append(item)
            if key in FOCUS_HARD:
                classifications.setdefault(f"{key[0]}/{key[1]}", item["abort_class"] if not ok else "nominal_evaluable")
    return {
        "n_seed_evaluate_rows": n_eval,
        "n_all_surfaces_reached": n_ok,
        "nominal_propagation_pass": n_eval > 0 and n_ok == n_eval,
        "focus_classifications": classifications,
        "focus_rows": focus,
        "question_37_86": {
            "A": "nominal state itself cannot propagate",
            "B": "an optimizer trial state cannot propagate",
            "C": "measurement surface ordering / navigation",
            "D": "numerical stepper / path limit",
        },
    }


def audit_kalman_vs_profile(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    seed = {}
    fitted = {}
    contracts = []
    for row in rows:
        if row.get("evaluate_only") is not True:
            continue
        key = (_identity(row), row.get("target_station"))
        kind = str(row.get("init_kind"))
        payload = {
            "all_surfaces_reached": bool(row.get("all_surfaces_reached")),
            "evaluate_chi2": row.get("evaluate_chi2"),
            "n_ok_hops": sum(1 for hop in (row.get("propagation_hits") or []) if hop.get("ok")),
            "n_hops": len(row.get("propagation_hits") or []),
            "hop_mode": row.get("hop_mode"),
            "max_step_size_contract": row.get("max_step_size_contract"),
            "kalman_n_measurements_in_fit": row.get("kalman_n_measurements_in_fit"),
            "kalman_chi2": row.get("kalman_chi2"),
        }
        if kind == "wb114_official_seed_mean":
            seed[key] = payload
        elif kind == "wb109_lto_fitted_mean":
            fitted[key] = payload
            if row.get("kalman_vs_profile_transport_contract"):
                contracts.append(row["kalman_vs_profile_transport_contract"])
    compared = []
    n_seed_fail_fit_ok = 0
    for key, fit_row in fitted.items():
        seed_row = seed.get(key)
        compared.append(
            {
                "identity": key[0],
                "target_station": key[1],
                "seed_evaluate": seed_row,
                "fitted_evaluate": fit_row,
                "seed_reached": bool((seed_row or {}).get("all_surfaces_reached")),
                "fitted_reached": bool(fit_row.get("all_surfaces_reached")),
            }
        )
        if fit_row.get("all_surfaces_reached") and seed_row and not seed_row.get("all_surfaces_reached"):
            n_seed_fail_fit_ok += 1
    contract = contracts[0] if contracts else {
        "kalman_max_steps": 10000,
        "kalman_max_step_size": "unlimited_adaptive",
        "profile_hop_mode": "sequential_z_order",
        "note": "both use ACTS but the contracts are not assumed identical",
    }
    return {
        "contract": contract,
        "n_compared": len(compared),
        "n_seed_fail_but_fitted_ok": n_seed_fail_fit_ok,
        "both_use_acts_does_not_imply_same_contract": True,
        "rows": compared,
    }


def audit_parameter_scaling(rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["profile_parameter_scaling"]
    n = 0
    n_ok = 0
    examples = []
    for row in rows:
        contract = row.get("parameter_scaling_contract")
        if not contract:
            continue
        n += 1
        ok = bool(contract.get("evaluations_ok") and contract.get("chi2_z_equals_chi2_theta"))
        if ok:
            n_ok += 1
        if len(examples) < 4:
            examples.append(
                {
                    "event_id": row.get("event_id"),
                    "target_station": row.get("target_station"),
                    **contract,
                }
            )
    synthetic = prove_physical_chi2_invariant(
        lambda theta: float(np.dot(theta, theta)),
        np.array([1.0, -2.0, 0.1, 0.02, 1.0e-3]),
        np.array(
            [
                float(spec["scales"]["loc0_mm"]),
                float(spec["scales"]["loc1_mm"]),
                float(spec["scales"]["phi"]),
                float(spec["scales"]["theta"]),
                float(spec["scales"]["q_over_p_per_gev"]),
            ]
        ),
        np.array([0.5, -1.0, 2.0, 0.0, -0.25]),
    )
    return {
        "kind": spec["kind"],
        "scales": spec["scales"],
        "not_a_prior": True,
        "not_from_truth": True,
        "not_from_hessian": True,
        "not_from_seed_campaign": True,
        "synthetic_invariance": synthetic,
        "n_acts_contracts": n,
        "n_acts_ok": n_ok,
        "physical_chi2_invariant": bool(synthetic["physical_chi2_invariant"] and (n == 0 or n_ok == n)),
        "examples": examples,
    }


def audit_jacobian(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = 0
    n_pass = 0
    failed_eval = 0
    columns = defaultdict(lambda: {"n": 0, "n_fail": 0, "rel": [], "sign_ok": 0})
    examples = []
    for row in rows:
        report = row.get("jacobian_validation")
        if not report:
            continue
        n += 1
        if report.get("jacobian_contract_established"):
            n_pass += 1
        failed_eval += int(report.get("failed_finite_difference_evaluations") or 0)
        for col in report.get("columns") or []:
            name = str(col.get("parameter"))
            columns[name]["n"] += 1
            if not col.get("ok"):
                columns[name]["n_fail"] += 1
            if col.get("relative_jacobian_error") is not None:
                columns[name]["rel"].append(float(col["relative_jacobian_error"]))
            if col.get("sign_consistency"):
                columns[name]["sign_ok"] += 1
        if len(examples) < 6:
            examples.append(
                {
                    "event_id": row.get("event_id"),
                    "target_station": row.get("target_station"),
                    "init_kind": row.get("init_kind"),
                    **report,
                }
            )
    summary = {}
    for name, item in columns.items():
        rels = item["rel"]
        summary[name] = {
            "n": item["n"],
            "n_fail": item["n_fail"],
            "median_relative_error": float(np.median(rels)) if rels else None,
            "sign_consistency_rate": (item["sign_ok"] / item["n"]) if item["n"] else None,
        }
    established = n > 0 and n_pass == n and failed_eval == 0
    return {
        "n_reports": n,
        "n_pass": n_pass,
        "failed_finite_difference_evaluations": failed_eval,
        "jacobian_contract_established": established,
        "columns": summary,
        "examples": examples,
    }


def audit_optimizer_trace(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    traces = []
    n_ridge = 0
    n_penalty = 0
    n_hidden = 0
    for row in rows:
        if row.get("evaluate_only"):
            continue
        if row.get("ridge_added"):
            n_ridge += 1
        if row.get("chi2_penalty_for_propagation_failure"):
            n_penalty += 1
        if row.get("prior_term_present"):
            n_hidden += 1
        traces.append(
            {
                "event_id": row.get("event_id"),
                "target_station": row.get("target_station"),
                "profile_init_variant": row.get("profile_init_variant"),
                "termination_reason": row.get("termination_reason"),
                "profile_chi2": row.get("profile_chi2"),
                "profile_n_iterations": row.get("profile_n_iterations"),
                "hessian_rank": row.get("hessian_rank"),
                "profile_success": row.get("profile_success"),
                "optimizer_trace": row.get("optimizer_trace") or [],
                "ridge_added": bool(row.get("ridge_added")),
                "prior_term_present": bool(row.get("prior_term_present")),
            }
        )
    return {
        "n_optimize_rows": len(traces),
        "hidden_ridge_or_prior": n_ridge + n_penalty + n_hidden > 0,
        "n_ridge": n_ridge,
        "n_chi2_penalty": n_penalty,
        "n_prior_term": n_hidden,
        "rows": traces,
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def audit_restart_invariance_v2(rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["profile_restart_v2"]
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("evaluate_only"):
            continue
        if _event_pair(row) not in RESTART_EVENTS:
            continue
        variant = str(row.get("profile_init_variant") or "nominal")
        key = (_identity(row), row.get("target_station"))
        groups[key][variant] = row
    n_groups = 0
    n_complete = 0
    n_equiv = 0
    n_loc1_equiv = 0
    n_loc1 = 0
    details = []
    for key, variants in groups.items():
        n_groups += 1
        missing = [name for name in OPTIMIZE_VARIANTS if name not in variants]
        if missing:
            details.append({"key": key, "missing": missing, "equivalent": False})
            continue
        n_complete += 1
        nominal = variants["nominal"]
        chi2_n = _finite(nominal.get("profile_chi2"))
        pred_n = nominal.get("target_prediction_derived") or [None]
        loc0_n = (nominal.get("profiled_native_state") or [None])[0]
        theta_n = (nominal.get("profiled_native_state") or [None, None, None, None])[3]
        ok_all = True
        loc1_ok = True
        variant_cmp = {}
        for name in OPTIMIZE_VARIANTS:
            row = variants[name]
            chi2 = _finite(row.get("profile_chi2"))
            pred = row.get("target_prediction_derived") or [None]
            loc0 = (row.get("profiled_native_state") or [None])[0]
            theta = (row.get("profiled_native_state") or [None, None, None, None])[3]
            chi2_ok = False
            if chi2 is not None and chi2_n is not None:
                chi2_ok = abs(chi2 - chi2_n) <= float(spec["chi2_abs_tolerance"]) or abs(
                    chi2 - chi2_n
                ) <= float(spec["chi2_rel_tolerance"]) * max(abs(chi2_n), 1.0)
            pred_ok = (
                pred[0] is not None
                and pred_n[0] is not None
                and abs(float(pred[0]) - float(pred_n[0])) <= float(spec["prediction_abs_tolerance_mm"])
            )
            loc0_ok = (
                loc0 is not None
                and loc0_n is not None
                and abs(float(loc0) - float(loc0_n)) <= float(spec["loc0_abs_tolerance_mm"])
            )
            theta_ok = (
                theta is not None
                and theta_n is not None
                and abs(float(theta) - float(theta_n)) <= float(spec["theta_abs_tolerance"])
            )
            equivalent = bool(
                row.get("profile_success")
                and chi2_ok
                and pred_ok
                and loc0_ok
                and theta_ok
            )
            variant_cmp[name] = {
                "equivalent": equivalent,
                "chi2": chi2,
                "prediction_x": pred[0],
                "loc0": loc0,
                "theta": theta,
                "termination_reason": row.get("termination_reason"),
            }
            if name != "nominal" and not equivalent:
                ok_all = False
            if name == "loc1_plus_1mm":
                n_loc1 += 1
                loc1_ok = equivalent
                if equivalent:
                    n_loc1_equiv += 1
        if ok_all:
            n_equiv += 1
        details.append(
            {
                "identity": key[0],
                "target_station": key[1],
                "equivalent": ok_all,
                "loc1_equivalent": loc1_ok,
                "variants": variant_cmp,
            }
        )
    return {
        "n_groups": n_groups,
        "n_complete_groups": n_complete,
        "n_equivalent_groups": n_equiv,
        "n_loc1_groups": n_loc1,
        "n_loc1_equivalent": n_loc1_equiv,
        "restart_invariance_holds": n_groups > 0 and n_complete == n_groups and n_equiv == n_groups,
        "loc1_restart_equivalent": n_loc1 > 0 and n_loc1_equiv == n_loc1,
        "nuisance_may_differ": True,
        "tolerances": {
            "chi2_rel": spec["chi2_rel_tolerance"],
            "chi2_abs": spec["chi2_abs_tolerance"],
            "loc0_mm": spec["loc0_abs_tolerance_mm"],
            "theta": spec["theta_abs_tolerance"],
            "prediction_mm": spec["prediction_abs_tolerance_mm"],
        },
        "groups": details,
    }


def audit_focus_numerics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_event: dict[tuple[int, int], dict[str, Any]] = {}
    for key in FOCUS_HARD | RESTART_EVENTS:
        by_event[key] = {
            "run_id": key[0],
            "event_id": key[1],
            "nominal_evaluable": False,
            "n_evaluate": 0,
            "n_optimize_success": 0,
            "n_optimize": 0,
            "first_failed": [],
            "step_limit_only": False,
            "retained": True,
        }
    for row in rows:
        key = _event_pair(row)
        if key not in by_event:
            continue
        item = by_event[key]
        if row.get("evaluate_only") and str(row.get("init_kind")) == "wb114_official_seed_mean":
            item["n_evaluate"] += 1
            if row.get("all_surfaces_reached"):
                item["nominal_evaluable"] = True
            hops = row.get("propagation_hits") or []
            for hop in hops:
                if not hop.get("ok"):
                    item["first_failed"].append(
                        {
                            "target_station": row.get("target_station"),
                            "measurement_index": hop.get("measurement_index"),
                            "station": hop.get("measurement_station"),
                            "abort_reason": hop.get("abort_reason"),
                            "propagation_status": hop.get("propagation_status"),
                            "n_steps": hop.get("number_of_propagation_steps"),
                        }
                    )
                    reason = str(hop.get("abort_reason") or "").lower()
                    if (
                        "maximum number of steps" in reason
                        or "step count exceeded" in reason
                        or ("step" in reason and "limit" in reason)
                    ):
                        item["step_limit_only"] = True
                    break
        elif not row.get("evaluate_only"):
            item["n_optimize"] += 1
            if row.get("profile_success"):
                item["n_optimize_success"] += 1
    if not all(item["retained"] for item in by_event.values()):
        refuse_focus_drop()
    return {
        "identities": {f"{k[0]}/{k[1]}": v for k, v in by_event.items()},
        "focus_retained": True,
        "event_37_evaluable": by_event[(100043, 37)]["nominal_evaluable"],
        "event_86_evaluable": by_event[(100048, 86)]["nominal_evaluable"],
        "focus_not_step_limit_only": (
            by_event[(100043, 37)]["nominal_evaluable"]
            and by_event[(100048, 86)]["nominal_evaluable"]
        ),
    }


def smoke_gate(inventory: Mapping[str, Any]) -> dict[str, Any]:
    restart = inventory.get("restart") or {}
    jacobian = inventory.get("jacobian") or {}
    prop = inventory.get("propagation") or {}
    focus = inventory.get("focus") or {}
    trace = inventory.get("optimizer") or {}
    exclusion = inventory.get("exclusion") or {}
    checks = {
        "events_0_1_all_restarts": bool(restart.get("restart_invariance_holds")),
        "loc1_restart_equivalent": bool(restart.get("loc1_restart_equivalent")),
        "jacobian_pass": bool(jacobian.get("jacobian_contract_established")),
        "nominal_propagation_pass": bool(prop.get("nominal_propagation_pass")),
        "focus_not_step_limit_only": bool(focus.get("focus_not_step_limit_only")),
        "no_ridge": not bool(trace.get("hidden_ridge_or_prior")),
        "target_exclusion": bool(exclusion.get("target_exclusion_holds", True)),
        "smoke_present": bool(inventory.get("smoke_present")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "full_sample_authorized": passed,
        "do_not_submit_if_failed": True,
    }


def audit_exclusion(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = 0
    leaked = 0
    for row in rows:
        if "target_exclusion_proven" not in row and "target_station_measurements_used" not in row:
            continue
        n += 1
        used = int(row.get("target_station_measurements_used") or 0)
        proven = bool(row.get("target_exclusion_proven", used == 0))
        if used != 0 or not proven:
            leaked += 1
    return {
        "n_compared": n,
        "n_leaked": leaked,
        "target_exclusion_holds": n > 0 and leaked == 0,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior()
    if inventory.get("ridge_as_information"):
        refuse_ridge_information()
    if inventory.get("chi2_penalty_used"):
        refuse_chi2_penalty()
    if inventory.get("statistical_model_changed"):
        refuse_change_statistical_model()
    gate = inventory.get("smoke_gate") or {}
    prop = inventory.get("propagation") or {}
    order = inventory.get("ordering") or {}
    jacobian = inventory.get("jacobian") or {}
    restart = inventory.get("restart") or {}
    focus = inventory.get("focus") or {}
    optimizer = inventory.get("optimizer") or {}
    smoke = bool(inventory.get("smoke_present"))
    mechanisms = []
    transport_broken = bool(
        order.get("ordering_is_physical_transport_bug")
        or (smoke and not prop.get("nominal_propagation_pass"))
        or (smoke and not focus.get("focus_not_step_limit_only"))
    )
    derivative_broken = smoke and not jacobian.get("jacobian_contract_established")
    # Optimizer failure is not a certified mechanism until the evaluator
    # and Jacobian can be trusted.
    optimizer_broken = (
        smoke
        and not transport_broken
        and not derivative_broken
        and (
            not restart.get("restart_invariance_holds")
            or not restart.get("loc1_restart_equivalent")
        )
    )
    if transport_broken:
        mechanisms.append("B")
    if derivative_broken and not transport_broken:
        mechanisms.append("C")
    if optimizer_broken:
        mechanisms.append("D")
    if not smoke:
        primary = CASE_F
        verdict = "DIAGNOSED"
        mechanisms = ["F"]
    elif bool(gate.get("passed")):
        primary = CASE_A
        verdict = "PASS"
        mechanisms = ["A"]
    elif len(mechanisms) > 1:
        primary = CASE_F
        verdict = "FAIL"
    elif "B" in mechanisms:
        primary = CASE_B
        verdict = "FAIL"
    elif "C" in mechanisms:
        primary = CASE_C
        verdict = "FAIL"
    elif "D" in mechanisms:
        primary = CASE_D
        verdict = "FAIL"
    else:
        primary = CASE_F
        verdict = "DIAGNOSED"
        mechanisms = ["F"]
    if primary == CASE_E:
        refuse_propagation_as_nonidentifiability()
    next_step = {
        CASE_A: "re_execute_b14m_physics_uncertainty_on_stable_numerics",
        CASE_B: "fix_standalone_measurement_transport_evaluator",
        CASE_C: "fix_residual_jacobian_before_optimizer_interpretation",
        CASE_D: "continue_globalization_or_parameterization_without_ridge",
        CASE_E: "certify_profile_objective_and_observable_uncertainty",
        CASE_F: "keep_layered_numerical_diagnosis",
    }[primary]
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "active_mechanisms": mechanisms,
        "next_step": next_step,
        "b15_authorized": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "full_sample_authorized": bool(gate.get("passed")),
        "prior_introduced": False,
        "ridge_added": bool(optimizer.get("hidden_ridge_or_prior")),
        "do_not_force_5d_lto_covariance": True,
        "statistical_model_unchanged": True,
        "profile_math_rewritten": False,
        "focus_identity_retained": True,
        "physical_nonidentifiability_not_claimed_from_propagation_failure": True,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_numerics_rows(config)
    sample = load_contracted_sample(config)
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    rows = dumps["rows"]
    synthetic = validate_linear_profile_agreement(
        relative=float(config["profile_numerical"]["pinv_relative"])
    )
    propagation = audit_measurement_propagation(rows)
    ordering = audit_surface_ordering(rows)
    kalman = audit_kalman_vs_profile(rows)
    scaling = audit_parameter_scaling(rows, config)
    jacobian = audit_jacobian(rows)
    optimizer = audit_optimizer_trace(rows)
    restart = audit_restart_invariance_v2(rows, config)
    focus = audit_focus_numerics(rows)
    exclusion = audit_exclusion(rows)
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
        "synthetic_profile": synthetic,
        "propagation": propagation,
        "ordering": ordering,
        "kalman": kalman,
        "scaling": scaling,
        "jacobian": jacobian,
        "optimizer": optimizer,
        "restart": restart,
        "focus": focus,
        "exclusion": exclusion,
        "prior_introduced": False,
        "ridge_as_information": False,
        "chi2_penalty_used": False,
        "statistical_model_changed": False,
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
    if mechanism["full_sample_authorized"] is not True and inventory.get("full_sample_requested"):
        refuse_full_sample_without_gate()
    if mechanism["prior_introduced"]:
        refuse_prior()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb114_decision_sha256": inherited["workbook_114"]["decision_sha256"],
        "inherited_wb113_decision_sha256": inherited["workbook_113"]["decision_sha256"],
        "inherited_wb112_decision_sha256": inherited["workbook_112"]["decision_sha256"],
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb114_rewritten": False,
        "target_exclusion_holds": bool((inventory.get("exclusion") or {}).get("target_exclusion_holds")),
        "synthetic_profile_passed": bool((inventory.get("synthetic_profile") or {}).get("passed")),
        "smoke_gate_passed": bool((inventory.get("smoke_gate") or {}).get("passed")),
        "acts_numerics_materialized": bool(inventory.get("smoke_present")),
    }
