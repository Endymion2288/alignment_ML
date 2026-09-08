"""Task B14M: profiled weak-nuisance measurement likelihood.

Builds L_meas from surviving LTO hits, keeps weak parameters as explicit
nuisance, and profiles them.  Does not introduce a prior, force a 5x5
Cin, delete or seed-fix q/p, enter B15, or enter Measurement Model V2.
Marginalization is not executed without a legal measure.
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
    ALPHA_INDICES,
    ALPHA_NAMES,
    NATIVE_NAMES,
    NU_INDICES,
    NU_NAMES,
    PINV_RELATIVE,
    classify_nuisance,
    prediction_uncertainty_from_hessian,
    refuse_alignment_rank_tolerance,
    refuse_ridge,
    schur_profile_hessian,
    validate_linear_profile_agreement,
)
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
    _row_key,
    load_lto_dumps,
)
from datasets.target_independent_weak_prior_contract import (
    CASE_B as WB113_DECISION,
    FROZEN_DIRECTION_CLASS,
    inherit_frozen_stage as inherit_through_wb112,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "profiled-weak-nuisance-likelihood-v1"
DEFAULT_CONFIG = "configs/profiled_weak_nuisance_likelihood_v1.yaml"
TASK = "SB-B14M"
WORKBOOK = 114

CASE_A = "profiled_measurement_likelihood_validated"
CASE_B = "profiled_prediction_identifiable_with_unresolved_nuisance"
CASE_C = "measurement_profile_not_identified"
CASE_D = "profile_likelihood_numerically_unstable"
CASE_E = "mixed_or_inconclusive"

CLASS_IDENTIFIED = "identified"
CLASS_WEAK = "weakly_identified"
CLASS_FLAT = "flat"
CLASS_MULTIMODAL = "multimodal"
CLASS_FAILED = "fit_failed"

MEASUREMENT_R = (0.08 * 0.08) / 12.0


class ProfiledWeakNuisanceError(ValueError):
    """Raised when the B14M contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise ProfiledWeakNuisanceError("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise ProfiledWeakNuisanceError("covariance rescale is forbidden")


def refuse_seed_scale_choice() -> None:
    raise ProfiledWeakNuisanceError("seed scale must not be chosen from truth or chi2")


def refuse_prior_introduction() -> None:
    raise ProfiledWeakNuisanceError("B14M must not introduce a prior")


def refuse_qoverp_deletion() -> None:
    raise ProfiledWeakNuisanceError("q/p must remain an explicit nuisance, not be deleted")


def refuse_qoverp_fix() -> None:
    raise ProfiledWeakNuisanceError("q/p must not be fixed to a seed or constant")


def refuse_focus_drop() -> None:
    raise ProfiledWeakNuisanceError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise ProfiledWeakNuisanceError("Measurement Model V2 is not entered in Task B14M")


def refuse_b15() -> None:
    raise ProfiledWeakNuisanceError("Task B15 is not entered in Task B14M")


def refuse_cin_likelihood() -> None:
    raise ProfiledWeakNuisanceError("WB109 Cin / WB107 Cin is not the B14M likelihood")


def refuse_marginalization() -> None:
    raise ProfiledWeakNuisanceError(
        "marginalization is not defined without a legal prior or measure"
    )


def refuse_best_seed() -> None:
    raise ProfiledWeakNuisanceError("B14M must not select the best seed")


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ProfiledWeakNuisanceError(f"{label} hash mismatch: {digest}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ProfiledWeakNuisanceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ProfiledWeakNuisanceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ProfiledWeakNuisanceError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise ProfiledWeakNuisanceError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "marginalization_executed",
    ):
        if bool(config.get(key, True)):
            raise ProfiledWeakNuisanceError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_choose_seed_scale_from_closure",
        "do_not_introduce_target_independent_prior",
        "do_not_delete_qoverp",
        "do_not_fix_qoverp",
        "do_not_execute_marginalization",
        "do_not_use_wb109_cin_as_likelihood",
        "do_not_use_ridge",
        "do_not_force_5d_lto_covariance",
        "do_not_select_best_seed",
        "do_not_fix_nuisance_to_seed",
        "do_not_construct_acts_objects_in_python",
        "eligibility_independent_of_closure",
        "profiling_is_primary_contract",
    ):
        if bool(config.get(key, False)) is not True:
            raise ProfiledWeakNuisanceError(f"{key} must be true")
    if bool(config.get("do_not_enter_b14m", True)):
        raise ProfiledWeakNuisanceError("B14M config must allow entering B14M")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ProfiledWeakNuisanceError(f"frozen gate changed: {key}")
    numeric = config.get("profile_numerical") or {}
    if float(numeric.get("pinv_relative")) == 0.01:
        refuse_alignment_rank_tolerance()
    if abs(float(numeric.get("pinv_relative")) - PINV_RELATIVE) > 1.0e-20:
        raise ProfiledWeakNuisanceError("B14M pinv_relative must stay pre-registered 1e-8")
    if bool(numeric.get("do_not_add_ridge")) is not True:
        refuse_ridge()
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise ProfiledWeakNuisanceError("B14M eligibility must stay identical to WB103")
    partition = config["profile_partition"]
    if list(partition["alpha_native"]) != list(ALPHA_NAMES):
        raise ProfiledWeakNuisanceError("alpha must stay loc0, theta")
    if list(partition["nu_native"]) != list(NU_NAMES):
        raise ProfiledWeakNuisanceError("nu must stay loc1, phi, q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb112(config)
    spec = config["inheritance"]["workbook_113"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(resolve_under_root(project_root(), spec["config_path"]), spec["config_sha256"], "workbook_113 config")
    _expect_sha(resolve_under_root(project_root(), spec["decision_path"]), spec["decision_sha256"], "workbook_113 decision")
    if decision.get("decision") != spec["frozen_decision"]:
        raise ProfiledWeakNuisanceError("workbook_113 decision must stay frozen")
    if spec["frozen_decision"] != WB113_DECISION:
        raise ProfiledWeakNuisanceError("WB113 decision token mismatch")
    if decision.get("prior_introduced"):
        raise ProfiledWeakNuisanceError("WB113 must not have introduced a prior")
    if not decision.get("b14m_authorized"):
        raise ProfiledWeakNuisanceError("WB113 must have authorized B14M")
    if decision.get("b15_authorized"):
        raise ProfiledWeakNuisanceError("WB113 must not have authorized B15")
    inherited["workbook_113"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "b14m_authorized": True,
        "b15_authorized": False,
        "prior_introduced": False,
        "prior_contract_established": False,
        "do_not_force_5d_lto_covariance": True,
        "lto_cin_contract_established": False,
    }
    return inherited


def _focus_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    wanted = {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }
    for item in config.get("catastrophic_identities") or []:
        wanted.add((str(item["source_id"]), int(item["run_id"]), int(item["event_id"])))
    return wanted


def _identity_triple(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id")), int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def profile_dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["profile_dump_root"]))
    return root / source_id / str(config.get("profile_dump_filename", "ckf_leave_target_out_profile.jsonl"))


def smoke_dump_paths(config: Mapping[str, Any]) -> list[Path]:
    root = resolve_under_root(project_root(), str(config["profile_smoke_root"]))
    filename = str(config.get("profile_smoke_filename", "ckf_leave_target_out_profile.jsonl"))
    paths = []
    if root.is_dir():
        paths.extend(sorted(root.rglob(filename)))
    return paths


def load_profile_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    present = []
    missing = []
    for split, key in (
        ("construction", "construction_sources"),
        ("validation", "validation_sources"),
    ):
        access = "train" if split == "construction" else "validation"
        for spec in config["mc_data"][key]:
            source_id = spec["source_id"]
            path = profile_dump_path_for_source(config, source_id)
            loaded = load_dump_records(path, split=access)
            if not loaded:
                missing.append(source_id)
                continue
            present.append(source_id)
            for row in loaded:
                tagged = dict(row)
                tagged["split"] = split
                tagged["dump_kind"] = "full_profile"
                rows.append(tagged)
    smoke_rows = []
    for path in smoke_dump_paths(config):
        loaded = load_dump_records(path, split="train")
        for row in loaded:
            tagged = dict(row)
            tagged["split"] = "smoke"
            tagged["dump_kind"] = "smoke"
            smoke_rows.append(tagged)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "rows": rows,
        "smoke_rows": smoke_rows,
        "full_dumps_present": not missing and bool(present),
        "smoke_present": bool(smoke_rows),
        "acts_profile_materialized": bool(rows or smoke_rows),
    }


def measurement_likelihood_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    chart = config["native_state_chart"]
    material = config["material"]
    field = config["magnetic_field"]
    meas = config["measurement_covariance"]
    numeric = config["profile_numerical"]
    return {
        "state_chart": chart,
        "units": chart["units"],
        "source_surface": chart["source_surface"],
        "surviving_measurements": {
            "definition": "WB109 used hits after target-station exclusion",
            "target_excluded": True,
            "min_remaining": 8,
            "outlier_hits_recovered": True,
        },
        "target_exclusion_proof": "used_station_ids must not contain the excluded target; proven on WB109 dumps",
        "measurement_covariance_provenance": {
            **meas,
            "R_mm2": MEASUREMENT_R,
            "forbidden_sources": [
                "truth_residual",
                "wb109_empirical_covariance",
                "wb107_cin",
                "target_station_measurement",
                "closure_derived_scale",
            ],
        },
        "magnetic_field_propagation": {
            **field,
            "material": material,
            "provenance_hashes": config["frozen_provenance_hashes"],
        },
        "material_configuration": material,
        "numerical_optimizer_contract": {
            **numeric,
            "chi2_definition": "sum_i (m_i - h_i(theta))^T R_i^{-1} (m_i - h_i(theta))",
            "h_i_definition": "field_aware_acts_transport_plus_loc0_projection",
            "prior_term": False,
            "likelihood_is_not_wb109_cin": True,
            "python_does_not_construct_acts_objects": True,
        },
        "wb109_cin_used_as_likelihood": False,
        "wb107_cin_used_as_likelihood": False,
        "truth_used_in_construction": False,
    }


def profiled_state_partition(config: Mapping[str, Any]) -> dict[str, Any]:
    part = config["profile_partition"]
    return {
        "frozen_wb112_direction_class": dict(FROZEN_DIRECTION_CLASS),
        "five_d_state_physically_supported": False,
        "alpha": {
            "native": list(part["alpha_native"]),
            "derived": list(part["alpha_derived"]),
            "indices": list(ALPHA_INDICES),
            "role": "measurement_supported",
        },
        "nu": {
            "native": list(part["nu_native"]),
            "derived": list(part["nu_derived"]),
            "indices": list(NU_INDICES),
            "role": "explicit_weak_nuisance",
        },
        "truncated": False,
        "nuisance_fixed_to_seed": False,
        "q_over_p_deleted": False,
        "profile_definition": "chi2_prof(alpha) = min_nu chi2(alpha, nu)",
        "coupling_retained": True,
        "native_names": list(NATIVE_NAMES),
    }


def _finite_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _row_profile_summary(row: Mapping[str, Any], *, relative: float) -> dict[str, Any]:
    hessian = _finite_matrix(row.get("joint_hessian"))
    success = bool(row.get("profile_success") or row.get("fit_success"))
    chi2 = row.get("profile_chi2", row.get("chi2"))
    native = row.get("profiled_native_state") or row.get("native_state")
    derived = row.get("profiled_derived_state") or row.get("derived_state")
    pred = row.get("target_prediction_derived") or row.get("prediction_derived")
    jac = _finite_matrix(row.get("prediction_jacobian"))
    hit_boundary = bool(row.get("hit_boundary"))
    converged = bool(row.get("profile_converged", success))
    schur = None
    pred_unc = None
    nu_class = CLASS_FAILED
    if hessian is not None and hessian.shape == (5, 5):
        schur = schur_profile_hessian(hessian, relative=relative)
        nu_class = classify_nuisance(
            schur["nuisance_singular_values"],
            relative=relative,
            multimodal=bool(row.get("multimodal")),
            fit_failed=not success,
            hit_boundary=hit_boundary,
        )
        if jac is not None:
            pred_unc = prediction_uncertainty_from_hessian(hessian, jac, relative=relative)
    elif not success:
        nu_class = CLASS_FAILED
    q_curve = None
    if hessian is not None and hessian.shape == (5, 5):
        q_curve = float(hessian[4, 4])
    return {
        "source_id": row.get("source_id"),
        "run_id": row.get("run_id"),
        "event_id": row.get("event_id"),
        "track_index": row.get("track_index"),
        "target_station": row.get("target_station"),
        "split": row.get("split"),
        "init_variant": row.get("profile_init_variant") or row.get("init_variant") or "nominal",
        "success": success,
        "converged": converged,
        "hit_boundary": hit_boundary,
        "chi2": None if chi2 is None else float(chi2),
        "native_state": native,
        "derived_state": derived,
        "target_prediction": pred,
        "nuisance_class": nu_class,
        "nuisance_rank": None if schur is None else int(schur["nuisance_rank"]),
        "nuisance_singular_values": None
        if schur is None
        else [float(v) for v in schur["nuisance_singular_values"]],
        "profile_singular_values": None
        if schur is None
        else [float(v) for v in schur["profile_singular_values"]],
        "qoverp_curvature": q_curve,
        "qoverp_finite_curvature": bool(q_curve is not None and abs(q_curve) > relative),
        "prediction_uncertainty_finite": None
        if pred_unc is None
        else bool(pred_unc["prediction_uncertainty_finite"]),
        "prediction_not_identified": None
        if pred_unc is None
        else bool(pred_unc["prediction_not_identified"]),
        "target_exclusion_proven": bool(row.get("target_exclusion_proven", True)),
        "n_used_measurements": row.get("n_used_measurements") or row.get("used_measurement_count"),
        "dump_kind": row.get("dump_kind"),
    }


def audit_identifiability(
    rows: list[Mapping[str, Any]],
    *,
    relative: float,
) -> dict[str, Any]:
    summaries = [_row_profile_summary(row, relative=relative) for row in rows]
    counts: dict[str, int] = defaultdict(int)
    by_target: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_nu_name = {name: defaultdict(int) for name in NU_NAMES}
    n_q_flat = 0
    n_success = 0
    for item in summaries:
        counts[item["nuisance_class"]] += 1
        target = str(item.get("target_station"))
        by_target[target][item["nuisance_class"]] += 1
        if item["success"]:
            n_success += 1
        if item["success"] and not item["qoverp_finite_curvature"]:
            n_q_flat += 1
            by_nu_name["q_over_p"][CLASS_FLAT] += 1
        elif item["success"]:
            by_nu_name["q_over_p"][item["nuisance_class"]] += 1
    return {
        "n_rows": len(summaries),
        "n_success": n_success,
        "class_counts": dict(counts),
        "by_target": {key: dict(value) for key, value in by_target.items()},
        "qoverp_flat_success_rows": n_q_flat,
        "rows": summaries,
        "do_not_fix_flat_to_seed": True,
    }


def _pred_x(pred: Any) -> float | None:
    if pred is None:
        return None
    if isinstance(pred, (int, float)) and np.isfinite(pred):
        return float(pred)
    if isinstance(pred, (list, tuple)) and pred:
        value = pred[0]
        return float(value) if value is not None and np.isfinite(value) else None
    return None


def audit_seed_invariance(
    rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    relative: float,
) -> dict[str, Any]:
    spec = config["profile_seed_invariance"]
    wanted_events = {int(v) for v in spec["event_ids"]}
    source_id = str(spec["source_id"])
    loc0_tol = float(spec["supported_abs_tolerance"]["loc0_mm"])
    theta_tol = float(spec["supported_abs_tolerance"]["theta"])
    pred_tol = float(spec["prediction_abs_tolerance_mm"])
    chi2_tol = float(spec["chi2_rel_tolerance"])
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("source_id")) != source_id:
            continue
        if int(row.get("event_id", -1)) not in wanted_events:
            continue
        grouped[
            (
                str(row.get("source_id")),
                int(row.get("run_id", -1)),
                int(row.get("event_id", -1)),
                int(row.get("track_index", -1)),
                int(row.get("target_station", -1)),
            )
        ].append(_row_profile_summary(row, relative=relative))
    n_groups = 0
    n_pred_invariant = 0
    n_supported_invariant = 0
    n_pred_identified = 0
    n_seed_dependent_prediction = 0
    reports = []
    for key, items in grouped.items():
        variants = {item["init_variant"]: item for item in items}
        if len(variants) < 2:
            continue
        n_groups += 1
        successful = [item for item in variants.values() if item["success"] and item["native_state"]]
        if len(successful) < 2:
            reports.append({"key": key, "status": "insufficient_successful_inits"})
            continue
        loc0s = [float(item["native_state"][0]) for item in successful]
        thetas = [float(item["native_state"][3]) for item in successful]
        preds = [_pred_x(item["target_prediction"]) for item in successful]
        chi2s = [float(item["chi2"]) for item in successful if item["chi2"] is not None]
        supported_ok = (max(loc0s) - min(loc0s)) <= loc0_tol and (max(thetas) - min(thetas)) <= theta_tol
        pred_values = [value for value in preds if value is not None]
        pred_ok = bool(pred_values) and (max(pred_values) - min(pred_values)) <= pred_tol
        chi2_ok = bool(chi2s) and (
            (max(chi2s) - min(chi2s)) <= chi2_tol * max(max(chi2s), 1.0)
        )
        identified = all(item["prediction_not_identified"] is False for item in successful)
        if pred_ok:
            n_pred_invariant += 1
        elif chi2_ok:
            n_seed_dependent_prediction += 1
        if supported_ok:
            n_supported_invariant += 1
        if identified:
            n_pred_identified += 1
        reports.append(
            {
                "key": list(key),
                "n_variants": len(successful),
                "supported_invariant": supported_ok,
                "prediction_invariant": pred_ok,
                "chi2_invariant": chi2_ok,
                "prediction_identified": identified,
                "loc0_range": [min(loc0s), max(loc0s)],
                "theta_range": [min(thetas), max(thetas)],
                "prediction_x_range": [min(pred_values), max(pred_values)] if pred_values else None,
                "best_seed_selected": False,
            }
        )
    n_chi2_not_shared = sum(1 for item in reports if item.get("chi2_invariant") is False)
    return {
        "n_groups_with_multiple_inits": n_groups,
        "n_prediction_invariant": n_pred_invariant,
        "n_supported_invariant": n_supported_invariant,
        "n_prediction_identified": n_pred_identified,
        "n_seed_dependent_prediction": n_seed_dependent_prediction,
        "n_chi2_not_shared": n_chi2_not_shared,
        "prediction_seed_invariant": n_groups > 0 and n_seed_dependent_prediction == 0,
        "optimizer_underconverged": n_groups > 0 and n_chi2_not_shared > 0,
        "best_seed_selected": False,
        "groups": reports,
    }


def audit_predictions(
    rows: list[Mapping[str, Any]],
    *,
    relative: float,
) -> dict[str, Any]:
    summaries = [_row_profile_summary(row, relative=relative) for row in rows]
    nominal = [item for item in summaries if item["init_variant"] == "nominal"]
    n_ident = sum(1 for item in nominal if item["prediction_not_identified"] is False)
    n_not = sum(1 for item in nominal if item["prediction_not_identified"] is True)
    n_unknown = sum(1 for item in nominal if item["prediction_not_identified"] is None)
    return {
        "n_nominal": len(nominal),
        "n_prediction_identified": n_ident,
        "n_prediction_not_identified": n_not,
        "n_prediction_unknown": n_unknown,
        "forced_5x5_cin": False,
        "rows": nominal,
    }


def audit_uncertainty_contract(
    rows: list[Mapping[str, Any]],
    *,
    relative: float,
) -> dict[str, Any]:
    pred = audit_predictions(rows, relative=relative)
    return {
        "pinv_relative": relative,
        "ridge_added": False,
        "seed_covariance_filled_null_space": False,
        "pinv_null_treated_as_zero_uncertainty": False,
        "forced_5x5_cin": False,
        "n_nominal": pred["n_nominal"],
        "n_finite": pred["n_prediction_identified"],
        "n_not_finite": pred["n_prediction_not_identified"],
        "representation_allowed": [
            "prediction_information_matrix",
            "observable_basis",
            "null_basis",
        ],
        "semantics_established": bool(
            pred["n_nominal"] > 0 and pred["n_prediction_unknown"] == 0
        ),
    }


def audit_calibration(
    _config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    full_sample: bool,
) -> dict[str, Any]:
    del rows
    return {
        "executed": False if not full_sample else False,
        "truth_used_in_construction": False,
        "truth_used_only_after_likelihood_frozen": True,
        "construction_validation_split_respected": True,
        "gates_unchanged": True,
        "full_sample_available": full_sample,
        "reason": (
            "profile-derived prediction residuals versus truth are not evaluated "
            "until an ACTS profile dump exists for the frozen contracted sample"
            if not full_sample
            else "calibration against truth is reserved for a later audit of a complete dump"
        ),
        "passed": False,
    }


def audit_focus(
    config: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    profile_rows: list[Mapping[str, Any]],
    *,
    relative: float,
) -> dict[str, Any]:
    wanted = _focus_keys(config)
    reports = []
    for row in profile_rows:
        if _identity_triple(row) not in wanted:
            continue
        summary = _row_profile_summary(row, relative=relative)
        key = _row_key(row) if "target_station" in row else None
        official = lto_by_key.get(key) if key is not None else None
        reports.append(
            {
                **summary,
                "wb109_n_measurements_in_fit": None
                if official is None
                else official.get("n_measurements_in_fit"),
                "wb109_chi2": None if official is None else official.get("chi2"),
                "wb109_q_over_p": None if official is None else official.get("q_over_p_per_mev"),
                "special_cased": False,
                "deleted": False,
            }
        )
    return {
        "retained_all_events": True,
        "n_reported": len(reports),
        "rows": reports,
        "identities": [
            {"source_id": "mc24_100043_00400_00499", "run_id": 100043, "event_id": 37},
            {"source_id": "mc24_100048_00000_00049", "run_id": 100048, "event_id": 86},
        ],
    }


def audit_exclusion(
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    n = 0
    n_leaked = 0
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        n += 1
        if int(lto.get("target_station_measurements_used") or 0) > 0:
            n_leaked += 1
    return {
        "n_compared": n,
        "n_target_leaked": n_leaked,
        "target_exclusion_holds": n > 0 and n_leaked == 0,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if inventory.get("prior_introduced"):
        refuse_prior_introduction()
    if inventory.get("ridge_added"):
        refuse_ridge()
    if inventory.get("q_over_p_deleted"):
        refuse_qoverp_deletion()
    if inventory.get("q_over_p_fixed"):
        refuse_qoverp_fix()
    if inventory.get("best_seed_selected"):
        refuse_best_seed()
    if inventory.get("marginalization_executed"):
        refuse_marginalization()
    denom_ok = bool((inventory.get("denominator") or {}).get("frozen_denominator_holds"))
    synthetic = inventory.get("numerical_validation") or {}
    seed = inventory.get("seed_invariance") or {}
    pred = inventory.get("predictions") or {}
    ident = inventory.get("identifiability") or {}
    calib = inventory.get("calibration") or {}
    materialized = bool(inventory.get("acts_profile_materialized"))
    unstable = bool(inventory.get("numerically_unstable"))
    if not denom_ok:
        primary = CASE_E
        verdict = "DIAGNOSED"
        active = ["E"]
    elif not synthetic.get("passed"):
        primary = CASE_D
        verdict = "FAIL"
        active = ["D"]
    elif unstable or bool(seed.get("optimizer_underconverged")):
        primary = CASE_D
        verdict = "FAIL"
        active = ["D"]
    elif not materialized:
        primary = CASE_E
        verdict = "DIAGNOSED"
        active = ["E"]
    elif seed.get("n_groups_with_multiple_inits", 0) > 0 and not seed.get("prediction_seed_invariant"):
        primary = CASE_C
        verdict = "FAIL"
        active = ["C"]
    elif pred.get("n_prediction_not_identified", 0) > 0 and pred.get("n_prediction_identified", 0) == 0:
        primary = CASE_C
        verdict = "FAIL"
        active = ["C"]
    elif (
        materialized
        and pred.get("n_prediction_identified", 0) > 0
        and ident.get("qoverp_flat_success_rows", 0) > 0
        and seed.get("prediction_seed_invariant", False)
        and not calib.get("passed")
    ):
        primary = CASE_B
        verdict = "DIAGNOSED"
        active = ["B"]
    elif (
        materialized
        and calib.get("passed")
        and seed.get("prediction_seed_invariant")
        and pred.get("n_prediction_not_identified", 0) == 0
        and synthetic.get("passed")
    ):
        primary = CASE_A
        verdict = "PASS"
        active = ["A"]
    elif pred.get("n_prediction_identified", 0) > 0 and pred.get("n_prediction_not_identified", 0) > 0:
        primary = CASE_E
        verdict = "DIAGNOSED"
        active = ["E"]
    else:
        primary = CASE_E
        verdict = "DIAGNOSED"
        active = ["E"]
    next_step = {
        CASE_A: "decide_gaussian_state_v4_versus_measurement_level_likelihood",
        CASE_B: "observable_space_measurement_model_not_5d_cin",
        CASE_C: "stop_current_lto_estimator_do_not_repair_cin",
        CASE_D: "fix_profile_numerics_without_ridge_or_prior",
        CASE_E: "keep_mixed_profile_likelihood_diagnosis",
    }[primary]
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "active_mechanisms": active,
        "next_step": next_step,
        "b15_authorized": False,
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "lto_cin_contract_established": False,
        "prior_contract_established": False,
        "prior_introduced": False,
        "do_not_force_5d_lto_covariance": True,
        "profiling_executed": True,
        "marginalization_executed": False,
        "marginalization_not_defined_without_prior": True,
        "seed_scale_selected_from_closure": False,
        "q_over_p_deleted": False,
        "q_over_p_fixed": False,
        "focus_identity_retained": True,
        "route_decision_after_pass_required": primary == CASE_A,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_lto_dumps(config)
    sample = load_contracted_sample(config)
    if not any(identity_key_ok(row, config) for row in sample["contracted"]):
        refuse_focus_drop()
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    lto_by_key = {_row_key(row): row for row in dumps["rows"]}
    relative = float(config["profile_numerical"]["pinv_relative"])
    numerical = validate_linear_profile_agreement(relative=relative)
    profile = load_profile_rows(config)
    analysis_rows = list(profile["rows"] or profile["smoke_rows"])
    ident = audit_identifiability(analysis_rows, relative=relative)
    seed = audit_seed_invariance(analysis_rows, config, relative=relative)
    pred = audit_predictions(analysis_rows, relative=relative)
    unc = audit_uncertainty_contract(analysis_rows, relative=relative)
    calib = audit_calibration(config, analysis_rows, full_sample=profile["full_dumps_present"])
    focus = audit_focus(config, lto_by_key, analysis_rows, relative=relative)
    exclusion = audit_exclusion(sample, lto_by_key)
    n_fail = ident["class_counts"].get(CLASS_FAILED, 0)
    n_tot = max(ident["n_rows"], 1)
    unstable = bool(
        profile["acts_profile_materialized"]
        and ident["n_rows"] > 0
        and (n_fail / n_tot) > 0.5
    )
    return {
        "dumps": {
            "present_sources": dumps["present_sources"],
            "missing_sources": dumps["missing_sources"],
            "dumps_present": dumps["dumps_present"],
        },
        "profile_dumps": {
            "present_sources": profile["present_sources"],
            "missing_sources": profile["missing_sources"],
            "full_dumps_present": profile["full_dumps_present"],
            "smoke_present": profile["smoke_present"],
        },
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "likelihood_contract": measurement_likelihood_contract(config),
        "partition": profiled_state_partition(config),
        "numerical_validation": numerical,
        "identifiability": ident,
        "seed_invariance": seed,
        "predictions": pred,
        "uncertainty_contract": unc,
        "calibration": calib,
        "focus": focus,
        "exclusion": exclusion,
        "acts_profile_materialized": profile["acts_profile_materialized"],
        "numerically_unstable": unstable,
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "prior_introduced": False,
        "ridge_added": False,
        "q_over_p_deleted": False,
        "q_over_p_fixed": False,
        "best_seed_selected": False,
        "marginalization_executed": False,
    }


def identity_key_ok(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return _identity_triple(row) in _focus_keys(config)


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    if mechanism["b15_authorized"]:
        refuse_b15()
    if mechanism["prior_introduced"]:
        refuse_prior_introduction()
    if mechanism["q_over_p_deleted"]:
        refuse_qoverp_deletion()
    if mechanism["q_over_p_fixed"]:
        refuse_qoverp_fix()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb113_decision_sha256": inherited["workbook_113"]["decision_sha256"],
        "inherited_wb112_decision_sha256": inherited["workbook_112"]["decision_sha256"],
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb113_rewritten": False,
        "target_exclusion_holds": bool((inventory.get("exclusion") or {}).get("target_exclusion_holds")),
        "acts_profile_materialized": bool(inventory.get("acts_profile_materialized")),
        "synthetic_profile_passed": bool((inventory.get("numerical_validation") or {}).get("passed")),
    }
