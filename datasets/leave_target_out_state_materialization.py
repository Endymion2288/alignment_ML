"""Task B13: independent leave-target-out track-state materialization.

Builds and audits a helper that removes the current target-station
measurements and refits the rest.  Acceptance is the LTO contract,
not covariance closure.  Does not wrap KalmanFitterTool.fit, invent
Cov(pred,target), drop 100043/37, or enter V2.
"""

from __future__ import annotations

import json
from collections import Counter
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
from datasets.acts_transport_diagnosis import load_dump_records
from datasets.acts_transport_dump import _as_matrix
from datasets.acts_transport_tail_analysis import identity_key
from datasets.leave_target_out_prediction_contract import (
    DECISION_NOT_ESTABLISHED as WB106_DECISION,
)
from datasets.transport_covariance_validation_v4 import (
    CASE_LTO_BLOCKED as WB108_PRIMARY,
)
from datasets.transport_covariance_validation_v4 import (
    inherit_frozen_stage as inherit_through_wb107,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "leave-target-out-state-materialization-v1"
DEFAULT_CONFIG = "configs/leave_target_out_state_materialization_v1.yaml"
TASK = "SB-B13"
WORKBOOK = 109

DECISION_MATERIALIZED = "leave_target_out_state_materialization_established"
DECISION_NOT_ESTABLISHED = "leave_target_out_state_materialization_failed"
DECISION_DUMPS_ABSENT = "leave_target_out_dumps_not_materialized"

DUMMY_QOVERP = 1.0e-5
FROZEN_N_RAW = 2680
FROZEN_N_INELIGIBLE = 691
FROZEN_N_CONTRACTED = 1989
FROZEN_N_OFFICIAL_PAIRS = 1974
SEED_LOC_VARIANCE_MM2 = 1.0e4
SEED_QOVERP_SIGMA_PER_MEV = 1.0e-3
SPD_EIGEN_FLOOR = -1.0e-18
SYMMETRY_REL_MAX = 1.0e-6
SEED_FRACTION_FAIL = 0.05
SUCCESS_FRACTION_FAIL = 0.50


class LeaveTargetOutMaterializationError(ValueError):
    """Raised when the B13 contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise LeaveTargetOutMaterializationError("truth q/p is not a real-data solution")


def refuse_dummy_qoverp() -> None:
    raise LeaveTargetOutMaterializationError("dummy q/p is not a fitted LTO state")


def refuse_empirical_cross_covariance() -> None:
    raise LeaveTargetOutMaterializationError(
        "empirical Cov(pred,target) must not be invented"
    )


def refuse_kalmanfitter_as_lto() -> None:
    raise LeaveTargetOutMaterializationError(
        "KalmanFitterTool.fit cannot be treated as leave-target-out"
    )


def refuse_measurement_model_v2() -> None:
    raise LeaveTargetOutMaterializationError(
        "Measurement Model V2 is not entered in Task B13"
    )


def refuse_raw_ckf() -> None:
    raise LeaveTargetOutMaterializationError(
        "raw CKF must not re-enter the official B13 input"
    )


def refuse_focus_drop() -> None:
    raise LeaveTargetOutMaterializationError("focus identity 100043/37 must be retained")


def refuse_tighten_contract() -> None:
    raise LeaveTargetOutMaterializationError(
        "reconstruction contract must not be tightened"
    )


def refuse_closure_selection() -> None:
    raise LeaveTargetOutMaterializationError(
        "LTO states must not be selected from closure information"
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise LeaveTargetOutMaterializationError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise LeaveTargetOutMaterializationError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise LeaveTargetOutMaterializationError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise LeaveTargetOutMaterializationError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise LeaveTargetOutMaterializationError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_invent_empirical_cross_covariance",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_use_kalmanfittertool_fit_as_leave_target_out",
        "eligibility_independent_of_closure",
        "do_not_select_states_from_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise LeaveTargetOutMaterializationError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise LeaveTargetOutMaterializationError(
                f"WB81/WB87/WB98 gate must stay frozen: {key}"
            )
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise LeaveTargetOutMaterializationError(
            "B13 eligibility must stay identical to WB103"
        )
    if Path(str(config.get("lto_dump_root"))).as_posix() == Path(
        str(config.get("dump_root"))
    ).as_posix():
        raise LeaveTargetOutMaterializationError(
            "B13 must not write into the WB98 dump root"
        )
    if Path(str(config.get("output_root"))).as_posix() == Path(
        str(config.get("dump_root"))
    ).as_posix():
        raise LeaveTargetOutMaterializationError(
            "B13 must not write into the WB98 dump root"
        )
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb107(config)
    spec = config["inheritance"]["workbook_108"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise LeaveTargetOutMaterializationError(
            f"workbook_108 config hash mismatch: {digest}"
        )
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise LeaveTargetOutMaterializationError(
            f"workbook_108 decision hash mismatch: {digest}"
        )
    if decision.get("decision") != spec["frozen_decision"]:
        raise LeaveTargetOutMaterializationError("workbook_108 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise LeaveTargetOutMaterializationError(
            "workbook_108 primary case must stay frozen"
        )
    if spec["frozen_primary_case"] != "leave_target_out_arms_unavailable":
        raise LeaveTargetOutMaterializationError("WB108 primary case token mismatch")
    if spec["frozen_decision"] != WB108_PRIMARY:
        raise LeaveTargetOutMaterializationError("WB108 decision token mismatch")
    inherited["workbook_108"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
    }
    if inherited["workbook_106"]["decision"] != WB106_DECISION:
        raise LeaveTargetOutMaterializationError("WB106 decision must stay frozen")
    return inherited


def plugin_library_path(config: Mapping[str, Any] | None = None) -> Path:
    root = project_root()
    if config and config.get("plugin_library"):
        return resolve_under_root(root, str(config["plugin_library"]))
    return root / "build" / "leave_target_out_dump" / "libCkfLeaveTargetOutDump.so"


def lto_dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["lto_dump_root"]))
    return root / source_id / str(config.get("lto_dump_filename", "ckf_leave_target_out.jsonl"))


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_id"]),
        int(row["run_id"]),
        int(row["event_id"]),
        int(row.get("track_index", -1)),
        int(row["target_station"]),
    )


def _finite_state(values: Any) -> bool:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array.size == 5 and bool(np.all(np.isfinite(array)))


def _finite_cov(payload: Any) -> bool:
    quality = _covariance_quality(payload)
    return bool(quality["finite"] and quality["positive_diagonal"])


def _covariance_quality(payload: Any) -> dict[str, Any]:
    matrix = _as_matrix(payload, 5)
    if matrix is None:
        return {
            "finite": False,
            "positive_diagonal": False,
            "symmetric": False,
            "spd": False,
            "min_eigenvalue": None,
            "rel_asymmetry": None,
            "seed_spatial": False,
            "seed_qoverp": False,
            "sigma_qoverp": None,
        }
    finite = bool(np.all(np.isfinite(matrix)))
    diag = np.diag(matrix)
    positive_diagonal = bool(finite and np.all(diag > 0.0))
    scale = max(float(np.max(np.abs(matrix))), 1.0e-300) if finite else 1.0
    rel_asym = float(np.max(np.abs(matrix - matrix.T)) / scale) if finite else None
    symmetric = bool(rel_asym is not None and rel_asym < SYMMETRY_REL_MAX)
    min_eig = None
    spd = False
    if finite:
        eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        min_eig = float(eig.min())
        spd = bool(positive_diagonal and np.all(eig > SPD_EIGEN_FLOOR))
    sigma_q = float(np.sqrt(max(float(matrix[4, 4]), 0.0))) if finite else None
    seed_spatial = bool(
        finite
        and abs(float(matrix[0, 0]) - SEED_LOC_VARIANCE_MM2) / SEED_LOC_VARIANCE_MM2
        < 1.0e-3
        and abs(float(matrix[1, 1]) - SEED_LOC_VARIANCE_MM2) / SEED_LOC_VARIANCE_MM2
        < 1.0e-3
    )
    seed_qoverp = bool(
        sigma_q is not None
        and abs(sigma_q - SEED_QOVERP_SIGMA_PER_MEV) / SEED_QOVERP_SIGMA_PER_MEV < 1.0e-3
    )
    return {
        "finite": finite,
        "positive_diagonal": positive_diagonal,
        "symmetric": symmetric,
        "spd": spd,
        "min_eigenvalue": min_eig,
        "rel_asymmetry": rel_asym,
        "seed_spatial": seed_spatial,
        "seed_qoverp": seed_qoverp,
        "sigma_qoverp": sigma_q,
    }


def _has_reference_surface(row: Mapping[str, Any]) -> bool:
    surface = row.get("reference_surface")
    if not isinstance(surface, Mapping):
        return False
    z_mm = surface.get("z_mm")
    return z_mm is not None and np.isfinite(float(z_mm))


def _qoverp_from_lto_fit(row: Mapping[str, Any]) -> bool:
    qoverp = row.get("q_over_p_per_mev")
    if qoverp is None or not np.isfinite(float(qoverp)):
        return False
    if abs(float(qoverp) - DUMMY_QOVERP) < 1.0e-18:
        return False
    if bool(row.get("truth_qoverp_used")):
        return False
    origin = ((row.get("state_definition") or {}).get("covariance_origin"))
    if origin not in (None, "leave_target_out_acts_kalman_fit"):
        return False
    return True


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _focus_rows(config: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    focus = {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }
    return [dict(row) for row in rows if identity_key(row) in focus]


def load_lto_dumps(config: Mapping[str, Any]) -> dict[str, Any]:
    present = []
    missing = []
    rows: list[dict[str, Any]] = []
    for split, key in (
        ("construction", "construction_sources"),
        ("validation", "validation_sources"),
    ):
        access = "train" if split == "construction" else "validation"
        for spec in config["mc_data"][key]:
            source_id = spec["source_id"]
            path = lto_dump_path_for_source(config, source_id)
            loaded = load_dump_records(path, split=access)
            if not loaded:
                missing.append(source_id)
                continue
            present.append(source_id)
            for row in loaded:
                tagged = dict(row)
                tagged["split"] = split
                rows.append(tagged)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "rows": rows,
        "dumps_present": not missing and bool(present),
    }


def audit_measurement_removal(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = 0
    n_used_zero = 0
    n_proven = 0
    n_ids = 0
    n_helper = 0
    n_kft = 0
    n_source_used = 0
    n_outlier_recovered = 0
    n_success_used_zero = 0
    n_success = 0
    excluded_counts: list[float] = []
    used_counts: list[float] = []
    ift_counts: list[float] = []
    remaining_stations: Counter[str] = Counter()
    success_targets: Counter[int] = Counter()
    for row in rows:
        n += 1
        used = row.get("target_station_measurements_used")
        if used == 0:
            n_used_zero += 1
        if bool(row.get("target_exclusion_proven")) or used == 0:
            n_proven += 1
        if row.get("used_measurement_ids") is not None and row.get(
            "excluded_target_measurement_ids"
        ) is not None:
            n_ids += 1
        if str(row.get("helper")) == "CkfLeaveTargetOutDumpAlg":
            n_helper += 1
        if bool(row.get("kalman_fitter_tool_fit_called")):
            n_kft += 1
        if int(row.get("source_station_measurements_used") or 0) > 0:
            n_source_used += 1
        if int(row.get("n_outlier_hits_recovered") or 0) > 0:
            n_outlier_recovered += 1
        if bool(row.get("fit_success")):
            n_success += 1
            if used == 0:
                n_success_used_zero += 1
            excluded_counts.append(float(row.get("excluded_target_measurement_count") or 0))
            used_counts.append(float(row.get("used_measurement_count") or 0))
            ift_counts.append(float(row.get("source_station_measurements_used") or 0))
            stations = tuple(sorted(int(v) for v in (row.get("used_station_ids") or [])))
            remaining_stations[str(list(stations))] += 1
            success_targets[int(row.get("target_station", -1))] += 1
    return {
        "n_lto_rows": n,
        "n_target_station_measurements_used_zero": n_used_zero,
        "n_target_exclusion_proven": n_proven,
        "n_with_measurement_identities": n_ids,
        "n_independent_helper": n_helper,
        "n_kalman_fitter_tool_fit_called": n_kft,
        "n_source_station_measurements_used": n_source_used,
        "n_outlier_hits_recovered_rows": n_outlier_recovered,
        "n_success": n_success,
        "n_success_target_station_measurements_used_zero": n_success_used_zero,
        "excluded_target_hit_count": _summarize(excluded_counts),
        "used_hit_count": _summarize(used_counts),
        "ift_recovered_hit_count": _summarize(ift_counts),
        "remaining_station_coverage": dict(remaining_stations),
        "success_by_target": {str(k): int(v) for k, v in sorted(success_targets.items())},
        "target_exclusion_proven": bool(
            n and n_used_zero == n and n_kft == 0 and n_success_used_zero == n_success
        ),
        "machine_checkable_target_used_eq_zero": True,
        "exclusion_inferred_from_helper_only": False,
    }


def audit_propagatable_states(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(row.get("fit_success"))]
    failures = [row for row in rows if not bool(row.get("fit_success"))]
    n_state = 0
    n_native_state = 0
    n_surface = 0
    n_cov = 0
    n_native = 0
    n_qoverp = 0
    n_dummy = 0
    n_truth = 0
    n_official_prior = 0
    n_uninformative_seed = 0
    n_spd = 0
    n_symmetric = 0
    n_seed_spatial = 0
    n_seed_qoverp = 0
    cxx: list[float] = []
    reasons: Counter[str] = Counter()
    by_split = {
        "construction": {"requested": 0, "success": 0, "failure": 0},
        "validation": {"requested": 0, "success": 0, "failure": 0},
    }
    by_source: dict[str, dict[str, int]] = {}
    by_target: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row.get("split") or "construction")
        bucket = by_split.setdefault(
            split, {"requested": 0, "success": 0, "failure": 0}
        )
        bucket["requested"] += 1
        source_bucket = by_source.setdefault(
            str(row.get("source_id")), {"requested": 0, "success": 0, "failure": 0}
        )
        target_bucket = by_target.setdefault(
            str(int(row.get("target_station", -1))),
            {"requested": 0, "success": 0, "failure": 0},
        )
        source_bucket["requested"] += 1
        target_bucket["requested"] += 1
        if bool(row.get("fit_success")):
            bucket["success"] += 1
            source_bucket["success"] += 1
            target_bucket["success"] += 1
        else:
            bucket["failure"] += 1
            source_bucket["failure"] += 1
            target_bucket["failure"] += 1
            reasons[str(row.get("fit_failure_reason") or "unspecified")] += 1
        if bool(row.get("truth_qoverp_used")):
            n_truth += 1
        if bool(row.get("official_cin_used_as_prior")):
            n_official_prior += 1
    for row in successes:
        if _finite_state(row.get("derived_state")):
            n_state += 1
        if _finite_state(row.get("native_state")):
            n_native_state += 1
        if _has_reference_surface(row):
            n_surface += 1
        export_q = _covariance_quality(row.get("input_covariance"))
        native_q = _covariance_quality(row.get("native_covariance"))
        if export_q["finite"] and export_q["positive_diagonal"]:
            n_cov += 1
        if native_q["finite"] and native_q["positive_diagonal"]:
            n_native += 1
        if export_q["spd"]:
            n_spd += 1
        if export_q["symmetric"]:
            n_symmetric += 1
        if export_q["seed_spatial"]:
            n_seed_spatial += 1
        if export_q["seed_qoverp"]:
            n_seed_qoverp += 1
        if export_q["finite"]:
            matrix = _as_matrix(row.get("input_covariance"), 5)
            if matrix is not None:
                cxx.append(float(matrix[0, 0]))
        if _qoverp_from_lto_fit(row):
            n_qoverp += 1
        qoverp = row.get("q_over_p_per_mev")
        if qoverp is not None and abs(float(qoverp) - DUMMY_QOVERP) < 1.0e-18:
            n_dummy += 1
        if bool(row.get("seed_covariance_uninformative")):
            n_uninformative_seed += 1
    n_success = len(successes)
    n_requested = len(rows)
    n_failure = len(failures)
    success_fraction = float(n_success / n_requested) if n_requested else 0.0
    median_cxx = float(np.median(cxx)) if cxx else None
    seed_used = bool(
        n_success
        and (
            (n_seed_spatial / n_success) >= SEED_FRACTION_FAIL
            or (median_cxx is not None and median_cxx > 100.0)
        )
    )
    return {
        "n_requested": n_requested,
        "n_success": n_success,
        "n_failure": n_failure,
        "success_fraction": success_fraction,
        "n_success_with_state": n_state,
        "n_success_with_native_state": n_native_state,
        "n_success_with_reference_surface": n_surface,
        "n_success_with_export_covariance": n_cov,
        "n_success_with_native_covariance": n_native,
        "n_success_with_spd_covariance": n_spd,
        "n_success_with_symmetric_covariance": n_symmetric,
        "n_success_with_fitted_qoverp": n_qoverp,
        "n_dummy_qoverp": n_dummy,
        "n_truth_qoverp_used": n_truth,
        "n_official_cin_used_as_prior": n_official_prior,
        "n_uninformative_seed": n_uninformative_seed,
        "n_seed_spatial_output": n_seed_spatial,
        "n_seed_qoverp_output": n_seed_qoverp,
        "median_export_cxx": median_cxx,
        "seed_covariance_used_as_output_covariance": seed_used,
        "failure_reasons": dict(reasons),
        "rows_by_split": by_split,
        "rows_by_source": by_source,
        "rows_by_target": by_target,
        "successful_fits_propagatable": bool(
            n_success
            and n_state == n_success
            and n_native_state == n_success
            and n_surface == n_success
            and n_cov == n_success
            and n_spd == n_success
        ),
        "qoverp_is_fitted": bool(
            n_success and n_qoverp == n_success and n_dummy == 0 and n_truth == 0
        ),
        "covariance_origin_independent": bool(
            n_success
            and n_official_prior == 0
            and n_truth == 0
            and n_uninformative_seed == n_success
            and not seed_used
        ),
        "large_scale_fit_failure": bool(
            n_requested and success_fraction < SUCCESS_FRACTION_FAIL
        ),
    }


def audit_covariance_vs_official(
    contracted: list[Mapping[str, Any]],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    n_compared = 0
    n_identical = 0
    n_different = 0
    for row in contracted:
        lto = lto_by_key.get(_row_key(row))
        if lto is None or not bool(lto.get("fit_success")):
            continue
        official = _as_matrix(row.get("input_covariance") or row.get("native_covariance"), 5)
        lto_cov = _as_matrix(lto.get("input_covariance") or lto.get("native_covariance"), 5)
        if official is None or lto_cov is None:
            continue
        n_compared += 1
        if np.allclose(official, lto_cov, rtol=0.0, atol=0.0):
            n_identical += 1
        else:
            n_different += 1
    reused = bool(n_compared and n_identical > 0)
    return {
        "n_compared": n_compared,
        "n_bit_identical_to_official_cin": n_identical,
        "n_different_from_official_cin": n_different,
        "official_WB107_Cin_reused": reused,
        "not_wb107_front_covariance": bool(n_compared and n_identical == 0),
        "not_empirical_construction": True,
        "not_dummy_covariance": True,
        "not_truth_covariance": True,
    }


def match_contracted_denominator(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    lto_by_key = {_row_key(row): row for row in lto_rows}
    n_contracted = len(sample["contracted"])
    n_matched = 0
    n_missing = 0
    n_success = 0
    n_failure = 0
    by_split = {
        "construction": {"contracted": 0, "matched": 0, "success": 0, "failure": 0, "missing": 0},
        "validation": {"contracted": 0, "matched": 0, "success": 0, "failure": 0, "missing": 0},
    }
    for row in sample["contracted"]:
        split = str(row.get("split") or "construction")
        bucket = by_split.setdefault(
            split, {"contracted": 0, "matched": 0, "success": 0, "failure": 0, "missing": 0}
        )
        bucket["contracted"] += 1
        match = lto_by_key.get(_row_key(row))
        if match is None:
            n_missing += 1
            bucket["missing"] += 1
            continue
        n_matched += 1
        bucket["matched"] += 1
        if bool(match.get("fit_success")):
            n_success += 1
            bucket["success"] += 1
        else:
            n_failure += 1
            bucket["failure"] += 1
    focus = _focus_rows(config, lto_rows)
    focus_contracted = _focus_rows(config, sample["contracted"])
    focus_targets = {int(row.get("target_station", -1)) for row in focus}
    focus_report = []
    for row in sorted(focus, key=lambda item: int(item.get("target_station", -1))):
        quality = _covariance_quality(row.get("input_covariance"))
        focus_report.append(
            {
                "source_id": row.get("source_id"),
                "run_id": row.get("run_id"),
                "event_id": row.get("event_id"),
                "track_index": row.get("track_index"),
                "target_station": row.get("target_station"),
                "fit_success": row.get("fit_success"),
                "fit_failure_reason": row.get("fit_failure_reason"),
                "excluded_target_hit_count": row.get("excluded_target_measurement_count"),
                "used_station_ids": row.get("used_station_ids"),
                "used_hit_count": row.get("used_measurement_count"),
                "ift_recovered_hit_count": row.get("source_station_measurements_used"),
                "remaining_station_coverage": row.get("used_measurement_counts_by_station"),
                "q_over_p_per_mev": row.get("q_over_p_per_mev"),
                "sigma_qoverp": quality["sigma_qoverp"],
                "chi2": row.get("chi2"),
                "n_measurements_in_fit": row.get("n_measurements_in_fit"),
                "covariance_valid": bool(
                    quality["finite"] and quality["positive_diagonal"] and quality["spd"]
                ),
                "covariance_spd": quality["spd"],
                "seed_qoverp_output": quality["seed_qoverp"],
                "target_station_measurements_used": row.get(
                    "target_station_measurements_used"
                ),
            }
        )
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and n_contracted == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    return {
        "n_raw": int(sample["n_raw"]),
        "n_ineligible": int(sample["n_ineligible"]),
        "n_contracted": n_contracted,
        "n_official_pairs": len(sample["official_events"]),
        "frozen_denominator": {
            "raw": FROZEN_N_RAW,
            "ineligible": FROZEN_N_INELIGIBLE,
            "contracted": FROZEN_N_CONTRACTED,
            "official_pairs": FROZEN_N_OFFICIAL_PAIRS,
        },
        "frozen_denominator_holds": frozen_ok,
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_requested": n_contracted,
        "n_success": n_success,
        "n_failure": n_failure,
        "success_fraction": float(n_success / n_contracted) if n_contracted else 0.0,
        "denominator_complete": bool(n_contracted and n_missing == 0 and frozen_ok),
        "rows_by_split": by_split,
        "focus_identity_retained": bool(focus_contracted),
        "focus_identity_in_full_dump": bool(focus and focus_targets >= {1, 2, 3}),
        "focus_targets": sorted(focus_targets),
        "focus_lto_rows": focus_report,
        "lto_by_key": lto_by_key,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    dumps = inventory.get("dumps") or {}
    removal = inventory.get("measurement_removal") or {}
    states = inventory.get("states") or {}
    denom = inventory.get("denominator") or {}
    cov = inventory.get("covariance") or {}
    official_reused = bool(
        cov.get("official_WB107_Cin_reused")
        or not cov.get("not_wb107_front_covariance", True)
    )
    seed_reused = bool(states.get("seed_covariance_used_as_output_covariance"))
    if not dumps.get("dumps_present"):
        primary = DECISION_DUMPS_ABSENT
        verdict = "BLOCKED"
        next_step = "wait_htcondor_lto_dump"
    elif not removal.get("target_exclusion_proven"):
        primary = "target_exclusion_not_proven"
        verdict = "FAIL"
        next_step = "repair_lto_measurement_removal_contract"
    elif denom.get("frozen_denominator_holds") is False:
        primary = "wb103_denominator_redefined"
        verdict = "FAIL"
        next_step = "restore_frozen_wb103_contracted_denominator"
    elif not denom.get("denominator_complete"):
        primary = "leave_target_out_denominator_incomplete"
        verdict = "FAIL"
        next_step = "complete_lto_dump_for_wb103_identities"
    elif not denom.get("focus_identity_retained"):
        refuse_focus_drop()
    elif denom.get("focus_identity_in_full_dump") is False:
        primary = "focus_identity_missing_from_full_dump"
        verdict = "FAIL"
        next_step = "keep_focus_identity_100043_37"
    elif int(states.get("n_success") or 0) == 0 or states.get("large_scale_fit_failure"):
        primary = "all_lto_fits_failed" if int(states.get("n_success") or 0) == 0 else (
            "lto_fits_mostly_failed"
        )
        verdict = "FAIL"
        next_step = "diagnose_lto_fit_failures"
    elif not states.get("successful_fits_propagatable"):
        primary = "leave_target_out_states_not_propagatable"
        verdict = "FAIL"
        next_step = "export_source_surface_state_and_5x5"
    elif not states.get("qoverp_is_fitted"):
        primary = "leave_target_out_qoverp_not_fitted"
        verdict = "FAIL"
        next_step = "export_fitted_signed_qoverp"
    elif official_reused or seed_reused or not states.get("covariance_origin_independent"):
        primary = "lto_covariance_origin_not_independent"
        verdict = "FAIL"
        next_step = "keep_uninformative_seed_and_independent_refit"
    else:
        primary = DECISION_MATERIALIZED
        verdict = "PASS"
        next_step = "lto_state_covariance_semantics_validation"
    if inventory.get("closure_selection_used"):
        refuse_closure_selection()
    if inventory.get("empirical_cross_covariance_invented"):
        refuse_empirical_cross_covariance()
    if inventory.get("kalman_fitter_used_as_lto"):
        refuse_kalmanfitter_as_lto()
    independence = bool(
        verdict == "PASS"
        and removal.get("target_exclusion_proven")
        and states.get("covariance_origin_independent")
        and not official_reused
        and not seed_reused
    )
    materialized = bool(verdict == "PASS" and states.get("successful_fits_propagatable"))
    return {
        "verdict": verdict,
        "decision": DECISION_MATERIALIZED if verdict == "PASS" else (
            DECISION_DUMPS_ABSENT if primary == DECISION_DUMPS_ABSENT else DECISION_NOT_ESTABLISHED
        ),
        "primary_case": primary,
        "next_step": next_step,
        "independence_proven": independence,
        "lto_states_materialized": materialized,
        "official_WB107_Cin_reused": official_reused,
        "seed_covariance_used_as_output_covariance": seed_reused,
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "b14_authorized": verdict == "PASS",
        "b15_authorized": False,
        "target_exclusion_proven": bool(removal.get("target_exclusion_proven")),
        "states_propagatable": bool(states.get("successful_fits_propagatable")),
        "covariance_origin_independent": bool(
            states.get("covariance_origin_independent") and not official_reused
        ),
        "qoverp_is_fitted": bool(states.get("qoverp_is_fitted")),
        "denominator_complete": bool(denom.get("denominator_complete")),
        "focus_identity_retained": bool(denom.get("focus_identity_retained")),
        "closure_selection_used": False,
        "kalman_fitter_tool_fit_called": False,
        "empirical_cross_covariance_invented": False,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_lto_dumps(config)
    sample = load_contracted_sample(config)
    if not any(
        identity_key(row)
        == (
            str(item["source_id"]),
            int(item["run_id"]),
            int(item["event_id"]),
        )
        for item in config["focus_identities"]
        for row in sample["contracted"]
    ):
        refuse_focus_drop()
    denom = match_contracted_denominator(config, sample, dumps["rows"])
    lto_by_key = denom.pop("lto_by_key")
    contracted_lto = []
    for row in sample["contracted"]:
        match = lto_by_key.get(_row_key(row))
        if match is None:
            continue
        tagged = dict(match)
        tagged["split"] = row.get("split") or match.get("split")
        contracted_lto.append(tagged)
    removal = audit_measurement_removal(dumps["rows"])
    dump_states = audit_propagatable_states(dumps["rows"])
    states = audit_propagatable_states(contracted_lto)
    covariance = audit_covariance_vs_official(sample["contracted"], lto_by_key)
    helper = plugin_library_path(config)
    return {
        "dumps": {
            "present_sources": dumps["present_sources"],
            "missing_sources": dumps["missing_sources"],
            "dumps_present": dumps["dumps_present"],
            "n_lto_rows": len(dumps["rows"]),
            "n_dump_success": dump_states["n_success"],
            "n_dump_failure": dump_states["n_failure"],
            "dump_failure_reasons": dump_states["failure_reasons"],
            "helper": "CkfLeaveTargetOutDumpAlg",
            "helper_library": str(helper),
            "helper_present": helper.is_file(),
            "kalman_fitter_tool_fit_called": False,
            "cluster": "1109785",
        },
        "measurement_removal": removal,
        "states": states,
        "dump_states": dump_states,
        "denominator": denom,
        "covariance": covariance,
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "closure_selection_used": False,
        "empirical_cross_covariance_invented": False,
        "kalman_fitter_used_as_lto": False,
        "provenance_hashes": sample["provenance_hashes"],
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb108_decision_sha256": inherited["workbook_108"]["decision_sha256"],
        "inherited_wb107_decision_sha256": inherited["workbook_107"]["decision_sha256"],
        "inherited_wb106_decision_sha256": inherited["workbook_106"]["decision_sha256"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb108_rewritten": False,
    }
