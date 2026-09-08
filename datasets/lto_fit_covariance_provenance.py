"""Task B14R: LTO fit-covariance provenance / information audit.

Asks what the helper 5×5 actually is.  Does not rescale Cin, choose a
seed scale from closure, drop 100043/37, enter B15, or enter V2.
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
from datasets.acts_transport_diagnosis import _stats, load_dump_records
from datasets.acts_transport_dump import _as_matrix
from datasets.acts_transport_tail_analysis import identity_key
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
    _row_key,
    load_lto_dumps,
)
from datasets.lto_state_covariance_semantics import (
    DECISION_NOT_VALIDATED as WB110_DECISION,
    _lto_residual,
    inherit_frozen_stage as inherit_through_wb109,
)
from datasets.transport_uncertainty_shape_diagnosis import (
    STATE_NAMES,
    load_contracted_sample,
)

SCHEMA_VERSION = "lto-fit-covariance-provenance-v1"
DEFAULT_CONFIG = "configs/lto_fit_covariance_provenance_v1.yaml"
TASK = "SB-B14R"
WORKBOOK = 111

CASE_A = "lto_exported_covariance_is_seed_or_predicted_state"
CASE_B = "lto_fit_information_insufficient"
CASE_C = "lto_fitter_covariance_semantics_mismatch"
CASE_D = "tail_dominated"
CASE_E = "mixed_or_inconclusive"
CASE_BLOCKED = "lto_provenance_smoke_incomplete"

PARAM_NAMES = ("x", "y", "tx", "ty", "q_over_p")
NATIVE_NAMES = ("loc0", "loc1", "phi", "theta", "q_over_p")
SEED_QOVERP_SIGMA = 1.0e-3
ACTS_MEV = 1.0e-3


class LtoProvenanceError(ValueError):
    """Raised when the B14R contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise LtoProvenanceError("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise LtoProvenanceError("covariance rescale is forbidden")


def refuse_seed_scale_choice() -> None:
    raise LtoProvenanceError("seed scale must not be chosen from truth or chi2")


def refuse_focus_drop() -> None:
    raise LtoProvenanceError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise LtoProvenanceError("Measurement Model V2 is not entered in Task B14R")


def refuse_b15() -> None:
    raise LtoProvenanceError("Task B15 is not entered in Task B14R")


def refuse_raw_ckf() -> None:
    raise LtoProvenanceError("raw CKF must not re-enter the official B14R input")


def refuse_tighten_contract() -> None:
    raise LtoProvenanceError("reconstruction contract must not be tightened")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise LtoProvenanceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise LtoProvenanceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise LtoProvenanceError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise LtoProvenanceError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise LtoProvenanceError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_invent_empirical_cross_covariance",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15",
        "do_not_choose_seed_scale_from_closure",
        "eligibility_independent_of_closure",
        "do_not_select_states_from_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise LtoProvenanceError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise LtoProvenanceError(f"frozen gate changed: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise LtoProvenanceError("B14R eligibility must stay identical to WB103")
    scales = [float(v) for v in config["seed_sensitivity"]["scales"]]
    if scales != [0.1, 1.0, 10.0]:
        raise LtoProvenanceError("B14R seed scales must stay 0.1 / 1 / 10")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb109(config)
    spec = config["inheritance"]["workbook_110"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise LtoProvenanceError(f"workbook_110 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise LtoProvenanceError(f"workbook_110 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise LtoProvenanceError("workbook_110 decision must stay frozen")
    if spec["frozen_decision"] != WB110_DECISION:
        raise LtoProvenanceError("WB110 decision token mismatch")
    if decision.get("b15_authorized"):
        raise LtoProvenanceError("WB110 must not have authorized B15")
    inherited["workbook_110"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "b15_authorized": False,
    }
    return inherited


def _focus_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }


def _sigma_q(matrix: np.ndarray | None) -> float | None:
    if matrix is None or matrix.shape[0] < 5:
        return None
    value = float(matrix[4, 4])
    if not np.isfinite(value) or value < 0.0:
        return None
    return float(np.sqrt(value))


def _bound_gev_to_mev(matrix: np.ndarray | None) -> np.ndarray | None:
    """Convert an ACTS bound 5×5 from GeV to Athena MeV, matching the helper."""
    if matrix is None or matrix.shape != (5, 5):
        return None
    converted = np.array(matrix, dtype=np.float64, copy=True)
    converted[:, 4] *= ACTS_MEV
    converted[4, :] *= ACTS_MEV
    return converted


def _eigen_spectrum(matrix: np.ndarray | None) -> list[float] | None:
    if matrix is None or matrix.shape[0] < 2:
        return None
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if not np.all(np.isfinite(values)):
        return None
    return [float(value) for value in values]


def _relative_change(current: float, reference: float) -> float | None:
    denom = max(abs(reference), 1.0e-12)
    if not np.isfinite(current) or not np.isfinite(reference):
        return None
    return abs(float(current) - float(reference)) / denom


def _matrices_close(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    rtol: float = 1.0e-3,
) -> bool:
    if left is None or right is None:
        return False
    if left.shape != right.shape:
        return False
    return bool(np.allclose(left, right, rtol=rtol, atol=0.0, equal_nan=False))


def _inherited_native_blocks(
    exported: np.ndarray | None,
    seed_mev: np.ndarray | None,
    threshold: float,
) -> list[str]:
    if exported is None or seed_mev is None:
        return []
    inherited = []
    for index, name in enumerate(NATIVE_NAMES):
        ref = float(np.sqrt(max(seed_mev[index, index], 0.0)))
        cur = float(np.sqrt(max(exported[index, index], 0.0)))
        if ref <= 0.0:
            continue
        if abs(cur - ref) / ref <= threshold:
            inherited.append(name)
    return inherited


def _percentiles(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0}
    abs_vals = np.abs(finite)
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "median_abs": float(np.median(abs_vals)),
        "p68": float(np.quantile(abs_vals, 0.68)),
        "p95": float(np.quantile(abs_vals, 0.95)),
        "p99": float(np.quantile(abs_vals, 0.99)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _top_share(values: list[float], fraction: float = 0.01) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    chi2 = finite * finite
    total = float(np.sum(chi2))
    if total <= 0.0:
        return None
    n_top = max(1, int(np.ceil(finite.size * fraction)))
    return float(np.sum(np.sort(chi2)[-n_top:]) / total)


def _station_z_span(row: Mapping[str, Any], config: Mapping[str, Any]) -> float | None:
    if row.get("measurement_z_span_mm") is not None:
        return float(row["measurement_z_span_mm"])
    stations = [int(v) for v in (row.get("used_station_ids") or [])]
    zs = []
    raw = config.get("station_z_mm") or {}
    for station in stations:
        value = raw.get(station, raw.get(str(station)))
        if value is not None:
            zs.append(float(value))
    if len(zs) < 2:
        return None
    return float(max(zs) - min(zs))


def audit_fit_information(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = []
    n_low = 0
    n_seedlike_q = 0
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        n_fit = lto.get("n_measurements_in_fit")
        used = lto.get("n_used_measurements", lto.get("used_measurement_count"))
        item = {
            "source_id": official.get("source_id"),
            "split": official.get("split"),
            "target_station": official.get("target_station"),
            "fit_success": lto.get("fit_success"),
            "n_input_measurements": lto.get("n_input_measurements"),
            "n_used_measurements": used,
            "n_measurements_in_fit": n_fit,
            "n_fit_states": lto.get("n_fit_states"),
            "ndof": lto.get("ndof"),
            "fit_chi2": lto.get("chi2"),
            "number_of_stations_used": (
                lto.get("number_of_stations_used")
                if lto.get("number_of_stations_used") is not None
                else len(lto.get("used_station_ids") or [])
            ),
            "measurement_z_span": _station_z_span(lto, config),
            "number_of_filtered_states": lto.get("number_of_filtered_states"),
            "number_of_smoothed_states": lto.get("number_of_smoothed_states"),
            "number_of_outlier_states": lto.get("number_of_outlier_states"),
        }
        rows.append(item)
        by_split[str(official.get("split"))].append(item)
        by_target[str(int(official.get("target_station", -1)))].append(item)
        if n_fit is not None and int(n_fit) <= 2:
            n_low += 1
        cov = _as_matrix(lto.get("input_covariance"), 5)
        sigma = _sigma_q(cov)
        if sigma is not None and abs(sigma - SEED_QOVERP_SIGMA) / SEED_QOVERP_SIGMA < 1.0e-3:
            n_seedlike_q += 1

    def _pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "n_success": sum(1 for item in items if item.get("fit_success")),
            "n_fit_le_2": sum(
                1
                for item in items
                if item.get("n_measurements_in_fit") is not None
                and int(item["n_measurements_in_fit"]) <= 2
            ),
            "n_input_measurements": _stats(
                [item["n_input_measurements"] for item in items if item.get("n_input_measurements") is not None]
            ),
            "n_used_measurements": _stats(
                [item["n_used_measurements"] for item in items if item.get("n_used_measurements") is not None]
            ),
            "n_measurements_in_fit": _stats(
                [item["n_measurements_in_fit"] for item in items if item.get("n_measurements_in_fit") is not None]
            ),
            "n_fit_states": _stats(
                [item["n_fit_states"] for item in items if item.get("n_fit_states") is not None]
            ),
            "ndof": _stats([item["ndof"] for item in items if item.get("ndof") is not None]),
            "fit_chi2": _stats(
                [item["fit_chi2"] for item in items if item.get("fit_chi2") is not None]
            ),
            "number_of_stations_used": _stats(
                [
                    item["number_of_stations_used"]
                    for item in items
                    if item.get("number_of_stations_used") is not None
                ]
            ),
            "measurement_z_span": _stats(
                [item["measurement_z_span"] for item in items if item.get("measurement_z_span") is not None]
            ),
            "n_measurements_in_fit_unavailable": sum(
                1 for item in items if item.get("n_measurements_in_fit") is None
            ),
            "n_fit_states_unavailable": sum(
                1 for item in items if item.get("n_fit_states") is None
            ),
            "filtered_smoothed_outlier_unavailable": sum(
                1
                for item in items
                if item.get("number_of_filtered_states") is None
            ),
        }

    return {
        "n_measurements_in_fit_definition": config.get(
            "n_measurements_in_fit_definition"
        ),
        "n_measurements_in_fit_is_not_input_hit_count": True,
        "n_contracted_matched": len(rows),
        "n_fit_le_2": n_low,
        "n_seedlike_qoverp_sigma": n_seedlike_q,
        "all": _pack(rows),
        "construction": _pack(by_split["construction"]),
        "validation": _pack(by_split["validation"]),
        "by_target": {key: _pack(items) for key, items in sorted(by_target.items())},
        "wb109_filtered_smoothed_outlier_fields": "unavailable_on_frozen_dumps",
    }


def _scale_label(scale: float) -> str:
    if abs(scale - 0.1) < 1.0e-12:
        return "scale_0p1"
    if abs(scale - 1.0) < 1.0e-12:
        return "scale_1p0"
    if abs(scale - 10.0) < 1.0e-12:
        return "scale_10p0"
    return f"scale_{scale}"


def load_seed_sensitivity_dumps(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["seed_sensitivity"]
    root = resolve_under_root(project_root(), spec["dump_root"])
    source_id = spec["source_id"]
    loaded = {}
    missing = []
    for scale in spec["scales"]:
        path = root / _scale_label(float(scale)) / source_id / "ckf_leave_target_out.jsonl"
        rows = load_dump_records(path, split="train") if path.is_file() else []
        if not rows:
            missing.append(_scale_label(float(scale)))
        loaded[float(scale)] = rows
    return {"rows_by_scale": loaded, "missing": missing, "present": not missing}


def audit_seed_sensitivity(config: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict[str, Any]:
    if not loaded.get("present"):
        return {
            "present": False,
            "missing": loaded.get("missing"),
            "scale_selected_from_closure": False,
        }
    wanted = {
        (str(config["seed_sensitivity"]["source_id"]), int(event))
        for event in config["seed_sensitivity"]["event_ids"]
    }
    by_key: dict[tuple[Any, ...], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for scale, rows in loaded["rows_by_scale"].items():
        for row in rows:
            identity = (str(row.get("source_id")), int(row.get("event_id", -1)))
            if identity not in wanted:
                continue
            key = (
                str(row.get("source_id")),
                int(row.get("event_id", -1)),
                int(row.get("track_index", -1)),
                int(row.get("target_station", -1)),
            )
            by_key[key][float(scale)] = row
    comparisons = []
    n_q_sensitive = 0
    n_spatial_sensitive = 0
    n_compared = 0
    threshold = float(config["provenance_thresholds"]["seed_relative_change_independent_max"])
    for key, by_scale in sorted(by_key.items()):
        if 1.0 not in by_scale or 0.1 not in by_scale or 10.0 not in by_scale:
            continue
        n_compared += 1
        ref = by_scale[1.0]
        ref_state = np.asarray(ref.get("derived_state"), dtype=np.float64).reshape(-1)
        ref_cov = _as_matrix(ref.get("input_covariance"), 5)
        ref_eig = _eigen_spectrum(ref_cov)
        item = {
            "source_id": key[0],
            "event_id": key[1],
            "target_station": key[3],
            "n_fit_1x": ref.get("n_measurements_in_fit"),
            "q_over_p_1x": ref.get("q_over_p_per_mev"),
            "sigma_qoverp_1x": _sigma_q(ref_cov),
            "eigen_spectrum_1x": ref_eig,
            "by_scale": {},
        }
        sensitive = {name: False for name in PARAM_NAMES}
        for scale, row in sorted(by_scale.items()):
            state = np.asarray(row.get("derived_state"), dtype=np.float64).reshape(-1)
            cov = _as_matrix(row.get("input_covariance"), 5)
            sigmas = {}
            mean_rel = {}
            sigma_rel = {}
            if state.size == 5 and ref_state.size == 5:
                for index, name in enumerate(PARAM_NAMES):
                    mean_rel[name] = _relative_change(
                        float(state[index]), float(ref_state[index])
                    )
            if cov is not None and ref_cov is not None:
                for index, name in enumerate(PARAM_NAMES):
                    ref_s = float(np.sqrt(max(ref_cov[index, index], 0.0)))
                    cur_s = float(np.sqrt(max(cov[index, index], 0.0)))
                    sigmas[name] = cur_s
                    rel = _relative_change(cur_s, ref_s)
                    sigma_rel[name] = rel
                    if rel is not None and rel > threshold:
                        sensitive[name] = True
            item["by_scale"][str(scale)] = {
                "q_over_p": row.get("q_over_p_per_mev"),
                "derived_state": state.tolist() if state.size else None,
                "sigma": sigmas,
                "mean_relative_change_vs_1x": mean_rel,
                "sigma_relative_change_vs_1x": sigma_rel,
                "eigen_spectrum": _eigen_spectrum(cov),
                "n_measurements_in_fit": row.get("n_measurements_in_fit"),
                "chi2": row.get("chi2"),
            }
        item["seed_sensitive_parameters"] = [
            name for name, flag in sensitive.items() if flag
        ]
        if sensitive["q_over_p"]:
            n_q_sensitive += 1
        if any(sensitive[name] for name in ("x", "y", "tx", "ty")):
            n_spatial_sensitive += 1
        comparisons.append(item)
    typical = [
        item
        for item in comparisons
        if int(item["event_id"]) != 37
    ]
    return {
        "present": True,
        "n_compared": n_compared,
        "n_qoverp_seed_sensitive": n_q_sensitive,
        "n_spatial_seed_sensitive": n_spatial_sensitive,
        "typical_qoverp_seed_sensitive": bool(
            typical
            and sum("q_over_p" in item["seed_sensitive_parameters"] for item in typical)
            >= max(1, len(typical) // 2)
        ),
        "typical_spatial_seed_sensitive": bool(
            typical
            and sum(
                any(name in item["seed_sensitive_parameters"] for name in ("x", "y", "tx", "ty"))
                for item in typical
            )
            >= max(1, len(typical) // 2)
        ),
        "comparisons": comparisons,
        "scale_selected_from_closure": False,
        "production_scale_unchanged": 1.0,
    }


def _optional_bound(payload: Any) -> np.ndarray | None:
    if payload in (None, "unavailable"):
        return None
    return _as_matrix(payload, 5)


def _row_state_matches(row: Mapping[str, Any]) -> dict[str, Any]:
    exported = _as_matrix(row.get("native_covariance") or row.get("exported_covariance"), 5)
    seed_mev = _bound_gev_to_mev(_optional_bound(row.get("seed_covariance")))
    predicted = _bound_gev_to_mev(_optional_bound(row.get("first_predicted_covariance")))
    first_filtered = _bound_gev_to_mev(_optional_bound(row.get("first_filtered_covariance")))
    last_filtered = _bound_gev_to_mev(_optional_bound(row.get("last_filtered_covariance")))
    first_smoothed = _bound_gev_to_mev(_optional_bound(row.get("first_smoothed_covariance")))
    last_smoothed = _bound_gev_to_mev(_optional_bound(row.get("last_smoothed_covariance")))
    threshold = 0.25
    return {
        "event_id": row.get("event_id"),
        "target_station": row.get("target_station"),
        "exported_state_type": row.get("exported_state_type"),
        "reference_surface": row.get("reference_surface"),
        "seed_covariance_scale": row.get("seed_covariance_scale"),
        "n_measurements_in_fit": row.get("n_measurements_in_fit"),
        "number_of_filtered_states": row.get("number_of_filtered_states"),
        "number_of_smoothed_states": row.get("number_of_smoothed_states"),
        "number_of_outlier_states": row.get("number_of_outlier_states"),
        "matches_seed": _matrices_close(exported, seed_mev),
        "matches_first_predicted": _matrices_close(exported, predicted),
        "matches_first_filtered": _matrices_close(exported, first_filtered),
        "matches_last_filtered": _matrices_close(exported, last_filtered),
        "matches_first_smoothed": _matrices_close(exported, first_smoothed),
        "matches_last_smoothed": _matrices_close(exported, last_smoothed),
        "native_blocks_near_seed": _inherited_native_blocks(
            exported, seed_mev, threshold
        ),
        "sigma_qoverp_exported": _sigma_q(exported),
        "sigma_qoverp_seed_mev": _sigma_q(seed_mev),
        "unavailable_fields": [
            name
            for name, value in (
                ("seed_covariance", row.get("seed_covariance")),
                ("first_predicted_covariance", row.get("first_predicted_covariance")),
                ("first_filtered_covariance", row.get("first_filtered_covariance")),
                ("last_filtered_covariance", row.get("last_filtered_covariance")),
                ("first_smoothed_covariance", row.get("first_smoothed_covariance")),
                ("last_smoothed_covariance", row.get("last_smoothed_covariance")),
            )
            if value in (None, "unavailable")
        ],
    }


def audit_covariance_provenance(loaded: Mapping[str, Any]) -> dict[str, Any]:
    code_path = {
        "acts_n_measurements": (
            "Acts::calculateTrackQuantities counts TrackStateFlag::MeasurementFlag"
        ),
        "acts_n_dof": "sum of calibratedSize() over MeasurementFlag states",
        "exported_parameters": (
            "KalmanFitter writes fittedParameters at the reference surface "
            "from the smoothed first measurement after transport"
        ),
        "reference_surface_strategy": "first",
        "reference_surface": "source_station_plane",
        "covariance_transported_to_reference": True,
        "back_propagation_used": "unavailable",
        "not_kalman_fitter_tool_fit": True,
        "comparison_frame": "native_bound_mev_after_official_qoverp_conversion",
    }
    rows = (loaded.get("rows_by_scale") or {}).get(1.0) or []
    if not rows:
        return {
            "present": False,
            "exported_state_type": "unavailable",
            "seed_covariance": "unavailable",
            "first_predicted_covariance": "unavailable",
            "first_filtered_covariance": "unavailable",
            "final_filtered_covariance": "unavailable",
            "smoothed_covariance": "unavailable",
            "exported_covariance": "unavailable",
            "code_path": code_path,
        }
    summaries = []
    types = set()
    n_seed_like = 0
    n_pred_like = 0
    n_success = 0
    for row in rows:
        if not row.get("fit_success"):
            continue
        n_success += 1
        if row.get("exported_state_type") is not None:
            types.add(str(row.get("exported_state_type")))
        summary = _row_state_matches(row)
        summaries.append(summary)
        if summary["matches_seed"]:
            n_seed_like += 1
        if summary["matches_first_predicted"]:
            n_pred_like += 1
    return {
        "present": True,
        "exported_state_type": sorted(types)[0] if len(types) == 1 else sorted(types),
        "n_success_smoke": n_success,
        "n_exported_bitclose_to_seed": n_seed_like,
        "n_exported_bitclose_to_first_predicted": n_pred_like,
        "is_seed_or_predicted": bool(
            n_success and (n_seed_like == n_success or n_pred_like == n_success)
        ),
        "reference_surface": "source_station_plane",
        "covariance_transported_to_reference": True,
        "back_propagation_used": "unavailable",
        "seed_covariance": "present_on_smoke_rows",
        "first_predicted_covariance": "present_on_smoke_rows",
        "first_filtered_covariance": "present_on_smoke_rows",
        "final_filtered_covariance": "present_on_smoke_rows_as_last_filtered",
        "smoothed_covariance": "present_on_smoke_rows",
        "exported_covariance": "native_bound_mev_and_derived_cartesian_mev",
        "rows": summaries,
        "code_path": code_path,
    }


def audit_parameter_information(
    info: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> dict[str, Any]:
    all_info = info.get("all") or {}
    n_fit = (all_info.get("n_measurements_in_fit") or {}).get("median")
    n_used = (all_info.get("n_used_measurements") or {}).get("median")
    ratio = None
    if n_fit is not None and n_used not in (None, 0):
        ratio = float(n_fit) / float(n_used)
    by_target = {}
    for target, payload in (info.get("by_target") or {}).items():
        fit = (payload.get("n_measurements_in_fit") or {}).get("median")
        used = (payload.get("n_used_measurements") or {}).get("median")
        span = (payload.get("measurement_z_span") or {}).get("median")
        by_target[target] = {
            "median_n_fit": fit,
            "median_n_used": used,
            "median_z_span": span,
            "n_fit_le_2": payload.get("n_fit_le_2"),
        }
    smoke_by_target: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "qoverp_sensitive": 0, "spatial_sensitive": 0}
    )
    for item in seed.get("comparisons") or []:
        target = str(int(item.get("target_station", -1)))
        smoke_by_target[target]["n"] += 1
        if "q_over_p" in item.get("seed_sensitive_parameters", []):
            smoke_by_target[target]["qoverp_sensitive"] += 1
        if any(
            name in item.get("seed_sensitive_parameters", [])
            for name in ("x", "y", "tx", "ty")
        ):
            smoke_by_target[target]["spatial_sensitive"] += 1
    return {
        "information_matrix": "unavailable",
        "singular_values": "unavailable",
        "numerical_condition_number": "unavailable",
        "weak_eigenvectors": "unavailable",
        "not_alignment_rank_gate": True,
        "median_n_fit_over_used": ratio,
        "n_fit_le_2": info.get("n_fit_le_2"),
        "typical_qoverp_seed_sensitive": seed.get("typical_qoverp_seed_sensitive"),
        "typical_spatial_seed_sensitive": seed.get("typical_spatial_seed_sensitive"),
        "by_target": by_target,
        "seed_sensitivity_by_target": dict(smoke_by_target),
        "diagnostic": (
            "ACTS normal/information matrix is unavailable. Observability uses "
            "n_fit vs n_used and pre-registered seed-scale finite differences. "
            "This is not the historical alignment rank_tolerance=0.01 gate."
        ),
    }


def audit_pull_correlation(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    pulls: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    q_pulls: list[float] = []
    by_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: {name: [] for name in STATE_NAMES})
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: {name: [] for name in STATE_NAMES})
    pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in (*STATE_NAMES, "q_over_p")}
    ndof_pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in (*STATE_NAMES, "q_over_p")}
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None or not lto.get("fit_success"):
            continue
        matched = _lto_residual(config, sample, official, lto)
        if matched is None:
            continue
        residual, cin, extra = matched
        n_fit = lto.get("n_measurements_in_fit")
        ndof = lto.get("ndof")
        for index, name in enumerate(STATE_NAMES):
            if index >= residual.size:
                continue
            sigma = float(np.sqrt(max(float(cin[index, index]), 0.0)))
            if sigma <= 0.0:
                continue
            pull = float(residual[index] / sigma)
            pulls[name].append(pull)
            by_target[str(int(official["target_station"]))][name].append(pull)
            by_source[str(official["source_id"])][name].append(pull)
            if n_fit is not None:
                pairs[name].append((float(n_fit), pull))
            if ndof is not None:
                ndof_pairs[name].append((float(ndof), pull))
        q_fit = lto.get("q_over_p_per_mev")
        q_truth = extra.get("q_over_p_truth")
        sigma_q = _sigma_q(cin)
        if q_fit is not None and q_truth is not None and sigma_q not in (None, 0.0):
            q_pull = float((float(q_fit) - float(q_truth)) / sigma_q)
            q_pulls.append(q_pull)
            if n_fit is not None:
                pairs["q_over_p"].append((float(n_fit), q_pull))
            if ndof is not None:
                ndof_pairs["q_over_p"].append((float(ndof), q_pull))
    packed = {name: _percentiles(values) for name, values in pulls.items()}
    packed["q_over_p"] = _percentiles(q_pulls)
    top = {name: _top_share(values) for name, values in pulls.items()}
    top["q_over_p"] = _top_share(q_pulls)

    def _corr(items: list[tuple[float, float]]) -> dict[str, Any]:
        if len(items) < 3:
            return {"n": len(items), "pearson_abs_pull": None}
        xs = np.asarray([item[0] for item in items], dtype=np.float64)
        ys = np.abs(np.asarray([item[1] for item in items], dtype=np.float64))
        if float(np.std(xs)) <= 0.0 or float(np.std(ys)) <= 0.0:
            return {"n": int(xs.size), "pearson_abs_pull": None}
        return {
            "n": int(xs.size),
            "pearson_abs_pull": float(np.corrcoef(xs, ys)[0, 1]),
        }

    def _binned(items: list[tuple[float, float]]) -> dict[str, Any]:
        buckets = {"le_2": [], "3_to_8": [], "9_to_14": [], "ge_15": []}
        for count, pull in items:
            if count <= 2:
                buckets["le_2"].append(abs(pull))
            elif count <= 8:
                buckets["3_to_8"].append(abs(pull))
            elif count <= 14:
                buckets["9_to_14"].append(abs(pull))
            else:
                buckets["ge_15"].append(abs(pull))
        return {name: _percentiles(values) for name, values in buckets.items()}

    return {
        "all": packed,
        "top1_chi2_share": top,
        "by_target": {
            key: {name: _percentiles(vals) for name, vals in payload.items()}
            for key, payload in sorted(by_target.items())
        },
        "by_source": {
            key: {name: _percentiles(vals) for name, vals in payload.items()}
            for key, payload in sorted(by_source.items())
        },
        "n_fit_vs_abs_pull": {
            name: {"correlation": _corr(items), "by_n_fit": _binned(items)}
            for name, items in pairs.items()
        },
        "ndof_vs_abs_pull": {
            name: _corr(items) for name, items in ndof_pairs.items()
        },
        "tails_deleted": False,
        "truth_used_as_diagnostic_only": True,
    }


def audit_focus(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    seed: Mapping[str, Any],
) -> dict[str, Any]:
    focus = _focus_keys(config)
    rows = []
    for official in sample["contracted"]:
        if identity_key(official) not in focus:
            continue
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        cov = _as_matrix(lto.get("input_covariance"), 5)
        rows.append(
            {
                "target_station": lto.get("target_station"),
                "fit_success": lto.get("fit_success"),
                "n_used_measurements": lto.get("used_measurement_count"),
                "n_measurements_in_fit": lto.get("n_measurements_in_fit"),
                "n_fit_states": lto.get("n_fit_states"),
                "ndof": lto.get("ndof"),
                "chi2": lto.get("chi2"),
                "official_q_over_p": official.get("q_over_p_per_mev"),
                "lto_q_over_p": lto.get("q_over_p_per_mev"),
                "sigma_qoverp": _sigma_q(cov),
                "used_station_ids": lto.get("used_station_ids"),
            }
        )
    q_vals = [float(row["lto_q_over_p"]) for row in rows if row.get("lto_q_over_p") is not None]
    official_q = [
        float(row["official_q_over_p"]) for row in rows if row.get("official_q_over_p") is not None
    ]
    smoke_focus = [
        item
        for item in (seed.get("comparisons") or [])
        if int(item.get("event_id", -1)) == 37
    ]
    return {
        "retained": bool(rows),
        "targets": rows,
        "lto_qoverp_identical_across_targets": bool(
            q_vals and max(q_vals) - min(q_vals) == 0.0
        ),
        "lto_qoverp_matches_official": bool(
            q_vals
            and official_q
            and abs(q_vals[0] - official_q[0]) / max(abs(official_q[0]), 1.0e-30) < 1.0e-6
        ),
        "n_fit_is_acts_measurement_flag_count": True,
        "n_fit_equals_2_means_two_measurement_updates": True,
        "sigma_qoverp_is_seed_scale": bool(
            rows
            and all(
                row.get("sigma_qoverp") is not None
                and abs(float(row["sigma_qoverp"]) - SEED_QOVERP_SIGMA) / SEED_QOVERP_SIGMA
                < 1.0e-3
                for row in rows
            )
        ),
        "seed_sensitivity_smoke": smoke_focus,
        "special_cased": False,
        "deleted": False,
        "downweighted": False,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    seed = inventory.get("seed_sensitivity") or {}
    provenance = inventory.get("provenance") or {}
    info = inventory.get("information") or {}
    pulls = inventory.get("pulls") or {}
    focus = inventory.get("focus") or {}
    denom = inventory.get("denominator") or {}
    if inventory.get("seed_scale_selected"):
        refuse_seed_scale_choice()
    active: list[str] = []
    if denom.get("frozen_denominator_holds") is False:
        primary = "wb103_denominator_redefined"
        verdict = "FAIL"
    elif seed.get("present") is False or int(seed.get("n_compared") or 0) == 0:
        primary = CASE_BLOCKED
        verdict = "BLOCKED"
    else:
        flags = {
            "A": bool(provenance.get("is_seed_or_predicted")),
            "B": bool(
                seed.get("typical_qoverp_seed_sensitive")
                or (
                    info.get("median_n_fit_over_used") is not None
                    and float(info["median_n_fit_over_used"])
                    < float(
                        (inventory.get("thresholds") or {}).get(
                            "n_fit_over_used_insufficient_max", 0.4
                        )
                    )
                )
            ),
            "C": False,
            "D": False,
        }
        all_pulls = pulls.get("all") or {}
        top = pulls.get("top1_chi2_share") or {}
        p68s = [
            (all_pulls.get(name) or {}).get("p68")
            for name in ("x", "y", "tx", "ty", "q_over_p")
        ]
        systematic = [value for value in p68s if value is not None and value >= 2.0]
        flags["C"] = bool(systematic) and not flags["A"]
        y_top = top.get("y")
        flags["D"] = bool(
            y_top is not None
            and y_top >= 0.5
            and (all_pulls.get("y") or {}).get("p68") is not None
            and float((all_pulls.get("y") or {})["p68"]) < 2.0
        )
        active = [name for name, flag in flags.items() if flag]
        if flags["A"] and not flags["B"] and not flags["C"]:
            primary = CASE_A
        elif flags["B"] and not flags["A"] and not flags["C"] and not flags["D"]:
            primary = CASE_B
        elif flags["C"] and not flags["A"] and not flags["B"] and not flags["D"]:
            primary = CASE_C
        elif flags["D"] and not flags["A"] and not flags["B"] and not flags["C"]:
            primary = CASE_D
        else:
            primary = CASE_E
        verdict = "DIAGNOSED"
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "active_mechanisms": (
            []
            if primary in (CASE_BLOCKED, "wb103_denominator_redefined")
            else active
        ),
        "next_step": (
            "repair_lto_state_extraction"
            if primary == CASE_A
            else "redefine_physically_supported_lto_state_model"
            if primary == CASE_B
            else "audit_acts_fitter_covariance_semantics"
            if primary == CASE_C
            else "fit_failure_provenance_keep_tails"
            if primary == CASE_D
            else "wait_b14r_smoke"
            if primary == CASE_BLOCKED
            else "keep_mixed_provenance_no_single_root_cause"
        ),
        "b15_authorized": False,
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "lto_cin_contract_established": False,
        "seed_scale_selected_from_closure": False,
        "focus_identity_retained": bool(focus.get("retained", True)),
        "focus_n_fit_equals_two_measurement_updates": bool(
            focus.get("n_fit_equals_2_means_two_measurement_updates")
        ),
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_lto_dumps(config)
    sample = load_contracted_sample(config)
    if not any(identity_key(row) in _focus_keys(config) for row in sample["contracted"]):
        refuse_focus_drop()
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    lto_by_key = {_row_key(row): row for row in dumps["rows"]}
    seed_loaded = load_seed_sensitivity_dumps(config)
    info = audit_fit_information(config, sample, lto_by_key)
    seed = audit_seed_sensitivity(config, seed_loaded)
    provenance = audit_covariance_provenance(seed_loaded)
    information = audit_parameter_information(info, seed)
    pulls = audit_pull_correlation(config, sample, lto_by_key)
    focus = audit_focus(config, sample, lto_by_key, seed)
    return {
        "dumps": {
            "present_sources": dumps["present_sources"],
            "missing_sources": dumps["missing_sources"],
            "dumps_present": dumps["dumps_present"],
        },
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "fit_information": info,
        "seed_sensitivity": seed,
        "provenance": provenance,
        "information": information,
        "pulls": pulls,
        "focus": focus,
        "thresholds": config["provenance_thresholds"],
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "seed_scale_selected": False,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    if mechanism["b15_authorized"]:
        refuse_b15()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb110_rewritten": False,
    }
