"""Task B14S: physically supported LTO track-state contract.

Asks which 5D directions remaining measurements actually support after
target removal.  Does not pick a seed scale, introduce a prior, delete
q/p, drop 100043/37, enter B14P, enter B15, or enter V2.
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
from datasets.acts_transport_dump import _as_matrix
from datasets.acts_transport_diagnosis import _stats, load_dump_records
from datasets.acts_transport_tail_analysis import identity_key
from datasets.leave_target_out_state_materialization import (
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
    SEED_LOC_VARIANCE_MM2,
    SEED_QOVERP_SIGMA_PER_MEV,
    _row_key,
    load_lto_dumps,
)
from datasets.lto_fit_covariance_provenance import (
    CASE_E as WB111_DECISION,
    PARAM_NAMES,
    SEED_QOVERP_SIGMA,
    _bound_gev_to_mev,
    _percentiles,
    _relative_change,
    _sigma_q,
    inherit_frozen_stage as inherit_through_wb110,
)
from datasets.transport_uncertainty_shape_diagnosis import load_contracted_sample

SCHEMA_VERSION = "lto-supported-state-contract-v1"
DEFAULT_CONFIG = "configs/lto_supported_state_contract_v1.yaml"
TASK = "SB-B14S"
WORKBOOK = 112

CASE_A = "full_5d_lto_state_measurement_supported"
CASE_B = "reduced_measurement_supported_state_with_weak_nuisance"
CASE_C = "lto_track_fit_information_insufficient"
CASE_D = "catastrophic_fit_mechanism_unresolved"
CASE_E = "mixed_or_inconclusive"
CASE_BLOCKED = "lto_directional_smoke_incomplete"

CLASS_MEASUREMENT = "measurement_dominated"
CLASS_WEAK = "weakly_measured"
CLASS_PRIOR = "prior_dominated"
CLASS_UNCONSTRAINED = "unconstrained"
CLASS_FAILED = "fit_failed"

SEED_BOUND_MEV = np.diag(
    [SEED_LOC_VARIANCE_MM2, SEED_LOC_VARIANCE_MM2, 2.5e-3, 2.5e-3, SEED_QOVERP_SIGMA_PER_MEV**2]
)
NATIVE_NAMES = ("loc0", "loc1", "phi", "theta", "q_over_p")


class LtoSupportedStateError(ValueError):
    """Raised when the B14S contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise LtoSupportedStateError("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise LtoSupportedStateError("covariance rescale is forbidden")


def refuse_seed_scale_choice() -> None:
    raise LtoSupportedStateError("seed scale must not be chosen from truth or chi2")


def refuse_prior_introduction() -> None:
    raise LtoSupportedStateError("B14S inventories priors but does not introduce one")


def refuse_qoverp_deletion() -> None:
    raise LtoSupportedStateError("q/p must remain an explicit nuisance, not be deleted")


def refuse_focus_drop() -> None:
    raise LtoSupportedStateError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise LtoSupportedStateError("Measurement Model V2 is not entered in Task B14S")


def refuse_b15() -> None:
    raise LtoSupportedStateError("Task B15 is not entered in Task B14S")


def refuse_b14p() -> None:
    raise LtoSupportedStateError("Task B14P is not entered until B14S Case B is official")


def refuse_raw_ckf() -> None:
    raise LtoSupportedStateError("raw CKF must not re-enter the official B14S input")


def refuse_full_track_prior() -> None:
    raise LtoSupportedStateError("full-track CKF q/p or WB107 Cin cannot be an LTO prior")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise LtoSupportedStateError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise LtoSupportedStateError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise LtoSupportedStateError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise LtoSupportedStateError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise LtoSupportedStateError(f"{key} must be false")
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
        "do_not_enter_b14p",
        "eligibility_independent_of_closure",
        "do_not_select_states_from_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise LtoSupportedStateError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise LtoSupportedStateError(f"frozen gate changed: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise LtoSupportedStateError("B14S eligibility must stay identical to WB103")
    scales = [float(v) for v in config["seed_sensitivity"]["scales"]]
    if scales != [0.1, 1.0, 10.0]:
        raise LtoSupportedStateError("B14S seed scales must stay 0.1 / 1 / 10")
    if list(config["seed_sensitivity"]["directions"]) != list(PARAM_NAMES):
        raise LtoSupportedStateError("B14S directions must be x y tx ty q_over_p")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb110(config)
    spec = config["inheritance"]["workbook_111"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise LtoSupportedStateError(f"workbook_111 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise LtoSupportedStateError(f"workbook_111 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise LtoSupportedStateError("workbook_111 decision must stay frozen")
    if spec["frozen_decision"] != WB111_DECISION:
        raise LtoSupportedStateError("WB111 decision token mismatch")
    if decision.get("b15_authorized"):
        raise LtoSupportedStateError("WB111 must not have authorized B15")
    inherited["workbook_111"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "active_mechanisms": list(spec.get("frozen_active_mechanisms") or []),
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


def _identity_triple(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("source_id")), int(row.get("run_id", -1)), int(row.get("event_id", -1)))


def _station_z(config: Mapping[str, Any], station: int) -> float | None:
    raw = config.get("station_z_mm") or {}
    value = raw.get(station, raw.get(str(station)))
    return None if value is None else float(value)


def _surviving_stations(target: int) -> list[int]:
    return [station for station in (0, 1, 2, 3) if station != target]


def _bending_path(surviving: list[int]) -> dict[str, Any]:
    crosses_magnet = 0 in surviving and any(station >= 1 for station in surviving)
    first_downstream = min((station for station in surviving if station >= 1), default=None)
    return {
        "surviving_stations": surviving,
        "crosses_spectrometer_magnet": crosses_magnet,
        "magnet_gap": [0, first_downstream] if crosses_magnet else None,
        "downstream_stations": [station for station in surviving if station >= 1],
        "source_only": surviving == [0],
    }


def audit_state_information_map(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {}
    for target in (1, 2, 3):
        surviving = _surviving_stations(target)
        items = [
            official
            for official in sample["contracted"]
            if int(official.get("target_station", -1)) == target
        ]
        used_counts = []
        n_fits = []
        zspans = []
        station_hits: dict[int, list[int]] = {station: [] for station in range(4)}
        for official in items:
            lto = lto_by_key.get(_row_key(official))
            if lto is None:
                continue
            used_counts.append(float(lto.get("n_used_measurements", lto.get("used_measurement_count") or 0)))
            if lto.get("n_measurements_in_fit") is not None:
                n_fits.append(float(lto["n_measurements_in_fit"]))
            zs = lto.get("measurement_z_span_mm")
            if zs is None:
                stations = [int(v) for v in (lto.get("used_station_ids") or [])]
                zvals = [_station_z(config, station) for station in stations]
                zvals = [value for value in zvals if value is not None]
                if len(zvals) >= 2:
                    zs = max(zvals) - min(zvals)
            if zs is not None:
                zspans.append(float(zs))
            counts = lto.get("used_measurement_counts_by_station") or []
            for station, count in enumerate(counts):
                station_hits[station].append(int(count))
        path = _bending_path(surviving)
        z_surviving = [_station_z(config, station) for station in surviving]
        by_target[str(target)] = {
            "excluded_target": target,
            "surviving_stations": surviving,
            "surviving_station_z_mm": z_surviving,
            "position_constraints": {
                "x": f"stereo/strip loc0 projections on stations {surviving}",
                "y": f"stereo combination on stations {surviving}",
            },
            "slope_lever_arms": {
                "tx": f"Δx/Δz among stations {surviving}",
                "ty": f"Δy/Δz among stations {surviving}",
                "max_dz_mm": (
                    None
                    if any(value is None for value in z_surviving) or len(z_surviving) < 2
                    else float(max(z_surviving) - min(z_surviving))  # type: ignore[arg-type]
                ),
            },
            "state_information_sources": {
                "x": {
                    "constraint": "position",
                    "surviving_stations": surviving,
                    "measurement_kind": "stereo/strip loc0 projections",
                },
                "y": {
                    "constraint": "position",
                    "surviving_stations": surviving,
                    "measurement_kind": "stereo combination",
                },
                "tx": {
                    "constraint": "slope",
                    "lever_arm": f"Δx/Δz among stations {surviving}",
                    "max_dz_mm": (
                        None
                        if any(value is None for value in z_surviving) or len(z_surviving) < 2
                        else float(max(z_surviving) - min(z_surviving))  # type: ignore[arg-type]
                    ),
                },
                "ty": {
                    "constraint": "slope",
                    "lever_arm": f"Δy/Δz among stations {surviving}",
                    "max_dz_mm": (
                        None
                        if any(value is None for value in z_surviving) or len(z_surviving) < 2
                        else float(max(z_surviving) - min(z_surviving))  # type: ignore[arg-type]
                    ),
                },
                "q_over_p": {
                    "constraint": "bending",
                    "spectrometer_magnet_between_stations": [0, 1],
                    "crosses_spectrometer_magnet": path["crosses_spectrometer_magnet"],
                    "remaining_bending_stations": path["magnet_gap"],
                    "field_integral_invented": False,
                },
            },
            "q_over_p_path": path,
            "residual_bending_lever_arm": (
                "magnet 0→first-downstream plus remaining tracker span"
                if path["crosses_spectrometer_magnet"]
                else "no spectrometer-magnet crossing remains"
            ),
            "magnet_gap_mm": (
                None
                if not path["crosses_spectrometer_magnet"] or path["magnet_gap"] is None
                else (
                    None
                    if _station_z(config, path["magnet_gap"][1]) is None
                    else float(_station_z(config, path["magnet_gap"][1]) - _station_z(config, 0))  # type: ignore[operator]
                )
            ),
            "geometric_degeneracy": {
                "one_d_strip_measurements": True,
                "qoverp_requires_magnet_and_lever": True,
                "source_only_qoverp_degenerate": path["source_only"],
                "target_3_shortens_downstream_lever": target == 3,
            },
            "n_contracted": len(items),
            "n_used_measurements": _stats(used_counts),
            "n_measurements_in_fit": _stats(n_fits),
            "measurement_z_span": _stats(zspans),
            "hits_by_surviving_station": {
                str(station): _stats([float(v) for v in values])
                for station, values in station_hits.items()
                if station in surviving
            },
        }
    return {
        "magnet_statement": config["geometry"]["magnet_statement"],
        "spectrometer_magnet_between_stations": list(
            config["geometry"]["spectrometer_magnet_between_stations"]
        ),
        "no_field_integral_invented": True,
        "by_target": by_target,
        "pooled_numbers_only": False,
    }


def load_directional_dumps(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["seed_sensitivity"]
    root = resolve_under_root(project_root(), spec["dump_root"])
    source_id = spec["source_id"]
    path = root / source_id / "ckf_leave_target_out.jsonl"
    rows = load_dump_records(path, split="train") if path.is_file() else []
    wanted = {
        (str(source_id), int(event))
        for event in spec["event_ids"]
    }
    kept = [
        row
        for row in rows
        if (str(row.get("source_id")), int(row.get("event_id", -1))) in wanted
    ]
    return {
        "present": bool(kept),
        "path": str(path),
        "n_rows": len(kept),
        "rows": kept,
    }


def audit_directional_sensitivity(
    config: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    if not loaded.get("present"):
        return {
            "present": False,
            "scale_selected_from_closure": False,
            "combination_search": False,
        }
    threshold = float(config["state_class_thresholds"]["measurement_dominated_seed_rel_max"])
    by_key: dict[tuple[Any, ...], dict[str, dict[float, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in loaded["rows"]:
        if not row.get("fit_success"):
            continue
        direction = str(row.get("seed_covariance_direction") or "")
        if direction not in PARAM_NAMES:
            continue
        scale = float(row.get("seed_covariance_scale", 1.0))
        key = (
            str(row.get("source_id")),
            int(row.get("event_id", -1)),
            int(row.get("target_station", -1)),
        )
        by_key[key][direction][scale] = row
    comparisons = []
    typical_flags = {name: [] for name in PARAM_NAMES}
    for key, by_direction in sorted(by_key.items()):
        item = {
            "source_id": key[0],
            "event_id": key[1],
            "target_station": key[2],
            "directions": {},
        }
        for direction in PARAM_NAMES:
            scales = by_direction.get(direction) or {}
            if 1.0 not in scales or 0.1 not in scales or 10.0 not in scales:
                continue
            ref = scales[1.0]
            ref_state = np.asarray(ref.get("derived_state"), dtype=np.float64).reshape(-1)
            ref_cov = _as_matrix(ref.get("input_covariance"), 5)
            payload = {
                "n_fit_1x": ref.get("n_measurements_in_fit"),
                "q_over_p_1x": ref.get("q_over_p_per_mev"),
                "seed_direction_bound_parameter": ref.get("seed_direction_bound_parameter"),
                "by_scale": {},
                "mean_dlog": {},
                "sigma_dlog": {},
                "prior_dominated_this_direction": False,
            }
            for scale, row in sorted(scales.items()):
                state = np.asarray(row.get("derived_state"), dtype=np.float64).reshape(-1)
                cov = _as_matrix(row.get("input_covariance"), 5)
                mean_rel = {}
                sigma_rel = {}
                sigmas = {}
                if state.size == 5 and ref_state.size == 5:
                    for index, name in enumerate(PARAM_NAMES):
                        mean_rel[name] = _relative_change(float(state[index]), float(ref_state[index]))
                if cov is not None and ref_cov is not None:
                    for index, name in enumerate(PARAM_NAMES):
                        ref_s = float(np.sqrt(max(ref_cov[index, index], 0.0)))
                        cur_s = float(np.sqrt(max(cov[index, index], 0.0)))
                        sigmas[name] = cur_s
                        sigma_rel[name] = _relative_change(cur_s, ref_s)
                payload["by_scale"][str(scale)] = {
                    "derived_state": state.tolist() if state.size else None,
                    "sigma": sigmas,
                    "mean_relative_change_vs_1x": mean_rel,
                    "sigma_relative_change_vs_1x": sigma_rel,
                    "n_measurements_in_fit": row.get("n_measurements_in_fit"),
                    "chi2": row.get("chi2"),
                    "q_over_p": row.get("q_over_p_per_mev"),
                }
            # Pre-registered finite difference: (f(10)-f(0.1)) / (ln 10 - ln 0.1)
            low = payload["by_scale"].get("0.1") or {}
            high = payload["by_scale"].get("10.0") or {}
            dlog = float(np.log(10.0) - np.log(0.1))
            low_state = np.asarray(low.get("derived_state") or [], dtype=np.float64).reshape(-1)
            high_state = np.asarray(high.get("derived_state") or [], dtype=np.float64).reshape(-1)
            for index, name in enumerate(PARAM_NAMES):
                if low_state.size == 5 and high_state.size == 5:
                    payload["mean_dlog"][name] = float(
                        (high_state[index] - low_state[index]) / dlog
                    )
                else:
                    payload["mean_dlog"][name] = None
                sig_hi = (high.get("sigma") or {}).get(name)
                sig_lo = (low.get("sigma") or {}).get(name)
                payload["sigma_dlog"][name] = (
                    None if sig_hi is None or sig_lo is None else float((sig_hi - sig_lo) / dlog)
                )
            own_changes = [
                (payload["by_scale"][str(scale)].get("sigma_relative_change_vs_1x") or {}).get(direction)
                for scale in (0.1, 10.0)
            ]
            payload["own_sigma_rel_max"] = max(
                [value for value in own_changes if value is not None],
                default=None,
            )
            payload["prior_dominated_this_direction"] = bool(
                payload["own_sigma_rel_max"] is not None
                and payload["own_sigma_rel_max"] > threshold
            )
            item["directions"][direction] = payload
            if int(key[1]) != 37:
                typical_flags[direction].append(payload["prior_dominated_this_direction"])
        comparisons.append(item)
    typical = {
        name: bool(flags) and sum(flags) >= max(1, len(flags) // 2)
        for name, flags in typical_flags.items()
    }
    return {
        "present": True,
        "n_compared": len(comparisons),
        "typical_direction_prior_dominated": typical,
        "comparisons": comparisons,
        "scale_selected_from_closure": False,
        "combination_search": False,
        "tx_ty_are_bound_phi_theta_proxies": True,
        "production_scale_unchanged": 1.0,
    }


def _spd_inverse(matrix: np.ndarray) -> np.ndarray | None:
    try:
        eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        if np.min(eig) <= 0.0:
            return None
        return np.linalg.inv(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return None


def _information_pair(c_fit: np.ndarray, c_seed: np.ndarray) -> dict[str, Any] | None:
    i_fit = _spd_inverse(c_fit)
    i_seed = _spd_inverse(c_seed)
    if i_fit is None or i_seed is None:
        return None
    delta = i_fit - i_seed
    try:
        gain = np.linalg.eigvalsh(c_seed @ i_fit)
        delta_eig = np.linalg.eigvalsh(0.5 * (delta + delta.T))
        _, vecs = np.linalg.eigh(0.5 * (delta + delta.T))
    except np.linalg.LinAlgError:
        return None
    diag_ratio = []
    for index in range(5):
        seed_var = float(c_seed[index, index])
        fit_var = float(c_fit[index, index])
        diag_ratio.append(None if fit_var <= 0.0 else seed_var / fit_var)
    return {
        "generalized_eigenvalues": [float(v) for v in gain],
        "delta_information_eigenvalues": [float(v) for v in delta_eig],
        "information_gain_eigenvectors": vecs.T.tolist(),
        "diagonal_variance_ratio_seed_over_fit": diag_ratio,
        "chart": "native_bound_mev_same_source_plane",
    }


def audit_information_gain(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    catastrophic = {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["catastrophic_identities"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    typical_diag: dict[str, list[float]] = defaultdict(list)
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None or not lto.get("fit_success"):
            continue
        native = _as_matrix(lto.get("native_covariance"), 5)
        derived = _as_matrix(lto.get("input_covariance"), 5)
        if native is None:
            continue
        pair = _information_pair(native, SEED_BOUND_MEV)
        if pair is None:
            continue
        identity = _identity_triple(official)
        n_fit = lto.get("n_measurements_in_fit")
        sigma_q = _sigma_q(derived if derived is not None else native)
        seedlike = (
            sigma_q is not None
            and abs(sigma_q - SEED_QOVERP_SIGMA) / SEED_QOVERP_SIGMA < 1.0e-3
        )
        record = {
            "source_id": official.get("source_id"),
            "event_id": official.get("event_id"),
            "target_station": official.get("target_station"),
            "n_measurements_in_fit": n_fit,
            "seedlike_qoverp": seedlike,
            "catastrophic_focus": identity in catastrophic,
            **pair,
        }
        grouped["all"].append(record)
        grouped[str(official.get("split"))].append(record)
        grouped[f"target_{int(official.get('target_station', -1))}"].append(record)
        grouped[str(official.get("source_id"))].append(record)
        if identity in catastrophic or (n_fit is not None and int(n_fit) <= 2) or seedlike:
            grouped["catastrophic"].append(record)
        else:
            grouped["typical"].append(record)
            for index, name in enumerate(NATIVE_NAMES):
                ratio = pair["diagonal_variance_ratio_seed_over_fit"][index]
                if ratio is not None:
                    typical_diag[name].append(float(ratio))

    def _pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        gains = [item["generalized_eigenvalues"] for item in items]
        array = np.asarray(gains, dtype=np.float64)
        return {
            "n": len(items),
            "generalized_eigenvalues": {
                "median": [float(v) for v in np.median(array, axis=0)],
                "p05": [float(v) for v in np.quantile(array, 0.05, axis=0)],
                "p95": [float(v) for v in np.quantile(array, 0.95, axis=0)],
            },
            "n_seedlike_qoverp": sum(1 for item in items if item.get("seedlike_qoverp")),
            "n_n_fit_le_2": sum(
                1
                for item in items
                if item.get("n_measurements_in_fit") is not None
                and int(item["n_measurements_in_fit"]) <= 2
            ),
        }

    return {
        "same_surface_as_seed": True,
        "same_chart": "native_bound_mev",
        "derived_chart_delta_i": (
            "seed_derived_covariance mapped on B14S smoke only; "
            "full-sample official gain uses native bound MeV"
        ),
        "c_seed_analytic": SEED_BOUND_MEV.tolist(),
        "all": _pack(grouped["all"]),
        "typical": _pack(grouped["typical"]),
        "catastrophic": _pack(grouped["catastrophic"]),
        "construction": _pack(grouped["construction"]),
        "validation": _pack(grouped["validation"]),
        "by_target": {
            key: _pack(grouped[f"target_{key}"])
            for key in ("1", "2", "3")
        },
        "typical_diagonal_gain_ratio": {
            name: _stats(values) for name, values in typical_diag.items()
        },
        "by_source": {
            source_id: _pack(grouped[source_id])
            for source_id in sorted(
                {
                    str(official.get("source_id"))
                    for official in sample["contracted"]
                    if official.get("source_id") is not None
                }
            )
        },
    }


def _classify_direction(
    *,
    gain_ratio: float | None,
    seed_rel: float | None,
    n_fit: int | None,
    mean_frozen: bool,
    thresholds: Mapping[str, Any],
) -> str:
    if n_fit is not None and int(n_fit) <= int(thresholds["unconstrained_n_fit_max"]) and (
        gain_ratio is None or gain_ratio <= float(thresholds["prior_dominated_gain_max"])
    ):
        return CLASS_UNCONSTRAINED
    if gain_ratio is not None and gain_ratio <= float(thresholds["prior_dominated_gain_max"]):
        return CLASS_PRIOR
    if mean_frozen and seed_rel is not None and seed_rel > 0.5:
        return CLASS_PRIOR
    if (
        gain_ratio is not None
        and gain_ratio >= float(thresholds["measurement_dominated_gain_min"])
        and seed_rel is not None
        and seed_rel <= float(thresholds["measurement_dominated_seed_rel_max"])
    ):
        return CLASS_MEASUREMENT
    if gain_ratio is not None and gain_ratio >= float(thresholds["weakly_measured_gain_min"]):
        return CLASS_WEAK
    if seed_rel is not None and seed_rel > float(thresholds["measurement_dominated_seed_rel_max"]):
        return CLASS_PRIOR
    return CLASS_WEAK


def audit_supported_contract(
    config: Mapping[str, Any],
    directional: Mapping[str, Any],
    gain: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["state_class_thresholds"]
    typical_gain = {
        name: (gain.get("typical_diagonal_gain_ratio") or {}).get(native, {}).get("median")
        for name, native in zip(PARAM_NAMES, NATIVE_NAMES)
    }
    typical_seed = directional.get("typical_direction_prior_dominated") or {}
    classes = {}
    for name in PARAM_NAMES:
        seed_rel_flag = typical_seed.get(name)
        # Recover a representative own_sigma_rel_max from typical comparisons.
        rels = []
        for item in directional.get("comparisons") or []:
            if int(item.get("event_id", -1)) == 37:
                continue
            payload = (item.get("directions") or {}).get(name) or {}
            if payload.get("own_sigma_rel_max") is not None:
                rels.append(float(payload["own_sigma_rel_max"]))
        seed_rel = float(np.median(rels)) if rels else (0.9 if seed_rel_flag else 0.1)
        classes[name] = _classify_direction(
            gain_ratio=typical_gain.get(name),
            seed_rel=seed_rel,
            n_fit=14,
            mean_frozen=False,
            thresholds=thresholds,
        )
    q_class = classes["q_over_p"]
    return {
        "thresholds": thresholds,
        "typical_direction_class": classes,
        "q_over_p_statement": (
            "q/p must remain an explicit nuisance / externally constrained latent parameter"
            if q_class in (CLASS_PRIOR, CLASS_WEAK, CLASS_UNCONSTRAINED)
            else "q/p is measurement-dominated on typical events"
        ),
        "q_over_p_deleted": False,
        "q_over_p_fixed_to_seed": False,
        "five_d_state_physically_supported": all(
            classes[name] == CLASS_MEASUREMENT for name in PARAM_NAMES
        ),
        "classification_used_truth_or_chi2": False,
        "classification_used_transport_gate": False,
    }


def audit_prior_inventory() -> dict[str, Any]:
    return {
        "prior_introduced": False,
        "authorized_as_production_prior": False,
        "candidates": [
            {
                "name": "beam_or_production_momentum",
                "usable_as_real_data_prior": False,
                "reason": "MC particle-gun energy is not a real-data prior; listed only as inventory",
                "uses_target_measurement": False,
                "uses_truth": True,
            },
            {
                "name": "source_only_ift_tracking",
                "usable_as_real_data_prior": False,
                "reason": "station 0 does not cross the spectrometer magnet; q/p unconstrained",
                "uses_target_measurement": False,
                "uses_truth": False,
            },
            {
                "name": "upstream_independent_reconstruction",
                "usable_as_real_data_prior": False,
                "reason": "no target-independent momentum spectrometer besides tracker+magnet",
                "uses_target_measurement": False,
                "uses_truth": False,
            },
            {
                "name": "full_track_ckf_qoverp",
                "usable_as_real_data_prior": False,
                "reason": "contains target-station measurements; forbidden",
                "uses_target_measurement": True,
                "uses_truth": False,
            },
            {
                "name": "official_wb107_cin",
                "usable_as_real_data_prior": False,
                "reason": "full-track KF front state; forbidden",
                "uses_target_measurement": True,
                "uses_truth": False,
            },
            {
                "name": "closure_inverted_or_smoke_scale",
                "usable_as_real_data_prior": False,
                "reason": "0.1/1/10 and truth/chi2 widths are forbidden production values",
                "uses_target_measurement": False,
                "uses_truth": False,
            },
        ],
    }


def audit_catastrophic(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    directional: Mapping[str, Any],
    gain: Mapping[str, Any],
) -> dict[str, Any]:
    wanted = {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["catastrophic_identities"]
    }
    wanted.update(_focus_keys(config))
    rows = []
    for official in sample["contracted"]:
        lto = lto_by_key.get(_row_key(official))
        if lto is None:
            continue
        identity = _identity_triple(official)
        cov = _as_matrix(lto.get("input_covariance") or lto.get("native_covariance"), 5)
        sigma_q = _sigma_q(cov)
        seedlike = (
            sigma_q is not None
            and abs(sigma_q - SEED_QOVERP_SIGMA) / SEED_QOVERP_SIGMA < 1.0e-3
        )
        if identity not in wanted and not seedlike:
            continue
        native = _as_matrix(lto.get("native_covariance"), 5)
        pair = _information_pair(native, SEED_BOUND_MEV) if native is not None else None
        q_gain = None
        if pair is not None:
            q_gain = pair["diagonal_variance_ratio_seed_over_fit"][4]
        rows.append(
            {
                "source_id": official.get("source_id"),
                "run_id": official.get("run_id"),
                "event_id": official.get("event_id"),
                "target_station": official.get("target_station"),
                "fit_success": lto.get("fit_success"),
                "used_measurement_count": lto.get("used_measurement_count"),
                "n_measurements_in_fit": lto.get("n_measurements_in_fit"),
                "used_station_ids": lto.get("used_station_ids"),
                "used_measurement_counts_by_station": lto.get(
                    "used_measurement_counts_by_station"
                ),
                "source_station_measurements_used": lto.get(
                    "source_station_measurements_used"
                ),
                "n_outlier_hits_recovered": lto.get("n_outlier_hits_recovered"),
                "measurement_flag_updates": lto.get("n_measurements_in_fit"),
                "filtered_smoothed_outlier_counts": "unavailable_on_frozen_wb109_dumps",
                "surface_ordering": lto.get("used_station_ids"),
                "residual_dimension": "unavailable_on_frozen_wb109_dumps",
                "chi2": lto.get("chi2"),
                "q_over_p_per_mev": lto.get("q_over_p_per_mev"),
                "official_q_over_p": official.get("q_over_p_per_mev"),
                "sigma_qoverp": sigma_q,
                "seedlike_qoverp": seedlike,
                "qoverp_information_gain_ratio": q_gain,
                "qoverp_information_gain_near_zero": bool(
                    q_gain is not None and q_gain <= 1.25
                ),
                "special_cased": False,
                "deleted": False,
            }
        )
    smoke37 = [
        item
        for item in directional.get("comparisons") or []
        if int(item.get("event_id", -1)) == 37
    ]
    return {
        "retained_all_events": True,
        "rejection_cut_designed": False,
        "n_reported": len(rows),
        "rows": rows,
        "focus_37_directional_smoke": smoke37,
        "qoverp_gain_near_zero_coincides_with_seedlike": bool(
            rows
            and all(
                (not row.get("seedlike_qoverp"))
                or row.get("qoverp_information_gain_near_zero")
                for row in rows
            )
        ),
        "typical_gain_summary": gain.get("typical"),
        "catastrophic_gain_summary": gain.get("catastrophic"),
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    directional = inventory.get("directional") or {}
    contract = inventory.get("contract") or {}
    catastrophic = inventory.get("catastrophic") or {}
    denom = inventory.get("denominator") or {}
    if inventory.get("seed_scale_selected"):
        refuse_seed_scale_choice()
    if inventory.get("prior_introduced"):
        refuse_prior_introduction()
    if denom.get("frozen_denominator_holds") is False:
        primary = "wb103_denominator_redefined"
        verdict = "FAIL"
        active: list[str] = []
    elif directional.get("present") is False:
        primary = CASE_BLOCKED
        verdict = "BLOCKED"
        active = []
    else:
        classes = contract.get("typical_direction_class") or {}
        n_meas = sum(1 for name in PARAM_NAMES if classes.get(name) == CLASS_MEASUREMENT)
        n_prior = sum(
            1
            for name in PARAM_NAMES
            if classes.get(name) in (CLASS_PRIOR, CLASS_UNCONSTRAINED)
        )
        n_weak = sum(1 for name in PARAM_NAMES if classes.get(name) == CLASS_WEAK)
        n_supported = n_meas + n_weak
        explained = bool(catastrophic.get("qoverp_gain_near_zero_coincides_with_seedlike"))
        flags = {
            "A": n_meas == 5,
            "B": n_supported >= 1 and (n_prior + n_weak) >= 1,
            "C": n_meas == 0 and n_weak == 0 and n_prior >= 3,
            "D": n_meas >= 4 and not explained,
        }
        if flags["A"] and not flags["D"]:
            primary = CASE_A
        elif flags["C"] and not flags["B"]:
            primary = CASE_C
        elif flags["B"] and not flags["C"] and (explained or not flags["D"]):
            primary = CASE_B
        elif flags["D"] and not flags["B"] and not flags["C"]:
            primary = CASE_D
        else:
            primary = CASE_E
        active = [name for name, flag in flags.items() if flag]
        verdict = "DIAGNOSED"
    return {
        "verdict": verdict,
        "decision": primary,
        "primary_case": primary,
        "active_mechanisms": active,
        "next_step": (
            "redo_b14_covariance_semantics"
            if primary == CASE_A
            else "b14p_target_independent_weak_parameter_prior_contract"
            if primary == CASE_B
            else "stop_lto_propagation_estimator"
            if primary == CASE_C
            else "continue_fit_state_provenance"
            if primary == CASE_D
            else "wait_b14s_directional_smoke"
            if primary == CASE_BLOCKED
            else "keep_mixed_state_support"
        ),
        "b14p_authorized": primary == CASE_B,
        "b15_authorized": False,
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "lto_cin_contract_established": False,
        "prior_introduced": False,
        "seed_scale_selected_from_closure": False,
        "q_over_p_deleted": False,
        "focus_identity_retained": True,
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
    directional_loaded = load_directional_dumps(config)
    info_map = audit_state_information_map(config, sample, lto_by_key)
    directional = audit_directional_sensitivity(config, directional_loaded)
    gain = audit_information_gain(config, sample, lto_by_key)
    contract = audit_supported_contract(config, directional, gain)
    prior = audit_prior_inventory()
    catastrophic = audit_catastrophic(config, sample, lto_by_key, directional, gain)
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
        "information_map": info_map,
        "directional": directional,
        "information_gain": gain,
        "contract": contract,
        "prior_inventory": prior,
        "catastrophic": catastrophic,
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "seed_scale_selected": False,
        "prior_introduced": False,
    }


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
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb111_decision_sha256": inherited["workbook_111"]["decision_sha256"],
        "inherited_wb110_decision_sha256": inherited["workbook_110"]["decision_sha256"],
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb111_rewritten": False,
    }
