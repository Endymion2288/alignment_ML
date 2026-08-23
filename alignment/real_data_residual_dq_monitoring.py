"""Production-like residual/DQ monitoring for frozen current geometry.

No Station calibration mode is constructed.  No C_dx Mode is started.
No FD probe, Newton step, or payload is written.  A long-term residual
shift on ``dy/rx`` can only be tagged ``alignment_drift_candidate``.
``dx/ry`` are cross-level-sensitive.  ``dz`` is not a track-driven
observable.  Runs with too few selected routes are
``insufficient_statistics_for_alignment_dq``, not alignment anomalies.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.calibration_modes import OPERATING_BAND_UM
from alignment.five_dof_sampling import SURVEY_PARAMETER
from alignment.real_data_station_mode_fullscale import _percentiles

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-residual-dq-monitoring"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_real_data_residual_dq_monitoring_v1.yaml"
)

DECISION_MONITORING_ONLY = "real_data_residual_dq_monitoring_only"

STATUS_NOMINAL = "nominal_monitoring"
STATUS_INSUFFICIENT = "insufficient_statistics_for_alignment_dq"
STATUS_ASSOCIATION_DEGRADATION = "association_or_reconstruction_degradation"
STATUS_DETECTOR_CONDITION = "detector_condition_change"
RUN_STATUSES = (
    STATUS_INSUFFICIENT,
    STATUS_ASSOCIATION_DEGRADATION,
    STATUS_DETECTOR_CONDITION,
    STATUS_NOMINAL,
)

ROLE_CALIBRATION_REFERENCE = "calibration_reference"
ROLE_MONITORING = "monitoring"

ISOLATION_CHANNELS = {
    "dy": "residual_y_mm",
    "rx": "residual_ty",
}
CROSS_LEVEL_CHANNELS = {
    "dx": "residual_x_mm",
    "ry": "residual_tx",
}
ALL_RESIDUAL_CHANNELS = {**ISOLATION_CHANNELS, **CROSS_LEVEL_CHANNELS}

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_db_write",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "joint_station_cdx_newton",
    "new_layer_or_module_dof",
    "track_driven_dz",
    "dz_track_driven_observable",
    "rz_direct_track_residual",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_enter_cdx_mode",
    "do_not_construct_station_calibration_mode",
    "do_not_retrain_v2",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "cannot_convert_drift_to_geometry",
    "do_not_extract_alignment_payload_from_self_nulling",
    "residual_reduction_is_not_alignment_success",
    "current_geometry_only",
    "do_not_open_sealed_test",
)


def load_monitoring_config(path: str | None = None) -> dict[str, Any]:
    from pathlib import Path

    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"monitoring config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected monitoring config schema: {source}")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"monitoring config must set {key}=false")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"monitoring config must set {key}=true")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False


def campaign_allows_geometry_write(decision: str) -> bool:
    return False


def campaign_allows_station_calibration_mode(decision: str) -> bool:
    return False


def dq_alarm_criteria(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "min_selected_routes_for_alignment_dq": int(config["min_selected_routes_for_alignment_dq"]),
        "max_event_share_of_selected_routes": float(config["max_event_share_of_selected_routes"]),
        "min_all_pairs_candidates": int(config["min_all_pairs_candidates"]),
        "robust_z_detector_condition": float(config["robust_z_detector_condition"]),
        "robust_z_drift": float(config["robust_z_drift"]),
        "min_nonreference_runs_for_repeatable_drift": int(config["min_nonreference_runs_for_repeatable_drift"]),
        "priority": [
            {
                "status": STATUS_ASSOCIATION_DEGRADATION,
                "when": "selected_routes==0 and n_all_pairs_candidates>=min_all_pairs",
                "means": "association_or_reconstruction_collapsed_despite_nonempty_candidate_graph",
            },
            {
                "status": STATUS_INSUFFICIENT,
                "when": "0<selected_routes<min_selected_routes_for_alignment_dq",
                "means": "too_few_selected_routes_to_judge_alignment_dq_not_an_alignment_anomaly",
            },
            {
                "status": STATUS_ASSOCIATION_DEGRADATION,
                "when": "selected_routes>=min and max_event_share>max_event_share_of_selected_routes",
                "means": "selected_single_event_dominated",
            },
            {
                "status": STATUS_DETECTOR_CONDITION,
                "when": "isolation |robust_z| vs 14973/14974 exceeds robust_z_detector_condition",
                "means": "dy_or_rx_residual_median_left_the_calibration_reference_band",
            },
            {
                "status": STATUS_NOMINAL,
                "when": "none of the alarms above",
                "means": "within_calibration_reference_dq_band",
            },
        ],
        "alignment_drift_candidate": {
            "when": (
                "same-sign isolation |robust_z|>=robust_z_drift on at least "
                "min_nonreference_runs_for_repeatable_drift non-reference runs "
                "that are not insufficient_statistics_for_alignment_dq"
            ),
            "action": "tag_only_never_invert_to_station_payload",
        },
        "dx_ry": "report_and_mark_cross_level_sensitive_never_a_track_driven_alignment_observable",
        "dz": "never_a_track_driven_observable",
        "rz": "no_direct_track_residual_channel_do_not_fabricate",
        "cannot_convert_drift_to_geometry": True,
    }


def _finite_column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def robust_location_scale(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n": 0,
            "median": None,
            "iqr": None,
            "mad": None,
            "robust_scale": None,
            "p16": None,
            "p84": None,
        }
    median = float(np.median(finite))
    p16 = float(np.percentile(finite, 16))
    p84 = float(np.percentile(finite, 84))
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0.0:
        scale = 1.4826 * mad
    elif iqr > 0.0:
        scale = iqr / 1.349
    else:
        scale = None
    return {
        "n": int(finite.size),
        "median": median,
        "iqr": iqr,
        "mad": mad,
        "robust_scale": None if scale is None or scale <= 0.0 else float(scale),
        "p16": p16,
        "p84": p84,
    }


def robust_standardized_shift(
    run_median: float | None,
    reference: Mapping[str, float | None],
) -> float | None:
    if run_median is None:
        return None
    ref_median = reference.get("median")
    scale = reference.get("robust_scale")
    if ref_median is None or scale in (None, 0.0):
        return None
    return float((float(run_median) - float(ref_median)) / float(scale))


def residual_channel_report(values: np.ndarray, *, cross_level_sensitive: bool) -> dict[str, Any]:
    location = robust_location_scale(values)
    return {
        **location,
        "tails": _percentiles(values),
        "cross_level_sensitive": bool(cross_level_sensitive),
        "track_driven_observable": not bool(cross_level_sensitive),
        "dq_observable_only": True,
        "not_a_geometry_correction": True,
    }


def station_residual_breakdown(field_edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median/IQR/tails per downstream station.  Never a geometry correction."""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in field_edges:
        raw = row.get("target_station_id")
        if raw in (None, ""):
            continue
        grouped[int(raw)].append(row)
    stations: dict[str, Any] = {}
    for station, rows in sorted(grouped.items()):
        isolation = {
            name: residual_channel_report(_finite_column(rows, column), cross_level_sensitive=False)
            for name, column in ISOLATION_CHANNELS.items()
        }
        cross_level = {
            name: residual_channel_report(_finite_column(rows, column), cross_level_sensitive=True)
            for name, column in CROSS_LEVEL_CHANNELS.items()
        }
        stations[str(station)] = {
            "target_station_id": int(station),
            "n_edges": int(len(rows)),
            "isolation_residual_observables": isolation,
            "cross_level_sensitive_residual_observables": cross_level,
            "not_a_geometry_correction": True,
        }
    return stations


def summarize_monitoring_run(
    *,
    run: int,
    role: str,
    source_id: str,
    n_events: int,
    n_tracklets: int,
    n_all_pairs_candidates: int,
    routes: Sequence[Mapping[str, Any]],
    field_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_selected = int(len(routes))
    lengths = [int(len(row.get("endpoint_stations") or row.get("endpoint_indices") or ())) for row in routes]
    length_hist = {str(length): int(count) for length, count in sorted(Counter(lengths).items())}
    n_complete = int(
        sum(
            1
            for row in routes
            if bool(row.get("is_complete_four_station_route")) or len(row.get("endpoint_stations") or ()) == 4
        )
    )
    per_event: dict[tuple[int, int], int] = defaultdict(int)
    utilities = _finite_column(routes, "utility")
    scores = _finite_column(field_edges, "score")
    for row in routes:
        per_event[(int(row["run_id"]), int(row["event_id"]))] += 1
    selected_total = float(sum(per_event.values()))
    max_share = 0.0 if selected_total <= 0.0 else max(per_event.values()) / selected_total
    isolation = {
        name: residual_channel_report(_finite_column(field_edges, column), cross_level_sensitive=False)
        for name, column in ISOLATION_CHANNELS.items()
    }
    cross_level = {
        name: residual_channel_report(_finite_column(field_edges, column), cross_level_sensitive=True)
        for name, column in CROSS_LEVEL_CHANNELS.items()
    }
    return {
        "run": int(run),
        "role": role,
        "source_id": source_id,
        "n_events": int(n_events),
        "n_tracklets": int(n_tracklets),
        "n_all_pairs_candidates": int(n_all_pairs_candidates),
        "candidate_graph_nonempty": int(n_all_pairs_candidates) > 0,
        "selected_routes": n_selected,
        "complete_four_station_routes": n_complete,
        "route_composition": {
            "2_station": int(length_hist.get("2", 0)),
            "3_station": int(length_hist.get("3", 0)),
            "4_station": int(length_hist.get("4", 0)),
        },
        "event_concentration": {
            "n_events_with_selected_routes": int(len(per_event)),
            "max_event_share_of_selected_routes": float(max_share),
        },
        "utility_quantiles": _percentiles(utilities),
        "score_quantiles": _percentiles(scores),
        "isolation_residual_observables": isolation,
        "cross_level_sensitive_residual_observables": cross_level,
        "station_residual_observables": station_residual_breakdown(field_edges),
        "rz_direct_track_residual": False,
        "dz_track_driven_observable": False,
        "survey_parameter": SURVEY_PARAMETER,
        "current_geometry_only": True,
        "fd_probes_generated": False,
        "newton_run": False,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "not_a_geometry_correction": True,
    }


def build_calibration_reference(
    residual_rows_by_run: Mapping[int, Sequence[Mapping[str, Any]]],
    utilities_by_run: Mapping[int, Sequence[float]] | None = None,
    *,
    reference_runs: Sequence[int],
) -> dict[str, Any]:
    if not list(reference_runs):
        raise ValueError("calibration reference must contain at least one run")
    pooled: dict[str, list[float]] = {name: [] for name in ALL_RESIDUAL_CHANNELS}
    utilities: list[float] = []
    selected: list[int] = []
    missing = [int(run) for run in reference_runs if int(run) not in residual_rows_by_run]
    if missing:
        raise ValueError(f"calibration reference residuals missing for runs {missing}")
    for run in reference_runs:
        edges = residual_rows_by_run[int(run)]
        selected.append(int(len(edges)))
        for name, column in ALL_RESIDUAL_CHANNELS.items():
            pooled[name].extend(_finite_column(edges, column).tolist())
        if utilities_by_run is not None:
            utilities.extend(float(value) for value in utilities_by_run.get(int(run), ()) if math.isfinite(float(value)))
    channels = {
        name: robust_location_scale(np.asarray(values, dtype=np.float64))
        for name, values in pooled.items()
    }
    return {
        "runs": [int(run) for run in reference_runs],
        "n_reference_field_edges": selected,
        "channels": channels,
        "utility": robust_location_scale(np.asarray(utilities, dtype=np.float64)),
        "used_for_geometry": False,
        "not_a_geometry_correction": True,
    }


def attach_reference_shifts(
    block: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    isolation_shift: dict[str, float | None] = {}
    cross_shift: dict[str, float | None] = {}
    for name in ISOLATION_CHANNELS:
        isolation_shift[name] = robust_standardized_shift(
            (block.get("isolation_residual_observables") or {}).get(name, {}).get("median"),
            (reference.get("channels") or {}).get(name) or {},
        )
    for name in CROSS_LEVEL_CHANNELS:
        cross_shift[name] = robust_standardized_shift(
            (block.get("cross_level_sensitive_residual_observables") or {}).get(name, {}).get("median"),
            (reference.get("channels") or {}).get(name) or {},
        )
    return {
        "isolation_robust_standardized_shift": isolation_shift,
        "cross_level_robust_standardized_shift": cross_shift,
        "not_a_geometry_correction": True,
        "cross_level_sensitive": True,
    }


def assign_run_status(
    block: Mapping[str, Any],
    shifts: Mapping[str, Any],
    *,
    min_selected_routes: int,
    max_event_share: float,
    min_all_pairs: int,
    robust_z_detector: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    selected = int(block.get("selected_routes") or 0)
    share = float((block.get("event_concentration") or {}).get("max_event_share_of_selected_routes") or 0.0)
    pairs = int(block.get("n_all_pairs_candidates") or 0)
    if selected == 0 and pairs >= int(min_all_pairs):
        status = STATUS_ASSOCIATION_DEGRADATION
        reasons.append("empty_selected_graph_despite_nonempty_candidate_graph")
    elif selected < int(min_selected_routes):
        status = STATUS_INSUFFICIENT
        reasons.append("too_few_selected_routes_for_alignment_dq")
    elif share > float(max_event_share):
        status = STATUS_ASSOCIATION_DEGRADATION
        reasons.append("selected_single_event_dominated")
    else:
        isolation_z = [
            abs(float(value))
            for value in (shifts.get("isolation_robust_standardized_shift") or {}).values()
            if value is not None and math.isfinite(float(value))
        ]
        if isolation_z and max(isolation_z) > float(robust_z_detector):
            status = STATUS_DETECTOR_CONDITION
            reasons.append("isolation_residual_median_exceeds_detector_condition_z")
        else:
            status = STATUS_NOMINAL
            reasons.append("within_calibration_reference_dq_band")
    return {
        "status": status,
        "reasons": reasons,
        "not_an_alignment_anomaly": status == STATUS_INSUFFICIENT,
        "not_a_geometry_correction": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
    }


def attach_consecutive_drift(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(blocks, key=lambda item: int(item["run"]))
    previous_medians: dict[str, float | None] = {}
    previous_run: int | None = None
    updated: list[dict[str, Any]] = []
    for block in ordered:
        current = dict(block)
        isolation = current.get("isolation_residual_observables") or {}
        current_medians = {
            name: (isolation.get(name) or {}).get("median") for name in ISOLATION_CHANNELS
        }
        deltas: dict[str, float | None] = {}
        for name, median in current_medians.items():
            prior = previous_medians.get(name)
            if median is None or prior is None:
                deltas[name] = None
            else:
                deltas[name] = float(median) - float(prior)
        current["run_to_run_drift"] = {
            "previous_run": previous_run,
            "isolation_median_delta": deltas,
            "not_a_geometry_correction": True,
        }
        updated.append(current)
        previous_medians = current_medians
        previous_run = int(current["run"])
    return updated


def detect_alignment_drift_candidates(
    blocks: Sequence[Mapping[str, Any]],
    *,
    reference_runs: Sequence[int],
    robust_z_drift: float,
    min_nonreference_runs: int,
) -> dict[str, Any]:
    """Repeatable isolation-channel drift.  Never a geometry correction."""
    reference = {int(run) for run in reference_runs}
    per_channel: dict[str, list[dict[str, Any]]] = {name: [] for name in ISOLATION_CHANNELS}
    for block in blocks:
        run = int(block["run"])
        if run in reference:
            continue
        if block.get("status") == STATUS_INSUFFICIENT:
            continue
        shifts = block.get("isolation_robust_standardized_shift") or {}
        for name in ISOLATION_CHANNELS:
            value = shifts.get(name)
            if value is None or not math.isfinite(float(value)):
                continue
            if abs(float(value)) >= float(robust_z_drift):
                per_channel[name].append({"run": run, "robust_z": float(value)})
    flagged: list[str] = []
    for name, rows in per_channel.items():
        if len(rows) < int(min_nonreference_runs):
            continue
        signs = [1 if float(row["robust_z"]) > 0.0 else -1 for row in rows]
        if abs(sum(signs)) == len(signs):
            flagged.append(name)
    return {
        "alignment_drift_candidate": bool(flagged),
        "channels": flagged,
        "supporting_runs": {name: per_channel[name] for name in flagged},
        "cannot_convert_to_geometry_correction": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "note": (
            "A repeatable dy/rx residual shift is an alignment_drift_candidate "
            "only.  It must not be inverted into a station payload."
        ),
    }


def time_stability_series(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(blocks, key=lambda item: int(item["run"]))
    series = []
    isolation_medians: dict[str, list[tuple[int, float]]] = {name: [] for name in ISOLATION_CHANNELS}
    for block in ordered:
        series.append(
            {
                "run": int(block["run"]),
                "role": block.get("role"),
                "status": block.get("status"),
                "selected_routes": block.get("selected_routes"),
                "route_composition": block.get("route_composition"),
                "isolation_medians": {
                    name: (block.get("isolation_residual_observables") or {}).get(name, {}).get("median")
                    for name in ISOLATION_CHANNELS
                },
                "isolation_robust_standardized_shift": block.get("isolation_robust_standardized_shift"),
                "cross_level_robust_standardized_shift": block.get("cross_level_robust_standardized_shift"),
                "run_to_run_drift": block.get("run_to_run_drift"),
            }
        )
        for name in ISOLATION_CHANNELS:
            median = (block.get("isolation_residual_observables") or {}).get(name, {}).get("median")
            if median is not None and math.isfinite(float(median)):
                isolation_medians[name].append((int(block["run"]), float(median)))
    slopes = {name: _theil_sen_slope(points) for name, points in isolation_medians.items()}
    return {
        "order": "run_number_as_time_proxy",
        "series": series,
        "isolation_theil_sen_slope_per_run": slopes,
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "not_a_geometry_correction": True,
    }


def _theil_sen_slope(points: Sequence[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    slopes: list[float] = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            dx = float(points[right][0] - points[left][0])
            if dx == 0.0:
                continue
            slopes.append((points[right][1] - points[left][1]) / dx)
    if not slopes:
        return None
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def geometry_reopen_prerequisites() -> dict[str, Any]:
    return {
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "geometry_write_allowed": False,
        "do_not_extract_alignment_payload_from_self_nulling": True,
        "required_any_of": [
            {
                "id": "external_station_constraints_and_cdx_budget",
                "independent_survey_or_external_station_constraints_fix": [
                    "ift_dx_mm",
                    "ift_ry_mrad",
                    "ift_dz_mm",
                ],
                "independent_C_dx_within_operating_band_um": list(OPERATING_BAND_UM),
            },
            {
                "id": "new_independent_real_data_topology",
                "new_independent_real_data_topology_or_track_sample": True,
                "must_empirically_demonstrate_cross_run_transferable_subspace": True,
            },
        ],
        "until_then": (
            "Do not extract any alignment payload from the current "
            "self-nulling residuals.  Continue residual/DQ monitoring only."
        ),
    }


def classify_monitoring_campaign(
    blocks: Sequence[Mapping[str, Any]],
    drift: Mapping[str, Any],
    *,
    residual_used_as_success: bool = False,
    payload_extracted: bool = False,
) -> dict[str, Any]:
    if residual_used_as_success:
        raise ValueError("residual reduction must not be used as alignment success")
    if payload_extracted:
        raise ValueError("self-nulling residuals must not be turned into an alignment payload")
    statuses = [str(block.get("status")) for block in blocks]
    reasons = [
        "entry_51_no_transferable_station_subspace",
        "current_geometry_only_monitoring",
    ]
    if STATUS_INSUFFICIENT in statuses:
        reasons.append("low_statistics_runs_classified_insufficient_not_alignment")
    if drift.get("alignment_drift_candidate"):
        reasons.append("isolation_residual_drift_tagged_candidate_only")
    return {
        "decision": DECISION_MONITORING_ONLY,
        "unique_class": DECISION_MONITORING_ONLY,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "official_conditions_db_write": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "do_not_retrain_v2": True,
        "do_not_generate_fd_probes": True,
        "do_not_run_newton": True,
        "do_not_write_payload": True,
        "do_not_open_sealed_test": True,
        "do_not_extract_alignment_payload_from_self_nulling": True,
        "cannot_convert_drift_to_geometry": True,
        "residual_reduction_is_not_alignment_success": True,
        "alignment_drift_candidate": bool(drift.get("alignment_drift_candidate")),
        "geometry_reopen_prerequisites": geometry_reopen_prerequisites(),
        "reasons": reasons,
    }


def assemble_monitored_run(
    summary: Mapping[str, Any],
    reference: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    shifts = attach_reference_shifts(summary, reference)
    status = assign_run_status(
        summary,
        shifts,
        min_selected_routes=int(config["min_selected_routes_for_alignment_dq"]),
        max_event_share=float(config["max_event_share_of_selected_routes"]),
        min_all_pairs=int(config["min_all_pairs_candidates"]),
        robust_z_detector=float(config["robust_z_detector_condition"]),
    )
    return {**dict(summary), **shifts, **status}
