"""Expansion of the frozen entry-52 residual/DQ monitoring protocol.

Applies the already-frozen occupancy rule, alarm priority, and 14973/14974
reference scale to independent 2024 r0022 runs.  Time order uses LHC fill
plus in-run skip, not a run-number proxy.  Drift can only be tagged
``alignment_drift_candidate``.  No Station or C_dx solve is opened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.real_data_occupancy_preflight import (
    format_segment,
    select_window_for_segment,
)
from alignment.real_data_residual_dq_monitoring import (
    DECISION_MONITORING_ONLY,
    ISOLATION_CHANNELS,
    STATUS_ASSOCIATION_DEGRADATION,
    STATUS_DETECTOR_CONDITION,
    STATUS_INSUFFICIENT,
    STATUS_NOMINAL,
    dq_alarm_criteria,
)
from alignment.real_data_v2_route_acceptance_scaling import plan_scale

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-residual-dq-monitoring-expansion"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml"
)
ROLE_MONITORING = "monitoring"
EXPANSION_RUNS = (14971, 14972, 14980, 14981, 14985, 14989, 15007)
STATISTICALLY_SUFFICIENT = (STATUS_NOMINAL, STATUS_DETECTOR_CONDITION)

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
    "do_not_reestimate_reference_scale",
    "do_not_reestimate_alarm_thresholds",
    "cannot_convert_drift_to_geometry",
    "do_not_extract_alignment_payload_from_self_nulling",
    "residual_reduction_is_not_alignment_success",
    "current_geometry_only",
    "do_not_repick_window_from_residual",
    "do_not_open_sealed_test",
)


def load_expansion_config(path: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expansion config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected expansion config schema: {source}")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"expansion config must set {key}=false")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"expansion config must set {key}=true")
    if payload.get("parent_decision") != DECISION_MONITORING_ONLY:
        raise ValueError("expansion must keep the frozen entry-52 decision")
    if list(payload.get("expansion_runs") or []) != list(EXPANSION_RUNS):
        raise ValueError("expansion_runs must remain the pre-listed independent r0022 set")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def load_frozen_calibration_reference(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    reference = payload.get("calibration_reference")
    if not isinstance(reference, Mapping):
        raise ValueError(f"parent run-level report lacks calibration_reference: {path}")
    if list(reference.get("runs") or []) != [14973, 14974]:
        raise ValueError("frozen calibration reference must stay 14973/14974")
    if reference.get("used_for_geometry") is not False:
        raise ValueError("frozen calibration reference must not be used for geometry")
    channels = reference.get("channels")
    if not isinstance(channels, Mapping) or "dy" not in channels or "rx" not in channels:
        raise ValueError("frozen calibration reference is missing isolation channels")
    return {
        "runs": [14973, 14974],
        "n_reference_field_edges": list(reference.get("n_reference_field_edges") or []),
        "channels": dict(channels),
        "utility": dict(reference.get("utility") or {}),
        "used_for_geometry": False,
        "not_a_geometry_correction": True,
        "reestimated": False,
        "source": str(Path(path).expanduser().resolve()),
        "parent_schema_version": payload.get("schema_version"),
    }


def frozen_alarm_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = dq_alarm_criteria(config)
    parent["reestimated"] = False
    parent["source"] = "entry_52_frozen"
    return parent


def assert_reference_not_reestimated(loaded: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    for name in ISOLATION_CHANNELS:
        left = (loaded.get("channels") or {}).get(name) or {}
        right = (parent.get("channels") or {}).get(name) or {}
        for key in ("median", "robust_scale"):
            if left.get(key) != right.get(key):
                raise ValueError(f"forbidden re-estimation of reference {name}.{key}")


def plan_full_remaining_monitoring(
    *,
    run: int,
    segment: str,
    skip_events: int,
    n_segment_events: int,
) -> dict[str, Any]:
    planned = plan_scale(
        run=run,
        role=ROLE_MONITORING,
        segment=segment,
        skip_events=skip_events,
        n_segment_events=n_segment_events,
        scale_name="full",
        requested_nevents="full_segment_remaining",
    )
    planned["current_geometry_only"] = True
    planned["needs_athena"] = True
    planned["fd_probes_generated"] = False
    planned["newton_run"] = False
    planned["geometry_write_allowed"] = False
    return planned


def select_expansion_window_from_scans(
    *,
    run: int,
    occupancy_root: Path,
    rule: Mapping[str, Any],
    rec_root: Path | None = None,
) -> dict[str, Any]:
    start = format_segment(str(rule["segment_policy"]["start_segment"]))
    chosen = None
    segment = start
    scanned: list[str] = []
    while True:
        scan_path = Path(occupancy_root) / "runs" / f"{int(run):05d}" / segment / "occupancy_windows.json"
        if not scan_path.is_file():
            payload = {
                "run": int(run),
                "role": ROLE_MONITORING,
                "segment": segment,
                "window_accepted": False,
                "status": "occupancy_scan_missing" if chosen is None else "need_next_segment",
                "next_segment": segment,
                "scanned_segments": scanned,
                "residual_blind": True,
            }
            if chosen is not None and chosen.get("status") == "need_next_segment":
                return {**chosen, "scanned_segments": scanned}
            return payload
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        if scan.get("residual_blind") is not True:
            raise ValueError(f"{scan_path} is not residual-blind")
        if scan.get("v2_scoring") is not False or scan.get("alignment_perturbation") is not False:
            raise ValueError(f"{scan_path} used V2 or alignment perturbation")
        result = select_window_for_segment(
            run=run,
            segment=segment,
            windows=list(scan.get("windows") or []),
            rule=rule,
            rec_root=rec_root,
            role=ROLE_MONITORING,
        )
        result["input_xaod"] = scan.get("input_xaod")
        result["occupancy_windows_json"] = str(scan_path)
        result["n_events_scanned"] = scan.get("n_events_scanned")
        occupancy_root_file = scan.get("occupancy_root") or str(scan_path.parent / "occupancy.root")
        result["occupancy_root"] = occupancy_root_file
        result["lhc_fill"] = _scan_fill({**scan, "occupancy_root": occupancy_root_file})
        scanned.append(segment)
        chosen = result
        if result["window_accepted"]:
            chosen["scanned_segments"] = scanned
            return chosen
        if result.get("status") != "need_next_segment":
            chosen["scanned_segments"] = scanned
            return chosen
        segment = str(result["next_segment"])


def _scan_fill(scan: Mapping[str, Any]) -> int | None:
    occupancy_root = scan.get("occupancy_root")
    if occupancy_root:
        fill = lhc_fill_from_occupancy_root(Path(str(occupancy_root)))
        if fill is not None:
            return fill
    for window in scan.get("windows") or []:
        fill = window.get("fill")
        if fill is not None:
            return int(fill)
    return None


def lhc_fill_from_occupancy_root(path: Path) -> int | None:
    if not Path(path).is_file():
        return None
    import numpy as np
    import uproot

    with uproot.open(path) as source:
        tree = source["occupancy"] if "occupancy" in source else None
        if tree is None:
            return None
        if "fill" not in tree:
            return None
        values = np.unique(np.asarray(tree["fill"].array(library="np")))
    finite = [int(value) for value in values.tolist() if int(value) > 0]
    if not finite:
        return None
    return int(finite[0])


def occupancy_time_key(
    *,
    lhc_fill: int | None,
    run: int,
    skip_events: int | None,
) -> tuple[int, int, int]:
    """Real data-taking order: LHC fill, then run, then in-file skip."""
    fill = 10**9 if lhc_fill is None else int(lhc_fill)
    skip = 0 if skip_events is None else int(skip_events)
    return (fill, int(run), skip)


def order_blocks_by_real_time(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decorated = []
    for block in blocks:
        key = occupancy_time_key(
            lhc_fill=None if block.get("lhc_fill") is None else int(block["lhc_fill"]),
            run=int(block["run"]),
            skip_events=None if block.get("skip_events") is None else int(block["skip_events"]),
        )
        item = dict(block)
        item["time_order_key"] = list(key)
        item["time_order"] = "lhc_fill_then_run_then_skip_events"
        decorated.append((key, item))
    decorated.sort(key=lambda pair: pair[0])
    return [item for _, item in decorated]


def time_stability_in_real_time(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = order_blocks_by_real_time(blocks)
    series = []
    isolation_points: dict[str, list[tuple[int, float]]] = {name: [] for name in ISOLATION_CHANNELS}
    for index, block in enumerate(ordered):
        series.append(
            {
                "run": int(block["run"]),
                "lhc_fill": block.get("lhc_fill"),
                "skip_events": block.get("skip_events"),
                "time_order_key": block.get("time_order_key"),
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
            if median is not None:
                isolation_points[name].append((index, float(median)))
    import numpy as np

    slopes: dict[str, float | None] = {}
    for name, points in isolation_points.items():
        if len(points) < 2:
            slopes[name] = None
            continue
        values = []
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                dx = float(points[right][0] - points[left][0])
                if dx == 0.0:
                    continue
                values.append((points[right][1] - points[left][1]) / dx)
        slopes[name] = None if not values else float(np.median(np.asarray(values, dtype=np.float64)))
    return {
        "order": "lhc_fill_then_run_then_skip_events",
        "series": series,
        "isolation_theil_sen_slope_per_time_step": slopes,
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "not_a_geometry_correction": True,
    }


def attach_consecutive_drift_in_time(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = order_blocks_by_real_time(blocks)
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
            "ordered_by": "real_time_lhc_fill_then_run_then_skip",
            "not_a_geometry_correction": True,
        }
        updated.append(current)
        previous_medians = current_medians
        previous_run = int(current["run"])
    return updated


def detect_adjacent_alignment_drift(
    blocks: Sequence[Mapping[str, Any]],
    *,
    robust_z_drift: float,
    min_adjacent_runs: int,
) -> dict[str, Any]:
    """Same-sign isolation drift on adjacent statistically sufficient runs."""
    sufficient = [
        block
        for block in order_blocks_by_real_time(blocks)
        if block.get("status") in STATISTICALLY_SUFFICIENT
    ]
    flagged: list[str] = []
    supporting: dict[str, list[dict[str, Any]]] = {}
    for name in ISOLATION_CHANNELS:
        streak: list[dict[str, Any]] = []
        best: list[dict[str, Any]] = []
        last_sign: int | None = None
        for block in sufficient:
            value = (block.get("isolation_robust_standardized_shift") or {}).get(name)
            if value is None:
                streak = []
                last_sign = None
                continue
            z = float(value)
            if abs(z) < float(robust_z_drift):
                streak = []
                last_sign = None
                continue
            sign = 1 if z > 0.0 else -1
            if last_sign is not None and sign != last_sign:
                streak = []
            streak.append({"run": int(block["run"]), "robust_z": z, "lhc_fill": block.get("lhc_fill")})
            last_sign = sign
            if len(streak) > len(best):
                best = list(streak)
        if len(best) >= int(min_adjacent_runs):
            flagged.append(name)
            supporting[name] = best
    return {
        "alignment_drift_candidate": bool(flagged),
        "channels": flagged,
        "supporting_runs": supporting,
        "order": "adjacent_statistically_sufficient_runs_in_real_time",
        "cannot_convert_to_geometry_correction": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "follow_up_if_true": "independent_survey_or_external_alignment_not_self_nulling",
        "note": (
            "A repeatable dy/rx residual shift on adjacent sufficient runs is "
            "an alignment_drift_candidate only.  It must not reopen self-nulling "
            "calibration or be inverted into a station payload."
        ),
    }


def interpret_expansion(
    expansion_blocks: Sequence[Mapping[str, Any]],
    drift: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = [str(block.get("status")) for block in expansion_blocks]
    n = len(statuses)
    n_nominal = statuses.count(STATUS_NOMINAL)
    n_detector = statuses.count(STATUS_DETECTOR_CONDITION)
    consecutive_alarm = bool(drift.get("alignment_drift_candidate")) or n_detector >= 2
    if n and n_nominal >= max(1, n - 1) and not consecutive_alarm:
        reading = "official_geometry_and_frozen_v2_can_carry_long_term_dq_monitoring"
    elif consecutive_alarm:
        reading = "independent_survey_or_external_alignment_follow_up"
    else:
        reading = "continue_current_geometry_residual_dq_monitoring"
    return {
        "reading": reading,
        "n_expansion_runs": n,
        "n_nominal": n_nominal,
        "n_detector_condition_change": n_detector,
        "n_association_or_reconstruction_degradation": statuses.count(STATUS_ASSOCIATION_DEGRADATION),
        "n_insufficient_statistics": statuses.count(STATUS_INSUFFICIENT),
        "alignment_drift_candidate": bool(drift.get("alignment_drift_candidate")),
        "reopen_self_nulling_calibration": False,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
    }


def first_wave_segments(config: Mapping[str, Any]) -> list[str]:
    start = format_segment(str(config["occupancy_preflight"]["start_segment"]))
    stop = format_segment(str(config["occupancy_preflight"]["first_wave_last_segment"]))
    return [format_segment(index) for index in range(int(start), int(stop) + 1)]
