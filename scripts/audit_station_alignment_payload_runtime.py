#!/usr/bin/env python3
"""Audit whether a Calypso station-alignment payload rebinds persisted tracklets.

The audit separates two questions that must not be conflated: whether the
SQLite/POOL payload reached the SCT and ACTS condition algorithms, and whether
existing xAOD local segment coordinates changed during ntuple export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from alignment.payload import load_station_alignment_payload


def _columns(path: Path, tree_name: str, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        available = {str(name).split(";")[0] for name in tree.keys()}
        missing = [field for field in fields if field not in available]
        if missing:
            raise ValueError(f"{path} is missing {tree_name} fields: {', '.join(missing)}")
        raw = tree.arrays(list(fields), library="np")
    if isinstance(raw, dict):
        return {field: np.asarray(raw[field]) for field in fields}
    if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
        return {field: np.asarray(raw[field]) for field in fields}
    raise TypeError("uproot returned an unsupported ROOT column container")


def _sorted(columns: dict[str, np.ndarray], identity: tuple[str, ...]) -> dict[str, np.ndarray]:
    order = np.lexsort(tuple(columns[field] for field in reversed(identity)))
    return {name: values[order] for name, values in columns.items()}


def _range(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(finite))),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-tracklets", required=True)
    parser.add_argument("--condition-tracklets", required=True)
    parser.add_argument("--nominal-propagations", required=True)
    parser.add_argument("--condition-propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--job-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    payload = load_station_alignment_payload(args.payload_manifest)
    job_log = Path(args.job_log).expanduser().resolve()
    if not job_log.is_file():
        raise FileNotFoundError(job_log)
    log_text = job_log.read_text(encoding="utf-8", errors="replace")
    required_log_markers = {
        "sqlite_override": "Reading folder /Tracker/Align from sqlite",
        "sct_alignment_store": "recorded new CDO SCTAlignmentStore",
        "acts_alignment_context": "Recorded new FaserActsAlignment",
        "job_success": "Execution succeeded",
    }
    condition_checks = {
        name: marker in log_text for name, marker in required_log_markers.items()
    }

    state_fields = ("x_mm", "y_mm", "z_mm", "tx", "ty")
    tracklet_identity = ("run_id", "event_id", "tracklet_id")
    tracklet_fields = (*tracklet_identity, "station_id", *state_fields)
    nominal_tracklets = _sorted(
        _columns(Path(args.nominal_tracklets).expanduser().resolve(), "tracklets", tracklet_fields),
        tracklet_identity,
    )
    condition_tracklets = _sorted(
        _columns(Path(args.condition_tracklets).expanduser().resolve(), "tracklets", tracklet_fields),
        tracklet_identity,
    )
    for field in (*tracklet_identity, "station_id"):
        if not np.array_equal(nominal_tracklets[field], condition_tracklets[field]):
            raise ValueError(f"tracklet identity differs between nominal and condition reruns: {field}")

    state_delta = {
        field: condition_tracklets[field] - nominal_tracklets[field] for field in state_fields
    }
    station_summary: dict[str, object] = {}
    for station in np.unique(nominal_tracklets["station_id"]):
        mask = nominal_tracklets["station_id"] == station
        expected_x, expected_y = payload.offset_for_station(int(station))
        station_summary[str(int(station))] = {
            "tracklets": int(np.count_nonzero(mask)),
            "expected_coordinate_delta_xy_mm": [expected_x, expected_y],
            "actual_delta": {field: _range(values[mask]) for field, values in state_delta.items()},
            "x_minus_expected_max_abs_mm": float(
                np.max(np.abs(state_delta["x_mm"][mask] - expected_x))
            ),
            "y_minus_expected_max_abs_mm": float(
                np.max(np.abs(state_delta["y_mm"][mask] - expected_y))
            ),
        }

    propagation_identity = (
        "run_id",
        "event_id",
        "source_tracklet_id",
        "target_tracklet_id",
        "q_over_p_mode",
    )
    propagation_fields = (*propagation_identity, "pred_x_mm", "pred_y_mm", "pred_tx", "pred_ty")
    nominal_propagations = _sorted(
        _columns(
            Path(args.nominal_propagations).expanduser().resolve(),
            "propagations",
            propagation_fields,
        ),
        propagation_identity,
    )
    condition_propagations = _sorted(
        _columns(
            Path(args.condition_propagations).expanduser().resolve(),
            "propagations",
            propagation_fields,
        ),
        propagation_identity,
    )
    for field in propagation_identity:
        if not np.array_equal(nominal_propagations[field], condition_propagations[field]):
            raise ValueError(f"propagation identity differs between nominal and condition reruns: {field}")
    propagation_delta = {
        field.removeprefix("pred_"): _range(condition_propagations[field] - nominal_propagations[field])
        for field in ("pred_x_mm", "pred_y_mm", "pred_tx", "pred_ty")
    }

    coordinate_rebound = all(
        values["x_minus_expected_max_abs_mm"] <= 1.0e-6
        and values["y_minus_expected_max_abs_mm"] <= 1.0e-6
        for values in station_summary.values()
    )
    summary = {
        "payload_manifest": str(payload.manifest_path),
        "payload_sqlite": str(payload.sqlite_path),
        "payload_pool": str(payload.pool_path),
        "job_log": str(job_log),
        "condition_chain_checks": condition_checks,
        "condition_chain_loaded": bool(all(condition_checks.values())),
        "persisted_tracklet_coordinate_rebound_observed": coordinate_rebound,
        "station_tracklet_deltas": station_summary,
        "propagation_prediction_deltas": propagation_delta,
        "interpretation": (
            "A true condition chain load with no coordinate rebound means this exporter reads persisted "
            "local segment states in their stored representation. Use the explicit coordinate-level payload "
            "surrogate for V1 closure; do not label it a raw-hit deformed-geometry reconstruction."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
