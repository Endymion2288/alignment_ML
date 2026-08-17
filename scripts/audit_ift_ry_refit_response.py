#!/usr/bin/env python3
"""Compare nominal and true IFT-R_y physical refit/Acts responses.

The input pair must come from independent ``/Tracker/Align`` payloads and the
full persisted-cluster -> SegmentFitRefit -> SegmentsRefit -> Acts chain.  The
audit deliberately reports observed state/residual changes rather than
predicting a rotation response from coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import EventTracklets, load_events
from evaluation.field_propagation import evaluate_field_propagation, field_propagation_summary


STATE_LABELS = ("x_mm", "y_mm", "z_mm", "tx", "ty")
RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")
PREDICTOR_LABELS = ("x_mm", "y_mm", "tx", "ty")


def _stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": None,
        "rms": None,
        "median": None,
        "p95": None,
        "minimum": None,
        "maximum": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95.0)),
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
            }
        )
    return result


def _state(event: EventTracklets, row: int) -> np.ndarray:
    return np.asarray(
        [
            event.state[row, 0],
            event.state[row, 1],
            event.z_mm[row],
            event.state[row, 2],
            event.state[row, 3],
        ],
        dtype=np.float64,
    )


def _tracklet_index(events: Iterable[EventTracklets]) -> dict[tuple[int, int, int], tuple[EventTracklets, int]]:
    index: dict[tuple[int, int, int], tuple[EventTracklets, int]] = {}
    for event in events:
        for row, tracklet_id in enumerate(event.tracklet_id):
            key = (int(event.run_id), int(event.event_id), int(tracklet_id))
            if key in index:
                raise ValueError(f"duplicate canonical tracklet identity: {key}")
            index[key] = (event, row)
    return index


def _linear_response(predictor: np.ndarray, response: np.ndarray) -> dict[str, float | int | None]:
    """Return an observed one-dimensional slope/correlation, never a model prior."""
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return {"count": int(x.size), "slope": None, "intercept": None, "correlation": None}
    variance = float(np.var(x))
    if variance <= 0.0:
        return {"count": int(x.size), "slope": None, "intercept": None, "correlation": None}
    slope = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    correlation = None if float(np.std(y)) == 0.0 else float(np.corrcoef(x, y)[0, 1])
    return {
        "count": int(x.size),
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
    }


def _positive_logdet(covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(covariance, dtype=np.float64))
    if sign <= 0.0 or not np.isfinite(logdet):
        raise ValueError("compared covariance is not positive definite")
    return float(logdet)


def _tracklet_response(
    nominal_events: list[EventTracklets], displaced_events: list[EventTracklets]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    nominal = _tracklet_index(nominal_events)
    displaced = _tracklet_index(displaced_events)
    common = sorted(set(nominal).intersection(displaced))
    rows: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    stations: list[int] = []
    covariance_logdet_delta: list[float] = []
    mismatch = 0
    for key in common:
        nominal_event, nominal_row = nominal[key]
        displaced_event, displaced_row = displaced[key]
        station = int(nominal_event.station_id[nominal_row])
        if station != int(displaced_event.station_id[displaced_row]):
            mismatch += 1
            continue
        if nominal_event.truth_particle_id is None or displaced_event.truth_particle_id is None:
            raise ValueError("IFT R_y response audit requires MC truth labels")
        truth = int(nominal_event.truth_particle_id[nominal_row])
        if truth != int(displaced_event.truth_particle_id[displaced_row]):
            mismatch += 1
            continue
        state = _state(nominal_event, nominal_row)
        changed = _state(displaced_event, displaced_row)
        delta = changed - state
        covariance_delta = _positive_logdet(displaced_event.covariance[displaced_row]) - _positive_logdet(
            nominal_event.covariance[nominal_row]
        )
        row: dict[str, object] = {
            "run_id": key[0],
            "event_id": key[1],
            "tracklet_id": key[2],
            "station_id": station,
            "truth_particle_id": truth,
            "covariance_logdet_change": covariance_delta,
            "covariance_max_abs_change": float(
                np.max(np.abs(displaced_event.covariance[displaced_row] - nominal_event.covariance[nominal_row]))
            ),
        }
        for index, label in enumerate(STATE_LABELS):
            row[f"nominal_{label}"] = float(state[index])
            row[f"displaced_{label}"] = float(changed[index])
            row[f"delta_{label}"] = float(delta[index])
        rows.append(row)
        states.append(state)
        deltas.append(delta)
        stations.append(station)
        covariance_logdet_delta.append(covariance_delta)
    if not rows:
        raise ValueError("no truth-stable tracklet identity survives the physical refit comparison")
    state_array = np.asarray(states, dtype=np.float64)
    delta_array = np.asarray(deltas, dtype=np.float64)
    station_array = np.asarray(stations, dtype=np.int16)
    by_station: dict[str, object] = {}
    for station in sorted(set(station_array.tolist())):
        mask = station_array == station
        regressions = {
            response_name: {
                predictor_name: _linear_response(state_array[mask, predictor_index], delta_array[mask, response_index])
                for predictor_index, predictor_name in enumerate(PREDICTOR_LABELS)
            }
            for response_index, response_name in enumerate(STATE_LABELS)
        }
        by_station[str(station)] = {
            "matched_tracklets": int(np.count_nonzero(mask)),
            "state_delta": {
                label: _stats(delta_array[mask, index]) for index, label in enumerate(STATE_LABELS)
            },
            "state_delta_vs_nominal_state": regressions,
            "covariance_logdet_change": _stats(np.asarray(covariance_logdet_delta)[mask]),
        }
    return rows, {
        "nominal_tracklets": len(nominal),
        "displaced_tracklets": len(displaced),
        "common_tracklet_ids": len(common),
        "missing_from_displaced": len(set(nominal) - set(displaced)),
        "extra_in_displaced": len(set(displaced) - set(nominal)),
        "station_or_truth_identity_mismatch": mismatch,
        "matched_tracklets": len(rows),
        "by_station": by_station,
    }


def _pair_key(evaluation, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(evaluation.run_id[row]),
        int(evaluation.event_id[row]),
        int(evaluation.source_tracklet_id[row]),
        int(evaluation.target_tracklet_id[row]),
        int(evaluation.q_over_p_mode[row]),
    )


def _propagation_response(nominal, displaced) -> tuple[list[dict[str, object]], dict[str, object]]:
    nominal_index = {_pair_key(nominal, row): row for row in range(nominal.size)}
    displaced_index = {_pair_key(displaced, row): row for row in range(displaced.size)}
    if len(nominal_index) != nominal.size or len(displaced_index) != displaced.size:
        raise ValueError("duplicate truth-pair identity in mode-0 propagation response")
    common = sorted(set(nominal_index).intersection(displaced_index))
    rows: list[dict[str, object]] = []
    deltas_by_pair: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    pulls_by_pair: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    chi2_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    covlog_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for key in common:
        nominal_row = nominal_index[key]
        displaced_row = displaced_index[key]
        pair = (int(nominal.source_station_id[nominal_row]), int(nominal.target_station_id[nominal_row]))
        if pair != (
            int(displaced.source_station_id[displaced_row]),
            int(displaced.target_station_id[displaced_row]),
        ):
            raise ValueError(f"station-pair identity drift for truth pair {key}")
        residual_delta = displaced.residual[displaced_row] - nominal.residual[nominal_row]
        pull_delta = displaced.pull[displaced_row] - nominal.pull[nominal_row]
        chi2_delta = float(displaced.chi2[displaced_row] - nominal.chi2[nominal_row])
        covlog_delta = _positive_logdet(displaced.combined_covariance[displaced_row]) - _positive_logdet(
            nominal.combined_covariance[nominal_row]
        )
        row: dict[str, object] = {
            "run_id": key[0],
            "event_id": key[1],
            "source_tracklet_id": key[2],
            "target_tracklet_id": key[3],
            "q_over_p_mode": key[4],
            "source_station_id": pair[0],
            "target_station_id": pair[1],
            "truth_particle_id": int(nominal.truth_particle_id[nominal_row]),
            "nominal_chi2": float(nominal.chi2[nominal_row]),
            "displaced_chi2": float(displaced.chi2[displaced_row]),
            "delta_chi2": chi2_delta,
            "combined_covariance_logdet_change": covlog_delta,
            "combined_covariance_max_abs_change": float(
                np.max(np.abs(displaced.combined_covariance[displaced_row] - nominal.combined_covariance[nominal_row]))
            ),
        }
        for index, label in enumerate(RESIDUAL_LABELS):
            row[f"nominal_{label}"] = float(nominal.residual[nominal_row, index])
            row[f"displaced_{label}"] = float(displaced.residual[displaced_row, index])
            row[f"delta_{label}"] = float(residual_delta[index])
            row[f"nominal_pull_{label}"] = float(nominal.pull[nominal_row, index])
            row[f"displaced_pull_{label}"] = float(displaced.pull[displaced_row, index])
            row[f"delta_pull_{label}"] = float(pull_delta[index])
        rows.append(row)
        deltas_by_pair[pair].append(residual_delta)
        pulls_by_pair[pair].append(pull_delta)
        chi2_by_pair[pair].append(chi2_delta)
        covlog_by_pair[pair].append(covlog_delta)
    if not rows:
        raise ValueError("no common truth-matched mode-0 propagation pair survives the response audit")
    by_pair = {
        f"{source}->{target}": {
            "truth_matched_pairs": len(deltas_by_pair[(source, target)]),
            "residual_delta": {
                label: _stats(np.asarray(deltas_by_pair[(source, target)])[:, index])
                for index, label in enumerate(RESIDUAL_LABELS)
            },
            "pull_delta": {
                label: _stats(np.asarray(pulls_by_pair[(source, target)])[:, index])
                for index, label in enumerate(RESIDUAL_LABELS)
            },
            "chi2_delta": _stats(np.asarray(chi2_by_pair[(source, target)])),
            "combined_covariance_logdet_change": _stats(np.asarray(covlog_by_pair[(source, target)])),
        }
        for source, target in sorted(deltas_by_pair)
    }
    return rows, {
        "nominal_truth_matched_pairs": nominal.size,
        "displaced_truth_matched_pairs": displaced.size,
        "common_truth_matched_pairs": len(common),
        "missing_from_displaced": len(set(nominal_index) - set(displaced_index)),
        "extra_in_displaced": len(set(displaced_index) - set(nominal_index)),
        "by_station_pair": by_pair,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-tracklets", required=True)
    parser.add_argument("--nominal-propagations", required=True)
    parser.add_argument("--displaced-tracklets", required=True)
    parser.add_argument("--displaced-propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = load_station_rigid_alignment_payload(args.payload_manifest)
    if any(abs(value) > 1.0e-15 for value in payload.transform_for_station(1)) or any(
        abs(value) > 1.0e-15 for value in payload.transform_for_station(2)
    ) or any(abs(value) > 1.0e-15 for value in payload.transform_for_station(3)):
        raise ValueError("IFT R_y response audit requires downstream stations 1--3 to be fixed")
    if abs(payload.transform_for_station(0)[4]) <= 1.0e-15:
        raise ValueError("IFT R_y response audit requires a non-zero station-0 R_y payload")
    nominal_events = load_events(args.nominal_tracklets, require_mc_labels=True)
    displaced_events = load_events(args.displaced_tracklets, require_mc_labels=True)
    tracklet_rows, tracklet_summary = _tracklet_response(nominal_events, displaced_events)
    evaluation_kwargs = {
        "require_truth_match": True,
        "q_over_p_mode": 0,
        "min_truth_match_fraction": float(args.min_truth_match_fraction),
    }
    nominal_evaluation = evaluate_field_propagation(
        nominal_events, load_propagation_records(args.nominal_propagations), **evaluation_kwargs
    )
    displaced_evaluation = evaluate_field_propagation(
        displaced_events, load_propagation_records(args.displaced_propagations), **evaluation_kwargs
    )
    propagation_rows, propagation_summary = _propagation_response(nominal_evaluation, displaced_evaluation)
    _write_csv(output / "tracklet_state_response.csv", tracklet_rows)
    _write_csv(output / "mode0_propagation_response.csv", propagation_rows)
    summary = {
        "method": "truth_matched_physical_ift_ry_refit_response_audit",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "payload_manifest": str(payload.manifest_path),
        "station_transforms": {str(station): list(payload.transform_for_station(station)) for station in (0, 1, 2, 3)},
        "ift_ry_mrad": float(payload.transform_for_station(0)[4] * 1.0e3),
        "tracklet_state_response": tracklet_summary,
        "mode0_propagation_response": propagation_summary,
        "nominal_field_summary": field_propagation_summary(nominal_evaluation),
        "displaced_field_summary": field_propagation_summary(displaced_evaluation),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                "nominal_tracklets": str(Path(args.nominal_tracklets).expanduser().resolve()),
                "nominal_propagations": str(Path(args.nominal_propagations).expanduser().resolve()),
                "displaced_tracklets": str(Path(args.displaced_tracklets).expanduser().resolve()),
                "displaced_propagations": str(Path(args.displaced_propagations).expanduser().resolve()),
                "payload_manifest": str(payload.manifest_path),
                "min_truth_match_fraction": float(args.min_truth_match_fraction),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "ift_ry_mrad": summary["ift_ry_mrad"]}, indent=2))


if __name__ == "__main__":
    main()
