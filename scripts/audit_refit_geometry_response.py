#!/usr/bin/env python3
"""Compare nominal and displaced-geometry segment-refit outputs truth by truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from alignment.payload import load_station_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import EventTracklets, load_events
from evaluation.field_propagation import FieldPropagationEvaluation, evaluate_field_propagation


TRACKLET_CSV_FIELDS = (
    "run_id",
    "event_id",
    "tracklet_id",
    "station_id",
    "truth_particle_id",
    "expected_dx_mm",
    "expected_dy_mm",
    "nominal_x_mm",
    "nominal_y_mm",
    "nominal_z_mm",
    "nominal_tx",
    "nominal_ty",
    "displaced_x_mm",
    "displaced_y_mm",
    "displaced_z_mm",
    "displaced_tx",
    "displaced_ty",
    "delta_x_mm",
    "delta_y_mm",
    "delta_z_mm",
    "delta_tx",
    "delta_ty",
    "position_response_error_x_mm",
    "position_response_error_y_mm",
    "nominal_covariance_det",
    "displaced_covariance_det",
    "covariance_logdet_change",
    "covariance_max_abs_change",
)

PROPAGATION_CSV_FIELDS = (
    "run_id",
    "event_id",
    "source_tracklet_id",
    "target_tracklet_id",
    "source_station_id",
    "target_station_id",
    "truth_particle_id",
    "q_over_p_mode",
    "expected_residual_increment_x_mm",
    "expected_residual_increment_y_mm",
    "nominal_rx_mm",
    "nominal_ry_mm",
    "nominal_rtx",
    "nominal_rty",
    "displaced_rx_mm",
    "displaced_ry_mm",
    "displaced_rtx",
    "displaced_rty",
    "delta_rx_mm",
    "delta_ry_mm",
    "delta_rtx",
    "delta_rty",
    "residual_response_error_x_mm",
    "residual_response_error_y_mm",
    "nominal_pull_x_mm",
    "nominal_pull_y_mm",
    "nominal_pull_tx",
    "nominal_pull_ty",
    "displaced_pull_x_mm",
    "displaced_pull_y_mm",
    "displaced_pull_tx",
    "displaced_pull_ty",
    "delta_pull_x_mm",
    "delta_pull_y_mm",
    "delta_pull_tx",
    "delta_pull_ty",
    "nominal_chi2",
    "displaced_chi2",
    "delta_chi2",
    "nominal_combined_covariance_det",
    "displaced_combined_covariance_det",
    "combined_covariance_logdet_change",
    "combined_covariance_max_abs_change",
)


def _stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95.0)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _state5(event: EventTracklets, row: int) -> np.ndarray:
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


def _tracklet_index(events: list[EventTracklets]) -> dict[tuple[int, int, int], tuple[EventTracklets, int]]:
    index: dict[tuple[int, int, int], tuple[EventTracklets, int]] = {}
    for event in events:
        for row, tracklet_id in enumerate(event.tracklet_id):
            key = (event.run_id, event.event_id, int(tracklet_id))
            if key in index:
                raise ValueError(f"duplicate canonical tracklet identity: {key}")
            index[key] = (event, row)
    return index


def _positive_determinant(covariance: np.ndarray) -> tuple[float, float]:
    sign, logdet = np.linalg.slogdet(np.asarray(covariance, dtype=np.float64))
    if not np.isfinite(logdet) or sign <= 0.0:
        raise ValueError("a compared covariance is not positive definite")
    return float(np.exp(logdet)), float(logdet)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tracklet_response(
    nominal_events: list[EventTracklets],
    displaced_events: list[EventTracklets],
    payload,
) -> tuple[list[dict[str, int | float]], dict[str, np.ndarray], dict[str, object]]:
    nominal_index = _tracklet_index(nominal_events)
    displaced_index = _tracklet_index(displaced_events)
    missing = sorted(set(nominal_index) - set(displaced_index))
    extra = sorted(set(displaced_index) - set(nominal_index))
    if missing or extra:
        raise ValueError(
            "nominal and displaced tracklet outputs do not have the same identities; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    rows: list[dict[str, int | float]] = []
    states_nominal: list[np.ndarray] = []
    states_displaced: list[np.ndarray] = []
    covariances_nominal: list[np.ndarray] = []
    covariances_displaced: list[np.ndarray] = []
    station_ids: list[int] = []
    identities: list[tuple[int, int, int, int, int]] = []
    for key in sorted(nominal_index):
        nominal_event, nominal_row = nominal_index[key]
        displaced_event, displaced_row = displaced_index[key]
        nominal_station = int(nominal_event.station_id[nominal_row])
        displaced_station = int(displaced_event.station_id[displaced_row])
        if nominal_station != displaced_station:
            raise ValueError(f"station identity drift after refit: {key}")
        if nominal_event.truth_particle_id is None or displaced_event.truth_particle_id is None:
            raise ValueError("truth labels are required for a refit geometry audit")
        truth_id = int(nominal_event.truth_particle_id[nominal_row])
        if truth_id != int(displaced_event.truth_particle_id[displaced_row]):
            raise ValueError(f"truth particle identity drift after refit: {key}")
        nominal_state = _state5(nominal_event, nominal_row)
        displaced_state = _state5(displaced_event, displaced_row)
        delta = displaced_state - nominal_state
        expected_offset = np.asarray(payload.offset_for_station(nominal_station), dtype=np.float64)
        nominal_covariance = np.asarray(nominal_event.covariance[nominal_row], dtype=np.float64)
        displaced_covariance = np.asarray(displaced_event.covariance[displaced_row], dtype=np.float64)
        nominal_det, nominal_logdet = _positive_determinant(nominal_covariance)
        displaced_det, displaced_logdet = _positive_determinant(displaced_covariance)
        row = {
            "run_id": key[0],
            "event_id": key[1],
            "tracklet_id": key[2],
            "station_id": nominal_station,
            "truth_particle_id": truth_id,
            "expected_dx_mm": float(expected_offset[0]),
            "expected_dy_mm": float(expected_offset[1]),
            "nominal_x_mm": float(nominal_state[0]),
            "nominal_y_mm": float(nominal_state[1]),
            "nominal_z_mm": float(nominal_state[2]),
            "nominal_tx": float(nominal_state[3]),
            "nominal_ty": float(nominal_state[4]),
            "displaced_x_mm": float(displaced_state[0]),
            "displaced_y_mm": float(displaced_state[1]),
            "displaced_z_mm": float(displaced_state[2]),
            "displaced_tx": float(displaced_state[3]),
            "displaced_ty": float(displaced_state[4]),
            "delta_x_mm": float(delta[0]),
            "delta_y_mm": float(delta[1]),
            "delta_z_mm": float(delta[2]),
            "delta_tx": float(delta[3]),
            "delta_ty": float(delta[4]),
            "position_response_error_x_mm": float(delta[0] - expected_offset[0]),
            "position_response_error_y_mm": float(delta[1] - expected_offset[1]),
            "nominal_covariance_det": nominal_det,
            "displaced_covariance_det": displaced_det,
            "covariance_logdet_change": displaced_logdet - nominal_logdet,
            "covariance_max_abs_change": float(
                np.max(np.abs(displaced_covariance - nominal_covariance))
            ),
        }
        rows.append(row)
        states_nominal.append(nominal_state)
        states_displaced.append(displaced_state)
        covariances_nominal.append(nominal_covariance)
        covariances_displaced.append(displaced_covariance)
        station_ids.append(nominal_station)
        identities.append((key[0], key[1], key[2], nominal_station, truth_id))

    nominal_array = np.asarray(states_nominal, dtype=np.float64)
    displaced_array = np.asarray(states_displaced, dtype=np.float64)
    delta_array = displaced_array - nominal_array
    station_array = np.asarray(station_ids, dtype=np.int16)
    covariance_nominal_array = np.asarray(covariances_nominal, dtype=np.float64)
    covariance_displaced_array = np.asarray(covariances_displaced, dtype=np.float64)
    by_station: dict[str, object] = {}
    for station in sorted(set(station_array.tolist())):
        mask = station_array == station
        by_station[str(station)] = {
            "tracklets": int(np.count_nonzero(mask)),
            "state_change": {
                name: _stats(delta_array[mask, index])
                for index, name in enumerate(("x_mm", "y_mm", "z_mm", "tx", "ty"))
            },
            "position_response_error_xy_mm": {
                "x_mm": _stats(delta_array[mask, 0] - payload.offset_for_station(station)[0]),
                "y_mm": _stats(delta_array[mask, 1] - payload.offset_for_station(station)[1]),
            },
            "covariance_max_abs_change": _stats(
                np.max(
                    np.abs(covariance_displaced_array[mask] - covariance_nominal_array[mask]),
                    axis=(1, 2),
                )
            ),
        }
    arrays = {
        "run_id": np.asarray([value[0] for value in identities], dtype=np.int64),
        "event_id": np.asarray([value[1] for value in identities], dtype=np.int64),
        "tracklet_id": np.asarray([value[2] for value in identities], dtype=np.int32),
        "station_id": station_array,
        "truth_particle_id": np.asarray([value[4] for value in identities], dtype=np.int64),
        "nominal_state": nominal_array,
        "displaced_state": displaced_array,
        "state_delta": delta_array,
        "nominal_covariance": covariance_nominal_array,
        "displaced_covariance": covariance_displaced_array,
        "covariance_delta": covariance_displaced_array - covariance_nominal_array,
    }
    return rows, arrays, {"matched_tracklets": len(rows), "by_station": by_station}


def _pair_key(evaluation: FieldPropagationEvaluation, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(evaluation.run_id[row]),
        int(evaluation.event_id[row]),
        int(evaluation.source_tracklet_id[row]),
        int(evaluation.target_tracklet_id[row]),
        int(evaluation.q_over_p_mode[row]),
    )


def _pair_response(
    nominal: FieldPropagationEvaluation,
    displaced: FieldPropagationEvaluation,
    payload,
) -> tuple[list[dict[str, int | float]], dict[str, np.ndarray], dict[str, object]]:
    nominal_index = {_pair_key(nominal, row): row for row in range(nominal.size)}
    displaced_index = {_pair_key(displaced, row): row for row in range(displaced.size)}
    if len(nominal_index) != nominal.size or len(displaced_index) != displaced.size:
        raise ValueError("duplicate field-propagation identities in a geometry response audit")
    missing = sorted(set(nominal_index) - set(displaced_index))
    extra = sorted(set(displaced_index) - set(nominal_index))
    if missing or extra:
        raise ValueError(
            "nominal and displaced propagation outputs do not select the same truth pairs; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    rows: list[dict[str, int | float]] = []
    metadata: list[tuple[int, int, int, int, int, int, int, int]] = []
    residual_nominal: list[np.ndarray] = []
    residual_displaced: list[np.ndarray] = []
    pull_nominal: list[np.ndarray] = []
    pull_displaced: list[np.ndarray] = []
    covariance_nominal: list[np.ndarray] = []
    covariance_displaced: list[np.ndarray] = []
    chi2_nominal: list[float] = []
    chi2_displaced: list[float] = []
    expected_increment: list[np.ndarray] = []
    labels = ("x_mm", "y_mm", "tx", "ty")
    residual_labels = ("rx_mm", "ry_mm", "rtx", "rty")
    for key in sorted(nominal_index):
        nominal_row = nominal_index[key]
        displaced_row = displaced_index[key]
        source_station = int(nominal.source_station_id[nominal_row])
        target_station = int(nominal.target_station_id[nominal_row])
        if (
            source_station != int(displaced.source_station_id[displaced_row])
            or target_station != int(displaced.target_station_id[displaced_row])
            or int(nominal.truth_particle_id[nominal_row])
            != int(displaced.truth_particle_id[displaced_row])
        ):
            raise ValueError(f"propagation identity metadata drift after refit: {key}")
        nominal_residual = np.asarray(nominal.residual[nominal_row], dtype=np.float64)
        displaced_residual = np.asarray(displaced.residual[displaced_row], dtype=np.float64)
        nominal_pull = np.asarray(nominal.pull[nominal_row], dtype=np.float64)
        displaced_pull = np.asarray(displaced.pull[displaced_row], dtype=np.float64)
        nominal_covariance_value = np.asarray(
            nominal.combined_covariance[nominal_row], dtype=np.float64
        )
        displaced_covariance_value = np.asarray(
            displaced.combined_covariance[displaced_row], dtype=np.float64
        )
        nominal_det, nominal_logdet = _positive_determinant(nominal_covariance_value)
        displaced_det, displaced_logdet = _positive_determinant(displaced_covariance_value)
        expected_xy = np.asarray(payload.offset_for_station(target_station), dtype=np.float64) - np.asarray(
            payload.offset_for_station(source_station), dtype=np.float64
        )
        residual_delta = displaced_residual - nominal_residual
        pull_delta = displaced_pull - nominal_pull
        row: dict[str, int | float] = {
            "run_id": key[0],
            "event_id": key[1],
            "source_tracklet_id": key[2],
            "target_tracklet_id": key[3],
            "source_station_id": source_station,
            "target_station_id": target_station,
            "truth_particle_id": int(nominal.truth_particle_id[nominal_row]),
            "q_over_p_mode": key[4],
            "expected_residual_increment_x_mm": float(expected_xy[0]),
            "expected_residual_increment_y_mm": float(expected_xy[1]),
            "residual_response_error_x_mm": float(residual_delta[0] - expected_xy[0]),
            "residual_response_error_y_mm": float(residual_delta[1] - expected_xy[1]),
            "nominal_chi2": float(nominal.chi2[nominal_row]),
            "displaced_chi2": float(displaced.chi2[displaced_row]),
            "delta_chi2": float(displaced.chi2[displaced_row] - nominal.chi2[nominal_row]),
            "nominal_combined_covariance_det": nominal_det,
            "displaced_combined_covariance_det": displaced_det,
            "combined_covariance_logdet_change": displaced_logdet - nominal_logdet,
            "combined_covariance_max_abs_change": float(
                np.max(np.abs(displaced_covariance_value - nominal_covariance_value))
            ),
        }
        for index, label in enumerate(labels):
            residual_label = residual_labels[index]
            row[f"nominal_{residual_label}"] = float(nominal_residual[index])
            row[f"displaced_{residual_label}"] = float(displaced_residual[index])
            row[f"delta_{residual_label}"] = float(residual_delta[index])
            row[f"nominal_pull_{label}"] = float(nominal_pull[index])
            row[f"displaced_pull_{label}"] = float(displaced_pull[index])
            row[f"delta_pull_{label}"] = float(pull_delta[index])
        rows.append(row)
        metadata.append(
            (
                key[0],
                key[1],
                key[2],
                key[3],
                source_station,
                target_station,
                int(nominal.truth_particle_id[nominal_row]),
                key[4],
            )
        )
        residual_nominal.append(nominal_residual)
        residual_displaced.append(displaced_residual)
        pull_nominal.append(nominal_pull)
        pull_displaced.append(displaced_pull)
        covariance_nominal.append(nominal_covariance_value)
        covariance_displaced.append(displaced_covariance_value)
        chi2_nominal.append(float(nominal.chi2[nominal_row]))
        chi2_displaced.append(float(displaced.chi2[displaced_row]))
        expected_increment.append(expected_xy)

    nominal_residual_array = np.asarray(residual_nominal, dtype=np.float64)
    displaced_residual_array = np.asarray(residual_displaced, dtype=np.float64)
    nominal_pull_array = np.asarray(pull_nominal, dtype=np.float64)
    displaced_pull_array = np.asarray(pull_displaced, dtype=np.float64)
    nominal_covariance_array = np.asarray(covariance_nominal, dtype=np.float64)
    displaced_covariance_array = np.asarray(covariance_displaced, dtype=np.float64)
    expected_array = np.asarray(expected_increment, dtype=np.float64)
    arrays = {
        "run_id": np.asarray([value[0] for value in metadata], dtype=np.int64),
        "event_id": np.asarray([value[1] for value in metadata], dtype=np.int64),
        "source_tracklet_id": np.asarray([value[2] for value in metadata], dtype=np.int32),
        "target_tracklet_id": np.asarray([value[3] for value in metadata], dtype=np.int32),
        "source_station_id": np.asarray([value[4] for value in metadata], dtype=np.int16),
        "target_station_id": np.asarray([value[5] for value in metadata], dtype=np.int16),
        "truth_particle_id": np.asarray([value[6] for value in metadata], dtype=np.int64),
        "q_over_p_mode": np.asarray([value[7] for value in metadata], dtype=np.int8),
        "nominal_residual": nominal_residual_array,
        "displaced_residual": displaced_residual_array,
        "residual_delta": displaced_residual_array - nominal_residual_array,
        "nominal_pull": nominal_pull_array,
        "displaced_pull": displaced_pull_array,
        "pull_delta": displaced_pull_array - nominal_pull_array,
        "nominal_combined_covariance": nominal_covariance_array,
        "displaced_combined_covariance": displaced_covariance_array,
        "combined_covariance_delta": displaced_covariance_array - nominal_covariance_array,
        "nominal_chi2": np.asarray(chi2_nominal, dtype=np.float64),
        "displaced_chi2": np.asarray(chi2_displaced, dtype=np.float64),
        "expected_residual_increment_xy": expected_array,
    }
    summary = {
        "accepted_truth_matched_pairs": len(rows),
        "residual_change": {
            name: _stats(arrays["residual_delta"][:, index])
            for index, name in enumerate(("x_mm", "y_mm", "tx", "ty"))
        },
        "pull_change": {
            name: _stats(arrays["pull_delta"][:, index])
            for index, name in enumerate(("x_mm", "y_mm", "tx", "ty"))
        },
        "residual_response_error_xy_mm": {
            "x_mm": _stats(arrays["residual_delta"][:, 0] - expected_array[:, 0]),
            "y_mm": _stats(arrays["residual_delta"][:, 1] - expected_array[:, 1]),
        },
        "chi2_change": _stats(arrays["displaced_chi2"] - arrays["nominal_chi2"]),
        "combined_covariance_max_abs_change": _stats(
            np.max(np.abs(arrays["combined_covariance_delta"]), axis=(1, 2))
        ),
    }
    return rows, arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-tracklets", required=True)
    parser.add_argument("--nominal-propagations", required=True)
    parser.add_argument("--displaced-tracklets", required=True)
    parser.add_argument("--displaced-propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--q-over-p-modes",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=(0, 1),
        help="Propagation q/p modes to compare (default: 0 1)",
    )
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()

    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    modes = tuple(sorted(set(args.q_over_p_modes)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_outputs = [
        output_dir / "tracklet_response.csv",
        output_dir / "tracklet_response.npz",
        output_dir / "metrics.json",
        output_dir / "resolved_config.yaml",
    ]
    expected_outputs.extend(output_dir / f"propagation_response_mode{mode}.csv" for mode in modes)
    expected_outputs.extend(output_dir / f"propagation_response_mode{mode}.npz" for mode in modes)
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite an existing response audit in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_station_alignment_payload(args.payload_manifest)
    nominal_events = load_events(args.nominal_tracklets, require_mc_labels=True)
    displaced_events = load_events(args.displaced_tracklets, require_mc_labels=True)
    tracklet_rows, tracklet_arrays, tracklet_summary = _tracklet_response(
        nominal_events, displaced_events, payload
    )
    _write_csv(output_dir / "tracklet_response.csv", TRACKLET_CSV_FIELDS, tracklet_rows)
    np.savez_compressed(output_dir / "tracklet_response.npz", **tracklet_arrays)

    nominal_records = load_propagation_records(args.nominal_propagations)
    displaced_records = load_propagation_records(args.displaced_propagations)
    propagation_summary: dict[str, object] = {}
    for mode in modes:
        evaluation_kwargs = {
            "require_truth_match": True,
            "q_over_p_mode": mode,
            "min_truth_match_fraction": args.min_truth_match_fraction,
        }
        nominal_evaluation = evaluate_field_propagation(
            nominal_events, nominal_records, **evaluation_kwargs
        )
        displaced_evaluation = evaluate_field_propagation(
            displaced_events, displaced_records, **evaluation_kwargs
        )
        rows, arrays, summary = _pair_response(nominal_evaluation, displaced_evaluation, payload)
        _write_csv(
            output_dir / f"propagation_response_mode{mode}.csv", PROPAGATION_CSV_FIELDS, rows
        )
        np.savez_compressed(output_dir / f"propagation_response_mode{mode}.npz", **arrays)
        propagation_summary[str(mode)] = {
            "nominal_rejected_counts": nominal_evaluation.rejected_counts,
            "displaced_rejected_counts": displaced_evaluation.rejected_counts,
            **summary,
        }

    resolved_config = {
        "nominal_tracklets": str(Path(args.nominal_tracklets).expanduser().resolve()),
        "nominal_propagations": str(Path(args.nominal_propagations).expanduser().resolve()),
        "displaced_tracklets": str(Path(args.displaced_tracklets).expanduser().resolve()),
        "displaced_propagations": str(Path(args.displaced_propagations).expanduser().resolve()),
        "payload_manifest": str(payload.manifest_path),
        "q_over_p_modes": list(modes),
        "min_truth_match_fraction": args.min_truth_match_fraction,
    }
    metrics = {
        "method": "truth-matched_displaced_geometry_segment_refit_response_audit",
        "translation_expectation": (
            "For a pure global station dx/dy translation, reconstructed global x/y and target-minus-"
            "source residuals should change by the corresponding station/pair offset; slopes and "
            "covariances are translation-invariant."
        ),
        **resolved_config,
        "tracklets": tracklet_summary,
        "propagation": propagation_summary,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
