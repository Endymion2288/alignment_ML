"""Collection and plotting helpers for physical refit capture scans."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")


def _number_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _point_status(
    scan_root: Path,
    point: Mapping[str, object],
    capture_tolerance_mm: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    relative_dir = Path(str(point["relative_point_dir"]))
    point_dir = scan_root / relative_dir
    closure_path = point_dir / "closure" / "closure.json"
    failure_path = point_dir / "failure.json"
    base = {
        "point_name": str(point["name"]),
        "relative_point_dir": str(relative_dir),
        "magnitude_mm": float(point["magnitude_mm"]),
        "direction_trial": str(point["direction_trial"]),
        "injected_offsets_xy_mm": point["injected_offsets_xy_mm"],
        "capture_tolerance_mm": float(capture_tolerance_mm),
    }
    if not closure_path.is_file():
        failure = _load_json(failure_path) if failure_path.is_file() else {}
        return (
            {
                **base,
                "status": str(failure.get("status", "missing")),
                "failure_phase": failure.get("phase"),
                "failure_message": failure.get("message"),
                "capture_success": False,
            },
            [],
        )

    closure = _load_json(closure_path)
    station_offsets = point["injected_offsets_xy_mm"]
    expected_rank = 2 * (len(station_offsets) - 1)
    rank = int(closure.get("normal_matrix_rank", -1))
    max_error = _number_or_none(closure.get("movable_station_max_norm_error_mm"))
    active_pairs = int(closure.get("active_truth_matched_pairs", 0))
    capture_success = bool(
        closure.get("physical_geometry_repropagation") is True
        and rank == expected_rank
        and active_pairs > 0
        and max_error is not None
        and max_error <= capture_tolerance_mm
    )
    station_errors = closure.get("absolute_error_norm_mm_by_station", {})
    record: dict[str, object] = {
        **base,
        "status": "complete",
        "capture_success": capture_success,
        "accepted_truth_matched_pairs": int(closure.get("accepted_truth_matched_pairs", 0)),
        "active_truth_matched_pairs": active_pairs,
        "wls_active_counts": closure.get("active_counts", []),
        "normal_matrix_rank": rank,
        "expected_normal_matrix_rank": expected_rank,
        "normal_matrix_condition_number": _number_or_none(
            closure.get("normal_matrix_condition_number")
        ),
        "recovery_error_max_norm_mm": max_error,
        "recovery_error_rms_mm": _number_or_none(
            closure.get("movable_station_rms_error_mm")
        ),
        "physical_response_increment_rms_error_mm": _number_or_none(
            closure.get("physical_response_increment_rms_error_mm")
        ),
        "absolute_error_norm_mm_by_station": station_errors,
        "recovered_offsets_xy_mm": closure.get("recovered_offsets_xy_mm", {}),
        "error_xy_mm": closure.get("error_xy_mm", {}),
        "closure_json": str(closure_path),
    }
    recovered_offsets = closure.get("recovered_offsets_xy_mm", {})
    component_errors = closure.get("error_xy_mm", {})
    for station, error in station_errors.items():
        record[f"station_{station}_absolute_error_norm_mm"] = _number_or_none(error)
        recovered = recovered_offsets.get(str(station), recovered_offsets.get(station, []))
        difference = component_errors.get(str(station), component_errors.get(station, []))
        if isinstance(recovered, Sequence) and len(recovered) == 2:
            record[f"station_{station}_recovered_dx_mm"] = _number_or_none(recovered[0])
            record[f"station_{station}_recovered_dy_mm"] = _number_or_none(recovered[1])
        if isinstance(difference, Sequence) and len(difference) == 2:
            record[f"station_{station}_error_dx_mm"] = _number_or_none(difference[0])
            record[f"station_{station}_error_dy_mm"] = _number_or_none(difference[1])

    diagnostics: list[dict[str, object]] = []
    by_pair = closure.get("displaced_field_aware_by_station_pair", {})
    if isinstance(by_pair, Mapping):
        for pair_name, pair_value in sorted(by_pair.items()):
            if not isinstance(pair_value, Mapping):
                continue
            try:
                source, target = (int(part) for part in str(pair_name).split("->", maxsplit=1))
            except ValueError:
                continue
            field_aware = pair_value.get("field_aware", {})
            if not isinstance(field_aware, Mapping):
                continue
            residual = field_aware.get("residual", {})
            pull = field_aware.get("pull", {})
            chi2 = field_aware.get("chi2", {})
            row: dict[str, object] = {
                "point_name": record["point_name"],
                "magnitude_mm": record["magnitude_mm"],
                "direction_trial": record["direction_trial"],
                "source_station": source,
                "target_station": target,
                "station_pair": str(pair_name),
                "records": int(pair_value.get("records", 0)),
                "chi2_mean": _number_or_none(chi2.get("mean")) if isinstance(chi2, Mapping) else None,
                "chi2_median": _number_or_none(chi2.get("median")) if isinstance(chi2, Mapping) else None,
                "chi2_rms": _number_or_none(chi2.get("rms")) if isinstance(chi2, Mapping) else None,
            }
            for state in STATE_NAMES:
                residual_stats = residual.get(state, {}) if isinstance(residual, Mapping) else {}
                pull_stats = pull.get(state, {}) if isinstance(pull, Mapping) else {}
                row[f"r_{state}_mean"] = _number_or_none(
                    residual_stats.get("mean") if isinstance(residual_stats, Mapping) else None
                )
                row[f"r_{state}_rms"] = _number_or_none(
                    residual_stats.get("rms") if isinstance(residual_stats, Mapping) else None
                )
                row[f"pull_{state}_mean"] = _number_or_none(
                    pull_stats.get("mean") if isinstance(pull_stats, Mapping) else None
                )
                row[f"pull_{state}_rms"] = _number_or_none(
                    pull_stats.get("rms") if isinstance(pull_stats, Mapping) else None
                )
            diagnostics.append(row)
    return record, diagnostics


def collect_physical_capture_scan(
    scan_root: str | Path,
    scan_plan: Mapping[str, object],
    capture_tolerance_mm: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Read one closure result per planned real-geometry injection."""
    root = Path(scan_root).expanduser().resolve()
    points = scan_plan.get("points")
    if not isinstance(points, Sequence):
        raise ValueError("scan plan must contain a points sequence")
    results: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError("each scan-plan point must be a mapping")
        result, point_diagnostics = _point_status(root, point, capture_tolerance_mm)
        results.append(result)
        diagnostics.extend(point_diagnostics)
    return results, diagnostics


def summarise_by_magnitude(points: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compute planned-trial capture fractions at every injected magnitude."""
    grouped: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        grouped[float(point["magnitude_mm"])].append(point)
    summary: list[dict[str, object]] = []
    for magnitude, rows in sorted(grouped.items()):
        successful = [row for row in rows if bool(row.get("capture_success"))]
        complete = [row for row in rows if row.get("status") == "complete"]
        errors = np.asarray(
            [
                float(row["recovery_error_max_norm_mm"])
                for row in complete
                if _number_or_none(row.get("recovery_error_max_norm_mm")) is not None
            ],
            dtype=np.float64,
        )
        summary.append(
            {
                "magnitude_mm": magnitude,
                "planned_trials": len(rows),
                "completed_trials": len(complete),
                "successful_trials": len(successful),
                "capture_fraction": float(len(successful) / len(rows)),
                "recovery_error_max_norm_mm_median": float(np.median(errors)) if errors.size else None,
                "recovery_error_max_norm_mm_max": float(np.max(errors)) if errors.size else None,
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def write_capture_scan_artifacts(
    output_dir: str | Path,
    points: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    capture_tolerance_mm: float,
) -> dict[str, object]:
    """Write JSON/CSV/PNG summaries for the physical capture scan."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    by_magnitude = summarise_by_magnitude(points)
    payload = {
        "method": "physical_refit_capture_range_scan",
        "physical_geometry_repropagation_required": True,
        "q_over_p_mode": 0,
        "capture_tolerance_mm": float(capture_tolerance_mm),
        "points": list(points),
        "by_magnitude": by_magnitude,
    }
    (directory / "capture_scan_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(directory / "capture_scan_points.csv", points)
    _write_csv(directory / "capture_scan_station_pair_diagnostics.csv", diagnostics)

    magnitudes = np.asarray([row["magnitude_mm"] for row in by_magnitude], dtype=np.float64)
    fractions = np.asarray([row["capture_fraction"] for row in by_magnitude], dtype=np.float64)
    median_errors = np.asarray(
        [
            np.nan if row["recovery_error_max_norm_mm_median"] is None
            else row["recovery_error_max_norm_mm_median"]
            for row in by_magnitude
        ],
        dtype=np.float64,
    )
    figure, (capture_axis, error_axis) = plt.subplots(
        2, 1, figsize=(7.0, 7.0), sharex=True, constrained_layout=True
    )
    capture_axis.plot(magnitudes, fractions, marker="o", color="tab:blue")
    capture_axis.set_ylabel("capture fraction")
    capture_axis.set_ylim(-0.05, 1.05)
    capture_axis.grid(True, alpha=0.3)
    plot_floor = max(float(capture_tolerance_mm) * 1.0e-3, 1.0e-15)
    plotted_errors = np.where(
        np.isfinite(median_errors), np.maximum(median_errors, plot_floor), np.nan
    )
    error_axis.plot(magnitudes, plotted_errors, marker="o", color="tab:red")
    error_axis.axhline(
        capture_tolerance_mm,
        color="0.35",
        linewidth=1.0,
        linestyle="--",
        label="capture tolerance",
    )
    error_axis.set_ylabel("median max recovery error [mm]")
    error_axis.set_xlabel("injected station-offset magnitude [mm]")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()
    if np.any(magnitudes > 0.0):
        maximum_magnitude = float(np.max(magnitudes))
        capture_axis.set_xlim(0.0, maximum_magnitude * 1.08)
        error_axis.set_xscale("symlog", linthresh=0.1)
        # Failed high-offset points otherwise hide the sub-millimetre capture
        # region that this scan is intended to resolve.  Exact-zero control
        # values use the plotting floor only; JSON and CSV retain zero.
        error_axis.set_yscale("log")
    figure.savefig(directory / "capture_scan.png", dpi=170)
    plt.close(figure)
    return payload
