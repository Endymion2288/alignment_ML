#!/usr/bin/env python3
"""Compare V2 route-aware validation artifacts without opening any event data.

The program consumes only validation-produced JSON/CSV artifacts.  It is a
post-selection report: it neither trains, calibrates, searches thresholds, nor
opens a physical manifest or a sealed test source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "candidate_complete_truth_chain_recall",
    "score_threshold_complete_truth_chain_recall",
    "complete_track_efficiency",
    "complete_track_purity",
    "track_fake_rate",
    "missing_station_recovery",
)
MAGNITUDES = (0.0, 0.1, 1.0, 5.0, 10.0, 50.0)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return dict(payload)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _magnitude_key(value: object) -> float:
    result = float(value)
    if result not in MAGNITUDES:
        raise ValueError(f"unexpected physical curriculum magnitude {result:g} mm")
    return result


def _capture(route: Mapping[str, object], criteria: Mapping[str, object]) -> bool:
    return (
        route.get("complete_track_efficiency") is not None
        and route.get("complete_track_purity") is not None
        and route.get("track_fake_rate") is not None
        and float(route["complete_track_efficiency"])
        >= float(criteria["minimum_complete_track_efficiency"])
        and float(route["complete_track_purity"])
        >= float(criteria["minimum_complete_track_purity"])
        and float(route["track_fake_rate"])
        <= float(criteria["maximum_track_fake_rate"])
    )


def _canonical_route(route: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in METRICS:
        result[metric] = route.get(metric)
    return result


def _load_v2_streams(
    v2_dir: Path,
) -> tuple[dict[str, dict[float, dict[str, object]]], dict[str, Any], dict[str, object]]:
    contract = _read_json(v2_dir / "validation_run_contract.json")
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V2 artifact was not produced from exactly train and validation")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError("V2 artifact violates the sealed-test boundary")
    if contract.get("same_control_calibration_refit") is not False:
        raise ValueError("same-checkpoint V2 control unexpectedly refit calibration")
    if contract.get("same_control_threshold_selection") is not False:
        raise ValueError("same-checkpoint V2 control unexpectedly selected thresholds")
    operating_point = _read_json(v2_dir / "validation_selected_operating_point.json")
    if operating_point.get("selection_split") != "validation_only" or operating_point.get("test_opened") is not False:
        raise ValueError("V2 route operating point is not validation-only")
    criteria = operating_point.get("capture_success_criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError("V2 operating point lacks capture criteria")
    required_criteria = (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    )
    if any(key not in criteria for key in required_criteria):
        raise ValueError("V2 capture criteria are incomplete")
    results = _read_json(v2_dir / "validation_results.json")
    streams = results.get("score_streams")
    if not isinstance(streams, Mapping):
        raise ValueError("V2 results have no score streams")
    result: dict[str, dict[float, dict[str, object]]] = {}
    for label in ("route_aware_v2", "same_checkpoint_base_edge_control"):
        stream = streams.get(label)
        if not isinstance(stream, Mapping):
            raise ValueError(f"V2 results lack '{label}' stream")
        rows = stream.get("route_magnitude_summary")
        if not isinstance(rows, list):
            raise ValueError(f"V2 stream '{label}' lacks magnitude summaries")
        by_magnitude: dict[float, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("V2 magnitude summary row is invalid")
            magnitude = _magnitude_key(row["magnitude_mm"])
            route = {
                metric: row.get(f"pooled_{metric}")
                for metric in METRICS
            }
            by_magnitude[magnitude] = route
        if tuple(sorted(by_magnitude)) != MAGNITUDES:
            raise ValueError(f"V2 stream '{label}' lacks a physical magnitude")
        result[label] = by_magnitude
    return result, contract, dict(criteria)


def _load_v1(
    v1_dir: Path, variant: str
) -> tuple[dict[float, dict[str, object]], dict[str, Any]]:
    payload = _read_json(v1_dir / variant / "validation_selected_operating_point.json")
    if payload.get("selection_split") != "validation_only" or payload.get("test_opened") is not False:
        raise ValueError(f"V1 '{variant}' operating point is not validation-only")
    nested = payload.get("validation_route_selection")
    if not isinstance(nested, Mapping):
        raise ValueError(f"V1 '{variant}' has no route selection")
    evaluations = nested.get("evaluation_by_magnitude")
    if not isinstance(evaluations, Mapping):
        raise ValueError(f"V1 '{variant}' has no magnitude evaluations")
    result: dict[float, dict[str, object]] = {}
    for raw_magnitude, evaluation in evaluations.items():
        if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("route"), Mapping):
            raise ValueError(f"V1 '{variant}' route evaluation is invalid")
        result[_magnitude_key(raw_magnitude)] = _canonical_route(evaluation["route"])
    if tuple(sorted(result)) != MAGNITUDES:
        raise ValueError(f"V1 '{variant}' lacks a physical magnitude")
    return result, payload


def _load_mlp(mlp_dir: Path) -> tuple[dict[float, dict[str, object]], dict[str, Any]]:
    contract = _read_json(mlp_dir / "validation_control_contract.json")
    if contract.get("loaded_event_splits") != ["validation"]:
        raise ValueError("MLP route control opened a non-validation split")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError("MLP route control violates the sealed-test boundary")
    selected = _read_json(mlp_dir / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError("MLP route operating point is not validation-only")
    evaluations = selected.get("evaluation_by_magnitude")
    if not isinstance(evaluations, Mapping):
        raise ValueError("MLP route control has no magnitude evaluations")
    result: dict[float, dict[str, object]] = {}
    for raw_magnitude, evaluation in evaluations.items():
        if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("route"), Mapping):
            raise ValueError("MLP route evaluation is invalid")
        result[_magnitude_key(raw_magnitude)] = _canonical_route(evaluation["route"])
    if tuple(sorted(result)) != MAGNITUDES:
        raise ValueError("MLP route control lacks a physical magnitude")
    return result, contract


def _load_direct_route_v2(
    direct_dir: Path,
) -> tuple[dict[str, dict[float, dict[str, object]]], dict[str, Any], dict[str, object]]:
    contract = _read_json(direct_dir / "validation_run_contract.json")
    if contract.get("loaded_event_splits") != ["validation"]:
        raise ValueError("direct V2 output opened a non-validation split")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError("direct V2 output violates the sealed-test boundary")
    if contract.get("edge_calibration_refit") is not False or contract.get("edge_threshold_selection_refit") is not False:
        raise ValueError("direct V2 unexpectedly refit frozen edge controls")
    if contract.get("same_checkpoint_control_edge_calibration_refit") is not False:
        raise ValueError("direct V2 same-checkpoint control refit edge calibration")
    if contract.get("same_checkpoint_control_edge_threshold_selection_refit") is not False:
        raise ValueError("direct V2 same-checkpoint control selected edge thresholds")
    selected = _read_json(direct_dir / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError("direct V2 operating point is not validation-only")
    criteria = selected.get("capture_success_criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError("direct V2 operating point lacks capture criteria")
    results = _read_json(direct_dir / "validation_results.json")
    raw_streams = results.get("score_streams")
    if not isinstance(raw_streams, Mapping):
        raise ValueError("direct V2 results have no score streams")
    expected = ("v2_direct_route_query", "same_checkpoint_edge_route_control")
    streams: dict[str, dict[float, dict[str, object]]] = {}
    for label in expected:
        stream = raw_streams.get(label)
        if not isinstance(stream, Mapping):
            raise ValueError(f"direct V2 results lack '{label}'")
        rows = stream.get("route_magnitude_summary")
        if not isinstance(rows, list):
            raise ValueError(f"direct V2 stream '{label}' lacks magnitude summaries")
        by_magnitude: dict[float, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("direct V2 magnitude row is invalid")
            magnitude = _magnitude_key(row["magnitude_mm"])
            by_magnitude[magnitude] = {
                metric: row.get(f"pooled_{metric}") for metric in METRICS
            }
        if tuple(sorted(by_magnitude)) != MAGNITUDES:
            raise ValueError(f"direct V2 stream '{label}' lacks a physical magnitude")
        streams[label] = by_magnitude
    return streams, contract, dict(criteria)


def _comparison_row(
    model: str,
    magnitude: float,
    route: Mapping[str, object],
    criteria: Mapping[str, object],
) -> dict[str, object]:
    return {
        "model": model,
        "magnitude_mm": magnitude,
        "capture_success": _capture(route, criteria),
        **dict(route),
    }


def _relative_gate(
    v2: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    efficiency_gain: float,
) -> dict[str, object]:
    if any(
        v2.get(metric) is None or baseline.get(metric) is None
        for metric in ("complete_track_efficiency", "complete_track_purity", "track_fake_rate")
    ):
        return {
            "efficiency_delta": None,
            "purity_delta": None,
            "fake_rate_delta": None,
            "clearly_better": False,
        }
    eff_delta = float(v2["complete_track_efficiency"]) - float(baseline["complete_track_efficiency"])
    purity_delta = float(v2["complete_track_purity"]) - float(baseline["complete_track_purity"])
    fake_delta = float(v2["track_fake_rate"]) - float(baseline["track_fake_rate"])
    return {
        "efficiency_delta": eff_delta,
        "purity_delta": purity_delta,
        "fake_rate_delta": fake_delta,
        # This is a predeclared report criterion, not a further model/control
        # search: demand a material efficiency gain without a purity/fake loss.
        "clearly_better": bool(
            eff_delta >= efficiency_gain and purity_delta >= 0.0 and fake_delta <= 0.0
        ),
    }


def _plot_comparison(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Plot validation metrics only after all controls have been frozen."""
    labels = (
        "mlp_pairwise_route",
        "v1_full_context",
        "v1_no_multistation_context",
        "v2_same_checkpoint_base_edge_control",
        "v2_edge_projection",
        "v2_direct_same_checkpoint_edge_control",
        "v2_direct_route_query",
    )
    titles = (
        ("complete_track_efficiency", "Complete-track efficiency"),
        ("complete_track_purity", "Complete-track purity"),
        ("track_fake_rate", "Track fake rate"),
    )
    by_label: dict[str, dict[float, Mapping[str, object]]] = {label: {} for label in labels}
    for row in rows:
        label = str(row["model"])
        if label in by_label:
            by_label[label][_magnitude_key(row["magnitude_mm"])] = row
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(MAGNITUDES), dtype=np.float64)
    for axis, (metric, title) in zip(axes, titles):
        for label in labels:
            values = [by_label[label][magnitude].get(metric) for magnitude in MAGNITUDES]
            numeric = [np.nan if value is None else float(value) for value in values]
            axis.plot(x, numeric, marker="o", linewidth=1.6, label=label)
        axis.set_title(title)
        axis.set_xlabel("Injected misalignment [mm]")
        axis.set_xticks(x, [f"{value:g}" for value in MAGNITUDES])
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation metric")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-output", required=True)
    parser.add_argument("--direct-route-output", required=True)
    parser.add_argument("--mlp-route-output", required=True)
    parser.add_argument("--v1-validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-efficiency-gain", type=float, default=0.03)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_efficiency_gain) or args.minimum_efficiency_gain < 0.0:
        raise ValueError("minimum efficiency gain must be finite and non-negative")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty V2 validation assessment")
    output_root.mkdir(parents=True, exist_ok=True)
    v2_streams, v2_contract, criteria = _load_v2_streams(
        Path(args.v2_output).expanduser().resolve()
    )
    direct_streams, direct_contract, direct_criteria = _load_direct_route_v2(
        Path(args.direct_route_output).expanduser().resolve()
    )
    if direct_criteria != criteria:
        raise ValueError("direct V2 and edge-projection V2 use different primary criteria")
    v1_full, v1_full_payload = _load_v1(
        Path(args.v1_validation_dir).expanduser().resolve(), "geometry_aware_transformer"
    )
    v1_no_context, v1_no_context_payload = _load_v1(
        Path(args.v1_validation_dir).expanduser().resolve(), "geometry_aware_no_multistation_context"
    )
    mlp, mlp_contract = _load_mlp(Path(args.mlp_route_output).expanduser().resolve())
    streams: dict[str, Mapping[float, Mapping[str, object]]] = {
        "mlp_pairwise_route": mlp,
        "v1_full_context": v1_full,
        "v1_no_multistation_context": v1_no_context,
        "v2_same_checkpoint_base_edge_control": v2_streams["same_checkpoint_base_edge_control"],
        "v2_edge_projection": v2_streams["route_aware_v2"],
        "v2_direct_route_query": direct_streams["v2_direct_route_query"],
        "v2_direct_same_checkpoint_edge_control": direct_streams[
            "same_checkpoint_edge_route_control"
        ],
    }
    rows = [
        _comparison_row(label, magnitude, values[magnitude], criteria)
        for label, values in streams.items()
        for magnitude in MAGNITUDES
    ]
    route_aware = direct_streams["v2_direct_route_query"]
    comparisons: dict[str, object] = {}
    for magnitude in (5.0, 10.0):
        comparisons[str(magnitude)] = {
            baseline: _relative_gate(
                route_aware[magnitude], values[magnitude], efficiency_gain=args.minimum_efficiency_gain
            )
            for baseline, values in streams.items()
            if baseline != "v2_direct_route_query"
        }
    nominal_pass = _capture(route_aware[0.0], criteria)
    required_baselines = (
        "mlp_pairwise_route",
        "v1_full_context",
        "v1_no_multistation_context",
    )
    broad_gain = all(
        bool(comparisons[str(magnitude)][baseline]["clearly_better"])
        for magnitude in (5.0, 10.0)
        for baseline in required_baselines
    )
    same_checkpoint_gain = all(
        bool(
            comparisons[str(magnitude)]["v2_direct_same_checkpoint_edge_control"]["clearly_better"]
        )
        for magnitude in (5.0, 10.0)
    )
    _write_csv(output_root / "validation_route_comparison.csv", rows)
    _plot_comparison(output_root / "validation_route_comparison.png", rows)
    _write_json(
        output_root / "validation_hypothesis_assessment.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-validation-assessment",
            "event_data_opened": False,
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "criteria": criteria,
            "relative_gain_definition": {
                "magnitudes_mm": [5.0, 10.0],
                "minimum_complete_track_efficiency_gain": args.minimum_efficiency_gain,
                "minimum_complete_track_purity_delta": 0.0,
                "maximum_track_fake_rate_delta": 0.0,
            },
            "nominal_primary_point_passed": nominal_pass,
            "all_5_10mm_baseline_gains_passed": broad_gain,
            "same_checkpoint_route_context_gain_passed": same_checkpoint_gain,
            "v2_hypothesis_passed": bool(nominal_pass and broad_gain and same_checkpoint_gain),
            "relative_comparisons": comparisons,
            "artifact_contracts": {
                "v2": v2_contract,
                "direct_v2": direct_contract,
                "mlp": mlp_contract,
                "v1_full_selection": {
                    "selection_split": v1_full_payload.get("selection_split"),
                    "test_opened": v1_full_payload.get("test_opened"),
                },
                "v1_no_context_selection": {
                    "selection_split": v1_no_context_payload.get("selection_split"),
                    "test_opened": v1_no_context_payload.get("test_opened"),
                },
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "nominal_primary_point_passed": nominal_pass,
                "all_5_10mm_baseline_gains_passed": broad_gain,
                "same_checkpoint_route_context_gain_passed": same_checkpoint_gain,
                "v2_hypothesis_passed": bool(nominal_pass and broad_gain and same_checkpoint_gain),
                "test_events_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
