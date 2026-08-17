#!/usr/bin/env python3
"""Summarize frozen IFT R_y validation controls without opening event data.

This is deliberately a post-selection reader.  It consumes only artifacts
produced by the MLP, V1, and V2 validation controls, rejects any artifact that
opened test data, and neither calibrates scores nor searches route controls.
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
V1_ABLATIONS = (
    ("v1_full_context", "geometry_aware_full_context"),
    ("v1_no_multistation_context", "geometry_aware_no_multistation_context"),
)
V2_STREAMS = (
    ("v2_bce_route_query", "route_aware_v2"),
    ("v2_same_checkpoint_base_edge", "same_checkpoint_base_edge_control"),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_sealed(contract: Mapping[str, object], label: str) -> None:
    if contract.get("test_events_loaded") is not False:
        raise ValueError(f"{label} is not sealed from test events")
    if contract.get("test_artifacts_opened") not in (False, None):
        raise ValueError(f"{label} opened test artifacts")
    if contract.get("test_opened") not in (False, None):
        raise ValueError(f"{label} opened test data")
    if contract.get("condition_axis") != "ift_ry_mrad":
        raise ValueError(f"{label} is not an explicit IFT R_y study")


def _route_from_mapping(route: Mapping[str, object]) -> dict[str, object]:
    return {metric: route.get(metric) for metric in METRICS}


def _capture(route: Mapping[str, object], criteria: Mapping[str, object]) -> bool:
    try:
        return bool(
            float(route["complete_track_efficiency"])
            >= float(criteria["minimum_complete_track_efficiency"])
            and float(route["complete_track_purity"])
            >= float(criteria["minimum_complete_track_purity"])
            and float(route["track_fake_rate"])
            <= float(criteria["maximum_track_fake_rate"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _criteria(payload: Mapping[str, object], label: str) -> dict[str, float]:
    raw = payload.get("capture_success_criteria")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} lacks capture-success criteria")
    required = (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    )
    if any(key not in raw for key in required):
        raise ValueError(f"{label} has incomplete capture-success criteria")
    return {key: float(raw[key]) for key in required}


def _row(
    model: str,
    magnitude: float,
    route: Mapping[str, object],
    criteria: Mapping[str, object],
) -> dict[str, object]:
    return {
        "model": model,
        "condition_axis": "ift_ry_mrad",
        "condition_magnitude": float(magnitude),
        "capture_success": _capture(route, criteria),
        **_route_from_mapping(route),
    }


def _load_mlp(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    contract = _read_json(root / "validation_control_contract.json")
    _assert_sealed(contract, "MLP route control")
    if contract.get("loaded_event_splits") != ["validation"]:
        raise ValueError("MLP route control must open validation only")
    selected = _read_json(root / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError("MLP route selection is not validation-only")
    criteria = _criteria(selected, "MLP route control")
    results = _read_json(root / "validation_results.json")
    raw_rows = results.get("route_magnitude_summary")
    if not isinstance(raw_rows, list):
        raise ValueError("MLP route control lacks magnitude summary")
    rows = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("MLP route magnitude row is malformed")
        if raw.get("condition_axis") != "ift_ry_mrad":
            raise ValueError("MLP route magnitude row has a wrong condition axis")
        route = {metric: raw.get(f"pooled_{metric}") for metric in METRICS}
        rows.append(_row("mlp_pairwise_route", float(raw["condition_magnitude"]), route, criteria))
    return rows, {"contract": contract, "criteria": criteria}


def _load_v1(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    contract = _read_json(root / "validation_run_contract.json")
    _assert_sealed(contract, "V1 validation control")
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V1 validation control must open train and validation only")
    rows: list[dict[str, object]] = []
    selection_metadata: dict[str, object] = {}
    for model, ablation in V1_ABLATIONS:
        selected = _read_json(root / ablation / "validation_selected_operating_point.json")
        if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
            raise ValueError(f"V1 {ablation} route selection is not validation-only")
        criteria = _criteria(selected, f"V1 {ablation}")
        nested = selected.get("validation_route_selection")
        if not isinstance(nested, Mapping):
            raise ValueError(f"V1 {ablation} has no route-selection result")
        evaluations = nested.get("evaluation_by_magnitude")
        if not isinstance(evaluations, Mapping):
            raise ValueError(f"V1 {ablation} has no magnitude evaluations")
        for raw_magnitude, evaluation in evaluations.items():
            if not isinstance(evaluation, Mapping):
                raise ValueError(f"V1 {ablation} has a malformed magnitude evaluation")
            route = evaluation.get("route")
            if not isinstance(route, Mapping):
                raise ValueError(f"V1 {ablation} magnitude evaluation lacks route metrics")
            rows.append(_row(model, float(raw_magnitude), route, criteria))
        selection_metadata[model] = {
            "criteria": criteria,
            "thresholds": selected.get("thresholds"),
            "unmatched_penalty": selected.get("unmatched_penalty"),
        }
    return rows, {"contract": contract, "selection": selection_metadata}


def _load_v2(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    contract = _read_json(root / "validation_run_contract.json")
    _assert_sealed(contract, "V2 validation control")
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V2 validation control must open train and validation only")
    selected = _read_json(root / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError("V2 route selection is not validation-only")
    criteria = _criteria(selected, "V2 validation control")
    results = _read_json(root / "validation_results.json")
    streams = results.get("score_streams")
    if not isinstance(streams, Mapping):
        raise ValueError("V2 validation results have no score streams")
    rows: list[dict[str, object]] = []
    for model, stream_name in V2_STREAMS:
        stream = streams.get(stream_name)
        if not isinstance(stream, Mapping):
            raise ValueError(f"V2 validation results lack stream '{stream_name}'")
        summaries = stream.get("route_magnitude_summary")
        if not isinstance(summaries, list):
            raise ValueError(f"V2 stream '{stream_name}' has no magnitude summaries")
        for raw in summaries:
            if not isinstance(raw, Mapping):
                raise ValueError(f"V2 stream '{stream_name}' has a malformed magnitude row")
            if raw.get("condition_axis") != "ift_ry_mrad":
                raise ValueError(f"V2 stream '{stream_name}' has a wrong condition axis")
            route = {metric: raw.get(f"pooled_{metric}") for metric in METRICS}
            rows.append(_row(model, float(raw["condition_magnitude"]), route, criteria))
    return rows, {"contract": contract, "criteria": criteria}


def _validate_grid(rows: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
    by_model: dict[str, set[float]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), set()).add(float(row["condition_magnitude"]))
    grids = {tuple(sorted(values)) for values in by_model.values()}
    if len(grids) != 1:
        detail = {model: sorted(values) for model, values in by_model.items()}
        raise ValueError(f"controls do not share one IFT R_y magnitude grid: {detail}")
    return next(iter(grids))


def _plot(path: Path, rows: Sequence[Mapping[str, object]], magnitudes: Sequence[float]) -> None:
    metrics = (
        (
            "candidate_complete_truth_chain_recall",
            "Candidate truth-chain retention",
            (0.90, 1.005),
        ),
        ("complete_track_efficiency", "Complete-track efficiency", (0.90, 1.005)),
        ("complete_track_purity", "Complete-track purity", (0.94, 1.005)),
        ("track_fake_rate", "Track fake rate", (-0.002, 0.08)),
    )
    models = sorted({str(row["model"]) for row in rows})
    by_model = {
        model: {float(row["condition_magnitude"]): row for row in rows if row["model"] == model}
        for model in models
    }
    figure, axes = plt.subplots(1, len(metrics), figsize=(18, 4.4), constrained_layout=True)
    x = np.arange(len(magnitudes), dtype=np.float64)
    for axis, (metric, title, limits) in zip(axes, metrics):
        for model in models:
            values = [by_model[model][magnitude].get(metric) for magnitude in magnitudes]
            axis.plot(
                x,
                [np.nan if value is None else float(value) for value in values],
                marker="o",
                linewidth=1.5,
                label=model,
            )
        axis.set_title(title)
        axis.set_xlabel("Injected |IFT R_y| [mrad]")
        axis.set_xticks(x, [f"{magnitude:g}" for magnitude in magnitudes])
        axis.set_ylim(*limits)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation metric")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-output", required=True)
    parser.add_argument("--v1-output", required=True)
    parser.add_argument("--v2-output", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty validation assessment directory")
    output_root.mkdir(parents=True, exist_ok=True)

    mlp_rows, mlp_metadata = _load_mlp(Path(args.mlp_output).expanduser().resolve())
    v1_rows, v1_metadata = _load_v1(Path(args.v1_output).expanduser().resolve())
    v2_rows, v2_metadata = _load_v2(Path(args.v2_output).expanduser().resolve())
    rows = [*mlp_rows, *v1_rows, *v2_rows]
    magnitudes = _validate_grid(rows)
    _write_csv(output_root / "ift_ry_validation_route_controls.csv", rows)
    _plot(output_root / "ift_ry_validation_route_controls.png", rows, magnitudes)
    _write_json(
        output_root / "summary.json",
        {
            "schema_version": "faser-ift-ry-validation-controls-assessment-v1",
            "event_data_opened": False,
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "condition_axis": "ift_ry_mrad",
            "condition_magnitudes": list(magnitudes),
            "models": sorted({str(row["model"]) for row in rows}),
            "rows": rows,
            "artifacts": {"mlp": mlp_metadata, "v1": v1_metadata, "v2": v2_metadata},
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "condition_magnitudes": list(magnitudes),
                "models": sorted({str(row["model"]) for row in rows}),
                "test_events_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
