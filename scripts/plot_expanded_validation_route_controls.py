#!/usr/bin/env python3
"""Plot completed expanded-corpus route controls without opening a test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "expanded_validation_route_controls.yaml"
METRICS = (
    ("pooled_complete_track_efficiency", "Complete-track efficiency", 0.70),
    ("pooled_complete_track_purity", "Complete-track purity", 0.95),
    ("pooled_track_fake_rate", "Track fake rate", 0.05),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("plot configuration must be a mapping")
    settings = payload.get("expanded_validation_route_controls")
    if not isinstance(settings, Mapping):
        raise ValueError("configuration lacks expanded_validation_route_controls")
    return settings


def _resolve(project_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("summary_csv must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path)


def _assert_validation_contract(summary_csv: Path) -> dict[str, object]:
    contract_paths = (
        summary_csv.parent / "validation_run_contract.json",
        summary_csv.parent / "validation_control_contract.json",
    )
    contract_path = next((path for path in contract_paths if path.is_file()), None)
    if contract_path is None:
        raise FileNotFoundError(
            "validation summary has no recognized sealed-test contract: "
            + ", ".join(str(path) for path in contract_paths)
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"validation contract is malformed: {contract_path}")
    if payload.get("test_events_loaded") is not False:
        raise ValueError(f"validation summary is not sealed from test events: {summary_csv}")
    if payload.get("test_artifacts_opened") not in (False, None):
        raise ValueError(f"validation summary opened test artifacts: {summary_csv}")
    return {
        "contract_present": True,
        "contract_path": str(contract_path),
        "loaded_event_splits": payload.get("loaded_event_splits"),
        "test_events_loaded": payload.get("test_events_loaded"),
        "test_artifacts_opened": payload.get("test_artifacts_opened"),
    }


def _float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def _load_method(
    item: Mapping[str, Any], magnitudes: Sequence[float]
) -> tuple[str, list[dict[str, float]], dict[str, object]]:
    label = item.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("every plot method requires a non-empty label")
    summary_csv = _resolve(PROJECT_ROOT, item.get("summary_csv"))
    if not summary_csv.is_file():
        raise FileNotFoundError(summary_csv)
    requested_stream = item.get("score_stream")
    if requested_stream is not None and not isinstance(requested_stream, str):
        raise ValueError(f"score_stream for {label} must be a string or null")
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    streams = {str(row.get("score_stream") or "") for row in rows}
    stream = str(requested_stream) if requested_stream is not None else ""
    if requested_stream is None and len(streams) != 1:
        raise ValueError(f"{label} has multiple score streams; declare score_stream explicitly")
    selected = [row for row in rows if str(row.get("score_stream") or "") == stream]
    by_magnitude = {_float(row, "magnitude_mm"): row for row in selected}
    if set(by_magnitude) != set(float(value) for value in magnitudes):
        raise ValueError(f"{label} does not cover the declared validation magnitudes")
    values: list[dict[str, float]] = []
    for magnitude in magnitudes:
        row = by_magnitude[float(magnitude)]
        values.append(
            {
                "magnitude_mm": float(magnitude),
                "candidate_truth_chain_recall": _float(
                    row, "pooled_candidate_complete_truth_chain_recall"
                ),
                "threshold_truth_chain_recall": _float(
                    row, "pooled_score_threshold_complete_truth_chain_recall"
                ),
                "complete_track_efficiency": _float(row, "pooled_complete_track_efficiency"),
                "complete_track_purity": _float(row, "pooled_complete_track_purity"),
                "track_fake_rate": _float(row, "pooled_track_fake_rate"),
                "capture_fraction": _float(row, "capture_fraction"),
            }
        )
    provenance = {
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": _sha256(summary_csv),
        "score_stream": stream,
        "validation_contract": _assert_validation_contract(summary_csv),
    }
    return label, values, provenance


def _write_summary_csv(path: Path, methods: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    fields = ["method", "magnitude_mm", "candidate_truth_chain_recall", "threshold_truth_chain_recall",
              "complete_track_efficiency", "complete_track_purity", "track_fake_rate", "capture_fraction"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, rows in methods.items():
            for row in rows:
                writer.writerow({"method": label, **row})


def _plot(path: Path, magnitudes: Sequence[float], methods: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    positions = np.arange(len(magnitudes), dtype=np.float64)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 5.0))
    colors = plt.get_cmap("tab10").colors
    for axis, (field, title, requirement) in zip(axes, METRICS):
        for index, (label, rows) in enumerate(methods.items()):
            axis.plot(
                positions,
                [float(row[field.removeprefix("pooled_")]) for row in rows],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=colors[index % len(colors)],
                label=label,
            )
        axis.axhline(requirement, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xticks(positions, [f"{value:g}" for value in magnitudes])
        axis.set_xlabel("Injected |dx, dy| scale [mm]")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(bottom=0.27, left=0.06, right=0.99, top=0.91, wspace=0.10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    settings = _read_yaml(config_path)
    raw_magnitudes = settings.get("magnitudes_mm")
    raw_methods = settings.get("methods")
    if not isinstance(raw_magnitudes, list) or not raw_magnitudes:
        raise ValueError("configuration requires non-empty magnitudes_mm")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("configuration requires non-empty methods")
    magnitudes = tuple(float(value) for value in raw_magnitudes)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty plot output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, list[dict[str, float]]] = {}
    provenance: dict[str, object] = {}
    for item in raw_methods:
        if not isinstance(item, Mapping):
            raise ValueError("every method entry must be a mapping")
        label, rows, metadata = _load_method(item, magnitudes)
        if label in methods:
            raise ValueError(f"duplicate plot method label: {label}")
        methods[label] = rows
        provenance[label] = metadata
    _write_summary_csv(output_dir / "route_validation_control_summary.csv", methods)
    _plot(output_dir / "route_validation_control_comparison.png", magnitudes, methods)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "faser-expanded-validation-route-controls-v1",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "magnitudes_mm": list(magnitudes),
                "methods": methods,
                "provenance": provenance,
                "test_events_loaded": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
