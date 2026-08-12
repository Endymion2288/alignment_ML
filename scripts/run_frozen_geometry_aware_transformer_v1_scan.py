#!/usr/bin/env python3
"""Evaluate validation-frozen Transformer V1 ablations on sealed test refits.

This program is test-only.  It neither trains nor calibrates a network, does
not scan a threshold, and does not choose a geometry-bias setting.  It uses the
unchanged adjacent unit-capacity route assignment backend and reports the same
multi-direction metrics as the frozen MLP route baseline.
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

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base
from scripts.run_frozen_route_level_scan import (
    _audit_sealed_samples,
    _capture_success,
    _configured_test_sources,
    _direction_summary,
    _json_value,
    _parse_thresholds,
    _sample_groups,
    _select_magnitude,
    _sha256,
    _station_pair_rows,
    _trial_row,
    _write_csv,
    _write_json,
)
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    candidate_score_metrics,
    graph_bundle_summary,
    load_transformer_artifact,
    predict_transformer_scores,
)
from training.route_assignment import evaluate_adjacent_route_assignment_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "frozen_geometry_aware_transformer_v1_multidirection_test.yaml"


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {resolved}")
    return resolved, dict(payload)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    supplied = load_yaml_with_base(path)
    root = supplied.get("physical_curriculum_mlp", supplied)
    transformer = supplied.get("geometry_aware_transformer_v1")
    frozen = supplied.get("frozen_geometry_aware_transformer_v1")
    if not all(isinstance(value, Mapping) for value in (root, transformer, frozen)):
        raise ValueError(
            "configuration requires physical_curriculum_mlp, geometry_aware_transformer_v1, "
            "and frozen_geometry_aware_transformer_v1"
        )
    return dict(root), dict(transformer), dict(frozen)


def _frozen_artifacts(
    ablation_root: Path,
    frozen_config: Mapping[str, Any],
    station_path: tuple[int, ...],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[tuple[int, int], float], float]:
    checkpoint = (ablation_root / "checkpoint.pt").resolve()
    model_selection_path = (ablation_root / "model_selection.json").resolve()
    calibration_path = (ablation_root / "calibration.json").resolve()
    operating_path = (ablation_root / "validation_selected_operating_point.json").resolve()
    for path in (checkpoint, model_selection_path, calibration_path, operating_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, model_selection = _read_json(model_selection_path)
    _, calibration_wrapper = _read_json(calibration_path)
    _, selected = _read_json(operating_path)
    contract = frozen_config.get("frozen_contract", {})
    if not isinstance(contract, Mapping):
        raise ValueError("frozen_contract must be a mapping")
    if bool(contract.get("require_validation_only_model_selection", True)):
        if model_selection.get("selection_split") != "validation_only" or model_selection.get("test_opened") is not False:
            raise ValueError(f"{ablation_root.name} model selection was not validation-only sealed")
        if model_selection.get("selected_checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError(f"{ablation_root.name} selected checkpoint differs from its validation contract")
    if bool(contract.get("require_validation_only_calibration", True)):
        if calibration_wrapper.get("fit_split") != "validation_only" or calibration_wrapper.get("test_opened") is not False:
            raise ValueError(f"{ablation_root.name} calibration was not validation-only sealed")
    calibration = calibration_wrapper.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("fit_split") != "validation_only":
        raise ValueError(f"{ablation_root.name} calibration payload is invalid")
    if bool(contract.get("require_validation_only_operating_point", True)):
        if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
            raise ValueError(f"{ablation_root.name} route operating point was not validation-only sealed")
    required_method = str(contract.get("required_assignment_method"))
    if selected.get("method") != required_method:
        raise ValueError(f"{ablation_root.name} route method differs from the frozen contract")
    thresholds = _parse_thresholds(selected, station_path)
    penalty = float(selected["unmatched_penalty"])
    if not math.isfinite(penalty):
        raise ValueError(f"{ablation_root.name} has a non-finite unmatched penalty")
    return checkpoint, model_selection, dict(calibration), selected, thresholds, penalty


def _model_directories(validation_root: Path, requested: Sequence[str] | None) -> list[Path]:
    if not validation_root.is_dir():
        raise FileNotFoundError(validation_root)
    names = (
        list(requested)
        if requested is not None
        else sorted(
            entry.name
            for entry in validation_root.iterdir()
            if entry.is_dir() and (entry / "checkpoint.pt").is_file()
        )
    )
    if not names:
        raise ValueError("no frozen Transformer ablation directories were selected")
    result = [(validation_root / name).resolve() for name in names]
    if any(not path.is_dir() for path in result):
        raise FileNotFoundError("requested Transformer ablation directory is absent")
    return result


def _plot_comparison(path: Path, rows: Sequence[Mapping[str, object]], metric: str, ylabel: str) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for model, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: float(row["magnitude_mm"]))
        x = np.asarray([float(row["magnitude_mm"]) for row in ordered], dtype=np.float64)
        y = np.asarray([float(row[metric]) for row in ordered], dtype=np.float64)
        axis.plot(x, y, marker="o", label=model)
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("Injected station misalignment magnitude [mm]")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _read_mlp_reference(
    transformer_config: Mapping[str, Any], test_manifest: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw = transformer_config.get("mlp_baseline_reference")
    if not isinstance(raw, Mapping):
        raise ValueError("Transformer configuration has no MLP baseline reference")
    baseline_root = (PROJECT_ROOT / str(raw["frozen_route_test"])).resolve()
    contract_path = baseline_root / "frozen_evaluation_contract.json"
    summary_path = baseline_root / "magnitude_summary.csv"
    if not contract_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("frozen MLP route reference is incomplete")
    _, contract = _read_json(contract_path)
    if contract.get("synthetic_manifest_sha256") != _sha256(test_manifest):
        raise ValueError("frozen MLP reference did not use the same controlled multi-direction test manifest")
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["model"] = "mlp_baseline"
    return rows, {
        "directory": str(baseline_root),
        "frozen_contract": str(contract_path),
        "frozen_contract_sha256": _sha256(contract_path),
        "magnitude_summary": str(summary_path),
        "same_test_manifest": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--validation-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ablation", action="append", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    root, transformer, frozen = _load_config(config_path)
    if not isinstance(root.get("refit"), Mapping) or int(root["refit"].get("q_over_p_mode", -1)) != 0:
        raise ValueError("frozen Transformer scan requires physical mode-0 propagation")
    if frozen.get("evaluation_split") != "test" or frozen.get("candidate_chi2_gate") is not None:
        raise ValueError("frozen Transformer scan is reserved for the ungated sealed test split")
    station_path = tuple(int(value) for value in frozen.get("station_path", (0, 1, 2, 3)))
    if adjacent_station_pairs(station_path) != ADJACENT_STATION_PAIRS:
        raise ValueError("Transformer V1 frozen scan requires IFT -> S1 -> S2 -> S3 adjacent routing")
    criteria = frozen.get("capture_success")
    if not isinstance(criteria, Mapping):
        raise ValueError("frozen Transformer scan requires capture_success criteria")
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest, require_all_splits=False
    )
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("test manifest does not certify physical mode-0 geometry repropagation")
    samples = [sample for sample in samples if sample.split == "test"]
    source_audit = _audit_sealed_samples(
        samples, "test", _configured_test_sources(root, "test")
    )
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty frozen Transformer test output")
    output_root.mkdir(parents=True, exist_ok=True)

    model_roots = _model_directories(Path(args.validation_output_dir).expanduser().resolve(), args.ablation)
    # Verify all frozen artifacts before opening the test ROOT files.
    frozen_models = []
    for model_root in model_roots:
        checkpoint, model_selection, calibration, selected, thresholds, penalty = _frozen_artifacts(
            model_root, frozen, station_path
        )
        frozen_models.append(
            (model_root, checkpoint, model_selection, calibration, selected, thresholds, penalty)
        )

    test_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    calibration_bins = int(transformer["calibration"]["bins"])
    device = str(transformer["training"]["device"])
    batch_size = int(transformer["training"]["batch_size"])
    all_summary_rows: list[dict[str, object]] = []
    all_results: dict[str, object] = {}
    model_contracts: dict[str, object] = {}
    for model_root, checkpoint, model_selection, calibration, selected, thresholds, penalty in frozen_models:
        name = model_root.name
        model, artifact = load_transformer_artifact(checkpoint, device=device)
        bundle = build_transformer_graph_bundle(test_sets, context_mode=artifact.context_mode)
        raw_scores = predict_transformer_scores(
            model,
            bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            device=device,
            batch_size=batch_size,
        )
        calibrated_scores = apply_frozen_transformer_calibration(
            bundle.adjacent_sets, raw_scores, calibration
        )
        route_assignment = RouteAssignmentConfig(
            score_threshold_by_pair=thresholds,
            unmatched_penalty=penalty,
            station_path=station_path,
            maximum_hypotheses=int(frozen.get("maximum_hypotheses", 100_000)),
        )
        groups = _sample_groups(samples, bundle.adjacent_sets, calibrated_scores)
        trial_rows: list[dict[str, object]] = []
        station_pair_rows: list[dict[str, object]] = []
        detailed_trials: list[dict[str, object]] = []
        for sample, sample_sets, sample_scores in groups:
            evaluation = evaluate_adjacent_route_assignment_sets(
                sample_sets,
                sample_scores,
                route_assignment,
                calibration_bins=calibration_bins,
            )
            success = _capture_success(evaluation["route"], criteria)
            trial_rows.append(_trial_row(sample, evaluation, success))
            station_pair_rows.extend(_station_pair_rows(sample, evaluation, success))
            detailed_trials.append(
                {
                    "source_id": sample.source_id,
                    "source_ids": list(sample.source_ids),
                    "payload_id": sample.payload_id,
                    "magnitude_mm": sample.magnitude_mm,
                    "direction_trial": sample.direction_trial,
                    "injected_offsets_xy_mm": dict(sample.injected_offsets_xy_mm),
                    "capture_success": success,
                    "evaluation": evaluation,
                }
            )
        summary_rows: list[dict[str, object]] = []
        detailed_magnitudes: list[dict[str, object]] = []
        for magnitude in sorted({float(sample.magnitude_mm) for sample in samples}):
            magnitude_samples, magnitude_sets, magnitude_scores = _select_magnitude(
                samples, bundle.adjacent_sets, calibrated_scores, magnitude
            )
            pooled = evaluate_adjacent_route_assignment_sets(
                magnitude_sets,
                magnitude_scores,
                route_assignment,
                calibration_bins=calibration_bins,
            )
            directions = [row for row in trial_rows if np.isclose(float(row["magnitude_mm"]), magnitude)]
            summary = _direction_summary(directions, pooled)
            summary_rows.append(summary)
            detailed_magnitudes.append(
                {
                    "magnitude_mm": magnitude,
                    "source_payloads": [sample.payload_id for sample in magnitude_samples],
                    "pooled_evaluation": pooled,
                    "direction_summary": summary,
                }
            )
        model_output = output_root / name
        model_output.mkdir(parents=True, exist_ok=False)
        _write_csv(model_output / "trial_metrics.csv", trial_rows)
        _write_csv(model_output / "station_pair_trial_metrics.csv", station_pair_rows)
        _write_csv(model_output / "magnitude_summary.csv", summary_rows)
        _write_json(
            model_output / "frozen_evaluation_contract.json",
            {
                "schema_version": "faser-geometry-aware-transformer-v1-frozen-route-scan",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "synthetic_manifest": str(manifest_path),
                "synthetic_manifest_sha256": _sha256(manifest_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "validation_model_selection": str(model_root / "model_selection.json"),
                "validation_model_selection_sha256": _sha256(model_root / "model_selection.json"),
                "validation_calibration": str(model_root / "calibration.json"),
                "validation_calibration_sha256": _sha256(model_root / "calibration.json"),
                "validation_operating_point": str(model_root / "validation_selected_operating_point.json"),
                "validation_operating_point_sha256": _sha256(
                    model_root / "validation_selected_operating_point.json"
                ),
                "source_audit": source_audit,
                "q_over_p_mode": 0,
                "station_path": list(station_path),
                "adjacent_station_pairs": [f"{left}->{right}" for left, right in ADJACENT_STATION_PAIRS],
                "context_mode": artifact.context_mode,
                "frozen_thresholds": {f"{left}->{right}": thresholds[(left, right)] for left, right in ADJACENT_STATION_PAIRS},
                "frozen_unmatched_penalty": penalty,
                "capture_success_criteria": dict(criteria),
                "route_solver": frozen["frozen_contract"]["route_solver"],
                "attention_edge_policy": frozen["frozen_contract"]["transformer_attention"],
                "test_time_training": False,
                "test_time_calibration": False,
                "test_time_threshold_selection": False,
            },
        )
        _write_json(
            model_output / "route_scan_results.json",
            {
                "schema_version": "faser-geometry-aware-transformer-v1-frozen-route-scan",
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "raw_candidate_score_metrics": candidate_score_metrics(
                    bundle.adjacent_sets, raw_scores, calibration_bins
                ),
                "calibrated_candidate_score_metrics": candidate_score_metrics(
                    bundle.adjacent_sets, calibrated_scores, calibration_bins
                ),
                "graph_bundle": graph_bundle_summary(bundle),
                "trial_evaluations": detailed_trials,
                "magnitude_evaluations": detailed_magnitudes,
            },
        )
        for row in summary_rows:
            all_summary_rows.append({"model": name, **row})
        all_results[name] = {
            "checkpoint": str(checkpoint),
            "context_mode": artifact.context_mode,
            "capture_fraction": {
                str(row["magnitude_mm"]): row["capture_fraction"] for row in summary_rows
            },
        }
        model_contracts[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "validation_model_selection": str(model_root / "model_selection.json"),
            "validation_calibration": str(model_root / "calibration.json"),
            "validation_operating_point": str(model_root / "validation_selected_operating_point.json"),
        }

    mlp_rows, mlp_reference = _read_mlp_reference(transformer, manifest_path)
    all_summary_rows.extend(mlp_rows)
    _write_csv(output_root / "comparison_magnitude_summary.csv", all_summary_rows)
    _plot_comparison(
        output_root / "comparison_capture_fraction_vs_misalignment.png",
        all_summary_rows,
        "capture_fraction",
        "Direction-trial capture fraction",
    )
    _plot_comparison(
        output_root / "comparison_complete_track_efficiency_vs_misalignment.png",
        all_summary_rows,
        "direction_mean_complete_track_efficiency",
        "Complete-track efficiency (direction mean)",
    )
    _write_json(
        output_root / "frozen_evaluation_contract.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v1-frozen-comparison",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "source_audit": source_audit,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "transformer_models": model_contracts,
            "mlp_baseline_reference": mlp_reference,
            "no_test_time_model_selection": True,
            "no_test_time_calibration": True,
            "no_test_time_threshold_selection": True,
        },
    )
    _write_json(
        output_root / "comparison_results.json",
        {
            "transformer_ablations": all_results,
            "mlp_baseline_reference": mlp_reference,
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "test_models": list(all_results),
                    "mlp_baseline_included": True,
                    "capture_fraction": {
                        name: payload["capture_fraction"] for name, payload in all_results.items()
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
