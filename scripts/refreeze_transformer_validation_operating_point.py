#!/usr/bin/env python3
"""Freeze a new validation-only calibration/route contract for a selected V1 checkpoint.

This is intentionally not a training or test-evaluation program.  It copies a
previously validation-selected Transformer checkpoint without changing its
weights, then re-fits the declared calibration family and route operating
point on the source-disjoint validation physical payloads only.  It exists so
that a calibration-method ablation does not silently retrain, reselect geometry
hyperparameters, or open the sealed test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    artifact_summary,
    build_transformer_graph_bundle,
    calibrate_transformer_scores,
    candidate_score_metrics,
    graph_bundle_summary,
    load_transformer_artifact,
    predict_transformer_scores,
    save_transformer_artifact,
    source_disjoint_audit,
)
from training.transformer_route_selection import select_route_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_SELECTION_POLICY = (
    "nominal_primary_then_validation_capture_count_then_maximum_magnitude_"
    "then_mean_route_quality"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return dict(payload)


def _pair_mapping(raw: Mapping[object, object]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        parts = str(key).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid adjacent station-pair key: {key!r}")
        values[(int(parts[0]), int(parts[1]))] = float(value)
    if set(values) != set(ADJACENT_STATION_PAIRS):
        raise ValueError("route initial thresholds do not define exactly 0->1, 1->2, 2->3")
    return values


def _selection_payload(result: object) -> dict[str, object]:
    return {
        "thresholds": {
            f"{left}->{right}": float(result.thresholds[(left, right)])
            for left, right in ADJACENT_STATION_PAIRS
        },
        "unmatched_penalty": float(result.unmatched_penalty),
        "rank": list(result.rank),
        "capture_by_magnitude": {
            str(magnitude): bool(value) for magnitude, value in sorted(result.capture_by_magnitude.items())
        },
        "evaluation_by_magnitude": {
            str(magnitude): value for magnitude, value in sorted(result.evaluation_by_magnitude.items())
        },
        "search_rows": list(result.search_rows),
        "selection_event_counts_by_magnitude": {
            str(magnitude): int(value)
            for magnitude, value in sorted(result.selection_event_counts_by_magnitude.items())
        },
        "full_validation_event_counts_by_magnitude": {
            str(magnitude): int(value)
            for magnitude, value in sorted(result.full_validation_event_counts_by_magnitude.items())
        },
        "full_validation_rerank_candidates_requested": int(
            result.full_validation_rerank_candidates_requested
        ),
        "full_validation_rerank_candidates_evaluated": int(
            result.full_validation_rerank_candidates_evaluated
        ),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    supplied = load_yaml_with_base(path)
    root = supplied.get("physical_curriculum_mlp", supplied)
    transformer = supplied.get("geometry_aware_transformer_v1")
    if not isinstance(root, Mapping) or not isinstance(transformer, Mapping):
        raise ValueError("configuration requires physical_curriculum_mlp and geometry_aware_transformer_v1")
    if not isinstance(root.get("refit"), Mapping):
        raise ValueError("configuration requires physical_curriculum_mlp.refit")
    return dict(supplied), dict(root), dict(transformer)


def _validate_manifest_contract(
    root: Mapping[str, Any], transformer: Mapping[str, Any], manifest: Mapping[str, Any], samples: list[object]
) -> None:
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("synthetic manifest does not certify physical geometry repropagation")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("validation refreeze requires the mode-0 physical manifest")
    if int(root["refit"].get("q_over_p_mode", -1)) != 0 or int(transformer.get("q_over_p_mode", -1)) != 0:
        raise ValueError("validation refreeze requires declared mode-0 propagation")
    if transformer.get("candidate_chi2_gate") is not None:
        raise ValueError("validation refreeze requires the existing ungated physical candidate graph")
    required = (0.0, 0.1, 1.0, 5.0, 10.0, 50.0)
    observed = tuple(sorted({float(sample.magnitude_mm) for sample in samples if sample.split == "validation"}))
    if observed != required:
        raise ValueError("validation split does not contain the fixed 0/0.1/1/5/10/50 mm curriculum")


def _validate_source_checkpoint(source_root: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = (source_root / "checkpoint.pt").resolve()
    selection_path = (source_root / "model_selection.json").resolve()
    if not checkpoint.is_file() or not selection_path.is_file():
        raise FileNotFoundError("source validation root lacks checkpoint.pt or model_selection.json")
    selection = _load_json(selection_path)
    if selection.get("selection_split") != "validation_only" or selection.get("test_opened") is not False:
        raise ValueError("source Transformer checkpoint was not selected on validation only")
    if selection.get("selected_checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("source model-selection hash does not match checkpoint.pt")
    return checkpoint, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--source-validation-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied, root, transformer = _load_config(config_path)
    calibration_config = transformer.get("calibration")
    route_config = transformer.get("route_selection")
    criteria = transformer.get("capture_success")
    if not all(isinstance(value, Mapping) for value in (calibration_config, route_config, criteria)):
        raise ValueError("configuration has no complete calibration/route/capture section")
    method = str(calibration_config.get("method", "temperature"))
    scope = str(calibration_config.get("scope", "station_pair"))
    if method not in {"temperature", "platt"} or scope not in {"global", "station_pair"}:
        raise ValueError("calibration method/scope is invalid")

    # A validation refreeze must not require, resolve, or open a sealed test
    # split.  The V3 expanded corpus deliberately has only train/validation
    # samples until its hypothesis is frozen.
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    _validate_manifest_contract(root, transformer, manifest, list(samples))
    source_audit = source_disjoint_audit(samples)
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    source_root = Path(args.source_validation_model_dir).expanduser().resolve()
    source_checkpoint, source_selection = _validate_source_checkpoint(source_root)
    model, artifact = load_transformer_artifact(source_checkpoint, device=str(transformer["training"]["device"]))
    if tuple(artifact.output_station_pairs) != ADJACENT_STATION_PAIRS:
        raise ValueError("source checkpoint does not score the V1 adjacent station path")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty validation refreeze output")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_refreeze_contract.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v1-validation-refreeze",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "loaded_event_splits": ["validation"],
            "test_events_loaded": False,
            "test_opened": False,
            "source_audit_from_manifest": source_audit,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_model_selection": str(source_root / "model_selection.json"),
            "source_model_selection_sha256": _sha256(source_root / "model_selection.json"),
            "weights_reused_without_retraining": True,
            "calibration_method": method,
            "calibration_scope": scope,
        },
    )

    # This is the same mode-0 Acts physical candidate construction as V1c.
    # Only validation samples are materialized; train/test ROOT payloads stay unopened.
    validation_sets = build_candidate_sets(
        validation_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    validation_bundle = build_transformer_graph_bundle(validation_sets, context_mode=artifact.context_mode)
    batch_size = int(transformer["training"]["batch_size"])
    raw_scores = predict_transformer_scores(
        model,
        validation_bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=str(transformer["training"]["device"]),
        batch_size=batch_size,
    )
    calibration_bins = int(calibration_config["bins"])
    calibrated_scores, calibration_payload = calibrate_transformer_scores(
        validation_bundle.adjacent_sets,
        raw_scores,
        calibration_bins,
        scope=scope,
        method=method,
    )
    initial_thresholds = _pair_mapping(dict(route_config["initial_thresholds"]))
    route_result = select_route_operating_point(
        validation_bundle.adjacent_sets,
        calibrated_scores,
        station_path=tuple(int(value) for value in transformer["station_path"]),
        threshold_grid=tuple(float(value) for value in route_config["threshold_grid"]),
        unmatched_penalties=tuple(float(value) for value in route_config["unmatched_penalties"]),
        initial_thresholds=initial_thresholds,
        maximum_sweeps=int(route_config["maximum_sweeps"]),
        capture_criteria=dict(criteria),
        calibration_bins=calibration_bins,
        maximum_hypotheses=int(route_config["maximum_hypotheses"]),
        selection_events_per_magnitude=(
            None
            if route_config.get("selection_events_per_magnitude") is None
            else int(route_config["selection_events_per_magnitude"])
        ),
        full_validation_rerank_candidates=int(
            route_config.get("full_validation_rerank_candidates", 1)
        ),
    )

    # Materialize an identical-weight checkpoint in the new immutable contract
    # directory.  Its provenance/hash records make the lack of retraining
    # explicit, while the frozen test evaluator can remain path-local.
    checkpoint = output_root / "checkpoint.pt"
    save_transformer_artifact(checkpoint, model, artifact)
    selection_payload = _selection_payload(route_result)
    _write_json(
        output_root / "model_selection.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v1-validation-refreeze",
            "selection_split": "validation_only",
            "test_opened": False,
            "model_weights": "reused_without_retraining",
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_model_selection": str(source_root / "model_selection.json"),
            "source_model_selection_sha256": _sha256(source_root / "model_selection.json"),
            "source_selected_geometric_hyperparameters": source_selection.get("selected"),
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": _sha256(checkpoint),
        },
    )
    _write_json(
        output_root / "calibration.json",
        {
            "schema_version": "faser-transformer-validation-calibration-v1",
            "fit_split": "validation_only",
            "test_opened": False,
            "calibration": calibration_payload,
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "schema_version": "faser-transformer-route-operating-point-v1",
            "selection_split": "validation_only",
            "test_opened": False,
            "method": str(route_config["method"]),
            "thresholds": selection_payload["thresholds"],
            "unmatched_penalty": selection_payload["unmatched_penalty"],
            "capture_success_criteria": dict(criteria),
            # Earlier V1 configurations declared this explanatory label
            # explicitly.  The expanded V3 control uses the same policy via
            # ``select_route_operating_point`` but does not duplicate the
            # optional key, so retain the exact documented default.
            "selection_policy": str(
                route_config.get("selection_policy", DEFAULT_ROUTE_SELECTION_POLICY)
            ),
            "validation_route_selection": selection_payload,
        },
    )
    _write_json(
        output_root / "validation_metrics.json",
        {
            "raw_candidate": candidate_score_metrics(
                validation_bundle.adjacent_sets, raw_scores, calibration_bins
            ),
            "calibrated_candidate": candidate_score_metrics(
                validation_bundle.adjacent_sets, calibrated_scores, calibration_bins
            ),
            "route_selection": selection_payload,
            "validation_graph_bundle": graph_bundle_summary(validation_bundle),
            "source_artifact": artifact_summary(artifact),
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "test_opened": False,
                    "weights_reused_without_retraining": True,
                    "calibration_method": method,
                    "validation_capture_by_magnitude": selection_payload["capture_by_magnitude"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
