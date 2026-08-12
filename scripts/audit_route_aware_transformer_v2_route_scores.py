#!/usr/bin/env python3
"""Audit V2 complete-route score scale on validation physical candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle, source_disjoint_audit
from training.route_aware_transformer import (
    calibrate_route_query_scores,
    load_route_aware_transformer_artifact,
    predict_route_aware_scores,
)


QUANTILES = (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
DEFAULT_THRESHOLDS = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_PENALTIES = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return dict(payload)


def _validate_v2_contract(v2_dir: Path) -> dict[str, Any]:
    contract = _read_json(v2_dir / "validation_run_contract.json")
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V2 checkpoint was not produced from exactly train and validation")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError("V2 checkpoint violates the sealed-test boundary")
    return contract


def _distribution(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return {
        "rows": int(selected.size),
        "mean": None if not selected.size else float(np.mean(selected)),
        "quantiles": {
            str(quantile): None if not selected.size else float(np.quantile(selected, quantile))
            for quantile in QUANTILES
        },
    }


def _threshold_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for threshold in DEFAULT_THRESHOLDS:
        keep = np.asarray(values, dtype=np.float64) >= threshold
        positives = np.asarray(labels, dtype=bool)
        result[str(threshold)] = {
            "retained_routes": int(np.count_nonzero(keep)),
            "retained_truth_routes": int(np.count_nonzero(keep & positives)),
            "truth_route_recall": (
                None if not np.any(positives) else float(np.count_nonzero(keep & positives) / np.count_nonzero(positives))
            ),
        }
    return result


def _utility_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    score = np.clip(np.asarray(values, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    target = np.asarray(labels, dtype=bool)
    log_odds = np.log(score) - np.log1p(-score)
    for penalty in DEFAULT_PENALTIES:
        utility = log_odds + 4.0 * penalty
        positive = utility > 0.0
        result[str(penalty)] = {
            "positive_utility_routes": int(np.count_nonzero(positive)),
            "positive_utility_truth_routes": int(np.count_nonzero(positive & target)),
            "positive_utility_truth_recall": (
                None
                if not np.any(target)
                else float(np.count_nonzero(positive & target) / np.count_nonzero(target))
            ),
            "positive_utility_false_routes": int(np.count_nonzero(positive & ~target)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--v2-validation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    v2_dir = Path(args.v2_validation_dir).expanduser().resolve()
    contract = _validate_v2_contract(v2_dir)
    checkpoint = v2_dir / "route_aware_transformer_v2.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V2 checkpoint is absent: {checkpoint}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite score audit: {output}")
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("route-score audit requires physical mode-0 repropagation")
    if {sample.split for sample in samples} != {"validation"}:
        raise ValueError("route-score audit opened a non-validation split")
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    model, artifact = load_route_aware_transformer_artifact(checkpoint, device=str(args.device))
    prediction = predict_route_aware_scores(
        model,
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=str(args.device),
        batch_size=32,
    )
    calibrated_by_graph, calibration = calibrate_route_query_scores(
        prediction, bins=15, method="platt"
    )
    calibrated = np.concatenate(calibrated_by_graph)
    raw = np.asarray(prediction.route_scores, dtype=np.float64)
    labels = np.asarray(prediction.route_labels, dtype=bool)
    fake = np.asarray(prediction.route_fake_endpoint, dtype=bool)
    hard = np.asarray(prediction.route_hard_negative, dtype=bool)
    if not (raw.shape == calibrated.shape == labels.shape == fake.shape == hard.shape):
        raise RuntimeError("route-score audit arrays are not aligned")
    _write_json(
        output,
        {
            "schema_version": "faser-route-aware-transformer-v2-route-score-audit",
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "v2_validation_dir": str(v2_dir),
            "v2_checkpoint": str(checkpoint),
            "v2_checkpoint_sha256": _sha256(checkpoint),
            "loaded_event_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "source_audit": source_disjoint_audit(samples),
            "frozen_v2_contract": contract,
            "route_calibration": calibration,
            "raw_score_distributions": {
                "all": _distribution(raw, np.ones_like(labels)),
                "truth_consistent": _distribution(raw, labels),
                "false": _distribution(raw, ~labels),
                "fake_endpoint": _distribution(raw, fake),
                "hard_negative": _distribution(raw, hard),
            },
            "platt_score_distributions": {
                "all": _distribution(calibrated, np.ones_like(labels)),
                "truth_consistent": _distribution(calibrated, labels),
                "false": _distribution(calibrated, ~labels),
                "fake_endpoint": _distribution(calibrated, fake),
                "hard_negative": _distribution(calibrated, hard),
            },
            "raw_threshold_retention": _threshold_summary(raw, labels),
            "platt_threshold_retention": _threshold_summary(calibrated, labels),
            "raw_complete_route_utility": _utility_summary(raw, labels),
            "platt_complete_route_utility": _utility_summary(calibrated, labels),
        },
    )
    print(json.dumps({"output": str(output), "test_events_loaded": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
