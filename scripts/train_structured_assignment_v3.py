#!/usr/bin/env python3
"""Train Geometry-Aware Transformer V3 with structured route assignment loss.

This entry point intentionally opens only source-disjoint train/validation
assets.  It writes an uncalibrated checkpoint and structured-loss history;
route-score calibration and route-solver controls are selected later by the
separate validation-only evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from models.route_transformer import RouteAwareTransformerConfig
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    CurriculumStage,
    build_transformer_graph_bundle,
    source_disjoint_audit,
)
from training.structured_assignment import (
    StructuredAssignmentTrainingConfig,
    save_structured_assignment_artifact,
    structured_assignment_artifact_summary,
    train_structured_assignment_v3,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v3.yaml"
MAGNITUDES = (0.0, 0.1, 1.0, 5.0, 10.0, 50.0)


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_value(list(rows)))


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires mapping '{key}'")
    return dict(value)


def _validate_input_contract(
    contract: Mapping[str, object], manifest: Mapping[str, object], samples: Sequence[CurriculumSample]
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("train", "validation"):
        raise ValueError("V3 training must allow exactly source-disjoint train and validation splits")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("V3 training must explicitly forbid the test split")
    if contract.get("physical_geometry_repropagation") is not True:
        raise ValueError("V3 requires physical geometry repropagation")
    if int(contract.get("q_over_p_mode", -1)) != 0 or contract.get("candidate_chi2_gate") is not None:
        raise ValueError("V3 requires the ungated mode-0 Acts physical candidate graph")
    if str(contract.get("feature_set")) != "residual_v1" or str(contract.get("context_mode")) != "full_event":
        raise ValueError("V3 requires the existing residual_v1 full-event graph contract")
    if tuple(int(value) for value in contract.get("station_path", ())) != (0, 1, 2, 3):
        raise ValueError("V3 requires the IFT -> S1 -> S2 -> S3 station path")
    declared = tuple(float(value) for value in contract.get("curriculum_magnitudes_mm", ()))
    if declared != MAGNITUDES:
        raise ValueError("V3 configuration must declare the fixed six-point physical curriculum")
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("V3 manifest does not certify physical mode-0 repropagation")
    if {sample.split for sample in samples} != {"train", "validation"}:
        raise ValueError("V3 loader did not return exactly train and validation samples")
    for split in ("train", "validation"):
        observed = tuple(sorted({float(sample.magnitude_mm) for sample in samples if sample.split == split}))
        if observed != MAGNITUDES:
            raise ValueError(f"V3 {split} samples are missing a physical curriculum magnitude")


def _validate_architecture(architecture: Mapping[str, object]) -> None:
    fixed = {"d_model": 128, "nhead": 8, "num_layers": 4, "ffn_dim": 256}
    for name, expected in fixed.items():
        if int(architecture.get(name, -1)) != expected:
            raise ValueError(f"V3 must retain the fixed V1 {name}={expected} backbone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "structured_assignment_v3")
    contract = _require_mapping(settings, "input_contract")
    architecture = _require_mapping(settings, "architecture")
    training = _require_mapping(settings, "training")
    calibration = _require_mapping(settings, "calibration")
    raw_stages = settings.get("curriculum_stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("V3 configuration requires non-empty curriculum_stages")
    _validate_architecture(architecture)
    if str(calibration.get("scope")) != "station_pair" or str(calibration.get("method")) not in {
        "temperature",
        "platt",
    }:
        raise ValueError("V3 later edge calibration must be station-pair temperature or Platt")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty V3 training output")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    _validate_input_contract(contract, manifest, samples)
    source_audit = source_disjoint_audit(samples)
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    train_sets = build_candidate_sets(train_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1")
    validation_sets = build_candidate_sets(
        validation_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    train_bundle = build_transformer_graph_bundle(train_sets, context_mode="full_event")
    validation_bundle = build_transformer_graph_bundle(validation_sets, context_mode="full_event")
    model_config = RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        **architecture,
    )
    training_config = StructuredAssignmentTrainingConfig(**training)
    stages = tuple(CurriculumStage(**dict(value)) for value in raw_stages)
    model, artifact, history = train_structured_assignment_v3(
        train_bundle,
        validation_bundle,
        model_config,
        training_config,
        stages,
    )
    checkpoint = output_root / "structured_assignment_v3.pt"
    save_structured_assignment_artifact(checkpoint, model, artifact)

    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_run_contract.json",
        {
            "schema_version": "faser-structured-assignment-v3-training",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "loaded_event_splits": ["train", "validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "route_candidates": "complete_chains_of_existing_adjacent_physical_edges_only",
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "training_objective": "exact_loss_augmented_structured_margin",
            "source_audit": source_audit,
        },
    )
    _write_json(output_root / "artifact_summary.json", structured_assignment_artifact_summary(artifact))
    _write_json(output_root / "training_history.json", {"history": history})
    _write_csv(output_root / "training_history.csv", history)
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "checkpoint": str(checkpoint),
                    "validation_best_epoch": artifact.training_summary["best_global_epoch"],
                    "test_events_loaded": False,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
