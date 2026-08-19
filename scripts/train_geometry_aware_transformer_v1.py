#!/usr/bin/env python3
"""Train and freeze Geometry-Aware Transformer V1 on train/validation only.

The program intentionally never constructs a test candidate set.  It reuses
the already materialized source-disjoint physical curriculum and mode-0 Acts
candidate graph, trains the required sparse-Transformer ablations, fits each
temperature map on validation only, and freezes route thresholds/penalties
before any separate test evaluator is allowed to run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from datasets.physical_curriculum import load_synthetic_curriculum_manifest, uniform_condition_axis
from models.transformer import SparseTransformerConfig
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
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
    stages_from_payload,
    train_transformer_v1,
    training_config_from_mapping,
)
from training.transformer_route_selection import select_route_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v1.yaml"
DEFAULT_ROUTE_SELECTION_POLICY = (
    "nominal_primary_then_validation_capture_count_then_maximum_magnitude_"
    "then_mean_route_quality"
)


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


def _write_history_csv(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    rows = []
    for entry in history:
        rows.append(
            {
                key: value
                for key, value in entry.items()
                if key != "validation_candidate" and not isinstance(value, (dict, list, tuple))
            }
        )
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_value(row) for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_mapping(raw: Mapping[object, object], expected: Sequence[tuple[int, int]]) -> dict[tuple[int, int], float]:
    parsed: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        text = str(key)
        parts = text.split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid station-pair key '{text}'")
        pair = (int(parts[0]), int(parts[1]))
        parsed[pair] = float(value)
    if set(parsed) != set(expected):
        raise ValueError(
            f"station-pair threshold mapping differs from expected adjacent pairs: {sorted(expected)}"
        )
    return parsed


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
    root: Mapping[str, Any],
    transformer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    samples: Sequence[object],
    q_over_p_mode: int = 0,
) -> None:
    declared = root.get("allowed_splits")
    if declared is not None and tuple(str(value) for value in declared) != ("train", "validation"):
        raise ValueError("Transformer V1 train/validation study must allow exactly train and validation")
    forbidden = {str(value) for value in root.get("forbidden_splits", ())}
    if forbidden and "test" not in forbidden:
        raise ValueError("Transformer V1 train/validation study must explicitly forbid test")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("synthetic manifest does not certify physical geometry repropagation")
    if int(manifest.get("q_over_p_mode", -1)) != q_over_p_mode or int(root["refit"].get("q_over_p_mode", -1)) != q_over_p_mode:
        raise ValueError(
            f"Transformer V1 requires the manifest and refit contract to declare q_over_p_mode={q_over_p_mode}"
        )
    if int(transformer.get("q_over_p_mode", -1)) != q_over_p_mode:
        raise ValueError(f"Transformer configuration must declare q_over_p_mode={q_over_p_mode}")
    if transformer.get("candidate_chi2_gate") is not None:
        raise ValueError("Transformer V1 must use the existing ungated physical candidate graph")
    rotation = root.get("rotation_curriculum")
    if isinstance(rotation, Mapping):
        expected_axis = str(rotation.get("condition_axis", ""))
        raw_points = rotation.get("rotation_points")
        if not expected_axis or not isinstance(raw_points, list) or not raw_points:
            raise ValueError("Transformer V1 rotation curriculum configuration is incomplete")
        expected = tuple(
            sorted(
                {
                    float(point.get("condition_magnitude", abs(float(point["ry_mrad"]))))
                    for point in raw_points
                    if isinstance(point, Mapping) and ("condition_magnitude" in point or "ry_mrad" in point)
                }
            )
        )
    else:
        expected_axis = "translation_xy_mm"
        expected = tuple(
            sorted(float(value) for value in root.get("payload_bank", {}).get("magnitudes_mm", ()))
        )
    if not expected:
        raise ValueError("Transformer V1 configuration has no declared physical curriculum")
    if uniform_condition_axis(samples) != expected_axis:
        raise ValueError("Transformer V1 manifest condition axis differs from the frozen configuration")
    for split in ("train", "validation"):
        observed = tuple(
            sorted({float(sample.curriculum_magnitude) for sample in samples if sample.split == split})
        )
        if observed != expected:
            raise ValueError(f"{split} split does not contain the full declared physical curriculum")
    if {str(sample.split) for sample in samples} != {"train", "validation"}:
        raise ValueError("Transformer V1 loader must return only train and validation samples")


def _candidate_grid(spec: Mapping[str, Any]) -> list[dict[str, object]]:
    values = spec.get("chi2_hyperparameter_candidates")
    if not isinstance(values, list) or not values:
        raise ValueError("an ablation requires at least one chi2_hyperparameter_candidates entry")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("chi2_hyperparameter candidate must be a mapping")
        value = dict(raw)
        label = str(value.pop("label", ""))
        if not label or label in names:
            raise ValueError("chi2_hyperparameter candidates need unique non-empty labels")
        names.add(label)
        if set(value) != {"chi2_lambda", "chi2_tau"}:
            raise ValueError("chi2_hyperparameter candidate must declare only lambda and tau")
        candidate = {
            "label": label,
            "chi2_lambda": float(value["chi2_lambda"]),
            "chi2_tau": float(value["chi2_tau"]),
        }
        if candidate["chi2_lambda"] < 0.0 or candidate["chi2_tau"] <= 0.0:
            raise ValueError("chi2 lambda/tau candidate is invalid")
        result.append(candidate)
    return result


def _selected_hyperparameters(
    spec: Mapping[str, Any], selected: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    inherited = spec.get("inherit_chi2_hyperparameters_from")
    if inherited is None:
        return _candidate_grid(spec)
    source = selected.get(str(inherited))
    if source is None:
        raise ValueError(f"ablation inherits unavailable geometry hyperparameters from '{inherited}'")
    return [
        {
            "label": f"inherited_{source['label']}",
            "chi2_lambda": float(source["chi2_lambda"]),
            "chi2_tau": float(source["chi2_tau"]),
        }
    ]


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


def _reference_artifacts(raw: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in raw.items():
        if name == "frozen_route_test":
            # The historical test reference is intentionally retained as a
            # label in legacy configuration only.  Do not resolve or inspect
            # a sealed test artifact during a train/validation study.
            result[name] = {"excluded": True, "reason": "sealed_test_boundary"}
            continue
        resolved = (PROJECT_ROOT / str(value)).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[name] = {"path": str(resolved), "sha256": _sha256(resolved)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--ablation",
        action="append",
        default=None,
        help="Train one named Transformer ablation; default trains every configured Transformer ablation.",
    )
    parser.add_argument(
        "--q-over-p-mode",
        type=int,
        default=0,
        choices=(0, 3),
        help=(
            "propagation record variant used to build the physical candidate graph; "
            "the configuration and synthetic manifest must declare the same mode"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied, root, transformer = _load_config(config_path)
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    _validate_manifest_contract(root, transformer, manifest, samples, q_over_p_mode=int(args.q_over_p_mode))
    source_audit = source_disjoint_audit(samples)
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    if not train_samples or not validation_samples:
        raise ValueError("Transformer V1 requires non-empty train and validation samples")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Transformer validation output")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_run_contract.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v1-validation",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "physical_geometry_repropagation": True,
            "q_over_p_mode": int(args.q_over_p_mode),
            "condition_axis": uniform_condition_axis(samples),
            "candidate_chi2_gate": None,
            "candidate_graph": f"existing_mode{int(args.q_over_p_mode)}_acts_physical_candidates_all_six_station_pairs",
            "output_station_pairs": [f"{left}->{right}" for left, right in ADJACENT_STATION_PAIRS],
            "local_edge_pretraining": {
                "enabled": bool(transformer.get("local_edge_pretrain_stages", [])),
                "stages": transformer.get("local_edge_pretrain_stages", []),
                "uses_message_edges": False,
            },
            "source_audit": source_audit,
            "loaded_event_splits": ["train", "validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_opened": False,
            "mlp_baseline_reference": _reference_artifacts(
                dict(transformer.get("mlp_baseline_reference", {}))
            ),
        },
    )

    # The MLP feature matrix is not used by the Transformer.  Asking the
    # existing candidate builder for its smallest auxiliary schema leaves the
    # actual mode-0 candidate definition unchanged and limits duplicate RAM.
    train_sets = build_candidate_sets(
        train_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    validation_sets = build_candidate_sets(
        validation_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    bundles: dict[str, tuple[object, object]] = {}

    def bundle_pair(context_mode: str) -> tuple[object, object]:
        cached = bundles.get(context_mode)
        if cached is None:
            cached = (
                build_transformer_graph_bundle(train_sets, context_mode=context_mode),
                build_transformer_graph_bundle(validation_sets, context_mode=context_mode),
            )
            bundles[context_mode] = cached
        return cached

    architecture = transformer.get("architecture")
    training = transformer.get("training")
    raw_stages = transformer.get("curriculum_stages")
    raw_local_pretrain_stages = transformer.get("local_edge_pretrain_stages", [])
    calibration = transformer.get("calibration")
    route_selection = transformer.get("route_selection")
    criteria = transformer.get("capture_success")
    ablations = transformer.get("ablations")
    if not all(isinstance(value, Mapping) for value in (architecture, training, calibration, route_selection, criteria, ablations)):
        raise ValueError("Transformer configuration has an incomplete architecture/training/selection section")
    if not isinstance(raw_stages, list):
        raise ValueError("Transformer configuration has no curriculum_stages list")
    if not isinstance(raw_local_pretrain_stages, list):
        raise ValueError("Transformer local_edge_pretrain_stages must be a list when supplied")
    training_config = training_config_from_mapping(dict(training))
    stages = stages_from_payload(raw_stages)
    local_pretrain_stages = stages_from_payload(raw_local_pretrain_stages)
    calibration_bins = int(calibration["bins"])
    calibration_scope = str(calibration["scope"])
    calibration_method = str(calibration.get("method", "temperature"))
    if calibration_method not in {"temperature", "platt"}:
        raise ValueError("Transformer calibration.method must be 'temperature' or 'platt'")
    initial_thresholds = _pair_mapping(
        dict(route_selection["initial_thresholds"]), ADJACENT_STATION_PAIRS
    )
    configured_names = list(ablations)
    selected_names = configured_names if args.ablation is None else list(args.ablation)
    if not selected_names or any(name not in ablations for name in selected_names):
        raise ValueError("requested ablation is absent from the Transformer configuration")

    selected_hyperparameters: dict[str, Mapping[str, object]] = {}
    overall: dict[str, object] = {}
    for name in selected_names:
        raw_spec = ablations[name]
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"ablation '{name}' is not a mapping")
        spec = dict(raw_spec)
        context_mode = str(spec.get("context_mode", ""))
        if context_mode not in {"full_event", "station_pair"}:
            raise ValueError(f"ablation '{name}' has an invalid context_mode")
        train_bundle, validation_bundle = bundle_pair(context_mode)
        ablation_training_config = (
            replace(training_config, batch_size=int(spec["batch_size"]))
            if "batch_size" in spec
            else training_config
        )
        if ablation_training_config.batch_size < 1:
            raise ValueError(f"ablation '{name}' has an invalid batch_size override")
        candidates = _selected_hyperparameters(spec, selected_hyperparameters)
        candidate_rows: list[dict[str, object]] = []
        ablation_root = output_root / name
        ablation_root.mkdir(parents=True, exist_ok=False)
        raw_architecture_overrides = spec.get("architecture_overrides", {})
        if not isinstance(raw_architecture_overrides, Mapping):
            raise ValueError(f"ablation '{name}' architecture_overrides must be a mapping")
        architecture_overrides = dict(raw_architecture_overrides)
        unsupported_overrides = set(architecture_overrides) - {"use_local_edge_residual"}
        if unsupported_overrides:
            raise ValueError(
                f"ablation '{name}' has unsupported architecture override(s): "
                + ", ".join(sorted(unsupported_overrides))
            )
        effective_architecture = {**dict(architecture), **architecture_overrides}
        for candidate in candidates:
            candidate_root = ablation_root / "candidates" / str(candidate["label"])
            candidate_root.mkdir(parents=True, exist_ok=False)
            model_config = SparseTransformerConfig(
                node_feature_dim=len(NODE_FEATURE_NAMES),
                edge_feature_dim=len(EDGE_FEATURE_NAMES),
                **effective_architecture,
                use_geometric_bias=bool(spec["use_geometric_bias"]),
                use_chi2_physics_term=bool(spec["use_chi2_physics_term"]),
                chi2_lambda=float(candidate["chi2_lambda"]),
                chi2_tau=float(candidate["chi2_tau"]),
            )
            model, artifact, history = train_transformer_v1(
                train_bundle,
                validation_bundle,
                model_config,
                ablation_training_config,
                stages,
                calibration_bins,
                # The ordinary sparse-Transformer ablation deliberately
                # disables the local residual decoder, so it cannot run a
                # decoder-specific pretraining phase.
                local_pretrain_stages=(
                    local_pretrain_stages if model_config.use_local_edge_residual else None
                ),
            )
            candidate_checkpoint = candidate_root / "checkpoint.pt"
            save_transformer_artifact(candidate_checkpoint, model, artifact)
            raw_validation_scores = predict_transformer_scores(
                model,
                validation_bundle,
                artifact.node_standardizer,
                artifact.edge_standardizer,
                device=ablation_training_config.device,
                batch_size=ablation_training_config.batch_size,
            )
            raw_validation_metrics = candidate_score_metrics(
                validation_bundle.adjacent_sets, raw_validation_scores, calibration_bins
            )
            _write_json(candidate_root / "artifact_summary.json", artifact_summary(artifact))
            _write_json(candidate_root / "training_history.json", {"history": history})
            _write_history_csv(candidate_root / "training_history.csv", history)
            _write_json(
                candidate_root / "validation_candidate_metrics.json",
                {"raw_validation_candidate_metrics": raw_validation_metrics},
            )
            candidate_rows.append(
                {
                    "label": str(candidate["label"]),
                    "chi2_lambda": float(candidate["chi2_lambda"]),
                    "chi2_tau": float(candidate["chi2_tau"]),
                    "checkpoint": str(candidate_checkpoint),
                    "checkpoint_sha256": _sha256(candidate_checkpoint),
                    "best_validation_average_precision": raw_validation_metrics["average_precision"],
                    "best_validation_roc_auc": raw_validation_metrics["roc_auc"],
                    "best_validation_ece": raw_validation_metrics["expected_calibration_error"],
                }
            )

        chosen = max(
            candidate_rows,
            key=lambda row: (
                float(row["best_validation_average_precision"]),
                float(row["best_validation_roc_auc"]),
                -float(row["chi2_tau"]),
            ),
        )
        selected_hyperparameters[name] = dict(chosen)
        model, artifact = load_transformer_artifact(
            str(chosen["checkpoint"]), device=training_config.device
        )
        validation_scores = predict_transformer_scores(
            model,
            validation_bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            device=ablation_training_config.device,
            batch_size=ablation_training_config.batch_size,
        )
        calibrated_scores, calibration_payload = calibrate_transformer_scores(
            validation_bundle.adjacent_sets,
            validation_scores,
            calibration_bins,
            scope=calibration_scope,
            method=calibration_method,
        )
        route_result = select_route_operating_point(
            validation_bundle.adjacent_sets,
            calibrated_scores,
            station_path=tuple(int(value) for value in transformer["station_path"]),
            threshold_grid=tuple(float(value) for value in route_selection["threshold_grid"]),
            unmatched_penalties=tuple(float(value) for value in route_selection["unmatched_penalties"]),
            initial_thresholds=initial_thresholds,
            maximum_sweeps=int(route_selection["maximum_sweeps"]),
            capture_criteria=dict(criteria),
            calibration_bins=calibration_bins,
            maximum_hypotheses=int(route_selection["maximum_hypotheses"]),
            selection_events_per_magnitude=(
                None
                if route_selection.get("selection_events_per_magnitude") is None
                else int(route_selection["selection_events_per_magnitude"])
            ),
            full_validation_rerank_candidates=int(
                route_selection.get("full_validation_rerank_candidates", 1)
            ),
        )
        calibrated_metrics = candidate_score_metrics(
            validation_bundle.adjacent_sets, calibrated_scores, calibration_bins
        )
        selected_checkpoint = ablation_root / "checkpoint.pt"
        save_transformer_artifact(selected_checkpoint, model, artifact)
        _write_json(
            ablation_root / "model_selection.json",
            {
                "selection_split": "validation_only",
                "test_opened": False,
                "geometric_bias_hyperparameter_selection_metric": "validation_candidate_average_precision_then_roc_auc",
                "candidates": candidate_rows,
                "selected": chosen,
                "selected_checkpoint": str(selected_checkpoint),
                "selected_checkpoint_sha256": _sha256(selected_checkpoint),
            },
        )
        _write_json(
            ablation_root / "calibration.json",
            {
                "schema_version": "faser-transformer-validation-calibration-v1",
                "fit_split": "validation_only",
                "test_opened": False,
                "calibration": calibration_payload,
            },
        )
        selection_payload = _selection_payload(route_result)
        _write_json(
            ablation_root / "validation_selected_operating_point.json",
            {
                "schema_version": "faser-transformer-route-operating-point-v1",
                "selection_split": "validation_only",
                "test_opened": False,
                "method": str(route_selection["method"]),
                "thresholds": selection_payload["thresholds"],
                "unmatched_penalty": selection_payload["unmatched_penalty"],
                "capture_success_criteria": dict(criteria),
                # The expanded controls use the same selection implementation
                # but need not repeat this legacy metadata-only label.
                "selection_policy": str(
                    route_selection.get("selection_policy", DEFAULT_ROUTE_SELECTION_POLICY)
                ),
                "validation_route_selection": selection_payload,
            },
        )
        _write_json(
            ablation_root / "validation_metrics.json",
            {
                "raw_candidate": candidate_score_metrics(
                    validation_bundle.adjacent_sets, validation_scores, calibration_bins
                ),
                "calibrated_candidate": calibrated_metrics,
                "route_selection": selection_payload,
                "train_graph_bundle": graph_bundle_summary(train_bundle),
                "validation_graph_bundle": graph_bundle_summary(validation_bundle),
            },
        )
        overall[name] = {
            "context_mode": context_mode,
            "batch_size": ablation_training_config.batch_size,
            "selected_hyperparameters": chosen,
            "checkpoint": str(selected_checkpoint),
            "checkpoint_sha256": _sha256(selected_checkpoint),
            "validation_candidate_average_precision": calibrated_metrics["average_precision"],
            "validation_candidate_roc_auc": calibrated_metrics["roc_auc"],
            "validation_candidate_expected_calibration_error": calibrated_metrics["expected_calibration_error"],
            "validation_route_capture_by_magnitude": selection_payload["capture_by_magnitude"],
            "validation_route_thresholds": selection_payload["thresholds"],
            "validation_route_unmatched_penalty": selection_payload["unmatched_penalty"],
        }

    _write_json(
        output_root / "validation_model_selection.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v1-validation",
            "selection_split": "validation_only",
            "test_opened": False,
            "trained_transformer_ablations": overall,
            "required_mlp_baseline_reference": _reference_artifacts(
                dict(transformer.get("mlp_baseline_reference", {}))
            ),
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "test_opened": False,
                    "trained_ablations": list(overall),
                    "validation_average_precision": {
                        name: payload["validation_candidate_average_precision"]
                        for name, payload in overall.items()
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
