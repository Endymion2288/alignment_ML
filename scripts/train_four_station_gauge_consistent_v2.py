#!/usr/bin/env python3
"""Train the pre-registered gauge-consistent V2 objective control.

Loads only the existing four-station train overlay.  Development validation
is not opened, not used for checkpoint selection, and not used to set loss
weights.  Operating convention is frozen before the first optimizer step:
identity Platt, threshold 0.001, unmatched_penalty -1.0.  GPU-only.
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

from datasets.physical_curriculum import (
    CurriculumSample,
    load_synthetic_curriculum_manifest,
    uniform_condition_axis,
)
from models.route_transformer import RouteAwareTransformerConfig
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, stages_from_payload
from training.gauge_consistent_route import (
    GaugeConsistentAuxConfig,
    identity_calibration_payload,
    train_gauge_consistent_v2,
)
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    route_aware_artifact_summary,
    save_route_aware_transformer_artifact,
)
from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    DEVELOPMENT_SOURCES,
    HISTORY_ONLY_SOURCES,
    OUTLIER_SOURCES,
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
    is_sealed_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_VALIDATION_SOURCES = tuple(sorted(DEVELOPMENT_SOURCES))
FORBIDDEN_TRAIN_SOURCES = (
    HISTORY_ONLY_SOURCES
    | set(RESERVED_BLIND_SOURCES)
    | set(UNUSED_RESERVE_SOURCES)
    | set(OUTLIER_SOURCES)
)


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
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires mapping '{key}'")
    return dict(value)


WB59_CHECKPOINT_SHA256 = "6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc"


def _copy_frozen_aux_weights(objective: Mapping[str, Any]) -> dict[str, float] | None:
    block = objective.get("objective")
    if not isinstance(block, Mapping):
        return None
    algorithm = block.get("loss_weight_algorithm")
    if not isinstance(algorithm, Mapping):
        return None
    if str(algorithm.get("method")) != "copy_frozen_workbook59_aux_weights":
        return None
    source = str(algorithm.get("source_checkpoint_sha256") or "")
    if source != WB59_CHECKPOINT_SHA256:
        raise ValueError("frozen aux weights must copy the workbook-59 checkpoint")
    return {
        "gauge_twin_consistency_weight": float(algorithm["gauge_twin_consistency_weight"]),
        "packing_route_competition_weight": float(algorithm["packing_route_competition_weight"]),
        "dustbin_aware_route_margin_weight": float(algorithm["dustbin_aware_route_margin_weight"]),
    }


def _training_config(raw: Mapping[str, Any]) -> tuple[RouteAwareTrainingConfig, float, bool, str]:
    payload = dict(raw)
    margin = float(payload.pop("packing_route_competition_margin", 1.0))
    enable_dustbin = bool(payload.pop("enable_dustbin_aware_route_margin", False))
    reduction = str(payload.pop("packing_route_competition_reduction", "mean"))
    if reduction not in {"mean", "max"}:
        raise ValueError("packing_route_competition_reduction must be mean or max")
    if str(payload.get("device", "cuda")) != "cuda":
        raise ValueError("gauge-consistent V2 training is GPU-only")
    payload["device"] = "cuda"
    allowed = {field.name for field in RouteAwareTrainingConfig.__dataclass_fields__.values()}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unknown V2 training keys: " + ", ".join(sorted(unknown)))
    return RouteAwareTrainingConfig(**payload), margin, enable_dustbin, reduction


def _validate_train_only(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    samples: Sequence[CurriculumSample],
    q_over_p_mode: int,
    *,
    control_id: str | None = None,
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("train",):
        raise ValueError("gauge-consistent V2 loads only the train overlay")
    forbidden = {str(value) for value in contract.get("forbidden_splits", ())}
    if "test" not in forbidden:
        raise ValueError("gauge-consistent V2 must forbid the sealed test split")
    if contract.get("physical_geometry_repropagation") is not True:
        raise ValueError("V2 requires physical geometry repropagation")
    if int(contract.get("q_over_p_mode", -1)) != q_over_p_mode or contract.get("candidate_chi2_gate") is not None:
        raise ValueError(f"V2 requires the ungated physical mode-{q_over_p_mode} candidate graph")
    if {sample.split for sample in samples} != {"train"}:
        raise ValueError("gauge-consistent V2 opened a non-train split")
    constituents = {str(source) for sample in samples for source in sample.source_ids}
    leaking = constituents & FORBIDDEN_TRAIN_SOURCES
    leaking.update(source for source in constituents if is_sealed_source(source))
    if leaking:
        raise ValueError("history-only, reserved-blind, sealed, or outlier sources leaked into training: " + ", ".join(sorted(leaking)))
    if str(control_id or "") == "retrained_v2_source_disjoint_diversity_v1":
        if constituents != set(AUTHORIZED_SIX_TRAIN_SOURCES):
            raise ValueError(
                "source-diversity V2 must train the authorized six-source set, got: "
                + ", ".join(sorted(constituents))
            )
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("train overlay lacks physical geometry repropagation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "geometry_aware_transformer_v2_four_station_gauge_consistent.yaml"),
    )
    parser.add_argument(
        "--objective-contract",
        default=str(PROJECT_ROOT / "configs" / "physical_four_station_gauge_consistent_route_training.yaml"),
    )
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    contract_path = Path(args.objective_contract).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    supplied = load_yaml_with_base(config_path)
    objective = load_yaml_with_base(contract_path)
    root = _require_mapping(supplied, "route_aware_transformer_v2")
    contract = _require_mapping(root, "input_contract")
    architecture = _require_mapping(root, "architecture")
    training, margin, enable_dustbin, reduction = _training_config(_require_mapping(root, "training"))
    raw_stages = root.get("curriculum_stages")
    if not isinstance(raw_stages, list):
        raise ValueError("curriculum_stages must be a list")
    calibration_config = _require_mapping(root, "calibration")
    if str(calibration_config.get("fit_split")) != "identity_frozen_pre_training":
        raise ValueError("this control freezes identity Platt before training")
    route_selection = _require_mapping(root, "route_selection")
    if str(route_selection.get("selection_split")) != "pre_registered_frozen_historical_packing":
        raise ValueError("operating point must be the pre-registered historical packing convention")
    if abs(float(route_selection["unmatched_penalty"]) + 1.0) > 1.0e-12:
        raise ValueError("unmatched_penalty must stay at the frozen historical -1.0")

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train",),
    )
    _validate_train_only(
        contract,
        manifest,
        samples,
        int(args.q_over_p_mode),
        control_id=None if not isinstance(objective, Mapping) else str(objective.get("control_id") or ""),
    )
    axis = uniform_condition_axis(samples)
    train_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    from training.geometry_aware_transformer import build_transformer_graph_bundle

    train_bundle = build_transformer_graph_bundle(train_sets, context_mode="full_event")
    model_config = RouteAwareTransformerConfig(node_feature_dim=17, edge_feature_dim=11, **architecture)
    aux = GaugeConsistentAuxConfig(
        packing_margin=margin,
        pair_threshold=0.001,
        unmatched_penalty=-1.0,
        enable_dustbin_aware_route_margin=enable_dustbin,
        route_competition_reduction=reduction,
        copy_frozen_aux_weights=_copy_frozen_aux_weights(objective),
        zero_mean_threshold=1.0e-6,
        zero_mean_fallback_weight=1.0,
    )
    def _persist_weight_contract(payload: Mapping[str, object]) -> None:
        _write_json(output_root / "aux_loss_weight_contract.json", payload)
        print(json.dumps({"aux_loss_weight_contract": payload}, sort_keys=False), flush=True)

    model, artifact, history, weight_contract = train_gauge_consistent_v2(
        train_bundle,
        model_config,
        training,
        stages_from_payload(raw_stages),
        aux,
        on_weight_contract=_persist_weight_contract,
    )
    checkpoint = output_root / "route_aware_transformer_v2.pt"
    save_route_aware_transformer_artifact(checkpoint, model, artifact)
    _write_json(output_root / "aux_loss_weight_contract.json", weight_contract)
    _write_json(
        output_root / "calibration.json",
        {
            "fit_split": "identity_frozen_pre_training",
            "test_opened": False,
            "identity_map": True,
            "calibration": identity_calibration_payload(),
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "method": "adjacent_contiguous_unit_capacity_set_packing",
            "selection_split": "pre_registered_frozen_historical_packing",
            "test_opened": False,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "complete_route_query_injected_into_packing": False,
        },
    )
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_run_contract.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-gauge-consistent",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "objective_contract": str(contract_path),
            "objective_contract_sha256": _sha256(contract_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "loaded_event_splits": ["train"],
            "forbidden_splits": ["validation", "test"],
            "development_validation_loaded": False,
            "development_validation_only_sources": list(DEVELOPMENT_VALIDATION_SOURCES),
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": int(args.q_over_p_mode),
            "condition_axis": axis,
            "candidate_chi2_gate": None,
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "checkpoint_selection": "last_completed_epoch_of_fixed_30_epoch_budget",
            "control_id": objective.get("control_id"),
            "loaded_source_ids": sorted(
                {str(source) for sample in samples for source in sample.source_ids}
            ),
            "history_only_sources_loaded": False,
            "reserved_blind_used_for_training": False,
            "reserved_blind_used_for_checkpoint_selection": False,
            "blind_not_used_for_training": True,
            "early_stopping": False,
            "objective_reestimated": False,
        },
    )
    _write_json(output_root / "artifact_summary.json", route_aware_artifact_summary(artifact))
    _write_json(output_root / "training_history.json", {"history": history})
    _write_csv(output_root / "training_history.csv", history)
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "epochs": len(history),
                "aux_loss_weights": {
                    "gauge_twin_consistency_weight": weight_contract["gauge_twin_consistency_weight"],
                    "packing_route_competition_weight": weight_contract["packing_route_competition_weight"],
                    "dustbin_aware_route_margin_weight": weight_contract.get(
                        "dustbin_aware_route_margin_weight"
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
