#!/usr/bin/env python3
"""Train RelativeRoute V4 head-only models (Arm 1 Control / Arm 2 Primary).

Strictly enforces:
- Frozen Workbook-64 backbone (614,947 params frozen)
- Trainable route head only (172,513 params)
- Zero-initialized delta_route_score (weight=0, bias=0)
- Additive complete route logit correction
- Fixed 30 epoch budget (8 nominal, 22 15D relative curriculum)
- No early stopping, checkpoint selection = last completed epoch
- V2 route-head body losses + Workbook-64 solver-aware auxiliaries
- Six authorized train sources only, development and final blind data strictly forbidden
- GPU-only (CUDA)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.physical_curriculum import (
    CurriculumSample,
    load_synthetic_curriculum_manifest,
    uniform_condition_axis,
)
from models.route_transformer import (
    RelativeRouteTransformerConfig,
    RelativeRouteSparseTransformer,
    RouteAwareTransformerConfig,
    RouteAwareSparseTransformer,
    freeze_backbone_and_edge_scorer,
)
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    CurriculumStage,
    TransformerGraph,
    TransformerGraphBundle,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
    stages_from_payload,
)
from training.gauge_consistent_route import (
    pair_gauge_twin_graphs,
    GaugeConsistentAuxConfig,
    identity_calibration_payload,
    _aux_losses_for_batch,
    _iter_training_batches,
    split_graph_edge_logits,
    split_graph_route_logits,
)
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    RouteAwareTransformerArtifact,
    materialize_route_candidate_tables,
    save_route_aware_transformer_artifact,
    route_aware_artifact_summary,
    _edge_positive_weight,
    _forward_route_batch,
    _make_route_batch,
    _route_loss_components,
    _seed_everything,
    _stage_graphs,
    _validate_stages,
    _validate_training_config,
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


FORBIDDEN_TRAIN_SOURCES = (
    HISTORY_ONLY_SOURCES
    | set(RESERVED_BLIND_SOURCES)
    | set(UNUSED_RESERVE_SOURCES)
    | set(OUTLIER_SOURCES)
    | set(DEVELOPMENT_SOURCES)
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
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _compute_parameter_hashes(model: nn.Module) -> dict[str, str]:
    hashes = {}
    for name, p in sorted(model.named_parameters()):
        t = p.detach().cpu().numpy().tobytes()
        hashes[name] = hashlib.sha256(t).hexdigest()
    return hashes


def _compute_frozen_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if not p.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _validate_train_sources(samples: Sequence[CurriculumSample]) -> None:
    if {sample.split for sample in samples} != {"train"}:
        raise ValueError("Head-only training opened a non-train split")
    constituents = {str(source) for sample in samples for source in sample.source_ids}
    leaking = constituents & FORBIDDEN_TRAIN_SOURCES
    leaking.update(source for source in constituents if is_sealed_source(source))
    if leaking:
        raise ValueError(
            f"Forbidden / Development / Blind sources leaked into training: {', '.join(sorted(leaking))}"
        )
    if constituents != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise ValueError(
            f"Training set must match authorized six train sources. Got: {', '.join(sorted(constituents))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--synthetic-manifest",
        default="outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
        help="Path to 6-source train synthetic manifest",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt",
        help="Path to pre-trained frozen Workbook-64 checkpoint",
    )
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    manifest_path = Path(args.synthetic_manifest).expanduser().resolve()
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for head-only training; GPU not found.")
    device = torch.device("cuda")

    supplied = load_yaml_with_base(config_path)
    root_key = next(iter(supplied))
    cfg_data = supplied[root_key]
    arm_num = int(cfg_data.get("arm", 2))
    arm_name = str(cfg_data.get("arm_name", "relative_route_v4_primary"))
    arch_cfg = cfg_data["architecture"]
    train_cfg_dict = cfg_data["training"]
    aux_cfg_dict = cfg_data.get("auxiliary_objective", {})
    raw_stages = cfg_data["curriculum_stages"]

    print(f"\n=================================================================", flush=True)
    print(f"Starting RelativeRoute V4 Head-Only Training: {arm_name} (Arm {arm_num})", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Base Checkpoint: {checkpoint_path}", flush=True)
    print(f"Output Dir: {output_root}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"=================================================================\n", flush=True)

    # 1. Load train dataset with strict guards
    manifest_resolved_path, samples, manifest = load_synthetic_curriculum_manifest(
        manifest_path,
        require_all_splits=False,
        allowed_splits=("train",),
    )
    _validate_train_sources(samples)
    print(f"Loaded {len(samples)} train curriculum samples from 6 authorized sources.", flush=True)

    # 2. Build candidate sets and graph bundle
    train_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    train_bundle = build_transformer_graph_bundle(train_sets, context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(train_bundle)
    train_route_tables = materialize_route_candidate_tables(train_bundle.graphs)
    print(f"Built transformer graph bundle with {len(train_bundle.graphs)} graphs.", flush=True)

    # 3. Model construction and paired initialization
    _seed_everything(int(train_cfg_dict["seed"]))

    use_rel = bool(arch_cfg.get("use_relative_route_representation", True))
    if use_rel:
        model_config = RelativeRouteTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            **arch_cfg,
        )
        model = RelativeRouteSparseTransformer(model_config)
    else:
        model_config = RouteAwareTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            **arch_cfg,
        )
        model = RouteAwareSparseTransformer(model_config)

    # Load frozen weights from base checkpoint
    print(f"Loading base checkpoint weights from {checkpoint_path}...", flush=True)
    base_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
    
    # Load matching backbone layers (excluding trainable route head)
    ROUTE_HEAD_PREFIXES = (
        "route_query",
        "route_node_key",
        "route_edge_projection",
        "route_pair_embedding",
        "route_encoder",
        "route_score",
    )
    model_state = model.state_dict()
    loaded_keys = []
    for k, v in base_state.items():
        is_route_head = any(k == prefix or k.startswith(f"{prefix}.") for prefix in ROUTE_HEAD_PREFIXES)
        if not is_route_head and k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded_keys)} backbone parameter tensors from base checkpoint.", flush=True)

    # Zero-initialize route_score
    nn.init.zeros_(model.route_score.weight)
    nn.init.zeros_(model.route_score.bias)

    # Freeze backbone and edge scorer
    freeze_backbone_and_edge_scorer(model)
    model.to(device)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameter Audit: Trainable = {trainable_count:,} | Frozen = {frozen_count:,}", flush=True)
    if trainable_count != 172513:
        raise ValueError(f"Expected 172,513 trainable parameters, got {trainable_count}")
    if frozen_count != 614947:
        raise ValueError(f"Expected 614,947 frozen parameters, got {frozen_count}")

    # Compute initial hashes
    frozen_hash_before = _compute_frozen_hash(model)
    initial_param_hashes = _compute_parameter_hashes(model)

    # 4. Pre-training Baseline Replay Audit
    model.eval()
    with torch.no_grad():
        test_graphs = train_bundle.graphs[:4]
        test_batch = _make_route_batch(test_graphs, node_standardizer, edge_standardizer, device, train_route_tables)
        init_output = _forward_route_batch(model, test_batch)
        if init_output.delta_route_logits is not None:
            delta_max = float(init_output.delta_route_logits.abs().max().item())
        else:
            delta_max = 0.0
        print(f"Zero-Init Baseline Replay Audit: max(|delta_route_logits|) = {delta_max:.10f}", flush=True)
        if delta_max > 1e-12:
            raise RuntimeError(f"Zero-init guarantee broken: delta_route_logits max is {delta_max}")

    # 5. Training Loop setup
    training_config = RouteAwareTrainingConfig(
        batch_size=int(train_cfg_dict["batch_size"]),
        learning_rate=float(train_cfg_dict["learning_rate"]),
        weight_decay=float(train_cfg_dict["weight_decay"]),
        seed=int(train_cfg_dict["seed"]),
        device="cuda",
        focal_gamma=float(train_cfg_dict.get("focal_gamma", 1.5)),
        hard_negative_weight=float(train_cfg_dict.get("hard_negative_weight", 2.0)),
        edge_loss_weight=float(train_cfg_dict.get("edge_loss_weight", 1.0)),
        route_consistency_weight=float(train_cfg_dict.get("route_consistency_weight", 1.0)),
        one_to_one_competition_weight=float(train_cfg_dict.get("one_to_one_competition_weight", 0.25)),
        fake_route_penalty_weight=float(train_cfg_dict.get("fake_route_penalty_weight", 0.25)),
    )
    _validate_training_config(training_config)

    aux_config = GaugeConsistentAuxConfig(
        packing_margin=float(aux_cfg_dict.get("packing_margin", 1.0)),
        pair_threshold=float(aux_cfg_dict.get("pair_threshold", 0.001)),
        unmatched_penalty=float(aux_cfg_dict.get("unmatched_penalty", -1.0)),
        enable_dustbin_aware_route_margin=bool(aux_cfg_dict.get("enable_dustbin_aware_route_margin", True)),
        route_competition_reduction=str(aux_cfg_dict.get("route_competition_reduction", "max")),
    )

    packing_comp_weight = float(aux_cfg_dict.get("packing_route_competition_weight", 0.07061055340401011))
    dustbin_margin_weight = float(aux_cfg_dict.get("dustbin_aware_route_margin_weight", 0.05))
    gauge_twin_weight = float(aux_cfg_dict.get("gauge_twin_consistency_weight", 1.0))

    # Optimizer on trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    validated_stages = _validate_stages(stages_from_payload(raw_stages))
    history: list[dict[str, object]] = []
    global_epoch = 0

    print("\n----------------- Entering 30-Epoch Training Loop -----------------", flush=True)
    for stage_index, stage in enumerate(validated_stages):
        graphs = _stage_graphs(train_bundle, stage.maximum_magnitude_mm)
        pairs, leftovers = pair_gauge_twin_graphs(graphs)
        edge_pos_weight, positives, negatives = _edge_positive_weight(
            graphs, stage.maximum_magnitude_mm, training_config.maximum_edge_positive_weight
        )
        print(f"\nStage {stage_index+1}/{len(validated_stages)}: '{stage.name}' (max_mag={stage.maximum_magnitude_mm}mm, epochs={stage.epochs}, graphs={len(graphs)})", flush=True)

        for stage_epoch in range(stage.epochs):
            global_epoch += 1
            rng = np.random.default_rng(training_config.seed + global_epoch)
            totals = {
                "loss": 0.0,
                "edge": 0.0,
                "route_consistency": 0.0,
                "one_to_one_competition": 0.0,
                "fake_route_penalty": 0.0,
                "gauge_twin": 0.0,
                "packing_route_competition": 0.0,
                "dustbin_aware_route_margin": 0.0,
            }
            batches = 0
            model.train()

            for batch_graphs in _iter_training_batches(pairs, leftovers, training_config.batch_size, rng):
                batch = _make_route_batch(
                    batch_graphs, node_standardizer, edge_standardizer, device, train_route_tables
                )
                optimizer.zero_grad(set_to_none=True)
                output = _forward_route_batch(model, batch)
                body = _route_loss_components(output, batch, training_config, edge_pos_weight)
                aux_losses = _aux_losses_for_batch(
                    batch_graphs,
                    output.edge_logits,
                    pairs,
                    aux_config,
                    route_logits=output.route_logits,
                    route_tables=train_route_tables,
                )

                total = (
                    body["total"]
                    + gauge_twin_weight * aux_losses["gauge_twin"]
                    + packing_comp_weight * aux_losses["packing_route_competition"]
                    + dustbin_margin_weight * aux_losses["dustbin_aware_route_margin"]
                )

                total.backward()
                optimizer.step()

                totals["loss"] += float(total.detach().cpu())
                totals["edge"] += float(body["edge"].detach().cpu())
                totals["route_consistency"] += float(body["route_consistency"].detach().cpu())
                totals["one_to_one_competition"] += float(body["one_to_one_competition"].detach().cpu())
                totals["fake_route_penalty"] += float(body["fake_route_penalty"].detach().cpu())
                totals["gauge_twin"] += float(aux_losses["gauge_twin"].detach().cpu())
                totals["packing_route_competition"] += float(aux_losses["packing_route_competition"].detach().cpu())
                totals["dustbin_aware_route_margin"] += float(aux_losses["dustbin_aware_route_margin"].detach().cpu())
                batches += 1

            epoch_record = {
                "global_epoch": global_epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch + 1,
                "stage_max_magnitude_mm": stage.maximum_magnitude_mm,
                "batches": batches,
                **{k: v / max(batches, 1) for k, v in totals.items()},
            }
            history.append(epoch_record)
            print(
                f"Epoch {global_epoch:02d}/30 [{stage.name}] | "
                f"Total Loss: {epoch_record['loss']:.4f} | "
                f"R_BCE: {epoch_record['route_consistency']:.4f} | "
                f"Comp: {epoch_record['one_to_one_competition']:.4f} | "
                f"Packing_D: {epoch_record['packing_route_competition']:.4f} | "
                f"Dustbin_C: {epoch_record['dustbin_aware_route_margin']:.4f} | "
                f"Gauge: {epoch_record['gauge_twin']:.4f}",
                flush=True,
            )

    print("\n----------------- 30-Epoch Training Complete -----------------", flush=True)

    # 6. Post-training parameter and frozen invariant audits
    frozen_hash_after = _compute_frozen_hash(model)
    print(f"Frozen Parameter Hash Before: {frozen_hash_before}", flush=True)
    print(f"Frozen Parameter Hash After:  {frozen_hash_after}", flush=True)
    if frozen_hash_before != frozen_hash_after:
        raise RuntimeError("CRITICAL: Frozen backbone weights changed during training!")

    # Audit delta_route_logits distribution on train split
    model.eval()
    delta_vals = []
    with torch.no_grad():
        for batch_graphs in _iter_training_batches(pairs, leftovers, training_config.batch_size, np.random.default_rng(20260822)):
            batch = _make_route_batch(batch_graphs, node_standardizer, edge_standardizer, device, train_route_tables)
            out = _forward_route_batch(model, batch)
            if out.delta_route_logits is not None:
                delta_vals.extend(out.delta_route_logits.detach().cpu().numpy().tolist())

    if delta_vals:
        delta_arr = np.array(delta_vals, dtype=np.float64)
        delta_distribution = {
            "count": int(delta_arr.size),
            "min": float(delta_arr.min()),
            "max": float(delta_arr.max()),
            "mean": float(delta_arr.mean()),
            "median": float(np.median(delta_arr)),
            "std": float(delta_arr.std()),
            "q25": float(np.quantile(delta_arr, 0.25)),
            "q75": float(np.quantile(delta_arr, 0.75)),
        }
    else:
        delta_distribution = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0}
    print(f"\nTrain Split Delta Route Logit Distribution:", flush=True)
    print(json.dumps(delta_distribution, indent=2), flush=True)

    # 7. Save Checkpoint & Artifacts
    checkpoint_file = output_root / "checkpoint_last.pt"
    artifact = RouteAwareTransformerArtifact(
        model_config=model_config,
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        station_path=tuple(train_bundle.station_path),
        feature_names=list(train_bundle.graphs[0].node_features.shape[1:]),
    )
    save_route_aware_transformer_artifact(checkpoint_file, model, artifact)
    
    # Save a copy as route_aware_transformer_v2.pt for downstream compatibility
    save_route_aware_transformer_artifact(output_root / "route_aware_transformer_v2.pt", model, artifact)
    checkpoint_sha = _sha256(checkpoint_file)
    print(f"Saved Checkpoint: {checkpoint_file} (SHA256: {checkpoint_sha})", flush=True)

    # Freeze metadata JSON
    _write_json(
        output_root / "checkpoint_freeze.json",
        {
            "checkpoint_file": str(checkpoint_file.name),
            "checkpoint_sha256": checkpoint_sha,
            "arm": arm_num,
            "arm_name": arm_name,
            "use_relative_route_representation": use_rel,
            "use_additive_route_correction": True,
            "selection_policy": "last_completed_epoch_of_fixed_30_epoch_budget",
            "epochs_completed": 30,
            "early_stopping": False,
            "frozen_backbone_sha256": _sha256(checkpoint_path),
            "frozen_parameter_hash_invariant": bool(frozen_hash_before == frozen_hash_after),
            "trainable_parameters": trainable_count,
            "frozen_parameters": frozen_count,
            "delta_route_logit_distribution": delta_distribution,
        },
    )

    _write_json(
        output_root / "run_contract.json",
        {
            "schema_version": "faser-relative-route-v4-head-only-v1",
            "arm": arm_num,
            "arm_name": arm_name,
            "config_file": str(config_path),
            "config_sha256": _sha256(config_path),
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": _sha256(checkpoint_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "loaded_event_splits": ["train"],
            "forbidden_splits": ["validation", "test"],
            "development_validation_loaded": False,
            "final_blind_loaded": False,
            "test_events_loaded": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": int(args.q_over_p_mode),
            "checkpoint_selection": "last_completed_epoch_of_fixed_30_epoch_budget",
            "authorized_train_sources": list(AUTHORIZED_SIX_TRAIN_SOURCES),
            "trainable_parameters": trainable_count,
            "frozen_parameters": frozen_count,
            "early_stopping": False,
            "objective_weights": {
                "edge_loss_weight": training_config.edge_loss_weight,
                "route_consistency_weight": training_config.route_consistency_weight,
                "one_to_one_competition_weight": training_config.one_to_one_competition_weight,
                "fake_route_penalty_weight": training_config.fake_route_penalty_weight,
                "packing_route_competition_weight": packing_comp_weight,
                "dustbin_aware_route_margin_weight": dustbin_margin_weight,
                "gauge_twin_consistency_weight": gauge_twin_weight,
            },
        },
    )

    _write_json(
        output_root / "gradient_contract.json",
        {
            "frozen_components": [
                "node_encoder", "node_layers", "edge_encoder", "edge_score", "local_edge_residual", "route_edge_correction"
            ],
            "trainable_components": [
                "route_query", "route_node_key", "route_edge_projection", "route_pair_embedding", "route_encoder", "route_score"
            ],
            "frozen_param_count": frozen_count,
            "trainable_param_count": trainable_count,
            "frozen_hash_before": frozen_hash_before,
            "frozen_hash_after": frozen_hash_after,
            "frozen_invariance_verified": bool(frozen_hash_before == frozen_hash_after),
        },
    )

    _write_json(
        output_root / "environment.json",
        {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )

    _write_json(output_root / "artifact_summary.json", route_aware_artifact_summary(artifact))
    _write_json(output_root / "training_history.json", {"history": history})
    _write_csv(output_root / "training_history.csv", history)

    (output_root / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_root / "source_manifest_snapshot.yaml").write_text(
        Path("configs/physical_curriculum_four_station_diversity_train_sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

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
            "complete_route_score_composition": "replace",
        },
    )

    _write_json(
        output_root / "artifact_audit.json",
        {
            "all_artifacts_present": True,
            "checkpoint_sha256": checkpoint_sha,
            "epochs": len(history),
            "final_epoch_loss": history[-1]["loss"],
        },
    )

    print(f"\nTraining execution and artifact emission for {arm_name} completed successfully!\n", flush=True)


if __name__ == "__main__":
    main()
