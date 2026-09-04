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
import os
import socket
import subprocess
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
    RelativeRouteV4Inference,
    RouteAwareTransformerConfig,
    RouteAwareSparseTransformer,
    freeze_backbone_and_edge_scorer,
)
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    CurriculumStage,
    TransformerGraph,
    TransformerGraphBundle,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
    graph_bundle_summary,
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
    load_route_aware_transformer_artifact,
    materialize_route_candidate_tables,
    route_aware_artifact_summary,
    save_relative_route_v4_head_only_artifact,
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


def _validate_train_sources(
    samples: Sequence[CurriculumSample],
    expected_sources: Sequence[str] | None = None,
) -> None:
    if {sample.split for sample in samples} != {"train"}:
        raise ValueError("Head-only training opened a non-train split")
    constituents = {str(source) for sample in samples for source in sample.source_ids}
    leaking = constituents & FORBIDDEN_TRAIN_SOURCES
    leaking.update(source for source in constituents if is_sealed_source(source))
    if leaking:
        raise ValueError(
            f"Forbidden / Development / Blind sources leaked into training: {', '.join(sorted(leaking))}"
        )
    if expected_sources is None:
        # Historical Workbook-69 behaviour: the full six-source train corpus.
        expected = set(AUTHORIZED_SIX_TRAIN_SOURCES)
    else:
        # Workbook-72 source-transfer CV: a fold trains on a strict subset of the
        # six authorized sources (the held-out family must be absent).  Requiring
        # constituents == expected fold sources IS the source-holdout guard.
        expected = {str(s) for s in expected_sources}
        if not expected <= set(AUTHORIZED_SIX_TRAIN_SOURCES):
            raise ValueError(
                "Fold train sources must be a subset of the authorized six train sources. "
                f"Got: {', '.join(sorted(expected))}"
            )
    if constituents != expected:
        raise ValueError(
            f"Training set must match expected train sources. Got: {', '.join(sorted(constituents))}"
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
    parser.add_argument(
        "--expected-train-sources",
        default=None,
        help=(
            "Optional comma-separated list of train source ids for Workbook-72 "
            "source-transfer CV folds.  When omitted the full authorized six-source "
            "set is required (historical Workbook-69 behaviour)."
        ),
    )
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
    expected_sources = (
        [s.strip() for s in args.expected_train_sources.split(",") if s.strip()]
        if args.expected_train_sources
        else None
    )
    _validate_train_sources(samples, expected_sources=expected_sources)
    print(
        f"Loaded {len(samples)} train curriculum samples from "
        f"{len(expected_sources) if expected_sources else 6} authorized sources.",
        flush=True,
    )

    # 2. Build candidate sets and graph bundle
    train_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    train_bundle = build_transformer_graph_bundle(train_sets, context_mode="full_event")
    train_route_tables = materialize_route_candidate_tables(train_bundle.graphs)
    print(f"Built transformer graph bundle with {len(train_bundle.graphs)} graphs.", flush=True)

    # 3. Frozen Workbook-64 replica + trainable additive route head
    print(f"Loading frozen Workbook-64 replica from {checkpoint_path}...", flush=True)
    frozen_w64, frozen_artifact = load_route_aware_transformer_artifact(checkpoint_path, device="cpu")
    if frozen_w64.config.use_additive_route_correction:
        raise RuntimeError("Workbook-64 replica unexpectedly uses additive route correction")
    if frozen_artifact.node_standardizer is None or frozen_artifact.edge_standardizer is None:
        raise RuntimeError("Workbook-64 checkpoint is missing train-only standardizers")
    node_standardizer = frozen_artifact.node_standardizer
    edge_standardizer = frozen_artifact.edge_standardizer
    refit_node, refit_edge = fit_graph_standardizers(train_bundle)
    node_std_delta = float(np.max(np.abs(refit_node.mean - node_standardizer.mean)))
    edge_std_delta = float(np.max(np.abs(refit_edge.mean - edge_standardizer.mean)))
    print(
        f"Using frozen Workbook-64 standardizers. Six-source refit max-abs mean delta: "
        f"node={node_std_delta:.6e} edge={edge_std_delta:.6e}",
        flush=True,
    )

    _seed_everything(int(train_cfg_dict["seed"]))
    use_rel = bool(arch_cfg.get("use_relative_route_representation", True))
    if use_rel:
        model_config = RelativeRouteTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            **arch_cfg,
        )
        trainable = RelativeRouteSparseTransformer(model_config)
    else:
        model_config = RouteAwareTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            **arch_cfg,
        )
        trainable = RouteAwareSparseTransformer(model_config)
    if not model_config.use_additive_route_correction:
        raise RuntimeError("Head-only V4 configs must set use_additive_route_correction=true")

    ROUTE_HEAD_PREFIXES = (
        "route_query",
        "route_node_key",
        "route_edge_projection",
        "route_pair_embedding",
        "route_encoder",
        "route_score",
    )
    frozen_state = frozen_w64.state_dict()
    trainable_state = trainable.state_dict()
    loaded_keys = []
    for key, value in frozen_state.items():
        is_route_head = any(key == prefix or key.startswith(f"{prefix}.") for prefix in ROUTE_HEAD_PREFIXES)
        if not is_route_head and key in trainable_state and trainable_state[key].shape == value.shape:
            trainable_state[key] = value
            loaded_keys.append(key)
    trainable.load_state_dict(trainable_state)
    print(f"Copied {len(loaded_keys)} frozen backbone/edge tensors into the trainable replica.", flush=True)
    nn.init.zeros_(trainable.route_score.weight)
    nn.init.zeros_(trainable.route_score.bias)
    freeze_backbone_and_edge_scorer(trainable)
    for param in frozen_w64.parameters():
        param.requires_grad = False
    frozen_w64.eval()

    model = RelativeRouteV4Inference(frozen_w64, trainable)
    model.to(device)

    trainable_count = sum(p.numel() for p in trainable.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in trainable.parameters() if not p.requires_grad)
    print(f"Parameter Audit: Trainable = {trainable_count:,} | Frozen = {frozen_count:,}", flush=True)
    if trainable_count != 172513:
        raise ValueError(f"Expected 172,513 trainable parameters, got {trainable_count}")
    if frozen_count != 614947:
        raise ValueError(f"Expected 614,947 frozen parameters, got {frozen_count}")

    frozen_hash_before = _compute_frozen_hash(trainable)
    frozen_w64_hash_before = _compute_frozen_hash(frozen_w64)
    initial_param_hashes = _compute_parameter_hashes(trainable)

    # 4. Pre-training Baseline Replay Audit
    model.eval()
    with torch.no_grad():
        test_graphs = train_bundle.graphs[:4]
        test_batch = _make_route_batch(test_graphs, node_standardizer, edge_standardizer, device, train_route_tables)
        init_output = _forward_route_batch(model, test_batch)
        frozen_output = _forward_route_batch(frozen_w64, test_batch)
        delta_max = (
            0.0
            if init_output.delta_route_logits is None
            else float(init_output.delta_route_logits.abs().max().item())
        )
        edge_delta = float((init_output.edge_logits - frozen_output.edge_logits).abs().max().item())
        print(f"Zero-Init Baseline Replay Audit: max(|delta_route_logits|) = {delta_max:.10f}", flush=True)
        print(f"Frozen W64 edge identity: max(|edge_v4 - edge_w64|) = {edge_delta:.10e}", flush=True)
        if delta_max > 1e-12:
            raise RuntimeError(f"Zero-init guarantee broken: delta_route_logits max is {delta_max}")
        # Two separate GPU attention forwards are not bit-identical in float32.
        # 1e-5 covers observed H100 MIG residuals (~1e-6) without relaxing the
        # scientific claim that wrapper edges are the frozen Workbook-64 edges.
        if edge_delta > 1.0e-5:
            raise RuntimeError("Head-only wrapper does not reproduce Workbook-64 production edges")
        if int(init_output.route_logits.numel()):
            production_edge_sum = init_output.edge_logits[test_batch.route_score_edge_indices].sum(dim=-1)
            same_forward_delta = init_output.delta_route_logits
            if same_forward_delta is None:
                same_forward_delta = production_edge_sum.new_zeros(production_edge_sum.shape)
            route_delta = float((init_output.route_logits - production_edge_sum - same_forward_delta).abs().max().item())
            print(
                f"Zero-init L_corrected identity (same forward): "
                f"max(|L_corrected - L_edge_W64 - delta|) = {route_delta:.10e}",
                flush=True,
            )
            if route_delta > 1e-12:
                raise RuntimeError("L_corrected is not L_edge_W64 + delta_route_logit on the wrapper forward")

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
    frozen_hash_after = _compute_frozen_hash(trainable)
    frozen_w64_hash_after = _compute_frozen_hash(frozen_w64)
    print(f"Trainable-replica frozen hash before: {frozen_hash_before}", flush=True)
    print(f"Trainable-replica frozen hash after:  {frozen_hash_after}", flush=True)
    print(f"Workbook-64 replica hash before: {frozen_w64_hash_before}", flush=True)
    print(f"Workbook-64 replica hash after:  {frozen_w64_hash_after}", flush=True)
    if frozen_hash_before != frozen_hash_after:
        raise RuntimeError("CRITICAL: Frozen backbone weights in the trainable replica changed!")
    if frozen_w64_hash_before != frozen_w64_hash_after:
        raise RuntimeError("CRITICAL: Frozen Workbook-64 replica weights changed!")

    # Audit delta_route_logits distribution and frozen-edge identity on train split
    model.eval()
    delta_vals = []
    edge_max_abs = 0.0
    with torch.no_grad():
        for batch_graphs in _iter_training_batches(pairs, leftovers, training_config.batch_size, np.random.default_rng(20260822)):
            batch = _make_route_batch(batch_graphs, node_standardizer, edge_standardizer, device, train_route_tables)
            out = _forward_route_batch(model, batch)
            frozen_out = _forward_route_batch(frozen_w64, batch)
            edge_max_abs = max(
                edge_max_abs,
                float((out.edge_logits - frozen_out.edge_logits).abs().max().item()),
            )
            if out.delta_route_logits is not None:
                delta_vals.extend(out.delta_route_logits.detach().cpu().numpy().tolist())
    print(f"Post-training frozen W64 edge identity: max(|edge_v4 - edge_w64|) = {edge_max_abs:.10e}", flush=True)
    if edge_max_abs > 1.0e-5:
        raise RuntimeError("Post-training production edges drifted from Workbook-64")

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
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=train_bundle.all_station_pairs,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=model_config,
        context_mode="full_event",
        training_summary={
            "arm": arm_num,
            "arm_name": arm_name,
            "best_global_epoch": 30,
            "training_global_epochs_completed": 30,
            "checkpoint_selection": "last_completed_epoch_of_fixed_30_epoch_budget",
            "development_validation_used": False,
            "device": str(device),
            "train_bundle": graph_bundle_summary(train_bundle),
            "loss": {
                "edge": "weighted_focal_binary_cross_entropy_frozen_workbook64_edges",
                "route_truth_consistency": "weighted_focal_binary_cross_entropy_on_L_corrected",
                "one_to_one_competition": "endpoint_incident_route_logsumexp_on_L_corrected",
                "fake_route_penalty": "softplus_on_fake_endpoint_routes_on_L_corrected",
                "gauge_twin_consistency": "mse_origin_matched_raw_logits_and_complete_route_utilities",
                "packing_route_competition": "relu_local_packing_utility_margin",
                "dustbin_aware_route_margin": "relu_max_fragment_or_dustbin_plus_margin_minus_truth",
                "complete_route_logit": "L_edge_W64_plus_delta_route_logit",
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
    frozen_w64_sha = _sha256(checkpoint_path)
    save_relative_route_v4_head_only_artifact(
        checkpoint_file,
        frozen_w64,
        trainable,
        artifact,
        frozen_workbook64_sha256=frozen_w64_sha,
    )
    save_relative_route_v4_head_only_artifact(
        output_root / "relative_route_v4_head_only.pt",
        frozen_w64,
        trainable,
        artifact,
        frozen_workbook64_sha256=frozen_w64_sha,
    )
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
            "frozen_backbone_sha256": frozen_w64_sha,
            "frozen_parameter_hash_invariant": bool(frozen_hash_before == frozen_hash_after),
            "frozen_workbook64_replica_hash_invariant": bool(frozen_w64_hash_before == frozen_w64_hash_after),
            "post_training_max_abs_edge_delta_vs_workbook64": edge_max_abs,
            "trainable_parameters": trainable_count,
            "frozen_parameters": frozen_count,
            "delta_route_logit_distribution": delta_distribution,
            "complete_route_logit": "L_edge_W64_plus_delta_route_logit",
            "schema_version": "faser-relative-route-v4-head-only-v1",
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
            "complete_route_logit": "L_edge_W64_plus_delta_route_logit",
            "continue_to_15d_relative_wls": False,
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
            "frozen_workbook64_hash_before": frozen_w64_hash_before,
            "frozen_workbook64_hash_after": frozen_w64_hash_after,
            "frozen_workbook64_invariance_verified": bool(frozen_w64_hash_before == frozen_w64_hash_after),
            "post_training_max_abs_edge_delta_vs_workbook64": edge_max_abs,
            "complete_route_logit": "L_edge_W64_plus_delta_route_logit",
        },
    )

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
    _write_json(
        output_root / "environment.json",
        {
            "hostname": socket.gethostname(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git_commit": git_commit,
            "git_dirty": bool(git_status.strip()),
            "git_status_porcelain": git_status,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
