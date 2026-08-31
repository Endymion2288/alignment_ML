from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_transformer import (
    RelativeRouteSparseTransformer,
    RelativeRouteTransformerConfig,
    RouteAwareSparseTransformer,
    RouteAwareTransformerConfig,
    freeze_backbone_and_edge_scorer,
)
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
)
from training.curriculum_mlp import CandidateSet
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    _forward_route_batch,
    _make_route_batch,
    _route_loss_components,
)


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="v4_dry_run_sample",
        source_ids=("v4_dry_run_sample",),
        split="validation",
        payload_id="v4_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("v4_dry:1",),
        physical_event_uids=("v4_dry:1",),
        physical_tracklets=Path("/tmp/v4_tracklets.root"),
        physical_propagations=Path("/tmp/v4_prop.root"),
        physical_payload_manifest=Path("/tmp/v4_payload.json"),
        synthetic_tracklets=Path("/tmp/v4_synthetic.root"),
        field_candidates=Path("/tmp/v4_candidates.root"),
    )


def _toy_event() -> EventTracklets:
    return EventTracklets(
        run_id=100,
        event_id=200,
        station_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.repeat(np.asarray([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64), 2),
        state=np.arange(32, dtype=np.float64).reshape(8, 4),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111111, dtype=np.uint64),
        truth_particle_id=np.tile(np.asarray([101, 202], dtype=np.int64), 4),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
        synthetic_role=np.zeros(8, dtype=np.int8),
    )


def _toy_candidate_sets() -> list[CandidateSet]:
    event = _toy_event()
    sample = _sample()
    result: list[CandidateSet] = []
    for source_station, target_station in ALL_STATION_PAIRS:
        candidates = []
        labels = []
        for source in event.indices_for_station(source_station):
            for target in event.indices_for_station(target_station):
                truth = bool(event.truth_particle_id[source] == event.truth_particle_id[target])
                candidates.append(
                    FieldCandidate(
                        source_index=int(source),
                        target_index=int(target),
                        source_station=source_station,
                        target_station=target_station,
                        chi2=1.0 if truth else 10.0,
                        residual=np.asarray([0.5, -0.5, 0.05, -0.05]),
                        pull=np.asarray([0.5, -0.5, 0.05, -0.05]),
                        combined_covariance=np.eye(4, dtype=np.float64),
                    )
                )
                labels.append(truth)
        result.append(
            CandidateSet(
                sample=sample,
                event=event,
                station_pair=(source_station, target_station),
                candidates=tuple(candidates),
                features=np.zeros((len(candidates), 1), dtype=np.float64),
                labels=np.asarray(labels, dtype=bool),
            )
        )
    return result


def run_dry_run_audit(device_str: str = "auto") -> dict[str, object]:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[DRY-RUN] PyTorch Version: {torch.__version__}")
    print(f"[DRY-RUN] Target Device: {device} (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"[DRY-RUN] GPU Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Build toy bundle and standardizers
    candidate_sets = _toy_candidate_sets()
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)

    # 2. Instantiate Arm 1 (Control) and Arm 2 (Primary)
    config_arm1 = RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        d_model=128,
        ffn_dim=256,
        route_hidden_dim=128,
        use_relative_route_representation=False,
        use_additive_route_correction=True,
    )
    config_arm2 = RelativeRouteTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        d_model=128,
        ffn_dim=256,
        route_hidden_dim=128,
        use_relative_route_representation=True,
        use_additive_route_correction=True,
    )

    model_arm1 = RouteAwareSparseTransformer(config_arm1).to(device)
    model_arm2 = RelativeRouteSparseTransformer(config_arm2).to(device)

    # 3. Apply parameter freeze
    report_arm1 = freeze_backbone_and_edge_scorer(model_arm1)
    report_arm2 = freeze_backbone_and_edge_scorer(model_arm2)

    print(f"[DRY-RUN] Arm 1 Trainable Params: {report_arm1['n_trainable_parameters']}, Frozen: {report_arm1['n_frozen_parameters']}")
    print(f"[DRY-RUN] Arm 2 Trainable Params: {report_arm2['n_trainable_parameters']}, Frozen: {report_arm2['n_frozen_parameters']}")

    assert report_arm1["n_trainable_parameters"] == report_arm2["n_trainable_parameters"], "Arm 1 and Arm 2 trainable param count mismatch!"

    # 4. Dry-run forward, loss and backward for Arm 2 (Primary)
    batch = _make_route_batch(bundle.graphs, node_std, edge_std, device)
    training_cfg = RouteAwareTrainingConfig(
        device=str(device),
        learning_rate=2e-4,
        weight_decay=1e-4,
        seed=20260822,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model_arm2.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    optimizer.zero_grad()

    model_arm2.train()
    output = _forward_route_batch(model_arm2, batch)
    loss_dict = _route_loss_components(output, batch, training_cfg, edge_positive_weight=1.0)
    total_loss = loss_dict["total"]

    print(f"[DRY-RUN] Arm 2 Loss: total={total_loss.item():.4f}, route_consistency={loss_dict['route_consistency'].item():.4f}, competition={loss_dict['one_to_one_competition'].item():.4f}")

    # Backward
    total_loss.backward()

    # 5. Gradient Audit
    frozen_with_grad: list[str] = []
    trainable_without_grad: list[str] = []

    for name, param in model_arm2.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                trainable_without_grad.append(name)
        else:
            if param.grad is not None:
                frozen_with_grad.append(name)

    print(f"[DRY-RUN] Frozen parameters with gradient: {len(frozen_with_grad)}")
    print(f"[DRY-RUN] Trainable parameters without gradient: {len(trainable_without_grad)}")

    assert len(frozen_with_grad) == 0, f"Frozen params received gradient: {frozen_with_grad}"
    assert len(trainable_without_grad) == 0, f"Trainable params missing gradient: {trainable_without_grad}"

    print("[DRY-RUN] SUCCESS: Forward, loss, backward, and parameter isolation verified without saving checkpoints!")
    return {
        "status": "PASS",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "n_trainable_parameters": report_arm2["n_trainable_parameters"],
        "n_frozen_parameters": report_arm2["n_frozen_parameters"],
        "loss_total": float(total_loss.item()),
    }


if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_dry_run_audit(device_arg)
