#!/usr/bin/env python3
"""Audit gradient flow from V2 body and Workbook-64 solver-aware losses to delta_route_head."""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from baselines.field_chi2_matching import FieldCandidate
from models.route_transformer import (
    RelativeRouteTransformerConfig,
    RelativeRouteSparseTransformer,
    RouteAwareTransformerConfig,
    RouteAwareSparseTransformer,
    freeze_backbone_and_edge_scorer,
)
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
)
from training.route_aware_transformer import (
    _make_route_batch,
    _forward_route_batch,
    _route_loss_components,
    RouteAwareTrainingConfig,
    materialize_route_candidate_tables,
    enumerate_complete_route_candidates,
)
from training.gauge_consistent_route import (
    pair_gauge_twin_graphs,
    GaugeConsistentAuxConfig,
    packing_route_competition_loss,
    dustbin_aware_route_margin_loss,
    gauge_twin_consistency_loss,
    _aux_losses_for_batch,
)


def _sample(payload_id: str = "v4_nominal") -> CurriculumSample:
    return CurriculumSample(
        source_id="v4_test_sample",
        source_ids=("v4_test_sample",),
        split="train",
        payload_id=payload_id,
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("v4_test:1",),
        physical_event_uids=("v4_test:1",),
        physical_tracklets=Path("/tmp/v4_tracklets.root"),
        physical_propagations=Path("/tmp/v4_prop.root"),
        physical_payload_manifest=Path("/tmp/v4_payload.json"),
        synthetic_tracklets=Path("/tmp/v4_synthetic.root"),
        field_candidates=Path("/tmp/v4_candidates.root"),
    )


def _toy_event(run_id: int = 100, event_id: int = 200) -> EventTracklets:
    return EventTracklets(
        run_id=run_id,
        event_id=event_id,
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


def _toy_candidate_sets(payload_id: str = "v4_nominal") -> list[CandidateSet]:
    event = _toy_event()
    sample = _sample(payload_id)
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


def _trainable_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            total_sq += float(p.grad.data.norm(2).item()) ** 2
    return math.sqrt(total_sq)


def _frozen_grad_count(model: nn.Module) -> int:
    return sum(
        1 for p in model.parameters()
        if not p.requires_grad and p.grad is not None and torch.any(p.grad != 0)
    )


def audit_losses(device_name: str = "cpu") -> dict[str, float]:
    device = torch.device(device_name)
    cfg = RelativeRouteTransformerConfig(
        node_feature_dim=17, edge_feature_dim=11, use_relative_route_representation=True
    )
    model = RelativeRouteSparseTransformer(cfg).to(device)
    freeze_backbone_and_edge_scorer(model)

    sets_chart = _toy_candidate_sets("chart_payload")
    sets_twin = _toy_candidate_sets("chart_payload_plus_common")
    all_sets = sets_chart + sets_twin
    bundle = build_transformer_graph_bundle(all_sets, context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)
    tables = materialize_route_candidate_tables(bundle.graphs)

    batch = _make_route_batch(bundle.graphs, node_std, edge_std, device, tables)
    train_cfg = RouteAwareTrainingConfig(device=device_name)
    aux_cfg = GaugeConsistentAuxConfig(
        packing_margin=1.0,
        pair_threshold=0.001,
        unmatched_penalty=-1.0,
        enable_dustbin_aware_route_margin=True,
        route_competition_reduction="max",
    )

    # 1. Forward pass
    output = _forward_route_batch(model, batch)
    body_losses = _route_loss_components(output, batch, train_cfg, edge_positive_weight=1.0)
    
    # Pair twin graphs
    pairs, leftovers = pair_gauge_twin_graphs(bundle.graphs)

    aux_losses = _aux_losses_for_batch(
        bundle.graphs, output.edge_logits, pairs, aux_cfg,
        route_logits=output.route_logits, route_tables=tables
    )

    results = {}

    # Audit each loss component gradient norm on route head
    named_losses = {
        "route_consistency (V2 BCE)": body_losses["route_consistency"],
        "one_to_one_competition (V2 LogSumExp)": body_losses["one_to_one_competition"],
        "fake_route_penalty (V2 Softplus)": body_losses["fake_route_penalty"],
        "packing_route_competition (WB64 / Problem D)": aux_losses["packing_route_competition"],
        "dustbin_aware_route_margin (WB64 / Problem C)": aux_losses["dustbin_aware_route_margin"],
        "gauge_twin_consistency (WB64 Gauge)": aux_losses["gauge_twin"],
    }

    print(f"\n================ Gradient Audit on {device_name.upper()} ================")
    for name, loss_tensor in named_losses.items():
        model.zero_grad(set_to_none=True)
        if loss_tensor.requires_grad:
            loss_tensor.backward(retain_graph=True)
            norm = _trainable_grad_norm(model)
            frozen_leaks = _frozen_grad_count(model)
            print(f"[{name}]")
            print(f"  loss value: {loss_tensor.item():.6f}")
            print(f"  grad_norm(trainable_head): {norm:.6f}")
            print(f"  frozen_param_leaks: {frozen_leaks}")
            results[name] = norm
            assert frozen_leaks == 0, f"Frozen param received grad in {name}!"
        else:
            print(f"[{name}] loss value: {loss_tensor.item():.6f} (no requires_grad)")
            results[name] = 0.0

    # Total composite loss backward
    model.zero_grad(set_to_none=True)
    total_loss = (
        body_losses["total"]
        + 0.07061055340401011 * aux_losses["packing_route_competition"]
        + 0.05 * aux_losses["dustbin_aware_route_margin"]
        + 1.0 * aux_losses["gauge_twin"]
    )
    total_loss.backward()
    total_norm = _trainable_grad_norm(model)
    total_frozen_leaks = _frozen_grad_count(model)
    print(f"\n[Total Composite Objective (WB68A Contract)]")
    print(f"  total loss value: {total_loss.item():.6f}")
    print(f"  grad_norm(trainable_head): {total_norm:.6f}")
    print(f"  frozen_param_leaks: {total_frozen_leaks}")
    results["total_loss_grad_norm"] = total_norm

    assert results["dustbin_aware_route_margin (WB64 / Problem C)"] > 0, "Problem C loss has 0 grad on route head!"
    assert results["packing_route_competition (WB64 / Problem D)"] > 0, "Problem D loss has 0 grad on route head!"
    assert results["total_loss_grad_norm"] > 0, "Total loss has 0 grad on route head!"
    assert total_frozen_leaks == 0, "Frozen backbone leaked gradients!"
    print("\n>>> ALL MANDATORY C/D GRADIENT AUDITS PASSED SUCCESSFULLY! <<<\n")
    return results


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    audit_losses(dev)
