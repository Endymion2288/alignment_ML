#!/usr/bin/env python3
"""Workbook 79: train Arm B and evaluate it against frozen W64 Arm A.

Family source-holdout only.  Does not open 00350, Final Blind, or sealed test.
Does not modify the W64 checkpoint.  Arm C is not implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.mlp_pair_classifier import FeatureStandardizer
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from models.explicit_route_energy import (
    FEATURE_DIM,
    HIDDEN_WIDTH,
    MODEL_CONTRACT,
    ExplicitRouteEnergyScorer,
    raw_route_utilities,
    route_feature_names,
)
from models.route_energy import CANONICAL_ENERGY_VERSION
from training.curriculum_mlp import build_candidate_sets
from training.explicit_route_energy import (
    account_event,
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    refuse_forbidden_experiment_path,
    route_feature_matrix,
    target_mask,
    w64_route_utilities,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.global_route_energy_loss import (
    differentiable_inclusion_gap_hinge,
    loss_augmented_structured_hinge,
)
from training.route_aware_transformer import (
    _forward_route_batch,
    _make_route_batch,
    load_route_aware_transformer_artifact,
    materialize_route_candidate_tables,
)
from training.source_diversity_audit import RESERVED_BLIND_SOURCES, UNUSED_RESERVE_SOURCES
from training.source_transfer_cv import (
    FAMILY1_DSID_PAIR,
    FAMILY2_DSID_PAIR,
    V5A_SOURCE_FAMILIES,
    validate_fold_sources,
)


W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
W64_ROOT = Path("outputs/mc24_four_station_source_diversity_v1/checkpoint")
FAMILY_CORPUS = {
    FAMILY1_DSID_PAIR: Path(
        "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    ),
    FAMILY2_DSID_PAIR: Path(
        "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family2/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    ),
}
FOLD_TRAIN_FAMILY = {
    "holdout_family1": FAMILY2_DSID_PAIR,
    "holdout_family2": FAMILY1_DSID_PAIR,
}
FOLD_HOLDOUT_FAMILY = {
    "holdout_family1": FAMILY1_DSID_PAIR,
    "holdout_family2": FAMILY2_DSID_PAIR,
}
UNMATCHED_PENALTY = -1.0
VAL_PAYLOAD_IDS = frozenset({"iteration_00_draw_01", "iteration_00_draw_01_plus_common"})
SEED = 20260905
EPOCHS = 20
PATIENCE = 6
LEARNING_RATE = 3.0e-4
LAI_MARGIN = 1.0
INCLUSION_MARGIN = 0.0


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_family_bundle(family: str):
    manifest = FAMILY_CORPUS[family]
    refuse_forbidden_experiment_path(manifest)
    expected = V5A_SOURCE_FAMILIES[family]
    _, samples, raw = load_synthetic_curriculum_manifest(
        manifest, require_all_splits=False, allowed_splits=("train",)
    )
    constituents = {str(source) for sample in samples for source in sample.source_ids}
    if constituents & (set(RESERVED_BLIND_SOURCES) | set(UNUSED_RESERVE_SOURCES)):
        raise SystemExit(f"{family} corpus leaked reserved/blind sources")
    if constituents != set(expected):
        raise SystemExit(f"{family} corpus sources {sorted(constituents)} != {list(expected)}")
    for sample in samples:
        refuse_forbidden_experiment_path(sample.synthetic_tracklets)
        refuse_forbidden_experiment_path(sample.field_candidates)
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1", q_over_p_mode=0
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    return samples, bundle, raw


def _split_graphs(bundle):
    train = []
    validation = []
    for graph in bundle.graphs:
        if str(graph.sample.payload_id) in VAL_PAYLOAD_IDS:
            validation.append(graph)
        else:
            train.append(graph)
    if not train or not validation:
        raise SystemExit("train/val payload split is empty; check draw_01 payloads")
    return train, validation


def _w64_base_logits(bundle, model, artifact, device: torch.device, batch_size: int) -> dict[tuple[int, int], float]:
    tables = materialize_route_candidate_tables(bundle.graphs)
    logits: dict[tuple[int, int], float] = {}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            graphs = bundle.graphs[start : start + batch_size]
            batch = _make_route_batch(graphs, artifact.node_standardizer, artifact.edge_standardizer, device, tables)
            output = _forward_route_batch(model, batch)
            values = output.base_edge_logits.detach().cpu().numpy().astype(np.float64)
            for value, owner, row in zip(values, batch.base.score_owner, batch.base.score_row):
                key = (int(owner), int(row))
                if key in logits:
                    raise RuntimeError("W64 base-logit reconstruction duplicated a candidate row")
                logits[key] = float(value)
    return logits


def _event_loss(model, standardizer, graph, unmatched_penalty: float, device: torch.device):
    routes = enumerate_contiguous_physical_routes(graph)
    mask = target_mask(routes)
    if not routes or not np.any(mask):
        return None
    features = standardizer.transform(route_feature_matrix(graph, routes))
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    cores = model(tensor)
    n_stations = [len(route.stations) for route in routes]
    utilities = raw_route_utilities(cores, n_stations, unmatched_penalty)
    table = energy_table_from_utilities(
        routes,
        utilities.detach().cpu().tolist(),
        unmatched_penalty=unmatched_penalty,
    )
    lai = loss_augmented_structured_hinge(
        utilities,
        [record.endpoints for record in table.records],
        mask.tolist(),
        margin=LAI_MARGIN,
    )
    target_ids = [index for index, flag in enumerate(mask.tolist()) if flag]
    inclusion = differentiable_inclusion_gap_hinge(
        utilities, table, target_ids, margin=INCLUSION_MARGIN
    )
    return {
        "loss": lai["loss"] + inclusion,
        "lai": float(lai["loss"].detach().cpu()),
        "inclusion": float(inclusion.detach().cpu()),
        "n_routes": len(routes),
        "n_targets": int(len(target_ids)),
    }


def _mean_val_loss(model, standardizer, graphs, unmatched_penalty: float, device: torch.device) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for graph in graphs:
            report = _event_loss(model, standardizer, graph, unmatched_penalty, device)
            if report is not None:
                values.append(float(report["loss"].detach().cpu()))
    if not values:
        raise SystemExit("validation split produced no structured-loss events")
    return float(np.mean(values))


def _fit_standardizer(graphs) -> FeatureStandardizer:
    blocks = []
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if routes:
            blocks.append(route_feature_matrix(graph, routes))
    if not blocks:
        raise SystemExit("no train routes to fit the Arm B standardizer")
    return FeatureStandardizer.fit(np.concatenate(blocks, axis=0))


def _evaluate_arm(name: str, utilities_fn, graphs, adjacent_sets) -> dict[str, Any]:
    accounting = None
    n_events = 0
    n_routes = 0
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        utilities = utilities_fn(graph, routes)
        table = energy_table_from_utilities(routes, utilities, unmatched_penalty=UNMATCHED_PENALTY)
        block = account_event(graph, routes, table, adjacent_sets)
        if accounting is None:
            accounting = block
        else:
            accounting.add(block)
        n_events += 1
        n_routes += len(routes)
    if accounting is None:
        raise SystemExit(f"{name} produced no evaluable events")
    payload = accounting.as_dict()
    payload.update(
        {
            "arm": name,
            "utility_contract": CANONICAL_ENERGY_VERSION,
            "metric_version": payload["metric_version"],
            "n_scored_events": n_events,
            "n_enumerated_routes": n_routes,
        }
    )
    return payload


def train_arm_b(train_graphs, val_graphs, device: torch.device, output: Path) -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    standardizer = _fit_standardizer(train_graphs)
    model = ExplicitRouteEnergyScorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_state = None
    best_val = None
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for graph in train_graphs:
            report = _event_loss(model, standardizer, graph, UNMATCHED_PENALTY, device)
            if report is None:
                continue
            optimizer.zero_grad()
            report["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(report["loss"].detach().cpu()))
        val_loss = _mean_val_loss(model, standardizer, val_graphs, UNMATCHED_PENALTY, device)
        row = {
            "epoch": epoch,
            "train_loss": None if not epoch_losses else float(np.mean(epoch_losses)),
            "validation_loss": val_loss,
            "n_train_events": len(epoch_losses),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise SystemExit("Arm B training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    artifact = {
        "model_contract": MODEL_CONTRACT,
        "feature_version": "raw_physical_route_v1",
        "feature_names": list(route_feature_names()),
        "feature_dim": FEATURE_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "unmatched_penalty": UNMATCHED_PENALTY,
        "seed": SEED,
        "standardizer_mean": standardizer.mean.tolist(),
        "standardizer_scale": standardizer.scale.tolist(),
        "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
        "best_validation_loss": best_val,
        "history": history,
    }
    _write_json(output / "arm_b_checkpoint.json", artifact)
    torch.save(
        {
            "model_contract": MODEL_CONTRACT,
            "state_dict": best_state,
            "standardizer_mean": standardizer.mean,
            "standardizer_scale": standardizer.scale,
        },
        output / "arm_b_checkpoint.pt",
    )
    return {"model": model, "standardizer": standardizer, "artifact": artifact}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLD_TRAIN_FAMILY), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-root",
        default="outputs/mc24_four_station_explicit_route_energy_v1",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    train_family = FOLD_TRAIN_FAMILY[args.fold]
    holdout_family = FOLD_HOLDOUT_FAMILY[args.fold]
    train_sources = V5A_SOURCE_FAMILIES[train_family]
    holdout_sources = V5A_SOURCE_FAMILIES[holdout_family]
    validate_fold_sources(train_sources, holdout_sources)

    output = Path(args.output_root) / args.fold
    output.mkdir(parents=True, exist_ok=True)

    w64_checkpoint = W64_ROOT / "route_aware_transformer_v2.pt"
    w64_sha = _sha256(w64_checkpoint)
    if w64_sha != W64_SHA:
        raise SystemExit(f"W64 checkpoint sha {w64_sha} != frozen {W64_SHA}")

    print(f"loading train family {train_family}", flush=True)
    train_samples, train_bundle, train_raw = _load_family_bundle(train_family)
    train_graphs, val_graphs = _split_graphs(train_bundle)
    n_train_samples = len(train_samples)
    n_train_graphs = len(train_graphs)
    n_val_graphs = len(val_graphs)
    train_overlay_seed = (train_raw.get("synthetic_multitrack") or {}).get("seed")

    print("training Arm B", flush=True)
    trained = train_arm_b(train_graphs, val_graphs, device, output)
    model = trained["model"]
    standardizer = trained["standardizer"]
    model.eval()
    del train_samples, train_bundle, train_raw, train_graphs, val_graphs

    print(f"loading holdout family {holdout_family}", flush=True)
    hold_samples, hold_bundle, hold_raw = _load_family_bundle(holdout_family)
    print("extracting frozen W64 base-edge logits (no sigmoid/clip decode)", flush=True)
    model_w64, artifact_w64 = load_route_aware_transformer_artifact(w64_checkpoint, device=str(device))
    hold_logits = _w64_base_logits(hold_bundle, model_w64, artifact_w64, device, args.batch_size)

    def arm_a_utilities(graph, routes):
        return w64_route_utilities(graph, routes, hold_logits, UNMATCHED_PENALTY)

    def arm_b_utilities(graph, routes):
        features = standardizer.transform(route_feature_matrix(graph, routes))
        with torch.no_grad():
            cores = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
        n_stations = np.asarray([len(route.stations) for route in routes], dtype=np.float64)
        return cores + n_stations * UNMATCHED_PENALTY

    print("evaluating Arm A / Arm B on holdout family", flush=True)
    arm_a = _evaluate_arm("A_frozen_w64_energy", arm_a_utilities, hold_bundle.graphs, hold_bundle.adjacent_sets)
    arm_b = _evaluate_arm("B_raw_physical_route_energy", arm_b_utilities, hold_bundle.graphs, hold_bundle.adjacent_sets)

    dataset_manifest = {
        "schema": "faser-dataset-contract-v2",
        "experiment": "wb79_explicit_route_energy_v1",
        "fold": args.fold,
        "train_family": train_family,
        "holdout_family": holdout_family,
        "train_sources": list(train_sources),
        "holdout_sources": list(holdout_sources),
        "train_manifest": str(FAMILY_CORPUS[train_family]),
        "holdout_manifest": str(FAMILY_CORPUS[holdout_family]),
        "validation_payload_ids": sorted(VAL_PAYLOAD_IDS),
        "development_00350_used": False,
        "final_blind_accessed": False,
        "sealed_test_accessed": False,
        "geometry_holdout_refit_used": False,
    }
    resolved_config = {
        "workbook": 79,
        "model_contract": MODEL_CONTRACT,
        "utility_contract": CANONICAL_ENERGY_VERSION,
        "metric_version": "route_accounting_v2",
        "solver": "exact_unit_capacity",
        "loss": ["loss_augmented_structured_hinge", "differentiable_inclusion_gap_hinge"],
        "unmatched_penalty": UNMATCHED_PENALTY,
        "hidden_width": HIDDEN_WIDTH,
        "seed": SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
        "arm_c_authorized": False,
        "legacy_decoder_used": False,
    }
    ancestry = {
        "w64_checkpoint": str(W64_ROOT / "route_aware_transformer_v2.pt"),
        "w64_sha256": w64_sha,
        "w64_used_as": "frozen_arm_a_baseline_and_not_trained",
        "arm_b_parent": None,
        "historical_v5a_head_used": False,
    }
    environment = {
        "hostname": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "cuda": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_sha": _git_sha(),
    }
    comparison = {
        "fold": args.fold,
        "holdout_family": holdout_family,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "delta_complete_efficiency_b_minus_a": (
            None
            if arm_a.get("complete_track_efficiency") is None or arm_b.get("complete_track_efficiency") is None
            else float(arm_b["complete_track_efficiency"] - arm_a["complete_track_efficiency"])
        ),
        "delta_complete_fake_rate_b_minus_a": (
            None
            if arm_a.get("complete_fake_rate") is None or arm_b.get("complete_fake_rate") is None
            else float(arm_b["complete_fake_rate"] - arm_a["complete_fake_rate"])
        ),
        "delta_all_route_purity_b_minus_a": (
            None
            if arm_a.get("all_route_purity") is None or arm_b.get("all_route_purity") is None
            else float(arm_b["all_route_purity"] - arm_a["all_route_purity"])
        ),
        "w64_saw_all_six_train_sources": True,
        "arm_b_trained_only_on_complement_family": True,
    }
    contract_gate = {
        "gate_pass": None,
        "note": "WB79 first phase reports A vs B; it does not promote a production operating point",
        "required_metric_version": "route_accounting_v2",
        "arm_a_metric_version": arm_a.get("metric_version"),
        "arm_b_metric_version": arm_b.get("metric_version"),
        "development_00350_used": False,
        "final_blind_accessed": False,
        "sealed_test_accessed": False,
        "legacy_decoder_used": False,
    }
    metadata = {
        "workbook": 79,
        "git_sha": _git_sha(),
        "fold": args.fold,
        "training_authorized": True,
        "final_blind_eval_authorized": False,
        "sealed_test_accessed": False,
        "continue_to_v5a_frozen_head": False,
        "continue_to_15d_relative_wls": False,
        "arm_c_authorized": False,
        "resolved_config": resolved_config,
        "dataset_manifest": dataset_manifest,
        "checkpoint_ancestry": ancestry,
        "n_train_graphs": n_train_graphs,
        "n_val_graphs": n_val_graphs,
        "n_holdout_graphs": len(hold_bundle.graphs),
        "n_train_samples": n_train_samples,
        "n_holdout_samples": len(hold_samples),
        "train_overlay_seed": train_overlay_seed,
        "holdout_overlay_seed": (hold_raw.get("synthetic_multitrack") or {}).get("seed"),
    }
    _write_json(output / "dataset_manifest.json", dataset_manifest)
    _write_json(output / "resolved_config.json", resolved_config)
    _write_json(output / "checkpoint_ancestry.json", ancestry)
    _write_json(output / "environment.json", environment)
    _write_json(output / "run_metadata.json", metadata)
    _write_json(output / "evaluation.json", comparison)
    _write_json(output / "contract_gate.json", contract_gate)
    print(json.dumps({"output": str(output), "comparison": comparison}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
