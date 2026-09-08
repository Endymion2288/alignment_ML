#!/usr/bin/env python3
"""Workbook 80: Arm B failure-mechanism audit and A/B/C offline scorers.

Does not retrain V5A/W64/Arm B.  Does not implement wp4 Arm C.  Does not
open 00350, Final Blind, or sealed test.  Ablation C is a low-capacity
hybrid energy scorer on physical features plus raw W64 edge logits.
"""

from __future__ import annotations

import argparse
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
from models.explicit_route_energy import ExplicitRouteEnergyScorer, raw_route_utilities
from models.hybrid_route_energy import (
    HYBRID_FEATURE_DIM,
    MODEL_CONTRACT,
    HybridRouteEnergyScorer,
    hybrid_feature_names,
)
from models.route_energy import CANONICAL_ENERGY_VERSION
from scripts.train_eval_wb79_explicit_route_energy import (
    EPOCHS,
    FAMILY_CORPUS,
    FOLD_HOLDOUT_FAMILY,
    FOLD_TRAIN_FAMILY,
    INCLUSION_MARGIN,
    LAI_MARGIN,
    LEARNING_RATE,
    PATIENCE,
    SEED,
    UNMATCHED_PENALTY,
    VAL_PAYLOAD_IDS,
    W64_ROOT,
    W64_SHA,
    _evaluate_arm,
    _load_family_bundle,
    _sha256,
    _split_graphs,
    _w64_base_logits,
    _write_json,
)
from training.explicit_route_energy import (
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    refuse_forbidden_experiment_path,
    route_feature_matrix,
    target_mask,
    w64_route_utilities,
)
from training.global_route_energy_loss import (
    differentiable_inclusion_gap_hinge,
    loss_augmented_structured_hinge,
)
from training.route_aware_transformer import load_route_aware_transformer_artifact
from training.source_transfer_cv import V5A_SOURCE_FAMILIES, validate_fold_sources
from training.wb80_failure_mechanism import (
    event_assignment_masks,
    hybrid_feature_matrix,
    interpret_ablation,
    selected_route_rows,
    stratify_holdout,
    truth_margin_rows,
)


WB79_ROOT = Path("outputs/mc24_four_station_explicit_route_energy_v1")
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb80_failure_mechanism_v1")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_arm_b(fold: str, device: torch.device) -> tuple[ExplicitRouteEnergyScorer, FeatureStandardizer]:
    artifact_path = WB79_ROOT / fold / "arm_b_checkpoint.json"
    if not artifact_path.is_file():
        raise SystemExit(f"missing WB79 Arm B checkpoint: {artifact_path}")
    refuse_forbidden_experiment_path(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    model = ExplicitRouteEnergyScorer()
    model.load_state_dict({key: torch.tensor(value) for key, value in payload["state_dict"].items()})
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    standardizer = FeatureStandardizer(
        mean=np.asarray(payload["standardizer_mean"], dtype=np.float64),
        scale=np.asarray(payload["standardizer_scale"], dtype=np.float64),
    )
    return model, standardizer


def _score_arm_b(model, standardizer, graph, routes, device: torch.device) -> np.ndarray:
    features = standardizer.transform(route_feature_matrix(graph, routes))
    with torch.no_grad():
        cores = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    n_stations = np.asarray([len(route.stations) for route in routes], dtype=np.float64)
    return cores + n_stations * UNMATCHED_PENALTY


def _score_arm_c(model, standardizer, graph, routes, logits, device: torch.device) -> np.ndarray:
    features = standardizer.transform(hybrid_feature_matrix(graph, routes, logits))
    with torch.no_grad():
        cores = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    n_stations = np.asarray([len(route.stations) for route in routes], dtype=np.float64)
    return cores + n_stations * UNMATCHED_PENALTY


def _hybrid_event_loss(model, standardizer, graph, logits, device: torch.device):
    routes = enumerate_contiguous_physical_routes(graph)
    mask = target_mask(routes)
    if not routes or not np.any(mask):
        return None
    features = standardizer.transform(hybrid_feature_matrix(graph, routes, logits))
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    cores = model(tensor)
    utilities = raw_route_utilities(cores, [len(route.stations) for route in routes], UNMATCHED_PENALTY)
    table = energy_table_from_utilities(routes, utilities.detach().cpu().tolist(), unmatched_penalty=UNMATCHED_PENALTY)
    lai = loss_augmented_structured_hinge(
        utilities,
        [record.endpoints for record in table.records],
        mask.tolist(),
        margin=LAI_MARGIN,
    )
    target_ids = [index for index, flag in enumerate(mask.tolist()) if flag]
    inclusion = differentiable_inclusion_gap_hinge(utilities, table, target_ids, margin=INCLUSION_MARGIN)
    return {"loss": lai["loss"] + inclusion}


def _mean_hybrid_val_loss(model, standardizer, graphs, logits, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for graph in graphs:
            report = _hybrid_event_loss(model, standardizer, graph, logits, device)
            if report is None:
                continue
            losses.append(float(report["loss"].detach().cpu()))
    if not losses:
        raise SystemExit("hybrid validation split has no complete-truth events")
    return float(np.mean(losses))


def _fit_hybrid_standardizer(graphs, logits) -> FeatureStandardizer:
    blocks = []
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if routes:
            blocks.append(hybrid_feature_matrix(graph, routes, logits))
    if not blocks:
        raise SystemExit("no hybrid training features")
    return FeatureStandardizer.fit(np.concatenate(blocks, axis=0))


def train_arm_c(train_graphs, val_graphs, logits, device: torch.device, output: Path) -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    standardizer = _fit_hybrid_standardizer(train_graphs, logits)
    model = HybridRouteEnergyScorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_state = None
    best_val = None
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for graph in train_graphs:
            report = _hybrid_event_loss(model, standardizer, graph, logits, device)
            if report is None:
                continue
            optimizer.zero_grad()
            report["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(report["loss"].detach().cpu()))
        val_loss = _mean_hybrid_val_loss(model, standardizer, val_graphs, logits, device)
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
        raise SystemExit("Arm C hybrid training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    artifact = {
        "model_contract": MODEL_CONTRACT,
        "feature_version": "raw_physical_plus_w64_logits_v1",
        "feature_names": list(hybrid_feature_names()),
        "feature_dim": HYBRID_FEATURE_DIM,
        "unmatched_penalty": UNMATCHED_PENALTY,
        "seed": SEED,
        "standardizer_mean": standardizer.mean.tolist(),
        "standardizer_scale": standardizer.scale.tolist(),
        "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
        "best_validation_loss": best_val,
        "history": history,
        "not_wp4_arm_c": True,
    }
    _write_json(output / "arm_c_checkpoint.json", artifact)
    torch.save(
        {
            "model_contract": MODEL_CONTRACT,
            "state_dict": best_state,
            "standardizer_mean": standardizer.mean,
            "standardizer_scale": standardizer.scale,
            "not_wp4_arm_c": True,
        },
        output / "arm_c_checkpoint.pt",
    )
    return {"model": model, "standardizer": standardizer, "artifact": artifact}


def _eval_three(graphs, adjacent_sets, logits, arm_b, arm_c, device: torch.device) -> dict[str, Any]:
    model_b, std_b = arm_b
    model_c, std_c = arm_c

    def utilities_a(graph, routes):
        return w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)

    def utilities_b(graph, routes):
        return _score_arm_b(model_b, std_b, graph, routes, device)

    def utilities_c(graph, routes):
        return _score_arm_c(model_c, std_c, graph, routes, logits, device)

    return {
        "A": _evaluate_arm("A_sum_raw_w64_edge_logits", utilities_a, graphs, adjacent_sets),
        "B": _evaluate_arm("B_raw_physical_route_mlp", utilities_b, graphs, adjacent_sets),
        "C": _evaluate_arm("C_physical_plus_w64_logits", utilities_c, graphs, adjacent_sets),
    }


def _audit_holdout(graphs, logits, arm_b, arm_c, device: torch.device) -> dict[str, Any]:
    model_b, std_b = arm_b
    model_c, std_c = arm_c
    length_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    truth_utilities = {"A": [], "B": [], "C": []}
    n_events = 0
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        utilities = {
            "A": w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY),
            "B": _score_arm_b(model_b, std_b, graph, routes, device),
            "C": _score_arm_c(model_c, std_c, graph, routes, logits, device),
        }
        selected = event_assignment_masks(routes, utilities, UNMATCHED_PENALTY)
        length_rows.extend(selected_route_rows(routes, utilities, selected))
        event_truth = truth_margin_rows(routes, utilities, selected, UNMATCHED_PENALTY)
        truth_rows.extend(event_truth)
        for row in event_truth:
            for arm in ("A", "B", "C"):
                truth_utilities[arm].append(float(row[f"U_{arm}"]))
        n_events += 1
    strata = stratify_holdout(length_rows, truth_rows, truth_utilities)
    extra_fakes_b = [
        row
        for row in length_rows
        if row["selected"].get("B") and not row["selected"].get("A") and not row["truth_consistent"]
    ]
    missed_truth_b = [row for row in truth_rows if row.get("selected_A") and not row.get("selected_B")]
    strata.update(
        {
            "n_events": n_events,
            "n_selected_rows": len(length_rows),
            "n_complete_truth_rows": len(truth_rows),
            "b_extra_fake_selected": len(extra_fakes_b),
            "b_extra_fake_near_dustbin_B": int(sum(1 for row in extra_fakes_b if row["near_dustbin"].get("B"))),
            "b_missed_truth_that_A_selected": len(missed_truth_b),
            "b_missed_truth_near_dustbin_A": int(sum(1 for row in missed_truth_b if row.get("near_dustbin_A"))),
        }
    )
    return strata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLD_TRAIN_FAMILY), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
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
    refuse_forbidden_experiment_path(w64_checkpoint)
    refuse_forbidden_experiment_path(output)
    w64_sha = _sha256(w64_checkpoint)
    if w64_sha != W64_SHA:
        raise SystemExit(f"W64 checkpoint sha {w64_sha} != frozen {W64_SHA}")

    arm_b = _load_arm_b(args.fold, device)
    print(f"loading train family {train_family}", flush=True)
    _, train_bundle, train_raw = _load_family_bundle(train_family)
    train_graphs, val_graphs = _split_graphs(train_bundle)
    n_train_graphs = len(train_graphs)
    n_val_graphs = len(val_graphs)
    print("extracting frozen W64 base-edge logits on the train family", flush=True)
    model_w64, artifact_w64 = load_route_aware_transformer_artifact(w64_checkpoint, device=str(device))
    train_logits = _w64_base_logits(train_bundle, model_w64, artifact_w64, device, args.batch_size)
    print("training hybrid Arm C", flush=True)
    trained_c = train_arm_c(train_graphs, val_graphs, train_logits, device, output)
    arm_c = (trained_c["model"], trained_c["standardizer"])
    print("evaluating A/B/C on same-family val payloads", flush=True)
    same_family = _eval_three(val_graphs, train_bundle.adjacent_sets, train_logits, arm_b, arm_c, device)
    del train_bundle, train_graphs, val_graphs, train_logits, model_w64

    print(f"loading holdout family {holdout_family}", flush=True)
    _, hold_bundle, hold_raw = _load_family_bundle(holdout_family)
    print("extracting frozen W64 base-edge logits on the holdout family", flush=True)
    model_w64, artifact_w64 = load_route_aware_transformer_artifact(w64_checkpoint, device=str(device))
    hold_logits = _w64_base_logits(hold_bundle, model_w64, artifact_w64, device, args.batch_size)
    print("evaluating A/B/C on holdout family", flush=True)
    holdout = _eval_three(hold_bundle.graphs, hold_bundle.adjacent_sets, hold_logits, arm_b, arm_c, device)
    print("writing route-level stratified audit", flush=True)
    audit = _audit_holdout(hold_bundle.graphs, hold_logits, arm_b, arm_c, device)
    reading = interpret_ablation(holdout, same_family, audit["complete_truth_energy_correlation"])

    dataset_manifest = {
        "schema": "faser-dataset-contract-v2",
        "experiment": "wb80_arm_b_failure_mechanism_v1",
        "fold": args.fold,
        "train_family": train_family,
        "holdout_family": holdout_family,
        "train_sources": list(train_sources),
        "holdout_sources": list(holdout_sources),
        "train_manifest": str(FAMILY_CORPUS[train_family]),
        "holdout_manifest": str(FAMILY_CORPUS[holdout_family]),
        "validation_payload_ids": sorted(VAL_PAYLOAD_IDS),
        "wb79_arm_b_checkpoint": str(WB79_ROOT / args.fold / "arm_b_checkpoint.json"),
        "development_00350_used": False,
        "final_blind_accessed": False,
        "sealed_test_accessed": False,
        "wp4_arm_c_authorized": False,
    }
    resolved_config = {
        "workbook": 80,
        "utility_contract": CANONICAL_ENERGY_VERSION,
        "metric_version": "route_accounting_v2",
        "solver": "exact_unit_capacity",
        "arms": {
            "A": "sum_raw_w64_edge_logits",
            "B": "frozen_wb79_physical_mlp",
            "C": "physical_plus_w64_logits_mlp",
        },
        "hybrid_model_contract": MODEL_CONTRACT,
        "arm_b_retrained": False,
        "v5a_retrained": False,
        "wp4_arm_c": False,
        "device": str(device),
        "seed": SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "unmatched_penalty": UNMATCHED_PENALTY,
    }
    _write_json(output / "dataset_manifest.json", dataset_manifest)
    _write_json(output / "resolved_config.json", resolved_config)
    _write_json(
        output / "checkpoint_ancestry.json",
        {
            "w64_sha256": w64_sha,
            "wb79_arm_b": str(WB79_ROOT / args.fold / "arm_b_checkpoint.json"),
            "arm_c_parent": "wb79_physical_features_plus_frozen_w64_logits",
            "historical_v5a_head_used": False,
        },
    )
    _write_json(
        output / "environment.json",
        {
            "hostname": socket.gethostname(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "cuda": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_sha": _git_sha(),
        },
    )
    def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str):
        if left.get(key) is None or right.get(key) is None:
            return None
        return float(left[key] - right[key])

    _write_json(
        output / "evaluation.json",
        {
            "fold": args.fold,
            "holdout_family": holdout_family,
            "same_family_val": same_family,
            "holdout": holdout,
            "delta_holdout": {
                "complete_efficiency_b_minus_a": _delta(holdout["B"], holdout["A"], "complete_track_efficiency"),
                "complete_efficiency_c_minus_a": _delta(holdout["C"], holdout["A"], "complete_track_efficiency"),
                "complete_fake_b_minus_a": _delta(holdout["B"], holdout["A"], "complete_fake_rate"),
                "complete_fake_c_minus_a": _delta(holdout["C"], holdout["A"], "complete_fake_rate"),
            },
            "delta_same_family_val": {
                "complete_efficiency_b_minus_a": _delta(same_family["B"], same_family["A"], "complete_track_efficiency"),
                "complete_efficiency_c_minus_a": _delta(same_family["C"], same_family["A"], "complete_track_efficiency"),
                "complete_fake_b_minus_a": _delta(same_family["B"], same_family["A"], "complete_fake_rate"),
                "complete_fake_c_minus_a": _delta(same_family["C"], same_family["A"], "complete_fake_rate"),
            },
            "w64_saw_all_six_train_sources": True,
            "arm_b_retrained": False,
            "wp4_arm_c_authorized": False,
        },
    )
    _write_json(output / "stratified_audit.json", audit)
    _write_json(output / "interpretation.json", reading)
    _write_json(
        output / "contract_gate.json",
        {
            "gate_pass": None,
            "note": "WB80 attributes the WB79 Arm B gap; it does not promote a production operating point",
            "required_metric_version": "route_accounting_v2",
            "arm_a_metric_version": holdout["A"].get("metric_version"),
            "arm_b_metric_version": holdout["B"].get("metric_version"),
            "arm_c_metric_version": holdout["C"].get("metric_version"),
            "development_00350_used": False,
            "final_blind_accessed": False,
            "sealed_test_accessed": False,
            "v5a_retrained": False,
            "wp4_arm_c_authorized": False,
            "legacy_decoder_used": False,
        },
    )
    _write_json(
        output / "run_metadata.json",
        {
            "workbook": 80,
            "git_sha": _git_sha(),
            "fold": args.fold,
            "training_authorized": True,
            "final_blind_eval_authorized": False,
            "sealed_test_accessed": False,
            "continue_to_v5a_frozen_head": False,
            "continue_to_15d_relative_wls": False,
            "arm_c_representation_authorized": False,
            "n_train_graphs": n_train_graphs,
            "n_val_graphs": n_val_graphs,
            "n_holdout_graphs": len(hold_bundle.graphs),
            "train_overlay_seed": (train_raw.get("synthetic_multitrack") or {}).get("seed"),
            "holdout_overlay_seed": (hold_raw.get("synthetic_multitrack") or {}).get("seed"),
        },
    )
    print(json.dumps({"output": str(output), "reading": reading}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
