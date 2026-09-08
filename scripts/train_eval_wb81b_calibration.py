#!/usr/bin/env python3
"""Workbook 81b: train bounded physics residual Arm D on frozen W64 energy.

Does not retrain V5A/W64/B/C.  Does not expand hybrid.  Does not open
00350, Final Blind, or sealed test.  Fake slack is zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.mlp_pair_classifier import FeatureStandardizer
from models.explicit_route_energy import FEATURE_DIM, route_feature_names
from models.physics_constrained_calibration import MODEL_CONTRACT, PhysicsConstrainedCalibrator
from models.route_energy import CANONICAL_ENERGY_VERSION
from scripts.materialize_wb81a_boundary import (
    FAMILY_CORPUS,
    VAL_PAYLOAD_IDS,
    W64_ROOT,
    W64_SHA,
    _load_family_bundle,
    _sha256,
    _w64_base_logits,
    _write_json,
)
from training.explicit_route_energy import (
    account_event,
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    route_feature_matrix,
    w64_route_utilities,
)
from training.route_aware_transformer import load_route_aware_transformer_artifact
from training.source_transfer_cv import (
    FAMILY1_DSID_PAIR,
    FAMILY2_DSID_PAIR,
    V5A_SOURCE_FAMILIES,
    validate_fold_sources,
)
from training.wb81_calibration_contract import (
    ACCEPT_EFF_GAIN,
    DELTA_MAX,
    UNMATCHED_PENALTY,
    fake_hard_gate,
    refuse_wb81a_path,
    verify_wb81a_checksum_file,
)
from training.wb81b_calibration import (
    audit_holdout,
    calibration_event_loss,
    fold_decision,
    identity_on_graphs,
    identity_with_model,
    score_routes,
)


WB81A_ROOT = Path("outputs/mc24_four_station_wb81a_calibration_foundation_v1")
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb81b_calibration_v1")
FOLD_TRAIN_FAMILY = {
    "holdout_family1": FAMILY2_DSID_PAIR,
    "holdout_family2": FAMILY1_DSID_PAIR,
}
FOLD_HOLDOUT_FAMILY = {
    "holdout_family1": FAMILY1_DSID_PAIR,
    "holdout_family2": FAMILY2_DSID_PAIR,
}
SEED = 20260905
EPOCHS = 20
LEARNING_RATE = 3.0e-4
IDENTITY_FAIL = "WB81b identity fail => stop.  No training."


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _split_graphs(graphs):
    train = []
    validation = []
    for graph in graphs:
        if str(graph.sample.payload_id) in VAL_PAYLOAD_IDS:
            validation.append(graph)
        else:
            train.append(graph)
    if not train or not validation:
        raise SystemExit("train/val payload split is empty")
    return train, validation


def _fit_standardizer(graphs) -> FeatureStandardizer:
    blocks = []
    for graph in graphs:
        routes = enumerate_contiguous_physical_routes(graph)
        if routes:
            blocks.append(route_feature_matrix(graph, routes))
    if not blocks:
        raise SystemExit("no train routes to fit the Arm D standardizer")
    return FeatureStandardizer.fit(np.concatenate(blocks, axis=0))


def _evaluate(graphs, adjacent_sets, utilities_fn) -> dict[str, Any]:
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
        raise SystemExit("evaluation produced no events")
    payload = accounting.as_dict()
    payload.update(
        {
            "n_scored_events": n_events,
            "n_enumerated_routes": n_routes,
            "utility_contract": CANONICAL_ENERGY_VERSION,
            "gate_pass": None,
        }
    )
    return payload


def _mean_val_loss(model, standardizer, graphs, logits, device: torch.device) -> float | None:
    model.eval()
    values = []
    with torch.no_grad():
        for graph in graphs:
            report = calibration_event_loss(model, standardizer, graph, logits, device)
            if report is not None:
                values.append(float(report["loss"].detach().cpu()))
    if not values:
        return None
    return float(np.mean(values))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    refuse_wb81a_path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verify_wb81a() -> dict[str, object]:
    reports = {}
    for family in (FAMILY1_DSID_PAIR, FAMILY2_DSID_PAIR):
        reports[family] = verify_wb81a_checksum_file(WB81A_ROOT / family / "checksums.json", family)
    return reports


def _load_scored_family(family: str, w64_checkpoint: Path, device: torch.device, batch_size: int):
    refuse_wb81a_path(FAMILY_CORPUS[family])
    print(f"loading family {family}", flush=True)
    _, bundle, raw = _load_family_bundle(family)
    print(f"extracting frozen W64 logits on {family}", flush=True)
    model_w64, artifact_w64 = load_route_aware_transformer_artifact(w64_checkpoint, device="cpu")
    logits = _w64_base_logits(bundle, model_w64, artifact_w64, device, batch_size)
    del model_w64
    gc.collect()
    return bundle, raw, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=sorted(FOLD_TRAIN_FAMILY), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit("WB81b formal path is CPU")
    device = torch.device("cpu")

    train_family = FOLD_TRAIN_FAMILY[args.fold]
    holdout_family = FOLD_HOLDOUT_FAMILY[args.fold]
    validate_fold_sources(V5A_SOURCE_FAMILIES[train_family], V5A_SOURCE_FAMILIES[holdout_family])
    output = Path(args.output_root) / args.fold
    output.mkdir(parents=True, exist_ok=True)
    refuse_wb81a_path(output)

    print("verifying WB81a checksums", flush=True)
    wb81a_checksums = _verify_wb81a()

    w64_checkpoint = W64_ROOT / "route_aware_transformer_v2.pt"
    refuse_wb81a_path(w64_checkpoint)
    w64_sha = _sha256(w64_checkpoint)
    if w64_sha != W64_SHA:
        raise SystemExit(f"W64 checkpoint sha {w64_sha} != frozen {W64_SHA}")

    print(f"preflight identity on holdout family {holdout_family}", flush=True)
    hold_bundle, hold_raw, hold_logits = _load_scored_family(holdout_family, w64_checkpoint, device, args.batch_size)
    identity_hold = identity_on_graphs(hold_bundle.graphs, hold_bundle.adjacent_sets, hold_logits, account_event)
    if not identity_hold["passed"]:
        raise SystemExit(f"{IDENTITY_FAIL} holdout family: {identity_hold}")
    del hold_bundle, hold_raw, hold_logits
    gc.collect()

    print(f"preflight identity on train family {train_family}", flush=True)
    train_bundle, train_raw, train_logits = _load_scored_family(train_family, w64_checkpoint, device, args.batch_size)
    identity_train = identity_on_graphs(train_bundle.graphs, train_bundle.adjacent_sets, train_logits, account_event)
    if not identity_train["passed"]:
        raise SystemExit(f"{IDENTITY_FAIL} train family: {identity_train}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = PhysicsConstrainedCalibrator().to(device)
    train_graphs, val_graphs = _split_graphs(train_bundle.graphs)
    standardizer = _fit_standardizer(train_graphs)
    print("identity replay on the official train path at theta=0", flush=True)
    identity_model = identity_with_model(
        train_bundle.graphs,
        train_bundle.adjacent_sets,
        train_logits,
        model,
        standardizer,
        device,
        account_event,
    )
    if not identity_model["passed"]:
        raise SystemExit(f"{IDENTITY_FAIL} zero-init model: {identity_model}")

    def arm_a_utilities(graph, routes, logits=train_logits):
        return w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)

    def arm_d_utilities(graph, routes, logits=train_logits):
        u_w64, delta = score_routes(model, standardizer, graph, routes, logits, device)
        return u_w64 + delta

    val_a = _evaluate(val_graphs, train_bundle.adjacent_sets, arm_a_utilities)
    fake_a_val = val_a.get("complete_fake_rate")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    admitted = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for graph in train_graphs:
            report = calibration_event_loss(model, standardizer, graph, train_logits, device)
            if report is None:
                continue
            optimizer.zero_grad()
            report["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(report["loss"].detach().cpu()))
        model.eval()
        val_d = _evaluate(val_graphs, train_bundle.adjacent_sets, arm_d_utilities)
        val_loss = _mean_val_loss(model, standardizer, val_graphs, train_logits, device)
        admitted_now = fake_hard_gate(val_d.get("complete_fake_rate"), fake_a_val)
        row = {
            "epoch": epoch,
            "train_loss": None if not epoch_losses else float(np.mean(epoch_losses)),
            "val_loss": val_loss,
            "val_complete_efficiency": val_d.get("complete_track_efficiency"),
            "val_complete_fake_rate": val_d.get("complete_fake_rate"),
            "val_complete_fake_rate_A": fake_a_val,
            "fake_hard_gate": admitted_now,
            "n_train_events": len(epoch_losses),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if admitted_now:
            admitted.append(
                {
                    "epoch": epoch,
                    "val_complete_efficiency": val_d.get("complete_track_efficiency"),
                    "val_complete_fake_rate": val_d.get("complete_fake_rate"),
                    "val_loss": val_loss,
                    "train_loss": row["train_loss"],
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
            )

    selected = None
    if admitted:
        admitted.sort(
            key=lambda item: (
                -(float("-inf") if item["val_complete_efficiency"] is None else float(item["val_complete_efficiency"])),
                float("inf") if item["val_loss"] is None else float(item["val_loss"]),
                int(item["epoch"]),
            )
        )
        selected = admitted[0]
        model.load_state_dict(selected["state_dict"])
    else:
        model = PhysicsConstrainedCalibrator().to(device)
    model.eval()

    n_train_graphs = len(train_graphs)
    n_val_graphs = len(val_graphs)
    train_overlay_seed = (train_raw.get("synthetic_multitrack") or {}).get("seed")
    del train_bundle, train_graphs, val_graphs, train_logits, train_raw
    gc.collect()

    print(f"loading holdout family {holdout_family} for Arm A / Arm D evaluation", flush=True)
    hold_bundle, hold_raw, hold_logits = _load_scored_family(holdout_family, w64_checkpoint, device, args.batch_size)
    print("identity replay on the holdout family at theta=0", flush=True)
    identity_hold_eval = identity_on_graphs(hold_bundle.graphs, hold_bundle.adjacent_sets, hold_logits, account_event)
    if not identity_hold_eval["passed"]:
        raise SystemExit(f"{IDENTITY_FAIL} holdout family at eval: {identity_hold_eval}")

    def hold_a(graph, routes):
        return w64_route_utilities(graph, routes, hold_logits, UNMATCHED_PENALTY)

    def hold_d(graph, routes):
        u_w64, delta = score_routes(model, standardizer, graph, routes, hold_logits, device)
        return u_w64 + delta

    print("evaluating Arm A and Arm D on holdout", flush=True)
    holdout_a = _evaluate(hold_bundle.graphs, hold_bundle.adjacent_sets, hold_a)
    holdout_d = _evaluate(hold_bundle.graphs, hold_bundle.adjacent_sets, hold_d)
    holdout_a["arm"] = "A_frozen_w64_raw_energy"
    holdout_d["arm"] = "D_physics_constrained_calibration"
    audit = audit_holdout(
        hold_bundle.graphs,
        hold_logits,
        model,
        standardizer,
        device,
        _load_jsonl(WB81A_ROOT / holdout_family / "complete_truths.jsonl"),
        _load_jsonl(WB81A_ROOT / holdout_family / "a_selected_fakes.jsonl"),
    )
    decision = fold_decision(
        identity_passed=bool(identity_train["passed"] and identity_hold["passed"] and identity_model["passed"]),
        admitted=bool(admitted),
        holdout_a=holdout_a,
        holdout_d=holdout_d,
    )
    artifact = {
        "model_contract": MODEL_CONTRACT,
        "feature_version": "raw_physical_route_v1",
        "feature_names": list(route_feature_names()),
        "feature_dim": FEATURE_DIM,
        "delta_max": DELTA_MAX,
        "seed": SEED,
        "standardizer_mean": standardizer.mean.tolist(),
        "standardizer_scale": standardizer.scale.tolist(),
        "state_dict": {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()},
        "selected_epoch": None if selected is None else selected["epoch"],
        "admitted": bool(admitted),
        "not_wp4_arm_c": True,
        "not_wb80_hybrid": True,
    }
    _write_json(output / "arm_d_checkpoint.json", artifact)
    torch.save(
        {
            "model_contract": MODEL_CONTRACT,
            "state_dict": model.state_dict(),
            "standardizer_mean": standardizer.mean,
            "standardizer_scale": standardizer.scale,
        },
        output / "arm_d_checkpoint.pt",
    )
    checkpoint_sha = hashlib.sha256((output / "arm_d_checkpoint.json").read_bytes()).hexdigest()
    _write_json(
        output / "dataset_manifest.json",
        {
            "schema": "faser-dataset-contract-v2",
            "experiment": "wb81b_calibration_v1",
            "fold": args.fold,
            "train_family": train_family,
            "holdout_family": holdout_family,
            "train_sources": list(V5A_SOURCE_FAMILIES[train_family]),
            "holdout_sources": list(V5A_SOURCE_FAMILIES[holdout_family]),
            "train_manifest": str(FAMILY_CORPUS[train_family]),
            "holdout_manifest": str(FAMILY_CORPUS[holdout_family]),
            "wb81a_root": str(WB81A_ROOT),
            "development_00350_used": False,
            "final_blind_accessed": False,
            "sealed_test_accessed": False,
            "wp4_arm_c_authorized": False,
        },
    )
    _write_json(
        output / "resolved_config.json",
        {
            "workbook": "81b",
            "model_contract": MODEL_CONTRACT,
            "utility_contract": CANONICAL_ENERGY_VERSION,
            "metric_version": "route_accounting_v2",
            "solver": "exact_unit_capacity",
            "loss": ["L_gap", "L_keep"],
            "lai_used": False,
            "delta_max": DELTA_MAX,
            "fake_slack": 0.0,
            "accept_eff_gain": ACCEPT_EFF_GAIN,
            "g_theta": "linear",
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "patience_used": False,
            "device": "cpu",
        },
    )
    _write_json(
        output / "checkpoint_ancestry.json",
        {
            "w64_sha256": w64_sha,
            "wb81a_checksums": wb81a_checksums,
            "arm_d_checkpoint_sha256": checkpoint_sha,
            "historical_v5a_head_used": False,
            "wb79_arm_b_used": False,
            "wb80_arm_c_used": False,
        },
    )
    _write_json(
        output / "identity.json",
        {
            "train_family": identity_train,
            "holdout_family": identity_hold,
            "holdout_family_at_eval": identity_hold_eval,
            "zero_init_model": identity_model,
            "passed": bool(
                identity_train["passed"]
                and identity_hold["passed"]
                and identity_hold_eval["passed"]
                and identity_model["passed"]
            ),
        },
    )
    _write_json(
        output / "training_history.json",
        {
            "history": history,
            "admitted_epochs": [item["epoch"] for item in admitted],
            "selected_epoch": None if selected is None else selected["epoch"],
        },
    )
    _write_json(
        output / "evaluation.json",
        {
            "fold": args.fold,
            "holdout_family": holdout_family,
            "arm_a": holdout_a,
            "arm_d": holdout_d,
            "same_family_val_a": val_a,
            "delta_complete_efficiency_d_minus_a": (
                None
                if holdout_a.get("complete_track_efficiency") is None
                or holdout_d.get("complete_track_efficiency") is None
                else float(holdout_d["complete_track_efficiency"] - holdout_a["complete_track_efficiency"])
            ),
            "delta_complete_fake_rate_d_minus_a": (
                None
                if holdout_a.get("complete_fake_rate") is None or holdout_d.get("complete_fake_rate") is None
                else float(holdout_d["complete_fake_rate"] - holdout_a["complete_fake_rate"])
            ),
            "w64_saw_all_six_train_sources": True,
            "admitted": bool(admitted),
            "selected_epoch": None if selected is None else selected["epoch"],
            "gate_pass": None,
        },
    )
    _write_json(output / "stratified_audit.json", audit)
    _write_json(output / "decision.json", decision)
    _write_json(
        output / "environment.json",
        {
            "hostname": socket.gethostname(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "cuda": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "condor_cluster": os.environ.get("CLUSTER"),
            "condor_process": os.environ.get("PROCESS"),
            "git_sha": _git_sha(),
            "git_status_porcelain": _git_porcelain(),
        },
    )
    _write_json(
        output / "run_metadata.json",
        {
            "workbook": "81b",
            "git_sha": _git_sha(),
            "git_status_porcelain": _git_porcelain(),
            "fold": args.fold,
            "training_authorized": True,
            "wb81b_authorized": True,
            "final_blind_eval_authorized": False,
            "sealed_test_accessed": False,
            "n_train_graphs": n_train_graphs,
            "n_val_graphs": n_val_graphs,
            "n_holdout_graphs": len(hold_bundle.graphs),
            "train_overlay_seed": train_overlay_seed,
            "holdout_overlay_seed": (hold_raw.get("synthetic_multitrack") or {}).get("seed"),
            "arm_d_checkpoint_sha256": checkpoint_sha,
        },
    )
    print(json.dumps({"output": str(output), "decision": decision, "audit_keys": list(audit)}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
