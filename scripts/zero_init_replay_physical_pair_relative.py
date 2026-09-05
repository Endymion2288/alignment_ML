#!/usr/bin/env python3
"""Workbook-74 Phase A: zero-init exact baseline replay for the Primary.

Before any optimizer step the Physical Pair-Relative Route Encoder has a
zero-initialized ``route_score`` terminal, so ``delta_route_logit == 0`` and
``L_corrected == L_edge_W64``.  This script verifies, on a handful of fixed
fold corpus events, that the Primary @ zero-init reproduces the frozen
Workbook-64 (Arm 0) solver assignment exactly:

    L_corrected, U_complete, selected routes, C/D decision class, fragment utility

Read-only.  It never opens development, final-blind, or sealed-test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.route_transformer import (
    RelativeRouteV4Inference,
    RouteAwareSparseTransformer,
    RouteAwareTransformerConfig,
    freeze_backbone_and_edge_scorer,
)
from scripts.config_loader import load_yaml_with_base
from scripts.audit_relative_route_v4_generalization import _load_split
from scripts.evaluate_relative_route_v4_development import (
    _assert_identity_platt,
    _score_arm,
    frozen_packing_config,
)
from scripts.evaluate_relative_route_v5a_source_transfer import (
    _evaluate_arm_single_pass,
    _load_arm_flexible,
)
from training.gauge_consistent_route import identity_calibration_payload
from training.route_aware_transformer import (
    RouteAwareTransformerArtifact,
    load_route_aware_transformer_artifact,
)
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)

W64_ROOT = "outputs/mc24_four_station_source_diversity_v1/checkpoint"
PRIMARY_CONFIG = "configs/wb74_physical_pair_relative_bounded_primary_source_transfer.yaml"
FAMILY_CORPUS = {
    "family1": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
    "family2": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family2/overlay_synthetic_v1/synthetic_corpus_manifest.json",
}


def _build_zero_init_primary(device: str) -> dict[str, object]:
    """Build the Physical Pair-Relative wrapper @ zero-init from the frozen W64 parent."""
    supplied = load_yaml_with_base(Path(PRIMARY_CONFIG))
    arch_cfg = next(iter(supplied.values()))["architecture"]
    frozen_w64, frozen_artifact = load_route_aware_transformer_artifact(
        Path(W64_ROOT) / "route_aware_transformer_v2.pt", device="cpu"
    )
    model_config = RouteAwareTransformerConfig(node_feature_dim=17, edge_feature_dim=11, **arch_cfg)
    if model_config.route_representation_mode != "physical_pair_relative":
        raise RuntimeError("primary config is not physical_pair_relative")
    trainable = RouteAwareSparseTransformer(model_config)
    # Copy frozen backbone/edge tensors (route head stays freshly initialized).
    prefixes = (
        "route_query", "route_node_key", "route_edge_projection",
        "route_pair_embedding", "route_encoder", "route_score",
    )
    frozen_state = frozen_w64.state_dict()
    trainable_state = trainable.state_dict()
    for key, value in frozen_state.items():
        is_head = any(key == p or key.startswith(f"{p}.") for p in prefixes)
        if not is_head and key in trainable_state and trainable_state[key].shape == value.shape:
            trainable_state[key] = value
    trainable.load_state_dict(trainable_state)
    torch.nn.init.zeros_(trainable.route_score.weight)
    torch.nn.init.zeros_(trainable.route_score.bias)
    freeze_backbone_and_edge_scorer(trainable)
    for param in frozen_w64.parameters():
        param.requires_grad = False
    frozen_w64.eval()
    wrapper = RelativeRouteV4Inference(frozen_w64, trainable).to(device)
    wrapper.eval()
    artifact = RouteAwareTransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=ALL_STATION_PAIRS,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=frozen_artifact.node_standardizer,
        edge_standardizer=frozen_artifact.edge_standardizer,
        model_config=model_config,
        context_mode="full_event",
        training_summary={"zero_init_replay": True},
    )
    return {
        "arm": "primary_zero_init",
        "name": "physical_pair_relative_zero_init",
        "model": wrapper,
        "artifact": artifact,
        "calibration": identity_calibration_payload(),
        "inject_complete_route_scores": True,
        "checkpoint": "<in-memory zero-init>",
        "checkpoint_sha256": "<none>",
    }


def _row_key(row: Mapping[str, Any]):
    return (
        str(row["payload_id"]),
        int(row["run_id"]),
        int(row["event_id"]),
        tuple(int(e["index"]) for e in row["endpoints"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILY_CORPUS), default="family2")
    parser.add_argument("--max-events-per-payload", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = args.device
    print("=== Workbook 74 zero-init exact baseline replay ===", flush=True)
    print(f"family={args.family} max_events_per_payload={args.max_events_per_payload} device={device}", flush=True)

    samples, bundle, tables, _ns = _load_split(
        FAMILY_CORPUS[args.family], "train", args.max_events_per_payload
    )
    print(f"loaded graphs={len(bundle.graphs)} routes={sum(t.size for t in tables.values())}", flush=True)

    config = frozen_packing_config()
    arms = {
        "w64": _load_arm_flexible("w64", w64_root=Path(W64_ROOT), head_checkpoint=None, device=device),
        "primary_zero_init": _build_zero_init_primary(device),
    }
    _assert_identity_platt(arms["primary_zero_init"]["calibration"])

    results: dict[str, Any] = {}
    for name, loaded in arms.items():
        raw, calibrated, route_maps = _score_arm(loaded, bundle, device, args.batch_size)
        rows, metrics = _evaluate_arm_single_pass(bundle, calibrated, route_maps, config)
        results[name] = {"rows": rows, "metrics": metrics}
        n_sel = sum(1 for r in rows if r.get("selected"))
        print(f"  {name}: truth_rows={len(rows)} selected={n_sel}", flush=True)

    # Compare per-truth-route assignment between Arm0 (W64) and Primary @ zero-init.
    w64_rows = {_row_key(r): r for r in results["w64"]["rows"]}
    p_rows = {_row_key(r): r for r in results["primary_zero_init"]["rows"]}
    shared = sorted(set(w64_rows) & set(p_rows), key=str)
    only_w64 = set(w64_rows) - set(p_rows)
    only_p = set(p_rows) - set(w64_rows)

    def _f(row, key):
        v = row.get(key)
        return None if v is None else float(v)

    mismatches = []
    max_du = 0.0
    max_dl = 0.0
    for key in shared:
        rw = w64_rows[key]
        rp = p_rows[key]
        uw = _f(rw, "complete_truth_route_utility")
        up = _f(rp, "complete_truth_route_utility")
        if uw is not None and up is not None:
            max_du = max(max_du, abs(uw - up))
        lw = _f(rw, "complete_truth_route_logit")
        lp = _f(rp, "complete_truth_route_logit")
        if lw is not None and lp is not None:
            max_dl = max(max_dl, abs(lw - lp))
        same_selected = bool(rw.get("selected")) == bool(rp.get("selected"))
        same_class = str(rw.get("loss_stage")) == str(rp.get("loss_stage"))
        if not (same_selected and same_class):
            mismatches.append({
                "key": key,
                "w64_selected": bool(rw.get("selected")),
                "primary_selected": bool(rp.get("selected")),
                "w64_class": str(rw.get("loss_stage")),
                "primary_class": str(rp.get("loss_stage")),
            })

    report = {
        "family": args.family,
        "n_truth_rows_w64": len(w64_rows),
        "n_truth_rows_primary": len(p_rows),
        "n_shared_truth_routes": len(shared),
        "n_only_w64": len(only_w64),
        "n_only_primary": len(only_p),
        "max_abs_delta_U_complete": max_du,
        "max_abs_delta_L_corrected": max_dl,
        "n_assignment_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    # Numerical contract: the selected routes, C/D decision class and the head's
    # L_corrected must match Arm0 exactly.  U_complete is computed via two
    # different numerical paths for the two arms -- Arm0 (edge-only) sums the
    # per-edge clipped packing log-odds, while the Primary @ zero-init injects
    # sigmoid(L_edge_W64) and the solver maps it back through solver_log_odds
    # (logit after clipping to the 1e-6 probability floor).  These agree to
    # ~1e-5, far below the O(1) solver decision margins, so the U_complete
    # tolerance is 1e-4 (the solver's numerical contract), not bitwise.
    ok = (
        len(only_w64) == 0
        and len(only_p) == 0
        and len(mismatches) == 0
        and max_du < 1e-4
        and max_dl < 1e-6
    )
    print(f"ZERO_INIT_REPLAY {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("zero-init replay mismatch: Primary @ init != Arm0 W64 assignment")


if __name__ == "__main__":
    main()
