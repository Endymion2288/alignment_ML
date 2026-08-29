#!/usr/bin/env python3
"""Freeze the workbook-64 checkpoint before any reserved-blind file is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    HISTORY_ONLY_SOURCES,
    RESERVED_BLIND_SOURCES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    root = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint = root / "route_aware_transformer_v2.pt"
    contract = json.loads((root / "validation_run_contract.json").read_text(encoding="utf-8"))
    weights = json.loads((root / "aux_loss_weight_contract.json").read_text(encoding="utf-8"))
    history = json.loads((root / "training_history.json").read_text(encoding="utf-8"))
    loaded = set(contract.get("loaded_source_ids") or [])
    if loaded != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise SystemExit("frozen checkpoint did not train the authorized six-source set")
    if loaded & HISTORY_ONLY_SOURCES or loaded & set(RESERVED_BLIND_SOURCES):
        raise SystemExit("checkpoint trained on history-only or reserved-blind sources")
    if contract.get("blind_not_used_for_training") is not True:
        raise SystemExit("training contract does not certify blind_not_used_for_training")
    epochs = len(history.get("history") or [])
    if epochs != 30:
        raise SystemExit(f"expected 30 epochs, got {epochs}")
    payload = {
        "control_id": contract.get("control_id"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "epochs": epochs,
        "checkpoint_selection": "last_completed_epoch_of_fixed_30_epoch_budget",
        "loaded_source_ids": sorted(loaded),
        "blind_not_used_for_training": True,
        "reserved_blind_used_for_checkpoint_selection": False,
        "history_only_sources_loaded": False,
        "early_stopping": False,
        "objective_reestimated": False,
        "aux_loss_weights": {
            "algorithm": weights.get("algorithm"),
            "packing_route_competition_weight": weights.get("packing_route_competition_weight"),
            "dustbin_aware_route_margin_weight": weights.get("dustbin_aware_route_margin_weight"),
            "gauge_twin_consistency_weight": weights.get("gauge_twin_consistency_weight"),
            "route_competition_reduction": weights.get("route_competition_reduction"),
            "packing_margin": weights.get("packing_margin"),
            "pair_threshold": weights.get("pair_threshold"),
            "unmatched_penalty": weights.get("unmatched_penalty"),
        },
        "operating_convention": {
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": 1.0,
        },
        "test_data_accessed": False,
        "continue_to_15d_relative_wls": False,
        "reserved_blind_may_be_opened": True,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
