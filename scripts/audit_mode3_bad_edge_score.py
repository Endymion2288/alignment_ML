#!/usr/bin/env python3
"""Score the known mis-associated 0->1 physical edge under a frozen or
retrained route-query backbone and report its calibrated score and rank.

The six synthetic copies of the pathological 0->1 mismatch were identified in
the iteration-1 validation backbone output.  For each copy this audit reports
the model's raw and calibrated edge score, the edge's rank among all 0->1
candidates of the same synthetic event (rank 1 = highest score), the truth
label, and whether the edge survives route selection.  The same audit runs
unchanged on the mode-0 and mode-3 candidate graphs so the matched-retraining
comparison can read off whether retraining actually demotes the bad edge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
)
from training.route_aware_transformer import (
    load_route_aware_transformer_artifact,
    predict_route_aware_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BAD_EDGE_IDENTITIES = {
    (996000, 5, 4, 10),
    (996000, 165, 3, 0),
    (996000, 297, 3, 6),
    (996000, 397, 3, 4),
    (996000, 670, 2, 3),
    (996000, 763, 10, 7),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(root: Path):
    from scripts.run_frozen_association_backbone import _load_calibration as _load

    return _load(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--model-artifact", required=True, help="V2 backbone artifact directory")
    parser.add_argument("--payload-id", default="iteration_01_anchor")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0, 3))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(args.split,),
    )
    if int(manifest.get("q_over_p_mode", -1)) != int(args.q_over_p_mode):
        raise ValueError("synthetic manifest mode does not match --q-over-p-mode")
    samples = [sample for sample in samples if str(sample.payload_id) == str(args.payload_id)]
    if not samples:
        raise ValueError(f"payload {args.payload_id} absent from split {args.split}")

    artifact_root = Path(args.model_artifact).expanduser().resolve()
    contract = _read_json(artifact_root / "validation_run_contract.json")
    model, artifact = load_route_aware_transformer_artifact(
        artifact_root / "route_aware_transformer_v2.pt", device=args.device
    )
    calibration = _load_calibration(artifact_root)

    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    prediction = predict_route_aware_scores(
        model,
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=args.device,
        batch_size=64,
    )
    calibrated = apply_frozen_transformer_calibration(bundle.adjacent_sets, list(prediction.edge_scores), calibration)

    copies: list[dict[str, Any]] = []
    for candidate_set, scores in zip(bundle.adjacent_sets, calibrated):
        if tuple(candidate_set.station_pair) != (0, 1):
            continue
        event = candidate_set.event
        key_prefix = (int(event.run_id), int(event.event_id))
        tracklet_ids = np.asarray(event.tracklet_id, dtype=np.int64)
        scored = []
        for index, (candidate, score) in enumerate(zip(candidate_set.candidates, scores)):
            source_id = int(tracklet_ids[candidate.source_index])
            target_id = int(tracklet_ids[candidate.target_index])
            scored.append(
                {
                    "candidate_index": index,
                    "source_tracklet_id": source_id,
                    "target_tracklet_id": target_id,
                    "calibrated_score": float(score),
                    "is_truth": bool(candidate_set.labels[index]),
                    "chi2": float(candidate.chi2),
                }
            )
        order = sorted(scored, key=lambda row: -row["calibrated_score"])
        for rank, row in enumerate(order, start=1):
            row["rank"] = rank
        for row in scored:
            identity = (*key_prefix, row["source_tracklet_id"], row["target_tracklet_id"])
            if identity in BAD_EDGE_IDENTITIES:
                copies.append(
                    {
                        "run_id": key_prefix[0],
                        "event_id": key_prefix[1],
                        "source_tracklet_id": row["source_tracklet_id"],
                        "target_tracklet_id": row["target_tracklet_id"],
                        "calibrated_score": row["calibrated_score"],
                        "rank_among_event_0_to_1_candidates": row["rank"],
                        "event_0_to_1_candidate_count": len(scored),
                        "is_truth": row["is_truth"],
                        "chi2": row["chi2"],
                        "best_truth_score": max(
                            (entry["calibrated_score"] for entry in scored if entry["is_truth"]),
                            default=None,
                        ),
                    }
                )

    summary = {
        "schema_version": "faser-mode3-bad-edge-score-audit-v1",
        "synthetic_manifest": str(manifest_path),
        "model_artifact": str(artifact_root),
        "model_artifact_q_over_p_mode": int(contract.get("q_over_p_mode", -1)),
        "data_q_over_p_mode": int(args.q_over_p_mode),
        "payload_id": str(args.payload_id),
        "split": str(args.split),
        "copies_found": len(copies),
        "copies": copies,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"copies_found": len(copies)}, indent=2))


if __name__ == "__main__":
    main()
