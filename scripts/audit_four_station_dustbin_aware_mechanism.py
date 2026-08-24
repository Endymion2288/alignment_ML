#!/usr/bin/env python3
"""Transfer-bank mechanism audit for workbook-59.  Does not train or retune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_four_station_dustbin_aware_objective import (
    DEVELOPMENT_SOURCES,
    FORBIDDEN_PAYLOADS,
    TRAIN_SOURCES,
    TRANSFER_SOURCES,
    _audit_model,
    _frozen_route_config,
    _score_model,
    _write_json,
)
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("mechanism audit is GPU-only")

    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if {sample.split for sample in samples} != {"validation"}:
        raise ValueError("mechanism audit must load the transfer validation split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    if constituents & DEVELOPMENT_SOURCES or constituents & TRAIN_SOURCES:
        raise SystemExit("mechanism audit must use the source-disjoint transfer pair")
    if constituents != TRANSFER_SOURCES:
        raise SystemExit("transfer overlay sources are not the unused μ± pair")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")

    root = Path(args.candidate_frozen_output).expanduser().resolve()
    frozen = _load_v2(root, args.device)
    calibration = frozen["calibration"]
    if calibration.get("fit_split") != "identity_frozen_pre_training" or calibration.get("identity_map") is not True:
        raise SystemExit("candidate checkpoint is not the frozen identity Platt")
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    scores = _score_model(frozen, bundle, args.device, args.batch_size)
    audit = _audit_model(
        "dustbin_aware_v2",
        frozen,
        bundle,
        scores,
        _frozen_route_config(),
        float(PACKING_MARGIN),
    )
    pooled = audit["pooled_feasibility"]
    ident = audit["pooled_identifiability"]
    payload = {
        "checkpoint": frozen["metadata"]["checkpoint"],
        "checkpoint_sha256": frozen["metadata"]["checkpoint_sha256"],
        "u_truth_median": pooled["u_truth_median"],
        "u_truth_fraction_nonpositive": pooled["u_truth_fraction_nonpositive"],
        "u_best_fragment_median": pooled["u_best_fragment_median"],
        "required_margin_to_dustbin_median": pooled["required_margin_to_dustbin_median"],
        "feasibility_family_counts": pooled["feasibility_family_counts"],
        "selected": ident["selected"],
        "score_retained": ident["score_retained"],
        "production_winner_counts": ident["production_winner_counts"],
        "production_admitted_fragment_wins": int(
            (ident.get("decision_class_counts") or {}).get("truth_loses_to_fragment") or 0
        ),
        "dustbin_winner_missed_by_training_miner": ident["dustbin_winner_missed_by_training_miner"],
        "payloads": {
            name: {
                "feasibility": report["feasibility"],
                "identifiability": {
                    "selected": report["identifiability"]["selected"],
                    "decision_class_counts": report["identifiability"]["decision_class_counts"],
                },
                "selected_summary": report["selected_summary"],
            }
            for name, report in audit["payloads"].items()
        },
    }
    _write_json(Path(args.output_json), payload)
    print(json.dumps(_json_ready(payload), indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
