#!/usr/bin/env python3
"""Origin-aligned chart/twin raw-logit and packing-utility shifts.

GPU-only.  Does not retune thresholds, penalty, or calibration.  Sealed test
is never opened.  Development validation is refused when --split is used on
the workbook-53 overlay unless --allow-development-validation is set, which
this workbook-56 gate does not pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.pairwise_metrics import probability_to_logit
from scripts.evaluate_four_station_matched_association import _families_from_plan
from scripts.run_frozen_association_backbone import _load_v2
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.gauge_consistent_route import origin_matched_score_shifts, pair_gauge_twin_graphs
from training.route_aware_transformer import predict_route_aware_scores


FORBIDDEN_PAYLOADS = {
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
}
DEVELOPMENT_SOURCES = {"mc24_100047_00050_00099", "mc24_100048_00050_00099"}


def _graph_logits(graph, scores: list[np.ndarray]) -> np.ndarray:
    values = np.empty(graph.score_labels.size, dtype=np.float64)
    for index, (owner, row) in enumerate(zip(graph.score_owner, graph.score_row)):
        probability = float(scores[int(owner)][int(row)])
        values[index] = float(probability_to_logit(np.asarray([probability]))[0])
    if not np.isfinite(values).all():
        raise RuntimeError("origin-aligned logit reconstruction produced non-finite values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    parser.add_argument("--allow-development-validation", action="store_true")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    if str(args.device) != "cuda":
        raise SystemExit("score-scale audit is GPU-only")

    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(args.split,),
    )
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    if constituents & DEVELOPMENT_SOURCES and not args.allow_development_validation:
        raise SystemExit("refusing development-validation sources for the workbook-56 gate")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")

    plan = json.loads(Path(args.iteration_manifest).read_text(encoding="utf-8"))["common_scan_plan"]
    families = _families_from_plan(plan)
    frozen = _load_v2(Path(args.frozen_output).expanduser().resolve(), args.device)
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    artifact = frozen["artifact"]
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode=artifact.context_mode)
    prediction = predict_route_aware_scores(
        frozen["model"],
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    scores = list(prediction.edge_scores)
    pairs, _ = pair_gauge_twin_graphs(bundle.graphs)
    pooled_edge: list[float] = []
    pooled_utility: list[float] = []
    for chart, twin in pairs:
        report = origin_matched_score_shifts(
            chart,
            twin,
            _graph_logits(chart, scores),
            _graph_logits(twin, scores),
        )
        pooled_edge.extend(float(value) for value in report["edge_abs_shifts"])
        pooled_utility.extend(float(value) for value in report["utility_abs_shifts"])

    def _pool(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "median": None, "p90": None, "mean": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "mean": float(np.mean(array)),
        }

    payload = {
        "split": args.split,
        "test_data_accessed": False,
        "frozen_output": str(Path(args.frozen_output).expanduser().resolve()),
        "families_in_plan": {name: list(pair) for name, pair in families.items()},
        "pooled_matched_origin_raw_logit_abs_shift": _pool(pooled_edge),
        "pooled_matched_complete_route_utility_abs_shift": _pool(pooled_utility),
        "event_pair_count": len(pairs),
        "development_validation_used": bool(constituents & DEVELOPMENT_SOURCES),
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready({"output_json": str(output), **payload}), indent=2)[:2000])


if __name__ == "__main__":
    main()
