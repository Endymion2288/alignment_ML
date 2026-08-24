#!/usr/bin/env python3
"""Workbook 58: train-only dustbin-aware objective feasibility.

Does not train, does not write a checkpoint, and does not retune
threshold, unmatched penalty, or Platt.  Sealed test is never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, _matrix_edge_lookup, adjacent_station_pairs
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import (
    PACKING_MARGIN,
    attach_dustbin_aware_margin,
    audit_workbook56_packing_loss_source,
    compare_scale_distributions,
    recommend_next_objective,
    summarize_feasibility,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_aware_transformer import predict_route_aware_scores
from training.route_operating_audit import assign_and_audit_payload
from training.route_utility_identifiability import (
    attach_truth_route_identifiability,
    compare_aligned_truth_routes,
    enumerate_threshold_feasible_routes,
    summarize_identifiability,
)


FORBIDDEN_PAYLOADS = {
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
}
DEVELOPMENT_SOURCES = {"mc24_100047_00050_00099", "mc24_100048_00050_00099"}
TRANSFER_SOURCES = {"mc24_100047_00300_00349", "mc24_100048_00300_00349"}
TRAIN_SOURCES = {"mc24_100043_00200_00299", "mc24_100044_00300_00399"}
CANDIDATE_SHA256 = "0e2ffe171e7cbbfd8ced426c9ce34759ffd6d0a2d6b26216df5814d48337cd80"
PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_hard_s3_ry",
    "iteration_00_hard_s3_ry_plus_common",
    "iteration_00_draw_00",
    "iteration_00_draw_00_plus_common",
    "iteration_00_draw_01",
    "iteration_00_draw_01_plus_common",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


def _frozen_route_config() -> RouteAssignmentConfig:
    pairs = adjacent_station_pairs((0, 1, 2, 3))
    return RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in pairs},
        unmatched_penalty=-1.0,
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=100_000,
    )


def _filter_sets(sets: Sequence[object], scores: Sequence[np.ndarray], payload_id: str):
    kept_sets = []
    kept_scores = []
    for candidate_set, values in zip(sets, scores):
        if str(candidate_set.sample.payload_id) == payload_id:
            kept_sets.append(candidate_set)
            kept_scores.append(np.asarray(values, dtype=np.float64))
    if not kept_sets:
        raise ValueError(f"payload {payload_id} is absent from the loaded train sets")
    return kept_sets, kept_scores


def _score_model(frozen: Mapping[str, Any], bundle, device: str, batch_size: int) -> list[np.ndarray]:
    prediction = predict_route_aware_scores(
        frozen["model"],
        bundle,
        frozen["artifact"].node_standardizer,
        frozen["artifact"].edge_standardizer,
        device=device,
        batch_size=int(batch_size),
    )
    return [np.asarray(values, dtype=np.float64) for values in prediction.edge_scores]


def _audit_model(
    name: str,
    frozen: Mapping[str, Any],
    bundle,
    raw_scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    margin: float,
) -> dict[str, object]:
    payload_reports: dict[str, object] = {}
    all_truth: list[dict[str, object]] = []
    for payload_id in PAYLOADS:
        sets, scores = _filter_sets(bundle.adjacent_sets, raw_scores, payload_id)
        report = assign_and_audit_payload(
            sets,
            scores,
            scores,
            config,
            complete_route_scores_by_event=None,
        )
        attached_truth: list[dict[str, object]] = []
        by_event: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for item in report["truth_chains"]:
            key = (item["sample_id"], item["payload_id"], int(item["run_id"]), int(item["event_id"]))
            by_event.setdefault(key, []).append(item)
        grouped_sets: dict[tuple[object, ...], list[object]] = {}
        grouped_scores: dict[tuple[object, ...], dict[tuple[int, int], np.ndarray]] = {}
        for candidate_set, values in zip(sets, scores):
            event = candidate_set.event
            key = (
                str(candidate_set.sample.source_id),
                str(candidate_set.sample.payload_id),
                int(event.run_id),
                int(event.event_id),
            )
            grouped_sets.setdefault(key, []).append(candidate_set)
            grouped_scores.setdefault(key, {})[tuple(int(value) for value in candidate_set.station_pair)] = values
        for key, rows in by_event.items():
            event = next(candidate_set.event for candidate_set in grouped_sets[key])
            lookups = {}
            for candidate_set in grouped_sets[key]:
                pair = tuple(int(value) for value in candidate_set.station_pair)
                if pair not in config.score_threshold_by_pair:
                    continue
                lookups[pair] = _matrix_edge_lookup(
                    full_score_matrix(
                        event,
                        candidate_set.candidates,
                        grouped_scores[key][pair].tolist(),
                        pair[0],
                        pair[1],
                    ),
                    candidate_set.candidates,
                    float(config.score_threshold_by_pair[pair]),
                )
            feasible = enumerate_threshold_feasible_routes(event, lookups, float(config.unmatched_penalty))
            for row in rows:
                identified = attach_truth_route_identifiability(row, feasible, float(config.unmatched_penalty))
                attached_truth.append(attach_dustbin_aware_margin(identified, margin=margin))
        payload_reports[payload_id] = {
            "feasibility": summarize_feasibility(attached_truth),
            "identifiability": summarize_identifiability(attached_truth),
            "operating_truth_summary": report["truth_summary"],
            "selected_summary": report["selected_summary"],
            "assigned_events": report["assigned_events"],
        }
        all_truth.extend(attached_truth)
    return {
        "model": name,
        "checkpoint": frozen["metadata"]["checkpoint"],
        "checkpoint_sha256": frozen["metadata"]["checkpoint_sha256"],
        "payloads": payload_reports,
        "pooled_feasibility": summarize_feasibility(all_truth),
        "pooled_identifiability": summarize_identifiability(all_truth),
        "truth_chains": all_truth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/physical_four_station_dustbin_aware_objective_feasibility.yaml",
    )
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--control-frozen-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if args.split != "train":
        raise SystemExit("workbook 58 is train-only")
    if str(args.device) != "cuda":
        raise SystemExit("dustbin-aware feasibility scoring is GPU-only")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("control_id") != "dustbin_aware_objective_feasibility_v1":
        raise SystemExit("unexpected feasibility contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    if contract.get("condor_training_submitted") is not False:
        raise SystemExit("this audit must not submit Condor training")
    if float(contract["frozen"]["packing_margin"]) != float(PACKING_MARGIN):
        raise SystemExit("packing margin must stay at the workbook-56 registered 1.0")

    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(args.split,),
    )
    if {sample.split for sample in samples} != {args.split}:
        raise ValueError("audit loaded an unexpected split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    if constituents & DEVELOPMENT_SOURCES:
        raise SystemExit("refusing development-validation sources")
    if constituents & TRANSFER_SOURCES:
        raise SystemExit("refusing transfer-validation sources on the train-only audit")
    if constituents != TRAIN_SOURCES:
        raise SystemExit("train overlay sources are not the workbook-56 train μ± pair")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError("train overlay is missing payloads: " + ", ".join(sorted(missing)))
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")

    candidate_root = Path(args.candidate_frozen_output).expanduser().resolve()
    control_root = Path(args.control_frozen_output).expanduser().resolve()
    candidate_checkpoint = candidate_root / "route_aware_transformer_v2.pt"
    observed = _sha256(candidate_checkpoint)
    if observed != CANDIDATE_SHA256:
        raise SystemExit(f"candidate checkpoint sha256 {observed} != frozen {CANDIDATE_SHA256}")

    candidate = _load_v2(candidate_root, args.device)
    control = _load_v2(control_root, args.device)
    if candidate["metadata"]["checkpoint_sha256"] != CANDIDATE_SHA256:
        raise SystemExit("loaded candidate metadata hash disagrees with the frozen sha256")
    calibration = candidate["calibration"]
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
    config = _frozen_route_config()
    packing_audit = audit_workbook56_packing_loss_source()
    candidate_scores = _score_model(candidate, bundle, args.device, args.batch_size)
    control_scores = _score_model(control, bundle, args.device, args.batch_size)
    candidate_audit = _audit_model(
        "gauge_consistent_v2", candidate, bundle, candidate_scores, config, float(PACKING_MARGIN)
    )
    control_audit = _audit_model(
        "workbook54_retrained_v2", control, bundle, control_scores, config, float(PACKING_MARGIN)
    )
    comparison = compare_aligned_truth_routes(candidate_audit["truth_chains"], control_audit["truth_chains"])
    scale = compare_scale_distributions(candidate_audit["truth_chains"], control_audit["truth_chains"])
    decision = recommend_next_objective(
        candidate_audit["pooled_feasibility"],
        candidate_audit["pooled_identifiability"],
    )

    mining = {
        **packing_audit,
        "training_enumerator_includes_2_3_station_and_mixed_routes": True,
        "training_includes_shared_endpoint_conflicts": True,
        "production_dustbin_winner_missed_when_negative_utility_fragments_exist": int(
            candidate_audit["pooled_identifiability"]["dustbin_winner_missed_by_training_miner"]
        ),
        "miner_understates_production_gap": int(
            candidate_audit["pooled_feasibility"]["miner_understates_production_gap"]
        ),
        "miner_satisfied_production_fails": int(
            candidate_audit["pooled_feasibility"]["miner_satisfied_production_fails"]
        ),
    }
    _write_json(
        output / "audit_contract.json",
        {
            "contract": contract,
            "candidate_checkpoint_sha256": observed,
            "control_checkpoint_sha256": control["metadata"]["checkpoint_sha256"],
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": float(PACKING_MARGIN),
            "complete_route_query_injected_into_packing": False,
            "split": "train",
            "test_data_accessed": False,
            "development_validation_used": False,
            "condor_training_submitted": False,
        },
    )
    _write_json(output / "packing_loss_audit.json", packing_audit)
    _write_json(
        output / "candidate_summary.json",
        {key: candidate_audit[key] for key in ("model", "checkpoint", "checkpoint_sha256", "payloads", "pooled_feasibility", "pooled_identifiability")},
    )
    _write_json(
        output / "control_summary.json",
        {key: control_audit[key] for key in ("model", "checkpoint", "checkpoint_sha256", "payloads", "pooled_feasibility", "pooled_identifiability")},
    )
    _write_json(output / "model_comparison.json", {**comparison, "scale": scale})
    _write_json(
        output / "feasibility_report.json",
        {
            "candidate": candidate_audit["pooled_feasibility"],
            "control": control_audit["pooled_feasibility"],
            "mining": mining,
            "scale": scale,
        },
    )
    _write_json(output / "decision.json", decision)
    _write_jsonl(output / "candidate_truth_routes.jsonl", candidate_audit["truth_chains"])
    _write_jsonl(output / "control_truth_routes.jsonl", control_audit["truth_chains"])
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "packing_loss_audit": packing_audit,
                    "candidate_feasibility": candidate_audit["pooled_feasibility"],
                    "control_feasibility": control_audit["pooled_feasibility"],
                    "comparison": comparison,
                    "scale": scale,
                    "decision": decision,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
