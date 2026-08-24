#!/usr/bin/env python3
"""Workbook 60: solver-generated hard-negative mining audit.

Scores the frozen workbook-59 checkpoint, then for every complete truth
route calls the production unit-capacity hypothesis set.  Transfer results
are not used to pick loss weights, margin, or the operating point.
Sealed test is never opened.  GPU-only scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.route_metrics import _unique_truth_by_station
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_aware_transformer import predict_route_aware_scores
from training.route_assignment import assign_adjacent_route_sets, prepare_route_assignment_context
from training.route_operating_audit import STATION_PATH, audit_event_truth_chains, pair_tables_from_sets
from training.solver_hard_negative_audit import (
    WB59_CHECKPOINT_SHA256,
    align_origin_twins,
    attach_solver_hard_negative,
    audit_workbook59_miner_does_not_unroll_solver,
    production_hypotheses,
    recommend_next_objective,
    summarize_hard_negatives,
    summarize_twin_efficiency_gap,
)


FORBIDDEN_PAYLOADS = {
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
}
DEVELOPMENT_SOURCES = {"mc24_100047_00050_00099", "mc24_100048_00050_00099"}
TRANSFER_SOURCES = {"mc24_100047_00300_00349", "mc24_100048_00300_00349"}
TRAIN_SOURCES = {"mc24_100043_00200_00299", "mc24_100044_00300_00399"}
PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_hard_s3_ry",
    "iteration_00_hard_s3_ry_plus_common",
    "iteration_00_draw_00",
    "iteration_00_draw_00_plus_common",
    "iteration_00_draw_01",
    "iteration_00_draw_01_plus_common",
)
TWIN_FAMILIES = {
    "draw_00": ("iteration_00_draw_00", "iteration_00_draw_00_plus_common"),
    "draw_01": ("iteration_00_draw_01", "iteration_00_draw_01_plus_common"),
    "hard_s3_ry": ("iteration_00_hard_s3_ry", "iteration_00_hard_s3_ry_plus_common"),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"payload {payload_id} is absent from the loaded sets")
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


def _audit_split(
    frozen: Mapping[str, Any],
    bundle,
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
) -> dict[str, object]:
    payload_rows: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    for payload_id in PAYLOADS:
        sets, values = _filter_sets(bundle.adjacent_sets, scores, payload_id)
        context = prepare_route_assignment_context(sets, values, 15, config.station_path)
        assigned = assign_adjacent_route_sets(sets, values, config, 15, context=context)
        tables = pair_tables_from_sets(sets, values, values)
        assigned_by_key = {item.key: item for item in assigned}
        groups_by_key = {group.key: group for group in context.groups}
        rows: list[dict[str, object]] = []
        event_truth_counts: dict[tuple[object, ...], int] = Counter()
        pending = []
        for group in context.groups:
            event = group.event
            item = assigned_by_key[group.key]
            pair_tables = tables[group.key]
            _, unique_by_index = _unique_truth_by_station(event, STATION_PATH)
            hypotheses = production_hypotheses(event, group.station_matrices, config)
            truth_rows = audit_event_truth_chains(
                event,
                group.station_matrices,
                pair_tables,
                config,
                item.result,
            )
            event_key = (group.key[2], group.key[3])
            event_truth_counts[event_key] += len(truth_rows)
            for row in truth_rows:
                row["sample_id"] = group.key[0]
                row["payload_id"] = group.key[1]
                row["run_id"] = group.key[2]
                row["event_id"] = group.key[3]
                pending.append((event_key, row, event, unique_by_index, pair_tables, hypotheses, item.result.routes))
        for event_key, row, event, unique_by_index, pair_tables, hypotheses, selected_routes in pending:
            attached = attach_solver_hard_negative(
                row,
                event=event,
                unique_by_index=unique_by_index,
                pair_tables=pair_tables,
                hypotheses=hypotheses,
                selected_routes=selected_routes,
                n_complete_truth_in_event=int(event_truth_counts[event_key]),
                margin=float(PACKING_MARGIN),
            )
            rows.append(attached)
        payload_rows[payload_id] = rows
        all_rows.extend(rows)
        del groups_by_key
    pooled = summarize_hard_negatives(all_rows)
    pooled["payloads"] = {
        name: summarize_hard_negatives(rows) for name, rows in payload_rows.items()
    }
    twins = {}
    for family, (chart, twin) in TWIN_FAMILIES.items():
        aligned, pair_stats = align_origin_twins(payload_rows[chart], payload_rows[twin])
        twins[family] = {
            "alignment": {**summarize_twin_efficiency_gap(aligned), **pair_stats},
            "lost_routes": [
                row
                for row in aligned
                if row["chart_selected"] and not row["twin_selected"]
            ],
        }
    return {
        "pooled": pooled,
        "twins": twins,
        "rows": all_rows,
        "payload_rows": payload_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/physical_four_station_solver_hard_negative_feasibility.yaml",
    )
    parser.add_argument("--split", choices=("transfer", "train"), required=True)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("solver hard-negative audit scoring is GPU-only")

    output = Path(args.output_dir).expanduser().resolve()
    split_dir = output / args.split
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("control_id") != "solver_hard_negative_mining_feasibility_v1":
        raise SystemExit("unexpected hard-negative feasibility contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    if contract.get("condor_training_submitted") is not False:
        raise SystemExit("this audit must not submit Condor training")
    if float(contract["frozen"]["packing_margin"]) != float(PACKING_MARGIN):
        raise SystemExit("packing margin must stay at the registered 1.0")

    allowed = "validation" if args.split == "transfer" else "train"
    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(allowed,),
    )
    if {sample.split for sample in samples} != {allowed}:
        raise ValueError("audit loaded an unexpected split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    if constituents & DEVELOPMENT_SOURCES:
        raise SystemExit("refusing development-validation sources")
    if args.split == "transfer":
        if constituents & TRAIN_SOURCES:
            raise SystemExit("transfer audit must not load the train pair")
        if constituents != TRANSFER_SOURCES:
            raise SystemExit("transfer overlay sources are not the unused μ± pair")
    else:
        if constituents & TRANSFER_SOURCES:
            raise SystemExit("train-only audit must not load the transfer pair")
        if constituents != TRAIN_SOURCES:
            raise SystemExit("train overlay sources are not the workbook-56 train μ± pair")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError("overlay is missing payloads: " + ", ".join(sorted(missing)))
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")

    root = Path(args.candidate_frozen_output).expanduser().resolve()
    observed = _sha256(root / "route_aware_transformer_v2.pt")
    if observed != WB59_CHECKPOINT_SHA256:
        raise SystemExit(f"candidate checkpoint sha256 {observed} != frozen workbook-59 {WB59_CHECKPOINT_SHA256}")
    frozen = _load_v2(root, args.device)
    if frozen["metadata"]["checkpoint_sha256"] != WB59_CHECKPOINT_SHA256:
        raise SystemExit("loaded metadata hash disagrees with the frozen workbook-59 sha256")
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
    audit = _audit_split(frozen, bundle, scores, _frozen_route_config())
    miner_source = audit_workbook59_miner_does_not_unroll_solver()
    decision = None
    if args.split == "train":
        decision = recommend_next_objective(audit["pooled"])

    _write_json(
        split_dir / "audit_contract.json",
        {
            "contract": contract,
            "split": args.split,
            "candidate_checkpoint_sha256": observed,
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": float(PACKING_MARGIN),
            "production_hypothesis_source": "baselines.route_assignment._route_hypotheses",
            "complete_route_query_injected_into_packing": False,
            "test_data_accessed": False,
            "development_validation_used": False,
            "transfer_used_to_pick_weights": False,
        },
    )
    _write_json(split_dir / "miner_source_audit.json", miner_source)
    _write_json(split_dir / "summary.json", audit["pooled"])
    _write_json(split_dir / "twin_alignment.json", audit["twins"])
    if decision is not None:
        _write_json(split_dir / "decision.json", decision)
    _write_jsonl(
        split_dir / "unselected_truth_routes.jsonl",
        [row for row in audit["rows"] if not row.get("selected")],
    )
    _write_jsonl(
        split_dir / "fragment_winner_truth_routes.jsonl",
        [row for row in audit["rows"] if row.get("production_fragment_winner")],
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "split": args.split,
                    "output_dir": str(split_dir),
                    "pooled": {
                        key: audit["pooled"][key]
                        for key in (
                            "complete_truth_chains",
                            "selected",
                            "production_fragment_winners",
                            "workbook59_truth_lt_fragment",
                            "fragment_winner_topology",
                            "fragment_winner_composition",
                            "fraction_winner_in_miner",
                            "fraction_winner_is_strongest",
                            "fraction_missed_training_signal",
                            "mean_dustbin_aware_loss_on_fragment_wins",
                            "mean_oracle_loss_on_fragment_wins",
                        )
                    },
                    "draw_01_twin": audit["twins"]["draw_01"]["alignment"],
                    "decision": decision,
                    "miner_source_audit": miner_source,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
