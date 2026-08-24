#!/usr/bin/env python3
"""Workbook 61: train-only weighting / reduction feasibility.

Scores the frozen workbook-59 checkpoint on the train overlay only.
Transfer is not loaded and is not used to pick a reduction or a weight.
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

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.route_metrics import _unique_truth_by_station
from scripts.audit_four_station_solver_hard_negatives import (
    DEVELOPMENT_SOURCES,
    FORBIDDEN_PAYLOADS,
    PAYLOADS,
    TRAIN_SOURCES,
    TRANSFER_SOURCES,
    _filter_sets,
    _frozen_route_config,
    _score_model,
    _write_json,
    _write_jsonl,
)
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_assignment import assign_adjacent_route_sets, prepare_route_assignment_context
from training.route_operating_audit import STATION_PATH, audit_event_truth_chains, pair_tables_from_sets
from training.route_reduction_audit import (
    attach_reduction_audit,
    recommend_next_objective,
    summarize_reduction_audit,
)
from training.solver_hard_negative_audit import WB59_CHECKPOINT_SHA256, production_hypotheses


def _audit_train(frozen: Mapping[str, Any], bundle, scores: Sequence[np.ndarray]) -> dict[str, object]:
    config = _frozen_route_config()
    payload_rows: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    for payload_id in PAYLOADS:
        sets, values = _filter_sets(bundle.adjacent_sets, scores, payload_id)
        context = prepare_route_assignment_context(sets, values, 15, config.station_path)
        assigned = assign_adjacent_route_sets(sets, values, config, 15, context=context)
        tables = pair_tables_from_sets(sets, values, values)
        assigned_by_key = {item.key: item for item in assigned}
        pending = []
        event_truth_counts: dict[tuple[object, ...], int] = Counter()
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
            event_key = (group.key[0], group.key[1], group.key[2], group.key[3])
            event_truth_counts[event_key] += len(truth_rows)
            for row in truth_rows:
                row["sample_id"] = group.key[0]
                row["payload_id"] = group.key[1]
                row["run_id"] = group.key[2]
                row["event_id"] = group.key[3]
                pending.append((event_key, row, event, unique_by_index, pair_tables, hypotheses))
        rows: list[dict[str, object]] = []
        for event_key, row, event, unique_by_index, pair_tables, hypotheses in pending:
            attached = attach_reduction_audit(
                row,
                event=event,
                unique_by_index=unique_by_index,
                pair_tables=pair_tables,
                hypotheses=hypotheses,
                selected_routes=assigned_by_key[event_key].result.routes,
                n_complete_truth_in_event=int(event_truth_counts[event_key]),
                margin=float(PACKING_MARGIN),
            )
            rows.append(attached)
        payload_rows[payload_id] = rows
        all_rows.extend(rows)
    pooled = summarize_reduction_audit(all_rows)
    pooled["payloads"] = {name: summarize_reduction_audit(rows) for name, rows in payload_rows.items()}
    return {"pooled": pooled, "rows": all_rows, "payload_rows": payload_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/physical_four_station_weighting_reduction_feasibility.yaml",
    )
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("weighting/reduction audit scoring is GPU-only")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("control_id") != "weighting_reduction_feasibility_v1":
        raise SystemExit("unexpected weighting/reduction feasibility contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    if contract.get("condor_training_submitted") is not False:
        raise SystemExit("this audit must not submit Condor training")
    if float(contract["frozen"]["packing_margin"]) != float(PACKING_MARGIN):
        raise SystemExit("packing margin must stay at the registered 1.0")

    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train",),
    )
    if {sample.split for sample in samples} != {"train"}:
        raise ValueError("audit loaded an unexpected split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    if constituents & DEVELOPMENT_SOURCES:
        raise SystemExit("refusing development-validation sources")
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
    audit = _audit_train(frozen, bundle, scores)
    decision = recommend_next_objective(audit["pooled"])
    _write_json(
        output / "audit_contract.json",
        {
            "contract": contract,
            "split": "train",
            "candidate_checkpoint_sha256": observed,
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": float(PACKING_MARGIN),
            "production_hypothesis_source": "baselines.route_assignment._route_hypotheses",
            "test_data_accessed": False,
            "development_validation_used": False,
            "transfer_loaded": False,
            "transfer_used_to_pick_reduction_or_weight": False,
        },
    )
    _write_json(output / "summary.json", audit["pooled"])
    _write_json(output / "decision.json", decision)
    compact = []
    for row in audit["rows"]:
        compact.append(
            {
                "payload_id": row.get("payload_id"),
                "payload_family": row.get("payload_family"),
                "run_id": row.get("run_id"),
                "event_id": row.get("event_id"),
                "origin_run_id": row.get("origin_run_id"),
                "selected": row.get("selected"),
                "u_truth": row.get("u_truth"),
                "u_best_solver_fragment": row.get("u_best_solver_fragment"),
                "production_margin": row.get("production_margin"),
                "margin_bin": row.get("margin_bin"),
                "short_station_bin": row.get("short_station_bin"),
                "competitor_n_stations": row.get("competitor_n_stations"),
                "n_complete_truth_in_event": row.get("n_complete_truth_in_event"),
                "dustbin_aware_margin_loss": row.get("dustbin_aware_margin_loss"),
                "workbook56_packing_loss": row.get("workbook56_packing_loss"),
                "dustbin_mean_share": row.get("dustbin_mean_share"),
                "production_by_length": row.get("production_by_length"),
                "production_fragment_winner": row.get("production_fragment_winner"),
            }
        )
    _write_jsonl(output / "train_truth_routes.jsonl", compact)
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "pooled": {
                        key: audit["pooled"][key]
                        for key in (
                            "complete_truth_chains",
                            "n_events",
                            "margin_bin_counts",
                            "hard",
                            "near_boundary",
                            "easy",
                            "hard_production_winners",
                            "short_station_hard",
                            "short_station_near_boundary",
                            "four_station_boundary",
                            "short_boundary_route_fraction",
                            "n_event_max_is_short_boundary",
                            "median_event_multiplicity_on_boundary",
                            "boundary_share_of_mean_dustbin_loss",
                            "boundary_share_of_max_dustbin_loss",
                            "short_boundary_share_of_mean_dustbin_loss",
                            "length_margin_median",
                            "length_dustbin_loss_sum",
                            "production_bin_by_competitor_length",
                            "event_max_competitor_length_when_loss_positive",
                        )
                    },
                    "decision": decision,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
