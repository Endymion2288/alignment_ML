#!/usr/bin/env python3
"""Workbook 63: source-disjoint four-station training-diversity audit.

Scores the frozen workbook-62 checkpoint on the already-opened train,
development-validation, and transfer overlays, then compares those
distributions with unused non-sealed identity banks.  Does not train,
does not retune the objective or operating point, and does not open
sealed test or reserved blind sources.  Workbook-56 transfer is a
development diagnostic only.
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
from datasets.root_loader import load_events
from evaluation.route_metrics import _unique_truth_by_station
from scripts.audit_four_station_solver_hard_negatives import (
    FORBIDDEN_PAYLOADS,
    PAYLOADS,
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
from training.route_reduction_audit import attach_reduction_audit
from training.solver_hard_negative_audit import production_hypotheses
from training.source_diversity_audit import (
    CANDIDATE_DIAGNOSTIC_SOURCES,
    CURRENT_TRAIN_SOURCES,
    DEVELOPMENT_SOURCES,
    RESERVED_BLIND_SOURCES,
    TRANSFER_DIAGNOSTIC_SOURCES,
    UNUSED_RESERVE_SOURCES,
    WB62_CHECKPOINT_SHA256,
    attach_route_phase_space,
    assert_sources_allowed,
    coverage_report,
    recommend_diversity_next,
    summarize_identity_events,
    summarize_overlay_rows,
)


V3_PHYSICAL_ROOT = Path(
    "/eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_physical_v1/sources"
)
FOUR_STATION_TRAIN_ROOT = Path(
    "outputs/mc24_four_station_relative_association_retrain_v1/sources"
)
FOUR_STATION_TRANSFER_ROOT = Path(
    "outputs/mc24_four_station_relative_transfer_validation_v1/sources"
)
CANDIDATE_IDENTITY_POINTS = {
    "mc24_100043_00300_00399": "mag_0_train_00",
    "mc24_100044_00200_00299": "mag_0_train_00",
    "mc24_100047_00100_00149": "mag_0_validation_00",
    "mc24_100048_00100_00149": "mag_0_validation_00",
}
NAMESPACE_BY_ROLE = {
    "train": {9000000000: "mc24_100043_00200_00299", 9000000100: "mc24_100044_00300_00399"},
    "development": {9000000000: "mc24_100047_00050_00099", 9000000100: "mc24_100048_00050_00099"},
    "transfer": {9000000000: "mc24_100047_00300_00349", 9000000100: "mc24_100048_00300_00349"},
}


def _namespace_map(samples, fallback: Mapping[int, str]) -> dict[int, str]:
    mapping = dict(fallback)
    for sample in samples:
        descriptor = Path(
            str(sample.physical_payload_manifest)
        ).with_name("pooled_physical_descriptor.json")
        # Some samples keep the descriptor next to the overlay sample, not the payload.
        candidates = [
            descriptor,
            Path(str(sample.synthetic_tracklets)).parent / "pooled_physical_descriptor.json",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for block in payload.get("origin_namespaces") or []:
                source_id = str(block.get("source_id") or "")
                for item in block.get("run_id_mapping") or []:
                    mapping[int(item["namespaced_run_id"])] = source_id
            break
    return mapping


def _score_overlay(
    frozen: Mapping[str, Any],
    bundle,
    scores: Sequence[np.ndarray],
    *,
    source_by_namespaced_run: Mapping[int, str],
) -> list[dict[str, object]]:
    config = _frozen_route_config()
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
            all_rows.append(
                attach_route_phase_space(
                    attached,
                    event,
                    source_by_namespaced_run=source_by_namespaced_run,
                )
            )
    return all_rows


def _load_overlay(manifest: str, split: str, expected: set[str], *, refuse: set[str]):
    _, samples, loaded = load_synthetic_curriculum_manifest(
        manifest,
        require_all_splits=False,
        allowed_splits=(split,),
    )
    if {sample.split for sample in samples} != {split}:
        raise ValueError(f"expected only the {split} split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    assert_sources_allowed(constituents)
    if constituents & refuse:
        raise SystemExit(f"{split} overlay loaded a forbidden source")
    if constituents != expected:
        raise SystemExit(f"{split} overlay sources are not the frozen pair")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError("overlay is missing payloads: " + ", ".join(sorted(missing)))
    if loaded.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")
    return samples, loaded


def _score_role(
    frozen: Mapping[str, Any],
    manifest: str,
    split: str,
    role: str,
    expected: set[str],
    refuse: set[str],
    device: str,
    batch_size: int,
) -> list[dict[str, object]]:
    samples, _loaded = _load_overlay(manifest, split, expected, refuse=refuse)
    namespace = _namespace_map(samples, NAMESPACE_BY_ROLE[role])
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    scores = _score_model(frozen, bundle, device, batch_size)
    return _score_overlay(frozen, bundle, scores, source_by_namespaced_run=namespace)


def _identity_tracklets(source_id: str, root: Path, point: str) -> Path:
    return root / source_id / "physical_scan" / "points" / point / "refit" / "tracklets.root"


def _load_identity(source_id: str, path: Path) -> dict[str, object]:
    assert_sources_allowed([source_id])
    events = load_events(path, require_mc_labels=True)
    summary = summarize_identity_events(events, source_id=source_id)
    summary["tracklets"] = str(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/physical_four_station_training_diversity_feasibility.yaml")
    parser.add_argument(
        "--train-manifest",
        default="outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
    )
    parser.add_argument(
        "--transfer-manifest",
        default="outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
    )
    parser.add_argument(
        "--candidate-frozen-output",
        default="outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_four_station_hard_aware_reduction_v1/training_diversity_audit_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("training-diversity overlay scoring is GPU-only")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("control_id") != "source_disjoint_training_diversity_audit_v1":
        raise SystemExit("unexpected training-diversity audit contract")
    if contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")
    if contract.get("workbook56_transfer_is_final_gate") is not False:
        raise SystemExit("workbook-56 transfer must not remain a final gate")
    if contract.get("retune_loss_weight_reduction_or_operating_point") is not False:
        raise SystemExit("diversity audit must not retune the frozen workbook-62 objective")
    if float(contract["frozen"]["packing_margin"]) != float(PACKING_MARGIN):
        raise SystemExit("packing margin must stay at the registered 1.0")
    reserved = set(contract.get("reserved_blind_sources") or [])
    if reserved != set(RESERVED_BLIND_SOURCES):
        raise SystemExit("reserved blind sources drifted from the frozen audit list")
    assert_sources_allowed(CURRENT_TRAIN_SOURCES | DEVELOPMENT_SOURCES | TRANSFER_DIAGNOSTIC_SOURCES | set(CANDIDATE_DIAGNOSTIC_SOURCES))

    root = Path(args.candidate_frozen_output).expanduser().resolve()
    observed = _sha256(root / "route_aware_transformer_v2.pt")
    if observed != WB62_CHECKPOINT_SHA256:
        raise SystemExit(f"candidate checkpoint sha256 {observed} != frozen workbook-62 {WB62_CHECKPOINT_SHA256}")
    frozen = _load_v2(root, args.device)
    if frozen["metadata"]["checkpoint_sha256"] != WB62_CHECKPOINT_SHA256:
        raise SystemExit("loaded metadata hash disagrees with the frozen workbook-62 sha256")
    calibration = frozen["calibration"]
    if calibration.get("fit_split") != "identity_frozen_pre_training" or calibration.get("identity_map") is not True:
        raise SystemExit("checkpoint is not the frozen identity Platt")

    overlays = {
        "train": {
            "manifest": args.train_manifest,
            "split": "train",
            "expected": set(CURRENT_TRAIN_SOURCES),
            "refuse": set(DEVELOPMENT_SOURCES) | set(TRANSFER_DIAGNOSTIC_SOURCES) | set(RESERVED_BLIND_SOURCES),
        },
        "development": {
            "manifest": args.train_manifest,
            "split": "validation",
            "expected": set(DEVELOPMENT_SOURCES),
            "refuse": set(CURRENT_TRAIN_SOURCES) | set(TRANSFER_DIAGNOSTIC_SOURCES) | set(RESERVED_BLIND_SOURCES),
        },
        "transfer": {
            "manifest": args.transfer_manifest,
            "split": "validation",
            "expected": set(TRANSFER_DIAGNOSTIC_SOURCES),
            "refuse": set(CURRENT_TRAIN_SOURCES) | set(DEVELOPMENT_SOURCES) | set(RESERVED_BLIND_SOURCES),
        },
    }
    scored: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    for role, spec in overlays.items():
        split_dir = output / role
        summary_path = split_dir / "summary.json"
        rows_path = split_dir / "truth_routes.jsonl"
        if summary_path.is_file() and rows_path.is_file():
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        else:
            if split_dir.exists() and any(split_dir.iterdir()):
                raise FileExistsError(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
            rows = _score_role(
                frozen,
                spec["manifest"],
                spec["split"],
                role,
                spec["expected"],
                spec["refuse"],
                args.device,
                args.batch_size,
            )
            _write_jsonl(rows_path, rows)
        scored[role] = rows
        summaries[role] = summarize_overlay_rows(rows)
        _write_json(summary_path, summaries[role])

    identity: dict[str, object] = {}
    for source_id, point in (
        ("mc24_100043_00200_00299", "iteration_00_reference"),
        ("mc24_100044_00300_00399", "iteration_00_reference"),
        ("mc24_100047_00050_00099", "iteration_00_reference"),
        ("mc24_100048_00050_00099", "iteration_00_reference"),
    ):
        path = _identity_tracklets(source_id, FOUR_STATION_TRAIN_ROOT, point)
        identity[source_id] = _load_identity(source_id, path)
    for source_id in TRANSFER_DIAGNOSTIC_SOURCES:
        path = _identity_tracklets(source_id, FOUR_STATION_TRANSFER_ROOT, "iteration_00_reference")
        identity[source_id] = _load_identity(source_id, path)
    candidates: dict[str, object] = {}
    for source_id, point in CANDIDATE_IDENTITY_POINTS.items():
        path = _identity_tracklets(source_id, V3_PHYSICAL_ROOT, point)
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates[source_id] = _load_identity(source_id, path)
        identity[f"candidate:{source_id}"] = candidates[source_id]

    report = coverage_report(scored["train"], scored["transfer"], development_rows=scored["development"])
    decision = recommend_diversity_next(report)
    if decision["workbook56_transfer_is_final_gate"] is not False:
        raise SystemExit("decision accidentally restored the burned transfer gate")
    if set(decision["reserved_blind_sources"]) & set(scored) or set(RESERVED_BLIND_SOURCES) & set(identity):
        raise SystemExit("reserved blind sources were loaded")

    _write_json(
        output / "audit_contract.json",
        {
            "contract": contract,
            "candidate_checkpoint_sha256": observed,
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": float(PACKING_MARGIN),
            "workbook56_transfer_role": "development_diagnostic_only",
            "reserved_blind_sources_loaded": False,
            "unused_reserve_sources_loaded": False,
            "sealed_sources_loaded": False,
            "test_data_accessed": False,
            "retune_loss_weight_reduction_or_operating_point": False,
        },
    )
    _write_json(output / "overlay_summaries.json", summaries)
    _write_json(output / "identity_summaries.json", {"four_station_or_v3": identity, "candidates": candidates})
    _write_json(output / "coverage_report.json", report)
    _write_json(output / "decision.json", decision)
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "coverage_class": decision["coverage_class"],
                    "authorize_new_training_sources": decision["authorize_new_training_sources"],
                    "next_step": decision["next_step"],
                    "phase_space_failures": decision["phase_space_failures"],
                    "source_characteristic_failures": decision["source_characteristic_failures"],
                    "reserved_blind_sources": list(RESERVED_BLIND_SOURCES),
                    "unused_reserve_not_loaded": list(UNUSED_RESERVE_SOURCES),
                    "workbook56_transfer_is_final_gate": False,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
