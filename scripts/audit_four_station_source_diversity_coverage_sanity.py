#!/usr/bin/env python3
"""Workbook 64 train-only coverage sanity check.

Scores the frozen workbook-62 checkpoint on the new six-source train
overlay and compares it with the already-written workbook-63 failure
core.  Does not reselect sources, does not change the curriculum, and
does not load reserved blind or sealed test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.audit_four_station_solver_hard_negatives import FORBIDDEN_PAYLOADS, _score_model
from scripts.audit_four_station_training_diversity import _namespace_map, _score_overlay
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    HISTORY_ONLY_SOURCES,
    RESERVED_BLIND_SOURCES,
    WB62_CHECKPOINT_SHA256,
    assert_sources_allowed,
    coverage_expansion_report,
    select_failure_core,
    summarize_overlay_rows,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-train-manifest", required=True)
    parser.add_argument(
        "--old-train-jsonl",
        default="outputs/mc24_four_station_hard_aware_reduction_v1/training_diversity_audit_v1/train/truth_routes.jsonl",
    )
    parser.add_argument(
        "--transfer-jsonl",
        default="outputs/mc24_four_station_hard_aware_reduction_v1/training_diversity_audit_v1/transfer/truth_routes.jsonl",
    )
    parser.add_argument(
        "--candidate-frozen-output",
        default="outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_four_station_source_diversity_train_v1/coverage_sanity_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("coverage sanity is GPU-only")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not (output / "coverage_expansion.json").is_file():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    old_rows = _read_jsonl(Path(args.old_train_jsonl).expanduser().resolve())
    transfer_rows = _read_jsonl(Path(args.transfer_jsonl).expanduser().resolve())
    failure = select_failure_core(transfer_rows)
    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.new_train_manifest,
        require_all_splits=False,
        allowed_splits=("train",),
    )
    if {sample.split for sample in samples} != {"train"}:
        raise ValueError("coverage sanity must load only the train overlay")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    constituents = {source for sample in samples for source in sample.source_ids}
    assert_sources_allowed(constituents)
    if constituents & HISTORY_ONLY_SOURCES or constituents & set(RESERVED_BLIND_SOURCES):
        raise SystemExit("six-source train overlay leaked history-only or reserved sources")
    if constituents != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise SystemExit("six-source train overlay is not the authorized set")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("train overlay lacks physical geometry repropagation")

    root = Path(args.candidate_frozen_output).expanduser().resolve()
    observed = _sha256(root / "route_aware_transformer_v2.pt")
    if observed != WB62_CHECKPOINT_SHA256:
        raise SystemExit(f"coverage sanity must score frozen workbook 62, got {observed}")
    frozen = _load_v2(root, args.device)
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    scores = _score_model(frozen, bundle, args.device, args.batch_size)
    namespace = _namespace_map(samples, {})
    new_rows = _score_overlay(frozen, bundle, scores, source_by_namespaced_run=namespace)
    expansion = coverage_expansion_report(old_rows, new_rows, failure)
    expansion["workbook63_transfer_used_as"] = "frozen_failure_region_history_only"
    expansion["reselect_sources"] = False
    expansion["modify_curriculum"] = False
    _write_json(output / "new_train_summary.json", summarize_overlay_rows(new_rows))
    _write_json(output / "coverage_expansion.json", expansion)
    print(json.dumps(_json_ready(expansion), indent=2))


if __name__ == "__main__":
    main()
