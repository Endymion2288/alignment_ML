#!/usr/bin/env python3
"""Workbook 62: matched mechanism comparison for the max-reduction control.

Scores the new checkpoint and the frozen workbook-59 checkpoint on the
same train overlay, then the new checkpoint on the already-opened
source-disjoint transfer overlay.  Does not pick a reduction, weight,
or operating point.  Sealed test is never opened.  GPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.audit_four_station_solver_hard_negatives import (
    DEVELOPMENT_SOURCES,
    FORBIDDEN_PAYLOADS,
    PAYLOADS,
    TRAIN_SOURCES,
    TRANSFER_SOURCES,
    TWIN_FAMILIES,
    _frozen_route_config,
    _score_model,
    _write_json,
    _write_jsonl,
)
from scripts.audit_four_station_weighting_reduction import _audit_train
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.hard_aware_reduction_eval import (
    bin_transition_counts,
    classify_stop_reason,
    classify_train_reduction_effect,
    compact_reduction_row,
    match_control_candidate,
)
from training.route_reduction_audit import summarize_reduction_audit
from training.solver_hard_negative_audit import align_origin_twins, summarize_twin_efficiency_gap


def _load_overlay(manifest: str, split: str, expected_sources: set[str], refuse_sources: set[str]):
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
    if constituents & DEVELOPMENT_SOURCES:
        raise SystemExit("refusing development-validation sources")
    if constituents & refuse_sources:
        raise SystemExit(f"{split} overlay loaded a forbidden source pair")
    if constituents != expected_sources:
        raise SystemExit(f"{split} overlay sources are not the frozen μ± pair")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError("overlay is missing payloads: " + ", ".join(sorted(missing)))
    if loaded.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest lacks physical geometry repropagation")
    return samples, loaded


def _score_split(frozen: Mapping[str, Any], manifest: str, split: str, expected: set[str], refuse: set[str], device: str, batch_size: int):
    samples, loaded = _load_overlay(manifest, split, expected, refuse)
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    scores = _score_model(frozen, bundle, device, batch_size)
    audit = _audit_train(frozen, bundle, scores)
    return audit, loaded


def _load_checkpoint(root: Path, device: str) -> dict[str, Any]:
    frozen = _load_v2(root, device)
    calibration = frozen["calibration"]
    if calibration.get("fit_split") != "identity_frozen_pre_training" or calibration.get("identity_map") is not True:
        raise SystemExit("checkpoint is not the frozen identity Platt")
    return frozen


def _twin_block(rows: list[dict[str, object]]) -> dict[str, object]:
    by_payload = {name: [row for row in rows if row.get("payload_id") == name] for name in PAYLOADS}
    result: dict[str, object] = {}
    for family, (chart_name, twin_name) in TWIN_FAMILIES.items():
        aligned, stats = align_origin_twins(by_payload[chart_name], by_payload[twin_name])
        result[family] = {
            "alignment": stats,
            "efficiency_gap": summarize_twin_efficiency_gap(aligned),
            "population": {
                "chart_u_truth_median": _median_field(by_payload[chart_name], "u_truth"),
                "twin_u_truth_median": _median_field(by_payload[twin_name], "u_truth"),
                "chart_selected": int(sum(1 for row in by_payload[chart_name] if row.get("selected"))),
                "twin_selected": int(sum(1 for row in by_payload[twin_name] if row.get("selected"))),
                "chart_fragment_winners": int(
                    sum(1 for row in by_payload[chart_name] if row.get("production_fragment_winner"))
                ),
                "twin_fragment_winners": int(
                    sum(1 for row in by_payload[twin_name] if row.get("production_fragment_winner"))
                ),
            },
        }
    return result


def _median_field(rows: list[Mapping[str, Any]], key: str) -> float | None:
    from training.route_utility_identifiability import _median

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(_median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--control-frozen-output", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--transfer-manifest", required=True)
    parser.add_argument("--gate-decision-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("hard-aware reduction comparison is GPU-only")
    if Path(args.transfer_manifest).resolve() == Path(args.train_manifest).resolve():
        raise SystemExit("train and transfer manifests must stay distinct")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    candidate_root = Path(args.candidate_frozen_output).expanduser().resolve()
    control_root = Path(args.control_frozen_output).expanduser().resolve()
    candidate = _load_checkpoint(candidate_root, args.device)
    control = _load_checkpoint(control_root, args.device)
    candidate_sha = _sha256(candidate_root / "route_aware_transformer_v2.pt")
    control_sha = _sha256(control_root / "route_aware_transformer_v2.pt")
    if control_sha != "6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc":
        raise SystemExit("control checkpoint is not the frozen workbook-59 artifact")

    train_control, _ = _score_split(
        control, args.train_manifest, "train", TRAIN_SOURCES, TRANSFER_SOURCES, args.device, args.batch_size
    )
    train_candidate, _ = _score_split(
        candidate, args.train_manifest, "train", TRAIN_SOURCES, TRANSFER_SOURCES, args.device, args.batch_size
    )
    transfer_candidate, _ = _score_split(
        candidate, args.transfer_manifest, "validation", TRANSFER_SOURCES, TRAIN_SOURCES, args.device, args.batch_size
    )
    matched = match_control_candidate(train_control["rows"], train_candidate["rows"])
    train_effect = classify_train_reduction_effect(matched, margin=float(PACKING_MARGIN))
    gates = json.loads(Path(args.gate_decision_json).read_text(encoding="utf-8"))
    if gates.get("test_data_accessed") is not False:
        raise SystemExit("gate decision opened sealed test")
    passed = bool(gates.get("continue_to_15d_relative_wls"))
    stop = classify_stop_reason(gates_passed=passed, train_effect=train_effect)
    stop["candidate_checkpoint_sha256"] = candidate_sha
    stop["control_checkpoint_sha256"] = control_sha
    stop["gates_layer1"] = bool((gates.get("layer1_raw_candidate_recall") or {}).get("ok"))
    stop["gates_layer2"] = bool((gates.get("layer2_association") or {}).get("ok"))
    stop["gates_layer3"] = bool((gates.get("layer3_gauge") or {}).get("ok"))

    control_train = summarize_reduction_audit(train_control["rows"])
    candidate_train = summarize_reduction_audit(train_candidate["rows"])
    candidate_transfer = summarize_reduction_audit(transfer_candidate["rows"])
    twins = _twin_block(transfer_candidate["rows"])

    _write_json(
        output / "comparison_contract.json",
        {
            "candidate_checkpoint_sha256": candidate_sha,
            "control_checkpoint_sha256": control_sha,
            "identity_platt": True,
            "thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
            "unmatched_penalty": -1.0,
            "packing_margin": float(PACKING_MARGIN),
            "reduction": "max_over_complete_truth_routes_in_event",
            "test_data_accessed": False,
            "development_validation_used": False,
            "transfer_used_to_pick_reduction_or_weight": False,
        },
    )
    _write_json(output / "train_control_summary.json", control_train)
    _write_json(output / "train_candidate_summary.json", candidate_train)
    _write_json(output / "transfer_candidate_summary.json", candidate_transfer)
    _write_json(
        output / "train_matched_short_boundary.json",
        {
            **{key: value for key, value in matched.items() if key != "rows"},
            "train_reduction_effect": train_effect,
            "short_bin_transitions": bin_transition_counts(matched["rows"], "control_short_bin", "candidate_short_bin"),
            "margin_bin_transitions": bin_transition_counts(matched["rows"], "control_bin", "candidate_bin"),
        },
    )
    _write_json(output / "transfer_twin_alignment.json", twins)
    _write_json(output / "decision.json", stop)
    _write_jsonl(output / "train_candidate_truth_routes.jsonl", [compact_reduction_row(row) for row in train_candidate["rows"]])
    _write_jsonl(
        output / "transfer_candidate_truth_routes.jsonl",
        [compact_reduction_row(row) for row in transfer_candidate["rows"]],
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "candidate_checkpoint_sha256": candidate_sha,
                    "train_effect": train_effect,
                    "decision": stop,
                    "train_candidate": {
                        "hard": candidate_train["hard"],
                        "near_boundary": candidate_train["near_boundary"],
                        "short_station_near_boundary": candidate_train["short_station_near_boundary"],
                        "n_event_max_is_short_boundary": candidate_train["n_event_max_is_short_boundary"],
                    },
                    "transfer_candidate": {
                        "hard": candidate_transfer["hard"],
                        "near_boundary": candidate_transfer["near_boundary"],
                        "short_station_near_boundary": candidate_transfer["short_station_near_boundary"],
                        "hard_production_winners": candidate_transfer["hard_production_winners"],
                    },
                    "draw_01_twin": twins["draw_01"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
