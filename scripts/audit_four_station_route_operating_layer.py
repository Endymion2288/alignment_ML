#!/usr/bin/env python3
"""Validation-only operating-layer audit for the matched four-station V2.

Reuses the frozen retrained V2 checkpoint, mode-0 candidate graph, unit-capacity
solver, and validation-selected packing constants.  It does not retune unmatched
penalty, does not open workbook-52 held-outs, and does not start 15-DoF WLS.

The audit aligns gauge twins that share ΔT_ij (chart vs left-SE(3) common
transform) event-by-event and decomposes complete-truth efficiency loss into
station-pair thresholds, non-positive route utility, or unit-capacity packing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
)
from training.route_aware_transformer import predict_route_aware_scores, route_query_score_maps_by_event
from training.route_operating_audit import (
    align_gauge_twins,
    assign_and_audit_payload,
    summarize_twin_alignment,
)


FORBIDDEN_PAYLOADS = {
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
}
FOCUS_CHART = "iteration_00_draw_00"
FOCUS_TWIN = "iteration_00_draw_00_plus_common"
NOMINAL = "iteration_00_reference"
ALLOWED_SPLITS = ("validation",)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


def _filter_sets(sets: Sequence[object], scores: Sequence[np.ndarray], payload_id: str):
    kept_sets = []
    kept_scores = []
    for candidate_set, values in zip(sets, scores):
        if str(candidate_set.sample.payload_id) == payload_id:
            kept_sets.append(candidate_set)
            kept_scores.append(np.asarray(values, dtype=np.float64))
    if not kept_sets:
        raise ValueError(f"payload {payload_id} is absent from the loaded validation sets")
    return kept_sets, kept_scores


def _filter_route_maps(maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]], payload_id: str):
    return {key: value for key, value in maps.items() if str(key[1]) == payload_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if args.batch_size < 1 or args.calibration_bins < 2:
        parser.error("batch size must be positive and calibration bins must be at least two")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    _manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=(args.split,),
    )
    if {sample.split for sample in samples} != {args.split}:
        raise ValueError("operating-layer audit loaded an unexpected split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    if manifest.get("physical_geometry_repropagation") is not True or int(
        manifest.get("q_over_p_mode", -1)
    ) != int(args.q_over_p_mode):
        raise ValueError("synthetic manifest lacks a physical mode-0 candidate contract")
    focus_payloads = (NOMINAL, FOCUS_CHART, FOCUS_TWIN)
    missing = set(focus_payloads) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError("validation manifest is missing required payloads: " + ", ".join(sorted(missing)))
    samples = [sample for sample in samples if str(sample.payload_id) in set(focus_payloads)]

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
        batch_size=args.batch_size,
    )
    raw_scores = list(prediction.edge_scores)
    calibrated = apply_frozen_transformer_calibration(bundle.adjacent_sets, raw_scores, frozen["calibration"])
    route_maps = route_query_score_maps_by_event(
        prediction.route_score_sets,
        tuple(np.asarray(score_set.scores, dtype=np.float64) for score_set in prediction.route_score_sets),
    )

    payload_reports: dict[str, object] = {}
    truth_by_payload: dict[str, list] = {}
    selected_by_payload: dict[str, list] = {}
    for payload_id in focus_payloads:
        sets, raw = _filter_sets(bundle.adjacent_sets, raw_scores, payload_id)
        _, cal = _filter_sets(bundle.adjacent_sets, calibrated, payload_id)
        report = assign_and_audit_payload(
            sets,
            raw,
            cal,
            frozen["route"],
            complete_route_scores_by_event=_filter_route_maps(route_maps, payload_id),
            calibration_bins=args.calibration_bins,
        )
        payload_reports[payload_id] = {
            "truth_summary": report["truth_summary"],
            "selected_summary": report["selected_summary"],
            "assigned_events": report["assigned_events"],
        }
        truth_by_payload[payload_id] = report["truth_chains"]
        selected_by_payload[payload_id] = report["selected_routes"]
        payload_dir = output / payload_id
        payload_dir.mkdir(parents=True, exist_ok=True)
        _write_json(payload_dir / "truth_summary.json", report["truth_summary"])
        _write_json(payload_dir / "selected_summary.json", report["selected_summary"])
        _write_jsonl(payload_dir / "truth_chains.jsonl", report["truth_chains"])
        _write_jsonl(payload_dir / "selected_routes.jsonl", report["selected_routes"])

    aligned = align_gauge_twins(truth_by_payload[FOCUS_CHART], truth_by_payload[FOCUS_TWIN])
    twin_summary = summarize_twin_alignment(aligned)
    _write_jsonl(output / "draw_00_gauge_twin_alignment.jsonl", aligned)
    _write_json(output / "draw_00_gauge_twin_summary.json", twin_summary)

    decision = {
        "continue_to_15d_relative_wls": False,
        "architecture_or_solver_changed": False,
        "unmatched_penalty_scanned": False,
        "workbook_52_held_outs_opened": False,
        "focus_payload": FOCUS_TWIN,
        "chart_payload": FOCUS_CHART,
        "nominal_payload": NOMINAL,
        "frozen_checkpoint_sha256": frozen["metadata"]["checkpoint_sha256"],
        "frozen_thresholds": frozen["metadata"]["thresholds"],
        "frozen_unmatched_penalty": frozen["metadata"]["unmatched_penalty"],
        "payloads": payload_reports,
        "gauge_twin_alignment": twin_summary,
        "interpretation_notes": {
            "packing_uses_edge_log_odds_only": True,
            "complete_route_query_is_diagnostic_not_in_solver": True,
            "historical_penalty_counterfactual_is_diagnostic_only": True,
            "platt_preserves_within_pair_ranking": True,
        },
    }
    _write_json(output / "operating_layer_audit.json", decision)
    (output / "frozen_checkpoint.sha256").write_text(
        str(frozen["metadata"]["checkpoint_sha256"]) + "\n", encoding="utf-8"
    )
    _write_json(
        output / "run_contract.json",
        {
            "split": args.split,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "test_opened": False,
            "unmatched_penalty_override": None,
            "threshold_scale_override": None,
            "checkpoint_sha256": _sha256(Path(args.frozen_output).expanduser().resolve() / "route_aware_transformer_v2.pt"),
        },
    )


if __name__ == "__main__":
    main()
