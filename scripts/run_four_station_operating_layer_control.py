#!/usr/bin/env python3
"""Fit and freeze-eval the pre-registered operating-layer control.

Train-only 3-pair Platt on frozen retrained-V2 logits, then one validation
evaluation with the historical packing constants.  Workbook-52 held-outs,
unmatched-penalty grids, architecture changes, and 15-DoF WLS are refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig
from datasets.physical_curriculum import load_synthetic_curriculum_manifest, uniform_condition_axis
from models.field_route_fitter import field_aware_route_summary, selected_field_aware_route
from scripts.evaluate_four_station_matched_association import (
    FORBIDDEN_PAYLOADS,
    _adjacent_recovery,
    _families_from_plan,
    _gates,
    _gauge_audit,
)
from scripts.run_frozen_association_backbone import (
    _covariance_row,
    _field_candidates_by_event,
    _load_v2,
    _maximum_hypotheses,
    _route_endpoint_metadata,
    _route_origin_signature,
    _sha256,
    _state_row,
    _write_csv,
)
from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import assess, summarize
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    calibrate_transformer_scores,
    candidate_score_metrics,
)
from training.operating_layer_control import (
    historical_packing_config,
    nominal_quality_ok,
    tag_train_only_calibration,
)
from training.route_assignment import assign_adjacent_route_sets, evaluate_adjacent_route_assignment_sets
from training.route_aware_transformer import predict_route_aware_scores


PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_hard_s3_ry",
    "iteration_00_hard_s3_ry_plus_common",
    "iteration_00_draw_00",
    "iteration_00_draw_00_plus_common",
    "iteration_00_draw_01",
    "iteration_00_draw_01_plus_common",
)
NOMINAL = "iteration_00_reference"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_split(manifest: Path, split: str):
    path, samples, payload = load_synthetic_curriculum_manifest(
        manifest,
        require_all_splits=False,
        allowed_splits=(split,),
    )
    if {sample.split for sample in samples} != {split}:
        raise ValueError(f"operating-layer control loaded an unexpected {split} split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise ValueError(f"{split} manifest is missing payloads: " + ", ".join(sorted(missing)))
    return path, samples, payload


def _predict(frozen: Mapping[str, Any], samples, *, device: str, batch_size: int, q_over_p_mode: int):
    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(q_over_p_mode),
    )
    artifact = frozen["artifact"]
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode=artifact.context_mode)
    prediction = predict_route_aware_scores(
        frozen["model"],
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=device,
        batch_size=batch_size,
    )
    return bundle, list(prediction.edge_scores)


def _filter(sets: Sequence[object], scores: Sequence[np.ndarray], payload_id: str):
    kept_sets = []
    kept_scores = []
    for candidate_set, values in zip(sets, scores):
        if str(candidate_set.sample.payload_id) == payload_id:
            kept_sets.append(candidate_set)
            kept_scores.append(np.asarray(values, dtype=np.float64))
    if not kept_sets:
        raise ValueError(f"payload {payload_id} is absent")
    return kept_sets, kept_scores


def _emit_payload(
    *,
    output: Path,
    payload_id: str,
    sets: Sequence[object],
    raw_scores: Sequence[np.ndarray],
    calibrated: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    frozen: Mapping[str, Any],
    manifest_path: Path,
    split: str,
    full_split_sample_count: int,
    calibration_bins: int,
    q_over_p_mode: int,
    control_metadata: Mapping[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    field_candidates = _field_candidates_by_event(sets)
    selections = assign_adjacent_route_sets(sets, calibrated, config, calibration_bins=calibration_bins)
    evaluation = evaluate_adjacent_route_assignment_sets(
        sets, calibrated, config, calibration_bins=calibration_bins
    )
    field_edge_rows: list[dict[str, object]] = []
    all_field_routes = []
    for assigned in selections:
        event_candidates = field_candidates.get(tuple(assigned.key))
        if event_candidates is None:
            raise RuntimeError("selected route event is absent from the physical candidate lookup")
        for route_index, route in enumerate(assigned.result.routes):
            endpoint_metadata = _route_endpoint_metadata(assigned.event, route.endpoints)
            metadata_by_event_row = {int(item["event_row"]): item for item in endpoint_metadata}
            origin_signature = _route_origin_signature(endpoint_metadata)
            field_route = selected_field_aware_route(route, event_candidates)
            all_field_routes.append(field_route)
            for edge in field_route.edges:
                source = metadata_by_event_row[int(edge.source_index)]
                target = metadata_by_event_row[int(edge.target_index)]
                row: dict[str, object] = {
                    "sample_id": assigned.key[0],
                    "payload_id": assigned.key[1],
                    "run_id": assigned.key[2],
                    "event_id": assigned.key[3],
                    "route_index": route_index,
                    "route_origin_signature": origin_signature,
                    "route_endpoint_count": len(route.endpoints),
                    "source_station_id": int(edge.source_station_id),
                    "target_station_id": int(edge.target_station_id),
                    "source_event_row": int(edge.source_index),
                    "target_event_row": int(edge.target_index),
                    "source_synthetic_tracklet_id": int(source["synthetic_tracklet_id"]),
                    "target_synthetic_tracklet_id": int(target["synthetic_tracklet_id"]),
                    "source_origin_run_id": int(source["origin_run_id"]),
                    "source_origin_event_id": int(source["origin_event_id"]),
                    "source_origin_tracklet_id": int(source["origin_tracklet_id"]),
                    "target_origin_run_id": int(target["origin_run_id"]),
                    "target_origin_event_id": int(target["origin_event_id"]),
                    "target_origin_tracklet_id": int(target["origin_tracklet_id"]),
                    "source_synthetic_role": int(source["synthetic_role"]),
                    "target_synthetic_role": int(target["synthetic_role"]),
                    "score": float(edge.score),
                    "chi2": float(edge.chi2),
                }
                row.update(_state_row("residual", edge.residual))
                row.update(_state_row("pull", edge.pull))
                row.update(_covariance_row("combined_cov", edge.combined_covariance))
                field_edge_rows.append(row)
    _write_csv(output / "selected_route_field_edge_residuals.csv", field_edge_rows)
    summary = {
        "method": "frozen_physical_association_then_field_aware_route_consistency",
        "synthetic_manifest": str(manifest_path),
        "split": split,
        "full_split_physical_payload_samples": full_split_sample_count,
        "inference_physical_payload_samples": 1,
        "inference_payload_filter": [payload_id],
        "inference_smoke_limit": None,
        "inference_event_limit_per_payload": None,
        "synthetic_condition_axis": uniform_condition_axis([item.sample for item in sets[:1]]),
        "physical_geometry_repropagation": True,
        "q_over_p_mode": int(q_over_p_mode),
        "test_opened": False,
        "architecture_or_threshold_tuning": False,
        "candidate_chi2_gate": None,
        "operating_layer_control": dict(control_metadata),
        "frozen_backbone": {
            **dict(frozen["metadata"]),
            "calibration_fit_split": "train_only",
            "packing": "pre_registered_historical_packing",
        },
        "route_solver": {
            "method": "adjacent_contiguous_unit_capacity_set_packing",
            "thresholds": {
                f"{source}->{target}": float(value)
                for (source, target), value in config.score_threshold_by_pair.items()
            },
            "unmatched_penalty": float(config.unmatched_penalty),
            "selection_split": "pre_registered_historical_packing",
        },
        "field_aware_route_consistency": field_aware_route_summary(all_field_routes),
        "truth_labelled_mc_evaluation": evaluation,
        "raw_candidate_score_metrics": candidate_score_metrics(sets, raw_scores, calibration_bins),
        "frozen_calibrated_score_metrics": candidate_score_metrics(sets, calibrated, calibration_bins),
    }
    _write_json(output / "association_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", default="configs/physical_four_station_operating_layer_control.yaml")
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
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

    control = yaml.safe_load(Path(args.control_config).expanduser().resolve().read_text(encoding="utf-8"))
    if str(control.get("control_id")) != "retrained_v2_train_pair_platt_historical_packing_v1":
        raise ValueError("unexpected operating-layer control id")
    if control.get("unmatched_penalty_grid_search_allowed") is not False:
        raise ValueError("control config must forbid unmatched-penalty search")
    gates_contract = yaml.safe_load(
        Path(control["association_gates"]).expanduser().resolve().read_text(encoding="utf-8")
    )
    packing = control["packing"]
    frozen_root = Path(args.frozen_output).expanduser().resolve()
    frozen = _load_v2(frozen_root, args.device)
    route_config = historical_packing_config(
        unmatched_penalty=float(packing["unmatched_penalty"]),
        pair_thresholds=packing["pair_thresholds"],
        maximum_hypotheses=_maximum_hypotheses(frozen_root, "route_aware_transformer_v2"),
    )

    manifest_path, train_samples, train_manifest = _load_split(Path(args.synthetic_manifest), "train")
    if train_manifest.get("physical_geometry_repropagation") is not True or int(
        train_manifest.get("q_over_p_mode", -1)
    ) != int(args.q_over_p_mode):
        raise ValueError("train synthetic manifest lacks a physical mode-0 candidate contract")
    train_bundle, train_raw = _predict(
        frozen, train_samples, device=args.device, batch_size=args.batch_size, q_over_p_mode=args.q_over_p_mode
    )
    _, calibration_payload = calibrate_transformer_scores(
        train_bundle.adjacent_sets,
        train_raw,
        args.calibration_bins,
        scope="station_pair",
        method="platt",
    )
    calibration_payload = tag_train_only_calibration(calibration_payload)
    expected_pairs = set(control["calibration"]["pairs"])
    observed_pairs = set(calibration_payload["platt_by_station_pair"])
    if observed_pairs != expected_pairs:
        raise ValueError(f"train Platt pairs {sorted(observed_pairs)} differ from the pre-registered set")
    calibration_wrapper = {
        "calibration": calibration_payload,
        "fit_split": "train_only",
        "schema_version": "faser-four-station-operating-layer-train-platt-v1",
        "test_opened": False,
        "control_id": control["control_id"],
    }
    _write_json(output / "calibration.json", calibration_wrapper)
    _write_json(
        output / "operating_point.json",
        {
            "method": packing["method"],
            "selection_split": "pre_registered_historical_packing",
            "selection_source": packing["source"],
            "test_opened": False,
            "unmatched_penalty": float(packing["unmatched_penalty"]),
            "thresholds": dict(packing["pair_thresholds"]),
            "control_id": control["control_id"],
        },
    )

    _, val_samples, val_manifest = _load_split(Path(args.synthetic_manifest), "validation")
    if val_manifest.get("physical_geometry_repropagation") is not True or int(
        val_manifest.get("q_over_p_mode", -1)
    ) != int(args.q_over_p_mode):
        raise ValueError("validation synthetic manifest lacks a physical mode-0 candidate contract")
    val_bundle, val_raw = _predict(
        frozen, val_samples, device=args.device, batch_size=args.batch_size, q_over_p_mode=args.q_over_p_mode
    )
    val_calibrated = apply_frozen_transformer_calibration(
        val_bundle.adjacent_sets, val_raw, calibration_payload
    )
    control_metadata = {
        "control_id": control["control_id"],
        "calibration_fit_split": "train_only",
        "packing_source": packing["source"],
        "unmatched_penalty_scanned": False,
        "checkpoint_sha256": frozen["metadata"]["checkpoint_sha256"],
    }
    validation_root = output / "validation"
    diagnostics = []
    for payload_id in PAYLOADS:
        sets, raw = _filter(val_bundle.adjacent_sets, val_raw, payload_id)
        _, cal = _filter(val_bundle.adjacent_sets, val_calibrated, payload_id)
        payload_dir = validation_root / payload_id
        _emit_payload(
            output=payload_dir,
            payload_id=payload_id,
            sets=sets,
            raw_scores=raw,
            calibrated=cal,
            config=route_config,
            frozen=frozen,
            manifest_path=manifest_path,
            split="validation",
            full_split_sample_count=len(val_samples),
            calibration_bins=args.calibration_bins,
            q_over_p_mode=args.q_over_p_mode,
            control_metadata=control_metadata,
        )
        diagnostic_path = validation_root / f"{payload_id}_diagnostics.json"
        report = summarize(
            association_dir=payload_dir,
            synthetic_manifest=Path(args.synthetic_manifest),
            payload_id=payload_id,
            split="validation",
        )
        _write_json(diagnostic_path, report)
        diagnostics.append(diagnostic_path)

    reports = {
        payload: _read_json(validation_root / f"{payload}_diagnostics.json") for payload in PAYLOADS
    }
    association_gates = _gates(gates_contract["association_gates"])
    association = assess(reports, association_gates)
    recovery = _adjacent_recovery(
        reports, float(association_gates["vs_nominal_efficiency_drop_max"])
    )
    plan = _read_json(Path(args.iteration_manifest).expanduser().resolve())["common_scan_plan"]
    gauge = _gauge_audit(reports, _families_from_plan(plan), gates_contract["gauge_invariance_audit"])
    quality = nominal_quality_ok(
        reports[NOMINAL]["selected_route"],
        maximum_track_fake_rate=float(control["nominal_quality_band"]["maximum_track_fake_rate"]),
        minimum_complete_track_purity=float(control["nominal_quality_band"]["minimum_complete_track_purity"]),
    )
    ok = bool(
        association["raw_candidate_graph_complete"]
        and association["frozen_v2_association_stable_vs_nominal"]
        and recovery["ok"]
        and gauge["ok"]
        and quality["ok"]
    )
    if not association["raw_candidate_graph_complete"]:
        failure = "candidate_or_propagation"
    elif not quality["ok"]:
        failure = "nominal_fake_or_purity_outside_pre_registered_band"
    elif not association["frozen_v2_association_stable_vs_nominal"]:
        failure = "association_domain_shift"
    elif not recovery["ok"]:
        failure = "adjacent_2_3_or_s3_not_recovered"
    elif not gauge["ok"]:
        failure = "gauge_invariance_failure"
    else:
        failure = None
    decision = {
        "control_id": control["control_id"],
        "split": "validation",
        "test_data_accessed": False,
        "unmatched_penalty_scanned": False,
        "workbook_52_held_outs_opened": False,
        "architecture_or_solver_changed": False,
        "continue_to_15d_relative_wls": bool(ok),
        "failure_class": failure,
        "record_if_fail": None
        if ok
        else str(control["if_control_fails"]["record"]),
        "do_not_loosen_efficiency_drop_gate": True,
        "checkpoint_sha256": _sha256(frozen_root / "route_aware_transformer_v2.pt"),
        "nominal_quality": quality,
        "association_vs_nominal": {
            "raw_candidate_graph_complete": association["raw_candidate_graph_complete"],
            "association_stable_vs_nominal": association["frozen_v2_association_stable_vs_nominal"],
            "payloads": association["payloads"],
        },
        "adjacent_2_3_and_s3_recovered": recovery,
        "gauge_invariance": gauge,
        "payloads": {
            name: {
                "complete_track_efficiency": report["selected_route"]["complete_track_efficiency"],
                "complete_track_purity": report["selected_route"]["complete_track_purity"],
                "track_fake_rate": report["selected_route"]["track_fake_rate"],
                "score_retained_complete_truth_chains": report["selected_route"][
                    "score_retained_complete_truth_chains"
                ],
                "complete_truth_chains": report["selected_route"]["complete_truth_chains"],
            }
            for name, report in reports.items()
        },
    }
    _write_json(output / "control_gate_decision.json", decision)
    print(json.dumps(_json_ready({
        "output_dir": str(output),
        "continue_to_15d_relative_wls": ok,
        "failure_class": failure,
        "nominal_quality_ok": quality["ok"],
        "association_vs_nominal_ok": association["frozen_v2_association_stable_vs_nominal"],
        "gauge_invariance_ok": gauge["ok"],
        "adjacent_2_3_ok": recovery["ok"],
    }), indent=2))


if __name__ == "__main__":
    main()
