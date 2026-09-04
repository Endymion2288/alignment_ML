#!/usr/bin/env python3
"""Workbook 70: frozen Arm 0/1/2 reserved-blind development evaluation.

Arm 0 is Workbook-64 edge-only packing.  Arm 1/2 inject additive
``L_corrected`` complete-route scores into the same unit-capacity solver.
Does not train, does not retune thresholds/Platt/penalty, and never opens
``00800_00849`` or sealed test.  ``continue_to_15d_relative_wls`` stays false.
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
from evaluation.route_metrics import RouteMetrics, _unique_truth_by_station, assess_adjacent_route_assignment
from scripts.audit_four_station_blind_failure_localization import (
    LOSS_STAGE_TO_PROBLEM,
    PAYLOADS,
    _failure_problems,
)
from scripts.evaluate_four_station_gauge_consistent_transfer import (
    NOMINAL,
    _absolute_nominal,
    _pair_and_s3,
)
from scripts.evaluate_four_station_matched_association import (
    FORBIDDEN_PAYLOADS,
    _adjacent_recovery,
    _families_from_plan,
    _gauge_audit,
    _gates,
)
from scripts.evaluate_four_station_source_diversity_blind import classify_blind_decision
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.summarize_four_station_frozen_association import _raw_candidate_audit, assess
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
)
from training.route_assignment import (
    assign_adjacent_route_sets,
    evaluate_adjacent_route_assignment_sets,
    prepare_route_assignment_context,
)
from training.route_aware_transformer import (
    load_relative_route_v4_head_only_artifact,
    predict_route_aware_scores,
    route_query_score_maps_by_event,
)
from training.route_operating_audit import STATION_PATH, audit_event_truth_chains, pair_tables_from_sets
from training.route_reduction_audit import attach_reduction_audit
from training.solver_hard_negative_audit import production_hypotheses
from training.source_diversity_audit import (
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
    assert_sources_allowed,
)


WORKBOOK64_SHA256 = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
ARM1_SHA256 = "e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8"
ARM2_SHA256 = "a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a"
UNUSED_FINAL_BLIND = tuple(UNUSED_RESERVE_SOURCES)
WB65_C_TOTAL = 200
WB65_D_TOTAL = 170


def refuse_forbidden_paths(paths: Sequence[Path]) -> None:
    needles = list(UNUSED_FINAL_BLIND) + ["00800_00849"]
    for path in paths:
        text = str(path.resolve())
        for needle in needles:
            if needle in text:
                raise SystemExit(f"refusing unused final-blind path: {text}")


def frozen_packing_config() -> RouteAssignmentConfig:
    pairs = adjacent_station_pairs((0, 1, 2, 3))
    return RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in pairs},
        unmatched_penalty=-1.0,
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=100_000,
        complete_route_score_composition="replace",
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


def _filter_route_maps(
    maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]] | None,
    payload_id: str,
):
    if maps is None:
        return None
    return {key: value for key, value in maps.items() if str(key[1]) == payload_id}


def selected_physical_edges(assigned: Sequence[object]) -> dict[str, object]:
    pair_counts: Counter[str] = Counter()
    station_counts: Counter[int] = Counter()
    physical_keys = []
    for item in assigned:
        event = item.event
        for route in item.result.routes:
            for (source_station, target_station), match in route.matches:
                pair_counts[f"{int(source_station)}_{int(target_station)}"] += 1
                station_counts[int(source_station)] += 1
                station_counts[int(target_station)] += 1
                physical_keys.append(
                    (
                        int(event.run_id),
                        int(event.event_id),
                        int(match.source_index),
                        int(match.target_index),
                        int(source_station),
                        int(target_station),
                    )
                )
    return {
        "selected_field_edges": len(physical_keys),
        "unique_physical_edges": len(set(physical_keys)),
        "station_pair_composition": dict(sorted(pair_counts.items())),
        "station_participation_edge_endpoints": {
            str(station): int(station_counts[station]) for station in sorted(station_counts)
        },
        "includes_adjacent_0_1": pair_counts.get("0_1", 0) > 0,
        "includes_adjacent_1_2": pair_counts.get("1_2", 0) > 0,
        "includes_adjacent_2_3": pair_counts.get("2_3", 0) > 0,
    }


def cd_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    problems = Counter(problem for row in rows for problem in _failure_problems(row))
    return {
        "A": int(problems.get("A", 0)),
        "B": int(problems.get("B", 0)),
        "C": int(problems.get("C", 0)),
        "D": int(sum(1 for row in rows if str(row.get("loss_stage")) == "packing_competition")),
        "unselected": int(sum(1 for row in rows if not row.get("selected"))),
        "selected": int(sum(1 for row in rows if row.get("selected"))),
        "complete_truth_chains": int(len(rows)),
    }


def compare_efficiency(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, object]:
    result = {}
    for payload_id in PAYLOADS:
        left_eff = ((left.get("payloads") or {}).get(payload_id) or {}).get("complete_track_efficiency")
        right_eff = ((right.get("payloads") or {}).get(payload_id) or {}).get("complete_track_efficiency")
        result[payload_id] = {
            "left": None if left_eff is None else float(left_eff),
            "right": None if right_eff is None else float(right_eff),
            "right_minus_left": None
            if left_eff is None or right_eff is None
            else float(right_eff) - float(left_eff),
        }
    return result


def compare_cd(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, int]:
    """Later-minus-earlier C/D counts.  Job 1108310 JSON used the opposite subtraction."""
    return {
        "C": int(right["C"]) - int(left["C"]),
        "D": int(right["D"]) - int(left["D"]),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_identity_platt(calibration: Mapping[str, Any]) -> None:
    if calibration.get("fit_split") != "identity_frozen_pre_training" or calibration.get("identity_map") is not True:
        raise SystemExit("evaluation requires the frozen identity Platt")


def _load_arm(
    arm: str,
    *,
    w64_root: Path,
    arm1_checkpoint: Path,
    arm2_checkpoint: Path,
    device: str,
) -> dict[str, object]:
    if arm == "arm0":
        frozen = _load_v2(w64_root, device)
        observed = str(frozen["metadata"]["checkpoint_sha256"])
        if observed != WORKBOOK64_SHA256:
            raise SystemExit(f"Arm 0 checkpoint sha256 {observed} != {WORKBOOK64_SHA256}")
        _assert_identity_platt(frozen["calibration"])
        return {
            "arm": "arm0",
            "name": "workbook64_edge_only",
            "model": frozen["model"],
            "artifact": frozen["artifact"],
            "calibration": frozen["calibration"],
            "inject_complete_route_scores": False,
            "checkpoint": frozen["metadata"]["checkpoint"],
            "checkpoint_sha256": observed,
        }
    if arm == "arm1":
        path = arm1_checkpoint
        expected = ARM1_SHA256
        name = "absolute_route_control"
    elif arm == "arm2":
        path = arm2_checkpoint
        expected = ARM2_SHA256
        name = "relative_route_v4_primary"
    else:
        raise SystemExit(f"unknown arm {arm}")
    observed = _sha256(path)
    if observed != expected:
        raise SystemExit(f"{arm} checkpoint sha256 {observed} != {expected}")
    wrapper, artifact = load_relative_route_v4_head_only_artifact(path, device=device)
    calibration_path = path.parent / "calibration.json"
    wrapper_cal = _read_json(calibration_path)
    calibration = dict(wrapper_cal.get("calibration") or wrapper_cal)
    _assert_identity_platt(calibration)
    return {
        "arm": arm,
        "name": name,
        "model": wrapper,
        "artifact": artifact,
        "calibration": calibration,
        "inject_complete_route_scores": True,
        "checkpoint": str(path),
        "checkpoint_sha256": observed,
    }


def _score_arm(loaded: Mapping[str, Any], bundle, device: str, batch_size: int):
    prediction = predict_route_aware_scores(
        loaded["model"],
        bundle,
        loaded["artifact"].node_standardizer,
        loaded["artifact"].edge_standardizer,
        device=device,
        batch_size=int(batch_size),
    )
    raw = [np.asarray(values, dtype=np.float64) for values in prediction.edge_scores]
    calibrated = apply_frozen_transformer_calibration(bundle.adjacent_sets, raw, loaded["calibration"])
    route_maps = None
    if loaded["inject_complete_route_scores"]:
        route_maps = route_query_score_maps_by_event(
            prediction.route_score_sets,
            tuple(np.asarray(score_set.scores, dtype=np.float64) for score_set in prediction.route_score_sets),
        )
    return raw, calibrated, route_maps


def _evaluate_arm(
    loaded: Mapping[str, Any],
    bundle,
    raw_scores: Sequence[np.ndarray],
    calibrated_scores: Sequence[np.ndarray],
    route_maps,
    config: RouteAssignmentConfig,
    raw_candidate_by_payload: Mapping[str, Mapping[str, Any]],
    payload_samples: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    payload_reports: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    payload_metrics: dict[str, RouteMetrics] = {}
    for payload_id in PAYLOADS:
        sets, values = _filter_sets(bundle.adjacent_sets, calibrated_scores, payload_id)
        maps = _filter_route_maps(route_maps, payload_id)
        evaluation = evaluate_adjacent_route_assignment_sets(
            sets,
            values,
            config,
            calibration_bins=15,
            complete_route_scores_by_event=maps,
        )
        assigned = assign_adjacent_route_sets(
            sets,
            values,
            config,
            15,
            complete_route_scores_by_event=maps,
        )
        tables = pair_tables_from_sets(sets, values, values)
        assigned_by_key = {item.key: item for item in assigned}
        context = prepare_route_assignment_context(sets, values, 15, config.station_path)
        rows: list[dict[str, object]] = []
        event_truth_counts: dict[tuple[object, ...], int] = Counter()
        pending = []
        metrics = RouteMetrics()
        for group in context.groups:
            event = group.event
            item = assigned_by_key[group.key]
            pair_tables = tables[group.key]
            _, unique_by_index = _unique_truth_by_station(event, STATION_PATH)
            query_map = None if maps is None else maps[group.key]
            hypotheses = production_hypotheses(
                event,
                group.station_matrices,
                config,
                complete_route_scores=query_map,
            )
            truth_rows = audit_event_truth_chains(
                event,
                group.station_matrices,
                pair_tables,
                config,
                item.result,
                complete_route_scores=query_map,
            )
            event_key = group.key
            event_truth_counts[event_key] += len(truth_rows)
            metrics.add(
                assess_adjacent_route_assignment(
                    event,
                    item.result,
                    group.station_matrices,
                    config.station_path,
                    config.score_threshold_by_pair,
                )
            )
            for row in truth_rows:
                row["sample_id"] = group.key[0]
                row["payload_id"] = group.key[1]
                row["run_id"] = group.key[2]
                row["event_id"] = group.key[3]
                pending.append((event_key, row, event, unique_by_index, pair_tables, hypotheses, item.result.routes))
        for event_key, row, event, unique_by_index, pair_tables, hypotheses, selected_routes in pending:
            attached = attach_reduction_audit(
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
        all_rows.extend(rows)
        payload_metrics[payload_id] = metrics
        route = evaluation["route"]
        payload_reports[payload_id] = {
            "payload_id": payload_id,
            "source_ids": sorted({str(sample.source_id) for sample in payload_samples[payload_id]}),
            "n_events": int((evaluation.get("assignment") or {}).get("events") or 0),
            "raw_candidate_graph": raw_candidate_by_payload[payload_id],
            "selected_route": {
                "complete_track_efficiency": route.get("complete_track_efficiency"),
                "complete_track_purity": route.get("complete_track_purity"),
                "track_purity": route.get("track_purity"),
                "track_fake_rate": route.get("track_fake_rate"),
                "selected_routes": route.get("selected_routes"),
                "correct_complete_routes": route.get("correct_complete_routes"),
                "selected_complete_routes": route.get("selected_complete_routes"),
                "duplicate_routes": route.get("duplicate_routes"),
                "missing_station_boundaries": route.get("missing_station_boundaries"),
                "complete_truth_chains": route.get("complete_truth_chains"),
                "candidate_retained_complete_truth_chains": route.get("candidate_retained_complete_truth_chains"),
                "score_retained_complete_truth_chains": route.get("score_retained_complete_truth_chains"),
            },
            "by_adjacent_station_pair": evaluation.get("by_station_pair"),
            "selected_physical_edges": selected_physical_edges(assigned),
            "cd": cd_counts(rows),
            "complete_track_efficiency": route.get("complete_track_efficiency"),
            "test_data_accessed": False,
            "architecture_or_threshold_tuning": False,
        }
    pooled_cd = cd_counts(all_rows)
    by_payload_cd = {
        payload_id: cd_counts([row for row in all_rows if str(row["payload_id"]) == payload_id])
        for payload_id in PAYLOADS
    }
    return {
        "arm": loaded["arm"],
        "name": loaded["name"],
        "checkpoint": loaded["checkpoint"],
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "inject_complete_route_scores": loaded["inject_complete_route_scores"],
        "payloads": payload_reports,
        "cd_pooled": pooled_cd,
        "cd_by_payload": by_payload_cd,
        "payload_metrics": {
            payload_id: {
                "complete_truth_chains": int(metrics.complete_truth_chains),
                "correct_complete_routes": int(metrics.correct_complete_routes),
                "selected_complete_routes": int(metrics.selected_complete_routes),
                "selected_routes": int(metrics.selected_routes),
            }
            for payload_id, metrics in payload_metrics.items()
        },
        "loss_stage_map": dict(LOSS_STAGE_TO_PROBLEM),
        "n_truth_rows": int(len(all_rows)),
    }


def _arm_gate(
    arm_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    families: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    reports = {payload_id: arm_result["payloads"][payload_id] for payload_id in PAYLOADS}
    association_gates = _gates(contract["gates"])
    association_gates["raw_complete_truth_chain_recall_min"] = float(
        contract["gates"]["raw_complete_truth_chain_recall_min"]
    )
    candidate = assess(reports, association_gates)
    recovery = _adjacent_recovery(
        reports, float(association_gates["vs_nominal_efficiency_drop_max"])
    )
    gauge = _gauge_audit(reports, families, contract["gates"]["gauge_invariance_audit"])
    nominal = _absolute_nominal(
        reports[NOMINAL],
        float(contract["gates"]["nominal_complete_track_purity_min"]),
        float(contract["gates"]["nominal_track_fake_rate_max"]),
    )
    pairs = _pair_and_s3(reports)
    layer1 = bool(candidate["raw_candidate_graph_complete"])
    layer2 = bool(candidate["frozen_v2_association_stable_vs_nominal"] and nominal["ok"] and recovery["ok"])
    layer3 = bool(gauge["ok"])
    passed = layer1 and layer2 and layer3
    plus_common_fail = any(
        (candidate.get("payloads") or {}).get(name, {}).get("association_vs_nominal_ok") is False
        for name in reports
        if str(name).endswith("_plus_common")
    )
    decision = classify_blind_decision(
        passed=passed,
        layer1_ok=layer1,
        same_common_se3_truth_utility_drop=bool(layer1 and (not layer2 or not layer3) and plus_common_fail),
    )
    decision["continue_to_15d_relative_wls"] = False
    return {
        "layer1_raw_candidate": layer1,
        "layer2_association_vs_own_nominal": layer2,
        "layer3_gauge": layer3,
        "passed": passed,
        "candidate": candidate,
        "nominal": nominal,
        "adjacent_23_and_s3": recovery,
        "pair_and_s3": pairs,
        "gauge": gauge,
        **decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/relative_route_v4_head_only_development_eval.yaml",
    )
    parser.add_argument("--gates", default="configs/physical_four_station_diversity_training.yaml")
    parser.add_argument(
        "--synthetic-manifest",
        default="outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
    )
    parser.add_argument(
        "--iteration-manifest",
        default="outputs/mc24_four_station_source_diversity_blind_v1/iteration_manifest.json",
    )
    parser.add_argument(
        "--workbook64-frozen-output",
        default="outputs/mc24_four_station_source_diversity_v1/checkpoint",
    )
    parser.add_argument(
        "--arm1-checkpoint",
        default="outputs/mc24_four_station_relative_route_v4_head_only/absolute_control/checkpoint_last.pt",
    )
    parser.add_argument(
        "--arm2-checkpoint",
        default="outputs/mc24_four_station_relative_route_v4_head_only/relative_primary/checkpoint_last.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_four_station_relative_route_v4_head_only/development_eval",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("Workbook 70 development evaluation is GPU-only")
    if int(args.batch_size) < 1:
        raise SystemExit("batch size must be positive")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    contract_path = Path(args.contract).expanduser().resolve()
    gates_path = Path(args.gates).expanduser().resolve()
    manifest_path = Path(args.synthetic_manifest).expanduser().resolve()
    iteration_path = Path(args.iteration_manifest).expanduser().resolve()
    w64_root = Path(args.workbook64_frozen_output).expanduser().resolve()
    arm1_checkpoint = Path(args.arm1_checkpoint).expanduser().resolve()
    arm2_checkpoint = Path(args.arm2_checkpoint).expanduser().resolve()
    refuse_forbidden_paths(
        [contract_path, gates_path, manifest_path, iteration_path, w64_root, arm1_checkpoint, arm2_checkpoint, output]
    )
    eval_contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    root = eval_contract.get("relative_route_v4_head_only_development_eval") or eval_contract
    if int(root.get("workbook", -1)) != 70:
        raise SystemExit("eval contract is not Workbook 70")
    if root.get("continue_to_15d_relative_wls") is not False:
        raise SystemExit("Workbook 70 must keep continue_to_15d_relative_wls false")
    gates_contract = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    if gates_contract.get("control_id") != "retrained_v2_source_disjoint_diversity_v1":
        raise SystemExit("gates file is not the workbook-64 source-diversity contract")
    if gates_contract.get("test_data_accessed") is not False:
        raise SystemExit("contract opened sealed test")

    _, samples, manifest = load_synthetic_curriculum_manifest(
        manifest_path,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if {sample.split for sample in samples} != {"validation"}:
        raise SystemExit("development evaluation must load the reserved-blind validation split")
    if any(str(sample.payload_id) in FORBIDDEN_PAYLOADS for sample in samples):
        raise SystemExit("refusing workbook-52 held-out payload")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise SystemExit("blind overlay is missing payloads: " + ", ".join(sorted(missing)))
    constituents = {source for sample in samples for source in sample.source_ids}
    assert_sources_allowed(constituents, allow_reserved_blind=True)
    if constituents != set(RESERVED_BLIND_SOURCES):
        raise SystemExit("development evaluation must use only the reserved blind pair")
    if constituents & set(UNUSED_FINAL_BLIND) or any("00800_00849" in str(source) for source in constituents):
        raise SystemExit("unused final-blind sources were loaded")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise SystemExit("manifest lacks physical geometry repropagation")
    iteration = _read_json(iteration_path)
    sources = {str(item.get("source_id")) for item in iteration.get("sources") or []}
    if sources != set(RESERVED_BLIND_SOURCES):
        raise SystemExit("iteration manifest is not the reserved pair")
    if iteration.get("reserved_blind_validation_only") is not True:
        raise SystemExit("iteration manifest is not reserved-blind-only")
    families = _families_from_plan(iteration["common_scan_plan"])

    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(args.q_over_p_mode),
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    payload_samples = {
        payload_id: [sample for sample in samples if str(sample.payload_id) == payload_id]
        for payload_id in PAYLOADS
    }
    raw_candidate_by_payload = {
        payload_id: _raw_candidate_audit(payload_samples[payload_id]) for payload_id in PAYLOADS
    }

    config = frozen_packing_config()
    arm_results: dict[str, object] = {}
    for arm in ("arm0", "arm1", "arm2"):
        loaded = _load_arm(
            arm,
            w64_root=w64_root,
            arm1_checkpoint=arm1_checkpoint,
            arm2_checkpoint=arm2_checkpoint,
            device=args.device,
        )
        raw, calibrated, route_maps = _score_arm(loaded, bundle, args.device, args.batch_size)
        result = _evaluate_arm(
            loaded,
            bundle,
            raw,
            calibrated,
            route_maps,
            config,
            raw_candidate_by_payload,
            payload_samples,
        )
        result["gate"] = _arm_gate(result, gates_contract, families)
        arm_results[arm] = result
        _write_json(output / f"{arm}_evaluation.json", {k: v for k, v in result.items() if k != "payload_metrics"})

    arm0_cd = arm_results["arm0"]["cd_pooled"]
    arm0_replay = {
        "C": int(arm0_cd["C"]),
        "D": int(arm0_cd["D"]),
        "expected_C": WB65_C_TOTAL,
        "expected_D": WB65_D_TOTAL,
        "ok": int(arm0_cd["C"]) == WB65_C_TOTAL and int(arm0_cd["D"]) == WB65_D_TOTAL,
    }
    arm1_minus_arm0_cd = compare_cd(arm_results["arm0"]["cd_pooled"], arm_results["arm1"]["cd_pooled"])
    arm2_minus_arm1_cd = compare_cd(arm_results["arm1"]["cd_pooled"], arm_results["arm2"]["cd_pooled"])
    paired = {
        "arm1_minus_arm0_efficiency": compare_efficiency(arm_results["arm0"], arm_results["arm1"]),
        "arm2_minus_arm1_efficiency": compare_efficiency(arm_results["arm1"], arm_results["arm2"]),
        "cd": {
            "arm0": arm_results["arm0"]["cd_pooled"],
            "arm1": arm_results["arm1"]["cd_pooled"],
            "arm2": arm_results["arm2"]["cd_pooled"],
            # Keys are later-minus-earlier, matching arm*_minus_*_efficiency.
            # Workbook 70 job 1108310 JSON used the opposite subtraction and is frozen.
            "arm1_minus_arm0_C": arm1_minus_arm0_cd["C"],
            "arm1_minus_arm0_D": arm1_minus_arm0_cd["D"],
            "arm2_minus_arm1_C": arm2_minus_arm1_cd["C"],
            "arm2_minus_arm1_D": arm2_minus_arm1_cd["D"],
        },
        "workbook64_gates_passed": {
            "arm0": bool(arm_results["arm0"]["gate"]["passed"]),
            "arm1": bool(arm_results["arm1"]["gate"]["passed"]),
            "arm2": bool(arm_results["arm2"]["gate"]["passed"]),
        },
    }
    decision = {
        "workbook": 70,
        "development_accessed": True,
        "reserved_blind_role": "development_evaluation_only",
        "test_data_accessed": False,
        "new_final_blind_content_accessed": False,
        "continue_to_15d_relative_wls": False,
        "do_not_retune_after_seeing_development": True,
        "arm0_workbook65_cd_replay": arm0_replay,
        "paired": paired,
        "checkpoint_sha256": {
            "arm0": WORKBOOK64_SHA256,
            "arm1": ARM1_SHA256,
            "arm2": ARM2_SHA256,
        },
        "complete_route_query_injected": {
            "arm0": False,
            "arm1": True,
            "arm2": True,
        },
    }
    _write_json(output / "paired_comparison.json", paired)
    _write_json(output / "decision.json", decision)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "arm0_replay_ok": arm0_replay["ok"],
                "gates": paired["workbook64_gates_passed"],
                "cd": paired["cd"],
                "continue_to_15d_relative_wls": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
