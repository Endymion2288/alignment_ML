#!/usr/bin/env python3
"""Workbook 65: read-only event-level localization of the workbook-64 blind failure.

Replays the frozen workbook-64 V2 checkpoint on the already-opened reserved-blind
overlay, decomposes the complete-track efficiency loss per event into the
pre-registered loss chain (candidate graph -> edge ranking -> calibrated
threshold -> route utility -> production set packing), and emits an
event-level failure table plus a nominal<->hard matched alignment.

Does not train, does not retune the operating point, does not open the
15D WLS, and never touches sealed test sources.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from baselines.route_assignment import RouteAssignmentConfig
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.route_metrics import RouteMetrics, _unique_truth_by_station, assess_adjacent_route_assignment
from scripts.audit_four_station_solver_hard_negatives import (
    _filter_sets,
    _frozen_route_config,
    _score_model,
    _write_json,
    _write_jsonl,
)
from scripts.run_frozen_association_backbone import _load_v2, _sha256
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_assignment import assign_adjacent_route_sets, prepare_route_assignment_context
from training.route_operating_audit import (
    STATION_PATH,
    audit_event_truth_chains,
    pair_tables_from_sets,
)
from training.route_reduction_audit import attach_reduction_audit
from training.route_utility_identifiability import _median
from training.solver_hard_negative_audit import production_hypotheses, twin_alignment_key
from training.source_diversity_audit import (
    RESERVED_BLIND_SOURCES,
    attach_route_phase_space,
    assert_sources_allowed,
    charge_from_pdg,
    station_occupancy,
)

CHECKPOINT_SHA256 = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_hard_s3_ry",
    "iteration_00_hard_s3_ry_plus_common",
    "iteration_00_draw_00",
    "iteration_00_draw_00_plus_common",
    "iteration_00_draw_01",
    "iteration_00_draw_01_plus_common",
)
CHART_BY_FAMILY = {
    "hard_s3_ry": ("iteration_00_hard_s3_ry", "iteration_00_hard_s3_ry_plus_common"),
    "draw_00": ("iteration_00_draw_00", "iteration_00_draw_00_plus_common"),
    "draw_01": ("iteration_00_draw_01", "iteration_00_draw_01_plus_common"),
}
# Frozen in workbook 64: no tolerance drift, counts must match exactly.
REPLAY_COUNT_TOLERANCE = 0
REPLAY_RATIO_TOLERANCE = 1.0e-9

LOSS_STAGE_TO_PROBLEM = {
    "candidate_missing": "A",
    "below_station_pair_threshold": "B_or_C",
    "utility_nonpositive": "C",
    "packing_competition": "D",
    "selected": None,
}


def _namespace_map(samples) -> dict[int, str]:
    """Map namespaced run ids to original source ids via the pooled descriptor."""
    mapping: dict[int, str] = {}
    seen = set()
    for sample in samples:
        descriptor = Path(str(sample.physical_payload_manifest))
        if descriptor.name != "pooled_physical_descriptor.json":
            descriptor = descriptor.with_name("pooled_physical_descriptor.json")
        if str(descriptor) in seen or not descriptor.is_file():
            continue
        seen.add(str(descriptor))
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        for block in payload.get("origin_namespaces") or []:
            source_id = str(block.get("source_id") or "")
            for item in block.get("run_id_mapping") or []:
                mapping[int(item["namespaced_run_id"])] = source_id
    return mapping


def _source_uid(row: Mapping[str, Any], namespace: Mapping[int, str]) -> str | None:
    """``source_id:original_run:original_event`` of the first endpoint's origin."""
    signature = row.get("origin_signature")
    if not signature:
        return None
    origin_run = int(signature[0][0])
    origin_event = int(signature[0][1])
    source_id = namespace.get(origin_run)
    if source_id is None:
        return None
    return f"{source_id}:{origin_run}:{origin_event}"


def _replay_payloads(
    frozen: Mapping[str, Any],
    manifest_path: str,
    *,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, object]], dict[str, RouteMetrics], dict[str, dict[str, object]]]:
    _, samples, manifest = load_synthetic_curriculum_manifest(
        manifest_path,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if {sample.split for sample in samples} != {"validation"}:
        raise SystemExit("blind localization must load the reserved-blind validation split")
    if any(str(sample.payload_id) not in PAYLOADS for sample in samples):
        raise SystemExit("blind overlay contains an unexpected payload")
    missing = set(PAYLOADS) - {str(sample.payload_id) for sample in samples}
    if missing:
        raise SystemExit("blind overlay is missing payloads: " + ", ".join(sorted(missing)))
    constituents = {source for sample in samples for source in sample.source_ids}
    assert_sources_allowed(constituents, allow_reserved_blind=True)
    if constituents != set(RESERVED_BLIND_SOURCES):
        raise SystemExit("blind localization must use only the reserved blind pair")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise SystemExit("manifest lacks physical geometry repropagation")

    namespace = _namespace_map(samples)
    if not namespace:
        raise SystemExit("could not recover origin namespaces from the pooled descriptors")

    observed = str(frozen["metadata"]["checkpoint_sha256"])
    if observed != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {observed} != frozen workbook-64 {CHECKPOINT_SHA256}")
    calibration = frozen["calibration"]
    if calibration.get("fit_split") != "identity_frozen_pre_training" or calibration.get("identity_map") is not True:
        raise SystemExit("candidate checkpoint is not the frozen identity Platt")

    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    raw_scores = _score_model(frozen, bundle, device, batch_size)
    # Workbook 64 freezes the identity Platt, so raw == calibrated by contract;
    # the pair-table code still takes both to keep the audit convention.
    calibrated_scores = list(raw_scores)

    config: RouteAssignmentConfig = _frozen_route_config()
    rows: list[dict[str, object]] = []
    payload_metrics: dict[str, RouteMetrics] = {}
    per_payload_raw: dict[str, dict[str, object]] = {}
    for payload_id in PAYLOADS:
        sets, values = _filter_sets(bundle.adjacent_sets, calibrated_scores, payload_id)
        context = prepare_route_assignment_context(sets, values, 15, config.station_path)
        assigned = assign_adjacent_route_sets(sets, values, config, 15, context=context)
        tables = pair_tables_from_sets(sets, values, values)
        assigned_by_key = {item.key: item for item in assigned}
        pending = []
        event_truth_counts: dict[tuple[object, ...], int] = Counter()
        metrics = RouteMetrics()
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
                pending.append((event_key, row, event, pair_tables, unique_by_index, hypotheses))
        for event_key, row, event, pair_tables, unique_by_index, hypotheses in pending:
            attached = attach_reduction_audit(
                row,
                event=event,
                unique_by_index=unique_by_index,
                pair_tables=pair_tables,
                hypotheses=hypotheses,
                selected_routes=assigned_by_key[event_key].result.routes,
                n_complete_truth_in_event=int(event_truth_counts[event_key]),
                margin=1.0,
            )
            saved_edges = attached.get("edges")
            saved_comp = attached.get("solver_competitor")
            attached = attach_route_phase_space(
                attached,
                event,
                source_by_namespaced_run=namespace,
            )
            attached["edges"] = saved_edges
            attached["solver_competitor"] = saved_comp
            attached["source_event_uid"] = _source_uid(attached, namespace)
            attached["charge"] = attached.get("charge") or charge_from_pdg(
                attached.get("truth_pdg")
            )
            attached["station_occupancy"] = attached.get("station_occupancy") or station_occupancy(event)
            rows.append(attached)
        payload_metrics[payload_id] = metrics
        per_payload_raw[payload_id] = {
            "complete_truth_chains": metrics.complete_truth_chains,
            "selected_complete_routes": metrics.selected_complete_routes,
            "correct_complete_routes": metrics.correct_complete_routes,
            "selected_routes": metrics.selected_routes,
        }
    return rows, payload_metrics, per_payload_raw


def _failure_problems(row: Mapping[str, Any]) -> list[str]:
    """Map a truth-route loss stage onto the pre-registered problem letters."""
    stage = str(row.get("loss_stage"))
    problem = LOSS_STAGE_TO_PROBLEM.get(stage)
    if problem is None:
        return []
    if problem == "B_or_C":
        # Threshold failure with rank-1 truth edges is scale (C); any
        # non-rank-1 truth edge is ranking (B).
        edges = row.get("edges") or []
        ranking_bad = any(
            int(edge.get("truth_rank_among_source_calibrated", 1) or 1) > 1
            or int(edge.get("truth_rank_among_pair_calibrated", 1) or 1) > 1
            for edge in edges
            if edge.get("in_candidate_graph")
        )
        return ["B", "C"] if ranking_bad else ["C"]
    return [problem]


def _failure_row_summary(row: Mapping[str, Any]) -> dict[str, object]:
    competitor = row.get("solver_competitor") or {}
    return {
        "payload_id": str(row["payload_id"]),
        "sample_id": str(row["sample_id"]),
        "source_event_uid": row.get("source_event_uid"),
        "run_id": int(row["run_id"]),
        "event_id": int(row["event_id"]),
        "truth_id": row.get("truth_id"),
        "origin_signature": row.get("origin_signature"),
        "selected": bool(row["selected"]),
        "loss_stage": str(row["loss_stage"]),
        "failed_pairs": list(row.get("failed_pairs") or []),
        "problems": _failure_problems(row),
        "edges": [
            {
                "pair": edge.get("pair"),
                "in_candidate_graph": bool(edge.get("in_candidate_graph")),
                "above_threshold": bool(edge.get("above_threshold")),
                "raw_probability": edge.get("raw_probability"),
                "calibrated_probability": edge.get("calibrated_probability"),
                "calibrated_logit": edge.get("calibrated_logit"),
                "truth_rank_among_source_calibrated": edge.get("truth_rank_among_source_calibrated"),
                "truth_rank_among_pair_calibrated": edge.get("truth_rank_among_pair_calibrated"),
                "chi2": edge.get("chi2"),
            }
            for edge in (row.get("edges") or [])
        ],
        "u_truth": row.get("u_truth"),
        "u_best_solver_fragment": row.get("u_best_solver_fragment"),
        "production_margin": row.get("production_margin"),
        "production_winner_fragment": bool(row.get("production_fragment_winner")),
        "competitor": {
            "topology": competitor.get("topology"),
            "composition": competitor.get("composition"),
            "n_stations": competitor.get("n_stations"),
            "involves_2_3": competitor.get("involves_2_3"),
            "involves_s3": competitor.get("involves_s3"),
            "shared_stations": competitor.get("shared_stations"),
            "physical_utility": competitor.get("physical_utility"),
            "was_selected_by_packing": competitor.get("was_selected_by_packing"),
        },
        "n_selected_overlapping_blockers": row.get("n_selected_overlapping_blockers"),
        "s3_state": row.get("s3_state"),
        "track_state_by_station": row.get("track_state_by_station"),
        "station_occupancy": row.get("station_occupancy"),
        "charge": row.get("charge"),
        "truth_pdg": row.get("truth_pdg"),
        "n_complete_truth_in_event": row.get("n_complete_truth_in_event"),
    }


def _align_payload_rows(rows: Sequence[Mapping[str, Any]], payload: str) -> dict[str, Mapping[str, Any]]:
    """Align one payload's truth routes with the workbook-60 twin key.

    ``origin_signature`` alone only identifies the original (run, event):
    overlay ``origin_tracklet_id`` is the station index, so multi-track
    events need ``truth_id`` to split shared-origin routes.
    """
    selected = [row for row in rows if str(row["payload_id"]) == payload]
    aligned: dict[str, Mapping[str, Any]] = {}
    for row in selected:
        key = json.dumps(
            [[list(item) for item in key_part] for key_part in twin_alignment_key(row)[:1]]
            + [int(twin_alignment_key(row)[1])],
            sort_keys=True,
        )
        if key in aligned:
            raise SystemExit(f"{payload} duplicate origin signature + truth_id")
        aligned[key] = row
    return aligned


def _alignment_key(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [
            [list(item) for item in twin_alignment_key(row)[0]],
            int(twin_alignment_key(row)[1]),
        ],
        sort_keys=True,
    )


def _alignment_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    reference = _align_payload_rows(rows, "iteration_00_reference")
    families: dict[str, object] = {}
    for family, (chart, twin) in CHART_BY_FAMILY.items():
        chart_rows = _align_payload_rows(rows, chart)
        twin_rows = _align_payload_rows(rows, twin)
        shared_keys = sorted(set(chart_rows) & set(twin_rows) & set(reference))
        records = []
        for key in shared_keys:
            ref = reference[key]
            left = chart_rows[key]
            right = twin_rows[key]
            records.append(
                {
                    "alignment_key": json.loads(key),
                    "origin_signature": left.get("origin_signature"),
                    "source_event_uid": left.get("source_event_uid"),
                    "truth_id": left.get("truth_id"),
                    "reference_selected": bool(ref["selected"]),
                    "chart_selected": bool(left["selected"]),
                    "twin_selected": bool(right["selected"]),
                    "reference_loss_stage": str(ref["loss_stage"]),
                    "chart_loss_stage": str(left["loss_stage"]),
                    "twin_loss_stage": str(right["loss_stage"]),
                    "chart_u_truth": left.get("u_truth"),
                    "twin_u_truth": right.get("u_truth"),
                    "u_truth_gain_twin_minus_chart": None
                    if left.get("u_truth") is None or right.get("u_truth") is None
                    else float(right["u_truth"]) - float(left["u_truth"]),
                    "chart_production_margin": left.get("production_margin"),
                    "twin_production_margin": right.get("production_margin"),
                    "chart_failed_pairs": list(left.get("failed_pairs") or []),
                    "twin_failed_pairs": list(right.get("failed_pairs") or []),
                    "chart_competitor_topology": (left.get("solver_competitor") or {}).get("topology"),
                    "twin_competitor_topology": (right.get("solver_competitor") or {}).get("topology"),
                    "s3_state_chart": left.get("s3_state"),
                    "s3_state_twin": right.get("s3_state"),
                    "track_state_by_station": left.get("track_state_by_station"),
                    "charge": left.get("charge"),
                    "station_occupancy": left.get("station_occupancy"),
                }
            )
        lost_chart = [record for record in records if record["reference_selected"] and not record["chart_selected"]]
        lost_both = [
            record
            for record in records
            if not record["reference_selected"] and not record["chart_selected"]
        ]
        recovered_twin = [record for record in lost_chart if record["twin_selected"]]
        families[family] = {
            "chart_total_complete_truth_routes": len(chart_rows),
            "twin_total_complete_truth_routes": len(twin_rows),
            "aligned_complete_truth_routes": len(records),
            "reference_selected": sum(1 for record in records if record["reference_selected"]),
            "chart_selected": sum(1 for record in records if record["chart_selected"]),
            "twin_selected": sum(1 for record in records if record["twin_selected"]),
            "reference_selected_chart_lost": len(lost_chart),
            "reference_and_chart_both_lost": len(lost_both),
            "chart_lost_twin_recovered": len(recovered_twin),
            "chart_lost_records": lost_chart,
            "both_lost_records": lost_both,
        }
    return {"families": families}


def _gate_replay_report(
    payload_metrics: Mapping[str, RouteMetrics],
    nominal: RouteMetrics,
    gate_decision: Mapping[str, Any],
) -> dict[str, object]:
    """Replay vs frozen workbook-64 gate numbers; any drift fails the audit."""
    checks: list[dict[str, object]] = []
    candidate_payloads = gate_decision["candidate"]["payloads"]
    adjacent_payloads = gate_decision["adjacent_23_and_s3"]["payloads"]
    for payload_id in PAYLOADS:
        metrics = payload_metrics[payload_id]
        frozen_route = candidate_payloads[payload_id]["selected_route"]
        for key in (
            "complete_truth_chains",
            "candidate_retained_complete_truth_chains",
            "score_retained_complete_truth_chains",
            "selected_complete_routes",
            "correct_complete_routes",
            "selected_routes",
            "duplicate_routes",
            "missing_station_boundaries",
        ):
            observed = int(getattr(metrics, key))
            expected = int(frozen_route[key])
            checks.append(
                {
                    "payload_id": payload_id,
                    "quantity": key,
                    "observed": observed,
                    "expected": expected,
                    "ok": abs(observed - expected) <= REPLAY_COUNT_TOLERANCE,
                }
            )
        for ratio_name, numerator, denominator in (
            ("complete_track_efficiency", metrics.correct_complete_routes, metrics.complete_truth_chains),
            ("complete_track_purity", metrics.correct_complete_routes, metrics.selected_complete_routes),
            ("track_fake_rate", metrics.selected_routes - metrics.truth_consistent_routes, metrics.selected_routes),
        ):
            observed = numerator / denominator if denominator else None
            expected = frozen_route[ratio_name]
            ok = observed is not None and abs(float(observed) - float(expected)) <= REPLAY_RATIO_TOLERANCE
            checks.append(
                {
                    "payload_id": payload_id,
                    "quantity": ratio_name,
                    "observed": observed,
                    "expected": expected,
                    "ok": ok,
                }
            )
        adjacent_frozen = adjacent_payloads[payload_id]
        adjacent_observed = (
            metrics.correctly_dustbinned_missing_station_boundaries,
            metrics.missing_station_boundaries,
        )
        if payload_id == "iteration_00_reference":
            efficiency_23_expected = adjacent_frozen["association_efficiency_2_3"]
        else:
            efficiency_23_expected = None
        checks.append(
            {
                "payload_id": payload_id,
                "quantity": "adjacent_23_replay_available",
                "observed": list(adjacent_observed),
                "expected": efficiency_23_expected,
                "ok": True,
            }
        )
    return {
        "checks": checks,
        "all_ok": all(bool(item["ok"]) for item in checks),
    }


def _failure_summary(
    rows: Sequence[Mapping[str, Any]],
    payload_metrics: Mapping[str, RouteMetrics],
    mechanism: Mapping[str, Any],
) -> dict[str, object]:
    nominal_metrics = payload_metrics["iteration_00_reference"]
    draw_metrics = payload_metrics["iteration_00_draw_00"]
    by_payload: dict[str, object] = {}
    for payload_id in PAYLOADS:
        payload_rows = [row for row in rows if str(row["payload_id"]) == payload_id]
        problems = Counter(
            problem
            for row in payload_rows
            for problem in _failure_problems(row)
        )
        stages = Counter(str(row["loss_stage"]) for row in payload_rows)
        failed_pairs = Counter(
            pair
            for row in payload_rows
            if str(row["loss_stage"]) == "below_station_pair_threshold"
            for pair in (row.get("failed_pairs") or [])
        )
        competitor_topologies = Counter(
            str((row.get("solver_competitor") or {}).get("topology") or "none")
            for row in payload_rows
            if row.get("production_fragment_winner")
        )
        competitor_pairs = Counter(
            str((row.get("solver_competitor") or {}).get("station_pairs") or [])
            for row in payload_rows
            if row.get("production_fragment_winner")
        )
        s3_involvement = Counter(
            bool((row.get("solver_competitor") or {}).get("involves_s3"))
            or any(pair.endswith("->3") for pair in (row.get("failed_pairs") or []))
            for row in payload_rows
            if not row.get("selected")
        )
        u_truth_values = [float(row["u_truth"]) for row in payload_rows if row.get("u_truth") is not None]
        margins = [
            float(row["production_margin"])
            for row in payload_rows
            if row.get("production_margin") is not None
        ]
        by_payload[payload_id] = {
            "complete_truth_chains": len(payload_rows),
            "loss_stage_counts": dict(stages),
            "problem_counts": dict(problems),
            "below_threshold_failed_pair_counts": dict(failed_pairs),
            "fragment_winner_competitor_topologies": dict(competitor_topologies),
            "fragment_winner_competitor_station_pairs": dict(competitor_pairs),
            "unselected_routes_with_s3_involvement": int(s3_involvement.get(True, 0)),
            "unselected_routes_total": int(sum(s3_involvement.values())),
            "u_truth_median": None if not u_truth_values else float(_median(u_truth_values)),
            "u_truth_fraction_nonpositive": (
                None if not u_truth_values else float(sum(1 for v in u_truth_values if v <= 0.0) / len(u_truth_values))
            ),
            "production_margin_median": None if not margins else float(_median(margins)),
        }
    nominal_rows = [row for row in rows if str(row["payload_id"]) == "iteration_00_reference"]
    draw_rows = [row for row in rows if str(row["payload_id"]) == "iteration_00_draw_00"]
    nominal_by_key = {_alignment_key(row): row for row in nominal_rows}
    draw_by_key = {_alignment_key(row): row for row in draw_rows}
    shared_keys = sorted(set(nominal_by_key) & set(draw_by_key))
    transitions = Counter()
    transition_examples = defaultdict(list)
    for key in shared_keys:
        row = nominal_by_key[key]
        counterpart = draw_by_key[key]
        transition = (
            bool(row["selected"]),
            bool(counterpart["selected"]),
            str(row["loss_stage"]),
            str(counterpart["loss_stage"]),
        )
        transitions[transition] += 1
        if len(transition_examples[str(transition)]) < 5:
            transition_examples[str(transition)].append(
                {
                    "source_event_uid": counterpart.get("source_event_uid"),
                    "truth_id": counterpart.get("truth_id"),
                    "run_id": int(counterpart["run_id"]),
                    "event_id": int(counterpart["event_id"]),
                    "u_truth_reference": row.get("u_truth"),
                    "u_truth_draw_00": counterpart.get("u_truth"),
                    "failed_pairs_draw_00": list(counterpart.get("failed_pairs") or []),
                    "s3_state_draw_00": counterpart.get("s3_state"),
                    "track_state_by_station": counterpart.get("track_state_by_station"),
                }
            )
    mechanism_payloads = mechanism.get("payloads") or {}
    u_truth_median_matches = {
        payload_id: (
            by_payload[payload_id]["u_truth_median"] is not None
            and mechanism_payloads.get(payload_id, {}).get("feasibility", {}).get("u_truth_median") is not None
            and abs(
                float(by_payload[payload_id]["u_truth_median"])
                - float(mechanism_payloads[payload_id]["feasibility"]["u_truth_median"])
            )
            <= 1.0e-6
        )
        for payload_id in PAYLOADS
    }
    return {
        "by_payload": by_payload,
        "reference_to_draw_00_transitions": {
            "counts": {str(key): int(value) for key, value in transitions.items()},
            "examples": dict(transition_examples),
        },
        "efficiency_decomposition_reference_minus_draw_00": {
            "reference_efficiency": float(nominal_metrics.correct_complete_routes / nominal_metrics.complete_truth_chains),
            "draw_00_efficiency": float(draw_metrics.correct_complete_routes / draw_metrics.complete_truth_chains),
            "efficiency_drop": float(
                nominal_metrics.correct_complete_routes / nominal_metrics.complete_truth_chains
                - draw_metrics.correct_complete_routes / draw_metrics.complete_truth_chains
            ),
            "reference_correct": int(nominal_metrics.correct_complete_routes),
            "draw_00_correct": int(draw_metrics.correct_complete_routes),
            "draw_00_candidate_missing": int(
                draw_metrics.complete_truth_chains - draw_metrics.candidate_retained_complete_truth_chains
            ),
            "draw_00_below_threshold": int(
                draw_metrics.candidate_retained_complete_truth_chains
                - draw_metrics.score_retained_complete_truth_chains
            ),
        },
        "mechanism_u_truth_median_consistent": u_truth_median_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--candidate-frozen-output", required=True)
    parser.add_argument("--gate-decision-json", required=True)
    parser.add_argument("--mechanism-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if str(args.device) != "cuda":
        raise SystemExit("blind failure localization is GPU-only (no silent CPU fallback)")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    gate_decision = json.loads(Path(args.gate_decision_json).read_text(encoding="utf-8"))
    mechanism = json.loads(Path(args.mechanism_json).read_text(encoding="utf-8"))
    checkpoint = Path(args.candidate_frozen_output).expanduser().resolve()
    if _sha256(checkpoint / "route_aware_transformer_v2.pt") != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint file hash disagrees with the frozen workbook-64 sha256")
    frozen = _load_v2(checkpoint, args.device)

    rows, payload_metrics, _per_payload_raw = _replay_payloads(
        frozen,
        args.synthetic_manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    gate_replay = _gate_replay_report(
        payload_metrics,
        payload_metrics["iteration_00_reference"],
        gate_decision,
    )
    if not gate_replay["all_ok"]:
        _write_json(output / "gate_replay.json", gate_replay)
        raise SystemExit("replay disagrees with the frozen workbook-64 gate numbers; refusing to interpret")

    alignment = _alignment_report(rows)
    failure_summary = _failure_summary(rows, payload_metrics, mechanism)
    failure_rows = [_failure_row_summary(row) for row in rows]
    _write_jsonl(output / "blind_failure_rows.jsonl", failure_rows)
    _write_json(output / "nominal_vs_hard_alignment.json", alignment)
    _write_json(output / "failure_summary.json", failure_summary)

    problem_totals = Counter(
        problem for row in rows for problem in _failure_problems(row)
    )
    draw_rows = [row for row in rows if str(row["payload_id"]) == "iteration_00_draw_00"]
    draw_unselected = [row for row in draw_rows if not row.get("selected")]
    decomposition = failure_summary["efficiency_decomposition_reference_minus_draw_00"]
    unexplained = [
        row
        for row in draw_unselected
        if not _failure_problems(row)
        and str(row["loss_stage"]) not in {"utility_nonpositive", "packing_competition"}
    ]
    decision = {
        "workbook": 65,
        "question": "where does the workbook-64 draw_00 complete-track efficiency drop of 0.117 come from",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "reserved_blind_role": "development_mechanism_diagnostic_only",
        "test_data_accessed": False,
        "continue_to_15d_relative_wls": False,
        "gate_replay_all_ok": bool(gate_replay["all_ok"]),
        "problem_totals_all_payloads": dict(problem_totals),
        "draw_00_unselected_routes": len(draw_unselected),
        "draw_00_unselected_problem_counts": dict(
            Counter(problem for row in draw_unselected for problem in _failure_problems(row))
        ),
        "draw_00_unselected_outside_pre_registered_stages": len(unexplained),
        "efficiency_decomposition_reference_minus_draw_00": decomposition,
        "classification": {
            "A_candidate_missing": int(problem_totals.get("A", 0)),
            "B_edge_ranking": int(
                sum(
                    1
                    for row in rows
                    if str(row["loss_stage"]) == "below_station_pair_threshold"
                    and "B" in _failure_problems(row)
                )
            ),
            "C_absolute_score_scale": int(
                sum(1 for row in rows if "C" in _failure_problems(row))
            ),
            "D_route_utility_or_packing": int(
                sum(1 for row in rows if str(row["loss_stage"]) == "packing_competition")
            ),
            "E_beyond_pre_registered_stages": len(unexplained),
        },
        "structural_representation_limitation_proven": None,
        "implementation_bug_suspected": False,
        "next_step": None,
    }
    _write_json(output / "decision.json", decision)
    print(json.dumps({"output_dir": str(output), "decision": decision}, indent=2)[:4000])


if __name__ == "__main__":
    main()
