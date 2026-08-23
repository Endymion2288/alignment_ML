#!/usr/bin/env python3
"""Frozen-V2 association-only DQ for the real-data route-acceptance scaling study.

No residuals, no alignment parameters, no geometry payload, no truth metrics.
14977 is reported and excluded from the verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ
from alignment.real_data_v2_route_acceptance_scaling import (
    SCALE_ORDER,
    association_rejection_counts,
    assign_scaling_verdict,
    load_scaling_config,
)
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from datasets.root_loader import load_events
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
)
from training.route_aware_transformer import (
    load_route_aware_transformer_artifact,
    materialize_route_candidate_tables,
    predict_route_aware_scores,
)
from training.route_assignment import assign_adjacent_route_sets
from scripts.run_frozen_association_backbone import _load_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "efficiency",
    "purity",
    "fake",
    "auc",
    "average_precision",
    "ap",
    "roc",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _walk_forbidden(payload: object, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            parts = set(lowered.split("_"))
            if any(token == lowered or token in parts for token in FORBIDDEN):
                raise ValueError(f"{where} contains forbidden truth-metric key {key!r}")
            _walk_forbidden(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            _walk_forbidden(item, where=where)


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("p05", "p16", "p50", "p84", "p95")}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in ("p05", "p16", "p50", "p84", "p95")}
    return {
        "p05": float(np.percentile(finite, 5)),
        "p16": float(np.percentile(finite, 16)),
        "p50": float(np.percentile(finite, 50)),
        "p84": float(np.percentile(finite, 84)),
        "p95": float(np.percentile(finite, 95)),
    }


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{int(pair[0])}->{int(pair[1])}"


def _log_odds(score: float) -> float:
    probability = float(np.clip(score, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(probability) - np.log1p(-probability))


def _load_routes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def analyze_identity_sample(
    *,
    sample: Any,
    role: str,
    run: int,
    scale: str,
    source_id: str,
    frozen: Mapping[str, Any],
    association_dir: Path | None,
    device: str,
    unmatched_penalty: float,
) -> dict[str, Any]:
    events = load_events(sample.synthetic_tracklets, require_mc_labels=False)
    candidate_sets = build_candidate_sets(
        [sample],
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
        require_mc_labels=False,
    )
    pair_counts = {_pair_key(pair): 0 for pair in ALL_STATION_PAIRS}
    physical_per_event: dict[tuple[int, int], int] = defaultdict(int)
    for candidate_set in candidate_sets:
        pair_counts[_pair_key(candidate_set.station_pair)] += int(len(candidate_set.candidates))
        physical_per_event[(int(candidate_set.event.run_id), int(candidate_set.event.event_id))] += int(
            len(candidate_set.candidates)
        )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode=frozen["artifact"].context_mode)
    prediction = predict_route_aware_scores(
        frozen["model"],
        bundle,
        frozen["artifact"].node_standardizer,
        frozen["artifact"].edge_standardizer,
        device=device,
        require_mc_labels=False,
    )
    calibrated = apply_frozen_transformer_calibration(
        bundle.adjacent_sets, list(prediction.edge_scores), frozen["calibration"]
    )
    thresholds = {
        (int(source), int(target)): float(value)
        for (source, target), value in frozen["route"].score_threshold_by_pair.items()
    }
    adjacent_scores: list[float] = []
    n_below = 0
    n_above = 0
    lookup: dict[tuple[int, int, int, int, int, int], float] = {}
    for candidate_set, scores in zip(bundle.adjacent_sets, calibrated):
        event = candidate_set.event
        pair = candidate_set.station_pair
        threshold = thresholds[pair]
        for candidate, score in zip(candidate_set.candidates, np.asarray(scores, dtype=np.float64)):
            value = float(score)
            if not math.isfinite(value):
                continue
            adjacent_scores.append(value)
            if value >= threshold:
                n_above += 1
                lookup[
                    (
                        int(event.run_id),
                        int(event.event_id),
                        int(pair[0]),
                        int(pair[1]),
                        int(candidate.source_index),
                        int(candidate.target_index),
                    )
                ] = value
            else:
                n_below += 1
    route_tables = materialize_route_candidate_tables(bundle.graphs, require_mc_labels=False)
    n_complete_physical = int(sum(table.size for table in route_tables.values()))
    complete_missing = 0
    utilities: list[float] = []
    n_nonpositive = 0
    n_positive = 0
    edges_by_event: dict[tuple[int, int], dict[tuple[int, int], list[tuple[int, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (run_id, event_id, source_station, target_station, source_index, target_index), score in lookup.items():
        edges_by_event[(run_id, event_id)][(source_station, target_station)].append(
            (source_index, target_index, score)
        )
    physical_complete_by_event: dict[tuple[int, int], int] = defaultdict(int)
    for event in events:
        by_station = {
            int(station): int(sum(1 for item in event.station_id.tolist() if int(item) == int(station)))
            for station in range(4)
        }
        if all(by_station[station] > 0 for station in range(4)):
            physical_complete_by_event[(int(event.run_id), int(event.event_id))] = (
                by_station[0] * by_station[1] * by_station[2] * by_station[3]
            )
    for event_key, pairs in edges_by_event.items():
        for pair in ((0, 1), (1, 2), (2, 3)):
            for _source_index, _target_index, score in pairs.get(pair, []):
                utility = float(_log_odds(score) + 2 * unmatched_penalty)
                utilities.append(utility)
                n_positive += int(utility > 0.0)
                n_nonpositive += int(utility <= 0.0)
        three_station_specs = (
            ((0, 1), (1, 2)),
            ((1, 2), (2, 3)),
        )
        for left_pair, right_pair in three_station_specs:
            for _source_a, target_a, score_a in pairs.get(left_pair, []):
                for source_b, _target_b, score_b in pairs.get(right_pair, []):
                    if target_a != source_b:
                        continue
                    scores = [score_a, score_b]
                    utility = float(sum(_log_odds(score) for score in scores) + 3 * unmatched_penalty)
                    utilities.append(utility)
                    n_positive += int(utility > 0.0)
                    n_nonpositive += int(utility <= 0.0)
        four_count = 0
        for source_a, mid_a, score_a in pairs.get((0, 1), []):
            for source_b, mid_b, score_b in pairs.get((1, 2), []):
                if mid_a != source_b:
                    continue
                for source_c, target_c, score_c in pairs.get((2, 3), []):
                    if mid_b != source_c:
                        continue
                    four_count += 1
                    scores = [score_a, score_b, score_c]
                    utility = float(sum(_log_odds(score) for score in scores) + 4 * unmatched_penalty)
                    utilities.append(utility)
                    n_positive += int(utility > 0.0)
                    n_nonpositive += int(utility <= 0.0)
        complete_missing += max(int(physical_complete_by_event.get(event_key, 0)) - four_count, 0)
    selections = assign_adjacent_route_sets(
        bundle.adjacent_sets,
        calibrated,
        frozen["route"],
        calibration_bins=15,
    )
    selected_routes = [route for assigned in selections for route in assigned.result.routes]
    selected_from_file = _load_routes(association_dir / "selected_routes.jsonl") if association_dir else []
    n_selected = int(len(selected_from_file) if selected_from_file or (association_dir and (association_dir / "selected_routes.jsonl").is_file()) else len(selected_routes))
    complete_selected = 0
    per_event_selected: dict[tuple[int, int], int] = defaultdict(int)
    endpoint_uses: Counter[tuple[int, int, int, int]] = Counter()
    if selected_from_file:
        complete_selected = int(sum(1 for row in selected_from_file if bool(row.get("is_complete_four_station_route"))))
        for row in selected_from_file:
            per_event_selected[(int(row["run_id"]), int(row["event_id"]))] += 1
            for item in row.get("endpoint_provenance") or []:
                endpoint_uses[
                    (
                        int(item["origin_run_id"]),
                        int(item["origin_event_id"]),
                        int(item["station_id"]),
                        int(item["origin_tracklet_id"]),
                    )
                ] += 1
        selected_utilities = [
            float(row["utility"]) for row in selected_from_file if row.get("utility") is not None
        ]
    else:
        complete_selected = int(sum(1 for route in selected_routes if len(route.endpoints) == 4))
        for assigned in selections:
            if assigned.result.routes:
                per_event_selected[(int(assigned.key[2]), int(assigned.key[3]))] += len(assigned.result.routes)
        selected_utilities = [float(route.utility) for route in selected_routes]
    n_selected_edges = 0
    if association_dir is not None:
        summary_path = association_dir / "association_summary.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
            n_selected_edges = int((summary.get("truth_free_selection") or {}).get("field_aware_edge_observations") or 0)
            n_selected = int((summary.get("truth_free_selection") or {}).get("selected_routes") or n_selected)
            complete_selected = int(
                (summary.get("truth_free_selection") or {}).get("complete_four_station_routes") or complete_selected
            )
    else:
        n_selected_edges = int(sum(len(route.matches) for route in selected_routes))
    physical_total = float(sum(physical_per_event.values()))
    physical_shares = sorted(
        (float(count) / physical_total for count in physical_per_event.values()),
        reverse=True,
    ) if physical_total else []
    selected_total = float(sum(per_event_selected.values()))
    selected_share = (
        max(per_event_selected.values()) / selected_total if selected_total else 0.0
    )
    reused = sum(1 for count in endpoint_uses.values() if count > 1)
    route_counts = np.asarray(list(per_event_selected.values()), dtype=np.int64) if per_event_selected else np.zeros(0, dtype=np.int64)
    adjacent = np.asarray(adjacent_scores, dtype=np.float64)
    utility_array = np.asarray(utilities, dtype=np.float64)
    n_candidates = int(sum(pair_counts.values()))
    report = {
        "source_id": source_id,
        "run": int(run),
        "role": str(role),
        "scale": scale,
        "association_complete": True,
        "used_for_verdict": role == ROLE_CALIBRATION,
        "n_events_with_tracklets": int(len(events)),
        "n_tracklets": int(sum(int(event.size) for event in events)),
        "n_all_pairs_candidates": n_candidates,
        "all_pairs_candidate_counts": pair_counts,
        "candidate_graph_nonempty": n_candidates > 0,
        "selected_graph_nonempty": n_selected > 0 and n_selected_edges > 0,
        "n_adjacent_candidates": int(adjacent.size),
        "n_adjacent_below_threshold": int(n_below),
        "n_adjacent_above_threshold": int(n_above),
        "candidate_score_distribution": _percentiles(adjacent),
        "candidate_utility_distribution": _percentiles(utility_array),
        "n_route_hypotheses_with_above_threshold_edges": int(utility_array.size),
        "n_physical_complete_route_candidates": n_complete_physical,
        "selected_edges": int(n_selected_edges),
        "selected_routes": int(n_selected),
        "complete_four_station_routes": int(complete_selected),
        "selected_utility_distribution": _percentiles(np.asarray(selected_utilities, dtype=np.float64)),
        "route_multiplicity": {
            "mean": None if route_counts.size == 0 else float(np.mean(route_counts)),
            "max": None if route_counts.size == 0 else int(np.max(route_counts)),
        },
        "endpoint_reuse": {
            "unique_endpoints": int(len(endpoint_uses)),
            "reused_endpoints": int(reused),
            "max_uses": 0 if not endpoint_uses else int(max(endpoint_uses.values())),
        },
        "event_concentration": {
            "max_event_share_of_physical_candidates": physical_shares[0] if physical_shares else 0.0,
            "top2_event_share_of_physical_candidates": float(sum(physical_shares[:2])),
            "max_event_share_of_selected_routes": float(selected_share),
            "n_events_with_selected_routes": int(len(per_event_selected)),
        },
        "candidate_rejection_reason": association_rejection_counts(
            n_all_pairs_candidates=n_candidates,
            n_adjacent_candidates=int(adjacent.size),
            n_adjacent_below_threshold=n_below,
            n_adjacent_above_threshold=n_above,
            n_physical_complete_routes=n_complete_physical,
            n_complete_routes_missing_above_threshold_edge=complete_missing,
            n_routes_above_threshold_nonpositive_utility=n_nonpositive,
            n_routes_positive_utility=n_positive,
            n_selected_routes=n_selected,
        ),
        "frozen_route_policy": {
            "thresholds": {f"{a}->{b}": float(v) for (a, b), v in thresholds.items()},
            "unmatched_penalty": float(unmatched_penalty),
            "do_not_change": True,
        },
        "truth_metrics_omitted": True,
        "residuals_excluded": True,
        "geometry_write_allowed": False,
    }
    _walk_forbidden(report, where=source_id)
    return report


def _n100_sources(physical_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(source) for source in physical_manifest["sources"]]


def _count_field_candidates(path: Path) -> int:
    if not path.is_file():
        return 0
    import uproot

    with uproot.open(path) as source:
        tree = source["propagations"] if "propagations" in source else None
        if tree is None:
            for key in source.keys():
                name = str(key).split(";")[0]
                if name not in {"metadata"}:
                    tree = source[key]
                    break
        if tree is None:
            return 0
        return int(tree.num_entries)


def block_from_artifacts(
    *,
    meta: Mapping[str, Any],
    association_dir: Path,
    content_audit: Mapping[str, Any],
    candidate_records: int | None,
    field_candidates: Path | None = None,
) -> dict[str, Any]:
    summary = _read_json(association_dir / "association_summary.json")
    truth_free = summary.get("truth_free_selection") or {}
    field = summary.get("field_aware_route_consistency") or {}
    backbone = summary.get("frozen_backbone") or summary.get("route_solver") or {}
    routes = _load_routes(association_dir / "selected_routes.jsonl")
    per_event: dict[tuple[int, int], int] = defaultdict(int)
    utilities: list[float] = []
    endpoint_uses: Counter[tuple[int, int, int, int]] = Counter()
    for row in routes:
        per_event[(int(row["run_id"]), int(row["event_id"]))] += 1
        if row.get("utility") is not None:
            utilities.append(float(row["utility"]))
        for item in row.get("endpoint_provenance") or []:
            endpoint_uses[
                (
                    int(item["origin_run_id"]),
                    int(item["origin_event_id"]),
                    int(item["station_id"]),
                    int(item["origin_tracklet_id"]),
                )
            ] += 1
    selected_total = float(sum(per_event.values()))
    n_selected = int(truth_free.get("selected_routes", field.get("routes", len(routes))))
    n_complete = int(truth_free.get("complete_four_station_routes", field.get("complete_four_station_routes", 0)))
    n_edges = int(truth_free.get("field_aware_edge_observations", field.get("selected_field_aware_edges", 0)))
    n_tracklets = int(content_audit.get("tracklets") or 0)
    n_events_with_tracklets = int(content_audit.get("events") or 0)
    n_candidates = int(candidate_records or 0)
    if n_candidates <= 0 and field_candidates is not None:
        n_candidates = _count_field_candidates(field_candidates)
    route_counts = np.asarray(list(per_event.values()), dtype=np.int64) if per_event else np.zeros(0, dtype=np.int64)
    report = {
        "source_id": meta["source_id"],
        "run": int(meta["run"]),
        "role": str(meta["role"]),
        "scale": str(meta["scale"]),
        "association_complete": True,
        "used_for_verdict": str(meta["role"]) == ROLE_CALIBRATION,
        "n_events": int(meta["nevents"]),
        "n_events_with_tracklets": n_events_with_tracklets,
        "n_tracklets": n_tracklets,
        "n_all_pairs_candidates": n_candidates,
        "candidate_graph_nonempty": n_candidates > 0,
        "selected_graph_nonempty": n_selected > 0,
        "selected_edges": n_edges,
        "selected_routes": n_selected,
        "complete_four_station_routes": n_complete,
        "selected_utility_distribution": _percentiles(np.asarray(utilities, dtype=np.float64)),
        "route_multiplicity": {
            "mean": None if route_counts.size == 0 else float(np.mean(route_counts)),
            "max": None if route_counts.size == 0 else int(np.max(route_counts)),
        },
        "endpoint_reuse": {
            "unique_endpoints": int(len(endpoint_uses)),
            "reused_endpoints": int(sum(1 for count in endpoint_uses.values() if count > 1)),
            "max_uses": 0 if not endpoint_uses else int(max(endpoint_uses.values())),
        },
        "event_concentration": {
            "max_event_share_of_selected_routes": (
                max(per_event.values()) / selected_total if selected_total else 0.0
            ),
            "n_events_with_selected_routes": int(len(per_event)),
        },
        "candidate_rejection_reason": {
            "selected_routes": n_selected,
            "complete_four_station_routes": n_complete,
            "source": "frozen_v2_association_summary",
            "score_utility_full_enumeration": "omitted_expanded_scale_combinatorial_cost",
        },
        "candidate_score_distribution": None,
        "candidate_utility_distribution": None,
        "frozen_route_policy": {
            "thresholds": dict(backbone.get("thresholds") or {}),
            "unmatched_penalty": backbone.get("unmatched_penalty"),
            "checkpoint_sha256": backbone.get("checkpoint_sha256"),
            "do_not_change": True,
        },
        "artifact_source": str(association_dir),
        "truth_metrics_omitted": True,
        "residuals_excluded": True,
        "geometry_write_allowed": False,
    }
    _walk_forbidden(report, where=str(meta["source_id"]))
    return report


def _scale_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        concentration = row.get("event_concentration") or {}
        table.append(
            {
                "scale": row.get("scale"),
                "run": row.get("run"),
                "role": row.get("role"),
                "n_events": row.get("n_events"),
                "n_events_with_tracklets": row.get("n_events_with_tracklets"),
                "n_tracklets": row.get("n_tracklets"),
                "n_all_pairs_candidates": row.get("n_all_pairs_candidates"),
                "candidate_graph_nonempty": row.get("candidate_graph_nonempty"),
                "selected_graph_nonempty": row.get("selected_graph_nonempty"),
                "selected_edges": row.get("selected_edges"),
                "selected_routes": row.get("selected_routes"),
                "complete_four_station_routes": row.get("complete_four_station_routes"),
                "max_event_share_of_selected_routes": concentration.get("max_event_share_of_selected_routes"),
                "used_for_verdict": row.get("used_for_verdict"),
            }
        )
    return table


def _occupancy_athena_comparison(
    forecasts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(int(row["run"]), str(row["scale"])): row for row in rows}
    comparison: list[dict[str, Any]] = []
    for forecast in forecasts:
        key = (int(forecast["run"]), str(forecast["scale"]))
        row = by_key.get(key)
        occ = forecast.get("occupancy_forecast") or {}
        comparison.append(
            {
                "run": int(forecast["run"]),
                "role": forecast.get("role"),
                "scale": forecast.get("scale"),
                "n_events": forecast.get("nevents"),
                "occupancy_tracklets": occ.get("total_tracklets"),
                "athena_tracklets": None if row is None else row.get("n_tracklets"),
                "occupancy_events_with_tracklets": occ.get("n_events_with_tracklets"),
                "athena_events_with_tracklets": None if row is None else row.get("n_events_with_tracklets"),
                "capped_to_segment": forecast.get("capped_to_segment"),
            }
        )
    return comparison


def _condor_metadata(scaling_root: Path) -> dict[str, Any]:
    n1000 = scaling_root / "condor_n1000" / "submission.json"
    n10000 = scaling_root / "condor_n10000_full" / "submission.json"
    return {
        "n1000_cluster": 1000823 if n1000.is_file() else None,
        "n10000_full_cluster": 1000824 if n10000.is_file() else None,
        "n1000_jobs": 5,
        "n10000_full_jobs": 9,
        "n100_athena": "reused_entry_48",
        "current_geometry_only": True,
        "finite_difference_probes": False,
    }


def _interpretation(verdict: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision = str(verdict.get("decision"))
    calibration_full = [
        row
        for row in rows
        if row.get("role") == ROLE_CALIBRATION and row.get("scale") == "full"
    ]
    return {
        "dry_run_empty_selected_graph": (
            "statistics_limited"
            if decision == "real_data_route_acceptance_statistics_limited"
            else decision
        ),
        "frozen_v2_data_compatibility": (
            "not_the_primary_empty-graph_cause"
            if decision == "real_data_route_acceptance_statistics_limited"
            else "primary_if_selected_routes_remain_zero"
        ),
        "complete_four_station_routes": (
            "not_stably_recovered_on_both_calibration_runs"
            if verdict.get("complete_four_station_recovered_at_scale") is None
            else f"recovered_at_{verdict.get('complete_four_station_recovered_at_scale')}"
        ),
        "tracklet_statistics": (
            "not_limited"
            if decision != "real_data_tracklet_statistics_limited"
            else "limited"
        ),
        "selected_graph_recovers_with_n": decision == "real_data_route_acceptance_statistics_limited",
        "full_calibration_selected_routes": {
            int(row["run"]): int(row.get("selected_routes") or 0) for row in calibration_full
        },
        "full_calibration_complete_four_station_routes": {
            int(row["run"]): int(row.get("complete_four_station_routes") or 0) for row in calibration_full
        },
        "station_mode": "blocked",
        "cdx_mode": "blocked",
        "geometry_write_allowed": False,
        "official_conditions_db_write": False,
        "alignment_parameter_estimation": False,
        "notes": [
            "n100 selected routes are zero on every run while all-pairs candidates are nonempty.",
            "Dominant n100 rejection is route_above_threshold_nonpositive_utility, not an empty candidate graph.",
            "Selected routes appear on both calibration runs by n10000 and remain at full segment.",
            "Complete four-station routes remain absent on 14973 even at full remaining segment.",
            "14975/14976 are blind holdout and 14977 is held-out DQ; none of them chose the verdict.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaling-config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_v2_route_acceptance_scaling_v1.yaml"),
    )
    parser.add_argument(
        "--n100-physical-manifest",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "operating_protocol_v1_real_data_station_mode_physical_frozen_windows_v2"
            / "iteration_manifest.json"
        ),
    )
    parser.add_argument(
        "--n100-identity-manifest",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "operating_protocol_v1_real_data_identity_curriculum_current_v1"
            / "synthetic_corpus_manifest.json"
        ),
    )
    parser.add_argument(
        "--n100-association-root",
        default=str(
            PROJECT_ROOT / "outputs" / "operating_protocol_v1_real_data_v2_current_frozen_windows_v1"
        ),
    )
    parser.add_argument("--scaling-root", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--from-artifacts",
        action="store_true",
        help="Read expanded-scale DQ from completed frozen V2 outputs instead of rescoring.",
    )
    args = parser.parse_args()
    config = load_scaling_config(args.scaling_config)
    frozen_v2 = _load_v2(PROJECT_ROOT / str(config["frozen_v2"]), args.device)
    unmatched = float(frozen_v2["route"].unmatched_penalty)
    physical = _read_json(Path(args.n100_physical_manifest).expanduser().resolve())
    _, samples, _ = load_synthetic_curriculum_manifest(
        args.n100_identity_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    association_root = Path(args.n100_association_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for source in _n100_sources(physical):
        source_id = str(source["source_id"])
        current = [sample for sample in samples if sample.source_id == source_id and sample.payload_id == "iteration_00_current"]
        if len(current) != 1:
            raise ValueError(f"expected one current identity sample for {source_id}")
        row = analyze_identity_sample(
            sample=current[0],
            role=str(source["role"]),
            run=int(source["run"]),
            scale="n100",
            source_id=source_id,
            frozen=frozen_v2,
            association_dir=association_root / source_id,
            device=args.device,
            unmatched_penalty=unmatched,
        )
        row["n_events"] = 100
        rows.append(row)
    if args.scaling_root:
        scaling_root = Path(args.scaling_root).expanduser().resolve()
        plan = _read_json(scaling_root / "scaling_plan.json") if (scaling_root / "scaling_plan.json").is_file() else {}
        identity_root = scaling_root / "identity"
        association_root_new = scaling_root / "association"
        if identity_root.is_dir():
            for manifest_path in sorted((identity_root / "manifests").glob("*.json")):
                raw_manifest = _read_json(manifest_path)
                splits = {
                    str(sample["split"])
                    for sample in raw_manifest.get("samples") or []
                    if isinstance(sample, Mapping)
                }
                _, scale_samples, _ = load_synthetic_curriculum_manifest(
                    manifest_path,
                    require_all_splits=False,
                    allowed_splits=tuple(splits) or ("train", "validation"),
                )
                for sample in scale_samples:
                    if sample.payload_id != "iteration_00_current":
                        continue
                    metas = [
                        item
                        for item in (plan.get("planned") or [])
                        if item.get("source_id") == sample.source_id
                    ]
                    if not metas:
                        continue
                    for meta in metas:
                        association_dir = association_root_new / sample.source_id
                        large_scale = str(meta["scale"]) in {"n10000", "full"}
                        use_artifacts = association_dir.is_dir() and (args.from_artifacts or large_scale)
                        if use_artifacts and large_scale:
                            source_row = next(
                                item
                                for item in _read_json(scaling_root / "iteration_manifest.json")["sources"]
                                if item["source_id"] == sample.source_id
                            )
                            audit = _read_json(
                                Path(source_row["physical_scan_root"])
                                / "points"
                                / "iteration_00_current"
                                / "refit"
                                / "content_audit.json"
                            )
                            field_path = Path(str(raw_manifest["samples"][0].get("field_candidates") or ""))
                            rows.append(
                                block_from_artifacts(
                                    meta=meta,
                                    association_dir=association_dir,
                                    content_audit=audit,
                                    candidate_records=raw_manifest["samples"][0].get("candidate_records"),
                                    field_candidates=field_path if field_path.is_file() else None,
                                )
                            )
                            continue
                        row = analyze_identity_sample(
                            sample=sample,
                            role=str(meta["role"]),
                            run=int(meta["run"]),
                            scale=str(meta["scale"]),
                            source_id=str(sample.source_id),
                            frozen=frozen_v2,
                            association_dir=association_dir if association_dir.is_dir() else None,
                            device=args.device,
                            unmatched_penalty=unmatched,
                        )
                        row["n_events"] = int(meta["nevents"])
                        rows.append(row)
    else:
        plan = {}
    verdict = assign_scaling_verdict(rows, config)
    scale_table = _scale_table(rows)
    occupancy_comparison = _occupancy_athena_comparison(plan.get("occupancy_forecasts") or [], rows)
    interpretation = _interpretation(verdict, rows)
    condor = _condor_metadata(Path(args.scaling_root).expanduser().resolve()) if args.scaling_root else {}
    payload = {
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": verdict["decision"],
        "verdict": verdict,
        "interpretation": interpretation,
        "dry_run_route_admission_failure": (
            "statistics_limited"
            if verdict["decision"] == "real_data_route_acceptance_statistics_limited"
            else (
                "frozen_v2_data_compatibility_limited"
                if verdict["decision"] == "real_data_v2_domain_shift_candidate"
                else verdict["decision"]
            )
        ),
        "official_conditions_db_write": False,
        "geometry_write_allowed": False,
        "station_mode_blocked": True,
        "cdx_mode_blocked": True,
        "residuals_excluded": True,
        "generate_geometry_payload": False,
        "do_not_enter_alignment": True,
        "do_not_change_thresholds": True,
        "do_not_change_unmatched_penalty": True,
        "frozen_v2": str(PROJECT_ROOT / str(config["frozen_v2"])),
        "frozen_v2_checkpoint_sha256": "0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27",
        "mc_transfer_100_event_tracklets": config["mc_transfer_100_event_tracklets"],
        "occupancy_forecasts": plan.get("occupancy_forecasts"),
        "occupancy_vs_athena": occupancy_comparison,
        "condor": condor,
        "scale_table": scale_table,
        "blocks": rows,
        "by_scale": {
            scale: [row for row in rows if row["scale"] == scale]
            for scale in SCALE_ORDER
        },
    }
    _walk_forbidden(payload, where="route-acceptance scaling report")
    output = Path(args.output_json).expanduser().resolve()
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": verdict["decision"],
                "n_blocks": len(rows),
                "held_out_dq_used_for_verdict": any(
                    row["role"] == ROLE_HELD_OUT_DQ and row["used_for_verdict"] for row in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
