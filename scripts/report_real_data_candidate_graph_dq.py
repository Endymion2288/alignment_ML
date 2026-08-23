#!/usr/bin/env python3
"""Observable-only candidate-graph DQ for real-data frozen V2 inference.

Real data has no truth.  This report never emits efficiency, purity, fake,
AP, or AUC.  Frozen training-reference feature moments are a read-only OOD
control: they do not re-standardize, recalibrate, or change thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from datasets.root_loader import load_events
from training.curriculum_mlp import build_candidate_sets
from alignment.real_data_candidate_graph_dq import (
    assign_campaign_decision,
    evaluate_candidate_graph_dq_gate,
    load_candidate_graph_dq_gate,
)
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    geometric_edge_features,
    build_transformer_graph_bundle,
)
from training.route_aware_transformer import (
    load_route_aware_transformer_artifact,
    materialize_route_candidate_tables,
    predict_route_aware_scores,
)


FORBIDDEN_TRUTH_KEYS = (
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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _walk_forbidden(payload: object, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token == lowered or token in lowered.split("_") for token in FORBIDDEN_TRUTH_KEYS):
                raise ValueError(f"{where} contains forbidden truth-metric key {key!r}")
            _walk_forbidden(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            _walk_forbidden(item, where=where)


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("p16", "p50", "p84", "p05", "p95")}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in ("p16", "p50", "p84", "p05", "p95")}
    return {
        "p05": float(np.percentile(finite, 5)),
        "p16": float(np.percentile(finite, 16)),
        "p50": float(np.percentile(finite, 50)),
        "p84": float(np.percentile(finite, 84)),
        "p95": float(np.percentile(finite, 95)),
    }


def _robust_z(values: np.ndarray, mean: float, scale: float) -> dict[str, float | None]:
    if not math.isfinite(scale) or abs(scale) < 1.0e-12 or values.size == 0:
        return {"median_z": None, "p16_z": None, "p84_z": None, "frac_abs_z_gt_3": None, "frac_abs_z_gt_5": None}
    z = (values - mean) / scale
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return {"median_z": None, "p16_z": None, "p84_z": None, "frac_abs_z_gt_3": None, "frac_abs_z_gt_5": None}
    return {
        "median_z": float(np.median(finite)),
        "p16_z": float(np.percentile(finite, 16)),
        "p84_z": float(np.percentile(finite, 84)),
        "frac_abs_z_gt_3": float(np.mean(np.abs(finite) > 3.0)),
        "frac_abs_z_gt_5": float(np.mean(np.abs(finite) > 5.0)),
    }


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
            if not isinstance(payload, Mapping):
                raise ValueError(f"route row is not a mapping in {path}")
            rows.append(dict(payload))
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
            if any(str(value).strip() for value in row.values())
        ]


def _finite_floats(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{int(pair[0])}->{int(pair[1])}"


def _current_sample(manifest_samples: Sequence[Any], source_id: str) -> Any:
    samples = [sample for sample in manifest_samples if sample.source_id == source_id]
    if not samples:
        raise ValueError(f"no curriculum sample for {source_id}")
    current = [sample for sample in samples if sample.payload_id == "iteration_00_current"]
    if len(current) != 1:
        raise ValueError(f"{source_id} must have exactly one current-geometry sample")
    return current[0]


def _physical_graph_observables(events: Sequence[Any], candidate_sets: Sequence[Any]) -> dict[str, Any]:
    pair_counts = {_pair_key(pair): 0 for pair in ALL_STATION_PAIRS}
    physical_per_event: dict[tuple[int, int], int] = defaultdict(int)
    for candidate_set in candidate_sets:
        pair_counts[_pair_key(candidate_set.station_pair)] += int(len(candidate_set.candidates))
        physical_per_event[(int(candidate_set.event.run_id), int(candidate_set.event.event_id))] += int(
            len(candidate_set.candidates)
        )
    physical_total = float(sum(physical_per_event.values()))
    physical_shares = sorted(
        (float(count) / physical_total for count in physical_per_event.values()),
        reverse=True,
    ) if physical_total else []
    station_coverage = Counter(
        int(len({int(station) for station in event.station_id.tolist()})) for event in events
    )
    station_counts = Counter(int(station) for event in events for station in event.station_id.tolist())
    return {
        "n_events_with_tracklets": int(len(events)),
        "n_tracklets": int(sum(int(event.size) for event in events)),
        "tracklets_by_station": {str(station): int(station_counts[station]) for station in range(4)},
        "all_pairs_candidate_counts": pair_counts,
        "n_all_pairs_candidates": int(sum(pair_counts.values())),
        "n_events_with_two_station_tracklets": int(station_coverage.get(2, 0)),
        "n_events_with_three_station_tracklets": int(station_coverage.get(3, 0)),
        "n_events_with_four_station_tracklets": int(station_coverage.get(4, 0)),
        "physical_event_concentration": {
            "n_events_with_candidates": int(len(physical_per_event)),
            "max_event_share": physical_shares[0] if physical_shares else 0.0,
            "top2_event_share": float(sum(physical_shares[:2])),
            "per_event_histogram": dict(Counter(int(count) for count in physical_per_event.values())),
        },
        "physical_all_pairs_candidate_graph_nonempty": int(sum(pair_counts.values())) > 0,
    }


def _ood_residual_aliases(block: Mapping[str, Any]) -> dict[str, Any]:
    ood = block.get("ood_vs_frozen_training_edge_features") or {}
    return {
        "rx": ood.get("residual_x_mm"),
        "ry": ood.get("residual_y_mm"),
        "rtx": ood.get("residual_tx"),
        "rty": ood.get("residual_ty"),
        "pull_x": ood.get("pull_x"),
        "pull_y": ood.get("pull_y"),
        "chi2": {
            "real_percentiles": block.get("chi2_percentiles"),
            "do_not_restandardize": True,
        },
        "dz": ood.get("delta_z_mm"),
    }


def _source_dq(
    *,
    source: Mapping[str, Any],
    manifest_samples: Sequence[Any],
    association_dir: Path,
    model: object,
    artifact: object,
    calibration: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    sample = _current_sample(manifest_samples, source_id)
    events = load_events(sample.synthetic_tracklets, require_mc_labels=False)
    candidate_sets = build_candidate_sets(
        [sample],
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
        require_mc_labels=False,
    )
    physical = _physical_graph_observables(events, candidate_sets)
    pair_counts = physical["all_pairs_candidate_counts"]
    edge_features: list[np.ndarray] = []
    residuals = []
    pulls = []
    chi2s = []
    dzs = []
    finite = True
    for candidate_set in candidate_sets:
        for candidate in candidate_set.candidates:
            if not (
                np.isfinite(candidate.residual).all()
                and np.isfinite(candidate.pull).all()
                and math.isfinite(float(candidate.chi2))
            ):
                finite = False
                continue
            features = geometric_edge_features(candidate_set.event, candidate, reverse=False)
            edge_features.append(features)
            residuals.append(np.asarray(candidate.residual, dtype=np.float64))
            pulls.append(np.asarray(candidate.pull, dtype=np.float64))
            chi2s.append(float(candidate.chi2))
            dzs.append(float(features[-1]))
    edge_names = tuple(artifact.edge_feature_names)
    node_mean = np.asarray(artifact.node_standardizer.mean, dtype=np.float64)
    node_scale = np.asarray(artifact.node_standardizer.scale, dtype=np.float64)
    edge_mean = np.asarray(artifact.edge_standardizer.mean, dtype=np.float64)
    edge_scale = np.asarray(artifact.edge_standardizer.scale, dtype=np.float64)
    feature_matrix = np.asarray(edge_features, dtype=np.float64) if edge_features else np.empty((0, len(edge_names)))
    residual_matrix = np.asarray(residuals, dtype=np.float64) if residuals else np.empty((0, 4))
    pull_matrix = np.asarray(pulls, dtype=np.float64) if pulls else np.empty((0, 4))
    chi2_array = np.asarray(chi2s, dtype=np.float64)
    dz_array = np.asarray(dzs, dtype=np.float64)
    ood: dict[str, Any] = {}
    for index, name in enumerate(edge_names):
        column = feature_matrix[:, index] if feature_matrix.size else np.empty(0)
        ood[name] = {
            "real_percentiles": _percentiles(column),
            "training_mean": float(edge_mean[index]),
            "training_scale": float(edge_scale[index]),
            "read_only_z_vs_training": _robust_z(column, float(edge_mean[index]), float(edge_scale[index])),
            "do_not_restandardize": True,
        }
    node_ood = {
        "training_mean": [float(value) for value in node_mean],
        "training_scale": [float(value) for value in node_scale],
        "feature_names": list(artifact.node_feature_names),
        "note": "Node OOD uses the frozen V2 node standardizer as a read-only control.",
        "do_not_restandardize": True,
    }
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode=artifact.context_mode)
    prediction = predict_route_aware_scores(
        model,
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=device,
        require_mc_labels=False,
    )
    calibrated = apply_frozen_transformer_calibration(
        bundle.adjacent_sets, list(prediction.edge_scores), calibration
    )
    raw_scores = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in prediction.edge_scores if np.asarray(values).size]
    ) if any(np.asarray(values).size for values in prediction.edge_scores) else np.empty(0)
    cal_scores = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in calibrated if np.asarray(values).size]
    ) if any(np.asarray(values).size for values in calibrated) else np.empty(0)
    thresholds = {
        "0->1": 0.001,
        "1->2": 0.001,
        "2->3": 0.001,
    }
    summary_path = association_dir / "association_summary.json"
    association = _read_json(summary_path) if summary_path.is_file() else {}
    frozen_thresholds = ((association.get("route_solver") or {}).get("thresholds") or thresholds)
    n_above = 0
    if cal_scores.size:
        # Adjacent-pair calibrated scores only; the frozen per-pair threshold is 0.001.
        n_above = int(np.sum(np.isfinite(cal_scores) & (cal_scores >= min(float(v) for v in frozen_thresholds.values()))))
    route_tables = materialize_route_candidate_tables(bundle.graphs, require_mc_labels=False)
    n_complete_physical_route_candidates = int(sum(table.size for table in route_tables.values()))
    if association.get("truth_labelled_mc_evaluation", {}).get("omitted") is not True:
        raise ValueError(f"{source_id} V2 output must omit truth-labelled MC evaluation")
    routes = _load_routes(association_dir / "selected_routes.jsonl")
    edges = _load_csv_rows(association_dir / "selected_route_field_edge_residuals.csv")
    complete = [row for row in routes if bool(row.get("is_complete_four_station_route"))]
    per_event_routes: dict[tuple[int, int], int] = defaultdict(int)
    for row in routes:
        per_event_routes[(int(row["run_id"]), int(row["event_id"]))] += 1
    route_counts = np.asarray(list(per_event_routes.values()), dtype=np.int64) if per_event_routes else np.zeros(0, dtype=np.int64)
    max_share = 0.0
    if routes:
        max_share = float(max(per_event_routes.values())) / float(len(routes))
    endpoint_uses: Counter[tuple[int, int, int, int]] = Counter()
    for row in routes:
        for item in row.get("endpoint_provenance") or []:
            endpoint_uses[
                (
                    int(item["origin_run_id"]),
                    int(item["origin_event_id"]),
                    int(item["station_id"]),
                    int(item["origin_tracklet_id"]),
                )
            ] += 1
    reused = sum(1 for count in endpoint_uses.values() if count > 1)
    selected_pair_counts = Counter(
        f"{int(row['source_station_id'])}->{int(row['target_station_id'])}" for row in edges if row.get("source_station_id")
    )
    scores = _finite_floats(edges, "score")
    utilities = _finite_floats(routes, "utility")
    n_candidates = int(sum(pair_counts.values()))
    n_selected_edges = int(len(edges))
    n_selected_routes = int(len(routes))
    physical_nonempty = n_candidates > 0
    selected_nonempty = n_selected_edges > 0 and n_selected_routes > 0
    nonempty = physical_nonempty and selected_nonempty
    partial_routes = [
        row
        for row in routes
        if not bool(row.get("is_complete_four_station_route"))
    ]
    report = {
        "source_id": source_id,
        "run": int(source["run"]),
        "role": str(source["role"]),
        "split": str(source["split"]),
        "payload_id": "iteration_00_current",
        "association_output": str(association_dir),
        "n_events_with_tracklets": physical["n_events_with_tracklets"],
        "n_tracklets": physical["n_tracklets"],
        "tracklets_by_station": physical["tracklets_by_station"],
        "all_pairs_candidate_counts": pair_counts,
        "n_all_pairs_candidates": n_candidates,
        "selected_edges": n_selected_edges,
        "selected_edges_by_adjacent_pair": dict(selected_pair_counts),
        "selected_routes": n_selected_routes,
        "selected_partial_routes": int(len(partial_routes)),
        "complete_four_station_routes": int(len(complete)),
        "complete_four_station_coverage_events": int(
            len({(int(row["run_id"]), int(row["event_id"])) for row in complete})
        ),
        "events_with_any_selected_route": int(len(per_event_routes)),
        "n_events_with_two_station_tracklets": physical["n_events_with_two_station_tracklets"],
        "n_events_with_three_station_tracklets": physical["n_events_with_three_station_tracklets"],
        "n_events_with_four_station_tracklets": physical["n_events_with_four_station_tracklets"],
        "route_multiplicity": {
            "mean": None if route_counts.size == 0 else float(np.mean(route_counts)),
            "max": None if route_counts.size == 0 else int(np.max(route_counts)),
            "per_event_histogram": dict(Counter(int(value) for value in route_counts.tolist())),
        },
        "endpoint_reuse": {
            "unique_endpoints": int(len(endpoint_uses)),
            "reused_endpoints": int(reused),
            "max_uses": 0 if not endpoint_uses else int(max(endpoint_uses.values())),
        },
        "physical_event_concentration": physical["physical_event_concentration"],
        "per_event_route_distribution": {
            "max_event_share_of_selected_routes": max_share,
            "n_events_with_selected_routes": int(len(per_event_routes)),
        },
        "score_distribution": _percentiles(scores),
        "utility_distribution": _percentiles(utilities),
        "adjacent_raw_score_distribution": _percentiles(raw_scores),
        "adjacent_calibrated_score_distribution": _percentiles(cal_scores),
        "n_adjacent_candidates": int(cal_scores.size),
        "n_adjacent_candidates_above_frozen_threshold": n_above,
        "n_complete_physical_route_candidates": n_complete_physical_route_candidates,
        "frozen_route_policy_selected_none": n_selected_routes == 0,
        "do_not_change_unmatched_penalty": True,
        "residual_percentiles": {
            "x_mm": _percentiles(residual_matrix[:, 0] if residual_matrix.size else np.empty(0)),
            "y_mm": _percentiles(residual_matrix[:, 1] if residual_matrix.size else np.empty(0)),
            "tx": _percentiles(residual_matrix[:, 2] if residual_matrix.size else np.empty(0)),
            "ty": _percentiles(residual_matrix[:, 3] if residual_matrix.size else np.empty(0)),
        },
        "pull_percentiles": {
            "x": _percentiles(pull_matrix[:, 0] if pull_matrix.size else np.empty(0)),
            "y": _percentiles(pull_matrix[:, 1] if pull_matrix.size else np.empty(0)),
            "tx": _percentiles(pull_matrix[:, 2] if pull_matrix.size else np.empty(0)),
            "ty": _percentiles(pull_matrix[:, 3] if pull_matrix.size else np.empty(0)),
        },
        "chi2_percentiles": _percentiles(chi2_array),
        "dz_percentiles": _percentiles(dz_array),
        "ood_vs_frozen_training_edge_features": ood,
        "ood_residual_aliases": _ood_residual_aliases(
            {"ood_vs_frozen_training_edge_features": ood, "chi2_percentiles": _percentiles(chi2_array)}
        ),
        "ood_vs_frozen_training_node_features": node_ood,
        "values_finite": bool(
            finite
            and (not feature_matrix.size or np.isfinite(feature_matrix).all())
            and (not raw_scores.size or np.isfinite(raw_scores).all())
            and (not cal_scores.size or np.isfinite(cal_scores).all())
        ),
        "physical_all_pairs_candidate_graph_nonempty": physical_nonempty,
        "selected_v2_graph_nonempty": selected_nonempty,
        "candidate_graph_nonempty": nonempty,
        "truth_metrics_omitted": True,
        "do_not_restandardize": True,
        "do_not_recalibrate": True,
        "do_not_change_thresholds": True,
        "frozen_v2_summary": {
            "q_over_p_mode": association.get("q_over_p_mode"),
            "truth_labelled_mc_evaluation": association.get("truth_labelled_mc_evaluation"),
            "raw_candidate_score_metrics": association.get("raw_candidate_score_metrics"),
            "frozen_calibrated_score_metrics": association.get("frozen_calibrated_score_metrics"),
        },
    }
    _walk_forbidden(report, where=source_id)
    return report


def _enrich_existing_block(
    block: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    manifest_samples: Sequence[Any],
    association_dir: Path,
) -> dict[str, Any]:
    """Reuse frozen V2 outputs: add physical observables, do not rescore."""
    sample = _current_sample(manifest_samples, str(source["source_id"]))
    events = load_events(sample.synthetic_tracklets, require_mc_labels=False)
    candidate_sets = build_candidate_sets(
        [sample],
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=0,
        require_mc_labels=False,
    )
    physical = _physical_graph_observables(events, candidate_sets)
    routes = _load_routes(association_dir / "selected_routes.jsonl")
    edges = _load_csv_rows(association_dir / "selected_route_field_edge_residuals.csv")
    association = _read_json(association_dir / "association_summary.json")
    truth_free = association.get("truth_free_selection") or {}
    n_selected_routes = int(truth_free.get("selected_routes", len(routes)))
    n_selected_edges = int(truth_free.get("field_aware_edge_observations", len(edges)))
    complete = [row for row in routes if bool(row.get("is_complete_four_station_route"))]
    selected_nonempty = n_selected_edges > 0 and n_selected_routes > 0
    enriched = dict(block)
    enriched.update(physical)
    enriched["selected_edges"] = n_selected_edges
    enriched["selected_routes"] = n_selected_routes
    enriched["selected_partial_routes"] = int(
        sum(1 for row in routes if not bool(row.get("is_complete_four_station_route")))
    )
    enriched["complete_four_station_routes"] = int(
        truth_free.get("complete_four_station_routes", len(complete))
    )
    enriched["selected_v2_graph_nonempty"] = selected_nonempty
    enriched["candidate_graph_nonempty"] = bool(
        physical["physical_all_pairs_candidate_graph_nonempty"] and selected_nonempty
    )
    enriched["frozen_route_policy_selected_none"] = n_selected_routes == 0
    enriched["ood_residual_aliases"] = _ood_residual_aliases(enriched)
    return enriched


def _finalize_payload(
    *,
    blocks: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    association_root: Path,
    frozen_v2: Path,
    gate_path: str | None,
) -> dict[str, Any]:
    calibration = [block for block in blocks if block["role"] == "calibration"]
    if len(calibration) != 2:
        raise ValueError("expected exactly two calibration blocks")
    gate = load_candidate_graph_dq_gate(gate_path)
    station_mode_gate = evaluate_candidate_graph_dq_gate(blocks, gate)
    gate_passed = bool(station_mode_gate["passed"])
    decision = (
        None
        if gate_passed
        else assign_campaign_decision(
            candidate_graph_dq_failed=True,
            station_fit_failed=False,
            implied_cdx_exceeds_operating_band=False,
            independent_cdx_evidence=False,
            station_and_holdout_dq_ok=False,
        )
    )
    payload = {
        "schema_version": "faser-operating-protocol-v1-real-data-candidate-graph-dq",
        "synthetic_manifest": str(manifest_path),
        "association_root": str(association_root),
        "frozen_v2": str(frozen_v2),
        "truth_metrics_omitted": True,
        "do_not_restandardize": True,
        "do_not_recalibrate": True,
        "do_not_change_thresholds": True,
        "do_not_change_unmatched_penalty": True,
        "blocks": [dict(block) for block in blocks],
        "station_mode_gate": station_mode_gate,
        "decision": decision,
        "station_mode_started": False,
        "cdx_mode_allowed": False,
        "geometry_write_allowed": False,
    }
    _walk_forbidden(payload, where="candidate-graph DQ report")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--physical-output-dir", required=True)
    parser.add_argument("--association-root", required=True)
    parser.add_argument("--frozen-v2", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--dq-gate",
        default=None,
        help="Frozen candidate-graph DQ gate YAML.  Default is the V1 real-data gate.",
    )
    parser.add_argument(
        "--reuse-existing-dq",
        default=None,
        help=(
            "Reuse a previous observable DQ JSON and the frozen V2 association "
            "outputs.  Recomputes physical concentration from identity tracklets "
            "and re-evaluates the frozen gate.  Does not rescore or overwrite V2."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path, samples, _manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    physical = _read_json(Path(args.physical_output_dir).expanduser().resolve() / "iteration_manifest.json")
    association_root = Path(args.association_root).expanduser().resolve()
    frozen_v2 = Path(args.frozen_v2).expanduser().resolve()
    blocks: list[dict[str, Any]] = []
    if args.reuse_existing_dq:
        existing = _read_json(Path(args.reuse_existing_dq).expanduser().resolve())
        by_id = {str(block["source_id"]): dict(block) for block in existing.get("blocks") or []}
        for source in physical["sources"]:
            source_id = str(source["source_id"])
            if source_id not in by_id:
                raise ValueError(f"existing DQ lacks {source_id}")
            blocks.append(
                _enrich_existing_block(
                    by_id[source_id],
                    source=source,
                    manifest_samples=samples,
                    association_dir=association_root / source_id,
                )
            )
    else:
        model, artifact = load_route_aware_transformer_artifact(
            frozen_v2 / "route_aware_transformer_v2.pt",
            device=args.device,
        )
        calibration_wrapper = _read_json(frozen_v2 / "calibration.json")
        calibration = dict(calibration_wrapper["calibration"])
        for source in physical["sources"]:
            blocks.append(
                _source_dq(
                    source=source,
                    manifest_samples=samples,
                    association_dir=association_root / source["source_id"],
                    model=model,
                    artifact=artifact,
                    calibration=calibration,
                    device=args.device,
                )
            )
    payload = _finalize_payload(
        blocks=blocks,
        manifest_path=manifest_path,
        association_root=association_root,
        frozen_v2=frozen_v2,
        gate_path=args.dq_gate,
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, _json_ready(payload))
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": payload["decision"],
                "station_mode_gate": payload["station_mode_gate"]["passed"],
                "blocks": [
                    {
                        "source_id": block["source_id"],
                        "selected_edges": block["selected_edges"],
                        "selected_routes": block["selected_routes"],
                        "complete_four_station_routes": block["complete_four_station_routes"],
                        "candidate_graph_nonempty": block["candidate_graph_nonempty"],
                    }
                    for block in blocks
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
