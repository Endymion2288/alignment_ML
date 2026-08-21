#!/usr/bin/env python3
"""Association-only diagnostics for frozen V2 on a four-station identity bank.

Reads one ``run_frozen_association_backbone.py`` output plus the identity
manifest.  Thresholds, calibration, and unmatched penalty are not touched.
The sealed test split is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignment.four_station import STATION_IDS, station_pair_ids
from baselines.field_chi2_matching import build_field_candidates
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.audit_physical_route_candidate_graph import _edge_set, _ratio, _unique_truth_indices
from scripts.run_refit_multidof_closure import _json_ready, _read_json


ADJACENT = ((0, 1), (1, 2), (2, 3))


def _physical_edge_key(row: Mapping[str, str]) -> tuple[object, ...]:
    return (
        int(row["source_origin_run_id"]),
        int(row["source_origin_event_id"]),
        int(row["source_origin_tracklet_id"]),
        int(row["target_origin_run_id"]),
        int(row["target_origin_event_id"]),
        int(row["target_origin_tracklet_id"]),
        int(row["source_station_id"]),
        int(row["target_station_id"]),
    )


def _selected_edge_audit(path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"selected-edge CSV has no header: {path}")
        rows = [dict(row) for row in reader if row.get("source_station_id")]
    pair_counts: Counter[str] = Counter()
    station_counts: Counter[int] = Counter()
    physical_keys = []
    for row in rows:
        source = int(row["source_station_id"])
        target = int(row["target_station_id"])
        pair_counts[f"{source}_{target}"] += 1
        station_counts[source] += 1
        station_counts[target] += 1
        physical_keys.append(_physical_edge_key(row))
    return {
        "selected_field_edges": len(rows),
        "unique_physical_edges": len(set(physical_keys)),
        "station_pair_composition": dict(sorted(pair_counts.items())),
        "station_participation_edge_endpoints": {str(station): int(station_counts[station]) for station in STATION_IDS},
        "includes_adjacent_1_2": pair_counts.get("1_2", 0) > 0,
        "includes_adjacent_2_3": pair_counts.get("2_3", 0) > 0,
        "includes_adjacent_0_1": pair_counts.get("0_1", 0) > 0,
    }


def _raw_candidate_audit(samples) -> dict[str, object]:
    pair_denominator: Counter[tuple[int, int]] = Counter()
    pair_retained: Counter[tuple[int, int]] = Counter()
    pair_candidates: Counter[tuple[int, int]] = Counter()
    chain_denominator = 0
    chain_retained = 0
    for sample in samples:
        events = load_events(sample.synthetic_tracklets, require_mc_labels=True)
        records = load_propagation_records(sample.field_candidates)
        for event in events:
            unique = {station: _unique_truth_indices(event, station) for station in STATION_IDS}
            edge_sets = {}
            for pair in station_pair_ids():
                candidates = build_field_candidates(
                    event,
                    records,
                    source_station=pair[0],
                    target_station=pair[1],
                    chi2_gate=None,
                    q_over_p_mode=0,
                )
                edge_sets[pair] = _edge_set(candidates)
                pair_candidates[pair] += len(candidates)
                for truth in set(unique[pair[0]]).intersection(unique[pair[1]]):
                    pair_denominator[pair] += 1
                    pair_retained[pair] += int(
                        (unique[pair[0]][truth], unique[pair[1]][truth]) in edge_sets[pair]
                    )
            common_truth = set.intersection(*(set(unique[station]) for station in STATION_IDS))
            chain_denominator += len(common_truth)
            for truth in common_truth:
                retained = all(
                    (unique[source][truth], unique[target][truth]) in edge_sets[(source, target)]
                    for source, target in ADJACENT
                )
                chain_retained += int(retained)
    return {
        "complete_truth_chains": chain_denominator,
        "candidate_retained_complete_truth_chains": chain_retained,
        "candidate_complete_truth_chain_recall": _ratio(chain_retained, chain_denominator),
        "by_station_pair": {
            f"{source}_{target}": {
                "truth_edges": int(pair_denominator[(source, target)]),
                "candidate_retained_truth_edges": int(pair_retained[(source, target)]),
                "truth_edge_recall": _ratio(
                    pair_retained[(source, target)], pair_denominator[(source, target)]
                ),
                "n_candidates": int(pair_candidates[(source, target)]),
                "adjacent": (source, target) in ADJACENT,
            }
            for source, target in station_pair_ids()
        },
    }


def summarize(
    *,
    association_dir: Path,
    synthetic_manifest: Path,
    payload_id: str,
) -> dict[str, object]:
    summary = _read_json(association_dir / "association_summary.json")
    if summary.get("test_opened") is not False or summary.get("architecture_or_threshold_tuning") is not False:
        raise ValueError("association output is not frozen from test/tuning")
    if int(summary.get("q_over_p_mode", -1)) != 0:
        raise ValueError("association output is not mode 0")
    _, samples, manifest = load_synthetic_curriculum_manifest(
        synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train",),
    )
    if manifest.get("q_over_p_mode") not in (0, None) and int(manifest.get("q_over_p_mode", 0)) != 0:
        raise ValueError("four-station association diagnostics require mode 0")
    overlay = bool(manifest.get("synthetic_multitrack")) or manifest.get("identity_physical_bank") is not True
    payload_samples = [sample for sample in samples if sample.payload_id == payload_id]
    if not payload_samples:
        raise ValueError(f"manifest has no payload '{payload_id}'")
    evaluation = summary.get("truth_labelled_mc_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("association summary lacks truth-labelled MC evaluation")
    route = dict(evaluation.get("route") or {})
    selected = _selected_edge_audit(association_dir / "selected_route_field_edge_residuals.csv")
    return {
        "payload_id": payload_id,
        "source_ids": sorted({sample.source_id for sample in payload_samples}),
        "n_events": int((evaluation.get("assignment") or {}).get("events") or 0),
        "raw_candidate_graph": _raw_candidate_audit(payload_samples),
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
        "selected_physical_edges": selected,
        "frozen_backbone": summary.get("frozen_backbone"),
        "route_solver": summary.get("route_solver"),
        "test_data_accessed": False,
        "architecture_or_threshold_tuning": False,
        "overlay": bool(overlay),
        "identity_physical_bank": manifest.get("identity_physical_bank") is True,
    }


def assess(reports: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, object]:
    nominal = reports["iteration_00_reference"]
    raw_ok = True
    assoc_ok = True
    rows = {}
    raw_chain_min = float(gates["raw_complete_truth_chain_recall_min"])
    raw_edge_min = float(gates["raw_adjacent_truth_edge_recall_min"])
    for name, report in reports.items():
        raw = report["raw_candidate_graph"]
        chain = raw["candidate_complete_truth_chain_recall"]
        adjacent = {
            pair: values["truth_edge_recall"]
            for pair, values in raw["by_station_pair"].items()
            if values["adjacent"]
        }
        payload_raw_ok = chain is not None and chain >= raw_chain_min and all(
            recall is not None and recall >= raw_edge_min for recall in adjacent.values()
        )
        raw_ok = raw_ok and payload_raw_ok
        route = report["selected_route"]
        vs_nominal = None
        payload_assoc_ok = True
        if name != "iteration_00_reference":
            def _drop(held, nom):
                if held is None or nom is None:
                    return None
                return float(nom) - float(held)

            def _increase(held, nom):
                if held is None or nom is None:
                    return None
                return float(held) - float(nom)

            vs_nominal = {
                "efficiency_drop": _drop(
                    route["complete_track_efficiency"],
                    nominal["selected_route"]["complete_track_efficiency"],
                ),
                "purity_drop": _drop(
                    route["complete_track_purity"],
                    nominal["selected_route"]["complete_track_purity"],
                ),
                "fake_rate_increase": _increase(
                    route["track_fake_rate"],
                    nominal["selected_route"]["track_fake_rate"],
                ),
            }
            payload_assoc_ok = all(value is not None for value in vs_nominal.values()) and (
                vs_nominal["efficiency_drop"] <= float(gates["vs_nominal_efficiency_drop_max"])
                and vs_nominal["purity_drop"] <= float(gates["vs_nominal_purity_drop_max"])
                and vs_nominal["fake_rate_increase"] <= float(gates["vs_nominal_fake_rate_increase_max"])
            )
            assoc_ok = assoc_ok and payload_assoc_ok
        selected = report["selected_physical_edges"]
        four_station_edges = bool(
            selected["includes_adjacent_0_1"]
            and selected["includes_adjacent_1_2"]
            and selected["includes_adjacent_2_3"]
        )
        rows[name] = {
            "raw_candidate_ok": payload_raw_ok,
            "association_vs_nominal_ok": payload_assoc_ok,
            "four_station_selected_edges": four_station_edges,
            "raw_complete_truth_chain_recall": chain,
            "adjacent_truth_edge_recall": adjacent,
            "vs_nominal": vs_nominal,
            "selected_route": route,
            "unique_physical_edges": selected["unique_physical_edges"],
        }
        assoc_ok = assoc_ok and four_station_edges
        if int(selected["selected_field_edges"] or 0) < 1:
            assoc_ok = False
            payload_assoc_ok = False
            rows[name]["association_vs_nominal_ok"] = False
            rows[name]["empty_selected_routes"] = True
    failure_class = None
    if not raw_ok:
        failure_class = "candidate_or_propagation"
    elif not assoc_ok:
        failure_class = "association_domain_shift"
    score_retained = {
        name: {
            "complete_truth_chains": report["selected_route"].get("complete_truth_chains"),
            "score_retained_complete_truth_chains": report["selected_route"].get(
                "score_retained_complete_truth_chains"
            ),
        }
        for name, report in reports.items()
    }
    return {
        "raw_candidate_graph_complete": raw_ok,
        "frozen_v2_association_stable_vs_nominal": assoc_ok,
        "continue_to_15d_relative_wls": bool(raw_ok and assoc_ok),
        "failure_class": failure_class,
        "payloads": rows,
        "gates": dict(gates),
        "score_threshold_retention": score_retained,
        "retrain_transformer": bool(raw_ok and not assoc_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association-dir", required=True)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--payload-id", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if args.split == "test" or args.payload_id == "test":
        raise SystemExit("refusing to open the sealed test split")
    report = summarize(
        association_dir=Path(args.association_dir).expanduser().resolve(),
        synthetic_manifest=Path(args.synthetic_manifest).expanduser().resolve(),
        payload_id=str(args.payload_id),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "payload_id": report["payload_id"],
                "raw_chain_recall": report["raw_candidate_graph"]["candidate_complete_truth_chain_recall"],
                "efficiency": report["selected_route"]["complete_track_efficiency"],
                "purity": report["selected_route"]["complete_track_purity"],
                "fake": report["selected_route"]["track_fake_rate"],
                "unique_physical_edges": report["selected_physical_edges"]["unique_physical_edges"],
                "keeps_1_2_and_2_3": report["selected_physical_edges"]["includes_adjacent_1_2"]
                and report["selected_physical_edges"]["includes_adjacent_2_3"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
