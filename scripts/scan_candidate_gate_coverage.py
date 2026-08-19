#!/usr/bin/env python3
"""Rescan the candidate chi2 gate under frozen covariance models.

For each named physical bank point, classify truth edges and fake candidates
exactly like the raw candidate-graph evaluation (same endpoint definitions,
same mode-0 records), but as a function of the Mahalanobis chi2 gate and of
the covariance model: the original exporter covariance or a frozen train-only
diagonal calibration.  Reports 0->1/1->2/2->3 truth-edge recall, complete
four-station truth-chain recall, propagation-failure and missing-covariance
counts (gate-independent), chi2-gate rejections, and fake-candidate growth.

All inputs are existing physical banks; nothing is retrained, retuned, or
repropagated.  Validation sources are only ever scanned read-only with
train-frozen calibrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.covariance_calibration import (
    CovarianceCalibration,
    apply_to_records,
    load_covariance_calibration,
)
from baselines.field_chi2_matching import _valid_covariance
from geometry.propagation import mahalanobis_chi2
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.audit_physical_route_candidate_graph import (
    ADJACENT_PAIRS,
    STATION_PATH,
    _unique_truth_indices,
)
from scripts.run_multisource_refit_multidof_local_step import _read_json

SCHEMA_VERSION = "faser-candidate-gate-coverage-scan-v1"


def _edge_chi2(
    event: Any,
    records: Any,
    rows_by_source: Mapping[tuple[int, int, int], list[int]],
    source_index: int,
    target_index: int,
) -> tuple[str, float | None]:
    """Outcome class and best truth-target chi2 for one edge."""
    source_tracklet = int(event.tracklet_id[source_index])
    target_tracklet = int(event.tracklet_id[target_index])
    rows = rows_by_source.get((int(event.run_id), int(event.event_id), source_tracklet), [])
    mode0 = [
        row
        for row in rows
        if (int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0) == 0
    ]
    if not mode0:
        return "no_propagation_record", None
    successful = [row for row in mode0 if bool(records.success[row])]
    if not successful:
        return "propagation_failed", None
    with_covariance = [row for row in successful if bool(records.has_covariance[row])]
    if not with_covariance:
        return "missing_covariance", None
    truth_rows = [
        row for row in with_covariance if int(records.target_tracklet_id[row]) == target_tracklet
    ]
    if not truth_rows:
        return "truth_target_not_recorded", None
    best = math.inf
    for row in truth_rows:
        propagated = np.asarray(records.covariance[row], dtype=np.float64)
        combined = propagated + np.asarray(event.covariance[target_index], dtype=np.float64)
        if not _valid_covariance(propagated) or not _valid_covariance(combined):
            return "invalid_covariance", None
        residual = np.asarray(event.state[target_index], dtype=np.float64) - np.asarray(
            records.prediction[row], dtype=np.float64
        )
        try:
            best = min(best, mahalanobis_chi2(residual, combined))
        except ValueError:
            return "chi2_exception", None
    return "evaluated", float(best)


def _scan_point(
    tracklets: Path,
    records: Any,
    gates: Sequence[float],
) -> dict[str, Any]:
    events = load_events(tracklets, require_mc_labels=True)
    rows_by_source: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in range(records.size):
        key = (int(records.run_id[row]), int(records.event_id[row]), int(records.source_tracklet_id[row]))
        rows_by_source[key].append(row)

    edge_outcomes: Counter[str] = Counter()
    edge_chi2: dict[tuple[int, int, int], float] = {}
    edge_counts: Counter[str] = Counter()
    fake_above_gate: Counter[tuple[str, float]] = Counter()
    fake_targets_evaluated: Counter[str] = Counter()
    chain_total = 0
    chain_edge_chi2: list[tuple[float, float, float]] = []

    for event in events:
        if event.truth_particle_id is None:
            continue
        per_pair_truth: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
        for source_station, target_station in ADJACENT_PAIRS:
            source_truth = _unique_truth_indices(event, source_station)
            target_truth = _unique_truth_indices(event, target_station)
            shared = sorted(set(source_truth) & set(target_truth))
            per_pair_truth[(source_station, target_station)] = {
                truth: (source_truth[truth], target_truth[truth]) for truth in shared
            }
        for pair in ADJACENT_PAIRS:
            pair_label = f"{pair[0]}->{pair[1]}"
            for truth, (source_index, target_index) in per_pair_truth[pair].items():
                edge_counts[pair_label] += 1
                outcome, chi2 = _edge_chi2(
                    event, records, rows_by_source, source_index, target_index
                )
                if outcome != "evaluated":
                    edge_outcomes[f"{pair_label}:{outcome}"] += 1
                else:
                    edge_chi2[(int(event.run_id), int(event.event_id), pair[0], truth)] = chi2
        # Fake growth: every recorded mode-0 source propagation fanned out to
        # all same-event target-station tracklets with mismatched truth.
        for pair in ADJACENT_PAIRS:
            pair_label = f"{pair[0]}->{pair[1]}"
            target_indices = event.indices_for_station(pair[1])
            if not len(target_indices):
                continue
            truth_by_target = {
                int(row): int(event.truth_particle_id[row]) for row in target_indices
            }
            for truth, (source_index, _target_index) in per_pair_truth[pair].items():
                source_tracklet = int(event.tracklet_id[source_index])
                rows = [
                    row
                    for row in rows_by_source.get(
                        (int(event.run_id), int(event.event_id), source_tracklet), []
                    )
                    if bool(records.success[row])
                    and bool(records.has_covariance[row])
                    and int(records.target_station_id[row]) == pair[1]
                ]
                if not rows:
                    continue
                row = rows[0]
                propagated = np.asarray(records.covariance[row], dtype=np.float64)
                if not _valid_covariance(propagated):
                    continue
                prediction = np.asarray(records.prediction[row], dtype=np.float64)
                for target_row, target_truth in truth_by_target.items():
                    if target_truth == truth:
                        continue
                    combined = propagated + np.asarray(event.covariance[target_row], dtype=np.float64)
                    if not _valid_covariance(combined):
                        continue
                    residual = np.asarray(event.state[target_row], dtype=np.float64) - prediction
                    try:
                        chi2 = mahalanobis_chi2(residual, combined)
                    except ValueError:
                        continue
                    fake_targets_evaluated[pair_label] += 1
                    for gate in gates:
                        if chi2 <= gate:
                            fake_above_gate[(pair_label, gate)] += 1
        # Complete chains: truth unique at all four stations.
        station_truth = {
            station: _unique_truth_indices(event, station) for station in STATION_PATH
        }
        common = set(station_truth[STATION_PATH[0]])
        for station in STATION_PATH[1:]:
            common &= set(station_truth[station])
        for truth in sorted(common):
            chain_total += 1
            chain_chi2 = []
            for pair in ADJACENT_PAIRS:
                value = edge_chi2.get((int(event.run_id), int(event.event_id), pair[0], truth))
                chain_chi2.append(value)
            chain_edge_chi2.append(tuple(chain_chi2))

    gates = list(gates)
    edge_recall: dict[str, dict[str, float | None]] = {}
    for pair in ADJACENT_PAIRS:
        pair_label = f"{pair[0]}->{pair[1]}"
        total = edge_counts[pair_label]
        per_gate = {}
        for gate in gates:
            retained = sum(
                1
                for (_run, _event, source_station, _truth), chi2 in edge_chi2.items()
                if source_station == pair[0] and chi2 <= gate
            )
            per_gate[str(gate)] = None if not total else retained / total
        edge_recall[pair_label] = {
            "truth_edges": total,
            "recall_by_gate": per_gate,
        }
    chain_recall = {}
    for gate in gates:
        retained = sum(
            1
            for chain in chain_edge_chi2
            if all(value is not None and value <= gate for value in chain)
        )
        chain_recall[str(gate)] = None if not chain_total else retained / chain_total
    return {
        "truth_edges_by_pair": {f"{p[0]}->{p[1]}": edge_counts[f"{p[0]}->{p[1]}"] for p in ADJACENT_PAIRS},
        "edge_outcomes": dict(edge_outcomes),
        "edge_recall": edge_recall,
        "complete_truth_chains": chain_total,
        "chain_recall_by_gate": chain_recall,
        "fake_targets_evaluated_by_pair": dict(fake_targets_evaluated),
        "fake_candidates_above_gate": {
            f"{pair}:{gate}": count for (pair, gate), count in sorted(fake_above_gate.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        metavar="MANIFEST:POINT:LABEL",
    )
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--gate", action="append", type=float, required=True)
    parser.add_argument("--covariance-calibration", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    gates = sorted(float(gate) for gate in args.gate)

    calibrations: dict[str, CovarianceCalibration | None] = {"original": None}
    for path in args.covariance_calibration or ():
        calibrations[Path(path).stem] = load_covariance_calibration(path)

    rows: list[dict[str, Any]] = []
    for spec in args.bank:
        manifest_path, _, rest = spec.partition(":")
        point_name, _, label = rest.partition(":")
        if not point_name or not label:
            raise ValueError("--bank must be MANIFEST:POINT:LABEL")
        manifest = _read_json(Path(manifest_path).expanduser().resolve())
        for entry in manifest["sources"]:
            source_id = str(entry["source_id"])
            selected = str(entry["split"]) == args.split or (
                args.source_id is not None and source_id in set(args.source_id)
            )
            if not selected:
                continue
            root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
            plan = _read_json(root / "scan_plan.json")
            points = {str(point["name"]): point for point in plan["points"]}
            point = points.get(point_name)
            if point is None:
                raise ValueError(f"point '{point_name}' absent from {root}/scan_plan.json")
            point_root = root / str(point["relative_point_dir"])
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            if not (tracklets.is_file() and propagations.is_file()):
                raise FileNotFoundError(f"{source_id}/{point_name} refit products missing")
            raw_records = load_propagation_records(propagations)
            for model_name, calibration in calibrations.items():
                records = (
                    raw_records
                    if calibration is None
                    else apply_to_records(raw_records, calibration)
                )
                result = _scan_point(tracklets, records, gates)
                rows.append(
                    {
                        "bank": label,
                        "point": point_name,
                        "source_id": source_id,
                        "split": str(entry["split"]),
                        "covariance_model": model_name,
                        **result,
                    }
                )

    json_path = output / "gate_coverage_scan.json"
    json_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "gates": gates,
        "covariance_models": sorted(calibrations),
        "test_data_accessed": False,
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat: list[dict[str, Any]] = []
    for row in rows:
        for pair_label, pair_data in row["edge_recall"].items():
            for gate, recall in pair_data["recall_by_gate"].items():
                flat.append(
                    {
                        "bank": row["bank"],
                        "source_id": row["source_id"],
                        "split": row["split"],
                        "covariance_model": row["covariance_model"],
                        "station_pair": pair_label,
                        "gate": gate,
                        "truth_edges": pair_data["truth_edges"],
                        "truth_edge_recall": recall,
                        "complete_chain_recall": row["chain_recall_by_gate"][gate],
                        "fake_candidates_above_gate": row["fake_candidates_above_gate"].get(
                            f"{pair_label}:{gate}", 0
                        ),
                    }
                )
    with (output / "gate_coverage_scan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"output_dir": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
