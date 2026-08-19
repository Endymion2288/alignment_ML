#!/usr/bin/env python3
"""Decompose raw physical candidate truth-chain coverage by failure mechanism.

Read-only diagnosis over an existing multi-source iteration bank.  For every
(source, alignment point, adjacent station pair, truth edge) the tool assigns
exactly one outcome class, strictly separating:

- ``missing_endpoint_tracklet``: the truth particle has no unique reconstructed
  tracklet at one of the two stations (nothing the candidate builder can do);
- ``no_propagation_record``: no mode-0 record row exists for the source
  tracklet on this pair (the exporter/Acts never produced a candidate row);
- ``propagation_failed``: a mode-0 row exists but ``success`` is false;
- ``missing_covariance``: the row succeeded but has no propagated covariance;
- ``truth_target_not_recorded``: the propagation succeeded with covariance but
  the recorded target tracklet is not the truth target (export-level matching
  picked another tracklet or none);
- ``invalid_covariance`` / ``chi2_exception``: numerically unusable rows;
- ``chi2_gate_rejection``: the truth target is the recorded target but the
  Mahalanobis chi2 exceeds the candidate gate;
- ``retained``: the truth edge is in the candidate graph.

Complete truth chains are then decomposed by the first failing edge.  All
breakdowns are reported per station pair, source, alignment point and source
tracklet kinematic bin.  The candidate graph itself is never modified.
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

from baselines.field_chi2_matching import _valid_covariance
from geometry.propagation import mahalanobis_chi2
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.audit_physical_route_candidate_graph import ADJACENT_PAIRS, STATION_PATH
from scripts.run_multisource_refit_multidof_local_step import _read_json


SCHEMA_VERSION = "faser-candidate-coverage-decomposition-v1"
OUTCOME_CLASSES = (
    "retained",
    "missing_endpoint_tracklet",
    "no_propagation_record",
    "propagation_failed",
    "missing_covariance",
    "truth_target_not_recorded",
    "invalid_covariance",
    "chi2_exception",
    "chi2_gate_rejection",
)


def _unique_truth_indices(event: Any, station: int) -> dict[int, int]:
    from scripts.audit_physical_route_candidate_graph import _unique_truth_indices as _impl

    return _impl(event, station)


def _classify_truth_edge(
    event: Any,
    records: Any,
    rows_by_source: Mapping[int, list[int]],
    source_station: int,
    target_station: int,
    source_index: int,
    target_index: int,
    chi2_gate: float,
) -> str:
    """Assign one outcome class to a truth edge with both endpoints present."""
    source_tracklet = int(event.tracklet_id[source_index])
    target_tracklet = int(event.tracklet_id[target_index])
    rows = rows_by_source.get(source_tracklet, [])
    if not rows:
        return "no_propagation_record"
    mode0_rows = [
        row
        for row in rows
        if (int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0) == 0
    ]
    if not mode0_rows:
        return "no_propagation_record"
    successful = [row for row in mode0_rows if bool(records.success[row])]
    if not successful:
        return "propagation_failed"
    with_covariance = [row for row in successful if bool(records.has_covariance[row])]
    if not with_covariance:
        return "missing_covariance"
    truth_rows = [
        row for row in with_covariance if int(records.target_tracklet_id[row]) == target_tracklet
    ]
    if not truth_rows:
        return "truth_target_not_recorded"
    best_chi2 = math.inf
    for row in truth_rows:
        propagated_covariance = records.covariance[row]
        combined_covariance = propagated_covariance + event.covariance[target_index]
        if not _valid_covariance(propagated_covariance) or not _valid_covariance(combined_covariance):
            return "invalid_covariance"
        residual = event.state[target_index] - records.prediction[row]
        try:
            best_chi2 = min(best_chi2, mahalanobis_chi2(residual, combined_covariance))
        except ValueError:
            return "chi2_exception"
    return "retained" if best_chi2 <= chi2_gate else "chi2_gate_rejection"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-chi2-gate", type=float, default=25.0)
    parser.add_argument("--max-points-per-source", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty coverage diagnosis directory")
    output_root.mkdir(parents=True, exist_ok=False)
    gate = float(args.candidate_chi2_gate)

    edge_counts: Counter[tuple[str, str, str, str]] = Counter()
    edge_kinematics: dict[tuple[str, str], list[tuple[float, float, float, float, str]]] = (
        defaultdict(list)
    )
    chain_counts: Counter[tuple[str, str, str]] = Counter()
    per_point_pair: Counter[tuple[str, str, str]] = Counter()
    per_point_pair_retained: Counter[tuple[str, str, str]] = Counter()

    for entry in manifest["sources"]:
        source_id = str(entry["source_id"])
        split = str(entry["split"])
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        points = plan["points"]
        if args.max_points_per_source is not None:
            points = points[: int(args.max_points_per_source)]
        for point in points:
            point_name = str(point["name"])
            point_root = root / str(point["relative_point_dir"])
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            if not (tracklets.is_file() and propagations.is_file()):
                continue
            events = load_events(tracklets, require_mc_labels=True)
            records = load_propagation_records(propagations)
            rows_by_event_pair: dict[tuple[int, int, int], dict[int, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for row in range(records.size):
                key = (
                    int(records.event_id[row]),
                    int(records.source_station_id[row]),
                    int(records.target_station_id[row]),
                )
                rows_by_event_pair[key][int(records.source_tracklet_id[row])].append(row)
            for event in events:
                unique = {station: _unique_truth_indices(event, station) for station in STATION_PATH}
                pair_outcome: dict[tuple[int, int], dict[int, str]] = {}
                for source_station, target_station in ADJACENT_PAIRS:
                    pair = f"{source_station}->{target_station}"
                    rows_by_source = rows_by_event_pair.get(
                        (event.event_id, source_station, target_station), {}
                    )
                    outcomes: dict[int, str] = {}
                    truths = set(unique[source_station]).intersection(unique[target_station])
                    for truth in truths:
                        source_index = unique[source_station][truth]
                        target_index = unique[target_station][truth]
                        outcome = _classify_truth_edge(
                            event,
                            records,
                            rows_by_source,
                            source_station,
                            target_station,
                            source_index,
                            target_index,
                            gate,
                        )
                        outcomes[truth] = outcome
                        edge_counts[(split, source_id, point_name, pair, outcome)] += 1
                        per_point_pair[(split, point_name, pair)] += 1
                        per_point_pair_retained[(split, point_name, pair)] += int(
                            outcome == "retained"
                        )
                        state = event.state[source_index]
                        edge_kinematics[(pair, outcome)].append(
                            (
                                float(state[0]),
                                float(state[1]),
                                float(state[2]),
                                float(state[3]),
                                source_id,
                            )
                        )
                    pair_outcome[(source_station, target_station)] = outcomes
                common_truth = set.intersection(*(set(unique[s]) for s in STATION_PATH))
                for truth in common_truth:
                    failing = "retained"
                    for source_station, target_station in ADJACENT_PAIRS:
                        outcome = pair_outcome[(source_station, target_station)].get(truth)
                        if outcome != "retained":
                            failing = f"{source_station}->{target_station}:{outcome}"
                            break
                    chain_counts[(split, point_name, failing)] += 1

    # ---- aggregate outputs -------------------------------------------------
    edge_rows = []
    for (split, source_id, point_name, pair, outcome), count in sorted(edge_counts.items()):
        edge_rows.append(
            {
                "split": split,
                "source_id": source_id,
                "point": point_name,
                "station_pair": pair,
                "outcome": outcome,
                "count": count,
            }
        )
    with (output_root / "edge_outcomes_by_source_point_pair.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source_id", "point", "station_pair", "outcome", "count"],
        )
        writer.writeheader()
        writer.writerows(edge_rows)

    pair_summary: dict[str, Any] = {}
    for split, point_name, pair in sorted(per_point_pair):
        total = per_point_pair[(split, point_name, pair)]
        retained = per_point_pair_retained[(split, point_name, pair)]
        key = f"{split}:{point_name}:{pair}"
        outcomes: Counter[str] = Counter()
        for (s, _src, p, pr, outcome), count in edge_counts.items():
            if s == split and p == point_name and pr == pair:
                outcomes[outcome] += count
        pair_summary[key] = {
            "truth_edges": total,
            "retained": retained,
            "recall": (None if total == 0 else retained / total),
            "outcome_shares": {
                outcome: (None if total == 0 else count / total)
                for outcome, count in sorted(outcomes.items())
            },
        }

    kinematic_summary: dict[str, Any] = {}
    tx_edges = (0.05, 0.1, 0.15)
    ty_edges = (0.03, 0.06, 0.09)
    for (pair, outcome), rows in sorted(edge_kinematics.items()):
        tx = np.asarray([row[2] for row in rows], dtype=np.float64)
        ty = np.asarray([row[3] for row in rows], dtype=np.float64)
        kinematic_summary[f"{pair}:{outcome}"] = {
            "count": int(tx.size),
            "mean_abs_tx": (None if tx.size == 0 else float(np.abs(tx).mean())),
            "mean_abs_ty": (None if ty.size == 0 else float(np.abs(ty).mean())),
        }

    chain_rows = []
    for (split, point_name, failing), count in sorted(chain_counts.items()):
        chain_rows.append({"split": split, "point": point_name, "first_failing_edge": failing, "count": count})
    with (output_root / "chain_first_failure_by_point.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "point", "first_failing_edge", "count"]
        )
        writer.writeheader()
        writer.writerows(chain_rows)

    report = {
        "schema_version": SCHEMA_VERSION,
        "iteration_manifest": str(manifest_path),
        "candidate_chi2_gate": gate,
        "outcome_classes": list(OUTCOME_CLASSES),
        "pair_summary_by_point": pair_summary,
        "kinematic_summary_by_pair_outcome": kinematic_summary,
        "chain_first_failure": chain_rows,
    }
    (output_root / "coverage_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(output_root, pair_summary)
    print(
        json.dumps(
            {"output_dir": str(output_root), "pair_points": len(pair_summary)}, indent=2
        )
    )


def _plot(output_root: Path, pair_summary: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [key for key in pair_summary if key.startswith("train:")]
    if not keys:
        return
    points = sorted({key.split(":")[1] for key in keys})
    pairs = sorted({key.split(":")[2] for key in keys})
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for pair in pairs:
        recalls = [
            pair_summary.get(f"train:{point}:{pair}", {}).get("recall") for point in points
        ]
        axis.plot(points, recalls, "o-", label=pair, markersize=4)
    axis.set_ylabel("truth edge recall (train)")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(range(len(points)))
    axis.set_xticklabels(points, rotation=45, ha="right", fontsize=7)
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "edge_recall_by_point.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
