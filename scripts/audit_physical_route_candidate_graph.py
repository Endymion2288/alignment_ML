#!/usr/bin/env python3
"""Audit raw mode-0 physical candidate and truth-route retention.

This read-only audit consumes one physically refitted tracklet/Acts pair.  It
does not create shifted coordinates, synthetic candidates, scores, or
assignments.  Truth labels are used only after candidate construction to ask
whether the actual adjacent IFT -> S1 -> S2 -> S3 edges remain available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from baselines.field_chi2_matching import FieldCandidate, build_field_candidates
from datasets.propagation_loader import PropagationRecords, load_propagation_records
from datasets.root_loader import EventTracklets, load_events


STATION_PATH = (0, 1, 2, 3)
ADJACENT_PAIRS = tuple(zip(STATION_PATH[:-1], STATION_PATH[1:]))


def _unique_truth_indices(event: EventTracklets, station: int) -> dict[int, int]:
    if event.truth_particle_id is None:
        raise ValueError("physical candidate-route audit requires MC truth labels")
    indices = event.indices_for_station(station)
    labels = [int(event.truth_particle_id[index]) for index in indices]
    counts = Counter(label for label in labels if label >= 0)
    return {
        label: int(index)
        for index, label in zip(indices, labels)
        if label >= 0 and counts[label] == 1
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _edge_set(candidates: Sequence[FieldCandidate]) -> set[tuple[int, int]]:
    return {(int(candidate.source_index), int(candidate.target_index)) for candidate in candidates}


def audit_candidate_graph(
    events: Sequence[EventTracklets],
    records: PropagationRecords,
    *,
    chi2_gate: float | None,
) -> dict[str, object]:
    """Return raw physical truth-edge and complete-chain retention metrics."""
    pair_denominator: Counter[tuple[int, int]] = Counter()
    pair_retained: Counter[tuple[int, int]] = Counter()
    pair_candidates: Counter[tuple[int, int]] = Counter()
    chain_denominator = 0
    chain_retained = 0
    no_truth_events = 0
    event_rows: list[dict[str, object]] = []
    for event in events:
        if event.truth_particle_id is None:
            no_truth_events += 1
            continue
        unique = {station: _unique_truth_indices(event, station) for station in STATION_PATH}
        edge_sets: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for pair in ADJACENT_PAIRS:
            candidates = build_field_candidates(
                event,
                records,
                source_station=pair[0],
                target_station=pair[1],
                chi2_gate=chi2_gate,
                q_over_p_mode=0,
            )
            edge_sets[pair] = _edge_set(candidates)
            pair_candidates[pair] += len(candidates)
            for truth in set(unique[pair[0]]).intersection(unique[pair[1]]):
                pair_denominator[pair] += 1
                pair_retained[pair] += int(
                    (unique[pair[0]][truth], unique[pair[1]][truth]) in edge_sets[pair]
                )
        common_truth = set.intersection(*(set(unique[station]) for station in STATION_PATH))
        event_chain_total = len(common_truth)
        event_chain_retained = 0
        for truth in common_truth:
            retained = all(
                (unique[source][truth], unique[target][truth]) in edge_sets[(source, target)]
                for source, target in ADJACENT_PAIRS
            )
            event_chain_retained += int(retained)
        chain_denominator += event_chain_total
        chain_retained += event_chain_retained
        event_rows.append(
            {
                "run_id": int(event.run_id),
                "event_id": int(event.event_id),
                "complete_truth_chains": event_chain_total,
                "candidate_retained_complete_truth_chains": event_chain_retained,
            }
        )
    return {
        "events": len(events),
        "events_without_truth": no_truth_events,
        "station_path": list(STATION_PATH),
        "adjacent_station_pairs": [f"{source}->{target}" for source, target in ADJACENT_PAIRS],
        "candidate_chi2_gate": chi2_gate,
        "by_station_pair": {
            f"{source}->{target}": {
                "truth_pairs_with_unique_endpoints": int(pair_denominator[(source, target)]),
                "candidate_retained_truth_pairs": int(pair_retained[(source, target)]),
                "candidate_truth_edge_recall": _ratio(
                    int(pair_retained[(source, target)]), int(pair_denominator[(source, target)])
                ),
                "physical_candidate_edges": int(pair_candidates[(source, target)]),
            }
            for source, target in ADJACENT_PAIRS
        },
        "complete_truth_chains": chain_denominator,
        "candidate_retained_complete_truth_chains": chain_retained,
        "candidate_complete_truth_chain_recall": _ratio(chain_retained, chain_denominator),
        "by_event": event_rows,
    }


def _acts_surface_summary(records: PropagationRecords) -> dict[str, object]:
    """Summarize actual exporter response before candidate eligibility cuts."""
    by_pair: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for row in range(records.size):
        if int(records.q_over_p_mode[row]) != 0:
            continue
        pair = (int(records.source_station_id[row]), int(records.target_station_id[row]))
        counts = by_pair[pair]
        counts["records"] += 1
        counts["success"] += int(bool(records.success[row]))
        counts["has_covariance"] += int(bool(records.has_covariance[row]))
        counts["successful_with_covariance"] += int(
            bool(records.success[row]) and bool(records.has_covariance[row])
        )
        counts["finite_prediction"] += int(np.isfinite(records.prediction[row]).all())
        counts["finite_covariance"] += int(np.isfinite(records.covariance[row]).all())
    return {
        f"{source}->{target}": {name: int(value) for name, value in sorted(counts.items())}
        for (source, target), counts in sorted(by_pair.items())
    }


def _condition_chain_checks(log_path: Path | None) -> dict[str, object] | None:
    if log_path is None:
        return None
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    markers = {
        "sqlite_override": "Reading folder /Tracker/Align from sqlite",
        "sct_alignment_store": "recorded new CDO SCTAlignmentStore",
        "acts_alignment_context": "Recorded new FaserActsAlignment",
        "job_success": "Execution succeeded",
    }
    return {
        "log": str(log_path),
        "checks": {name: marker in log for name, marker in markers.items()},
        # These are deliberately counts rather than a Boolean job verdict.
        # A broad tracking geometry can log unrelated failed surface attempts
        # while the exact mode-0 truth edge used by the candidate graph still
        # has a finite state and covariance.  Report both layers separately.
        "acts_layer_overlap_error_count": int(log.count("Layers are overlapping")),
        "acts_surface_error_count": int(log.count("SurfaceError:")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--refit-log", default=None)
    parser.add_argument("--chi2-gate", type=float, default=None)
    args = parser.parse_args()
    if args.chi2_gate is not None and (
        not np.isfinite(args.chi2_gate) or args.chi2_gate <= 0.0
    ):
        parser.error("--chi2-gate must be finite and positive")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = load_station_rigid_alignment_payload(args.payload_manifest)
    events = load_events(args.tracklets, require_mc_labels=True)
    records = load_propagation_records(args.propagations)
    report = {
        "method": "read_only_physical_mode0_candidate_route_audit",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "payload_manifest": str(payload.manifest_path),
        "station_transforms": {
            str(station): list(payload.transform_for_station(station)) for station in STATION_PATH
        },
        "acts_surface_response": _acts_surface_summary(records),
        "condition_chain": _condition_chain_checks(
            None if args.refit_log is None else Path(args.refit_log).expanduser().resolve()
        ),
        "candidate_graph": audit_candidate_graph(events, records, chi2_gate=args.chi2_gate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["candidate_graph"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
