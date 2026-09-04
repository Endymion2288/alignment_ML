#!/usr/bin/env python3
"""Audit canonical FASER tracklet content before association training.

The command deliberately reports observed station IDs and z positions without
using z to infer a station label. Candidate diagnostics are optional because a
straight-line model is only a temporary V1 baseline, especially across the
spectrometer field.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from baselines.chi2_matching import build_candidates
from datasets.root_loader import EventTracklets, load_events


def _finite_stats(values: Iterable[float] | np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _candidate_summary(
    events: Iterable[EventTracklets],
    source_station: int,
    target_station: int,
    chi2_gate: float,
) -> dict[str, object]:
    all_chi2: list[float] = []
    same_truth_chi2: list[float] = []
    other_truth_chi2: list[float] = []
    for event in events:
        candidates = build_candidates(
            event,
            source_station=source_station,
            target_station=target_station,
            chi2_gate=chi2_gate,
        )
        for candidate in candidates:
            all_chi2.append(candidate.chi2)
            if event.truth_particle_id is None:
                continue
            source_truth = int(event.truth_particle_id[candidate.source_index])
            target_truth = int(event.truth_particle_id[candidate.target_index])
            if source_truth >= 0 and source_truth == target_truth:
                same_truth_chi2.append(candidate.chi2)
            else:
                other_truth_chi2.append(candidate.chi2)
    return {
        "source_station": source_station,
        "target_station": target_station,
        "chi2_gate": chi2_gate,
        "candidate_pairs": len(all_chi2),
        "chi2": _finite_stats(all_chi2),
        "same_truth_id_chi2": _finite_stats(same_truth_chi2),
        "different_or_unknown_truth_chi2": _finite_stats(other_truth_chi2),
    }


def summarize_events(
    events: list[EventTracklets],
    input_path: str | Path,
    source_station: int | None = None,
    target_station: int | None = None,
    chi2_gate: float = 1.0e6,
) -> dict[str, object]:
    """Create a JSON-safe content summary from canonical event groups."""
    station_counts: Counter[int] = Counter()
    station_event_counts: Counter[int] = Counter()
    station_z: defaultdict[int, list[float]] = defaultdict(list)
    tracklets_per_event: list[float] = []
    n_hit: list[float] = []
    local_chi2: list[float] = []
    local_ndof: list[float] = []
    covariances: list[np.ndarray] = []
    truth_particle_ids: list[int] = []
    truth_pdgs: list[int] = []
    truth_match_fractions: list[float] = []

    for event in events:
        tracklets_per_event.append(float(event.size))
        unique_stations, counts = np.unique(event.station_id, return_counts=True)
        for station, count in zip(unique_stations, counts):
            station_id = int(station)
            station_counts[station_id] += int(count)
            station_event_counts[station_id] += 1
        for station, z_mm in zip(event.station_id, event.z_mm):
            station_z[int(station)].append(float(z_mm))
        n_hit.extend(event.n_hit.astype(np.float64))
        local_chi2.extend(event.chi2)
        local_ndof.extend(event.ndof)
        covariances.append(event.covariance)
        if event.truth_particle_id is not None:
            truth_particle_ids.extend(map(int, event.truth_particle_id))
            truth_pdgs.extend(map(int, event.truth_pdg))
            truth_match_fractions.extend(event.truth_match_fraction)

    if covariances:
        covariance_array = np.concatenate(covariances, axis=0)
        eigenvalues = np.linalg.eigvalsh(covariance_array)
        covariance_summary: dict[str, object] = {
            "rows": int(covariance_array.shape[0]),
            "positive_definite_rows": int(np.count_nonzero(np.all(eigenvalues > 0.0, axis=1))),
            "minimum_eigenvalue": _finite_stats(eigenvalues[:, 0]),
        }
    else:
        covariance_summary = {
            "rows": 0,
            "positive_definite_rows": 0,
            "minimum_eigenvalue": _finite_stats([]),
        }

    has_mc_labels = bool(events and events[0].truth_particle_id is not None)
    summary: dict[str, object] = {
        "input": str(Path(input_path).expanduser().resolve()),
        "events": len(events),
        "tracklets": int(sum(event.size for event in events)),
        "tracklets_per_event": _finite_stats(tracklets_per_event),
        "station_counts": {str(station): count for station, count in sorted(station_counts.items())},
        "events_with_station": {
            str(station): count for station, count in sorted(station_event_counts.items())
        },
        "z_mm_by_station": {
            str(station): _finite_stats(values) for station, values in sorted(station_z.items())
        },
        "n_hit": _finite_stats(n_hit),
        "local_chi2": _finite_stats(local_chi2),
        "local_ndof": _finite_stats(local_ndof),
        "covariance": covariance_summary,
        "has_mc_labels": has_mc_labels,
    }
    if has_mc_labels:
        labels = np.asarray(truth_particle_ids, dtype=np.int64)
        summary["truth"] = {
            "tracklets_with_known_particle_id": int(np.count_nonzero(labels >= 0)),
            "tracklets_with_unknown_particle_id": int(np.count_nonzero(labels < 0)),
            "truth_particle_id_counts": {
                str(label): count for label, count in sorted(Counter(truth_particle_ids).items())
            },
            "truth_pdg_counts": {
                str(pdg): count for pdg, count in sorted(Counter(truth_pdgs).items())
            },
            "truth_match_fraction": _finite_stats(truth_match_fractions),
        }

    if source_station is not None and target_station is not None:
        summary["candidate_diagnostics"] = _candidate_summary(
            events,
            source_station=source_station,
            target_station=target_station,
            chi2_gate=chi2_gate,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Canonical flat ROOT tracklet file")
    parser.add_argument("--output", default=None, help="Optional JSON destination")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--require-mc-labels", action="store_true")
    parser.add_argument("--source-station", type=int, default=None)
    parser.add_argument("--target-station", type=int, default=None)
    parser.add_argument("--chi2-gate", type=float, default=1.0e6)
    parser.add_argument(
        "--physical-order",
        action="store_true",
        help=(
            "Group events as consecutive equal-(run_id, event_id) blocks in "
            "file order instead of sorting.  Required for merged MC24 rec "
            "productions that reuse generator-job event numbers."
        ),
    )
    args = parser.parse_args()
    if (args.source_station is None) != (args.target_station is None):
        parser.error("--source-station and --target-station must be supplied together")
    if args.chi2_gate <= 0.0:
        parser.error("--chi2-gate must be positive")

    events = load_events(
        args.input,
        max_events=args.max_events,
        require_mc_labels=args.require_mc_labels,
        preserve_file_order=args.physical_order,
    )
    summary = summarize_events(
        events,
        input_path=args.input,
        source_station=args.source_station,
        target_station=args.target_station,
        chi2_gate=args.chi2_gate,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
