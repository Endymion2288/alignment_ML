#!/usr/bin/env python3
"""Audit kinematic-state coverage of physically refitted MC sources.

The input is the canonical tracklet output of the real cluster -> segment
refit -> Acts workflow, not an xAOD filename heuristic.  It reports only
observed truth-matched muon state distributions and never infers source
particle properties from production names.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from datasets.root_loader import EventTracklets, load_events


STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")


def _distribution(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "p01": None,
        "p50": None,
        "p99": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "p01": float(np.quantile(finite, 0.01)),
                "p50": float(np.quantile(finite, 0.50)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
            }
        )
    return result


def _state_summary(states: np.ndarray) -> dict[str, dict[str, int | float | None]]:
    matrix = np.asarray(states, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(STATE_NAMES):
        raise ValueError("state coverage input must have shape [rows, 4]")
    return {name: _distribution(matrix[:, column]) for column, name in enumerate(STATE_NAMES)}


def summarize_refitted_source(
    source_id: str,
    events: Sequence[EventTracklets],
    *,
    min_truth_match_fraction: float,
) -> dict[str, object]:
    """Summarize observed truth-matched muon state coverage for one source."""
    if not source_id:
        raise ValueError("source_id must be non-empty")
    if not 0.0 <= min_truth_match_fraction <= 1.0:
        raise ValueError("min_truth_match_fraction must be in [0, 1]")
    if not events:
        raise ValueError(f"source '{source_id}' has no exported events")
    if any(
        event.truth_pdg is None or event.truth_match_fraction is None or event.truth_particle_id is None
        for event in events
    ):
        raise ValueError(f"source '{source_id}' lacks required MC truth labels")

    states = np.concatenate([event.state for event in events], axis=0)
    stations = np.concatenate([event.station_id for event in events], axis=0)
    pdgs = np.concatenate([np.asarray(event.truth_pdg, dtype=np.int32) for event in events], axis=0)
    fractions = np.concatenate(
        [np.asarray(event.truth_match_fraction, dtype=np.float64) for event in events], axis=0
    )
    particle_ids = np.concatenate(
        [np.asarray(event.truth_particle_id, dtype=np.int64) for event in events], axis=0
    )
    truth_muon = (np.abs(pdgs) == 13) & (fractions >= min_truth_match_fraction) & (particle_ids >= 0)
    station_event_counts: Counter[int] = Counter()
    for event in events:
        station_event_counts.update(int(value) for value in np.unique(event.station_id))
    by_station = {
        str(station): _state_summary(states[truth_muon & (stations == station)])
        for station in sorted(int(value) for value in np.unique(stations))
    }
    return {
        "source_id": source_id,
        "events": int(len(events)),
        "tracklets": int(states.shape[0]),
        "station_event_counts": {
            str(station): int(count) for station, count in sorted(station_event_counts.items())
        },
        "truth_pdg_counts": {str(pdg): int(count) for pdg, count in sorted(Counter(pdgs.tolist()).items())},
        "truth_matched_muon_tracklets": int(np.count_nonzero(truth_muon)),
        "min_truth_match_fraction": float(min_truth_match_fraction),
        "truth_matched_muon_state": _state_summary(states[truth_muon]),
        "truth_matched_muon_state_by_station": by_station,
    }


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be SOURCE_ID=TRACKLETS_ROOT")
    source_id, raw_path = value.split("=", 1)
    source_id = source_id.strip()
    path = Path(raw_path).expanduser().resolve()
    if not source_id or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid source specification: {value}")
    return source_id, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_source,
        metavar="SOURCE_ID=TRACKLETS_ROOT",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    source_ids = [source_id for source_id, _ in args.source]
    if len(source_ids) != len(set(source_ids)):
        parser.error("--source IDs must be unique")

    source_rows: list[dict[str, object]] = []
    for source_id, path in args.source:
        source_rows.append(
            {
                "input": str(path),
                **summarize_refitted_source(
                    source_id,
                    load_events(path, require_mc_labels=True),
                    min_truth_match_fraction=args.min_truth_match_fraction,
                ),
            }
        )
    payload: Mapping[str, object] = {
        "schema_version": "faser-refitted-source-diversity-audit-v1",
        "source_split_unit": "original_xaod_file",
        "summary_scope": "observed_truth_matched_muon_refitted_tracklets",
        "sources": source_rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sources": len(source_rows)}, indent=2))


if __name__ == "__main__":
    main()
