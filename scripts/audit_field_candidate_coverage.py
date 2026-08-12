#!/usr/bin/env python3
"""Audit truth-pair coverage of physical mode-0 field candidate exports.

The audit is intentionally read-only.  It distinguishes an absent synthetic
candidate from a failed Acts propagation, missing covariance, an incompatible
target plane, or a numerical covariance rejection.  This makes the maximum
association efficiency imposed by the physical candidate graph explicit before
selecting an MLP or global-assignment operating point.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from datasets.propagation_loader import PropagationRecords, load_propagation_records
from datasets.root_loader import EventTracklets, load_events
from geometry.propagation import mahalanobis_chi2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "faser-field-candidate-coverage-audit-v1"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _unique_truth_indices(event: EventTracklets, station: int) -> dict[int, int]:
    """Return MC labels that occur exactly once in one station collection."""
    if event.truth_particle_id is None:
        raise ValueError("field candidate coverage requires MC truth labels")
    indices = event.indices_for_station(station)
    truth_ids = [int(event.truth_particle_id[index]) for index in indices]
    counts = Counter(value for value in truth_ids if value >= 0)
    return {
        truth_id: int(index)
        for index, truth_id in zip(indices, truth_ids)
        if truth_id >= 0 and counts[truth_id] == 1
    }


def _valid_covariance(covariance: np.ndarray) -> bool:
    values = np.asarray(covariance, dtype=np.float64)
    return bool(
        values.shape == (4, 4)
        and np.isfinite(values).all()
        and np.allclose(values, values.T, rtol=1.0e-7, atol=1.0e-12)
        and np.all(np.diag(values) > 0.0)
    )


def _record_index(records: PropagationRecords) -> dict[tuple[int, int, int, int, int, int], int]:
    """Build an exact source/target record index and reject duplicate identities."""
    index: dict[tuple[int, int, int, int, int, int], int] = {}
    for row in range(records.size):
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_station_id[row]),
            int(records.target_station_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_tracklet_id[row]),
        )
        if key in index:
            raise ValueError(f"duplicate synthetic field-candidate record identity: {key}")
        index[key] = row
    return index


def _candidate_reason(
    event: EventTracklets,
    source_index: int,
    target_index: int,
    source_station: int,
    target_station: int,
    record_row: int | None,
    records: PropagationRecords,
    target_z_tolerance_mm: float,
) -> str:
    """Return the first physical eligibility failure used by candidate construction."""
    if record_row is None:
        return "no_exact_exported_record"
    row = int(record_row)
    if int(records.q_over_p_mode[row]) != 0:
        return "wrong_q_over_p_mode"
    if not bool(records.success[row]):
        return "acts_propagation_failed"
    if not bool(records.has_covariance[row]):
        return "acts_covariance_missing"
    if (
        int(records.source_station_id[row]) != source_station
        or int(records.target_station_id[row]) != target_station
    ):
        return "station_pair_metadata_mismatch"
    if not np.isclose(
        float(event.z_mm[target_index]),
        float(records.target_z_mm[row]),
        rtol=0.0,
        atol=target_z_tolerance_mm,
    ):
        return "target_z_mismatch"
    if not _valid_covariance(records.covariance[row]):
        return "invalid_propagated_covariance"
    combined = records.covariance[row] + event.covariance[target_index]
    if not _valid_covariance(combined):
        return "invalid_combined_covariance"
    try:
        mahalanobis_chi2(event.state[target_index] - records.prediction[row], combined)
    except ValueError:
        return "mahalanobis_failure"
    return "candidate_eligible"


def audit_sample(
    sample: CurriculumSample,
    target_z_tolerance_mm: float,
    example_limit: int,
) -> tuple[Counter[str], dict[tuple[int, int], Counter[str]], list[dict[str, object]]]:
    """Classify every unique truth pair in one physical synthetic payload."""
    events = load_events(sample.synthetic_tracklets, require_mc_labels=True)
    records = load_propagation_records(sample.field_candidates)
    index = _record_index(records)
    totals: Counter[str] = Counter()
    by_pair: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, object]] = []

    stations = tuple(sorted({int(station) for event in events for station in event.station_id}))
    for event in events:
        truth_by_station = {station: _unique_truth_indices(event, station) for station in stations}
        for source_station in stations:
            source_truth = truth_by_station[source_station]
            for target_station in stations:
                if source_station >= target_station:
                    continue
                target_truth = truth_by_station[target_station]
                for truth_id in sorted(set(source_truth).intersection(target_truth)):
                    source_index = source_truth[truth_id]
                    target_index = target_truth[truth_id]
                    identity = (
                        int(event.run_id),
                        int(event.event_id),
                        source_station,
                        target_station,
                        int(event.tracklet_id[source_index]),
                        int(event.tracklet_id[target_index]),
                    )
                    record_row = index.get(identity)
                    reason = _candidate_reason(
                        event,
                        source_index,
                        target_index,
                        source_station,
                        target_station,
                        record_row,
                        records,
                        target_z_tolerance_mm,
                    )
                    totals[reason] += 1
                    by_pair[(source_station, target_station)][reason] += 1
                    if reason != "candidate_eligible" and len(examples) < example_limit:
                        examples.append(
                            {
                                "source_station": source_station,
                                "target_station": target_station,
                                "reason": reason,
                                "synthetic_run_id": int(event.run_id),
                                "synthetic_event_id": int(event.event_id),
                                "truth_particle_id": int(truth_id),
                                "source_tracklet_id": int(event.tracklet_id[source_index]),
                                "target_tracklet_id": int(event.tracklet_id[target_index]),
                                "field_candidate_record_row": None if record_row is None else int(record_row),
                            }
                        )
    return totals, by_pair, examples


def _coverage_payload(counts: Mapping[str, int]) -> dict[str, object]:
    total = int(sum(counts.values()))
    eligible = int(counts.get("candidate_eligible", 0))
    return {
        "truth_pairs_with_unique_known_endpoints": total,
        "field_candidate_eligible_truth_pairs": eligible,
        "raw_physical_candidate_truth_recall": None if not total else eligible / total,
        "eligibility_reasons": {key: int(value) for key, value in sorted(counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
        help="audit one split only; validation is the non-test default",
    )
    parser.add_argument("--target-z-tolerance-mm", type=float, default=1.0e-6)
    parser.add_argument("--example-limit", type=int, default=25)
    args = parser.parse_args()

    if not np.isfinite(args.target_z_tolerance_mm) or args.target_z_tolerance_mm < 0.0:
        raise ValueError("target-z tolerance must be finite and non-negative")
    if args.example_limit < 0:
        raise ValueError("example limit must be non-negative")
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("coverage audit is restricted to physical mode-0 propagation")
    selected = [sample for sample in samples if sample.split == args.split]
    if not selected:
        raise ValueError(f"manifest has no {args.split} samples")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate: Counter[str] = Counter()
    by_magnitude: dict[float, Counter[str]] = defaultdict(Counter)
    by_pair: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    by_magnitude_pair: dict[tuple[float, tuple[int, int]], Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, object]] = []
    sample_reports: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []

    for sample in sorted(selected, key=lambda value: (value.magnitude_mm, value.payload_id)):
        totals, pair_counts, sample_examples = audit_sample(
            sample,
            target_z_tolerance_mm=float(args.target_z_tolerance_mm),
            example_limit=args.example_limit,
        )
        aggregate.update(totals)
        by_magnitude[float(sample.magnitude_mm)].update(totals)
        for pair, counts in pair_counts.items():
            by_pair[pair].update(counts)
            by_magnitude_pair[(float(sample.magnitude_mm), pair)].update(counts)
            for reason, count in sorted(counts.items()):
                rows.append(
                    {
                        "split": sample.split,
                        "payload_id": sample.payload_id,
                        "magnitude_mm": float(sample.magnitude_mm),
                        "station_pair": f"{pair[0]}->{pair[1]}",
                        "reason": reason,
                        "count": int(count),
                    }
                )
        sample_reports.append(
            {
                "payload_id": sample.payload_id,
                "magnitude_mm": float(sample.magnitude_mm),
                **_coverage_payload(totals),
            }
        )
        examples.extend(
            {"payload_id": sample.payload_id, "magnitude_mm": float(sample.magnitude_mm), **row}
            for row in sample_examples
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_manifest": str(manifest_path),
        "split": args.split,
        "q_over_p_mode": 0,
        "physical_geometry_repropagation": True,
        "target_z_tolerance_mm": float(args.target_z_tolerance_mm),
        "overall": _coverage_payload(aggregate),
        "by_magnitude_mm": {
            f"{magnitude:g}": _coverage_payload(counts)
            for magnitude, counts in sorted(by_magnitude.items())
        },
        "by_station_pair": {
            f"{pair[0]}->{pair[1]}": _coverage_payload(counts)
            for pair, counts in sorted(by_pair.items())
        },
        "by_magnitude_and_station_pair": {
            f"{magnitude:g}:{pair[0]}->{pair[1]}": _coverage_payload(counts)
            for (magnitude, pair), counts in sorted(by_magnitude_pair.items())
        },
        "by_payload": sample_reports,
        "failure_examples": examples,
    }
    _write_json(output_root / "coverage_audit.json", report)
    _write_csv(output_root / "coverage_by_payload_pair_reason.csv", rows)
    _write_json(
        output_root / "provenance.json",
        {
            "generator": "scripts/audit_field_candidate_coverage.py",
            "project_root": str(PROJECT_ROOT),
            "manifest": str(manifest_path),
            "split": args.split,
            "test_opened": args.split == "test",
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
        },
    )
    print(json.dumps({"output_dir": str(output_root), "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
