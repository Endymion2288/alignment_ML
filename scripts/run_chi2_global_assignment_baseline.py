#!/usr/bin/env python3
"""Evaluate conventional field-chi2 greedy matching on validation only.

This is the non-learned reference for the pairwise MLP study.  Candidate
edges are the same mode-0 FaserActs records used by the MLP; at a fixed
physical chi2 gate, descending ``1 / (1 + chi2)`` is exactly equivalent to
ascending chi2 greedy one-to-one matching.  The program never opens a test
split, so it is safe to use while repairing a nominal baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from baselines.global_assignment import AssignmentConfig
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets, filter_candidate_sets
from training.global_assignment import choose_global_operating_point, evaluate_global_assignment_sets


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_gates(text: str) -> list[float | None]:
    values: list[float | None] = []
    for value in text.split(","):
        token = value.strip().lower()
        gate = None if token in {"none", "ungated", "null"} else float(token)
        if gate is not None and (not np.isfinite(gate) or gate <= 0.0):
            raise ValueError("chi2 gates must be positive or 'none'")
        if any(
            existing is None and gate is None
            or existing is not None and gate is not None and np.isclose(existing, gate)
            for existing in values
        ):
            raise ValueError("chi2 gate list contains a duplicate")
        values.append(gate)
    if not values:
        raise ValueError("at least one chi2 gate is required")
    return values


def _scores(candidate_sets: Sequence[object]) -> list[np.ndarray]:
    """Map finite chi2 monotonically to valid score-matrix probabilities."""
    result: list[np.ndarray] = []
    for candidate_set in candidate_sets:
        chi2 = np.asarray([candidate.chi2 for candidate in candidate_set.candidates], dtype=np.float64)
        if not np.isfinite(chi2).all() or np.any(chi2 < 0.0):
            raise ValueError("physical candidate chi2 must be finite and non-negative")
        result.append(1.0 / (1.0 + chi2))
    return result


def _selected(sets: Sequence[object], scores: Sequence[np.ndarray], magnitude: float) -> tuple[list[object], list[np.ndarray]]:
    selected_sets: list[object] = []
    selected_scores: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        if np.isclose(float(candidate_set.sample.magnitude_mm), magnitude):
            selected_sets.append(candidate_set)
            selected_scores.append(values)
    return selected_sets, selected_scores


def _row(
    result: Mapping[str, object],
    gate: float | None,
    magnitude: float,
) -> dict[str, object]:
    candidate = result["candidate"]
    association = result["association"]
    unmatched = result["unmatched"]
    if not isinstance(candidate, Mapping) or not isinstance(association, Mapping) or not isinstance(unmatched, Mapping):
        raise RuntimeError("unexpected global-assignment evaluation payload")
    return {
        "assignment_method": "traditional_field_chi2_greedy",
        "candidate_chi2_gate": gate,
        "magnitude_mm": magnitude,
        "score_mapping": "1/(1+chi2); descending score == ascending chi2",
        "score_threshold": 0.0,
        "candidate_truth_recall": candidate["candidate_truth_recall"],
        "candidate_rows": candidate["candidate_rows"],
        "roc_auc": candidate["roc_auc"],
        "average_precision": candidate["average_precision"],
        **association,
        **unmatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chi2-gates", default="25,250,1000,5000,none")
    parser.add_argument("--maximum-inclusive-fake-rate", type=float, default=0.05)
    parser.add_argument("--minimum-inclusive-purity", type=float, default=0.95)
    parser.add_argument("--minimum-nominal-association-efficiency", type=float, default=0.70)
    args = parser.parse_args()

    for name, value in (
        ("maximum inclusive fake rate", args.maximum_inclusive_fake_rate),
        ("minimum inclusive purity", args.minimum_inclusive_purity),
        ("minimum nominal association efficiency", args.minimum_nominal_association_efficiency),
    ):
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("traditional chi2 baseline is restricted to physical mode-0 propagation")
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    if not validation_samples:
        raise ValueError("synthetic manifest contains no validation sample")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_config = manifest.get("synthetic_multitrack")
    if not isinstance(synthetic_config, Mapping):
        raise ValueError("synthetic manifest lacks its station configuration")
    raw_stations = synthetic_config.get("stations")
    if not isinstance(raw_stations, list) or len(raw_stations) < 2:
        raise ValueError("synthetic manifest has invalid station configuration")
    stations = tuple(sorted({int(station) for station in raw_stations}))
    station_pairs = tuple((left, right) for left in stations for right in stations if left < right)
    all_sets = build_candidate_sets(validation_samples, station_pairs, chi2_gate=None)
    gates = _parse_gates(args.chi2_gates)
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for gate in gates:
        gated_sets = filter_candidate_sets(all_sets, gate)
        gated_scores = _scores(gated_sets)
        for magnitude in sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in gated_sets}):
            selected_sets, selected_scores = _selected(gated_sets, gated_scores, magnitude)
            evaluation = evaluate_global_assignment_sets(
                selected_sets,
                selected_scores,
                AssignmentConfig(method="greedy", score_threshold=0.0),
                calibration_bins=15,
            )
            rows.append(_row(evaluation, gate, magnitude))
            by_pair = evaluation["by_station_pair"]
            if not isinstance(by_pair, Mapping):
                raise RuntimeError("unexpected station-pair evaluation payload")
            for pair, payload in sorted(by_pair.items()):
                if not isinstance(payload, Mapping):
                    raise RuntimeError("unexpected pair payload")
                candidate = payload["candidate"]
                association = payload["association"]
                unmatched = payload["unmatched"]
                if not isinstance(candidate, Mapping) or not isinstance(association, Mapping) or not isinstance(unmatched, Mapping):
                    raise RuntimeError("unexpected pair metric payload")
                pair_rows.append(
                    {
                        "assignment_method": "traditional_field_chi2_greedy",
                        "candidate_chi2_gate": gate,
                        "magnitude_mm": magnitude,
                        "station_pair": pair,
                        "candidate_truth_recall": candidate["candidate_truth_recall"],
                        "candidate_rows": candidate["candidate_rows"],
                        "roc_auc": candidate["roc_auc"],
                        "average_precision": candidate["average_precision"],
                        **association,
                        **unmatched,
                    }
                )

    nominal_rows = [row for row in rows if np.isclose(float(row["magnitude_mm"]), 0.0)]
    operating_point = choose_global_operating_point(
        nominal_rows,
        maximum_inclusive_fake_rate=float(args.maximum_inclusive_fake_rate),
        minimum_inclusive_purity=float(args.minimum_inclusive_purity),
        minimum_association_efficiency=float(args.minimum_nominal_association_efficiency),
    )
    _write_csv(output_root / "validation_chi2_matching_by_magnitude.csv", rows)
    _write_csv(output_root / "validation_chi2_matching_by_station_pair.csv", pair_rows)
    _write_json(
        output_root / "metrics.json",
        {
            "method": "traditional_field_chi2_greedy",
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "split": "validation",
            "test_opened": False,
            "test_withheld_reason": "validation_only_baseline_diagnosis",
            "synthetic_manifest": str(manifest_path),
            "chi2_gates": gates,
            "operating_constraints": {
                "maximum_inclusive_fake_rate": float(args.maximum_inclusive_fake_rate),
                "minimum_inclusive_purity": float(args.minimum_inclusive_purity),
                "minimum_nominal_association_efficiency": float(args.minimum_nominal_association_efficiency),
            },
            "nominal_validation_operating_point": operating_point,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "nominal_validation_operating_point": operating_point,
                "test_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
