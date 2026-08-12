#!/usr/bin/env python3
"""Audit synthetic missingness and fake populations using physical Acts candidates.

The audit is evaluation-only.  It reads provenance branches written by
``synthetic_overlay`` and recomputes candidate chi2 values from the same
payload's mode-0 FaserActs propagation records; it never modifies coordinates
or feeds synthetic-role labels to the MLP.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from baselines.field_chi2_matching import build_field_candidates
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from datasets.synthetic_overlay import SYNTHETIC_ROLE_FIELD_HARD_FAKE, SYNTHETIC_ROLE_NAMES
from evaluation.metrics import UnmatchedEndpointMetrics, assess_event_unmatched_endpoints


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: list[float], prefix: str = "chi2") -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_{name}": None
            for name in ("min", "p10", "p50", "p90", "p99", "max")
        }
    array = np.asarray(values, dtype=np.float64)
    quantile = np.quantile(array, [0.0, 0.10, 0.50, 0.90, 0.99, 1.0])
    return {
        f"{prefix}_min": float(quantile[0]),
        f"{prefix}_p10": float(quantile[1]),
        f"{prefix}_p50": float(quantile[2]),
        f"{prefix}_p90": float(quantile[3]),
        f"{prefix}_p99": float(quantile[4]),
        f"{prefix}_max": float(quantile[5]),
    }


def _overlay_summary(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"overlay summary is not a mapping: {path}")
    return payload


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [
        row
        for row in rows
        if row["station_pair"] == "0->1"
        and row["candidate_category"] in {
            "true_tracklet->true_tracklet",
            "true_tracklet->random_easy_fake",
            "true_tracklet->field_aware_hard_negative",
        }
        and row["chi2_p50"] is not None
    ]
    if not selected:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["candidate_category"])].append(row)
    for category, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["magnitude_mm"]))
        axis.plot(
            [float(row["magnitude_mm"]) for row in ordered],
            [float(row["chi2_p50"]) for row in ordered],
            marker="o",
            label=category,
        )
    axis.set_xlabel("injected misalignment magnitude [mm]")
    axis.set_ylabel("candidate chi2 median, station pair 0->1")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        default=None,
        help="Audit only the requested split; repeat to include multiple splits",
    )
    args = parser.parse_args()
    # A sealed multi-direction test scan intentionally materializes only the
    # test split after the MLP, calibration, and thresholds have been frozen.
    # The default loader remains strict for training; this audit is read-only.
    _, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
    )
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("background audit supports only mode-0 physical propagation")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty audit directory")
    output_root.mkdir(parents=True, exist_ok=True)
    requested_splits = None if args.split is None else set(args.split)
    selected_samples = [
        sample for sample in samples if requested_splits is None or sample.split in requested_splits
    ]
    if not selected_samples:
        raise ValueError("no synthetic samples match the requested split selection")
    configured_synthetic = manifest.get("synthetic_multitrack", {})
    if not isinstance(configured_synthetic, Mapping):
        raise ValueError("synthetic manifest lacks a synthetic_multitrack mapping")

    tracklet_counts: dict[tuple[str, float, int, str], int] = defaultdict(int)
    candidate_chi2: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    candidate_counts: dict[tuple[str, float, str, str], int] = defaultdict(int)
    candidate_gates: dict[tuple[str, float, str, str], dict[float, int]] = defaultdict(
        lambda: {25.0: 0, 250.0: 0, 1000.0: 0, 5000.0: 0}
    )
    missing_by_group: dict[tuple[str, float, str], UnmatchedEndpointMetrics] = defaultdict(
        UnmatchedEndpointMetrics
    )
    hard_anchor_values: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    hard_relative: dict[tuple[str, float, str], dict[str, object]] = defaultdict(
        lambda: {
            "hard_negative_tracklets": 0,
            "anchor_candidate_found": 0,
            "truth_counterpart_found": 0,
            "truth_candidate_found": 0,
            "hard_chi2_le_truth_chi2": 0,
            "hard_to_truth_chi2_ratio": [],
            "hard_minus_truth_chi2": [],
        }
    )
    overlay_summaries: list[dict[str, object]] = []

    for sample in selected_samples:
        events = load_events(sample.synthetic_tracklets, require_mc_labels=True)
        records = load_propagation_records(sample.field_candidates)
        for event in events:
            if event.synthetic_role is None:
                raise ValueError(
                    f"synthetic provenance branches are absent in {sample.synthetic_tracklets}; "
                    "materialize the explicit fake populations first"
                )
            for row in range(event.size):
                role = int(event.synthetic_role[row])
                role_name = SYNTHETIC_ROLE_NAMES.get(role, f"unknown_role_{role}")
                tracklet_counts[(sample.split, sample.magnitude_mm, int(event.station_id[row]), role_name)] += 1
                if role == SYNTHETIC_ROLE_FIELD_HARD_FAKE:
                    if event.synthetic_hard_anchor_chi2 is None:
                        raise ValueError("hard-negative role is present without anchor chi2 provenance")
                    value = float(event.synthetic_hard_anchor_chi2[row])
                    if np.isfinite(value):
                        hard_anchor_values[(sample.split, sample.magnitude_mm, int(event.station_id[row]))].append(value)
            stations = sorted(set(int(value) for value in event.station_id.tolist()))
            for source in stations:
                for target in stations:
                    if source >= target:
                        continue
                    pair_name = f"{source}->{target}"
                    missing_by_group[(sample.split, sample.magnitude_mm, pair_name)].add(
                        assess_event_unmatched_endpoints(event, (), source, target)
                    )
                    candidates = build_field_candidates(
                        event,
                        records,
                        source_station=source,
                        target_station=target,
                        chi2_gate=None,
                        q_over_p_mode=0,
                    )
                    candidate_by_endpoints = {
                        (candidate.source_index, candidate.target_index): candidate
                        for candidate in candidates
                    }
                    for candidate in candidates:
                        source_role = SYNTHETIC_ROLE_NAMES.get(
                            int(event.synthetic_role[candidate.source_index]),
                            f"unknown_role_{int(event.synthetic_role[candidate.source_index])}",
                        )
                        target_role = SYNTHETIC_ROLE_NAMES.get(
                            int(event.synthetic_role[candidate.target_index]),
                            f"unknown_role_{int(event.synthetic_role[candidate.target_index])}",
                        )
                        category = f"{source_role}->{target_role}"
                        key = (sample.split, sample.magnitude_mm, pair_name, category)
                        candidate_counts[key] += 1
                        candidate_chi2[key].append(float(candidate.chi2))
                        for gate in candidate_gates[key]:
                            candidate_gates[key][gate] += int(candidate.chi2 <= gate)

                    # A hard fake is selected with an actual Acts chi2 to one
                    # true source.  Compare that score to the physical score
                    # of that source's real same-truth endpoint, if present.
                    # This exposes whether the synthetic hard population is
                    # merely nearby or systematically more compatible than a
                    # genuine segment, without exposing roles to training.
                    if event.synthetic_hard_anchor_station is None or event.truth_particle_id is None:
                        continue
                    hard_target_rows = [
                        int(row)
                        for row in event.indices_for_station(target).tolist()
                        if int(event.synthetic_role[row]) == SYNTHETIC_ROLE_FIELD_HARD_FAKE
                        and int(event.synthetic_hard_anchor_station[row]) == source
                    ]
                    if not hard_target_rows:
                        continue
                    true_source_rows = [
                        int(row)
                        for row in event.indices_for_station(source).tolist()
                        if int(event.synthetic_role[row]) != SYNTHETIC_ROLE_FIELD_HARD_FAKE
                        and int(event.truth_particle_id[row]) >= 0
                    ]
                    true_target_by_truth = {
                        int(event.truth_particle_id[row]): int(row)
                        for row in event.indices_for_station(target).tolist()
                        if int(event.truth_particle_id[row]) >= 0
                    }
                    pair_key = (sample.split, sample.magnitude_mm, pair_name)
                    for hard_target in hard_target_rows:
                        payload = hard_relative[pair_key]
                        payload["hard_negative_tracklets"] = int(payload["hard_negative_tracklets"]) + 1
                        anchor_chi2 = float(event.synthetic_hard_anchor_chi2[hard_target])
                        if not np.isfinite(anchor_chi2):
                            continue
                        anchor_candidates = [
                            candidate_by_endpoints[(source_row, hard_target)]
                            for source_row in true_source_rows
                            if (source_row, hard_target) in candidate_by_endpoints
                        ]
                        if not anchor_candidates:
                            continue
                        anchor = min(
                            anchor_candidates,
                            key=lambda candidate: abs(float(candidate.chi2) - anchor_chi2),
                        )
                        payload["anchor_candidate_found"] = int(payload["anchor_candidate_found"]) + 1
                        truth_id = int(event.truth_particle_id[anchor.source_index])
                        truth_target = true_target_by_truth.get(truth_id)
                        if truth_target is None:
                            continue
                        payload["truth_counterpart_found"] = int(payload["truth_counterpart_found"]) + 1
                        truth_candidate = candidate_by_endpoints.get((anchor.source_index, truth_target))
                        if truth_candidate is None:
                            continue
                        payload["truth_candidate_found"] = int(payload["truth_candidate_found"]) + 1
                        hard_chi2 = float(anchor.chi2)
                        truth_chi2 = float(truth_candidate.chi2)
                        payload["hard_chi2_le_truth_chi2"] = int(
                            payload["hard_chi2_le_truth_chi2"]
                        ) + int(hard_chi2 <= truth_chi2)
                        ratios = payload["hard_to_truth_chi2_ratio"]
                        deltas = payload["hard_minus_truth_chi2"]
                        if not isinstance(ratios, list) or not isinstance(deltas, list):  # pragma: no cover
                            raise RuntimeError("invalid hard-negative relative-score audit payload")
                        if truth_chi2 > 0.0:
                            ratios.append(hard_chi2 / truth_chi2)
                        deltas.append(hard_chi2 - truth_chi2)
        summary_path = sample.synthetic_tracklets.parent / "overlay_summary.json"
        summary = _overlay_summary(summary_path)
        overlay_summaries.append(
            {
                "source_id": sample.source_id,
                "split": sample.split,
                "payload_id": sample.payload_id,
                "magnitude_mm": sample.magnitude_mm,
                **summary,
            }
        )

    tracklet_rows = [
        {
            "split": split,
            "magnitude_mm": magnitude,
            "station_id": station,
            "synthetic_role": role,
            "tracklet_count": count,
        }
        for (split, magnitude, station, role), count in sorted(tracklet_counts.items())
    ]
    candidate_rows: list[dict[str, object]] = []
    for key, count in sorted(candidate_counts.items()):
        split, magnitude, pair, category = key
        candidate_rows.append(
            {
                "split": split,
                "magnitude_mm": magnitude,
                "station_pair": pair,
                "candidate_category": category,
                "candidate_count": count,
                **_quantiles(candidate_chi2[key]),
                **{f"fraction_chi2_le_{int(gate)}": candidate_gates[key][gate] / count for gate in candidate_gates[key]},
            }
        )
    missing_rows = [
        {
            "split": split,
            "magnitude_mm": magnitude,
            "station_pair": pair,
            **metrics.as_dict(),
        }
        for (split, magnitude, pair), metrics in sorted(missing_by_group.items())
    ]
    hard_rows = [
        {
            "split": split,
            "magnitude_mm": magnitude,
            "station_id": station,
            "hard_negative_tracklets": len(values),
            **_quantiles(values),
        }
        for (split, magnitude, station), values in sorted(hard_anchor_values.items())
    ]
    hard_relative_rows: list[dict[str, object]] = []
    for (split, magnitude, pair), payload in sorted(hard_relative.items()):
        hard_count = int(payload["hard_negative_tracklets"])
        truth_candidate_count = int(payload["truth_candidate_found"])
        ratios = payload["hard_to_truth_chi2_ratio"]
        deltas = payload["hard_minus_truth_chi2"]
        if not isinstance(ratios, list) or not isinstance(deltas, list):  # pragma: no cover
            raise RuntimeError("invalid hard-negative relative-score audit payload")
        hard_relative_rows.append(
            {
                "split": split,
                "magnitude_mm": magnitude,
                "station_pair": pair,
                "hard_negative_tracklets": hard_count,
                "anchor_candidate_found": int(payload["anchor_candidate_found"]),
                "truth_counterpart_found": int(payload["truth_counterpart_found"]),
                "truth_candidate_found": truth_candidate_count,
                "hard_chi2_le_truth_chi2": int(payload["hard_chi2_le_truth_chi2"]),
                "fraction_hard_chi2_le_truth_chi2": (
                    None
                    if not truth_candidate_count
                    else int(payload["hard_chi2_le_truth_chi2"]) / truth_candidate_count
                ),
                **_quantiles(ratios, prefix="hard_to_truth_chi2_ratio"),
                **_quantiles(deltas, prefix="hard_minus_truth_chi2"),
            }
        )
    _write_csv(output_root / "tracklet_roles_by_station.csv", tracklet_rows)
    _write_csv(output_root / "candidate_chi2_by_fake_category.csv", candidate_rows)
    _write_csv(output_root / "missing_endpoint_audit.csv", missing_rows)
    _write_csv(output_root / "hard_negative_anchor_chi2.csv", hard_rows)
    _write_csv(output_root / "hard_negative_truth_relative_chi2.csv", hard_relative_rows)
    _write_csv(output_root / "overlay_summaries.csv", overlay_summaries)
    _plot(output_root / "fake_category_chi2_median.png", candidate_rows)
    (output_root / "audit.json").write_text(
        json.dumps(
            {
                "synthetic_manifest": str(Path(args.synthetic_manifest).expanduser().resolve()),
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "synthetic_role_codes": SYNTHETIC_ROLE_NAMES,
                "configured_hard_negative": {
                    key: configured_synthetic.get(key)
                    for key in (
                        "hard_negative_mean_per_target_station",
                        "hard_negative_chi2_min",
                        "hard_negative_chi2_max",
                        "hard_negative_min_truth_chi2_ratio",
                        "hard_negative_max_truth_chi2_ratio",
                    )
                },
                "samples": len(selected_samples),
                "requested_splits": (
                    ["train", "validation", "test"]
                    if requested_splits is None
                    else sorted(requested_splits)
                ),
                "test_opened": bool(requested_splits is None or "test" in requested_splits),
                "role_rows": len(tracklet_rows),
                "candidate_category_rows": len(candidate_rows),
                "missing_rows": len(missing_rows),
                "hard_anchor_rows": len(hard_rows),
                "hard_truth_relative_rows": len(hard_relative_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "samples": len(selected_samples),
                "test_opened": bool(requested_splits is None or "test" in requested_splits),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
