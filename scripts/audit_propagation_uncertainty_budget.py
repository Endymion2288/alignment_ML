#!/usr/bin/env python3
"""Per-station-pair uncertainty budget of mode-0 propagation (root-cause audit).

Read-only decomposition of the combined covariance for truth-matched adjacent
station-pair edges into

    source tracklet covariance  --(Acts transport + material)-->  propagated
    covariance  +  target tracklet covariance  =  combined covariance,

evaluated separately for every q/p seed mode present in the bank (mode 0 =
fixed 100 GeV seed, mode 1/2 = truth-q/p controls).  Same-edge controls hold
the reconstructed position/direction fixed and swap only the q/p seed, so
residual degradation and covariance inflation are measured independently.

For each station pair, component and mode the budget table reports robust
medians of: source-tracklet sigma, propagated sigma, target-tracklet sigma,
combined sigma, the propagated/source inflation factor, the target share of
the combined variance, residual robust sigma, pull width, and the chi2 tail
(fraction above 100).  No propagation behaviour is modified; nothing is fit
or retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events

ADJACENT_PAIRS = ((0, 1), (1, 2), (2, 3))
COMPONENTS = ("x_mm", "y_mm", "tx", "ty")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _robust_sigma(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values)))


def collect_budget(
    tracklets: Path, propagations: Path
) -> dict[tuple[str, int], dict[str, list[float]]]:
    """Per (pair, mode) budget ingredients for truth-matched edges."""
    events = load_events(tracklets, require_mc_labels=True)
    records = load_propagation_records(propagations)
    rows_by_key: dict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
    for row in range(records.size):
        if not (bool(records.success[row]) and bool(records.has_covariance[row])):
            continue
        mode = int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_tracklet_id[row]),
            mode,
        )
        rows_by_key[key].append(row)

    out: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        if event.truth_particle_id is None:
            continue
        for source_station, target_station in ADJACENT_PAIRS:
            pair = f"{source_station}->{target_station}"
            per_station: dict[int, dict[int, int]] = {}
            for index in range(event.size):
                station = int(event.station_id[index])
                truth = int(event.truth_particle_id[index])
                per_station.setdefault(station, {})[truth] = index
            shared = sorted(
                set(per_station.get(source_station, {}))
                & set(per_station.get(target_station, {}))
            )
            for truth_id in shared:
                source_index = per_station[source_station][truth_id]
                target_index = per_station[target_station][truth_id]
                modes_present = sorted(
                    {
                        key[4]
                        for key in rows_by_key
                        if key[:4]
                        == (
                            int(event.run_id),
                            int(event.event_id),
                            int(event.tracklet_id[source_index]),
                            int(event.tracklet_id[target_index]),
                        )
                    }
                )
                for mode in modes_present:
                    key = (
                        int(event.run_id),
                        int(event.event_id),
                        int(event.tracklet_id[source_index]),
                        int(event.tracklet_id[target_index]),
                        mode,
                    )
                    row = rows_by_key[key][0]
                    propagated = np.asarray(records.covariance[row], dtype=np.float64)
                    target_cov = np.asarray(event.covariance[target_index], dtype=np.float64)
                    source_cov = np.asarray(event.covariance[source_index], dtype=np.float64)
                    combined = propagated + target_cov
                    residual = np.asarray(event.state[target_index], dtype=np.float64) - np.asarray(
                        records.prediction[row], dtype=np.float64
                    )
                    budget = out[(pair, mode)]
                    budget["count"].append(1.0)
                    for i, component in enumerate(COMPONENTS):
                        budget[f"source_sigma_{component}"].append(
                            math.sqrt(max(source_cov[i, i], 0.0))
                        )
                        budget[f"prop_sigma_{component}"].append(
                            math.sqrt(max(propagated[i, i], 0.0))
                        )
                        budget[f"target_sigma_{component}"].append(
                            math.sqrt(max(target_cov[i, i], 0.0))
                        )
                        budget[f"combined_sigma_{component}"].append(
                            math.sqrt(max(combined[i, i], 0.0))
                        )
                        budget[f"residual_{component}"].append(float(residual[i]))
                        if combined[i, i] > 0:
                            budget[f"pull_{component}"].append(
                                float(residual[i] / math.sqrt(combined[i, i]))
                            )
                        if source_cov[i, i] > 0 and propagated[i, i] > 0:
                            budget[f"inflation_{component}"].append(
                                math.sqrt(propagated[i, i] / source_cov[i, i])
                            )
                        if combined[i, i] > 0:
                            budget[f"target_share_{component}"].append(
                                float(target_cov[i, i] / combined[i, i])
                            )
                    try:
                        inverse = np.linalg.inv(combined)
                        budget["chi2"].append(float(residual @ inverse @ residual))
                    except np.linalg.LinAlgError:
                        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    pooled: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in manifest["sources"]:
        if str(entry["split"]) != args.split:
            continue
        source_id = str(entry["source_id"])
        if args.source_id is not None and source_id not in set(args.source_id):
            continue
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        points = {str(point["name"]): point for point in plan["points"]}
        point = points.get(args.point)
        if point is None:
            raise ValueError(f"point '{args.point}' absent from {root}/scan_plan.json")
        point_root = root / str(point["relative_point_dir"])
        source_budget = collect_budget(
            point_root / "refit" / "tracklets.root",
            point_root / "refit" / "propagations.root",
        )
        for key, payload in source_budget.items():
            for name, values in payload.items():
                pooled[key][name].extend(values)

    rows: list[dict[str, Any]] = []
    for (pair, mode), payload in sorted(pooled.items()):
        chi2 = np.asarray(payload["chi2"], dtype=np.float64)
        for component in COMPONENTS:
            rows.append(
                {
                    "station_pair": pair,
                    "q_over_p_mode": mode,
                    "component": component,
                    "count": int(len(payload["count"])),
                    "source_sigma_median": _median(payload[f"source_sigma_{component}"]),
                    "propagated_sigma_median": _median(payload[f"prop_sigma_{component}"]),
                    "target_sigma_median": _median(payload[f"target_sigma_{component}"]),
                    "combined_sigma_median": _median(payload[f"combined_sigma_{component}"]),
                    "transport_inflation_median": _median(payload[f"inflation_{component}"]),
                    "target_variance_share_median": _median(
                        payload[f"target_share_{component}"]
                    ),
                    "residual_robust_sigma": _robust_sigma(
                        np.asarray(payload[f"residual_{component}"])
                    ),
                    "pull_robust_sigma": _robust_sigma(np.asarray(payload[f"pull_{component}"])),
                    "chi2_median": float(np.median(chi2)) if chi2.size else None,
                    "chi2_frac_above_100": float(np.mean(chi2 > 100)) if chi2.size else None,
                }
            )

    csv_path = output / "uncertainty_budget.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split,
        "point": args.point,
        "rows": len(rows),
        "pairs_modes": sorted({f"{pair}/mode{mode}" for pair, mode in pooled}),
        "test_data_accessed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
