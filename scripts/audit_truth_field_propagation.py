#!/usr/bin/env python3
"""Check FaserActs truth-to-truth field propagation centre closure.

The exporter writes propagation ``q_over_p_mode=2`` records from the source
truth state to a global-z plane at the target truth station-average z.  Truth
has no measurement covariance, so this command reports residuals only.  Use
``evaluate_field_propagation.py`` mode 1 for reco-endpoint pulls and chi2.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")
REQUIRED_BRANCHES = (
    "run",
    "eventID",
    "truth_barcode",
    *(f"truth_st{station}_{component}" for station in range(4) for component in (
        "x", "y", "z", "px", "py", "pz"
    )),
    "TrackletPropagation_source_station_id",
    "TrackletPropagation_target_station_id",
    "TrackletPropagation_truth_particle_id",
    "TrackletPropagation_q_over_p_mode",
    "TrackletPropagation_target_z_mm",
    "TrackletPropagation_x_mm",
    "TrackletPropagation_y_mm",
    "TrackletPropagation_tx",
    "TrackletPropagation_ty",
    "TrackletPropagation_success",
)


def _stats(values: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict[str, int | float | None] = {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": None,
        "std": None,
        "rms": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95.0)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _truth_state(
    values: dict[str, np.ndarray],
    station: int,
    truth_row: int,
) -> tuple[np.ndarray, float] | None:
    x = values[f"truth_st{station}_x"][truth_row]
    y = values[f"truth_st{station}_y"][truth_row]
    z = values[f"truth_st{station}_z"][truth_row]
    px = values[f"truth_st{station}_px"][truth_row]
    py = values[f"truth_st{station}_py"][truth_row]
    pz = values[f"truth_st{station}_pz"][truth_row]
    if not np.isfinite([x, y, z, px, py, pz]).all() or abs(pz) < 1.0e-12:
        return None
    return np.array([x, y, px / pz, py / pz], dtype=np.float64), float(z)


def _summary(residual: np.ndarray, target_z_difference: np.ndarray) -> dict[str, object]:
    return {
        "records": int(residual.shape[0]),
        "residual": {
            name: _stats(residual[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "record_minus_truth_target_z_mm": _stats(target_z_difference),
    }


def audit_truth_field_propagation(
    input_path: str | Path,
    tree_name: str = "nt",
) -> dict[str, object]:
    """Evaluate ``q_over_p_mode=2`` propagation records against truth states."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        available = {str(name).split(";")[0] for name in tree.keys()}
        missing = [name for name in REQUIRED_BRANCHES if name not in available]
        if missing:
            raise ValueError("missing truth propagation branches: " + ", ".join(missing))
        raw = tree.arrays(REQUIRED_BRANCHES, library="ak")

    rejected: Counter[str] = Counter()
    accepted: dict[str, list[int | float | np.ndarray]] = {
        "source_station": [],
        "target_station": [],
        "residual": [],
        "target_z_difference": [],
    }
    for event_row in range(len(raw["eventID"])):
        propagation = {
            name: np.asarray(ak.to_numpy(raw[name][event_row]))
            for name in REQUIRED_BRANCHES
            if name.startswith("TrackletPropagation_")
        }
        truth_barcodes = np.asarray(ak.to_numpy(raw["truth_barcode"][event_row]), dtype=np.int64)
        barcode_to_index = {int(barcode): index for index, barcode in enumerate(truth_barcodes)}
        if len(barcode_to_index) != truth_barcodes.size:
            rejected["duplicate_truth_barcode"] += 1
            continue
        truth_values = {
            name: np.asarray(ak.to_numpy(raw[name][event_row]), dtype=np.float64)
            for name in REQUIRED_BRANCHES
            if name.startswith("truth_st")
        }

        for row in range(propagation["TrackletPropagation_q_over_p_mode"].size):
            rejected["total_records"] += 1
            if int(propagation["TrackletPropagation_q_over_p_mode"][row]) != 2:
                rejected["not_truth_to_truth_mode"] += 1
                continue
            if not bool(propagation["TrackletPropagation_success"][row]):
                rejected["propagation_not_successful"] += 1
                continue
            source_station = int(propagation["TrackletPropagation_source_station_id"][row])
            target_station = int(propagation["TrackletPropagation_target_station_id"][row])
            barcode = int(propagation["TrackletPropagation_truth_particle_id"][row])
            if source_station not in range(4) or target_station not in range(4):
                rejected["invalid_station"] += 1
                continue
            if barcode < 0 or barcode not in barcode_to_index:
                rejected["truth_barcode_not_found"] += 1
                continue
            truth = _truth_state(truth_values, target_station, barcode_to_index[barcode])
            if truth is None:
                rejected["invalid_target_truth_state"] += 1
                continue
            truth_state, truth_z = truth
            prediction = np.array(
                [
                    propagation["TrackletPropagation_x_mm"][row],
                    propagation["TrackletPropagation_y_mm"][row],
                    propagation["TrackletPropagation_tx"][row],
                    propagation["TrackletPropagation_ty"][row],
                ],
                dtype=np.float64,
            )
            target_z = float(propagation["TrackletPropagation_target_z_mm"][row])
            if not np.isfinite(prediction).all() or not np.isfinite(target_z):
                rejected["invalid_prediction"] += 1
                continue
            accepted["source_station"].append(source_station)
            accepted["target_station"].append(target_station)
            accepted["residual"].append(truth_state - prediction)
            accepted["target_z_difference"].append(target_z - truth_z)
            rejected["accepted"] += 1

    if not accepted["residual"]:
        return {
            "input": str(path),
            "tree": tree_name,
            "accepted_records": 0,
            "rejected_counts": dict(sorted(rejected.items())),
        }
    residual = np.asarray(accepted["residual"], dtype=np.float64)
    target_z_difference = np.asarray(accepted["target_z_difference"], dtype=np.float64)
    source_station = np.asarray(accepted["source_station"], dtype=np.int16)
    target_station = np.asarray(accepted["target_station"], dtype=np.int16)
    by_station_pair = {}
    for source, target in sorted(set(zip(source_station.tolist(), target_station.tolist()))):
        mask = (source_station == source) & (target_station == target)
        by_station_pair[f"{source}->{target}"] = _summary(
            residual[mask], target_z_difference[mask]
        )
    return {
        "input": str(path),
        "tree": tree_name,
        "q_over_p_mode": 2,
        "comparison": (
            "target MC truth station-average state minus FaserActs propagation from "
            "the source MC truth station-average state"
        ),
        "units": {"x_mm": "mm", "y_mm": "mm", "tx": "1", "ty": "1"},
        "statistical_scope": (
            "Residual-only centre closure: truth covariance is intentionally absent, "
            "so pull and chi2 are not defined for this control."
        ),
        "accepted_records": int(residual.shape[0]),
        "rejected_counts": dict(sorted(rejected.items())),
        "overall": _summary(residual, target_z_difference),
        "by_station_pair": by_station_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Enhanced NtupleDumper ROOT output")
    parser.add_argument("--tree", default="nt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit_truth_field_propagation(args.input, tree_name=args.tree)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
