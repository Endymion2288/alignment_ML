#!/usr/bin/env python3
"""Audit exported local-segment states against matched MC station truth.

This intentionally reads the enhanced NtupleDumper output rather than the
canonical flat file.  The per-tracklet truth barcode is joined to the existing
``truth_*`` station arrays in the same event, which preserves the actual
Calypso exporter path used for the audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from geometry.propagation import mahalanobis_chi2


STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")
TRACKLET_COVARIANCE_BRANCHES = (
    "Tracklet_cov_xx_mm2",
    "Tracklet_cov_xy_mm2",
    "Tracklet_cov_xtx_mm",
    "Tracklet_cov_xty_mm",
    "Tracklet_cov_yy_mm2",
    "Tracklet_cov_ytx_mm",
    "Tracklet_cov_yty_mm",
    "Tracklet_cov_txtx",
    "Tracklet_cov_txty",
    "Tracklet_cov_tyty",
)
TRACKLET_BRANCHES = (
    "Tracklet_id",
    "Tracklet_station_id",
    "Tracklet_x_mm",
    "Tracklet_y_mm",
    "Tracklet_z_mm",
    "Tracklet_tx",
    "Tracklet_ty",
    "Tracklet_truth_particle_id",
    "Tracklet_truth_match_fraction",
    *TRACKLET_COVARIANCE_BRANCHES,
)
TRUTH_BRANCHES = (
    "truth_barcode",
    "truth_pdg",
    *(f"truth_st{station}_{component}" for station in range(4) for component in (
        "x", "y", "z", "px", "py", "pz"
    )),
)
REQUIRED_BRANCHES = ("run", "eventID", *TRACKLET_BRANCHES, *TRUTH_BRANCHES)


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


def _covariance(values: dict[str, np.ndarray], row: int) -> np.ndarray:
    xx, xy, xtx, xty, yy, ytx, yty, txtx, txty, tyty = (
        float(values[name][row]) for name in TRACKLET_COVARIANCE_BRANCHES
    )
    return np.array(
        [[xx, xy, xtx, xty], [xy, yy, ytx, yty], [xtx, ytx, txtx, txty], [xty, yty, txty, tyty]],
        dtype=np.float64,
    )


def _valid_covariance(covariance: np.ndarray) -> bool:
    return bool(
        covariance.shape == (4, 4)
        and np.isfinite(covariance).all()
        and np.allclose(covariance, covariance.T, rtol=1.0e-7, atol=1.0e-12)
        and np.all(np.diag(covariance) > 0.0)
    )


def _summary(
    residual: np.ndarray,
    pull: np.ndarray,
    chi2: np.ndarray,
    correlation_condition: np.ndarray,
    reference_delta_z_mm: np.ndarray,
) -> dict[str, object]:
    return {
        "records": int(residual.shape[0]),
        "residual": {
            name: _stats(residual[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "pull": {name: _stats(pull[:, index]) for index, name in enumerate(STATE_NAMES)},
        "chi2": _stats(chi2),
        "correlation_condition_number": _stats(correlation_condition),
        "reco_minus_truth_reference_z_mm": _stats(reference_delta_z_mm),
    }


def audit_local_tracklet_truth(
    input_path: str | Path,
    tree_name: str = "nt",
    min_truth_match_fraction: float = 1.0,
) -> dict[str, object]:
    """Return a truth-state audit for local tracklets in enhanced output."""
    if not 0.0 <= min_truth_match_fraction <= 1.0:
        raise ValueError("min_truth_match_fraction must be in [0, 1]")
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
            raise ValueError("missing audit branches: " + ", ".join(missing))
        raw = tree.arrays(REQUIRED_BRANCHES, library="ak")

    rejected: Counter[str] = Counter()
    accepted: dict[str, list[float | np.ndarray | int]] = {
        "station": [],
        "residual": [],
        "pull": [],
        "chi2": [],
        "correlation_condition": [],
        "reference_delta_z_mm": [],
    }
    for event_row in range(len(raw["eventID"])):
        tracklet_values = {
            name: np.asarray(ak.to_numpy(raw[name][event_row])) for name in TRACKLET_BRANCHES
        }
        truth_barcodes = np.asarray(ak.to_numpy(raw["truth_barcode"][event_row]), dtype=np.int64)
        barcode_to_index = {int(barcode): index for index, barcode in enumerate(truth_barcodes)}
        if len(barcode_to_index) != truth_barcodes.size:
            rejected["duplicate_truth_barcode"] += 1
            continue

        truth_values = {
            name: np.asarray(ak.to_numpy(raw[name][event_row]), dtype=np.float64)
            for name in TRUTH_BRANCHES
            if name != "truth_barcode" and name != "truth_pdg"
        }
        for row in range(tracklet_values["Tracklet_id"].size):
            rejected["total_tracklets"] += 1
            station = int(tracklet_values["Tracklet_station_id"][row])
            barcode = int(tracklet_values["Tracklet_truth_particle_id"][row])
            fraction = float(tracklet_values["Tracklet_truth_match_fraction"][row])
            if station not in range(4):
                rejected["invalid_station"] += 1
                continue
            if barcode < 0 or barcode not in barcode_to_index:
                rejected["truth_barcode_not_found"] += 1
                continue
            if not np.isfinite(fraction) or fraction < min_truth_match_fraction:
                rejected["truth_match_fraction_below_threshold"] += 1
                continue

            truth_row = barcode_to_index[barcode]
            truth_x = truth_values[f"truth_st{station}_x"][truth_row]
            truth_y = truth_values[f"truth_st{station}_y"][truth_row]
            truth_z = truth_values[f"truth_st{station}_z"][truth_row]
            truth_px = truth_values[f"truth_st{station}_px"][truth_row]
            truth_py = truth_values[f"truth_st{station}_py"][truth_row]
            truth_pz = truth_values[f"truth_st{station}_pz"][truth_row]
            reco_z = float(tracklet_values["Tracklet_z_mm"][row])
            if (
                not np.isfinite([truth_x, truth_y, truth_z, truth_px, truth_py, truth_pz, reco_z]).all()
                or abs(truth_pz) < 1.0e-12
            ):
                rejected["invalid_truth_state"] += 1
                continue

            truth_tx = truth_px / truth_pz
            truth_ty = truth_py / truth_pz
            # Truth positions/momenta are station averages.  Shift this local
            # state to the segment's declared reference z before comparing it.
            delta_z = reco_z - truth_z
            truth_state = np.array(
                [truth_x + truth_tx * delta_z, truth_y + truth_ty * delta_z, truth_tx, truth_ty],
                dtype=np.float64,
            )
            reco_state = np.array(
                [
                    tracklet_values["Tracklet_x_mm"][row],
                    tracklet_values["Tracklet_y_mm"][row],
                    tracklet_values["Tracklet_tx"][row],
                    tracklet_values["Tracklet_ty"][row],
                ],
                dtype=np.float64,
            )
            if not np.isfinite(reco_state).all():
                rejected["invalid_reconstructed_state"] += 1
                continue
            covariance = _covariance(tracklet_values, row)
            if not _valid_covariance(covariance):
                rejected["invalid_reconstructed_covariance"] += 1
                continue
            residual = reco_state - truth_state
            try:
                chi2 = mahalanobis_chi2(residual, covariance)
            except ValueError:
                rejected["singular_reconstructed_covariance"] += 1
                continue
            diagonal_sigma = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(diagonal_sigma, diagonal_sigma)
            accepted["station"].append(station)
            accepted["residual"].append(residual)
            accepted["pull"].append(residual / diagonal_sigma)
            accepted["chi2"].append(chi2)
            accepted["correlation_condition"].append(float(np.linalg.cond(correlation)))
            accepted["reference_delta_z_mm"].append(delta_z)
            rejected["accepted"] += 1

    if not accepted["residual"]:
        return {
            "input": str(path),
            "tree": tree_name,
            "min_truth_match_fraction": min_truth_match_fraction,
            "accepted_records": 0,
            "rejected_counts": dict(sorted(rejected.items())),
        }
    residual = np.asarray(accepted["residual"], dtype=np.float64)
    pull = np.asarray(accepted["pull"], dtype=np.float64)
    chi2 = np.asarray(accepted["chi2"], dtype=np.float64)
    correlation_condition = np.asarray(accepted["correlation_condition"], dtype=np.float64)
    reference_delta_z = np.asarray(accepted["reference_delta_z_mm"], dtype=np.float64)
    station = np.asarray(accepted["station"], dtype=np.int16)
    by_station = {}
    for station_id in sorted(set(map(int, station.tolist()))):
        mask = station == station_id
        by_station[str(station_id)] = _summary(
            residual[mask], pull[mask], chi2[mask], correlation_condition[mask], reference_delta_z[mask]
        )
    return {
        "input": str(path),
        "tree": tree_name,
        "min_truth_match_fraction": min_truth_match_fraction,
        "reference_convention": (
            "MC truth station-average state is linearly shifted to Tracklet_z_mm; "
            "this removes the small within-station reference-z difference only."
        ),
        "accepted_records": int(residual.shape[0]),
        "rejected_counts": dict(sorted(rejected.items())),
        "overall": _summary(residual, pull, chi2, correlation_condition, reference_delta_z),
        "by_station": by_station,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Enhanced NtupleDumper ROOT output")
    parser.add_argument("--tree", default="nt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-truth-match-fraction", type=float, default=1.0)
    args = parser.parse_args()
    result = audit_local_tracklet_truth(
        args.input,
        tree_name=args.tree,
        min_truth_match_fraction=args.min_truth_match_fraction,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
