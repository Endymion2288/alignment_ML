#!/usr/bin/env python3
"""Build the 2-column C_dx/C_rx Jacobian from existing layer0/layer2 central FD.

This command reuses the identifiability pilot's already-produced layer0/layer2
dx and rx probes.  It does not reproduce the 40-parameter map, does not open
validation, and does not open the sealed test.  The two contrast columns are

    J_C_dx = J_L0_dx - J_L2_dx
    J_C_rx = J_L0_rx - J_L2_rx

matching the payload expansion L0=+C, L2=-C.  Column cosine, rank, and
condition number are the pre-flight degeneracy diagnostics for the 2-D
mini-curriculum.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.layer_hierarchy import NEAR_DEGENERACY_COSINE
from scripts.audit_6dof_identifiability import _column_cosines, _fit_summary, _source_entries
from scripts.audit_layer_identifiability import _fit, _json_ready, _pool_layer_banks, _source_bank
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-ift-layer-contrast-2d-jacobian-from-existing-fd-v1"
LAYER_FD_PARAMETERS = (
    "ift_layer0_dx_mm",
    "ift_layer2_dx_mm",
    "ift_layer0_rx_mrad",
    "ift_layer2_rx_mrad",
)
CONTRAST_NAMES = ("C_dx", "C_rx")
CONTRAST_SCALES = np.asarray([0.12, 0.70], dtype=np.float64)


def _contrast_column_transform() -> np.ndarray:
    """Map (C_dx, C_rx) onto (L0_dx, L2_dx, L0_rx, L2_rx) as L0=+C, L2=-C."""
    transform = np.zeros((4, 2), dtype=np.float64)
    transform[0, 0] = 1.0
    transform[1, 0] = -1.0
    transform[2, 1] = 1.0
    transform[3, 1] = -1.0
    return transform


def _flattened_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-point", default="iteration_00_reference")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("identifiability manifest accessed test data")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("identifiability manifest is not mode-0")
    entries = _source_entries(manifest)
    if any(str(entry.get("split")) != "train" for entry in entries):
        raise ValueError("contrast Jacobian pre-flight must use only the existing train FD sources")
    banks = [
        _source_bank(
            entry,
            anchor_point=str(args.anchor_point),
            target_points=(str(args.anchor_point),),
            min_truth_match_fraction=float(args.min_truth_match_fraction),
            only_parameters=LAYER_FD_PARAMETERS,
        )
        for entry in entries
    ]
    pooled = _pool_layer_banks(banks)
    transform = _contrast_column_transform()
    fit = _fit(
        pooled,
        rcond=1.0e-10,
        names=CONTRAST_NAMES,
        scales=CONTRAST_SCALES,
        column_transform=transform,
    )
    cosines = _column_cosines(fit)
    unweighted = _flattened_cosine(fit.derivative_native[:, :, 0], fit.derivative_native[:, :, 1])
    information_cosine = float(cosines[0, 1])
    degenerate = bool(
        abs(information_cosine) >= NEAR_DEGENERACY_COSINE or abs(unweighted) >= NEAR_DEGENERACY_COSINE
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration_manifest": str(manifest_path),
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "validation_used": False,
        "reproduced_40_identifiability_probes": False,
        "layer_fd_parameters": list(LAYER_FD_PARAMETERS),
        "contrast_parameters": list(CONTRAST_NAMES),
        "payload_expansion": "L0=(+C_dx,+C_rx), L1=0, L2=(-C_dx,-C_rx); station six-vector identically 0",
        "jacobian_map": "J_C = J_L0 - J_L2",
        "sources": [str(entry["source_id"]) for entry in entries],
        "observations": int(fit.used_pairs),
        "fit": _fit_summary(fit, CONTRAST_NAMES),
        "column_cosine_information": information_cosine,
        "column_cosine_unweighted": unweighted,
        "near_degeneracy_cosine": NEAR_DEGENERACY_COSINE,
        "near_degenerate": degenerate,
        "rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "condition_number": fit.normal_matrix_condition_number,
        "passed_preflight": bool(fit.full_rank and int(fit.normal_matrix_rank) == 2 and not degenerate),
        "note": (
            "Pre-flight only: this 3-source Jacobian is not transferred onto the 10/8 curriculum. "
            "The 10/8 bank measures its own four contrast-space FD points."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "observations": report["observations"],
                "column_cosine_information": information_cosine,
                "column_cosine_unweighted": unweighted,
                "rank": report["rank"],
                "condition_number": report["condition_number"],
                "near_degenerate": degenerate,
                "passed_preflight": report["passed_preflight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
