#!/usr/bin/env python3
"""Small-scale joint physical closure on the admitted 6-DoF subset.

Solves the local multi-parameter update from the pilot anchor toward one
held-out joint closure payload with truth-selected physical edges only.
Only parameters passed via ``--admit`` enter the finite-difference Jacobian;
the held-out point injects every DoF jointly, so parameters outside the
admitted set appear as unmodelled injection (negligible exactly when the
identifiability gate excluded them for weak/degenerate response).

Reports per-parameter recovered local delta versus the injected delta
(closure value minus anchor value), the native data sigma, the closure pull,
and the post-fit residual response, pooled and per source.  This script never
touches association outputs, synthetic overlays, or test data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.audit_6dof_identifiability import (
    _fit,
    _pooled_bank,
    _source_bank,
    _source_entries,
)
from scripts.run_refit_multidof_closure import _point_map, _read_json


SCHEMA_VERSION = "faser-station0-6dof-joint-closure-v1"


def _closure_values(manifest: Mapping[str, object], *, target_point: str) -> dict[str, float]:
    """Read the injected parameter values of the held-out closure point."""
    for entry in _source_entries(manifest):
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        point = _point_map(plan).get(target_point)
        if point is None:
            raise ValueError(f"scan plan lacks closure target '{target_point}'")
        values = point.get("alignment_parameter_values")
        if not isinstance(values, Mapping):
            raise ValueError(f"closure target '{target_point}' lacks alignment_parameter_values")
        return {str(key): float(value) for key, value in values.items()}
    raise ValueError("iteration manifest has no sources")


def _closure_report(fit, names: Sequence[str], anchor: np.ndarray, injected: Mapping[str, float]) -> dict[str, Any]:
    expected = np.asarray([float(injected[name]) - float(anchor[index]) for index, name in enumerate(names)])
    recovered = np.asarray(fit.recovered_parameters, dtype=np.float64)
    error = recovered - expected
    diagonal = np.diag(fit.covariance_native)
    sigma = np.asarray(
        [math.sqrt(value) if math.isfinite(value) and value >= 0.0 else math.nan for value in diagonal],
        dtype=np.float64,
    )
    pull = np.where(np.isfinite(sigma) & (sigma > 0.0), error / sigma, np.nan)
    return {
        "observations": int(fit.used_pairs),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "post_fit_residual_rms": {
            label: float(np.sqrt(np.mean(np.square(fit.residual_response[:, index]))))
            for index, label in enumerate(("rx_mm", "ry_mm", "rtx", "rty"))
        },
        "parameters": {
            name: {
                "anchor_value": float(anchor[index]),
                "injected_value": float(injected[name]),
                "expected_local_delta": float(expected[index]),
                "recovered_local_delta": float(recovered[index]),
                "closure_error": float(error[index]),
                "sigma_native": None if not np.isfinite(sigma[index]) else float(sigma[index]),
                "closure_pull": None if not np.isfinite(pull[index]) else float(pull[index]),
            }
            for index, name in enumerate(names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--closure-target", required=True)
    parser.add_argument("--admit", action="append", required=True, metavar="PARAMETER")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    args = parser.parse_args()

    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    admitted = [str(name) for name in args.admit]
    injected = _closure_values(manifest, target_point=str(args.closure_target))

    banks = [
        _source_bank(
            entry,
            anchor_point=str(args.anchor_point),
            target_point=str(args.closure_target),
            min_truth_match_fraction=float(args.min_truth_match_fraction),
            only_parameters=admitted,
        )
        for entry in _source_entries(manifest)
    ]
    pooled = _pooled_bank(banks)
    names = pooled["names"]
    missing_injection = [name for name in names if name not in injected]
    if missing_injection:
        raise ValueError(f"closure target lacks injected values for: {', '.join(missing_injection)}")

    pooled_fit = _fit(pooled, rcond=float(args.rcond))
    anchor_values = np.asarray(pooled["anchor_values"], dtype=np.float64)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration_manifest": str(manifest_path),
        "anchor_point": str(args.anchor_point),
        "closure_target": str(args.closure_target),
        "admitted_parameters": list(names),
        "observation_semantics": "truth_selected_physical_edge",
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "pooled": _closure_report(pooled_fit, names, anchor_values, injected),
        "per_source": [
            {
                "source_id": str(bank["source_id"]),
                **_closure_report(_fit(bank, rcond=float(args.rcond)), names, anchor_values, injected),
            }
            for bank in banks
        ],
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"joint_closure_{args.closure_target}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"closure target: {args.closure_target}; admitted: {', '.join(names)}")
    for name in names:
        row = report["pooled"]["parameters"][name]
        pull = row["closure_pull"]
        print(
            f"  {name}: injected {row['expected_local_delta']:+.4f}, recovered "
            f"{row['recovered_local_delta']:+.4f}, error {row['closure_error']:+.4f}, "
            f"pull {'n/a' if pull is None else f'{pull:+.2f}'}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
