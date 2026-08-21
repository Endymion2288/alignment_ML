#!/usr/bin/env python3
"""Truth-selected outer-contrast update on a contrast-space physical bank.

Default remains the historical 2-D C_dx+C_rx Newton step.  Sequential
hierarchy alignment passes exactly one of ``--only-parameters C_dx`` or
``C_rx``; the other contrast is not a nuisance column.  Optional
``--observed-iteration-manifest`` keeps the Jacobian on the iteration-00
finite-difference bank while measuring the residual on a later remaining
physical geometry.  Station six-vectors stay identically zero.
Frozen-station reference_layer is not solved.  Capture criteria must have
been frozen from the 3-source train 1-D route-selected results before any
10/8 numbers are produced; this command only evaluates them.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.capture_criteria import attach_capture_and_prior, load_capture_criteria
from alignment.contrast_sampling import CONTRAST_PARAMETERS
from alignment.layer_hierarchy import NEAR_DEGENERACY_COSINE, contrast_layer_six_vectors
from alignment.sequential_contrast import remaining_after_block_step
from scripts.audit_6dof_identifiability import _column_cosines, _fit_summary, _source_entries
from scripts.audit_layer_identifiability import (
    _bank_for_target,
    _fit,
    _json_ready,
    _pool_layer_banks,
    _source_bank,
)
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-ift-layer-contrast-2d-truth-selected-v1"
SCHEMA_VERSION_1D = "faser-ift-layer-contrast-1d-sequential-truth-selected-v1"


def _rms(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def _source_spread(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.max(array) - np.min(array))


def _injected_contrast(entry: Mapping[str, Any], target_point: str) -> dict[str, float]:
    root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    for point in plan.get("points", ()):
        if not isinstance(point, Mapping):
            continue
        if str(point.get("name")) != str(target_point):
            continue
        values = point.get("alignment_parameter_values")
        if not isinstance(values, Mapping):
            raise ValueError(f"target '{target_point}' lacks alignment_parameter_values")
        present = {str(key): float(values[key]) for key in values}
        missing = [name for name in CONTRAST_PARAMETERS if name not in present]
        if missing == ["C_rx"]:
            present["C_rx"] = 0.0
            missing = []
        if missing:
            raise ValueError(f"target '{target_point}' is missing " + ", ".join(missing))
        return {name: float(present[name]) for name in CONTRAST_PARAMETERS}
    raise ValueError(f"scan plan lacks target '{target_point}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capture-criteria", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--anchor-point", default="iteration_00_reference")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument(
        "--only-parameters",
        nargs="+",
        default=None,
        help="Float only these contrast coordinates (default: both C_dx and C_rx).",
    )
    parser.add_argument(
        "--observed-iteration-manifest",
        default=None,
        help=(
            "Optional remaining/held-out-only bank that contains the target point. "
            "Finite-difference probes stay on --iteration-manifest.  Required so a "
            "sequential 1-D step measures the new physical residual instead of "
            "algebraically subtracting on iteration-00 data."
        ),
    )
    args = parser.parse_args()
    manifest = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest accessed test data")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("iteration manifest is not mode-0")
    if "test" not in set(manifest.get("forbidden_splits", ())):
        raise ValueError("iteration manifest does not seal test data")
    observed_manifest = None
    if args.observed_iteration_manifest is not None:
        observed_manifest = _read_json(Path(args.observed_iteration_manifest).expanduser().resolve())
        if observed_manifest.get("test_data_accessed") is not False:
            raise ValueError("observed iteration manifest accessed test data")
        if int(observed_manifest.get("q_over_p_mode", -1)) != 0:
            raise ValueError("observed iteration manifest is not mode-0")
    criteria = load_capture_criteria(Path(args.capture_criteria).expanduser().resolve())
    if criteria.get("validation_used_in_registration") is not False:
        raise ValueError("capture criteria used validation during registration")
    entries = [
        entry
        for entry in _source_entries(manifest)
        if str(entry.get("split")) == str(args.split)
    ]
    if not entries:
        raise ValueError(f"no {args.split} sources in the iteration manifest")
    if args.only_parameters is None:
        names = tuple(CONTRAST_PARAMETERS)
    else:
        names = tuple(str(name) for name in args.only_parameters)
        unknown = [name for name in names if name not in CONTRAST_PARAMETERS]
        if unknown or len(set(names)) != len(names):
            raise ValueError("only-parameters must be a unique subset of C_dx C_rx")
        if not names:
            raise ValueError("only-parameters must not be empty")
    observed_by_id: dict[str, Any] | None = None
    if observed_manifest is not None:
        from scripts.run_layer_linear_internal_closure import _combined_source_bank

        observed_by_id = {
            str(entry["source_id"]): entry
            for entry in _source_entries(observed_manifest)
            if str(entry.get("split")) == str(args.split)
        }
        missing = [str(entry["source_id"]) for entry in entries if str(entry["source_id"]) not in observed_by_id]
        if missing:
            raise ValueError("observed bank missing sources: " + ", ".join(missing))
        banks = [
            _combined_source_bank(
                entry,
                observed_by_id[str(entry["source_id"])],
                anchor_point=str(args.anchor_point),
                target_point=str(args.target_point),
                only_parameters=names,
                min_truth_match_fraction=float(args.min_truth_match_fraction),
            )
            for entry in entries
        ]
    else:
        banks = [
            _source_bank(
                entry,
                anchor_point=str(args.anchor_point),
                target_points=(str(args.target_point),),
                min_truth_match_fraction=float(args.min_truth_match_fraction),
                only_parameters=names,
            )
            for entry in entries
        ]
    inject_entry = entries[0] if observed_by_id is None else observed_by_id[str(entries[0]["source_id"])]
    injected = _injected_contrast(inject_entry, str(args.target_point))
    pooled = _bank_for_target(_pool_layer_banks(banks), str(args.target_point))
    fit = _fit(pooled, rcond=float(args.rcond), names=names, scales=pooled["scales"])
    if len(names) == 2:
        information_cosine = float(_column_cosines(fit)[0, 1])
        near_degenerate = bool(abs(information_cosine) >= NEAR_DEGENERACY_COSINE)
    else:
        information_cosine = None
        near_degenerate = False
    expected = {
        name: float(pooled["reference_values"][index] - pooled["anchor_values"][index])
        for index, name in enumerate(names)
    }
    recovered = {name: float(fit.recovered_parameters[index]) for index, name in enumerate(names)}
    errors = {name: recovered[name] - expected[name] for name in names}
    sigmas = []
    for index, name in enumerate(names):
        variance = float(fit.covariance_native[index, index])
        sigmas.append(math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None)
    parameter_rows = [
        {
            "name": name,
            "expected": expected[name],
            "recovered": recovered[name],
            "error": errors[name],
            "sigma": sigmas[index],
            "pull": None if sigmas[index] in (None, 0.0) else errors[name] / float(sigmas[index]),
        }
        for index, name in enumerate(names)
    ]
    parameter_rows, capture_aggregate, prior_rows = attach_capture_and_prior(
        parameter_rows,
        names=names,
        errors=[errors[name] for name in names],
        fit_sigmas=sigmas,
        criteria=criteria,
        normal_matrix_native=fit.normal_matrix_native,
        covariance_native=fit.covariance_native,
        prior_sigma_native=None,
    )
    per_source = []
    for bank in banks:
        source_fit = _fit(
            _bank_for_target(bank, str(args.target_point)),
            rcond=float(args.rcond),
            names=names,
            scales=bank["scales"],
        )
        per_source.append(
            {
                "source_id": bank["source_id"],
                "split": bank["split"],
                "observations": int(source_fit.used_pairs),
                "recovered": {
                    name: float(source_fit.recovered_parameters[index]) for index, name in enumerate(names)
                },
            }
        )
    source_spread = {
        name: _source_spread([row["recovered"][name] for row in per_source]) for name in names
    }
    pre_rms = _rms(fit.response)
    post_rms = _rms(fit.residual_response)
    sequential = len(names) == 1
    remaining = None
    floated = None
    fixed = []
    if sequential:
        floated = names[0]
        remaining = remaining_after_block_step(
            injected=injected,
            recovered_floated=float(recovered[floated]),
            floated=floated,
        )
        fixed = [name for name in CONTRAST_PARAMETERS if name != floated]
        layers = contrast_layer_six_vectors(remaining["C_dx"], remaining["C_rx"])
    else:
        layers = contrast_layer_six_vectors(recovered["C_dx"], recovered["C_rx"])
    station_zero = True
    report = {
        "schema_version": SCHEMA_VERSION_1D if sequential else SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "validation_used_in_registration": False,
        "split": str(args.split),
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "observed_iteration_manifest": (
            None
            if args.observed_iteration_manifest is None
            else str(Path(args.observed_iteration_manifest).expanduser().resolve())
        ),
        "canonical_internal_basis": "outer_contrast",
        "payload_expansion": "L0=(+C_dx,+C_rx), L1=0, L2=(-C_dx,-C_rx); station six-vector identically 0",
        "joint_2d_newton": not sequential,
        "block_coordinate": sequential,
        "floated": floated,
        "fixed": fixed,
        "only_parameters": list(names),
        "sources": [str(entry["source_id"]) for entry in entries],
        "fit": _fit_summary(fit, names),
        "column_cosine_information": information_cosine,
        "near_degenerate": near_degenerate,
        "rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "condition_number": fit.normal_matrix_condition_number,
        "injected_contrast": injected,
        "expected": expected,
        "recovered": recovered,
        "errors": errors,
        "remaining_contrast": remaining,
        "source_spread": source_spread,
        "per_source": per_source,
        "parameters": parameter_rows,
        "prior_audit": prior_rows,
        "capture_aggregate": capture_aggregate,
        "prefit_residual_rms": pre_rms,
        "postfit_residual_rms": post_rms,
        "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else None,
        "expanded_layer_transforms": {"0": layers},
        "station_six_vector_zero": station_zero,
        "layer1_zero": True,
        "reference_layer_used_as_gauge": False,
        "proposed_next_is_not_remaining": sequential,
        "capture_success": None if capture_aggregate is None else bool(capture_aggregate["capture_success"]),
    }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    report_name = "truth_selected_contrast_1d.json" if sequential else "truth_selected_contrast_2d.json"
    (output / report_name).write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output / report_name),
                "split": args.split,
                "target_point": args.target_point,
                "floated": floated,
                "rank": report["rank"],
                "condition_number": report["condition_number"],
                "column_cosine": information_cosine,
                "near_degenerate": report["near_degenerate"],
                "injected_contrast": injected,
                "errors": errors,
                "remaining_contrast": remaining,
                "source_spread": source_spread,
                "post_over_pre_rms": report["post_over_pre_rms"],
                "capture_success": report["capture_success"],
                "joint_2d_newton": not sequential,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
