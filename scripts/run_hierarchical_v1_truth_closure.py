#!/usr/bin/env python3
"""Truth-selected hierarchical V1 block update: station 5-DoF or 1-D C_dx.

Exactly one level is floated.  The other stays at the current geometry and is
not a nuisance column.  Optional ``--observed-iteration-manifest`` keeps the
Jacobian on the iteration-00 finite-difference bank while measuring the
residual on a later remaining physical geometry.  Capture criteria are the
already-frozen station 5-DoF or 1-D C_dx contracts; this command only
evaluates them.  Test stays sealed.
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
from alignment.calibration_modes import (
    DEFAULT_CONTRACT_RELATIVE,
    IFT_INTERNAL_MODE,
    STATION_MODE,
    evaluate_mode_validity,
    load_mode_validity_contract,
    normalize_mode,
    read_station_framework_capture_success,
)
from alignment.five_dof_sampling import five_dof_severity
from alignment.hierarchical_v1 import (
    C_DX,
    HIERARCHICAL_V1_PARAMETERS,
    LAYER_LEVEL,
    STATION_FREE_PARAMETERS,
    STATION_LEVEL,
    STATION_SOLVE_PARAMETERS,
    STATION_SURVEY_PARAMETER,
    complete_hierarchical_values,
    layer_transforms_for_cdx,
    remaining_after_level_step,
    station_absorption_of_cdx,
    station_payload_stability,
)
from scripts.audit_6dof_identifiability import _fit_summary, _source_entries
from scripts.audit_layer_identifiability import (
    _bank_for_target,
    _fit,
    _json_ready,
    _pool_layer_banks,
    _source_bank,
)
from scripts.run_refit_multidof_closure import _read_json


SCHEMA_VERSION = "faser-ift-hierarchical-v1-truth-selected-v1"
STATION_ONLY = STATION_SOLVE_PARAMETERS
LAYER_ONLY = (C_DX,)


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


def _parse_prior(values: Sequence[str] | None, names: Sequence[str]) -> np.ndarray | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, raw_value = str(raw).partition(":")
        if not separator or not name:
            raise ValueError("--prior-sigma entries must have form PARAMETER:VALUE")
        parsed[name] = float(raw_value)
    unknown = set(parsed) - set(names)
    if unknown:
        raise ValueError("unknown --prior-sigma: " + ", ".join(sorted(unknown)))
    return np.asarray([parsed[name] if name in parsed else math.nan for name in names], dtype=np.float64)


def _floated_level(names: Sequence[str]) -> str:
    if tuple(names) == LAYER_ONLY:
        return LAYER_LEVEL
    if C_DX in names:
        raise ValueError("do not float station 5-DoF and C_dx in one Newton step")
    if set(names) <= set(STATION_SOLVE_PARAMETERS) and set(STATION_FREE_PARAMETERS) <= set(names):
        return STATION_LEVEL
    raise ValueError("only-parameters must be station 5-DoF (+ survey dz) or C_dx")


def _injected_hierarchical(entry: Mapping[str, Any], target_point: str) -> dict[str, float]:
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
        return complete_hierarchical_values({name: float(values[name]) for name in HIERARCHICAL_V1_PARAMETERS})
    raise ValueError(f"scan plan lacks target '{target_point}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capture-criteria", required=True)
    parser.add_argument(
        "--calibration-mode",
        choices=(STATION_MODE, IFT_INTERNAL_MODE, "C_dx"),
        default=None,
    )
    parser.add_argument("--mode-validity-contract", default=None)
    parser.add_argument("--declared-unmodeled-cdx-um", type=float, default=None)
    parser.add_argument(
        "--cdx-fixed-by",
        choices=("external_geometry", "dedicated_calibration", "isolation_zero"),
        default=None,
    )
    parser.add_argument("--station-capture-artifact", default=None)
    parser.add_argument("--station-framework-capture-success", choices=("true", "false"), default=None)
    parser.add_argument("--same-data-stage-as-other-mode", action="store_true")
    parser.add_argument("--require-mode-valid", action="store_true")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--anchor-point", default="iteration_00_reference")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--only-parameters", nargs="+", required=True)
    parser.add_argument("--prior-sigma", action="append", default=None)
    parser.add_argument("--observed-iteration-manifest", default=None)
    args = parser.parse_args()
    names = tuple(str(name) for name in args.only_parameters)
    if len(set(names)) != len(names) or not names:
        raise ValueError("only-parameters must be a unique non-empty subset")
    floated_level = _floated_level(names)
    calibration_mode = None if args.calibration_mode is None else normalize_mode(args.calibration_mode)
    mode_contract = None
    mode_contract_path = None
    if calibration_mode is not None:
        from alignment.calibration_modes import assert_exclusive_parameters

        expected = STATION_LEVEL if calibration_mode == STATION_MODE else LAYER_LEVEL
        if floated_level != expected:
            raise ValueError(f"--calibration-mode {calibration_mode} does not match floated parameters")
        mode_contract_path = (
            Path(args.mode_validity_contract).expanduser().resolve()
            if args.mode_validity_contract
            else Path(__file__).resolve().parents[1] / DEFAULT_CONTRACT_RELATIVE
        )
        mode_contract = load_mode_validity_contract(mode_contract_path)
        assert_exclusive_parameters(calibration_mode, names)
    if floated_level == STATION_LEVEL and STATION_SURVEY_PARAMETER in names and not args.prior_sigma:
        raise ValueError("station hierarchical V1 step requires --prior-sigma ift_dz_mm:5.0")
    if floated_level == LAYER_LEVEL and args.prior_sigma:
        raise ValueError("C_dx step does not take a survey prior")
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
    entries = [entry for entry in _source_entries(manifest) if str(entry.get("split")) == str(args.split)]
    if not entries:
        raise ValueError(f"no {args.split} sources in the iteration manifest")
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
    injected = _injected_hierarchical(inject_entry, str(args.target_point))
    pooled = _bank_for_target(_pool_layer_banks(banks), str(args.target_point))
    prior = _parse_prior(args.prior_sigma, names)
    fit = _fit(pooled, rcond=float(args.rcond), names=names, scales=pooled["scales"], prior_sigma_native=prior)
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
        prior_sigma_native=prior,
    )
    per_source = []
    for bank in banks:
        source_fit = _fit(
            _bank_for_target(bank, str(args.target_point)),
            rcond=float(args.rcond),
            names=names,
            scales=bank["scales"],
            prior_sigma_native=prior,
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
    source_spread = {name: _source_spread([row["recovered"][name] for row in per_source]) for name in names}
    remaining = remaining_after_level_step(injected=injected, recovered=recovered, floated_level=floated_level)
    pre_rms = _rms(fit.response)
    post_rms = _rms(fit.residual_response)
    leakage_station_absorbs_cdx = None
    leakage_layer_moves_station = None
    if floated_level == STATION_LEVEL:
        leakage_station_absorbs_cdx = station_absorption_of_cdx(
            injected=injected, recovered_station=recovered
        )
    else:
        leakage_layer_moves_station = station_payload_stability(injected, remaining)
    mode_validity = None
    if mode_contract is not None and calibration_mode is not None:
        if args.declared_unmodeled_cdx_um is not None:
            unmodeled_um = float(args.declared_unmodeled_cdx_um)
        else:
            unmodeled_um = abs(float(injected[C_DX])) * 1.0e3
        cdx_fixed_by = args.cdx_fixed_by
        if cdx_fixed_by is None and unmodeled_um == 0.0:
            cdx_fixed_by = "isolation_zero"
        station_capture = None
        if args.station_framework_capture_success is not None:
            station_capture = args.station_framework_capture_success == "true"
        elif args.station_capture_artifact is not None:
            station_capture = read_station_framework_capture_success(
                Path(args.station_capture_artifact).expanduser().resolve()
            )
        mode_validity = evaluate_mode_validity(
            mode_contract,
            mode=calibration_mode,
            floated_parameters=names,
            unmodeled_abs_C_dx_um=unmodeled_um,
            cdx_fixed_by=cdx_fixed_by,
            station_framework_capture_success=station_capture,
            same_data_stage_as_other_mode=bool(args.same_data_stage_as_other_mode),
        )
        mode_validity = {**mode_validity, "contract": str(mode_contract_path)}
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
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
        "fit_basis": "hierarchical_v1",
        "joint_station_cdx_newton": False,
        "calibration_mode": calibration_mode,
        "mode_validity": mode_validity,
        "geometry_write_allowed": None if mode_validity is None else bool(mode_validity["geometry_write_allowed"]),
        "residual_reduction_is_not_alignment_success": True,
        "block_coordinate": True,
        "floated_level": floated_level,
        "fixed_level": LAYER_LEVEL if floated_level == STATION_LEVEL else STATION_LEVEL,
        "only_parameters": list(names),
        "payload_expansion": "station 5-DoF + dz=0; L0=+C_dx, L1=0, L2=-C_dx; C_rx identically 0",
        "sources": [str(entry["source_id"]) for entry in entries],
        "fit": _fit_summary(fit, names),
        "parameters": parameter_rows,
        "prior_audit": prior_rows,
        "capture_aggregate": capture_aggregate,
        "capture_criteria": str(Path(args.capture_criteria).expanduser().resolve()),
        "injected_hierarchical_v1": injected,
        "recovered": recovered,
        "remaining_hierarchical_v1": remaining,
        "remaining_layer_transforms": {"0": layer_transforms_for_cdx(remaining[C_DX])},
        "proposed_next_is_not_remaining": True,
        "station_five_dof_severity_injected": five_dof_severity(injected),
        "station_five_dof_severity_remaining": five_dof_severity(remaining),
        "leakage_station_absorbs_C_dx": leakage_station_absorbs_cdx,
        "leakage_layer_moves_station": leakage_layer_moves_station,
        "per_source": per_source,
        "source_spread": source_spread,
        "unique_truth_pairs": int(fit.used_pairs),
        "prefit_residual_rms": pre_rms,
        "postfit_residual_rms": post_rms,
        "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else float("nan"),
        "capture_success": None if capture_aggregate is None else bool(capture_aggregate["capture_success"]),
    }
    (output / "hierarchical_v1_truth_selected.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "split": str(args.split),
                "floated_level": floated_level,
                "capture_success": report["capture_success"],
                "post_over_pre_rms": report["post_over_pre_rms"],
                "remaining_C_dx": remaining[C_DX],
                "station_five_dof_severity_remaining": report["station_five_dof_severity_remaining"],
                "calibration_mode": calibration_mode,
                "geometry_write_allowed": report["geometry_write_allowed"],
                "cross_level_contaminated": (
                    None if mode_validity is None else bool(mode_validity["cross_level_contaminated"])
                ),
            },
            indent=2,
        )
    )
    if args.require_mode_valid and mode_validity is not None and not bool(mode_validity["geometry_write_allowed"]):
        raise ValueError(
            "mode-validity contract forbids writing this geometry: " + str(mode_validity.get("status"))
        )


if __name__ == "__main__":
    main()
