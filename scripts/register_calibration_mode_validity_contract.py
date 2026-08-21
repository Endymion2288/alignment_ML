#!/usr/bin/env python3
"""Freeze the two exclusive calibration modes' validity contract from workbook 45.

Reads the already-computed leakage audit (train route-selected production A6),
the frozen station 5-DoF capture criteria, and the C_dx row of the contrast-2d
criteria.  Validation never participates.  The sealed test is not opened.
A and capture sigmas are not retuned.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.calibration_modes import A_STABILITY_MAX_REL_DEVIATION, build_mode_validity_contract
from alignment.capture_criteria import load_capture_criteria
from alignment.hierarchical_v1 import C_DX, STATION_SOLVE_PARAMETERS
from alignment.hierarchical_v1_leakage import reverse_leakage_from_A_and_covariance
from alignment.layer_contrast_capture import load_layer_contrast_capture_criteria


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "outputs/mc24_ift_hierarchical_v1_leakage_identifiability_audit_v1/hierarchical_v1_leakage_audit.json"
DEFAULT_SUMMARY = ROOT / "outputs/mc24_ift_hierarchical_v1_leakage_identifiability_audit_v1/summary.json"
DEFAULT_STATION = ROOT / "outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json"
DEFAULT_CDX = ROOT / "outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/capture_criteria.json"
DEFAULT_OUTPUT = ROOT / "outputs/mc24_ift_calibration_mode_validity_contract_v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return dict(payload)


def _primary_corner(audit: Mapping[str, Any]) -> dict[str, Any]:
    for corner in audit.get("corners", ()):
        if (
            corner.get("observation_kind") == "route_selected"
            and corner.get("split") == "train"
            and corner.get("target") == "iteration_00_start"
            and corner.get("geometry") == "iteration_00"
        ):
            return dict(corner)
    raise ValueError("leakage audit lacks the train route-selected iteration_00_start primary corner")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leakage-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--leakage-summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--station-capture-criteria", default=str(DEFAULT_STATION))
    parser.add_argument("--cdx-capture-criteria", default=str(DEFAULT_CDX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    audit_path = Path(args.leakage_audit).expanduser().resolve()
    summary_path = Path(args.leakage_summary).expanduser().resolve()
    station_path = Path(args.station_capture_criteria).expanduser().resolve()
    cdx_path = Path(args.cdx_capture_criteria).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)

    audit = _read_json(audit_path)
    if audit.get("test_data_accessed") is not False:
        raise ValueError("leakage audit accessed test data")
    summary = _read_json(summary_path)
    station_criteria = load_capture_criteria(station_path)
    cdx_criteria = load_layer_contrast_capture_criteria(cdx_path)
    if station_criteria.get("validation_used_in_registration") is not False:
        raise ValueError("station capture criteria used validation")
    if cdx_criteria.get("validation_used_in_registration") is not False:
        raise ValueError("C_dx capture criteria used validation")

    operator = audit["primary_leakage_operator_production_six_dof"]
    corner = _primary_corner(audit)
    ordinary_cov = corner["result"]["pooled"]["ordinary_station"]["fit"]["parameter_covariance_native"]
    reverse_b = reverse_leakage_from_A_and_covariance(
        A_native_per_mm=operator["A_native_per_mm_C_dx"],
        station_covariance_native=ordinary_cov,
        normal_cc_native=float(operator["normal_cc_native"]),
        station_names=STATION_SOLVE_PARAMETERS,
    )
    station_sigma = {
        name: float(station_criteria["per_parameter"][name]["registered_sigma"])
        for name in STATION_SOLVE_PARAMETERS
    }
    realizability = summary["realizability"]
    a_spread = summary["decision"]["A6_dx_across_iteration00_corners"]
    ry_spread = summary["decision"]["A6_ry_across_iteration00_corners"]
    contract = build_mode_validity_contract(
        leakage_operator=operator,
        reverse_B_native=reverse_b,
        station_capture_path=station_path,
        cdx_capture_path=cdx_path,
        station_registered_sigma=station_sigma,
        cdx_registered_sigma_mm=float(cdx_criteria["per_parameter"][C_DX]["registered_sigma"]),
        statistical_max_abs_C_dx_um=float(realizability["binding_statistical_C_dx_um"]),
        engineering_max_abs_C_dx_um=float(realizability["binding_engineering_C_dx_um"]),
        A_stability={
            "freeze_point": "train_route_selected_iteration_00_start_A6",
            "observed_A_dx_min": float(a_spread["min"]),
            "observed_A_dx_max": float(a_spread["max"]),
            "observed_A_dx_rel_spread": float(a_spread["rel_spread"]),
            "observed_A_ry_min": float(ry_spread["min"]),
            "observed_A_ry_max": float(ry_spread["max"]),
            "observed_A_ry_rel_spread": float(ry_spread["rel_spread"]),
            "transfer_max_rel_deviation": A_STABILITY_MAX_REL_DEVIATION,
            "do_not_retune": True,
        },
        provenance={
            "workbook": 45,
            "leakage_audit": str(audit_path),
            "leakage_summary": str(summary_path),
            "station_capture_criteria": str(station_path),
            "cdx_capture_criteria": str(cdx_path),
            "primary_corner": {
                "split": "train",
                "observation_kind": "route_selected",
                "target": "iteration_00_start",
                "geometry": "iteration_00",
            },
        },
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "mode_validity_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_out = {
        "output_dir": str(output),
        "schema_version": contract["schema_version"],
        "A_dx": contract["leakage_operator"]["A_native_per_mm_C_dx"]["ift_dx_mm"],
        "station_mode_statistical_max_abs_C_dx_um": contract["station_mode"]["unmodeled_C_dx"]["statistical_max_abs_um"],
        "station_mode_engineering_max_abs_C_dx_um": contract["station_mode"]["unmodeled_C_dx"]["engineering_max_abs_um"],
        "ift_internal_C_dx_systematic_rss_1sigma_um": contract["ift_internal_mode"]["station_to_C_dx_propagation"][
            "free_station_rss"
        ]["rss_1sigma_um"],
        "registered_C_dx_sigma_um": contract["station_mode"]["prerequisite"]["registered_C_dx_sigma_um"],
        "hierarchical_v1_rescue_closed": True,
        "test_data_accessed": False,
        "validation_used_in_registration": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
