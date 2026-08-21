#!/usr/bin/env python3
"""Score frozen-inference isolation closures against exclusive calibration modes.

Does not retrain V2, retune capture, or reopen the sealed test.  Residual
reduction is recorded and is never treated as alignment success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from alignment.calibration_modes import (
    C_DX,
    DEFAULT_CONTRACT_RELATIVE,
    IFT_INTERNAL_MODE,
    STATION_MODE,
    evaluate_mode_validity,
    load_mode_validity_contract,
)
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return dict(payload)


def _pull(error: float | None, sigma: float | None) -> float | None:
    if error is None or sigma is None or not math.isfinite(float(sigma)) or float(sigma) == 0.0:
        return None
    return float(error) / float(sigma)


def _association(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = payload.get("association_contract")
    if not isinstance(contract, Mapping):
        return {}
    overlap = contract.get("selected_route_overlap")
    roles = contract.get("target_synthetic_role_counts_by_bank")
    return {
        "selected_route_overlap": overlap,
        "target_synthetic_role_counts_by_bank": roles,
    }


def score_route_selected(path: Path, *, mode: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("test_opened") is True or payload.get("q_over_p_mode") not in (0, None):
        raise ValueError(f"{path} is not a sealed-test-safe mode-0 closure")
    parameters = []
    for row in payload.get("parameters") or ():
        if not isinstance(row, Mapping):
            continue
        error = row.get("local_delta_error")
        sigma = row.get("recovered_sigma")
        parameters.append(
            {
                "name": row.get("name"),
                "error": None if error is None else float(error),
                "fit_sigma": None if sigma is None else float(sigma),
                "pull": _pull(None if error is None else float(error), None if sigma is None else float(sigma)),
            }
        )
    obs = payload.get("observation_statistics") if isinstance(payload.get("observation_statistics"), Mapping) else {}
    if mode == STATION_MODE:
        validity = evaluate_mode_validity(
            contract,
            mode=STATION_MODE,
            floated_parameters=STATION_SOLVE_PARAMETERS,
            unmodeled_abs_C_dx_um=0.0,
            cdx_fixed_by="isolation_zero",
        )
    else:
        validity = evaluate_mode_validity(
            contract,
            mode=IFT_INTERNAL_MODE,
            floated_parameters=[C_DX],
            station_framework_capture_success=True,
        )
    hierarchy = payload.get("hierarchy_internal_audit")
    post_over_pre = None
    if isinstance(hierarchy, Mapping) and hierarchy.get("post_over_pre_rms") is not None:
        post_over_pre = float(hierarchy["post_over_pre_rms"])
    return {
        "path": str(path),
        "calibration_mode": mode,
        "q_over_p_mode": payload.get("q_over_p_mode"),
        "observation_kind": payload.get("observation_kind"),
        "observation_semantics": obs.get("semantics"),
        "unique_physical_edges": obs.get("unique_physical_edges"),
        "replicas_before_deduplication": obs.get("replicas_before_deduplication"),
        "normal_matrix_rank": payload.get("normal_matrix_rank"),
        "normal_matrix_condition_number": (
            None
            if payload.get("normal_matrix_condition_number") is None
            or not math.isfinite(float(payload["normal_matrix_condition_number"]))
            else float(payload["normal_matrix_condition_number"])
        ),
        "identifiable_subspace_condition_number": (
            None
            if payload.get("identifiable_subspace_condition_number") is None
            or not math.isfinite(float(payload["identifiable_subspace_condition_number"]))
            else float(payload["identifiable_subspace_condition_number"])
        ),
        "capture_success": payload.get("capture_success"),
        "capture_aggregate": payload.get("capture_aggregate"),
        "parameters": parameters,
        "association": _association(payload),
        "post_over_pre_rms": post_over_pre,
        "residual_reduction_is_not_alignment_success": True,
        "architecture_or_threshold_tuning": payload.get("architecture_or_threshold_tuning"),
        "test_opened": payload.get("test_opened"),
        "mode_validity": validity,
        "independent_closure": bool(payload.get("capture_success") is True and validity["geometry_write_allowed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--station-train", required=True)
    parser.add_argument("--station-validation", required=True)
    parser.add_argument("--cdx-train", required=True)
    parser.add_argument("--cdx-validation", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract_path = (
        Path(args.contract).expanduser().resolve()
        if args.contract
        else Path(__file__).resolve().parents[1] / DEFAULT_CONTRACT_RELATIVE
    )
    contract = load_mode_validity_contract(contract_path)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    station_train = score_route_selected(
        Path(args.station_train).expanduser().resolve(), mode=STATION_MODE, contract=contract
    )
    station_val = score_route_selected(
        Path(args.station_validation).expanduser().resolve(), mode=STATION_MODE, contract=contract
    )
    cdx_train = score_route_selected(Path(args.cdx_train).expanduser().resolve(), mode=IFT_INTERNAL_MODE, contract=contract)
    cdx_val = None
    if args.cdx_validation:
        cdx_val = score_route_selected(
            Path(args.cdx_validation).expanduser().resolve(), mode=IFT_INTERNAL_MODE, contract=contract
        )
    report = {
        "schema_version": "faser-ift-calibration-mode-transfer-v1",
        "contract": str(contract_path),
        "retrain": False,
        "test_data_accessed": False,
        "residual_reduction_is_not_alignment_success": True,
        "corpus": "current_v3_expanded_source_disjoint_isolation",
        "station_mode": {"train": station_train, "validation": station_val},
        "ift_internal_mode": {"train": cdx_train, "validation": cdx_val},
        "both_modes_independent_closure": bool(
            station_train["independent_closure"]
            and station_val["independent_closure"]
            and cdx_train["independent_closure"]
        ),
        "large_statistics_extra_sources_pending": True,
        "note": (
            "This scores already-closed isolation banks with frozen inference. "
            "Extra 100043/044/047/048 files still need the physical chain before "
            "a larger source-disjoint transfer. Do not mix the two modes."
        ),
    }
    (output / "transfer_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output),
        "station_train_independent_closure": station_train["independent_closure"],
        "station_validation_independent_closure": station_val["independent_closure"],
        "cdx_train_independent_closure": cdx_train["independent_closure"],
        "station_train_unique_edges": station_train["unique_physical_edges"],
        "cdx_train_unique_edges": cdx_train["unique_physical_edges"],
        "both_modes_independent_closure": report["both_modes_independent_closure"],
        "large_statistics_extra_sources_pending": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
