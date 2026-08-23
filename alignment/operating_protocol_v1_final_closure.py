"""Operating Protocol V1 final real-data closure and reproducibility freeze.

Assembles entries 47-53 into an immutable evidence package.  Does not
reopen Station Mode, reduced Station Mode, or C_dx Mode.  Does not
retrain V2, retune thresholds, re-estimate the 14973/14974 residual
scale, or extract an alignment payload from self-nulling residuals.

Any residual decrease copied into this package is labeled
``DQ observable`` only.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.calibration_modes import OPERATING_BAND_UM
from alignment.real_data_residual_dq_monitoring import (
    DECISION_MONITORING_ONLY,
    geometry_reopen_prerequisites,
)
from alignment.real_data_residual_dq_monitoring_expansion import EXPANSION_RUNS

SCHEMA_VERSION = "faser-operating-protocol-v1-final-real-data-closure"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_final_real_data_closure_v1.yaml"
)
PROTOCOL_VERSION = "operating_protocol_v1"
OPERATING_MODE = "residual_dq_monitoring_only"
RESIDUAL_DECREASE_LABEL = "DQ observable"
FROZEN_V2_CHECKPOINT_SHA256 = (
    "0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27"
)
FORBIDDEN_TRUTH_METRICS = (
    "efficiency",
    "purity",
    "fake",
    "auc",
    "average_precision",
    "ap",
    "roc",
)

EVIDENCE_CHAIN = (
    {
        "id": "mc_transfer",
        "label": "MC transfer PASS",
        "workbook": [47],
        "verdict": "PASS",
        "status": "PASS",
    },
    {
        "id": "real_data_frozen_v2_acceptance",
        "label": "real-data frozen-V2 acceptance statistics-limited but recovered at scale",
        "workbook": [48, 49],
        "verdict": "statistics_limited_but_recovered_at_scale",
        "status": "PASS_AT_SCALE",
    },
    {
        "id": "full_segment_station_calibration",
        "label": (
            "full-segment Station calibration REJECTED by "
            "physical_nonidentifiability/cross_level_contamination"
        ),
        "workbook": [49, 50],
        "verdict": "REJECTED",
        "status": "REJECTED",
        "rejection_class": [
            "physical_nonidentifiability",
            "cross_level_contamination",
        ],
    },
    {
        "id": "reduced_mode_calibration",
        "label": "reduced-mode calibration REJECTED by cross-run non-transferability",
        "workbook": [51],
        "verdict": "REJECTED",
        "status": "REJECTED",
        "rejection_class": ["cross_run_non_transferability"],
    },
    {
        "id": "current_geometry_residual_dq_monitoring",
        "label": "current-geometry residual/DQ monitoring PASS on independent runs",
        "workbook": [52, 53],
        "verdict": "PASS",
        "status": "PASS",
    },
)

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_db_write",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "joint_station_cdx_newton",
    "new_layer_or_module_dof",
    "track_driven_dz",
    "dz_track_driven_observable",
    "rz_direct_track_residual",
    "alignment_payload_from_self_nulling_residuals",
    "schur_projection_production",
    "mc_truth_used",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_enter_cdx_mode",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_retrain_v2",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "do_not_reestimate_reference_scale",
    "do_not_reestimate_alarm_thresholds",
    "do_not_retune_A",
    "cannot_convert_drift_to_geometry",
    "do_not_extract_alignment_payload_from_self_nulling",
    "residual_reduction_is_not_alignment_success",
    "current_geometry_only",
    "do_not_repick_window_from_residual",
    "do_not_open_sealed_test",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under_root(root: Path, relative: str | Path) -> Path:
    source = Path(relative)
    if source.is_absolute():
        return source
    return (root / source).resolve()


def load_closure_config(path: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"closure config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected closure config schema: {source}")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("closure config must freeze residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("closure config must keep the frozen V2 checkpoint SHA256")
    if str(payload.get("residual_decrease_label")) != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must be labeled 'DQ observable'")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"closure config must set {key}=false")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"closure config must set {key}=true")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False


def campaign_allows_geometry_write(decision: str) -> bool:
    return False


def campaign_allows_station_calibration_mode(decision: str) -> bool:
    return False


def campaign_allows_alignment_payload(decision: str) -> bool:
    return False


def frozen_operating_state() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "real_data_operating_mode": OPERATING_MODE,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_drift_candidate": False,
        "official_conditions_db_write": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "do_not_extract_alignment_payload_from_self_nulling": True,
        "alignment_payload_from_self_nulling_residuals": False,
        "residual_reduction_is_not_alignment_success": True,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "code_path": [
            "current_official_geometry",
            "frozen_v2",
            "residual_dq_monitoring",
        ],
    }


def unlock_criteria() -> dict[str, Any]:
    prerequisites = geometry_reopen_prerequisites()
    return {
        "schema_version": f"{SCHEMA_VERSION}-unlock",
        "currently_met": False,
        "opens": "operating_protocol_v2",
        "does_not_write_geometry_in_this_package": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "required_any_of": [
            {
                "id": "external_station_constraints_and_cdx_budget",
                "met": False,
                "all_of": [
                    {
                        "independent_survey_or_external_station_constraints_fix": [
                            "ift_dx_mm",
                            "ift_ry_mrad",
                            "ift_dz_mm",
                        ]
                    },
                    {
                        "independent_real_C_dx_abs_within_operating_band_um": list(
                            OPERATING_BAND_UM
                        )
                    },
                ],
                "note": (
                    "Survey or other external station constraints must fix "
                    "dx/ry/dz, and an independent measurement must prove that "
                    "the true |C_dx| sits inside the 1.5-1.7 µm isolation budget."
                ),
            },
            {
                "id": "new_independent_real_data_topology",
                "met": False,
                "all_of": [
                    {"new_independent_real_data_topology_or_track_sample": True},
                    {
                        "empirically_demonstrates_cross_run_transferable_calibration_subspace": True
                    },
                ],
                "note": (
                    "A new independent real-data track topology must empirically "
                    "demonstrate a calibration subspace that transfers across runs."
                ),
            },
        ],
        "until_then": {
            "real_data_operating_mode": OPERATING_MODE,
            "code_path": [
                "current_official_geometry",
                "frozen_v2",
                "residual_dq_monitoring",
            ],
            "geometry_write_allowed": False,
            "station_calibration_mode_available": False,
            "cdx_mode_allowed": False,
            "alignment_payload_from_self_nulling_residuals": False,
            "text": prerequisites["until_then"],
        },
        "geometry_reopen_prerequisites": prerequisites,
    }


def evaluate_unlock(claims: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return whether Operating Protocol V2 may be opened.

    This package never writes geometry.  A true result only permits a
    new protocol; it does not emit a payload.
    """
    claims = {} if claims is None else dict(claims)
    survey = bool(claims.get("independent_survey_or_external_station_constraints"))
    fixed = set(claims.get("fixed_station_parameters") or [])
    cdx = claims.get("independent_real_C_dx_abs_um")
    cdx_ok = (
        cdx is not None
        and math.isfinite(float(cdx))
        and OPERATING_BAND_UM[0] <= float(cdx) <= OPERATING_BAND_UM[1]
    )
    first = survey and fixed.issuperset({"ift_dx_mm", "ift_ry_mrad", "ift_dz_mm"}) and cdx_ok
    second = bool(claims.get("new_independent_real_data_topology")) and bool(
        claims.get("cross_run_transferable_calibration_subspace")
    )
    allowed = bool(first or second)
    return {
        "currently_met": allowed,
        "opens_operating_protocol_v2": allowed,
        "geometry_write_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "matched": {
            "external_station_constraints_and_cdx_budget": first,
            "new_independent_real_data_topology": second,
        },
        "until_then_if_unmet": unlock_criteria()["until_then"],
    }


def walk_forbidden(payload: object, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            parts = set(lowered.split("_"))
            if any(token == lowered or token in parts for token in FORBIDDEN_TRUTH_METRICS):
                raise ValueError(f"{where} contains forbidden truth-metric key {key!r}")
            walk_forbidden(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            walk_forbidden(item, where=where)


def evidence_chain_text() -> str:
    return (
        "MC transfer PASS → real-data frozen-V2 acceptance statistics-limited "
        "but recovered at scale → full-segment Station calibration REJECTED by "
        "physical_nonidentifiability/cross_level_contamination → reduced-mode "
        "calibration REJECTED by cross-run non-transferability → "
        "current-geometry residual/DQ monitoring PASS on independent runs"
    )


def label_dq_observable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(payload),
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "residual_reduction_is_not_alignment_success": True,
        "not_a_geometry_correction": True,
        "dq_observable_only": True,
    }


def frozen_alarm_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "min_selected_routes_for_alignment_dq": int(
            config["min_selected_routes_for_alignment_dq"]
        ),
        "max_event_share_of_selected_routes": float(
            config["max_event_share_of_selected_routes"]
        ),
        "min_all_pairs_candidates": int(config["min_all_pairs_candidates"]),
        "robust_z_detector_condition": float(config["robust_z_detector_condition"]),
        "robust_z_drift": float(config["robust_z_drift"]),
        "min_nonreference_runs_for_repeatable_drift": int(
            config["min_nonreference_runs_for_repeatable_drift"]
        ),
        "reestimated": False,
        "source": "entry_52_frozen",
    }


def extract_frozen_A(contract: Mapping[str, Any]) -> dict[str, Any]:
    leakage = contract.get("leakage_operator") or {}
    native = dict(leakage.get("A_native_per_mm_C_dx") or {})
    station = (contract.get("station_mode") or {}).get("unmodeled_C_dx") or {}
    return {
        "do_not_retune": True,
        "source": leakage.get("source"),
        "A_native_per_mm_C_dx": native,
        "A_dx": float(native.get("ift_dx_mm", station.get("A_dx"))),
        "A_ry": float(native.get("ift_ry_mrad", -31.90780811244653)),
        "operating_band_um": list(OPERATING_BAND_UM),
        "statistical_max_abs_um": station.get("statistical_max_abs_um"),
        "engineering_max_abs_um": station.get("engineering_max_abs_um"),
    }


def extract_reference_scale(parent_report: Mapping[str, Any]) -> dict[str, Any]:
    reference = parent_report.get("calibration_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("parent monitoring report lacks calibration_reference")
    if list(reference.get("runs") or []) != [14973, 14974]:
        raise ValueError("frozen calibration reference must stay 14973/14974")
    channels = reference.get("channels") or {}
    dy = channels.get("dy") or {}
    rx = channels.get("rx") or {}
    return {
        "runs": [14973, 14974],
        "n_reference_field_edges": list(reference.get("n_reference_field_edges") or []),
        "dy": {
            "median": dy.get("median"),
            "robust_scale": dy.get("robust_scale"),
            "n": dy.get("n"),
        },
        "rx": {
            "median": rx.get("median"),
            "robust_scale": rx.get("robust_scale"),
            "n": rx.get("n"),
        },
        "used_for_geometry": False,
        "reestimated": False,
        "not_a_geometry_correction": True,
    }


def extract_selected_route_scaling(
    scaling_report: Mapping[str, Any],
    expansion_blocks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = []
    for row in scaling_report.get("scale_table") or []:
        rows.append(
            {
                "run": int(row["run"]),
                "role": row.get("role"),
                "scale": row.get("scale"),
                "n_events": row.get("n_events"),
                "n_tracklets": row.get("n_tracklets"),
                "n_all_pairs_candidates": row.get("n_all_pairs_candidates"),
                "selected_routes": row.get("selected_routes"),
                "complete_four_station_routes": row.get("complete_four_station_routes"),
                "campaign": "entry_49_scaling",
            }
        )
    for block in expansion_blocks or []:
        if int(block.get("run", -1)) not in EXPANSION_RUNS:
            continue
        rows.append(
            {
                "run": int(block["run"]),
                "role": block.get("role"),
                "scale": "full",
                "n_events": block.get("n_events"),
                "n_tracklets": block.get("n_tracklets"),
                "n_all_pairs_candidates": block.get("n_all_pairs_candidates"),
                "selected_routes": block.get("selected_routes"),
                "complete_four_station_routes": block.get("complete_four_station_routes"),
                "campaign": "entry_53_expansion",
            }
        )
    return {
        "title": "selected-route count versus event statistics",
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "not_a_geometry_correction": True,
        "rows": rows,
    }


def extract_route_composition(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for block in blocks:
        composition = block.get("route_composition") or {}
        rows.append(
            {
                "run": int(block["run"]),
                "role": block.get("role"),
                "lhc_fill": block.get("lhc_fill"),
                "selected_routes": block.get("selected_routes"),
                "n_2_station": composition.get("2_station"),
                "n_3_station": composition.get("3_station"),
                "n_4_station": composition.get("4_station"),
                "status": block.get("status"),
                "dq_observable_only": True,
            }
        )
    return {
        "title": "2/3/4-station selected-route composition",
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "not_a_geometry_correction": True,
        "rows": rows,
    }


def extract_robust_z_timeseries(series: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in series:
        isolation = item.get("isolation_robust_standardized_shift") or {}
        rows.append(
            {
                "run": int(item["run"]),
                "lhc_fill": item.get("lhc_fill"),
                "skip_events": item.get("skip_events"),
                "role": item.get("role"),
                "selected_routes": item.get("selected_routes"),
                "status": item.get("status"),
                "dy_robust_z": isolation.get("dy"),
                "rx_robust_z": isolation.get("rx"),
                "dy_median_mm": (item.get("isolation_medians") or {}).get("dy"),
                "rx_median": (item.get("isolation_medians") or {}).get("rx"),
                "dq_observable_only": True,
                "not_a_geometry_correction": True,
            }
        )
    return {
        "title": "dy/rx robust-z time series",
        "order": "lhc_fill_then_run_then_skip_events",
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "not_a_geometry_correction": True,
        "rows": rows,
    }


def extract_jacobian_singular_spectrum(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for block in audit.get("blocks") or []:
        ident = block.get("identifiability") or {}
        five = ident.get("five_dof_excluding_survey_dz") or {}
        rows.append(
            {
                "run": int(block["run"]),
                "six_dof_singular_values": list(ident.get("jacobian_singular_values") or []),
                "six_dof_condition_number": ident.get("condition_number"),
                "six_dof_near_degenerate": (ident.get("near_degenerate") or {}).get(
                    "near_degenerate"
                ),
                "five_dof_singular_values": list(five.get("singular_values") or []),
                "five_dof_condition_number": five.get("condition_number"),
                "weak_direction_cosine_with_A": five.get("weak_direction_cosine_with_A"),
                "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
                "identifiability_diagnostic_not_closure": True,
            }
        )
    return {
        "title": "Station Jacobian singular spectrum",
        "note": "Identifiability diagnostic only.  Not a geometry correction.",
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "rows": rows,
    }


def extract_reduced_mode_inconsistency(
    identifiability: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> dict[str, Any]:
    mode = (identifiability.get("predeclared_modes") or {}).get("three_dof_dy_rx_rz") or {}
    per_run = mode.get("per_run") or {}
    consistency = mode.get("run_to_run_consistency") or {}
    transfers = (transfer.get("linearized_calibration_transfer") or {}).get(
        "three_dof_dy_rx_rz"
    ) or []
    rows = []
    for run, payload in per_run.items():
        rows.append(
            {
                "run": int(run),
                "condition_number": payload.get("condition_number"),
                "delta": dict(payload.get("delta") or {}),
                "sigma": dict(payload.get("sigma") or {}),
                "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
                "not_a_geometry_correction": True,
            }
        )
    labeled_transfers = [
        label_dq_observable(
            {
                "from_run": item.get("from_run"),
                "to_run": item.get("to_run"),
                "chi2_before": item.get("chi2_before"),
                "chi2_after": item.get("chi2_after"),
                "chi2_ratio": item.get("chi2_ratio"),
            }
        )
        for item in transfers
    ]
    return {
        "title": "reduced-mode cross-run correction inconsistency",
        "mode": "three_dof_dy_rx_rz",
        "floated_parameters": list(mode.get("floated_parameters") or []),
        "max_nsigma": consistency.get("max_nsigma"),
        "nsigma": dict(consistency.get("nsigma") or {}),
        "sign_agreement": dict(consistency.get("sign_agreement") or {}),
        "consistent": False,
        "per_run": rows,
        "linearized_calibration_transfer": labeled_transfers,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "note": (
            "A linearized chi2 drop is a DQ observable only and is never "
            "alignment success.  The 14974→14973 transfer worsens."
        ),
    }


def extract_a_vs_weak_direction(
    frozen_a: Mapping[str, Any],
    audit: Mapping[str, Any],
    reduced: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    native = dict(frozen_a.get("A_native_per_mm_C_dx") or {})
    rows = []
    for block in audit.get("blocks") or []:
        ident = block.get("identifiability") or {}
        five = ident.get("five_dof_excluding_survey_dz") or {}
        six_small = ((ident.get("near_degenerate") or {}).get("smallest_direction")) or {}
        five_small = ((five.get("near_degenerate") or {}).get("smallest_direction")) or {}
        rows.append(
            {
                "run": int(block["run"]),
                "six_dof_smallest_direction": dict(six_small),
                "five_dof_smallest_direction": dict(five_small),
                "five_dof_weak_direction_cosine_with_A": five.get(
                    "weak_direction_cosine_with_A"
                ),
            }
        )
    common = (reduced or {}).get("common_identifiable_subspace") or {}
    return {
        "title": "frozen A leakage direction versus station weak direction",
        "frozen_A_native_per_mm_C_dx": native,
        "A_dx": frozen_a.get("A_dx"),
        "A_ry": frozen_a.get("A_ry"),
        "do_not_retune": True,
        "per_run": rows,
        "reduced_mode_axis_cosine_with_A": dict(common.get("axis_cosine_with_A") or {}),
        "isolation_safe_axes": list(common.get("isolation_safe_axes") or []),
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "identifiability_diagnostic_not_closure": True,
    }


def _window_runs(windows: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(windows.get("runs"), Mapping):
        return windows["runs"]
    nested = windows.get("frozen_windows")
    if isinstance(nested, Mapping) and isinstance(nested.get("runs"), Mapping):
        return nested["runs"]
    return {}


def occupancy_provenance(windows: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run, payload in sorted(_window_runs(windows).items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "run": int(run),
                "role": payload.get("role"),
                "segment": payload.get("segment"),
                "skip_events": payload.get("skip_events"),
                "lhc_fill": payload.get("lhc_fill"),
                "input_xaod": payload.get("input_xaod"),
                "status": payload.get("status"),
                "residual_blind": payload.get("residual_blind", True),
                "selection_order": payload.get("selection_order"),
            }
        )
    return rows


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    record = {
        "path": str(resolved),
        "role": role,
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        record["sha256"] = sha256_file(resolved)
        record["bytes"] = resolved.stat().st_size
    return record


def assert_no_alignment_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("geometry_write_allowed") is True:
        raise ValueError("closure package must not allow geometry write")
    if payload.get("alignment_payload_from_self_nulling_residuals") is True:
        raise ValueError("closure package must not emit a self-nulling payload")
    if payload.get("station_calibration_mode_available") is True:
        raise ValueError("closure package must not reopen Station Mode")
    if payload.get("cdx_mode_allowed") is True:
        raise ValueError("closure package must not reopen C_dx Mode")
    forbidden_payload_keys = {
        "alignment_payload",
        "station_payload",
        "cdx_payload",
        "newton_delta",
        "conditions_payload",
    }
    extra = forbidden_payload_keys.intersection(payload)
    if extra:
        raise ValueError(f"closure package contains alignment payload keys {sorted(extra)}")


def build_evidence_matrix(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    transfer = artifacts.get("transfer_report") or {}
    scaling = artifacts.get("scaling_report") or {}
    fullscale = artifacts.get("fullscale_decision") or {}
    failure = artifacts.get("failure_classification") or {}
    reduced = artifacts.get("reduced_decision") or {}
    monitoring = artifacts.get("monitoring_decision") or {}
    expansion = artifacts.get("expansion_decision") or {}
    expansion_blocks = list((artifacts.get("expansion_run_level") or {}).get("blocks") or [])
    chain = []
    for step, item in enumerate(EVIDENCE_CHAIN, start=1):
        row = dict(item)
        row["step"] = step
        if item["id"] == "mc_transfer":
            row["independent_closure"] = transfer.get("both_modes_independent_closure")
            row["A_updated"] = False
            row["sealed_test_opened"] = False
        elif item["id"] == "real_data_frozen_v2_acceptance":
            row["n100_selected_empty"] = True
            row["full_segment_selected"] = {
                str(block.get("run")): block.get("selected_routes")
                for block in (scaling.get("scale_table") or [])
                if block.get("scale") == "full"
            }
            row["decision"] = scaling.get("decision")
        elif item["id"] == "full_segment_station_calibration":
            row["fullscale_decision"] = fullscale.get("decision")
            row["failure_decision"] = failure.get("decision")
            row["unique_class"] = failure.get("decision")
            row["geometry_write_allowed"] = False
        elif item["id"] == "reduced_mode_calibration":
            row["decision"] = reduced.get("decision")
            row["unique_class"] = reduced.get("unique_class")
            row["v2_candidate_mode"] = reduced.get("v2_candidate_mode")
        elif item["id"] == "current_geometry_residual_dq_monitoring":
            row["parent_decision"] = monitoring.get("decision")
            row["expansion_decision"] = expansion.get("decision")
            row["alignment_drift_candidate"] = bool(
                expansion.get("alignment_drift_candidate")
            )
            row["n_expansion_nominal"] = sum(
                1
                for block in expansion_blocks
                if block.get("expansion_run") and block.get("status") == "nominal_monitoring"
            )
        chain.append(row)
    return {
        "schema_version": f"{SCHEMA_VERSION}-evidence-matrix",
        "evidence_chain_text": evidence_chain_text(),
        "chain": chain,
        "real_data_operating_mode": OPERATING_MODE,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_drift_candidate": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "residual_reduction_is_not_alignment_success": True,
    }


def build_chart_data(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    frozen_a = extract_frozen_A(artifacts.get("mode_validity_contract") or {})
    expansion_run_level = artifacts.get("expansion_run_level") or {}
    blocks = list(expansion_run_level.get("blocks") or [])
    time_series = list((artifacts.get("time_stability") or {}).get("series") or [])
    return {
        "schema_version": f"{SCHEMA_VERSION}-chart-data",
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "residual_reduction_is_not_alignment_success": True,
        "not_a_geometry_correction": True,
        "selected_route_scaling": extract_selected_route_scaling(
            artifacts.get("scaling_report") or {},
            blocks,
        ),
        "route_composition": extract_route_composition(blocks),
        "robust_z_timeseries": extract_robust_z_timeseries(time_series),
        "station_jacobian_singular_spectrum": extract_jacobian_singular_spectrum(
            artifacts.get("identifiability_audit") or {}
        ),
        "reduced_mode_cross_run_inconsistency": extract_reduced_mode_inconsistency(
            artifacts.get("reduced_identifiability") or {},
            artifacts.get("reduced_transfer") or {},
        ),
        "frozen_A_versus_station_weak_direction": extract_a_vs_weak_direction(
            frozen_a,
            artifacts.get("identifiability_audit") or {},
            artifacts.get("reduced_feasibility") or {},
        ),
    }


def build_reproducibility_manifest(
    *,
    config: Mapping[str, Any],
    file_records: Sequence[Mapping[str, Any]],
    occupancy_rows: Sequence[Mapping[str, Any]],
    reference_scale: Mapping[str, Any],
    frozen_a: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if checkpoint_sha256 != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("frozen V2 checkpoint SHA256 mismatch")
    return {
        "schema_version": f"{SCHEMA_VERSION}-reproducibility-manifest",
        "frozen": True,
        "protocol_version": PROTOCOL_VERSION,
        "real_data_operating_mode": OPERATING_MODE,
        "q_over_p_mode": 0,
        "observation_kind": config.get("observation_kind"),
        "observation_statistics": config.get("observation_statistics"),
        "geometry_tag": config.get("geometry_tag"),
        "conditions_tag": config.get("conditions_tag"),
        "reconstruction_tag": config.get("reconstruction_tag"),
        "athena_flags": dict(config.get("athena_flags") or {}),
        "frozen_v2": {
            "root": config.get("frozen_v2"),
            "checkpoint": config.get("frozen_v2_checkpoint"),
            "sha256": checkpoint_sha256,
            "thresholds": config.get("frozen_v2_thresholds"),
            "unmatched_penalty": config.get("frozen_v2_unmatched_penalty"),
            "do_not_retrain": True,
        },
        "occupancy_selection_rule": {
            "path": config.get("window_rule"),
            "residual_blind": True,
            "selection_order": "first_contiguous_window_in_skip_events_order",
            "minima": dict(config.get("occupancy_minima") or {}),
            "do_not_repick_from_residual": True,
        },
        "reference_residual_scale": dict(reference_scale),
        "alarm_thresholds": frozen_alarm_thresholds(config),
        "mode_validity_contract": config.get("mode_validity_contract"),
        "A_operator": dict(frozen_a),
        "input_run_provenance": list(occupancy_rows),
        "files": list(file_records),
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "do_not_extract_alignment_payload_from_self_nulling": True,
    }


def build_final_report(
    *,
    config: Mapping[str, Any],
    evidence_matrix: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expansion_decision: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    state = frozen_operating_state()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "frozen": True,
        "protocol_version": PROTOCOL_VERSION,
        **state,
        "evidence_chain_text": evidence_chain_text(),
        "evidence_chain": list(evidence_matrix.get("chain") or EVIDENCE_CHAIN),
        "unique_decision": DECISION_MONITORING_ONLY,
        "alignment_drift_candidate": bool(
            expansion_decision.get("alignment_drift_candidate", False)
        ),
        "unlock_criteria": unlock_criteria(),
        "unlock_evaluation": evaluate_unlock(),
        "frozen_v2_checkpoint_sha256": FROZEN_V2_CHECKPOINT_SHA256,
        "mode_validity_contract": config.get("mode_validity_contract"),
        "occupancy_selection_rule": manifest.get("occupancy_selection_rule"),
        "reference_residual_scale": manifest.get("reference_residual_scale"),
        "alarm_thresholds": manifest.get("alarm_thresholds"),
        "A_operator": manifest.get("A_operator"),
        "input_run_provenance": manifest.get("input_run_provenance"),
        "campaign_roots": {
            "47_mc_transfer": config.get("mc_transfer_root"),
            "48_occupancy_preflight": config.get("occupancy_preflight_root"),
            "49_scaling_and_fullscale": [
                config.get("scaling_root"),
                config.get("self_nulling_root"),
            ],
            "50_failure_audit": config.get("failure_audit_root"),
            "51_reduced_mode": config.get("reduced_mode_root"),
            "52_monitoring": config.get("monitoring_root"),
            "53_expansion": config.get("expansion_root"),
        },
        "next_allowed_step": (
            "Keep real_data_residual_dq_monitoring_only.  Do not construct a "
            "Station calibration mode.  Do not start C_dx Mode.  Do not extract "
            "any alignment payload from real-data self-nulling residuals.  "
            "Official current geometry plus frozen V2 remains the only code path "
            "until an unlock criterion is met and Operating Protocol V2 is opened."
        ),
        "sealed_test_opened": False,
        "do_not_open_sealed_test": True,
        "do_not_retrain_v2": True,
        "do_not_generate_fd_probes": True,
        "do_not_run_newton": True,
        "do_not_write_payload": True,
        "cannot_convert_drift_to_geometry": True,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "official_conditions_db_modified": False,
    }
    if payload["alignment_drift_candidate"]:
        raise ValueError("closure freeze requires alignment_drift_candidate=false")
    assert_no_alignment_payload(payload)
    walk_forbidden(payload, where="operating_protocol_v1_final_report")
    return payload
