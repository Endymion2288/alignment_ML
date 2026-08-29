"""Survey-candidate database, frame validation, and prior interface.

Reads the existing 2021/2022 slide summaries and the entry-62 constraint
schema.  Does not expand ML, module maps, or collision-track statistics,
does not map unconfirmed-frame tilts to Calypso station ry, and does not
emit a geometry candidate or alignment payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.external_constraint_iov_feasibility import combined_identifiability, unlocked
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
)
from alignment.survey_metrology_iov_infrastructure import (
    constraint_prior_row,
    covariance_matched_collinear_jacobian,
    empty_constraint_catalog,
    load_infrastructure_config,
    synthetic_feasibility_constraint,
    validate_constraint,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-survey-derived-prior-interface-frame-validation-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "survey_derived_prior_interface_frame_validation_v1.yaml"
DECISION = (
    "survey_prior_interface_ready_candidates_lack_validated_ry_mapping_and_measurement_covariance"
)
SCAN_PARAMETERS = ("station_dx", "station_ry", "C_dx")
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_survey_candidate",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_enter_cdx_mode",
    "do_not_run_newton",
    "do_not_solve_alignment_correction",
    "do_not_write_official_conditions",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_enter_full_module_identifiability_map",
    "do_not_invent_new_cosine_cut",
    "do_not_select_events_from_residual_or_cosine",
    "do_not_select_prior_from_residual",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_rebuild_2024_r0022_jacobian",
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "do_not_average_across_iovs",
    "do_not_invent_survey_numbers",
    "do_not_use_design_as_survey",
    "do_not_use_software_gauge_as_survey",
    "do_not_treat_conditions_as_independent_survey",
    "do_not_upgrade_numeric_agreement_to_survey_derived",
    "do_not_treat_ppt_as_alignment_prior",
    "do_not_treat_ppt_as_geometry",
    "do_not_map_unconfirmed_tilt_to_ry",
    "do_not_use_station_sd_as_measurement_sigma",
    "do_not_equate_layer_trend_to_calypso_station_ry",
    "do_not_expand_ml_or_module_map",
    "do_not_expand_collision_track_statistics",
    "synthetic_prior_feasibility_only",
    "slide_summary_feasibility_only",
    "software_fd_sensitivity_only",
)


class UnconfirmedRyMappingError(ValueError):
    """Raised when a tilt without a validated Calypso-ry frame is mapped to ry."""


def load_interface_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"interface config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected interface schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"interface config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"interface config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def common_audit_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "emits_alignment_payload": False,
            "geometry_candidate": False,
            "did_train_model": False,
            "did_run_alignment_fit": False,
            "did_write_geometry": False,
            "did_rebuild_2024_r0022_jacobian": False,
            "did_expand_ml_or_module_map": False,
            "did_expand_collision_track_statistics": False,
            "ppt_used_as_alignment_prior": False,
            "ppt_used_as_geometry": False,
            "unconfirmed_tilt_mapped_to_ry": False,
            "conditions_treated_as_independent_survey": False,
            "survey_numbers_were_invented": False,
            "slide_summary_feasibility_only": True,
        }
    )
    return state


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a mapping: {path}")
    return dict(payload)


def load_upstream_summaries(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    directory = resolve_under_root(root, str(config["upstream"]["survey_summary_dir"]))
    required = {
        "survey_2021": directory / "survey_2021_summary.json",
        "survey_2022": directory / "survey_2022_summary.json",
        "contrast": directory / "ift_layer_contrast_reconstruction.json",
        "frames": directory / "survey_calypso_frame_reconciliation.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"survey summaries are missing: {missing}")
    loaded = {name: _load_json(path) for name, path in required.items()}
    loaded["paths"] = {name: str(path) for name, path in required.items()}
    conditions = resolve_under_root(root, str(config["upstream"]["conditions_summary"]))
    loaded["conditions"] = _load_json(conditions) if conditions.is_file() else None
    return loaded


def _candidate(
    *,
    candidate_id: str,
    value: Any,
    unit: str | None,
    source: str,
    source_page: int | None,
    frame: str,
    frame_validated: bool,
    parameter_mapping: Mapping[str, Any],
    provenance: str,
    candidate_status: str,
    may_map_to_ry: bool,
    may_map_to_cdx: bool,
    ingest_availability: str,
    note: str,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "value": value,
        "sigma": None,
        "covariance": None,
        "unit": unit,
        "source": source,
        "source_page": source_page,
        "coordinate_frame": frame,
        "frame_validated": frame_validated,
        "parameter_mapping": dict(parameter_mapping),
        "provenance": provenance,
        "candidate_status": candidate_status,
        "may_map_to_station_ry": may_map_to_ry,
        "may_map_to_C_dx": may_map_to_cdx,
        "ingest_availability": ingest_availability,
        "independent_of_track_residual": True,
        "may_enter_geometry_candidate": False,
        "may_be_used_as_alignment_prior": False,
        "label": "slide_summary_not_alignment_prior",
        "note": note,
    }
    if may_map_to_ry:
        raise UnconfirmedRyMappingError(
            f"{candidate_id}: no current slide quantity may map to station ry"
        )
    return payload


def build_survey_candidate_database(
    config: Mapping[str, Any],
    summaries: Mapping[str, Any],
) -> dict[str, Any]:
    s21 = summaries["survey_2021"]
    s22 = summaries["survey_2022"]
    contrast = summaries["contrast"]
    angles = s21["normals_and_provisional_angles"]
    source_2021 = s21["source_pdf"]["relative"]
    source_2022 = s22["source_pdf"]["relative"]
    candidates = [
        _candidate(
            candidate_id="2021_ift_if_measured_normal_cad",
            value=list(angles["measured_normal_cad"]),
            unit="dimensionless",
            source=source_2021,
            source_page=7,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={"scan_parameter": None, "reason": "plane normal, not a scan DoF"},
            provenance="CERN 4-point I/F survey, 2021-11-25; Cadoux 2021-12-10 slide 7",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="I/F-plane measured normal in CAD. Do not map to Calypso station ry.",
        ),
        _candidate(
            candidate_id="2021_ift_if_total_tilt_mrad",
            value=angles["total_tilt_from_transverse_normal_mrad"],
            unit="mrad",
            source=source_2021,
            source_page=7,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={"scan_parameter": None, "forbidden": "station_ry"},
            provenance="Angle between CAD n0=(0,0,-1) and the measured I/F normal",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="Out-of-plane I/F tilt. Under the default CAD→Calypso permutation this is rx/rz, not ry.",
        ),
        _candidate(
            candidate_id="2021_ift_if_provisional_rx_cad_mrad",
            value=angles["provisional_rx_cad_mrad"],
            unit="mrad",
            source=source_2021,
            source_page=7,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={
                "cad_axis": "rx_cad",
                "default_calypso_image": "rx_cal",
                "scan_parameter": None,
                "forbidden": "station_ry",
            },
            provenance="Small-angle T*Rz*Ry*Rx on CAD n0=(0,0,-1), rz=0",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="Provisional CAD rx. Signs/origin are not unique. Forbidden as station ry.",
        ),
        _candidate(
            candidate_id="2021_ift_if_provisional_ry_cad_mrad",
            value=angles["provisional_ry_cad_mrad"],
            unit="mrad",
            source=source_2021,
            source_page=7,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={
                "cad_axis": "ry_cad",
                "default_calypso_image": "rz_cal",
                "scan_parameter": None,
                "forbidden": "station_ry",
            },
            provenance="Small-angle T*Rz*Ry*Rx on CAD n0=(0,0,-1), rz=0",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="CAD ry is rotation about the CAD beam axis. Default image is Calypso rz, not station ry.",
        ),
        _candidate(
            candidate_id="2021_ift_if_inplane_yaw_cad_rz",
            value=None,
            unit="mrad",
            source=source_2021,
            source_page=6,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={
                "cad_axis": "rz_cad",
                "default_calypso_image": "ry_cal",
                "scan_parameter": "station_ry",
                "mapping_blocked": True,
            },
            provenance="Cadoux slide 6: small rotation / z axis, no number quoted",
            candidate_status="unquantified",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="This is the only 2021 object that could become Calypso station ry after a unique transform. It has no value.",
        ),
        _candidate(
            candidate_id="2021_ift_if_center_shift_cad_mm",
            value=list(s21["center_shift_cad_mm"]["xyz_mm"]),
            unit="mm",
            source=source_2021,
            source_page=7,
            frame="2021_survey_CAD",
            frame_validated=False,
            parameter_mapping={"scan_parameter": None},
            provenance="Cadoux slide 7 inset: Shift / center about 0.4 mm",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="I/F-plane center shift in CAD. Not a ry or C_dx constraint.",
        ),
        _candidate(
            candidate_id="2022_station0_mean_offset_mm",
            value=s22["station0_ift_mean_offset_mm"],
            unit="mm",
            source=source_2022,
            source_page=5,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": "station_dx", "also": ["station_dy", "station_dz"]},
            provenance="Casper 2022-11-04 table Average Offsets by Station; IFT=station 0",
            candidate_status="recovered_value_covariance_missing",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="feasibility_only",
            note="Translation of IFT sensor means after stations 1-3 average match. Not a tilt.",
        ),
        _candidate(
            candidate_id="2022_station0_offset_width_mm",
            value=s22["station_offset_width_mm"]["0"],
            unit="mm",
            source=source_2022,
            source_page=6,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": None, "forbidden": "gaussian_prior_sigma"},
            provenance="Casper table Standard Deviation of Offsets by Station",
            candidate_status="population_spread_not_sigma",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="Module-to-module population spread. Forbidden as Gaussian prior uncertainty.",
        ),
        _candidate(
            candidate_id="2022_ift_layer_x_mm",
            value=contrast["ift_layer_x_mm"],
            unit="mm",
            source=source_2022,
            source_page=10,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": "C_dx", "via": "(x_L0-x_L2)/2"},
            provenance="Casper slide Mean Shifts by Layer; layers 0-2 reproduce Station-0",
            candidate_status="recovered_value_covariance_missing",
            may_map_to_ry=False,
            may_map_to_cdx=True,
            ingest_availability="feasibility_only",
            note="IFT layer-mean x. Layers 0-2 confirmed as IFT by station-average check.",
        ),
        _candidate(
            candidate_id="2022_ift_C_dx_slide_mm",
            value=contrast["C_dx_slide_mm"],
            unit="mm",
            source=source_2022,
            source_page=10,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": "C_dx", "scan_column": 2, "definition": "(x_L0-x_L2)/2"},
            provenance="Derived from 2022 layer-mean x after IFT identity check",
            candidate_status="recovered_value_covariance_missing",
            may_map_to_ry=False,
            may_map_to_cdx=True,
            ingest_availability="feasibility_only",
            note="The only current slide quantity that may attach to the C_dx prior slot. Sigma is missing.",
        ),
        _candidate(
            candidate_id="2022_ift_delta_x_L0_minus_L2_mm",
            value=contrast["delta_x_L0_minus_L2_mm"],
            unit="mm",
            source=source_2022,
            source_page=10,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": "C_dx", "scan_column": 2, "sigma_C_dx": "sigma(dx_L0-dx_L2)/2"},
            provenance="Same L0/L2 x contrast as C_dx_slide",
            candidate_status="recovered_value_covariance_missing",
            may_map_to_ry=False,
            may_map_to_cdx=True,
            ingest_availability="feasibility_only",
            note="Same degree of freedom as 2022_ift_C_dx_slide_mm, not an extra constraint.",
        ),
        _candidate(
            candidate_id="2022_layer_mean_coherent_tilt_candidate_mrad",
            value=contrast["layer_mean_coherent_tilt_candidate_mrad"],
            unit="mrad",
            source=source_2022,
            source_page=10,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": None, "forbidden": "station_ry", "identity": "slope=-C_dx/LAYERPITCH"},
            provenance="Linear x(z) using design LAYERPITCH=31.5 mm, not surveyed z",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="Named layer_mean_coherent_tilt_candidate only. Same DoF as C_dx_slide. Forbidden as Calypso station ry.",
        ),
        _candidate(
            candidate_id="2022_support_beam_horizontal_misalignment_urad",
            value=s22["support_beam_horizontal_misalignment_urad"],
            unit="urad",
            source=source_2022,
            source_page=7,
            frame="2022_survey_adjusted_FASER",
            frame_validated=True,
            parameter_mapping={"scan_parameter": None, "forbidden": "station_ry"},
            provenance="Casper X Offsets: Stations 1-3 consistent with 350 µrad support-beam tilt",
            candidate_status="frame_unconfirmed_ry_mapping_forbidden",
            may_map_to_ry=False,
            may_map_to_cdx=False,
            ingest_availability="unavailable",
            note="Residual CAD-to-FASER rotation of the support beam, not IFT station ry.",
        ),
    ]
    by_id = {row["candidate_id"]: row for row in candidates}
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "by_id": by_id,
        "n_may_map_to_C_dx": sum(1 for row in candidates if row["may_map_to_C_dx"]),
        "n_may_map_to_station_ry": sum(1 for row in candidates if row["may_map_to_station_ry"]),
        "source_summaries": summaries["paths"],
        "any_candidate_maps_to_ry": False,
    }


def attach_candidate_to_scan_parameter(
    candidate: Mapping[str, Any],
    scan_parameter: str,
) -> dict[str, Any]:
    """Attach a candidate to a scan parameter, or refuse an illegal ry map."""
    if scan_parameter not in SCAN_PARAMETERS:
        raise ValueError(f"unknown scan parameter: {scan_parameter}")
    if scan_parameter == "station_ry":
        raise UnconfirmedRyMappingError(
            f"cannot map {candidate['candidate_id']} to station_ry: "
            "unconfirmed-frame and same-DoF tilts are forbidden"
        )
    if scan_parameter == "C_dx" and not candidate.get("may_map_to_C_dx"):
        raise ValueError(f"{candidate['candidate_id']} may not map to C_dx")
    if not candidate.get("frame_validated"):
        raise ValueError(f"{candidate['candidate_id']} frame is not validated")
    return {
        "candidate_id": candidate["candidate_id"],
        "scan_parameter": scan_parameter,
        "value": candidate["value"],
        "sigma": None,
        "availability": "feasibility_only",
        "may_enter_geometry_candidate": False,
    }


def validate_frames(
    config: Mapping[str, Any],
    summaries: Mapping[str, Any],
    database: Mapping[str, Any],
) -> dict[str, Any]:
    frames = summaries["frames"]
    calypso = config["calypso"]
    rules = []
    for candidate in database["candidates"]:
        if candidate["parameter_mapping"].get("forbidden") == "station_ry" or not candidate["may_map_to_station_ry"]:
            if "tilt" in candidate["candidate_id"] or "ry_cad" in candidate["candidate_id"] or "yaw" in candidate["candidate_id"]:
                rules.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "maps_to_station_ry": False,
                        "reason": candidate["note"],
                    }
                )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "frames": {
            "survey_2021_cad": frames["frames"]["survey_2021_cad"],
            "survey_2022_adjusted_faser": frames["frames"]["survey_2022_adjusted_faser"],
            "calypso_geomodel_global": frames["frames"]["calypso_global"],
            "tracker_align": {
                "folder": calypso["align_folder"],
                "rotation_convention": calypso["rotation_convention"],
                "station_constants_order": list(calypso["station_constants_order"]),
                "stations_planes_frame": calypso["stations_planes_frame"],
                "interface_modules_frame": calypso["interface_modules_frame"],
                "detector_factory": (
                    "SCT_DetectorFactory addChannel: Stations/Planes level 3/2 global; "
                    "Interface1/2/3 level 1 local"
                ),
                "align_db_tool": (
                    "TrackerAlignDBTool stationAlignment: "
                    "T(dx,dy,dz)*Rz(rz)*Ry(ry)*Rx(rx), mm/rad, constants [dx,dy,dz,rx,ry,rz]"
                ),
                "dirkey_note": (
                    "dirkey inverts DetectorFactory level labels. Schema frames follow addChannel."
                ),
            },
        },
        "candidate_2021_cad_to_calypso": frames["candidate_2021_cad_to_calypso"],
        "explicit_transform_completed": False,
        "unconfirmed_tilt_mapped_to_ry": False,
        "ry_mapping_refusals": rules,
        "allowed_scan_attachments": [
            {
                "candidate_id": "2022_ift_C_dx_slide_mm",
                "scan_parameter": "C_dx",
                "frame": "2022_survey_adjusted_FASER ≈ Calypso global after stations 1-3 translation",
                "sigma": None,
            }
        ],
        "forbidden_scan_attachments": [
            {"candidate_id": "2021_ift_if_provisional_ry_cad_mrad", "scan_parameter": "station_ry"},
            {"candidate_id": "2021_ift_if_total_tilt_mrad", "scan_parameter": "station_ry"},
            {"candidate_id": "2022_layer_mean_coherent_tilt_candidate_mrad", "scan_parameter": "station_ry"},
            {"candidate_id": "2022_support_beam_horizontal_misalignment_urad", "scan_parameter": "station_ry"},
        ],
        "existing_conditions_are_not_survey": True,
        "may_enter_geometry_candidate": False,
    }


def official_constraint_slots(
    config: Mapping[str, Any],
    database: Mapping[str, Any],
) -> list[dict[str, Any]]:
    infrastructure = load_infrastructure_config(
        resolve_under_root(project_root(), str(config["upstream"]["infrastructure_config"]))
    )
    slots = empty_constraint_catalog(infrastructure)
    c_dx = database["by_id"]["2022_ift_C_dx_slide_mm"]
    delta = database["by_id"]["2022_ift_delta_x_L0_minus_L2_mm"]
    filled = []
    for slot in slots:
        item = dict(slot)
        if item["constraint_id"] == "ift_C_dx":
            item.update(
                {
                    "value": [float(c_dx["value"])],
                    "sigma": None,
                    "covariance": None,
                    "availability": "feasibility_only",
                    "source_file": c_dx["source"],
                    "provenance": c_dx["provenance"],
                    "frame": "2022_survey_adjusted_FASER",
                    "may_enter_geometry_candidate": False,
                    "central_value_is_survey": False,
                    "note": "Slide-derived C_dx central value. Sigma missing. Not a geometry candidate.",
                }
            )
        elif item["constraint_id"] == "ift_l0_minus_l2_dx":
            item.update(
                {
                    "value": [float(delta["value"])],
                    "sigma": None,
                    "covariance": None,
                    "availability": "feasibility_only",
                    "source_file": delta["source"],
                    "provenance": delta["provenance"],
                    "frame": "2022_survey_adjusted_FASER",
                    "may_enter_geometry_candidate": False,
                    "central_value_is_survey": False,
                    "note": "Same DoF as ift_C_dx. Sigma missing.",
                }
            )
        validate_constraint(item)
        filled.append(item)
    return filled


def track_fisher_external_prior_feasibility(
    config: Mapping[str, Any],
    database: Mapping[str, Any],
) -> dict[str, Any]:
    jacobian = covariance_matched_collinear_jacobian(config)
    physics = config["physics_scales"]
    rules = config["unlock"]
    frozen = config["entry_61_frozen"]
    points = []
    families = [("none", None, None)]
    for sigma_ry in config["prior_scan"]["sigma_ry_mrad"]:
        families.append(("ry_only", float(sigma_ry), None))
    for sigma_c in config["prior_scan"]["sigma_cdx_mm"]:
        families.append(("cdx_only", None, float(sigma_c)))
    for sigma_ry in config["prior_scan"]["sigma_ry_mrad"]:
        for sigma_c in config["prior_scan"]["sigma_cdx_mm"]:
            families.append(("both_hypothetical_independent", float(sigma_ry), float(sigma_c)))
    for family, sigma_ry, sigma_c in families:
        metrics = combined_identifiability(
            jacobian,
            sigma_r_mm=float(physics["measurement_sigma_mm"]),
            sigma_ry_mrad=sigma_ry,
            sigma_cdx_mm=sigma_c,
            rank_tolerance=float(rules["rank_tolerance"]),
        )
        points.append(
            {
                "family": family,
                "sigma_ry_mrad": sigma_ry,
                "sigma_cdx_mm": sigma_c,
                "unlocked": unlocked(metrics, rules),
                "rank": metrics["rank"],
                "ry_cdx_correlation": metrics["ry_cdx_correlation"],
                "sigma3_over_sigma1": metrics["sigma3_over_sigma1"],
                "marginal_sigma_ry_mrad": metrics["marginal_sigma_ry_mrad"],
                "marginal_sigma_cdx_mm": metrics["marginal_sigma_cdx_mm"],
                "geometry_candidate": False,
            }
        )
    ry_unlock = [row for row in points if row["family"] == "ry_only" and row["unlocked"]]
    cdx_unlock = [row for row in points if row["family"] == "cdx_only" and row["unlocked"]]
    catalog_priors = []
    for candidate_id in ("2022_ift_C_dx_slide_mm", "2021_ift_if_provisional_ry_cad_mrad"):
        candidate = database["by_id"][candidate_id]
        catalog_priors.append(
            {
                "candidate_id": candidate_id,
                "would_enter_fisher_now": False,
                "reason": (
                    "sigma is null"
                    if candidate_id.startswith("2022")
                    else "ry mapping forbidden"
                ),
            }
        )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "mode": "slide_summary_feasibility_only",
        "did_rebuild_2024_r0022_jacobian": False,
        "central_values_recorded_not_used_by_fisher": {
            "C_dx_slide_mm": database["by_id"]["2022_ift_C_dx_slide_mm"]["value"],
            "no_validated_ry_central_value": True,
        },
        "catalog_priors_do_not_enter_fisher_without_sigma": catalog_priors,
        "same_2022_contrast_cannot_supply_both_priors": True,
        "station_sd_used_as_sigma": False,
        "unconfirmed_tilt_mapped_to_ry": False,
        "predeclared_grid": config["prior_scan"],
        "unlock_rules": rules,
        "points": points,
        "coarsest_unlocking_sigma_ry_mrad": max(row["sigma_ry_mrad"] for row in ry_unlock) if ry_unlock else None,
        "coarsest_unlocking_sigma_cdx_mm": max(row["sigma_cdx_mm"] for row in cdx_unlock) if cdx_unlock else None,
        "degeneracy_breaking_frozen": {
            "sigma_ry_mrad": frozen["degeneracy_breaking_sigma_ry_mrad"],
            "sigma_cdx_mm": frozen["degeneracy_breaking_sigma_cdx_mm"],
        },
        "useful_precision": config["useful_precision"],
        "geometry_candidate": False,
        "emits_alignment_payload": False,
    }


def metrology_covariance_request(
    config: Mapping[str, Any],
    database: Mapping[str, Any],
    feasibility: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = config["entry_61_frozen"]
    useful = config["useful_precision"]
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "audience": "FASER hardware / alignment / metrology",
        "do_not_quote_20_mrad_or_0_315_mm_as_the_final_target": True,
        "candidates_do_not_yet_satisfy_information_requirement": True,
        "requests": [
            {
                "priority": 1,
                "what": "IFT L0 versus L2 transverse x, or C_dx=(dx_L0-dx_L2)/2",
                "frame": "Calypso global / 2022 survey-adjusted FASER, with an explicit unique transform",
                "why": (
                    "The slides already give a central value C_dx_slide="
                    f"{database['by_id']['2022_ift_C_dx_slide_mm']['value']:.7f} mm. "
                    "The missing object is the measurement covariance, not another track sample."
                ),
                "sigma_needed_to_unlock": feasibility["coarsest_unlocking_sigma_cdx_mm"],
                "sigma_useful": useful["cdx_mm"],
                "do_not_send": "2022 station/module standard deviations as if they were measurement sigma",
            },
            {
                "priority": 1,
                "what": "IFT / station-0 rotation about Calypso global y (station ry)",
                "frame": "Calypso GeoModel global after T*Rz*Ry*Rx",
                "why": (
                    "No current slide quantity may map to ry. The 2021 I/F normal tilt "
                    "is CAD rx/rz under the only candidate permutation; the page-6 "
                    "in-plane yaw is unquantified; the 2022 x(z) slope is the same "
                    "DoF as C_dx_slide."
                ),
                "sigma_needed_to_unlock": frozen["degeneracy_breaking_sigma_ry_mrad"],
                "sigma_useful": useful["ry_mrad"],
                "also_needed": [
                    "sign of 2021 Beam L+ versus Calypso +z",
                    "sign of 2021 CAD x versus Calypso +x",
                    "original CERN 4-point I/F coordinates with covariance",
                ],
            },
            {
                "priority": 2,
                "what": "Original 2021 Camille survey file and UNIGE plane-by-plane / assembled tables",
                "frame": "native survey/CAD plus the unique CAD→Calypso transform",
                "why": "Screenshot tables were not digitized as measurements.",
            },
            {
                "priority": 2,
                "what": "Original 2022 wafer-level front-sensor table used by Casper",
                "frame": "the same survey-adjusted FASER frame as the 2022 slides",
                "why": "Layer means exist; per-wafer covariance does not.",
            },
        ],
        "not_requested_as_measurements": [
            "geomdb_layerpitch",
            "design_stereo",
            "write_alignment_neutral_pool",
            "software_dz_gauge_5mm",
            "nominal_station_z",
            "OFLCOND-FASER-04/05/06 /Tracker/Align constants",
            "2022 station offset widths",
            "2022 layer_mean_coherent_tilt_candidate as station ry",
            "2021 I/F normal tilt as station ry",
        ],
        "either_priority_one_constraint_with_covariance_is_enough_to_start": True,
    }


def assess_information_requirement(
    database: Mapping[str, Any],
    frames: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    slots: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ry_slot = next(item for item in slots if item["constraint_id"] == "ift_station0_ry")
    cdx_slot = next(item for item in slots if item["constraint_id"] == "ift_C_dx")
    possess = (
        ry_slot["availability"] == "measured"
        or (
            cdx_slot["availability"] == "measured"
            and cdx_slot.get("sigma") is not None
        )
    )
    return {
        "survey_candidates_possess_degeneracy_breaking_information_requirement": possess,
        "have_slide_central_value_for_C_dx": True,
        "have_measurement_covariance": False,
        "have_validated_ry_mapping": False,
        "official_ry_slot_availability": ry_slot["availability"],
        "official_cdx_slot_availability": cdx_slot["availability"],
        "official_cdx_sigma": cdx_slot.get("sigma"),
        "explicit_transform_completed": frames["explicit_transform_completed"],
        "catalog_prior_rows_that_enter_fisher": [
            constraint_prior_row(item) for item in slots if constraint_prior_row(item) is not None
        ],
        "reason": (
            "A C_dx or ry Gaussian prior with the frozen σ would unlock the "
            "collinear Jacobian, and the slides already name a C_dx central "
            "value. The information requirement is not met because that value "
            "has no measurement covariance and no slide quantity has a validated "
            "Calypso-ry mapping. Track degeneracy is a known Fisher property, "
            "not the missing object."
        ),
        "coarsest_unlocking_sigma_ry_mrad": feasibility["coarsest_unlocking_sigma_ry_mrad"],
        "coarsest_unlocking_sigma_cdx_mm": feasibility["coarsest_unlocking_sigma_cdx_mm"],
    }


def decide_next_stage(
    *,
    assessment: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "survey_candidates_possess_degeneracy_breaking_information_requirement": assessment[
            "survey_candidates_possess_degeneracy_breaking_information_requirement"
        ],
        "have_validated_ry_mapping": False,
        "have_measurement_covariance": False,
        "enough_to_write_alignment_prior": False,
        "enough_to_write_geometry": False,
        "survey_derived": False,
        "next_request": request["requests"][:2],
        "next_allowed_step": (
            "Keep residual_dq_monitoring_only. Ask the hardware/alignment team "
            "for the original 2021/2022 tables with measurement covariance and "
            "a unique CAD→Calypso transform. Do not expand ML, module maps, or "
            "collision-track statistics. Do not promote slide numbers to a prior."
        ),
        "reason": assessment["reason"],
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    summaries = load_upstream_summaries(config)
    database = build_survey_candidate_database(config, summaries)
    frames = validate_frames(config, summaries, database)
    slots = official_constraint_slots(config, database)
    feasibility = track_fisher_external_prior_feasibility(config, database)
    request = metrology_covariance_request(config, database, feasibility)
    assessment = assess_information_requirement(database, frames, feasibility, slots)
    decision = decide_next_stage(assessment=assessment, request=request)
    database = dict(database)
    database["official_constraint_slots"] = [
        {
            "constraint_id": item["constraint_id"],
            "availability": item["availability"],
            "value": item.get("value"),
            "sigma": item.get("sigma"),
            "frame": item.get("frame"),
            "may_enter_geometry_candidate": item.get("may_enter_geometry_candidate"),
        }
        for item in slots
    ]
    decision["information_requirement"] = assessment
    return {
        "survey_candidate_database": database,
        "survey_frame_reconciliation_validation": frames,
        "track_fisher_external_prior_feasibility": feasibility,
        "metrology_covariance_request": request,
        "next_stage_decision": decision,
    }
