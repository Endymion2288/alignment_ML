"""Reconstruct 2021/2022 survey-slide summaries and reconcile frames.

Reads only the two existing FASER survey PDFs plus frozen entry-61 Fisher
thresholds.  Does not train, refit, write geometry, treat slide numbers as
an alignment prior, or emit an alignment payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.external_constraint_iov_feasibility import combined_identifiability, unlocked
from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.survey_metrology_iov_infrastructure import covariance_matched_collinear_jacobian
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-survey-summary-reconstruction-frame-reconciliation-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "survey_summary_reconstruction_frame_reconciliation_v1.yaml"
DECISION = (
    "slide_summaries_define_ift_l0l2_contrast_and_tilt_candidates_"
    "frames_not_uniquely_reconciled_remaining_uncertainty_is_measurement_covariance"
)
STATION_FROM_LAYER = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_slide_summary",
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
    "do_not_wait_for_original_survey_excel",
    "do_not_use_station_sd_as_measurement_sigma",
    "do_not_equate_layer_trend_to_calypso_station_ry",
    "do_not_digitize_screenshot_tables_as_measurements",
    "synthetic_prior_feasibility_only",
    "slide_summary_feasibility_only",
    "software_fd_sensitivity_only",
)


def load_reconstruction_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"reconstruction config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected reconstruction schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"reconstruction config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"reconstruction config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    physics = payload.get("physics_scales") or {}
    if physics.get("ift_layer_pitch_is_survey") is not False:
        raise ValueError("LAYERPITCH is design, not survey")
    if payload.get("slide_2022_recovered", {}).get("station_offset_width_label") != (
        "population_spread_not_measurement_sigma"
    ):
        raise ValueError("2022 station widths must stay population_spread_not_measurement_sigma")
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
            "ppt_used_as_alignment_prior": False,
            "ppt_used_as_geometry": False,
            "conditions_treated_as_independent_survey": False,
            "survey_numbers_were_invented": False,
            "slide_summary_feasibility_only": True,
        }
    )
    return state


def _int_keyed(mapping: Mapping[Any, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in mapping.items()}


def pdf_provenance(config: Mapping[str, Any], year: int) -> dict[str, Any]:
    meta = config["source_pdfs"][year]
    path = resolve_under_root(project_root(), str(meta["relative"]))
    return {
        **dict(meta),
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def c_dx_from_layer_x(x_l0: float, x_l2: float) -> float:
    return (float(x_l0) - float(x_l2)) / 2.0


def cad_normal_to_provisional_rx_ry(
    nominal: SequenceLike,
    measured: SequenceLike,
) -> dict[str, Any]:
    """Small-angle rx, ry in the PPT/CAD frame from a plane normal.

    Uses the same composition as Calypso, T * Rz * Ry * Rx, applied to
    n0=(0,0,-1) inside the CAD axes.  Result is CAD-frame and
    ``frame_not_yet_reconciled``.
    """
    n0 = np.asarray(nominal, dtype=np.float64)
    n_meas = np.asarray(measured, dtype=np.float64)
    n0 = n0 / np.linalg.norm(n0)
    n_meas = n_meas / np.linalg.norm(n_meas)
    cosine = float(np.clip(np.dot(n0, n_meas), -1.0, 1.0))
    total_tilt_mrad = 1000.0 * math.acos(cosine)
    transverse_mrad = 1000.0 * float(np.linalg.norm(n_meas[:2]))
    # n' = R * n0 = -R[:,2] ≈ (-ry, rx, -1) for rz=0
    ry_cad_mrad = -1000.0 * float(n_meas[0])
    rx_cad_mrad = 1000.0 * float(n_meas[1])
    return {
        "nominal_normal_cad": [float(item) for item in n0],
        "measured_normal_cad": [float(item) for item in measured],
        "measured_normal_cad_unit": [float(item) for item in n_meas],
        "total_tilt_from_dot_product_mrad": total_tilt_mrad,
        "total_tilt_from_transverse_normal_mrad": transverse_mrad,
        "provisional_rx_cad_mrad": rx_cad_mrad,
        "provisional_ry_cad_mrad": ry_cad_mrad,
        "composition": "T * Rz * Ry * Rx applied to CAD n0=(0,0,-1), rz=0",
        "label": "frame_not_yet_reconciled",
        "not_calypso_station_ry": True,
        "not_alignment_prior": True,
    }


SequenceLike = Any


def fit_layer_x_trend(xs: Mapping[int, float], *, pitch_mm: float) -> dict[str, Any]:
    layers = sorted(xs)
    z_mm = np.asarray([float(layer) * float(pitch_mm) for layer in layers], dtype=np.float64)
    x_mm = np.asarray([float(xs[layer]) for layer in layers], dtype=np.float64)
    slope, intercept = np.polyfit(z_mm, x_mm, 1)
    predicted = intercept + slope * z_mm
    residual = x_mm - predicted
    return {
        "z_assignment": "z_rel = layer_id * LAYERPITCH, LAYERPITCH is design not survey",
        "layer_pitch_mm": float(pitch_mm),
        "layer_pitch_is_survey": False,
        "z_rel_mm": [float(item) for item in z_mm],
        "x_mm": [float(item) for item in x_mm],
        "slope_dx_over_dz": float(slope),
        "intercept_mm": float(intercept),
        "residual_mm": [float(item) for item in residual],
        "rms_residual_mm": float(np.sqrt(np.mean(residual**2))),
        "layer_mean_coherent_tilt_candidate_mrad": 1000.0 * float(slope),
        "two_point_l0_l2_slope_mrad": 1000.0 * (float(xs[2]) - float(xs[0])) / (2.0 * float(pitch_mm)),
        "name": "layer_mean_coherent_tilt_candidate",
        "not_calypso_station_ry": True,
        "not_alignment_prior": True,
    }


def station_layer_average_check(
    layer_shifts: Mapping[int, SequenceLike],
    station_means: Mapping[int, SequenceLike],
    *,
    atol_mm: float = 5.0e-4,
) -> dict[str, Any]:
    grouped: dict[int, list[np.ndarray]] = {}
    for layer, shift in layer_shifts.items():
        grouped.setdefault(STATION_FROM_LAYER[int(layer)], []).append(np.asarray(shift, dtype=np.float64))
    rows = []
    all_ok = True
    for station, shifts in sorted(grouped.items()):
        mean = np.mean(np.vstack(shifts), axis=0)
        quoted = np.asarray(station_means[station], dtype=np.float64)
        match = bool(np.allclose(mean, quoted, atol=atol_mm, rtol=0.0))
        all_ok = all_ok and match
        rows.append(
            {
                "station_id": station,
                "n_layers": len(shifts),
                "layer_average_mm": [float(item) for item in mean],
                "quoted_station_mean_mm": [float(item) for item in quoted],
                "delta_mm": [float(item) for item in (mean - quoted)],
                "reproduces_quoted_station_mean": match,
            }
        )
    ift = next(row for row in rows if row["station_id"] == IFT_STATION_ID)
    return {
        "stations": rows,
        "all_station_means_reproduced": all_ok,
        "layers_0_1_2_are_ift": bool(ift["reproduces_quoted_station_mean"] and all_ok),
        "atol_mm": atol_mm,
    }


def survey_2021_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    recovered = config["slide_2021_recovered"]
    pdf = pdf_provenance(config, 2021)
    angles = cad_normal_to_provisional_rx_ry(
        recovered["nominal_normal_cad"],
        recovered["measured_normal_cad"],
    )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "source_pdf": pdf,
        "label": "slide_summary_not_alignment_prior",
        "availability": "slide_summary",
        "frame": "2021_survey_CAD",
        "frame_status": "frame_not_yet_reconciled",
        "cern_survey_date": recovered["cern_survey_date"],
        "surveyed_by": recovered["surveyed_by"],
        "survey_file_author": recovered["survey_file_author"],
        "survey_file_units": "meter / nominal CAD positions",
        "ift_if_plane": {
            "n_measured_points": recovered["n_if_points"],
            "provenance": (
                "CERN surveyed 4 positions onto the IFT I/F plane on 2021-11-25. "
                "CAD is in its nominal position (no 12 mm vertical shift used for "
                "the three tracker stations). Green volume is the simplified IFT."
            ),
            "cad_in_nominal_position": recovered["cad_in_nominal_position"],
            "cad_has_12mm_vertical_shift_of_three_trk_stations": recovered[
                "cad_has_12mm_vertical_shift_of_three_trk_stations"
            ],
            "source_page": recovered["source_pages"]["four_point_if_and_cad_axes"],
        },
        "cad_axes": recovered["cad_axes"],
        "normals_and_provisional_angles": angles,
        "center_shift_cad_mm": {
            "xyz_mm": list(recovered["center_shift_cad_mm"]),
            "slide_text": "Shift / center: about 0.4 mm",
            "source_page": recovered["source_pages"]["side_view_normals_and_center_shift"],
            "frame_status": "frame_not_yet_reconciled",
        },
        "top_view": {
            "note": "Small rotation / z axis (white nominal vs red measured I/F rectangle).",
            "quantified_in_slide": False,
            "would_map_to_calypso_ry_if_cad_z_is_vertical": True,
            "source_page": recovered["source_pages"]["top_view_small_rz"],
        },
        "reference_chain": {
            "steps": [
                "CERN 4-point survey of each station I/F",
                "I/F points used as the mechanical reference",
                "UNIGE metrology performed with respect to those I/F points",
            ],
            "ift_treated_as_separated_system_from_station_mechanics": True,
            "unge_metrology_steps": [
                "Metrology#1: plane by plane",
                "Metrology#2: three planes assembled with top plate (Plan_sup)",
            ],
            "unge_screenshot_tables_digitized_as_measurements": False,
            "source_pages": [
                recovered["source_pages"]["unge_two_step"],
                recovered["source_pages"]["reference_chain"],
                recovered["source_pages"]["camille_file"],
            ],
        },
        "other_station_backbone_context": recovered["other_station_mean_plane_angles_deg"],
        "may_enter_geometry_candidate": False,
        "may_be_used_as_alignment_prior": False,
    }


def survey_2022_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    recovered = config["slide_2022_recovered"]
    pdf = pdf_provenance(config, 2022)
    station_means = _int_keyed(recovered["station_mean_offset_mm"])
    station_widths = _int_keyed(recovered["station_offset_width_mm"])
    layer_shifts = _int_keyed(recovered["layer_mean_shift_side0_mm"])
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "source_pdf": pdf,
        "label": "slide_summary_not_alignment_prior",
        "availability": "slide_summary",
        "frame": "2022_survey_adjusted_FASER",
        "frame_construction": (
            "Support beam held at nominal location relates CAD to FASER up to a "
            "translation. The three translations are fixed by matching average "
            "nominal and average survey positions of front sensors (pigtail side) "
            "in Stations 1-3. IFT/station 0 is not used in that translation. "
            "GeoModel dump has no alignment applied."
        ),
        "front_sensor_pigtail_side_only": recovered["front_sensor_pigtail_side_only"],
        "translation_from_stations_1_to_3_front_sensor_average": recovered[
            "translation_from_stations_1_to_3_front_sensor_average"
        ],
        "ift_not_used_in_translation": recovered["ift_not_used_in_translation"],
        "station_mean_offset_mm": {
            str(station): {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
            for station, xyz in sorted(station_means.items())
        },
        "station0_ift_mean_offset_mm": {
            "x": station_means[0][0],
            "y": station_means[0][1],
            "z": station_means[0][2],
            "is_ift": True,
        },
        "station_offset_width_mm": {
            str(station): {
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "label": "population_spread_not_measurement_sigma",
                "may_be_used_as_gaussian_prior_uncertainty": False,
            }
            for station, xyz in sorted(station_widths.items())
        },
        "layer_mean_shift_side0_mm": {
            str(layer): {
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "station_id": STATION_FROM_LAYER[int(layer)],
            }
            for layer, xyz in sorted(layer_shifts.items())
        },
        "support_beam_horizontal_misalignment_urad": recovered[
            "support_beam_horizontal_misalignment_urad"
        ],
        "support_beam_note": (
            "Stations 1-3 X offsets are described as consistent with a 350 µrad "
            "horizontal misalignment of the support beam. That is a residual "
            "CAD-to-FASER rotation, not a measurement sigma."
        ),
        "ift_z_comment": (
            "The IFT z shift is the only one the 2022 talk says one may want to "
            "account for in the nominal geometry. This stage does not write geometry."
        ),
        "may_enter_geometry_candidate": False,
        "may_be_used_as_alignment_prior": False,
    }


def ift_layer_contrast_reconstruction(
    config: Mapping[str, Any],
    summary_2022: Mapping[str, Any],
) -> dict[str, Any]:
    recovered = config["slide_2022_recovered"]
    station_means = _int_keyed(recovered["station_mean_offset_mm"])
    layer_shifts = _int_keyed(recovered["layer_mean_shift_side0_mm"])
    check = station_layer_average_check(layer_shifts, station_means)
    xs = {layer: float(layer_shifts[layer][0]) for layer in IFT_LAYER_IDS}
    c_dx = c_dx_from_layer_x(xs[0], xs[2])
    delta = float(xs[0]) - float(xs[2])
    pitch = float(config["physics_scales"]["ift_layer_pitch_mm"])
    trend = fit_layer_x_trend(xs, pitch_mm=pitch)
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "label": "slide_summary_not_alignment_prior",
        "station_average_check": check,
        "layers_0_1_2_confirmed_as_ift": check["layers_0_1_2_are_ift"],
        "ift_layer_x_mm": xs,
        "ift_layer_xyz_mm": {str(layer): list(layer_shifts[layer]) for layer in IFT_LAYER_IDS},
        "C_dx_slide_mm": c_dx,
        "C_dx_definition": "C_dx_slide=(x_L0-x_L2)/2",
        "delta_x_L0_minus_L2_mm": delta,
        "layer_x_trend": trend,
        "same_l0_l2_contrast_expressed_two_ways": {
            "C_dx_slide_mm": c_dx,
            "two_point_tilt_mrad": trend["two_point_l0_l2_slope_mrad"],
            "identity": "two_point_slope = -C_dx_slide / LAYERPITCH",
            "independent_constraints": False,
            "note": (
                "The 2022 L0/L2 x contrast can be written as C_dx_slide or as an "
                "x(z) slope. Those are the same degree of freedom, not a global "
                "tilt plus an independent internal contrast."
            ),
        },
        "layer_mean_coherent_tilt_candidate_mrad": trend["layer_mean_coherent_tilt_candidate_mrad"],
        "not_calypso_station_ry": True,
        "may_enter_geometry_candidate": False,
        "may_be_used_as_alignment_prior": False,
        "frame": summary_2022["frame"],
    }


def survey_calypso_frame_reconciliation(
    *,
    config: Mapping[str, Any],
    summary_2021: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    physics = config["physics_scales"]
    angles = summary_2021["normals_and_provisional_angles"]
    existing_path = resolve_under_root(project_root(), str(config["existing_conditions_summary"]))
    existing = None
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    comparison = None
    if existing is not None:
        comparison = {
            "allowed": True,
            "reason": (
                "2022 offsets already live in the survey-adjusted FASER frame "
                "that was translated onto GeoModel stations 1-3. Numeric "
                "comparison with /Tracker/Align is allowed only as "
                "slide_summary vs existing_conditions_state."
            ),
            "slide_C_dx_mm": contrast["C_dx_slide_mm"],
            "conditions_C_dx_cond_mm": existing.get("C_dx_cond_mm"),
            "slide_layer_mean_coherent_tilt_candidate_mrad": contrast[
                "layer_mean_coherent_tilt_candidate_mrad"
            ],
            "conditions_station0_Stations_ry_mrad": existing.get("station0_Stations_ry_cond_mrad"),
            "conditions_ift_plane_ry_mrad": existing.get("ift_plane_ry_used_by_reconstruction_mrad"),
            "slide_label": "slide_summary_not_alignment_prior",
            "conditions_label": "existing_conditions_state",
            "origin_upgrade": None,
            "numeric_closeness_does_not_imply_survey_origin": True,
        }
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "frames": {
            "survey_2021_cad": {
                "x": "horizontal",
                "y": "beam / LoS",
                "z": "vertical up",
                "origin": "nominal CAD / I/F plane",
                "evidence_pages": [5, 6, 7, 8],
                "status": "frame_not_yet_reconciled",
            },
            "survey_2022_adjusted_faser": {
                "x": "intended FASER / Calypso x (horizontal)",
                "y": "intended FASER / Calypso y (vertical)",
                "z": "intended FASER / Calypso z (beam)",
                "origin": (
                    "translation fixed so <front sensors, stations 1-3> match "
                    "unaligned GeoModel; IFT not used"
                ),
                "residual_rotation": "350 µrad support-beam horizontal misalignment quoted for stations 1-3",
                "status": "translation_reconciled_axes_intended_equal",
            },
            "calypso_global": {
                "x": "horizontal",
                "y": "vertical",
                "z": "beam, downstream positive",
                "evidence": (
                    f"TrackerAlignDBTool applies T*Rz*Ry*Rx. A +10 mrad IFT ry "
                    f"moves IFT x by {physics['calypso_plus_10mrad_ry_observed_dx_mm']} mm, "
                    f"matching dx ≈ ry * z_station0 with z_station0="
                    f"{physics['station0_nominal_z_mm']} mm (nominal, not survey)."
                ),
                "stations_planes_frame": "global",
                "interface_modules_frame": "local",
                "rotation_convention": "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx); mm / rad",
            },
            "calypso_local_interface": {
                "meaning": "/Tracker/Align/Interface1/2/3 module-in-plane transforms",
                "not_used_as_survey": True,
            },
        },
        "candidate_2021_cad_to_calypso": {
            "coordinate_permutation": "(x,y,z)_cal = (s_x * x_cad, s_y * z_cad, s_z * y_cad)",
            "default_sign_hypothesis": [1, 1, 1],
            "default_sign_meaning": "CAD x→Calypso x, CAD z→Calypso y, CAD y→Calypso z if Beam L+ is +z and V+ is +y",
            "uniquely_determined": False,
            "missing": [
                "sign of Beam L+ versus Calypso +z",
                "sign of CAD x versus Calypso +x",
                "common origin after the 2022 stations-1-3 translation",
            ],
            "rotation_map_under_default_signs": {
                "rx_cad": "rx_cal",
                "ry_cad": "rz_cal (about the beam)",
                "rz_cad": "ry_cal (about vertical; this is the alignment-relevant station ry)",
            },
            "implication": (
                "The 2021 I/F measured-plane normal tilt is CAD rx/ry, which under "
                "the default permutation is Calypso rx/rz, not Calypso station ry. "
                "The Calypso-ry candidate in 2021 is the unquantified page-6 in-plane yaw."
            ),
        },
        "mapped_2021_under_default_signs_only": {
            "label": "frame_not_yet_reconciled",
            "provisional_rx_cad_mrad": angles["provisional_rx_cad_mrad"],
            "provisional_ry_cad_mrad": angles["provisional_ry_cad_mrad"],
            "maps_to_calypso_rx_mrad": angles["provisional_rx_cad_mrad"],
            "maps_to_calypso_rz_mrad": angles["provisional_ry_cad_mrad"],
            "maps_to_calypso_station_ry": False,
            "numeric_comparison_with_2022_or_conditions_allowed": False,
        },
        "comparison_gates": {
            "2021_tilt_vs_2022_layer_trend": False,
            "2021_tilt_vs_tracker_align": False,
            "2022_cdx_or_tilt_vs_tracker_align_as_existing_conditions_state": True,
        },
        "existing_conditions_numeric_comparison": comparison,
        "explicit_transform_completed": False,
        "may_compare_2021_with_2022_as_same_number": False,
        "may_enter_geometry_candidate": False,
    }


def _scan_point(sigma_ry: float | None, sigma_cdx: float | None, family: str) -> dict[str, Any]:
    return {"family": family, "sigma_ry_mrad": sigma_ry, "sigma_cdx_mm": sigma_cdx}


def prior_scan_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    scan = config["prior_scan"]
    points = [_scan_point(None, None, "none")]
    for sigma_ry in scan["sigma_ry_mrad"]:
        points.append(_scan_point(float(sigma_ry), None, "ry_only"))
    for sigma_c in scan["sigma_cdx_mm"]:
        points.append(_scan_point(None, float(sigma_c), "cdx_only"))
    for sigma_ry in scan["sigma_ry_mrad"]:
        for sigma_c in scan["sigma_cdx_mm"]:
            points.append(_scan_point(float(sigma_ry), float(sigma_c), "both_hypothetical_independent"))
    return points


def slide_summary_prior_feasibility(
    *,
    config: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    jacobian = covariance_matched_collinear_jacobian(config)
    physics = config["physics_scales"]
    rules = config["unlock"]
    frozen = config["entry_61_frozen"]
    points = []
    for item in prior_scan_grid(config):
        metrics = combined_identifiability(
            jacobian,
            sigma_r_mm=float(physics["measurement_sigma_mm"]),
            sigma_ry_mrad=item["sigma_ry_mrad"],
            sigma_cdx_mm=item["sigma_cdx_mm"],
            rank_tolerance=float(rules["rank_tolerance"]),
        )
        points.append(
            {
                **item,
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
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "mode": "slide_summary_feasibility_only",
        "did_rebuild_2024_r0022_jacobian": False,
        "central_values": {
            "C_dx_slide_mm": contrast["C_dx_slide_mm"],
            "layer_mean_coherent_tilt_candidate_mrad": contrast[
                "layer_mean_coherent_tilt_candidate_mrad"
            ],
            "label": "slide_summary_feasibility_only",
            "not_alignment_prior": True,
            "not_calypso_station_ry": True,
            "note": (
                "Fisher information does not use the central values. They are "
                "recorded only as the slide-derived numbers that would sit at "
                "the center if the original survey later confirms them."
            ),
        },
        "same_2022_contrast_cannot_supply_both_priors": True,
        "both_family_is_hypothetical_independent_measurements": True,
        "station_sd_used_as_sigma": False,
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
        "leftover_after_tight_complement": {
            "sigma_cdx_mm_after_tight_ry": frozen["leftover_sigma_cdx_mm_after_tight_ry"],
            "sigma_ry_mrad_after_tight_cdx": frozen["leftover_sigma_ry_mrad_after_tight_cdx"],
            "leftover_cdx_finer_than_strip_pitch": frozen["leftover_cdx_finer_than_strip_pitch"],
        },
        "answer": (
            "If the slide-derived central values are later confirmed by the "
            "original survey, a single independent Gaussian prior unlocks the "
            "frozen collinear Jacobian at the coarsest predeclared grid point "
            "σ(ry)=20 mrad or σ(C_dx)=0.345 mm (the leftover-after-tight-ry "
            "value). The frozen entry-61 degeneracy-breaking threshold "
            "σ(C_dx)=0.315 mm also unlocks. Useful precision remains "
            "0.5 mrad / 0.080 mm. After a tight ry prior the leftover σ(C_dx) "
            "is still 0.345 mm, so a useful C_dx must be a direct L0/L2 "
            "metrology covariance, not this track sample. The 2022 "
            "station/module widths are population spread and were not used as σ."
        ),
        "geometry_candidate": False,
        "may_enter_geometry_candidate": False,
    }


def decide_next_stage(
    *,
    summary_2021: Mapping[str, Any],
    contrast: Mapping[str, Any],
    frames: Mapping[str, Any],
    feasibility: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "question": (
            "Using only the two existing survey slides, can we obtain mutually "
            "coordinate-consistent, physically well-defined IFT global tilt and "
            "internal L0/L2 contrast candidates, and converge the remaining "
            "uncertainty onto missing measurement covariance rather than "
            "treating track degeneracy as an unknown source?"
        ),
        "answer": {
            "internal_l0_l2_contrast_defined": True,
            "C_dx_slide_mm": contrast["C_dx_slide_mm"],
            "global_tilt_candidates_defined_per_slide": True,
            "mutually_coordinate_consistent": False,
            "remaining_uncertainty": "missing_measurement_covariance",
            "track_degeneracy_is_unknown_source": False,
        },
        "detail": (
            "2022 gives a physically clear IFT L0/L2 x contrast "
            f"C_dx_slide={contrast['C_dx_slide_mm']:.7f} mm after the "
            "layers-0-1-2 average was shown to reproduce the Station-0 mean. "
            "The same contrast can be written as a layer-mean x(z) slope of "
            f"{contrast['layer_mean_coherent_tilt_candidate_mrad']:.2f} mrad; "
            "that slope is a layer_mean_coherent_tilt_candidate, not Calypso "
            "station ry, and is not a second independent constraint. 2021 "
            "gives a physically clear I/F-plane normal tilt of about 5.0 mrad "
            f"(provisional CAD rx={summary_2021['normals_and_provisional_angles']['provisional_rx_cad_mrad']:.2f} mrad, "
            f"ry={summary_2021['normals_and_provisional_angles']['provisional_ry_cad_mrad']:.2f} mrad) "
            "but that object is the top I/F plate in the CAD frame; under the "
            "only candidate axis permutation it maps to Calypso rx/rz, not "
            "station ry, and the signs/origin are not unique. Therefore the "
            "two slides do not yet share one reconciled coordinate system. "
            "The remaining unknown is the original survey measurement "
            "covariance (and those frame signs), not an unidentified track "
            "degeneracy: the Jacobian degeneracy is already quantified, and "
            "the slides already name the physical candidates."
        ),
        "explicit_transform_completed": frames["explicit_transform_completed"],
        "enough_to_write_alignment_prior": False,
        "enough_to_write_geometry": False,
        "survey_derived": False,
        "next_allowed_step": (
            "Keep residual_dq_monitoring_only. Ingest the original 2021 CERN/"
            "UNIGE table and 2022 wafer-level survey file with measurement "
            "covariance, then apply an explicit unique CAD→Calypso transform. "
            "Do not promote these slide summaries to a prior or geometry."
        ),
        "feasibility_coarsest_unlocking_sigma_ry_mrad": feasibility["coarsest_unlocking_sigma_ry_mrad"],
        "feasibility_coarsest_unlocking_sigma_cdx_mm": feasibility["coarsest_unlocking_sigma_cdx_mm"],
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    summary_2021 = survey_2021_summary(config)
    summary_2022 = survey_2022_summary(config)
    contrast = ift_layer_contrast_reconstruction(config, summary_2022)
    if not contrast["layers_0_1_2_confirmed_as_ift"]:
        raise ValueError("layers 0-2 do not reproduce the Station-0 average")
    frames = survey_calypso_frame_reconciliation(
        config=config,
        summary_2021=summary_2021,
        contrast=contrast,
    )
    feasibility = slide_summary_prior_feasibility(config=config, contrast=contrast)
    decision = decide_next_stage(
        summary_2021=summary_2021,
        contrast=contrast,
        frames=frames,
        feasibility=feasibility,
    )
    return {
        "survey_2021_summary": summary_2021,
        "survey_2022_summary": summary_2022,
        "ift_layer_contrast_reconstruction": contrast,
        "survey_calypso_frame_reconciliation": frames,
        "slide_summary_prior_feasibility": feasibility,
        "next_stage_decision": decision,
    }
