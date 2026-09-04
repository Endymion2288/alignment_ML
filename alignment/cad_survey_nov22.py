"""Parse cad_survey_nov22, audit frames/covariance, and test prior ingest.

Treats the Nov 2022 per-sensor CAD/survey dump as immutable evidence.
Does not retrain, change the frozen association policy, map unconfirmed
tilts to Calypso station ry, treat population Sigma as a Gaussian prior,
or emit a geometry candidate / alignment payload.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.survey_derived_prior_interface import UnconfirmedRyMappingError
from alignment.survey_metrology_iov_infrastructure import (
    empty_constraint_catalog,
    load_infrastructure_config,
    validate_constraint,
)
from alignment.survey_summary_reconstruction import c_dx_from_layer_x
from alignment.true_cluster_local_residual import RESIDUAL_KIND, common_operating_state

SCHEMA_VERSION = "faser-cad-survey-nov22-frame-covariance-audit-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "cad_survey_nov22_frame_covariance_audit_v1.yaml"
DECISION = (
    "cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_"
    "measurement_covariance_and_cross_year_iov"
)
STATION_NAMES = {0: "IFT_Interface", 1: "Upstream", 2: "Central", 3: "Downstream"}
PHI_MODULE_LABELS = {0: "Bottom", 1: "LowerMiddle", 2: "UpperMiddle", 3: "Top"}
ETA_MODULE_LABELS = {-1: "Starboard", 1: "Port"}
SIDE_LABELS = {0: "Upper_pigtail_front", 1: "Lower"}
GLOBAL_LAYER_TO_STATION = {index: index // 3 for index in range(12)}
GLOBAL_LAYER_TO_STATION_LAYER = {index: index % 3 for index in range(12)}
VECTOR_RE = re.compile(
    r"Vector\s*\(\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)
SENSOR_RE = re.compile(
    r"^\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]\s*:\s*"
    r"\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]\s*:\s*"
    r"\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]"
)
STATION_SUMMARY_RE = re.compile(
    r"^Station\s+(\d+)\s*:\s*Delta \(x, y, z\) = \(\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\),\s*"
    r"Sigma \(x, y, z\) = \(\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)
LAYER_SUMMARY_RE = re.compile(
    r"^Layer\s+(\d+)\s*:\s*Delta \(x, y, z\) = \(\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "geometry_candidate_from_cad_survey",
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
    "do_not_map_stereo_or_layerpitch_to_station_ry",
    "do_not_map_planes_rotations_to_station_ry",
    "do_not_guess_index_tuple_meanings",
    "do_not_modify_original_survey_file",
    "do_not_expand_ml_or_module_map",
    "do_not_expand_collision_track_statistics",
    "synthetic_prior_feasibility_only",
    "slide_summary_feasibility_only",
    "software_fd_sensitivity_only",
)


class IndexTupleGuessError(ValueError):
    """Raised when an index-tuple meaning is asserted without source evidence."""


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"cad_survey_nov22 config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected cad_survey_nov22 schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"cad_survey_nov22 config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"cad_survey_nov22 config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    frozen = payload.get("entry_61_frozen") or {}
    if float(frozen.get("degeneracy_breaking_sigma_ry_mrad") or 0.0) != 20.0:
        raise ValueError("entry-61 degeneracy-breaking σ(ry) must stay 20 mrad")
    if float(frozen.get("degeneracy_breaking_sigma_cdx_mm") or 0.0) != 0.315:
        raise ValueError("entry-61 degeneracy-breaking σ(C_dx) must stay 0.315 mm")
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
            "station_sd_used_as_sigma": False,
            "slide_summary_feasibility_only": True,
            "original_survey_file_modified": False,
        }
    )
    return state


def git_head_sha(root: Path | None = None) -> str | None:
    cwd = project_root() if root is None else Path(root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _xyz(values: Sequence[float]) -> dict[str, float]:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def _mean_and_spreads(rows: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    array = np.asarray(rows, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute mean/std of an empty sensor set")
    mean = np.mean(array, axis=0)
    if array.shape[0] == 1:
        std_sample = np.zeros(3, dtype=np.float64)
        std_pop = np.zeros(3, dtype=np.float64)
    else:
        std_sample = np.std(array, axis=0, ddof=1)
        std_pop = np.std(array, axis=0, ddof=0)
    return {"mean": mean, "std_sample_ddof1": std_sample, "std_population_ddof0": std_pop}


def parse_cad_survey_nov22_text(text: str) -> dict[str, Any]:
    """Parse the immutable dump.  Does not interpret index physics."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("cad_survey_nov22 is empty")
    delta_match = VECTOR_RE.search(lines[0])
    if delta_match is None:
        raise ValueError("cad_survey_nov22 is missing the Delta FASER vector")
    delta = [float(delta_match.group(i)) for i in range(1, 4)]
    sensors: list[dict[str, Any]] = []
    quoted_stations: dict[int, dict[str, Any]] = {}
    quoted_layers: dict[int, dict[str, Any]] = {}
    for line in lines[1:]:
        sensor_match = SENSOR_RE.match(line.strip())
        if sensor_match:
            station, layer, phi, eta, side = (int(sensor_match.group(i)) for i in range(1, 6))
            corrected = [float(sensor_match.group(i)) for i in range(6, 9)]
            offset = [float(sensor_match.group(i)) for i in range(9, 12)]
            nominal = [corrected[i] - offset[i] for i in range(3)]
            sensors.append(
                {
                    "station_id": station,
                    "layer_id": layer,
                    "phi_module": phi,
                    "eta_module": eta,
                    "side": side,
                    "index_tuple": [station, layer, phi, eta, side],
                    "corrected_mm": corrected,
                    "offset_from_nominal_mm": offset,
                    "implied_nominal_mm": nominal,
                    "source_line": line.rstrip("\n"),
                }
            )
            continue
        station_match = STATION_SUMMARY_RE.match(line.strip())
        if station_match:
            station = int(station_match.group(1))
            quoted_stations[station] = {
                "delta_mm": [float(station_match.group(i)) for i in range(2, 5)],
                "sigma_mm": [float(station_match.group(i)) for i in range(5, 8)],
                "side_selection": 0,
                "sigma_label": "population_spread_not_measurement_sigma",
            }
            continue
        layer_match = LAYER_SUMMARY_RE.match(line.strip())
        if layer_match:
            layer = int(layer_match.group(1))
            quoted_layers[layer] = {
                "delta_mm": [float(layer_match.group(i)) for i in range(2, 5)],
                "side_selection": 0,
                "station_id": GLOBAL_LAYER_TO_STATION[layer],
                "station_layer_id": GLOBAL_LAYER_TO_STATION_LAYER[layer],
            }
    if len(sensors) != 192:
        raise ValueError(f"expected 192 sensors, parsed {len(sensors)}")
    unique_tuples = {tuple(row["index_tuple"]) for row in sensors}
    if len(unique_tuples) != 192:
        raise ValueError(f"expected 192 unique index tuples, parsed {len(unique_tuples)}")
    if set(quoted_stations) != {0, 1, 2, 3}:
        raise ValueError(f"incomplete station summaries: {sorted(quoted_stations)}")
    if set(quoted_layers) != set(range(12)):
        raise ValueError(f"incomplete layer summaries: {sorted(quoted_layers)}")
    return {
        "delta_faser_mm": delta,
        "delta_faser_source_line": lines[0].rstrip("\n"),
        "n_sensors": len(sensors),
        "sensors": sensors,
        "quoted_station_summaries": quoted_stations,
        "quoted_layer_summaries": quoted_layers,
        "summary_side_selection": 0,
        "file_states_summaries_use_side_0_only": True,
    }


def parse_cad_survey_nov22_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    parsed = parse_cad_survey_nov22_text(raw)
    parsed["source_path"] = str(source)
    parsed["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    parsed["size_bytes"] = source.stat().st_size
    parsed["n_lines"] = raw.count("\n") + (0 if raw.endswith("\n") or not raw else 1)
    parsed["original_bytes_unmodified"] = True
    return parsed


def reproduce_quoted_summaries(
    parsed: Mapping[str, Any],
    *,
    reprint_atol_mm: float = 5.0e-4,
) -> dict[str, Any]:
    """Reproduce the file's own summary block.

    Per-sensor lines are printed to 3 decimals.  Station/layer summaries keep
    higher precision.  Exact reproduction is the parsed summary block itself;
    recomputation from the rounded sensor lines is a rounding-consistency check.
    """
    side0 = [row for row in parsed["sensors"] if int(row["side"]) == 0]
    by_station: dict[int, list[list[float]]] = {}
    by_global_layer: dict[int, list[list[float]]] = {}
    for row in side0:
        by_station.setdefault(int(row["station_id"]), []).append(row["offset_from_nominal_mm"])
        global_layer = 3 * int(row["station_id"]) + int(row["layer_id"])
        by_global_layer.setdefault(global_layer, []).append(row["offset_from_nominal_mm"])
    station_rows = []
    means_ok = True
    sigma_pop_closer = True
    for station in sorted(by_station):
        stats = _mean_and_spreads(by_station[station])
        quoted = parsed["quoted_station_summaries"][station]
        mean_residual = stats["mean"] - np.asarray(quoted["delta_mm"], dtype=np.float64)
        pop_residual = stats["std_population_ddof0"] - np.asarray(quoted["sigma_mm"], dtype=np.float64)
        sample_residual = stats["std_sample_ddof1"] - np.asarray(quoted["sigma_mm"], dtype=np.float64)
        mean_ok = bool(np.all(np.abs(mean_residual) <= reprint_atol_mm))
        pop_closer = bool(np.linalg.norm(pop_residual) < np.linalg.norm(sample_residual))
        means_ok = means_ok and mean_ok
        sigma_pop_closer = sigma_pop_closer and pop_closer
        station_rows.append(
            {
                "station_id": station,
                "n_side0_sensors": len(by_station[station]),
                "recomputed_from_printed_sensors_delta_mm": [float(item) for item in stats["mean"]],
                "recomputed_population_std_ddof0_mm": [float(item) for item in stats["std_population_ddof0"]],
                "recomputed_sample_std_ddof1_mm": [float(item) for item in stats["std_sample_ddof1"]],
                "quoted_delta_mm": list(quoted["delta_mm"]),
                "quoted_sigma_mm": list(quoted["sigma_mm"]),
                "mean_residual_vs_quoted_mm": [float(item) for item in mean_residual],
                "population_std_residual_vs_quoted_mm": [float(item) for item in pop_residual],
                "mean_matches_quoted_within_print_rounding": mean_ok,
                "population_std_closer_to_quoted_than_sample_std": pop_closer,
            }
        )
    layer_rows = []
    layers_ok = True
    for layer in sorted(by_global_layer):
        stats = _mean_and_spreads(by_global_layer[layer])
        quoted = parsed["quoted_layer_summaries"][layer]["delta_mm"]
        residual = stats["mean"] - np.asarray(quoted, dtype=np.float64)
        match = bool(np.all(np.abs(residual) <= reprint_atol_mm))
        layers_ok = layers_ok and match
        layer_rows.append(
            {
                "global_layer_id": layer,
                "station_id": GLOBAL_LAYER_TO_STATION[layer],
                "station_layer_id": GLOBAL_LAYER_TO_STATION_LAYER[layer],
                "n_side0_sensors": len(by_global_layer[layer]),
                "recomputed_from_printed_sensors_delta_mm": [float(item) for item in stats["mean"]],
                "quoted_delta_mm": list(quoted),
                "mean_residual_vs_quoted_mm": [float(item) for item in residual],
                "mean_matches_quoted_within_print_rounding": match,
            }
        )
    quoted_xs = {
        layer: float(parsed["quoted_layer_summaries"][layer]["delta_mm"][0])
        for layer in IFT_LAYER_IDS
    }
    quoted_c_dx = c_dx_from_layer_x(quoted_xs[0], quoted_xs[2])
    recomputed_xs = {
        layer: float(next(row["recomputed_from_printed_sensors_delta_mm"][0] for row in layer_rows if row["global_layer_id"] == layer))
        for layer in IFT_LAYER_IDS
    }
    station_from_layers = []
    layers_average_to_stations = True
    for station in range(4):
        layer_ids = [3 * station + layer for layer in range(3)]
        stacked = np.asarray(
            [parsed["quoted_layer_summaries"][layer]["delta_mm"] for layer in layer_ids],
            dtype=np.float64,
        )
        mean = stacked.mean(axis=0)
        quoted = np.asarray(parsed["quoted_station_summaries"][station]["delta_mm"], dtype=np.float64)
        residual = mean - quoted
        match = bool(np.allclose(mean, quoted, atol=1.0e-12, rtol=0.0))
        layers_average_to_stations = layers_average_to_stations and match
        station_from_layers.append(
            {
                "station_id": station,
                "global_layer_ids": layer_ids,
                "mean_of_quoted_layers_mm": [float(item) for item in mean],
                "quoted_station_delta_mm": [float(item) for item in quoted],
                "residual_mm": [float(item) for item in residual],
                "quoted_layers_average_to_quoted_station": match,
            }
        )
    return {
        "n_side0_sensors": len(side0),
        "n_side0_sensors_per_station": 24,
        "n_side0_sensors_per_layer": 8,
        "stations": station_rows,
        "layers": layer_rows,
        "quoted_summaries_parsed_exactly": True,
        "printed_sensors_are_3_decimal": True,
        "all_station_summaries_reproduced": bool(means_ok),
        "all_layer_summaries_reproduced": bool(layers_ok),
        "recomputed_means_match_quoted_within_print_rounding": bool(means_ok and layers_ok),
        "reprint_atol_mm": reprint_atol_mm,
        "quoted_ift_layer_x_mm": quoted_xs,
        "quoted_C_dx_mm": quoted_c_dx,
        "quoted_delta_x_L0_minus_L2_mm": float(quoted_xs[0] - quoted_xs[2]),
        "recomputed_ift_layer_x_mm": recomputed_xs,
        "recomputed_C_dx_mm": c_dx_from_layer_x(recomputed_xs[0], recomputed_xs[2]),
        "entry64_regression_uses_quoted_layer_means_not_printed_sensors": True,
        "sigma_definition_in_file": (
            "population standard deviation (ddof=0) of side=0 sensor offsets from nominal, by station"
        ),
        "sigma_matches_population_std_of_printed_sensors_better_than_sample_std": sigma_pop_closer,
        "sigma_is_population_std_ddof0": True,
        "sigma_is_sample_std_ddof_1": False,
        "sigma_is_population_spread_not_measurement_sigma": True,
        "sigma_may_be_used_as_gaussian_prior_uncertainty": False,
        "quoted_layers_average_to_quoted_stations": station_from_layers,
        "quoted_layers_0_1_2_are_ift": bool(station_from_layers[0]["quoted_layers_average_to_quoted_station"]),
        "all_quoted_layer_triples_average_to_quoted_stations": layers_average_to_stations,
    }


def index_tuple_evidence() -> dict[str, Any]:
    """Record source-backed meanings.  Nothing here is inferred from plots."""
    fields = [
        {
            "tuple_index": 0,
            "name": "station_id",
            "observed_values": [0, 1, 2, 3],
            "calypso_name": "station",
            "physical_meaning": (
                "SCT station: 0=Interface/IFT, 1=Upstream, 2=Central, 3=Downstream"
            ),
            "evidence": [
                "TrackerIdentifier/FaserSCT_ID.h: station / layer / phi_module / eta_module / side",
                "TrackerIdDictFiles/data/IdDictInterface.xml station labels Interface=0 ... Downstream=3",
                "SCT_Identifier.cxx print: 2/1/station/layer/phi/eta/side",
                "SCT_DetectorFactory.cxx partNames Interface, StationA, StationB, StationC with iStation 0-3",
                "GeoModelTestAlg.cxx wafer_id(station, plane, row, module, sensor)",
            ],
            "status": "confirmed",
        },
        {
            "tuple_index": 1,
            "name": "layer_id",
            "observed_values": [0, 1, 2],
            "calypso_name": "layer",
            "physical_meaning": "three planes per station, upstream to downstream inside the station",
            "evidence": [
                "FaserSCT_ID.h: layer 0 to 2, three layers per station",
                "IdDictInterface.xml layer Upstream=0 Central=1 Downstream=2",
                "SCT_Station.cxx: id.setLayer(iLayer); GeoIdentifierTag(iLayer); iLayer in [0, numLayers)",
                "numerology.setNumLayers from SCTFASERGENERAL NUMLAYERS=3",
            ],
            "status": "confirmed",
        },
        {
            "tuple_index": 2,
            "name": "phi_module",
            "observed_values": [0, 1, 2, 3],
            "calypso_name": "phi_module",
            "physical_meaning": "precision / vertical module row: Bottom=0 ... Top=3",
            "evidence": [
                "FaserSCT_ID.h: phi_module 0 to 3, precision/vertical direction",
                "IdDictInterface.xml Bottom=0 LowerMiddle=1 UpperMiddle=2 Top=3",
                "SCT_Frame.cxx: id.setPhiModule(module) with module in 0..3; iy = -3 + 2*module along local y",
                "GeoModelTestAlg.cxx maps the third wafer_id argument as row = phi_module",
            ],
            "status": "confirmed",
        },
        {
            "tuple_index": 3,
            "name": "eta_module",
            "observed_values": [-1, 1],
            "calypso_name": "eta_module",
            "physical_meaning": (
                "non-precision / horizontal module: Starboard=-1, Port=+1, facing downstream"
            ),
            "evidence": [
                "FaserSCT_ID.h: eta_module -1 starboard = right facing beam, +1 port = left facing beam",
                "IdDictInterface.xml Starboard=-1 Port=+1; x increases right to left for a right-handed system",
                "SCT_Frame.cxx: id.setEtaModule(iEta) with iEta = iz * etaSign and iz = ±1, never 0",
                "GeoModelTestAlg.cxx skips iModule==0 when NumSCTModulesPerRow is even",
            ],
            "status": "confirmed",
        },
        {
            "tuple_index": 4,
            "name": "side",
            "observed_values": [0, 1],
            "calypso_name": "side",
            "physical_meaning": (
                "sensor side of a module pair: 0=Upper/pigtail/front, 1=Lower. "
                "geomDB SCTBRLMODULE SIDEUPPER=0 so the pigtail/outer side is identifier 0"
            ),
            "evidence": [
                "FaserSCT_ID.h: side 0 to 1, upstream/downstream of pairs of Si crystals",
                "IdDictInterface.xml: Upper=0 Lower=1; 'The upper side is the side with the pigtail'",
                "geomDB.sql SCTBRLMODULE_DATA SIDEUPPER=0",
                "SCT_Module.cxx: outerSideNumber = m_upperSide; innerSideNumber = (m_upperSide) ? 0 : 1",
                "docs/04Nov2022_Survey.pdf: statistics use the front sensor (pigtail side) only",
                "cad_survey_nov22.txt station/layer summaries are labelled 'side = 0 only'",
            ],
            "status": "confirmed",
        },
    ]
    return {
        "index_order": "station, layer, phi_module, eta_module, side",
        "source_of_order": (
            "FaserSCT_ID::wafer_id(station, layer, phi_module, eta_module, side) "
            "and SCT_Identifier::print 2/1/station/layer/phi/eta/side"
        ),
        "guessed": False,
        "fields": fields,
        "file_consistency_not_used_as_definition": (
            "Observed x increases with eta_module and y increases with phi_module "
            "in the dump; that is a consistency check against IdDict, not the "
            "definition of the tuple."
        ),
    }


def source_provenance(config: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    source = resolve_under_root(root, str(config["source"]["relative"]))
    pdf_2022 = resolve_under_root(root, str(config["adjacent_sources"]["survey_2022_pdf"]))
    pdf_2021 = resolve_under_root(root, str(config["adjacent_sources"]["survey_2021_pdf"]))
    expected = str(config["source"]["expected_sha256"])
    digest = sha256_file(source) if source.is_file() else None
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_head_sha(root),
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "source_file": {
            "relative": config["source"]["relative"],
            "path": str(source),
            "exists": source.is_file(),
            "immutable": True,
            "sha256": digest,
            "expected_sha256": expected,
            "sha256_matches_config": digest == expected,
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "n_sensors": parsed["n_sensors"],
            "original_content_modified": False,
        },
        "adjacent_pdfs": {
            "2022": {
                "relative": config["adjacent_sources"]["survey_2022_pdf"],
                "path": str(pdf_2022),
                "exists": pdf_2022.is_file(),
                "sha256": sha256_file(pdf_2022) if pdf_2022.is_file() else None,
                "role": "Casper 2022-11-04 slide source of the station/layer summary tables",
            },
            "2021": {
                "relative": config["adjacent_sources"]["survey_2021_pdf"],
                "path": str(pdf_2021),
                "exists": pdf_2021.is_file(),
                "sha256": sha256_file(pdf_2021) if pdf_2021.is_file() else None,
                "role": "Cadoux 2021-12-10 CAD-frame I/F survey; not this dump",
            },
        },
        "producing_code": {
            "identified": False,
            "candidates": [
                {
                    "what": "CalypsoExamples/GeoModelTest as-built sensor dump",
                    "evidence": (
                        "docs/04Nov2022_Survey.pdf page 4: "
                        "'Use CalypsoExamples/GeoModelTest program to dump as-built sensor positions. "
                        "No alignment applied yet.'"
                    ),
                    "status": "nominal-geometry-half-of-the-offsets-only",
                },
                {
                    "what": "Casper comparison of Franck CAD/survey vs GeoModelTest",
                    "evidence": (
                        "Same PDF: survey data from Franck; support-beam translation from "
                        "Stations 1-3 front-sensor averages; this dump's station/layer numbers "
                        "match the slide tables at higher precision."
                    ),
                    "status": "dump_is_the_comparison_output_not_the_raw_instrument_file",
                },
            ],
            "raw_instrument_file_present": False,
            "per_point_measurement_error_present": False,
        },
        "year_iov": {
            "survey_talk_date": "2022-11-04",
            "survey_data_from": "Franck / Cadoux CAD+survey",
            "may_be_used_as_2024_or_2025_station_rigid_body": False,
            "reason": (
                "Station rigid motion is IOV-specific. A 2022 survey is not a 2024/2025 "
                "station correction without mechanical-stability evidence."
            ),
            "C_dx_policy": "common_static_candidate_until_opening_or_thermal_evidence",
        },
        "index_tuple": index_tuple_evidence(),
        "assumptions": [
            "Per-sensor lines are 3-decimal reprints; quoted station/layer blocks keep the higher-precision means used for C_dx.",
            "File Sigma is the population (ddof=0) spread of side=0 offsets, not a measurement sigma.",
            "This dump is already Casper's 2022 survey-adjusted FASER output, not the native 2021 CAD table.",
            "Native CAD axis signs before that translation remain unresolved and are not used.",
            "Kabsch ry and dz-vs-x ry are diagnostics, not Calypso /Tracker/Align/Stations ry.",
            "A 2022 survey cannot constrain 2024/2025 station rigid motion without mechanical-stability evidence.",
            "C_dx is a common-static candidate until opening/thermal/metrology evidence exists.",
        ],
    }


def _kabsch(nominal: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """Rigid map measured ≈ R @ nominal + t, centroid-centered."""
    nom = np.asarray(nominal, dtype=np.float64)
    meas = np.asarray(measured, dtype=np.float64)
    nom_c = nom.mean(axis=0)
    meas_c = meas.mean(axis=0)
    h_matrix = (nom - nom_c).T @ (meas - meas_c)
    u_mat, _s, vt = np.linalg.svd(h_matrix)
    rotation = vt.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = vt.T @ u_mat.T
    translation = meas_c - rotation @ nom_c
    predicted = (rotation @ nom.T).T + translation
    residual = meas - predicted
    # TrackerAlignDBTool extractAlphaBetaGamma on row-major R.
    siny = max(-1.0, min(1.0, float(rotation[0, 2])))
    ry = math.asin(siny)
    rx = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
    rz = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    return {
        "centroid_nominal_mm": [float(item) for item in nom_c],
        "centroid_measured_mm": [float(item) for item in meas_c],
        "translation_mm": [float(item) for item in translation],
        "rotation_matrix": [[float(item) for item in row] for row in rotation],
        "rx_mrad": 1000.0 * rx,
        "ry_mrad": 1000.0 * ry,
        "rz_mrad": 1000.0 * rz,
        "rms_residual_mm": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "n_points": int(nom.shape[0]),
        "composition": "Kabsch R, then Calypso extractAlphaBetaGamma: ry=asin(R_xz), rx=atan2(-R_yz,R_zz), rz=atan2(-R_xy,R_xx)",
        "not_calypso_station_ry": True,
        "not_alignment_prior": True,
    }


def _linear_ry_from_dz_vs_x(nominal: np.ndarray, offset: np.ndarray) -> dict[str, Any]:
    """Candidate ry from δz ≈ -ry (x-x0), the component layer-x slope cannot fake.

    A rigid Calypso Ry also produces δx ≈ ry (z-z0).  That δx(z) piece is the
    same family as C_dx / layer-mean x, so it is not used as a ry proof.
    """
    x = np.asarray(nominal, dtype=np.float64)[:, 0]
    dz = np.asarray(offset, dtype=np.float64)[:, 2]
    x0 = float(np.mean(x))
    dx = x - x0
    denom = float(np.dot(dx, dx))
    slope = float(np.dot(dx, dz) / denom) if denom > 0.0 else float("nan")
    # δz = -ry * (x-x0)  =>  ry = -slope
    ry_mrad = -1000.0 * slope
    residual = dz - slope * dx
    return {
        "slope_dz_over_dx": slope,
        "ry_candidate_mrad": ry_mrad,
        "rms_residual_mm": float(np.sqrt(np.mean(residual**2))),
        "n_points": int(x.size),
        "identity": "ry_from_dz_vs_x = -d(delta_z)/d(x_nominal)",
        "not_calypso_station_ry": True,
        "not_from_layer_x_slope": True,
        "not_alignment_prior": True,
    }


def geometry_analysis(parsed: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    sensors = parsed["sensors"]
    side0 = [row for row in sensors if int(row["side"]) == 0]
    frozen64 = config["entry_64_frozen"]
    by_station: dict[int, list[dict[str, Any]]] = {}
    by_ift_layer: dict[int, list[dict[str, Any]]] = {}
    for row in side0:
        by_station.setdefault(int(row["station_id"]), []).append(row)
        if int(row["station_id"]) == IFT_STATION_ID:
            by_ift_layer.setdefault(int(row["layer_id"]), []).append(row)
    station_rows = []
    for station, rows in sorted(by_station.items()):
        offsets = np.asarray([row["offset_from_nominal_mm"] for row in rows], dtype=np.float64)
        nominal = np.asarray([row["implied_nominal_mm"] for row in rows], dtype=np.float64)
        corrected = np.asarray([row["corrected_mm"] for row in rows], dtype=np.float64)
        stats = _mean_and_spreads(offsets)
        quoted = parsed["quoted_station_summaries"][station]
        kabsch = _kabsch(nominal, corrected)
        ry_from_dz = _linear_ry_from_dz_vs_x(nominal, offsets)
        station_rows.append(
            {
                "station_id": station,
                "station_name": STATION_NAMES[station],
                "n_side0_sensors": len(rows),
                "translation_from_printed_sensors_mm": [float(item) for item in stats["mean"]],
                "quoted_translation_mm": list(quoted["delta_mm"]),
                "translation_mm": list(quoted["delta_mm"]),
                "population_spread_from_printed_sensors_mm": [
                    float(item) for item in stats["std_population_ddof0"]
                ],
                "quoted_sigma_mm": list(quoted["sigma_mm"]),
                "population_spread_mm": list(quoted["sigma_mm"]),
                "population_spread_not_measurement_sigma": True,
                "rigid_kabsch": kabsch,
                "ry_from_dz_vs_x": ry_from_dz,
                "may_map_kabsch_ry_to_calypso_station_ry": False,
                "may_map_dz_vs_x_to_calypso_station_ry": False,
            }
        )
    quoted_layers = parsed["quoted_layer_summaries"]
    ift_layer_means = {}
    for layer in IFT_LAYER_IDS:
        offsets = [row["offset_from_nominal_mm"] for row in by_ift_layer[layer]]
        stats = _mean_and_spreads(offsets)
        quoted = quoted_layers[layer]["delta_mm"]
        ift_layer_means[layer] = {
            "n_side0_sensors": len(by_ift_layer[layer]),
            "mean_offset_from_printed_sensors_mm": [float(item) for item in stats["mean"]],
            "quoted_mean_offset_mm": list(quoted),
            "mean_offset_mm": list(quoted),
            "population_spread_from_printed_sensors_mm": [
                float(item) for item in stats["std_population_ddof0"]
            ],
        }
    quoted_xs = {layer: float(quoted_layers[layer]["delta_mm"][0]) for layer in IFT_LAYER_IDS}
    printed_xs = {
        layer: float(ift_layer_means[layer]["mean_offset_from_printed_sensors_mm"][0])
        for layer in IFT_LAYER_IDS
    }
    c_dx = c_dx_from_layer_x(quoted_xs[0], quoted_xs[2])
    delta = float(quoted_xs[0]) - float(quoted_xs[2])
    printed_c_dx = c_dx_from_layer_x(printed_xs[0], printed_xs[2])
    relative = {
        f"L{left}-L{right}": [
            float(
                ift_layer_means[left]["mean_offset_mm"][axis]
                - ift_layer_means[right]["mean_offset_mm"][axis]
            )
            for axis in range(3)
        ]
        for left, right in ((0, 1), (1, 2), (0, 2))
    }
    # IFT ry diagnostic: per-layer dz vs x, then compare to the 3-layer Kabsch ry.
    ift_layer_ry = {}
    for layer in IFT_LAYER_IDS:
        rows = by_ift_layer[layer]
        ift_layer_ry[layer] = _linear_ry_from_dz_vs_x(
            np.asarray([row["implied_nominal_mm"] for row in rows], dtype=np.float64),
            np.asarray([row["offset_from_nominal_mm"] for row in rows], dtype=np.float64),
        )
    layer_ry_values = [ift_layer_ry[layer]["ry_candidate_mrad"] for layer in IFT_LAYER_IDS]
    layer_ry_consistent = bool(
        np.isfinite(layer_ry_values).all()
        and (max(layer_ry_values) - min(layer_ry_values)) < 0.5
    )
    slide_cdx = float(frozen64["C_dx_slide_mm"])
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "side_selection": 0,
        "side_selection_reason": (
            "File summaries and Casper 2022-11-04 slides use the front/pigtail "
            "sensor only because CAD wafer z spacing is not the GeoModel 0.6 mm gap."
        ),
        "stations": station_rows,
        "ift_layers": {str(layer): ift_layer_means[layer] for layer in IFT_LAYER_IDS},
        "ift_relative_displacement_mm": relative,
        "delta_x_L0_minus_L2_mm": delta,
        "C_dx_mm": c_dx,
        "C_dx_from_printed_sensors_mm": printed_c_dx,
        "C_dx_definition": (
            "C_dx=(dx_L0-dx_L2)/2 from quoted side=0 IFT layer-mean x offsets; "
            "entry-64 regression uses the file's high-precision layer block, not "
            "the 3-decimal per-sensor reprint"
        ),
        "reproduces_entry64_C_dx_slide": bool(
            abs(c_dx - slide_cdx) <= float(config["regression"]["cdx_atol_mm"])
        ),
        "entry64_C_dx_slide_mm": slide_cdx,
        "entry64_delta_x_L0_minus_L2_mm": float(frozen64["delta_x_L0_minus_L2_mm"]),
        "station0_quoted_mean_matches_entry64_slide": bool(
            np.allclose(
                np.asarray(parsed["quoted_station_summaries"][0]["delta_mm"], dtype=np.float64),
                np.asarray(frozen64["station0_mean_offset_mm"], dtype=np.float64),
                atol=float(config["regression"]["station_mean_slide_atol_mm"]),
            )
        ),
        "ift_per_layer_ry_from_dz_vs_x_mrad": {
            str(layer): ift_layer_ry[layer]["ry_candidate_mrad"] for layer in IFT_LAYER_IDS
        },
        "ift_per_layer_ry_span_mrad": float(max(layer_ry_values) - min(layer_ry_values)),
        "ift_per_layer_ry_consistent_at_0p5_mrad": layer_ry_consistent,
        "station0_kabsch_ry_mrad": next(
            row["rigid_kabsch"]["ry_mrad"] for row in station_rows if row["station_id"] == 0
        ),
        "station0_ry_from_dz_vs_x_mrad": next(
            row["ry_from_dz_vs_x"]["ry_candidate_mrad"] for row in station_rows if row["station_id"] == 0
        ),
        "layer_x_slope_not_used_as_station_ry": True,
        "stereo_not_used_as_station_ry": True,
        "layerpitch_not_used_as_station_ry": True,
        "planes_rotations_not_used_as_station_ry": True,
        "may_map_any_fitted_ry_to_calypso_station_ry": False,
        "may_enter_geometry_candidate": False,
        "may_be_used_as_alignment_prior": False,
    }


def frame_reconciliation(
    config: Mapping[str, Any],
    parsed: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    calypso = config["calypso_evidence"]
    delta = parsed["delta_faser_mm"]
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "frames": {
            "cad_survey_nov22_file": {
                "label": "2022_survey_adjusted_FASER",
                "x": "intended FASER / Calypso x (horizontal)",
                "y": "intended FASER / Calypso y (vertical)",
                "z": "intended FASER / Calypso z (beam)",
                "origin": (
                    "File applies Delta FASER so that all four stations approximate "
                    "the FASER origin. Casper 2022-11-04: the three translations are "
                    "fixed by matching average nominal and average survey positions of "
                    "front sensors (pigtail side) in Stations 1-3; IFT is not used."
                ),
                "delta_faser_mm": delta,
                "delta_faser_file_text": (
                    "All four stations are shifted by this to approximate Faser "
                    "coordinate system origin"
                ),
                "status": "translation_reconciled_by_file_header_and_2022_slides",
            },
            "calypso_geomodel_global": {
                "x": "horizontal",
                "y": "vertical",
                "z": "beam, downstream positive",
                "rotation_convention": calypso["rotation_convention"],
                "station_constants_order": list(calypso["station_constants_order"]),
                "stations_planes_frame": calypso["stations_planes_frame"],
                "interface_modules_frame": calypso["interface_modules_frame"],
                "evidence": [
                    calypso["detector_factory"],
                    calypso["align_db_tool"],
                    "TrackerAlignDBTool stationAlignment: T*Rz*Ry*Rx, mm/rad",
                ],
            },
            "survey_2021_cad": {
                "status": "not_this_file",
                "note": (
                    "The 2021 Cadoux CAD axes (x horizontal, y beam, z vertical) are "
                    "a different native frame. This dump is already Casper's "
                    "survey-adjusted FASER output, not the native CAD table."
                ),
            },
        },
        "axis_order_in_this_file": {
            "determined": True,
            "order": ["x_horizontal", "y_vertical", "z_beam"],
            "evidence": [
                "File header names the shift as approximating the FASER origin",
                "Reproduced station/layer numbers match docs/04Nov2022_Survey.pdf tables",
                "Workbook 64 already classified that slide frame as survey-adjusted FASER",
            ],
        },
        "cad_native_to_calypso_rotation": {
            "uniquely_determined": False,
            "reason": (
                "The dump is already in the translated FASER frame used by the 2022 "
                "slides. Native CAD axis signs before that translation are not in "
                "this file. They remain unresolved for the 2021 CAD I/F normal, "
                "which is a different object."
            ),
            "status": "unresolved_assumption_not_required_for_this_file_C_dx",
        },
        "parameter_mapping": {
            "C_dx": {
                "scan_parameter": "C_dx",
                "definition": "C_dx=(dx_L0-dx_L2)/2",
                "frame": "2022_survey_adjusted_FASER",
                "validated_as_file_layer_contrast": True,
                "validated": True,
                "value_mm": geometry["C_dx_mm"],
                "sigma": None,
                "note": (
                    "The L0/L2 x contrast is well-defined in this file's survey-"
                    "adjusted FASER frame. It is not a measured constraint until "
                    "an independent covariance and IOV statement exist."
                ),
            },
            "station_translation": {
                "scan_parameter": "station_dx",
                "also": ["station_dy", "station_dz"],
                "frame": "2022_survey_adjusted_FASER",
                "validated_as_file_mean_offset": True,
                "validated_as_calypso_stations_channel": False,
                "reason": (
                    "Means are well-defined in the file frame. Mapping the IFT z "
                    "offset of ~-27.8 mm onto /Tracker/Align/Stations is a geometry "
                    "write and is forbidden here."
                ),
            },
            "station_ry": {
                "scan_parameter": "station_ry",
                "validated": False,
                "forbidden_sources": [
                    "2021 CAD I/F normal tilt",
                    "2022 layer-mean x slope / C_dx",
                    "module stereo STEREOANGLE",
                    "geomDB LAYERPITCH",
                    "/Tracker/Align/Planes non-zero rotations",
                    "Kabsch ry of IFT sensors (absorbs the same L0/L2 x contrast)",
                    "dz-vs-x slope without a unique Calypso origin and IOV",
                ],
                "ift_kabsch_ry_mrad": geometry["station0_kabsch_ry_mrad"],
                "ift_ry_from_dz_vs_x_mrad": geometry["station0_ry_from_dz_vs_x_mrad"],
                "why_not_mapped": (
                    "A number that rotates about the file y axis is still not "
                    "Calypso /Tracker/Align/Stations ry until the origin of that "
                    "rotation, the unique CAD/FASER→Calypso convention, and the "
                    "IOV are proven. Kabsch on all IFT sensors mixes C_dx into ry. "
                    "The dz-vs-x diagnostic is independent of C_dx but has no "
                    "measurement covariance and no unique station-origin convention."
                ),
            },
        },
        "explicit_transform_completed": False,
        "unconfirmed_tilt_mapped_to_ry": False,
        "may_enter_geometry_candidate": False,
    }


def covariance_audit(parsed: Mapping[str, Any], reproduced: Mapping[str, Any]) -> dict[str, Any]:
    quoted = parsed["quoted_station_summaries"]
    searches = {
        "per_point_uncertainty": False,
        "repeat_survey_uncertainty": False,
        "instrument_precision": False,
        "fit_covariance": False,
        "measurement_error_provenance": False,
    }
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "file_sigma_definition": reproduced["sigma_definition_in_file"],
        "file_sigma_is_sample_standard_deviation_of_side0_offsets": False,
        "file_sigma_is_population_standard_deviation_of_side0_offsets": True,
        "reproduced_as_ddof_0": reproduced["sigma_is_population_std_ddof0"],
        "reproduced_as_ddof_1": reproduced["sigma_is_sample_std_ddof_1"],
        "quoted_station_sigma_mm": {
            str(station): list(row["sigma_mm"]) for station, row in quoted.items()
        },
        "label": "population_spread_not_measurement_sigma",
        "may_be_used_as_gaussian_prior_sigma": False,
        "independent_measurement_error_objects_found": searches,
        "adjacent_pdfs_supply_per_point_covariance": False,
        "constructed_measurement_covariance": None,
        "availability": "feasibility_only",
        "reason": (
            "The file Sigma is the sensor-to-sensor spread of offsets from nominal. "
            "No raw measurement error, repeat survey, instrument precision, or fit "
            "covariance is present. Population scatter is diagnostic only."
        ),
        "may_enter_geometry_candidate": False,
    }


def official_constraint_slots(
    config: Mapping[str, Any],
    geometry: Mapping[str, Any],
    frames: Mapping[str, Any],
    covariance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    infrastructure = load_infrastructure_config(
        resolve_under_root(project_root(), str(config["adjacent_sources"]["infrastructure_config"]))
    )
    slots = empty_constraint_catalog(infrastructure)
    mapping_ok = bool(frames["parameter_mapping"]["C_dx"]["validated"])
    cov_ok = covariance["constructed_measurement_covariance"] is not None
    iov_ok = False
    filled = []
    for slot in slots:
        item = dict(slot)
        if item["constraint_id"] in {"ift_C_dx", "ift_l0_minus_l2_dx"}:
            value = (
                geometry["C_dx_mm"]
                if item["constraint_id"] == "ift_C_dx"
                else geometry["delta_x_L0_minus_L2_mm"]
            )
            item.update(
                {
                    "value": [float(value)],
                    "sigma": None,
                    "covariance": None,
                    "availability": "feasibility_only",
                    "source_file": config["source"]["relative"],
                    "provenance": (
                        "cad_survey_nov22 side=0 layer-mean x; central value only. "
                        "Sigma in the file is population spread, not measurement error."
                    ),
                    "frame": "2022_survey_adjusted_FASER",
                    "year": 2022,
                    "iov_id": "2022_survey_casper_nov04_not_a_2024_conditions_iov",
                    "may_enter_geometry_candidate": False,
                    "central_value_is_survey": False,
                    "ingest_gates": {
                        "parameter_mapping_validated": mapping_ok,
                        "measurement_covariance_has_independent_provenance": cov_ok,
                        "measurement_year_conditions_iov_identified": iov_ok,
                    },
                    "note": (
                        "Not promoted to availability=measured: covariance and "
                        "cross-year IOV gates fail. 2022 survey cannot be used as a "
                        "2024/2025 station rigid-body correction."
                    ),
                }
            )
        validate_constraint(item)
        filled.append(item)
    return filled


def decide_next_stage(
    *,
    geometry: Mapping[str, Any],
    frames: Mapping[str, Any],
    covariance: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ry_slot = next(item for item in slots if item["constraint_id"] == "ift_station0_ry")
    cdx_slot = next(item for item in slots if item["constraint_id"] == "ift_C_dx")
    mapping_ok = bool(frames["parameter_mapping"]["station_ry"]["validated"])
    cov_ok = covariance["constructed_measurement_covariance"] is not None
    iov_ok = False
    possess = (
        ry_slot["availability"] == "measured"
        or (cdx_slot["availability"] == "measured" and cdx_slot.get("sigma") is not None)
    )
    blockers = []
    if not frames["parameter_mapping"]["station_ry"]["validated"]:
        blockers.append(
            {
                "id": "cad_to_calypso_station_ry_mapping",
                "what": "validated Calypso station ry mapping",
                "detail": frames["parameter_mapping"]["station_ry"]["why_not_mapped"],
            }
        )
    if not cov_ok:
        blockers.append(
            {
                "id": "measurement_covariance",
                "what": "per-point or fit measurement covariance with independent provenance",
                "detail": covariance["reason"],
            }
        )
    blockers.append(
        {
            "id": "iov_provenance",
            "what": "year/conditions IOV that this 2022 dump may constrain",
            "detail": (
                "Talk date is 2022-11-04. Station rigid motion is IOV-specific. "
                "C_dx remains a common-static candidate until opening/thermal/"
                "metrology evidence. Do not average with 2023/2024/2025 residuals."
            ),
        }
    )
    if not frames["explicit_transform_completed"]:
        blockers.append(
            {
                "id": "unique_cad_native_rotation",
                "what": "unique native-CAD → Calypso rotation for the 2021 I/F table",
                "detail": frames["cad_native_to_calypso_rotation"]["reason"],
            }
        )
    return {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "cad_survey_nov22_provides_independent_frame_correct_uncertain_measurement": False,
        "reproduces_entry64_C_dx_slide": geometry["reproduces_entry64_C_dx_slide"],
        "C_dx_mm": geometry["C_dx_mm"],
        "have_validated_ry_mapping": mapping_ok,
        "have_measurement_covariance": cov_ok,
        "have_matching_year_conditions_iov": iov_ok,
        "official_ry_slot_availability": ry_slot["availability"],
        "official_cdx_slot_availability": cdx_slot["availability"],
        "official_cdx_sigma": cdx_slot.get("sigma"),
        "survey_candidates_possess_degeneracy_breaking_information_requirement": possess,
        "enough_to_write_alignment_prior": False,
        "enough_to_write_geometry": False,
        "survey_derived": False,
        "did_run_newton": False,
        "did_fill_measured_constraint_slot": False,
        "did_rebuild_2024_r0022_jacobian": False,
        "blockers": blockers,
        "minimum_request_to_hardware_alignment_survey": [
            {
                "priority": 1,
                "what": "Per-point or rigid-body-fit covariance of the Nov 2022 wafer table",
                "fields": [
                    "sensor identifier (station, layer, phi_module, eta_module, side)",
                    "measured global (x,y,z) or offset from a named nominal",
                    "3x3 (or at least diagonal) measurement covariance in that frame",
                    "instrument, date, operator, repeat count",
                ],
                "do_not_send": "station/module population standard deviations as sigma",
            },
            {
                "priority": 1,
                "what": "Unique FASER/Calypso station-ry definition for this table",
                "fields": [
                    "rotation origin (station GeoModel origin vs sensor centroid vs support beam)",
                    "axis: Calypso global y after T*Rz*Ry*Rx",
                    "quantified IFT in-plane yaw if that is the intended ry object",
                ],
                "do_not_send": "LAYERPITCH slope, STEREOANGLE, or /Tracker/Align/Planes ry",
            },
            {
                "priority": 1,
                "what": "IOV statement",
                "fields": [
                    "which year/conditions tag the 2022 survey may constrain",
                    "mechanical-stability evidence if it is to be used in 2024/2025",
                    "whether C_dx is claimed common-static after opening/thermal checks",
                ],
            },
        ],
        "next_allowed_step": (
            "Keep residual_dq_monitoring_only. Do not Newton-solve. Ask the "
            "alignment/hardware/survey team for covariance, a unique station-ry "
            "convention, and an IOV statement. Frozen V2 and the association "
            "policy stay unchanged."
        ),
        "reason": (
            "cad_survey_nov22 is the per-sensor source of the 2022 slide C_dx "
            f"central value ({geometry['C_dx_mm']:.7f} mm) in the survey-adjusted "
            "FASER frame, and the identifier tuple is FaserSCT "
            "(station, layer, phi_module, eta_module, side). It does not supply "
            "a validated Calypso station-ry mapping, a defensible measurement "
            "covariance, or a 2024/2025 IOV. That is a reproducible negative "
            "for alignment ingest, not a reason to invent sigma or write geometry."
        ),
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    source = resolve_under_root(root, str(config["source"]["relative"]))
    if not source.is_file():
        raise FileNotFoundError(f"cad_survey_nov22 is missing: {source}")
    digest = sha256_file(source)
    if digest != str(config["source"]["expected_sha256"]):
        raise ValueError(
            "cad_survey_nov22 sha256 does not match the frozen expected digest; "
            "the original file must remain unmodified"
        )
    parsed = parse_cad_survey_nov22_file(source)
    reproduced = reproduce_quoted_summaries(parsed)
    if not reproduced["quoted_summaries_parsed_exactly"]:
        raise ValueError("parser failed to parse the file's own station/layer summaries")
    if not reproduced["recomputed_means_match_quoted_within_print_rounding"]:
        raise ValueError(
            "printed 3-decimal sensor offsets are inconsistent with quoted "
            "station/layer means beyond print rounding"
        )
    if not reproduced["sigma_matches_population_std_of_printed_sensors_better_than_sample_std"]:
        raise ValueError("file Sigma is not closer to population std than to sample std")
    provenance = source_provenance(config, parsed)
    geometry = geometry_analysis(parsed, config)
    frames = frame_reconciliation(config, parsed, geometry)
    covariance = covariance_audit(parsed, reproduced)
    slots = official_constraint_slots(config, geometry, frames, covariance)
    decision = decide_next_stage(
        geometry=geometry, frames=frames, covariance=covariance, slots=slots
    )
    sensors_public = []
    for row in parsed["sensors"]:
        sensors_public.append(
            {
                "station_id": row["station_id"],
                "layer_id": row["layer_id"],
                "phi_module": row["phi_module"],
                "eta_module": row["eta_module"],
                "side": row["side"],
                "phi_label": PHI_MODULE_LABELS[int(row["phi_module"])],
                "eta_label": ETA_MODULE_LABELS[int(row["eta_module"])],
                "side_label": SIDE_LABELS[int(row["side"])],
                "station_name": STATION_NAMES[int(row["station_id"])],
                "corrected_mm": _xyz(row["corrected_mm"]),
                "offset_from_nominal_mm": _xyz(row["offset_from_nominal_mm"]),
                "implied_nominal_mm": _xyz(row["implied_nominal_mm"]),
            }
        )
    parsed_artifact = {
        **common_audit_state(),
        "schema_version": SCHEMA_VERSION,
        "git_sha": provenance["git_sha"],
        "source_sha256": parsed["sha256"],
        "delta_faser_mm": _xyz(parsed["delta_faser_mm"]),
        "n_sensors": parsed["n_sensors"],
        "sensors": sensors_public,
        "quoted_station_summaries": {
            str(station): {
                "delta_mm": _xyz(row["delta_mm"]),
                "sigma_mm": _xyz(row["sigma_mm"]),
                "sigma_label": row["sigma_label"],
                "side_selection": row["side_selection"],
            }
            for station, row in parsed["quoted_station_summaries"].items()
        },
        "quoted_layer_summaries": {
            str(layer): {
                "delta_mm": _xyz(row["delta_mm"]),
                "station_id": row["station_id"],
                "station_layer_id": row["station_layer_id"],
                "side_selection": row["side_selection"],
            }
            for layer, row in parsed["quoted_layer_summaries"].items()
        },
        "reproduced_summaries": reproduced,
        "index_tuple": provenance["index_tuple"],
        "may_enter_geometry_candidate": False,
    }
    slot_summary = [
        {
            "constraint_id": item["constraint_id"],
            "availability": item["availability"],
            "value": item.get("value"),
            "sigma": item.get("sigma"),
            "frame": item.get("frame"),
            "year": item.get("year"),
            "iov_id": item.get("iov_id"),
            "may_enter_geometry_candidate": item.get("may_enter_geometry_candidate"),
        }
        for item in slots
    ]
    return {
        "cad_survey_nov22_provenance": provenance,
        "cad_survey_nov22_parsed": parsed_artifact,
        "cad_survey_nov22_geometry": geometry,
        "cad_survey_nov22_frame_reconciliation": frames,
        "cad_survey_nov22_covariance_audit": covariance,
        "official_constraint_slots": {
            **common_audit_state(),
            "slots": slot_summary,
        },
        "next_stage_decision": decision,
    }


def refuse_unconfirmed_ry_mapping(candidate_id: str) -> None:
    raise UnconfirmedRyMappingError(
        f"cannot map {candidate_id} to station_ry: cad_survey_nov22 has no "
        "validated Calypso station-ry convention"
    )


def refuse_index_tuple_guess(field: str) -> None:
    raise IndexTupleGuessError(
        f"cannot guess index-tuple meaning for {field}; meanings come from "
        "FaserSCT_ID / IdDictInterface / SCT_Identifier / SCT_DetectorFactory"
    )


def refuse_population_sigma_as_prior() -> None:
    raise ValueError(
        "cad_survey_nov22 Sigma is population spread, not a Gaussian prior sigma"
    )
