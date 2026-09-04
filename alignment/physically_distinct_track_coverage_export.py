"""Physically-distinct track-coverage residual-blind export & reinventory V1.

Workbook 75 (2026-09-04).  This module executes the single step authorized
by workbook 74 (``residual_blind_export_authorized_fd_not_opened``): the
residual-blind HTCondor tracklet export of the two metadata-distinct
candidates ``mc24_100120_muon_floor`` and ``mc24_100130_kshort_end_fasernu``,
followed by a repeated residual-blind coverage inventory against the frozen
workbook-74 gates and the canonical ``(tx, ty)`` envelope.

This stage does not construct ``A = W^{1/2} J S``, does not SVD, does not
inspect rank, does not inject an alignment payload, does not build a
physical FD point, and does not select events from residuals / Jacobian /
singular values / cosine / alignment response.  Workbooks 59-74 remain
frozen.  No coverage gate is modified.
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE
from alignment.noncollision_crossyear_topology import parse_job_log, probe_xaod
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.physically_distinct_track_coverage import (
    REQUIRED_STATIONS,
    ResidualBlindStageError,
    common_inventory_state,
    load_inventory_config,
    refuse_sealed_source,
    _distribution,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND
from datasets.root_loader import EventTracklets
from datasets.schema import CANONICAL_TREE_NAME, MC_LABEL_FIELDS, required_fields, validate_tracklet_tree

SCHEMA_VERSION = (
    "faser-physically-distinct-track-coverage-residual-blind-export-reinventory-v1"
)
DEFAULT_CONFIG_RELATIVE = Path(
    "configs"
) / "physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml"
INHERITED_ENTRY_74 = "residual_blind_export_authorized_fd_not_opened"
WORKBOOK_74_CONFIG_SHA256 = (
    "2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf"
)

DECISION_ADMIT_FD_PREREGISTRATION = (
    "candidate_admitted_for_separate_5dof_fd_preregistration"
)
DECISION_INSUFFICIENT = (
    "current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof"
)

VERDICT_ADMITTED = "admitted_for_separate_5dof_fd_preregistration"
VERDICT_INSUFFICIENT = "insufficient_statistics"
VERDICT_NOT_DISTINCT = "not_observed_phase_space_distinct"
VERDICT_SEGMENTFIT_EMPTY = "segmentfit_empty"
VERDICT_EXPORTER_INCOMPATIBLE = "exporter_contract_incompatible"
VERDICT_PROVENANCE_FAILURE = "provenance_failure"
VERDICT_MISSING_OUTPUT = "missing_output"
VERDICT_EXPORT_FAILED = "export_job_failed"

EXPORT_JOB_SCHEMA = "faser-residual-blind-tracklet-export-job-v1"
MANIFEST_SCHEMA = "faser-residual-blind-export-input-manifest-v1"


class ExportContractError(RuntimeError):
    """The residual-blind export contract was violated."""


class ProvenanceValidationError(RuntimeError):
    """Deterministic provenance validation failed for an exported source."""


# ---------------------------------------------------------------------------
# Config loading and workbook-74 inheritance verification
# ---------------------------------------------------------------------------

def load_export_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the stage-75 config, verifying WB74 inheritance."""
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"export/reinventory config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected export/reinventory schema: {source}")
    if payload.get("stage") != "residual_blind_htcondor_export_and_reinventory":
        raise ValueError("stage must be residual_blind_htcondor_export_and_reinventory")

    # Reuse the workbook-74 frozen-flag validation on this config: every flag
    # that workbook 74 froze must keep the same value here.
    wb74_path = resolve_under_root(
        project_root(), str(payload.get("inherited_workbook_74_config"))
    )
    inherited_sha = sha256_file(wb74_path)
    if inherited_sha != str(payload.get("inherited_workbook_74_config_sha256")):
        raise ValueError(
            "inherited workbook-74 config SHA256 mismatch: "
            f"{inherited_sha} != {payload.get('inherited_workbook_74_config_sha256')}"
        )
    if inherited_sha != WORKBOOK_74_CONFIG_SHA256:
        raise ValueError("inherited workbook-74 config SHA256 is not the frozen value")
    wb74 = load_inventory_config(wb74_path)

    for key, expected in (
        ("min_events", 200),
        ("min_ift_events", 200),
        ("min_complete_four_station_events", 80),
        ("min_independent_sources_or_runs", 2),
        ("min_xaod_files_for_export", 2),
        ("min_xaod_events_per_file_for_export", 1000),
    ):
        if int(payload["coverage_gates"][key]) != expected:
            raise ValueError(f"coverage gate {key} must stay at the frozen value {expected}")
    for key in (
        "min_events",
        "min_ift_events",
        "min_complete_four_station_events",
        "min_independent_sources_or_runs",
        "require_ift_station_0",
        "require_stations_0_1_2_3_present",
        "min_outside_canonical_envelope_fraction",
        "max_histogram_intersection_for_distinct",
        "min_wide_local_slope_fraction",
        "do_not_lower_gates",
        "collision_like_metadata_cannot_admit",
        "same_production_as_canonical_cannot_admit",
        "sealed_test_cannot_admit",
        "coverage_unmeasured_cannot_admit",
        "pooled_or_mixed_rank_is_not_admission",
        "min_xaod_files_for_export",
        "min_xaod_events_per_file_for_export",
        "require_segmentfit_for_export",
        "require_ift_geometry_for_export",
    ):
        if payload["coverage_gates"].get(key) != wb74["coverage_gates"].get(key):
            raise ValueError(f"coverage gate {key} differs from the frozen workbook-74 value")
    for key in (
        "coordinates",
        "preferred_station",
        "envelope_quantiles",
        "histogram_range_tx",
        "histogram_range_ty",
        "histogram_bins",
        "azimuth_bins",
        "min_truth_match_fraction",
    ):
        if payload["phase_space"].get(key) != wb74["phase_space"].get(key):
            raise ValueError(f"phase_space {key} differs from the frozen workbook-74 value")
    contract = payload.get("rigid_station_five_dof_contract") or {}
    if tuple(contract.get("parameter_names") or ()) != FREE_PARAMETERS:
        raise ValueError("rigid-station 5DoF names must stay frozen")
    if float(contract.get("rank_tolerance")) != float(FROZEN_RANK_TOLERANCE):
        raise ValueError("rank_tolerance=0.01 must stay frozen")
    scales = contract.get("scale_matrix_S") or {}
    if float(scales.get("ift_dx_mm")) != 5.0 or float(scales.get("ift_ry_mrad")) != 60.0:
        raise ValueError("frozen S = (5, 5, 60, 60, 60) must not be retuned")
    if "ift_rz_mrad" not in scales:
        raise ValueError("rz remains in the frozen 5DoF model; it is not deleted")
    if contract.get("do_not_reconstruct_this_stage") is not True:
        raise ValueError("stage 75 must not reconstruct A = W^{1/2} J S")

    must_be_true = (
        "frozen",
        "do_not_construct_fd_identifiability_this_stage",
        "do_not_compute_svd_or_rank_this_stage",
        "do_not_select_events_from_residual_or_cosine",
        "do_not_open_sealed_test",
        "do_not_lower_statistical_gates",
        "do_not_restack_2024_r0022_collision_like",
        "do_not_restack_canonical_5mrad_unused_files",
        "do_not_inject_alignment_payload_in_export",
        "do_not_construct_physical_fd_point_this_stage",
        "do_not_compute_residual_based_selection_this_stage",
        "do_not_pool_the_two_export_candidates",
        "do_not_split_one_xaod_file_into_fake_independent_sources",
        "do_not_modify_truth_matching_or_route_definition_post_hoc",
        "do_not_reinterpret_local_hypot_tx_ty_as_spectrometer_wide_angle",
        "export_is_not_fd_authorization",
        "do_not_open_gauge_branch_until_inventory_closes",
        "do_not_rescue_seven_d_observable",
        "do_not_rescue_cluster_local_observable",
        "do_not_delete_rz_or_further_dof",
        "filename_5mrad_is_not_phase_space_proof",
    )
    for key in must_be_true:
        if payload.get(key) is not True:
            raise ValueError(f"export/reinventory config must set {key}=true")
    must_be_false = (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "three_arm_authorized",
        "frozen_v2_alignment_loop_authorized",
        "real_data_correction_authorized",
        "survey_is_alignment_input",
    )
    for key in must_be_false:
        if payload.get(key) is not False:
            raise ValueError(f"export/reinventory config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("real data remains residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("frozen V2 checkpoint SHA256 must not be replaced")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("residual kind must stay the true cluster-local residual")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("inherited_entry_74_decision") != INHERITED_ENTRY_74:
        raise ValueError("workbook 74 decision must stay frozen")

    candidate_ids = [str(row["id"]) for row in payload.get("export_candidates") or []]
    if candidate_ids != ["mc24_100120_muon_floor", "mc24_100130_kshort_end_fasernu"]:
        raise ValueError(
            "only the two workbook-74 export-authorized candidates may be exported: "
            + ", ".join(candidate_ids)
        )
    sealed = {
        str(item)
        for item in (payload.get("sealed_test") or {}).get("forbidden_source_ids") or []
    }
    for candidate in payload.get("export_candidates") or []:
        for row in candidate.get("inputs") or []:
            refuse_sealed_source(str(row["source_id"]), sorted(sealed))
    config = dict(payload)
    config["config_path"] = str(source)
    return config


# ---------------------------------------------------------------------------
# Immutable input manifest
# ---------------------------------------------------------------------------

def _xrdfs_adler32(path: Path) -> str | None:
    """Return the EOS-stored adler32 checksum without reading the file."""
    try:
        result = subprocess.run(
            ["xrdfs", "root://eospublic.cern.ch", "query", "checksum", str(path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    tokens = result.stdout.strip().split()
    if len(tokens) == 2 and tokens[0] == "adler32":
        return tokens[1]
    return None


def build_input_manifest(
    config: Mapping[str, Any],
    *,
    probe_events: bool = True,
) -> dict[str, Any]:
    """Build the immutable input manifest from EOS/xAOD metadata only.

    For every declared input file this records the full EOS path, the
    file/production/source identifiers, the xAOD event count, the generator
    and reconstruction provenance (from the pre-declared candidate metadata
    and the job logs), the geometry/reco tags, the file size, and the
    EOS-stored adler32 checksum.  No residual, Jacobian, SVD, cosine, rank,
    or alignment response is consulted.
    """
    created = datetime.now(timezone.utc).isoformat()
    candidates = []
    for candidate in config.get("export_candidates") or []:
        cid = str(candidate["id"])
        gen_log = parse_job_log(candidate["representative_log"])
        rec_log = parse_job_log(candidate["representative_rec_log"])
        files = []
        for row in candidate.get("inputs") or []:
            source_id = str(row["source_id"])
            path = Path(str(row["path"]))
            entry: dict[str, Any] = {
                "source_id": source_id,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": None,
                "n_events": None,
                "adler32": None,
                "collection_keys_of_interest": {},
                "production_dsid": candidate.get("production_dsid"),
                "production_name": candidate.get("production_name"),
                "sim_tag": candidate.get("sim_tag"),
                "rec_tag": candidate.get("rec_tag"),
            }
            if path.is_file():
                stat = path.stat()
                entry["size_bytes"] = int(stat.st_size)
                entry["mtime_utc"] = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()
                entry["adler32"] = _xrdfs_adler32(path)
                if probe_events:
                    probe = probe_xaod(str(path))
                    entry["n_events"] = probe.get("n_events")
                    entry["collection_keys_of_interest"] = probe.get(
                        "collection_keys_of_interest"
                    )
                    entry["file_metadata"] = probe.get("file_metadata")
            files.append(entry)
        candidates.append(
            {
                "id": cid,
                "metadata_class": candidate.get("metadata_class"),
                "physically_distinct_hypothesis": candidate.get(
                    "physically_distinct_hypothesis"
                ),
                "production_dsid": candidate.get("production_dsid"),
                "production_name": candidate.get("production_name"),
                "sim_tag": candidate.get("sim_tag"),
                "rec_tag": candidate.get("rec_tag"),
                "geometry_tag": candidate.get("geometry_tag"),
                "geom_flag": candidate.get("geom_flag"),
                "conditions_tag": candidate.get("conditions_tag"),
                "generator_provenance": candidate.get("generator_provenance"),
                "angular_population": candidate.get("angular_population"),
                "generator_log": {
                    "path": gen_log.get("path"),
                    "exists": gen_log.get("exists"),
                    "geom_flag": gen_log.get("geom_flag"),
                },
                "reconstruction_log": {
                    "path": rec_log.get("path"),
                    "exists": rec_log.get("exists"),
                    "geometry_tag": rec_log.get("geometry_tag"),
                    "conditions_tag": rec_log.get("conditions_tag"),
                    "geom_flag": rec_log.get("geom_flag"),
                },
                "inputs": files,
                "n_input_files": len(files),
                "total_events": sum(int(row["n_events"] or 0) for row in files),
                "total_size_bytes": sum(int(row["size_bytes"] or 0) for row in files),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": created,
        "git_head": git_head_sha(project_root()),
        "config_path": config.get("config_path"),
        "config_sha256": sha256_file(Path(config["config_path"])) if config.get("config_path") else None,
        "inherited_workbook_74_config_sha256": config.get(
            "inherited_workbook_74_config_sha256"
        ),
        "residual_blind": True,
        "built_from": "eos_xaod_metadata_only",
        "candidates": candidates,
    }


def manifest_sha256(manifest_path: str | Path) -> str:
    return sha256_file(Path(manifest_path))


def verify_export_gate(
    manifest: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the frozen workbook-74 export gate for each candidate.

    The gate requires at least ``min_xaod_files_for_export`` input files, at
    least ``min_xaod_events_per_file_for_export`` events per file, SegmentFit
    present in every file, and IFT-capable geometry (TI12MC04/FASERNU-04 with
    the export-time ``--useIFT``).  The gate is not lowered.
    """
    min_files = int(gates["min_xaod_files_for_export"])
    min_events = int(gates["min_xaod_events_per_file_for_export"])
    rows = []
    for candidate in manifest.get("candidates") or []:
        cid = str(candidate["id"])
        files = [row for row in candidate.get("inputs") or [] if row.get("exists")]
        n_files = len(files)
        per_file_events = [int(row.get("n_events") or 0) for row in files]
        segmentfit_all = all(
            bool((row.get("collection_keys_of_interest") or {}).get("SegmentFit"))
            for row in files
        )
        geometry_ok = (
            str(candidate.get("geom_flag")) == "TI12MC04"
            and str(candidate.get("geometry_tag")) == "FASERNU-04"
        )
        reasons = []
        if n_files < min_files:
            reasons.append(f"n_xaod_files {n_files} < {min_files}")
        low = [row["source_id"] for row in files if int(row.get("n_events") or 0) < min_events]
        if low:
            reasons.append(f"files below {min_events} events: {low}")
        if files and not segmentfit_all:
            reasons.append("SegmentFit missing in at least one input file")
        if not geometry_ok:
            reasons.append("IFT-capable geometry (TI12MC04/FASERNU-04) not confirmed")
        rows.append(
            {
                "id": cid,
                "n_input_files": n_files,
                "per_file_events": per_file_events,
                "segmentfit_present_all_files": bool(segmentfit_all),
                "ift_geometry_confirmed": bool(geometry_ok),
                "export_gate_satisfied": not reasons,
                "reason": "; ".join(reasons) if reasons else "export gate satisfied",
                "gates_not_lowered": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "residual_blind": True,
        "gate": {
            "min_xaod_files_for_export": min_files,
            "min_xaod_events_per_file_for_export": min_events,
            "require_segmentfit_for_export": bool(gates.get("require_segmentfit_for_export")),
            "require_ift_geometry_for_export": bool(gates.get("require_ift_geometry_for_export")),
        },
        "candidates": rows,
        "all_candidates_satisfied": all(row["export_gate_satisfied"] for row in rows),
    }


# ---------------------------------------------------------------------------
# Exporter-contract audit
# ---------------------------------------------------------------------------

def exporter_contract_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record the exporter-contract audit for the two export candidates.

    The audit is code-level and pre-registered: it documents whether the
    tracklet exporter, the truth matching, and the SegmentFit population
    definition assume a single muon, PDG=13, or a single truth parent.  It
    does not modify truth matching, particle selection, or route definition.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "residual_blind": True,
        "audited_before_large_scale_submission": True,
        "chain": {
            "exporter": {
                "component": "NtupleDumperAlg::appendDetailedTracklet",
                "path": "calypso/PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx",
                "input_collection": "GhostBustedTrackSegmentCollection (persisted SegmentFit after GhostBusters)",
                "single_muon_assumption": False,
                "pdg_13_requirement": False,
                "single_truth_parent_requirement": False,
                "population_definition": (
                    "any ghost-busted single-station segment with valid global "
                    "state, covariance, and at least one SCT cluster-on-track"
                ),
            },
            "truth_matching": {
                "component": "TrackTruthMatchingTool::getTruthParticle",
                "path": "calypso/Tracking/Acts/FaserActsKalmanFilter/src/TrackTruthMatchingTool.cxx",
                "method": "majority barcode from SCT SDO deposits, label = that particle's pdgId",
                "single_muon_assumption": False,
                "pdg_13_requirement": False,
                "single_truth_parent_requirement": False,
                "unmatched_label": "truth_particle_id=-1, truth_pdg=0",
            },
            "segmentfit_population": {
                "component": "persisted SegmentFit (standard reconstruction)",
                "single_muon_assumption": False,
                "pdg_13_requirement": False,
                "note": (
                    "station-local straight-line segment fits from SCT clusters; "
                    "no particle hypothesis at pattern-recognition level"
                ),
            },
            "event_selection": {
                "component": "NtupleDumperAlg::execute",
                "mc_requires_grl": False,
                "mc_requires_n_track_filter": False,
                "note": "every MC event with a valid McEventCollection is written",
            },
            "propagation_mass_hypothesis": {
                "component": "FaserActsExtrapolationTool",
                "note": (
                    "Acts propagation uses a muon mass hypothesis for material "
                    "effects; this is a propagation setting, not an export "
                    "population cut, and is not used by the coverage inventory"
                ),
            },
        },
        "workbook_74_reporter_angular_mask": {
            "mask": "abs(truth_pdg)==13 and truth_match_fraction>=0.99 and truth_particle_id>=0",
            "muon_specific": True,
            "consequence_for_neutral_gun": (
                "for pid=310 the strict workbook-74 angular population is empty "
                "by construction (smoke: 0 muon-tagged tracklets in 500 events); "
                "the pre-registered 100130 angular population is the "
                "truth-matched charged pion daughters |pdg|=211, declared in "
                "the stage-75 config before the full export"
            ),
        },
        "candidates": {
            "mc24_100120_muon_floor": {
                "gun_pid": [-13, 13],
                "exporter_contract_compatible": True,
                "angular_population_abs_pdg": [13],
                "note": "muon gun; identical to the workbook-74 population definition",
            },
            "mc24_100130_kshort_end_fasernu": {
                "gun_pid": 310,
                "exporter_contract_compatible": True,
                "angular_population_abs_pdg": [211],
                "strict_muon_mask_crosscheck": "reported as a labelled cross-check only",
                "note": (
                    "neutral kaon decays to charged pions which produce tracker "
                    "tracks; the generic reconstructed-tracklet chain natively "
                    "supports this topology"
                ),
            },
        },
        "truth_matching_modified": False,
        "particle_selection_modified_post_hoc": False,
        "route_definition_modified": False,
    }


# ---------------------------------------------------------------------------
# Physical-order event loading (merged-rec event-number collisions)
# ---------------------------------------------------------------------------

def load_events_physical_order(
    root_path: str | Path,
    *,
    tree_name: str = CANONICAL_TREE_NAME,
    max_events: int | None = None,
) -> tuple[list[EventTracklets], dict[str, Any]]:
    """Load canonical tracklets as physical events in exporter file order.

    Merged MC24 rec productions reuse generator-job event numbers, so the
    same ``(run_id, event_id)`` pair appears once per merged generator job.
    The sorted grouping of ``datasets.root_loader.load_events`` would merge
    those distinct physical events into one logical event (and raise on the
    resulting duplicate tracklet ids).  Here each maximal consecutive block
    of equal ``(run_id, event_id)`` rows in file order is one physical
    event, exactly matching the exporter's per-entry order.  No fuzzy join
    and no nearest-neighbour matching is involved.
    """
    import uproot

    from datasets.schema import covariance_from_columns

    path = Path(root_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        report = validate_tracklet_tree(tree)
        report.require_valid(require_mc_labels=False)
        fields = list(required_fields(require_mc_labels=False))
        if report.has_mc_labels:
            fields.extend(MC_LABEL_FIELDS)
        raw = tree.arrays(fields, library="np")
    columns = {name: np.asarray(raw[name]) for name in fields}
    n_rows = int(columns["run_id"].size)
    run_ids = columns["run_id"]
    event_ids = columns["event_id"]
    boundaries = np.flatnonzero(
        (run_ids[1:] != run_ids[:-1]) | (event_ids[1:] != event_ids[:-1])
    ) + 1
    groups = np.split(np.arange(n_rows), boundaries) if n_rows else []
    covariance = covariance_from_columns(columns)
    has_mc = report.has_mc_labels

    events: list[EventTracklets] = []
    n_duplicate_tracklet_id = 0
    for group in groups:
        if group.size == 0:
            continue
        first = int(group[0])
        tracklet_ids = np.asarray(columns["tracklet_id"][group], dtype=np.int32)
        if np.unique(tracklet_ids).size != tracklet_ids.size:
            n_duplicate_tracklet_id += 1
            raise ProvenanceValidationError(
                f"duplicate tracklet_id inside one physical event block in {path}: "
                f"run={int(run_ids[first])} event={int(event_ids[first])}"
            )
        state = np.column_stack(
            [
                columns["x_mm"][group],
                columns["y_mm"][group],
                columns["tx"][group],
                columns["ty"][group],
            ]
        ).astype(np.float64, copy=False)
        if not np.isfinite(state).all():
            raise ProvenanceValidationError(
                f"non-finite tracklet state in {path}: "
                f"run={int(run_ids[first])} event={int(event_ids[first])}"
            )
        events.append(
            EventTracklets(
                run_id=int(run_ids[first]),
                event_id=int(event_ids[first]),
                station_id=np.asarray(columns["station_id"][group], dtype=np.int16),
                tracklet_id=tracklet_ids,
                z_mm=np.asarray(columns["z_mm"][group], dtype=np.float64),
                state=state,
                covariance=np.asarray(covariance[group], dtype=np.float64),
                chi2=np.asarray(columns["chi2"][group], dtype=np.float64),
                ndof=np.asarray(columns["ndof"][group], dtype=np.float64),
                n_hit=np.asarray(columns["n_hit"][group], dtype=np.int16),
                hit_pattern=np.asarray(columns["hit_pattern"][group], dtype=np.uint64),
                truth_particle_id=(
                    np.asarray(columns["truth_particle_id"][group], dtype=np.int64)
                    if has_mc
                    else None
                ),
                truth_pdg=(
                    np.asarray(columns["truth_pdg"][group], dtype=np.int32)
                    if has_mc
                    else None
                ),
                truth_match_fraction=(
                    np.asarray(columns["truth_match_fraction"][group], dtype=np.float64)
                    if has_mc
                    else None
                ),
            )
        )
        if max_events is not None and len(events) >= max_events:
            break
    provenance = {
        "path": str(path),
        "n_rows": n_rows,
        "n_physical_events": len(events),
        "n_unique_event_ids": int(np.unique(event_ids).size) if n_rows else 0,
        "event_id_reused_across_generator_jobs": bool(
            n_rows and np.unique(event_ids).size < len(events)
        ),
        "grouping": "consecutive_equal_run_event_blocks_in_file_order",
        "sorted_merge_used": False,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
        "duplicate_tracklet_id_blocks": int(n_duplicate_tracklet_id),
    }
    return events, provenance


# ---------------------------------------------------------------------------
# Residual-blind coverage summaries with the pre-registered population
# ---------------------------------------------------------------------------

def _truth_population_mask(
    event: EventTracklets,
    min_truth_match_fraction: float,
    abs_pdg: Sequence[int],
) -> np.ndarray:
    """Workbook-74 truth-match mask with the pre-registered particle id set.

    For ``abs_pdg=(13,)`` this is exactly the workbook-74 truth-matched muon
    mask; the regression tests prove the two implementations agree.
    """
    n = int(event.size)
    if event.truth_pdg is None or event.truth_match_fraction is None or event.truth_particle_id is None:
        return np.ones(n, dtype=bool)
    pdg = np.abs(np.asarray(event.truth_pdg))
    pdg_ok = np.zeros(n, dtype=bool)
    for value in abs_pdg:
        pdg_ok |= pdg == int(value)
    return (
        pdg_ok
        & (np.asarray(event.truth_match_fraction) >= float(min_truth_match_fraction))
        & (np.asarray(event.truth_particle_id) >= 0)
    )


def summarize_events_population(
    events: Sequence[EventTracklets],
    *,
    source_id: str,
    physics: Mapping[str, Any],
    phase_space: Mapping[str, Any],
    angular_abs_pdg: Sequence[int] = (13,),
) -> dict[str, Any]:
    """Workbook-74 residual-blind coverage summary for one tracklet file.

    Identical to ``alignment.physically_distinct_track_coverage.summarize_events``
    except that the truth-matched angular population uses the pre-registered
    ``angular_abs_pdg`` set instead of the hard-coded muon id.  Station
    coverage, event counts, and all gate-relevant numbers are computed from
    all tracklets exactly as in workbook 74.  No residual, Jacobian, SVD,
    cosine, rank, or alignment response is consulted.
    """
    preferred = int(phase_space.get("preferred_station") or 0)
    min_frac = float(phase_space.get("min_truth_match_fraction") or 0.99)
    wide_cut = float(physics["wide_angle_min_slope"])
    ip_cut = float(physics["ip_like_max_slope"])
    n_events = int(len(events))
    station_event_counts = {str(station): 0 for station in REQUIRED_STATIONS}
    complete = 0
    ift_events = 0
    tx_values: list[float] = []
    ty_values: list[float] = []
    slope_values: list[float] = []
    azimuth_values: list[float] = []
    n_tracklets = 0
    n_selected_tracklets = 0
    for event in events:
        n_tracklets += int(event.size)
        mask = _truth_population_mask(event, min_frac, angular_abs_pdg)
        stations = set(int(value) for value in np.unique(event.station_id))
        for station in REQUIRED_STATIONS:
            if station in stations:
                station_event_counts[str(station)] += 1
        if 0 in stations:
            ift_events += 1
        if set(REQUIRED_STATIONS).issubset(stations):
            complete += 1
        selected = mask
        if preferred in stations:
            selected = selected & (event.station_id == preferred)
        elif np.any(selected):
            first = int(event.station_id[np.flatnonzero(selected)[0]])
            selected = selected & (event.station_id == first)
        chosen = np.flatnonzero(selected)
        if chosen.size == 0:
            continue
        index = int(chosen[0])
        tx = float(event.state[index, 2])
        ty = float(event.state[index, 3])
        if not math.isfinite(tx) or not math.isfinite(ty):
            continue
        n_selected_tracklets += 1
        tx_values.append(tx)
        ty_values.append(ty)
        slope_values.append(float(math.hypot(tx, ty)))
        azimuth_values.append(float(math.atan2(ty, tx)))
    tx_arr = np.asarray(tx_values, dtype=np.float64)
    ty_arr = np.asarray(ty_values, dtype=np.float64)
    slope_arr = np.asarray(slope_values, dtype=np.float64)
    n_slope = int(slope_arr.size)
    wide_n = int(np.count_nonzero(slope_arr >= wide_cut)) if n_slope else 0
    ip_n = int(np.count_nonzero(slope_arr < ip_cut)) if n_slope else 0
    return {
        "source_id": source_id,
        "n_events": n_events,
        "n_tracklets": int(n_tracklets),
        "n_angular_tracklets": n_selected_tracklets,
        "station_event_counts": station_event_counts,
        "n_ift_events": int(ift_events),
        "n_complete_four_station_events": int(complete),
        "stations_present": [
            station for station in REQUIRED_STATIONS if station_event_counts[str(station)] > 0
        ],
        "tx": _distribution(tx_arr),
        "ty": _distribution(ty_arr),
        "slope": _distribution(slope_arr),
        "azimuth": _distribution(np.asarray(azimuth_values, dtype=np.float64)),
        "wide_local_slope_fraction": (wide_n / n_slope) if n_slope else None,
        "ip_like_local_slope_fraction": (ip_n / n_slope) if n_slope else None,
        "slope_definition": "hypot_source_tracklet_tx_ty",
        "slope_source": "canonical_tracklet_local_state",
        "slope_is_not_leftover_residual_rtx_rty": True,
        "slope_is_not_spectrometer_delta_x_over_delta_z": True,
        "residual_blind": True,
        "angular_population_abs_pdg": [int(value) for value in angular_abs_pdg],
        "tx_values": tx_arr,
        "ty_values": ty_arr,
    }


# ---------------------------------------------------------------------------
# Deterministic provenance validation of exported sources
# ---------------------------------------------------------------------------

def _read_metadata_source_file(handle: Any) -> str | None:
    """Read ``metadata/source_file`` from a canonical tracklets file.

    Tolerates both TTree (dict-like ``arrays``) and RNTuple (structured
    ndarray) backends.
    """
    if "metadata" not in handle:
        return None
    meta = handle["metadata"]
    try:
        data = meta.arrays(library="np")
    except Exception:
        return None
    try:
        raw = data["source_file"]
    except Exception:
        return None
    try:
        if len(raw) == 0:
            return None
        return str(raw[0])
    except Exception:
        return None


def validate_exported_source(
    *,
    source_id: str,
    expected_input_path: str,
    export_dir: Path,
    job_provenance: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic provenance validation for one exported source.

    Every check is exact: the canonical output must back-reference the exact
    input xAOD, the event content must match the enhanced ntuple, station
    z positions must map to IFT/0/1/2/3 within the frozen tolerance, and
    tracklet (tx, ty) must be finite.  No fuzzy join, no nearest-neighbour
    matching.
    """
    export_dir = Path(export_dir)
    tracklets = export_dir / "tracklets.root"
    enhanced = export_dir / "enhanced_tracklets.root"
    audit = export_dir / "content_audit.json"
    checks: dict[str, Any] = {}
    failure_reasons: list[str] = []

    checks["output_tracklets_exists"] = tracklets.is_file()
    checks["enhanced_exists"] = enhanced.is_file()
    checks["content_audit_exists"] = audit.is_file()
    if not checks["output_tracklets_exists"]:
        return {
            "source_id": source_id,
            "status": VERDICT_MISSING_OUTPUT,
            "provenance_valid": False,
            "checks": checks,
            "failure_reasons": ["tracklets.root missing"],
        }

    import uproot

    # Exact back-reference: the canonical file metadata must name the exact
    # input xAOD path declared for this source.
    with uproot.open(tracklets) as handle:
        source_file = _read_metadata_source_file(handle)
    checks["metadata_source_file"] = source_file
    checks["metadata_matches_declared_input"] = source_file == str(Path(expected_input_path))
    if not checks["metadata_matches_declared_input"]:
        failure_reasons.append(
            f"canonical metadata source_file '{source_file}' != declared input '{expected_input_path}'"
        )

    try:
        events, load_prov = load_events_physical_order(tracklets)
    except (ProvenanceValidationError, ValueError) as exc:
        checks["loader_error"] = str(exc)
        return {
            "source_id": source_id,
            "status": VERDICT_PROVENANCE_FAILURE,
            "provenance_valid": False,
            "checks": checks,
            "failure_reasons": failure_reasons + [f"loader: {exc}"],
        }
    checks["loader"] = load_prov
    n_physical = int(load_prov["n_physical_events"])

    # Enhanced ntuple entry count == physical events processed by the exporter.
    n_enhanced_entries = None
    if enhanced.is_file():
        with uproot.open(enhanced) as handle:
            if "nt" in handle:
                n_enhanced_entries = int(handle["nt"].num_entries)
    checks["enhanced_entries"] = n_enhanced_entries
    if job_provenance:
        checks["job_events_processed"] = job_provenance.get("events_processed")
        checks["job_exit_status"] = job_provenance.get("exit_status")
        if job_provenance.get("exit_status") != 0:
            failure_reasons.append("export job exit status != 0")
        job_events = job_provenance.get("events_processed")
        if n_enhanced_entries is not None and job_events is not None:
            checks["enhanced_entries_match_job"] = int(job_events) == n_enhanced_entries
            if not checks["enhanced_entries_match_job"]:
                failure_reasons.append(
                    f"enhanced entries {n_enhanced_entries} != job events processed {job_events}"
                )
    # Physical events with tracklets cannot exceed the events processed.
    if n_enhanced_entries is not None:
        checks["physical_events_within_processed"] = n_physical <= n_enhanced_entries
        if not checks["physical_events_within_processed"]:
            failure_reasons.append(
                f"physical events with tracklets {n_physical} exceed processed {n_enhanced_entries}"
            )

    # Run id must be the declared production DSID; station z must map to the
    # frozen station table; tracklet (tx, ty) finite (enforced at load).
    run_ids = sorted({int(event.run_id) for event in events})
    checks["run_ids"] = run_ids
    expected_run = None
    for candidate in config.get("export_candidates") or []:
        for row in candidate.get("inputs") or []:
            if str(row["source_id"]) == source_id:
                expected_run = int(candidate.get("production_dsid"))
    if expected_run is not None:
        checks["run_id_matches_production"] = run_ids == [expected_run]
        if not checks["run_id_matches_production"]:
            failure_reasons.append(f"run ids {run_ids} != production DSID {expected_run}")
    station_z = {int(k): float(v) for k, v in config["physics_scales"]["station_z_mm"].items()}
    tolerance = float(config["physics_scales"].get("station_z_match_tolerance_mm") or 400.0)
    z_ok = True
    stations_seen: set[int] = set()
    n_tracklets_total = 0
    for event in events:
        n_tracklets_total += int(event.size)
        for station, z in zip(event.station_id, event.z_mm):
            station = int(station)
            stations_seen.add(station)
            if station not in station_z or abs(float(z) - station_z[station]) > tolerance:
                z_ok = False
    checks["station_z_mapping_valid"] = bool(z_ok)
    if not z_ok:
        failure_reasons.append("tracklet z does not map to the frozen station z table")
    checks["stations_seen"] = sorted(stations_seen)
    checks["n_tracklets"] = int(n_tracklets_total)
    checks["n_physical_events_with_tracklets"] = n_physical
    if n_tracklets_total == 0:
        return {
            "source_id": source_id,
            "status": VERDICT_SEGMENTFIT_EMPTY,
            "provenance_valid": not failure_reasons,
            "checks": checks,
            "failure_reasons": failure_reasons,
        }
    status = "ok" if not failure_reasons else VERDICT_PROVENANCE_FAILURE
    return {
        "source_id": source_id,
        "status": status,
        "provenance_valid": not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def wrong_source_negative_control(
    *,
    source_id: str,
    wrong_input_path: str,
    export_dir: Path,
) -> dict[str, Any]:
    """Negative control: validating against the wrong input must fail."""
    tracklets = Path(export_dir) / "tracklets.root"
    if not tracklets.is_file():
        return {"control": "wrong_source", "ran": False, "reason": "no output to validate"}
    import uproot

    with uproot.open(tracklets) as handle:
        source_file = _read_metadata_source_file(handle)
    mismatch = source_file != str(Path(wrong_input_path))
    return {
        "control": "wrong_source",
        "ran": True,
        "source_id": source_id,
        "declared_wrong_input": str(wrong_input_path),
        "metadata_source_file": source_file,
        "mismatch_detected": bool(mismatch),
        "control_passed": bool(mismatch),
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
    }


# ---------------------------------------------------------------------------
# Reinventory and per-candidate verdicts
# ---------------------------------------------------------------------------

def classify_candidate_verdict(
    admission: Mapping[str, Any],
    *,
    provenance_valid: bool,
    exporter_compatible: bool,
    n_tracklets: int,
    n_sources_with_output: int,
) -> str:
    """Map the workbook-74 admission result to the stage-75 verdict label."""
    if not provenance_valid:
        return VERDICT_PROVENANCE_FAILURE
    if not exporter_compatible:
        return VERDICT_EXPORTER_INCOMPATIBLE
    if n_sources_with_output == 0:
        return VERDICT_MISSING_OUTPUT
    if n_tracklets == 0:
        return VERDICT_SEGMENTFIT_EMPTY
    if admission.get("admitted_to_separate_fd_campaign"):
        return VERDICT_ADMITTED
    stats_ok = (
        int(admission.get("n_events") or 0) >= 200
        and int(admission.get("n_ift_events") or 0) >= 200
        and int(admission.get("n_complete_four_station_events") or 0) >= 80
        and int(admission.get("n_independent_sources") or 0) >= 2
    )
    if stats_ok and not admission.get("observed_phase_space_distinct"):
        return VERDICT_NOT_DISTINCT
    return VERDICT_INSUFFICIENT


def decide_next_stage_75(
    verdicts: Sequence[Mapping[str, Any]],
    *,
    fd_executed: bool = False,
) -> dict[str, Any]:
    """Stage-75 next-stage decision from per-candidate verdicts."""
    if fd_executed:
        raise ResidualBlindStageError(
            "Stage 75 is residual-blind export + reinventory.  FD identifiability, "
            "SVD, and rank are forbidden in this stage."
        )
    admitted = [row for row in verdicts if row.get("verdict") == VERDICT_ADMITTED]
    if admitted:
        decision = DECISION_ADMIT_FD_PREREGISTRATION
        answer = (
            "At least one candidate satisfies every frozen workbook-74 gate.  "
            "Stage 75 does not open FD; a separate pre-registered native 5DoF FD "
            "campaign config and workbook entry are required first."
        )
        next_step = "preregister_separate_native_five_dof_fd_campaign_per_admitted_candidate"
        freeze_insufficient = False
        open_gauge = False
    else:
        decision = DECISION_INSUFFICIENT
        answer = "No"
        next_step = "preregister_gauge_constrained_or_external_constraint_alignment_feasibility"
        freeze_insufficient = True
        open_gauge = True
    return {
        **common_inventory_state(),
        "schema_version": SCHEMA_VERSION,
        "stage": "residual_blind_htcondor_export_and_reinventory",
        "inherited_entry_74_decision": INHERITED_ENTRY_74,
        "decision": decision,
        "answer": answer,
        "per_candidate_verdicts": [dict(row) for row in verdicts],
        "admitted_candidates": [row.get("id") for row in admitted],
        "fd_identifiability_executed": False,
        "svd_or_rank_computed": False,
        "freeze_current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof": freeze_insufficient,
        "open_gauge_constrained_or_external_constraint_branch": open_gauge,
        "gauge_convention_is_not_a_physical_measurement": True,
        "survey_may_enter_only_as_independent_external_constraint": True,
        "do_not_treat_population_spread_as_prior": True,
        "three_arm_authorized": False,
        "frozen_v2_alignment_loop_authorized": False,
        "real_data_correction_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "next_allowed_step": next_step,
        "reason": (
            "Per-candidate verdicts use the frozen workbook-74 gates: "
            "metadata-distinct, observed-(tx,ty)-distinct, 200 events, 200 IFT "
            "events, 80 complete-four-station events, at least 2 independent "
            "file-level sources, IFT and stations 0-3 coverage, and no metadata "
            "prohibition.  Candidates are never pooled.  Stage 75 does not open "
            "FD; an admitted candidate requires a separate pre-registered "
            "native 5DoF FD campaign.  If no candidate is admitted, the "
            "current track coverage is frozen as insufficient for "
            "unconstrained rigid-station 5DoF and only then may a "
            "gauge/external-constraint branch be pre-registered."
        ),
    }
