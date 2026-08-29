"""Non-collision / cross-year track-topology inventory.

Residual-blind provenance and topology search for a real FASER population
that is physically different from ordinary 2024 r0022 collision-like tracks.
Does not train, write geometry, invent a cosine cut, mix cross-year
residuals, or restack 2024 r0022 collision windows.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
)
from alignment.true_cluster_local_residual import (
    RESIDUAL_KIND,
    common_operating_state,
)

SCHEMA_VERSION = "faser-noncollision-crossyear-topology-identifiability-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "noncollision_crossyear_topology_identifiability_v1.yaml"
DECISION_FOUND = "portable_high_lever_arm_topology_found"
DECISION_INSUFFICIENT = "real_track_topology_insufficient_for_ry_cdx_separation"
GO_NO_GO_QUESTION = (
    "Does existing FASER real data contain a topology different from ordinary "
    "2024 r0022 collision-like tracks, repeatable, with enough angular / "
    "lever-arm information to stably lift ry↔C_dx degeneracy?"
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
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
    "do_not_restack_2024_r0022_collision_like",
    "do_not_mix_cross_year_residuals_or_alignment_constants",
    "software_fd_sensitivity_only",
)
XAOD_KEY_INTEREST = (
    "SCT_ClusterContainer",
    "SegmentFit",
    "Segments",
    "EventInfo",
    "FaserLHCData",
    "CKFTrackCollection",
    "CKFTrackCollectionWithoutIFT",
    "TruthParticles",
)
LOG_LINE_LIMIT = 400


def load_inventory_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"cross-year topology config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected cross-year topology schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"cross-year topology config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"cross-year topology config must set {key}=false")
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
            "do_not_enter_full_module_identifiability_map": True,
            "do_not_invent_new_cosine_cut": True,
            "do_not_select_events_from_residual_or_cosine": True,
            "do_not_restack_2024_r0022_collision_like": True,
            "do_not_mix_cross_year_residuals_or_alignment_constants": True,
            "schema_version": SCHEMA_VERSION,
            "go_no_go_question": GO_NO_GO_QUESTION,
        }
    )
    return state


def _safe_listdir(path: Path) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []
    except OSError:
        return []


def _count_run_dirs(path: Path) -> dict[str, Any]:
    names = _safe_listdir(path)
    runs = [name for name in names if name.isdigit()]
    return {
        "n_children": len(names),
        "n_run_directories": len(runs),
        "first_runs": runs[:8],
        "last_runs": runs[-4:] if len(runs) > 8 else [],
    }


def inventory_eos_layout(config: Mapping[str, Any]) -> dict[str, Any]:
    eos = config["eos"]
    rec_root = Path(str(eos["rec_root"]))
    data0_rec = Path(str(eos["data0_rec_root"]))
    phys_root = Path(str(eos["phys_root"]))
    years = {}
    for year in eos.get("years") or []:
        rec_path = rec_root / str(year)
        tags = [name for name in _safe_listdir(rec_path) if not name.endswith("_log")]
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = _count_run_dirs(rec_path / tag)
        years[str(year)] = {
            "rec_path": str(rec_path),
            "exists": rec_path.is_dir(),
            "tags": tags,
            "tag_run_counts": tag_counts,
            "data0_exists": (data0_rec / str(year)).is_dir(),
        }
    extras = {}
    for name in eos.get("extra_rec_trees") or []:
        path = rec_root / str(name)
        tags = [item for item in _safe_listdir(path) if not item.endswith("_log")]
        extras[str(name)] = {
            "path": str(path),
            "exists": path.is_dir(),
            "tags": tags,
            "tag_run_counts": {tag: _count_run_dirs(path / tag) for tag in tags},
        }
    phys_streams = {}
    if phys_root.is_dir():
        for stream in _safe_listdir(phys_root):
            stream_path = phys_root / stream
            if not stream_path.is_dir():
                continue
            children = [name for name in _safe_listdir(stream_path) if not name.endswith("_log")]
            phys_streams[stream] = {
                "path": str(stream_path),
                "n_children": len(children),
                "children": children[:12],
            }
    rec_keyword_hits = []
    keywords = ("cosmic", "halo", "calib", "special", "beamtest", "testbeam")
    for year, block in years.items():
        for tag in block.get("tags") or []:
            if any(key in tag.lower() for key in keywords):
                rec_keyword_hits.append(f"{year}/{tag}")
    for name, block in extras.items():
        for tag in block.get("tags") or []:
            if any(key in tag.lower() for key in keywords):
                rec_keyword_hits.append(f"{name}/{tag}")
    return {
        "rec_root": str(rec_root),
        "data0_rec_root": str(data0_rec),
        "phys_root": str(phys_root),
        "years": years,
        "extra_rec_trees": extras,
        "phys_streams": phys_streams,
        "named_cosmic_or_halo_stream_in_rec_tree": bool(rec_keyword_hits),
        "rec_keyword_directory_hits": rec_keyword_hits,
        "named_cosmic_phys_streams": [
            name for name in phys_streams if name.endswith("_cos") or "_cos" in name
        ],
        "note": (
            "PHYS stream names are ntuple filters, not rec-tree run types. "
            "2024_cos / 2023_cos are topology sources; they must not be mixed "
            "into 2024 r0022 alignment constants."
        ),
    }


def parse_job_log(path: str | Path, *, n_lines: int = LOG_LINE_LIMIT) -> dict[str, Any]:
    source = Path(path)
    report: dict[str, Any] = {
        "path": str(source),
        "exists": source.is_file(),
        "geometry_tag": None,
        "conditions_tag": None,
        "geom_flag": None,
        "no_ift_four_station_ckf": None,
        "remaining_args": None,
        "cosmics_only": None,
        "use_ift": None,
        "stable_beams_flag": None,
        "backward": None,
        "trigger_mask": None,
    }
    if not source.is_file():
        return report
    text_lines: list[str] = []
    with source.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= int(n_lines):
                break
            text_lines.append(line.rstrip("\n"))
    blob = "\n".join(text_lines)
    remaining = re.search(r"Remaining:\s*(.*)", blob)
    if remaining:
        report["remaining_args"] = remaining.group(1).strip()
        args = report["remaining_args"]
        report["no_ift_four_station_ckf"] = "--noIFT" in args
        report["use_ift"] = "--useIFT" in args
        report["cosmics_only"] = "--cosmics" in args
        report["backward"] = "--backward" in args
        geom = re.search(r"--geom\s+(\S+)", args)
        if geom:
            report["geom_flag"] = geom.group(1)
        mask = re.search(r"--triggerMask\s+(\d+)", args)
        if mask:
            report["trigger_mask"] = int(mask.group(1))
        if "--no_stable" in args:
            report["stable_beams_flag"] = False
    if report["geom_flag"] is None:
        typed = re.search(r"with type\s+(TI12\S+|FASER\S+)", blob)
        named = re.search(r"Geom:\s+(\S+)", blob)
        if typed:
            report["geom_flag"] = typed.group(1)
        elif named:
            report["geom_flag"] = named.group(1)
    conditions = re.search(r"Global tag:\s*(\S+)", blob)
    if conditions:
        report["conditions_tag"] = conditions.group(1)
    geometry = re.search(r"Version Tag:\s*(FASERNU-\d+|FASER-TB\d+)", blob)
    if geometry:
        report["geometry_tag"] = geometry.group(1)
    if re.search(r"Cosmics = True", blob):
        report["cosmics_only"] = True
    if re.search(r"Cosmics = False", blob) and report["cosmics_only"] is None:
        report["cosmics_only"] = False
    if re.search(r"Use IFT = True", blob):
        report["use_ift"] = True
    if re.search(r"Stable Beams = True", blob):
        report["stable_beams_flag"] = True
    if re.search(r"Stable Beams = False", blob):
        report["stable_beams_flag"] = False
    if re.search(r"Backward = True", blob):
        report["backward"] = True
    return report


def probe_xaod(path: str | Path, *, year: int | None = None, rec_tag: str | None = None) -> dict[str, Any]:
    import uproot

    source = Path(path)
    report: dict[str, Any] = {
        "path": str(source),
        "exists": source.is_file(),
        "year": year,
        "rec_tag": rec_tag,
        "n_events": None,
        "collection_keys_of_interest": {},
        "file_metadata": {},
        "usable_as_cluster_local_jacobian_input": False,
        "missing_required_for_cdx_jacobian": [],
    }
    if not source.is_file():
        report["missing_required_for_cdx_jacobian"] = ["file"]
        return report
    with uproot.open(source) as handle:
        if "CollectionTree" not in handle:
            report["missing_required_for_cdx_jacobian"] = ["CollectionTree"]
            return report
        tree = handle["CollectionTree"]
        keys = {str(name).split(";")[0] for name in tree.keys()}
        report["n_events"] = int(tree.num_entries)
        report["collection_keys_of_interest"] = {key: (key in keys) for key in XAOD_KEY_INTEREST}
        if "MetaData" in handle:
            meta = handle["MetaData"]
            meta_keys = [str(name).split(";")[0] for name in meta.keys()]
            for key in meta_keys:
                if key.endswith("conditionsTag") or key.endswith("productionRelease") or key.endswith("dataType"):
                    try:
                        values = meta[key].array(library="np")
                        report["file_metadata"][key] = [
                            str(item) for item in (values.tolist() if hasattr(values, "tolist") else [values])
                        ][:4]
                    except Exception as exc:  # noqa: BLE001 — inventory must continue
                        report["file_metadata"][key] = f"unreadable:{type(exc).__name__}"
    required = ["SCT_ClusterContainer", "SegmentFit", "EventInfo"]
    missing = [key for key in required if not report["collection_keys_of_interest"].get(key)]
    if report["collection_keys_of_interest"].get("TruthParticles"):
        missing.append("looks_like_mc")
    report["missing_required_for_cdx_jacobian"] = missing
    report["usable_as_cluster_local_jacobian_input"] = not missing
    report["has_ift_ckf_collection"] = bool(report["collection_keys_of_interest"].get("CKFTrackCollection"))
    report["has_three_station_ckf"] = bool(
        report["collection_keys_of_interest"].get("CKFTrackCollectionWithoutIFT")
    )
    report["has_lhc_data"] = bool(report["collection_keys_of_interest"].get("FaserLHCData"))
    return report


def assign_station_from_z(z_mm: float, physics: Mapping[str, Any]) -> int | None:
    table = physics["station_z_mm"]
    tolerance = float(physics.get("station_z_match_tolerance_mm") or 400.0)
    best: int | None = None
    best_distance = float("inf")
    for station, nominal in table.items():
        distance = abs(float(z_mm) - float(nominal))
        if distance < best_distance:
            best_distance = distance
            best = int(station)
    if best is None or best_distance > tolerance:
        return None
    return best


def _as_seq(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, (bytes, str)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _finite_slope(dx: float, dy: float, dz: float) -> float:
    if not math.isfinite(dx) or not math.isfinite(dy) or not math.isfinite(dz) or abs(dz) < 1.0:
        return float("nan")
    return float(math.hypot(dx / dz, dy / dz))


def summarize_phys_topology(
    path: str | Path,
    physics: Mapping[str, Any],
    *,
    max_events: int = 20000,
) -> dict[str, Any]:
    import uproot

    source = Path(path)
    wide_cut = float(physics["wide_angle_min_slope"])
    ip_cut = float(physics["ip_like_max_slope"])
    report: dict[str, Any] = {
        "path": str(source),
        "exists": source.is_file(),
        "n_file_events": 0,
        "n_used": 0,
        "residual_blind": True,
        "used_unassociated_segments_for_gate": False,
    }
    if not source.is_file():
        report["error"] = "missing_file"
        return report
    with uproot.open(source) as handle:
        if "nt" not in handle:
            report["error"] = "missing_nt_tree"
            return report
        tree = handle["nt"]
        n_file = int(tree.num_entries)
        report["n_file_events"] = n_file
        keys = {str(name).split(";")[0] for name in tree.keys()}
        branches = [name for name in ("nClusters0", "nClusters1", "nClusters2", "nClusters3") if name in keys]
        has_ckf = all(
            name in keys
            for name in ("Track_x0", "Track_y0", "Track_z0", "Track_x1", "Track_y1", "Track_z1")
        )
        if has_ckf:
            branches.extend(["Track_x0", "Track_y0", "Track_z0", "Track_x1", "Track_y1", "Track_z1"])
        has_seg = all(name in keys for name in ("TrackSegment_x", "TrackSegment_y", "TrackSegment_z"))
        if has_seg:
            branches.extend(["TrackSegment_z"])
        if not branches:
            report["error"] = "missing_cluster_and_track_branches"
            return report
        stop = min(n_file, int(max_events))
        arrays = tree.arrays(branches, entry_stop=stop, library="np")
    n_used = int(len(next(iter(arrays.values()))))
    report["n_used"] = n_used
    if n_used == 0:
        report["empty_tree"] = True
        return report

    def cluster_count(station: int) -> np.ndarray:
        key = f"nClusters{station}"
        if key not in arrays:
            return np.zeros(n_used, dtype=np.int32)
        return np.asarray(arrays[key])

    n0, n1, n2, n3 = (cluster_count(station) for station in range(4))
    ift = n0 > 0
    four = (n0 > 0) & (n1 > 0) & (n2 > 0) & (n3 > 0)
    n_stations = (n0 > 0).astype(int) + (n1 > 0).astype(int) + (n2 > 0).astype(int) + (n3 > 0).astype(int)
    ge3 = n_stations >= 3
    long_lever = (n0 > 0) & (n3 > 0)
    report["cluster_occupancy"] = {
        "ift_fraction": float(np.mean(ift)),
        "four_station_coincidence": float(np.mean(four)),
        "ge3_station_fraction": float(np.mean(ge3)),
        "station0_and_station3_fraction": float(np.mean(long_lever)),
        "n_ge3_ift_events": int(np.sum(ge3 & ift)),
        "n_four_station_events": int(np.sum(four)),
        "observable": "nClusters0-3 in PHYS nt",
        "note": "Track-filtered PHYS is not unbiased occupancy. Compare only like-with-like.",
    }

    ckf_slopes: list[float] = []
    if has_ckf:
        for index in range(n_used):
            x0 = _as_seq(arrays["Track_x0"][index])
            y0 = _as_seq(arrays["Track_y0"][index])
            z0 = _as_seq(arrays["Track_z0"][index])
            x1 = _as_seq(arrays["Track_x1"][index])
            y1 = _as_seq(arrays["Track_y1"][index])
            z1 = _as_seq(arrays["Track_z1"][index])
            if not x0 or not x1 or not z0 or not z1:
                continue
            slope = _finite_slope(
                float(x1[0]) - float(x0[0]),
                float(y1[0]) - float(y0[0]),
                float(z1[0]) - float(z0[0]),
            )
            if math.isfinite(slope):
                ckf_slopes.append(slope)
    report["ckf_single_track_slope"] = _slope_pack(ckf_slopes, wide_cut, ip_cut)
    report["ckf_single_track_slope"]["definition"] = (
        "first CKF track endpoint (x0,y0,z0)→(x1,y1,z1); single-track spectrometer analog"
    )

    z_values: list[float] = []
    station_hits = defaultdict(int)
    unmatched_z = 0
    if has_seg:
        for row in arrays["TrackSegment_z"][: min(n_used, 5000)]:
            for z_mm in _as_seq(row):
                z_values.append(float(z_mm))
                station = assign_station_from_z(float(z_mm), physics)
                if station is None:
                    unmatched_z += 1
                else:
                    station_hits[station] += 1
    report["segment_z_coverage"] = {
        "n_points_sampled": len(z_values),
        "z_min": min(z_values) if z_values else None,
        "z_max": max(z_values) if z_values else None,
        "z_median": float(np.median(z_values)) if z_values else None,
        "assigned_station_counts": {str(key): int(value) for key, value in sorted(station_hits.items())},
        "unmatched_z_points": unmatched_z,
        "matches_fasernu04_station_z": bool(z_values) and unmatched_z == 0 and len(station_hits) > 0,
        "note": "Segment z is residual-blind coverage, not a single-track slope.",
    }
    return report


def _slope_pack(slopes: Sequence[float], wide_cut: float, ip_cut: float) -> dict[str, Any]:
    values = [float(item) for item in slopes if math.isfinite(float(item))]
    if not values:
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "wide_fraction": None,
            "ip_like_fraction": None,
            "intermediate_fraction": None,
        }
    array = np.asarray(values, dtype=np.float64)
    n_wide = int(np.sum(array >= wide_cut))
    n_ip = int(np.sum(array < ip_cut))
    n_mid = int(len(array) - n_wide - n_ip)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "wide_fraction": n_wide / len(array),
        "ip_like_fraction": n_ip / len(array),
        "intermediate_fraction": n_mid / len(array),
        "wide_cut": wide_cut,
        "ip_like_cut": ip_cut,
    }


def jacobian_admission_for_probe(
    probe: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    rules: Mapping[str, Any],
    xaod_by_year_tag: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    topology = str(probe.get("topology_id") or "")
    role = str(probe.get("role") or "candidate")
    stats = probe.get("topology") or {}
    ckf = stats.get("ckf_single_track_slope") or {}
    occ = stats.get("cluster_occupancy") or {}
    reasons: list[str] = []
    if role == "collision_phys_baseline":
        return {
            "id": probe.get("id"),
            "topology_id": topology,
            "admitted_to_jacobian": False,
            "reason": "collision PHYS baseline; not an independent non-collision topology",
        }
    if topology in {"phys_ift_collision", "rec_2022_collision", "rec_2023_collision", "rec_2025_collision"}:
        reasons.append("metadata class is still collision-like, not a distinct non-collision topology")
    if topology == "phys_backward_ckf":
        reasons.append("backward CKF is reverse propagation of collision tracks, not a new incident-angle source")
    if topology == "phys_alps_filter":
        reasons.append("ALPS is a physics filter, not an angular/lever-arm topology")
    n_used = int(stats.get("n_used") or 0)
    n_file = int(stats.get("n_file_events") or 0)
    n_ge3_ift = int(occ.get("n_ge3_ift_events") or 0)
    ckf_wide = ckf.get("wide_fraction")
    occupancy_four = occ.get("four_station_coincidence")
    baseline_occ_four = float(baseline["occupancy_four_station_coincidence_in_100_event_window"])
    min_wide = float(rules["min_wide_dominated_ckf_fraction"])
    min_four_ratio = float(rules["min_unbiased_four_station_coincidence_ratio_vs_r0022_occupancy"])
    min_ge3 = int(rules["min_ge3_ift_events_per_scanned_file"])
    four_ratio = None
    if occupancy_four is not None and baseline_occ_four > 0:
        four_ratio = float(occupancy_four) / baseline_occ_four
    cosmic_like = topology in {"phys_2024_cosmic_beam_mode", "phys_2023_cosmic_like"}
    if cosmic_like:
        wide_ok = ckf_wide is not None and float(ckf_wide) >= min_wide
        four_ok = four_ratio is not None and four_ratio >= min_four_ratio
        if not wide_ok and not four_ok:
            reasons.append(
                f"cosmic-like sample is neither wide-dominated (CKF wide>={min_wide}) "
                f"nor >= {min_four_ratio:.1f}x r0022 occupancy four-station"
            )
        if n_ge3_ift < min_ge3:
            reasons.append(f"ge3+IFT events {n_ge3_ift} < {min_ge3}")
        if n_file < 1000 and n_used < 1000:
            reasons.append(f"scanned sample is tiny (n_file={n_file})")
    if bool(rules.get("require_ift_for_cdx_jacobian")) and n_ge3_ift <= 0 and n_used > 0:
        reasons.append("no >=3-station+IFT cluster events in the scanned file")
    if topology == "testbeam_2021":
        reasons.append("TestBeam uses FASER-TB00; IFT/C_dx Jacobian does not apply")
    admitted = not reasons
    return {
        "id": probe.get("id"),
        "topology_id": topology,
        "admitted_to_jacobian": admitted,
        "n_file_events": n_file,
        "n_used": n_used,
        "n_ge3_ift_events": n_ge3_ift,
        "ckf_wide_fraction": ckf_wide,
        "four_station_coincidence": occupancy_four,
        "four_station_ratio_vs_r0022_occupancy": four_ratio,
        "xaod_required": bool(rules.get("require_xaod_with_clusters_and_segmentfit")),
        "cross_year_residual_mix_forbidden": bool(rules.get("forbid_cross_year_residual_or_constant_mix")),
        "reason": "admitted" if admitted else "; ".join(reasons),
    }


def decide_next_stage(
    *,
    admissions: Sequence[Mapping[str, Any]],
    jacobian_executed: bool,
    portable_within_period: bool,
) -> dict[str, Any]:
    admitted = [row for row in admissions if row.get("admitted_to_jacobian")]
    if jacobian_executed and portable_within_period:
        return {
            **common_audit_state(),
            "go_no_go_question": GO_NO_GO_QUESTION,
            "answer": "Yes",
            "decision": DECISION_FOUND,
            "admitted_sources": [row.get("id") for row in admitted],
            "go_to_full_module_identifiability_map": True,
            "still_no_new_network": True,
            "next_allowed_step": "full_module_level_identifiability_map_no_new_network",
            "reason": (
                "A predeclared real topology is rank-3, bootstrap-concentrated, "
                "and transfers across independent runs of the same data period."
            ),
        }
    return {
        **common_audit_state(),
        "go_no_go_question": GO_NO_GO_QUESTION,
        "answer": "No",
        "decision": DECISION_INSUFFICIENT,
        "admitted_sources": [row.get("id") for row in admitted],
        "jacobian_executed": bool(jacobian_executed),
        "go_to_full_module_identifiability_map": False,
        "still_no_new_network": True,
        "do_not_sink_ml_to_more_complex_cluster_architecture": True,
        "next_allowed_step": (
            "external_survey_mechanical_constraints_and_year_iov_dependent_alignment_framework"
        ),
        "reason": (
            "No existing real-data source supplies a repeatable, physically "
            "different topology with enough angular / lever-arm information to "
            "stably lift ry↔C_dx. Cosmic PHYS streams are occupancy-empty; "
            "backward/ALPS/IFT PHYS streams remain collision-like; other years "
            "are the same TI12 collision topology on different alignment IOVs; "
            "TestBeam2021 is FASER-TB00 without a C_dx IFT lever. Do not add ML "
            "complexity. Mainline constraints are external survey / mechanical "
            "limits and a year/IOV-dependent alignment framework."
        ),
    }


def predeclared_topology_report(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **common_audit_state(),
        "frozen_before_jacobian": True,
        "frozen_before_residual_inspection": True,
        "selection_forbidden": [
            "unbiased_residual_u_mm",
            "station_ry_vs_C_dx",
            "station_dx_vs_C_dx",
            "cosine",
            "unassociated_event_segment_endpoint_slope",
        ],
        "physics_scales": dict(config["physics_scales"]),
        "categories": list(config["predeclared_topologies"]),
        "jacobian_admission": dict(config["jacobian_admission"]),
        "note": (
            "Categories come from named PHYS streams, ntuple flags, rec geom/conditions "
            "tags, and detector configuration. Cuts are stereo/10 and IP acceptance. "
            "They are not entry-58 tertiles and are not tuned on |cos(ry,C_dx)|."
        ),
    }
