"""Frozen V2 real-data route-acceptance scaling study.

Expands event counts from the entry-48 occupancy windows without changing
V2, thresholds, unmatched penalty, route logic, or occupancy selection.
14977 is reported but never used to choose a verdict.  Residuals and
alignment parameters are out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.real_data_occupancy_preflight import (
    assert_occupancy_tree_is_residual_blind,
    occupancy_window_summary,
)
from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-v2-route-acceptance-scaling"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "operating_protocol_v1_real_data_v2_route_acceptance_scaling_v1.yaml"

VERDICT_STATISTICS_LIMITED = "real_data_route_acceptance_statistics_limited"
VERDICT_DOMAIN_SHIFT = "real_data_v2_domain_shift_candidate"
VERDICT_TRACKLET_LIMITED = "real_data_tracklet_statistics_limited"
VERDICT_PENDING = "real_data_route_acceptance_scaling_pending"
VERDICTS = (
    VERDICT_STATISTICS_LIMITED,
    VERDICT_DOMAIN_SHIFT,
    VERDICT_TRACKLET_LIMITED,
    VERDICT_PENDING,
)
SCALE_ORDER = ("n100", "n1000", "n10000", "full")


def load_scaling_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"scaling config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected scaling config schema: {source}")
    if payload.get("frozen") is not True:
        raise ValueError("scaling config must be frozen")
    if payload.get("do_not_enter_alignment") is not True:
        raise ValueError("scaling config must forbid alignment")
    if payload.get("do_not_change_thresholds") is not True or payload.get("do_not_change_unmatched_penalty") is not True:
        raise ValueError("scaling config must forbid threshold/penalty changes")
    if payload.get("geometry_write_allowed") is not False:
        raise ValueError("scaling config must keep geometry_write_allowed false")
    if payload.get("use_residual") is not False:
        raise ValueError("scaling config must not use residuals")
    return {
        "path": str(source),
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        **{key: payload[key] for key in payload},
    }


def load_occupancy_table(path: Path) -> dict[str, np.ndarray]:
    import uproot

    with uproot.open(path) as source:
        tree_name = "occupancy" if "occupancy" in source else None
        if tree_name is None:
            for key in source.keys():
                name = str(key).split(";")[0]
                if name.endswith("occupancy") or name == "occupancy":
                    tree_name = name
                    break
        if tree_name is None:
            raise ValueError(f"occupancy tree is absent from {path}")
        tree = source[tree_name]
        assert_occupancy_tree_is_residual_blind([str(name) for name in tree.keys()])
        arrays = tree.arrays(library="np")
    return {str(key): np.asarray(value) for key, value in arrays.items()}


def occupancy_slice_summary(
    table: Mapping[str, np.ndarray],
    *,
    skip_events: int,
    nevents: int,
) -> dict[str, Any]:
    n_rows = int(np.asarray(table["n_sct_clusters"]).shape[0])
    start = int(skip_events)
    stop = min(n_rows, start + int(nevents))
    if start < 0 or start >= n_rows or stop <= start:
        raise ValueError(f"occupancy slice [{start}, {start}+{nevents}) is empty")
    chunk = {key: np.asarray(value)[start:stop] for key, value in table.items()}
    summary = occupancy_window_summary(chunk)
    summary["skip_events"] = start
    summary["requested_nevents"] = int(nevents)
    summary["n_events_available"] = int(stop - start)
    summary["segment_events"] = n_rows
    summary["capped_to_segment"] = bool(start + int(nevents) > n_rows)
    return summary


def _remaining_events(n_segment: int, skip_events: int) -> int:
    remaining = int(n_segment) - int(skip_events)
    if remaining < 1:
        raise ValueError("frozen skip leaves no events in the segment")
    return remaining


def plan_scale(
    *,
    run: int,
    role: str,
    segment: str,
    skip_events: int,
    n_segment_events: int,
    scale_name: str,
    requested_nevents: int | str,
) -> dict[str, Any]:
    remaining = _remaining_events(n_segment_events, skip_events)
    if requested_nevents == "full_segment_remaining":
        actual = remaining
        requested = remaining
    else:
        requested = int(requested_nevents)
        actual = min(requested, remaining)
    capped = actual < requested if requested_nevents != "full_segment_remaining" else False
    is_full = actual == remaining
    reuse = scale_name == "n100"
    source_id = (
        f"data24_r{int(run):05d}_{segment}_skip{int(skip_events):05d}"
        if reuse
        else f"data24_r{int(run):05d}_{segment}_skip{int(skip_events):05d}_n{int(actual):05d}"
    )
    return {
        "scale": scale_name,
        "run": int(run),
        "role": str(role),
        "segment": str(segment),
        "skip_events": int(skip_events),
        "requested_nevents": requested if requested_nevents != "full_segment_remaining" else "full_segment_remaining",
        "nevents": int(actual),
        "remaining_after_skip": int(remaining),
        "capped_to_segment": capped or is_full and requested_nevents != "full_segment_remaining" and requested > remaining,
        "is_full_segment": is_full,
        "reuse_existing_100_event": reuse,
        "needs_athena": not reuse,
        "source_id": source_id,
        "geometry_write_allowed": False,
        "station_mode_blocked": True,
        "cdx_mode_blocked": True,
    }


def plan_campaign(
    frozen_windows: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    roles = config["blind_roles"]
    runs = frozen_windows["runs"]
    planned: list[dict[str, Any]] = []
    seen_jobs: set[tuple[int, int]] = set()
    for scale in config["scales"]:
        scale_name = str(scale["name"])
        requested = scale["requested_nevents"]
        for run_key, row in runs.items():
            run = int(row["run"])
            role = str(roles[str(run)])
            if str(row.get("role")) != role:
                raise ValueError(f"frozen role drift for run {run}")
            item = plan_scale(
                run=run,
                role=role,
                segment=str(row["segment"]),
                skip_events=int(row["skip_events"]),
                n_segment_events=int(row["n_events_scanned"]),
                scale_name=scale_name,
                requested_nevents=requested,
            )
            job_key = (run, int(item["nevents"]))
            if job_key in seen_jobs and item["needs_athena"]:
                item = dict(item)
                item["needs_athena"] = False
                item["duplicate_of_existing_scale_job"] = True
            elif item["needs_athena"]:
                seen_jobs.add(job_key)
            planned.append(item)
    return planned


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("p05", "p16", "p50", "p84", "p95")}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in ("p05", "p16", "p50", "p84", "p95")}
    return {
        "p05": float(np.percentile(finite, 5)),
        "p16": float(np.percentile(finite, 16)),
        "p50": float(np.percentile(finite, 50)),
        "p84": float(np.percentile(finite, 84)),
        "p95": float(np.percentile(finite, 95)),
    }


def association_rejection_counts(
    *,
    n_all_pairs_candidates: int,
    n_adjacent_candidates: int,
    n_adjacent_below_threshold: int,
    n_adjacent_above_threshold: int,
    n_physical_complete_routes: int,
    n_complete_routes_missing_above_threshold_edge: int,
    n_routes_above_threshold_nonpositive_utility: int,
    n_routes_positive_utility: int,
    n_selected_routes: int,
) -> dict[str, int]:
    """Truth-free candidate rejection tally.  No residual fields."""
    return {
        "no_all_pairs_candidates": int(n_all_pairs_candidates <= 0),
        "adjacent_score_below_threshold": int(n_adjacent_below_threshold),
        "adjacent_score_above_threshold": int(n_adjacent_above_threshold),
        "complete_route_missing_above_threshold_edge": int(n_complete_routes_missing_above_threshold_edge),
        "route_above_threshold_nonpositive_utility": int(n_routes_above_threshold_nonpositive_utility),
        "route_positive_utility_not_selected": int(max(n_routes_positive_utility - n_selected_routes, 0)),
        "selected_routes": int(n_selected_routes),
        "physical_complete_routes": int(n_physical_complete_routes),
        "adjacent_candidates": int(n_adjacent_candidates),
    }


def _calibration_blocks(rows: Sequence[Mapping[str, Any]], scale: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("scale") == scale
        and row.get("role") == ROLE_CALIBRATION
        and row.get("association_complete") is True
    ]


def _stable_selected(blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    if len(blocks) != 2:
        return False
    max_share = float(config.get("max_event_share_of_selected_routes", 0.5))
    for block in blocks:
        share = float(
            ((block.get("event_concentration") or {}).get("max_event_share_of_selected_routes")) or 0.0
        )
        if int(block.get("selected_routes") or 0) < 1:
            return False
        if share > max_share:
            return False
    return True


def _stable_complete(blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    if not _stable_selected(blocks, config):
        return False
    min_complete = int(config.get("stable_complete_route_min", 1))
    return all(int(block.get("complete_four_station_routes") or 0) >= min_complete for block in blocks)


def _both_zero_selected(blocks: Sequence[Mapping[str, Any]]) -> bool:
    return len(blocks) == 2 and all(int(block.get("selected_routes") or 0) == 0 for block in blocks)


def _many_all_pairs(blocks: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    floor = int(config.get("many_all_pairs_candidates", 50))
    return len(blocks) == 2 and all(int(block.get("n_all_pairs_candidates") or 0) >= floor for block in blocks)


def assign_scaling_verdict(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide from calibration only.  Holdout/14977 never select the label."""
    excluded = {ROLE_HOLDOUT, ROLE_HELD_OUT_DQ}
    if any(row.get("role") in excluded and row.get("used_for_verdict") for row in rows):
        raise ValueError("holdout and held-out DQ must not be used for verdict selection")
    calibration = {
        scale: _calibration_blocks(rows, scale)
        for scale in SCALE_ORDER
    }
    largest = next((scale for scale in reversed(SCALE_ORDER) if len(calibration[scale]) == 2), None)
    if largest is None:
        return {
            "decision": VERDICT_PENDING,
            "largest_complete_calibration_scale": None,
            "geometry_write_allowed": False,
            "station_mode_blocked": True,
            "cdx_mode_blocked": True,
            "verdict_roles": [ROLE_CALIBRATION],
            "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
            "reasons": ["calibration_association_incomplete"],
        }
    if largest == "n100":
        return {
            "decision": VERDICT_PENDING,
            "largest_complete_calibration_scale": "n100",
            "geometry_write_allowed": False,
            "station_mode_blocked": True,
            "cdx_mode_blocked": True,
            "verdict_roles": [ROLE_CALIBRATION],
            "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
            "reasons": ["expanded_scales_not_yet_associated"],
        }
    largest_blocks = calibration[largest]
    mc_ref = float(config.get("mc_transfer_100_event_tracklets", 376))
    fraction = float(config.get("tracklet_significantly_below_mc_fraction", 0.5))
    mean_tracklets = float(np.mean([int(block["n_tracklets"]) for block in largest_blocks]))
    reasons: list[str] = []
    if mean_tracklets < fraction * mc_ref:
        return {
            "decision": VERDICT_TRACKLET_LIMITED,
            "largest_complete_calibration_scale": largest,
            "mean_calibration_tracklets": mean_tracklets,
            "mc_transfer_100_event_tracklets": mc_ref,
            "geometry_write_allowed": False,
            "station_mode_blocked": True,
            "cdx_mode_blocked": True,
            "verdict_roles": [ROLE_CALIBRATION],
            "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
            "reasons": ["largest_scale_tracklets_significantly_below_mc_transfer"],
        }
    n100 = calibration["n100"]
    recovered_selected = next(
        (scale for scale in ("n1000", "n10000", "full") if _stable_selected(calibration[scale], config)),
        None,
    )
    recovered_complete = next(
        (scale for scale in ("n1000", "n10000", "full") if _stable_complete(calibration[scale], config)),
        None,
    )
    if _both_zero_selected(n100) and recovered_selected is not None:
        reasons.append("n100_selected_routes_zero")
        reasons.append(f"stable_selected_routes_at_{recovered_selected}")
        if recovered_complete is None:
            reasons.append("complete_four_station_routes_still_absent_on_expanded_calibration")
        else:
            reasons.append(f"stable_complete_routes_at_{recovered_complete}")
        return {
            "decision": VERDICT_STATISTICS_LIMITED,
            "largest_complete_calibration_scale": largest,
            "recovered_at_scale": recovered_selected,
            "complete_four_station_recovered_at_scale": recovered_complete,
            "mean_calibration_tracklets": mean_tracklets,
            "mc_transfer_100_event_tracklets": mc_ref,
            "geometry_write_allowed": False,
            "station_mode_blocked": True,
            "cdx_mode_blocked": True,
            "verdict_roles": [ROLE_CALIBRATION],
            "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
            "reasons": reasons,
        }
    if _many_all_pairs(largest_blocks, config) and _both_zero_selected(largest_blocks):
        reasons.append("largest_scale_has_many_all_pairs_candidates")
        reasons.append("largest_scale_selected_routes_zero")
        return {
            "decision": VERDICT_DOMAIN_SHIFT,
            "largest_complete_calibration_scale": largest,
            "mean_calibration_tracklets": mean_tracklets,
            "mc_transfer_100_event_tracklets": mc_ref,
            "geometry_write_allowed": False,
            "station_mode_blocked": True,
            "cdx_mode_blocked": True,
            "verdict_roles": [ROLE_CALIBRATION],
            "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
            "reasons": reasons,
        }
    return {
        "decision": VERDICT_PENDING,
        "largest_complete_calibration_scale": largest,
        "mean_calibration_tracklets": mean_tracklets,
        "mc_transfer_100_event_tracklets": mc_ref,
        "geometry_write_allowed": False,
        "station_mode_blocked": True,
        "cdx_mode_blocked": True,
        "verdict_roles": [ROLE_CALIBRATION],
        "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
        "reasons": ["calibration_pattern_not_yet_decisive"],
    }


def campaign_allows_station_mode(decision: str) -> bool:
    return False


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False
