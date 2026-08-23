"""Residual-blind occupancy preflight and frozen window selection.

Window selection may use cluster and SegmentFit *counts* only.  Residuals,
V2 scores, chi2, and alignment parameters are forbidden both as TTree
branches and as selection inputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.real_data_operating_protocol import (
    BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY,
    BLOCK_ROLES,
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    ROLE_HOLDOUT,
)

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-occupancy-window-rule"
FORBIDDEN_BRANCH_SUBSTRINGS = (
    "residual",
    "v2",
    "score",
    "chi2",
    "chi_2",
    "alignment",
    "parameter",
    "dx_mm",
    "dy_mm",
    "ry_mrad",
)
DEFAULT_RULE_PATH = Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml")
XAOD_NAME = "Faser-Physics-{run:06d}-{segment}-r0022-xAOD.root"
REC_ROOT = Path("/eos/experiment/faser/rec/2024/r0022")
BLIND_ROLES = {
    14973: ROLE_CALIBRATION,
    14974: ROLE_CALIBRATION,
    14975: ROLE_HOLDOUT,
    14976: ROLE_HOLDOUT,
    14977: ROLE_HELD_OUT_DQ,
}


def load_window_rule(path: Path | None = None) -> dict[str, Any]:
    rule_path = Path(path) if path is not None else DEFAULT_RULE_PATH
    with rule_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"window rule must be a mapping: {rule_path}")
    raw = payload.get("operating_protocol_v1_real_data_occupancy_window_rule", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("occupancy window rule root is missing")
    rule = dict(raw)
    if rule.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected occupancy window-rule schema: {rule.get('schema_version')}")
    if rule.get("residual_blind") is not True:
        raise ValueError("occupancy window rule must be residual_blind")
    if rule.get("alignment_perturbation") is not False:
        raise ValueError("occupancy window rule forbids alignment perturbation")
    if rule.get("v2_scoring") is not False:
        raise ValueError("occupancy window rule forbids V2 scoring")
    forbidden = {str(item) for item in rule.get("selection_inputs_forbidden", ())}
    for token in ("residual", "v2_score", "fit_chi2", "alignment_parameter"):
        if token not in forbidden:
            raise ValueError(f"window rule must forbid {token}")
    roles = rule.get("blind_roles")
    if not isinstance(roles, Mapping):
        raise ValueError("window rule must freeze blind_roles")
    parsed_roles = {int(run): str(role) for run, role in roles.items()}
    if parsed_roles != BLIND_ROLES:
        raise ValueError(f"blind roles must remain {BLIND_ROLES}, got {parsed_roles}")
    for role in parsed_roles.values():
        if role not in BLOCK_ROLES:
            raise ValueError(f"unknown frozen role {role!r}")
    return rule


def assert_occupancy_tree_is_residual_blind(branch_names: Sequence[str]) -> None:
    for name in branch_names:
        lowered = str(name).lower()
        for token in FORBIDDEN_BRANCH_SUBSTRINGS:
            if token in lowered:
                raise ValueError(
                    f"occupancy tree branch {name!r} is not residual-blind ({token})"
                )


def window_minima(rule: Mapping[str, Any]) -> dict[str, int]:
    raw = rule.get("minima")
    if not isinstance(raw, Mapping):
        raise ValueError("window rule minima are required")
    required = (
        "n_events_with_clusters",
        "n_events_with_tracklets",
        "total_tracklets",
        "n_events_with_four_station_tracklets",
    )
    minima = {key: int(raw[key]) for key in required}
    if any(value < 1 for value in minima.values()):
        raise ValueError("pre-registered occupancy minima must be positive")
    return minima


def occupancy_window_summary(table: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n_clusters = np.asarray(table["n_sct_clusters"])
    n_tracklets = np.asarray(table["n_refit_tracklets"])
    four = np.asarray(table["four_station_refit_multiplicity"])
    n_events = int(n_clusters.shape[0])
    station_clusters = np.asarray(table.get("n_clusters_station", np.zeros((n_events, 4))))
    station_tracklets = np.asarray(table.get("n_refit_station", np.zeros((n_events, 4))))
    if station_clusters.ndim == 1:
        station_clusters = station_clusters.reshape(n_events, -1)
    if station_tracklets.ndim == 1:
        station_tracklets = station_tracklets.reshape(n_events, -1)
    summary = {
        "n_events": n_events,
        "n_events_with_clusters": int(np.count_nonzero(n_clusters > 0)),
        "n_events_with_tracklets": int(np.count_nonzero(n_tracklets > 0)),
        "total_clusters": int(n_clusters.sum()),
        "total_tracklets": int(n_tracklets.sum()),
        "n_events_with_four_station_tracklets": int(np.count_nonzero(four > 0)),
        "total_four_station_multiplicity": int(four.sum()),
        "n_clusters_station": [int(station_clusters[:, index].sum()) for index in range(min(4, station_clusters.shape[1]))],
        "n_tracklets_station": [int(station_tracklets[:, index].sum()) for index in range(min(4, station_tracklets.shape[1]))],
    }
    if "stable_beams" in table:
        summary["n_stable_beam_events"] = int(np.count_nonzero(np.asarray(table["stable_beams"]) > 0))
    if "run" in table:
        runs = np.unique(np.asarray(table["run"]))
        summary["run_ids"] = [int(value) for value in runs.tolist()]
    if "lumi_block" in table:
        lbs = np.unique(np.asarray(table["lumi_block"]))
        summary["lumi_blocks"] = [int(value) for value in lbs.tolist()]
        summary["n_lumi_blocks"] = int(lbs.size)
    return summary


def window_meets_minima(summary: Mapping[str, Any], minima: Mapping[str, int]) -> bool:
    return (
        int(summary.get("n_events", 0)) >= 1
        and int(summary.get("n_events_with_clusters", 0)) >= int(minima["n_events_with_clusters"])
        and int(summary.get("n_events_with_tracklets", 0)) >= int(minima["n_events_with_tracklets"])
        and int(summary.get("total_tracklets", 0)) >= int(minima["total_tracklets"])
        and int(summary.get("n_events_with_four_station_tracklets", 0))
        >= int(minima["n_events_with_four_station_tracklets"])
    )


def contiguous_windows(
    table: Mapping[str, np.ndarray],
    *,
    nevents: int,
    skip_start: int,
    skip_stride: int,
    job_skip_events: int = 0,
) -> list[dict[str, Any]]:
    if nevents < 1 or skip_stride < 1:
        raise ValueError("window nevents and skip stride must be positive")
    n_rows = int(np.asarray(table["n_sct_clusters"]).shape[0])
    windows: list[dict[str, Any]] = []
    skip = skip_start
    while skip + nevents <= n_rows:
        sl = slice(skip, skip + nevents)
        chunk = {key: np.asarray(value)[sl] for key, value in table.items()}
        summary = occupancy_window_summary(chunk)
        windows.append(
            {
                "skip_events": int(job_skip_events + skip),
                "nevents": int(nevents),
                **summary,
            }
        )
        skip += skip_stride
    return windows


def first_passing_window(
    windows: Sequence[Mapping[str, Any]],
    minima: Mapping[str, int],
) -> dict[str, Any] | None:
    for window in windows:
        if window_meets_minima(window, minima):
            return dict(window)
    return None


def format_segment(segment: int | str) -> str:
    text = str(segment)
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"invalid r0022 segment {segment!r}")
    return f"{int(text):05d}"


def next_segment(segment: int | str) -> str:
    return format_segment(int(format_segment(segment)) + 1)


def xaod_path_for_segment(run: int, segment: int | str, *, rec_root: Path | None = None) -> Path:
    root = Path(rec_root) if rec_root is not None else REC_ROOT
    name = XAOD_NAME.format(run=int(run), segment=format_segment(segment))
    return root / f"{int(run):06d}" / name


def list_r0022_segments(run: int, *, rec_root: Path | None = None) -> list[str]:
    root = Path(rec_root) if rec_root is not None else REC_ROOT
    directory = root / f"{int(run):06d}"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    prefix = f"Faser-Physics-{int(run):06d}-"
    suffix = "-r0022-xAOD.root"
    segments: list[str] = []
    for path in directory.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : -len(suffix)]
        if re.fullmatch(r"\d{5}", middle):
            segments.append(middle)
    return sorted(segments)


def select_window_for_segment(
    *,
    run: int,
    segment: str,
    windows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    rec_root: Path | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    minima = window_minima(rule)
    accepted = first_passing_window(windows, minima)
    if role is None:
        if int(run) not in BLIND_ROLES:
            raise ValueError(f"run {run} has no frozen blind role; pass role= explicitly")
        role = BLIND_ROLES[int(run)]
    payload: dict[str, Any] = {
        "run": int(run),
        "role": role,
        "segment": format_segment(segment),
        "window_accepted": accepted is not None,
        "selection_order": "first_contiguous_window_in_skip_events_order",
        "residual_blind": True,
        "minima": minima,
        "n_windows_scanned": len(windows),
    }
    if accepted is not None:
        payload["skip_events"] = int(accepted["skip_events"])
        payload["nevents"] = int(accepted["nevents"])
        payload["occupancy_summary"] = {
            key: accepted[key]
            for key in (
                "n_events",
                "n_events_with_clusters",
                "n_events_with_tracklets",
                "total_clusters",
                "total_tracklets",
                "n_events_with_four_station_tracklets",
                "n_clusters_station",
                "n_tracklets_station",
            )
            if key in accepted
        }
        payload["status"] = "frozen_window"
        return payload
    try:
        available = list_r0022_segments(run, rec_root=rec_root)
    except FileNotFoundError:
        available = []
    nxt = next_segment(segment)
    payload["skip_events"] = None
    payload["nevents"] = int(rule["window"]["nevents"])
    payload["occupancy_summary"] = None
    payload["status"] = BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
    payload["segment_exhausted"] = True
    later = [item for item in available if int(item) > int(format_segment(segment))]
    if nxt in available:
        payload["status"] = "need_next_segment"
        payload["next_segment"] = nxt
    elif later:
        payload["status"] = "need_next_segment"
        payload["next_segment"] = later[0]
    return payload
