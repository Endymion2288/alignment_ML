"""Identity provenance samples for real FASER data.

Real data has no MC truth overlay.  The physical refit already is the
sample: copy states, stamp origin_* to the reconstructed run/event/tracklet
identity, and refuse any MC label columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import uproot

from datasets.root_loader import load_events
from datasets.schema import CANONICAL_TREE_NAME, SCHEMA_VERSION, covariance_columns
from datasets.synthetic_overlay import Q_OVER_P_FIELDS, SYNTHETIC_ROLE_TRUE


def write_real_data_identity_tracklets(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write origin-stamped tracklets without MC labels or synthetic overlay."""
    events = load_events(source, require_mc_labels=False)
    if any(event.truth_particle_id is not None for event in events):
        raise ValueError("real-data identity samples must not carry MC truth labels")
    columns: dict[str, list[object]] = {
        "run_id": [],
        "event_id": [],
        "station_id": [],
        "tracklet_id": [],
        "z_mm": [],
        "chi2": [],
        "ndof": [],
        "n_hit": [],
        "hit_pattern": [],
        "origin_run_id": [],
        "origin_event_id": [],
        "origin_tracklet_id": [],
        "synthetic_role": [],
        "synthetic_hard_anchor_station": [],
        "synthetic_hard_anchor_chi2": [],
    }
    q_over_p_fields = []
    if events and events[0].q_over_p_per_mev is not None:
        q_over_p_fields = list(Q_OVER_P_FIELDS)
        for field in q_over_p_fields:
            columns[field] = []
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for event in events:
        for row in range(int(event.size)):
            columns["run_id"].append(int(event.run_id))
            columns["event_id"].append(int(event.event_id))
            columns["station_id"].append(int(event.station_id[row]))
            columns["tracklet_id"].append(int(event.tracklet_id[row]))
            columns["z_mm"].append(float(event.z_mm[row]))
            columns["chi2"].append(float(event.chi2[row]))
            columns["ndof"].append(float(event.ndof[row]))
            columns["n_hit"].append(int(event.n_hit[row]))
            columns["hit_pattern"].append(int(event.hit_pattern[row]))
            columns["origin_run_id"].append(int(event.run_id))
            columns["origin_event_id"].append(int(event.event_id))
            columns["origin_tracklet_id"].append(int(event.tracklet_id[row]))
            columns["synthetic_role"].append(int(SYNTHETIC_ROLE_TRUE))
            columns["synthetic_hard_anchor_station"].append(-1)
            columns["synthetic_hard_anchor_chi2"].append(float("nan"))
            states.append(np.asarray(event.state[row], dtype=np.float64))
            covariances.append(np.asarray(event.covariance[row], dtype=np.float64))
            if "q_over_p_per_mev" in columns and event.q_over_p_per_mev is not None:
                columns["q_over_p_per_mev"].append(float(event.q_over_p_per_mev[row]))
                columns["q_over_p_from_momentum_per_mev"].append(
                    float(event.q_over_p_from_momentum_per_mev[row])
                    if event.q_over_p_from_momentum_per_mev is not None
                    else float("nan")
                )
                columns["q_over_p_variance_per_mev2"].append(
                    float(event.q_over_p_variance_per_mev2[row])
                    if event.q_over_p_variance_per_mev2 is not None
                    else float("nan")
                )
                columns["has_q_over_p_covariance"].append(
                    bool(event.has_q_over_p_covariance[row])
                    if event.has_q_over_p_covariance is not None
                    else False
                )
    root_columns: dict[str, np.ndarray] = {
        "run_id": np.asarray(columns["run_id"], dtype=np.int64),
        "event_id": np.asarray(columns["event_id"], dtype=np.int64),
        "station_id": np.asarray(columns["station_id"], dtype=np.int8),
        "tracklet_id": np.asarray(columns["tracklet_id"], dtype=np.int32),
        "z_mm": np.asarray(columns["z_mm"], dtype=np.float64),
        "chi2": np.asarray(columns["chi2"], dtype=np.float64),
        "ndof": np.asarray(columns["ndof"], dtype=np.float64),
        "n_hit": np.asarray(columns["n_hit"], dtype=np.int16),
        "hit_pattern": np.asarray(columns["hit_pattern"], dtype=np.uint64),
        "origin_run_id": np.asarray(columns["origin_run_id"], dtype=np.int64),
        "origin_event_id": np.asarray(columns["origin_event_id"], dtype=np.int64),
        "origin_tracklet_id": np.asarray(columns["origin_tracklet_id"], dtype=np.int32),
        "synthetic_role": np.asarray(columns["synthetic_role"], dtype=np.int8),
        "synthetic_hard_anchor_station": np.asarray(
            columns["synthetic_hard_anchor_station"], dtype=np.int8
        ),
        "synthetic_hard_anchor_chi2": np.asarray(
            columns["synthetic_hard_anchor_chi2"], dtype=np.float64
        ),
    }
    if states:
        state_array = np.asarray(states, dtype=np.float64)
        covariance_array = np.asarray(covariances, dtype=np.float64)
    else:
        state_array = np.empty((0, 4), dtype=np.float64)
        covariance_array = np.empty((0, 4, 4), dtype=np.float64)
    root_columns.update(
        {
            "x_mm": state_array[:, 0],
            "y_mm": state_array[:, 1],
            "tx": state_array[:, 2],
            "ty": state_array[:, 3],
        }
    )
    root_columns.update(covariance_columns(covariance_array))
    for field in q_over_p_fields:
        dtype = np.bool_ if field == "has_q_over_p_covariance" else np.float64
        root_columns[field] = np.asarray(columns[field], dtype=dtype)
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(destination_path) as output_file:
        output_file[CANONICAL_TREE_NAME] = root_columns
        output_file["metadata"] = {
            "schema_version": np.asarray([SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "source_file": np.asarray([str(Path(source).expanduser().resolve())]),
            "converter": np.asarray(["datasets.real_data_identity"]),
            "real_data": np.asarray([True]),
            "mc_labels": np.asarray([False]),
            "overlay": np.asarray(["identity"]),
            "optional_fields": np.asarray([json.dumps(sorted(q_over_p_fields))]),
        }
    return {
        "source": str(Path(source).expanduser().resolve()),
        "destination": str(destination_path),
        "events": int(len(events)),
        "tracklets": int(state_array.shape[0]),
        "has_mc_labels": False,
        "overlay": "identity",
    }
