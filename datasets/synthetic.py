"""Small deterministic canonical dataset for loader and baseline validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

from .schema import CANONICAL_TREE_NAME, SCHEMA_VERSION, covariance_columns


def make_synthetic_tracklet_root(path: str | Path, seed: int = 7) -> Path:
    """Write a two-station sample with known associations and fake tracklets."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    states: list[list[float]] = []
    covariances: list[np.ndarray] = []
    rows: dict[str, list[float | int]] = {
        "run_id": [],
        "event_id": [],
        "station_id": [],
        "tracklet_id": [],
        "z_mm": [],
        "chi2": [],
        "ndof": [],
        "n_hit": [],
        "hit_pattern": [],
        "truth_particle_id": [],
        "truth_pdg": [],
        "truth_match_fraction": [],
    }

    covariance = np.diag([0.05**2, 0.05**2, 2.0e-5**2, 2.0e-5**2])
    for event_id in range(4):
        truth_id = 1000 + event_id
        source_state = np.array(
            [5.0 * event_id, -2.0 * event_id, 1.0e-3, -0.6e-3],
            dtype=np.float64,
        )
        target_state = source_state.copy()
        target_state[0] += 1000.0 * source_state[2]
        target_state[1] += 1000.0 * source_state[3]
        target_state += rng.normal(0.0, [0.01, 0.01, 5.0e-6, 5.0e-6])

        payload = (
            (0, 0.0, source_state, truth_id),
            (1, 1000.0, target_state, truth_id),
            (1, 1000.0, np.array([60.0, -55.0, 0.0, 0.0]), -1),
        )
        for tracklet_id, (station_id, z_mm, state, label) in enumerate(payload):
            rows["run_id"].append(1)
            rows["event_id"].append(event_id)
            rows["station_id"].append(station_id)
            rows["tracklet_id"].append(tracklet_id)
            rows["z_mm"].append(z_mm)
            rows["chi2"].append(1.0)
            rows["ndof"].append(1.0)
            rows["n_hit"].append(3)
            rows["hit_pattern"].append(0b111)
            rows["truth_particle_id"].append(label)
            rows["truth_pdg"].append(11 if label >= 0 else 0)
            rows["truth_match_fraction"].append(1.0 if label >= 0 else 0.0)
            states.append(state.tolist())
            covariances.append(covariance)

    state_array = np.asarray(states, dtype=np.float64)
    root_columns: dict[str, np.ndarray] = {
        "run_id": np.asarray(rows["run_id"], dtype=np.int64),
        "event_id": np.asarray(rows["event_id"], dtype=np.int64),
        "station_id": np.asarray(rows["station_id"], dtype=np.int8),
        "tracklet_id": np.asarray(rows["tracklet_id"], dtype=np.int32),
        "x_mm": state_array[:, 0],
        "y_mm": state_array[:, 1],
        "z_mm": np.asarray(rows["z_mm"], dtype=np.float64),
        "tx": state_array[:, 2],
        "ty": state_array[:, 3],
        "chi2": np.asarray(rows["chi2"], dtype=np.float64),
        "ndof": np.asarray(rows["ndof"], dtype=np.float64),
        "n_hit": np.asarray(rows["n_hit"], dtype=np.int16),
        "hit_pattern": np.asarray(rows["hit_pattern"], dtype=np.uint64),
        "truth_particle_id": np.asarray(rows["truth_particle_id"], dtype=np.int64),
        "truth_pdg": np.asarray(rows["truth_pdg"], dtype=np.int32),
        "truth_match_fraction": np.asarray(
            rows["truth_match_fraction"], dtype=np.float64
        ),
    }
    root_columns.update(covariance_columns(np.asarray(covariances)))
    with uproot.recreate(output) as root_file:
        root_file[CANONICAL_TREE_NAME] = root_columns
        root_file["metadata"] = {
            "schema_version": np.asarray([SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "generator": np.asarray(["datasets.synthetic.make_synthetic_tracklet_root"]),
        }
    return output

