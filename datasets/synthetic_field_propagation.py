"""Build field-aware candidate records for synthetic multi-track overlays.

The source prediction is an exact mode-0 FaserActsExtrapolationTool record
exported from the same refitted geometry payload.  Synthetic overlay changes
event membership only, so that source prediction can be compared with every
target tracklet on the same fixed station reference plane.  A record is copied
only when its exported target z agrees with the synthetic target z.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import uproot

from .propagation_loader import (
    PROPAGATION_SCHEMA_VERSION,
    PROPAGATION_TREE_NAME,
    PropagationRecords,
    load_propagation_records,
)
from .root_loader import EventTracklets, load_events
from .schema import covariance_columns


@dataclass(frozen=True)
class SyntheticFieldCandidateSummary:
    """Accounting for one synthetic field-candidate export."""

    synthetic_tracklets: str
    source_propagations: str
    destination: str
    q_over_p_mode: int
    synthetic_events: int
    source_tracklets: int
    candidate_records: int
    missing_source_prediction: int
    target_z_mismatch: int
    records_by_station_pair: dict[str, int]


def _origin_key(event: EventTracklets, row: int) -> tuple[int, int, int]:
    if (
        event.origin_run_id is None
        or event.origin_event_id is None
        or event.origin_tracklet_id is None
    ):
        raise ValueError(
            "synthetic tracklets need origin_run_id, origin_event_id, and "
            "origin_tracklet_id provenance fields"
        )
    return (
        int(event.origin_run_id[row]),
        int(event.origin_event_id[row]),
        int(event.origin_tracklet_id[row]),
    )


def _source_prediction_index(
    records: PropagationRecords,
    q_over_p_mode: int,
) -> dict[tuple[int, int, int, int], int]:
    index: dict[tuple[int, int, int, int], int] = {}
    for row in range(records.size):
        mode = int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0
        if mode != q_over_p_mode:
            continue
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_station_id[row]),
        )
        if key in index:
            raise ValueError(
                "source propagation is not unique for origin source and target station: "
                f"{key}"
            )
        index[key] = row
    if not index:
        raise ValueError(f"no q/p mode {q_over_p_mode} source propagation records are available")
    return index


def _empty_columns() -> dict[str, list[int | float | bool]]:
    return {
        "run_id": [],
        "event_id": [],
        "source_tracklet_id": [],
        "target_tracklet_id": [],
        "source_station_id": [],
        "target_station_id": [],
        # Deliberately do not expose synthetic truth labels through the
        # candidate-propagation input used by matching baselines.
        "truth_particle_id": [],
        "target_z_mm": [],
        "pred_x_mm": [],
        "pred_y_mm": [],
        "pred_tx": [],
        "pred_ty": [],
        "success": [],
        "has_covariance": [],
        "q_over_p_mode": [],
        "source_q_over_p_per_mev": [],
        **{f"pred_{field}": [] for field in covariance_columns(np.empty((0, 4, 4))).keys()},
    }


def _root_columns(
    columns: dict[str, list[int | float | bool]],
) -> dict[str, np.ndarray]:
    dtypes: dict[str, type[np.generic]] = {
        "run_id": np.int64,
        "event_id": np.int64,
        "source_tracklet_id": np.int32,
        "target_tracklet_id": np.int32,
        "source_station_id": np.int16,
        "target_station_id": np.int16,
        "truth_particle_id": np.int64,
        "target_z_mm": np.float64,
        "pred_x_mm": np.float64,
        "pred_y_mm": np.float64,
        "pred_tx": np.float64,
        "pred_ty": np.float64,
        "success": np.bool_,
        "has_covariance": np.bool_,
        "q_over_p_mode": np.int8,
        "source_q_over_p_per_mev": np.float64,
    }
    for field in covariance_columns(np.empty((0, 4, 4))).keys():
        dtypes[f"pred_{field}"] = np.float64
    return {name: np.asarray(values, dtype=dtypes[name]) for name, values in columns.items()}


def write_synthetic_field_candidate_root(
    synthetic_tracklets: str | Path,
    source_propagations: str | Path,
    destination: str | Path,
    q_over_p_mode: int = 0,
    target_z_tolerance_mm: float = 1.0e-6,
    require_mc_labels: bool = True,
) -> SyntheticFieldCandidateSummary:
    """Fan out exact source Acts predictions to compatible synthetic targets.

    This is valid for the current mode-0 export because the station reference
    z is common across events.  The explicit z check makes a geometry or
    exporter convention change fail closed rather than silently approximating
    a propagation to a different target plane.
    """
    if q_over_p_mode not in (0, 1, 2, 3):
        raise ValueError("synthetic physical candidate export requires a known q_over_p_mode")
    if not np.isfinite(target_z_tolerance_mm) or target_z_tolerance_mm < 0.0:
        raise ValueError("target_z_tolerance_mm must be finite and non-negative")
    synthetic_path = Path(synthetic_tracklets).expanduser().resolve()
    source_path = Path(source_propagations).expanduser().resolve()
    output_path = Path(destination).expanduser().resolve()
    events = load_events(synthetic_path, require_mc_labels=require_mc_labels)
    if require_mc_labels is False and any(event.truth_particle_id is not None for event in events):
        raise ValueError("real-data field-candidate export received MC truth labels")
    records = load_propagation_records(source_path)
    prediction_index = _source_prediction_index(records, q_over_p_mode=q_over_p_mode)
    columns = _empty_columns()
    records_by_station_pair: Counter[str] = Counter()
    missing_source_prediction = 0
    target_z_mismatch = 0
    source_tracklets = 0

    for event in events:
        for source_row in range(event.size):
            source_tracklets += 1
            source_station = int(event.station_id[source_row])
            origin = _origin_key(event, source_row)
            for target_row in range(event.size):
                target_station = int(event.station_id[target_row])
                if source_station >= target_station:
                    continue
                record_row = prediction_index.get((*origin, target_station))
                if record_row is None:
                    missing_source_prediction += 1
                    continue
                if not np.isclose(
                    float(records.target_z_mm[record_row]),
                    float(event.z_mm[target_row]),
                    rtol=0.0,
                    atol=target_z_tolerance_mm,
                ):
                    target_z_mismatch += 1
                    continue
                source_prediction = records.prediction[record_row]
                source_covariance = records.covariance[record_row]
                columns["run_id"].append(event.run_id)
                columns["event_id"].append(event.event_id)
                columns["source_tracklet_id"].append(int(event.tracklet_id[source_row]))
                columns["target_tracklet_id"].append(int(event.tracklet_id[target_row]))
                columns["source_station_id"].append(source_station)
                columns["target_station_id"].append(target_station)
                columns["truth_particle_id"].append(-1)
                columns["target_z_mm"].append(float(event.z_mm[target_row]))
                columns["pred_x_mm"].append(float(source_prediction[0]))
                columns["pred_y_mm"].append(float(source_prediction[1]))
                columns["pred_tx"].append(float(source_prediction[2]))
                columns["pred_ty"].append(float(source_prediction[3]))
                flattened_covariance = covariance_columns(source_covariance[np.newaxis, ...])
                for field, values in flattened_covariance.items():
                    columns[f"pred_{field}"].append(float(values[0]))
                columns["success"].append(bool(records.success[record_row]))
                columns["has_covariance"].append(bool(records.has_covariance[record_row]))
                columns["q_over_p_mode"].append(q_over_p_mode)
                if records.source_q_over_p_per_mev is None:
                    columns["source_q_over_p_per_mev"].append(float("nan"))
                else:
                    columns["source_q_over_p_per_mev"].append(
                        float(records.source_q_over_p_per_mev[record_row])
                    )
                records_by_station_pair[f"{source_station}->{target_station}"] += 1

    root_columns = _root_columns(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = SyntheticFieldCandidateSummary(
        synthetic_tracklets=str(synthetic_path),
        source_propagations=str(source_path),
        destination=str(output_path),
        q_over_p_mode=q_over_p_mode,
        synthetic_events=len(events),
        source_tracklets=source_tracklets,
        candidate_records=len(columns["run_id"]),
        missing_source_prediction=missing_source_prediction,
        target_z_mismatch=target_z_mismatch,
        records_by_station_pair=dict(sorted(records_by_station_pair.items())),
    )
    with uproot.recreate(output_path) as root_file:
        root_file[PROPAGATION_TREE_NAME] = root_columns
        root_file["metadata"] = {
            "schema_version": np.asarray([PROPAGATION_SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "generator": np.asarray(
                ["datasets.synthetic_field_propagation.write_synthetic_field_candidate_root"]
            ),
            "source_propagations": np.asarray([str(source_path)]),
            "q_over_p_mode": np.asarray([q_over_p_mode], dtype=np.int8),
            "target_z_tolerance_mm": np.asarray([target_z_tolerance_mm], dtype=np.float64),
            "prediction_semantics": np.asarray(
                [
                    "Exact mode-0 FaserActs source prediction copied from the same "
                    "physical payload only after target-z compatibility validation."
                ]
            ),
            "summary": np.asarray([json.dumps(asdict(summary), sort_keys=True)]),
        }
    return summary
