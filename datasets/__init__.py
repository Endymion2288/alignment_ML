"""Dataset schemas and ROOT readers for FASER tracklet studies."""

from .root_loader import EventTracklets, load_events
from .propagation_loader import PropagationRecords, load_propagation_records
from .schema import DatasetSchemaError, validate_tracklet_tree
from .synthetic_overlay import (
    SyntheticOverlaySummary,
    select_complete_truth_tracks,
    write_synthetic_multitrack_root,
)

__all__ = [
    "DatasetSchemaError",
    "EventTracklets",
    "PropagationRecords",
    "SyntheticOverlaySummary",
    "load_events",
    "load_propagation_records",
    "select_complete_truth_tracks",
    "validate_tracklet_tree",
    "write_synthetic_multitrack_root",
]
