from pathlib import Path

import pytest

from datasets.physical_curriculum import CurriculumSample
from scripts.audit_frozen_v1_ift_ry_routes import _condition_fields, _parse_route_config


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="source",
        source_ids=("source",),
        split="validation",
        payload_id="ry_p40_joint",
        magnitude_mm=40.0,
        direction_trial="joint_dxdy_a",
        injected_offsets_xy_mm={},
        source_event_uids=("event",),
        physical_event_uids=("event",),
        physical_tracklets=Path("/tmp/tracklets.root"),
        physical_propagations=Path("/tmp/propagations.root"),
        physical_payload_manifest=Path("/tmp/payload.json"),
        synthetic_tracklets=Path("/tmp/synthetic.root"),
        field_candidates=Path("/tmp/candidates.root"),
        condition_axis="ift_ry_mrad",
        condition_value=40.0,
        condition_magnitude=40.0,
        injected_station_transforms={"0": [1.0, -1.0, 0.0, 0.0, 0.04, 0.0]},
    )


def test_frozen_v1_rotation_audit_preserves_signed_condition_and_joint_translation():
    fields = _condition_fields(_sample())

    assert fields == {
        "condition_axis": "ift_ry_mrad",
        "condition_value": 40.0,
        "condition_magnitude": 40.0,
        "direction_trial": "joint_dxdy_a",
        "joint_dx_mm": 1.0,
        "joint_dy_mm": -1.0,
        "pure_ry_trial": False,
    }


def test_frozen_v1_rotation_audit_requires_all_adjacent_frozen_thresholds():
    with pytest.raises(KeyError):
        _parse_route_config(
            {
                "method": "adjacent_contiguous_unit_capacity_set_packing",
                "thresholds": {"0->1": 0.1},
                "unmatched_penalty": 0.0,
            },
            100,
        )
