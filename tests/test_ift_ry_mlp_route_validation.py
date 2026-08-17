from pathlib import Path

import pytest

from datasets.physical_curriculum import CurriculumSample
from scripts.evaluate_pairwise_mlp_route_validation import _validate_contract


def _sample(*, condition_axis: str, magnitude: float) -> CurriculumSample:
    return CurriculumSample(
        source_id="source",
        source_ids=("source",),
        split="validation",
        payload_id=f"payload_{magnitude:g}",
        magnitude_mm=magnitude,
        direction_trial="trial",
        injected_offsets_xy_mm={},
        source_event_uids=("event",),
        physical_event_uids=("event",),
        physical_tracklets=Path("/tmp/tracklets.root"),
        physical_propagations=Path("/tmp/propagations.root"),
        physical_payload_manifest=Path("/tmp/payload.json"),
        synthetic_tracklets=Path("/tmp/synthetic.root"),
        field_candidates=Path("/tmp/candidates.root"),
        condition_axis=condition_axis,
        condition_value=magnitude,
        condition_magnitude=magnitude,
    )


def test_mlp_route_control_accepts_explicit_ift_ry_condition_contract():
    contract = {
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "condition_axis": "ift_ry_mrad",
        "curriculum_condition_magnitudes": [0.0, 10.0, 25.0, 40.0, 60.0],
    }
    manifest = {"physical_geometry_repropagation": True, "q_over_p_mode": 0}
    samples = [
        _sample(condition_axis="ift_ry_mrad", magnitude=value)
        for value in (0.0, 10.0, 25.0, 40.0, 60.0)
    ]

    _validate_contract(contract, manifest, samples)


def test_mlp_route_control_rejects_mismatched_condition_axis():
    contract = {
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "condition_axis": "ift_ry_mrad",
        "curriculum_condition_magnitudes": [0.0],
    }
    manifest = {"physical_geometry_repropagation": True, "q_over_p_mode": 0}

    with pytest.raises(ValueError, match="condition axis"):
        _validate_contract(
            contract,
            manifest,
            [_sample(condition_axis="translation_xy_mm", magnitude=0.0)],
        )
