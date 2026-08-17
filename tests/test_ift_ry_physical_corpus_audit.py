from __future__ import annotations

import pytest

from scripts.audit_ift_ry_physical_corpus import _candidate_summary, _load_manifest


def _point(source_id: str, *, retained: int, total: int) -> dict[str, object]:
    row: dict[str, object] = {
        "source_id": source_id,
        "split": "validation",
        "payload_id": "ry_p40",
        "condition_axis": "ift_ry_mrad",
        "condition_value": 40.0,
        "condition_magnitude": 40.0,
        "direction_trial": "positive",
        "joint_dx_mm": 0.0,
        "joint_dy_mm": 0.0,
        "complete_truth_chains": total,
        "candidate_retained_complete_truth_chains": retained,
        "acts_surface_error_count": 3,
        "acts_layer_overlap_error_count": 1,
        "condition_chain_complete": True,
    }
    for pair in ("0->1", "1->2", "2->3"):
        row[f"{pair}_truth_pairs"] = total
        row[f"{pair}_retained_truth_pairs"] = retained
        row[f"{pair}_physical_candidate_edges"] = total + 2
    return row


def test_candidate_summary_uses_global_truth_denominators():
    summary = _candidate_summary(
        [_point("small", retained=1, total=1), _point("large", retained=1, total=9)]
    )

    assert len(summary) == 1
    assert summary[0]["complete_truth_chains"] == 10
    assert summary[0]["candidate_retained_complete_truth_chains"] == 2
    assert summary[0]["candidate_complete_truth_chain_recall"] == 0.2
    assert summary[0]["0->1_candidate_truth_edge_recall"] == 0.2


def test_corpus_audit_requires_explicit_test_exclusion(tmp_path):
    path = tmp_path / "physical.json"
    path.write_text(
        '{"physical_geometry_repropagation": true, "q_over_p_mode": 0, "forbidden_splits": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbid test"):
        _load_manifest(path)
