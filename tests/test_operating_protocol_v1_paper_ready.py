from __future__ import annotations

from pathlib import Path

import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    RESIDUAL_DECREASE_LABEL,
)
from alignment.operating_protocol_v1_paper_ready import (
    DECISION_FLOW_TEXT,
    assert_no_forbidden_positive_claims,
    build_main_claims,
    campaign_allows_geometry_write,
    figure_captions,
    load_paper_config,
    render_all_documents,
    reviewer_risks,
    three_layer_claims,
    why_not_writing_geometry_is_a_result,
)


def _config():
    return load_paper_config(Path("configs/operating_protocol_v1_paper_ready_validation_v1.yaml"))


def test_config_is_read_only_and_forbids_rescue():
    config = _config()
    assert config["do_not_search_geometry_write_subset"] is True
    assert config["do_not_retrain_v2"] is True
    assert config["do_not_reestimate_alarm_thresholds"] is True
    assert config["geometry_write_allowed"] is False
    assert config["implied_cdx_is_not_a_measurement"] is True
    assert config["residual_decrease_label"] == RESIDUAL_DECREASE_LABEL
    assert campaign_allows_geometry_write("any") is False


def test_three_layers_and_decision_flow_stay_strict():
    layers = three_layer_claims()
    assert [item["layer"] for item in layers] == [1, 2, 3]
    assert layers[0]["status"] == "PASS"
    assert layers[1]["status"] == "REJECT_GEOMETRY_WRITE"
    assert layers[2]["status"] == "PASS"
    assert "statistics-limited" in layers[0]["proves"]
    assert "does not transfer" in layers[1]["proves"]
    assert "14977" in layers[2]["proves"]
    assert "insufficient statistics" in layers[2]["proves"]
    assert DECISION_FLOW_TEXT == (
        "MC transfer PASS → real-data association PASS → "
        "Station calibration REJECT → reduced calibration REJECT → "
        "residual/DQ monitoring PASS"
    )


def test_not_writing_geometry_is_a_result_not_an_unfinished_state():
    why = why_not_writing_geometry_is_a_result()
    assert why["geometry_write_allowed"] is False
    assert "not an unfinished state" in why["statement"]
    assert "an overly conservative alarm threshold" in why["not_caused_by"]
    assert "failure to search a quieter event subset" in why["not_caused_by"]


def test_claims_and_documents_forbid_residual_closure_and_cdx_measurement():
    frozen = {
        "final_report": {
            "frozen_v2_checkpoint_sha256": FROZEN_V2_CHECKPOINT_SHA256,
            "geometry_write_allowed": False,
        },
        "unlock": {"currently_met": False},
    }
    claims = build_main_claims(frozen)
    assert claims["implied_cdx_is_not_a_measurement"] is True
    assert claims["residual_reduction_is_not_alignment_success"] is True
    assert claims["unlock_evaluation"]["currently_met"] is False
    unsupported_ids = {item["id"] for item in claims["unsupported_claims"]}
    assert "no_cdx_measurement" in unsupported_ids
    assert "no_residual_as_closure" in unsupported_ids
    assert "no_subset_rescue" in unsupported_ids
    documents = render_all_documents(claims)
    blob = "\n".join(documents.values())
    assert_no_forbidden_positive_claims(blob, where="rendered documents")
    assert "dq observable" in blob.lower()
    assert "1.5–1.7" in documents["paper_results_summary.md"] or "1.5-1.7" in documents["paper_results_summary.md"]
    assert any("14977" in item["en"] for item in figure_captions())
    assert any(item["id"] == "not_writing_looks_unfinished" for item in reviewer_risks())
    with pytest.raises(ValueError, match="forbidden positive claim"):
        assert_no_forbidden_positive_claims("this is alignment closure", where="bad")
