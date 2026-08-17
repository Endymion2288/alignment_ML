from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_expanded_validation_route_controls import _assert_validation_contract


def test_plot_summary_accepts_legacy_validation_control_contract(tmp_path: Path):
    summary = tmp_path / "validation_route_magnitude_summary.csv"
    summary.write_text("magnitude_mm\n0.0\n", encoding="utf-8")
    (tmp_path / "validation_control_contract.json").write_text(
        json.dumps(
            {
                "loaded_event_splits": ["validation"],
                "test_events_loaded": False,
                "test_artifacts_opened": False,
            }
        ),
        encoding="utf-8",
    )

    evidence = _assert_validation_contract(summary)

    assert evidence["contract_present"] is True
    assert evidence["test_events_loaded"] is False
    assert str(evidence["contract_path"]).endswith("validation_control_contract.json")


def test_plot_summary_rejects_contract_that_opened_test_events(tmp_path: Path):
    summary = tmp_path / "validation_route_magnitude_summary.csv"
    summary.write_text("magnitude_mm\n0.0\n", encoding="utf-8")
    (tmp_path / "validation_run_contract.json").write_text(
        json.dumps({"test_events_loaded": True, "test_artifacts_opened": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sealed from test events"):
        _assert_validation_contract(summary)
