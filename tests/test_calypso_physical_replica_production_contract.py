"""Design-only physical replica contract.  Does not produce or qualify."""

from __future__ import annotations

import json
from pathlib import Path

from alignment.wb83_qualification import refuse_wb83_path


DESIGN = Path("outputs/mc24_four_station_calypso_physical_replica_design_v1")
FILES = (
    "calypso_physical_replica_production_protocol.md",
    "calypso_physical_replica_production_plan.json",
    "physical_replica_capacity_estimate.json",
    "provenance_contract.json",
)
V1_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v1/truth_oracle_qualification.json")
V2_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v2/truth_oracle_qualification.json")
CLOSURE = Path("outputs/mc24_four_station_wb83_final_closure_v1/wb83_final_closure.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_required_design_files_exist_and_are_not_jobs():
    names = {path.name for path in DESIGN.iterdir() if path.is_file()}
    assert set(FILES) <= names
    for name in FILES:
        refuse_wb83_path(DESIGN / name)
        text = (DESIGN / name).read_text(encoding="utf-8")
        assert "production_authorized" in text or name.endswith(".md")
        assert '"alignment_oracle_qualified": true' not in text
        assert "alignment_oracle_qualified = true\n" not in text


def test_contract_keeps_qualification_blocked_and_production_unstarted():
    provenance = _load(DESIGN / "provenance_contract.json")
    plan = _load(DESIGN / "calypso_physical_replica_production_plan.json")
    capacity = _load(DESIGN / "physical_replica_capacity_estimate.json")
    for payload in (provenance, plan, capacity):
        assert payload["physical_alignment_qualification_blocked_by_data"] is True
        assert payload["physical_replica_production_feasible"] is True
        assert payload["alignment_oracle_qualified"] is False
        assert payload.get("production_authorized") is False
        assert payload.get("production_started") is False
        assert payload.get("executable") is False
    assert provenance["forbidden_assets"]["opened"] is False
    assert provenance["source_policy"]["association_source_unseen_evaluation"]["is_source_unseen_ml_evaluation"] is False
    assert capacity["trained_on_wb83_failure_numbers"] is False
    assert capacity["planning_table"]["number_of_independent_events_total"] == 3200
    assert capacity["sample_size_basis"]["alternative_not_taken_from_wb83_cells"] is True
    assert plan["is_wb83_rerun"] is False
    assert plan["new_replicas_authorized"] is False


def test_wb83_gates_remain_fail_after_design():
    v1 = _load(V1_GATE)
    v2 = _load(V2_GATE)
    closure = _load(CLOSURE)
    assert v1["qualification"] == "FAIL"
    assert v2["qualification"] == "FAIL"
    assert v2["common_track_solver_qualified_under_toy_model"] is False
    assert v2["alignment_oracle_qualified_for_physical_FASER"] is False
    assert closure["toy_qualification"] == "FAIL"
    assert closure["physical_FASER_oracle_qualified"] is False
    assert closure["further_identical_rerun_authorized"] is False
    assert not Path("outputs/mc24_four_station_wb83_truth_only_replicas_v3").exists()
