"""WB83 final closure: FAIL stays FAIL; no identical rerun; no forbidden opens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignment.wb83_qualification import PULL_MEAN_MAX, refuse_wb83_path


CLOSURE = Path("outputs/mc24_four_station_wb83_final_closure_v1")
V1_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v1/truth_oracle_qualification.json")
V2_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v2/truth_oracle_qualification.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_closure_seals_wb83_as_fail():
    closure = _load(CLOSURE / "wb83_final_closure.json")
    assert closure["v1_status"] == "FAIL"
    assert closure["v2_status"] == "FAIL"
    assert closure["solver_implementation_fixed"] is True
    assert closure["convergence_qualified"] is True
    assert closure["fd_stability_qualified"] is True
    assert closure["bias_screening_qualified"] is True
    assert closure["pulls_qualified"] is False
    assert closure["coverage_qualified"] is False
    assert closure["toy_qualification"] == "FAIL"
    assert closure["physical_FASER_oracle_qualified"] is False
    assert closure["further_identical_rerun_authorized"] is False
    assert closure["ml_alignment_eval_authorized"] is False
    assert closure["systematic_bias_not_established"] is True
    assert closure["remaining_failures_are_frozen_ensemble_statistical_failures"] is True
    assert PULL_MEAN_MAX == 0.2
    assert closure["qualification_rules"]["pull_mean_max"] == 0.2


def test_v1_and_v2_gates_are_not_rewritten_to_pass():
    v1 = _load(V1_GATE)
    v2 = _load(V2_GATE)
    assert v1["qualification"] == "FAIL"
    assert v2["qualification"] == "FAIL"
    assert v2["common_track_solver_qualified_under_toy_model"] is False
    assert v2["alignment_oracle_qualified_for_physical_FASER"] is False


def test_protocol_design_is_not_an_experiment():
    protocol = _load(CLOSURE / "prospective_statistical_protocol_design.json")
    assert protocol["executable"] is False
    assert protocol["new_replicas_authorized"] is False
    assert protocol["is_wb83_rerun"] is False
    assert protocol["trained_on_wb83_failure_numbers"] is False
    assert "M1_multiplicity_controlled_cellwise" in protocol["prospective_methods"]
    assert "M2_global_pull_gof" in protocol["prospective_methods"]
    assert "M3_hierarchical_random_effects" in protocol["prospective_methods"]
    assert protocol["new_experiment_must_use"]["not_a_wb83_rerun"] is True


def test_physical_feasibility_is_blocked_and_does_not_open_forbidden():
    audit = _load(CLOSURE / "calypso_physical_replica_feasibility.json")
    assert audit["physical_alignment_qualification_blocked_by_data"] is True
    assert audit["opened_forbidden_assets"] is False
    assert audit["overlay_pseudo_replicas_used"] is False
    assert audit["development_00350_used"] is False
    assert audit["final_blind_opened"] is False
    assert audit["sealed_opened"] is False
    assert audit["not_started"]["new_qualification_run"] is True


def test_closure_writer_refuses_replica_roots():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_wb83_final_closure.py",
            "--output-root",
            "outputs/mc24_four_station_wb83_truth_only_replicas_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing" in (result.stderr + result.stdout).lower() or "refuses" in (
        result.stderr + result.stdout
    ).lower()


def test_still_refuses_forbidden_alignment_paths():
    with pytest.raises(Exception, match="refuses"):
        refuse_wb83_path("outputs/family1/overlay_synthetic_v1/x.json")
    with pytest.raises(Exception, match="refuses"):
        refuse_wb83_path("mc24_100047_00350_00399/x.root")
    with pytest.raises(Exception, match="refuses"):
        refuse_wb83_path("00800_00849")
