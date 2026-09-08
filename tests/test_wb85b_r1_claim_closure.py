"""WB85b-r1: claim wording and source-freeze checks.  No WB86."""

from __future__ import annotations

import subprocess
import sys

import pytest

from alignment.wb85b_fwer_protocol import (
    ALPHA_COV,
    ALPHA_LS,
    CRITICAL_COV,
    CRITICAL_LS,
    EXPECTED_CRITICAL_COV,
    EXPECTED_CRITICAL_LS,
    HISTORICAL_WB85A,
    OUTPUT_ROOT as WB85B_OUTPUT,
)
from alignment.wb85b_r1_claim_closure import (
    RUNTIME_SOURCE_SNAPSHOT,
    WB85BR1Error,
    claim_correction,
    refuse_wb85b_rewrite,
    worktree_source_identity,
)


def test_decision_rule_is_unchanged():
    claim = claim_correction()
    assert claim["decision_rule_unchanged"] is True
    assert claim["alpha_LS"] == ALPHA_LS == 0.025
    assert claim["alpha_cov"] == ALPHA_COV == 0.025
    assert CRITICAL_LS == EXPECTED_CRITICAL_LS
    assert CRITICAL_COV == EXPECTED_CRITICAL_COV
    assert claim["exact_finite_sample_FWER_proven"] is False
    assert claim["finite_sample_FWER_mathematically_proven"] is False
    assert claim["nominal_primary_statistical_FWER_cap"] == 0.05
    assert claim["fwer_le_0.05_is_conditional_on_actual_individual_gate_sizes_le_0.025"] is True
    assert claim["thresholds_not_retuned_from_this_wording_correction"] is True


def test_worktree_matches_runtime_source_snapshot():
    identity = worktree_source_identity()
    assert identity["runtime_source_snapshot_sha256"] == RUNTIME_SOURCE_SNAPSHOT
    assert identity["worktree_source_snapshot_sha256"] == RUNTIME_SOURCE_SNAPSHOT
    assert identity["byte_identical_to_runtime_snapshot"] is True
    assert not identity["missing_source_files"]
    assert all(identity["file_matches_runtime_record"].values())


def test_writer_refuses_history_execute_and_wb86():
    with pytest.raises(WB85BR1Error, match="historical"):
        refuse_wb85b_rewrite(WB85B_OUTPUT / "wb85b_protocol_qa.json")
    with pytest.raises(Exception, match="historical"):
        refuse_wb85b_rewrite(HISTORICAL_WB85A / "wb85a_protocol_qa.json")
    execute = subprocess.run(
        [sys.executable, "scripts/run_wb85b_r1_claim_closure.py", "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert execute.returncode != 0
    wb86 = subprocess.run(
        [sys.executable, "scripts/run_wb85b_r1_claim_closure.py", "--run-wb86"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wb86.returncode != 0
    mc = subprocess.run(
        [sys.executable, "scripts/run_wb85b_r1_claim_closure.py", "--rerun-monte-carlo"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert mc.returncode != 0
