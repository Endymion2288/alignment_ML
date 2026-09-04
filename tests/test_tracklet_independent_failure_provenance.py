from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, ForcedRankError, refuse_forced_identifiable_rank
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.tracklet_independent_failure_provenance import (
    CLASS_CORRUPTION,
    CLASS_COVERAGE,
    INHERITED_ENTRY_69,
    SCHEMA_VERSION,
    audit_bank,
    build_all_reports,
    classify_source,
    load_audit_config,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_audit_config(Path("configs/tracklet_independent_failure_provenance_audit_v1.yaml"))


def _synthetic_derivative(rank_like="full"):
    derivative = np.zeros((8, 4, 7), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.4
    derivative[:, 0, 2] = 0.0 if rank_like == "weak_dz" else 0.02
    derivative[:, 2, 3] = 0.5
    derivative[:, 3, 4] = 0.6
    derivative[:, 0, 5] = 0.3
    derivative[:, 1, 6] = 0.8
    if rank_like == "two":
        derivative[:, :, 1:] = 0.0
        derivative[:, 0, 6] = 1.0
    return derivative


def _synthetic_bank(derivative, *, source_id="s0", split="train", scales=None):
    pairs, _dim, parameters = derivative.shape
    if scales is None:
        scales = (5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 0.12)[:parameters]
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step * derivative[:, :, index]
        negative[index] = nominal - step * derivative[:, :, index]
    names = (
        "ift_dx_mm",
        "ift_dy_mm",
        "ift_dz_mm",
        "ift_rx_mrad",
        "ift_ry_mrad",
        "ift_rz_mrad",
        "C_dx",
    )[:parameters]
    run_offset = 1000 * (sum(ord(char) for char in source_id) + 1)
    return {
        "source_id": source_id,
        "split": split,
        "specs": (),
        "names": names,
        "scales": np.asarray(scales, dtype=np.float64),
        "anchor_values": np.zeros(parameters, dtype=np.float64),
        "reference_values": np.zeros(parameters, dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": np.full(parameters, step, dtype=np.float64),
        "negative_values": np.full(parameters, -step, dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.arange(pairs, dtype=np.int64) + run_offset,
        "event_id": np.arange(pairs, dtype=np.int64),
        "truth_particle_id": np.ones(pairs, dtype=np.int64),
        "source_station_id": np.zeros(pairs, dtype=np.int64),
        "target_station_id": np.ones(pairs, dtype=np.int64),
        "layer_weights": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "overlap": {"pairs": int(pairs)},
        "test_data_accessed": False,
        "held_out_physical_points_loaded": False,
    }


def test_config_is_read_only_and_does_not_reopen_stable_core():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["inherited_entry_69_decision"] == INHERITED_ENTRY_69
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert config["read_only_audit"] is True
    assert config["do_not_drop_failure_sources"] is True
    assert config["do_not_force_identifiable_rank_five"] is True
    assert config["do_not_use_as_stable_core_confirmation"] is True
    assert config["do_not_merge_into_cluster_local_gate"] is True
    failures = set(config["jacobian_corpus"]["failure_source_ids"])
    siblings = set(config["jacobian_corpus"]["sibling_control_source_ids"])
    assert failures == {
        "mc24_100043_00400_00499",
        "mc24_100043_00500_00599",
        "mc24_100044_00200_00299",
    }
    assert siblings == {
        "mc24_100043_00300_00399",
        "mc24_100044_00400_00499",
    }
    assert not (failures & siblings)


def test_refuses_to_force_rank_five():
    with pytest.raises(ForcedRankError):
        refuse_forced_identifiable_rank(native_rank=2, requested_rank=5)


def test_missing_probes_are_pipeline_corruption():
    audit = {
        "artifact_flags": ["missing_fd_probes"],
        "low_rank_consistent_with_weak_columns": False,
    }
    assert classify_source(audit) == CLASS_CORRUPTION
    coverage = classify_source({"artifact_flags": ["near_zero_jacobian_column"], "low_rank_consistent_with_weak_columns": True})
    assert coverage == CLASS_COVERAGE


def test_audit_bank_does_not_truncate_or_drop_low_rank_source():
    bank = _synthetic_bank(_synthetic_derivative("two"), source_id="mc24_100043_00400_00499")
    payload = audit_bank(bank, config=_config())
    assert payload["native_identifiable_rank"] < 5
    assert payload["truncated_to_five"] is False
    assert payload["not_dropped"] is True
    assert payload["not_used_as_stable_core_confirmation"] is True
    assert payload["classification"] in {CLASS_COVERAGE, CLASS_CORRUPTION}


def test_build_all_reports_on_synthetic_banks_never_authorizes_chase():
    failures = [
        _synthetic_bank(_synthetic_derivative("two"), source_id="mc24_100043_00400_00499"),
        _synthetic_bank(_synthetic_derivative("two"), source_id="mc24_100043_00500_00599"),
        _synthetic_bank(_synthetic_derivative("two"), source_id="mc24_100044_00200_00299"),
    ]
    siblings = [
        _synthetic_bank(_synthetic_derivative("full"), source_id="mc24_100043_00300_00399"),
        _synthetic_bank(_synthetic_derivative("full"), source_id="mc24_100044_00400_00499"),
    ]
    reports = build_all_reports(
        _config(),
        banks_by_role={"failure": failures, "sibling": siblings},
    )
    assert_no_alignment_payload(reports["next_stage_decision"])
    decision = reports["next_stage_decision"]
    assert decision["inherited_entry_69_decision"] == INHERITED_ENTRY_69
    assert decision["sources_dropped"] is False
    assert decision["rank_forced_to_five"] is False
    assert decision["used_as_cluster_local_confirmation"] is False
    assert decision["geometry_write_allowed"] is False
    assert decision["if_failed_do_not_chase_by_dropping_sources"] is True
    assert len(reports["failure_sources"]) == 3
    assert len(reports["sibling_controls"]) == 2
