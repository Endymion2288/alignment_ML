from __future__ import annotations

from pathlib import Path

import pytest

from alignment.cluster_local_jacobian_transfer_repair import (
    DECISION_RESTORED,
    DECISION_UNAVAILABLE,
    EXACT_JOIN_KEY,
    INHERITED_ENTRY_58,
    INHERITED_SKIP,
    SCHEMA_VERSION,
    decide_repair,
    dump_run_inventory,
    load_repair_config,
)
from alignment.identifiable_subspace import FuzzyJoinError, refuse_fuzzy_or_nearest_neighbour_join
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_repair_config(Path("configs/cluster_local_jacobian_transfer_repair_v1.yaml"))


def test_config_is_provenance_repair_not_identifiability_or_stable_core():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["not_a_stable_core_gate"] is True
    assert config["not_a_cluster_local_identifiability_campaign"] is True
    assert config["do_not_fuzzy_match_join"] is True
    assert config["do_not_nearest_neighbour_join"] is True
    assert config["do_not_reselect_population_to_force_nonzero_join"] is True
    assert config["do_not_merge_into_stable_core_gate"] is True
    assert config["do_not_reopen_entry_58_identifiability_decision"] is True
    assert config["inherited_v1_skip_reason"] == INHERITED_SKIP
    assert config["inherited_entry_58_decision"] == INHERITED_ENTRY_58
    assert config["exact_join_key"] == EXACT_JOIN_KEY
    assert config["runs"]["reference"]["run"] == 14973
    assert config["runs"]["transfer"]["run"] == 14974
    assert "cluster_local_r14974.root" in config["runs"]["transfer"]["cluster_dump"]
    assert config["runs"]["reference"]["cluster_dump"].endswith("true_cluster_local_residual_feasibility_v1/cluster_local.root")


def test_refuses_fuzzy_and_nearest_neighbour_join():
    with pytest.raises(FuzzyJoinError):
        refuse_fuzzy_or_nearest_neighbour_join()


def test_wrong_dump_control_stays_zero_and_matched_dump_restores_join():
    reference = {
        "join_complete": True,
        "n_measurements": 454,
        "exact_join_nonzero": True,
    }
    transfer = {
        "join_complete": True,
        "n_measurements": 403,
        "exact_join_nonzero": True,
    }
    wrong = {
        "join_complete": False,
        "n_measurements": 0,
        "exact_join_nonzero": False,
        "mismatch_reason": "run_id_disjoint_dump_does_not_contain_route_run",
    }
    decision = decide_repair(reference=reference, transfer=transfer, wrong=wrong)
    assert_no_alignment_payload(decision)
    assert decision["decision"] == DECISION_RESTORED
    assert decision["exact_join_restored"] is True
    assert decision["wrong_dump_control_still_zero"] is True
    assert decision["authorize_cluster_local_identifiability_campaign"] is False
    assert decision["not_a_stable_core_gate"] is True
    assert decision["entry_58_identifiability_not_reopened"] is True
    assert decision["geometry_write_allowed"] is False
    still_zero = decide_repair(
        reference=reference,
        transfer={"join_complete": False, "n_measurements": 0, "exact_join_nonzero": False},
        wrong=wrong,
    )
    assert still_zero["decision"] == DECISION_UNAVAILABLE
    assert still_zero["if_failed_do_not_chase_by_fuzzy_join"] is True


def test_dump_inventories_are_run_disjoint():
    reference = dump_run_inventory(
        Path("outputs/true_cluster_local_residual_feasibility_v1/cluster_local.root")
    )
    transfer = dump_run_inventory(
        Path("outputs/true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root")
    )
    assert reference["runs"] == {"14973": reference["n_clusters_with_surface"]}
    assert transfer["runs"] == {"14974": transfer["n_clusters_with_surface"]}
    assert set(reference["runs"]) != set(transfer["runs"])
    assert reference["fuzzy_join_used"] is False
    assert transfer["nearest_neighbour_join_used"] is False
    assert reference["join_key"] == EXACT_JOIN_KEY
