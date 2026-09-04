from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.physically_distinct_track_coverage import (
    ResidualBlindStageError,
    summarize_events,
)
from alignment.physically_distinct_track_coverage_export import (
    DECISION_ADMIT_FD_PREREGISTRATION,
    DECISION_INSUFFICIENT,
    SCHEMA_VERSION,
    VERDICT_ADMITTED,
    VERDICT_INSUFFICIENT,
    VERDICT_MISSING_OUTPUT,
    VERDICT_NOT_DISTINCT,
    VERDICT_PROVENANCE_FAILURE,
    VERDICT_SEGMENTFIT_EMPTY,
    WORKBOOK_74_CONFIG_SHA256,
    ProvenanceValidationError,
    classify_candidate_verdict,
    decide_next_stage_75,
    exporter_contract_audit,
    load_events_physical_order,
    load_export_config,
    summarize_events_population,
    validate_exported_source,
    verify_export_gate,
    wrong_source_negative_control,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload
from datasets.root_loader import EventTracklets

CONFIG_PATH = Path(
    "configs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml"
)


def _config():
    return load_export_config(CONFIG_PATH)


def _event(stations, tx, ty, *, event_id=1, run_id=100120, pdg=13) -> EventTracklets:
    size = len(stations)
    state = np.zeros((size, 4), dtype=np.float64)
    state[:, 2] = tx
    state[:, 3] = ty
    z_table = {0: -1860.15, 1: 47.4, 2: 1237.4, 3: 2427.4}
    return EventTracklets(
        run_id=int(run_id),
        event_id=int(event_id),
        station_id=np.asarray(stations, dtype=np.int16),
        tracklet_id=np.arange(size, dtype=np.int32),
        z_mm=np.asarray([z_table[int(s)] for s in stations], dtype=np.float64),
        state=state,
        covariance=np.tile(np.eye(4, dtype=np.float64), (size, 1, 1)),
        chi2=np.ones(size, dtype=np.float64),
        ndof=np.ones(size, dtype=np.float64),
        n_hit=np.full(size, 6, dtype=np.int16),
        hit_pattern=np.full(size, 0b111, dtype=np.uint64),
        truth_particle_id=np.arange(10, 10 + size, dtype=np.int64),
        truth_pdg=np.full(size, int(pdg), dtype=np.int32),
        truth_match_fraction=np.ones(size, dtype=np.float64),
    )


def _write_tracklets_root(path: Path, blocks, *, source_file: str = "/input/x.root") -> None:
    """Write a synthetic canonical tracklets file.

    ``blocks`` is a list of ``(run_id, event_id, n_tracklets, pdg)`` tuples;
    each tuple becomes one consecutive physical-event block in file order.
    """
    import uproot

    from datasets.schema import covariance_columns

    columns: dict[str, list[np.ndarray]] = {}

    def append(name, values):
        columns.setdefault(name, []).append(np.asarray(values))

    for run_id, event_id, n_trk, pdg in blocks:
        n = int(n_trk)
        append("run_id", np.full(n, run_id, dtype=np.int32))
        append("event_id", np.full(n, event_id, dtype=np.int32))
        append("station_id", np.zeros(n, dtype=np.int16))
        append("tracklet_id", np.arange(n, dtype=np.int32))
        append("x_mm", np.zeros(n))
        append("y_mm", np.zeros(n))
        append("z_mm", np.full(n, -1860.15))
        append("tx", np.full(n, 0.001))
        append("ty", np.full(n, 0.001))
        cov = covariance_columns(np.tile(np.eye(4), (n, 1, 1)))
        for key, values in cov.items():
            append(key, values)
        append("chi2", np.ones(n))
        append("ndof", np.ones(n))
        append("n_hit", np.full(n, 6, dtype=np.int16))
        append("hit_pattern", np.full(n, 0b111, dtype=np.uint64))
        append("truth_particle_id", np.arange(n, dtype=np.int64) + 1)
        append("truth_pdg", np.full(n, pdg, dtype=np.int32))
        append("truth_match_fraction", np.ones(n))
    if columns:
        payload = {key: np.concatenate(values) for key, values in columns.items()}
    else:
        # Empty SegmentFit control: all canonical fields present, zero rows.
        from datasets.schema import COVARIANCE_FIELDS

        payload = {
            "run_id": np.asarray([], dtype=np.int32),
            "event_id": np.asarray([], dtype=np.int32),
            "station_id": np.asarray([], dtype=np.int16),
            "tracklet_id": np.asarray([], dtype=np.int32),
            "x_mm": np.asarray([], dtype=np.float64),
            "y_mm": np.asarray([], dtype=np.float64),
            "z_mm": np.asarray([], dtype=np.float64),
            "tx": np.asarray([], dtype=np.float64),
            "ty": np.asarray([], dtype=np.float64),
            "chi2": np.asarray([], dtype=np.float64),
            "ndof": np.asarray([], dtype=np.float64),
            "n_hit": np.asarray([], dtype=np.int16),
            "hit_pattern": np.asarray([], dtype=np.uint64),
            "truth_particle_id": np.asarray([], dtype=np.int64),
            "truth_pdg": np.asarray([], dtype=np.int32),
            "truth_match_fraction": np.asarray([], dtype=np.float64),
        }
        for name in COVARIANCE_FIELDS:
            payload[name] = np.asarray([], dtype=np.float64)
    with uproot.recreate(path) as handle:
        handle.mktree("tracklets", {key: values.dtype for key, values in payload.items()})
        handle["tracklets"].extend(payload)
        handle["metadata"] = {
            "source_file": np.asarray([source_file]),
            "converter": np.asarray(["test"]),
        }


# ---------------------------------------------------------------------------
# Config freezing
# ---------------------------------------------------------------------------

def test_config_freezes_workbook74_contract_and_gates():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["stage"] == "residual_blind_htcondor_export_and_reinventory"
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["inherited_workbook_74_config_sha256"] == WORKBOOK_74_CONFIG_SHA256
    gates = config["coverage_gates"]
    assert gates["min_events"] == 200
    assert gates["min_ift_events"] == 200
    assert gates["min_complete_four_station_events"] == 80
    assert gates["min_independent_sources_or_runs"] == 2
    assert gates["min_outside_canonical_envelope_fraction"] == pytest.approx(0.20)
    assert gates["max_histogram_intersection_for_distinct"] == pytest.approx(0.80)
    assert gates["do_not_lower_gates"] is True
    assert gates["min_xaod_files_for_export"] == 2
    assert gates["min_xaod_events_per_file_for_export"] == 1000
    contract = config["rigid_station_five_dof_contract"]
    assert tuple(contract["parameter_names"]) == FREE_PARAMETERS
    assert contract["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert contract["scale_matrix_S"]["ift_dx_mm"] == pytest.approx(5.0)
    assert contract["scale_matrix_S"]["ift_rz_mrad"] == pytest.approx(60.0)
    assert contract["do_not_reconstruct_this_stage"] is True
    assert config["do_not_construct_fd_identifiability_this_stage"] is True
    assert config["do_not_compute_svd_or_rank_this_stage"] is True
    assert config["do_not_inject_alignment_payload_in_export"] is True
    assert config["do_not_pool_the_two_export_candidates"] is True
    assert config["do_not_split_one_xaod_file_into_fake_independent_sources"] is True
    assert config["export_is_not_fd_authorization"] is True
    assert config["geometry_write_allowed"] is False
    assert config["real_data_correction_authorized"] is False
    ids = [row["id"] for row in config["export_candidates"]]
    assert ids == ["mc24_100120_muon_floor", "mc24_100130_kshort_end_fasernu"]
    by_id = {row["id"]: row for row in config["export_candidates"]}
    assert by_id["mc24_100120_muon_floor"]["angular_population"]["abs_pdg"] == [13]
    assert by_id["mc24_100130_kshort_end_fasernu"]["angular_population"]["abs_pdg"] == [211]
    assert by_id["mc24_100130_kshort_end_fasernu"]["crosscheck_angular_population"]["abs_pdg"] == [13]
    for row in config["export_candidates"]:
        assert len(row["inputs"]) == 10
        source_ids = [str(item["source_id"]) for item in row["inputs"]]
        assert len(set(source_ids)) == 10


def test_config_rejects_gate_or_candidate_tampering(tmp_path):
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["coverage_gates"]["min_events"] = 100
    bad = tmp_path / "bad_gate.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="min_events"):
        load_export_config(bad)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["export_candidates"] = payload["export_candidates"][:1]
    bad = tmp_path / "bad_candidates.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only the two"):
        load_export_config(bad)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["coverage_gates"]["min_outside_canonical_envelope_fraction"] = 0.10
    bad = tmp_path / "bad_distinct.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="min_outside"):
        load_export_config(bad)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["export_candidates"][0]["inputs"][0]["source_id"] = "mc24_100116_00030_00039"
    bad = tmp_path / "bad_sealed.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(Exception, match="[Ss]ealed"):
        load_export_config(bad)


# ---------------------------------------------------------------------------
# Physical-order loader (merged-rec event-number collisions)
# ---------------------------------------------------------------------------

def test_physical_order_loader_splits_reused_event_ids(tmp_path):
    path = tmp_path / "tracklets.root"
    # Merged rec file: generator job A events 0,1 then generator job B
    # restarts at 0,1.  Sorted (run, event) grouping would merge the two
    # physical event-0 blocks into one logical event.
    _write_tracklets_root(
        path,
        [
            (100120, 0, 2, 13),
            (100120, 1, 1, 13),
            (100120, 0, 3, 13),
            (100120, 1, 2, 13),
        ],
    )
    events, provenance = load_events_physical_order(path)
    assert len(events) == 4
    assert [int(ev.event_id) for ev in events] == [0, 1, 0, 1]
    assert [int(ev.size) for ev in events] == [2, 1, 3, 2]
    assert provenance["n_physical_events"] == 4
    assert provenance["n_unique_event_ids"] == 2
    assert provenance["event_id_reused_across_generator_jobs"] is True
    assert provenance["sorted_merge_used"] is False
    assert provenance["fuzzy_join_used"] is False


def test_physical_order_loader_rejects_duplicate_tracklet_id_in_block(tmp_path):
    import uproot

    from datasets.schema import covariance_columns

    path = tmp_path / "tracklets.root"
    n = 2
    payload = {
        "run_id": np.full(n, 100120, dtype=np.int32),
        "event_id": np.zeros(n, dtype=np.int32),
        "station_id": np.zeros(n, dtype=np.int16),
        "tracklet_id": np.zeros(n, dtype=np.int32),  # duplicate inside one block
        "x_mm": np.zeros(n),
        "y_mm": np.zeros(n),
        "z_mm": np.full(n, -1860.15),
        "tx": np.full(n, 0.001),
        "ty": np.full(n, 0.001),
        "chi2": np.ones(n),
        "ndof": np.ones(n),
        "n_hit": np.full(n, 6, dtype=np.int16),
        "hit_pattern": np.full(n, 0b111, dtype=np.uint64),
        "truth_particle_id": np.arange(n, dtype=np.int64) + 1,
        "truth_pdg": np.full(n, 13, dtype=np.int32),
        "truth_match_fraction": np.ones(n),
    }
    payload.update(covariance_columns(np.tile(np.eye(4), (n, 1, 1))))
    with uproot.recreate(path) as handle:
        handle["tracklets"] = payload
    with pytest.raises(ProvenanceValidationError, match="duplicate tracklet_id"):
        load_events_physical_order(path)


# ---------------------------------------------------------------------------
# Population-parameterized summary regression
# ---------------------------------------------------------------------------

def test_population_summary_matches_workbook74_muon_mask():
    config = _config()
    events = [
        _event([0, 1, 2, 3], 0.01, 0.0, event_id=1, pdg=13),
        _event([0, 1], 0.02, 0.01, event_id=2, pdg=-13),
        _event([1, 2, 3], 0.03, 0.0, event_id=3, pdg=211),
    ]
    reference = summarize_events(
        events,
        source_id="s0",
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
    )
    ours = summarize_events_population(
        events,
        source_id="s0",
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
        angular_abs_pdg=(13,),
    )
    for key in (
        "n_events",
        "n_tracklets",
        "n_angular_tracklets",
        "n_ift_events",
        "n_complete_four_station_events",
        "station_event_counts",
        "stations_present",
        "tx",
        "ty",
        "slope",
        "azimuth",
        "wide_local_slope_fraction",
        "ip_like_local_slope_fraction",
    ):
        assert ours[key] == reference[key], key
    assert ours["residual_blind"] is True
    assert ours["angular_population_abs_pdg"] == [13]

    pion = summarize_events_population(
        events,
        source_id="s0",
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
        angular_abs_pdg=(211,),
    )
    # The pion event has no station-0 tracklet; the representative tracklet
    # falls back to the first selected station, exactly as in workbook 74.
    assert pion["n_angular_tracklets"] == 1
    assert pion["tx"]["count"] == 1
    # Station coverage is population-independent and identical to the muon mask.
    assert pion["station_event_counts"] == reference["station_event_counts"]
    assert pion["n_ift_events"] == reference["n_ift_events"]
    assert_no_alignment_payload({k: v for k, v in ours.items() if k not in {"tx_values", "ty_values"}})


# ---------------------------------------------------------------------------
# Export gate
# ---------------------------------------------------------------------------

def _manifest_row(n_events=100000, segmentfit=True, exists=True):
    return {
        "source_id": "s",
        "exists": exists,
        "n_events": n_events,
        "collection_keys_of_interest": {"SegmentFit": segmentfit},
    }


def test_verify_export_gate_uses_frozen_thresholds():
    config = _config()
    gates = config["coverage_gates"]
    manifest = {
        "candidates": [
            {
                "id": "c",
                "geom_flag": "TI12MC04",
                "geometry_tag": "FASERNU-04",
                "inputs": [_manifest_row(), _manifest_row()],
            }
        ]
    }
    gate = verify_export_gate(manifest, gates)
    assert gate["all_candidates_satisfied"] is True
    assert gate["candidates"][0]["export_gate_satisfied"] is True

    manifest["candidates"][0]["inputs"] = [_manifest_row()]
    gate = verify_export_gate(manifest, gates)
    assert gate["all_candidates_satisfied"] is False
    assert "n_xaod_files" in gate["candidates"][0]["reason"]

    manifest["candidates"][0]["inputs"] = [_manifest_row(), _manifest_row(n_events=10)]
    gate = verify_export_gate(manifest, gates)
    assert gate["all_candidates_satisfied"] is False
    assert "below 1000 events" in gate["candidates"][0]["reason"]

    manifest["candidates"][0]["inputs"] = [_manifest_row(segmentfit=False), _manifest_row()]
    gate = verify_export_gate(manifest, gates)
    assert gate["all_candidates_satisfied"] is False
    assert "SegmentFit" in gate["candidates"][0]["reason"]


# ---------------------------------------------------------------------------
# Exporter-contract audit
# ---------------------------------------------------------------------------

def test_exporter_contract_audit_is_generic_and_unmodified():
    audit = exporter_contract_audit(_config())
    chain = audit["chain"]
    assert chain["exporter"]["single_muon_assumption"] is False
    assert chain["exporter"]["pdg_13_requirement"] is False
    assert chain["truth_matching"]["single_truth_parent_requirement"] is False
    assert chain["segmentfit_population"]["single_muon_assumption"] is False
    assert audit["truth_matching_modified"] is False
    assert audit["particle_selection_modified_post_hoc"] is False
    assert audit["route_definition_modified"] is False
    kshort = audit["candidates"]["mc24_100130_kshort_end_fasernu"]
    assert kshort["exporter_contract_compatible"] is True
    assert kshort["angular_population_abs_pdg"] == [211]
    assert audit["workbook_74_reporter_angular_mask"]["muon_specific"] is True


# ---------------------------------------------------------------------------
# Provenance validation and negative controls
# ---------------------------------------------------------------------------

def test_validate_exported_source_and_wrong_source_control(tmp_path):
    config = _config()
    candidate = config["export_candidates"][0]
    source = candidate["inputs"][0]
    export_dir = tmp_path / "exports" / "mc24_100120_muon_floor" / source["source_id"]
    export_dir.mkdir(parents=True)
    _write_tracklets_root(
        export_dir / "tracklets.root",
        [(100120, 0, 2, 13), (100120, 1, 1, 13), (100120, 0, 1, 13)],
        source_file=source["path"],
    )
    result = validate_exported_source(
        source_id=source["source_id"],
        expected_input_path=source["path"],
        export_dir=export_dir,
        job_provenance={"exit_status": 0, "events_processed": 4},
        config=config,
    )
    assert result["provenance_valid"] is True
    assert result["status"] == "ok"
    assert result["checks"]["metadata_matches_declared_input"] is True
    assert result["checks"]["run_id_matches_production"] is True
    assert result["checks"]["station_z_mapping_valid"] is True
    assert result["checks"]["n_physical_events_with_tracklets"] == 3

    control = wrong_source_negative_control(
        source_id=source["source_id"],
        wrong_input_path=candidate["inputs"][1]["path"],
        export_dir=export_dir,
    )
    assert control["control_passed"] is True
    assert control["mismatch_detected"] is True
    assert control["fuzzy_join_used"] is False

    missing = validate_exported_source(
        source_id="never_exported",
        expected_input_path="/nonexistent.root",
        export_dir=tmp_path / "absent",
        job_provenance=None,
        config=config,
    )
    assert missing["status"] == VERDICT_MISSING_OUTPUT
    assert missing["provenance_valid"] is False

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    _write_tracklets_root(empty_dir / "tracklets.root", [], source_file=source["path"])
    empty = validate_exported_source(
        source_id=source["source_id"],
        expected_input_path=source["path"],
        export_dir=empty_dir,
        job_provenance={"exit_status": 0, "events_processed": 4},
        config=config,
    )
    assert empty["status"] == VERDICT_SEGMENTFIT_EMPTY


def test_validate_exported_source_detects_wrong_run(tmp_path):
    config = _config()
    candidate = config["export_candidates"][0]
    source = candidate["inputs"][0]
    export_dir = tmp_path / "wrong_run"
    export_dir.mkdir()
    _write_tracklets_root(
        export_dir / "tracklets.root",
        [(999999, 0, 2, 13)],
        source_file=source["path"],
    )
    result = validate_exported_source(
        source_id=source["source_id"],
        expected_input_path=source["path"],
        export_dir=export_dir,
        job_provenance={"exit_status": 0},
        config=config,
    )
    assert result["provenance_valid"] is False
    assert result["status"] == VERDICT_PROVENANCE_FAILURE
    assert any("run ids" in reason for reason in result["failure_reasons"])


# ---------------------------------------------------------------------------
# Verdict classification and stage decision
# ---------------------------------------------------------------------------

def _admission(**overrides):
    payload = {
        "admitted_to_separate_fd_campaign": False,
        "physically_distinct_hypothesis": True,
        "observed_phase_space_distinct": True,
        "coverage_measured": True,
        "n_events": 1000,
        "n_ift_events": 500,
        "n_complete_four_station_events": 200,
        "n_independent_sources": 10,
        "reason": "admitted",
    }
    payload.update(overrides)
    return payload


def test_classify_candidate_verdict_labels():
    admitted = classify_candidate_verdict(
        _admission(admitted_to_separate_fd_campaign=True),
        provenance_valid=True,
        exporter_compatible=True,
        n_tracklets=100,
        n_sources_with_output=10,
    )
    assert admitted == VERDICT_ADMITTED
    insufficient = classify_candidate_verdict(
        _admission(n_events=50, reason="n_events 50 < 200"),
        provenance_valid=True,
        exporter_compatible=True,
        n_tracklets=100,
        n_sources_with_output=10,
    )
    assert insufficient == VERDICT_INSUFFICIENT
    not_distinct = classify_candidate_verdict(
        _admission(observed_phase_space_distinct=False),
        provenance_valid=True,
        exporter_compatible=True,
        n_tracklets=100,
        n_sources_with_output=10,
    )
    assert not_distinct == VERDICT_NOT_DISTINCT
    empty = classify_candidate_verdict(
        _admission(),
        provenance_valid=True,
        exporter_compatible=True,
        n_tracklets=0,
        n_sources_with_output=5,
    )
    assert empty == VERDICT_SEGMENTFIT_EMPTY
    missing = classify_candidate_verdict(
        _admission(),
        provenance_valid=True,
        exporter_compatible=True,
        n_tracklets=0,
        n_sources_with_output=0,
    )
    assert missing == VERDICT_MISSING_OUTPUT
    bad_prov = classify_candidate_verdict(
        _admission(admitted_to_separate_fd_campaign=True),
        provenance_valid=False,
        exporter_compatible=True,
        n_tracklets=100,
        n_sources_with_output=10,
    )
    assert bad_prov == VERDICT_PROVENANCE_FAILURE


def test_decide_next_stage_75_branches_and_refusals():
    admitted = decide_next_stage_75(
        [{"id": "c", "verdict": VERDICT_ADMITTED}]
    )
    assert admitted["decision"] == DECISION_ADMIT_FD_PREREGISTRATION
    assert admitted["admitted_candidates"] == ["c"]
    assert admitted["fd_identifiability_executed"] is False
    assert admitted["svd_or_rank_computed"] is False
    assert admitted["open_gauge_constrained_or_external_constraint_branch"] is False
    assert admitted["three_arm_authorized"] is False
    assert_no_alignment_payload(admitted)

    insufficient = decide_next_stage_75(
        [
            {"id": "a", "verdict": VERDICT_INSUFFICIENT},
            {"id": "b", "verdict": VERDICT_NOT_DISTINCT},
        ]
    )
    assert insufficient["decision"] == DECISION_INSUFFICIENT
    assert insufficient["answer"] == "No"
    assert (
        insufficient[
            "freeze_current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof"
        ]
        is True
    )
    assert insufficient["open_gauge_constrained_or_external_constraint_branch"] is True
    assert insufficient["gauge_convention_is_not_a_physical_measurement"] is True
    assert insufficient["geometry_write_allowed"] is False

    with pytest.raises(ResidualBlindStageError):
        decide_next_stage_75([], fd_executed=True)
