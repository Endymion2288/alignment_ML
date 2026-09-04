"""Workbook-76 K-short rigid-station 5DoF FD complementarity tests.

Covers the pre-registered config freeze, the merged-rec physical-event
identity contract (occurrence-augmented exact join bridged through the
enhanced ntuple), the population/complementarity gate logic on synthetic
banks, and the campaign decision branches.  No K-short FD spectrum is
consulted anywhere in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, frozen_scales_for
from alignment.module_level_residual_poc import json_ready
from alignment.kshort_rigid_station_5dof_complementarity import (
    DECISION_CANONICAL_REGRESSION_FAILED,
    DECISION_COMPLEMENTARITY_FAIL,
    DECISION_COMPLEMENTARITY_PASS,
    DECISION_PHYSICAL_FAILURE,
    SCHEMA_VERSION,
    STATION_FIVE_NAMES,
    decide_campaign,
    joint_complementarity_report,
    load_config,
    population_report,
)
from datasets.physical_event_identity import (
    assert_no_duplicate_physical_identity,
    flat_row_uids,
    load_events_ntuple_identity,
    load_propagation_records_ntuple_identity,
    read_ntuple_event_index,
)
from datasets.root_loader import (
    PHYSICAL_EVENT_UID_STRIDE,
    load_events,
    physical_event_occurrence_uids,
)
from datasets.schema import DatasetSchemaError, covariance_columns


def _config():
    return load_config()


# ---------------------------------------------------------------------------
# Config freezing
# ---------------------------------------------------------------------------


def test_config_freezes_preregistered_contract():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["frozen"] is True
    assert float(config["rank_tolerance"]) == float(FROZEN_RANK_TOLERANCE)
    assert tuple(config["parameter_names"]) == STATION_FIVE_NAMES
    scales = config["scale_matrix_S"]
    assert [scales[name] for name in STATION_FIVE_NAMES] == [5.0, 5.0, 60.0, 60.0, 60.0]
    steps = config["finite_difference_steps"]
    assert [steps[name] for name in STATION_FIVE_NAMES] == [0.5, 0.5, 10.0, 10.0, 10.0]
    assert config["three_arm_authorized"] is False
    assert config["geometry_write_allowed"] is False
    assert config["real_data_correction_authorized"] is False
    assert config["joint_pooled_rank_five_alone_is_not_success"] is True
    assert config["do_not_change_pion_track_or_event_selection_after_fd"] is True
    identity = config["physical_event_identity"]
    assert identity["contract"] == "occurrence_augmented_file_order"
    assert identity["duplicate_physical_event_identity"] == "hard_failure"
    assert identity["fuzzy_join_allowed"] is False
    assert identity["nearest_neighbour_join_allowed"] is False
    assert identity["residual_proximity_join_allowed"] is False
    assert identity["sorted_run_event_grouping_allowed"] is False
    kshort = config["kshort_corpus"]
    assert len(kshort["train_source_ids"]) == 5
    assert len(kshort["validation_source_ids"]) == 5
    assert kshort["nevents_per_source"] == 10000
    assert kshort["fd_observable_pdg_mask"] == "none"
    canonical = config["canonical_corpus"]
    assert len(canonical["train_source_ids"]) == 10
    assert len(canonical["validation_source_ids"]) == 8
    contract = config["complementarity_contract"]
    assert contract["require_joint_pooled_rank"] == 5
    assert contract["require_canonical_hypothesis_core_dimension"] == 5
    assert contract["joint_source_stability_loso_mode"] == "pooled_remainder"
    core = config["stable_core"]
    assert core["min_consensus_eigenvalue"] == pytest.approx(0.70)
    assert core["min_mode_persistence"] == pytest.approx(0.85)
    assert core["independent_min_source_support_fraction"] == pytest.approx(0.80)
    assert core["independent_max_principal_angle_deg"] == pytest.approx(15.0)
    assert core["independent_max_missing_frobenius"] == pytest.approx(0.75)
    assert core["source_weighting"] == "equal_per_source"


def test_config_rejects_rank_tolerance_retune(tmp_path):
    import yaml

    config = _config()
    config.pop("config_path", None)
    config["rank_tolerance"] = 0.02
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="rank_tolerance"):
        load_config(path)


def test_config_rejects_sorted_grouping(tmp_path):
    import yaml

    config = _config()
    config.pop("config_path", None)
    config["physical_event_identity"]["sorted_run_event_grouping_allowed"] = True
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted_run_event_grouping_allowed"):
        load_config(path)


def test_config_rejects_step_change(tmp_path):
    import yaml

    config = _config()
    config.pop("config_path", None)
    config["finite_difference_steps"]["ift_rz_mrad"] = 5.0
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="finite-difference steps"):
        load_config(path)


# ---------------------------------------------------------------------------
# Physical-event identity primitive
# ---------------------------------------------------------------------------


def test_occurrence_uids_are_identity_without_collisions():
    runs = np.asarray([100043, 100043, 100043], dtype=np.int64)
    events = np.asarray([0, 1, 2], dtype=np.int64)
    uids = physical_event_occurrence_uids(runs, events)
    assert np.array_equal(uids, events)


def test_occurrence_uids_resolve_merged_rec_collisions():
    # Two generator jobs merged: event ids 0,1,2 then 0,1,2 again.
    runs = np.full(6, 100130, dtype=np.int64)
    events = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    uids = physical_event_occurrence_uids(runs, events)
    assert np.array_equal(
        uids,
        np.asarray(
            [0, 1, 2, PHYSICAL_EVENT_UID_STRIDE, PHYSICAL_EVENT_UID_STRIDE + 1, PHYSICAL_EVENT_UID_STRIDE + 2],
            dtype=np.int64,
        ),
    )
    assert np.unique(uids).size == uids.size


def test_occurrence_uids_reject_overflow():
    runs = np.asarray([1], dtype=np.int64)
    events = np.asarray([PHYSICAL_EVENT_UID_STRIDE], dtype=np.int64)
    with pytest.raises(DatasetSchemaError, match="stride"):
        physical_event_occurrence_uids(runs, events)


# ---------------------------------------------------------------------------
# Ntuple-bridged physical-event identity
# ---------------------------------------------------------------------------


def _write_enhanced_ntuple(path: Path, entries, validity_masks=None) -> None:
    """entries: list of (run, event_id, n_tracklet_rows, n_propagation_rows).

    ``validity_masks`` optionally gives one bool list per entry for
    ``Tracklet_has_covariance`` (length must equal n_tracklet_rows).
    """
    import awkward as ak
    import uproot

    path.parent.mkdir(parents=True, exist_ok=True)
    branches = {
        "run": ak.Array([int(entry[0]) for entry in entries]),
        "eventID": ak.Array([int(entry[1]) for entry in entries]),
        "Tracklet_z_mm": ak.Array(
            [np.full(int(entry[2]), -1860.15).tolist() for entry in entries]
        ),
        "TrackletPropagation_success": ak.Array(
            [[True] * int(entry[3]) for entry in entries]
        ),
    }
    if validity_masks is not None:
        branches["Tracklet_has_covariance"] = ak.Array(
            [[bool(value) for value in mask] for mask in validity_masks]
        )
    with uproot.recreate(path) as handle:
        handle["nt"] = branches


def _write_tracklets(path: Path, blocks) -> None:
    """blocks: list of (run, event_id, [(tracklet_id, station, x, truth), ...])."""
    import uproot

    columns: dict[str, list[np.ndarray]] = {}

    def append(name, values):
        columns.setdefault(name, []).append(np.asarray(values))

    for run, event_id, rows in blocks:
        n = len(rows)
        append("run_id", np.full(n, run, dtype=np.int32))
        append("event_id", np.full(n, event_id, dtype=np.int32))
        append("station_id", np.asarray([row[1] for row in rows], dtype=np.int16))
        append("tracklet_id", np.asarray([row[0] for row in rows], dtype=np.int32))
        append("x_mm", np.asarray([row[2] for row in rows], dtype=np.float64))
        append("y_mm", np.zeros(n))
        append(
            "z_mm",
            np.asarray([-1860.15 if row[1] == 0 else -1455.35 for row in rows]),
        )
        append("tx", np.full(n, 1.0e-3))
        append("ty", np.full(n, 2.0e-3))
        cov = covariance_columns(np.tile(0.01 * np.eye(4), (n, 1, 1)))
        for key, values in cov.items():
            append(key, values)
        append("chi2", np.ones(n))
        append("ndof", np.ones(n))
        append("n_hit", np.full(n, 6, dtype=np.int16))
        append("hit_pattern", np.full(n, 0b111, dtype=np.uint64))
        append("truth_particle_id", np.asarray([row[3] for row in rows], dtype=np.int64))
        append("truth_pdg", np.full(n, 211, dtype=np.int32))
        append("truth_match_fraction", np.ones(n))
    payload = {key: np.concatenate(values) for key, values in columns.items()}
    with uproot.recreate(path) as handle:
        handle.mktree("tracklets", {key: values.dtype for key, values in payload.items()})
        handle["tracklets"].extend(payload)


def _write_propagations(path: Path, rows) -> None:
    """rows: list of (run, event_id, src_trk, trg_trk, pred_x, truth)."""
    import uproot

    n = len(rows)
    cov = covariance_columns(np.tile(0.01 * np.eye(4), (n, 1, 1)))
    payload = {
        "run_id": np.asarray([row[0] for row in rows], dtype=np.int64),
        "event_id": np.asarray([row[1] for row in rows], dtype=np.int64),
        "source_tracklet_id": np.asarray([row[2] for row in rows], dtype=np.int32),
        "target_tracklet_id": np.asarray([row[3] for row in rows], dtype=np.int32),
        "source_station_id": np.zeros(n, dtype=np.int16),
        "target_station_id": np.ones(n, dtype=np.int16),
        "truth_particle_id": np.asarray([row[5] for row in rows], dtype=np.int64),
        "target_z_mm": np.full(n, -1455.35),
        "pred_x_mm": np.asarray([row[4] for row in rows], dtype=np.float64),
        "pred_y_mm": np.zeros(n),
        "pred_tx": np.full(n, 1.0e-3),
        "pred_ty": np.full(n, 2.0e-3),
        "success": np.ones(n, dtype=bool),
        "has_covariance": np.ones(n, dtype=bool),
    }
    for key, values in cov.items():
        payload[f"pred_{key}"] = values
    with uproot.recreate(path) as handle:
        handle.mktree("propagations", {key: values.dtype for key, values in payload.items()})
        handle["propagations"].extend(payload)


def test_ntuple_bridge_keeps_files_consistent_with_tracklet_only_event(tmp_path):
    # Three physical events share (run=100130, event=7); the middle one has
    # tracklets but no propagation rows.  File-relative occurrence counting
    # would shift the third event's propagation rows; the ntuple bridge must
    # not.
    entries = [
        (100130, 7, 2, 1),
        (100130, 7, 2, 0),
        (100130, 7, 2, 1),
    ]
    enhanced = tmp_path / "enhanced_tracklets.root"
    _write_enhanced_ntuple(enhanced, entries)
    index = read_ntuple_event_index(enhanced)
    assert index.size == 3
    assert np.array_equal(index.occurrence, [0, 1, 2])
    assert np.array_equal(
        index.uid,
        [7, 7 + PHYSICAL_EVENT_UID_STRIDE, 7 + 2 * PHYSICAL_EVENT_UID_STRIDE],
    )

    tracklet_rows = [
        (100130, 7, [(0, 0, 1.0, 5), (1, 1, 10.0, 5)]),
        (100130, 7, [(0, 0, 2.0, 5), (1, 1, 20.0, 5)]),
        (100130, 7, [(0, 0, 3.0, 5), (1, 1, 30.0, 5)]),
    ]
    tracklets = tmp_path / "tracklets.root"
    _write_tracklets(tracklets, tracklet_rows)
    events = load_events_ntuple_identity(tracklets, index)
    assert [int(event.event_id) for event in events] == [
        7,
        7 + PHYSICAL_EVENT_UID_STRIDE,
        7 + 2 * PHYSICAL_EVENT_UID_STRIDE,
    ]
    assert_no_duplicate_physical_identity(events)

    propagation_rows = [
        (100130, 7, 0, 1, 9.5, 5),
        (100130, 7, 0, 1, 29.5, 5),
    ]
    propagations = tmp_path / "propagations.root"
    _write_propagations(propagations, propagation_rows)
    records = load_propagation_records_ntuple_identity(propagations, index)
    # The second propagation row belongs to the THIRD physical event
    # (occurrence 2), not to the tracklet-only middle event.
    assert np.array_equal(
        records.event_id,
        [7, 7 + 2 * PHYSICAL_EVENT_UID_STRIDE],
    )


def test_ntuple_bridge_follows_has_covariance_drop_mask(tmp_path):
    # The canonical converter drops tracklets with missing covariance.  The
    # bridge must count per-entry tracklet rows through the same mask, or a
    # dropped tracklet would shift every later block.
    entries = [
        (100130, 7, 2, 1),
        (100130, 7, 2, 1),
    ]
    enhanced = tmp_path / "enhanced_tracklets.root"
    _write_enhanced_ntuple(
        enhanced,
        entries,
        validity_masks=[[True, True], [False, True]],
    )
    index = read_ntuple_event_index(enhanced)
    assert index.tracklet_count_rule == "has_covariance_mask"
    assert np.array_equal(index.n_tracklet_rows, [2, 1])

    # Flat file contains the surviving rows only: both tracklets of
    # occurrence 0, then the single valid tracklet (id 1) of occurrence 1.
    tracklets = tmp_path / "tracklets.root"
    _write_tracklets(
        tracklets,
        [
            (100130, 7, [(0, 0, 1.0, 5), (1, 1, 10.0, 5)]),
            (100130, 7, [(1, 1, 20.0, 5)]),
        ],
    )
    events = load_events_ntuple_identity(tracklets, index)
    assert [int(event.event_id) for event in events] == [
        7,
        7 + PHYSICAL_EVENT_UID_STRIDE,
    ]
    assert [int(event.size) for event in events] == [2, 1]
    assert_no_duplicate_physical_identity(events)


def test_ntuple_bridge_rejects_row_count_mismatch(tmp_path):
    entries = [(100130, 7, 2, 1)]
    enhanced = tmp_path / "enhanced_tracklets.root"
    _write_enhanced_ntuple(enhanced, entries)
    index = read_ntuple_event_index(enhanced)
    with pytest.raises(DatasetSchemaError, match="does not match"):
        flat_row_uids(
            index,
            kind="tracklets",
            n_rows=3,
            run_ids=np.asarray([100130, 100130, 100130]),
            event_ids=np.asarray([7, 7, 7]),
        )


def test_sorted_grouping_hard_fails_on_merged_rec_tracklets(tmp_path):
    # The pre-workbook-76 sorted loader must refuse this file (duplicate
    # tracklet identity), proving no silent cross-occurrence merge.
    tracklet_rows = [
        (100130, 7, [(0, 0, 1.0, 5), (1, 1, 10.0, 5)]),
        (100130, 7, [(0, 0, 2.0, 5), (1, 1, 20.0, 5)]),
    ]
    tracklets = tmp_path / "tracklets.root"
    _write_tracklets(tracklets, tracklet_rows)
    with pytest.raises(DatasetSchemaError, match="duplicate tracklet_id"):
        load_events(tracklets, require_mc_labels=True)


def test_physical_identity_evaluation_joins_correct_occurrence(tmp_path):
    from alignment.kshort_rigid_station_5dof_complementarity import (
        physical_identity_evaluation,
    )

    entries = [
        (100130, 7, 2, 1),
        (100130, 7, 2, 1),
    ]
    refit = tmp_path / "refit"
    _write_enhanced_ntuple(refit / "enhanced_tracklets.root", entries)
    _write_tracklets(
        refit / "tracklets.root",
        [
            (100130, 7, [(0, 0, 1.0, 5), (1, 1, 10.0, 5)]),
            (100130, 7, [(0, 0, 2.0, 5), (1, 1, 20.0, 5)]),
        ],
    )
    _write_propagations(
        refit / "propagations.root",
        [
            (100130, 7, 0, 1, 9.5, 5),
            (100130, 7, 0, 1, 19.5, 5),
        ],
    )
    evaluation = physical_identity_evaluation(
        refit / "tracklets.root", refit / "propagations.root", 0.99
    )
    assert evaluation.size == 2
    by_event = {
        int(evaluation.event_id[row]): float(evaluation.residual[row][0])
        for row in range(evaluation.size)
    }
    # Correct occurrence pairing: occurrence 0 has target x=10, pred 9.5;
    # occurrence 1 has target x=20, pred 19.5.  A cross-occurrence misjoin
    # would produce residual +-10 instead of +0.5.
    assert by_event[7] == pytest.approx(0.5)
    assert by_event[7 + PHYSICAL_EVENT_UID_STRIDE] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Synthetic banks and population reports
# ---------------------------------------------------------------------------


def _synthetic_bank(
    direction_matrix,
    *,
    source_id,
    split,
    run,
    pairs=90,
    seed=0,
):
    """Bank with J = G @ B: identifiable row space = row space of B.

    ``run`` must be unique across banks so that pooled (run, event) groups
    never merge distinct synthetic sources.
    """
    rng = np.random.default_rng(seed)
    basis = np.asarray(direction_matrix, dtype=np.float64)
    rank_dirs = basis.shape[0]
    generator = rng.normal(size=(pairs, 4, rank_dirs))
    derivative = generator @ basis
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = np.asarray([0.5, 0.5, 10.0, 10.0, 10.0], dtype=np.float64)
    positive = np.zeros((5, pairs, 4), dtype=np.float64)
    negative = np.zeros((5, pairs, 4), dtype=np.float64)
    for index in range(5):
        positive[index] = nominal + step[index] * derivative[:, :, index]
        negative[index] = nominal - step[index] * derivative[:, :, index]
    events = np.arange(pairs, dtype=np.int64) // 3
    truth = np.arange(pairs, dtype=np.int64) // 3 + 1
    targets = np.asarray([1, 2, 3] * (pairs // 3), dtype=np.int64)
    slopes = np.linspace(1.0e-4, 3.0e-3, pairs)
    return {
        "source_id": source_id,
        "split": split,
        "specs": (),
        "names": STATION_FIVE_NAMES,
        "scales": frozen_scales_for(STATION_FIVE_NAMES),
        "anchor_values": np.zeros(5, dtype=np.float64),
        "reference_values": np.zeros(5, dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": step,
        "negative_values": -step,
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.full(pairs, int(run), dtype=np.int64),
        "event_id": events,
        "source_tracklet_id": np.arange(pairs, dtype=np.int32),
        "target_tracklet_id": np.arange(pairs, dtype=np.int32) + 100,
        "truth_particle_id": truth,
        "source_station_id": np.zeros(pairs, dtype=np.int64),
        "target_station_id": targets,
        "layer_weights": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "overlap": {"pairs": int(pairs)},
        "source_tx": slopes,
        "source_ty": np.zeros(pairs, dtype=np.float64),
        "source_slope": slopes,
        "source_slope_missing_pairs": 0,
        "held_out_physical_points_loaded": False,
        "test_data_accessed": False,
    }


def _rank5_banks(prefix, count, split_of, run_base):
    return [
        _synthetic_bank(
            np.eye(5),
            source_id=f"{prefix}{index}",
            split=split_of(index),
            run=run_base + index,
            seed=index,
        )
        for index in range(count)
    ]


def test_population_report_native_rank_reference_not_five():
    config = _config()
    # Rank-3-stable population: every source spans the same three
    # directions; the pooled native rank is 3 and source stability is
    # referenced to 3, not to the frozen rank-5 criterion.
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    banks = [
        _synthetic_bank(
            directions,
            source_id=f"k{index}",
            split="train" if index < 2 else "validation",
            run=200000 + index,
            seed=index,
        )
        for index in range(3)
    ]
    report = population_report(banks, config=config, label="kshort_only", required_rank=None)
    assert report["pooled"]["identifiable_rank"] == 3
    assert report["stability_reference_rank"] == 3
    assert report["stability_reference"] == "population_pooled_native_rank"
    assert all(row["identifiable_rank"] == 3 for row in report["per_source"])
    assert report["source_stability"]["stable"] is True
    assert report["three_arm_opened"] is False


def test_population_report_detects_native_rank_instability():
    config = _config()
    full = np.eye(5)
    missing_rz = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    banks = [
        _synthetic_bank(full, source_id="k0", split="train", run=200000, seed=0),
        _synthetic_bank(full, source_id="k1", split="train", run=200001, seed=1),
        _synthetic_bank(missing_rz, source_id="k2", split="validation", run=200002, seed=2),
    ]
    report = population_report(banks, config=config, label="kshort_only", required_rank=None)
    assert report["pooled"]["identifiable_rank"] == 5
    assert report["source_stability"]["stable"] is False


def _reports_for_joint(config, canonical_banks, kshort_banks):
    canonical_report = population_report(
        canonical_banks, config=config, label="canonical_only", required_rank=5
    )
    kshort_report = population_report(
        kshort_banks, config=config, label="kshort_only", required_rank=None
    )
    joint_report = population_report(
        list(canonical_banks) + list(kshort_banks),
        config=config,
        label="joint_canonical_kshort",
        required_rank=5,
        loso_mode="pooled_remainder",
    )
    return canonical_report, kshort_report, joint_report


def test_joint_complementarity_passes_on_stable_rank_five_pair():
    config = _config()
    canonical = _rank5_banks("c", 3, lambda i: "train" if i < 2 else "validation", run_base=100000)
    kshort = _rank5_banks("k", 2, lambda i: "train" if i == 0 else "validation", run_base=200000)
    canonical_report, kshort_report, joint_report = _reports_for_joint(config, canonical, kshort)
    report = joint_complementarity_report(
        config,
        canonical_report=canonical_report,
        kshort_report=kshort_report,
        joint_report=joint_report,
    )
    assert report["canonical_hypothesis_core_dimension"] == 5
    assert report["independent_validation"]["pass"] is True
    assert all(bool(value) for value in report["checks"].values()), report["checks"]
    # The report driver writes this payload as JSON; no IdentifiableSubspace
    # or numpy object may leak through public keys.
    from scripts.report_kshort_rigid_station_5dof_complementarity import _strip_private

    json.dumps(json_ready(_strip_private(report)))


def test_joint_complementarity_fails_when_canonical_core_collapses():
    config = _config()
    missing_rz = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    # Two of three canonical sources miss the rz direction, so the frozen
    # consensus-eigenvalue cut (0.70) keeps the core at dimension 4.
    canonical = [
        _synthetic_bank(missing_rz, source_id="c0", split="train", run=100000, seed=0),
        _synthetic_bank(missing_rz, source_id="c1", split="train", run=100001, seed=1),
        _synthetic_bank(np.eye(5), source_id="c2", split="validation", run=100002, seed=2),
    ]
    kshort = _rank5_banks("k", 2, lambda i: "train" if i == 0 else "validation", run_base=200000)
    canonical_report, kshort_report, joint_report = _reports_for_joint(config, canonical, kshort)
    report = joint_complementarity_report(
        config,
        canonical_report=canonical_report,
        kshort_report=kshort_report,
        joint_report=joint_report,
    )
    assert report["canonical_hypothesis_core_dimension"] == 4
    assert report["checks"]["canonical_hypothesis_core_dimension_five"] is False
    assert not all(bool(value) for value in report["checks"].values())


# ---------------------------------------------------------------------------
# Campaign decision branches
# ---------------------------------------------------------------------------


def _closure(pass_=True):
    return {
        "pass": pass_,
        "failure_reasons": [] if pass_ else ["incomplete_fd_points:mc24_100130_00000_00009"],
    }


def _regression(pass_=True):
    return {
        "pass": pass_,
        "failure_reasons": [] if pass_ else ["canonical_decision_changed"],
    }


def _complementarity(checks):
    return {"checks": checks}


_ALL_CHECKS = {
    "kshort_pooled_rank_at_least": True,
    "kshort_source_stability_at_native_rank": True,
    "kshort_coverage_stability_at_native_rank": True,
    "joint_pooled_rank_five": True,
    "joint_source_stability": True,
    "joint_coverage_stability": True,
    "joint_bootstrap_stability": True,
    "canonical_hypothesis_core_dimension_five": True,
    "kshort_independent_validation_of_canonical_core": True,
}


def test_decide_campaign_pass():
    decision = decide_campaign(
        physical_closure=_closure(),
        canonical_regression_report=_regression(),
        complementarity=_complementarity(dict(_ALL_CHECKS)),
    )
    assert decision["decision"] == DECISION_COMPLEMENTARITY_PASS
    assert decision["complementarity_pass"] is True
    assert decision["real_data_correction_remains_closed"] is True


def test_decide_campaign_physical_failure_claims_no_rank():
    decision = decide_campaign(
        physical_closure=_closure(False),
        canonical_regression_report=_regression(),
        complementarity=_complementarity(dict(_ALL_CHECKS)),
    )
    assert decision["decision"] == DECISION_PHYSICAL_FAILURE
    assert decision["no_rank_claimed"] is True


def test_decide_campaign_canonical_regression_failure():
    decision = decide_campaign(
        physical_closure=_closure(),
        canonical_regression_report=_regression(False),
        complementarity=_complementarity(dict(_ALL_CHECKS)),
    )
    assert decision["decision"] == DECISION_CANONICAL_REGRESSION_FAILED
    assert decision["no_rank_claimed"] is True


def test_decide_campaign_complementarity_fail_enables_gauge_branch_only():
    checks = dict(_ALL_CHECKS)
    checks["kshort_independent_validation_of_canonical_core"] = False
    decision = decide_campaign(
        physical_closure=_closure(),
        canonical_regression_report=_regression(),
        complementarity=_complementarity(checks),
    )
    assert decision["decision"] == DECISION_COMPLEMENTARITY_FAIL
    assert decision["complementarity_pass"] is False
    assert decision["enables"] == "gauge_constrained_external_constraint_alignment_feasibility"
    assert "rank_tolerance_retune" in decision["forbidden_after_failure"]
    assert "source_dropping" in decision["forbidden_after_failure"]
    assert "pion_selection_change" in decision["forbidden_after_failure"]
