"""WB86 execution-only tests.  No Calypso job and no WB84 outcome inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.physical_common_track_execution import (
    OFFICIAL_ENGINE,
    PhysicalHit,
    PhysicalMeasurements,
    PhysicalTrack,
    identity_payload,
)
from alignment.wb85_physical_qualification_protocol import (
    ENGINEERING_TRANSLATION_MM,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    chart_parameter_names,
    identifiable_strata,
    weak_strata,
)
from alignment.wb85a_protocol_qa import healthy_identifiable_row
from alignment.wb85b_fwer_protocol import CRITICAL_COV, CRITICAL_LS, evaluate_wb85b_qualification
from alignment.wb86_official_execution import (
    WB86OfficialPhysicalBackend,
    chart_vector_from_payload,
    iterate_official_event,
    payloads_close,
    true_payload,
)
from alignment.wb86_physical_qualification import (
    EXPECTED_HASHES,
    SOURCE_FREEZE_COMMIT,
    WB86Error,
    chart_kind_for_family,
    decision_function,
    execution_layer,
    locked_execution_flags,
    physics_screening,
    refuse_wb84_wb85b_rewrite,
    require_remote_verification,
    snapshot_verify,
    split_condition_id,
)


class _LinearOfficialDouble(WB86OfficialPhysicalBackend):
    """Hermetic official-engine double.  Not a Calypso job."""

    def __init__(self, truth, names):
        self.engine_name = OFFICIAL_ENGINE
        self.is_official = True
        self.work_root = Path("/tmp/wb86_test")
        self.n_official_refits = 0
        self.last_refit = {"executor": "test_double", "sqlite_sha256": "0" * 64}
        self._truth = {int(station): tuple(float(v) for v in values) for station, values in truth.items()}
        self._names = list(names)
        self._seq = 0
        self._cache = {}

    def predict(self, event, payload, measurements):
        return {
            int(track.truth_particle_id): self._h(payload, track)
            for track in event.tracks
        }

    def rerefit_and_repropagate(self, event, payload, *, work_dir):
        self.n_official_refits += 1
        tracks = []
        for track in event.tracks:
            predicted = self._h(payload, track)
            hits = []
            offset = 0
            for old in track.hits:
                hits.append(
                    PhysicalHit(
                        station=old.station,
                        measurement_id=old.measurement_id,
                        observed=predicted[offset : offset + 4],
                        covariance=old.covariance,
                        z_mm=old.z_mm,
                        q_over_p=old.q_over_p,
                    )
                )
                offset += 4
            tracks.append(
                PhysicalTrack(
                    track_uid=track.track_uid,
                    truth_particle_id=track.truth_particle_id,
                    match_fraction=track.match_fraction,
                    hits=tuple(hits),
                )
            )
        packed = {int(station): tuple(float(v) for v in payload[int(station)]) for station in payload}
        return PhysicalMeasurements(
            event_uid=event.event_uid,
            payload=packed,
            tracks=tuple(tracks),
            engine=OFFICIAL_ENGINE,
            provenance="wb86_official_calypso_acts_rerefit",
        )

    def _h(self, payload, track):
        observed = []
        truth = chart_vector_from_payload(self._truth, self._names)
        trial = chart_vector_from_payload(payload, self._names)
        delta = np.asarray(truth, dtype=np.float64) - np.asarray(trial, dtype=np.float64)
        for hit in track.hits:
            extra = np.zeros(4, dtype=np.float64)
            for index, name in enumerate(self._names):
                if name.endswith("dx_mm") and f"s{hit.station}_" in name:
                    extra[0] += delta[index]
                if name.endswith("dy_mm") and f"s{hit.station}_" in name:
                    extra[1] += delta[index]
                if name.endswith("rz_mrad") and f"s{hit.station}_" in name:
                    extra[2] += 0.01 * delta[index]
            observed.append(hit.observed + extra)
        return np.concatenate(observed)


def _healthy_bundle(n: int = 220) -> dict:
    z = {}
    coverage = {}
    rows = {}
    rng = np.random.default_rng(20260908)
    for row in identifiable_strata():
        values = rng.normal(0.0, 1.0, size=n)
        z[row["stratum_id"]] = [float(v) for v in values]
        coverage[row["stratum_id"]] = (int(round(0.95 * n)), n)
        rows[row["stratum_id"]] = healthy_identifiable_row(n, int(round(0.95 * n)))
    weak = {
        row["stratum_id"]: {"standardized_bias_dx": 0.1, "standardized_bias_ry": -0.2}
        for row in weak_strata()
    }
    return {"z": z, "coverage": coverage, "identifiable_strata": rows, "weak_strata": weak}


def test_locked_flags_authorize_wb86_only():
    flags = locked_execution_flags()
    assert flags["qualification_authorized"] is True
    assert flags["executable"] is True
    assert flags["alignment_oracle_qualified_for_physical_FASER"] is False
    assert flags["ml_alignment_eval_authorized"] is False
    assert flags["looked_at_physical_alignment_outcomes_before_authorization"] is False
    assert flags["uses_evaluate_wb85b_qualification"] is True


def test_decision_function_is_frozen_wb85b():
    assert decision_function() is evaluate_wb85b_qualification


def test_refuse_historical_rewrite():
    with pytest.raises(Exception):
        refuse_wb84_wb85b_rewrite("outputs/mc24_four_station_wb85b_fwer_protocol_v1/x.json")
    with pytest.raises(Exception):
        refuse_wb84_wb85b_rewrite("outputs/mc24_four_station_calypso_physical_replicas_v1/x.json")


def test_condition_and_chart_maps():
    assert split_condition_id("identifiable_translation_fixed_dz") == (
        "identifiable_translation",
        "fixed_dz",
    )
    assert chart_kind_for_family("identifiable_rotation") == "rotation"
    assert chart_kind_for_family("weak_jg_diagnostic") == "weak_jg"


def test_remote_verification_is_present():
    payload = require_remote_verification()
    assert payload["remote_commit_verification"] is True
    assert payload["post_run_source_freeze_commit"] == SOURCE_FREEZE_COMMIT


def test_snapshot_hashes_match_frozen_expected():
    report = snapshot_verify()
    assert report["hashes_expected"] == EXPECTED_HASHES
    assert report["snapshot_verify_pass"] is True
    assert report["identity"]["critical_LS"] == pytest.approx(CRITICAL_LS)
    assert report["identity"]["critical_cov"] == pytest.approx(CRITICAL_COV)


def test_three_layer_unknown_when_n_below_200():
    bundle = _healthy_bundle(150)
    execution = execution_layer(bundle, snapshot_pass=True, corpus_pass=True, integrity_pass=True)
    assert execution["status"] == "UNKNOWN"
    assert N_MIN_PER_IDENTIFIABLE_STRATUM == 200
    statistical = evaluate_wb85b_qualification(bundle)
    assert statistical["qualification_status"] == "UNKNOWN"


def test_physics_screening_is_research_only():
    bundle = _healthy_bundle()
    name = next(row["stratum_id"] for row in identifiable_strata() if row["family"] == "identity")
    bundle["identifiable_strata"][name]["engineering_bias"] = ENGINEERING_TRANSLATION_MM + 0.05
    screening = physics_screening(bundle)
    assert screening["status"] == "FAIL"
    assert screening["research_screening_not_collaboration_approved"] is True


def test_gate_uses_wb85b_not_legacy_alpha():
    bundle = _healthy_bundle()
    result = evaluate_wb85b_qualification(bundle)
    assert result["qualification_status"] == "PASS"
    assert result["decision_family"] == "primary_statistical_bonferroni"
    assert result["wb85_accepted_flags_ignored"]["location_scale_gof.accepted"] in {True, False}


def test_official_backend_does_not_refuse_by_feeding_forbidden_needles():
    from alignment.physical_common_track_execution import refuse_toy_official_fallback

    with pytest.raises(Exception):
        refuse_toy_official_fallback("toy_uniform_By")
    refuse_toy_official_fallback("WB85bOfficialCalypsoActsBackend.execute_refit")


def test_load_replica_record_may_read_frozen_wb84_corpus():
    from alignment.wb86_physical_qualification import load_catalog_event, load_replica_record

    row = load_catalog_event(0)
    record = load_replica_record(row)
    assert record["event_uid"] == row["event_uid"]
    assert record["input_xaod"] == row["input_xaod"]
    assert "tracks" in record


def test_payloads_close_and_linear_iterate(tmp_path):
    family = "identifiable_translation"
    survey = "fixed_dz"
    chart = "translation"
    truth = true_payload(family)
    names = list(chart_parameter_names(chart, survey))
    cov = np.eye(4) * 1.0e-4
    hits = []
    for station in (0, 1, 2, 3):
        hits.append(
            {
                "station_id": station,
                "measurement_id": f"m{station}",
                "x_mm": 1.0 + 0.1 * station,
                "y_mm": -0.2,
                "tx": 0.0,
                "ty": 0.0,
                "covariance_4x4": cov.tolist(),
                "z_mm": 1000.0 * station,
                "q_over_p_per_mev": 0.001,
            }
        )
    record = {
        "event_uid": "test:1:1",
        "source_id": "mc24_100043_00200_00299",
        "run_id": 1,
        "event_id": 1,
        "xaod_entry_index": 0,
        "condition_id": f"{family}_{survey}",
        "input_xaod": "/tmp/missing.xAOD.root",
        "tracks": [
            {
                "track_uid": "t0",
                "truth": {"particle_id": 7, "match_fraction": 1.0},
                "hits": hits,
            }
        ],
    }
    backend = _LinearOfficialDouble(truth, names)
    result = iterate_official_event(
        record,
        family=family,
        survey_mode=survey,
        chart_kind=chart,
        work_dir=tmp_path,
        backend=backend,
    )
    assert result["engine"] == OFFICIAL_ENGINE
    assert result["reused_first_step_measurements"] is False
    assert result["automatically_passed_at_max_iterations"] is False
    assert result["solver_ok"] is True
    assert abs(result["theta_hat"][0] - 0.30) < 5.0e-2
    assert payloads_close(identity_payload(), identity_payload())
