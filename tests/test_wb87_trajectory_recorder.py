"""WB87 trajectory bookkeeping.  Uses the hermetic linear double, not Calypso."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alignment.physical_common_track_execution import OFFICIAL_ENGINE, stacked_observations
from alignment.wb85_physical_qualification_protocol import chart_parameter_names
from alignment.wb86_official_execution import json_payload, true_payload
from alignment.wb87_convergence_autopsy import record_frozen_trajectory
from tests.test_wb86_physical_qualification import _LinearOfficialDouble


class _RecordingLinearDouble:
    """Official-shaped cache over the hermetic linear response.  Not Calypso."""

    engine_name = OFFICIAL_ENGINE
    is_official = True

    def __init__(self, truth, names, work_root: Path) -> None:
        self.work_root = work_root
        self._inner = _LinearOfficialDouble(truth, names)
        self._cache: dict[str, object] = {}
        self.accepted: list[dict] = []
        self.n_official_refits = 0
        self.last_refit = None

    def _cached(self, event, payload, kind: str):
        from alignment.wb86_official_execution import payload_digest

        packed = {int(station): tuple(float(value) for value in payload[int(station)]) for station in payload}
        key = payload_digest(packed)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._inner.rerefit_and_repropagate(event, payload, work_dir=self.work_root)
            self._cache[key] = cached
            self.n_official_refits += 1
            self.accepted.append(
                {
                    "kind": kind,
                    "payload_sha256": key,
                    "stacked": {
                        int(particle): np.asarray(values, dtype=np.float64).copy()
                        for particle, values in stacked_observations(cached.tracks).items()
                    },
                }
            )
        return cached

    def predict(self, event, payload, measurements):
        return stacked_observations(self._cached(event, payload, "predict").tracks)

    def rerefit_and_repropagate(self, event, payload, *, work_dir):
        return self._cached(event, payload, "rerefit")


def _inputs():
    family = "identifiable_translation"
    survey = "fixed_dz"
    chart = "translation"
    names = list(chart_parameter_names(chart, survey))
    covariance = (np.eye(4) * 1.0e-4).tolist()
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
                "covariance_4x4": covariance,
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
        "tracks": [{"track_uid": "t0", "truth": {"particle_id": 7, "match_fraction": 1.0}, "hits": hits}],
    }
    return record, family, survey, chart, names


def test_recorder_keeps_per_iteration_fields(tmp_path, monkeypatch):
    record, family, survey, chart, names = _inputs()
    backend = _RecordingLinearDouble(true_payload(family), names, tmp_path)
    monkeypatch.setattr(
        "alignment.wb87_convergence_autopsy.RecordingOfficialBackend",
        lambda *args, **kwargs: backend,
    )
    result = record_frozen_trajectory(
        record,
        family=family,
        survey_mode=survey,
        chart_kind=chart,
        work_dir=tmp_path,
    )
    assert result["qualifies_wb86"] is False
    assert result["history"]
    recorded = [row for row in result["history"] if row["trajectory_status"] == "recorded"]
    assert recorded
    assert all(row["rho"] < 1.0e-6 for row in recorded)
    assert result["history"][0]["trajectory_status"] == "recorded"
    assert result["history"][0]["validation_relative_change"] is None
    assert result["history"][1]["trajectory_status"] == "recorded"
    assert result["history"][2]["trajectory_status"] == "cached_repeat"
    assert all(row["trajectory_status"] == "cached_repeat" for row in result["history"][2:])
    assert all(row["update_rerefit_cache"] == "hit" for row in result["history"][2:])
    zero = result["history"][3]
    assert zero["scaled_update_norm"] == 0.0
    assert zero["chi2"] == 0.0
    assert zero["previous_chi2"] == 0.0
    assert zero["validation_relative_change"] is None
    assert zero["consecutive_convergence_count"] == 0
    assert "J_delta_norm" in result["history"][0]
    assert "actual_prediction_change_norm" in result["history"][0]
    assert "linearization_error_norm" in result["history"][0]
    assert result["history"][0]["right_singular_vectors"] == "spectrum_unavailable"
    assert abs(result["theta_hat"][names.index("s1_dx_mm")] - 0.30) < 5.0e-2
    assert json_payload({0: (0.0,) * 6})
