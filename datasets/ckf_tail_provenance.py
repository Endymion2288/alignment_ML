"""Task B5: CKF tail provenance / association audit.

Uses the frozen WB99/WB100 tail lists.  Does not rescreen, drop tails,
tune C/Q, design quality cuts, or enter Measurement Model V2.
Truth q/p is diagnostic only and never enters reconstruction.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_transport_diagnosis import jsonable
from datasets.acts_transport_dump import (
    DECISION_NOT_ESTABLISHED as WB98_DECISION,
    SITUATION_B,
)
from datasets.acts_transport_diagnosis import (
    CASE_C as WB99_CASE_C,
    DECISION_DIAGNOSED as WB99_DECISION,
    dump_path_for_source,
    load_dump_records,
)
from datasets.acts_transport_tail_analysis import (
    CASE_1 as WB100_CASE_1,
    DECISION_ANALYZED as WB100_DECISION,
    event_key,
    identity_key,
    is_wrong_momentum,
    load_frozen_tails,
    p_ratio_log10,
)
from datasets.qoverp_covariance_export import DECISION_ESTABLISHED as WB96_DECISION
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)

SCHEMA_VERSION = "ckf-tail-provenance-v1"
DEFAULT_CONFIG = "configs/ckf_tail_provenance_v1.yaml"
TASK = "SB-B5"
WORKBOOK = 101

CLASS_A = "wrong_association_or_identity"
CLASS_B = "ckf_reconstruction_failure"
CLASS_C = "remaining_physics_uncertainty"

NEXT_STEP = {
    CLASS_A: "association_quality_control",
    CLASS_B: "tracking_reconstruction_audit",
    CLASS_C: "acts_material_diagnosis",
}

DECISION_AUDITED = "ckf_tail_provenance_audited"

NTUPLE_BRANCHES = (
    "run",
    "eventID",
    "longTracks",
    "Track_Chi2",
    "Track_nDoF",
    "Track_p0",
    "Track_nMeasurements",
    "Track_nLayers",
    "Track_InStation0",
    "Track_InStation1",
    "Track_InStation2",
    "Track_InStation3",
    "Track_charge",
    "Track_hitSet",
    "Track_isForward",
    "Track_PropagationError",
    "TrackSegments",
    "nClusters0",
    "nClusters1",
    "nClusters2",
    "nClusters3",
    "Tracklet_station_id",
    "Tracklet_n_hit",
    "Tracklet_hit_pattern",
    "Tracklet_truth_match_fraction",
    "Tracklet_truth_particle_id",
    "Tracklet_chi2",
    "Tracklet_ndof",
)


class CkfProvenanceError(ValueError):
    """Raised when the B5 provenance contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise CkfProvenanceError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise CkfProvenanceError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise CkfProvenanceError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise CkfProvenanceError("process noise must not be adjusted to chi2")


def refuse_outlier_rejection() -> None:
    raise CkfProvenanceError("outlier rejection and chi2 clipping are forbidden")


def refuse_tail_rescreen() -> None:
    raise CkfProvenanceError("WB99/WB100 tail lists must not be rescreened")


def refuse_quality_cuts() -> None:
    raise CkfProvenanceError("quality cuts must not be designed from B5 counts")


def refuse_measurement_model_v2() -> None:
    raise CkfProvenanceError("Measurement Model V2 is not entered in Task B5")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CkfProvenanceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise CkfProvenanceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise CkfProvenanceError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise CkfProvenanceError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_rescreen_wb99_tails",
        "do_not_design_quality_cuts",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_reread_sealed_test",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise CkfProvenanceError(f"{key} must be true")
    if float(config["wrong_momentum"]["abs_log10_p_ratio_min"]) != 1.0:
        raise CkfProvenanceError("wrong-momentum decade gate must stay frozen")
    roots = {
        Path(str(config.get("output_root"))).as_posix(),
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb99_output_root"))).as_posix(),
        Path(str(config.get("wb100_output_root"))).as_posix(),
    }
    if len(roots) < 4:
        raise CkfProvenanceError("B5 must not write into WB98/WB99/WB100 artifact roots")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise CkfProvenanceError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    checks = (
        ("workbook_87", None, None, None),
        ("workbook_95", "mechanism", None, None),
        ("workbook_96", None, None, None),
        ("workbook_97", "mechanism", "failure_type", None),
        ("workbook_98", "mechanism", "failure_type", "frozen_situation"),
        ("workbook_99", None, None, None),
        ("workbook_100", None, None, None),
    )
    for name, mechanism_key, failure_key, situation_key in checks:
        spec = config["inheritance"][name]
        decision = json.loads(
            resolve_under_root(project_root(), spec["decision_path"]).read_text(
                encoding="utf-8"
            )
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{name} config",
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["decision_path"]),
            spec["decision_sha256"],
            f"{name} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise CkfProvenanceError(f"{name} decision must stay frozen")
        if mechanism_key and spec.get("frozen_mechanism"):
            if decision.get("mechanism") != spec.get("frozen_mechanism"):
                raise CkfProvenanceError(f"{name} mechanism must stay frozen")
        if failure_key and spec.get("frozen_failure_type"):
            if decision.get(failure_key) != spec.get("frozen_failure_type"):
                raise CkfProvenanceError(f"{name} failure_type must stay frozen")
        if situation_key and spec.get(situation_key):
            if decision.get("situation") != spec.get(situation_key):
                raise CkfProvenanceError(f"{name} situation must stay frozen")
        inherited[name] = {
            "decision": spec["frozen_decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
        if spec.get("frozen_mechanism"):
            inherited[name]["mechanism"] = spec["frozen_mechanism"]
        if spec.get("frozen_failure_type"):
            inherited[name]["failure_type"] = spec["frozen_failure_type"]
        if spec.get("frozen_situation"):
            inherited[name]["situation"] = spec["frozen_situation"]
        if spec.get("frozen_primary_case"):
            inherited[name]["primary_case"] = spec["frozen_primary_case"]
        if spec.get("tail_sha256"):
            inherited[name]["tail_sha256"] = spec["tail_sha256"]
        if spec.get("classification_sha256"):
            inherited[name]["classification_sha256"] = spec["classification_sha256"]
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise CkfProvenanceError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise CkfProvenanceError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise CkfProvenanceError("WB96 must keep the CKF export established")
    if inherited["workbook_98"]["decision"] != WB98_DECISION:
        raise CkfProvenanceError("WB98 decision must stay not-established")
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise CkfProvenanceError("WB98 situation must stay frozen")
    if inherited["workbook_99"]["decision"] != WB99_DECISION:
        raise CkfProvenanceError("WB99 decision must stay diagnosed")
    if inherited["workbook_99"].get("primary_case") != WB99_CASE_C:
        raise CkfProvenanceError("WB99 primary case must stay frozen")
    if inherited["workbook_100"]["decision"] != WB100_DECISION:
        raise CkfProvenanceError("WB100 decision must stay analyzed")
    if inherited["workbook_100"].get("primary_case") != WB100_CASE_1:
        raise CkfProvenanceError("WB100 primary case must stay wrong_track_state_dominated")
    spec99 = config["inheritance"]["workbook_99"]
    _expect_sha(
        resolve_under_root(project_root(), spec99["tail_path"]),
        spec99["tail_sha256"],
        "WB99 tail_provenance",
    )
    spec100 = config["inheritance"]["workbook_100"]
    _expect_sha(
        resolve_under_root(project_root(), spec100["classification_path"]),
        spec100["classification_sha256"],
        "WB100 tail classification",
    )
    return inherited


def load_frozen_classified_tails(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = load_frozen_tails(config)
    spec = config["inheritance"]["workbook_100"]
    path = resolve_under_root(project_root(), spec["classification_path"])
    _expect_sha(path, spec["classification_sha256"], "WB100 tail classification")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bool(payload.get("rescreened")):
        raise CkfProvenanceError("WB100 classification must not be rescreened")
    catalogs = {}
    for quantile in config["tail_quantiles"]:
        wb99 = frozen["catalogs"][str(quantile)]
        wb100 = (payload.get("catalogs") or {}).get(str(quantile))
        if wb100 is None:
            raise CkfProvenanceError(f"WB100 classification missing quantile {quantile}")
        events = list(wb100.get("events") or [])
        keys = [event_key(item) for item in events]
        if set(keys) != set(wb99["keys"]):
            raise CkfProvenanceError("WB100 classification keys drifted from WB99 tails")
        catalogs[str(quantile)] = {
            "quantile": float(quantile),
            "n": len(events),
            "n_retained": len(events),
            "rejected": 0,
            "rescreened": False,
            "keys": keys,
            "identity_keys": [identity_key(item) for item in events],
            "events": events,
        }
    return {
        "wb99_tail_sha256": frozen["sha256"],
        "wb100_classification_sha256": spec["classification_sha256"],
        "rescreened": False,
        "catalogs": catalogs,
    }


def _as_1d(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def load_ntuple_event_table(enhanced_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    import uproot

    with uproot.open(enhanced_path) as handle:
        arrays = handle["nt"].arrays(list(NTUPLE_BRANCHES), library="np")
    table = {}
    for i in range(len(arrays["run"])):
        table[(int(arrays["run"][i]), int(arrays["eventID"][i]))] = {
            name: arrays[name][i] for name in NTUPLE_BRANCHES
        }
    return table


def extract_reconstruction(record: Mapping[str, Any] | None, p_reco: float) -> dict[str, Any]:
    empty = {
        "n_long_tracks": 0,
        "n_segments": None,
        "n_clusters": [None, None, None, None],
        "ckf_joined": False,
        "ckf_chi2": None,
        "ckf_ndof": None,
        "ckf_chi2_per_ndof": None,
        "ckf_n_measurements": None,
        "ckf_n_layers": None,
        "ckf_p0": None,
        "ckf_charge": None,
        "in_station": [None, None, None, None],
        "n_stations_present": None,
        "hit_set": None,
        "is_forward": None,
        "propagation_error": None,
        "smoothing_status": None,
        "outlier_measurements": None,
        "material_interactions": None,
        "tracklet_truth_ids": [],
        "tracklet_truth_match": [],
        "tracklet_n_hit": [],
        "tracklet_hit_pattern": [],
        "tracklet_stations": [],
        "seed_station": None,
    }
    if record is None:
        return empty
    p0 = _as_1d(record.get("Track_p0"))
    payload = dict(empty)
    payload["n_long_tracks"] = int(p0.size)
    segs = _as_1d(record.get("TrackSegments"))
    payload["n_segments"] = int(segs[0]) if segs.size else None
    payload["n_clusters"] = [
        int(_as_1d(record.get(f"nClusters{s}"))[0])
        if _as_1d(record.get(f"nClusters{s}")).size
        else None
        for s in range(4)
    ]
    payload["propagation_error"] = (
        float(_as_1d(record.get("Track_PropagationError"))[0])
        if _as_1d(record.get("Track_PropagationError")).size
        else None
    )
    if p0.size and np.isfinite(p_reco):
        index = int(np.argmin(np.abs(p0.astype(np.float64) - float(p_reco))))
        payload["ckf_joined"] = True
        payload["ckf_p0"] = float(p0[index])
        chi2 = _as_1d(record.get("Track_Chi2"))
        ndof = _as_1d(record.get("Track_nDoF"))
        if index < chi2.size:
            payload["ckf_chi2"] = float(chi2[index])
        if index < ndof.size:
            payload["ckf_ndof"] = float(ndof[index])
        if payload["ckf_chi2"] is not None and payload["ckf_ndof"] not in (None, 0.0):
            payload["ckf_chi2_per_ndof"] = float(payload["ckf_chi2"] / payload["ckf_ndof"])
        nmeas = _as_1d(record.get("Track_nMeasurements"))
        nlay = _as_1d(record.get("Track_nLayers"))
        charge = _as_1d(record.get("Track_charge"))
        hitset = _as_1d(record.get("Track_hitSet"))
        fwd = _as_1d(record.get("Track_isForward"))
        if index < nmeas.size:
            payload["ckf_n_measurements"] = float(nmeas[index])
        if index < nlay.size:
            payload["ckf_n_layers"] = float(nlay[index])
        if index < charge.size:
            payload["ckf_charge"] = float(charge[index])
        if index < hitset.size:
            payload["hit_set"] = int(hitset[index])
        if index < fwd.size:
            payload["is_forward"] = bool(fwd[index])
        inst = []
        for station in range(4):
            values = _as_1d(record.get(f"Track_InStation{station}"))
            inst.append(int(values[index]) if index < values.size else None)
        payload["in_station"] = inst
        payload["n_stations_present"] = int(sum(1 for value in inst if value))
    stations = _as_1d(record.get("Tracklet_station_id"))
    payload["tracklet_stations"] = [int(v) for v in stations]
    payload["seed_station"] = int(stations[0]) if stations.size else None
    payload["tracklet_n_hit"] = [int(v) for v in _as_1d(record.get("Tracklet_n_hit"))]
    payload["tracklet_hit_pattern"] = [int(v) for v in _as_1d(record.get("Tracklet_hit_pattern"))]
    payload["tracklet_truth_ids"] = [
        int(v) for v in _as_1d(record.get("Tracklet_truth_particle_id"))
    ]
    payload["tracklet_truth_match"] = [
        float(v) for v in _as_1d(record.get("Tracklet_truth_match_fraction"))
    ]
    return payload


def classify_provenance_flags(
    event: Mapping[str, Any],
    reco: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    n_ckf_tracks: int,
    n_pairs_for_identity: int,
) -> dict[str, Any]:
    truth_ids = [int(v) for v in reco.get("tracklet_truth_ids") or [] if int(v) > 0]
    unique_ids = set(truth_ids)
    unmatched = any(int(v) <= 0 for v in reco.get("tracklet_truth_ids") or [])
    tmf = [float(v) for v in reco.get("tracklet_truth_match") or [] if np.isfinite(float(v))]
    low_tmf = any(value < float(config["low_truth_match_fraction_max"]) for value in tmf)
    n_seg = reco.get("n_segments")
    n_st = reco.get("n_stations_present")
    flags = {
        "duplicate_ckf_candidate": n_ckf_tracks > 1,
        "duplicate_long_track": int(reco.get("n_long_tracks") or 0) > 1,
        "conflicting_tracklet_truth_id": len(unique_ids) > 1,
        "unmatched_tracklet": unmatched,
        "low_truth_match": low_tmf,
        "not_a_long_track": int(reco.get("n_long_tracks") or 0) == 0,
        "incomplete_segments": (
            n_seg is not None and int(n_seg) < int(config["expected_n_segments"])
        ),
        "missing_station": (
            n_st is not None and int(n_st) < int(config["expected_n_stations"])
        ),
        "poor_ckf_fit": (
            reco.get("ckf_chi2_per_ndof") is not None
            and float(reco["ckf_chi2_per_ndof"]) >= float(config["poor_ckf_chi2_per_ndof_min"])
        ),
        "same_state_multi_pair": n_pairs_for_identity > 1,
        "wrong_momentum": is_wrong_momentum(
            float(event.get("p_mev", np.nan)),
            float(event.get("p_truth_mev", np.nan)),
            config,
        ),
        "qoverp_pull_anomaly": (
            event.get("pull", {}).get("q_over_p") is not None
            and np.isfinite(float(event["pull"]["q_over_p"]))
            and abs(float(event["pull"]["q_over_p"]))
            >= float(config["qoverp_pull_anomaly_abs_min"])
        ),
        "high_material_proxy": bool((event.get("flags") or {}).get("high_material_proxy")),
        "large_slope": bool((event.get("flags") or {}).get("large_slope")),
    }
    association = (
        flags["duplicate_ckf_candidate"]
        or flags["duplicate_long_track"]
        or flags["conflicting_tracklet_truth_id"]
        or flags["unmatched_tracklet"]
        or flags["low_truth_match"]
    )
    reconstruction = (
        flags["not_a_long_track"]
        or flags["incomplete_segments"]
        or flags["missing_station"]
        or flags["poor_ckf_fit"]
    )
    physics = flags["high_material_proxy"] or flags["large_slope"]
    if association:
        primary = CLASS_A
    elif reconstruction:
        primary = CLASS_B
    elif physics:
        primary = CLASS_C
    else:
        primary = CLASS_C
    return {
        "flags": flags,
        "classes": {"A": association, "B": reconstruction, "C": physics and not association and not reconstruction},
        "primary_class": primary,
    }


def build_provenance_record(
    event: Mapping[str, Any],
    reco: Mapping[str, Any],
    dump_summary: Mapping[str, Any],
    config: Mapping[str, Any],
    n_pairs_for_identity: int,
) -> dict[str, Any]:
    taxonomy = classify_provenance_flags(
        event,
        reco,
        config=config,
        n_ckf_tracks=int(dump_summary.get("n_ckf_tracks") or 0),
        n_pairs_for_identity=n_pairs_for_identity,
    )
    qop_reco = float(event.get("q_over_p_per_mev", np.nan))
    p_truth = float(event.get("p_truth_mev", np.nan))
    charge = float(event.get("charge", np.nan))
    qop_truth = (
        float(charge / p_truth)
        if np.isfinite(charge) and np.isfinite(p_truth) and p_truth > 0.0
        else None
    )
    residual = (
        float(qop_reco - qop_truth)
        if qop_truth is not None and np.isfinite(qop_reco)
        else None
    )
    return {
        "source_id": event["source_id"],
        "run_id": event["run_id"],
        "event_id": event["event_id"],
        "track_index": event.get("track_index"),
        "candidate_id": event.get("track_index"),
        "collection": dump_summary.get("collection"),
        "station_pair": event["station_pair"],
        "source_station": event.get("source_station"),
        "n_ckf_tracks": dump_summary.get("n_ckf_tracks"),
        "ckf_track_indices": dump_summary.get("track_indices"),
        "seed_station": reco.get("seed_station"),
        "n_segments": reco.get("n_segments"),
        "n_measurements": reco.get("ckf_n_measurements"),
        "n_layers": reco.get("ckf_n_layers"),
        "n_long_tracks": reco.get("n_long_tracks"),
        "in_station": reco.get("in_station"),
        "n_clusters": reco.get("n_clusters"),
        "tracklet_stations": reco.get("tracklet_stations"),
        "tracklet_n_hit": reco.get("tracklet_n_hit"),
        "tracklet_hit_pattern": reco.get("tracklet_hit_pattern"),
        "tracklet_truth_ids": reco.get("tracklet_truth_ids"),
        "tracklet_truth_match": reco.get("tracklet_truth_match"),
        "same_state_propagated_to_pairs": n_pairs_for_identity,
        "q_over_p_reco_per_mev": qop_reco if np.isfinite(qop_reco) else None,
        "q_over_p_truth_per_mev": qop_truth,
        "q_over_p_residual": residual,
        "q_over_p_pull": event.get("pull", {}).get("q_over_p"),
        "p_reco_mev": event.get("p_mev"),
        "p_truth_mev": event.get("p_truth_mev"),
        "p_log10_ratio": p_ratio_log10(
            float(event.get("p_mev", np.nan)), float(event.get("p_truth_mev", np.nan))
        ),
        "charge": charge if np.isfinite(charge) else None,
        "ckf_chi2": reco.get("ckf_chi2"),
        "ckf_ndof": reco.get("ckf_ndof"),
        "ckf_chi2_per_ndof": reco.get("ckf_chi2_per_ndof"),
        "hit_set": reco.get("hit_set"),
        "is_forward": reco.get("is_forward"),
        "propagation_error": reco.get("propagation_error"),
        "smoothing_status": reco.get("smoothing_status"),
        "outlier_measurements": reco.get("outlier_measurements"),
        "material_interactions": reco.get("material_interactions"),
        "truth_qoverp_used_in_reconstruction": False,
        "original_residual_kept": True,
        **taxonomy,
    }


def _dump_event_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indices = sorted({int(row.get("track_index", -1)) for row in rows})
    collection = rows[0].get("collection") if rows else None
    return {
        "n_ckf_tracks": len(indices),
        "track_indices": indices,
        "collection": collection,
    }


def audit_frozen_tails(
    classified: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    ntuple_tables: Mapping[str, Mapping[tuple[int, int], Any]],
    dump_rows_by_identity: Mapping[tuple[Any, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    catalogs = {}
    for quantile, cell in classified["catalogs"].items():
        identity_pairs = Counter(cell["identity_keys"])
        records = []
        for event in cell["events"]:
            ident = identity_key(event)
            reco = extract_reconstruction(
                ntuple_tables.get(str(event["source_id"]), {}).get(
                    (int(event["run_id"]), int(event["event_id"]))
                ),
                float(event.get("p_mev", np.nan)),
            )
            dump_summary = _dump_event_summary(dump_rows_by_identity.get(ident, []))
            records.append(
                build_provenance_record(
                    event,
                    reco,
                    dump_summary,
                    config,
                    n_pairs_for_identity=identity_pairs[ident],
                )
            )
        counts = Counter(item["primary_class"] for item in records)
        catalogs[quantile] = {
            "quantile": float(quantile),
            "n": len(records),
            "n_retained": len(records),
            "rejected": 0,
            "rescreened": False,
            "cuts_designed": False,
            "primary_class_counts": dict(counts),
            "n_class_a": int(sum(1 for item in records if item["classes"]["A"])),
            "n_class_b": int(sum(1 for item in records if item["classes"]["B"])),
            "n_class_c": int(sum(1 for item in records if item["classes"]["C"])),
            "n_not_a_long_track": int(sum(1 for item in records if item["flags"]["not_a_long_track"])),
            "n_same_state_multi_pair": int(
                sum(1 for item in records if item["flags"]["same_state_multi_pair"])
            ),
            "n_qoverp_pull_anomaly": int(
                sum(1 for item in records if item["flags"]["qoverp_pull_anomaly"])
            ),
            "events": records,
        }
    return {
        "wb99_tail_sha256": classified["wb99_tail_sha256"],
        "wb100_classification_sha256": classified["wb100_classification_sha256"],
        "rescreened": False,
        "catalogs": catalogs,
    }


def qoverp_failure_report(provenance: Mapping[str, Any]) -> dict[str, Any]:
    rows = provenance["catalogs"]["0.99"]["events"]
    residuals = [item["q_over_p_residual"] for item in rows if item["q_over_p_residual"] is not None]
    pulls = [item["q_over_p_pull"] for item in rows if item["q_over_p_pull"] is not None]
    ratios = [item["p_log10_ratio"] for item in rows if item["p_log10_ratio"] is not None]
    return {
        "n": len(rows),
        "truth_qoverp_used_in_reconstruction": False,
        "n_qoverp_pull_anomaly": int(sum(1 for item in rows if item["flags"]["qoverp_pull_anomaly"])),
        "n_wrong_momentum": int(sum(1 for item in rows if item["flags"]["wrong_momentum"])),
        "q_over_p_residual": {
            "mean": float(np.mean(residuals)) if residuals else None,
            "rms": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
        },
        "q_over_p_pull": {
            "mean": float(np.mean(pulls)) if pulls else None,
            "rms": float(np.sqrt(np.mean(np.square(pulls)))) if pulls else None,
            "n_abs_ge_5": int(sum(1 for value in pulls if abs(float(value)) >= 5.0)),
        },
        "p_log10_ratio": {
            "mean": float(np.mean(ratios)) if ratios else None,
            "n_decade": int(sum(1 for value in ratios if abs(float(value)) >= 1.0)),
        },
    }


def association_report(provenance: Mapping[str, Any]) -> dict[str, Any]:
    rows = provenance["catalogs"]["0.99"]["events"]
    identities = defaultdict(list)
    for item in rows:
        identities[identity_key(item)].append(item["station_pair"])
    multi = {
        f"{sid}/{run}/{ev}": sorted(set(pairs))
        for (sid, run, ev), pairs in identities.items()
        if len(set(pairs)) > 1
    }
    return {
        "n_rows": len(rows),
        "n_identities": len(identities),
        "n_multi_pair_identities": len(multi),
        "multi_pair_identities": multi,
        "n_duplicate_ckf": int(sum(1 for item in rows if item["flags"]["duplicate_ckf_candidate"])),
        "n_conflicting_truth_id": int(
            sum(1 for item in rows if item["flags"]["conflicting_tracklet_truth_id"])
        ),
        "n_unmatched_tracklet": int(sum(1 for item in rows if item["flags"]["unmatched_tracklet"])),
        "n_not_a_long_track": int(sum(1 for item in rows if item["flags"]["not_a_long_track"])),
        "n_incomplete_segments": int(sum(1 for item in rows if item["flags"]["incomplete_segments"])),
        "same_state_multi_surface": True,
        "events_dropped": 0,
    }


def decide_case(provenance: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rows = provenance["catalogs"]["0.99"]["events"]
    n = len(rows) or 1
    frac_a = float(sum(1 for item in rows if item["classes"]["A"]) / n)
    frac_b = float(sum(1 for item in rows if item["primary_class"] == CLASS_B) / n)
    frac_c = float(sum(1 for item in rows if item["primary_class"] == CLASS_C) / n)
    majority = float(config["case_gates"]["majority_fraction_min"])
    if frac_a >= majority:
        primary = CLASS_A
    elif frac_b >= majority:
        primary = CLASS_B
    elif frac_c >= majority:
        primary = CLASS_C
    else:
        primary = max(
            ((frac_a, CLASS_A), (frac_b, CLASS_B), (frac_c, CLASS_C)),
            key=lambda item: item[0],
        )[1]
    secondaries = [
        name
        for name, frac in ((CLASS_A, frac_a), (CLASS_B, frac_b), (CLASS_C, frac_c))
        if name != primary and frac >= 0.20
    ]
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "fraction_class_a": frac_a,
            "fraction_primary_b": frac_b,
            "fraction_primary_c": frac_c,
            "n_not_a_long_track": int(sum(1 for item in rows if item["flags"]["not_a_long_track"])),
            "n_same_state_multi_pair": int(
                sum(1 for item in rows if item["flags"]["same_state_multi_pair"])
            ),
        },
        "measurement_model_v2_entered": False,
        "tails_dropped": False,
        "quality_cuts_designed": False,
        "covariance_retuned": False,
    }


def _all_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(config["mc_data"]["construction_sources"]) + list(
        config["mc_data"]["validation_sources"]
    )


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    classified = load_frozen_classified_tails(config)
    needed = {identity_key(event) for cell in classified["catalogs"].values() for event in cell["events"]}
    sources = {sid for sid, _, _ in needed}
    ntuple_tables: dict[str, Any] = {}
    dump_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    present = []
    missing = []
    for spec in _all_sources(config):
        source_id = spec["source_id"]
        if source_id not in sources:
            continue
        split = (
            "train"
            if source_id in config["mc_data"]["construction_source_ids"]
            else "validation"
        )
        authorize_path(spec["input_xaod"], AccessScope.DEVELOPMENT_VALIDATION, split=split)
        path = dump_path_for_source(config, source_id)
        rows = load_dump_records(path, split=split)
        if not rows:
            missing.append(source_id)
            continue
        present.append(source_id)
        for row in rows:
            dump_by_identity[identity_key(row)].append(row)
        refit = resolve_under_root(
            project_root(),
            str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
        )
        ntuple_tables[source_id] = load_ntuple_event_table(
            refit / config["mc_data"]["enhanced_file"]
        )
    provenance = audit_frozen_tails(
        classified,
        config=config,
        ntuple_tables=ntuple_tables,
        dump_rows_by_identity=dump_by_identity,
    )
    qop = qoverp_failure_report(provenance)
    association = association_report(provenance)
    mechanism = decide_case(provenance, config)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "ckf_tail_provenance": provenance,
        "qoverp_failure": qop,
        "association": association,
        "mechanism": mechanism,
        "wb98_dumps_read_only": True,
        "wb99_tails_read_only": True,
        "wb100_classification_read_only": True,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("tails_dropped")):
        refuse_outlier_rejection()
    if bool(mechanism.get("quality_cuts_designed")):
        refuse_quality_cuts()
    return {
        "verdict": "AUDITED",
        "decision": DECISION_AUDITED,
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "closure_pass": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb100_decision": inherited["workbook_100"]["decision"],
        "inherited_wb100_primary_case": inherited["workbook_100"]["primary_case"],
        "inherited_wb100_decision_sha256": inherited["workbook_100"]["decision_sha256"],
        "inherited_wb100_classification_sha256": inherited["workbook_100"]["classification_sha256"],
        "inherited_wb99_tail_sha256": inherited["workbook_99"]["tail_sha256"],
        "wb87_through_wb100_rewritten": False,
        "evidence": mechanism["evidence"],
    }
