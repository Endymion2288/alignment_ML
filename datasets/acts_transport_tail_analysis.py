"""Task B4: tail / uncertainty model analysis.

Uses the frozen WB99 top 1% and 0.1% lists.  Does not rescreen, drop,
clip, inflate C, rescale Q, or enter Measurement Model V2.
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
from alignment.propagated_covariance_closure import _load_truth_table
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_process_noise_contract import MODEL1, _as_matrix
from datasets.acts_transport_diagnosis import (
    CASE_C as WB99_CASE_C,
    DECISION_DIAGNOSED as WB99_DECISION,
    _bin_name,
    _eigen_spectrum,
    _safe_corr,
    _stats,
    collect_matched_events,
    dump_path_for_source,
    jsonable,
    load_dump_records,
)
from datasets.acts_transport_dump import (
    DECISION_NOT_ESTABLISHED as WB98_DECISION,
    SITUATION_B,
)
from datasets.qoverp_covariance_export import DECISION_ESTABLISHED as WB96_DECISION
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)

SCHEMA_VERSION = "acts-transport-tail-analysis-v1"
DEFAULT_CONFIG = "configs/acts_transport_tail_analysis_v1.yaml"
TASK = "SB-B4"
WORKBOOK = 100

PULL_NAMES = ("x", "y", "tx", "ty", "q_over_p")

CLASS_A = "track_reconstruction_anomaly"
CLASS_B = "transport_anomaly"
CLASS_C = "geometry_material_anomaly"
CLASS_UNEXPLAINED = "unexplained"

CASE_1 = "wrong_track_state_dominated"
CASE_2 = "material_slope_correlated_tail"
CASE_3 = "uncertainty_model_mismatch"

NEXT_STEP = {
    CASE_1: "reconstruction_association_quality_control",
    CASE_2: "acts_material_diagnosis",
    CASE_3: "reassess_measurement_model_v2_input_model",
}

DECISION_ANALYZED = "acts_transport_tail_uncertainty_analyzed"


class TailAnalysisError(ValueError):
    """Raised when the B4 analysis contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise TailAnalysisError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise TailAnalysisError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise TailAnalysisError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise TailAnalysisError("process noise must not be adjusted to chi2")


def refuse_outlier_rejection() -> None:
    raise TailAnalysisError("outlier rejection and chi2 clipping are forbidden")


def refuse_tail_rescreen() -> None:
    raise TailAnalysisError("WB99 tail lists must not be rescreened")


def refuse_highland_replacement() -> None:
    raise TailAnalysisError("Highland Q must not replace ACTS Q")


def refuse_measurement_model_v2() -> None:
    raise TailAnalysisError("Measurement Model V2 is not entered in Task B4")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TailAnalysisError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TailAnalysisError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TailAnalysisError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise TailAnalysisError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_replace_acts_q_with_highland",
        "do_not_delete_state_variables",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_rescreen_wb99_tails",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_reread_sealed_test",
        "do_not_start_new_source_campaign",
        "do_not_write_geometry_or_conditions_payload",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise TailAnalysisError(f"{key} must be true")
    bins = config.get("qoverp_bins_abs_per_mev") or {}
    expected = {
        "high_momentum": [0.0, 1.0e-6],
        "medium_momentum": [1.0e-6, 5.0e-6],
        "low_momentum": [5.0e-6, 1.0],
    }
    for name, edges in expected.items():
        got = [float(v) for v in bins.get(name, [])]
        if got != edges:
            raise TailAnalysisError(f"q/p bin {name} must stay frozen")
    if float(config["wrong_momentum"]["abs_log10_p_ratio_min"]) != 1.0:
        raise TailAnalysisError("wrong-momentum decade gate must stay frozen at 1.0")
    if Path(str(config.get("output_root"))).as_posix() in {
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb99_output_root"))).as_posix(),
    }:
        raise TailAnalysisError("B4 must not write into WB98 dumps or WB99 artifacts")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise TailAnalysisError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    checks = (
        ("workbook_87", None, None, None),
        ("workbook_95", "mechanism", None, None),
        ("workbook_96", None, None, None),
        ("workbook_97", "mechanism", "failure_type", None),
        ("workbook_98", "mechanism", "failure_type", "frozen_situation"),
        ("workbook_99", None, None, None),
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
            raise TailAnalysisError(f"{name} decision must stay frozen")
        if mechanism_key and spec.get("frozen_mechanism"):
            if decision.get("mechanism") != spec.get("frozen_mechanism"):
                raise TailAnalysisError(f"{name} mechanism must stay frozen")
        if failure_key and spec.get("frozen_failure_type"):
            if decision.get(failure_key) != spec.get("frozen_failure_type"):
                raise TailAnalysisError(f"{name} failure_type must stay frozen")
        if situation_key and spec.get(situation_key):
            if decision.get("situation") != spec.get(situation_key):
                raise TailAnalysisError(f"{name} situation must stay frozen")
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
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise TailAnalysisError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise TailAnalysisError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise TailAnalysisError("WB96 must keep the CKF export established")
    if inherited["workbook_98"]["decision"] != WB98_DECISION:
        raise TailAnalysisError("WB98 decision must stay not-established")
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise TailAnalysisError("WB98 situation must stay acts_q_materialized_closure_failed")
    if inherited["workbook_99"]["decision"] != WB99_DECISION:
        raise TailAnalysisError("WB99 decision must stay diagnosed")
    if inherited["workbook_99"].get("primary_case") != WB99_CASE_C:
        raise TailAnalysisError("WB99 primary case must stay high_chi2_tail_dominated")
    tail_spec = config["inheritance"]["workbook_99"]
    _expect_sha(
        resolve_under_root(project_root(), tail_spec["tail_path"]),
        tail_spec["tail_sha256"],
        "WB99 tail_provenance",
    )
    return inherited


def event_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_id"]),
        int(row["run_id"]),
        int(row["event_id"]),
        str(row["station_pair"]),
        int(row.get("track_index", -1)),
    )


def identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row["source_id"]), int(row["run_id"]), int(row["event_id"]))


def load_frozen_tails(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["inheritance"]["workbook_99"]
    path = resolve_under_root(project_root(), spec["tail_path"])
    _expect_sha(path, spec["tail_sha256"], "WB99 tail_provenance")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bool(payload.get("outlier_rejection")):
        raise TailAnalysisError("WB99 tail file must not reject outliers")
    official = payload.get("tails") or {}
    catalogs = {}
    for quantile in config["tail_quantiles"]:
        cell = official.get(str(quantile))
        if cell is None:
            raise TailAnalysisError(f"frozen tail quantile {quantile} is absent")
        events = list(cell.get("events") or [])
        catalogs[str(quantile)] = {
            "quantile": float(quantile),
            "n": len(events),
            "n_retained": int(cell.get("n_retained", len(events))),
            "rejected": int(cell.get("rejected", 0)),
            "clipped": int(cell.get("clipped", 0)),
            "chi2_share": cell.get("chi2_share"),
            "threshold_chi2_per_ndof": cell.get("threshold_chi2_per_ndof"),
            "keys": [event_key(item) for item in events],
            "identity_keys": [identity_key(item) for item in events],
            "events": events,
        }
        if catalogs[str(quantile)]["rejected"] or catalogs[str(quantile)]["clipped"]:
            raise TailAnalysisError("frozen tails were rejected or clipped")
    return {
        "path": str(spec["tail_path"]),
        "sha256": spec["tail_sha256"],
        "rescreened": False,
        "catalogs": catalogs,
    }


def p_ratio_log10(p_reco: float, p_truth: float) -> float | None:
    if not np.isfinite(p_reco) or not np.isfinite(p_truth) or p_reco <= 0.0 or p_truth <= 0.0:
        return None
    return float(np.log10(p_reco / p_truth))


def is_wrong_momentum(p_reco: float, p_truth: float, config: Mapping[str, Any]) -> bool:
    ratio = p_ratio_log10(p_reco, p_truth)
    if ratio is None:
        return False
    return abs(ratio) >= float(config["wrong_momentum"]["abs_log10_p_ratio_min"])


def _dump_index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index = {}
    for row in records:
        block = row.get(MODEL1) or {}
        key = (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
            f"({int(row.get('source_station', -1))},{int(row.get('target_station', -1))})",
            int(row.get("track_index", -1)),
        )
        index[key] = row
        if block:
            index[key] = row
    return index


def _load_track_quality(enhanced_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    import uproot

    branches = [
        "run",
        "eventID",
        "Track_Chi2",
        "Track_nDoF",
        "Track_p0",
        "Track_nMeasurements",
        "Track_nLayers",
        "Tracklet_station_id",
        "Tracklet_chi2",
        "Tracklet_ndof",
        "Tracklet_truth_match_fraction",
    ]
    with uproot.open(enhanced_path) as handle:
        arrays = handle["nt"].arrays(branches, library="np")
    table: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(len(arrays["run"])):
        table[(int(arrays["run"][i]), int(arrays["eventID"][i]))] = {
            "track_chi2": np.asarray(arrays["Track_Chi2"][i], dtype=np.float64),
            "track_ndof": np.asarray(arrays["Track_nDoF"][i], dtype=np.float64),
            "track_p0": np.asarray(arrays["Track_p0"][i], dtype=np.float64),
            "track_n_measurements": np.asarray(arrays["Track_nMeasurements"][i], dtype=np.float64),
            "track_n_layers": np.asarray(arrays["Track_nLayers"][i], dtype=np.float64),
            "tracklet_station": np.asarray(arrays["Tracklet_station_id"][i], dtype=np.int32),
            "tracklet_chi2": np.asarray(arrays["Tracklet_chi2"][i], dtype=np.float64),
            "tracklet_ndof": np.asarray(arrays["Tracklet_ndof"][i], dtype=np.float64),
            "tracklet_truth_match": np.asarray(
                arrays["Tracklet_truth_match_fraction"][i], dtype=np.float64
            ),
        }
    return table


def _attach_quality(
    event: Mapping[str, Any],
    quality: Mapping[tuple[int, int], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_long_tracks": None,
        "ckf_chi2": None,
        "ckf_ndof": None,
        "ckf_chi2_per_ndof": None,
        "ckf_n_measurements": None,
        "ckf_n_layers": None,
        "source_tracklet_chi2": None,
        "source_tracklet_ndof": None,
        "source_tracklet_truth_match_fraction": None,
    }
    if quality is None:
        return payload
    entry = quality.get((int(event["run_id"]), int(event["event_id"])))
    if entry is None:
        return payload
    p0 = np.asarray(entry["track_p0"], dtype=np.float64)
    payload["n_long_tracks"] = int(p0.size)
    if p0.size:
        p_reco = float(event.get("p_mev", np.nan))
        index = int(np.argmin(np.abs(p0 - p_reco))) if np.isfinite(p_reco) else 0
        chi2 = float(entry["track_chi2"][index]) if index < len(entry["track_chi2"]) else None
        ndof = float(entry["track_ndof"][index]) if index < len(entry["track_ndof"]) else None
        payload["ckf_chi2"] = chi2
        payload["ckf_ndof"] = ndof
        payload["ckf_chi2_per_ndof"] = (
            float(chi2 / ndof) if chi2 is not None and ndof not in (None, 0.0) else None
        )
        if index < len(entry["track_n_measurements"]):
            payload["ckf_n_measurements"] = float(entry["track_n_measurements"][index])
        if index < len(entry["track_n_layers"]):
            payload["ckf_n_layers"] = float(entry["track_n_layers"][index])
    stations = np.asarray(entry["tracklet_station"], dtype=np.int32)
    source = int(event.get("source_station", 0))
    matches = np.nonzero(stations == source)[0]
    if matches.size:
        pick = int(matches[0])
        payload["source_tracklet_chi2"] = float(entry["tracklet_chi2"][pick])
        payload["source_tracklet_ndof"] = float(entry["tracklet_ndof"][pick])
        payload["source_tracklet_truth_match_fraction"] = float(
            entry["tracklet_truth_match"][pick]
        )
    return payload


def _transport_fields(dump_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if dump_row is None:
        return {
            "c0_jacobian_frobenius_rel": None,
            "transport_jacobian": None,
            "q_not_psd": None,
            "q_min_eigenvalue": None,
            "lever_arm_mm": None,
            "f_tx_lever": None,
            "surface": None,
        }
    jacobian = _as_matrix(dump_row.get("transport_jacobian"), 5)
    q_mat = _as_matrix(dump_row.get("process_noise"), 5)
    spectrum = _eigen_spectrum(q_mat) if q_mat is not None else None
    source_z = float(dump_row.get("source_z_mm", np.nan))
    target_z = float(dump_row.get("target_z_mm", np.nan))
    lever = target_z - source_z if np.isfinite(source_z) and np.isfinite(target_z) else None
    f_tx = float(jacobian[0, 2]) if jacobian is not None else None
    return {
        "c0_jacobian_frobenius_rel": dump_row.get("c0_jacobian_frobenius_rel"),
        "transport_jacobian": jacobian.tolist() if jacobian is not None else None,
        "q_not_psd": None if spectrum is None else (not spectrum["psd"]),
        "q_min_eigenvalue": None if spectrum is None else spectrum["min"],
        "lever_arm_mm": lever,
        "f_tx_lever": f_tx,
        "surface": dump_row.get("surface"),
        "source_z_mm": source_z,
        "target_z_mm": target_z,
    }


def classify_tail_event(
    event: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    repeated: bool,
    dump_row: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    quality = quality or {}
    transport = _transport_fields(dump_row)
    log_ratio = p_ratio_log10(float(event.get("p_mev", np.nan)), float(event.get("p_truth_mev", np.nan)))
    qop_pull = event.get("pull", {}).get("q_over_p")
    jacobian_rel = transport.get("c0_jacobian_frobenius_rel")
    lever = transport.get("lever_arm_mm")
    f_tx = transport.get("f_tx_lever")
    lever_rel = None
    if lever not in (None, 0.0) and f_tx is not None:
        lever_rel = abs(float(f_tx) - float(lever)) / abs(float(lever))
    flags = {
        "wrong_momentum": is_wrong_momentum(
            float(event.get("p_mev", np.nan)), float(event.get("p_truth_mev", np.nan)), config
        ),
        "qoverp_pull_anomaly": (
            qop_pull is not None
            and np.isfinite(float(qop_pull))
            and abs(float(qop_pull)) >= float(config["qoverp_pull_anomaly_abs_min"])
        ),
        "poor_ckf_fit": (
            quality.get("ckf_chi2_per_ndof") is not None
            and float(quality["ckf_chi2_per_ndof"]) >= float(config["poor_ckf_chi2_per_ndof_min"])
        ),
        "low_truth_match": (
            quality.get("source_tracklet_truth_match_fraction") is not None
            and float(quality["source_tracklet_truth_match_fraction"])
            < float(config["low_truth_match_fraction_max"])
        ),
        "repeated_event": bool(repeated),
        "jacobian_anomaly": (
            jacobian_rel is not None
            and np.isfinite(float(jacobian_rel))
            and float(jacobian_rel) >= float(config["jacobian_anomaly_rel_min"])
        ),
        "lever_arm_anomaly": (
            lever_rel is not None and lever_rel >= float(config["lever_arm_rel_anomaly_min"])
        ),
        "q_not_psd": bool(transport.get("q_not_psd")),
        "high_material_proxy": (
            event.get("q_frobenius") is not None
            and float(event["q_frobenius"]) >= float(config["q_frobenius_bins"]["high"][0])
        ),
        "large_slope": (
            event.get("abs_slope") is not None
            and float(event["abs_slope"]) >= float(config["incidence_abs_slope_bins"]["large"][0])
        ),
        "low_momentum": (
            event.get("abs_q_over_p") is not None
            and float(event["abs_q_over_p"]) >= float(config["qoverp_bins_abs_per_mev"]["low_momentum"][0])
        ),
    }
    class_a = flags["wrong_momentum"] or flags["qoverp_pull_anomaly"] or flags["poor_ckf_fit"] or flags["low_truth_match"]
    class_b = flags["jacobian_anomaly"] or flags["lever_arm_anomaly"]
    class_c = flags["high_material_proxy"] or flags["large_slope"]
    if class_a:
        primary = CLASS_A
    elif class_b:
        primary = CLASS_B
    elif class_c:
        primary = CLASS_C
    else:
        primary = CLASS_UNEXPLAINED
    return {
        **dict(event),
        **quality,
        "p_log10_ratio": log_ratio,
        "repeated_event": repeated,
        "flags": flags,
        "classes": {
            "A": class_a,
            "B": class_b,
            "C": class_c,
        },
        "primary_class": primary,
        "transport": {
            "c0_jacobian_frobenius_rel": jacobian_rel,
            "q_not_psd": transport.get("q_not_psd"),
            "q_min_eigenvalue": transport.get("q_min_eigenvalue"),
            "lever_arm_mm": lever,
            "f_tx_lever": f_tx,
            "lever_arm_rel_error": lever_rel,
            "surface": transport.get("surface"),
        },
        "original_residual_kept": True,
        "original_covariance_kept": True,
        "original_q_kept": True,
    }


def classify_frozen_tails(
    frozen: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    dump_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    quality_tables: Mapping[str, Mapping[tuple[int, int], Any]],
) -> dict[str, Any]:
    catalogs = {}
    for quantile, cell in frozen["catalogs"].items():
        identity_counts = Counter(cell["identity_keys"])
        classified = []
        for event in cell["events"]:
            key = event_key(event)
            quality = _attach_quality(
                event, quality_tables.get(str(event["source_id"]))
            )
            classified.append(
                classify_tail_event(
                    event,
                    config=config,
                    repeated=identity_counts[identity_key(event)] > 1,
                    dump_row=dump_index.get(key),
                    quality=quality,
                )
            )
        counts = Counter(item["primary_class"] for item in classified)
        catalogs[quantile] = {
            "quantile": float(quantile),
            "n": len(classified),
            "n_retained": len(classified),
            "rejected": 0,
            "rescreened": False,
            "primary_class_counts": dict(counts),
            "n_wrong_momentum": int(sum(1 for item in classified if item["flags"]["wrong_momentum"])),
            "n_repeated_identity": int(sum(1 for item in classified if item["repeated_event"])),
            "n_class_a": int(sum(1 for item in classified if item["classes"]["A"])),
            "n_class_b": int(sum(1 for item in classified if item["classes"]["B"])),
            "n_class_c": int(sum(1 for item in classified if item["classes"]["C"])),
            "events": classified,
        }
    return {
        "frozen_tail_sha256": frozen["sha256"],
        "rescreened": False,
        "catalogs": catalogs,
    }


def reconstruction_quality_report(
    official_events: Sequence[Mapping[str, Any]],
    classified_tails: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    official = [event for event in official_events if event.get("on_frozen_station_pair")]
    tail_keys = {event_key(item) for item in classified_tails["catalogs"]["0.99"]["events"]}
    report = {}
    for split in ("construction", "validation", "all"):
        selected = official if split == "all" else [event for event in official if event["split"] == split]
        tail_rows = [event for event in selected if event_key(event) in tail_keys]
        identities = [identity_key(event) for event in tail_rows]
        identity_counts = Counter(identities)
        n_repeated = int(sum(1 for key in identities if identity_counts[key] > 1))
        n_wrong = int(
            sum(
                1
                for event in tail_rows
                if is_wrong_momentum(
                    float(event.get("p_mev", np.nan)),
                    float(event.get("p_truth_mev", np.nan)),
                    config,
                )
            )
        )
        pair_of_identity = defaultdict(set)
        for event in tail_rows:
            pair_of_identity[identity_key(event)].add(event["station_pair"])
        multi_pair = {
            f"{sid}/{run}/{ev}": sorted(pairs)
            for (sid, run, ev), pairs in pair_of_identity.items()
            if len(pairs) > 1
        }
        tail_chi2 = [float(event["chi2_per_ndof"]) for event in tail_rows if event.get("chi2_per_ndof") is not None]
        wrong_chi2 = [
            float(event["chi2_per_ndof"])
            for event in tail_rows
            if event.get("chi2_per_ndof") is not None
            and is_wrong_momentum(
                float(event.get("p_mev", np.nan)),
                float(event.get("p_truth_mev", np.nan)),
                config,
            )
        ]
        all_chi2 = [float(event["chi2_per_ndof"]) for event in selected if event.get("chi2_per_ndof") is not None]
        report[split] = {
            "n_official": len(selected),
            "n_tail_1pct": len(tail_rows),
            "tail_fraction": (len(tail_rows) / len(selected)) if selected else None,
            "n_wrong_momentum": n_wrong,
            "wrong_momentum_fraction_of_tail": (n_wrong / len(tail_rows)) if tail_rows else None,
            "n_repeated_event_rows": n_repeated,
            "repeated_event_fraction_of_tail": (n_repeated / len(tail_rows)) if tail_rows else None,
            "n_multi_pair_identities": len(multi_pair),
            "multi_pair_identities": multi_pair,
            "tail_chi2_share_of_official": (
                float(sum(tail_chi2) / sum(all_chi2)) if all_chi2 and sum(all_chi2) else None
            ),
            "wrong_momentum_chi2_share_of_tail": (
                float(sum(wrong_chi2) / sum(tail_chi2)) if tail_chi2 and sum(tail_chi2) else None
            ),
            "wrong_momentum_chi2_share_of_official": (
                float(sum(wrong_chi2) / sum(all_chi2)) if all_chi2 and sum(all_chi2) else None
            ),
            "events_dropped": 0,
        }
    return {
        "tails_rescreened": False,
        "events_dropped": 0,
        "splits": report,
    }


def _pull_block(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    block = {"n": len(events)}
    for name in PULL_NAMES:
        values = [
            event.get("pull", {}).get(name)
            for event in events
            if event.get("pull", {}).get(name) is not None
        ]
        finite = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
        stats = _stats(finite)
        if finite.size:
            centered = finite - float(np.mean(finite))
            moment4 = float(np.mean(centered**4))
            var = float(np.mean(centered**2))
            excess = (moment4 / var**2 - 3.0) if var > 0.0 else None
            stats["excess_kurtosis"] = excess
            stats["fraction_abs_gt_3"] = float(np.mean(np.abs(finite) > 3.0))
            stats["fraction_abs_gt_5"] = float(np.mean(np.abs(finite) > 5.0))
            stats["overcovered"] = float(stats["rms"] or 0.0) < 1.0
        else:
            stats["excess_kurtosis"] = None
            stats["fraction_abs_gt_3"] = None
            stats["fraction_abs_gt_5"] = None
            stats["overcovered"] = None
        block[name] = stats
    return block


def pull_distribution(
    official_events: Sequence[Mapping[str, Any]],
    classified_tails: Mapping[str, Any],
) -> dict[str, Any]:
    official = [event for event in official_events if event.get("on_frozen_station_pair")]
    tail_keys = {event_key(item) for item in classified_tails["catalogs"]["0.99"]["events"]}
    bulk = [event for event in official if event_key(event) not in tail_keys]
    tail = [event for event in official if event_key(event) in tail_keys]
    return {
        "n_official_kept": len(official),
        "n_dropped": 0,
        "tail_removed_is_diagnostic_only": True,
        "full_sample": _pull_block(official),
        "bulk_diagnostic_tail_removed": _pull_block(bulk),
        "tail_1pct": _pull_block(tail),
        "bulk_overcovered": all(
            bool((_pull_block(bulk).get(name) or {}).get("overcovered"))
            for name in ("x", "y", "tx", "ty")
        ),
        "tail_non_gaussian": any(
            ((_pull_block(tail).get(name) or {}).get("excess_kurtosis") or 0.0) > 5.0
            or ((_pull_block(tail).get(name) or {}).get("fraction_abs_gt_5") or 0.0) > 0.2
            for name in PULL_NAMES
        ),
    }


def q_eigenvalue_diagnostic(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_pairs = {f"({int(a)},{int(b)})" for a, b in config["station_pairs"]}
    samples = []
    for row in records:
        pair = f"({int(row.get('source_station', -1))},{int(row.get('target_station', -1))})"
        if pair not in frozen_pairs:
            continue
        q_mat = _as_matrix(row.get("process_noise"), 5)
        if q_mat is None:
            continue
        spectrum = _eigen_spectrum(q_mat)
        pred = (row.get(MODEL1) or {}).get("state_xy_tx_ty") or [0.0, 0.0, 0.0, 0.0]
        tx, ty = float(pred[2]), float(pred[3])
        samples.append(
            {
                "min_eigenvalue": spectrum["min"],
                "n_negative": spectrum["n_negative"],
                "psd": spectrum["psd"],
                "abs_q_over_p": abs(float(row.get("q_over_p_per_mev", np.nan))),
                "q_frobenius": float(row["q_frobenius"]) if row.get("q_frobenius") is not None else float(np.linalg.norm(q_mat)),
                "abs_slope": float(np.hypot(tx, ty)),
                "station_pair": pair,
            }
        )
    mins = [item["min_eigenvalue"] for item in samples]
    n_not_psd = int(sum(1 for item in samples if not item["psd"]))
    by_bin = {}
    for field, bins in (
        ("abs_q_over_p", config["qoverp_bins_abs_per_mev"]),
        ("q_frobenius", config["q_frobenius_bins"]),
        ("abs_slope", config["incidence_abs_slope_bins"]),
    ):
        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in bins}
        for item in samples:
            name = _bin_name(float(item[field]), bins)
            if name is not None:
                grouped[name].append(item)
        by_bin[field] = {
            name: {
                "n": len(items),
                "n_not_psd": int(sum(1 for item in items if not item["psd"])),
                "fraction_not_psd": (
                    float(sum(1 for item in items if not item["psd"]) / len(items))
                    if items
                    else None
                ),
                "min_eigenvalue": _stats([item["min_eigenvalue"] for item in items]),
            }
            for name, items in grouped.items()
        }
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        by_pair[item["station_pair"]].append(item)
    return {
        "n": len(samples),
        "n_not_psd": n_not_psd,
        "fraction_not_psd": (n_not_psd / len(samples)) if samples else None,
        "min_eigenvalue": _stats(mins),
        "q_replaced_by_highland": False,
        "acts_process_noise_modified": False,
        "correlation_min_eig_vs_abs_qoverp": _safe_corr(
            [item["min_eigenvalue"] for item in samples],
            [item["abs_q_over_p"] for item in samples],
        ),
        "correlation_min_eig_vs_slope": _safe_corr(
            [item["min_eigenvalue"] for item in samples],
            [item["abs_slope"] for item in samples],
        ),
        "by_bin": by_bin,
        "by_station_pair": {
            pair: {
                "n": len(items),
                "fraction_not_psd": float(sum(1 for item in items if not item["psd"]) / len(items)),
            }
            for pair, items in by_pair.items()
        },
    }


def _enrichment(tail_events: Sequence[Mapping[str, Any]], official: Sequence[Mapping[str, Any]], flag: str, config: Mapping[str, Any]) -> float | None:
    def _has(event: Mapping[str, Any]) -> bool:
        if flag == "high_q":
            return event.get("q_frobenius") is not None and float(event["q_frobenius"]) >= float(
                config["q_frobenius_bins"]["high"][0]
            )
        return event.get("abs_slope") is not None and float(event["abs_slope"]) >= float(
            config["incidence_abs_slope_bins"]["large"][0]
        )

    p_tail = np.mean([_has(event) for event in tail_events]) if tail_events else 0.0
    p_all = np.mean([_has(event) for event in official]) if official else 0.0
    if p_all <= 0.0:
        return None
    return float(p_tail / p_all)


def decide_mechanism(
    classified_tails: Mapping[str, Any],
    quality: Mapping[str, Any],
    pulls: Mapping[str, Any],
    config: Mapping[str, Any],
    official_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = config["case_gates"]
    tail = classified_tails["catalogs"]["0.99"]["events"]
    all_quality = quality["splits"]["all"]
    wrong_frac = all_quality.get("wrong_momentum_fraction_of_tail")
    wrong_share = all_quality.get("wrong_momentum_chi2_share_of_tail")
    class_a_frac = (tail and float(sum(1 for item in tail if item["classes"]["A"]) / len(tail))) or 0.0
    case1 = bool(
        (wrong_frac is not None and float(wrong_frac) >= float(gates["case1_wrong_state_fraction_min"]))
        or (wrong_share is not None and float(wrong_share) >= float(gates["case1_wrong_state_chi2_share_min"]))
        or class_a_frac >= float(gates["case1_wrong_state_fraction_min"])
    )
    official = [event for event in official_events if event.get("on_frozen_station_pair")]
    q_enr = _enrichment(tail, official, "high_q", config)
    slope_enr = _enrichment(tail, official, "large_slope", config)
    case2 = bool(
        (q_enr is not None and q_enr >= float(gates["case2_material_or_slope_enrichment_min"]))
        or (slope_enr is not None and slope_enr >= float(gates["case2_material_or_slope_enrichment_min"]))
    )
    if case1:
        primary = CASE_1
    elif case2:
        primary = CASE_2
    else:
        primary = CASE_3
    secondaries = []
    if primary != CASE_2 and case2:
        secondaries.append(CASE_2)
    if primary != CASE_3 and pulls.get("bulk_overcovered"):
        secondaries.append(CASE_3)
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "wrong_momentum_fraction_of_tail": wrong_frac,
            "wrong_momentum_chi2_share_of_tail": wrong_share,
            "class_a_fraction_of_tail": class_a_frac,
            "high_q_enrichment": q_enr,
            "large_slope_enrichment": slope_enr,
            "bulk_overcovered": pulls.get("bulk_overcovered"),
            "tail_non_gaussian": pulls.get("tail_non_gaussian"),
        },
        "case_flags": {"1": case1, "2": case2, "3": (not case1) and (not case2)},
        "measurement_model_v2_entered": False,
        "tails_dropped": False,
        "covariance_retuned": False,
        "q_retuned": False,
    }


def _all_sources(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for spec in config["mc_data"]["construction_sources"]:
        items.append(("construction", spec))
    for spec in config["mc_data"]["validation_sources"]:
        items.append(("validation", spec))
    return items


def inventory_and_analyze(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = load_frozen_tails(config)
    records_by_split: dict[str, list[dict[str, Any]]] = {"construction": [], "validation": []}
    truth_tables: dict[str, Any] = {}
    quality_tables: dict[str, Any] = {}
    present = []
    missing = []
    for split, spec in _all_sources(config):
        source_id = spec["source_id"]
        access = "train" if split == "construction" else "validation"
        authorize_path(spec["input_xaod"], AccessScope.DEVELOPMENT_VALIDATION, split=access)
        path = dump_path_for_source(config, source_id)
        rows = load_dump_records(path, split=access)
        if not rows:
            missing.append(source_id)
            continue
        present.append(source_id)
        records_by_split[split].extend(rows)
        refit = resolve_under_root(
            project_root(),
            str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
        )
        enhanced = refit / config["mc_data"]["enhanced_file"]
        truth_tables[source_id] = _load_truth_table(
            enhanced, config["mc_data"]["enhanced_tree"], config
        )
        quality_tables[source_id] = _load_track_quality(enhanced)
    all_rows = records_by_split["construction"] + records_by_split["validation"]
    dump_index = _dump_index(all_rows)
    events: list[dict[str, Any]] = []
    for split, rows in records_by_split.items():
        events.extend(
            collect_matched_events(
                config, records=rows, truth_tables=truth_tables, split=split
            )
        )
    classified = classify_frozen_tails(
        frozen, config=config, dump_index=dump_index, quality_tables=quality_tables
    )
    missing_keys = []
    official_keys = {event_key(event) for event in events if event.get("on_frozen_station_pair")}
    for item in classified["catalogs"]["0.99"]["events"]:
        if event_key(item) not in official_keys:
            missing_keys.append(list(event_key(item)))
    quality = reconstruction_quality_report(events, classified, config)
    pulls = pull_distribution(events, classified)
    q_diag = q_eigenvalue_diagnostic(all_rows, config)
    mechanism = decide_mechanism(classified, quality, pulls, config, events)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "n_dump_rows": len(all_rows),
        "n_matched_events": len(events),
        "n_official_pair_events": int(sum(1 for event in events if event.get("on_frozen_station_pair"))),
        "n_frozen_tail_missing_from_official": len(missing_keys),
        "frozen_tail_missing_keys": missing_keys,
        "tail_failure_classification": classified,
        "reconstruction_quality": quality,
        "pull_distribution": pulls,
        "q_eigenvalue_diagnostic": q_diag,
        "mechanism": mechanism,
        "wb98_dumps_read_only": True,
        "wb99_tails_read_only": True,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("tails_dropped")):
        refuse_outlier_rejection()
    return {
        "verdict": "ANALYZED",
        "decision": DECISION_ANALYZED,
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "closure_pass": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb99_decision": inherited["workbook_99"]["decision"],
        "inherited_wb99_primary_case": inherited["workbook_99"]["primary_case"],
        "inherited_wb99_decision_sha256": inherited["workbook_99"]["decision_sha256"],
        "inherited_wb99_tail_sha256": inherited["workbook_99"]["tail_sha256"],
        "wb87_through_wb99_rewritten": False,
        "evidence": mechanism["evidence"],
        "case_flags": mechanism["case_flags"],
    }
