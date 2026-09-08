"""Task B6: CKF reconstruction input contract audit.

Defines which CKF states currently enter transport covariance validation
and alignment likelihood.  Does not implement a filter, design cuts,
retune C/Q, drop tails, or enter Measurement Model V2.
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
from alignment.propagated_covariance_closure import _load_truth_table
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_process_noise_contract import _as_matrix
from datasets.acts_transport_diagnosis import (
    CASE_C as WB99_CASE_C,
    DECISION_DIAGNOSED as WB99_DECISION,
    _stats,
    collect_matched_events,
    dump_path_for_source,
    jsonable,
    load_dump_records,
    residual_decomposition,
)
from datasets.acts_transport_dump import (
    DECISION_NOT_ESTABLISHED as WB98_DECISION,
    SITUATION_B,
)
from datasets.acts_transport_tail_analysis import (
    CASE_1 as WB100_CASE_1,
    DECISION_ANALYZED as WB100_DECISION,
    event_key,
    identity_key,
    is_wrong_momentum,
)
from datasets.ckf_tail_provenance import (
    CLASS_B as WB101_CASE_B,
    DECISION_AUDITED as WB101_DECISION,
    NTUPLE_BRANCHES,
    extract_reconstruction,
    load_frozen_classified_tails,
    load_ntuple_event_table,
)
from datasets.qoverp_covariance_export import DECISION_ESTABLISHED as WB96_DECISION
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)

SCHEMA_VERSION = "ckf-reconstruction-contract-audit-v1"
DEFAULT_CONFIG = "configs/ckf_reconstruction_contract_audit_v1.yaml"
TASK = "SB-B6"
WORKBOOK = 102

CASE_A = "reconstruction_input_contract_missing"
CASE_B = "ckf_fitting_quality"
CASE_C = "remaining_physics_uncertainty"

NEXT_STEP = {
    CASE_A: "transport_covariance_v3_after_input_scope",
    CASE_B: "ckf_fitting_quality_audit",
    CASE_C: "acts_material_diagnosis",
}

DECISION_AUDITED = "ckf_reconstruction_contract_audited"

REJECTION_REASONS = (
    "missing_station",
    "short_segment",
    "low_measurement_count",
    "no_truth_match",
    "fit_failure",
    "other",
)


class CkfContractError(ValueError):
    """Raised when the B6 reconstruction-contract audit is illegal."""


def refuse_truth_qoverp() -> None:
    raise CkfContractError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise CkfContractError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise CkfContractError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise CkfContractError("process noise must not be adjusted to chi2")


def refuse_outlier_rejection() -> None:
    raise CkfContractError("outlier rejection and chi2 clipping are forbidden")


def refuse_tail_rescreen() -> None:
    raise CkfContractError("WB99/WB100/WB101 tail lists must not be rescreened")


def refuse_quality_cuts() -> None:
    raise CkfContractError("quality cuts must not be designed from B6 counts")


def refuse_input_filter() -> None:
    raise CkfContractError("eligible_for_transport_validation must not be implemented as a filter")


def refuse_closure_pass_after_subset() -> None:
    raise CkfContractError("long-track subset must not be declared closure PASS")


def refuse_measurement_model_v2() -> None:
    raise CkfContractError("Measurement Model V2 is not entered in Task B6")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CkfContractError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise CkfContractError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise CkfContractError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise CkfContractError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_delete_state_variables",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_rescreen_wb99_tails",
        "do_not_design_quality_cuts",
        "do_not_implement_input_filter",
        "do_not_declare_closure_pass_after_subset",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_reread_sealed_test",
        "do_not_start_new_source_campaign",
        "do_not_write_geometry_or_conditions_payload",
        "do_not_modify_faseracts_extrapolation_tool_source",
        "do_not_construct_acts_objects_in_python",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise CkfContractError(f"{key} must be true")
    if float(config["wrong_momentum"]["abs_log10_p_ratio_min"]) != 1.0:
        raise CkfContractError("wrong-momentum decade gate must stay frozen")
    roots = {
        Path(str(config.get("output_root"))).as_posix(),
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb99_output_root"))).as_posix(),
        Path(str(config.get("wb100_output_root"))).as_posix(),
        Path(str(config.get("wb101_output_root"))).as_posix(),
    }
    if len(roots) < 5:
        raise CkfContractError("B6 must not write into WB98–WB101 artifact roots")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise CkfContractError(f"{label} hash mismatch: {digest}")


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
        ("workbook_101", None, None, None),
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
            raise CkfContractError(f"{name} decision must stay frozen")
        if mechanism_key and spec.get("frozen_mechanism"):
            if decision.get("mechanism") != spec.get("frozen_mechanism"):
                raise CkfContractError(f"{name} mechanism must stay frozen")
        if failure_key and spec.get("frozen_failure_type"):
            if decision.get(failure_key) != spec.get("frozen_failure_type"):
                raise CkfContractError(f"{name} failure_type must stay frozen")
        if situation_key and spec.get(situation_key):
            if decision.get("situation") != spec.get(situation_key):
                raise CkfContractError(f"{name} situation must stay frozen")
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
        if spec.get("provenance_sha256"):
            inherited[name]["provenance_sha256"] = spec["provenance_sha256"]
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise CkfContractError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise CkfContractError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise CkfContractError("WB96 must keep the CKF export established")
    if inherited["workbook_98"]["decision"] != WB98_DECISION:
        raise CkfContractError("WB98 decision must stay not-established")
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise CkfContractError("WB98 situation must stay frozen")
    if inherited["workbook_99"]["decision"] != WB99_DECISION:
        raise CkfContractError("WB99 decision must stay diagnosed")
    if inherited["workbook_99"].get("primary_case") != WB99_CASE_C:
        raise CkfContractError("WB99 primary case must stay frozen")
    if inherited["workbook_100"]["decision"] != WB100_DECISION:
        raise CkfContractError("WB100 decision must stay analyzed")
    if inherited["workbook_100"].get("primary_case") != WB100_CASE_1:
        raise CkfContractError("WB100 primary case must stay wrong_track_state_dominated")
    if inherited["workbook_101"]["decision"] != WB101_DECISION:
        raise CkfContractError("WB101 decision must stay audited")
    if inherited["workbook_101"].get("primary_case") != WB101_CASE_B:
        raise CkfContractError("WB101 primary case must stay ckf_reconstruction_failure")
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
    spec101 = config["inheritance"]["workbook_101"]
    _expect_sha(
        resolve_under_root(project_root(), spec101["provenance_path"]),
        spec101["provenance_sha256"],
        "WB101 provenance",
    )
    return inherited


def load_frozen_wb101_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["inheritance"]["workbook_101"]
    path = resolve_under_root(project_root(), spec["provenance_path"])
    _expect_sha(path, spec["provenance_sha256"], "WB101 provenance")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bool(payload.get("rescreened")):
        raise CkfContractError("WB101 provenance must not be rescreened")
    catalogs = payload.get("catalogs") or {}
    for quantile in config["tail_quantiles"]:
        cell = catalogs.get(str(quantile))
        if cell is None:
            raise CkfContractError(f"WB101 provenance missing quantile {quantile}")
        if bool(cell.get("rescreened")) or int(cell.get("rejected") or 0):
            raise CkfContractError("WB101 provenance tails were rescreened or rejected")
    return {
        "sha256": spec["provenance_sha256"],
        "rescreened": False,
        "catalogs": catalogs,
    }


def ckf_track_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_id"]),
        int(row["run_id"]),
        int(row["event_id"]),
        int(row.get("track_index", 0)),
    )


def _as_1d(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def attach_tracklet_fit(
    reco: Mapping[str, Any],
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(reco)
    if record is None:
        payload["tracklet_chi2"] = []
        payload["tracklet_ndof"] = []
        payload["tracklet_chi2_per_ndof"] = []
        return payload
    chi2 = _as_1d(record.get("Tracklet_chi2"))
    ndof = _as_1d(record.get("Tracklet_ndof"))
    payload["tracklet_chi2"] = [float(v) for v in chi2] if chi2.size else []
    payload["tracklet_ndof"] = [float(v) for v in ndof] if ndof.size else []
    ratios = []
    for value, dof in zip(payload["tracklet_chi2"], payload["tracklet_ndof"]):
        if dof not in (0.0, None) and np.isfinite(value) and np.isfinite(dof):
            ratios.append(float(value / dof))
    payload["tracklet_chi2_per_ndof"] = ratios
    return payload


def classify_rejection_reason(
    reco: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Descriptive taxonomy only.  Not a production cut."""
    stations = [int(v) for v in reco.get("tracklet_stations") or [] if v is not None]
    n_unique_stations = len(set(stations))
    n_seg = reco.get("n_segments")
    n_hit = int(sum(int(v) for v in reco.get("tracklet_n_hit") or [] if v is not None))
    truth_ids = [int(v) for v in reco.get("tracklet_truth_ids") or [] if v is not None]
    tmf = [
        float(v)
        for v in reco.get("tracklet_truth_match") or []
        if v is not None and np.isfinite(float(v))
    ]
    tracklet_poor = any(
        value >= float(config["poor_ckf_chi2_per_ndof_min"])
        for value in reco.get("tracklet_chi2_per_ndof") or []
    )
    ckf_poor = (
        reco.get("ckf_chi2_per_ndof") is not None
        and float(reco["ckf_chi2_per_ndof"]) >= float(config["poor_ckf_chi2_per_ndof_min"])
    )
    flags = {
        "missing_station": n_unique_stations < int(config["expected_n_stations"]),
        "short_segment": n_seg is not None and int(n_seg) < int(config["expected_n_segments"]),
        "low_measurement_count": 0 < n_hit < int(config["descriptive_min_tracklet_hits"]),
        "no_truth_match": (
            any(value <= 0 for value in truth_ids)
            or any(value < float(config["low_truth_match_fraction_max"]) for value in tmf)
        ),
        "fit_failure": bool(tracklet_poor or ckf_poor),
        "ntuple_event_missing": reco.get("n_long_tracks") is None,
    }
    primary = next((name for name in REJECTION_REASONS[:-1] if flags[name]), "other")
    return {
        "primary_reason": primary,
        "flags": flags,
        "n_unique_tracklet_stations": n_unique_stations,
        "n_tracklet_hits": n_hit,
        "cuts_designed": False,
    }


def is_long_track_accepted(reco: Mapping[str, Any]) -> bool:
    return int(reco.get("n_long_tracks") or 0) > 0


def qoverp_sigma(row: Mapping[str, Any]) -> float | None:
    cin = _as_matrix(row.get("input_covariance"), 5)
    if cin is None:
        return None
    variance = float(cin[4, 4])
    if variance <= 0.0 or not np.isfinite(variance):
        return None
    return float(np.sqrt(variance))


def define_reconstruction_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record the current vs intended input scope.  Does not apply a filter."""
    return {
        "implemented_as_filter": False,
        "production_filter_exists": False,
        "current_transport_input": "raw_CKF_collection",
        "current_alignment_likelihood_input": "raw_CKF_collection",
        "raw_CKF_collection": {
            "name": "CKFTrackCollection",
            "currently_used_for_transport_validation": True,
            "currently_used_for_alignment_likelihood": True,
            "requirements_enforced": [
                "state_5_available",
                "covariance_5x5_available",
                "q_over_p_finite_signed",
            ],
            "long_track_selection_applied": False,
            "n_stations_required": None,
            "minimum_measurements_required": None,
            "fit_status_required": None,
        },
        "eligible_for_transport_validation": {
            "defined_in_production": False,
            "implemented_as_filter": False,
            "descriptive_requirements": {
                "n_stations": int(config["expected_n_stations"]),
                "n_segments": int(config["expected_n_segments"]),
                "minimum_measurements": "unspecified_not_a_cut",
                "descriptive_min_tracklet_hits": int(config["descriptive_min_tracklet_hits"]),
                "state_available": True,
                "covariance_5x5_available": True,
                "q_over_p_finite_signed": True,
                "q_over_p_valid": True,
                "fit_status_available": True,
                "long_track_equivalent": True,
            },
            "note": (
                "These requirements document the long-track reconstruction "
                "contract.  Task B6 does not apply them as a filter."
            ),
        },
        "difference": (
            "Transport validation and alignment currently consume the raw "
            "CKFTrackCollection.  The ntuple Track_* block is the long-track "
            "filtered subset.  eligible_for_transport_validation is undefined "
            "in production."
        ),
        "not_a_cut_sheet": True,
    }


def _all_sources(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for spec in config["mc_data"]["construction_sources"]:
        items.append(("construction", spec))
    for spec in config["mc_data"]["validation_sources"]:
        items.append(("validation", spec))
    return items


def _sample_stats(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chi2 = [event["chi2_per_ndof"] for event in events if event.get("chi2_per_ndof") is not None]
    pulls = {
        name: _stats(
            [
                event["pull"][name]
                for event in events
                if event.get("pull", {}).get(name) is not None
            ]
        )
        for name in ("x", "y", "tx", "ty", "q_over_p")
    }
    return {
        "n": len(events),
        "chi2_per_ndof": _stats(chi2),
        "pull": pulls,
        "residual_decomposition": residual_decomposition(events),
    }


def _frozen_membership(
    events: Sequence[Mapping[str, Any]],
    frozen_keys: set[tuple[Any, ...]],
) -> dict[str, Any]:
    members = [event for event in events if event_key(event) in frozen_keys]
    n = len(events)
    return {
        "n_frozen_members": len(members),
        "frozen_fraction_of_sample": (float(len(members) / n) if n else None),
        "keys": [list(event_key(event)) for event in members],
        "identities": [list(key) for key in sorted({identity_key(event) for event in members})],
    }


def audit_input_scope(
    *,
    config: Mapping[str, Any],
    dump_rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    ntuple_tables: Mapping[str, Mapping[tuple[int, int], Any]],
) -> dict[str, Any]:
    per_source = []
    all_tracks = []
    ntuple_long_keys: set[tuple[Any, ...]] = set()
    ckf_keys: set[tuple[Any, ...]] = set()
    for source_id, rows in dump_rows_by_source.items():
        unique: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            if bool(row.get("is_truth", False)):
                continue
            unique.setdefault(ckf_track_key(row), row)
        table = ntuple_tables.get(source_id, {})
        for (run_id, event_id), record in table.items():
            n_long = int(_as_1d(record.get("Track_p0")).size)
            if n_long > 0:
                ntuple_long_keys.add((source_id, int(run_id), int(event_id)))
        source_tracks = []
        for key, row in unique.items():
            ckf_keys.add(key[:3])
            record = table.get((int(row["run_id"]), int(row["event_id"])))
            reco = attach_tracklet_fit(
                extract_reconstruction(record, float(row.get("p_mev", np.nan))),
                record,
            )
            accepted = is_long_track_accepted(reco)
            taxonomy = classify_rejection_reason(reco, config=config)
            item = {
                "source_id": source_id,
                "run_id": int(row["run_id"]),
                "event_id": int(row["event_id"]),
                "track_index": int(row.get("track_index", 0)),
                "collection": row.get("collection"),
                "p_mev": row.get("p_mev"),
                "q_over_p_per_mev": row.get("q_over_p_per_mev"),
                "q_over_p_sigma": qoverp_sigma(row),
                "accepted_long_track": accepted,
                "rejection_reason": None if accepted else taxonomy["primary_reason"],
                "rejection_flags": taxonomy["flags"],
                "n_long_tracks": reco.get("n_long_tracks"),
                "n_segments": reco.get("n_segments"),
                "n_unique_tracklet_stations": taxonomy["n_unique_tracklet_stations"],
                "n_tracklet_hits": taxonomy["n_tracklet_hits"],
                "n_measurements": reco.get("ckf_n_measurements"),
                "in_station": reco.get("in_station"),
                "tracklet_stations": reco.get("tracklet_stations"),
                "ckf_chi2_per_ndof": reco.get("ckf_chi2_per_ndof"),
            }
            source_tracks.append(item)
            all_tracks.append(item)
        reasons = Counter(
            item["rejection_reason"] for item in source_tracks if item["rejection_reason"]
        )
        per_source.append(
            {
                "source_id": source_id,
                "n_dump_rows": len(rows),
                "n_unique_ckf_tracks": len(source_tracks),
                "n_accepted_long_tracks": int(
                    sum(1 for item in source_tracks if item["accepted_long_track"])
                ),
                "n_rejected_ckf_tracks": int(
                    sum(1 for item in source_tracks if not item["accepted_long_track"])
                ),
                "rejection_reason_counts": {name: int(reasons.get(name, 0)) for name in REJECTION_REASONS},
            }
        )
    reasons = Counter(item["rejection_reason"] for item in all_tracks if item["rejection_reason"])
    flag_counts = Counter()
    for item in all_tracks:
        if item["accepted_long_track"]:
            continue
        for name, value in (item.get("rejection_flags") or {}).items():
            if value:
                flag_counts[name] += 1
    n_ckf = len(all_tracks)
    n_accepted = int(sum(1 for item in all_tracks if item["accepted_long_track"]))
    n_rejected = n_ckf - n_accepted
    ntuple_without_ckf = [
        list(key) for key in sorted(ntuple_long_keys) if key not in ckf_keys
    ]
    return {
        "n_unique_ckf_tracks": n_ckf,
        "n_accepted_long_tracks": n_accepted,
        "n_rejected_ckf_tracks": n_rejected,
        "ckf_not_long_fraction": float(n_rejected / n_ckf) if n_ckf else None,
        "n_ntuple_long_track_events": len(ntuple_long_keys),
        "n_ntuple_long_tracks_without_ckf_dump": len(ntuple_without_ckf),
        "ntuple_long_tracks_without_ckf_dump": ntuple_without_ckf,
        "rejection_reason_counts": {name: int(reasons.get(name, 0)) for name in REJECTION_REASONS},
        "rejection_flag_counts": dict(flag_counts),
        "per_source": per_source,
        "cuts_designed": False,
        "filter_implemented": False,
        "tracks": all_tracks,
    }


def audit_tail_reproduction(
    *,
    config: Mapping[str, Any],
    official_events: Sequence[Mapping[str, Any]],
    ntuple_tables: Mapping[str, Mapping[tuple[int, int], Any]],
    frozen_1pct: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen_keys = {event_key(item) for item in frozen_1pct}
    long_identities = set()
    for event in official_events:
        reco = extract_reconstruction(
            ntuple_tables.get(str(event["source_id"]), {}).get(
                (int(event["run_id"]), int(event["event_id"]))
            ),
            float(event.get("p_mev", np.nan)),
        )
        if is_long_track_accepted(reco):
            long_identities.add(identity_key(event))
    long_events = [event for event in official_events if identity_key(event) in long_identities]
    full_stats = _sample_stats(official_events)
    long_stats = _sample_stats(long_events)
    full_frozen = _frozen_membership(official_events, frozen_keys)
    long_frozen = _frozen_membership(long_events, frozen_keys)
    remaining_long_tails = [
        item
        for item in frozen_1pct
        if identity_key(item) in long_identities
    ]
    remaining_qop = [
        item
        for item in remaining_long_tails
        if item.get("pull", {}).get("q_over_p") is not None
        and abs(float(item["pull"]["q_over_p"])) >= float(config["qoverp_pull_anomaly_abs_min"])
    ]
    remaining_wrong_p = [
        item
        for item in remaining_long_tails
        if is_wrong_momentum(
            float(item.get("p_mev", np.nan)),
            float(item.get("p_truth_mev", np.nan)),
            config,
        )
    ]
    return {
        "diagnostic_only": True,
        "not_a_repair": True,
        "closure_not_claimed": True,
        "closure_pass": False,
        "tails_deleted": False,
        "rescreened": False,
        "filter_implemented": False,
        "raw_official": {
            **full_stats,
            "frozen_1pct": full_frozen,
        },
        "long_track_equivalent_diagnostic": {
            **long_stats,
            "frozen_1pct": long_frozen,
            "n_remaining_frozen_1pct": long_frozen["n_frozen_members"],
            "n_remaining_qoverp_pull_anomaly": len(remaining_qop),
            "n_remaining_wrong_momentum": len(remaining_wrong_p),
            "remaining_identities": [
                list(key) for key in sorted({identity_key(item) for item in remaining_long_tails})
            ],
        },
        "note": (
            "The long-track equivalent sample is a diagnostic contrast only.  "
            "It is not a repair, not an implemented filter, and not a closure PASS."
        ),
    }


def audit_short_tracks(
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    dump_by_identity: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = list((provenance.get("catalogs") or {}).get("0.99", {}).get("events") or [])
    short = [item for item in rows if bool((item.get("flags") or {}).get("not_a_long_track"))]
    events = []
    for item in short:
        dump_rows = list(dump_by_identity.get(identity_key(item), []))
        representative = dump_rows[0] if dump_rows else {}
        taxonomy = classify_rejection_reason(item, config=config)
        stations = [int(v) for v in item.get("tracklet_stations") or []]
        expected = set(range(int(config["expected_n_stations"])))
        missing = sorted(expected.difference(stations))
        events.append(
            {
                "source_id": item["source_id"],
                "run_id": item["run_id"],
                "event_id": item["event_id"],
                "track_index": item.get("track_index"),
                "station_pair": item.get("station_pair"),
                "n_segments": item.get("n_segments"),
                "n_measurements": item.get("n_measurements"),
                "n_long_tracks": item.get("n_long_tracks"),
                "tracklet_stations": stations,
                "tracklet_n_hit": item.get("tracklet_n_hit"),
                "tracklet_hit_pattern": item.get("tracklet_hit_pattern"),
                "missing_stations": missing,
                "track_length_n_stations": len(set(stations)),
                "track_length_n_hits": int(sum(int(v) for v in item.get("tracklet_n_hit") or [])),
                "q_over_p_reco_per_mev": item.get("q_over_p_reco_per_mev"),
                "q_over_p_pull": item.get("q_over_p_pull"),
                "q_over_p_sigma": qoverp_sigma(representative),
                "p_reco_mev": item.get("p_reco_mev"),
                "p_truth_mev": item.get("p_truth_mev"),
                "primary_reason": taxonomy["primary_reason"],
                "rejection_flags": taxonomy["flags"],
                "wb101_primary_class": item.get("primary_class"),
            }
        )
    reasons = Counter(item["primary_reason"] for item in events)
    return {
        "n_frozen_1pct": len(rows),
        "n_not_a_long_track": len(events),
        "rescreened": False,
        "tails_deleted": False,
        "rejection_reason_counts": {name: int(reasons.get(name, 0)) for name in REJECTION_REASONS},
        "events": events,
    }


def decide_case(
    scope: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    short: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if bool(reproduction.get("closure_pass")):
        refuse_closure_pass_after_subset()
    if bool(scope.get("filter_implemented")) or bool(reproduction.get("filter_implemented")):
        refuse_input_filter()
    n_frozen = int(short.get("n_frozen_1pct") or 0) or 1
    tail_not_long = float(int(short.get("n_not_a_long_track") or 0) / n_frozen)
    ckf_not_long = float(scope.get("ckf_not_long_fraction") or 0.0)
    remaining = reproduction["long_track_equivalent_diagnostic"]
    n_long_qop = int(remaining.get("n_remaining_qoverp_pull_anomaly") or 0)
    n_long_wrong_p = int(remaining.get("n_remaining_wrong_momentum") or 0)
    n_long_frozen = int(remaining.get("n_remaining_frozen_1pct") or 0)
    case_a = (
        tail_not_long >= float(config["case_gates"]["case_a_tail_not_long_fraction_min"])
        or ckf_not_long >= float(config["case_gates"]["case_a_ckf_not_long_fraction_min"])
    )
    case_b = (n_long_qop + n_long_wrong_p + n_long_frozen) >= int(
        config["case_gates"]["case_b_long_track_tail_or_qop_min"]
    ) and n_long_frozen > 0
    case_c = (not case_a) and (not case_b) and n_long_frozen > 0
    if case_a:
        primary = CASE_A
    elif case_b:
        primary = CASE_B
    else:
        primary = CASE_C
    secondaries = []
    if primary != CASE_B and case_b:
        secondaries.append(CASE_B)
    if primary != CASE_C and case_c:
        secondaries.append(CASE_C)
    if primary != CASE_A and case_a:
        secondaries.append(CASE_A)
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "tail_not_long_fraction": tail_not_long,
            "ckf_not_long_fraction": ckf_not_long,
            "n_unique_ckf_tracks": scope.get("n_unique_ckf_tracks"),
            "n_accepted_long_tracks": scope.get("n_accepted_long_tracks"),
            "n_rejected_ckf_tracks": scope.get("n_rejected_ckf_tracks"),
            "n_frozen_1pct_not_a_long_track": short.get("n_not_a_long_track"),
            "n_remaining_frozen_1pct_on_long_track": n_long_frozen,
            "n_remaining_qoverp_pull_anomaly": n_long_qop,
            "n_remaining_wrong_momentum": n_long_wrong_p,
            "raw_chi2_mean": (reproduction["raw_official"]["chi2_per_ndof"] or {}).get("mean"),
            "raw_chi2_median": (reproduction["raw_official"]["chi2_per_ndof"] or {}).get("median"),
            "long_track_chi2_mean": (remaining["chi2_per_ndof"] or {}).get("mean"),
            "long_track_chi2_median": (remaining["chi2_per_ndof"] or {}).get("median"),
        },
        "case_flags": {"A": case_a, "B": case_b, "C": case_c},
        "measurement_model_v2_entered": False,
        "tails_dropped": False,
        "quality_cuts_designed": False,
        "input_filter_implemented": False,
        "closure_pass": False,
        "closure_claimed_after_subset": False,
        "covariance_retuned": False,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    classified = load_frozen_classified_tails(config)
    provenance = load_frozen_wb101_provenance(config)
    frozen_1pct = list(classified["catalogs"]["0.99"]["events"])
    dump_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    dump_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    ntuple_tables: dict[str, Any] = {}
    truth_tables: dict[str, Any] = {}
    present = []
    missing = []
    records_by_split: dict[str, list[dict[str, Any]]] = {
        "construction": [],
        "validation": [],
    }
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
        dump_rows_by_source[source_id] = rows
        records_by_split[split].extend(rows)
        for row in rows:
            dump_by_identity[identity_key(row)].append(row)
        refit = resolve_under_root(
            project_root(),
            str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
        )
        enhanced = refit / config["mc_data"]["enhanced_file"]
        ntuple_tables[source_id] = load_ntuple_event_table(enhanced)
        truth_tables[source_id] = _load_truth_table(
            enhanced,
            config["mc_data"]["enhanced_tree"],
            config,
        )
    scope = audit_input_scope(
        config=config,
        dump_rows_by_source=dump_rows_by_source,
        ntuple_tables=ntuple_tables,
    )
    matched: list[dict[str, Any]] = []
    for split, rows in records_by_split.items():
        matched.extend(
            collect_matched_events(
                config, records=rows, truth_tables=truth_tables, split=split
            )
        )
    official = [event for event in matched if event.get("on_frozen_station_pair")]
    reproduction = audit_tail_reproduction(
        config=config,
        official_events=official,
        ntuple_tables=ntuple_tables,
        frozen_1pct=frozen_1pct,
    )
    short = audit_short_tracks(
        config=config,
        provenance=provenance,
        dump_by_identity=dump_by_identity,
    )
    contract = define_reconstruction_contract(config)
    mechanism = decide_case(scope, reproduction, short, config)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "n_dump_rows": sum(len(rows) for rows in dump_rows_by_source.values()),
        "n_matched_events": len(matched),
        "n_official_pair_events": len(official),
        "ckf_input_scope_audit": {
            **{key: value for key, value in scope.items() if key != "tracks"},
            "n_tracks_recorded": len(scope["tracks"]),
        },
        "ckf_input_scope_tracks": scope["tracks"],
        "ckf_reconstruction_contract": contract,
        "tail_reproduction": reproduction,
        "short_track_failure_analysis": short,
        "mechanism": mechanism,
        "wb98_dumps_read_only": True,
        "wb99_tails_read_only": True,
        "wb100_classification_read_only": True,
        "wb101_provenance_read_only": True,
        "frozen_1pct_n": len(frozen_1pct),
        "frozen_1pct_rescreened": False,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("tails_dropped")):
        refuse_outlier_rejection()
    if bool(mechanism.get("quality_cuts_designed")):
        refuse_quality_cuts()
    if bool(mechanism.get("input_filter_implemented")):
        refuse_input_filter()
    if bool(mechanism.get("closure_claimed_after_subset")) or bool(mechanism.get("closure_pass")):
        refuse_closure_pass_after_subset()
    return {
        "verdict": "AUDITED",
        "decision": DECISION_AUDITED,
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "closure_pass": False,
        "closure_claimed_after_subset": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "input_filter_implemented": False,
        "inherited_wb101_decision": inherited["workbook_101"]["decision"],
        "inherited_wb101_primary_case": inherited["workbook_101"]["primary_case"],
        "inherited_wb101_decision_sha256": inherited["workbook_101"]["decision_sha256"],
        "inherited_wb101_provenance_sha256": inherited["workbook_101"]["provenance_sha256"],
        "inherited_wb100_decision_sha256": inherited["workbook_100"]["decision_sha256"],
        "inherited_wb99_tail_sha256": inherited["workbook_99"]["tail_sha256"],
        "wb87_through_wb101_rewritten": False,
        "evidence": mechanism["evidence"],
        "case_flags": mechanism["case_flags"],
    }
