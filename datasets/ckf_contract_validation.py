"""Task B7: reconstruction contract implementation validation.

Applies eligible_for_transport_validation independently of closure.
Replays the frozen WB98 dump / C0 / C1 / Q / gates with only the input
scope changed.  Does not retune C/Q, drop tails, or enter V2.
Truth is diagnostic only and never enters eligibility.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.propagated_covariance_closure import _load_truth_table
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_process_noise_contract import MODEL1, _as_matrix, evaluate_model_records
from datasets.acts_transport_diagnosis import (
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
    _principal_direction_report,
)
from datasets.acts_transport_tail_analysis import event_key, identity_key
from datasets.ckf_reconstruction_contract import (
    CASE_A as WB102_CASE_A,
    DECISION_AUDITED as WB102_DECISION,
    ckf_track_key,
    inherit_frozen_stage as inherit_through_wb101,
    is_long_track_accepted,
    load_frozen_classified_tails,
    qoverp_sigma,
)
from datasets.ckf_tail_provenance import extract_reconstruction, load_ntuple_event_table

SCHEMA_VERSION = "ckf-reconstruction-contract-v1"
DEFAULT_CONFIG = "configs/ckf_reconstruction_contract_v1.yaml"
TASK = "SB-B7"
WORKBOOK = 103

SAMPLE_A = "raw_CKF"
SAMPLE_B = "contract_eligible"
SAMPLE_C = "reference_longTrack"

CASE_A = "input_scope_mismatch_confirmed"
CASE_B = "ckf_fitting_tail_remains"
CASE_C = "transport_model_remains"

NEXT_STEP = {
    CASE_A: "transport_covariance_v3",
    CASE_B: "ckf_fitting_quality_audit",
    CASE_C: "transport_covariance_v3_uncertainty_model",
}

DECISION_VALIDATED = "ckf_reconstruction_contract_validated"

INELIGIBLE_REASONS = (
    "not_long_track_equivalent",
    "incomplete_stations",
    "incomplete_segments",
    "invalid_covariance_5x5",
    "invalid_q_over_p",
    "invalid_state_surface",
)


class CkfValidationError(ValueError):
    """Raised when the B7 contract validation is illegal."""


def refuse_truth_qoverp() -> None:
    raise CkfValidationError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise CkfValidationError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise CkfValidationError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise CkfValidationError("process noise must not be adjusted to chi2")


def refuse_outlier_rejection() -> None:
    raise CkfValidationError("outlier rejection and chi2 clipping are forbidden")


def refuse_tail_rescreen() -> None:
    raise CkfValidationError("WB99/WB102 tail lists must not be rescreened")


def refuse_closure_designed_selection() -> None:
    raise CkfValidationError("eligibility must not be designed from closure")


def refuse_filter_pass_claim() -> None:
    raise CkfValidationError("must not write filter-then-PASS")


def refuse_covariance_model_fixed() -> None:
    raise CkfValidationError("must not declare the covariance model fixed")


def refuse_measurement_model_v2() -> None:
    raise CkfValidationError("Measurement Model V2 is not entered in Task B7")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise CkfValidationError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise CkfValidationError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise CkfValidationError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise CkfValidationError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_rescreen_wb99_tails",
        "do_not_design_selection_from_closure",
        "do_not_use_chi2_in_eligibility",
        "do_not_use_pull_in_eligibility",
        "do_not_use_residual_in_eligibility",
        "do_not_use_truth_in_eligibility",
        "do_not_claim_filter_pass",
        "do_not_declare_covariance_model_fixed",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_reread_sealed_test",
        "eligibility_independent_of_closure",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise CkfValidationError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise CkfValidationError(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    bins = config.get("qoverp_bins_abs_per_mev") or {}
    expected_bins = {
        "high_momentum": [0.0, 1.0e-6],
        "medium_momentum": [1.0e-6, 5.0e-6],
        "low_momentum": [5.0e-6, 1.0],
    }
    for name, edges in expected_bins.items():
        got = [float(v) for v in bins.get(name, [])]
        if got != edges:
            raise CkfValidationError(f"q/p bin {name} must stay frozen")
    roots = {
        Path(str(config.get("output_root"))).as_posix(),
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb102_output_root"))).as_posix(),
    }
    if len(roots) < 3:
        raise CkfValidationError("B7 must not write into the WB98 dump or WB102 roots")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb101(config)
    spec = config["inheritance"]["workbook_102"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise CkfValidationError(f"workbook_102 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise CkfValidationError(f"workbook_102 decision hash mismatch: {digest}")
    if spec.get("contract_path") and spec.get("contract_sha256"):
        digest = sha256_file(resolve_under_root(project_root(), spec["contract_path"]))
        if digest != spec["contract_sha256"]:
            raise CkfValidationError(f"workbook_102 contract hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise CkfValidationError("workbook_102 decision must stay frozen")
    if spec.get("frozen_primary_case") and decision.get("primary_case") != spec["frozen_primary_case"]:
        raise CkfValidationError("workbook_102 primary case must stay frozen")
    if decision.get("decision") != WB102_DECISION:
        raise CkfValidationError("WB102 decision token mismatch")
    if spec["frozen_primary_case"] != WB102_CASE_A:
        raise CkfValidationError("WB102 primary case token mismatch")
    inherited["workbook_102"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "contract_sha256": spec.get("contract_sha256"),
    }
    return inherited


def _surface_z_mm(row: Mapping[str, Any]) -> float | None:
    surface = row.get("surface") or {}
    if surface.get("z_mm") is not None:
        value = float(surface["z_mm"])
        return value if np.isfinite(value) else None
    origin = surface.get("origin_mm")
    if isinstance(origin, Sequence) and len(origin) >= 3:
        value = float(origin[2])
        return value if np.isfinite(value) else None
    return None


def valid_covariance_5x5(row: Mapping[str, Any]) -> bool:
    cin = _as_matrix(row.get("input_covariance"), 5)
    if cin is None or not np.isfinite(cin).all():
        return False
    return bool(np.all(np.diag(cin) > 0.0))


def valid_signed_q_over_p(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    qop = float(row.get("q_over_p_per_mev", np.nan))
    if not np.isfinite(qop) or qop == 0.0:
        return False
    dummy = float(config["acceptance"]["dummy_qoverp_per_mev"])
    return abs(qop - dummy) > 1.0e-18


def valid_state_surface(row: Mapping[str, Any]) -> bool:
    if _surface_z_mm(row) is None:
        return False
    try:
        source = int(row["source_station"])
        target = int(row["target_station"])
    except (KeyError, TypeError, ValueError):
        return False
    if source not in range(4) or target not in range(4):
        return False
    state = row.get("derived_state") or row.get("native_state")
    array = np.asarray(state, dtype=np.float64).reshape(-1)
    return array.size >= 5 and bool(np.all(np.isfinite(array[:5])))


def evaluate_eligibility(
    row: Mapping[str, Any],
    reco: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruction-only predicate.  No χ², pull, residual, or truth."""
    spec = config["eligibility"]
    tracklet_stations = {
        int(v) for v in reco.get("tracklet_stations") or [] if v is not None
    }
    in_station_count = reco.get("n_stations_present")
    flags = {
        "long_track_equivalent": is_long_track_accepted(reco),
        "station_complete": (
            len(tracklet_stations) >= int(spec["expected_n_stations"])
            or (
                in_station_count is not None
                and int(in_station_count) >= int(spec["expected_n_stations"])
            )
        ),
        "segment_complete": (
            reco.get("n_segments") is not None
            and int(reco["n_segments"]) >= int(spec["expected_n_segments"])
        ),
        "covariance_5x5_valid": valid_covariance_5x5(row),
        "q_over_p_finite_signed": valid_signed_q_over_p(row, config),
        "state_surface_valid": valid_state_surface(row),
    }
    required = {
        "not_long_track_equivalent": spec["require_long_track_equivalent"]
        and not flags["long_track_equivalent"],
        "incomplete_stations": spec["require_station_completeness"]
        and not flags["station_complete"],
        "incomplete_segments": spec["require_segment_completeness"]
        and not flags["segment_complete"],
        "invalid_covariance_5x5": spec["require_covariance_5x5"]
        and not flags["covariance_5x5_valid"],
        "invalid_q_over_p": spec["require_finite_signed_q_over_p"]
        and not flags["q_over_p_finite_signed"],
        "invalid_state_surface": spec["require_valid_state_surface"]
        and not flags["state_surface_valid"],
    }
    reason = next((name for name in INELIGIBLE_REASONS if required[name]), None)
    return {
        "eligible": reason is None,
        "ineligible_reason": reason,
        "flags": flags,
        "used_chi2": False,
        "used_pull": False,
        "used_residual": False,
        "used_truth": False,
    }


def _all_sources(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for spec in config["mc_data"]["construction_sources"]:
        items.append(("construction", spec))
    for spec in config["mc_data"]["validation_sources"]:
        items.append(("validation", spec))
    return items


def _eig_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mins: list[float] = []
    maxs: list[float] = []
    for row in rows:
        cin = _as_matrix(row.get("input_covariance"), 5)
        if cin is None:
            continue
        values = np.linalg.eigvalsh(0.5 * (cin + cin.T))
        mins.append(float(values[0]))
        maxs.append(float(values[-1]))
    return {"min_eigenvalue": _stats(mins), "max_eigenvalue": _stats(maxs)}


def _station_distribution(recos: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for reco in recos:
        pattern = reco.get("in_station")
        if pattern and all(value is not None for value in pattern):
            key = "".join("1" if int(value) else "0" for value in pattern)
        else:
            stations = sorted({int(v) for v in reco.get("tracklet_stations") or [] if v is not None})
            key = "".join(str(v) for v in stations) if stations else "none"
        counts[key] += 1
    return dict(counts)


def _sample_kinematics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qops = [float(row.get("q_over_p_per_mev", np.nan)) for row in rows]
    return {
        "n_dump_rows": len(rows),
        "n_unique_ckf_tracks": len({ckf_track_key(row) for row in rows}),
        "n_unique_events": len({identity_key(row) for row in rows}),
        "q_over_p": _stats(qops),
        "abs_q_over_p": _stats([abs(v) for v in qops if np.isfinite(v)]),
        "covariance_eigenvalues": _eig_stats(rows),
    }


def summarize_event_sample(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    official = [event for event in events if event.get("on_frozen_station_pair")]
    chi2 = [event["chi2_per_ndof"] for event in official if event.get("chi2_per_ndof") is not None]
    return {
        "n_matched": len(events),
        "n_official_pairs": len(official),
        "chi2_per_ndof": _stats(chi2),
        "pull": {
            name: _stats(
                [
                    event["pull"][name]
                    for event in official
                    if event.get("pull", {}).get(name) is not None
                ]
            )
            for name in ("x", "y", "tx", "ty", "q_over_p")
        },
        "residual_decomposition": residual_decomposition(official),
    }


def summarize_gate_replay(model: Mapping[str, Any]) -> dict[str, Any]:
    per_pair = {}
    means = []
    medians = []
    pencils = []
    for label, cell in (model.get("per_pair") or {}).items():
        pencil = (cell.get("pencil") or {}).get("variance_ratio_prop_over_emp")
        gen = cell.get("generalized_eigenvalues_cemp_over_cprop")
        per_pair[label] = {
            "n_pairs": cell.get("n_pairs") or cell.get("n_matched"),
            "chi2_per_ndof": cell.get("chi2_per_ndof"),
            "chi2_per_ndof_mean": cell.get("chi2_per_ndof_mean"),
            "chi2_per_ndof_median": cell.get("chi2_per_ndof_median"),
            "pencil_ratio": pencil,
            "eigenvalue_ratio": gen,
            "cov_z_eigenvalues": cell.get("cov_z_eigenvalues"),
            "calibrated": bool((cell.get("gate_verdict") or {}).get("calibrated")),
            "gates": (cell.get("gate_verdict") or {}).get("gates"),
        }
        if cell.get("chi2_per_ndof_mean") is not None:
            means.append(float(cell["chi2_per_ndof_mean"]))
        if cell.get("chi2_per_ndof_median") is not None:
            medians.append(float(cell["chi2_per_ndof_median"]))
        if pencil is not None:
            pencils.append(float(pencil))
    return {
        "n_success": model.get("n_success"),
        "n_dummy_rejected": model.get("n_dummy_rejected"),
        "n_q_added": model.get("n_q_added"),
        "all_calibrated": bool(model.get("all_calibrated")),
        "chi2_per_ndof_mean": float(np.mean(means)) if means else None,
        "chi2_per_ndof_median": float(np.median(medians)) if medians else None,
        "pencil_ratio": _stats(pencils),
        "per_pair": per_pair,
        "principal_direction": _principal_direction_report(model.get("per_pair") or {}),
        "covariance_unmodified": True,
        "same_wb98_dump": True,
        "same_c0_c1_q": True,
        "same_gates": True,
    }


def compare_samples(
    *,
    tagged_rows: Sequence[Mapping[str, Any]],
    reco_by_track: Mapping[tuple[Any, ...], Mapping[str, Any]],
    events_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    samples = {}
    for name, predicate in (
        (SAMPLE_A, lambda row: True),
        (SAMPLE_B, lambda row: bool(row.get("contract_eligible"))),
        (SAMPLE_C, lambda row: bool(row.get("reference_long_track"))),
    ):
        rows = [row for row in tagged_rows if not bool(row.get("is_truth", False)) and predicate(row)]
        recos = [reco_by_track[ckf_track_key(row)] for row in rows if ckf_track_key(row) in reco_by_track]
        samples[name] = {
            **_sample_kinematics(rows),
            "station_distribution": _station_distribution(recos),
            "matched": summarize_event_sample(events_by_sample[name]),
        }
    return {
        "samples": samples,
        "n_contract_not_longtrack": int(
            sum(
                1
                for row in tagged_rows
                if row.get("contract_eligible") and not row.get("reference_long_track")
            )
        ),
        "n_longtrack_not_contract": int(
            sum(
                1
                for row in tagged_rows
                if row.get("reference_long_track") and not row.get("contract_eligible")
            )
        ),
        "eligibility_used_closure": False,
        "eligibility_used_truth": False,
    }


def replay_closure(
    *,
    config: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    truth_tables: Mapping[str, Any],
    sample_name: str,
    predicate,
) -> dict[str, Any]:
    splits = {}
    for split, rows in records_by_split.items():
        selected = [
            row
            for row in rows
            if not bool(row.get("is_truth", False)) and predicate(row)
        ]
        model = evaluate_model_records(
            selected, config, model_key=MODEL1, truth_tables=truth_tables
        )
        splits[split] = summarize_gate_replay(model)
    return {
        "sample": sample_name,
        "filter_then_pass_claimed": False,
        "covariance_model_fixed": False,
        "splits": splits,
        "all_calibrated": bool(splits["construction"]["all_calibrated"])
        and bool(splits["validation"]["all_calibrated"]),
    }


def audit_remaining_tails(
    *,
    config: Mapping[str, Any],
    frozen_1pct: Sequence[Mapping[str, Any]],
    contract_events: Sequence[Mapping[str, Any]],
    reco_by_identity: Mapping[tuple[Any, ...], Mapping[str, Any]],
    dump_by_identity: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    official = [event for event in contract_events if event.get("on_frozen_station_pair")]
    official_keys = {event_key(event) for event in official}
    remaining_frozen = [item for item in frozen_1pct if event_key(item) in official_keys]
    qop_anomalies = [
        event
        for event in official
        if event.get("pull", {}).get("q_over_p") is not None
        and abs(float(event["pull"]["q_over_p"])) >= float(config["qoverp_pull_anomaly_abs_min"])
    ]
    focus_rows = []
    for spec in config["focus_identities"]:
        ident = (str(spec["source_id"]), int(spec["run_id"]), int(spec["event_id"]))
        reco = reco_by_identity.get(ident) or {}
        dump_rows = list(dump_by_identity.get(ident) or [])
        matched = [event for event in official if identity_key(event) == ident]
        focus_rows.append(
            {
                "source_id": spec["source_id"],
                "run_id": spec["run_id"],
                "event_id": spec["event_id"],
                "in_contract_sample": bool(matched),
                "n_contract_pairs": len(matched),
                "long_track": is_long_track_accepted(reco),
                "n_stations_present": reco.get("n_stations_present"),
                "n_segments": reco.get("n_segments"),
                "n_measurements": reco.get("ckf_n_measurements"),
                "ckf_chi2_per_ndof": reco.get("ckf_chi2_per_ndof"),
                "q_over_p_sigma": qoverp_sigma(dump_rows[0]) if dump_rows else None,
                "pairs": [
                    {
                        "station_pair": event["station_pair"],
                        "chi2_per_ndof": event.get("chi2_per_ndof"),
                        "q_over_p_pull": event.get("pull", {}).get("q_over_p"),
                        "p_reco_mev": event.get("p_mev"),
                        "p_truth_mev": event.get("p_truth_mev"),
                    }
                    for event in matched
                ],
                "truth_used_in_eligibility": False,
                "truth_used_as_diagnostic": True,
            }
        )
    return {
        "rescreened": False,
        "tails_deleted": False,
        "clipped": False,
        "n_frozen_1pct": len(frozen_1pct),
        "n_frozen_1pct_remaining": len(remaining_frozen),
        "remaining_frozen_identities": [
            list(key) for key in sorted({identity_key(item) for item in remaining_frozen})
        ],
        "n_contract_qoverp_pull_anomaly": len(qop_anomalies),
        "qoverp_pull_anomaly_identities": [
            list(key) for key in sorted({identity_key(event) for event in qop_anomalies})
        ],
        "focus": focus_rows,
        "note": (
            "Remaining tails are retained.  High q/p pull is diagnostic and "
            "was not used as an eligibility cut."
        ),
    }


def decide_case(
    comparison: Mapping[str, Any],
    raw_replay: Mapping[str, Any],
    contract_replay: Mapping[str, Any],
    remaining: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if bool(contract_replay.get("filter_then_pass_claimed")):
        refuse_filter_pass_claim()
    if bool(contract_replay.get("covariance_model_fixed")):
        refuse_covariance_model_fixed()
    raw_mean = (comparison["samples"][SAMPLE_A]["matched"]["chi2_per_ndof"] or {}).get("mean")
    contract_mean = (comparison["samples"][SAMPLE_B]["matched"]["chi2_per_ndof"] or {}).get("mean")
    contract_median = (comparison["samples"][SAMPLE_B]["matched"]["chi2_per_ndof"] or {}).get("median")
    relative_drop = None
    if raw_mean and contract_mean is not None and float(raw_mean) > 0.0:
        relative_drop = float((float(raw_mean) - float(contract_mean)) / float(raw_mean))
    case_a = relative_drop is not None and relative_drop >= float(
        config["case_gates"]["case_a_relative_mean_drop_min"]
    )
    case_b = int(remaining.get("n_contract_qoverp_pull_anomaly") or 0) >= int(
        config["case_gates"]["case_b_remaining_qop_pull_min"]
    ) or int(remaining.get("n_frozen_1pct_remaining") or 0) > 0
    bulk_fail = contract_median is not None and float(contract_median) >= float(
        config["case_gates"]["case_c_median_chi2_min"]
    )
    case_c = (not bool(contract_replay.get("all_calibrated"))) and (
        (not case_a) or bulk_fail
    )
    if case_a:
        primary = CASE_A
    elif case_b:
        primary = CASE_B
    else:
        primary = CASE_C
    secondaries = [
        name
        for name, flag in ((CASE_B, case_b), (CASE_C, case_c), (CASE_A, case_a))
        if flag and name != primary
    ]
    contracted_calibrated = bool(contract_replay.get("all_calibrated"))
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "raw_chi2_mean": raw_mean,
            "contract_chi2_mean": contract_mean,
            "contract_chi2_median": contract_median,
            "relative_mean_drop": relative_drop,
            "raw_all_calibrated": raw_replay.get("all_calibrated"),
            "contract_all_calibrated": contracted_calibrated,
            "n_frozen_1pct_remaining": remaining.get("n_frozen_1pct_remaining"),
            "n_contract_qoverp_pull_anomaly": remaining.get("n_contract_qoverp_pull_anomaly"),
        },
        "case_flags": {"A": case_a, "B": case_b, "C": case_c},
        "closure_pass": False,
        "filter_then_pass_claimed": False,
        "closure_under_contracted_input_scope": contracted_calibrated,
        "covariance_model_fixed": False,
        "measurement_model_v2_entered": False,
        "tails_dropped": False,
        "selection_designed_from_closure": False,
    }


def inventory_and_validate(config: Mapping[str, Any]) -> dict[str, Any]:
    classified = load_frozen_classified_tails(config)
    frozen_1pct = list(classified["catalogs"]["0.99"]["events"])
    records_by_split: dict[str, list[dict[str, Any]]] = {"construction": [], "validation": []}
    truth_tables: dict[str, Any] = {}
    ntuple_tables: dict[str, Any] = {}
    reco_by_track: dict[tuple[Any, ...], dict[str, Any]] = {}
    reco_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    dump_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    tagged: list[dict[str, Any]] = []
    present = []
    missing = []
    reason_counts: Counter[str] = Counter()
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
        refit = resolve_under_root(
            project_root(),
            str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
        )
        enhanced = refit / config["mc_data"]["enhanced_file"]
        ntuple_tables[source_id] = load_ntuple_event_table(enhanced)
        truth_tables[source_id] = _load_truth_table(
            enhanced, config["mc_data"]["enhanced_tree"], config
        )
        table = ntuple_tables[source_id]
        split_rows = []
        for row in rows:
            record = table.get((int(row["run_id"]), int(row["event_id"])))
            reco = extract_reconstruction(record, float(row.get("p_mev", np.nan)))
            eligibility = evaluate_eligibility(row, reco, config)
            tagged_row = dict(row)
            tagged_row["split"] = split
            tagged_row["contract_eligible"] = bool(eligibility["eligible"])
            tagged_row["reference_long_track"] = is_long_track_accepted(reco)
            tagged_row["ineligible_reason"] = eligibility["ineligible_reason"]
            tagged_row["eligibility_flags"] = eligibility["flags"]
            if eligibility["ineligible_reason"]:
                reason_counts[eligibility["ineligible_reason"]] += 1
            reco_by_track[ckf_track_key(row)] = reco
            reco_by_identity[identity_key(row)] = reco
            dump_by_identity.setdefault(identity_key(row), []).append(row)
            tagged.append(tagged_row)
            split_rows.append(tagged_row)
        records_by_split[split].extend(split_rows)
    events_by_sample: dict[str, list[dict[str, Any]]] = {}
    for name, predicate in (
        (SAMPLE_A, lambda row: True),
        (SAMPLE_B, lambda row: bool(row.get("contract_eligible"))),
        (SAMPLE_C, lambda row: bool(row.get("reference_long_track"))),
    ):
        events: list[dict[str, Any]] = []
        for split, rows in records_by_split.items():
            selected = [row for row in rows if predicate(row)]
            events.extend(
                collect_matched_events(
                    config, records=selected, truth_tables=truth_tables, split=split
                )
            )
        events_by_sample[name] = events
    comparison = compare_samples(
        tagged_rows=tagged,
        reco_by_track=reco_by_track,
        events_by_sample=events_by_sample,
    )
    comparison["ineligible_reason_counts"] = {
        name: int(reason_counts.get(name, 0)) for name in INELIGIBLE_REASONS
    }
    raw_replay = replay_closure(
        config=config,
        records_by_split=records_by_split,
        truth_tables=truth_tables,
        sample_name=SAMPLE_A,
        predicate=lambda row: True,
    )
    contract_replay = replay_closure(
        config=config,
        records_by_split=records_by_split,
        truth_tables=truth_tables,
        sample_name=SAMPLE_B,
        predicate=lambda row: bool(row.get("contract_eligible")),
    )
    remaining = audit_remaining_tails(
        config=config,
        frozen_1pct=frozen_1pct,
        contract_events=events_by_sample[SAMPLE_B],
        reco_by_identity=reco_by_identity,
        dump_by_identity=dump_by_identity,
    )
    mechanism = decide_case(comparison, raw_replay, contract_replay, remaining, config)
    return {
        "present_sources": present,
        "missing_sources": missing,
        "n_dump_rows": len(tagged),
        "comparison": comparison,
        "closure_replay": {SAMPLE_A: raw_replay, SAMPLE_B: contract_replay},
        "remaining_tails": remaining,
        "mechanism": mechanism,
        "wb98_dumps_read_only": True,
        "wb102_read_only": True,
        "frozen_1pct_rescreened": False,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("tails_dropped")):
        refuse_outlier_rejection()
    if bool(mechanism.get("selection_designed_from_closure")):
        refuse_closure_designed_selection()
    if bool(mechanism.get("filter_then_pass_claimed")):
        refuse_filter_pass_claim()
    if bool(mechanism.get("covariance_model_fixed")):
        refuse_covariance_model_fixed()
    if bool(mechanism.get("closure_pass")) and not bool(
        mechanism.get("closure_under_contracted_input_scope")
    ):
        refuse_filter_pass_claim()
    return {
        "verdict": "VALIDATED",
        "decision": DECISION_VALIDATED,
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "closure_pass": False,
        "filter_then_pass_claimed": False,
        "closure_under_contracted_input_scope": mechanism["closure_under_contracted_input_scope"],
        "covariance_model_fixed": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb102_decision": inherited["workbook_102"]["decision"],
        "inherited_wb102_primary_case": inherited["workbook_102"]["primary_case"],
        "inherited_wb102_decision_sha256": inherited["workbook_102"]["decision_sha256"],
        "inherited_wb102_contract_sha256": inherited["workbook_102"].get("contract_sha256"),
        "inherited_wb101_decision_sha256": inherited["workbook_101"]["decision_sha256"],
        "inherited_wb98_decision_sha256": inherited["workbook_98"]["decision_sha256"],
        "wb87_through_wb102_rewritten": False,
        "evidence": mechanism["evidence"],
        "case_flags": mechanism["case_flags"],
    }
