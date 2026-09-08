"""Task B8: Transport Covariance Validation V3.

Official ACTS propagated-covariance closure under the frozen WB103
contracted input.  Does not retune C/Q, drop 100043/37, or enter V2.
Truth is diagnostic only and never enters eligibility.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
from datasets.acts_process_noise_contract import (
    MODEL0,
    MODEL1,
    _as_matrix,
    evaluate_model_records,
)
from datasets.acts_transport_diagnosis import (
    _stats,
    collect_matched_events,
    dump_path_for_source,
    jsonable,
    load_dump_records,
)
from datasets.acts_transport_dump import SITUATION_B
from datasets.acts_transport_tail_analysis import identity_key
from datasets.ckf_contract_validation import (
    CASE_A as WB103_CASE_A,
    DECISION_VALIDATED as WB103_DECISION,
    evaluate_eligibility,
    inherit_frozen_stage as inherit_through_wb102,
    summarize_event_sample,
    summarize_gate_replay,
)
from datasets.ckf_reconstruction_contract import qoverp_sigma
from datasets.ckf_tail_provenance import extract_reconstruction, load_ntuple_event_table

SCHEMA_VERSION = "transport-covariance-validation-v3"
DEFAULT_CONFIG = "configs/transport_covariance_validation_v3.yaml"
TASK = "SB-B8"
WORKBOOK = 104
STATE_NAMES = ("x", "y", "tx", "ty")

CASE_A = "transport_covariance_validated_under_contracted_input"
CASE_B = "ckf_fitting_tail_blocks_transport_validation"
CASE_C = "transport_covariance_shape_not_validated"
CASE_D = "sample_or_contract_inconclusive"

NEXT_STEP = {
    CASE_A: "measurement_model_v2",
    CASE_B: "ckf_momentum_state_audit",
    CASE_C: "transport_material_uncertainty_diagnosis",
    CASE_D: "no_new_source_campaign",
}


class TransportV3Error(ValueError):
    """Raised when the V3 contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise TransportV3Error("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise TransportV3Error("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise TransportV3Error("process noise must not be adjusted to chi2")


def refuse_q_psd_projection() -> None:
    raise TransportV3Error("Q must not be PSD-projected to pass a gate")


def refuse_outlier_rejection() -> None:
    raise TransportV3Error("outlier rejection and chi2 clipping are forbidden")


def refuse_raw_ckf() -> None:
    raise TransportV3Error("raw CKF must not re-enter the official V3 input")


def refuse_focus_drop() -> None:
    raise TransportV3Error("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise TransportV3Error("Measurement Model V2 is not entered in Task B8")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TransportV3Error(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TransportV3Error(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TransportV3Error(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise TransportV3Error("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise TransportV3Error(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_project_q_to_psd",
        "do_not_reject_outliers",
        "do_not_clip_chi2_tails",
        "do_not_use_chi2_in_eligibility",
        "do_not_use_truth_in_eligibility",
        "do_not_drop_focus_identity",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "eligibility_independent_of_closure",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise TransportV3Error(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise TransportV3Error(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    bins = config.get("qoverp_bins_abs_per_mev") or {}
    expected_bins = {
        "high_momentum": [0.0, 1.0e-6],
        "medium_momentum": [1.0e-6, 5.0e-6],
        "low_momentum": [5.0e-6, 1.0],
    }
    for name, edges in expected_bins.items():
        got = [float(v) for v in bins.get(name, [])]
        if got != edges:
            raise TransportV3Error(f"q/p bin {name} must stay frozen")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise TransportV3Error("V3 eligibility must stay identical to WB103")
    roots = {
        Path(str(config.get("output_root"))).as_posix(),
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb103_output_root"))).as_posix(),
    }
    if len(roots) < 3:
        raise TransportV3Error("V3 must not write into the WB98 dump or WB103 roots")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb102(config)
    spec = config["inheritance"]["workbook_103"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise TransportV3Error(f"workbook_103 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise TransportV3Error(f"workbook_103 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise TransportV3Error("workbook_103 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise TransportV3Error("workbook_103 primary case must stay frozen")
    if decision.get("decision") != WB103_DECISION:
        raise TransportV3Error("WB103 decision token mismatch")
    if spec["frozen_primary_case"] != WB103_CASE_A:
        raise TransportV3Error("WB103 primary case token mismatch")
    inherited["workbook_103"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "contract_sha256": spec["config_sha256"],
    }
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise TransportV3Error("WB98 situation must stay frozen")
    return inherited


def _all_sources(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for spec in config["mc_data"]["construction_sources"]:
        items.append(("construction", spec))
    for spec in config["mc_data"]["validation_sources"]:
        items.append(("validation", spec))
    return items


def _focus_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }


def _pair_shape_holds(cell: Mapping[str, Any]) -> bool:
    gates = (cell.get("gates") or {})
    return bool(gates.get("generalized_eigenvalue")) and bool(gates.get("pencil_variance_ratio"))


def _pair_chi2_holds(cell: Mapping[str, Any]) -> bool:
    return bool((cell.get("gates") or {}).get("whitened_chi2_per_ndof"))


def _dominant_axis(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    if array.size < 4:
        return None
    index = int(np.argmax(np.abs(array[:4])))
    return STATE_NAMES[index]


def _generalized_modes(c_emp: np.ndarray, c_prop: np.ndarray) -> list[dict[str, Any]]:
    from scipy.linalg import eigh

    try:
        values, vectors = eigh(c_emp, c_prop)
    except Exception:
        values = np.sort(np.linalg.eigvals(np.linalg.solve(c_prop, c_emp)).real)
        vectors = np.eye(4)
    modes = []
    for value, vector in zip(values, vectors.T):
        axis = _dominant_axis(vector)
        modes.append(
            {
                "lambda_cemp_over_cprop": float(value),
                "dominant_component": axis,
                "weights": {
                    name: float(vector[i]) for i, name in enumerate(STATE_NAMES)
                },
                "overwide_c_prop": bool(np.isfinite(value) and value < 0.25),
                "underwide_c_prop": bool(np.isfinite(value) and value > 4.0),
            }
        )
    return modes


def _row_abs_slope(row: Mapping[str, Any]) -> float | None:
    state = row.get("derived_state") or []
    if len(state) < 4:
        return None
    value = float(np.hypot(float(state[2]), float(state[3])))
    return value if np.isfinite(value) else None


def _row_abs_qoverp(row: Mapping[str, Any]) -> float | None:
    value = abs(float(row.get("q_over_p_per_mev", np.nan)))
    return value if np.isfinite(value) else None


def _qoverp_bin_name(abs_qoverp: float, bins: Mapping[str, Sequence]) -> str | None:
    for name, edges in bins.items():
        if float(edges[0]) <= abs_qoverp < float(edges[1]):
            return str(name)
    return None


def analyze_material_increment(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    q_min = []
    q_frob = []
    c0_frob = []
    c1_frob = []
    n_q_not_psd = 0
    n = 0
    by_qoverp: dict[str, list[float]] = defaultdict(list)
    by_slope: dict[str, list[float]] = {"low": [], "high": []}
    for row in rows:
        q_mat = _as_matrix(row.get("process_noise"), 5)
        c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
        c1 = _as_matrix(row.get("output_covariance_with_material"), 5)
        if q_mat is None or c0 is None or c1 is None:
            continue
        n += 1
        eigs = np.linalg.eigvalsh(0.5 * (q_mat + q_mat.T))
        q_min.append(float(eigs[0]))
        if float(eigs[0]) < -1.0e-18:
            n_q_not_psd += 1
        q_norm = float(np.linalg.norm(q_mat))
        q_frob.append(q_norm)
        c0_frob.append(float(np.linalg.norm(c0)))
        c1_frob.append(float(np.linalg.norm(c1)))
        abs_q = _row_abs_qoverp(row)
        if abs_q is not None and config is not None:
            name = _qoverp_bin_name(abs_q, config.get("qoverp_bins_abs_per_mev") or {})
            if name is not None:
                by_qoverp[name].append(q_norm)
        slope = _row_abs_slope(row)
        if slope is not None:
            by_slope["high" if slope >= 0.002 else "low"].append(q_norm)
    return {
        "n": n,
        "q_not_psd_fraction": float(n_q_not_psd / n) if n else None,
        "q_min_eigenvalue": _stats(q_min),
        "q_frobenius": _stats(q_frob),
        "c0_frobenius": _stats(c0_frob),
        "c1_frobenius": _stats(c1_frob),
        "q_frobenius_by_qoverp_bin": {name: _stats(values) for name, values in sorted(by_qoverp.items())},
        "q_frobenius_by_abs_slope": {name: _stats(values) for name, values in by_slope.items()},
        "q_interpreted_as_fixed_psd_process_noise": False,
        "q_psd_projected": False,
        "q_nonpsd_does_not_fail_v3": True,
        "note": (
            "Q_ACTS := C1-C0 is the honest increment between two possibly "
            "re-linearized trajectories, not a PSD process-noise matrix."
        ),
    }


def analyze_eigen_pencil(
    *,
    official_replays: Mapping[str, Mapping[str, Any]],
    loo_construction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    splits = {}
    for split, replay in official_replays.items():
        model1 = replay["model1"]
        pairs = {}
        for label, cell in (model1.get("per_pair") or {}).items():
            pencil = cell.get("pencil") or {}
            direction = pencil.get("direction")
            c_emp = np.asarray(cell.get("c_emp_eigenvalues") or [], dtype=np.float64)
            c_prop = np.asarray(cell.get("c_prop_mean_eigenvalues") or [], dtype=np.float64)
            gen = (
                cell.get("generalized_eigenvalues_cemp_over_cprop")
                or cell.get("eigenvalue_ratio")
                or []
            )
            pairs[label] = {
                "n_pairs": cell.get("n_pairs") or cell.get("n_matched"),
                "calibrated": cell.get("calibrated"),
                "gates": cell.get("gates"),
                "chi2_per_ndof_mean": cell.get("chi2_per_ndof_mean") or cell.get("chi2_per_ndof"),
                "chi2_per_ndof_median": cell.get("chi2_per_ndof_median"),
                "coverage_probability_95": cell.get("coverage_probability_95"),
                "pencil_ratio": pencil.get("variance_ratio_prop_over_emp") or cell.get("pencil_ratio"),
                "pencil_direction": direction,
                "pencil_dominant_component": _dominant_axis(direction),
                "c_emp": cell.get("c_emp"),
                "c_prop_mean": cell.get("c_prop_mean"),
                "c_emp_eigenvalues": c_emp.tolist() if c_emp.size else cell.get("eigenvalue_ratio"),
                "c_prop_mean_eigenvalues": c_prop.tolist() if c_prop.size else None,
                "generalized_eigenvalues": gen,
                "pull_rms": cell.get("pull_rms"),
                "overcoverage": bool(
                    (cell.get("pencil_ratio") is not None and float(cell["pencil_ratio"]) > 4.0)
                    or any(float(v) < 0.25 for v in gen if v is not None)
                ),
            }
        splits[split] = {
            "all_calibrated": model1.get("all_calibrated"),
            "shape_holds": all(_pair_shape_holds(cell) for cell in (model1.get("per_pair") or {}).values())
            and bool(model1.get("per_pair")),
            "chi2_holds": all(_pair_chi2_holds(cell) for cell in (model1.get("per_pair") or {}).values())
            and bool(model1.get("per_pair")),
            "per_pair": pairs,
        }
    return {
        "scalar_rescale_forbidden": True,
        "official_includes_focus_identity": True,
        "splits": splits,
        "leave_one_focus_out_diagnostic": loo_construction,
        "leave_one_focus_out_is_not_a_cut": True,
    }


def _source_breakdown(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("on_frozen_station_pair"):
            grouped[str(event["source_id"])].append(event)
    return {source: summarize_event_sample(items) for source, items in sorted(grouped.items())}


def _chi2_share(events: Sequence[Mapping[str, Any]], keys: set[tuple[Any, ...]]) -> dict[str, Any]:
    official = [event for event in events if event.get("on_frozen_station_pair")]
    values = [
        float(event["chi2_per_ndof"])
        for event in official
        if event.get("chi2_per_ndof") is not None
    ]
    selected = [
        float(event["chi2_per_ndof"])
        for event in official
        if identity_key(event) in keys and event.get("chi2_per_ndof") is not None
    ]
    total = float(sum(values)) if values else 0.0
    return {
        "n_events": len(official),
        "n_focus_rows": len(selected),
        "chi2_share": float(sum(selected) / total) if total else None,
        "focus_chi2": _stats(selected),
        "all_chi2": _stats(values),
    }


def build_focus_case_study(
    *,
    config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reco_by_identity: Mapping[tuple[Any, ...], Mapping[str, Any]],
    dump_by_identity: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    focus = _focus_keys(config)
    official = [event for event in events if event.get("on_frozen_station_pair")]
    rows = []
    for spec in config["focus_identities"]:
        ident = (str(spec["source_id"]), int(spec["run_id"]), int(spec["event_id"]))
        reco = reco_by_identity.get(ident) or {}
        dump_rows = list(dump_by_identity.get(ident) or [])
        matched = [event for event in official if identity_key(event) == ident]
        if not matched:
            raise TransportV3Error("focus identity 100043/37 must remain in the contracted sample")
        rows.append(
            {
                "source_id": spec["source_id"],
                "run_id": spec["run_id"],
                "event_id": spec["event_id"],
                "retained": True,
                "downweighted": False,
                "truth_qoverp_substituted": False,
                "covariance_inflated": False,
                "in_contract_sample": True,
                "long_track": True,
                "n_stations_present": reco.get("n_stations_present"),
                "n_segments": reco.get("n_segments"),
                "n_measurements": reco.get("ckf_n_measurements"),
                "ckf_chi2_per_ndof": reco.get("ckf_chi2_per_ndof"),
                "q_over_p_sigma": qoverp_sigma(dump_rows[0]) if dump_rows else None,
                "q_over_p_reco": dump_rows[0].get("q_over_p_per_mev") if dump_rows else None,
                "pairs": [
                    {
                        "station_pair": event["station_pair"],
                        "split": event.get("split"),
                        "chi2_per_ndof": event.get("chi2_per_ndof"),
                        "residual": event.get("residual"),
                        "pull": event.get("pull"),
                        "p_reco_mev": event.get("p_mev"),
                        "p_truth_mev": event.get("p_truth_mev"),
                    }
                    for event in matched
                ],
                "aggregate_chi2_share": _chi2_share(official, {ident}),
                "truth_used_in_eligibility": False,
                "truth_used_as_diagnostic": True,
            }
        )
    return {
        "retained": True,
        "selection_redesigned": False,
        "events": rows,
        "note": (
            "100043/37 is a legal contracted long track.  It is a frozen "
            "diagnostic case, not a licence to drop or rescale."
        ),
    }


def decide_case(
    official: Mapping[str, Any],
    eigen: Mapping[str, Any],
    focus: Mapping[str, Any],
    event_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(focus.get("retained", True)):
        refuse_focus_drop()
    construction = official["splits"]["construction"]["model1"]
    validation = official["splits"]["validation"]["model1"]
    official_ok = bool(construction.get("all_calibrated")) and bool(validation.get("all_calibrated"))
    val_shape = bool(eigen["splits"]["validation"].get("shape_holds"))
    val_chi2 = bool(eigen["splits"]["validation"].get("chi2_holds"))
    con_shape = bool(eigen["splits"]["construction"].get("shape_holds"))
    n_official = int((event_summary.get("chi2_per_ndof") or {}).get("finite_count") or 0)
    median = (event_summary.get("chi2_per_ndof") or {}).get("median")
    typical_overcover = median is not None and float(median) < 1.0
    focus_share = None
    if focus.get("events"):
        focus_share = (focus["events"][0].get("aggregate_chi2_share") or {}).get("chi2_share")
    tail_present = bool(focus.get("events")) and focus_share is not None and float(focus_share) > 0.0
    typical_shape_holds = val_shape
    typical_chi2_holds = val_chi2
    if official_ok:
        primary = CASE_A
    elif n_official < 40:
        primary = CASE_D
    elif typical_shape_holds and typical_chi2_holds and tail_present:
        primary = CASE_B
    elif not typical_shape_holds:
        primary = CASE_C
    else:
        primary = CASE_C
    secondaries = []
    if primary != CASE_B and tail_present and not official_ok:
        secondaries.append(CASE_B)
    if primary != CASE_C and not typical_shape_holds:
        secondaries.append(CASE_C)
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "evidence": {
            "construction_all_calibrated": construction.get("all_calibrated"),
            "validation_all_calibrated": validation.get("all_calibrated"),
            "validation_shape_holds": val_shape,
            "validation_chi2_holds": val_chi2,
            "construction_shape_holds": con_shape,
            "event_chi2_mean": (event_summary.get("chi2_per_ndof") or {}).get("mean"),
            "event_chi2_median": median,
            "typical_overcoverage": typical_overcover,
            "focus_chi2_share": focus_share,
            "n_official_pairs": n_official,
        },
        "case_flags": {
            "A": primary == CASE_A,
            "B": primary == CASE_B or CASE_B in secondaries,
            "C": primary == CASE_C or CASE_C in secondaries,
            "D": primary == CASE_D,
        },
        "closure_pass": official_ok,
        "measurement_model_v2_authorized": official_ok,
        "measurement_model_v2_entered": False,
        "focus_retained": True,
        "raw_ckf_used": False,
        "covariance_rescaled": False,
        "q_psd_projected": False,
        "tails_dropped": False,
    }


def inventory_and_validate(config: Mapping[str, Any]) -> dict[str, Any]:
    records_by_split: dict[str, list[dict[str, Any]]] = {"construction": [], "validation": []}
    truth_tables: dict[str, Any] = {}
    reco_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    dump_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    contracted: list[dict[str, Any]] = []
    n_raw = 0
    n_ineligible = 0
    present = []
    missing = []
    hashes = {"geometry_hash": set(), "field_hash": set(), "material_hash": set(), "conditions_hash": set()}
    focus = _focus_keys(config)
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
        table = load_ntuple_event_table(enhanced)
        truth_tables[source_id] = _load_truth_table(
            enhanced, config["mc_data"]["enhanced_tree"], config
        )
        for row in rows:
            if bool(row.get("is_truth", False)):
                continue
            n_raw += 1
            record = table.get((int(row["run_id"]), int(row["event_id"])))
            reco = extract_reconstruction(record, float(row.get("p_mev", np.nan)))
            eligibility = evaluate_eligibility(row, reco, config)
            if not eligibility["eligible"]:
                n_ineligible += 1
                continue
            tagged = dict(row)
            tagged["split"] = split
            tagged["contract_eligible"] = True
            contracted.append(tagged)
            records_by_split[split].append(tagged)
            reco_by_identity[identity_key(row)] = reco
            dump_by_identity.setdefault(identity_key(row), []).append(row)
            for key in hashes:
                if row.get(key) or row.get("material_map_hash") and key == "material_hash":
                    value = row.get(key) or row.get("material_map_hash")
                    if value:
                        hashes[key].add(str(value))
    if n_ineligible == 0 and n_raw and len(contracted) == n_raw:
        refuse_raw_ckf()
    if not any(identity_key(row) in focus for row in contracted):
        refuse_focus_drop()
    events: list[dict[str, Any]] = []
    for split, rows in records_by_split.items():
        events.extend(
            collect_matched_events(config, records=rows, truth_tables=truth_tables, split=split)
        )
    official_events = [event for event in events if event.get("on_frozen_station_pair")]
    event_summary = summarize_event_sample(official_events)
    official_replays = {}
    for split, rows in records_by_split.items():
        model1 = summarize_gate_replay(
            evaluate_model_records(rows, config, model_key=MODEL1, truth_tables=truth_tables)
        )
        model0 = summarize_gate_replay(
            evaluate_model_records(rows, config, model_key=MODEL0, truth_tables=truth_tables)
        )
        official_replays[split] = {"model1": model1, "model0": model0}
    loo_rows = [
        row
        for row in records_by_split["construction"]
        if identity_key(row) not in focus
    ]
    loo = summarize_gate_replay(
        evaluate_model_records(loo_rows, config, model_key=MODEL1, truth_tables=truth_tables)
    )
    loo_report = {
        "diagnostic_only": True,
        "not_a_cut": True,
        "focus_retained_in_official": True,
        "model1": loo,
        "shape_holds": all(_pair_shape_holds(cell) for cell in (loo.get("per_pair") or {}).values()),
        "chi2_holds": all(_pair_chi2_holds(cell) for cell in (loo.get("per_pair") or {}).values()),
        "all_calibrated": loo.get("all_calibrated"),
    }
    for split, rows in records_by_split.items():
        raw = evaluate_model_records(rows, config, model_key=MODEL1, truth_tables=truth_tables)
        split_events = [event for event in official_events if event.get("split") == split]
        for label, cell in (raw.get("per_pair") or {}).items():
            dest = official_replays[split]["model1"]["per_pair"].setdefault(label, {})
            dest["c_emp_eigenvalues"] = cell.get("c_emp_eigenvalues")
            dest["c_prop_mean_eigenvalues"] = cell.get("c_prop_mean_eigenvalues")
            dest["generalized_eigenvalues_cemp_over_cprop"] = cell.get(
                "generalized_eigenvalues_cemp_over_cprop"
            )
            dest["coverage_probability_95"] = cell.get("coverage_probability_95")
            dest["pencil"] = cell.get("pencil")
            dest["n_matched"] = cell.get("n_matched") or cell.get("n_pairs")
            bundle = pair_covariance_bundle(split_events, rows, label)
            dest["generalized_modes"] = bundle.get("modes") or []
            dest["c_emp"] = bundle.get("c_emp")
            dest["c_prop_mean"] = bundle.get("c_prop_mean")
            dest["pull_rms"] = bundle.get("pull_rms")
    eigen = analyze_eigen_pencil(official_replays=official_replays, loo_construction=loo_report)
    for split in official_replays:
        for label, cell in official_replays[split]["model1"]["per_pair"].items():
            dest = eigen["splits"][split]["per_pair"][label]
            dest["generalized_modes"] = cell.get("generalized_modes")
            dest["c_emp"] = cell.get("c_emp")
            dest["c_prop_mean"] = cell.get("c_prop_mean")
            dest["pull_rms"] = cell.get("pull_rms")
    focus_study = build_focus_case_study(
        config=config,
        events=official_events,
        reco_by_identity=reco_by_identity,
        dump_by_identity=dump_by_identity,
    )
    material = analyze_material_increment(contracted, config)
    mechanism = decide_case(
        {"splits": official_replays},
        eigen,
        focus_study,
        event_summary,
    )
    provenance = {
        key: sorted(values)[0] if len(values) == 1 else sorted(values)
        for key, values in hashes.items()
    }
    expected = config["frozen_provenance_hashes"]
    for name, key in (
        ("geometry_hash", "geometry_hash"),
        ("field_hash", "field_hash"),
        ("material_map_hash", "material_hash"),
        ("conditions_hash", "conditions_hash"),
    ):
        got = provenance.get(key)
        if got != expected[name] and got != [expected[name]]:
            raise TransportV3Error(f"{name} drifted from the frozen WB98 dump")
    return {
        "present_sources": present,
        "missing_sources": missing,
        "inventory": {
            "official_input_scope": "contract_eligible",
            "n_raw_dump_rows_seen": n_raw,
            "n_ineligible_excluded": n_ineligible,
            "n_contracted_rows": len(contracted),
            "n_official_pair_events": len(official_events),
            "n_construction_rows": len(records_by_split["construction"]),
            "n_validation_rows": len(records_by_split["validation"]),
            "raw_ckf_used_officially": False,
            "eligibility_used_chi2": False,
            "eligibility_used_truth": False,
            "focus_identity_retained": True,
            "provenance_hashes": {
                "geometry_hash": expected["geometry_hash"],
                "field_hash": expected["field_hash"],
                "material_map_hash": expected["material_map_hash"],
                "conditions_hash": expected["conditions_hash"],
            },
        },
        "metrics": {
            "event_level": event_summary,
            "source_level": _source_breakdown(official_events),
            "gate_level": official_replays,
            "typical": {
                "chi2_median": (event_summary.get("chi2_per_ndof") or {}).get("median"),
                "validation_chi2_holds": eigen["splits"]["validation"]["chi2_holds"],
                "validation_shape_holds": eigen["splits"]["validation"]["shape_holds"],
                "overcoverage": (
                    (event_summary.get("chi2_per_ndof") or {}).get("median") is not None
                    and float((event_summary["chi2_per_ndof"] or {}).get("median") or 1.0) < 1.0
                ),
            },
            "tail": focus_study["events"][0]["aggregate_chi2_share"] if focus_study["events"] else None,
            "failure_layers": {
                "typical_event_calibration": {
                    "median_chi2_low": (
                        (event_summary.get("chi2_per_ndof") or {}).get("median") is not None
                        and float((event_summary["chi2_per_ndof"] or {}).get("median") or 1.0) < 1.0
                    ),
                    "validation_chi2_holds": eigen["splits"]["validation"]["chi2_holds"],
                },
                "tail_contribution": focus_study["events"][0]["aggregate_chi2_share"]
                if focus_study["events"]
                else None,
                "covariance_shape_mismatch": {
                    "validation_shape_holds": eigen["splits"]["validation"]["shape_holds"],
                    "construction_shape_holds": eigen["splits"]["construction"]["shape_holds"],
                    "scalar_rescale_forbidden": True,
                },
            },
            "material_increment": material,
            "same_wb98_dump": True,
            "same_c0_c1_q": True,
            "same_gates": True,
        },
        "eigen": eigen,
        "focus": focus_study,
        "mechanism": mechanism,
        "wb98_dumps_read_only": True,
        "wb103_contract_read_only": True,
    }


def pair_covariance_bundle(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    pair_label: str,
) -> dict[str, Any]:
    """C_emp, mean C_prop, λ(C_emp, C_prop), and pull RMS.  No rescale."""
    index = {
        (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
            f"({int(row.get('source_station', -1))},{int(row.get('target_station', -1))})",
        ): row
        for row in rows
    }
    residuals = []
    covs = []
    pulls: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    for event in events:
        if str(event.get("station_pair")) != pair_label:
            continue
        residual = [event.get("residual", {}).get(name) for name in STATE_NAMES]
        if any(value is None or not np.isfinite(float(value)) for value in residual):
            continue
        row = index.get(
            (
                str(event["source_id"]),
                int(event["run_id"]),
                int(event["event_id"]),
                str(event["station_pair"]),
            )
        )
        if row is None:
            continue
        cov = _as_matrix((row.get(MODEL1) or {}).get("covariance_4x4"), 4)
        if cov is None:
            continue
        residuals.append([float(value) for value in residual])
        covs.append(cov)
        for name in STATE_NAMES:
            pull = (event.get("pull") or {}).get(name)
            if pull is not None and np.isfinite(float(pull)):
                pulls[name].append(float(pull))
    if len(residuals) < 8:
        return {"n": len(residuals), "modes": [], "c_emp": None, "c_prop_mean": None}
    residual_array = np.asarray(residuals, dtype=np.float64)
    demeaned = residual_array - residual_array.mean(axis=0)
    c_emp = np.cov(demeaned.T)
    c_prop = np.mean(np.asarray(covs, dtype=np.float64), axis=0)
    return {
        "n": len(residuals),
        "modes": _generalized_modes(c_emp, c_prop),
        "c_emp": c_emp.tolist(),
        "c_prop_mean": c_prop.tolist(),
        "pull_rms": {name: _stats(values) for name, values in pulls.items()},
    }


def pair_generalized_modes(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    pair_label: str,
) -> list[dict[str, Any]]:
    """λ(C_emp, C_prop) with dominant physical component.  No rescale."""
    return list(pair_covariance_bundle(events, rows, pair_label).get("modes") or [])


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("raw_ckf_used")):
        refuse_raw_ckf()
    if bool(mechanism.get("tails_dropped")) or not bool(mechanism.get("focus_retained", True)):
        refuse_focus_drop()
    if bool(mechanism.get("covariance_rescaled")):
        refuse_covariance_rescale()
    if bool(mechanism.get("q_psd_projected")):
        refuse_q_psd_projection()
    primary = mechanism["primary_case"]
    return {
        "verdict": "ESTABLISHED" if primary == CASE_A else "NOT_ESTABLISHED",
        "decision": primary,
        "primary_case": primary,
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "closure_pass": bool(mechanism.get("closure_pass")),
        "measurement_model_v2_authorized": bool(mechanism.get("measurement_model_v2_authorized")),
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "focus_identity_retained": True,
        "covariance_rescaled": False,
        "q_psd_projected": False,
        "inherited_wb103_decision": inherited["workbook_103"]["decision"],
        "inherited_wb103_primary_case": inherited["workbook_103"]["primary_case"],
        "inherited_wb103_decision_sha256": inherited["workbook_103"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "inherited_wb98_decision_sha256": inherited["workbook_98"]["decision_sha256"],
        "wb87_through_wb103_rewritten": False,
        "evidence": mechanism["evidence"],
        "case_flags": mechanism["case_flags"],
    }
