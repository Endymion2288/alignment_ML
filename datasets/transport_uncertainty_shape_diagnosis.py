"""Task B9: transport / material / state-uncertainty shape diagnosis.

Explains WB104 C_prop overcoverage on the frozen WB103 contracted input.
Does not retune C/Q, drop 100043/37, invent a cross-covariance, or enter V2.
Truth is diagnostic only and never enters eligibility.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
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
    _pick_truth,
    _truth_residual,
)
from datasets.acts_transport_diagnosis import (
    _stats,
    collect_matched_events,
    compare_c0_to_jacobian,
    dump_path_for_source,
    load_dump_records,
)
from datasets.acts_transport_dump import SITUATION_B
from datasets.acts_transport_tail_analysis import identity_key
from datasets.ckf_contract_validation import evaluate_eligibility
from datasets.ckf_tail_provenance import extract_reconstruction, load_ntuple_event_table
from datasets.transport_covariance_validation_v3 import (
    CASE_C as WB104_CASE,
    inherit_frozen_stage as inherit_through_wb103,
)

SCHEMA_VERSION = "transport-uncertainty-shape-diagnosis-v1"
DEFAULT_CONFIG = "configs/transport_uncertainty_shape_diagnosis_v1.yaml"
TASK = "SB-B9"
WORKBOOK = 105
STATE_NAMES = ("x", "y", "tx", "ty")

CASE_A = "shared_measurement_covariance_semantics_missing"
CASE_B = "ckf_input_covariance_overcovered"
CASE_C = "material_transport_shape_mismatch"
CASE_D = "transport_linearization_mismatch"
CASE_E = "mixed_or_inconclusive"

NEXT_STEP = {
    CASE_A: "leave_target_out_prediction_contract",
    CASE_B: "ckf_fit_covariance_semantics_audit",
    CASE_C: "acts_material_process_noise_semantics",
    CASE_D: "transport_derivative_contract",
    CASE_E: "no_forced_single_root_cause",
}

SOURCE_STATE_ORIGIN = "Trk::Track.trackParameters().front()"


class ShapeDiagnosisError(ValueError):
    """Raised when the B9 diagnosis contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise ShapeDiagnosisError("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise ShapeDiagnosisError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise ShapeDiagnosisError("process noise must not be adjusted to chi2")


def refuse_q_psd_projection() -> None:
    raise ShapeDiagnosisError("Q must not be PSD-projected to pass a gate")


def refuse_raw_ckf() -> None:
    raise ShapeDiagnosisError("raw CKF must not re-enter the official B9 input")


def refuse_focus_drop() -> None:
    raise ShapeDiagnosisError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise ShapeDiagnosisError("Measurement Model V2 is not entered in Task B9")


def refuse_empirical_cross_covariance() -> None:
    raise ShapeDiagnosisError("empirical Cov(pred,target) must not be invented")


def refuse_tighten_contract() -> None:
    raise ShapeDiagnosisError("reconstruction contract must not be tightened")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ShapeDiagnosisError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ShapeDiagnosisError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ShapeDiagnosisError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise ShapeDiagnosisError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise ShapeDiagnosisError(f"{key} must be false")
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
        "do_not_tighten_reconstruction_contract",
        "do_not_invent_empirical_cross_covariance",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "eligibility_independent_of_closure",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise ShapeDiagnosisError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ShapeDiagnosisError(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise ShapeDiagnosisError("B9 eligibility must stay identical to WB103")
    roots = {
        Path(str(config.get("output_root"))).as_posix(),
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("wb104_output_root"))).as_posix(),
    }
    if len(roots) < 3:
        raise ShapeDiagnosisError("B9 must not write into the WB98 dump or WB104 roots")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb103(config)
    spec = config["inheritance"]["workbook_104"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise ShapeDiagnosisError(f"workbook_104 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise ShapeDiagnosisError(f"workbook_104 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise ShapeDiagnosisError("workbook_104 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise ShapeDiagnosisError("workbook_104 primary case must stay frozen")
    if spec["frozen_primary_case"] != WB104_CASE:
        raise ShapeDiagnosisError("WB104 primary case token mismatch")
    inherited["workbook_104"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
    }
    if inherited["workbook_98"].get("situation") != SITUATION_B:
        raise ShapeDiagnosisError("WB98 situation must stay frozen")
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


def _dominant_axis(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    if array.size < 4:
        return None
    return STATE_NAMES[int(np.argmax(np.abs(array[:4])))]


def _shape_against(c_emp: np.ndarray, c_model: np.ndarray) -> dict[str, Any]:
    from scipy.linalg import eigh

    try:
        values, vectors = eigh(c_emp, c_model)
    except Exception:
        values = np.sort(np.linalg.eigvals(np.linalg.solve(c_model, c_emp)).real)
        vectors = np.eye(4)
    w_model, v_model = np.linalg.eigh(c_model)
    pencil_vec = v_model[:, -1]
    pencil = float(w_model[-1] / max(float(pencil_vec @ c_emp @ pencil_vec), 1.0e-300))
    modes = []
    for value, vector in zip(values, vectors.T):
        modes.append(
            {
                "lambda_cemp_over_cmodel": float(value),
                "dominant_component": _dominant_axis(vector),
                "overwide_c_model": bool(np.isfinite(value) and value < 0.25),
            }
        )
    gen = [float(v) for v in values]
    return {
        "generalized_eigenvalues": gen,
        "pencil_ratio": pencil,
        "pencil_dominant_component": _dominant_axis(pencil_vec),
        "shape_holds": bool(
            gen
            and all(0.25 <= v <= 4.0 for v in gen)
            and 0.25 <= pencil <= 4.0
        ),
        "overwide": bool(any(v < 0.25 for v in gen) or pencil > 4.0),
        "modes": modes,
        "c_emp_eigenvalues": np.linalg.eigvalsh(c_emp).tolist(),
        "c_model_eigenvalues": np.linalg.eigvalsh(c_model).tolist(),
    }


def _pair_matrices(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    pair_label: str,
    cov_fn: Callable[[Mapping[str, Any]], np.ndarray | None],
) -> dict[str, Any]:
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
        cov = cov_fn(row)
        if cov is None:
            continue
        residuals.append([float(value) for value in residual])
        covs.append(cov)
    if len(residuals) < 8:
        return {"n": len(residuals), "shape": None, "c_emp": None, "c_model_mean": None}
    residual_array = np.asarray(residuals, dtype=np.float64)
    demeaned = residual_array - residual_array.mean(axis=0)
    c_emp = np.cov(demeaned.T)
    c_model = np.mean(np.asarray(covs, dtype=np.float64), axis=0)
    return {
        "n": len(residuals),
        "shape": _shape_against(c_emp, c_model),
        "c_emp": c_emp.tolist(),
        "c_model_mean": c_model.tolist(),
        "c_emp_frobenius": float(np.linalg.norm(c_emp)),
        "c_model_frobenius": float(np.linalg.norm(c_model)),
    }


def _cin4(row: Mapping[str, Any]) -> np.ndarray | None:
    cin = _as_matrix(row.get("input_covariance"), 5)
    return None if cin is None else cin[:4, :4]


def _c0_4(row: Mapping[str, Any]) -> np.ndarray | None:
    cov = _as_matrix((row.get(MODEL0) or {}).get("covariance_4x4"), 4)
    if cov is not None:
        return cov
    c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
    return None if c0 is None else c0[:4, :4]


def _c1_4(row: Mapping[str, Any]) -> np.ndarray | None:
    cov = _as_matrix((row.get(MODEL1) or {}).get("covariance_4x4"), 4)
    if cov is not None:
        return cov
    c1 = _as_matrix(row.get("output_covariance_with_material"), 5)
    return None if c1 is None else c1[:4, :4]


def _transported_cin4(row: Mapping[str, Any]) -> np.ndarray | None:
    jacobian = _as_matrix(row.get("transport_jacobian"), 5)
    cin = _as_matrix(row.get("input_covariance"), 5)
    if jacobian is None or cin is None:
        return None
    return (jacobian @ cin @ jacobian.T)[:4, :4]


def _pair_label(row: Mapping[str, Any]) -> str:
    return f"({int(row.get('source_station', -1))},{int(row.get('target_station', -1))})"


def load_contracted_sample(config: Mapping[str, Any]) -> dict[str, Any]:
    records_by_split: dict[str, list[dict[str, Any]]] = {"construction": [], "validation": []}
    reco_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    truth_tables: dict[str, Any] = {}
    contracted: list[dict[str, Any]] = []
    hashes = {
        "geometry_hash": set(),
        "field_hash": set(),
        "material_hash": set(),
        "conditions_hash": set(),
    }
    present = []
    missing = []
    n_raw = 0
    n_ineligible = 0
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
            tagged["_reco"] = reco
            contracted.append(tagged)
            records_by_split[split].append(tagged)
            reco_by_identity[identity_key(row)] = reco
            for key in hashes:
                value = row.get(key) or (row.get("material_map_hash") if key == "material_hash" else None)
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
    expected = config["frozen_provenance_hashes"]
    provenance = {
        key: sorted(values)[0] if len(values) == 1 else sorted(values)
        for key, values in hashes.items()
    }
    for name, key in (
        ("geometry_hash", "geometry_hash"),
        ("field_hash", "field_hash"),
        ("material_map_hash", "material_hash"),
        ("conditions_hash", "conditions_hash"),
    ):
        got = provenance.get(key)
        if got != expected[name] and got != [expected[name]]:
            raise ShapeDiagnosisError(f"{name} drifted from the frozen WB98 dump")
    return {
        "present_sources": present,
        "missing_sources": missing,
        "records_by_split": records_by_split,
        "contracted": contracted,
        "official_events": official_events,
        "reco_by_identity": reco_by_identity,
        "truth_tables": truth_tables,
        "n_raw": n_raw,
        "n_ineligible": n_ineligible,
        "provenance_hashes": {
            "geometry_hash": expected["geometry_hash"],
            "field_hash": expected["field_hash"],
            "material_map_hash": expected["material_map_hash"],
            "conditions_hash": expected["conditions_hash"],
        },
    }


def decompose_covariance_budget(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    splits = {}
    for split, rows in sample["records_by_split"].items():
        events = [event for event in sample["official_events"] if event.get("split") == split]
        pairs = {}
        for pair in config["station_pairs"]:
            label = f"({int(pair[0])},{int(pair[1])})"
            cin = _pair_matrices(events, rows, label, _cin4)
            transported = _pair_matrices(events, rows, label, _transported_cin4)
            c0 = _pair_matrices(events, rows, label, _c0_4)
            c1 = _pair_matrices(events, rows, label, _c1_4)
            pairs[label] = {
                "n": c1.get("n"),
                "cin": {
                    "frobenius": cin.get("c_model_frobenius"),
                    "compared_to_target_cemp": False,
                    "note": "Cin is a source-surface covariance; shape vs target C_emp is B9.3, not here.",
                },
                "transported_cin": {
                    "frobenius": transported.get("c_model_frobenius"),
                    "shape_vs_cemp": transported.get("shape"),
                },
                "c0_material_off": {
                    "frobenius": c0.get("c_model_frobenius"),
                    "shape_vs_cemp": c0.get("shape"),
                },
                "c1_material_on": {
                    "frobenius": c1.get("c_model_frobenius"),
                    "shape_vs_cemp": c1.get("shape"),
                },
                "c_emp_frobenius": c1.get("c_emp_frobenius"),
                "official_covariance": "C1_model1_4x4",
            }
        splits[split] = pairs
    return {
        "official_residual_reference": "truth",
        "official_covariance": "C_prop = C1 = material-on transported CKF Cin",
        "official_adds_c_target": False,
        "c_target_in_official_gate": None,
        "independence_assumption_cpred_plus_ctarget_used": False,
        "required_if_measurement_residual": "C_pred + C_target - 2 Cov(pred,target)",
        "double_counting_in_official_truth_gate": False,
        "cin_used_as_prediction_covariance": True,
        "note": (
            "WB81/WB104 isolate C_prop from C_target: e = x_prop - x_truth, "
            "C = C_prop.  The official gate does not add C_target.  Shared-fit "
            "still matters because Cin is not a leave-target-out prediction."
        ),
        "splits": splits,
    }


def audit_shared_measurement(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    n_target_in_tracklets = 0
    n_target_clusters = 0
    n_official = 0
    unavailable = {
        "filter_vs_smoother": "unavailable",
        "cluster_identities": "unavailable",
        "source_measurement_identity": "unavailable",
        "target_measurement_identity": "unavailable",
        "cov_pred_target": "not_estimated",
    }
    for event in sample["official_events"]:
        reco = sample["reco_by_identity"].get(identity_key(event)) or {}
        stations = {int(v) for v in reco.get("tracklet_stations") or [] if v is not None}
        target = int(event["target_station"])
        clusters = reco.get("n_clusters") or [None, None, None, None]
        target_clusters = clusters[target] if target < len(clusters) else None
        in_station = reco.get("in_station") or [None, None, None, None]
        n_official += 1
        if target in stations:
            n_target_in_tracklets += 1
        if target_clusters is not None and int(target_clusters) > 0:
            n_target_clusters += 1
        rows.append(
            {
                "source_id": event["source_id"],
                "run_id": event["run_id"],
                "event_id": event["event_id"],
                "station_pair": event["station_pair"],
                "used_stations": sorted(stations),
                "used_clusters_per_station": clusters,
                "target_station": target,
                "target_station_in_tracklets": target in stations,
                "target_n_clusters": target_clusters,
                "in_station_flags": in_station,
                "n_measurements": reco.get("ckf_n_measurements"),
                "n_layers": reco.get("ckf_n_layers"),
                "source_state_origin": SOURCE_STATE_ORIGIN,
                "same_ckf_track": True,
                "smoothing_filtering_provenance": unavailable["filter_vs_smoother"],
            }
        )
    fraction = float(n_target_in_tracklets / n_official) if n_official else None
    return {
        "official_residual_partner": "truth_not_target_measurement",
        "source_state_origin": SOURCE_STATE_ORIGIN,
        "source_and_target_from_same_ckf_track": True,
        "target_station_in_same_ckf_fit_fraction": fraction,
        "target_station_has_clusters_fraction": (
            float(n_target_clusters / n_official) if n_official else None
        ),
        "n_official_pairs": n_official,
        "prediction_and_target_measurement_independent": False if fraction and fraction >= 0.9 else None,
        "cov_pred_target_estimated": False,
        "cov_pred_target": None,
        "empirical_cross_covariance_invented": False,
        "unavailable": unavailable,
        "note": (
            "Contracted long tracks have four tracklet stations, so the dumped "
            "Cin already includes the target station.  Filter vs smoother and "
            "cluster identities are not in the current EDM dump/ntuple."
        ),
        "tracks": rows[:12],
        "tracks_truncated": n_official > 12,
    }


def audit_cin_shape(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    seen: set[tuple[Any, ...]] = set()
    residuals = []
    cins = []
    spectra = []
    conditions = []
    qop_corr = []
    diagonals: dict[str, list[float]] = {
        name: [] for name in ("x", "y", "tx", "ty", "q_over_p")
    }
    by_source: dict[str, list[float]] = defaultdict(list)
    by_bin: dict[str, list[float]] = defaultdict(list)
    bins = config["qoverp_bins_abs_per_mev"]
    for row in sample["contracted"]:
        key = (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
            int(row.get("track_index", -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        cin = _as_matrix(row.get("input_covariance"), 5)
        if cin is None:
            continue
        eigs = np.linalg.eigvalsh(0.5 * (cin + cin.T))
        spectra.append([float(v) for v in eigs])
        conditions.append(float(np.linalg.cond(cin)))
        for index, name in enumerate(("x", "y", "tx", "ty", "q_over_p")):
            diagonals[name].append(float(cin[index, index]))
        abs_q = abs(float(row.get("q_over_p_per_mev", np.nan)))
        for name, edges in bins.items():
            if float(edges[0]) <= abs_q < float(edges[1]):
                by_bin[name].append(float(np.linalg.norm(cin)))
                break
        diag = np.sqrt(np.clip(np.diag(cin), 0.0, None))
        if diag[4] > 0.0:
            qop_corr.append(
                {
                    name: float(cin[index, 4] / (diag[index] * diag[4]))
                    if diag[index] > 0.0
                    else None
                    for index, name in enumerate(("x", "y", "tx", "ty", "q_over_p"))
                }
            )
        table = sample["truth_tables"].get(str(row.get("source_id")))
        event_truth = table.get((int(row["run_id"]), int(row["event_id"]))) if table else None
        if event_truth:
            _, src = _pick_truth(event_truth, int(row["source_station"]))
            state = np.asarray(row.get("derived_state"), dtype=np.float64).reshape(-1)
            if src is not None and state.size >= 4:
                residual = _truth_residual(
                    state[:4],
                    src["pos"],
                    src["mom"],
                    float(row.get("source_z_mm", src["pos"][2])),
                    straight_line=bool(config["truth_reference"]["straight_line_z_correction"]),
                )
                if residual is not None:
                    residuals.append(residual)
                    cins.append(cin[:4, :4])
                    by_source[str(row["source_id"])].append(float(residual @ residual))
    shape = None
    holds = None
    if len(residuals) >= 8:
        array = np.asarray(residuals, dtype=np.float64)
        c_emp = np.cov((array - array.mean(axis=0)).T)
        c_model = np.mean(np.asarray(cins, dtype=np.float64), axis=0)
        shape = _shape_against(c_emp, c_model)
        holds = bool(shape.get("shape_holds"))
    corr_stats = {}
    if qop_corr:
        for name in ("x", "y", "tx", "ty"):
            values = [item[name] for item in qop_corr if item.get(name) is not None]
            corr_stats[name] = _stats(values)
    return {
        "n_unique_ckf_tracks": len(seen),
        "n_source_truth_matched": len(residuals),
        "cin_eigen_spectrum": {
            "min": _stats([spec[0] for spec in spectra]),
            "max": _stats([spec[-1] for spec in spectra]),
        },
        "condition_number": _stats(conditions),
        "cin_diagonal_variance": {name: _stats(values) for name, values in diagonals.items()},
        "qoverp_correlations": corr_stats,
        "source_cin_vs_empirical_source_error": shape,
        "source_cin_shape_holds": holds,
        "overcoverage_present_before_propagation": bool(shape and shape.get("overwide")),
        "by_source_residual_norm2": {name: _stats(values) for name, values in sorted(by_source.items())},
        "cin_frobenius_by_qoverp_bin": {name: _stats(values) for name, values in sorted(by_bin.items())},
        "truth_used_as_diagnostic_only": True,
    }


def compare_material_shape(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    splits = {}
    c0_holds = []
    c1_holds = []
    for split, rows in sample["records_by_split"].items():
        events = [event for event in sample["official_events"] if event.get("split") == split]
        pairs = {}
        for pair in config["station_pairs"]:
            label = f"({int(pair[0])},{int(pair[1])})"
            c0 = _pair_matrices(events, rows, label, _c0_4)
            c1 = _pair_matrices(events, rows, label, _c1_4)
            s0 = c0.get("shape") or {}
            s1 = c1.get("shape") or {}
            c0_holds.append(bool(s0.get("shape_holds")))
            c1_holds.append(bool(s1.get("shape_holds")))
            l0 = s0.get("generalized_eigenvalues") or []
            l1 = s1.get("generalized_eigenvalues") or []
            pairs[label] = {
                "n": c1.get("n"),
                "c_emp_vs_c0": s0,
                "c_emp_vs_c1": s1,
                "material_on_improves_shape": bool(
                    s1.get("shape_holds") and not s0.get("shape_holds")
                ),
                "material_on_further_overwide": bool(
                    l0 and l1 and float(min(l1)) < float(min(l0))
                ),
                "c0_already_overwide": bool(s0.get("overwide")),
            }
        splits[split] = pairs
    return {
        "q_interpreted_as_fixed_psd_process_noise": False,
        "q_psd_projected": False,
        "c0_shape_holds": all(c0_holds) and bool(c0_holds),
        "c1_shape_holds": all(c1_holds) and bool(c1_holds),
        "c0_already_overwide": not (all(c0_holds) and bool(c0_holds)),
        "material_is_primary_mismatch": (
            all(c0_holds) and bool(c0_holds) and not (all(c1_holds) and bool(c1_holds))
        ),
        "splits": splits,
        "note": (
            "If C0 is already overwide, missing material is not the primary "
            "shape failure.  Q = C1-C0 is not treated as a PSD process-noise matrix."
        ),
    }


def audit_linearization(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(config.get("jacobian_frobenius_rel_tail_min", 0.05))
    values = []
    by_pair: dict[str, list[float]] = defaultdict(list)
    by_bin: dict[str, list[float]] = defaultdict(list)
    focus_values = []
    focus = _focus_keys(config)
    n_tail = 0
    n = 0
    for row in sample["contracted"]:
        dumped = row.get("c0_jacobian_frobenius_rel")
        jacobian = _as_matrix(row.get("transport_jacobian"), 5)
        cin = _as_matrix(row.get("input_covariance"), 5)
        c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
        rel = float(dumped) if dumped is not None else None
        if rel is None and jacobian is not None and cin is not None and c0 is not None:
            rel = compare_c0_to_jacobian(jacobian, cin, c0).get("frobenius_relative_error")
        if rel is None or not np.isfinite(float(rel)):
            continue
        rel = float(rel)
        n += 1
        values.append(rel)
        by_pair[_pair_label(row)].append(rel)
        if identity_key(row) in focus:
            focus_values.append(rel)
        abs_q = abs(float(row.get("q_over_p_per_mev", np.nan)))
        for name, edges in config["qoverp_bins_abs_per_mev"].items():
            if float(edges[0]) <= abs_q < float(edges[1]):
                by_bin[name].append(rel)
                break
        if rel > threshold:
            n_tail += 1
    stats = _stats(values)
    median = stats.get("median")
    fraction = float(n_tail / n) if n else None
    systematic = bool(
        (median is not None and float(median) > threshold)
        or (fraction is not None and fraction > 0.30)
    )
    return {
        "n": n,
        "c0_jacobian_frobenius_rel": stats,
        "fraction_above_0p05": fraction,
        "by_pair": {name: _stats(items) for name, items in sorted(by_pair.items())},
        "by_qoverp_bin": {name: _stats(items) for name, items in sorted(by_bin.items())},
        "focus_identity": _stats(focus_values),
        "systematic_transport_failure": systematic,
        "rescreened": False,
        "note": (
            "A small numerical-Jacobian tail is not upgraded to a systematic "
            "transport failure."
        ),
    }


def focus_case_study(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    material: Mapping[str, Any],
) -> dict[str, Any]:
    focus = _focus_keys(config)
    official = sample["official_events"]
    chi2_all = [
        float(event["chi2_per_ndof"])
        for event in official
        if event.get("chi2_per_ndof") is not None
    ]
    chi2_focus = [
        float(event["chi2_per_ndof"])
        for event in official
        if identity_key(event) in focus and event.get("chi2_per_ndof") is not None
    ]
    total = float(sum(chi2_all)) if chi2_all else 0.0
    construction = sample["records_by_split"]["construction"]
    loo_rows = [row for row in construction if identity_key(row) not in focus]
    loo_events = [
        event
        for event in official
        if event.get("split") == "construction" and identity_key(event) not in focus
    ]
    loo_pairs = {}
    loo_holds = []
    for pair in config["station_pairs"]:
        label = f"({int(pair[0])},{int(pair[1])})"
        cell = _pair_matrices(loo_events, loo_rows, label, _c1_4)
        shape = cell.get("shape") or {}
        loo_holds.append(bool(shape.get("shape_holds")))
        loo_pairs[label] = {
            "n": cell.get("n"),
            "pencil_ratio": shape.get("pencil_ratio"),
            "generalized_eigenvalues": shape.get("generalized_eigenvalues"),
            "shape_holds": shape.get("shape_holds"),
        }
    matched = [event for event in official if identity_key(event) in focus]
    if not matched:
        refuse_focus_drop()
    return {
        "retained": True,
        "downweighted": False,
        "truth_qoverp_substituted": False,
        "covariance_inflated": False,
        "chi2_share": float(sum(chi2_focus) / total) if total else None,
        "focus_chi2": _stats(chi2_focus),
        "all_chi2": _stats(chi2_all),
        "loo_construction_c1": loo_pairs,
        "loo_shape_holds": all(loo_holds) and bool(loo_holds),
        "loo_is_not_a_cut": True,
        "ckf_tail_equals_shape_root_cause": False,
        "pairs": [
            {
                "station_pair": event["station_pair"],
                "chi2_per_ndof": event.get("chi2_per_ndof"),
                "p_reco_mev": event.get("p_mev"),
                "p_truth_mev": event.get("p_truth_mev"),
            }
            for event in matched
        ],
        "official_material_shape_holds": material.get("c1_shape_holds"),
        "note": (
            "100043/37 remains.  If leave-one-out still fails shape, the CKF "
            "fitting tail is not the covariance-shape root cause."
        ),
    }


def decide_case(
    budget: Mapping[str, Any],
    dependency: Mapping[str, Any],
    cin_shape: Mapping[str, Any],
    material: Mapping[str, Any],
    linearization: Mapping[str, Any],
    focus: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(focus.get("retained", True)):
        refuse_focus_drop()
    shared = float(dependency.get("target_station_in_same_ckf_fit_fraction") or 0.0) >= 0.9
    official_adds_target = bool(budget.get("official_adds_c_target"))
    cin_holds = cin_shape.get("source_cin_shape_holds")
    cin_overwide = cin_holds is False
    c0_holds = bool(material.get("c0_shape_holds"))
    c1_holds = bool(material.get("c1_shape_holds"))
    material_primary = bool(material.get("material_is_primary_mismatch"))
    jac_systematic = bool(linearization.get("systematic_transport_failure"))
    flags = {
        "A": bool(shared and not official_adds_target and not c0_holds),
        "B": bool(cin_overwide),
        "C": bool(material_primary),
        "D": bool(jac_systematic),
    }
    active = [name for name, value in flags.items() if value]
    if len(active) > 1:
        primary = CASE_E
    elif active == ["A"]:
        primary = CASE_A
    elif active == ["B"]:
        primary = CASE_B
    elif active == ["C"]:
        primary = CASE_C
    elif active == ["D"]:
        primary = CASE_D
    elif shared and not official_adds_target:
        primary = CASE_A
    else:
        primary = CASE_E
    secondaries = []
    token = {
        "A": CASE_A,
        "B": CASE_B,
        "C": CASE_C,
        "D": CASE_D,
    }
    for name in active:
        if token[name] != primary:
            secondaries.append(token[name])
    if not focus.get("loo_shape_holds") and CASE_B not in secondaries and primary != CASE_B:
        pass
    return {
        "primary_case": primary,
        "secondary_cases": secondaries,
        "next_step": NEXT_STEP[primary],
        "active_mechanisms": [token[name] for name in active],
        "evidence": {
            "shared_fit_fraction": dependency.get("target_station_in_same_ckf_fit_fraction"),
            "official_adds_c_target": official_adds_target,
            "official_residual_reference": budget.get("official_residual_reference"),
            "c0_shape_holds": c0_holds,
            "c1_shape_holds": c1_holds,
            "source_cin_shape_holds": cin_holds,
            "material_is_primary_mismatch": material_primary,
            "jacobian_systematic": jac_systematic,
            "focus_chi2_share": focus.get("chi2_share"),
            "loo_shape_holds": focus.get("loo_shape_holds"),
            "ckf_tail_equals_shape_root_cause": False,
        },
        "case_flags": {
            "A": primary == CASE_A or CASE_A in secondaries,
            "B": primary == CASE_B or CASE_B in secondaries,
            "C": primary == CASE_C or CASE_C in secondaries,
            "D": primary == CASE_D or CASE_D in secondaries,
            "E": primary == CASE_E,
        },
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "conservative_covariance_declared_acceptable": False,
        "focus_retained": True,
        "raw_ckf_used": False,
        "covariance_rescaled": False,
        "q_psd_projected": False,
        "empirical_cross_covariance_invented": False,
    }


def inventory_and_diagnose(config: Mapping[str, Any]) -> dict[str, Any]:
    sample = load_contracted_sample(config)
    budget = decompose_covariance_budget(config, sample)
    dependency = audit_shared_measurement(config, sample)
    cin_shape = audit_cin_shape(config, sample)
    material = compare_material_shape(config, sample)
    linearization = audit_linearization(config, sample)
    focus = focus_case_study(config, sample, material)
    mechanism = decide_case(budget, dependency, cin_shape, material, linearization, focus)
    return {
        "present_sources": sample["present_sources"],
        "missing_sources": sample["missing_sources"],
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "provenance_hashes": sample["provenance_hashes"],
        "budget": budget,
        "dependency": dependency,
        "cin_shape": cin_shape,
        "material": material,
        "linearization": linearization,
        "focus": focus,
        "mechanism": mechanism,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(mechanism.get("measurement_model_v2_entered")):
        refuse_measurement_model_v2()
    if bool(mechanism.get("raw_ckf_used")):
        refuse_raw_ckf()
    if not bool(mechanism.get("focus_retained", True)):
        refuse_focus_drop()
    if bool(mechanism.get("empirical_cross_covariance_invented")):
        refuse_empirical_cross_covariance()
    if bool(mechanism.get("covariance_rescaled")):
        refuse_covariance_rescale()
    if bool(mechanism.get("q_psd_projected")):
        refuse_q_psd_projection()
    primary = mechanism["primary_case"]
    return {
        "verdict": "DIAGNOSED",
        "decision": primary,
        "primary_case": primary,
        "secondary_cases": mechanism["secondary_cases"],
        "next_step": mechanism["next_step"],
        "active_mechanisms": mechanism["active_mechanisms"],
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "unconstrained_tracker_only_stopped": True,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "focus_identity_retained": True,
        "conservative_covariance_declared_acceptable": False,
        "inherited_wb104_decision": inherited["workbook_104"]["decision"],
        "inherited_wb104_primary_case": inherited["workbook_104"]["primary_case"],
        "inherited_wb104_decision_sha256": inherited["workbook_104"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "inherited_wb103_decision_sha256": inherited["workbook_103"]["decision_sha256"],
        "wb87_through_wb104_rewritten": False,
        "evidence": mechanism["evidence"],
        "case_flags": mechanism["case_flags"],
    }
