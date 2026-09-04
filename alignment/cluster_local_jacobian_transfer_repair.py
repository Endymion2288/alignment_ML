"""Cluster-local Jacobian transfer exact-join repair V1.

Workbook 68 skipped the diagnostic because r14974 routes were joined
against the r14973 feasibility dump.  This campaign restores the
already-produced run-matched dump and audits the exact
``run+event+cluster_identifier`` join.  Fuzzy / nearest-neighbour
matching is forbidden.  A restored join does not reopen entry 58 and
is not a stable-core gate.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.identifiable_subspace import refuse_fuzzy_or_nearest_neighbour_join
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.true_cluster_local_residual import (
    RESIDUAL_KIND,
    assert_no_alignment_payload,
    common_operating_state,
    load_cluster_local_hits,
)
from alignment.true_cluster_local_stability_transfer import load_run_measurements
from alignment.module_level_residual_poc import load_selected_routes, load_tracklet_hits


SCHEMA_VERSION = "faser-cluster-local-jacobian-transfer-repair-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "cluster_local_jacobian_transfer_repair_v1.yaml"
EXACT_JOIN_KEY = "run + event + TrackletHit_cluster_identifier = FaserSCT_Cluster.identify().get_compact()"
INHERITED_SKIP = "cluster_local_reference_or_transfer_unavailable"
INHERITED_ENTRY_58 = "cluster_local_identifiability_not_transferable"
DECISION_RESTORED = "cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign"
DECISION_UNAVAILABLE = "cluster_local_transfer_exact_join_still_unavailable"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "survey_is_alignment_input",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_run_newton",
    "do_not_solve_alignment_correction",
    "do_not_write_official_conditions",
    "do_not_use_tracklet_intercept_as_cluster_residual",
    "do_not_fuzzy_match_join",
    "do_not_nearest_neighbour_join",
    "do_not_reselect_population_to_force_nonzero_join",
    "do_not_merge_into_stable_core_gate",
    "do_not_reopen_entry_58_identifiability_decision",
    "not_a_stable_core_gate",
    "not_a_cluster_local_identifiability_campaign",
    "survey_is_external_cross_check_only",
)


def load_repair_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"transfer-repair config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected transfer-repair schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"transfer-repair config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"transfer-repair config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("campaign must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("campaign must not replace the frozen V2 checkpoint SHA256")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("campaign must keep the true cluster-local residual label")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if payload.get("inherited_v1_skip_reason") != INHERITED_SKIP:
        raise ValueError("campaign must inherit the workbook-68 skip reason")
    if payload.get("inherited_entry_58_decision") != INHERITED_ENTRY_58:
        raise ValueError("campaign must inherit the entry-58 identifiability freeze")
    if payload.get("exact_join_key") != EXACT_JOIN_KEY:
        raise ValueError("exact join key must remain run+event+cluster_identifier")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def dump_run_inventory(path: str | Path) -> dict[str, Any]:
    hits = load_cluster_local_hits(path)
    runs = Counter(int(hit.run_id) for hit in hits.values())
    events = {(int(hit.run_id), int(hit.event_id)) for hit in hits.values()}
    stations = Counter(int(hit.station_id) for hit in hits.values())
    return {
        "dump_path": str(Path(path).expanduser().resolve()),
        "n_clusters_with_surface": int(len(hits)),
        "n_unique_events": int(len(events)),
        "runs": {str(run): int(count) for run, count in sorted(runs.items())},
        "stations": {str(station): int(count) for station, count in sorted(stations.items())},
        "join_key": EXACT_JOIN_KEY,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
    }


def exact_join_audit(
    spec: Mapping[str, Any],
    *,
    cluster_dump: str | Path,
    representative_station: int = 0,
) -> dict[str, Any]:
    root = project_root()
    dump_path = resolve_under_root(root, str(cluster_dump))
    selected_path = resolve_under_root(root, str(spec["selected_routes"]))
    enhanced_path = resolve_under_root(root, str(spec["enhanced_tracklets"]))
    routes = load_selected_routes(selected_path)
    wanted_events = {(int(row["run_id"]), int(row["event_id"])) for row in routes}
    route_runs = Counter(int(row["run_id"]) for row in routes)
    hits = load_tracklet_hits(enhanced_path, wanted_events)
    dump_hits = load_cluster_local_hits(dump_path)
    dump_runs = Counter(int(hit.run_id) for hit in dump_hits.values())
    wanted_keys: list[tuple[int, int, int]] = []
    for route in routes:
        key = (int(route["run_id"]), int(route["event_id"]))
        for hit in hits.get(key, ()):
            wanted_keys.append((int(hit.run_id), int(hit.event_id), int(hit.cluster_identifier)))
    unique_wanted = set(wanted_keys)
    matched = [key for key in unique_wanted if key in dump_hits]
    missing = [key for key in unique_wanted if key not in dump_hits]
    dump_event_set = {(int(hit.run_id), int(hit.event_id)) for hit in dump_hits.values()}
    event_overlap = wanted_events & dump_event_set
    run_overlap = set(int(run) for run in route_runs) & set(int(run) for run in dump_runs)
    mismatch_reason = None
    if not run_overlap:
        mismatch_reason = "run_id_disjoint_dump_does_not_contain_route_run"
    elif not event_overlap:
        mismatch_reason = "run_matches_but_event_keys_disjoint"
    elif missing and not matched:
        mismatch_reason = "run_and_event_overlap_but_cluster_identifier_miss"
    bundle = load_run_measurements(
        spec,
        root=root,
        cluster_dump=dump_path,
        representative_station=int(representative_station),
    )
    return {
        "run": int(spec["run"]),
        "source_id": str(spec["source_id"]),
        "role": str(spec.get("role", "")),
        "selected_routes": str(selected_path),
        "enhanced_tracklets": str(enhanced_path),
        "cluster_dump": str(dump_path),
        "reconstruction": str(spec.get("reconstruction", "")),
        "n_selected_routes": int(len(routes)),
        "n_unique_route_events": int(len(wanted_events)),
        "route_runs": {str(run): int(count) for run, count in sorted(route_runs.items())},
        "dump": dump_run_inventory(dump_path),
        "n_wanted_cluster_keys": int(len(unique_wanted)),
        "n_exact_joined_cluster_keys": int(len(matched)),
        "n_missing_cluster_keys": int(len(missing)),
        "n_run_overlap": int(len(run_overlap)),
        "n_event_overlap": int(len(event_overlap)),
        "join_complete": bool(unique_wanted and not missing),
        "n_measurements": int(bundle["n_measurements"]),
        "join_key": EXACT_JOIN_KEY,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
        "population_reselected": False,
        "mismatch_reason": mismatch_reason,
        "exact_join_nonzero": bool(matched),
    }


def decide_repair(*, reference: Mapping[str, Any], transfer: Mapping[str, Any], wrong: Mapping[str, Any]) -> dict[str, Any]:
    restored = bool(
        reference["join_complete"]
        and reference["n_measurements"] > 0
        and transfer["join_complete"]
        and transfer["n_measurements"] > 0
        and transfer["exact_join_nonzero"]
    )
    control_still_zero = bool(
        not wrong["exact_join_nonzero"] and wrong.get("mismatch_reason") == "run_id_disjoint_dump_does_not_contain_route_run"
    )
    if restored:
        decision = DECISION_RESTORED
        reasons = [
            "exact_run_event_cluster_identifier_join_nonzero",
            "wrong_dump_control_reproduces_workbook_68_zero_join",
            "entry_58_identifiability_decision_not_reopened",
            "not_a_stable_core_gate",
        ]
    else:
        decision = DECISION_UNAVAILABLE
        reasons = [
            "exact_join_still_zero_or_incomplete",
            "do_not_fuzzy_match_or_reselect_population",
            "not_a_stable_core_gate",
        ]
    return {
        "decision": decision,
        "exact_join_restored": restored,
        "wrong_dump_control_still_zero": control_still_zero,
        "n_reference_measurements": int(reference["n_measurements"]),
        "n_transfer_measurements": int(transfer["n_measurements"]),
        "n_wrong_dump_measurements": int(wrong["n_measurements"]),
        "authorize_cluster_local_identifiability_campaign": False,
        "authorize_frozen_v2_unknown_association_closure": False,
        "real_data_alignment_correction_authorized": False,
        "geometry_write_allowed": False,
        "not_a_stable_core_gate": True,
        "entry_58_identifiability_not_reopened": True,
        "fuzzy_join_used": False,
        "reasons": reasons,
        "if_failed_do_not_chase_by_fuzzy_join": True,
    }


def build_all_reports(config: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root()
    created = datetime.now(timezone.utc).isoformat()
    reference_spec = config["runs"]["reference"]
    transfer_spec = config["runs"]["transfer"]
    reference = exact_join_audit(reference_spec, cluster_dump=reference_spec["cluster_dump"])
    transfer = exact_join_audit(transfer_spec, cluster_dump=transfer_spec["cluster_dump"])
    wrong = exact_join_audit(transfer_spec, cluster_dump=reference_spec["cluster_dump"])
    decision = decide_repair(reference=reference, transfer=transfer, wrong=wrong)
    provenance = {
        "reference_input_xaod": str(reference_spec["input_xaod"]),
        "transfer_input_xaod": str(transfer_spec["input_xaod"]),
        "reference_reconstruction": str(reference_spec.get("reconstruction", "")),
        "transfer_reconstruction": str(transfer_spec.get("reconstruction", "")),
        "join_key": EXACT_JOIN_KEY,
        "workbook_68_failure_mode": (
            "r14974 routes joined against r14973 feasibility dump; dump runs={14973}"
        ),
        "repair": "wire already-produced cluster_local_r14974.root; do not invent a new key",
    }
    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": common_operating_state(),
            "inherited_v1_skip_reason": INHERITED_SKIP,
            "inherited_entry_58_decision": INHERITED_ENTRY_58,
            "exact_join_key": EXACT_JOIN_KEY,
            "not_a_stable_core_gate": True,
            "not_a_cluster_local_identifiability_campaign": True,
            "assumptions": list(config.get("assumptions", [])),
            "provenance": provenance,
        },
        "reference_join": reference,
        "transfer_join": transfer,
        "wrong_dump_control": wrong,
        "next_stage_decision": decision,
    }
    for payload in reports.values():
        if isinstance(payload, Mapping):
            assert_no_alignment_payload(payload)
    return reports


def refuse_forbidden_operations() -> dict[str, Any]:
    try:
        refuse_fuzzy_or_nearest_neighbour_join()
        fuzzy = False
    except Exception:
        fuzzy = True
    return {
        "refused_fuzzy_or_nearest_neighbour_join": fuzzy,
        "geometry_write_allowed": False,
        "real_data_alignment_correction": False,
        "merged_into_stable_core_gate": False,
        "reopened_entry_58": False,
    }
