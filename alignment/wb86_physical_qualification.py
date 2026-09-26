"""WB86: execution-only physical common-track qualification.

Does not invent a new protocol, threshold, corpus, or statistic.
Calls frozen evaluate_wb85b_qualification.  Does not rewrite WB84/WB85b.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.calypso_physical_replica_production import (
    AUTHORIZED_SOURCES,
    FORBIDDEN_PATH_NEEDLES,
    OUTPUT_ROOT as WB84_ROOT,
    PAYLOAD_FAMILIES,
    PROVENANCE,
    SURVEY_MODES,
    sha256_file,
)
from alignment.wb85_physical_qualification_protocol import (
    COVERAGE_GROSS_CP_UPPER_MIN,
    COVERAGE_GROSS_EMPIRICAL_MIN,
    ENGINEERING_RZ_MRAD,
    ENGINEERING_TRANSLATION_MM,
    FD_REL_MAX,
    MAX_NONCONVERGENCE_FRACTION,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    WB84_CONDITION_COUNTS,
    WB84_QUALIFIED_EXPORTS,
    WEAK_STANDARDIZED_CATASTROPHIC,
    clopper_pearson,
    diagnostic_decomposition,
    identifiable_strata,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    weak_strata,
    write_json,
)
from alignment.wb85a_protocol_qa import WB84_BUNDLE, WB84_GEOMETRY, WB84_MANIFEST, WB84_PROVENANCE, WB84_QA, WB84_SNAPSHOT
from alignment.wb85b_fwer_protocol import (
    ALPHA_COV,
    ALPHA_LS,
    CRITICAL_COV,
    CRITICAL_LS,
    EXPECTED_CRITICAL_COV,
    EXPECTED_CRITICAL_LS,
    OUTPUT_ROOT as WB85B_OUTPUT,
    evaluate_wb85b_qualification,
    require_frozen_criticals,
)
from alignment.wb85b_r1_claim_closure import RUNTIME_SOURCE_SNAPSHOT


WORKBOOK = "86"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb86_physical_qualification_v1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FREEZE_COMMIT = "6e065aeed01a6bdf353baea0565b59dd86966d90"
SOURCE_SNAPSHOT = RUNTIME_SOURCE_SNAPSHOT
ORIGIN_4STATION_AT_AUTHORIZATION = "3bd3073e3ffe66a74aed5dc55a6b551853d7dc98"
REMOTE_VERIFICATION = Path(
    "outputs/mc24_four_station_wb85b_r1_remote_verification_v1/wb85b_r1_remote_commit_verification.json"
)

EXPECTED_HASHES = {
    "wb84_corpus_manifest_sha256": "05cf0c66f97b94ee1948fc5ad8cf00fad1b171832fa5586680eab3c712a8fca0",
    "wb84_corpus_qa_sha256": "1a9a9bc69ef2e02bcecf2a70a8bc184bb10c2e8a8b1c1b03275cb99743c95876",
    "wb84_geometry_manifest_sha256": "df8361fb8760944f98c6b8b231c5c5b563d52680f09e1920583332cf5064ec12",
    "wb84_snapshot_sha256": "8706e244c850c5d1d97a7bd03779d1d94e5eb5a4c01681689da9d2e6aff17158",
    "wb84_calypso_bundle_sha256": "0ee0dcf948a0f3392a2a19d27fbc5d667632493bbe64c477aa983a5254239a52",
    "wb84_provenance_sha256": "e25864f486710531a011697fa0da6660cf2a0af76e8b50c41132ce4788f60e17",
    "wb85b_protocol_sha256": "1bac20ffdd26b7fbc4cd912ce5d75db266683b279e860837eab55f0da519af21",
    "wb85b_acceptance_rule_sha256": "a3adf56709a474ac963da3d45ff627b562465127e3146d642f99916905e83427",
    "wb85b_fwer_protocol_py_sha256": "757955097f41a5598e01f0e86f2128224b68bf0c3a199161c14936f42edbe756",
    "common_track_solver_sha256": "b8d2bc31fd736d84fb7a0c18a229d34c8acd6b7a5be774a4ef3bd0b98a793bf4",
    "official_runner_sha256": "6cc919202be3cabf4e61df499231593f4f830bd83116d3ab73db2e5441f0babe",
    "physical_execution_sha256": "08770a252f1a0ea8adef9ee120289cb91fec2597d7d377717ffd591436bf8072",
}

HASH_PATHS = {
    "wb84_corpus_manifest_sha256": WB84_MANIFEST,
    "wb84_corpus_qa_sha256": WB84_QA,
    "wb84_geometry_manifest_sha256": WB84_GEOMETRY,
    "wb84_snapshot_sha256": WB84_SNAPSHOT,
    "wb84_calypso_bundle_sha256": WB84_BUNDLE,
    "wb84_provenance_sha256": WB84_PROVENANCE,
    "wb85b_protocol_sha256": WB85B_OUTPUT / "wb85b_fwer_protocol.json",
    "wb85b_acceptance_rule_sha256": WB85B_OUTPUT / "wb85b_acceptance_rule.json",
    "wb85b_fwer_protocol_py_sha256": PROJECT_ROOT / "alignment" / "wb85b_fwer_protocol.py",
    "common_track_solver_sha256": PROJECT_ROOT / "alignment" / "common_track_solver.py",
    "official_runner_sha256": PROJECT_ROOT / "alignment" / "wb85b_official_runner.py",
    "physical_execution_sha256": PROJECT_ROOT / "alignment" / "physical_common_track_execution.py",
}


class WB86Error(ValueError):
    """Fail-closed WB86 execution contract."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def refuse_wb84_wb85b_rewrite(path: object) -> None:
    text = str(path)
    if "wb86" in text:
        refuse_forbidden_path(path)
        return
    refuse_wb83_wb84_write(path)
    blocked = (
        "wb85_physical_alignment_protocol_v1",
        "wb85a_protocol_qa_v1",
        "wb85a_r1_protocol_qa_v1",
        "wb85a_r1_runtime_smoke_v1",
        "wb85b_fwer_protocol_v1",
        "wb85b_official_runner_smoke_v1",
        "wb85b_r1_claim_closure_v1",
    )
    if any(marker in text for marker in blocked):
        raise WB86Error(f"WB86 must not rewrite historical artifacts: {path}")
    refuse_forbidden_path(path)


def locked_execution_flags() -> dict[str, Any]:
    return {
        "workbook": WORKBOOK,
        "qualification_authorized": True,
        "executable": True,
        "looked_at_physical_alignment_outcomes_before_authorization": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "association_default_system": "frozen_W64_raw_energy_plus_exact_solver",
        "wb86_automatically_authorized": False,
        "research_screening_thresholds_not_collaboration_approved": True,
        "no_new_model_protocol_threshold_or_dataset": True,
        "uses_evaluate_wb85b_qualification": True,
        "does_not_use_legacy_alpha_0.05_accepted_flag": True,
    }


def split_condition_id(condition_id: str) -> tuple[str, str]:
    for survey in SURVEY_MODES:
        suffix = f"_{survey}"
        if condition_id.endswith(suffix):
            return condition_id[: -len(suffix)], survey
    raise WB86Error(f"unrecognized condition_id: {condition_id}")


def chart_kind_for_family(family: str) -> str:
    if family in {"identity", "identifiable_translation"}:
        return "translation"
    if family == "identifiable_rotation":
        return "rotation"
    if family == "weak_jg_diagnostic":
        return "weak_jg"
    raise WB86Error(f"unknown family: {family}")


def authorized_xaod(source_id: str) -> str:
    for row in AUTHORIZED_SOURCES:
        if row["source_id"] == source_id:
            refuse_forbidden_path(row["path"])
            return str(row["path"])
    raise WB86Error(f"source is not on the WB84 allowlist: {source_id}")


def snapshot_verify() -> dict[str, Any]:
    require_frozen_criticals()
    observed = {name: sha256_file(path) for name, path in HASH_PATHS.items()}
    matches = {name: observed[name] == EXPECTED_HASHES[name] for name in EXPECTED_HASHES}
    snapshot = json.loads(WB84_SNAPSHOT.read_text(encoding="utf-8"))
    qa = json.loads(WB84_QA.read_text(encoding="utf-8"))
    identity = {
        "provenance": PROVENANCE,
        "geometry_name": snapshot.get("tracker_align_configuration", {}).get("geometry"),
        "global_tag": snapshot.get("tracker_align_configuration", {}).get("global_tag"),
        "athena_version": snapshot.get("athena_version"),
        "acts_tool": snapshot.get("acts", {}).get("tool"),
        "q_over_p_mode": snapshot.get("acts", {}).get("q_over_p_mode"),
        "field_map_exists": bool(snapshot.get("field_map", {}).get("exists")),
        "material_map_hashed_files_left_empty": not bool(snapshot.get("material_map", {}).get("hashed_files")),
        "n_qualified_exports": qa.get("n_qualified_exports"),
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "source_snapshot_sha256": SOURCE_SNAPSHOT,
        "alpha_LS": ALPHA_LS,
        "alpha_cov": ALPHA_COV,
        "critical_LS": CRITICAL_LS,
        "critical_cov": CRITICAL_COV,
    }
    expected_identity = {
        "athena_version": "24.0.41",
        "acts_tool": "FaserActsExtrapolationTool",
        "q_over_p_mode": 0,
        "geometry_name": "FASERNU-04",
        "global_tag": "OFLCOND-FASER-06",
        "n_qualified_exports": WB84_QUALIFIED_EXPORTS,
        "critical_LS": EXPECTED_CRITICAL_LS,
        "critical_cov": EXPECTED_CRITICAL_COV,
    }
    identity_ok = (
        identity["athena_version"] == expected_identity["athena_version"]
        and identity["acts_tool"] == expected_identity["acts_tool"]
        and identity["q_over_p_mode"] == expected_identity["q_over_p_mode"]
        and identity["geometry_name"] == expected_identity["geometry_name"]
        and identity["global_tag"] == expected_identity["global_tag"]
        and identity["n_qualified_exports"] == expected_identity["n_qualified_exports"]
        and abs(float(identity["critical_LS"]) - expected_identity["critical_LS"]) < 1.0e-12
        and abs(float(identity["critical_cov"]) - expected_identity["critical_cov"]) < 1.0e-12
    )
    passed = all(matches.values()) and identity_ok
    return {
        **locked_execution_flags(),
        "schema": "wb86_snapshot_verify_v1",
        "utc": _utc(),
        "hashes_observed": observed,
        "hashes_expected": dict(EXPECTED_HASHES),
        "hash_matches": matches,
        "identity": identity,
        "snapshot_verify_pass": bool(passed),
        "execution_qualification": "PASS" if passed else "FAIL",
        "may_not_substitute_latest_for_frozen": True,
    }


def load_replica_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    shards = WB84_ROOT / "shards"
    if not shards.is_dir():
        raise WB86Error("WB84 shard directory is absent")
    for shard in sorted(shards.iterdir()):
        replicas = shard / "replicas.jsonl"
        if not replicas.is_file():
            continue
        refuse_forbidden_path(replicas)
        for line in replicas.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record["event_uid"])
            if uid in index:
                raise WB86Error(f"duplicate WB84 event_uid: {uid}")
            index[uid] = {"record": record, "replica_path": str(replicas)}
    return index


def build_event_catalog() -> dict[str, Any]:
    manifest = json.loads(WB84_MANIFEST.read_text(encoding="utf-8"))
    uids = [str(uid) for uid in manifest["event_uids"]]
    if len(uids) != WB84_QUALIFIED_EXPORTS:
        raise WB86Error(f"WB84 manifest has {len(uids)} UIDs, expected {WB84_QUALIFIED_EXPORTS}")
    index = load_replica_index()
    events = []
    forbidden = []
    counts: dict[str, int] = {}
    for position, uid in enumerate(uids):
        if uid not in index:
            raise WB86Error(f"manifest UID missing from shards: {uid}")
        record = index[uid]["record"]
        text = json.dumps(record, sort_keys=True).lower() + index[uid]["replica_path"].lower()
        if any(needle.lower() in text for needle in FORBIDDEN_PATH_NEEDLES):
            forbidden.append(uid)
            continue
        condition = str(record["condition_id"])
        family, survey = split_condition_id(condition)
        source_id = str(record["source_id"])
        events.append(
            {
                "index": position,
                "event_uid": uid,
                "condition_id": condition,
                "family": family,
                "survey_mode": survey,
                "chart_kind": chart_kind_for_family(family),
                "source_id": source_id,
                "input_xaod": authorized_xaod(source_id),
                "xaod_entry_index": int(record["xaod_entry_index"]),
                "replica_path": index[uid]["replica_path"],
                "enters_primary_global_test": family != "weak_jg_diagnostic",
            }
        )
        counts[condition] = counts.get(condition, 0) + 1
    if forbidden:
        raise WB86Error(f"catalog contains forbidden assets: {forbidden[:5]}")
    if counts != dict(WB84_CONDITION_COUNTS):
        raise WB86Error(f"catalog counts drifted from WB84: {counts}")
    return {
        **locked_execution_flags(),
        "schema": "wb86_event_catalog_v1",
        "provenance": PROVENANCE,
        "n_events": len(events),
        "events": events,
        "counts": counts,
        "no_replicas_added_or_dropped": True,
    }


def corpus_verify() -> dict[str, Any]:
    catalog = build_event_catalog()
    qa = json.loads(WB84_QA.read_text(encoding="utf-8"))
    passed = (
        catalog["n_events"] == WB84_QUALIFIED_EXPORTS
        and qa["n_qualified_exports"] == WB84_QUALIFIED_EXPORTS
        and catalog["counts"] == dict(WB84_CONDITION_COUNTS)
    )
    return {
        **locked_execution_flags(),
        "schema": "wb86_corpus_verify_v1",
        "utc": _utc(),
        "n_events": catalog["n_events"],
        "counts": catalog["counts"],
        "catalog_without_events": {key: catalog[key] for key in catalog if key != "events"},
        "corpus_verify_pass": bool(passed),
        "execution_qualification": "PASS" if passed else "FAIL",
        "overlay_forbidden": True,
        "final_blind_forbidden": True,
        "sealed_forbidden": True,
    }


def write_prereq_artifacts(output_root: Path | None = None) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_verify()
    write_json(root / "wb86_snapshot_verify.json", snapshot)
    if not snapshot["snapshot_verify_pass"]:
        write_json(root / "wb86_execution_qualification.json", {**locked_execution_flags(), "execution_qualification": "FAIL", "reason": "snapshot mismatch"})
        return root
    corpus = corpus_verify()
    write_json(root / "wb86_corpus_verify.json", corpus)
    catalog = build_event_catalog()
    write_json(root / "wb86_event_catalog.json", catalog)
    if not corpus["corpus_verify_pass"]:
        write_json(root / "wb86_execution_qualification.json", {**locked_execution_flags(), "execution_qualification": "FAIL", "reason": "corpus mismatch"})
    return root


def load_catalog_event(index: int, output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    catalog = json.loads((root / "wb86_event_catalog.json").read_text(encoding="utf-8"))
    row = catalog["events"][int(index)]
    if int(row["index"]) != int(index):
        raise WB86Error("catalog index is not dense")
    return row


def load_replica_record(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(row["replica_path"])
    refuse_forbidden_path(path)
    # Read-only ingest of the frozen WB84 corpus.  refuse_wb83_wb84_write()
    # blocks any path containing calypso_physical_replicas_v1 and must not
    # be used on this read.
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if str(record["event_uid"]) == str(row["event_uid"]):
            record["input_xaod"] = row["input_xaod"]
            return record
    raise WB86Error(f"replica record missing: {row['event_uid']}")


def technical_event_dir(output_root: Path, index: int) -> Path:
    return Path(output_root) / "events" / f"{int(index):04d}"


def sealed_event_path(output_root: Path, index: int) -> Path:
    return Path(output_root) / "sealed" / f"{int(index):04d}.json"


def list_technical_reports(output_root: Path) -> list[Path]:
    root = Path(output_root) / "events"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/technical.json"))


def decision_function():
    """The only allowed scientific decision function."""
    return evaluate_wb85b_qualification


def require_remote_verification() -> dict[str, Any]:
    if not REMOTE_VERIFICATION.is_file():
        raise WB86Error("WB85b-r1 remote commit verification artifact is absent")
    payload = json.loads(REMOTE_VERIFICATION.read_text(encoding="utf-8"))
    if not bool(payload.get("remote_commit_verification")):
        raise WB86Error("remote_commit_verification is not true")
    if int(payload.get("merge_base_is_ancestor_exit_code", 1)) != 0:
        raise WB86Error("source-freeze commit is not an ancestor of origin/4station")
    if payload.get("post_run_source_freeze_commit") != SOURCE_FREEZE_COMMIT:
        raise WB86Error("remote verification freeze commit drifted")
    return payload


def write_prereq_or_fail(output_root: Path | None = None) -> dict[str, Any]:
    require_remote_verification()
    root = write_prereq_artifacts(output_root)
    snapshot = json.loads((root / "wb86_snapshot_verify.json").read_text(encoding="utf-8"))
    if not snapshot.get("snapshot_verify_pass"):
        raise WB86Error("SNAPSHOT_VERIFY failed; scientific alignment must not start")
    corpus = json.loads((root / "wb86_corpus_verify.json").read_text(encoding="utf-8"))
    if not corpus.get("corpus_verify_pass"):
        raise WB86Error("CORPUS_VERIFY failed; scientific alignment must not start")
    return {"root": str(root), "snapshot": snapshot, "corpus": corpus}


def result_integrity(output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    catalog = json.loads((root / "wb86_event_catalog.json").read_text(encoding="utf-8"))
    missing = []
    hash_mismatch = []
    technical_fail = []
    success = 0
    for row in catalog["events"]:
        index = int(row["index"])
        technical_path = technical_event_dir(root, index) / "technical.json"
        sealed_path = sealed_event_path(root, index)
        if not technical_path.is_file() or not sealed_path.is_file():
            missing.append(index)
            continue
        technical = json.loads(technical_path.read_text(encoding="utf-8"))
        observed = sha256_file(sealed_path)
        if technical.get("sealed_sha256") != observed:
            hash_mismatch.append(index)
        if technical.get("technical_status") == "SUCCESS":
            success += 1
        else:
            technical_fail.append(index)
    passed = not missing and not hash_mismatch
    report = {
        **locked_execution_flags(),
        "schema": "wb86_result_integrity_v1",
        "utc": _utc(),
        "n_expected": catalog["n_events"],
        "n_technical_success": success,
        "n_technical_fail": len(technical_fail),
        "n_missing": len(missing),
        "n_hash_mismatch": len(hash_mismatch),
        "missing_indices_head": missing[:20],
        "hash_mismatch_indices_head": hash_mismatch[:20],
        "result_integrity_pass": bool(passed),
        "scientific_aggregates_not_computed_here": True,
    }
    write_json(root / "wb86_result_integrity.json", report)
    if not passed:
        write_json(
            root / "wb86_execution_qualification.json",
            {
                **locked_execution_flags(),
                "execution_qualification": "FAIL",
                "reason": "result integrity failed",
            },
        )
    return report


def _finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}


def _stratum_accumulator() -> dict[str, Any]:
    return {
        "n_attempted": 0,
        "n_converged": 0,
        "n_nan": 0,
        "n_inf": 0,
        "n_non_spd": 0,
        "n_dropped_parameters": 0,
        "fd_rel_max_observed": 0.0,
        "sign_frame_failure": False,
        "z": [],
        "covered": 0,
        "bias": [],
        "weak_dx": [],
        "weak_ry": [],
        "dropped": False,
    }


def global_summary(output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    integrity = json.loads((root / "wb86_result_integrity.json").read_text(encoding="utf-8"))
    if not integrity.get("result_integrity_pass"):
        raise WB86Error("GLOBAL_SUMMARY refuses to run before result integrity")
    catalog = json.loads((root / "wb86_event_catalog.json").read_text(encoding="utf-8"))
    identifiable = {row["stratum_id"]: _stratum_accumulator() for row in identifiable_strata()}
    weak = {row["stratum_id"]: _stratum_accumulator() for row in weak_strata()}
    for row in catalog["events"]:
        sealed = json.loads(sealed_event_path(root, int(row["index"])).read_text(encoding="utf-8"))
        bucket = identifiable if row["enters_primary_global_test"] else weak
        target = bucket[str(sealed["stratum_id"])]
        target["n_attempted"] += 1
        target["fd_rel_max_observed"] = max(
            float(target["fd_rel_max_observed"]), float(sealed.get("fd_rel_max_observed") or 0.0)
        )
        target["sign_frame_failure"] = bool(target["sign_frame_failure"] or sealed.get("sign_frame_failure"))
        if int(sealed.get("n_dropped") or 0) > 0:
            target["n_dropped_parameters"] += 1
        if sealed.get("solver_status") == "non_spd":
            target["n_non_spd"] += 1
        analyzable = bool(sealed.get("analyzable"))
        projected = sealed.get("projected") or {}
        z_value = projected.get("z")
        if z_value is None:
            continue
        if not _finite_number(z_value):
            if z_value == z_value:
                target["n_inf"] += 1
            else:
                target["n_nan"] += 1
            continue
        if not analyzable:
            continue
        target["n_converged"] += 1
        target["z"].append(float(z_value))
        target["bias"].append(float(projected["a_hat"]) - float(projected["a_true"]))
        if bool(projected.get("covered_95")):
            target["covered"] += 1
        per_z = sealed.get("per_parameter_z") or {}
        if "s3_dx_mm" in per_z:
            target["weak_dx"].append(float(per_z["s3_dx_mm"]))
        if "s3_ry_mrad" in per_z:
            target["weak_ry"].append(float(per_z["s3_ry_mrad"]))
    z_bundle = {}
    coverage = {}
    identifiable_rows = {}
    for name, acc in identifiable.items():
        z_bundle[name] = list(acc["z"])
        n = int(acc["n_converged"])
        k = int(acc["covered"])
        coverage[name] = (k, n) if n > 0 else (0, 0)
        lo, hi = (0.0, 1.0) if n <= 0 else clopper_pearson(k, n)
        identifiable_rows[name] = {
            "n_attempted": acc["n_attempted"],
            "n_converged": n,
            "n_nan": acc["n_nan"],
            "n_inf": acc["n_inf"],
            "n_non_spd": acc["n_non_spd"],
            "fd_rel_max_observed": acc["fd_rel_max_observed"],
            "sign_frame_failure": acc["sign_frame_failure"],
            "engineering_bias": float(sum(acc["bias"]) / n) if n else 0.0,
            "empirical_coverage": (k / float(n)) if n else 0.0,
            "cp_lo": lo,
            "cp_hi": hi,
            "n_dropped_parameters": acc["n_dropped_parameters"],
            "dropped": bool(acc["n_attempted"] <= 0),
            "pull_mean": float(sum(acc["z"]) / n) if n else None,
            "pull_width": (
                float((sum((value - (sum(acc["z"]) / n)) ** 2 for value in acc["z"]) / (n - 1)) ** 0.5)
                if n > 1
                else None
            ),
        }
    weak_rows = {}
    for name, acc in weak.items():
        n = int(acc["n_converged"])
        weak_rows[name] = {
            "n_attempted": acc["n_attempted"],
            "n_converged": n,
            "standardized_bias_dx": float(sum(acc["weak_dx"]) / len(acc["weak_dx"])) if acc["weak_dx"] else 0.0,
            "standardized_bias_ry": float(sum(acc["weak_ry"]) / len(acc["weak_ry"])) if acc["weak_ry"] else 0.0,
        }
    bundle = {
        "z": z_bundle,
        "coverage": coverage,
        "identifiable_strata": identifiable_rows,
        "weak_strata": weak_rows,
    }
    diagnostics = None
    if all(len(values) >= 2 for values in z_bundle.values()):
        diagnostics = diagnostic_decomposition(z_bundle)
        diagnostics["may_reselect_model"] = False
        diagnostics["holm_is_diagnostic_only"] = True
    summary = {
        **locked_execution_flags(),
        "schema": "wb86_global_summary_v1",
        "utc": _utc(),
        "bundle": bundle,
        "diagnostics_only": diagnostics,
        "decision_function": "alignment.wb85b_fwer_protocol.evaluate_wb85b_qualification",
        "wb85_accepted_flags_ignored": True,
    }
    write_json(root / "wb86_global_summary.json", summary)
    return summary


def execution_layer(bundle: Mapping[str, Any], *, snapshot_pass: bool, corpus_pass: bool, integrity_pass: bool) -> dict[str, Any]:
    failures = []
    unknown = []
    if not snapshot_pass:
        failures.append("snapshot mismatch")
    if not corpus_pass:
        failures.append("corpus mismatch")
    if not integrity_pass:
        failures.append("result integrity")
    rows = bundle["identifiable_strata"]
    for name in [row["stratum_id"] for row in identifiable_strata()]:
        row = rows[name]
        n_attempted = int(row["n_attempted"])
        n_converged = int(row["n_converged"])
        if n_attempted <= 0 or bool(row.get("dropped")):
            failures.append(f"missing stratum {name}")
            continue
        if n_converged < N_MIN_PER_IDENTIFIABLE_STRATUM:
            unknown.append(f"{name} n={n_converged} < {N_MIN_PER_IDENTIFIABLE_STRATUM}")
        if (n_attempted - n_converged) / float(n_attempted) > MAX_NONCONVERGENCE_FRACTION:
            failures.append(f"nonconvergence {name}")
        if int(row.get("n_nan", 0)) > 0 or int(row.get("n_inf", 0)) > 0:
            failures.append(f"NaN/Inf {name}")
        if int(row.get("n_non_spd", 0)) > 0:
            failures.append(f"non-SPD {name}")
        if float(row.get("fd_rel_max_observed", 0.0)) > FD_REL_MAX:
            failures.append(f"FD {name}")
        if int(row.get("n_dropped_parameters", 0)) > 0:
            failures.append(f"dropped identifiable mode {name}")
    if unknown and not failures:
        status = "UNKNOWN"
    elif failures:
        status = "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "analyzable": status == "PASS",
        "failures": failures,
        "unknown_reasons": unknown,
    }


def physics_screening(bundle: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for name, row in bundle["identifiable_strata"].items():
        bias = abs(float(row.get("engineering_bias") or 0.0))
        if name.startswith("identifiable_rotation"):
            if bias > ENGINEERING_RZ_MRAD:
                failures.append(f"rotation rz screening {name}")
        else:
            if bias > ENGINEERING_TRANSLATION_MM:
                failures.append(f"translation/identity screening {name}")
        if bool(row.get("sign_frame_failure")):
            failures.append(f"sign/frame {name}")
        empirical = float(row.get("empirical_coverage") or 0.0)
        cp_hi = float(row.get("cp_hi") or 0.0)
        if empirical < COVERAGE_GROSS_EMPIRICAL_MIN or cp_hi < COVERAGE_GROSS_CP_UPPER_MIN:
            failures.append(f"gross coverage {name}")
    for name, row in bundle.get("weak_strata", {}).items():
        if abs(float(row.get("standardized_bias_dx") or 0.0)) > WEAK_STANDARDIZED_CATASTROPHIC:
            failures.append(f"weak dx {name}")
        if abs(float(row.get("standardized_bias_ry") or 0.0)) > WEAK_STANDARDIZED_CATASTROPHIC:
            failures.append(f"weak ry {name}")
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "research_screening_not_collaboration_approved": True,
        "thresholds": {
            "translation_identity_mm": ENGINEERING_TRANSLATION_MM,
            "rotation_rz_mrad": ENGINEERING_RZ_MRAD,
            "weak_standardized_abs_z": WEAK_STANDARDIZED_CATASTROPHIC,
        },
    }


def frozen_qualification_gate(output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    snapshot = json.loads((root / "wb86_snapshot_verify.json").read_text(encoding="utf-8"))
    corpus = json.loads((root / "wb86_corpus_verify.json").read_text(encoding="utf-8"))
    integrity = json.loads((root / "wb86_result_integrity.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "wb86_global_summary.json").read_text(encoding="utf-8"))
    bundle = summary["bundle"]
    execution = execution_layer(
        bundle,
        snapshot_pass=bool(snapshot.get("snapshot_verify_pass")),
        corpus_pass=bool(corpus.get("corpus_verify_pass")),
        integrity_pass=bool(integrity.get("result_integrity_pass")),
    )
    statistical = None
    if execution["analyzable"] or execution["status"] == "UNKNOWN":
        statistical = evaluate_wb85b_qualification(bundle)
    screening = physics_screening(bundle)
    if execution["status"] == "UNKNOWN" or (statistical or {}).get("qualification_status") == "UNKNOWN":
        overall = "UNKNOWN"
        oracle = False
    else:
        statistical_pass = bool(statistical) and statistical.get("qualification_status") == "PASS"
        overall = (
            "PASS"
            if execution["status"] == "PASS" and statistical_pass and screening["status"] == "PASS"
            else "FAIL"
        )
        oracle = overall == "PASS"
    flags = locked_execution_flags()
    flags["alignment_oracle_qualified_for_physical_FASER"] = bool(oracle)
    report = {
        **flags,
        "schema": "wb86_final_qualification_v1",
        "utc": _utc(),
        "execution_qualification": execution,
        "statistical_qualification": statistical,
        "physics_screening": screening,
        "overall_qualification": overall,
        "alignment_oracle_qualified_for_physical_FASER": bool(oracle),
        "ml_alignment_eval_authorized": False,
        "decision_function": "alignment.wb85b_fwer_protocol.evaluate_wb85b_qualification",
        "no_retune_on_fail": True,
        "no_new_corpus_or_protocol": True,
    }
    write_json(root / "wb86_final_qualification.json", report)
    write_json(root / "wb86_execution_qualification.json", {**flags, "execution_qualification": execution["status"]})
    return report


def write_closure(output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    final = json.loads((root / "wb86_final_qualification.json").read_text(encoding="utf-8"))
    oracle = bool(final["alignment_oracle_qualified_for_physical_FASER"])
    closure = {
        **locked_execution_flags(),
        "schema": "physical_alignment_oracle_closure_v1",
        "utc": _utc(),
        "alignment_oracle_qualified_for_physical_FASER": oracle,
        "overall_qualification": final["overall_qualification"],
        "execution_qualification": final["execution_qualification"]["status"],
        "statistical_qualification": (final.get("statistical_qualification") or {}).get("qualification_status"),
        "physics_screening": final["physics_screening"]["status"],
        "ml_alignment_eval_authorized": False,
        "next_stage_requires_separate_authorization": True,
        "on_fail_read_only_attribution_only": final["overall_qualification"] != "PASS",
        "final_qualification_sha256": sha256_file(root / "wb86_final_qualification.json"),
    }
    closure["alignment_oracle_qualified_for_physical_FASER"] = oracle
    write_json(root / "physical_alignment_oracle_closure.json", closure)
    return closure
