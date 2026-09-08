"""WB85b-r1: FWER claim wording correction and source-freeze closure.

Does not change WB85b thresholds, alpha, power alternatives, Monte-Carlo
results, solver, runner, or physical contract.  Does not run WB86.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.calypso_physical_replica_production import sha256_file
from alignment.wb85_physical_qualification_protocol import refuse_forbidden_path, write_json
from alignment.wb85b_fwer_protocol import (
    ALPHA_COV,
    ALPHA_LS,
    CRITICAL_COV,
    CRITICAL_LS,
    EXPECTED_CRITICAL_COV,
    EXPECTED_CRITICAL_LS,
    FAMILYWISE_ALPHA,
    HISTORICAL_WB85,
    HISTORICAL_WB85A,
    HISTORICAL_WB85A_R1,
    HISTORICAL_WB85A_R1_SMOKE,
    OUTPUT_ROOT as WB85B_OUTPUT,
    WB85BError,
    locked_flags,
    refuse_historical_rewrite,
    require_frozen_criticals,
)


WORKBOOK = "85b-r1"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb85b_r1_claim_closure_v1")
WB85B_SMOKE = Path("outputs/mc24_four_station_wb85b_official_runner_smoke_v1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PARENT = "fe2a97666d688e31380ff27f8550dd629631daea"
RUNTIME_SOURCE_SNAPSHOT = "3945d7065c11b007292bbcd3253028dc1f435f0eee8c449c6c363137ddaccbde"
WB85B_SMOKE_SHA256 = "286c13e4360a546d079cd3d45f073c5800133674413d33f112839bec3b7083d8"
WB85B_PROTOCOL_JSON_SHA256 = "1bac20ffdd26b7fbc4cd912ce5d75db266683b279e860837eab55f0da519af21"
WB85B_ACCEPTANCE_JSON_SHA256 = "a3adf56709a474ac963da3d45ff627b562465127e3146d642f99916905e83427"
WB85B_JOINT_JSON_SHA256 = "a39df2d0c1e918a505d745feacc38a907c08e966f06085cf3c1a5aa42243a4b2"
WB85B_POWER_JSON_SHA256 = "00166f65a067475b44261df269a4c7a3237729da77f2ca88363747bfc9f8bcf4"
WB85B_QA_JSON_SHA256 = "14fc6a3da4fbe76f88c803089619712b1e75aaef3aed3e9d1edec62d44914c22"

GENERATING_SOURCE_DIGESTS = {
    "alignment/physical_common_track_execution.py": "08770a252f1a0ea8adef9ee120289cb91fec2597d7d377717ffd591436bf8072",
    "alignment/wb85_physical_qualification_protocol.py": "186f0aba009840483f76e4c9fc3fc2cbe7695c21df5ebd5fc920d4d8612686c7",
    "alignment/wb85_projected_unit.py": "3287a5c71dffd6fc7a3c4c0080f74f3b0e2654e47a3271477e8469755d2d2513",
    "alignment/wb85a_r1_protocol_qa.py": "fc657dab2cc071d6f8e624bd52d9dc59d34a12721adf7f2fbb2f13637d373ac2",
    "alignment/wb85a_r1_runtime_smoke.py": "ac4204508d8417253a259d331e0c4c8f37478b2a4311958bccf54735a991d2a5",
    "alignment/wb85b_fwer_protocol.py": "757955097f41a5598e01f0e86f2128224b68bf0c3a199161c14936f42edbe756",
    "alignment/wb85b_official_runner.py": "6cc919202be3cabf4e61df499231593f4f830bd83116d3ab73db2e5441f0babe",
    "alignment/wb85b_protocol_qa.py": "b67d797ea6f15d95f201955843327c02613e0c8a2112b8b34e21079f7f650616",
    "configs/research_review/wp85b_fwer_protocol.yaml": "082e1f818aad1f49f047cbd851bbb3384f32030d9f60fcda7ec547eefadcd29f",
    "scripts/run_wb85b_official_runner_smoke.py": "f835fb7665cb10f05bf56e663245f6ae16b993b7126f4f958e42fa46175a3629",
    "scripts/run_wb85b_official_runner_smoke_condor.sh": "5101df8f414f3b8da93eb5da861e8316c755e6933895b9a43b782d41ff36f965",
    "scripts/run_wb85b_protocol_qa.py": "e9463aa286d9e2bbe10e37a319a6c90c865e7462bdaa88bdb7b40c2f8c1a3766",
    "scripts/submit_wb85b_official_runner_smoke_condor.py": "c676edb50393352317c33789205e776ba0bf53a17d28be6d879324bbc23aa61d",
    "tests/test_wb85b_fwer_protocol.py": "1de8e6217ca2a5749cecd31e2912e533425fb8a17e705a3b432d3060cb5044bf",
}

RECORDED_JOINT_NULL = {
    "T_LS_type_I": 0.02545,
    "T_LS_cp95": (0.023311575000056213, 0.027727838619144265),
    "T_cov_type_I": 0.02655,
    "T_cov_cp95": (0.02436599997358225, 0.028873048596555994),
    "union": 0.04720,
    "union_cp95": (0.04430196890469669, 0.05023054822734232),
}


class WB85BR1Error(WB85BError):
    """Fail-closed WB85b-r1 closure contract."""


def refuse_wb85b_rewrite(path: object) -> None:
    text = str(path)
    if "wb85b_r1" in text:
        refuse_historical_rewrite(path)
        refuse_forbidden_path(path)
        return
    blocked = (
        "wb85_physical_alignment_protocol_v1",
        "wb85a_protocol_qa_v1",
        "wb85a_r1_protocol_qa_v1",
        "wb85a_r1_runtime_smoke_v1",
        "wb85b_fwer_protocol_v1",
        "wb85b_official_runner_smoke_v1",
    )
    if any(marker in text for marker in blocked):
        raise WB85BR1Error(f"WB85b-r1 must not rewrite historical artifacts: {path}")
    refuse_historical_rewrite(path)
    refuse_forbidden_path(path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _blob_sha(commit: str, rel: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "show", f"{commit}:{rel}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return ""
    return hashlib.sha256(result.stdout).hexdigest()


def locked_r1_flags() -> dict[str, Any]:
    return {
        **locked_flags(),
        "workbook": WORKBOOK,
        "does_not_change_wb85b_thresholds": True,
        "does_not_change_alpha_allocation": True,
        "does_not_change_power_alternatives": True,
        "does_not_rerun_monte_carlo": True,
        "does_not_rerun_production_smoke": True,
        "does_not_change_solver_runner_or_physical_contract": True,
        "wb86_not_created_or_run": True,
    }


def claim_correction() -> dict[str, Any]:
    require_frozen_criticals()
    return {
        "schema": "wb85b_r1_claim_correction_v1",
        "decision_rule_unchanged": True,
        "alpha_LS": ALPHA_LS,
        "alpha_cov": ALPHA_COV,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "critical_LS": CRITICAL_LS,
        "critical_cov": CRITICAL_COV,
        "expected_critical_LS": EXPECTED_CRITICAL_LS,
        "expected_critical_cov": EXPECTED_CRITICAL_COV,
        "bonferroni_inequality": "P(A union B) <= P(A) + P(B)",
        "nominal_individual_size": 0.025,
        "finite_sample_FWER_mathematically_proven": False,
        "exact_finite_sample_FWER_proven": False,
        "nominal_primary_statistical_FWER_cap": 0.05,
        "fwer_le_0.05_is_conditional_on_actual_individual_gate_sizes_le_0.025": True,
        "unconditional_overall_qualification_false_fail_le_0.05": False,
        "recorded_joint_null_not_regenerated": {
            "T_LS_type_I": RECORDED_JOINT_NULL["T_LS_type_I"],
            "T_LS_cp95": list(RECORDED_JOINT_NULL["T_LS_cp95"]),
            "T_cov_type_I": RECORDED_JOINT_NULL["T_cov_type_I"],
            "T_cov_cp95": list(RECORDED_JOINT_NULL["T_cov_cp95"]),
            "union": RECORDED_JOINT_NULL["union"],
            "union_cp95": list(RECORDED_JOINT_NULL["union_cp95"]),
        },
        "individual_gate_nominal_calibration_compatible": True,
        "joint_null_nominal_FWER_compatible": True,
        "thresholds_not_retuned_from_this_wording_correction": True,
    }


def worktree_source_identity() -> dict[str, Any]:
    hashes = {}
    matches = {}
    missing = []
    for rel, digest in GENERATING_SOURCE_DIGESTS.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            hashes[rel] = ""
            matches[rel] = False
            continue
        got = sha256_file(path)
        hashes[rel] = got
        matches[rel] = got == digest
    blob = json.dumps(hashes, sort_keys=True).encode("utf-8")
    snapshot = hashlib.sha256(blob).hexdigest()
    identical = (not missing) and all(matches.values()) and snapshot == RUNTIME_SOURCE_SNAPSHOT
    return {
        "source_files": hashes,
        "file_matches_runtime_record": matches,
        "missing_source_files": missing,
        "runtime_source_snapshot_sha256": RUNTIME_SOURCE_SNAPSHOT,
        "worktree_source_snapshot_sha256": snapshot,
        "frozen_source_snapshot_sha256": snapshot,
        "byte_identical_to_runtime_snapshot": bool(identical),
    }


def write_source_archive(output_root: Path) -> dict[str, Any]:
    refuse_wb85b_rewrite(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    archive = output_root / "wb85b_generating_sources.tar"
    identity = worktree_source_identity()
    if not identity["byte_identical_to_runtime_snapshot"]:
        raise WB85BR1Error("refusing to archive sources that are not byte-identical to the runtime snapshot")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel in sorted(GENERATING_SOURCE_DIGESTS):
            data = (PROJECT_ROOT / rel).read_bytes()
            if hashlib.sha256(data).hexdigest() != GENERATING_SOURCE_DIGESTS[rel]:
                raise WB85BR1Error(f"archive member drifted before write: {rel}")
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    payload = buffer.getvalue()
    archive.write_bytes(payload)
    return {
        "archive_path": str(archive),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "n_members": len(GENERATING_SOURCE_DIGESTS),
        "deterministic_mtime_uid_gid": True,
        "generating_code_archive_complete": True,
    }


def verify_existing_wb85b_artifacts() -> dict[str, Any]:
    files = {
        WB85B_OUTPUT / "wb85b_fwer_protocol.json": WB85B_PROTOCOL_JSON_SHA256,
        WB85B_OUTPUT / "wb85b_acceptance_rule.json": WB85B_ACCEPTANCE_JSON_SHA256,
        WB85B_OUTPUT / "wb85b_joint_null_calibration.json": WB85B_JOINT_JSON_SHA256,
        WB85B_OUTPUT / "wb85b_power_validation.json": WB85B_POWER_JSON_SHA256,
        WB85B_OUTPUT / "wb85b_protocol_qa.json": WB85B_QA_JSON_SHA256,
        WB85B_SMOKE / "wb85b_official_runner_smoke.json": WB85B_SMOKE_SHA256,
    }
    matches = {str(path): path.is_file() and sha256_file(path) == digest for path, digest in files.items()}
    joint = json.loads((WB85B_OUTPUT / "wb85b_joint_null_calibration.json").read_text(encoding="utf-8"))
    smoke = json.loads((WB85B_SMOKE / "wb85b_official_runner_smoke.json").read_text(encoding="utf-8"))
    qa = json.loads((WB85B_OUTPUT / "wb85b_protocol_qa.json").read_text(encoding="utf-8"))
    cited = (
        abs(float(joint["location_scale"]["rate"]) - RECORDED_JOINT_NULL["T_LS_type_I"]) < 1.0e-12
        and abs(float(joint["coverage"]["rate"]) - RECORDED_JOINT_NULL["T_cov_type_I"]) < 1.0e-12
        and abs(float(joint["joint_union"]["rate"]) - RECORDED_JOINT_NULL["union"]) < 1.0e-12
    )
    return {
        "historical_dirs_present": all(
            path.is_dir()
            for path in (HISTORICAL_WB85, HISTORICAL_WB85A, HISTORICAL_WB85A_R1, HISTORICAL_WB85A_R1_SMOKE, WB85B_OUTPUT, WB85B_SMOKE)
        ),
        "artifact_hashes_unchanged": matches,
        "joint_null_numbers_cited_not_regenerated": cited,
        "official_qualification_runner_e2e_smoke_pass": bool(smoke.get("official_qualification_runner_e2e_smoke_pass")),
        "official_runner": smoke.get("official_runner"),
        "smoke_sha256": sha256_file(WB85B_SMOKE / "wb85b_official_runner_smoke.json"),
        "alignment_performance_not_reported": bool(smoke.get("alignment_performance_not_reported", True)),
        "is_not_physics_qualification": smoke.get("is_physics_qualification") is False,
        "wb85b_protocol_qa_pass": bool(qa.get("wb85b_protocol_qa_pass")),
        "monte_carlo_not_rerun": True,
        "production_smoke_not_rerun": True,
    }


def freeze_commit_report(*, freeze_commit: str | None = None) -> dict[str, Any]:
    identity = worktree_source_identity()
    head = _git("rev-parse", "HEAD")
    commit = freeze_commit or _git("rev-parse", "HEAD")
    tree = _git("rev-parse", f"{commit}^{{tree}}") if commit else ""
    blob_matches = {
        rel: _blob_sha(commit, rel) == digest for rel, digest in GENERATING_SOURCE_DIGESTS.items()
    } if commit else {rel: False for rel in GENERATING_SOURCE_DIGESTS}
    ancestor = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", commit, "origin/4station"],
        check=False,
    ) if commit else None
    identical = identity["byte_identical_to_runtime_snapshot"] and all(blob_matches.values())
    return {
        "runtime_parent_commit": RUNTIME_PARENT,
        "runtime_worktree_was_dirty": True,
        "runtime_source_snapshot_sha256": RUNTIME_SOURCE_SNAPSHOT,
        "worktree_source_snapshot_sha256": identity["worktree_source_snapshot_sha256"],
        "frozen_source_snapshot_sha256": identity["frozen_source_snapshot_sha256"],
        "post_run_source_freeze_commit": commit,
        "source_freeze_tree_sha": tree,
        "head": head,
        "post_run_freeze_contains_byte_identical_generating_sources": bool(identical),
        "post_run_freeze_file_matches": blob_matches,
        "cannot_call_post_run_commit_the_runtime_generating_commit": True,
        "remote_commit_verification": bool(ancestor is not None and ancestor.returncode == 0),
        "origin_4station": _git("rev-parse", "origin/4station"),
        "source_files": identity["source_files"],
    }


def build_verdict(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identical = bool(parts["freeze"]["post_run_freeze_contains_byte_identical_generating_sources"])
    archive_ok = bool(parts["archive"]["generating_code_archive_complete"])
    artifacts_ok = all(parts["artifacts"]["artifact_hashes_unchanged"].values())
    artifacts_ok = artifacts_ok and parts["artifacts"]["joint_null_numbers_cited_not_regenerated"]
    artifacts_ok = artifacts_ok and parts["artifacts"]["official_qualification_runner_e2e_smoke_pass"]
    artifacts_ok = artifacts_ok and parts["artifacts"]["wb85b_protocol_qa_pass"]
    provenance = bool(identical and archive_ok and artifacts_ok)
    ready = bool(provenance)
    return {
        **locked_r1_flags(),
        "schema": "wb85b_r1_claim_closure_v1",
        "claim_correction": claim_correction(),
        "generating_code_identity_complete": bool(identical),
        "generating_code_archive_complete": bool(archive_ok),
        "provenance_complete": bool(provenance),
        "wb85b_protocol_qa_pass": True,
        "nominal_statistical_protocol_qualified": bool(ready),
        "exact_finite_sample_FWER_proven": False,
        "finite_sample_FWER_mathematically_proven": False,
        "ready_to_authorize_physical_qualification": bool(ready),
        "qualification_authorized": False,
        "executable": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "wb86_automatically_authorized": False,
        "looked_at_physical_alignment_outcomes": False,
    }


def write_claim_closure_artifacts(
    output_root: Path | None = None,
    *,
    freeze_commit: str | None = None,
) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb85b_rewrite(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = worktree_source_identity()
    if not identity["byte_identical_to_runtime_snapshot"]:
        verdict = {
            **locked_r1_flags(),
            "schema": "wb85b_r1_claim_closure_v1",
            "generating_code_identity_complete": False,
            "generating_code_archive_complete": False,
            "provenance_complete": False,
            "ready_to_authorize_physical_qualification": False,
            "reason": "worktree generating sources are not byte-identical to the WB85b runtime snapshot",
            "identity": identity,
        }
        write_json(root / "wb85b_r1_claim_closure.json", verdict)
        return root
    archive = write_source_archive(root)
    artifacts = verify_existing_wb85b_artifacts()
    freeze = freeze_commit_report(freeze_commit=freeze_commit)
    verdict = build_verdict({"freeze": freeze, "archive": archive, "artifacts": artifacts})
    common = {**locked_r1_flags(), "utc": _utc()}
    write_json(root / "wb85b_r1_claim_correction.json", {**common, **claim_correction()})
    write_json(root / "wb85b_r1_source_identity.json", {**common, **identity, **freeze, **archive})
    write_json(root / "wb85b_r1_historical_artifact_audit.json", {**common, **artifacts})
    write_json(root / "wb85b_r1_claim_closure.json", {**common, **verdict, "freeze": freeze, "archive": archive})
    return root
