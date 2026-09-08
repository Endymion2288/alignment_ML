"""WB85b prospective protocol QA.  Synthetic only.  Does not run WB86."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import stats

from alignment.calypso_physical_replica_production import sha256_file
from alignment.wb85_physical_qualification_protocol import (
    ALT_MEAN_PRIMARY,
    ALT_SCALE,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    POWER_TARGET,
    refuse_forbidden_path,
    write_json,
)
from alignment.wb85a_protocol_qa import (
    GROSS_COVERAGE_FRACTION,
    IDENTIFIABLE_STRATUM_COUNTS,
    N_OFFICIAL_CROSSCHECK,
    N_POWER_MONTE_CARLO,
    POWER_ABS_TOL,
    WB84_BUNDLE,
    WB84_GEOMETRY,
    WB84_MANIFEST,
    WB84_PROVENANCE,
    WB84_QA,
    WB84_SNAPSHOT,
    WB85_PYTHON,
    binomial_report,
    fail_closed_suite,
    healthy_identifiable_row,
    identifiable_ids,
    nuisance_covariance_report,
    statistical_unit_contract_report,
    tcov_statistic,
    type_i_accept,
    weak_jg_contract_report,
)
from alignment.wb85a_r1_protocol_qa import (
    FROZEN_PARENT,
    draw_coherent_null,
)
from alignment.wb85b_fwer_protocol import (
    ALPHA_COV,
    ALPHA_LS,
    CRITICAL_COV,
    CRITICAL_LS,
    FAMILYWISE_ALPHA,
    HISTORICAL_WB85,
    HISTORICAL_WB85A,
    HISTORICAL_WB85A_R1,
    HISTORICAL_WB85A_R1_SMOKE,
    OUTPUT_ROOT,
    WB85BError,
    analytic_power_table,
    apply_wb85b_statistical_gate,
    evaluate_wb85b_qualification,
    fwer_claim_scope,
    historical_revision_semantics,
    locked_flags,
    protocol_freeze,
    refuse_historical_rewrite,
    require_frozen_criticals,
    statistical_family,
)
from alignment.wb85b_official_runner import (
    SMOKE_OUTPUT as OFFICIAL_SMOKE_OUTPUT,
    official_runner_contract,
    planner_execute_stays_false,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_NULL_SEED = 2026090820
POWER_VALIDATION_SEED = 2026090821
N_JOINT_NULL_MONTE_CARLO = 20000
TYPE_I_POINT_BAND = (0.015, 0.040)
SCALE_DEFLATION = 0.7632762616558972
R1_SOURCE_SNAPSHOT = "1a5e0f3f75e6a9e399930756f9572b3abbb1e35459577c3772cd1180b197a540"
R1_RUNTIME_PARENT = "fe00c0a9674b861f408e5c7e6c0d00163a42d116"
R1_POST_RUN_FREEZE = "582316b662fe9f5d7ad1bd1f5957a3258af7e6bf"
R1_HEAD_AFTER_WORKBOOK = "fe2a97666d688e31380ff27f8550dd629631daea"

GENERATING_SOURCES = (
    PROJECT_ROOT / "alignment" / "wb85b_fwer_protocol.py",
    PROJECT_ROOT / "alignment" / "wb85b_protocol_qa.py",
    PROJECT_ROOT / "alignment" / "wb85b_official_runner.py",
    PROJECT_ROOT / "alignment" / "wb85a_r1_runtime_smoke.py",
    PROJECT_ROOT / "alignment" / "wb85a_r1_protocol_qa.py",
    PROJECT_ROOT / "alignment" / "wb85_projected_unit.py",
    PROJECT_ROOT / "alignment" / "wb85_physical_qualification_protocol.py",
    PROJECT_ROOT / "alignment" / "physical_common_track_execution.py",
    PROJECT_ROOT / "scripts" / "run_wb85b_protocol_qa.py",
    PROJECT_ROOT / "scripts" / "run_wb85b_official_runner_smoke.py",
    PROJECT_ROOT / "scripts" / "run_wb85b_official_runner_smoke_condor.sh",
    PROJECT_ROOT / "scripts" / "submit_wb85b_official_runner_smoke_condor.py",
    PROJECT_ROOT / "configs" / "research_review" / "wp85b_fwer_protocol.yaml",
    PROJECT_ROOT / "tests" / "test_wb85b_fwer_protocol.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_counts() -> dict[str, int]:
    if tuple(IDENTIFIABLE_STRATUM_COUNTS[name] for name in identifiable_ids()) != (
        299,
        303,
        311,
        298,
        300,
        283,
    ):
        raise WB85BError("WB85b must use the frozen WB84 identifiable-stratum counts")
    return dict(IDENTIFIABLE_STRATUM_COUNTS)


def acceptance_rule() -> dict[str, Any]:
    require_frozen_criticals()
    return {
        "schema": "wb85b_acceptance_rule_v1",
        "frozen_before_any_new_monte_carlo": True,
        "may_retune_from_wb85a_r1_correlations": False,
        "alpha_allocation_sweep_forbidden": True,
        "may_silently_change_alpha": False,
        "n_joint_monte_carlo": N_JOINT_NULL_MONTE_CARLO,
        "joint_null_seed": JOINT_NULL_SEED,
        "stratum_counts": _require_counts(),
        "statistical_family": statistical_family(),
        "fwer_claim_scope": fwer_claim_scope(),
        "historical_revision_semantics": historical_revision_semantics(),
        "location_scale_type_i": {
            "target": ALPHA_LS,
            "accept_if_cp95_contains_target": True,
            "point_estimate_band": list(TYPE_I_POINT_BAND),
        },
        "coverage_type_i": {
            "target": ALPHA_COV,
            "accept_if_cp95_contains_target": True,
            "point_estimate_band": list(TYPE_I_POINT_BAND),
        },
        "joint_union": {
            "primary_guarantee": "bonferroni_FWER_le_0.05",
            "simulation_is_not_the_fwer_proof": True,
            "implementation_or_null_model_inconsistency_if": "union 95% CP lower bound > 0.05",
        },
        "power_validation": {
            "n_monte_carlo": N_POWER_MONTE_CARLO,
            "seed": POWER_VALIDATION_SEED,
            "compare_at_n": N_MIN_PER_IDENTIFIABLE_STRATUM,
            "must_meet_power_target": POWER_TARGET,
            "required_alternatives": (
                "persistent_mean_0.35_one_stratum",
                "scale_1.25_one_stratum",
            ),
            "analytic_power_is_approximation": True,
            "mc_is_empirical_qa": True,
            "may_retune_alternative_or_sample_size": False,
        },
    }


def run_joint_null_calibration(
    *,
    n_mc: int = N_JOINT_NULL_MONTE_CARLO,
    seed: int = JOINT_NULL_SEED,
) -> dict[str, Any]:
    counts = _require_counts()
    rng = np.random.default_rng(seed)
    reject_ls = np.empty(int(n_mc), dtype=np.int8)
    reject_cov = np.empty(int(n_mc), dtype=np.int8)
    t_ls_values = np.empty(int(n_mc), dtype=np.float64)
    t_cov_values = np.empty(int(n_mc), dtype=np.float64)
    official_mismatch = 0
    from alignment.wb85_physical_qualification_protocol import location_scale_gof

    for trial in range(int(n_mc)):
        z, covered = draw_coherent_null(rng, counts)
        z_lists = {name: values.tolist() for name, values in z.items()}
        gof = location_scale_gof(z_lists)
        t_cov = float(tcov_statistic(covered))
        gate = apply_wb85b_statistical_gate(t_ls=float(gof["T_LS"]), t_cov=t_cov)
        t_ls_values[trial] = float(gate["T_LS"])
        t_cov_values[trial] = float(gate["T_cov"])
        reject_ls[trial] = int(gate["reject_LS"])
        reject_cov[trial] = int(gate["reject_cov"])
        if trial < N_OFFICIAL_CROSSCHECK:
            bundle = {
                "z": z_lists,
                "coverage": covered,
                "identifiable_strata": {
                    name: healthy_identifiable_row(n, covered[name][0]) for name, n in counts.items()
                },
                "weak_strata": {},
            }
            future = evaluate_wb85b_qualification(bundle)
            official_gate = future.get("statistical_gate")
            if official_gate is None:
                official_mismatch += 1
            else:
                if abs(float(official_gate["T_cov"]) - t_cov) > 1.0e-12:
                    official_mismatch += 1
                official = future["qualification_status"] != "PASS"
                if official != bool(gate["primary_statistical_fail"]) and future["qualification_status"] != "UNKNOWN":
                    official_mismatch += 1
    union = reject_ls | reject_cov
    ls_report = binomial_report(int(reject_ls.sum()), n_mc)
    cov_report = binomial_report(int(reject_cov.sum()), n_mc)
    union_report = binomial_report(int(union.sum()), n_mc)
    ls_ok = type_i_accept(ls_report, target=ALPHA_LS, band=TYPE_I_POINT_BAND)
    cov_ok = type_i_accept(cov_report, target=ALPHA_COV, band=TYPE_I_POINT_BAND)
    inconsistency = float(union_report["cp95_lo"]) > FAMILYWISE_ALPHA
    calibrated = ls_ok and cov_ok and official_mismatch == 0 and not inconsistency

    def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
        if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "n_monte_carlo": int(n_mc),
        "seed": int(seed),
        "stratum_counts": counts,
        "reads_physical_alignment_output": False,
        "critical_LS": CRITICAL_LS,
        "critical_cov": CRITICAL_COV,
        "location_scale": {**ls_report, "accepted": ls_ok, "target": ALPHA_LS},
        "coverage": {**cov_report, "accepted": cov_ok, "target": ALPHA_COV},
        "joint_union": {
            **union_report,
            "primary_guarantee": "bonferroni_FWER_le_0.05",
            "simulation_is_not_the_fwer_proof": True,
            "implementation_or_null_model_inconsistency": inconsistency,
        },
        "corr_T_LS_T_cov": _corr(t_ls_values, t_cov_values),
        "corr_reject_LS_reject_cov": _corr(reject_ls.astype(np.float64), reject_cov.astype(np.float64)),
        "official_function_crosscheck": {"n": N_OFFICIAL_CROSSCHECK, "mismatches": official_mismatch},
        "joint_synthetic_null_calibrated": bool(calibrated),
        "implementation_or_null_model_inconsistency": bool(inconsistency),
        "thresholds_not_retuned_from_this_simulation": True,
    }


def _power_reject(
    rng: np.random.Generator,
    counts: Mapping[str, int],
    *,
    mean_shift: Mapping[str, float] | None = None,
    scale: Mapping[str, float] | None = None,
    coverage_p: Mapping[str, float] | None = None,
    use_full_gate: bool = True,
) -> bool:
    from alignment.wb85a_protocol_qa import draw_null_coverage, draw_null_z
    from alignment.wb85_physical_qualification_protocol import clopper_pearson

    z = draw_null_z(rng, counts, mean_shift=mean_shift, scale=scale)
    if use_full_gate:
        covered = draw_null_coverage(rng, counts, p=coverage_p)
        bundle = {
            "z": {name: values.tolist() for name, values in z.items()},
            "coverage": covered,
            "identifiable_strata": {
                name: healthy_identifiable_row(n, covered[name][0]) for name, n in counts.items()
            },
            "weak_strata": {},
        }
        if coverage_p:
            for name, prob in coverage_p.items():
                bundle["identifiable_strata"][name]["empirical_coverage"] = float(prob)
                k, n = covered[name]
                _lo, hi = clopper_pearson(k, n)
                bundle["identifiable_strata"][name]["cp_hi"] = hi
        return evaluate_wb85b_qualification(bundle)["qualification_status"] != "PASS"
    from alignment.wb85_physical_qualification_protocol import location_scale_gof

    gof_z = {name: values.tolist() for name, values in z.items()}
    return float(location_scale_gof(gof_z)["T_LS"]) > CRITICAL_LS


def run_power_validation(
    *,
    n_mc: int = N_POWER_MONTE_CARLO,
    seed: int = POWER_VALIDATION_SEED,
) -> dict[str, Any]:
    n = N_MIN_PER_IDENTIFIABLE_STRATUM
    counts = {name: n for name in identifiable_ids()}
    target = identifiable_ids()[0]
    rng = np.random.default_rng(seed)
    analytic = analytic_power_table(n=n)
    cases = {
        "persistent_mean_0.35_one_stratum": {
            "analytic": float(analytic["persistent_mean_0.35_one_stratum"]),
            "mean_shift": {target: ALT_MEAN_PRIMARY},
            "required": True,
        },
        "scale_1.25_one_stratum": {
            "analytic": float(analytic["scale_1.25_one_stratum"]),
            "scale": {target: ALT_SCALE},
            "required": True,
        },
        "scale_deflation_0.763_one_stratum": {
            "analytic": float(analytic["scale_deflation_0.763_one_stratum"]),
            "scale": {target: SCALE_DEFLATION},
            "required": False,
        },
    }
    reports = {}
    required_ok = True
    for name, spec in cases.items():
        rejects = 0
        for _ in range(int(n_mc)):
            if _power_reject(
                rng,
                counts,
                mean_shift=spec.get("mean_shift"),
                scale=spec.get("scale"),
                use_full_gate=False,
            ):
                rejects += 1
        report = binomial_report(rejects, n_mc)
        sufficient = report["rate"] >= POWER_TARGET
        analytic_sufficient = spec["analytic"] >= POWER_TARGET
        reports[name] = {
            **report,
            "analytic": spec["analytic"],
            "analytic_power_is_approximation": True,
            "mc_is_empirical_qa": True,
            "mc_meets_80_percent": sufficient,
            "analytic_meets_80_percent": analytic_sufficient,
            "absolute_gap": abs(report["rate"] - spec["analytic"]),
            "absolute_tolerance_not_used_to_retune": POWER_ABS_TOL,
            "required": spec["required"],
            "injected_stratum": target,
            "n_per_stratum": n,
            "decision": "T_LS > critical_LS",
        }
        if spec["required"] and not (sufficient and analytic_sufficient):
            required_ok = False
    from alignment.wb85_physical_qualification_protocol import (
        COVERAGE_GROSS_CP_UPPER_MIN,
        COVERAGE_GROSS_EMPIRICAL_MIN,
        clopper_pearson,
    )
    from alignment.wb85a_protocol_qa import draw_null_coverage, draw_null_z

    gross_rejects = 0
    official_gross_mismatch = 0
    actual = _require_counts()
    for trial in range(int(n_mc)):
        z = draw_null_z(rng, actual)
        covered = draw_null_coverage(rng, actual, p={target: GROSS_COVERAGE_FRACTION})
        k, n_covered = covered[target]
        empirical = k / float(n_covered)
        _lo, hi = clopper_pearson(k, n_covered)
        gross_fail = empirical < COVERAGE_GROSS_EMPIRICAL_MIN or hi < COVERAGE_GROSS_CP_UPPER_MIN
        if gross_fail:
            gross_rejects += 1
        if trial < N_OFFICIAL_CROSSCHECK:
            bundle = {
                "z": {name: values.tolist() for name, values in z.items()},
                "coverage": covered,
                "identifiable_strata": {
                    name: healthy_identifiable_row(count, covered[name][0]) for name, count in actual.items()
                },
                "weak_strata": {},
            }
            bundle["identifiable_strata"][target]["empirical_coverage"] = float(GROSS_COVERAGE_FRACTION)
            bundle["identifiable_strata"][target]["cp_hi"] = hi
            official = evaluate_wb85b_qualification(bundle)["qualification_status"] != "PASS"
            if official != bool(gross_fail):
                official_gross_mismatch += 1
    gross = binomial_report(gross_rejects, n_mc)
    gross_ok = gross["rate"] >= 0.99 and official_gross_mismatch == 0
    reports["gross_coverage_deficit"] = {
        **gross,
        "coverage_p": GROSS_COVERAGE_FRACTION,
        "required": True,
        "decision": "evaluate_wb85b_qualification guardrail",
        "official_function_crosscheck": {"n": N_OFFICIAL_CROSSCHECK, "mismatches": official_gross_mismatch},
        "mc_meets_80_percent": gross_ok,
    }
    required_ok = required_ok and gross_ok
    return {
        "n_monte_carlo": int(n_mc),
        "seed": int(seed),
        "compare_at_n": N_MIN_PER_IDENTIFIABLE_STRATUM,
        "gross_coverage_uses_wb84_stratum_counts": True,
        "old_alpha_0.05_power_not_assumed": True,
        "analytic_table": analytic,
        "reports": reports,
        "power_requirement_pass": bool(required_ok),
        "power_validation_pass": bool(required_ok),
        "alternatives_not_retuned": True,
        "sample_size_claim_not_changed": True,
    }


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


def r1_runtime_provenance_correction() -> dict[str, Any]:
    r1_files = {
        "alignment/wb85a_r1_runtime_smoke.py": "ac4204508d8417253a259d331e0c4c8f37478b2a4311958bccf54735a991d2a5",
        "alignment/wb85a_r1_protocol_qa.py": "fc657dab2cc071d6f8e624bd52d9dc59d34a12721adf7f2fbb2f13637d373ac2",
        "alignment/wb85_projected_unit.py": "3287a5c71dffd6fc7a3c4c0080f74f3b0e2654e47a3271477e8469755d2d2513",
        "alignment/physical_common_track_execution.py": "08770a252f1a0ea8adef9ee120289cb91fec2597d7d377717ffd591436bf8072",
        "alignment/wb85a_protocol_qa.py": "563e000aef861d0a4ac6cb55f78cfcb08a8ca285e3ff60291efb970fd8992983",
    }
    freeze_matches = {
        rel: _blob_sha(R1_POST_RUN_FREEZE, rel) == digest for rel, digest in r1_files.items()
    }
    ancestor = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", R1_POST_RUN_FREEZE, "origin/4station"],
        check=False,
    )
    return {
        "runtime_parent_commit": R1_RUNTIME_PARENT,
        "runtime_source_snapshot_sha256": R1_SOURCE_SNAPSHOT,
        "runtime_worktree_was_dirty_untracked": True,
        "post_run_source_freeze_commit": R1_POST_RUN_FREEZE,
        "post_run_freeze_contains_byte_identical_generating_sources": all(freeze_matches.values()),
        "post_run_freeze_file_matches": freeze_matches,
        "cannot_call_post_run_commit_the_runtime_generating_commit": True,
        "remote_commit_verification": ancestor.returncode == 0,
        "origin_4station": _git("rev-parse", "origin/4station"),
        "workbook_note_commit": R1_HEAD_AFTER_WORKBOOK,
    }


def generating_code_identity() -> dict[str, Any]:
    hashes = {}
    missing = []
    for path in GENERATING_SOURCES:
        rel = str(path.relative_to(PROJECT_ROOT))
        if path.is_file():
            hashes[rel] = sha256_file(path)
        else:
            missing.append(rel)
    blob = json.dumps(hashes, sort_keys=True).encode("utf-8")
    return {
        "source_files": hashes,
        "missing_source_files": missing,
        "source_snapshot_sha256": hashlib.sha256(blob).hexdigest(),
        "wb85b_protocol_sha256": sha256_file(PROJECT_ROOT / "alignment" / "wb85b_fwer_protocol.py"),
        "wb85_python_sha256": sha256_file(WB85_PYTHON),
    }


def git_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    head = _git("rev-parse", "HEAD")
    return {
        "git_sha": head,
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "frozen_parent": FROZEN_PARENT,
        "head_is_frozen_parent": head == FROZEN_PARENT,
    }


def load_official_runner_smoke(path: Path | None = None) -> dict[str, Any]:
    target = OFFICIAL_SMOKE_OUTPUT / "wb85b_official_runner_smoke.json" if path is None else Path(path)
    contract = official_runner_contract()
    historical = HISTORICAL_WB85A_R1 / "wb85a_r1_runtime_smoke_gate.json"
    engine_pass = False
    if historical.is_file():
        engine_pass = bool(json.loads(historical.read_text(encoding="utf-8")).get("physical_execution_runtime_smoke_pass"))
    r1_smoke = HISTORICAL_WB85A_R1_SMOKE / "wb85a_r1_runtime_smoke.json"
    if not engine_pass and r1_smoke.is_file():
        engine_pass = bool(json.loads(r1_smoke.read_text(encoding="utf-8")).get("physical_execution_runtime_smoke_pass"))
    if not target.is_file():
        return {
            **contract,
            "report_present": False,
            "calypso_acts_engine_runtime_smoke_pass": engine_pass,
            "official_qualification_runner_e2e_smoke_pass": False,
            "reason": "official qualification runner e2e smoke report is absent",
        }
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {
        **contract,
        "report_present": True,
        "report_path": str(target),
        "calypso_acts_engine_runtime_smoke_pass": engine_pass,
        "official_qualification_runner_e2e_smoke_pass": bool(
            payload.get("official_qualification_runner_e2e_smoke_pass")
        ),
        "updated_sqlite_contains_preregistered_s1_dx": payload.get("updated_sqlite_contains_preregistered_s1_dx"),
        "iteration2_consumed_updated_sqlite": payload.get("iteration2_consumed_updated_sqlite"),
        "alignment_performance_not_reported": payload.get("alignment_performance_not_reported", True),
    }


def provenance_freeze_report(*, output_root: Path | None = None) -> dict[str, Any]:
    refuse_historical_rewrite(output_root or OUTPUT_ROOT)
    snapshot = json.loads(WB84_SNAPSHOT.read_text(encoding="utf-8"))
    qa = json.loads(WB84_QA.read_text(encoding="utf-8"))
    generating = generating_code_identity()
    git = git_identity()
    r1 = r1_runtime_provenance_correction()
    historical_present = all(path.is_dir() for path in (HISTORICAL_WB85, HISTORICAL_WB85A, HISTORICAL_WB85A_R1))
    complete = (
        bool(generating["source_files"])
        and not generating["missing_source_files"]
        and historical_present
        and qa["n_qualified_exports"] == 2394
        and bool(snapshot.get("field_map", {}).get("hashed_files"))
        and r1["post_run_freeze_contains_byte_identical_generating_sources"]
    )
    return {
        **locked_flags(),
        "schema": "wb85b_provenance_freeze_v1",
        "utc": _utc(),
        "git": git,
        "generating_code": generating,
        "wb85a_r1_runtime_provenance": r1,
        "wb84_material_map_hashed_files_left_empty": not bool(snapshot.get("material_map", {}).get("hashed_files")),
        "wb84_material_hashes_not_backfilled": True,
        "historical_artifacts_present": historical_present,
        "historical_wb84_hashes": {
            "wb84_corpus_manifest_sha256": sha256_file(WB84_MANIFEST),
            "wb84_corpus_qa_sha256": sha256_file(WB84_QA),
            "wb84_provenance_freeze_sha256": sha256_file(WB84_PROVENANCE),
            "calypso_source_bundle_sha256": sha256_file(WB84_BUNDLE),
            "wb84_geometry_manifest_sha256": sha256_file(WB84_GEOMETRY),
            "wb84_snapshot_sha256": sha256_file(WB84_SNAPSHOT),
        },
        "calypso_planner_execute_stays_false": planner_execute_stays_false(),
        "provenance_complete": bool(complete),
    }


def wb85b_fail_closed_report() -> dict[str, Any]:
    """Same frozen guardrail cases, decided by evaluate_wb85b_qualification."""
    original = fail_closed_suite()
    names = identifiable_ids()
    unit = np.ones(N_MIN_PER_IDENTIFIABLE_STRATUM, dtype=np.float64)
    centered = np.concatenate(
        [-unit[: N_MIN_PER_IDENTIFIABLE_STRATUM // 2], unit[N_MIN_PER_IDENTIFIABLE_STRATUM // 2 :]]
    )
    centered = centered - centered.mean()
    centered = centered / centered.std(ddof=1)
    z = {name: centered.tolist() for name in names}
    coverage = {name: (190, 200) for name in names}
    healthy = {name: healthy_identifiable_row(200, 190) for name in names}

    def _status(**overrides: Any) -> str:
        bundle = {
            "z": dict(z),
            "coverage": dict(coverage),
            "identifiable_strata": {name: dict(row) for name, row in healthy.items()},
            "weak_strata": {},
        }
        bundle.update(overrides)
        try:
            return str(evaluate_wb85b_qualification(bundle)["qualification_status"])
        except Exception:
            return "FAIL"

    cases = {
        "missing_stratum": _status(z={name: z[name] for name in names[:-1]}),
        "n_below_200": _status(z={name: centered[:199].tolist() for name in names}),
        "nan": _status(z={**z, names[0]: [float("nan")] + centered.tolist()[1:]}),
        "inf": _status(z={**z, names[1]: [float("inf")] + centered.tolist()[1:]}),
        "nonconvergence_gt_5pct": _status(
            identifiable_strata={**{name: dict(row) for name, row in healthy.items()}, names[0]: {**healthy[names[0]], "n_converged": 189}}
        ),
        "unresolved_non_spd": _status(
            identifiable_strata={**{name: dict(row) for name, row in healthy.items()}, names[1]: {**healthy[names[1]], "n_non_spd": 1}}
        ),
        "dropped_identifiable_mode": _status(
            identifiable_strata={**{name: dict(row) for name, row in healthy.items()}, names[0]: {**healthy[names[0]], "dropped": True}}
        ),
    }
    wb85b_ok = all(status in {"FAIL", "UNKNOWN"} for status in cases.values())
    wb85b_ok = wb85b_ok and cases["n_below_200"] == "UNKNOWN" and cases["missing_stratum"] == "FAIL"
    return {
        **original,
        "wb85b_cases": cases,
        "wb85b_guardrails_use_evaluate_wb85b_qualification": True,
        "fail_closed_tests_pass": bool(original["fail_closed_tests_pass"] and wb85b_ok),
    }


def build_verdict(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "protocol_freeze_pass": True,
        "joint_synthetic_null_calibrated": bool(parts["joint"]["joint_synthetic_null_calibrated"]),
        "power_validation_pass": bool(parts["power"]["power_validation_pass"]),
        "statistical_unit_contract_pass": bool(parts["units"]["statistical_unit_contract_pass"]),
        "nuisance_covariance_pass": bool(parts["nuisance"]["nuisance_covariance_pass"]),
        "fail_closed_tests_pass": bool(parts["gates"]["fail_closed_tests_pass"]),
        "weak_jg_contract_pass": bool(parts["weak"]["weak_jg_contract_pass"]),
        "calypso_acts_engine_runtime_smoke_pass": bool(parts["smoke"]["calypso_acts_engine_runtime_smoke_pass"]),
        "official_qualification_runner_e2e_smoke_pass": bool(
            parts["smoke"]["official_qualification_runner_e2e_smoke_pass"]
        ),
        "provenance_complete": bool(parts["provenance"]["provenance_complete"]),
    }
    inconsistency = bool(parts["joint"]["implementation_or_null_model_inconsistency"])
    power_req = bool(parts["power"]["power_requirement_pass"])
    all_true = all(required.values()) and (not inconsistency) and power_req
    return {
        **locked_flags(),
        "schema": "wb85b_protocol_qa_v1",
        "acceptance_rule_frozen_before_calibration": True,
        "acceptance_rule": acceptance_rule(),
        "checks": required,
        "implementation_or_null_model_inconsistency": inconsistency,
        "power_requirement_pass": power_req,
        "wb85b_protocol_qa_pass": bool(all_true),
        "ready_to_authorize_physical_qualification": bool(all_true),
        "qualification_authorized": False,
        "executable": False,
        "wb86_automatically_authorized": False,
        "wb86_not_created_or_run": True,
    }


def run_protocol_qa(
    *,
    n_joint: int = N_JOINT_NULL_MONTE_CARLO,
    n_power: int = N_POWER_MONTE_CARLO,
    smoke_path: Path | None = None,
) -> dict[str, Any]:
    require_frozen_criticals()
    joint = run_joint_null_calibration(n_mc=n_joint)
    power = run_power_validation(n_mc=n_power)
    units = statistical_unit_contract_report()
    nuisance = nuisance_covariance_report()
    gates = wb85b_fail_closed_report()
    weak = weak_jg_contract_report()
    smoke = load_official_runner_smoke(smoke_path)
    provenance = provenance_freeze_report()
    verdict = build_verdict(
        {
            "joint": joint,
            "power": power,
            "units": units,
            "nuisance": nuisance,
            "gates": gates,
            "weak": weak,
            "smoke": smoke,
            "provenance": provenance,
        }
    )
    return {
        "verdict": verdict,
        "joint": joint,
        "power": power,
        "units": units,
        "nuisance": nuisance,
        "gates": gates,
        "weak": weak,
        "smoke": smoke,
        "provenance": provenance,
        "protocol": protocol_freeze(),
    }


def write_protocol_qa_artifacts(
    output_root: Path | None = None,
    *,
    n_joint: int = N_JOINT_NULL_MONTE_CARLO,
    n_power: int = N_POWER_MONTE_CARLO,
    smoke_path: Path | None = None,
) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_historical_rewrite(root)
    refuse_forbidden_path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "wb85b_acceptance_rule.json", {**locked_flags(), **acceptance_rule()})
    write_json(root / "wb85b_fwer_protocol.json", protocol_freeze())
    freeze_hashes = {
        **locked_flags(),
        "schema": "wb85b_protocol_freeze_hashes_v1",
        "utc_before_monte_carlo": _utc(),
        "frozen_before_any_new_monte_carlo": True,
        "wb85b_fwer_protocol_sha256": sha256_file(root / "wb85b_fwer_protocol.json"),
        "wb85b_acceptance_rule_sha256": sha256_file(root / "wb85b_acceptance_rule.json"),
    }
    write_json(root / "wb85b_protocol_freeze_hashes.json", freeze_hashes)
    bundle = run_protocol_qa(n_joint=n_joint, n_power=n_power, smoke_path=smoke_path)
    if sha256_file(root / "wb85b_fwer_protocol.json") != freeze_hashes["wb85b_fwer_protocol_sha256"]:
        raise WB85BError("WB85b protocol file changed after the pre-MC freeze")
    common = {**locked_flags(), "generating_code": bundle["provenance"]["generating_code"], "git": bundle["provenance"]["git"]}
    artifacts = {
        "wb85b_protocol_qa.json": {**common, **bundle["verdict"], "protocol_freeze_hashes": freeze_hashes},
        "wb85b_joint_null_calibration.json": {**common, **bundle["joint"]},
        "wb85b_power_validation.json": {**common, **bundle["power"]},
        "wb85b_official_runner_contract.json": {**common, **bundle["smoke"]},
        "wb85b_provenance_freeze.json": bundle["provenance"],
    }
    for name, payload in artifacts.items():
        if payload.get("qualification_authorized") is not False:
            raise WB85BError("WB85b artifacts must keep qualification unauthorized")
        if payload.get("wb86_automatically_authorized") is not False:
            raise WB85BError("WB85b must not automatically authorize WB86")
        write_json(root / name, payload)
    return root


def refresh_verdict_with_smoke(output_root: Path | None = None, *, smoke_path: Path | None = None) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_historical_rewrite(root)
    joint = json.loads((root / "wb85b_joint_null_calibration.json").read_text(encoding="utf-8"))
    power = json.loads((root / "wb85b_power_validation.json").read_text(encoding="utf-8"))
    units = statistical_unit_contract_report()
    nuisance = nuisance_covariance_report()
    gates = wb85b_fail_closed_report()
    weak = weak_jg_contract_report()
    smoke = load_official_runner_smoke(smoke_path)
    provenance = provenance_freeze_report(output_root=root)
    verdict = build_verdict(
        {
            "joint": joint,
            "power": power,
            "units": units,
            "nuisance": nuisance,
            "gates": gates,
            "weak": weak,
            "smoke": smoke,
            "provenance": provenance,
        }
    )
    common = {**locked_flags(), "generating_code": provenance["generating_code"], "git": provenance["git"]}
    write_json(root / "wb85b_protocol_qa.json", {**common, **verdict})
    write_json(root / "wb85b_official_runner_contract.json", {**common, **smoke})
    write_json(root / "wb85b_provenance_freeze.json", provenance)
    return root
