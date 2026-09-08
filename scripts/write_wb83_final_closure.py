#!/usr/bin/env python3
"""Write the immutable WB83 closure and read-only physical-data audit.

Reads v1/v2 artifacts only.  Does not overwrite them, does not generate
replicas, and does not change frozen gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb83_qualification import (
    DESIGN_SEED,
    MASTER_SEED,
    N_TARGET_REPLICAS,
    RECONSTRUCTION_PROVENANCE,
    refuse_wb83_path,
    frozen_criteria,
    write_json,
)

V1 = PROJECT_ROOT / "outputs/mc24_four_station_wb83_truth_only_replicas_v1"
V2 = PROJECT_ROOT / "outputs/mc24_four_station_wb83_truth_only_replicas_v2"
CLOSURE = PROJECT_ROOT / "outputs/mc24_four_station_wb83_final_closure_v1"
FORBIDDEN_WRITE_ROOTS = (V1, V2)
OFFICIAL_JSON = (
    "truth_oracle_qualification.json",
    "coverage_report.json",
    "convergence_report.json",
    "geometry_qualification_matrix.json",
    "replica_manifest.json",
    "weak_mode_report.json",
    "measurement_likelihood_contract.json",
    "snapshot.json",
)
SOLVER_FILES = (
    "alignment/common_track_solver.py",
    "alignment/common_track_geometry.py",
    "alignment/wb83_qualification.py",
    "alignment/measurement_likelihood_contract.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, pattern: str) -> dict[str, object]:
    files = []
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            refuse_wb83_path(path)
            files.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)})
    combined = hashlib.sha256()
    for item in files:
        combined.update(item["sha256"].encode("ascii"))
        combined.update(b"\n")
    return {"n_files": len(files), "combined_sha256": combined.hexdigest(), "files": files}


def _refuse_overwrite(path: Path) -> None:
    resolved = path.resolve()
    for root in FORBIDDEN_WRITE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise SystemExit(f"WB83 closure refuses to write inside {root}")


def _load(root: Path, name: str) -> dict:
    path = root / name
    refuse_wb83_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_bundle(root: Path, *, extra: tuple[str, ...] = ()) -> dict[str, object]:
    names = list(OFFICIAL_JSON) + list(extra)
    files = {}
    for name in names:
        path = root / name
        if path.exists():
            refuse_wb83_path(path)
            files[name] = sha256_file(path)
    replicas = sha256_tree(root / "replicas", "*/*.jsonl")
    return {
        "root": str(root.relative_to(PROJECT_ROOT)),
        "official_json_sha256": files,
        "replica_jsonl": replicas,
    }


def closure_payload() -> dict[str, object]:
    v1_gate = _load(V1, "truth_oracle_qualification.json")
    v2_gate = _load(V2, "truth_oracle_qualification.json")
    v2_cov = _load(V2, "coverage_report.json")
    v2_matrix = _load(V2, "geometry_qualification_matrix.json")
    v1_matrix = _load(V1, "geometry_qualification_matrix.json")
    diagnostics = _load(V2, "remaining_alignment_mechanism.json")
    if v1_matrix["matrix_sha256"] != v2_matrix["matrix_sha256"]:
        raise SystemExit("v1/v2 matrix SHA mismatch")
    failing = [
        {
            "cell_id": cell["cell_id"],
            "qualification": cell["qualification"],
            "reason": cell["reason"],
            "pull_mean": cell.get("pull_mean"),
            "coverage": cell.get("coverage"),
            "n_covered_95": cell.get("n_covered_95"),
            "n_replicas": cell.get("n_replicas"),
        }
        for cell in v2_cov["cells"]
        if cell.get("qualification") != "PASS"
    ]
    return {
        "workbook": "83",
        "schema": "wb83_final_closure_v1",
        "immutable": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "v1_status": "FAIL",
        "v2_status": "FAIL",
        "v1_dag_cluster": 1109787,
        "v2_dag_cluster": 1109806,
        "v2_label": "final solver-fix rerun",
        "solver_implementation_fixed": True,
        "convergence_qualified": True,
        "fd_stability_qualified": True,
        "bias_screening_qualified": True,
        "pulls_qualified": False,
        "coverage_qualified": False,
        "toy_qualification": "FAIL",
        "physical_FASER_oracle_qualified": False,
        "common_track_solver_qualified_under_toy_model": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "alignment_oracle_qualified": False,
        "ml_alignment_eval_authorized": False,
        "further_identical_rerun_authorized": False,
        "reinterpreting_wb83_as_pass_forbidden": True,
        "systematic_bias_not_established": True,
        "remaining_failures_are_frozen_ensemble_statistical_failures": True,
        "association_default_system": "frozen_W64_raw_energy_plus_exact_solver",
        "reconstruction_provenance": RECONSTRUCTION_PROVENANCE,
        "matrix_sha256": v2_matrix["matrix_sha256"],
        "seeds": {
            "master_seed": MASTER_SEED,
            "design_seed": DESIGN_SEED,
            "n_target_replicas": N_TARGET_REPLICAS,
            "replica_seed_formula": "sha256(f'{MASTER}|{condition_id}|{amplitude:.6f}|{survey_mode}|{replica_id}') -> 31-bit",
        },
        "qualification_rules": frozen_criteria(),
        "solver_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in SOLVER_FILES},
        "v1_artifacts": artifact_bundle(V1),
        "v2_artifacts": artifact_bundle(V2, extra=("remaining_alignment_mechanism.json",)),
        "v1_gate": {
            "qualification": v1_gate["qualification"],
            "alignment_oracle_qualified": v1_gate["alignment_oracle_qualified"],
            "gates": v1_gate["gates"],
        },
        "v2_gate": {
            "qualification": v2_gate["qualification"],
            "solver_implementation_fixed": v2_gate.get("solver_implementation_fixed"),
            "common_track_solver_qualified_under_toy_model": v2_gate.get(
                "common_track_solver_qualified_under_toy_model"
            ),
            "alignment_oracle_qualified_for_physical_FASER": v2_gate.get(
                "alignment_oracle_qualified_for_physical_FASER"
            ),
            "gates": v2_gate["gates"],
        },
        "remaining_official_fails": failing,
        "diagnostics": {
            "path": "outputs/mc24_four_station_wb83_truth_only_replicas_v2/remaining_alignment_mechanism.json",
            "sha256": sha256_file(V2 / "remaining_alignment_mechanism.json"),
            "conclusions": diagnostics.get("conclusions"),
        },
        "forbidden_asset_status": {
            "overlay_used": False,
            "w64_used": False,
            "ml_association_used": False,
            "development_00350_used": False,
            "final_blind_eval_authorized": False,
            "sealed_test_accessed": False,
            "no_forbidden_assets_accessed": True,
        },
        "forbidden_next": [
            "identical_matrix_seed_threshold_rerun",
            "relax_pull_mean_0.2",
            "change_coverage_rule",
            "add_wb83_replicas",
            "change_weak_direction",
            "change_seeds",
            "change_prior_sigma",
            "change_damping",
            "new_solver_patch",
            "reinterpret_as_pass",
            "claim_proven_systematic_bias",
            "return_to_association",
            "open_00350_final_blind_sealed",
        ],
    }


def protocol_design() -> dict[str, object]:
    return {
        "schema": "prospective_alignment_qualification_protocol_design_v0",
        "workbook": "83",
        "executable": False,
        "new_replicas_authorized": False,
        "is_wb83_rerun": False,
        "trained_on_wb83_failure_numbers": False,
        "purpose": (
            "Design space for a future independent common-track qualification. "
            "Not an authorization to generate replicas."
        ),
        "motivation_from_wb83_without_reusing_its_gate": {
            "old_rule": "every cell must independently satisfy an unadjusted threshold",
            "statistical_issue": (
                "An intersection-union of many unadjusted cellwise tests has a "
                "family-wise false-rejection rate much larger than any single-cell "
                "nominal level, even when the estimator is unbiased and calibrated."
            ),
            "wb83_may_illustrate_old_behavior_only": True,
            "wb83_numbers_are_not_tuning_targets": True,
        },
        "must_distinguish": {
            "single_cell_calibration": (
                "Per-cell mean/width/coverage estimates and their sampling "
                "uncertainty.  Used for diagnostics and for power calculations."
            ),
            "global_multi_cell_qualification": (
                "One pre-registered decision for the solver under a stated "
                "simultaneous error rate.  Cellwise numbers do not each have "
                "to clear an unadjusted threshold."
            ),
        },
        "required_freeze_before_any_new_replica": [
            "global_null",
            "cell_mode_family",
            "simultaneous_error_rate",
            "sample_size",
            "acceptance_statistic",
            "multiplicity_handling",
            "unknown_criterion",
            "bias_coverage_pull_reporting_rules",
            "weak_mode_treatment",
            "new_independent_replica_seeds",
        ],
        "new_experiment_must_use": {
            "new_independent_replica_seeds": True,
            "seeds_frozen_before_execution": True,
            "must_not_reuse_wb83_master_seed": 20260907,
            "must_not_reuse_wb83_design_seed": 202609070,
            "must_not_reuse_wb83_matrix_as_a_new_gate": True,
            "label": "new prospective qualification experiment",
            "not_a_wb83_rerun": True,
        },
        "global_null_candidates": {
            "N0_unbiased_calibrated": (
                "For every pre-registered identifiable mode, the iterated "
                "estimator's projected residual is zero-mean with the stated "
                "identity-linearized variance, up to the declared approximation."
            ),
            "N0_no_persistent_bias": (
                "Cell means are exchangeable around zero after studentization; "
                "any apparent cell shift is finite-replica noise."
            ),
        },
        "prospective_methods": {
            "M1_multiplicity_controlled_cellwise": {
                "idea": (
                    "Keep cellwise tests, but control a pre-registered family-wise "
                    "or false-discovery rate instead of requiring every unadjusted "
                    "gate to pass."
                ),
                "examples": [
                    "Holm-Bonferroni on pre-declared cellwise p-values",
                    "Hochberg if positive dependence is justified a priori",
                    "hierarchical closed testing: global first, then families, then cells",
                ],
                "must_pre_register": [
                    "which statistics are tests vs reports",
                    "the family partition (identifiable translation / rotation / weak JG / survey mode)",
                    "FWER or FDR level",
                    "how many tests enter the correction",
                ],
                "strength": "Preserves interpretability of individual cells.",
                "weakness": "Still fragments information; weak modes dominate the correction if treated as ordinary cells.",
            },
            "M2_global_pull_gof": {
                "idea": (
                    "Jointly test mean and width calibration of standardized "
                    "residuals across all pre-registered modes/cells."
                ),
                "examples": [
                    "Hotelling or chi-square on the vector of cellwise pull means",
                    "global scale test on pooled studentized residuals after a declared covariance",
                    "combined location-scale statistic with a single critical value",
                ],
                "must_pre_register": [
                    "the stacked residual map",
                    "whether cells are treated as independent (they are if seeds are)",
                    "the single acceptance region",
                    "how coverage is folded in, if at all",
                ],
                "strength": "Matches the scientific claim 'the solver is calibrated' rather than 14 separate claims.",
                "weakness": "A single bad cell can be diluted; must pre-declare a complementary lack-of-fit breakdown, not a post-hoc one.",
            },
            "M3_hierarchical_random_effects": {
                "idea": (
                    "Separate persistent cell bias from finite-replica ensemble "
                    "fluctuation.  Qualify on the persistent-bias component."
                ),
                "examples": [
                    "cell mean ~ N(0, tau^2) with sampling variance sigma_c^2 / n",
                    "reject only if tau^2 is incompatible with 0 at the declared level",
                    "posterior predictive check of ensemble fluctuation vs persistent shift",
                ],
                "must_pre_register": [
                    "the random-effects law",
                    "the estimator of tau^2",
                    "the decision threshold on tau^2 or on a Bayes factor",
                    "whether weak modes have their own variance component",
                ],
                "strength": "Directly targets the scientific distinction WB83 diagnostics pointed at.",
                "weakness": "Needs an a priori variance-component model; easy to overfit if tuned after seeing cells.",
            },
        },
        "method_choice_rule": (
            "Choose M1, M2, or M3 — or a pre-declared hybrid — before generating "
            "any new replica.  Do not pick the method that would have passed WB83."
        ),
        "sample_size_rule": (
            "Fix n from power against a scientifically relevant alternative "
            "(for example a persistent bias of a declared fraction of the mode "
            "sigma), after multiplicity or hierarchical structure.  Do not set n "
            "so that a previous FAIL flips."
        ),
        "unknown_criterion": (
            "If the realized independent replica count is below the frozen n, "
            "the experiment is UNKNOWN, never PASS."
        ),
        "weak_mode_treatment_options": {
            "separate_family": "Own error rate; not pooled with identifiable translation cells.",
            "variance_component": "Enter M3 as a high-sigma random effect, not as a 0.1 mm screening.",
            "report_only": "Pre-declare that weak JG is diagnostic, not part of the global pass/fail.",
            "forbidden": "Do not invent a new weak direction after seeing data.",
        },
        "forbidden_uses_of_wb83": [
            "threshold optimization on pull_mean = -0.262893",
            "threshold optimization on 89/100 coverage",
            "seed reuse",
            "adding replicas to the same matrix",
            "declaring WB83 PASS under a new rule",
        ],
    }


def physical_feasibility() -> dict[str, object]:
    return {
        "schema": "calypso_physical_replica_feasibility_v1",
        "workbook": "83",
        "read_only": True,
        "opened_forbidden_assets": False,
        "overlay_pseudo_replicas_used": False,
        "development_00350_used": False,
        "final_blind_opened": False,
        "sealed_opened": False,
        "question": (
            "Without overlays, 00350, Final Blind, or sealed assets, can we "
            "obtain enough independent Calypso physical/simulation events for "
            "a real physical-model alignment qualification?"
        ),
        "physical_alignment_qualification_blocked_by_data": True,
        "calypso_physical_replicas_available": False,
        "reason": (
            "No independent Calypso physical-replica corpus exists under the "
            "current data contract.  Existing large samples are overlay "
            "pseudo-replicas or forbidden ranges.  Source-pure physical refit "
            "subsets already audited are O(50) events per source, far below a "
            "prospective multi-cell qualification, and held-out geometry has "
            "physical_refit_authorized = false."
        ),
        "available_independent_sources_events": {
            "authorized_train_xaod_ids": [
                "mc24_100043_00200_00299",
                "mc24_100044_00300_00399",
                "mc24_100043_00300_00399",
                "mc24_100044_00200_00299",
                "mc24_100047_00100_00149",
                "mc24_100048_00100_00149",
            ],
            "generated_event_scale_from_wb64_tables": {
                "mc24_100043_00200_00299": 500000,
                "mc24_100044_00300_00399": 500000,
                "mc24_100043_00300_00399": 500000,
                "mc24_100044_00200_00299": 500000,
                "mc24_100047_00100_00149": 250000,
                "mc24_100048_00100_00149": 250000,
                "note": "Generated xAOD counts, not alignment-ready reconstructed replicas.",
            },
            "already_used_by_w64_association": True,
            "usable_today_as_alignment_replicas": False,
            "source_pure_refit_subsets_from_wb78": {
                "mc24_100047_00100_00149_identity": {"events": 50, "tracklets": 188},
                "mc24_100048_00100_00149_identity": {"events": 50, "tracklets": 189},
                "note": "Read-only historical audit counts.  00350 rows are listed only as already-opened development diagnostics and are not available for qualification.",
            },
            "overlay_corpora": {
                "status": "exist_but_forbidden_as_replicas",
                "historical_overlay_seed": 20260813,
                "wb80_holdout_overlay_events": {"family1": 3360, "family2": 1680},
                "independence": "same_event_geometry_payload_reuse",
            },
        },
        "missing_samples": [
            "independent Calypso events reconstructed once, not overlaid",
            "station (x,y,tx,ty) plus physical covariance",
            "physical field / material / ACTS transport (not toy uniform By)",
            "known injected or survey alignment payload per event",
            "truth association for truth-only alignment, without W64 selection",
            "enough events for the future global statistical protocol after n is frozen",
        ],
        "required_new_simulation_reconstruction_production": {
            "pipeline": (
                "Calypso /Tracker/Align -> physical refit -> ACTS "
                "(q_over_p_mode=0, s0013-r0022 or a newly declared tag) "
                "on independent events, one reconstruction per event"
            ),
            "must_not_use": [
                "overlay_synthetic",
                "mc24_100047_00350_00399",
                "mc24_100048_00350_00399",
                "mc24_100047_00800_00849",
                "mc24_100048_00800_00849",
                "mc24_100116_*",
                "mc24_100117_*",
            ],
            "heldout_geometry_seed_314159": {
                "table_exists": True,
                "physical_refit_authorized": False,
                "events_produced": 0,
            },
            "if_reusing_train_range_xaod": (
                "Disclose that W64 saw these six sources.  Truth-only alignment "
                "may still use newly reconstructed events, but they are not "
                "source-unseen for the default association system."
            ),
        },
        "estimated_provenance_contract": {
            "reconstruction_provenance_required": "calypso_physical_independent_v1",
            "current_wb83_provenance": RECONSTRUCTION_PROVENANCE,
            "field_material": "Calypso physical, not toy_uniform_By_v1",
            "overlay_replicas_forbidden": True,
            "same_event_payload_reuse_forbidden": True,
            "association": "truth_only",
            "default_association_system_unchanged": "frozen_W64_raw_energy_plus_exact_solver",
            "alignment_not_a_certified_downstream_oracle": True,
        },
        "not_started": {
            "new_qualification_run": True,
            "new_physical_production": True,
            "wb83_replica_rerun": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(CLOSURE))
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    _refuse_overwrite(output)
    refuse_wb83_path(output)
    if output == V1.resolve() or output == V2.resolve():
        raise SystemExit("refusing to write closure into a replica output root")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "wb83_final_closure.json", closure_payload())
    write_json(output / "prospective_statistical_protocol_design.json", protocol_design())
    write_json(output / "calypso_physical_replica_feasibility.json", physical_feasibility())
    print(
        json.dumps(
            {
                "output_root": str(output.relative_to(PROJECT_ROOT)),
                "v1_status": "FAIL",
                "v2_status": "FAIL",
                "toy_qualification": "FAIL",
                "further_identical_rerun_authorized": False,
                "physical_alignment_qualification_blocked_by_data": True,
                "new_replicas_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
