#!/usr/bin/env python3
"""Workbook-75 residual-blind export & reinventory report driver.

Stages (each writes its own artifact under the configured output directory):

* ``manifest``      -- immutable input manifest from EOS/xAOD metadata only.
* ``export-gate``   -- frozen workbook-74 export gate verification.
* ``audit``         -- exporter-contract audit (code-level, pre-registered).
* ``validate``      -- deterministic provenance validation of exported
                       sources plus wrong-source / missing-output /
                       empty-SegmentFit negative controls.
* ``reinventory``   -- per-candidate residual-blind coverage recomputation
                       with the workbook-74 reporter and canonical envelope,
                       per-candidate verdicts, and the stage decision.
* ``all``           -- validate + reinventory (plus manifest/gate/audit when
                       their artifacts are absent).

This stage does not construct A = W^{1/2} J S, does not SVD, does not
inspect rank, does not inject an alignment payload, does not build a
physical FD point, and does not select events from residuals / Jacobian /
singular values / cosine / alignment response.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignment.cad_survey_nov22 import git_head_sha
from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.physically_distinct_track_coverage import (
    load_source_coverage,
    overlap_with_canonical,
    pool_population,
    refuse_forbidden_operations,
    admit_candidate,
    _empty_source_summary,
    _public_population,
    _strip_arrays,
)
from alignment.physically_distinct_track_coverage_export import (
    SCHEMA_VERSION,
    VERDICT_MISSING_OUTPUT,
    build_input_manifest,
    classify_candidate_verdict,
    decide_next_stage_75,
    exporter_contract_audit,
    load_events_physical_order,
    load_export_config,
    summarize_events_population,
    validate_exported_source,
    verify_export_gate,
    wrong_source_negative_control,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extras(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "stage": "residual_blind_htcondor_export_and_reinventory",
        "git_head": git_head_sha(project_root()),
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "inherited_workbook_74_config_sha256": config.get(
            "inherited_workbook_74_config_sha256"
        ),
        "forbidden_operations": refuse_forbidden_operations(),
        "fd_identifiability_executed": False,
        "svd_or_rank_computed": False,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _job_provenance(export_dir: Path) -> dict[str, Any] | None:
    return _load_json(export_dir / "job_provenance.json")


# ---------------------------------------------------------------------------
# Stage: manifest
# ---------------------------------------------------------------------------

def stage_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    manifest = build_input_manifest(config)
    payload = {**manifest, **_extras(config)}
    _write_json(output / "input_manifest.json", payload)
    payload["input_manifest_sha256"] = sha256_file(output / "input_manifest.json")
    _write_json(output / "input_manifest.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: export gate
# ---------------------------------------------------------------------------

def stage_export_gate(
    config: Mapping[str, Any], output: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    gate = verify_export_gate(manifest, config["coverage_gates"])
    payload = {**gate, **_extras(config)}
    _write_json(output / "export_gate.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: exporter-contract audit
# ---------------------------------------------------------------------------

def stage_audit(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    audit = exporter_contract_audit(config)
    payload = {**audit, **_extras(config)}
    _write_json(output / "exporter_contract_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: provenance validation
# ---------------------------------------------------------------------------

def stage_validate(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    root = project_root()
    candidates_report = []
    for candidate in config.get("export_candidates") or []:
        cid = str(candidate["id"])
        source_rows = []
        for row in candidate.get("inputs") or []:
            source_id = str(row["source_id"])
            export_dir = output / "exports" / cid / source_id
            job = _job_provenance(export_dir)
            result = validate_exported_source(
                source_id=source_id,
                expected_input_path=str(row["path"]),
                export_dir=export_dir,
                job_provenance=job,
                config=config,
            )
            source_rows.append(result)
        # Negative controls: wrong-source (validate source 0 against the
        # declared input of source 1) and missing-output (a source id that
        # was never exported).  Both must be detected; no fuzzy join.
        inputs = candidate.get("inputs") or []
        controls = []
        if len(inputs) >= 2:
            first = str(inputs[0]["source_id"])
            wrong_path = str(inputs[1]["path"])
            controls.append(
                wrong_source_negative_control(
                    source_id=first,
                    wrong_input_path=wrong_path,
                    export_dir=output / "exports" / cid / first,
                )
            )
        missing_dir = output / "exports" / cid / "negative_control_missing_output"
        missing = validate_exported_source(
            source_id="negative_control_missing_output",
            expected_input_path="/nonexistent/input.root",
            export_dir=missing_dir,
            job_provenance=None,
            config=config,
        )
        controls.append(
            {
                "control": "missing_output",
                "ran": True,
                "status": missing.get("status"),
                "control_passed": missing.get("status") == VERDICT_MISSING_OUTPUT,
            }
        )
        candidates_report.append(
            {
                "id": cid,
                "sources": source_rows,
                "n_sources_valid": sum(
                    1 for row in source_rows if row.get("provenance_valid")
                ),
                "n_sources_total": len(source_rows),
                "negative_controls": controls,
                "all_negative_controls_passed": all(
                    bool(row.get("control_passed")) for row in controls
                ),
            }
        )
    payload = {
        **_extras(config),
        "residual_blind": True,
        "deterministic": True,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
        "candidates": candidates_report,
        "all_sources_valid": all(
            row["n_sources_valid"] == row["n_sources_total"]
            for row in candidates_report
        ),
        "all_negative_controls_passed": all(
            row["all_negative_controls_passed"] for row in candidates_report
        ),
    }
    _write_json(output / "provenance_validation.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage: reinventory
# ---------------------------------------------------------------------------

def _canonical_pooled(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the canonical population with the workbook-74 reporter."""
    root = project_root()
    rows = [
        load_source_coverage(source, config=config, root=root)
        for source in config["canonical_population"]["sources"]
    ]
    return pool_population(
        rows, population_id=str(config["canonical_population"]["id"]), config=config
    )


def _canonical_regression(
    pooled: Mapping[str, Any], reference_path: Path
) -> dict[str, Any]:
    """Regression-check the rebuilt canonical envelope against workbook 74."""
    reference = _load_json(reference_path)
    if reference is None:
        return {"ran": False, "reason": f"reference absent: {reference_path}"}
    ref = reference.get("pooled") or {}
    keys = (
        "n_events",
        "n_tracklets",
        "n_angular_tracklets",
        "n_ift_events",
        "n_complete_four_station_events",
    )
    mismatches = []
    for key in keys:
        if int(pooled.get(key) or 0) != int(ref.get(key) or 0):
            mismatches.append(f"{key}: {pooled.get(key)} != {ref.get(key)}")
    for station in ("0", "1", "2", "3"):
        got = int((pooled.get("station_event_counts") or {}).get(station) or 0)
        want = int((ref.get("station_event_counts") or {}).get(station) or 0)
        if got != want:
            mismatches.append(f"station_event_counts[{station}]: {got} != {want}")
    ref_box = ref.get("quantile_box") or {}
    got_box = pooled.get("quantile_box") or {}
    for key in ("tx_lo", "tx_hi", "ty_lo", "ty_hi"):
        got = got_box.get(key)
        want = ref_box.get(key)
        if got is None or want is None or abs(float(got) - float(want)) > 1e-12:
            mismatches.append(f"quantile_box.{key}: {got} != {want}")
    return {
        "ran": True,
        "reference": str(reference_path),
        "matches_workbook_74": not mismatches,
        "mismatches": mismatches,
    }


def _candidate_summaries(
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
    *,
    abs_pdg: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    cid = str(candidate["id"])
    for row in candidate.get("inputs") or []:
        source_id = str(row["source_id"])
        tracklets = output / "exports" / cid / source_id / "tracklets.root"
        if not tracklets.is_file():
            summary = _empty_source_summary(source_id, "tracklets_file_missing")
            summary["tracklets"] = str(tracklets)
            rows.append(summary)
            continue
        events, load_prov = load_events_physical_order(tracklets)
        summary = summarize_events_population(
            events,
            source_id=source_id,
            physics=config["physics_scales"],
            phase_space=config["phase_space"],
            angular_abs_pdg=abs_pdg,
        )
        summary["tracklets"] = str(tracklets)
        summary["loader_provenance"] = load_prov
        rows.append(summary)
    return rows


def stage_reinventory(
    config: Mapping[str, Any],
    output: Path,
    validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    canonical = _canonical_pooled(config)
    regression = _canonical_regression(
        canonical,
        resolve_under_root(
            project_root(), str(config["inherited_workbook_74_canonical_coverage"])
        ),
    )
    validation_by_source: dict[str, Mapping[str, Any]] = {}
    if validation:
        for cand in validation.get("candidates") or []:
            for row in cand.get("sources") or []:
                validation_by_source[str(row["source_id"])] = row

    candidate_reports = []
    verdicts = []
    for candidate in config.get("export_candidates") or []:
        cid = str(candidate["id"])
        angular = candidate.get("angular_population") or {}
        abs_pdg = [int(value) for value in (angular.get("abs_pdg") or [13])]
        rows = _candidate_summaries(candidate, config, output, abs_pdg=abs_pdg)
        pooled = pool_population(rows, population_id=cid, config=config)
        overlap = overlap_with_canonical(pooled, canonical)
        admission = admit_candidate(
            candidate,
            pooled=pooled,
            overlap=overlap,
            gates=config["coverage_gates"],
        )
        source_valid = [
            bool(validation_by_source.get(str(row["source_id"]), {}).get("provenance_valid"))
            for row in candidate.get("inputs") or []
        ]
        provenance_valid = bool(validation_by_source) and all(source_valid)
        n_sources_with_output = sum(
            1 for row in rows if int(row.get("n_tracklets") or 0) > 0
        )
        verdict = classify_candidate_verdict(
            admission,
            provenance_valid=provenance_valid if validation_by_source else True,
            exporter_compatible=True,
            n_tracklets=int(pooled.get("n_tracklets") or 0),
            n_sources_with_output=n_sources_with_output,
        )
        verdict_row = {
            "id": cid,
            "verdict": verdict,
            "admitted_for_separate_5dof_fd_preregistration": verdict
            == "admitted_for_separate_5dof_fd_preregistration",
            "admission": admission,
            "provenance_valid": provenance_valid if validation_by_source else None,
            "n_sources_with_output": n_sources_with_output,
            "gates_not_lowered": True,
            "stage_75_does_not_open_fd": True,
        }
        verdicts.append(verdict_row)

        crosscheck = None
        cross_spec = candidate.get("crosscheck_angular_population")
        if cross_spec:
            cross_pdg = [int(value) for value in (cross_spec.get("abs_pdg") or [13])]
            cross_rows = _candidate_summaries(candidate, config, output, abs_pdg=cross_pdg)
            cross_pooled = pool_population(
                cross_rows, population_id=f"{cid}_muon_mask_crosscheck", config=config
            )
            crosscheck = {
                "abs_pdg": cross_pdg,
                "definition": cross_spec.get("definition"),
                "not_an_admission_input": True,
                "n_angular_tracklets": cross_pooled.get("n_angular_tracklets"),
                "tx": cross_pooled.get("tx"),
                "ty": cross_pooled.get("ty"),
            }
        candidate_reports.append(
            {
                "id": cid,
                "metadata_class": candidate.get("metadata_class"),
                "physically_distinct_hypothesis": candidate.get(
                    "physically_distinct_hypothesis"
                ),
                "angular_population": angular,
                "sources": [_strip_arrays(row) for row in rows],
                "pooled": _public_population(pooled),
                "overlap_with_canonical": overlap,
                "admission": admission,
                "verdict": verdict_row,
                "muon_mask_crosscheck": crosscheck,
                "residual_blind": True,
            }
        )

    decision = decide_next_stage_75(verdicts, fd_executed=False)
    payload = {
        **_extras(config),
        "residual_blind": True,
        "canonical_population": _public_population(canonical),
        "canonical_envelope_regression": regression,
        "candidates": candidate_reports,
        "verdicts": verdicts,
        "next_stage_decision": decision,
    }
    _write_json(output / "reinventory.json", payload)
    _write_json(output / "next_stage_decision.json", {**decision, **_extras(config)})
    return payload


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            project_root()
            / "configs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("manifest", "export-gate", "audit", "validate", "reinventory", "all"),
        default="all",
    )
    args = parser.parse_args()
    config = load_export_config(args.config)
    output = resolve_under_root(project_root(), str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["config_path"], output / "config.yaml")

    manifest = _load_json(output / "input_manifest.json")
    if args.stage in ("manifest", "all") and (args.stage == "manifest" or manifest is None):
        manifest = stage_manifest(config, output)
    if args.stage in ("export-gate", "all"):
        if manifest is None:
            raise RuntimeError("input_manifest.json is required; run --stage manifest first")
        gate = stage_export_gate(config, output, manifest)
        if args.stage == "export-gate":
            print(json.dumps({"all_candidates_satisfied": gate["all_candidates_satisfied"]}, indent=2))
            return
    if args.stage in ("audit", "all"):
        stage_audit(config, output)
        if args.stage == "audit":
            print(json.dumps({"audit": "written"}, indent=2))
            return
    if args.stage == "manifest":
        print(json.dumps({"manifest": "written", "sha256": manifest.get("input_manifest_sha256")}, indent=2))
        return

    validation = _load_json(output / "provenance_validation.json")
    if args.stage in ("validate", "all"):
        validation = stage_validate(config, output)
        if args.stage == "validate":
            print(
                json.dumps(
                    {
                        "all_sources_valid": validation["all_sources_valid"],
                        "all_negative_controls_passed": validation[
                            "all_negative_controls_passed"
                        ],
                    },
                    indent=2,
                )
            )
            return
    if args.stage in ("reinventory", "all"):
        payload = stage_reinventory(config, output, validation)
        decision = payload["next_stage_decision"]
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "canonical_envelope_matches_workbook_74": payload[
                        "canonical_envelope_regression"
                    ].get("matches_workbook_74"),
                    "verdicts": {
                        row["id"]: row["verdict"] for row in payload["verdicts"]
                    },
                    "decision": decision["decision"],
                    "admitted_candidates": decision.get("admitted_candidates"),
                    "fd_identifiability_executed": False,
                    "svd_or_rank_computed": False,
                    "three_arm_authorized": False,
                    "frozen_v2_alignment_loop_authorized": False,
                    "real_data_correction_authorized": False,
                    "geometry_write_allowed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
