#!/usr/bin/env python3
"""Classify and explicitly retry failed workbook-75 export jobs.

No failed job is silently skipped.  Each failed job is classified from its
``job_provenance.json`` step exit codes:

* ``audit_loader_event_id_collision`` -- the Calypso export, conversion,
  and schema check all succeeded; only the content audit failed because
  the sorted ``load_events`` grouping merges the distinct physical events
  that merged MC24 rec files produce under reused generator-job event
  numbers.  Retried by rerunning only the content audit with
  ``--physical-order`` on the existing ``tracklets.root``.  The Calypso
  export is never rerun.
* ``export_chain_failed`` -- any earlier step failed; not retried here.

Every retry writes ``content_audit_retry.json`` with the classification,
the exact command, the git SHA, and the exit status.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import project_root
from alignment.physically_distinct_track_coverage_export import load_export_config

AUDIT_COLLISION = "audit_loader_event_id_collision"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_export_config(args.config)
    output = Path(args.output_dir).expanduser().resolve()
    root = project_root()

    rows = []
    for candidate in config.get("export_candidates") or []:
        cid = str(candidate["id"])
        for row in candidate.get("inputs") or []:
            source_id = str(row["source_id"])
            export_dir = output / "exports" / cid / source_id
            job = _load(export_dir / "job_provenance.json")
            record: dict[str, Any] = {
                "schema_version": "faser-residual-blind-tracklet-export-audit-retry-v1",
                "source_id": source_id,
                "candidate": cid,
                "residual_blind": True,
            }
            if job is None:
                record["classification"] = "missing_job_provenance"
                record["retried"] = False
                rows.append(record)
                continue
            record["original_exit_status"] = job.get("exit_status")
            record["original_step_exit_codes"] = job.get("step_exit_codes")
            if job.get("exit_status") == 0:
                record["classification"] = "ok_no_retry_needed"
                record["retried"] = False
                rows.append(record)
                continue
            steps = job.get("step_exit_codes") or {}
            upstream_ok = (
                int(steps.get("calypso_export", 99)) == 0
                and int(steps.get("convert", 99)) == 0
                and int(steps.get("schema_check", 99)) == 0
            )
            tracklets = export_dir / "tracklets.root"
            if upstream_ok and int(steps.get("content_audit", 99)) != 0 and tracklets.is_file():
                record["classification"] = AUDIT_COLLISION
                record["classification_detail"] = (
                    "Calypso export, conversion, and schema check succeeded; the "
                    "content audit used the sorted load_events grouping, which "
                    "merges distinct physical events under reused generator-job "
                    "event numbers in merged MC24 rec files.  Retried with "
                    "--physical-order; the Calypso export is not rerun."
                )
                command = [
                    "python",
                    "-m",
                    "scripts.audit_tracklets",
                    str(tracklets),
                    "--output",
                    str(export_dir / "content_audit.json"),
                    "--physical-order",
                ]
                started = datetime.now(timezone.utc).isoformat()
                result = subprocess.run(
                    command,
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                record.update(
                    {
                        "retried": True,
                        "command": " ".join(command),
                        "git_head": git_head_sha(root),
                        "retry_start_utc": started,
                        "retry_end_utc": datetime.now(timezone.utc).isoformat(),
                        "retry_exit_status": int(result.returncode),
                        "retry_output_tail": result.stdout[-4000:] if result.stdout else "",
                        "content_audit_exists": (export_dir / "content_audit.json").is_file(),
                    }
                )
            else:
                record["classification"] = "export_chain_failed"
                record["retried"] = False
                record["detail"] = (
                    "an upstream export step failed; rerunning the audit cannot "
                    "repair it and the job is left classified, not skipped"
                )
            (export_dir / "content_audit_retry.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            rows.append(record)

    summary = {
        "schema_version": "faser-residual-blind-tracklet-export-audit-retry-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head_sha(root),
        "config_path": config["config_path"],
        "n_jobs": len(rows),
        "n_ok_no_retry": sum(1 for r in rows if r["classification"] == "ok_no_retry_needed"),
        "n_audit_collision_retried": sum(
            1 for r in rows if r["classification"] == AUDIT_COLLISION and r.get("retried")
        ),
        "n_retry_succeeded": sum(
            1 for r in rows if r.get("retried") and r.get("retry_exit_status") == 0
        ),
        "n_export_chain_failed": sum(
            1 for r in rows if r["classification"] == "export_chain_failed"
        ),
        "n_missing_job_provenance": sum(
            1 for r in rows if r["classification"] == "missing_job_provenance"
        ),
        "rows": [
            {
                "source_id": r["source_id"],
                "classification": r["classification"],
                "retried": r.get("retried"),
                "retry_exit_status": r.get("retry_exit_status"),
            }
            for r in rows
        ],
        "no_failed_job_silently_skipped": True,
    }
    (output / "audit_retry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
