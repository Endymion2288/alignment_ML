#!/usr/bin/env python3
"""T10 reference-oracle gap.  Frozen arrays only; no refit, no sealed test."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.reference_oracle_gap import (
    decide,
    evaluate_split,
    load_config,
    load_frozen_arrays,
    load_frozen_report,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reference_oracle_gap_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "t10_reference_oracle",
    )
    hashes = {
        "kind": "input_hashes",
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    results = {}
    paired_rows = []
    for split, spec in config["splits"].items():
        arrays = load_frozen_arrays(spec["arrays"])
        report_path = resolve_under_root(project_root(), spec["report"])
        report_sha = sha256_file(report_path)
        if arrays["sha256"] != spec["arrays_sha256"]:
            raise ValueError(f"{split} arrays hash mismatch")
        if report_sha != spec["report_sha256"]:
            raise ValueError(f"{split} report hash mismatch")
        hashes["files"][split] = {
            "arrays": spec["arrays"],
            "arrays_sha256": arrays["sha256"],
            "report": spec["report"],
            "report_sha256": report_sha,
        }
        report = load_frozen_report(spec["report"])
        results[split] = evaluate_split(arrays, report, split=split)
        for row in results[split]["source_effects_zero"]:
            paired_rows.append({"split": split, "estimator": "zero_target", **row})
        for row in results[split]["source_effects_paired"]:
            paired_rows.append({"split": split, "estimator": "paired_target", **row})
    decision = decide(results)
    store.write_json("input_hashes.json", hashes)
    store.write_json("paired_vs_absolute.json", {"kind": "paired_vs_absolute", "splits": results})
    store.write_json("decision.json", decision)
    csv_path = store.run_dir / "source_effects.csv"
    if csv_path.exists():
        raise FileExistsError(csv_path)
    fields = sorted({key for row in paired_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "T10",
            "old_wls_authorized_for_data_correction": False,
        }
    )
    print(f"T10 verdict={decision['verdict']} run_id={store.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
