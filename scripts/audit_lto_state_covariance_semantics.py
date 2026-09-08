#!/usr/bin/env python3
"""Task B14 audit.  LTO state / covariance semantics before propagation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_diagnosis import jsonable
from datasets.lto_state_covariance_semantics import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lto_state_covariance_semantics_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14_lto_covariance_semantics",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
    }
    store.write_json(
        "official_cin_semantics.json",
        jsonable({"kind": "official_cin_semantics", **common, **inventory["official_shape"]}),
    )
    store.write_json(
        "lto_cin_semantics.json",
        jsonable({"kind": "lto_cin_semantics", **common, **inventory["lto_shape"]}),
    )
    store.write_json(
        "official_vs_lto_cin_comparison.json",
        jsonable(
            {"kind": "official_vs_lto_cin_comparison", **common, **inventory["comparison"]}
        ),
    )
    store.write_json(
        "focus_identity_lto_semantics.json",
        jsonable({"kind": "focus_identity_lto_semantics", **common, **inventory["focus"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "lto_state_covariance_semantics_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                "n_raw": inventory["n_raw"],
                "n_ineligible": inventory["n_ineligible"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14",
            "workbook": 110,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb109_decision_sha256": decision["inherited_wb109_decision_sha256"],
            "inherited_wb105_decision_sha256": decision["inherited_wb105_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "transport_covariance_validated": False,
            "lto_cin_contract_established": decision["lto_cin_contract_established"],
            "b15_authorized": decision["b15_authorized"],
            "focus_identity_retained": True,
            "truth_used_as_diagnostic_only": True,
        }
    )
    print(
        f"SB-B14 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
