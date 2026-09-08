#!/usr/bin/env python3
"""Task B14P audit.  Target-independent weak-parameter prior admissibility."""

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
from datasets.target_independent_weak_prior_contract import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/target_independent_weak_prior_contract_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14p_target_independent_prior",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
        "wb109_dumps_not_overwritten": True,
        "prior_introduced": False,
    }
    store.write_json(
        "weak_parameter_prior_requirements.json",
        jsonable({"kind": "weak_parameter_prior_requirements", **common, **inventory["requirements"]}),
    )
    store.write_json(
        "target_independent_prior_inventory_v2.json",
        jsonable(
            {
                "kind": "target_independent_prior_inventory_v2",
                **common,
                **inventory["inventory"],
                "dependency_graph": inventory["dependency_graph"],
            }
        ),
    )
    store.write_json(
        "prior_level_semantics.json",
        jsonable({"kind": "prior_level_semantics", **common, **inventory["levels"]}),
    )
    store.write_json(
        "lto_measurement_prior_likelihood_contract.json",
        jsonable(
            {
                "kind": "lto_measurement_prior_likelihood_contract",
                **common,
                **inventory["likelihood_contract"],
            }
        ),
    )
    store.write_json(
        "prior_influence_falsification.json",
        jsonable({"kind": "prior_influence_falsification", **common, **inventory["influence"]}),
    )
    store.write_json(
        "focus_identity_prior_diagnostic.json",
        jsonable({"kind": "focus_identity_prior_diagnostic", **common, **inventory["focus"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "target_independent_prior_decision.json",
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
            "task": "SB-B14P",
            "workbook": 113,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb112_decision_sha256": decision["inherited_wb112_decision_sha256"],
            "inherited_wb111_decision_sha256": decision["inherited_wb111_decision_sha256"],
            "inherited_wb110_decision_sha256": decision["inherited_wb110_decision_sha256"],
            "inherited_wb109_decision_sha256": decision["inherited_wb109_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "transport_covariance_validated": False,
            "b15_authorized": False,
            "b14m_authorized": decision["b14m_authorized"],
            "lto_cin_contract_established": False,
            "prior_contract_established": decision["prior_contract_established"],
            "prior_introduced": False,
            "focus_identity_retained": True,
            "seed_scale_selected_from_closure": False,
            "wb109_dumps_not_overwritten": True,
        }
    )
    print(
        f"SB-B14P verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
