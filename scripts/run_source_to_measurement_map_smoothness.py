#!/usr/bin/env python3
"""Login-safe B14K entry: inheritance and contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_source_to_measurement_map_smoothness.py.  ACTS smoke is
run_source_to_measurement_map_smoothness_smoke.sh.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.source_to_measurement_map_smoothness import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/source_to_measurement_map_smoothness_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14K",
        "workbook": 118,
        "inherited_wb117": inherited["workbook_117"]["decision"],
        "inherited_wb117_jacobian_contract_established": inherited["workbook_117"][
            "jacobian_contract_established"
        ],
        "statistical_model_unchanged": True,
        "b15_authorized": False,
        "b14m_reopen_authorized": False,
        "restart_invariance_authorized": False,
        "full_sample_authorized": False,
        "login_node_full_sample_forbidden": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_switch_to_direct": True,
        "five_d_cin_not_the_objective": True,
        "wb117_not_a_physical_conclusion": True,
        "likelihood_contract": likelihood_contract(),
        "control_event": "100048/44",
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
