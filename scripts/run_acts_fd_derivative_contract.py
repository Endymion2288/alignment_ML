#!/usr/bin/env python3
"""Login-safe B14L entry: inheritance and contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_acts_fd_derivative_contract.py.  ACTS smoke is
run_acts_fd_derivative_contract_smoke.sh.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.acts_fd_derivative_contract import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/acts_fd_derivative_contract_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14L",
        "workbook": 119,
        "inherited_wb118": inherited["workbook_118"]["decision"],
        "inherited_wb118_jacobian_contract_established": inherited["workbook_118"][
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
        "do_not_replace_official_likelihood": True,
        "five_d_cin_not_the_objective": True,
        "wb118_not_a_physical_conclusion": True,
        "likelihood_contract": likelihood_contract(),
        "control_events": ["100043/0", "100043/1", "100043/37"],
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
