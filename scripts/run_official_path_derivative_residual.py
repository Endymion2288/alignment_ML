#!/usr/bin/env python3
"""Login-safe B14V entry: inheritance and residual-contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_official_path_derivative_residual.py.  Frozen WB120 dumps are
the only smoke input.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.official_path_derivative_residual import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/official_path_derivative_residual_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14V",
        "workbook": 121,
        "inherited_wb120": inherited["workbook_120"]["decision"],
        "inherited_wb120_jacobian_contract_established": inherited["workbook_120"][
            "jacobian_contract_established"
        ],
        "inherited_wb120_same_path_jacobian_available": inherited["workbook_120"][
            "same_path_jacobian_available"
        ],
        "inherited_wb119_took_wrong_jacobian": inherited["workbook_119"][
            "took_wrong_jacobian"
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
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_change_jacobian_implementation": True,
        "five_d_cin_not_the_objective": True,
        "wb120_not_a_physical_conclusion": True,
        "likelihood_contract": likelihood_contract(),
        "control_events": ["100043/0", "100043/1", "100043/37"],
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
