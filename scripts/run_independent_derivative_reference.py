#!/usr/bin/env python3
"""Login-safe B14W entry: inheritance and residual-reference printout.

Does not launch Athena, does not enter B15, and does not submit 1989.
Official writes go through audit_independent_derivative_reference.py.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.independent_derivative_reference import (
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/independent_derivative_reference_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14W",
        "workbook": 122,
        "inherited_wb121": inherited["workbook_121"]["decision"],
        "inherited_wb121_jacobian_contract_established": inherited["workbook_121"][
            "jacobian_contract_established"
        ],
        "inherited_wb120_same_path_jacobian_available": inherited["workbook_120"][
            "same_path_jacobian_available"
        ],
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "do_not_change_jacobian_implementation": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_shrink_fd_step": True,
        "likelihood_contract": likelihood_contract(),
        "control_event": "100043/1",
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
