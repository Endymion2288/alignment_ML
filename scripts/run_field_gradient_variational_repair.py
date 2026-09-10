#!/usr/bin/env python3
"""Login-safe B14X entry: inheritance and source-audit printout.

Does not launch Athena, does not enter B15, and does not submit 1989.
Official writes go through audit_field_gradient_variational_repair.py.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.field_gradient_variational_repair import (
    audit_acts_source,
    audit_equation_contract,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/field_gradient_variational_repair_v1.yaml"
    )
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    payload = {
        "task": "SB-B14X",
        "workbook": 123,
        "inherited_wb122": inherited["workbook_122"]["decision"],
        "inherited_wb122_jacobian_contract_established": inherited["workbook_122"][
            "jacobian_contract_established"
        ],
        "b14m_reopen_authorized": False,
        "full_sample_authorized": False,
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_tune_field_step_from_track_jacobian": True,
        "source_audit": audit_acts_source(),
        "equation": audit_equation_contract(),
        "likelihood_contract": likelihood_contract(),
        "control_event": "100043/1",
        "focus_event": "100048/86",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
