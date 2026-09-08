#!/usr/bin/env python3
"""Login-safe B14N entry: synthetic contract printout.

Does not launch Athena, does not enter B15, and does not submit the
1989-row campaign.  Official writes go through
audit_profile_likelihood_numerics.py.  ACTS smoke is
run_profile_likelihood_numerics_smoke.sh.
"""

from __future__ import annotations

import argparse
import json

from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.profiled_measurement_likelihood import validate_linear_profile_agreement
from datasets.profile_likelihood_numerics import inherit_frozen_stage, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profile_likelihood_numerics_v1.yaml")
    args = parser.parse_args()
    config = load_config(resolve_under_root(project_root(), args.config))
    inherited = inherit_frozen_stage(config)
    numerical = validate_linear_profile_agreement(
        relative=float(config["profile_numerical"]["pinv_relative"])
    )
    payload = {
        "task": "SB-B14N",
        "workbook": 115,
        "synthetic_passed": numerical["passed"],
        "inherited_wb114": inherited["workbook_114"]["decision"],
        "statistical_model_unchanged": True,
        "b15_authorized": False,
        "full_sample_authorized": False,
        "login_node_full_sample_forbidden": True,
        "numerical_validation": numerical,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if numerical["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
