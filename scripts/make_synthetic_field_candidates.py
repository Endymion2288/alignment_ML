#!/usr/bin/env python3
"""Create field-aware candidate records for a synthetic multi-track ROOT file."""

from __future__ import annotations

import argparse
import json

from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-tracklets", required=True)
    parser.add_argument("--source-propagations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--q-over-p-mode", type=int, default=0, choices=(0,))
    parser.add_argument("--target-z-tolerance-mm", type=float, default=1.0e-6)
    args = parser.parse_args()
    summary = write_synthetic_field_candidate_root(
        synthetic_tracklets=args.synthetic_tracklets,
        source_propagations=args.source_propagations,
        destination=args.output,
        q_over_p_mode=args.q_over_p_mode,
        target_z_tolerance_mm=args.target_z_tolerance_mm,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
