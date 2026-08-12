#!/usr/bin/env python3
"""Audit optional q/p fields in a canonical FASER tracklet dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.root_loader import load_events
from evaluation.qoverp import audit_q_over_p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Canonical flat ROOT tracklet file")
    parser.add_argument("--output", default=None, help="Optional JSON destination")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--absolute-floor-per-mev", type=float, default=1.0e-15)
    args = parser.parse_args()

    result = audit_q_over_p(
        load_events(args.input, max_events=args.max_events),
        relative_tolerance=args.relative_tolerance,
        absolute_floor_per_mev=args.absolute_floor_per_mev,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
