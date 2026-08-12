#!/usr/bin/env python3
"""Create the deterministic synthetic sample used by the baseline smoke test."""

from __future__ import annotations

import argparse

from datasets.synthetic import make_synthetic_tracklet_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output canonical ROOT file")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = make_synthetic_tracklet_root(args.output, seed=args.seed)
    print(output)


if __name__ == "__main__":
    main()

