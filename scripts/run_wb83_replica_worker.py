#!/usr/bin/env python3
"""Run one WB83 geometry cell over a replica range.  Truth-only.  No W64."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb83_qualification import EXPERIMENT, refuse_wb83_path, run_one_replica, write_json


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _git_porcelain() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / f"outputs/mc24_four_station_{EXPERIMENT}"))
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--replica-begin", type=int, default=0)
    parser.add_argument("--replica-end", type=int, default=100)
    parser.add_argument("--cluster-id", default=os.environ.get("CLUSTER", os.environ.get("CLUSTERID")))
    parser.add_argument("--proc-id", default=os.environ.get("PROCESS", os.environ.get("PROCID")))
    args = parser.parse_args()
    output = Path(args.output_root)
    refuse_wb83_path(output)
    matrix_path = output / "geometry_qualification_matrix.json"
    refuse_wb83_path(matrix_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cells = {cell["cell_id"]: cell for cell in matrix["cells"]}
    if args.cell_id not in cells:
        raise SystemExit(f"unknown cell_id {args.cell_id}")
    cell = cells[args.cell_id]
    shard_dir = output / "replicas" / args.cell_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    jsonl = shard_dir / f"{args.replica_begin:04d}_{args.replica_end:04d}.jsonl"
    refuse_wb83_path(jsonl)
    rows = []
    with jsonl.open("w", encoding="utf-8") as handle:
        for replica_id in range(int(args.replica_begin), int(args.replica_end)):
            row = run_one_replica(cell, replica_id, matrix_sha256=str(matrix["matrix_sha256"]))
            row["git_sha"] = _git_sha()
            row["git_status_porcelain"] = _git_porcelain()
            row["hostname"] = socket.gethostname()
            row["utc"] = datetime.now(timezone.utc).isoformat()
            row["condor_cluster"] = args.cluster_id
            row["condor_proc"] = args.proc_id
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            rows.append(row)
    write_json(
        shard_dir / f"{args.replica_begin:04d}_{args.replica_end:04d}.manifest.json",
        {
            "cell_id": args.cell_id,
            "replica_begin": int(args.replica_begin),
            "replica_end": int(args.replica_end),
            "n_written": len(rows),
            "matrix_sha256": matrix["matrix_sha256"],
            "jsonl": str(jsonl),
        },
    )
    print(json.dumps({"cell_id": args.cell_id, "n_written": len(rows), "jsonl": str(jsonl)}, indent=2))


if __name__ == "__main__":
    main()
