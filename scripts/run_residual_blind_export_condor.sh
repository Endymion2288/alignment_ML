#!/usr/bin/env bash
# Run one residual-blind tracklet export job inside an isolated Condor worker.
#
# Workbook 75: each job exports exactly one declared input xAOD through the
# validated nominal chain (persisted SegmentFit -> GhostBusters ->
# NtupleDumperAlg detailed tracklets -> canonical converter -> content audit).
# No alignment payload is injected, no physical FD point is constructed, no
# residual-based selection is applied, and the chi-square baseline is not run.
# Every job writes job_provenance.json recording the input provenance, the
# exact command, the environment/Calypso revision, the git SHA, the exit
# status, and the output ROOT paths.  A failed job still writes provenance
# and exits nonzero so it is explicitly classified, never silently skipped.

set -uo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 INPUT_XAOD SOURCE_ID OUTPUT_DIR PROJECT_ROOT" >&2
  exit 2
fi

input_xaod="$1"
source_id="$2"
output_dir="$3"
project_root="$4"

job_start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
hostname_fqdn="$(hostname -f 2>/dev/null || hostname)"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$input_xaod" ]]; then
  echo "Input xAOD is not a file: $input_xaod" >&2
  exit 2
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
enhanced_root="$output_dir/enhanced_tracklets.root"
canonical_root="$output_dir/tracklets.root"
content_audit="$output_dir/content_audit.json"
provenance_json="$output_dir/job_provenance.json"
for output in "$enhanced_root" "$canonical_root" "$content_audit" "$provenance_json"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 2
  fi
done

git_head="$(git -C "$project_root" rev-parse HEAD 2>/dev/null || echo unknown)"
calypso_root="${CALYPSO_ROOT:-$(cd "$project_root/.." && pwd)/calypso}"

export_step="not_started"
export_exit=99
convert_exit=99
audit_exit=99
schema_exit=99
calypso_head="unknown"
faser_ntuple_maker_path="unknown"

# --- Step 1: Calypso residual-blind tracklet export -------------------------
set +u
source "$project_root/scripts/setup_environment.sh" calypso
setup_status=$?
set -u
if [[ "$setup_status" -ne 0 ]]; then
  export_step="calypso_setup_failed"
  export_exit="$setup_status"
else
  calypso_head="$(git -C "$calypso_root" rev-parse HEAD 2>/dev/null || echo unknown)"
  faser_ntuple_maker_path="$(command -v faser_ntuple_maker.py || echo unknown)"
  export_step="calypso_export"
  faser_ntuple_maker.py "$input_xaod" \
    --isMC --useIFT --export-tracklets --nevents -1 \
    --outfile "$enhanced_root"
  export_exit=$?
fi

# --- Step 2: canonical conversion + schema + content audit ------------------
if [[ "$export_exit" -eq 0 ]]; then
  set +u
  source "$project_root/scripts/setup_environment.sh" ml
  setup_status=$?
  set -u
  if [[ "$setup_status" -ne 0 ]]; then
    export_step="ml_setup_failed"
    convert_exit="$setup_status"
  else
    cd "$project_root"
    export_step="convert"
    python -m scripts.convert_ntuple_tracklets "$enhanced_root" \
      --output "$canonical_root" --include-truth
    convert_exit=$?
    if [[ "$convert_exit" -eq 0 ]]; then
      export_step="schema_check"
      python -m scripts.inspect_root_schema "$canonical_root"
      schema_exit=$?
      export_step="content_audit"
      # Merged MC24 rec files reuse generator-job event numbers; the audit
      # must group physical events in file order, not sorted (run, event).
      python -m scripts.audit_tracklets "$canonical_root" \
        --output "$content_audit" --physical-order
      audit_exit=$?
    fi
  fi
fi

job_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$export_exit" -eq 0 && "$convert_exit" -eq 0 && "$schema_exit" -eq 0 && "$audit_exit" -eq 0 ]]; then
  chain_exit=0
  chain_status="ok"
else
  chain_exit=1
  chain_status="failed_at_${export_step}"
fi

# --- Provenance record ------------------------------------------------------
python3 - "$provenance_json" \
    "$source_id" "$input_xaod" "$output_dir" \
    "$enhanced_root" "$canonical_root" "$content_audit" \
    "$project_root" "$git_head" "$calypso_root" "$calypso_head" \
    "$faser_ntuple_maker_path" "$hostname_fqdn" \
    "$job_start_utc" "$job_end_utc" \
    "$export_exit" "$convert_exit" "$schema_exit" "$audit_exit" \
    "$export_step" "$chain_exit" "$chain_status" <<'EOF'
import json
import os
import sys

(
    out_path, source_id, input_xaod, output_dir,
    enhanced_root, canonical_root, content_audit,
    project_root, git_head, calypso_root, calypso_head,
    maker_path, hostname_fqdn, job_start_utc, job_end_utc,
    export_exit, convert_exit, schema_exit, audit_exit,
    failed_step, chain_exit, chain_status,
) = sys.argv[1:23]
payload = {
    "schema_version": "faser-residual-blind-tracklet-export-job-v1",
    "residual_blind": True,
    "source_id": source_id,
    "input_xaod": input_xaod,
    "output_dir": output_dir,
    "outputs": {
        "enhanced_tracklets": enhanced_root,
        "tracklets": canonical_root,
        "content_audit": content_audit,
    },
    "command": (
        "faser_ntuple_maker.py INPUT --isMC --useIFT --export-tracklets "
        "--nevents -1 --outfile enhanced_tracklets.root && "
        "python -m scripts.convert_ntuple_tracklets --include-truth && "
        "python -m scripts.inspect_root_schema && "
        "python -m scripts.audit_tracklets --physical-order"
    ),
    "project_root": project_root,
    "git_head": git_head,
    "calypso_root": calypso_root,
    "calypso_head": calypso_head,
    "faser_ntuple_maker": maker_path,
    "hostname": hostname_fqdn,
    "job_start_utc": job_start_utc,
    "job_end_utc": job_end_utc,
    "condor_cluster": os.environ.get("CONDOR_CLUSTER_ID", ""),
    "condor_proc": os.environ.get("CONDOR_PROCESS", ""),
    "step_exit_codes": {
        "calypso_export": int(export_exit),
        "convert": int(convert_exit),
        "schema_check": int(schema_exit),
        "content_audit": int(audit_exit),
    },
    "failed_step": failed_step if chain_status != "ok" else "",
    "exit_status": int(chain_exit),
    "status": chain_status,
    "alignment_payload_injected": False,
    "physical_fd_point_constructed": False,
    "residual_based_selection_applied": False,
    "chi2_baseline_run": False,
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
EOF

echo "Residual-blind export job finished: source=${source_id} status=${chain_status}"
exit "$chain_exit"
