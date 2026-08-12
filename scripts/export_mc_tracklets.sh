#!/usr/bin/env bash
# Export MC local tracklets with Calypso, convert them, and run the V1 baseline.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_mc_tracklets.sh --input X_AOD.root --output-dir DIRECTORY [options]

Required:
  --input PATH                 MC xAOD input file
  --output-dir DIRECTORY       New or empty experiment directory

Options:
  --nevents N                  Number of events to export (default: 10)
  --baseline-config PATH       chi-square YAML config
  --source-station ID          Override baseline source station ID
  --target-station ID          Override baseline target station ID
  --chi2-gate VALUE            Override baseline candidate chi-square gate
  --geometry NAME              Optional faser_ntuple_maker geometry selector
  -h, --help                   Show this help

The script requires a completed Calypso installation in calypso/run.  It
creates enhanced_tracklets.root, tracklets.root, content_audit.json, and
chi2_baseline/ below the requested output directory. Existing output files
are never overwritten.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ml_root="$(cd "$script_dir/.." && pwd)"
setup_script="$script_dir/setup_environment.sh"

input=""
output_dir=""
nevents=10
baseline_config="$ml_root/configs/baseline_chi2.yaml"
geometry=""
source_station=""
target_station=""
chi2_gate=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --nevents)
      nevents="$2"
      shift 2
      ;;
    --baseline-config)
      baseline_config="$2"
      shift 2
      ;;
    --source-station)
      source_station="$2"
      shift 2
      ;;
    --target-station)
      target_station="$2"
      shift 2
      ;;
    --chi2-gate)
      chi2_gate="$2"
      shift 2
      ;;
    --geometry)
      geometry="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$input" || -z "$output_dir" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$input" ]]; then
  echo "MC xAOD input is not a file: $input" >&2
  exit 2
fi
if [[ ! "$nevents" =~ ^[0-9]+$ ]] || [[ "$nevents" -eq 0 ]]; then
  echo "--nevents must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$baseline_config" ]]; then
  echo "Baseline config is not a file: $baseline_config" >&2
  exit 2
fi
if [[ -n "$source_station" && ! "$source_station" =~ ^-?[0-9]+$ ]]; then
  echo "--source-station must be an integer" >&2
  exit 2
fi
if [[ -n "$target_station" && ! "$target_station" =~ ^-?[0-9]+$ ]]; then
  echo "--target-station must be an integer" >&2
  exit 2
fi
if [[ -n "$source_station" && -z "$target_station" ]] || \
   [[ -z "$source_station" && -n "$target_station" ]]; then
  echo "--source-station and --target-station must be supplied together" >&2
  exit 2
fi

output_dir="$(mkdir -p "$output_dir" && cd "$output_dir" && pwd)"
enhanced_root="$output_dir/enhanced_tracklets.root"
canonical_root="$output_dir/tracklets.root"
content_audit="$output_dir/content_audit.json"
baseline_dir="$output_dir/chi2_baseline"
for output in "$enhanced_root" "$canonical_root" "$content_audit"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 2
  fi
done
if [[ -e "$baseline_dir" ]]; then
  echo "Refusing to overwrite existing baseline directory: $baseline_dir" >&2
  exit 2
fi

geometry_args=()
if [[ -n "$geometry" ]]; then
  geometry_args=(--geom "$geometry")
fi
baseline_args=()
if [[ -n "$source_station" ]]; then
  baseline_args+=(--source-station "$source_station" --target-station "$target_station")
fi
if [[ -n "$chi2_gate" ]]; then
  baseline_args+=(--chi2-gate "$chi2_gate")
fi

(
  # atlasLocalSetup accesses optional interactive-shell variables. Do not
  # inherit nounset/errexit while sourcing it; restore errexit for the job.
  set +eu
  source "$setup_script" calypso
  setup_status=$?
  if [[ "$setup_status" -ne 0 ]]; then
    exit "$setup_status"
  fi
  set -e
  faser_ntuple_maker.py "$input" \
    --isMC --useIFT --export-tracklets --nevents "$nevents" \
    --outfile "$enhanced_root" "${geometry_args[@]}"
)

(
  set +eu
  source "$setup_script" ml
  setup_status=$?
  if [[ "$setup_status" -ne 0 ]]; then
    exit "$setup_status"
  fi
  set -e
  cd "$ml_root"
  python -m scripts.convert_ntuple_tracklets "$enhanced_root" \
    --output "$canonical_root" --include-truth
  python -m scripts.inspect_root_schema "$canonical_root"
  python -m scripts.audit_tracklets "$canonical_root" \
    --output "$content_audit"
  python -m scripts.run_chi2_baseline \
    --input "$canonical_root" --config "$baseline_config" \
    --output-dir "$baseline_dir" "${baseline_args[@]}"
)

printf 'Enhanced Calypso output: %s\n' "$enhanced_root"
printf 'Canonical tracklets: %s\n' "$canonical_root"
printf 'Baseline artifacts: %s\n' "$baseline_dir"
