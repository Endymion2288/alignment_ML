#!/usr/bin/env bash
# Merge the current build's Configurable metadata for workspace-only components.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_dir/../.." && pwd)"
calypso_root="$workspace/calypso"
build_dir="$calypso_root/build"
merge_script="$calypso_root/run/cmake/modules/scripts/mergeConfdb2.py"
merge_inputs="$build_dir/CMakeFiles/CalypsoConfdb2MergeFiles.txt"
output_dir="$build_dir/x86_64-el9-gcc13-opt/alignment_ml_confdb"
output="$output_dir/Calypso.confdb2"

if [[ ! -f "$merge_script" ]]; then
  echo "Calypso confdb merge helper is absent: $merge_script" >&2
  exit 1
fi
if [[ ! -f "$merge_inputs" ]]; then
  echo "Calypso build metadata is absent: $merge_inputs" >&2
  exit 1
fi

# mergeConfdb2 imports Gaudi classes. The build script calls this helper after
# `asetup`; direct use must therefore source the same environment first.
python_executable="${PYTHON:-python}"
if ! command -v "$python_executable" >/dev/null 2>&1; then
  echo "No Athena Python is available. Source setup_environment.sh calypso first." >&2
  exit 1
fi
if ! "$python_executable" -c "import GaudiKernel" >/dev/null 2>&1; then
  echo "GaudiKernel is unavailable. Source setup_environment.sh calypso first." >&2
  exit 1
fi

mkdir -p "$output_dir"
"$python_executable" "$merge_script" "$output" "$merge_inputs"
printf 'Refreshed workspace Configurable metadata: %s\n' "$output"
