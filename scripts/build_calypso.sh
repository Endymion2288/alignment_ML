#!/usr/bin/env bash
# Build Calypso into calypso/run. Execute this script from any directory.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_dir/../.." && pwd)"
calypso_root="$workspace/calypso"
build_dir="$calypso_root/build"
jobs="${FASER_BUILD_JOBS:-8}"

if [[ ! -f "$calypso_root/faser-common/EventFormats/EventFormats/DAQFormats.hpp" ]]; then
  # The parent repository uses an SSH remote, but lxplus batch shells may not
  # have an SSH key.  The public submodule can be fetched reproducibly over HTTPS.
  git -C "$calypso_root" config submodule.faser-common.url \
    https://gitlab.cern.ch/faser/faser-common.git
  git -C "$calypso_root" submodule update --init --recursive
fi

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source "$ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh"
if command -v amsg >/dev/null 2>&1; then
  amsg seen >/dev/null 2>&1 || true
fi
asetup --input="$calypso_root/asetup.faser" Athena,24.0.41

# atlasLocalSetup must be sourced before enabling errexit; it explicitly rejects
# shells that already have -e enabled.
set -eo pipefail

mkdir -p "$build_dir"
cmake -S "$calypso_root" -B "$build_dir" -DCMAKE_INSTALL_PREFIX="$calypso_root/run"
cmake --build "$build_dir" --target install --parallel "$jobs"
bash "$script_dir/refresh_calypso_confdb.sh"
