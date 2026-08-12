#!/usr/bin/env bash
# Source this script: source alignment_ML/scripts/setup_environment.sh <ml|calypso>

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced, not executed." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  mode="ml"
else
  mode="$1"
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FASER_ML_ROOT="$(cd "$script_dir/.." && pwd)"
export FASER_WORKSPACE="$(cd "$FASER_ML_ROOT/.." && pwd)"
export CALYPSO_ROOT="$FASER_WORKSPACE/calypso"

case "$mode" in
  ml)
    source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
    if [[ -n "${PYTHONPATH:-}" ]]; then
      export PYTHONPATH="$FASER_ML_ROOT:$PYTHONPATH"
    else
      export PYTHONPATH="$FASER_ML_ROOT"
    fi
    export MPLBACKEND=Agg
    echo "Loaded LCG_110_cuda ML environment."
    ;;
  calypso)
    calypso_platform="${CALYPSO_PLATFORM:-x86_64-el9-gcc13-opt}"
    lcg_setup="/cvmfs/sft.cern.ch/lcg/views/LCG_104d_ATLAS_7/$calypso_platform/setup.sh"
    athena_setup="/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/$calypso_platform/setup.sh"
    if [[ -f "$lcg_setup" && -f "$athena_setup" ]]; then
      # This is the exact EL9/GCC13 stack used by the existing Calypso build.
      # It avoids the slow interactive asetup path in batch or GPU-node shells.
      source "$lcg_setup"
      export ATLAS_RELEASEDATA=/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData
      export ATLAS_POOLCOND_PATH=/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current
      source "$athena_setup"
    else
      echo "Direct Calypso setup is unavailable for $calypso_platform; falling back to asetup." >&2
      export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
      source "$ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh"
      # Acknowledge the CERN setup notice in scripted shells when the helper is
      # available. It is harmless after the acknowledgement has been recorded.
      if command -v amsg >/dev/null 2>&1; then
        amsg seen >/dev/null 2>&1 || true
      fi
      asetup --input="$CALYPSO_ROOT/asetup.faser" Athena,24.0.41
    fi
    if [[ ! -f "$CALYPSO_ROOT/run/setup.sh" ]]; then
      echo "Calypso is not built. Run alignment_ML/scripts/build_calypso.sh first." >&2
      return 1
    fi
    source "$CALYPSO_ROOT/run/setup.sh"
    # Keep the workspace NtupleDumper CLI ahead of the installed copy.  This
    # permits Python-only exporter fixes without rebuilding Calypso libraries.
    export PATH="$CALYPSO_ROOT/PhysicsAnalysis/NtupleDumper/scripts:$PATH"
    confdb_dir="$CALYPSO_ROOT/build/x86_64-el9-gcc13-opt/alignment_ml_confdb"
    build_lib_dir="$CALYPSO_ROOT/build/x86_64-el9-gcc13-opt/lib"
    if [[ -d "$build_lib_dir" ]]; then
      # Targeted builds are intentionally used during exporter development.
      # Put their libraries before run/lib so a rebuilt component and its
      # linked implementation are loaded together.
      export LD_LIBRARY_PATH="$build_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [[ -f "$confdb_dir/Calypso.confdb2" ]]; then
      export LD_LIBRARY_PATH="$confdb_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    echo "Loaded Calypso Athena environment."
    ;;
  *)
    echo "Usage: source ${BASH_SOURCE[0]} <ml|calypso>" >&2
    return 2
    ;;
esac
