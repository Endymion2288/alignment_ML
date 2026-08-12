#!/usr/bin/env bash
# Install the Python package into the LCG_110_cuda user site once per account.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ml_root="$(cd "$script_dir/.." && pwd)"

# atlasLocalSetup is sensitive to shell options. The ML setup only loads LCG,
# but use the same defensive pattern as the export scripts.
set +eu
source "$script_dir/setup_environment.sh" ml
setup_status=$?
if [[ "$setup_status" -ne 0 ]]; then
  exit "$setup_status"
fi
set -e

python -m pip install --user -e "$ml_root"
python - <<'PY'
import awkward
import torch
import uproot

print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
print(f"awkward {awkward.__version__}; uproot {uproot.__version__}")
PY
