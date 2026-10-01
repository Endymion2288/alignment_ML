#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb97_out="$2"
mkdir "$wb97_out/execution_lock" || exit 70
wb97_exit_receipt() {
  wb97_exit_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb97_exit_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb97_out/worker_exit.json")
}
trap wb97_exit_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb97_contract.py run --output-root "$wb97_out"
