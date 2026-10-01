#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb100_out="$2"
mkdir "$wb100_out/execution_lock" || exit 70
wb100_receipt() {
  wb100_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb100_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb100_out/worker_exit.json")
}
trap wb100_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb100_contract.py run --output-root "$wb100_out"
