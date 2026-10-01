#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb98_out="$2"
mkdir "$wb98_out/execution_lock" || exit 70
wb98_exit_receipt() {
  wb98_exit_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb98_exit_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb98_out/worker_exit.json")
}
trap wb98_exit_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb98_contract.py run --output-root "$wb98_out"
