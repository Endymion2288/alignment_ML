#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb101_out="$2"
mkdir "$wb101_out/execution_lock" || exit 70
wb101_receipt() {
  wb101_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb101_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb101_out/worker_exit.json")
}
trap wb101_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb101_contract.py run --output-root "$wb101_out"
