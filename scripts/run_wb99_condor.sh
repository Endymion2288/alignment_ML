#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb99_out="$2"
mkdir "$wb99_out/execution_lock" || exit 70
wb99_exit_receipt() {
  wb99_exit_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb99_exit_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb99_out/worker_exit.json")
}
trap wb99_exit_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb99_contract.py run --output-root "$wb99_out"
