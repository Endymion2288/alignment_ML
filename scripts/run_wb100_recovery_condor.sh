#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb100_recovery_out="$2"
mkdir "$wb100_recovery_out/execution_lock" || exit 70
wb100_recovery_receipt() {
  wb100_recovery_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb100_recovery_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb100_recovery_out/worker_exit.json")
}
trap wb100_recovery_receipt EXIT
source scripts/setup_environment.sh ml
python scripts/wb100_aggregation_recovery.py run --output-root "$wb100_recovery_out"
