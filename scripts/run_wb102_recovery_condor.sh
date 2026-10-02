#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb102_recovery_out="$2"
mkdir "$wb102_recovery_out/execution_lock" || exit 70
wb102_recovery_receipt() {
  wb102_recovery_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb102_recovery_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb102_recovery_out/worker_exit.json")
}
trap wb102_recovery_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb102_recovery.py run --output-root "$wb102_recovery_out"
