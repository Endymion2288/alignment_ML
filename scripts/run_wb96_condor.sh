#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
wb96_out="$2"
mkdir "$wb96_out/execution_lock" || exit 70
wb96_exit_receipt() {
  wb96_exit_code=$?
  (set -o noclobber; printf '{"exit_code":%d,"completed_utc":"%s","worker":"%s"}\n' "$wb96_exit_code" "$(date -u +%FT%TZ)" "$(hostname)" > "$wb96_out/worker_exit.json")
}
trap wb96_exit_receipt EXIT
source scripts/setup_environment.sh calypso
python scripts/wb96_contract.py run --output-root "$wb96_out"
