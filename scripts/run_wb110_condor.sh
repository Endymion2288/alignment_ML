#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
source scripts/setup_environment.sh calypso
set +e
python scripts/wb110_contract.py run --output-root "$2"
worker_code=$?
set -e
python - "$2" "$worker_code" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
with (Path(sys.argv[1])/'worker_exit.json').open('x') as f:
    json.dump({'exit_code':int(sys.argv[2]),'completed_utc':datetime.now(timezone.utc).isoformat()},f)
PY
exit "$worker_code"
