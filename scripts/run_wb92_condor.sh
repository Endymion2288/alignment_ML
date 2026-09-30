#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
source scripts/setup_environment.sh calypso
python scripts/wb92_contract.py "$3" --output-root "$2" --index "$4" --axis "$5" --multiplier "$6" --build-root "$7"
