#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
source scripts/setup_environment.sh calypso
python scripts/wb93_contract.py run --output-root "$2"
