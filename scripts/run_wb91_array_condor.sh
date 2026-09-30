#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
cd "$1"
source scripts/setup_environment.sh calypso
python scripts/wb91_array.py event --output-root "$2" --index "$3"
