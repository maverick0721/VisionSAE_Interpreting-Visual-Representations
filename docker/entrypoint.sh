#!/usr/bin/env bash
set -euo pipefail

cd /app
python -m scripts.run_end_to_end "$@"
