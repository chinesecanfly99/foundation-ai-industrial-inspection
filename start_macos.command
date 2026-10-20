#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'First run: bash setup_macos.sh'
  exit 1
fi
exec .venv/bin/python -m app.app --host 127.0.0.1 --port 7860
