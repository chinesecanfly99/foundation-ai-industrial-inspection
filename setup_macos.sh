#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/base.txt -r requirements/torch-macos.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m industrial_inspection prepare-demo
.venv/bin/python -m industrial_inspection check-env
echo 'Setup finished. macOS execution still requires a real-machine acceptance run.'
