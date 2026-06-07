#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LITERATURE_SHOWCASE_PORT:-8051}"

cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

export LITERATURE_SHOWCASE_PORT="$PORT"
echo "Starting paper reproduction showcase at http://127.0.0.1:${PORT}/?view=reproduction#repro-nasri_2016_ac_uc_benders"
python literature_showcase/app.py
