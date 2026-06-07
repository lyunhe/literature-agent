from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PORT = os.environ.get("LITERATURE_SHOWCASE_PORT", "8051")


def main() -> int:
    env = dict(os.environ, LITERATURE_SHOWCASE_PORT=PORT)
    print(
        f"Starting paper reproduction showcase at "
        f"http://127.0.0.1:{PORT}/?view=reproduction#repro-nasri_2016_ac_uc_benders"
    )
    return subprocess.call([sys.executable, "literature_showcase/app.py"], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
