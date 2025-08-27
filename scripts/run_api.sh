#!/usr/bin/env bash
set -euo pipefail
source ~/venvs/fashion3d/bin/activate
cd ~/fashion3d
exec python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
