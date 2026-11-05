#!/usr/bin/env bash
# Full pipeline: build -> replicate -> extend -> interact -> validate
set -euo pipefail
cd "$(dirname "$0")"
python3 build_data.py
python3 replicate.py
python3 extend.py
python3 interact.py
python3 validate.py
