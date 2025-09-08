#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/server.py" \
  --address 0.0.0.0:8080 \
  --rounds 9999 \
  --threshold 0.5 \
  --win-stride 8 \
  --padding 8 \
  --scale 1.05