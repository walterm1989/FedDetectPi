#!/usr/bin/env bash
set -euo pipefail

python FlowerAI_BoundingBoxes/server.py \
  --address 0.0.0.0:8080 \
  --rounds 9999 \
  --conf-thr 0.40 \
  --nms-thr 0.45 \
  --input-size 416