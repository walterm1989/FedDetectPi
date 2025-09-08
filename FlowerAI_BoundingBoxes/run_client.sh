#!/usr/bin/env bash
set -euo pipefail

# Edita la IP del portátil
SERVER_IP_PORT="<IP_PORTATIL>:8080"

python FlowerAI_BoundingBoxes/client_raspberry.py \
  --server "${SERVER_IP_PORT}" \
  --source webcam \
  --cam-index 0 \
  --duration 60 \
  --out-dir ./FlowerAI_BoundingBoxes/Metrics