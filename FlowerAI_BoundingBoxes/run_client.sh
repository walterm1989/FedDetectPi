#!/usr/bin/env bash
set -euo pipefail

# Ejemplos de uso:
# Webcam:
# ./run_client.sh webcam 0 "" 60 ./FlowerAI_BoundingBoxes/Metrics
# Vídeo:
# ./run_client.sh video -1 ./samples/people.mp4 60 ./FlowerAI_BoundingBoxes/Metrics

SERVER_ADDR="${SERVER_ADDR:-127.0.0.1:8080}"
SOURCE="${1:-webcam}"            # webcam | video
CAM_INDEX="${2:-0}"              # índice webcam (para webcam)
VIDEO_PATH="${3:-}"              # ruta vídeo (para video)
DURATION="${4:-60}"
OUT_DIR="${5:-./FlowerAI_BoundingBoxes/Metrics}"

if [[ "$SOURCE" == "webcam" ]]; then
  python "$(dirname "$0")/client_raspberry.py" \
    --server "$SERVER_ADDR" \
    --source webcam --cam-index "$CAM_INDEX" \
    --duration "$DURATION" \
    --out-dir "$OUT_DIR"
else
  python "$(dirname "$0")/client_raspberry.py" \
    --server "$SERVER_ADDR" \
    --source video --video-path "$VIDEO_PATH" \
    --duration "$DURATION" \
    --out-dir "$OUT_DIR"
fi