from __future__ import annotations
import os, time, csv
from datetime import datetime
from pathlib import Path

import cv2
import psutil

# ----------------- Configuración -----------------
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_W = int(os.getenv("CAM_W", "640"))
CAM_H = int(os.getenv("CAM_H", "480"))
CAM_FPS = int(os.getenv("CAM_FPS", "30"))
SHOW = os.getenv("SHOW", "0") == "1"

METRICS_SECONDS = int(os.getenv("METRICS_SECONDS", "60"))
METRICS_INTERVAL = float(os.getenv("METRICS_INTERVAL", "1.0"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", "Metrics"))
csv_env = os.getenv("METRICS_CSV", "")

if csv_env:
    METRICS_CSV = Path(csv_env)
else:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_CSV = METRICS_DIR / f"bb_rpi_metrics_{stamp}.csv"

# ----------------- Inicialización -----------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

proc = psutil.Process(os.getpid())
proc.cpu_percent(None)  # prime

def write_header(writer):
    writer.writerow(["timestamp_iso","elapsed_s","fps","cpu_percent","rss_mb","detections"])

def main() -> None:
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (VideoCapture)")

    t0 = time.time()
    last_metrics = t0
    frames = 0

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        write_header(w)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detección (HOG+SVM)
            rects, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
            det_n = len(rects)

            # FPS por ventana METRICS_INTERVAL
            frames += 1
            now = time.time()
            elapsed = now - t0
            if now - last_metrics >= METRICS_INTERVAL:
                window = now - last_metrics
                fps = frames / window if window > 0 else 0.0
                cpu = proc.cpu_percent(interval=None)
                rss_mb = proc.memory_info().rss / (1024*1024)
                w.writerow([datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                            round(elapsed,2), round(fps,2), round(cpu,2), round(rss_mb,2), det_n])
                f.flush()
                last_metrics = now
                frames = 0

            if SHOW:
                for (x,y,wid,ht) in rects:
                    cv2.rectangle(frame, (x,y), (x+wid,y+ht), (0,255,0), 2)
                cv2.imshow("BB - HOG (q para salir)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if elapsed >= METRICS_SECONDS:
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()
    print(f"[BB][RPi] Métricas guardadas en: {METRICS_CSV}")

if __name__ == "__main__":
    main()