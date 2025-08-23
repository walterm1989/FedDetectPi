from __future__ import annotations
import os, time, csv, threading
from datetime import datetime
from pathlib import Path

import psutil
import numpy as np
import flwr as fl
import cv2  # cámara OBLIGATORIA

# ---------------- Configuración ----------------
SERVER_ADDRESS_DEFAULT = "127.0.0.1:8080"

# Métricas
METRICS_SECONDS = int(os.getenv("METRICS_SECONDS", "60"))
METRICS_INTERVAL = float(os.getenv("METRICS_INTERVAL", "1.0"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", "Metrics"))

csv_env = os.getenv("METRICS_CSV", "")
if csv_env:
    METRICS_CSV = Path(csv_env)
else:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_CSV = METRICS_DIR / f"flower_rpi_metrics_{stamp}.csv"

# Cámara
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_W = int(os.getenv("CAM_W", "640"))
CAM_H = int(os.getenv("CAM_H", "480"))
CAM_FPS_TARGET = int(os.getenv("CAM_FPS", "30"))
SHOW = os.getenv("SHOW", "0") == "1"

# Parámetros de HOG (ajustables por entorno)
HOG_HIT_THRESHOLD = float(os.getenv("HOG_HIT_THRESHOLD", "0.0"))
HOG_WIN_STRIDE = tuple(map(int, os.getenv("HOG_WIN_STRIDE", "8,8").split(",")))
HOG_PADDING = tuple(map(int, os.getenv("HOG_PADDING", "8,8").split(",")))
HOG_SCALE = float(os.getenv("HOG_SCALE", "1.05"))
HOG_FINAL_THRESHOLD = float(os.getenv("HOG_FINAL_THRESHOLD", "2.0"))
HOG_UPSCALE = float(os.getenv("HOG_UPSCALE", "1.0"))  # 1.0 = sin upsample

# Contadores federados
FIT_CALLS = 0
EVAL_CALLS = 0

# Métricas compartidas de cámara
_cam_fps = 0.0
_cam_det = 0
_cam_lock = threading.Lock()

class PassiveClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.zeros(1, dtype=np.float32)]

    def fit(self, parameters, config):
        global FIT_CALLS
        FIT_CALLS += 1
        print(f"[FlowerAI][RPi] Fit -> params: {len(parameters)} | sin entrenamiento local | fit_calls={FIT_CALLS}")
        return parameters, 1, {}  # 1 ejemplo para evitar div/0

    def evaluate(self, parameters, config):
        global EVAL_CALLS
        EVAL_CALLS += 1
        print(f"[FlowerAI][RPi] Evaluate -> sin datos locales | eval_calls={EVAL_CALLS}")
        return 0.0, 1, {}  # 1 ejemplo para evitar div/0

def _camera_worker(stop_event: threading.Event) -> None:
    """Captura cámara y ejecuta HOG+SVM, actualizando fps/detecciones por segundo."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_TARGET)

    if not cap.isOpened():
        raise RuntimeError("[FlowerAI][RPi] No se pudo abrir la cámara (VideoCapture).")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    last = time.time()
    frames = 0
    det_n = 0
    fail = 0  # Contador de fallos consecutivos

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                fail += 1
                if fail >= 30:
                    cap.release()
                    time.sleep(0.1)
                    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(CAM_INDEX)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_TARGET)
                    fail = 0
                time.sleep(0.02)
                continue
            else:
                fail = 0

            # Procesamiento de upscaling si corresponde
            frame_proc = frame
            if HOG_UPSCALE != 1.0:
                frame_proc = cv2.resize(
                    frame,
                    None,
                    fx=HOG_UPSCALE,
                    fy=HOG_UPSCALE,
                    interpolation=cv2.INTER_LINEAR
                )

            rects, _ = hog.detectMultiScale(
                frame_proc,
                hitThreshold=HOG_HIT_THRESHOLD,
                winStride=HOG_WIN_STRIDE,
                padding=HOG_PADDING,
                scale=HOG_SCALE,
                finalThreshold=HOG_FINAL_THRESHOLD,
            )
            det_n = len(rects)
            frames += 1

            now = time.time()
            if now - last >= 1.0:
                fps = frames / (now - last) if now > last else 0.0
                with _cam_lock:
                    global _cam_fps, _cam_det
                    _cam_fps = fps
                    _cam_det = det_n
                last = now
                frames = 0

            if SHOW:
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame_proc, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("FlowerAI Camera (q para salir)", frame_proc)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    finally:
        cap.release()
        if SHOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

def _metrics_sampler(stop_event: threading.Event) -> None:
    """CPU/RAM + FPS/detecciones -> CSV cada METRICS_INTERVAL durante METRICS_SECONDS."""
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(None)

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_iso","elapsed_s","cpu_percent","rss_mb",
            "fit_calls","eval_calls","cam_fps","cam_det"
        ])
        t0 = time.time()
        while not stop_event.is_set():
            now = time.time()
            elapsed = now - t0
            if elapsed > METRICS_SECONDS:
                break
            cpu = proc.cpu_percent(interval=None)
            rss_mb = proc.memory_info().rss / (1024*1024)
            with _cam_lock:
                cam_fps = _cam_fps
                cam_det = _cam_det
            writer.writerow([
                datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                round(elapsed,2), round(cpu,2), round(rss_mb,2),
                FIT_CALLS, EVAL_CALLS, round(cam_fps,2), int(cam_det)
            ])
            f.flush()
            time.sleep(METRICS_INTERVAL)

def main() -> None:
    addr = os.getenv("FLOWER_SERVER_ADDRESS", SERVER_ADDRESS_DEFAULT)
    print(f"[FlowerAI][RPi] Conectando con servidor en {addr}")

    stop_event = threading.Event()

    # Hilo de métricas (60 s por defecto)
    th_metrics = threading.Thread(target=_metrics_sampler, args=(stop_event,), daemon=True)
    th_metrics.start()

    # Hilo de cámara (OBLIGATORIO)
    th_cam = threading.Thread(target=_camera_worker, args=(stop_event,), daemon=True)
    th_cam.start()

    try:
        fl.client.start_client(server_address=addr, client=PassiveClient().to_client())
    finally:
        stop_event.set()
        th_cam.join(timeout=2.0)
        th_metrics.join(timeout=2.0)
        print(f"[FlowerAI][RPi] Métricas guardadas en: {METRICS_CSV}")

if __name__ == "__main__":
    main()