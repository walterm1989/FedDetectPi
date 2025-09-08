#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cliente Raspberry/portátil:
- Conecta (opcional) al servidor Flower para recibir parámetros de configuración por ronda.
- Ejecuta detección de personas mediante OpenCV HOG+SVM (Bounding Boxes).
- Mide métricas por frame y las guarda en CSV con cabecera exacta.
- Si el servidor no está disponible, continúa en modo local con parámetros por defecto.

CSV: timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections
"""

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import psutil

import flwr as fl

from FlowerAI_BoundingBoxes.utils.metrics import MetricsWriter
from FlowerAI_BoundingBoxes.utils.camera import (
    open_webcam_with_fallback,
    open_video_file,
)


METHOD_NAME = "BBoxes-HOG"


# -------------------------
# Config compartida (thread-safe)
# -------------------------
@dataclass
class DetectorConfig:
    threshold: float = 0.5
    win_stride: int = 8
    padding: int = 8
    scale: float = 1.05


class ConfigManager:
    def __init__(self, initial: DetectorConfig) -> None:
        self._cfg = initial
        self._lock = threading.Lock()

    def get(self) -> DetectorConfig:
        with self._lock:
            return DetectorConfig(
                threshold=self._cfg.threshold,
                win_stride=self._cfg.win_stride,
                padding=self._cfg.padding,
                scale=self._cfg.scale,
            )

    def update_from_dict(self, d: dict) -> None:
        with self._lock:
            if "threshold" in d:
                self._cfg.threshold = float(d["threshold"])
            if "win_stride" in d:
                self._cfg.win_stride = int(d["win_stride"])
            if "padding" in d:
                self._cfg.padding = int(d["padding"])
            if "scale" in d:
                self._cfg.scale = float(d["scale"])


# -------------------------
# Cliente Flower "vacío" (solo consume config)
# -------------------------
class ControlClient(fl.client.NumPyClient):
    def __init__(self, cfg_mgr: ConfigManager) -> None:
        self.cfg_mgr = cfg_mgr
        # Sin parámetros reales; este cliente no entrena nada
        self._params: Tuple[np.ndarray, ...] = tuple()

    def get_parameters(self, config):
        return self._params

    def fit(self, parameters, config):
        # Actualizar configuración recibida
        self.cfg_mgr.update_from_dict(config or {})
        # Devolver tal cual
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

    @property
    def parameters(self):
        return self._params


def can_connect(address: str, timeout: float = 2.0) -> bool:
    """Comprueba si se puede abrir un socket TCP al host:puerto."""
    try:
        host, port_s = address.split(":")
        port = int(port_s)
    except Exception:
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def flower_thread(server_addr: str, cfg_mgr: ConfigManager) -> None:
    """Hilo que conecta al servidor Flower y queda a la espera de rondas."""
    try:
        fl.client.start_numpy_client(
            server_address=server_addr,
            client=ControlClient(cfg_mgr),
        )
    except Exception as e:
        print(f"[Flower] Finalizó el hilo de control: {e}")


# -------------------------
# Detección HOG+SVM
# -------------------------
def create_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people(frame, hog, cfg: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta HOG detectMultiScale y devuelve (rects, weights) filtrados por threshold.
    """
    ws = max(1, int(cfg.win_stride))
    pad = max(0, int(cfg.padding))
    scale = float(cfg.scale)

    # detectMultiScale retorna rects, weights
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(ws, ws),
        padding=(pad, pad),
        scale=scale,
        hitThreshold=0.0,
    )

    if weights is None or len(weights) == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32)

    weights = np.array(weights).reshape(-1)
    rects = np.array(rects).reshape(-1, 4)

    mask = weights >= float(cfg.threshold)
    return rects[mask], weights[mask]


def build_source_label(args: argparse.Namespace) -> str:
    if args.source == "webcam":
        return "webcam"
    else:
        # Ruta relativa o nombre
        p = Path(args.video_path).resolve()
        try:
            rel = p.relative_to(Path.cwd())
            return f"video:{str(rel)}"
        except Exception:
            return f"video:{p.name}"


def csv_output_path(start_ts: datetime, args: argparse.Namespace) -> Path:
    ts = start_ts.strftime("%Y%m%d_%H%M%S")
    if args.source == "webcam":
        name = f"{ts}_{METHOD_NAME}_webcam.csv"
    else:
        p = Path(args.video_path)
        name = f"{ts}_{METHOD_NAME}_video-{p.name}.csv"
    return Path(args.out_dir) / name


def draw_bboxes(frame, rects):
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cliente HOG+SVM con control Flower")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="host:puerto del servidor Flower")
    parser.add_argument("--source", type=str, choices=["webcam", "video"], required=True, help="Fuente de vídeo")
    parser.add_argument("--cam-index", type=int, default=0, help="Índice de la webcam (si --source=webcam)")
    parser.add_argument("--video-path", type=str, default="", help="Ruta del archivo de vídeo (si --source=video)")
    parser.add_argument("--duration", type=int, default=60, help="Duración de la captura en segundos (por defecto 60)")
    parser.add_argument("--out-dir", type=str, default="./FlowerAI_BoundingBoxes/Metrics", help="Directorio de salida para CSV")
    parser.add_argument("--show", action="store_true", help="Muestra ventana con bounding boxes (evitar en RPi salvo demo)")

    # Valores por defecto locales (pueden sobreescribirse por Flower)
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de weight para filtrar detecciones")
    parser.add_argument("--win-stride", type=int, default=8, help="winStride para HOG")
    parser.add_argument("--padding", type=int, default=8, help="padding para HOG")
    parser.add_argument("--scale", type=float, default=1.05, help="scale para HOG")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Config inicial (local)
    cfg_mgr = ConfigManager(
        DetectorConfig(
            threshold=args.threshold,
            win_stride=args.win_stride,
            padding=args.padding,
            scale=args.scale,
        )
    )

    # Intentar conexión con servidor Flower (no bloquear si no está disponible)
    flower_ok = can_connect(args.server)
    if flower_ok:
        print(f"[Flower] Conectando a {args.server} en hilo de control...")
        th = threading.Thread(target=flower_thread, args=(args.server, cfg_mgr), daemon=True)
        th.start()
    else:
        print(f"[Flower] Servidor no disponible ({args.server}). Continuando en modo local con valores por defecto.")

    # Preparar captura de vídeo
    if args.source == "webcam":
        cap = open_webcam_with_fallback(index=args.cam_index, try_range=(0, 3))
    else:
        cap = open_video_file(args.video_path)

    if cap is None or not cap.isOpened():
        print("Error: no se pudo abrir la fuente de vídeo. Abortando.")
        return

    source_label = build_source_label(args)
    start_ts = datetime.now()
    out_path = csv_output_path(start_ts, args)
    print(f"Escribiendo métricas en: {out_path}")

    process = psutil.Process()
    hog = create_hog_detector()

    frame_idx = 0
    t0 = time.perf_counter()
    last_log = 0
    running = True

    try:
        with MetricsWriter(out_path) as mw:
            while running:
                if args.duration > 0 and (time.perf_counter() - t0) >= args.duration:
                    break

                t_frame_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Si el vídeo termina, salimos
                    if args.source == "video":
                        break
                    # En webcam, intentar continuar
                    continue

                cfg = cfg_mgr.get()

                rects, weights = detect_people(frame, hog, cfg)
                detections = int(rects.shape[0])

                if args.show:
                    draw_bboxes(frame, rects)
                    cv2.imshow("BBoxes-HOG (HOG+SVM)", frame)
                    # Salir con 'q'
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Métricas
                t_frame_end = time.perf_counter()
                latency_ms = (t_frame_end - t_frame_start) * 1000.0
                fps_inst = 1000.0 / max(1e-6, latency_ms)
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_mb = process.memory_info().rss / (1024 * 1024)
                timestamp_iso = datetime.now().isoformat(timespec="milliseconds")

                # CSV: timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections
                mw.writerow(
                    [
                        timestamp_iso,
                        METHOD_NAME,
                        source_label,
                        frame_idx,
                        round(latency_ms, 3),
                        round(fps_inst, 3),
                        round(cpu_pct, 2),
                        round(ram_mb, 2),
                        detections,
                    ]
                )

                # Log cada ~30 frames
                if frame_idx - last_log >= 30:
                    elapsed = time.perf_counter() - t0
                    fps_mean = (frame_idx + 1) / max(1e-6, elapsed)
                    print(
                        f"[{METHOD_NAME}] frame={frame_idx} | fps_mean={fps_mean:.2f} | "
                        f"cpu={cpu_pct:.1f}% | last_detections={detections} | "
                        f"cfg(th={cfg.threshold}, ws={cfg.win_stride}, pad={cfg.padding}, sc={cfg.scale})"
                    )
                    last_log = frame_idx

                frame_idx += 1

    except KeyboardInterrupt:
        print("Interrumpido por el usuario (Ctrl+C). Cerrando de forma segura...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("Finalizado.")
    

if __name__ == "__main__":
    main()