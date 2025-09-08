#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cliente Raspberry para KeyPoints (Keypoint R-CNN ResNet50-FPN, torchvision, CPU).
# - Conecta a servidor Flower para recibir configuración (no entrena).
# - Si no hay servidor, funciona en modo local con defaults.
# - Fuente: webcam (robusto en /dev/video*).
# - CSV por frame con cabecera fija:
#   timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch

import flwr as fl

from FlowerAI_KeyPoints.utils.metrics import open_metrics_csv, write_metrics_row
from FlowerAI_KeyPoints.utils.draw import draw_boxes, draw_keypoints_and_skeleton

METHOD_NAME = "KeyPoints-resnet50"


# Utilidades locales de cámara (evitamos dependencias cruzadas)
def list_video_devices_linux() -> list:
    if os.name != "posix":
        return []
    import glob
    return sorted(glob.glob("/dev/video*"))

def try_open_camera(index: int, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[Optional[cv2.VideoCapture], int]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, -1
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, index

def open_webcam_with_fallback(preferred_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[cv2.VideoCapture, int]:
    indices = [preferred_index] + [i for i in range(0, 4) if i != preferred_index]
    for idx in indices:
        cap, opened_idx = try_open_camera(idx, width=width, height=height)
        if cap is not None:
            return cap, opened_idx
    raise RuntimeError("No se pudo abrir ninguna cámara en índices 0..3")


@dataclass
class ControlConfig:
    conf_thr: float = 0.5
    input_size: int = 640
    max_frames: int = 0
    draw: int = 0  # 0/1
    server_round: int = 0

class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cfg = ControlConfig()

    def update_from_dict(self, d: Dict) -> None:
        with self.lock:
            self.cfg.conf_thr = float(d.get("conf_thr", self.cfg.conf_thr))
            self.cfg.input_size = int(d.get("input_size", self.cfg.input_size))
            self.cfg.max_frames = int(d.get("max_frames", self.cfg.max_frames))
            self.cfg.draw = int(d.get("draw", self.cfg.draw))
            self.cfg.server_round = int(d.get("server_round", self.cfg.server_round))

    def snapshot(self) -> ControlConfig:
        with self.lock:
            return ControlConfig(
                conf_thr=self.cfg.conf_thr,
                input_size=self.cfg.input_size,
                max_frames=self.cfg.max_frames,
                draw=self.cfg.draw,
                server_round=self.cfg.server_round,
            )

class ControlPlaneClient(fl.client.NumPyClient):
    def __init__(self, shared: SharedState) -> None:
        super().__init__()
        self.shared = shared

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        self.shared.update_from_dict(config or {})
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

def flower_thread_fn(server_addr: str, shared: SharedState) -> None:
    while True:
        try:
            print(f"[Flower][Client] Conectando a {server_addr}...")
            fl.client.start_numpy_client(server_address=server_addr, client=ControlPlaneClient(shared))
        except Exception as e:
            print(f"[Flower][Client] No se pudo conectar o conexión cerrada: {e}")
            time.sleep(5.0)
        time.sleep(1.0)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cliente Raspberry - KeyPoints (Keypoint R-CNN ResNet50-FPN, CPU)")
    p.add_argument("--server", type=str, default=None, help="Dirección host:puerto del servidor Flower (opcional)")
    p.add_argument("--cam-index", type=int, default=0, help="Índice de cámara (webcam)")
    p.add_argument("--duration", type=int, default=60, help="Duración mínima en segundos (por defecto 60)")
    p.add_argument("--metrics-dir", type=str, default="Metrics", help="Directorio de salida para CSV")
    p.add_argument("--weights", type=str, default=None, help="Ruta local a checkpoint del modelo (fallback si no hay red)")
    p.add_argument("--draw", type=int, choices=[0, 1], default=None, help="Forzar dibujado local (override); por defecto usa config del servidor")
    return p


def build_model(weights_path: Optional[str] = None):
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    try:
        # Intentar con pesos por defecto (requiere descarga/caché disponible)
        model = keypointrcnn_resnet50_fpn(weights="KeypointRCNN_ResNet50_FPN_Weights.DEFAULT")
    except Exception:
        try:
            # Compatibilidad con versiones antiguas
            model = keypointrcnn_resnet50_fpn(pretrained=True)
        except Exception as e:
            if weights_path is None:
                raise RuntimeError(f"No se pudieron cargar pesos por defecto y no se proporcionó --weights: {e}")
            # Crear modelo sin pesos y cargar estado desde ruta local
            model = keypointrcnn_resnet50_fpn(weights=None, pretrained=False)
    # Cargar pesos locales si se proporcionan
    if weights_path:
        print(f"[Model] Cargando pesos locales desde: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        # Permitir que el checkpoint sea 'state_dict' o dict plano
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_bgr_to_tensor(frame_bgr: np.ndarray, input_short_side: int) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray, float]:
    """
    Convierte BGR->RGB, reescala manteniendo aspecto de forma que lado corto = input_short_side,
    y retorna:
      - tensor (C,H,W) float32 [0..1]
      - tamaño (H,W) de la imagen redimensionada
      - frame_rgb redimensionado (para posible visualización)
      - escala aplicada respecto a tamaño original (para depuración)
    """
    h0, w0 = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Calcular nuevo tamaño manteniendo aspecto
    if min(h0, w0) != input_short_side:
        scale = input_short_side / float(min(h0, w0))
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_rgb
        scale = 1.0
    img = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)
    return tensor, (resized.shape[0], resized.shape[1]), resized, scale


def postprocess_and_filter(output: Dict, conf_thr: float) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Filtra por clase 'person' (label==1) y conf >= conf_thr.
    Retorna:
      - boxes (x1,y1,x2,y2)
      - keypoints list (N x (17,3)) con (x,y,score)
    """
    boxes_out: List[Tuple[int, int, int, int]] = []
    kps_out: List[np.ndarray] = []

    labels = output.get("labels", torch.empty(0))
    scores = output.get("scores", torch.empty(0))
    boxes = output.get("boxes", torch.empty(0))
    keypoints = output.get("keypoints", torch.empty(0))

    if len(scores) == 0:
        return boxes_out, kps_out

    for i in range(len(scores)):
        lbl = int(labels[i].item()) if labels is not None and len(labels) > i else -1
        sc = float(scores[i].item())
        if lbl == 1 and sc >= conf_thr:
            x1, y1, x2, y2 = boxes[i].tolist()
            boxes_out.append((int(x1), int(y1), int(x2), int(y2)))
            if keypoints is not None and len(keypoints) > i:
                kps = keypoints[i].detach().cpu().numpy()  # (17,3)
                kps_out.append(kps)
    return boxes_out, kps_out


def main() -> None:
    args = build_argparser().parse_args()

    shared = SharedState()

    # Hilo de Flower (no bloqueante)
    if args.server:
        t = threading.Thread(target=flower_thread_fn, args=(args.server, shared), daemon=True)
        t.start()
    else:
        print("[Flower][Client] Sin servidor: ejecutando con valores por defecto.")

    # Cargar modelo
    print("[Model] Cargando Keypoint R-CNN (CPU)...")
    try:
        model = build_model(weights_path=args.weights)
    except Exception as e:
        print(f"[Model] Error al cargar modelo: {e}")
        sys.exit(1)
    device = torch.device("cpu")
    model.to(device)

    # Abrir webcam
    try:
        devs = list_video_devices_linux()
        if devs:
            print("[Cam] Dispositivos detectados:", ", ".join(devs))
        cap, used_idx = open_webcam_with_fallback(args.cam_index)
        print(f"[Cam] Webcam abierta en índice {used_idx}")
        source_desc = "webcam"
    except Exception as e:
        print(f"[Cam] Error abriendo webcam: {e}")
        sys.exit(1)

    # CSV de métricas
    start_ts = datetime.now()
    ts_prefix = start_ts.strftime("%Y%m%d_%H%M%S")
    csv_name = f"{ts_prefix}_{METHOD_NAME}_webcam.csv"
    f_csv = open_metrics_csv(args.metrics_dir, csv_name)
    print(f"[CSV] Escribiendo métricas en {os.path.join(args.metrics_dir, csv_name)}")

    proc = psutil.Process(os.getpid())
    frame_idx = 0
    t0 = perf_counter()
    last_log = t0

    try:
        end_time = time.time() + int(args.duration)
        while True:
            if time.time() >= end_time:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            t_start = perf_counter()

            # Config actual (plano de control)
            cfg_now = shared.snapshot()
            conf_thr = cfg_now.conf_thr
            input_size = cfg_now.input_size
            draw_flag = cfg_now.draw if args.draw is None else int(args.draw)

            # Preprocess
            img_tensor, (rh, rw), img_rgb_resized, scale = preprocess_bgr_to_tensor(frame_bgr, input_size)
            inputs = [img_tensor.to(device)]

            # Inferencia (CPU)
            with torch.no_grad():
                outputs = model(inputs)
            out = outputs[0]

            # Postproceso: filtrar personas
            boxes, keypoints = postprocess_and_filter(out, conf_thr)
            detections = len(boxes)

            # Visualización opcional (no bloquear CSV)
            if draw_flag == 1:
                try:
                    vis = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)
                    vis = draw_boxes(vis, boxes)
                    if keypoints:
                        vis = draw_keypoints_and_skeleton(vis, keypoints, kp_thresh=conf_thr)
                    cv2.imshow("KeyPoints - Keypoint R-CNN (person)", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except Exception:
                    pass

            # Métricas
            t_end = perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            fps_inst = 1000.0 / max(latency_ms, 1e-6)
            # Suma por núcleo
            cpu_list = psutil.cpu_percent(percpu=True)
            cpu_pct = float(sum(cpu_list))
            ram_mb = proc.memory_info().rss / (1024 * 1024)
            timestamp_iso = datetime.now().isoformat(timespec="milliseconds")

            write_metrics_row(f_csv, {
                "timestamp": timestamp_iso,
                "method": METHOD_NAME,
                "source": source_desc,
                "frame_idx": frame_idx,
                "latency_ms": round(latency_ms, 3),
                "fps_inst": round(fps_inst, 3),
                "cpu_pct": round(cpu_pct, 2),
                "ram_mb": round(ram_mb, 2),
                "detections": int(detections),
            })

            frame_idx += 1

            # Límite de frames (si publicado por servidor)
            if cfg_now.max_frames > 0 and frame_idx >= cfg_now.max_frames:
                print("[Loop] max_frames alcanzado, saliendo.")
                break

            # Log periódico
            now = perf_counter()
            if now - last_log >= 2.0 and frame_idx > 0:
                elapsed = now - t0
                fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"[Loop] frames={frame_idx} fps_avg={fps_avg:.2f} cpu_sum={cpu_pct:.1f}% det={detections} (round={cfg_now.server_round} sz={input_size})")
                last_log = now

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupción por usuario.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            f_csv.close()
        except Exception:
            pass

    print("[Fin] Cliente terminado.")


if __name__ == "__main__":
    main()