#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cliente Raspberry:
# - Ejecuta detección de personas con YOLOv4-tiny (OpenCV DNN, CPU).
# - Conecta a servidor Flower (si disponible) para recibir parámetros dinámicos.
# - Genera CSV de métricas por frame con cabecera exacta.
#
# Robustez:
# - Si no hay servidor, usa valores por defecto y sigue funcionando.
# - Auto-descarga assets si faltan (cfg, weights, names) a assets/.
# - Webcam: intenta índices 0..3, lista /dev/video* en Linux.

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import psutil

# Flower es opcional: si no está accesible el servidor, seguimos en local
import flwr as fl

# Utilidades locales
from FlowerAI_BoundingBoxes.utils.metrics import open_metrics_csv, write_metrics_row
from FlowerAI_BoundingBoxes.utils.camera import open_webcam_with_fallback, list_video_devices_linux
from FlowerAI_BoundingBoxes.download_assets import download_all as download_yolo_assets


METHOD_NAME = "BBoxes-YOLOv4tiny"


@dataclass
class ControlConfig:
    conf_thr: float = 0.40
    nms_thr: float = 0.45
    input_size: int = 416
    person_class_name: str = "person"
    server_round: int = 0


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cfg = ControlConfig()

    def update_from_dict(self, d: Dict) -> None:
        with self.lock:
            self.cfg.conf_thr = float(d.get("conf_thr", self.cfg.conf_thr))
            self.cfg.nms_thr = float(d.get("nms_thr", self.cfg.nms_thr))
            self.cfg.input_size = int(d.get("input_size", self.cfg.input_size))
            self.cfg.person_class_name = str(d.get("person_class_name", self.cfg.person_class_name))
            self.cfg.server_round = int(d.get("server_round", self.cfg.server_round))

    def snapshot(self) -> ControlConfig:
        with self.lock:
            return ControlConfig(
                conf_thr=self.cfg.conf_thr,
                nms_thr=self.cfg.nms_thr,
                input_size=self.cfg.input_size,
                person_class_name=self.cfg.person_class_name,
                server_round=self.cfg.server_round,
            )


class ControlPlaneClient(fl.client.NumPyClient):
    """Cliente Flower que solo recibe configuración en fit()."""

    def __init__(self, shared: SharedState) -> None:
        super().__init__()
        self.shared = shared

    def get_parameters(self, config):  # noqa: D401
        # No entrenamos, devolvemos params vacíos
        return []

    def fit(self, parameters, config):
        # Recibimos configuración del servidor y la volcamos en el estado compartido.
        # Devolvemos parámetros sin cambios.
        self.shared.update_from_dict(config or {})
        # No entrenamos, devolvemos lo que recibimos
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        # Sin evaluación
        return 0.0, 0, {}


def flower_thread_fn(server_addr: str, shared: SharedState) -> None:
    """Hilo de conexión al servidor Flower. Reintenta automáticamente."""
    while True:
        try:
            print(f"[Flower][Client] Conectando a {server_addr}...")
            fl.client.start_numpy_client(server_address=server_addr, client=ControlPlaneClient(shared))
        except Exception as e:
            print(f"[Flower][Client] No se pudo conectar o conexión cerrada: {e}")
            # Esperar antes de reintentar
            time.sleep(5.0)
        # Si start_numpy_client retorna, volvemos a intentar para seguir recibiendo config
        time.sleep(1.0)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cliente Raspberry - BBoxes YOLOv4-tiny con OpenCV DNN (CPU)")
    p.add_argument("--server", type=str, default=None, help="Dirección IP:puerto del servidor Flower (opcional)")
    p.add_argument("--source", type=str, required=True, choices=["webcam", "video"], help="Fuente de vídeo")
    p.add_argument("--cam-index", type=int, default=0, help="Índice de cámara (webcam)")
    p.add_argument("--video-path", type=str, default="", help="Ruta a vídeo (si --source video)")
    p.add_argument("--show", action="store_true", help="Mostrar ventana con detecciones (desactivado por defecto)")
    p.add_argument("--duration", type=int, default=60, help="Duración en segundos (por defecto 60)")
    p.add_argument("--out-dir", type=str, default="./FlowerAI_BoundingBoxes/Metrics", help="Directorio de salida para CSV")
    p.add_argument("--cfg", type=str, default=None, help="Ruta alternativa a yolov4-tiny.cfg")
    p.add_argument("--weights", type=str, default=None, help="Ruta alternativa a yolov4-tiny.weights")
    p.add_argument("--names", type=str, default=None, help="Ruta alternativa a coco.names")
    p.add_argument("--yolo-dir", type=str, default=os.environ.get("YOLO_DIR"), help="Directorio base de assets YOLO")
    return p


def ensure_assets(cfg_path: Optional[str], weights_path: Optional[str], names_path: Optional[str], yolo_dir: Optional[str]) -> Tuple[str, str, str]:
    """Verifica o descarga assets. Permite override por CLI y env."""
    base_dir = yolo_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(base_dir, exist_ok=True)
    # Descarga si faltan
    download_yolo_assets()

    cfg = cfg_path or os.path.join(base_dir, "yolov4-tiny.cfg")
    weights = weights_path or os.path.join(base_dir, "yolov4-tiny.weights")
    names = names_path or os.path.join(base_dir, "coco.names")
    for pth in [cfg, weights, names]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"No existe asset requerido: {pth}")
    return cfg, weights, names


def load_net(cfg: str, weights: str) -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_class_names(names_path: str) -> List[str]:
    with open(names_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names


def person_class_index(names: List[str], person_name: str = "person") -> int:
    try:
        return names.index(person_name)
    except ValueError:
        return -1


def preprocess(frame: np.ndarray, input_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(input_size, input_size),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    return blob, (w, h)


def infer_persons(net: cv2.dnn_Net, frame: np.ndarray, input_size: int, conf_thr: float, nms_thr: float, person_idx: int) -> Tuple[List[List[int]], List[float]]:
    """Retorna (boxes, confidences) tras filtrar clase persona y aplicar NMS."""
    blob, (w, h) = preprocess(frame, input_size)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    boxes = []
    confidences = []

    # Parseo YOLO: outs es lista de salidas
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if class_id == person_idx and confidence >= conf_thr:
                center_x = int(det[0] * w)
                center_y = int(det[1] * h)
                width = int(det[2] * w)
                height = int(det[3] * h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(confidence)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    final_boxes = []
    final_confidences = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])

    return final_boxes, final_confidences


def draw_boxes(frame: np.ndarray, boxes: List[List[int]], color=(0, 255, 0)) -> np.ndarray:
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame


def main() -> None:
    args = build_argparser().parse_args()

    # Estado compartido con el plano de control
    shared = SharedState()

    # Lanzar hilo de Flower si se especifica servidor
    if args.server:
        t = threading.Thread(target=flower_thread_fn, args=(args.server, shared), daemon=True)
        t.start()
    else:
        print("[Flower][Client] Sin servidor: ejecutando con valores por defecto.")

    # Assets
    try:
        cfg_path, weights_path, names_path = ensure_assets(args.cfg, args.weights, args.names, args.yolo_dir)
    except Exception as e:
        print(f"[Assets] Error con assets: {e}")
        sys.exit(1)

    # Cargar modelo y clases
    print("[YOLO] Cargando red y clases...")
    net = load_net(cfg_path, weights_path)
    class_names = load_class_names(names_path)
    p_idx = person_class_index(class_names, "person")
    if p_idx < 0:
        print("[YOLO] 'person' no está en coco.names")
        sys.exit(1)
    print("[YOLO] Listo. Clase 'person' idx =", p_idx)

    # Fuente de vídeo
    source_desc = ""
    cap = None
    try:
        if args.source == "webcam":
            # Informativo en Linux
            devs = list_video_devices_linux()
            if devs:
                print("[Cam] Dispositivos detectados:", ", ".join(devs))
            cap, used_idx = open_webcam_with_fallback(args.cam_index)
            source_desc = "webcam"
            print(f"[Cam] Webcam abierta en índice {used_idx}")
        else:
            if not os.path.isfile(args.video_path):
                print(f"[Video] No existe el archivo: {args.video_path}")
                sys.exit(1)
            cap = cv2.VideoCapture(args.video_path)
            if not cap.isOpened():
                print("[Video] No se pudo abrir el vídeo")
                sys.exit(1)
            source_desc = f"video:{os.path.basename(args.video_path)}"
            print(f"[Video] Reproduciendo: {args.video_path}")
    except Exception as e:
        print(f"[Video] Error abriendo fuente: {e}")
        sys.exit(1)

    # CSV de métricas
    start_ts = datetime.now()
    ts_prefix = start_ts.strftime("%Y%m%d_%H%M%S")
    if args.source == "webcam":
        csv_name = f"{ts_prefix}_{METHOD_NAME}_webcam.csv"
    else:
        base = os.path.basename(args.video_path)
        csv_name = f"{ts_prefix}_{METHOD_NAME}_video-{base}.csv"

    f_csv = open_metrics_csv(args.out_dir, csv_name)
    print(f"[CSV] Escribiendo métricas en {os.path.join(args.out_dir, csv_name)}")

    proc = psutil.Process(os.getpid())
    frame_idx = 0
    t0 = perf_counter()
    last_log = t0

    try:
        end_time = time.time() + int(args.duration)
        while time.time() &lt; end_time:
            ret, frame = cap.read()
            if not ret:
                # Para vídeo, si termina, detenemos
                if args.source == "video":
                    print("[Video] Fin del archivo de vídeo.")
                    break
                else:
                    # Webcam: intentar esperar un poco
                    time.sleep(0.01)
                    continue

            t_start = perf_counter()

            # Copia de la config actual
            cfg_now = shared.snapshot()

            # Inferencia
            boxes, confidences = infer_persons(
                net=net,
                frame=frame,
                input_size=cfg_now.input_size,
                conf_thr=cfg_now.conf_thr,
                nms_thr=cfg_now.nms_thr,
                person_idx=p_idx,
            )
            detections = len(boxes)

            if args.show:
                vis = draw_boxes(frame.copy(), boxes)
                cv2.imshow("BBoxes - YOLOv4-tiny (person)", vis)
                if cv2.waitKey(1) &amp; 0xFF == 27:
                    # ESC para salir
                    break

            # Métricas
            t_end = perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            fps_inst = 1000.0 / latency_ms if latency_ms &gt; 0 else 0.0
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_mb = proc.memory_info().rss / (1024 * 1024)
            timestamp_iso = datetime.now().isoformat()

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

            # Log periódico
            now = perf_counter()
            if now - last_log &gt;= 2.0 and frame_idx &gt; 0:
                elapsed = now - t0
                fps_avg = frame_idx / elapsed if elapsed &gt; 0 else 0.0
                print(f"[Loop] frames={frame_idx} fps_avg={fps_avg:.2f} cpu={cpu_pct:.1f}% det={detections} (round={cfg_now.server_round} sz={cfg_now.input_size})")
                last_log = now

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupción por usuario.")
    finally:
        if cap:
            cap.release()
        if args.show:
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