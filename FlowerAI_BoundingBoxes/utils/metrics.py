# -*- coding: utf-8 -*-
# Utilidad para gestionar CSV de métricas con cabecera exacta.

import csv
import os
from typing import IO, Dict, Any, Optional


HEADER = [
    "timestamp",
    "method",
    "source",
    "frame_idx",
    "latency_ms",
    "fps_inst",
    "cpu_pct",
    "ram_mb",
    "detections",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_metrics_csv(out_dir: str, filename: str) -> IO[str]:
    """Abre un CSV en out_dir/filename y escribe la cabecera si el archivo no existía."""
    ensure_dir(out_dir)
    fpath = os.path.join(out_dir, filename)
    file_exists = os.path.isfile(fpath)
    f = open(fpath, mode="a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(HEADER)
        f.flush()
        os.fsync(f.fileno())
    return f


def write_metrics_row(f: IO[str], row: Dict[str, Any]) -> None:
    """Escribe una fila según el orden exacto de la cabecera y hace flush."""
    writer = csv.writer(f)
    writer.writerow([
        row.get("timestamp"),
        row.get("method"),
        row.get("source"),
        row.get("frame_idx"),
        row.get("latency_ms"),
        row.get("fps_inst"),
        row.get("cpu_pct"),
        row.get("ram_mb"),
        row.get("detections"),
    ])
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        # En algunos FS no es necesario/posible
        pass