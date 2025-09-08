#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilidades para crear y escribir métricas en CSV con cabecera exacta:

timestamp,method,source,frame_idx,latency_ms,fps_inst,cpu_pct,ram_mb,detections
"""

import csv
from pathlib import Path
from typing import Iterable, Optional


CSV_HEADER = [
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


class MetricsWriter:
    """Abre un CSV, escribe la cabecera si es nuevo y permite añadir filas con flush seguro."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if self.csv_path.stat().st_size == 0:
            # Archivo nuevo, escribir cabecera exacta
            self._writer.writerow(CSV_HEADER)
            self._file.flush()

    def writerow(self, row: Iterable) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None