#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilidades para gestionar fuentes de vídeo:
- Abrir webcam probando índices si falla el solicitado
- Abrir archivo de vídeo
- Listar dispositivos /dev/video* en Linux (opcional)
"""

from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import os
import glob


def list_video_devices_linux() -> List[str]:
    """Lista dispositivos /dev/video* en Linux."""
    if os.name != "posix":
        return []
    return sorted(glob.glob("/dev/video*"))


def open_webcam_with_fallback(index: int = 0, try_range: Tuple[int, int] = (0, 3)) -> Optional[cv2.VideoCapture]:
    """
    Intenta abrir la webcam en `index`. Si falla, prueba índices en el rango [a..b].
    """
    cap = cv2.VideoCapture(index)
    if cap is not None and cap.isOpened():
        return cap

    if cap is not None:
        cap.release()

    start, end = try_range
    for i in range(start, end + 1):
        if i == index:
            continue
        tmp = cv2.VideoCapture(i)
        if tmp is not None and tmp.isOpened():
            print(f"Webcam abierta en índice alternativo {i}")
            return tmp
        if tmp is not None:
            tmp.release()

    print("No se pudo abrir ninguna webcam. En Linux, pruebe con:")
    devs = list_video_devices_linux()
    if devs:
        print("Dispositivos detectados:", ", ".join(devs))
    else:
        print("No se detectaron /dev/video*")
    return None


def open_video_file(path: str) -> Optional[cv2.VideoCapture]:
    """Abre un archivo de vídeo si existe."""
    p = Path(path)
    if not p.exists():
        print(f"Archivo de vídeo no encontrado: {path}")
        return None
    cap = cv2.VideoCapture(str(p))
    if cap is None or not cap.isOpened():
        print(f"No se pudo abrir el vídeo: {path}")
        return None
    return cap