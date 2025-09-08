# -*- coding: utf-8 -*-
# Utilidades para abrir webcam y listar dispositivos de vídeo.

import glob
import os
from typing import Optional, Tuple

import cv2


def list_video_devices_linux() -> list:
    """Lista /dev/video* (Linux)."""
    if os.name != "posix":
        return []
    return sorted(glob.glob("/dev/video*"))


def try_open_camera(index: int, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[Optional[cv2.VideoCapture], int]:
    """Intenta abrir la cámara en un índice y retorna (cap, index_abierto) o (None, -1)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, -1
    # Opcionalmente, configurar resolución
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, index


def open_webcam_with_fallback(preferred_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[cv2.VideoCapture, int]:
    """Abre webcam probando índices 0..3 con preferencia por preferred_index."""
    indices = [preferred_index] + [i for i in range(0, 4) if i != preferred_index]
    last_err = None
    for idx in indices:
        cap, opened_idx = try_open_camera(idx, width=width, height=height)
        if cap is not None:
            return cap, opened_idx
    raise RuntimeError("No se pudo abrir ninguna cámara en índices 0..3")