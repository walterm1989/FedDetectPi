#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Descarga automática de assets de YOLOv4-tiny:
#  - yolov4-tiny.cfg
#  - yolov4-tiny.weights
#  - coco.names
#
# Directorio destino: FlowerAI_BoundingBoxes/assets/
# Se salta la descarga si ya existen.

import os
import sys
import urllib.request


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(THIS_DIR, "assets")

URLS = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    # Pesos oficiales (mirror en GitHub releases de AlexeyAB)
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dst_path: str) -> None:
    print(f"[Assets] Descargando {url} -> {dst_path}")
    with urllib.request.urlopen(url) as response, open(dst_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"[Assets] OK: {dst_path}")


def download_all() -> None:
    ensure_dir(ASSETS_DIR)
    for fname, url in URLS.items():
        fpath = os.path.join(ASSETS_DIR, fname)
        if os.path.isfile(fpath):
            print(f"[Assets] Ya existe: {fpath}, saltando")
            continue
        download_file(url, fpath)
    print("[Assets] Descarga completada.")


if __name__ == "__main__":
    try:
        download_all()
    except KeyboardInterrupt:
        print("\nCancelado por el usuario")
        sys.exit(1)