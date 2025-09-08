# -*- coding: utf-8 -*-
# Dibujo de esqueletos COCO (17 puntos clave) y cajas.

from typing import List, Tuple
import cv2
import numpy as np

# Índices COCO Keypoints (17):
# 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
# 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
# 9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
# 13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle

COCO_EDGES = [
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
    (5, 6),                  # shoulders
    (11, 12),                # hips
    (5, 11), (6, 12),        # torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
]

def draw_boxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img

def draw_keypoints_and_skeleton(
    img: np.ndarray,
    keypoints: List[np.ndarray],
    scores: List[float] = None,
    kp_thresh: float = 0.5,
    color_kp=(0, 255, 255),
    color_limb=(255, 0, 0),
) -> np.ndarray:
    """
    keypoints: lista de arrays (K, 3) donde cada fila es (x, y, v) o (x, y, prob)
    """
    for kps in keypoints:
        # Esperamos (17,3)
        for i in range(kps.shape[0]):
            x, y, v = kps[i]
            if v >= kp_thresh:
                cv2.circle(img, (int(x), int(y)), 2, color_kp, -1)
        # Conexiones
        for i, j in COCO_EDGES:
            x1, y1, v1 = kps[i]
            x2, y2, v2 = kps[j]
            if v1 >= kp_thresh and v2 >= kp_thresh:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color_limb, 2)
    return img