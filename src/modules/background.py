"""background.py — поиск локального низоконтрастного фона для каждой маски

Дополнение: сохраняется не только центр, но и радиус фонового круга.

Выход: каждая запись в instances содержит:
    'background': {'center': (x, y), 'radius': int} или None
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


def _is_background_valid(full_forbidden: np.ndarray, center: Tuple[int, int], radius: int, shape: Tuple[int, int]) -> bool:
    h, w = shape
    cx, cy = center
    if cx < radius or cy < radius or cx + radius >= w or cy + radius >= h:
        return False

    circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle, center, radius, 255, -1)
    overlap = cv2.bitwise_and(circle, full_forbidden)
    return np.count_nonzero(overlap) == 0


def find_background(
    image: np.ndarray,
    instances: List[Dict],
    cfg: Dict,
) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    init_r = cfg.get("initial_ring_radius", 40)
    patch_r = cfg.get("patch_radius", 15)
    angle_step = cfg.get("angle_step", 20)
    buffer_px = cfg.get("buffer_px", 2)
    max_r = cfg.get("max_ring_radius", 300)
    ring_step = cfg.get("ring_step", 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_px + 1, 2 * buffer_px + 1))

    union_all = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        union_all = cv2.bitwise_or(union_all, inst["mask"].astype(np.uint8))

    for inst in instances:
        cell_mask = inst["mask"].astype(np.uint8)
        buffered_current = cv2.dilate(cell_mask * 255, kernel)
        others_mask = cv2.bitwise_and(union_all, cv2.bitwise_not(cell_mask)) * 255
        full_forbidden = cv2.bitwise_or(buffered_current, others_mask)

        M = cv2.moments(cell_mask)
        if M["m00"] == 0:
            inst["background"] = None
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        best_patch: Optional[Tuple[int, int]] = None

        for ring_r in range(init_r, max_r + 1, ring_step):
            candidates = []
            for angle in range(0, 360, angle_step):
                theta = np.deg2rad(angle)
                sx = int(cx + ring_r * np.cos(theta))
                sy = int(cy + ring_r * np.sin(theta))

                if not _is_background_valid(full_forbidden, (sx, sy), patch_r, (h, w)):
                    continue

                circle = np.zeros_like(gray)
                cv2.circle(circle, (sx, sy), patch_r, 255, -1)
                roi = gray[circle == 255]
                candidates.append(((sx, sy), np.std(roi)))

            if candidates:
                best_patch = min(candidates, key=lambda x: x[1])[0]
                break

        if best_patch:
            inst["background"] = {"center": best_patch, "radius": patch_r}
        else:
            inst["background"] = None

    return instances
