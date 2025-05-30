"""metrics.py — расчёт CSI, BSI и BGAV по найденным маскам и фоновым патчам

Вход:
    image: np.ndarray — BGR изображение (cv2.imread)
    instances: List[Dict] — из filter_masks + find_background

Выход:
    Обновлённый список instances с полями:
        - 'csi'
        - 'bsi'
        - 'bgav'
"""

import numpy as np
import cv2
import math
from typing import List, Dict, Tuple


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Безопасное поэлементное деление с заменой NaN и inf на 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(numerator, denominator)
        ratio[~np.isfinite(ratio)] = 0
    return ratio


def compute_csi_bsi_bgav(image: np.ndarray, preds: List[Dict]) -> List[Dict]:
    """
    Добавляет CSI, BSI и BGAV к каждому объекту с найденным фоном.
    """
    for inst in preds:
        mask = inst['mask'] == 1  # бинарная маска
        bg_info = inst.get('background')

        if not bg_info or 'center' not in bg_info or 'radius' not in bg_info:
            inst['csi'] = inst['bsi'] = inst['bgav'] = None
            continue

        # --- клетка ---
        cell_pixels = image[mask]  # (N, 3)
        if len(cell_pixels) == 0:
            inst['csi'] = inst['bsi'] = inst['bgav'] = None
            continue

        cell_rgb_sum = np.sum(cell_pixels, axis=1)
        cell_gb_sum = cell_pixels[:, 1] + cell_pixels[:, 0]
        csi = np.sum(safe_ratio(cell_gb_sum, cell_rgb_sum)) / len(cell_pixels)

        # --- фон ---
        h, w, _ = image.shape
        sx, sy = bg_info['center']
        patch_radius = bg_info['radius']

        circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle, (sx, sy), patch_radius, 255, -1)
        bg_pixels = image[circle == 255]

        if len(bg_pixels) == 0:
            inst['csi'] = csi
            inst['bsi'] = inst['bgav'] = None
            continue

        bg_rgb_sum = np.sum(bg_pixels, axis=1)
        bg_gb_sum = bg_pixels[:, 1] + bg_pixels[:, 0]
        bsi = np.sum(safe_ratio(bg_gb_sum, bg_rgb_sum)) / len(bg_pixels)

        bgav = math.log(csi / bsi) if bsi > 0 else None

        inst['csi'] = csi
        inst['bsi'] = bsi
        inst['bgav'] = bgav

    return preds
