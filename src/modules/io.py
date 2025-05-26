# io.py — image I/O and utility functions

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import os
import colorsys
import torch


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file as BGR numpy array.
    """
    path = Path(path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return image


def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR image (H, W, 3) to normalized torch tensor (3, H, W) in range [0.0, 1.0].
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()  # (3, H, W)
    return tensor


def get_distinct_colors(n: int) -> list:
    """
    Generate visually distinct RGB colors using evenly spaced hues in HSV.
    Returned as (B, G, R) tuples for OpenCV.
    """
    hues = [i / n for i in range(n)]
    return [
        tuple(int(c * 255) for c in reversed(colorsys.hsv_to_rgb(h, 1, 1)))  # RGB -> BGR
        for h in hues
    ]


def save_overlay(image_bgr: np.ndarray, masks: list, out_path: Path) -> None:
    """
    Save image with all masks overlayed in distinct colors and ID labels.

    Args:
        image_bgr: BGR np.ndarray (H, W, 3)
        masks: list of dicts with keys 'mask' and 'id'
        out_path: full path to output PNG file (e.g. path/to/img_overlay.png)
    """
    overlay = image_bgr.copy()
    colors = get_distinct_colors(len(masks))

    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        contours, _ = cv2.findContours(m['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        if contours:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(
                    overlay,
                    str(m['id']),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA
                )

    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


def save_binary_masks(name: str, masks: list, out_dir: Path) -> None:
    """
    Save all masks as a single RGB image with distinct colors and ID labels.
    """
    import cv2

    if not masks:
        return

    h, w = masks[0]['mask'].shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    colors = get_distinct_colors(len(masks))

    for i, m in enumerate(masks):
        mask = m['mask']
        color = colors[i % len(colors)]
        for c in range(3):
            canvas[:, :, c][mask.astype(bool)] = color[c]

        # Label with ID
        ys, xs = np.nonzero(mask)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            cv2.putText(
                canvas,
                str(m['id']),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )

    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{name}_masks.png"
    cv2.imwrite(str(out_path), canvas)