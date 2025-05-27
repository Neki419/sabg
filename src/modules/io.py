# io.py — image I/O and utility functions

from pathlib import Path
import pandas as pd
from typing import Union, List, Dict, Optional
import cv2
import numpy as np
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


def get_distinct_colors(n: int) -> List[tuple[int, int, int]]:
    """
    Generate visually distinct RGB colors using evenly spaced hues in HSV.
    Returned as (B, G, R) tuples for OpenCV.
    """
    hues = [i / n for i in range(n)]
    return [
        tuple(int(c * 255) for c in reversed(colorsys.hsv_to_rgb(h, 1, 1)))  # RGB → BGR
        for h in hues
    ]


def save_overlay(image_bgr: np.ndarray, instances: List[Dict], out_path: Path, background=False) -> None:
    """
    Save image with all masks overlayed in distinct colors and ID labels.
    If 'background' is present in an instance, draws a circle at that location.
    """
    overlay = image_bgr.copy()
    colors = get_distinct_colors(len(instances))

    for i, inst in enumerate(instances):
        color = colors[i % len(colors)]

        # Рисуем контуры маски
        contours, _ = cv2.findContours(inst['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Рисуем ID
        if contours:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(
                    overlay,
                    str(inst['id']),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA
                )

        # Рисуем фоновый круг, если он есть
        if background and 'background' in inst and inst['background'] is not None:
            bg = inst['background']
            center = tuple(map(int, bg['center']))
            radius = int(bg['radius'])
            cv2.circle(overlay, center, radius, color, 1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)



def save_binary_masks(name: str, instances: List[Dict], out_dir: Path) -> None:
    """
    Save all masks as a single RGB image with distinct colors and ID labels.
    """
    if not instances:
        return

    h, w = instances[0]['mask'].shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    colors = get_distinct_colors(len(instances))

    for i, inst in enumerate(instances):
        mask = inst['mask']
        color = colors[i % len(colors)]
        for c in range(3):
            canvas[:, :, c][mask.astype(bool)] = color[c]

        ys, xs = np.nonzero(mask)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            cv2.putText(
                canvas,
                str(inst['id']),
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



def save_metrics(
    folder: str,
    image: str,
    instances: List[Dict],
    path: Optional[Path] = None
) -> None:
    """
    Save CSI, BSI, and BGAV metrics to a CSV or Excel file.

    Each row includes:
        folder, image, id, csi, bsi, bgav

    If file exists, appends new rows. Otherwise creates it with headers.
    If path is not specified, defaults to './<folder>.xlsx'.
    """
    # Подставляем путь по умолчанию
    if path is None:
        path = Path(f"{folder}.xlsx")

    records = []
    for inst in instances:
        records.append({
            "folder": folder,
            "image": image,
            "id": inst["id"],
            "csi": inst["csi"],
            "bsi": inst["bsi"],
            "bgav": inst["bgav"],
        })

    df_new = pd.DataFrame(records)

    if path.suffix == ".csv":
        if path.exists():
            df_new.to_csv(path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(path, index=False)

    elif path.suffix in [".xls", ".xlsx"]:
        from openpyxl import load_workbook

        if path.exists():
            with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df_new.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            df_new.to_excel(path, index=False)

    else:
        raise ValueError("Unsupported file extension. Use .csv or .xlsx")
