# io.py — image I/O and utility functions

from pathlib import Path
import json
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


def save_overlay(image_bgr: np.ndarray, instances: List[Dict], out_path: Path, draw_id=False, draw_back=False) -> None:
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
        if contours and draw_id:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(
                    overlay,
                    str(inst['id']),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    2,
                    lineType=cv2.LINE_AA
                )

        # Рисуем фоновый круг, если он есть
        if draw_back and 'background' in inst and inst['background'] is not None:
            bg = inst['background']
            center = tuple(map(int, bg['center']))
            radius = int(bg['radius'])
            cv2.circle(overlay, center, radius, color, 1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)



def save_binary_masks(preds: List[Dict], out_path: Path, draw_id=False) -> None:
    """
    Save all masks as a single RGB image with distinct colors and ID labels.
    """
    if not preds:
        return

    h, w = preds[0]['mask'].shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    colors = get_distinct_colors(len(preds))

    for i, inst in enumerate(preds):
        mask = inst['mask']
        color = colors[i % len(colors)]
        for c in range(3):
            canvas[:, :, c][mask.astype(bool)] = color[c]

        ys, xs = np.nonzero(mask)
        if draw_id and len(xs) > 0 and len(ys) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            cv2.putText(
                canvas,
                str(inst['id']),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)



def save_metrics(
    preds: List[Dict],
    folder: str,
    image_name: str,
    metrics_path: Path,
) -> None:
    """
    Save CSI, BSI, and BGAV metrics to a CSV or Excel file.

    If file exists, removes previous rows with same folder+image,
    then appends new rows.
    """

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Формируем новые строки
    records = [{
        "folder": folder,
        "image": image_name,
        "id": inst["id"],
        "csi": inst["csi"],
        "bsi": inst["bsi"],
        "bgav": inst["bgav"],
    } for inst in preds]
    df_new = pd.DataFrame(records)

    if metrics_path.exists():
        if metrics_path.suffix == ".csv":
            df_old = pd.read_csv(metrics_path)
            # Удаляем старые строки с этим folder+image
            df_old = df_old[~((df_old["folder"] == folder) & (df_old["image"] == image_name))]
            df_result = pd.concat([df_old, df_new], ignore_index=True)
            df_result.to_csv(metrics_path, index=False)

        elif metrics_path.suffix in [".xls", ".xlsx"]:
            df_old = pd.read_excel(metrics_path)
            df_old = df_old[~((df_old["folder"] == folder) & (df_old["image"] == image_name))]
            df_result = pd.concat([df_old, df_new], ignore_index=True)
            df_result.to_excel(metrics_path, index=False)

        else:
            raise ValueError("Unsupported file extension. Use .csv or .xlsx")
    else:
        # Файл ещё не существует — просто сохраняем
        if metrics_path.suffix == ".csv":
            df_new.to_csv(metrics_path, index=False)
        elif metrics_path.suffix in [".xls", ".xlsx"]:
            df_new.to_excel(metrics_path, index=False)
        else:
            raise ValueError("Unsupported file extension. Use .csv or .xlsx")



def _next_id(items: list[dict]) -> int:
    """Вернуть следующий свободный id в секции COCO."""
    return (max((it["id"] for it in items), default=0) + 1) if items else 1


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """Превратить бинарную маску в COCO-polygons (List[List[xy...]])"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 6:      # нужно ≥3 точки
            continue
        poly = cnt.flatten().astype(float).tolist()
        polygons.append(poly)
    return polygons


def append_to_coco(
    preds: List[Dict],
    img_rel_path: str,
    ann_path: Path | str
) -> None:
    """
    • Если изображение с таким file_name уже есть в COCO-json,
      его аннотации удаляются и заменяются новыми.
    • Если изображения ещё нет, создаётся новая запись и новые аннотации.
    """
    ann_path = Path(ann_path)

    # ─── загрузка ───────────────────────────────────────────────
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    coco.setdefault("images", [])
    coco.setdefault("annotations", [])

    # ─── ищем, есть ли уже такое изображение ───────────────────
    img_entry = next(
        (img for img in coco["images"] if img["file_name"] == img_rel_path),
        None,
    )

    if img_entry is not None:
        # ── изображение уже есть: перезаписываем аннотации ─────
        img_id = img_entry["id"]
        # Удаляем старые аннотации для этого image_id
        coco["annotations"] = [
            ann for ann in coco["annotations"] if ann["image_id"] != img_id
        ]
    else:
        # ── изображение отсутствует: добавляем новое ───────────
        img_id = _next_id(coco["images"])
        h, w = preds[0]["mask"].shape
        coco["images"].append(
            {
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": img_rel_path,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

    # ─── добавляем (или заново добавляем) аннотации ────────────
    ann_id = _next_id(coco["annotations"])
    for inst in preds:
        mask = inst["mask"]
        area = int(mask.sum())
        x1, y1, x2, y2 = inst["box"]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        segm = _mask_to_polygons(mask)

        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # одна категория
                "segmentation": segm,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    # ─── сохранение ────────────────────────────────────────────
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

