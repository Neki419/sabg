# coco_append.py
"""
Добавить результаты одного изображения в готовый COCO-json.

Аргументы
---------
preds        : List[Dict]  – вывод filter_masks ( masks → np.uint8 )
folder       : str         – название под-папки (сохраняется в file_name)
image_name   : str         – имя файла (пример: "img_0001.png")
ann_path     : Path | str  – путь к существующему COCO-json

Файл ann_path будет обновлён «на месте»:  
в sections  *images*  и  *annotations*  появятся новые записи с корректно
сдвинутыми `id`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np


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

