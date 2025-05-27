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
    folder: str,
    image_name: str,
    ann_path: Path | str
) -> None:
    ann_path = Path(ann_path)

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # ───── новый image entry ─────
    img_id = _next_id(coco.get("images", []))
    h, w = preds[0]["mask"].shape
    coco.setdefault("images", []).append({
        "id": img_id,
        "width":  w,
        "height": h,
        "file_name": f"{folder}/{image_name}",
        "license":  0,
        "flickr_url": "",
        "coco_url":   "",
        "date_captured": 0
    })

    # ───── новые annotations ─────
    ann_id = _next_id(coco.get("annotations", []))
    for inst in preds:
        mask   = inst["mask"]
        area   = int(mask.sum())
        x1, y1, x2, y2 = inst["box"]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        segm   = _mask_to_polygons(mask)

        coco.setdefault("annotations", []).append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,        # при одной категории
            "segmentation": segm,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        })
        ann_id += 1

    # ───── сохранить обратно ─────
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
