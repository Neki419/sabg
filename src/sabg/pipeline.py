# pipeline.py — ties every module together according to config.yaml
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union
import tqdm

# internal modules
from sabg.modules.model      import Model
from sabg.modules.mask_utils import filter_masks, assign_ids
from sabg.modules.background import find_background
from sabg.modules.metrics    import compute_csi_bsi_bgav
from sabg.modules.io         import (
    load_image,
    save_overlay,
    save_binary_masks,
    save_metrics,
    append_to_coco,
)


# ───────────────────────── helpers ──────────────────────────
_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_images(folder: Path) -> List[Path]:
    """Return all images inside *folder* (recursively)."""
    images: list[Path] = [
        p for p in sorted(folder.rglob("*"))
        if p.suffix.lower() in _IMG_EXT
    ]
    if not images:
        raise FileNotFoundError(f"No images in {folder}")
    return images



# ───────────────────────── main API ─────────────────────────
def run_pipeline(folder_path: Union[str, Path], cfg: Dict) -> None:
    folder_path = Path(folder_path)
    images      = _collect_images(folder_path)
    folder_name = folder_path.name

    # ── output: root/folder
    out_root = Path(cfg["paths"]["output_dir"]) / folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    # ── metrics
    metrics_path =  Path(cfg["paths"]["metrics_file_path"])
       
    # ── COCO
    coco_path = None
    if cfg["output"].get("save_annotations", False):
        coco_path = Path(cfg["paths"]["annotations_file_path"])

        if not coco_path.exists():
            coco_path.write_text(
                '{"info": {}, "licenses": [], '
                '"categories":[{"id":1,"name":"cell"}],'
                '"images":[], "annotations":[]}'
            )

    # ── model
    model = Model(
        weights_name=cfg["model"],
        device=cfg["device"],
    )

    # ── per-image processing
    for img_path in tqdm.tqdm(images, desc="inference"):
        img_bgr  = load_image(img_path)
        preds    = model.predict(img_bgr)

        inst     = filter_masks(preds, cfg["filter"])
        inst     = assign_ids(inst)
        inst     = find_background(img_bgr, inst, cfg["background"])
        inst     = compute_csi_bsi_bgav(img_bgr, inst)

        stem = img_path.stem

        if cfg["output"]["save_overlay"]:
            save_overlay(
                img_bgr,
                inst,
                out_root / f"{stem}_overlay.png",
                draw_id   = cfg["output"]["overlay_draw_id"],
                draw_back = cfg["output"]["overlay_draw_background"],
            )

        if cfg["output"]["save_binary_masks"]:
            save_binary_masks(
                inst,
                out_root / f"{stem}_masks.png",
                draw_id = cfg["output"]["binary_mask_draw_id"],
            )

        save_metrics(
            inst,
            folder=folder_name,
            image_name=img_path.name,
            metrics_path=metrics_path,
        )

        if coco_path:
            append_to_coco(
                preds=inst,
                img_rel_path=f"{folder_name}/{img_path.name}",
                ann_path=coco_path,
            )

    print("✅  Done. Results saved to", out_root)
