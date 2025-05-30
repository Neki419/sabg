# mask_utils.py — filtering and mask postprocessing

from typing import List, Dict
import numpy as np


def filter_masks(preds: Dict, cfg: Dict) -> List[Dict]:
    """
    Filters predicted instances based on confidence, area, and optional removal of non-contiguous (ripped) masks.

    Args:
        preds: Dict with keys 'masks', 'scores', 'boxes', 'labels' from model.predict()
        cfg: Dict with filtering thresholds, e.g.:
            {
                'min_conf': 0.7,
                'min_area': 200,
                'remove_ripped': True,
                'pixel_thr': 0.5
            }

    Returns:
        List of kept masks with fields: id, mask, score, box, etc.
    """
    masks = preds['masks']         # Tensor[N, 1, H, W]
    scores = preds['scores']       # Tensor[N]
    boxes = preds['boxes']         # Tensor[N, 4]

    keep = []
    for i in range(len(masks)):
        score = scores[i].item()
        if score < cfg.get('min_conf', 0.5):
            continue

        mask = masks[i][0].cpu().numpy() > cfg.get('pixel_thr', 0.5)  # binarize
        area = mask.sum()
        if cfg.get('min_area', None) is not None and area < cfg['min_area']:
            continue

        # Optional: remove masks with more than one connected component
        if cfg.get("remove_ripped", False):
            from skimage.measure import label
            num_components = label(mask.astype(np.uint8)).max()
            if num_components > 1:
                continue

        keep.append({
            'id': None,           # will be set later
            'mask': mask.astype(np.uint8),
            'score': score,
            'box': boxes[i].cpu().numpy(),
            'background': None,   # will be set later
            'csi': None,          # will be set later
            'bsi': None,          # will be set later
            'bgav': None,         # will be set later
        })

    return keep



def assign_ids(instances: List[Dict]) -> List[Dict]:
    """
    Assign unique ID to each instance for tracking.
    """
    for idx, inst in enumerate(instances):
        inst['id'] = idx + 1
    return instances
