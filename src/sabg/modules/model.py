# model.py — Mask R-CNN wrapper

import os
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Model:
    def __init__(self, weights_name: str, device: str = "cuda"):
        self.device = torch.device(device)

        # Build model architecture (NUM_CLASSES = 2 assumed: background + cell)
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        # Custom anchor generator
        anchor_gen = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
        )
        self.model.rpn.anchor_generator = anchor_gen

        # Custom RPN head
        in_channels = self.model.backbone.out_channels
        num_anchors = len((0.25, 0.5, 1.0, 2.0, 4.0))
        self.model.rpn.head = RPNHead(in_channels, num_anchors)

        # Custom mask head
        self.model.roi_heads.mask_head = MaskRCNNHeads(256, [256, 256, 256, 256], dilation=1)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)

        # Load weights
        weights_path = hf_hub_download(repo_id="Neki419/sabg", filename=weights_name)
        weights = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()

    def predict(self, image_bgr: np.ndarray) -> Dict:
        """
        Run inference on a single image.
        Returns a dict with masks, scores, boxes, labels, etc.
        """
        from modules.io import to_tensor  # avoid circular import
        tensor = to_tensor(image_bgr)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            preds = self.model([tensor])[0]  # batch of 1

        return preds
