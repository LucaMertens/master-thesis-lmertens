"""Promptable SAM for the labeling UI, with per-frame embeddings cached in an LRU."""

import os
import threading
from collections import OrderedDict

import cv2
import numpy as np
import torch
from transformers import SamModel, SamProcessor

from common import rle, unrle  # noqa: F401  (re-exported for callers of sam.rle)

MODEL = "facebook/sam-vit-large"
# below this IoU the decoder found no coherent region (chained refinement can collapse)
MIN_IOU = 0.4
# SAM often ranks a sliver top for a ground point, so area gates the choice before the score
MIN_AREA = 0.005
# keyframe JPEGs are 960x720 while clicks arrive in native 1440x1080 pixels
JPEG_SCALE = 1.5


class Prompter:
    """SAM held on one GPU with an LRU of frame embeddings."""

    def __init__(self, frames_dir, device="cuda:3", cache=64):
        self.frames_dir = frames_dir
        self.device = device
        self.proc = SamProcessor.from_pretrained(MODEL)
        self.model = SamModel.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
        self.cache, self.cap = OrderedDict(), cache
        # prefetch thread and clicks share one GPU
        self.lock = threading.Lock()
        # warmup: the first CUDA call is slow (750 ms vs 50 ms)
        with torch.no_grad():
            self.model.get_image_embeddings(
                torch.zeros(1, 3, 1024, 1024, dtype=torch.float16, device=device))

    def _embed(self, name):
        # whole body under the lock, concurrent encodes are slower than waiting
        with self.lock:
            if name in self.cache:
                self.cache.move_to_end(name)
                return self.cache[name]
            path = os.path.join(self.frames_dir, name + ".jpg")
            img = cv2.imread(path)
            if img is None:
                raise KeyError(name)
            # BGR -> RGB; torch refuses the negative stride
            img = np.ascontiguousarray(img[:, :, ::-1])
            inp = self.proc(img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_embeddings(inp["pixel_values"].to(torch.float16))
            # sizes kept so clicks skip preprocessing
            entry = (emb, inp["original_sizes"], inp["reshaped_input_sizes"])
            self.cache[name] = entry
            while len(self.cache) > self.cap:
                self.cache.popitem(last=False)
            return entry

    def warm(self, name):
        """Encode ahead of the annotator. Returns whether it had to encode."""
        if name in self.cache:
            self.cache.move_to_end(name)
            return False
        self._embed(name)
        return True

    def mask(self, name, points, labels, out_size, prev=None):
        """Binary mask at ``out_size`` (w, h) for native-pixel prompts.

        ``labels``: 1 inside, 0 outside. ``prev``: previous logits, so clicks refine one
        region. Returns (mask, logits, score).
        """
        emb, orig, reshaped = self._embed(name)
        pts = torch.tensor([[[[u / JPEG_SCALE, v / JPEG_SCALE] for u, v in points]]],
                           dtype=torch.float32, device=self.device)
        lbl = torch.tensor([[list(labels)]], dtype=torch.int32, device=self.device)
        kw = {}
        if prev is not None:
            # must match the model dtype
            kw["input_masks"] = prev.to(self.device, torch.float16)
        with self.lock, torch.no_grad():
            out = self.model(image_embeddings=emb, input_points=pts, input_labels=lbl,
                             multimask_output=prev is None, **kw)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), orig.cpu(), reshaped.cpu())[0][0]
        scores = out.iou_scores.flatten().tolist()
        areas = [float(x.float().mean()) for x in masks]
        # best score among proposals above MIN_AREA, else the largest
        live = [j for j in range(len(masks)) if areas[j] >= MIN_AREA]
        i = max(live, key=lambda j: scores[j]) if live else int(np.argmax(areas))
        m = masks[i].numpy().astype(np.uint8)
        if (m.shape[1], m.shape[0]) != tuple(out_size):
            m = cv2.resize(m, tuple(out_size), interpolation=cv2.INTER_NEAREST)
        # (batch, point_batch, masks, 256, 256) -> (batch, 1, 256, 256)
        logits = out.pred_masks[:, 0, i : i + 1].float().cpu()
        return m, logits, float(scores[i])
