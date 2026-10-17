"""Human pair labels as (imgA, imgB, [img, u, v, img, u, v, t]) samples in W-RIZZ's format."""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from traversability_network.adaptations import human_pairs as HP
from traversability_network.adaptations.npy_cache import imread

# (H, W); matches the stored resized copies (4:3 camera), so nothing is resampled twice
RESOLUTION = (360, 480)
NATIVE = (1080, 1440)
REGION_UV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "region_uv.npz")


def frame_index(data_root, resized_root, mission, camera="front_center"):
    """ts_ns -> image path for one mission, preferring the pre-resized local copy."""
    mission_dir = os.path.join(data_root, mission)
    manifest = np.load(os.path.join(mission_dir, "image_manifest.npz"))[camera]
    root = mission_dir
    if resized_root and os.path.isdir(os.path.join(resized_root, mission)):
        root = os.path.join(resized_root, mission)
    return {int(ts): os.path.join(root, p) for ts, p in zip(manifest["ts_ns"], manifest["path"])}


def _load_regions():
    """(mission, label_id, side) -> (native pixels, validity) for the annotated frame."""
    d = np.load(REGION_UV)
    out = {}
    for key in d.files:
        if not key.endswith("/uv"):
            continue
        mission = key[: -len("/uv")]
        uv, ok = d[f"{mission}/uv"], d[f"{mission}/ok"]
        for i, (lid, side) in enumerate(zip(d[f"{mission}/label_id"], d[f"{mission}/side"])):
            out[(mission, str(lid), str(side))] = (uv[i], ok[i])
    return out


class PairDataset(Dataset):
    """Human pairs of the given missions. With n_samples, draws that many region pixels per side."""

    def __init__(self, cfg, missions, table=None, source=None, trainable=None,
                 resolution=RESOLUTION, augmentation=None, normalize=None, n_samples=0,
                 consistency_augmentation=None, frames=None):
        self.table = table or HP.PairTable(cfg.DATASET.HUMAN_INDEX_ROOT)
        self.resolution = resolution
        self.augmentation = augmentation
        self.normalize = normalize
        self.n_samples = n_samples
        self.regions = _load_regions() if n_samples else None
        # student and teacher share geometry, photometric noise is drawn independently
        self.consistency_augmentation = consistency_augmentation
        self.frames = frames
        missions = set(missions)
        self.index = {m: frame_index(cfg.DATASET.DATA_ROOT, cfg.DATASET.RESIZED_ROOT, m)
                      for m in missions}
        self.items = []
        for i in range(len(self.table)):
            _, a, b = self.table.entries[i]
            if not all(e["mission"] in missions for e in (a, b)):
                continue
            if source is not None and self.table.source[i] != source:
                continue
            if trainable is not None and bool(self.table.trainable[i]) != trainable:
                continue
            if not all(e["ts_ns"] in self.index[e["mission"]] for e in (a, b)):
                continue
            # an answer belongs to a split only if both its frames do
            if frames is not None and not all((e["mission"], e["ts_ns"]) in frames
                                              for e in (a, b)):
                continue
            self.items.append(i)
        self.labels = self.table.t[self.items]

    def __len__(self):
        return len(self.items)

    def frame_set(self):
        """Every (mission, ts_ns) an answer in this table touches."""
        out = set()
        for i in range(len(self.table)):
            for e in self.table.entries[i][1:]:
                out.add((e["mission"], e["ts_ns"]))
        return out

    def _image(self, e):
        img = imread(self.index[e["mission"]][e["ts_ns"]], True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self.resolution
        if (img.shape[0], img.shape[1]) != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0

    def _uv(self, e):
        """Native click -> pixel index at the network's resolution, clamped in range."""
        h, w = self.resolution
        u = int(round(e["uv"][0] * w / NATIVE[1]))
        v = int(round(e["uv"][1] * h / NATIVE[0]))
        return min(max(u, 0), w - 1), min(max(v, 0), h - 1)

    def _draw(self, lid, e, side, rng):
        """``n_samples`` pixel indices drawn from this endpoint's region, with repeats."""
        uv, ok = self.regions[(e["mission"], str(lid), side)]
        idx = np.flatnonzero(ok)
        pick = rng.choice(idx, self.n_samples, replace=True) if len(idx) else None
        h, w = self.resolution
        if pick is None:
            return np.tile(np.array(self._uv(e)), (self.n_samples, 1))
        u = np.clip(np.rint(uv[pick, 0] * w / NATIVE[1]), 0, w - 1)
        v = np.clip(np.rint(uv[pick, 1] * h / NATIVE[0]), 0, h - 1)
        return np.stack([u, v], 1).astype(np.int64)

    def __getitem__(self, k):
        i = self.items[k]
        lid, a, b = self.table.entries[i]
        image_a = self._image(a)
        same = (a["mission"], a["ts_ns"]) == (b["mission"], b["ts_ns"])
        image_b = image_a if same else self._image(b)
        ua, va = self._uv(a)
        ub, vb = self._uv(b)
        t, jb = int(self.table.t[i]), 0 if same else 1
        # column order: (img_a, u_a, v_a, img_b, u_b, v_b, t)
        if self.n_samples:
            # redrawn on every access
            rng = np.random.default_rng()
            pa, pb = self._draw(lid, a, "a", rng), self._draw(lid, b, "b", rng)
            anno = torch.from_numpy(np.concatenate(
                [np.zeros((self.n_samples, 1), np.int64), pa,
                 np.full((self.n_samples, 1), jb, np.int64), pb,
                 np.full((self.n_samples, 1), t, np.int64)], 1))
        else:
            anno = torch.tensor([[0, ua, va, jb, ub, vb, t]], dtype=torch.long)
        if self.augmentation is not None:
            image_a, image_b, anno = self.augmentation((image_a, image_b, anno))
        if self.consistency_augmentation is not None:
            views = [self.consistency_augmentation(v)
                     for v in (image_a, image_b, image_a, image_b)]
            if self.normalize is not None:
                views = [self.normalize(v) for v in views]
            return (*views, anno, i)
        if self.normalize is not None:
            image_a, image_b = self.normalize(image_a), self.normalize(image_b)
        return image_a, image_b, anno, i
