"""Does smoothing the test-side score map before pooling help? Detectors are fitted as in eval.run.

    uv run python -m ood_detection.eval.denoise_sweep --features-dir <cache with cellmaps>
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared_utils.tiers import ID_TRAIN, tiers
from ood_detection.common import denoise
from ood_detection.common.splits import HELDOUT_TIER
from ood_detection.common.retier import TIERS, assign
from ood_detection.eval.metrics import evaluate
from ood_detection.eval.run import cell_rows, load_cellmaps, load_tier
from ood_detection.mahalanobis.detector import MahalanobisPP

SCORED_TIERS = [HELDOUT_TIER, *(t for t in tiers() if t not in (ID_TRAIN, HELDOUT_TIER))]
DETECTORS = {"mahalanobis": "mahalanobis", "kmeans": "euclidean"}
SLIC_N = [50, 150, 300]
METHODS = ["none", "box3", "box5", "box7"] + [f"slic{n}" for n in SLIC_N]
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
CHUNK = 400
# same gate as the 512-cell path, so the `none` row matches the reported median
MIN_OBSERVED = 512


def score_maps(det, maps, observed):
    """(N, X, Y) per-cell distance, observed cells only, NaN elsewhere."""
    out = np.full(maps.shape[:3], np.nan, dtype=np.float32)
    for i in range(0, len(maps), CHUNK):
        m, obs = maps[i:i + CHUNK], observed[i:i + CHUNK]
        sel = obs.reshape(len(m), -1)
        flat = m.reshape(len(m), -1, m.shape[-1])
        block = flat[sel].astype(np.float32)
        if len(block) == 0:
            continue
        chunk = np.full(sel.shape, np.nan, dtype=np.float32)
        chunk[sel] = det.cell_scores(block)
        out[i:i + CHUNK] = chunk.reshape(m.shape[:3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    id_train = load_tier(args.features_dir, "id_train")
    train_missions = set(np.unique(id_train["mission"]))
    fit_cells = cell_rows(id_train)
    dets = {name: MahalanobisPP(args.modes, metric=metric).fit(fit_cells)
            for name, metric in DETECTORS.items()}
    print(f"fitted on {len(fit_cells)} in-distribution training cells, unchanged\n")

    basis, pooled, meta = None, [], []
    for src in tqdm(SCORED_TIERS, desc="tiers"):
        data = load_cellmaps(args.features_dir, src)
        if data is None:
            print(f"{src}: no cell maps, skipping")
            continue
        maps, observed = data["maps"], data["observed"]
        enough = observed.reshape(len(observed), -1).sum(1) >= MIN_OBSERVED
        if basis is None:
            basis = denoise.pca_basis(maps)
        labels = {n: denoise.slic_labels(maps, basis, n) for n in SLIC_N}
        meta.append(pd.DataFrame({"mission": data["mission"], "timestamp": data["timestamp"]}))
        block = {}
        for name, det in dets.items():
            s = score_maps(det, maps, observed)
            for method in METHODS:
                lab = labels.get(int(method[4:])) if method.startswith("slic") else None
                sm = denoise.apply(method, s, observed, lab)
                for q in QUANTILES:
                    block[(name, method, q)] = np.where(enough, denoise.pool(sm, q), np.nan)
        pooled.append(pd.DataFrame(block))
        del maps, observed, labels, data

    meta = pd.concat(meta, ignore_index=True)
    pooled = pd.concat(pooled, ignore_index=True)
    tier = assign(meta["mission"].to_numpy(), meta["timestamp"].to_numpy(), train_missions)

    rows = []
    for col in pooled.columns:
        name, method, q = col
        s = pooled[col].to_numpy()
        ref = s[tier == "id_heldout"]
        for t in TIERS:
            if t not in ("id_train", "id_heldout") and (tier == t).any():
                rows.append({"detector": name, "method": method, "q": q, "tier": t,
                             **evaluate(ref, s[tier == t])})
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print("\n== AUROC averaged over tiers ==")
    print(df.pivot_table(index=["detector", "method"], columns="q", values="auroc")
          .round(3).to_string())

    out = args.out or os.path.join(args.features_dir, "eval")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "denoise_sweep.csv"), index=False)
    print(f"\n-> {out}/denoise_sweep.csv")


if __name__ == "__main__":
    main()
