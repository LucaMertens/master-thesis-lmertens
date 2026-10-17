"""Builds the OOD detection report (typst) from the CSVs of eval.run and eval.scores.

    uv run python -m ood_detection.report.make_report --features-dir <cache> --out <dir>
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd

from shared_utils.tiers import ID_TRAIN, tiers
from ood_detection.common.splits import HELDOUT_TIER

TIERS = [t for t in tiers() if t not in (ID_TRAIN, HELDOUT_TIER)]
TIER_LABEL = {t: t.replace("_", " ") for t in TIERS}
DETECTORS = ["mahalanobis_wayfaster", "kmeans_wayfaster", "mahalanobis_dino", "kmeans_dino",
             "viewpoint_disagreement", "modality_disagreement", "predicted_trav"]
DET_LABEL = {
    "mahalanobis_wayfaster": "Mahalanobis++ (BEV)", "kmeans_wayfaster": "k-means (BEV)",
    "mahalanobis_dino": "Mahalanobis++ (DINOv3)", "kmeans_dino": "k-means (DINOv3)",
    "viewpoint_disagreement": "viewpoint disagreement", "modality_disagreement": "modality disagreement",
    "predicted_trav": "predicted traversability",
}
METRICS = [("auroc", "AUROC", "higher"), ("fpr95", "FPR95", "lower"), ("aupr", "AUPR", "higher")]


def cell(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def render(head, body, n):
    """A borderless table: one label column and n right-aligned value columns."""
    return (f'#table(\n  columns: (auto, ..({n} * (1fr,))), align: (left, ..({n} * (right,))),\n'
            f'  stroke: none, inset: (x: 4pt, y: 5pt),\n  table.hline(),\n  [], {head},\n'
            f'  table.hline(stroke: 0.5pt),\n' + "\n".join(body) + "\n  table.hline(),\n)")


def bold_best(wide, better, axis):
    """Bold the best cell per column (axis 0) or row (axis 1), at display precision; all-tied is unmarked."""
    r = wide.round(3).to_numpy(dtype=float)
    lo, hi = np.nanmin(r, axis=axis), np.nanmax(r, axis=axis)
    best = lo if better == "lower" else hi
    hit = (r == np.expand_dims(best, axis)) & np.expand_dims(lo != hi, axis)
    marked = pd.DataFrame(hit, index=wide.index, columns=wide.columns)
    return wide.map(cell).where(~marked, wide.map(lambda v: f"*{cell(v)}*"))


def metric_table(df, metric, cols, better, col_key="tier", label=None, axis=0):
    """Rows detector, columns `cols`, bolding the best of each column (axis 0) or row."""
    label = label or (lambda c: TIER_LABEL.get(c, c))
    wide = df.pivot_table(index="detector", columns=col_key, values=metric).reindex(
        index=[d for d in DETECTORS if d in set(df.detector)], columns=cols)
    body = [f'  [{DET_LABEL.get(det, det)}], ' + ", ".join(f"[{v}]" for v in row) + ","
            for det, row in bold_best(wide, better, axis).iterrows()]
    return render(", ".join(f"[*{label(c)}*]" for c in cols), body, len(cols))


def dropout_table(df):
    """One tier, all three metrics as the columns."""
    d = df[df.tier == "sensor_dropout"].set_index("detector")
    body = [f'  [{DET_LABEL[det]}], ' + ", ".join(
        f"[{cell(d[m][det]) if det in d.index else '--'}]" for m, _, _ in METRICS) + ","
        for det in DETECTORS]
    return render(", ".join(f"[*{lab}*]" for _, lab, _ in METRICS), body, len(METRICS))


def sweep_tables(sweep):
    """Mean over tiers, per k and detector; one table per metric."""
    g = sweep.groupby(["detector", "n_modes"])[["auroc", "fpr95", "aupr"]].mean().reset_index()
    ks = sorted(sweep.n_modes.unique())
    return [(label, metric_table(g, metric, ks, better, "n_modes", lambda k: f"k = {k}", 1))
            for metric, label, better in METRICS]


def raw_table(raw, detectors):
    tiers = ["id_heldout"] + TIERS
    body = []
    for det in detectors:
        d = raw[raw.detector == det].set_index("tier")
        vals = [f"{d.p50[t]:.3g}" if t in d.index else "--" for t in tiers]
        body.append(f'  [{DET_LABEL.get(det, det)}], ' + ", ".join(f"[{v}]" for v in vals) + ",")
    head = ", ".join(f"[*{TIER_LABEL.get(t, 'held-out ID')}*]" for t in tiers)
    return render(head, body, len(tiers))


def null_table(fp):
    """Per detector: the ID x ID session null, and how often each tier clears it."""
    tiers = [t for t in TIERS if t in set(fp.tier)]
    body = []
    for det in [d for d in DETECTORS if d in set(fp.detector)]:
        g = fp[fp.detector == det].set_index("tier")
        vals = [f"{g.null_mean.iloc[0]:.3f}"] + [
            cell(g.beats_null[t]) if t in g.index else "--" for t in tiers]
        body.append(f'  [{DET_LABEL[det]}], ' + ", ".join(f"[{v}]" for v in vals) + ",")
    head = "[*null*], " + ", ".join(f"[*{TIER_LABEL.get(t, t)}*]" for t in tiers)
    return render(head, body, len(tiers) + 1)


PREAMBLE = """#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, hyphenate: false)
#set par(justify: true)
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em)[#text(size: 13pt)[#it.body]]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em)[#text(size: 11pt)[#it.body]]
"""


def baselines(features_dir, k):
    """Median distance travelled between a window and its partner k steps later."""
    out = {}
    for tier in ["id_heldout"] + TIERS:
        path = os.path.join(features_dir, f"{tier}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path, allow_pickle=True) as f:
            pose, mission = f["pose"], f["mission"]
        same = mission[k:] == mission[:-k]
        out[tier] = float(np.median(np.linalg.norm(pose[k:, :2] - pose[:-k, :2], axis=1)[same]))
    return out


def build(features_dir, out_dir, checkpoint, vp_k, n_modes):
    ev = os.path.join(features_dir, "eval")
    res = pd.read_csv(os.path.join(ev, "results.csv"))
    raw = pd.read_csv(os.path.join(ev, "raw_scores.csv"))
    sweep = pd.read_csv(os.path.join(ev, "modes_sweep.csv"))
    fp_path = os.path.join(ev, "fingerprint_corrected.csv")
    fp = pd.read_csv(fp_path) if os.path.exists(fp_path) else None
    main = res[~res.detector.str.endswith("+ema")]
    counts = main.groupby("tier")[["n_id", "n_ood"]].max()

    title = ('#align(center, block(width: 88%)[#set par(justify: false)\n'
             '#text(size: 16pt)[*Out-of-distribution detection on the GrandTour traversability '
             'estimator*]\n#v(0.5em)\n'
             f'#text(size: 9pt)[checkpoint `{checkpoint}`]])\n#v(0.6em)\n')
    doc = [PREAMBLE, title]

    doc.append("= Setup\n")
    doc.append(
        f"The estimator's penultimate BEV feature is a $64 times 100 times 100$ map, so one "
        f"window is $10^4$ samples of dimension 64. Every detector scores cells, not frames: a "
        f"window's score is the mean of the worst decile of the cells at least one input frame "
        f"observes. In-distribution statistics are fitted on the checkpoint's own training "
        f"missions; the negative class is its held-out validation missions "
        f"({int(counts.n_id.max())} windows).\n")
    doc.append(
        f"Seven detectors. *Mahalanobis++* fits $k = {n_modes}$ k-means modes with one shared "
        f"Ledoit-Wolf covariance on L2-normalized features, in the estimator's BEV space and in "
        f"a frozen DINOv3 ViT-B/16 space. *k-means* is the ablation: the same modes, Euclidean "
        f"distance, no covariance. *Viewpoint disagreement* needs no reference distribution at "
        f"all: the BEV frame is anchored on the window's last input frame, so a window and the "
        f"window ${vp_k}$ steps later cover overlapping ground in different frames; the score is "
        f"$|Delta mu|$ over the worst decile of the cells both observe. *Modality disagreement* "
        f"is the same idea across sensors (depth on vs. off), and *predicted traversability* is "
        f"the estimator's own output, the baseline any detector has to beat.\n")
    base = baselines(features_dir, vp_k)
    doc.append(
        f"The frame stride is fixed, not speed-matched, so the viewpoint baseline is whatever the "
        f"robot covered in ${vp_k}$ steps: a median of "
        f"{min(base.values()):.1f} to {max(base.values()):.1f} m across tiers "
        f"({', '.join(f'{TIER_LABEL[t]} {v:.1f} m' for t, v in base.items())}). "
        f"That score therefore tracks speed as well as novelty.\n")

    doc.append("= Detection\n")
    doc.append(
        "OOD is the positive class, so AUROC and AUPR are higher-is-better and FPR95 is "
        "lower-is-better. AUPR moves with the OOD share of the pool, which differs per tier "
        "(last table), so it compares detectors within a column, never across "
        "columns. Best per column in bold.\n")
    for metric, label, better in METRICS:
        doc.append(f"== {label}\n")
        doc.append(metric_table(main, metric, TIERS, better) + "\n")

    drop = main[main.tier == "sensor_dropout"]
    if not drop.empty:
        doc.append("== Sensor dropout\n")
        doc.append("Held-out ID windows re-scored with the depth input zeroed, against the same "
                   "windows with depth. Depth-blind detectors are not applicable.\n")
        doc.append(dropout_table(main) + "\n")

    doc.append("= Ablation: number of modes\n")
    doc.append(f"Mean over the {len(TIERS)} tiers. $k$ is the number of k-means modes standing in "
               f"for the classes of the original method; traversability is a regression, so there "
               f"are none to use.\n")
    for label, table in sweep_tables(sweep):
        doc.append(f"== {label}\n")
        doc.append(table + "\n")

    if fp is not None:
        doc.append("= Session floor\n")
        doc.append(
            "Every ID mission is also a separate recording, so two ID recordings already "
            "separate on any of these scores. The null column is the mean AUROC between pairs "
            "of same-domain ID recordings: the separation session identity alone buys. "
            "`beats null` is the share of (OOD mission, ID pair) comparisons the OOD mission "
            "wins; 0.5 means the tier shift is indistinguishable from a change of recording.\n")
        doc.append(null_table(fp) + "\n")
        doc.append(
            f"Subtracting each recording's own median score leaves every detector on every tier "
            f"between {fp.auroc_centered.min():.2f} and {fp.auroc_centered.max():.2f} AUROC. What "
            f"the tables above measure is almost entirely a per-recording offset, not a "
            f"within-recording ordering of windows.\n")

    doc.append("= Raw scores\n")
    doc.append("Median window score per tier, on each detector's own scale. AUROC is invariant to "
               "any monotone rescaling, so it hides whether a tier is far from the ID mass or only "
               "ordered above it.\n")
    doc.append(raw_table(raw, DETECTORS) + "\n")

    doc.append("= Windows per tier\n")
    rows = "\n  ".join(f"[{TIER_LABEL.get(t, t)}], [{int(counts.n_ood[t])}], "
                        f"[{counts.n_ood[t] / (counts.n_id[t] + counts.n_ood[t]):.3f}],"
                        for t in TIERS if t in counts.index)
    doc.append(f'#table(\n  columns: 3, align: (left, right, right), stroke: none, inset: 5pt,\n'
               f'  table.hline(),\n  [], [*windows*], [*base rate*],\n'
               f'  table.hline(stroke: 0.5pt),\n  {rows}\n  table.hline(),\n)\n')

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ood_report.typ")
    with open(path, "w") as f:
        f.write("\n".join(doc))
    subprocess.run(["typst", "compile", path], check=True)
    print(f"report -> {path.replace('.typ', '.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="gt_human_ineq075_rhe005_f2 (tidgo49o)")
    ap.add_argument("--vp-k", type=int, default=10)
    ap.add_argument("--modes", type=int, default=3)
    args = ap.parse_args()
    build(args.features_dir, args.out, args.checkpoint, args.vp_k, args.modes)


if __name__ == "__main__":
    main()
