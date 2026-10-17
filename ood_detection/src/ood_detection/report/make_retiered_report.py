"""Builds the segment-retiered OOD report (typst) from the CSVs of eval.retiered.

    uv run python -m ood_detection.report.make_retiered_report --features-dir <cache> --out <dir>
"""

import argparse
import os
import subprocess

import pandas as pd

from shared_utils.tiers import ID_TRAIN, tiers
from ood_detection.common.splits import HELDOUT_TIER
from ood_detection.report.plots import (SERIES, aggregator_figure, detection_figure,
                                        shift_class_figure, typicality_all_figure,
                                        typicality_figure)
from ood_detection.report.shading import fill, legend

# shift tiers in the order of shared_utils/tiers.json
TIERS = [t for t in tiers() if t not in (ID_TRAIN, HELDOUT_TIER)]


def _tier_label(t):
    """Near/Roots, Far/Construction Outdoor."""
    head, _, rest = t.partition("_")
    tail = " ".join(w.capitalize() for w in rest.split("_"))
    return f"{head.capitalize()}/{tail}" if tail else head.capitalize()


TIER_LABEL = {t: _tier_label(t) for t in tiers()}
TIER_LABEL[HELDOUT_TIER] = "Held-out ID"
TIER_LABEL["far_construction_outdoor"] = "Far/Construction"
# break at the slash for axis ticks
TIER_TICK = {k: v.replace("/", "\n") for k, v in TIER_LABEL.items()}
# short table headers (nine columns, no hyphenation)
TIER_HEAD = {**TIER_LABEL,
             "near_roots": "Roots", "near_undergrowth": "Undergr.",
             "near_meadow": "Meadow", "medium_alpine": "Alpine",
             "far_construction_outdoor": "Constr.", "far_indoor": "Indoor",
             "far_urban": "Urban", "far_snow": "Snow"}
# Reporting view only (no retiering): tiers grouped by input shift vs traversability shift
# (columns of documentation/ood_split.md)
SHIFT_CLASSES = {
    "silent_hazard": ["near_roots", "near_undergrowth"],
    "benign_novelty": ["near_meadow", "far_indoor", "far_urban"],
    "both": ["medium_alpine", "far_construction_outdoor", "far_snow"],
}
CLASS_LABEL = {"silent_hazard": "Silent Hazard", "benign_novelty": "Benign Novelty",
               "both": "Both"}
CLASS_GLOSS = {"silent_hazard": "looks familiar, less traversable",
               "benign_novelty": "looks new, just as traversable",
               "both": "looks new and less traversable"}
RAW_NOTE_HEAD = "; ".join(f"{CLASS_LABEL[k]}: {CLASS_GLOSS[k]}"
                          for k in ("silent_hazard", "benign_novelty"))
RAW_NOTE = (RAW_NOTE_HEAD + ".\nRaw AUROC against the held-out in-distribution windows, "
            "with no correction for what\nrecording identity alone scores. 0.5 is chance.")
CLASS_NOTE = ("; ".join(f"{CLASS_LABEL[k]}: {CLASS_GLOSS[k]}"
                        for k in ("silent_hazard", "benign_novelty"))
              + ".\nBaseline: AUROC between two in-distribution recordings of the same "
                "terrain.\nAt 0 a detector is only telling the recordings apart.")
DISTANCE = ["bev_mahalanobis", "bev_euclidean", "bev_cosine",
            "image_mahalanobis", "image_euclidean", "image_cosine",
            "dino_mahalanobis", "dino_euclidean", "dino_cosine"]
# no in-distribution fit, so no feature space and no aggregator
FREE = ["viewpoint_range_med", "viewpoint_range", "viewpoint_disagreement",
        "viewpoint_variance", "viewpoint_variance_med",
        "modality_disagreement", "depth_entropy", "predicted_trav"]
DETECTORS = DISTANCE + FREE
FIGURE_DETECTORS = [k for k, *_ in SERIES]
DET_LABEL = {
    "bev_mahalanobis": "BEV Mahalanobis++",
    "bev_euclidean": "BEV k-means",
    "bev_cosine": "BEV cosine",
    "image_mahalanobis": "Encoder Mahalanobis++",
    "image_euclidean": "Encoder k-means",
    "image_cosine": "Encoder cosine",
    "dino_mahalanobis": "DINOv3 Mahalanobis++",
    "dino_euclidean": "DINOv3 k-means",
    "dino_cosine": "DINOv3 cosine",
    "viewpoint_disagreement": "viewpoint disagreement",
    "viewpoint_range": "viewpoint range",
    "viewpoint_range_med": "viewpoint range (median)",
    "viewpoint_variance": "viewpoint variance",
    "viewpoint_variance_med": "viewpoint variance (median)",
    "modality_disagreement": "modality disagreement",
    "depth_entropy": "depth entropy",
    "predicted_trav": "predicted traversability",
}
SHORT = {**{d: DET_LABEL[d].split(" ", 1)[1] for d in DISTANCE},
         "bev_mahalanobis": "Mahalanobis++", "dino_mahalanobis": "Mahalanobis++",
         "image_mahalanobis": "Mahalanobis++",
         "viewpoint_disagreement": "viewpoint", "viewpoint_variance": "vp variance",
         "viewpoint_range": "vp range", "viewpoint_range_med": "vp range med.",
         "viewpoint_variance_med": "vp var. med.", "modality_disagreement": "modality",
         "depth_entropy": "depth entropy", "predicted_trav": "predicted trav."}
METRICS = [("auroc", "AUROC", "higher"), ("fpr95", "FPR95", "lower"), ("aupr", "AUPR", "higher")]


def boxed(body):
    """Keeps a table and its legend on one page with smaller text."""
    return ("#block(breakable: false)[\n#set text(size: 8.5pt)\n#set par(justify: false)\n"
            + body + "\n]")


def table(head, body, n):
    return (f'#table(\n  columns: (auto, ..({n} * (1fr,))), align: (left, ..({n} * (right,))),\n'
            f'  stroke: none, inset: (x: 4pt, y: 5pt),\n  table.hline(),\n  [], {head},\n'
            f'  table.hline(stroke: 0.5pt),\n' + "\n".join(body) + "\n  table.hline(),\n")


def detection_table(res, metric, better, base_rates, nulls):
    """Rows detector, columns tier. Shading reference: session floor (AUROC), base rate (AUPR), column best (FPR95)."""
    wide = res.pivot_table(index="detector", columns="tier", values=metric).reindex(
        index=[d for d in DETECTORS if d in set(res.detector)],
        columns=[t for t in TIERS if t in set(res.tier)])
    body = []
    for det, row in wide.iterrows():
        cells = []
        for t, v in row.items():
            if pd.isna(v):
                cells.append("[--]")
            elif metric == "aupr":
                cells.append(fill(f"{v:.3f}", v - base_rates[t]))
            elif metric == "auroc":
                cells.append(fill(f"{v:.3f}", v - nulls[det]))
            else:
                cells.append(fill(f"{v:.3f}", wide[t].min() - v))
        body.append(f'  [{DET_LABEL[det]}], ' + ", ".join(cells) + ",")
    head = ", ".join(f"[*{TIER_HEAD[c]}*]" for c in wide.columns)
    ref = {"auroc": "that detector's session floor, the mean AUROC between two "
                    "in-distribution recordings",
           "aupr": "the tier's base rate, what a no-skill detector scores",
           "fpr95": "the best value in the column"}[metric]
    arrow = ("higher is better" if better == "higher" else "lower is better") + \
        ". Columns are the shift tiers in the order of @fig-detection"
    n = len(wide.columns)
    return boxed(table(head, body, n) + "  "
                 + legend(f"Shaded against {ref}; {arrow}.", n + 1, "worse", "better") + ")")


def typicality_table(typ, group, train_missions):
    """Rows ID mission, columns detector. Only the bad half is tinted."""
    wide = typ.pivot_table(index="mission", columns="detector", values="auroc_vs_id")
    keep = FREE if group == "free" else [d for d in DISTANCE if d.startswith(group)]
    cols = [d for d in keep if d in wide.columns]
    held = [m for m in wide.index if m not in train_missions]
    body = []
    for group, missions in (("train (inside the fit set)", sorted(train_missions)),
                            ("held out", held)):
        body.append(f'  table.cell(colspan: {len(cols) + 1})[_{group}_],')
        for m in missions:
            cells = [fill(f"{wide[c][m]:.3f}", 0.5 - wide[c][m], full=0.25, one_sided=True)
                     for c in cols]
            body.append(f'  [{m}], ' + ", ".join(cells) + ",")
    head = ", ".join(f'[*{SHORT[c].replace("Mahalanobis++", "Mah.")}*]' for c in cols)
    cap = "Shaded above 0.5 only: a mission scoring below its peers is not a problem."
    return boxed(table(head, body, len(cols)) + "  "
                 + legend(cap, len(cols) + 1, "typical (0.5 or below)", "reads as OOD",
                          one_sided=True, full=0.25, reverse=True) + ")")


def raw_table(raw):
    """Median window score, shaded by its distance from the held-out ID median."""
    tiers = ["id_heldout"] + [t for t in TIERS if t in set(raw.tier)]
    wide = raw.pivot_table(index="detector", columns="tier", values="p50").reindex(
        index=[d for d in DETECTORS if d in set(raw.detector)], columns=tiers)
    body = []
    for det, row in wide.iterrows():
        ref = row["id_heldout"]
        cells = [f"[#text(fill: luma(90))[{ref:.3g}]]"] + [
            "[--]" if pd.isna(v) else fill(f"{v:.3g}", (v - ref) / ref, full=0.5)
            for t, v in row.items() if t != "id_heldout"]
        body.append(f'  [{DET_LABEL[det]}], ' + ", ".join(cells) + ",")
    head = ", ".join(f"[*{TIER_HEAD[t]}*]" for t in tiers)
    cap = ("Shaded by the relative distance from that detector's held-out ID median; "
           "further from the ID mass is better. Columns are the shift tiers in the order "
           "of @fig-detection.")
    n = len(tiers)
    return boxed(table(head, body, n) + "  " + legend(cap, n + 1, "closer", "further") + ")")


MEDIAN = ("quantile", 0.5)


def agg_label(kind, q):
    if kind == "mean":
        return "mean of all cells"
    return "median" if q == 0.5 else f"q{round(q * 100):02d}"


def sweep_table(sweep):
    """Mean AUROC over tiers per aggregator, shaded against the median of that column."""
    g = sweep[sweep["space"] == "bev"].groupby(
        ["metric", "kind", "q"])["auroc"].mean().reset_index()
    cols = ["mahalanobis", "euclidean", "cosine"]
    head_label = {"mahalanobis": "Mahalanobis++", "euclidean": "k-means", "cosine": "cosine"}
    wide = g.pivot_table(index=["kind", "q"], columns="metric", values="auroc")
    order = ([("quantile", q) for q in sorted(g[g["kind"] == "quantile"]["q"].unique())]
             + [("mean", 1.0)])
    body = []
    for spec in order:
        cells = []
        for c in cols:
            v = wide[c][spec]
            cells.append(f"[#text(fill: luma(90))[{v:.3f}]]" if spec == MEDIAN
                         else fill(f"{v:.3f}", v - wide[c][MEDIAN], dead=0.005, full=0.05))
        body.append(f'  [{agg_label(*spec)}], ' + ", ".join(cells) + ",")
    head = ", ".join(f"[*BEV {head_label[c]}*]" for c in cols)
    cap = ("Shaded against the median of the same column, on a scale ten times finer than "
           "the other tables; higher is better.")
    return boxed(table(head, body, len(cols)) + "  "
                 + legend(cap, len(cols) + 1, "worse", "better", full=0.05) + ")")


def counts_table(res):
    counts = res.groupby("tier")[["n_ood", "base_rate"]].max()
    rows = "\n  ".join(f"[{TIER_LABEL[t]}], [{int(counts.n_ood[t])}], [{counts.base_rate[t]:.3f}],"
                       for t in TIERS if t in counts.index)
    return (f'#table(\n  columns: 3, align: (left, right, right), stroke: none, inset: 5pt,\n'
            f'  table.hline(),\n  [], [*windows*], [*base rate*],\n'
            f'  table.hline(stroke: 0.5pt),\n  {rows}\n  table.hline(),\n)')


PREAMBLE = """#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, hyphenate: false)
#set par(justify: true)
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em)[#text(size: 13pt)[#it.body]]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em)[#text(size: 11pt)[#it.body]]
"""


def bullets(items):
    return "".join(f"- {t}\n" for t in items)


def typicality_bullets(typ, train_missions):
    wide = typ.pivot_table(index="mission", columns="detector", values="auroc_vs_id")
    held = wide.loc[[m for m in wide.index if m not in train_missions]]
    out = []
    for group, label in (("dino", "DINOv3"), ("bev", "BEV")):
        cols = [c for c in held.columns if c.startswith(group)]
        row = held[cols].mean(axis=1).sort_values(ascending=False)
        out.append(f"{row.index[0]} is the least typical held-out mission in {label} space, "
                   f"at {row.iloc[0]:.3f}; {row.index[-1]} the most, at {row.iloc[-1]:.3f}.")
    n_bad = int((held.mean(axis=1) > 0.5).sum())
    out.append(f"{n_bad} of {len(held)} held-out missions average above 0.5, which puts a "
               "floor under every tier number computed against them.")
    out.append("Train missions sit inside each detector's fit set, so only the held-out "
               "rows compare with each other.")
    return bullets(out)


def detection_bullets(res, nulls):
    """Which detectors clear their own session floor, and where."""
    wide = res.pivot_table(index="detector", columns="tier", values="auroc").reindex(
        columns=[t for t in TIERS if t in set(res.tier)])
    lift = wide.sub(pd.Series(nulls), axis=0)
    shown = [d for d in lift.index if d in set(FIGURE_DETECTORS)]
    out = []
    clean = [DET_LABEL[d] for d in shown if (lift.loc[d] > 0).all()]
    joined = " and ".join(filter(None, [", ".join(clean[:-1]), clean[-1]])) if clean else ""
    out.append(f"{joined} clear{'s' if len(clean) == 1 else ''} the session floor on all "
               f"{len(lift.columns)} tiers." if clean else
               "No detector clears its session floor on every tier.")
    best = lift.loc[shown].idxmax()
    for det in dict.fromkeys(best):
        ts = [t for t in lift.columns if best[t] == det]
        gains = ", ".join(f"{TIER_LABEL[t]} ({lift.loc[det, t]:+.3f})" for t in ts)
        out.append(f"{DET_LABEL[det]} leads on {gains}.")
    hardest = lift.loc[shown].max().idxmin()
    n_above = int((lift.loc[shown, hardest] > 0).sum())
    out.append(f"{TIER_LABEL[hardest]} is the hardest tier: {n_above} of {len(shown)} shown "
               f"detectors are above the floor at all, the best by "
               f"{lift.loc[shown, hardest].max():+.3f}.")
    base = "predicted_trav"
    if base in lift.index:
        out.append(f"Predicted traversability has the highest floor of any detector, "
                   f"{nulls[base]:.3f}, and falls below it on "
                   f"{int((lift.loc[base] < 0).sum())} of {len(lift.columns)} tiers: the "
                   "baseline is a per-recording offset. It is left out of the figure.")
    for space in ("bev", "dino"):
        d = lift.loc[f"{space}_mahalanobis"] - lift.loc[f"{space}_euclidean"]
        verb = "helps" if d.mean() > 0 else "hurts"
        out.append(f"The shared covariance {verb} in {'BEV' if space == 'bev' else 'DINOv3'} "
                   f"space, by {abs(d.mean()):.3f} AUROC averaged over tiers.")
    return bullets(out)


def aggregator_bullets(sweep, n_tiers):
    g = sweep[(sweep["space"] == "bev") & (sweep["metric"] == "mahalanobis")].groupby(
        ["kind", "q"])["auroc"].mean()
    flat = g["quantile"][(g["quantile"].index >= 0.4) & (g["quantile"].index <= 0.8)]
    q99 = g["quantile"].get(0.99)
    med = g["quantile"][0.5]
    return bullets([
        "One parameter with one meaning. The median is $q_(50)$, and a localized hazard is "
        "the same statistic at a higher quantile.",
        "Less sensitive to the cells the pipeline cannot fully trust, behind occluders or at the "
        "frustum edge, which move a mean and not a median.",
        f"Nothing to tune. Mahalanobis++ is flat from $q_(40)$ to $q_(80)$, {flat.min():.3f} "
        f"to {flat.max():.3f}, and two independent extractions of this checkpoint disagree "
        "by up to 0.005, so taking the argmax would be fitting the benchmark.",
        f"The upper tail is the worst choice: $q_(99)$ costs {med - q99:.2f} against the "
        "median.",
        f"This benchmark cannot settle it. All {n_tiers} tiers are whole-scene shifts, which "
        "reward any bulk statistic.",
    ])


def shift_class_table(res, nulls):
    """Mean lift over the session floor per shift class."""
    wide = res.pivot_table(index="detector", columns="tier", values="auroc")
    lift = wide.sub(pd.Series(nulls), axis=0)
    keys = list(SHIFT_CLASSES)
    body = []
    for det in [d for d in DETECTORS if d in lift.index]:
        cells = [fill(f"{lift.loc[det, SHIFT_CLASSES[k]].mean():+.3f}",
                      lift.loc[det, SHIFT_CLASSES[k]].mean(), dead=0.0, full=0.25)
                 for k in keys]
        body.append(f'  [{DET_LABEL[det]}], ' + ", ".join(cells) + ",")
    head = ", ".join(f"[*{CLASS_LABEL[k]}*]" for k in keys)
    cap = ("Mean AUROC minus that detector's session floor, over the tiers of each class ("
           + "; ".join(f"{CLASS_LABEL[k]}: {CLASS_GLOSS[k]}" for k in keys)
           + "). Above zero the detector sees more than recording identity.")
    return boxed(table(head, body, len(keys)) + "  "
                 + legend(cap, len(keys) + 1, "below floor", "above floor", full=0.25) + ")")


def shift_class_bullets(res, nulls):
    wide = res.pivot_table(index="detector", columns="tier", values="auroc")
    lift = wide.sub(pd.Series(nulls), axis=0)
    shown = [d for d in FIGURE_DETECTORS if d in lift.index]
    h = {d: lift.loc[d, SHIFT_CLASSES["silent_hazard"]].mean() for d in shown}
    b = {d: lift.loc[d, SHIFT_CLASSES["benign_novelty"]].mean() for d in shown}
    above = [d for d in shown if h[d] > b[d] and h[d] > 0]
    best_h = max(shown, key=lambda d: h[d])
    best_b = max(shown, key=lambda d: b[d])
    out = [
        f"{DET_LABEL[best_h]} is the strongest on terrain that looks familiar and is less "
        f"traversable ({h[best_h]:+.3f}), {DET_LABEL[best_b]} on terrain that looks new and "
        f"is just as traversable ({b[best_b]:+.3f}). They are not the same detector.",
        f"{DET_LABEL[best_b]} scores {b[best_b]:+.3f} on the new-looking terrain against "
        f"{h[best_b]:+.3f} on the less traversable terrain: it is ordered by how far the "
        "image moved, which is the quantity it was trained on and not the one the robot "
        "needs.",
    ]
    names = [DET_LABEL[d] for d in above]
    out.append((" and ".join(filter(None, [", ".join(names[:-1]), names[-1]]))
                + (" sits" if len(names) == 1 else " sit")
                + " above the diagonal, answering the terrain question more strongly "
                  "than the view question.")
               if above else "No detector sits above the diagonal.")
    both = {d: lift.loc[d, SHIFT_CLASSES["both"]].mean() for d in shown}
    n_pos = sum(v > 0 for v in both.values())
    out.append(f"Where the terrain both looks new and is less traversable, {n_pos} of "
               f"{len(shown)} shown detectors are "
               "above their floor, but a detection there cannot be attributed to either "
               "cause, which is why the class is reported separately rather than pooled.")
    return bullets(out)


def build(features_dir, out_dir, checkpoint, n_modes):
    ev = os.path.join(features_dir, "eval")
    res = pd.read_csv(os.path.join(ev, "retiered_results.csv"))
    typ = pd.read_csv(os.path.join(ev, "retiered_typicality.csv"))
    raw = pd.read_csv(os.path.join(ev, "retiered_raw.csv"))
    sweep = pd.read_csv(os.path.join(ev, "retiered_aggregator.csv"))
    typ_all = pd.read_csv(os.path.join(ev, "retiered_typicality_all.csv"))
    base_rates = res.groupby("tier")["base_rate"].max().to_dict()
    nulls = res.groupby("detector")["null_mean"].max().to_dict()
    train_missions = set(typ[typ.split == "train"]["mission"])
    os.makedirs(out_dir, exist_ok=True)

    typicality_figure(typ, os.path.join(out_dir, "typicality.pdf"))
    typicality_all_figure(typ_all, ["id_train", "id_heldout"] + TIERS,
                          {**TIER_LABEL, "id_train": "in-distribution, train",
                           "id_heldout": "in-distribution, held out"},
                          os.path.join(out_dir, "typicality_all.pdf"))
    detection_figure(res, TIERS, TIER_TICK, os.path.join(out_dir, "detection.pdf"))
    shift_class_figure(res, nulls, SHIFT_CLASSES, CLASS_LABEL, DET_LABEL,
                       os.path.join(out_dir, "shift_classes.pdf"), CLASS_NOTE)
    # raw AUROC variant
    shift_class_figure(res, nulls, SHIFT_CLASSES, CLASS_LABEL, DET_LABEL,
                       os.path.join(out_dir, "shift_classes_raw.pdf"), RAW_NOTE,
                       baseline=False)
    aggregator_figure(sweep, os.path.join(out_dir, "aggregator.pdf"))

    doc = [PREAMBLE,
           '#align(center, block(width: 88%)[#set par(justify: false)\n'
           '#text(size: 16pt)[*Out-of-distribution detection under segment-level tiers*]\n'
           '#v(0.5em)\n'
           f'#text(size: 9pt)[checkpoint `{checkpoint}`, $k = {n_modes}$ modes]])\n#v(0.6em)\n']

    doc.append("= In-distribution typicality\n")
    doc.append('#figure(image("typicality.pdf", width: 100%), caption: [Each ID mission '
               'scored against every other ID window. Above 0.5 the mission reads as more '
               'out-of-distribution than its own peers, which puts a floor under every tier '
               'number computed against it.]) <fig-typicality>\n')
    doc.append(typicality_bullets(typ, train_missions))

    doc.append("= Detection\n")
    doc.append('#figure(image("detection.pdf", width: 100%), caption: [AUROC per tier, and '
               'the same values as a lift over the session floor. The floor is the mean '
               'folded AUROC between two in-distribution recordings, so it is what '
               'recording identity alone buys on that detector. Cosine is left out of the '
               'figure, sitting within 0.002 of k-means on every tier.]) <fig-detection>\n')
    doc.append("- Cosine is left out of the figure, and one representative per distance "
               "family is shown; the full grid is in the appendix.\n"
               + detection_bullets(res, nulls))

    doc.append("= What the detectors are ordered by\n")
    doc.append("The tier table places every tier on two shifts that do not move together: "
               "how far the image is from the training images, and how far the ground "
               "truth is from what the estimator learned. Reporting near to far orders the "
               "tiers by the first of them, so a detector that reads appearance tracks that "
               "order by construction. Grouping the same tiers by which shift is the large "
               "one separates the two questions. Nothing is rescored and "
               "`shared_utils/tiers.json` is unchanged; this is a second reading of "
               "@fig-detection.\n")
    doc.append('#figure(image("shift_classes.pdf", width: 82%), caption: [Mean lift over '
               'the session floor on the tiers where only the terrain shifted, against the '
               'tiers where only the view did. Above the dashed diagonal a detector answers '
               'the question the robot has; below it, the one the camera has. Predicted '
               'traversability is a baseline rather than a detector and is left out.]) '
               '<fig-shift-classes>\n')
    doc.append(shift_class_bullets(res, nulls))
    doc.append(shift_class_table(res, nulls) + "\n")
    doc.append("The classes follow the two columns of the tier table. Terrain-only: "
               + ", ".join(TIER_LABEL[t] for t in SHIFT_CLASSES["silent_hazard"])
               + ". View-only: "
               + ", ".join(TIER_LABEL[t] for t in SHIFT_CLASSES["benign_novelty"])
               + ". Both: "
               + ", ".join(TIER_LABEL[t] for t in SHIFT_CLASSES["both"]) + ".\n")

    doc.append("= Raw scores\n")
    doc.append(raw_table(raw) + "\n")

    doc.append("= Appendix: tiers\n")
    doc.append(counts_table(res) + "\n")
    doc.append("= Appendix: detection\n")
    for metric, label, better in METRICS:
        doc.append(f"== {label}\n")
        doc.append(detection_table(res, metric, better, base_rates, nulls) + "\n")

    doc.append("= Appendix: every mission and mission part\n")
    multi = sorted(m for m, n in typ_all.groupby("mission")["tier"].nunique().items() if n > 1)
    doc.append('#figure(image("typicality_all.pdf", width: 100%), caption: [The typicality '
               'reading of @fig-typicality applied to every mission, grouped by the tier it '
               'sits in and scored against every in-distribution window. A recording split '
               'across tiers contributes one row per part, so '
               + ", ".join(multi) + ' appear more than once.]) <fig-typicality-all>\n')

    doc.append("= Appendix: in-distribution typicality\n")
    for group, label in (("bev", "BEV"), ("dino", "DINOv3"), ("free", "Reference-free")):
        doc.append(f"== {label}\n")
        doc.append(typicality_table(typ, group, train_missions) + "\n")

    doc.append("= Appendix: aggregator\n")
    doc.append('#figure(image("aggregator.pdf", width: 100%), caption: [Mean AUROC '
               'over the shifted tiers across the quantiles of a window\'s cell '
               'distances. Stars mark the median, the score reported everywhere else; '
               'dotted lines are the mean of all cells. Cosine lies underneath k-means. '
               'DINOv3 is pooled to one vector per window at extraction, so it has no '
               'aggregator and is left out.]) <fig-aggregator>\n')
    doc.append(sweep_table(sweep) + "\n")
    doc.append("Why the median:\n\n" + aggregator_bullets(sweep, len(TIERS)))

    path = os.path.join(out_dir, "ood_retiered_report.typ")
    with open(path, "w") as f:
        f.write("\n".join(doc))
    subprocess.run(["typst", "compile", path], check=True)
    print(f"report -> {path.replace('.typ', '.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="gt_human_mt_ineq075_f2 (owbc12tm)")
    ap.add_argument("--modes", type=int, default=3)
    args = ap.parse_args()
    build(args.features_dir, args.out, args.checkpoint, args.modes)


if __name__ == "__main__":
    main()
