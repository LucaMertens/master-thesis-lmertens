"""Post-extraction red-tint correction for the front_center color images.

Some missions (e.g. ARC-5/6/7, SPX-3) have a strong red tint from failed
camera auto-white-balance in dark scenes. The mission's own non-tinted (clean)
frames define the true color balance; we map each tinted frame onto that guide
and match its brightness to the clean frames.

Corrected images overwrite the originals in-place. Originals are backed up to
extracted_raw/<mission>/ before the first overwrite, so re-runs with different
parameters always work from pristine data, never from a previously-corrected file.
Statistics ignore the black borders left by fisheye rectification.
"""

import argparse
import pathlib
import shutil

import cv2
import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done

VERSION = 3
CAM = "front_center"


def masked_means(img: np.ndarray) -> np.ndarray:
    valid = img.any(axis=2)
    if not valid.any():
        return np.zeros(3, np.float64)
    return img[valid].mean(0)


def redness_ratio(means: np.ndarray) -> np.ndarray:
    """Red dominance over green/blue from BGR channel means; >1 means red-tinted."""
    means = np.atleast_2d(means)
    return means[:, 2] / (0.5 * (means[:, 0] + means[:, 1]) + 1e-6)


def correct(output_dir: pathlib.Path, threshold: float, local_window: float, force: bool) -> None:
    if not force and is_step_done(output_dir, "tint_correction", VERSION):
        print(f"Skipping tint_correction (up to date): {output_dir}")
        return

    manifest = np.load(output_dir / "image_manifest.npz")
    fc = manifest[CAM]
    order = np.argsort(fc["ts_ns"])
    ts_ns = fc["ts_ns"][order]
    rel_paths = [str(fc["path"][i]) for i in order]

    # Pristine originals live here; sibling of the extracted/ root.
    backup_dir = output_dir.parent.parent / "extracted_raw" / output_dir.name

    def src(p: str) -> pathlib.Path:
        """Backup if it exists (already-backed-up original), else the live file."""
        bak = backup_dir / p
        return bak if bak.exists() else output_dir / p

    with log_step("tint_correction", output_dir):
        # Remove stale corrected-manifest approach (VERSION < 3).
        for stale in (output_dir / "image_manifest_corrected.npz", output_dir / "images_corrected"):
            if stale.is_file():
                stale.unlink()
            elif stale.is_dir():
                shutil.rmtree(stale)

        means = np.array([
            masked_means(cv2.imread(str(src(p)), cv2.IMREAD_UNCHANGED)) for p in rel_paths
        ])
        ratios = redness_ratio(means)
        to_correct = ratios > threshold
        clean = ~to_correct
        n_correct = int(to_correct.sum())

        if n_correct == 0:
            # Restore any frames corrected by a previous run (gate may have changed).
            if backup_dir.exists():
                for p in rel_paths:
                    bak = backup_dir / p
                    if bak.exists():
                        shutil.copyfile(bak, output_dir / p)
            print(f"  no tinted frames (max ratio {ratios.max():.2f})")
            write_step_done(output_dir, "tint_correction", VERSION, corrected=False, tinted=0)
            return

        if clean.sum() == 0:
            print(f"  {n_correct} tinted but no clean reference; cannot correct")
            write_step_done(output_dir, "tint_correction", VERSION, corrected=False, tinted=n_correct)
            return

        t = ts_ns.astype(np.float64) / 1e9
        clean_means = means[clean]
        clean_chroma = clean_means / clean_means.mean(1, keepdims=True)
        clean_t = t[clean]
        ref_global = np.median(clean_chroma, axis=0)
        ref_lum = float(np.median(clean_means.mean(1)))

        for i, p in enumerate(rel_paths):
            live = output_dir / p
            bak = backup_dir / p
            if not to_correct[i]:
                # Restore from backup if this frame was corrected in a previous run.
                if bak.exists():
                    shutil.copyfile(bak, live)
                continue
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(live, bak)
            img = cv2.imread(str(bak), cv2.IMREAD_UNCHANGED)
            m = means[i]
            L = float(m.mean())
            near = np.abs(clean_t - t[i]) <= local_window
            ref = np.median(clean_chroma[near], axis=0) if near.any() else ref_global
            gains = ref * L / (m + 1e-6) * (ref_lum / (L + 1e-6))
            cv2.imwrite(str(live), np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8))

        print(f"  corrected {n_correct}/{len(means)} tinted frames (clean reference: {int(clean.sum())})")
        write_step_done(output_dir, "tint_correction", VERSION, corrected=True, tinted=n_correct)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--threshold", type=float, default=1.5, help="Redness ratio above which a frame is tinted.")
    p.add_argument("--local-window", type=float, default=5.0, help="Seconds: prefer clean frames within this window.")
    p.add_argument("--force", action="store_true", help="Re-run even if up to date.")
    args = p.parse_args()
    correct(args.output_dir, args.threshold, args.local_window, args.force)


if __name__ == "__main__":
    main()
