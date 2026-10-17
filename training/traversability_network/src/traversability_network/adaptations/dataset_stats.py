"""Size and mu distribution of a built dataset, logged at startup."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def supervised_mu(dataset):
    """mu of every cell the loss can score, once per (window, horizon state).

    Same observability and BEV-grid test as _sample_path."""
    xb, yb = dataset.grid_bounds["xbound"], dataset.grid_bounds["ybound"]
    offsets = np.arange(dataset.horizon) * dataset.dt
    idx = np.asarray(dataset.indices)
    starts = np.asarray(dataset.bag_sizes)
    bag_of = np.searchsorted(starts, idx, "right") - 1

    out = []
    for i, bag in enumerate(dataset.rosbags):
        w = idx[bag_of == i] - starts[i]
        if not len(w):
            continue
        last = bag.get("frame_stride", dataset.frame_stride) * (dataset.n_frames - 1)
        st = np.asarray(bag["states_timestamp"])
        img = np.asarray(bag["image_timestamp"])
        pos = np.asarray(bag["position"])[:, :2]
        yaw = R.from_quat(bag["quaternion"]).as_euler("ZYX")[:, 0]
        mu = np.asarray(bag["traversability"])[:, 0]
        obs = np.asarray(bag["mu_observability"])

        times = img[w][:, None] + offsets
        sync = np.searchsorted(st, times, "left").clip(max=len(st) - 1)
        anchor = np.searchsorted(st, img[w + last], "left").clip(max=len(st) - 1)

        d = pos[sync] - pos[anchor][:, None]
        c, s = np.cos(yaw[anchor])[:, None], np.sin(yaw[anchor])[:, None]
        x, y = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        keep = (
            (times < st[-1])
            & (obs[sync] > 0.5)
            & (x >= xb[0]) & (x <= xb[1]) & (y >= yb[0]) & (y <= yb[1])
        )
        out.append(mu[sync][keep])
    return np.concatenate(out) if out else np.zeros(0)


def dataset_stats(dataset, n_bins=10):
    """Windows, duration and the mu histogram of mu-observable label states."""
    mu, hours = [], 0.0
    for bag in dataset.rosbags:
        obs = np.asarray(bag["mu_observability"])
        mu.append(np.asarray(bag["traversability"])[:, 0][obs > 0.5])
        ts = np.asarray(bag["image_timestamp"], dtype=float)
        hours += (ts[-1] - ts[0]) / 3600.0
    mu = np.concatenate(mu) if mu else np.zeros(0)
    hist = np.histogram(mu, bins=n_bins, range=(0.0, 1.0))[0]
    return {
        "windows": len(dataset.indices),
        "segments": len(dataset.rosbags),
        "hours": hours,
        "states": int(len(mu)),
        "low_states": int((mu < 0.3).sum()),
        "low_frac": float((mu < 0.3).mean()) if len(mu) else 0.0,
        "mu_mean": float(mu.mean()) if len(mu) else 0.0,
        "hist": hist,
    }


def format_stats(name, s, n_bins=10):
    bw = 1.0 / n_bins
    total = max(s["hist"].sum(), 1)
    bins = "  ".join(
        f"{i * bw:.1f}:{c / total:.3%}" for i, c in enumerate(s["hist"]) if c
    )
    return (
        f"{name}: {s['windows']} windows, {s['segments']} segments, {s['hours']:.2f} h, "
        f"{s['states']} observable states, {s['low_states']} low mu "
        f"({s['low_frac']:.2%}), mean mu {s['mu_mean']:.3f}\n  {bins}"
    )


def log_stats(logger, name, s, n_bins=10):
    """Flatten into the wandb run summary."""
    bw = 1.0 / n_bins
    flat = {f"data/{name}_{k}": v for k, v in s.items() if k != "hist"}
    total = max(s["hist"].sum(), 1)
    for i, c in enumerate(s["hist"]):
        flat[f"data/{name}_mu_{i * bw:.1f}-{(i + 1) * bw:.1f}"] = float(c / total)
    logger.experiment.summary.update(flat)
