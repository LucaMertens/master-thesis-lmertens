"""W-RIZZ's own network, loss, metric and augmentations trained on our human labels (image space).

Run: uv run python side_experiments/wrizz_image/train.py --mean-teacher
"""

import argparse
import copy
import math
import os
import random
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as data_mod
from traversability_network.config import get_cfg

WRIZZ_REPO = os.path.expanduser("~/git-repos/W-RIZZ")
assert os.path.isdir(WRIZZ_REPO), f"clone W-RIZZ to {WRIZZ_REPO} first"
sys.path.insert(0, WRIZZ_REPO)
import augmentations as wr_aug  # noqa: E402
from architectures.travnet import TravNetUp3NNRGB  # noqa: E402
from losses import LRIZZLoss  # noqa: E402
from metrics import SparseDisagreementScore  # noqa: E402

TAUS = (0.1, 0.25, 0.5)
HEADLINE_TAU = 0.25


# W-RIZZ ColorJitter args (train.py:82, :161)
JITTER = ((0.75, 1.75), 0.5, 0.5, 0.1)


class Augmentation:
    """Sparse flip, sparse crop, optional colour jitter. A class so it pickles to workers."""

    def __init__(self, jitter=JITTER):
        self.flip = wr_aug.SparseHorizontalFlip()
        self.crop = wr_aug.SparseRandomCrop()
        self.jitter = (wr_aug.ImageOnlyAugment(torchvision.transforms.ColorJitter(*jitter))
                       if jitter else None)

    def __call__(self, data):
        data = self.crop(self.flip(data))
        return self.jitter(data) if self.jitter is not None else data


class Ema:
    """Teacher EMA over parameters only (no torch_ema). BatchNorm buffers stay the teacher's own."""

    def __init__(self, model, decay=0.99):
        self.teacher = copy.deepcopy(model).requires_grad_(False).train()
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for t, s in zip(self.teacher.parameters(), model.parameters()):
            t.lerp_(s.float(), 1 - d)


def predict(net, image_a, image_b):
    """Stack the pair, run as one batch, unflatten to [B,2,H,W]."""
    b = image_a.shape[0]
    out = net(torch.stack([image_a, image_b], dim=1).flatten(0, 1))
    return {k: v.unflatten(0, (b, 2)) for k, v in out.items()}


@torch.no_grad()
def evaluate(net, loader, device):
    """HDR per tau, split by tie/inequality, plus the balanced headline."""
    net.eval()
    scores = {(tau, mode): SparseDisagreementScore(tau, mode)
              for tau in TAUS for mode in (None, "eq", "neq")}
    n = 0
    for item in loader:
        image_a, image_b, anno = item[0].to(device), item[1].to(device), item[-2].to(device)
        pred = predict(net, image_a, image_b)["prediction"]
        for m in scores.values():
            m(pred, anno)
        n += anno.shape[0]
    out = {}
    for (tau, mode), m in scores.items():
        name = {None: "", "eq": "_eq", "neq": "_ineq"}[mode]
        try:
            out[f"hdr{name}_{tau:g}"] = float(m.compute())
        except ValueError:
            pass
    # ~55% of answers are ties, so the raw rate is dominated by them
    out["hdr"] = 0.5 * (out[f"hdr_eq_{HEADLINE_TAU:g}"] + out[f"hdr_ineq_{HEADLINE_TAU:g}"])
    out["pairs"] = n
    net.train()
    return out


def rebalanced(dataset):
    """Oversample the minority class (tie vs inequality) up to the majority."""
    eq = np.flatnonzero(dataset.labels == 0)
    ineq = np.flatnonzero(dataset.labels != 0)
    minority, majority = (ineq, eq) if len(ineq) < len(eq) else (eq, ineq)
    extra = np.random.default_rng(0).choice(minority, len(majority) - len(minority))
    return np.concatenate([np.arange(len(dataset)), extra])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_file", default="training/traversability_network/configs/gt_human_f2.yaml")
    ap.add_argument("--tag", default="wrizz_image_f2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)   # pairs, so 16 images
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay_step", type=int, default=20)
    ap.add_argument("--lr_decay_gamma", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.5)   # W-RIZZ's l
    ap.add_argument("--mean-teacher", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--rebalance", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=0,
                    help="region pixel pairs drawn per item; 0 uses the clicked pixel")
    ap.add_argument("--ckpt_dir", default="/datahome/lmertens/checkpoints")
    ap.add_argument("--ramp_epochs", type=int, default=10)
    ap.add_argument("--split", choices=("mission", "iid"), default="mission",
                    help="mission = hold out whole missions; iid = random frame-level "
                         "split of every mission (W-RIZZ's setting)")
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true",
                    help="continue this tag from last.pt and its wandb run "
                         "(optimizer state is not checkpointed)")
    args = ap.parse_args()

    # determinism, as in W-RIZZ train.py:43
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = get_cfg(args.cfg_file)
    device = args.device
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # with mean teacher the jitter becomes the per-view consistency augmentation
    aug = Augmentation(jitter=None if args.mean_teacher else JITTER)
    consistency = (transforms.RandomApply([transforms.ColorJitter(*JITTER)], p=0.5)
                   if args.mean_teacher else None)

    if args.split == "iid":
        # all missions on both sides, split on frames
        missions = list(cfg.DATASET.TRAIN_MISSIONS) + list(cfg.DATASET.VALID_MISSIONS)
        all_frames = sorted(data_mod.PairDataset(cfg, missions, normalize=normalize).frame_set())
        rng = np.random.default_rng(args.seed)
        cut = int(round((1 - args.valid_frac) * len(all_frames)))
        order = rng.permutation(len(all_frames))
        train_frames = {all_frames[i] for i in order[:cut]}
        valid_frames = {all_frames[i] for i in order[cut:]}
        train_missions = valid_missions = missions
        print(f"iid split: {len(train_frames)} train frames, {len(valid_frames)} valid")
    else:
        train_missions, valid_missions = cfg.DATASET.TRAIN_MISSIONS, cfg.DATASET.VALID_MISSIONS
        train_frames = valid_frames = None
    trainable = None if args.split == "iid" else True

    train_set = data_mod.PairDataset(cfg, train_missions, trainable=trainable,
                                     augmentation=aug, normalize=normalize,
                                     n_samples=args.n_samples, frames=train_frames,
                                     consistency_augmentation=consistency)
    # always scored on the clicked pixel, whatever the training draw
    eval_train_set = data_mod.PairDataset(cfg, train_missions, table=train_set.table,
                                          trainable=trainable, normalize=normalize,
                                          frames=train_frames)
    valid_set = data_mod.PairDataset(cfg, valid_missions, table=train_set.table,
                                     source="random", normalize=normalize,
                                     frames=valid_frames)
    print(f"train pairs {len(train_set)} ({int((train_set.labels == 0).sum())} tie / "
          f"{int((train_set.labels != 0).sum())} ineq), valid pairs {len(valid_set)}")

    sampler = None
    if args.rebalance:
        idx = rebalanced(train_set)
        sampler = torch.utils.data.SubsetRandomSampler(idx.tolist())
        print(f"rebalanced train length {len(idx)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, num_workers=args.workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)

    net = TravNetUp3NNRGB(pretrained=True, output_size=data_mod.RESOLUTION,
                          output_channels=1).to(device)
    ema = Ema(net) if args.mean_teacher else None
    if ema is not None:
        ema.teacher.to(device)
    # the teacher is validated and saved, not the student
    scored = (lambda: ema.teacher) if ema is not None else (lambda: net)
    loss_fn = LRIZZLoss(l=args.margin)
    consistency_loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.lr_decay_step, args.lr_decay_gamma)
    ramp_epochs = args.ramp_epochs

    ckpt_dir = os.path.join(args.ckpt_dir, args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    best = float("inf")

    start_epoch, resume_id = 0, None
    if args.resume:
        ckpt = torch.load(os.path.join(ckpt_dir, "last.pt"), map_location=device,
                          weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        if ema is not None:
            ema.teacher.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        bp = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(bp):
            best = torch.load(bp, map_location="cpu", weights_only=False)["hdr"]
        import wandb as _w
        found = _w.Api().runs("luca-mertens-kiel-university/master-thesis-lmertens",
                              filters={"display_name": args.tag})
        resume_id = found[0].id if len(found) else None
        print(f"resuming {args.tag} from epoch {start_epoch} (best {best:.4f}, "
              f"wandb {resume_id})")

    import wandb
    run = wandb.init(project="master-thesis-lmertens", name=args.tag,
                     dir="/datahome/lmertens",
                     **({"id": resume_id, "resume": "must"} if resume_id else {}),
                     config={**vars(args), "resolution": data_mod.RESOLUTION,
                             "n_train": len(train_set), "n_valid": len(valid_set)},
                     notes="W-RIZZ network, loss, metric and augmentations on our human labels.")
    for epoch in range(start_epoch, args.epochs):
        # beta ramps as exp(-5(1-e/R)^2); EMA decay is slower after the ramp
        ramp = 0.0
        if args.mean_teacher:
            e = epoch + 1
            ramp = math.exp(-5 * (1 - e / max(ramp_epochs, 1)) ** 2) if e <= ramp_epochs else 1.0
            ema.decay = 0.99 if e <= ramp_epochs else 0.999
        totals = {"acc": 0.0, "con": 0.0, "n": 0}
        for item in train_loader:
            anno = item[-2].to(device)
            image_a, image_b = item[0].to(device), item[1].to(device)
            pred = predict(net, image_a, image_b)
            acc = loss_fn(predictions=pred, targets=anno)
            loss = args.alpha * acc
            con = torch.zeros((), device=device)
            if ema is not None:
                # teacher gets its own photometric view
                with torch.no_grad():
                    teacher = predict(ema.teacher, item[2].to(device), item[3].to(device))
                con = consistency_loss(pred["prediction"], teacher["prediction"])
                loss = loss + ramp * args.beta * con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(net)
            totals["acc"] += float(acc.detach())
            totals["con"] += float(con.detach())
            totals["n"] += 1
        sched.step()
        train_hdr = evaluate(scored(), DataLoader(eval_train_set, batch_size=args.batch_size,
                                             num_workers=args.workers), device)
        valid_hdr = evaluate(scored(), valid_loader, device)
        log = {"epoch": epoch, "lr": sched.get_last_lr()[0],
               "human/train_loss": totals["acc"] / max(totals["n"], 1),
               "train/mt_consistency": totals["con"] / max(totals["n"], 1)}
        log |= {f"human/valid_{k}": v for k, v in valid_hdr.items()}
        log |= {f"human/train_{k}": v for k, v in train_hdr.items()}
        run.log(log)
        # weights only; best and last
        if valid_hdr["hdr"] < best:
            best = valid_hdr["hdr"]
            torch.save({"epoch": epoch, "hdr": best, "state_dict": scored().state_dict()},
                       os.path.join(ckpt_dir, "best.pt"))
        torch.save({"epoch": epoch, "state_dict": scored().state_dict()},
                   os.path.join(ckpt_dir, "last.pt"))
        print(f"epoch {epoch:3d}  loss {log['human/train_loss']:.4f}  "
              f"train_hdr {train_hdr['hdr']:.4f}  valid_hdr {valid_hdr['hdr']:.4f}  "
              f"(eq {valid_hdr['hdr_eq_0.25']:.3f} / ineq {valid_hdr['hdr_ineq_0.25']:.3f})",
              flush=True)
    run.finish()


if __name__ == "__main__":
    main()
