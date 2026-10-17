"""Mu loss variants and loss reduction added on top of the weighted smooth-L1 trav_criterion."""
import math

import torch
import torch.nn.functional as F


def mu_two_hot(mu, K):
    """Soft two-hot encoding of continuous mu over the K bin centers."""
    x = (mu * K - 0.5).clamp(0, K - 1)
    lo = x.floor()
    frac = (x - lo).unsqueeze(-1)
    lo = lo.long().unsqueeze(-1)
    hi = (lo + 1).clamp(max=K - 1)
    target = torch.zeros(*mu.shape, K, device=mu.device, dtype=mu.dtype)
    target.scatter_(-1, lo, 1 - frac)
    target.scatter_add_(-1, hi, frac)
    return target


def mu_gaussian(mu, K, sigma):
    """Gaussian target over the K bin centers, normalized to sum to one."""
    centers = (torch.arange(K, device=mu.device, dtype=mu.dtype) + 0.5) / K
    target = (-0.5 * ((mu.unsqueeze(-1) - centers) / sigma).square()).exp()
    return target / target.sum(-1, keepdim=True)


def binned_ce(logits, target_mu, K, sigma=0.0):
    """Cross-entropy of the binned mu head against the two-hot or Gaussian target."""
    target = mu_gaussian(target_mu, K, sigma) if sigma > 0 else mu_two_hot(target_mu, K)
    return -(target * logits.log_softmax(-1)).sum(-1)


def focal_r(error, residual, beta, gamma):
    """Focal-R (Yang et al. 2021): scale error by (2 sigmoid(beta |e|) - 1)^gamma.

    The paper's beta=0.2 is for labels spanning ~100 units; mu lives in [0, 1], so beta must be larger."""
    return (2 * torch.sigmoid(beta * residual.abs()) - 1).pow(gamma) * error


def class3_ce(logits, target_mu, thresholds, log_priors):
    """Logit-adjusted CE (Menon et al. 2021) for the hard-class mu head."""
    target_cls = torch.bucketize(target_mu, thresholds, right=True)
    adjusted = (logits + log_priors).log_softmax(-1)
    return -adjusted.gather(-1, target_cls.unsqueeze(-1)).squeeze(-1)


def evidential_loss(gamma, y, aux, lam):
    """Deep Evidential Regression loss (Amini et al. 2020), NIG (gamma, v, alpha, beta)."""
    v, alpha, beta = aux[..., 0], aux[..., 1], aux[..., 2]
    omega = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(math.pi / v)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma).square() * v + omega)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )
    reg = (y - gamma).abs() * (2 * v + alpha)
    return nll + lam * reg


def bmc_loss(pred, target, weight, sigma, eps):
    """Balanced MSE, BMC variant (Ren et al. 2022), learnable noise scale ``sigma``."""
    sel = weight > 0
    pred, target, weight = pred[sel], target[sel], weight[sel]
    if pred.numel() == 0:
        return pred.sum()
    noise_var = sigma.square().clamp_min(1e-4)
    logits = -(pred[:, None] - target[None, :]).square() / (2 * noise_var)
    ce = F.cross_entropy(
        logits, torch.arange(len(pred), device=pred.device), reduction="none"
    )
    ce = ce * (2 * noise_var).detach()
    return (ce * weight).sum() / (weight.sum() + eps)


def overpred_weight(train_error, pred, target, weight):
    """Scale optimistic errors (pred > target) by ``weight``; identity at 1.0."""
    if weight == 1.0:
        return train_error
    over = (pred > target).detach().to(train_error.dtype)
    return train_error * (1 + (weight - 1) * over)


def reduce_trav_loss(error, weight, mask, mode, eps):
    """``weighted`` divides by the weight mass; ``mean`` (original) is the plain mean of error*weight."""
    weighted = (error * weight)[mask]
    if mode == "mean":
        return weighted.mean()
    return weighted.sum() / (weight[mask].sum() + eps)
