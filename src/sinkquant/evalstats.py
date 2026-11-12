"""Small statistics helpers for the end-to-end protection kill test (numpy only)."""
import numpy as np


def paired_bootstrap(diff, n_boot=5000, seed=0, alpha=0.05):
    """Mean of per-item paired differences with a percentile bootstrap CI over items."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def kl(p, q, eps=1e-12):
    """KL(p || q) for probability vectors along the last axis."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def excludes_zero(ci):
    _, lo, hi = ci
    return lo > 0 or hi < 0
