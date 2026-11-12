import numpy as np

from sinkquant import evalstats as E


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    shifted = rng.normal(0.5, 1.0, size=400)
    null = rng.normal(0.0, 1.0, size=400)
    assert E.excludes_zero(E.paired_bootstrap(shifted))
    assert not E.excludes_zero(E.paired_bootstrap(null))
    m, lo, hi = E.paired_bootstrap(np.ones(50))
    assert m == 1.0 and lo == 1.0 and hi == 1.0


def test_kl_basic():
    p = np.array([0.5, 0.5])
    assert abs(E.kl(p, p)) < 1e-9
    assert E.kl(np.array([0.9, 0.1]), np.array([0.5, 0.5])) > 0
