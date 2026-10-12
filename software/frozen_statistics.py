"""Verbatim function bodies extracted from the byte-frozen stats.py; NumPy only."""
import numpy as np


def _finite_vector(values, name="values"):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D vector")
    return x


def bootstrap_mean_ci(values, bootstrap=10000, seed=20261009):
    """Patient bootstrap percentile 95% CI; reusable for paired differences."""
    x = _finite_vector(values)
    if int(bootstrap) != bootstrap or bootstrap < 100:
        raise ValueError("bootstrap must be an integer >= 100")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(bootstrap), dtype=np.float64)
    # Keep memory bounded even if the function is used on a larger cohort.
    for start in range(0, int(bootstrap), 1000):
        end = min(start + 1000, int(bootstrap))
        samples = rng.integers(0, len(x), size=(end - start, len(x)))
        estimates[start:end] = x[samples].mean(axis=1)
    lo, hi = np.percentile(estimates, (2.5, 97.5))
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(proposed, baseline, bootstrap=10000, seed=20261009):
    """Resample patient-paired differences, never the two methods separately."""
    p, b = _finite_vector(proposed, "proposed"), _finite_vector(baseline, "baseline")
    if p.shape != b.shape:
        raise ValueError("Paired method vectors must have identical length")
    mean, lower, upper = bootstrap_mean_ci(p - b, bootstrap, seed)
    return {"mean_delta_dice": mean, "ci95_lower": lower, "ci95_upper": upper,
            "n_patients": len(p)}


def sign_flip_pvalue(differences, permutations=20000, seed=20261009):
    """Two-sided Monte Carlo paired sign-flip p, with add-one correction.

    All-zero differences return 1. Small n still uses the prespecified Monte
    Carlo estimator; the returned p-value is not labelled an exact test.
    """
    delta = _finite_vector(differences, "differences")
    if int(permutations) != permutations or permutations < 100:
        raise ValueError("permutations must be an integer >= 100")
    observed = abs(delta.mean())
    if observed == 0 and np.all(delta == 0):
        return 1.0
    rng, extreme = np.random.default_rng(seed), 0
    for start in range(0, int(permutations), 1000):
        count = min(1000, int(permutations) - start)
        signs = 2 * rng.integers(0, 2, size=(count, len(delta))) - 1
        simulated = np.abs((signs * delta).mean(axis=1))
        extreme += int(np.count_nonzero(simulated >= observed - 1e-14))
    return float((extreme + 1) / (int(permutations) + 1))


def holm_adjust(pvalues):
    """Holm step-down family-wise adjustment, returned in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("pvalues must be a finite 1-D vector in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilson_rate_ci(events, n_patients):
    """95% binomial Wilson score interval, including zero/all-event samples."""
    if n_patients < 1 or not 0 <= events <= n_patients:
        raise ValueError("Events must be between 0 and a positive patient count")
    z = 1.959963984540054
    rate = events / n_patients
    denominator = 1 + z * z / n_patients
    centre = (rate + z * z / (2 * n_patients)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n_patients + z * z / (4 * n_patients ** 2)) / denominator
    return float(rate), float(max(0, centre - half)), float(min(1, centre + half))
