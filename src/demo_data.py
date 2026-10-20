"""Synthetic experiment data generators (deterministic, seeded)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def conversion_experiment(
    n_per_arm: int = 5000,
    baseline: float = 0.10,
    true_uplift: float = 0.05,
    srm_shift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Two-arm conversion experiment.

    srm_shift > 0 injects a sample ratio mismatch (e.g. 0.02 => 52/48 split
    instead of 50/50) so the SRM check has something to catch.
    """
    rng = np.random.default_rng(seed)
    share_b = 0.5 + srm_shift
    variant = rng.choice(["control", "treatment"], size=2 * n_per_arm,
                         p=[1 - share_b, share_b])
    rate_c = baseline
    rate_t = baseline * (1.0 + true_uplift)
    probs = np.where(variant == "control", rate_c, rate_t)
    converted = rng.binomial(1, probs)
    # Device mix differs across arms only through randomness of assignment
    device = rng.choice(["mobile", "desktop", "tablet"], size=2 * n_per_arm,
                        p=[0.6, 0.3, 0.1])
    return pd.DataFrame({
        "variant": variant,
        "converted": converted,
        "device": device,
    })


def revenue_experiment(
    n_per_arm: int = 3000,
    mean_control: float = 4.0,
    lift: float = 0.30,
    heavy_tail: bool = True,
    seed: int = 7,
) -> pd.DataFrame:
    """Two-arm revenue-per-user experiment with optional heavy tail.

    Lift is applied multiplicatively to the mean via a lognormal shift.
    """
    rng = np.random.default_rng(seed)
    variant = np.array(["control"] * n_per_arm + ["treatment"] * n_per_arm)
    if heavy_tail:
        base = rng.lognormal(mean=np.log(mean_control), sigma=1.2, size=variant.size)
    else:
        base = rng.normal(mean_control, 2.0, size=variant.size)
        base = np.clip(base, 0, None)
    treat_mult = np.where(variant == "treatment", 1.0 + lift, 1.0)
    revenue = base * treat_mult
    return pd.DataFrame({"variant": variant, "revenue": revenue})


def cuped_experiment(
    n: int = 8000,
    lift: float = 0.02,
    rho: float = 0.6,
    seed: int = 123,
) -> pd.DataFrame:
    """Experiment with a correlated pre-period metric for CUPED demos.

    rho controls correlation between pre-period and in-experiment metric.
    """
    rng = np.random.default_rng(seed)
    pre = rng.normal(10.0, 2.0, size=n)
    noise = rng.normal(0.0, 2.0 * np.sqrt(1 - rho**2), size=n)
    variant = rng.choice(["control", "treatment"], size=n)
    effect = np.where(variant == "treatment", lift * 10.0, 0.0)
    y = 10.0 + rho * (pre - 10.0) + noise + effect
    return pd.DataFrame({"variant": variant, "metric": y, "pre_metric": pre})
