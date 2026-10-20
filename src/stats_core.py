"""Core statistical routines for experimentation analysis.

Implemented from first principles (numpy/scipy only) so the logic is
testable independently of the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Alt = Literal["two-sided", "larger", "smaller"]


# --------------------------------------------------------------------------
# Power / sample size
# --------------------------------------------------------------------------

def required_sample_size(
    baseline_rate: float,
    mde_rel: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Per-arm sample size for detecting a relative lift on a proportion.

    Uses the normal approximation for two independent proportions with a
    pooled variance under H0.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1.0 + mde_rel)
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("baseline_rate * (1 + mde_rel) must lie in (0, 1)")
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2.0
    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return int(np.ceil(numerator / (p2 - p1) ** 2))


def achieved_power(
    baseline_rate: float,
    mde_rel: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Power achieved for a given per-arm sample size."""
    p1 = baseline_rate
    p2 = baseline_rate * (1.0 + mde_rel)
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("baseline_rate * (1 + mde_rel) must lie in (0, 1)")
    p_bar = (p1 + p2) / 2.0
    se_null = np.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    # P(reject H0 | true effect)
    return float(stats.norm.cdf((abs(p2 - p1) - z_alpha * se_null) / se_alt))


def mde_for_power(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest relative lift detectable at the given n (grid search)."""
    grid = np.linspace(0.001, 1.0, 2000)
    for rel in grid:
        if baseline_rate * (1.0 + rel) >= 1.0:
            break
        if achieved_power(baseline_rate, float(rel), n_per_arm, alpha) >= power:
            return float(rel)
    return float("nan")


# --------------------------------------------------------------------------
# Frequentist tests
# --------------------------------------------------------------------------

@dataclass
class TestResult:
    method: str
    estimate_a: float
    estimate_b: float
    absolute_uplift: float
    relative_uplift: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    extras: dict = field(default_factory=dict)


def _z_test_proportions(
    x_a: int, n_a: int, x_b: int, n_b: int, alpha: float, alternative: Alt
) -> TestResult:
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_pool == 0:
        z, p = 0.0, 1.0
    else:
        z = (p_b - p_a) / se_pool
        if alternative == "two-sided":
            p = 2.0 * stats.norm.sf(abs(z))
        elif alternative == "larger":
            p = stats.norm.sf(z)
        else:
            p = stats.norm.cdf(z)
    # CI on the untransformed difference (Wald)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return TestResult(
        method="Two-proportion z-test",
        estimate_a=float(p_a),
        estimate_b=float(p_b),
        absolute_uplift=float(diff),
        relative_uplift=float(diff / p_a) if p_a else float("nan"),
        statistic=float(z),
        p_value=float(p),
        ci_low=float(diff - z_crit * se_diff),
        ci_high=float(diff + z_crit * se_diff),
        significant=bool(p < alpha),
        extras={"se_diff": float(se_diff), "n_a": n_a, "n_b": n_b},
    )


def _welch_t_test(a: np.ndarray, b: np.ndarray, alpha: float, alternative: Alt) -> TestResult:
    res = stats.ttest_ind(b, a, equal_var=False, alternative=alternative)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    diff = mb - ma
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(
        method="Welch's t-test",
        estimate_a=ma,
        estimate_b=mb,
        absolute_uplift=diff,
        relative_uplift=diff / ma if ma else float("nan"),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        ci_low=float(diff - t_crit * se),
        ci_high=float(diff + t_crit * se),
        significant=bool(res.pvalue < alpha),
        extras={"df": float(df), "se_diff": float(se), "n_a": na, "n_b": nb},
    )


def _mann_whitney(a: np.ndarray, b: np.ndarray, alpha: float, alternative: Alt) -> TestResult:
    res = stats.mannwhitneyu(b, a, alternative=alternative)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    diff = mb - ma
    # Bootstrap CI for the difference in means (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    draws = 2000
    idx_a = rng.integers(0, len(a), size=(draws, len(a)))
    idx_b = rng.integers(0, len(b), size=(draws, len(b)))
    boot = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return TestResult(
        method="Mann-Whitney U test",
        estimate_a=ma,
        estimate_b=mb,
        absolute_uplift=diff,
        relative_uplift=diff / ma if ma else float("nan"),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(res.pvalue < alpha),
        extras={"n_a": len(a), "n_b": len(b), "ci_source": "bootstrap (2000 draws)"},
    )


# --------------------------------------------------------------------------
# Sample ratio mismatch (SRM)
# --------------------------------------------------------------------------

def srm_p_value(observed: dict[str, int], expected_ratio: dict[str, float]) -> float:
    """Chi-square goodness-of-fit p-value for a sample ratio mismatch check."""
    total = sum(observed.values())
    exp_frac = {k: v / sum(expected_ratio.values()) for k, v in expected_ratio.items()}
    observed_counts = np.array([observed[k] for k in exp_frac], dtype=float)
    expected_counts = np.array([exp_frac[k] * total for k in exp_frac], dtype=float)
    if (expected_counts <= 0).any():
        raise ValueError("expected ratio must be positive for every variant")
    chi2 = float(((observed_counts - expected_counts) ** 2 / expected_counts).sum())
    dof = len(observed_counts) - 1
    return float(stats.chi2.sf(chi2, dof))


def frequentist_test(
    df: pd.DataFrame,
    variant_col: str,
    value_col: str,
    control: str,
    treatment: str,
    metric_type: Literal["conversion", "continuous"] = "conversion",
    method: Literal["z", "t", "mw"] = "z",
    alpha: float = 0.05,
    alternative: Alt = "two-sided",
) -> TestResult:
    """Run the appropriate test for control vs treatment."""
    sub = df[df[variant_col].isin([control, treatment])]
    a = sub[sub[variant_col] == control][value_col]
    b = sub[sub[variant_col] == treatment][value_col]

    if metric_type == "conversion":
        x_a, n_a = int(a.sum()), len(a)
        x_b, n_b = int(b.sum()), len(b)
        if min(x_a, x_b, n_a - x_a, n_b - x_b) < 5:
            raise ValueError(
                "Cell counts < 5: consider Fisher's exact test instead."
            )
        return _z_test_proportions(x_a, n_a, x_b, n_b, alpha, alternative)

    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if method == "t":
        return _welch_t_test(av, bv, alpha, alternative)
    if method == "mw":
        return _mann_whitney(av, bv, alpha, alternative)
    raise ValueError(f"unknown method: {method}")


# --------------------------------------------------------------------------
# Bayesian (beta-binomial) test for conversion metrics
# --------------------------------------------------------------------------

@dataclass
class BayesianResult:
    prob_b_beats_a: float
    expected_uplift: float
    ci_low: float
    ci_high: float
    risk: float  # expected loss of shipping B
    samples: int = 0


def bayesian_test(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    ci: float = 0.95,
    n_samples: int = 200_000,
    seed: int = 0,
) -> BayesianResult:
    """Beta-Binomial posterior comparison with uniform Beta(1,1) priors."""
    rng = np.random.default_rng(seed)
    post_a = rng.beta(1 + x_a, 1 + n_a - x_a, size=n_samples)
    post_b = rng.beta(1 + x_b, 1 + n_b - x_b, size=n_samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = (post_b - post_a) / np.where(post_a == 0, np.nan, post_a)
    prob_b = float((post_b > post_a).mean())
    lo, hi = (1 - ci) / 2 * 100, 100 - (1 - ci) / 2 * 100
    ci_low, ci_high = np.nanpercentile(uplift, [lo, hi])
    # Expected loss = E[max(0, a - b)] if we ship B
    risk = float(np.maximum(post_a - post_b, 0).mean())
    return BayesianResult(
        prob_b_beats_a=prob_b,
        expected_uplift=float(np.nanmean(uplift)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        risk=risk,
        samples=n_samples,
    )


# --------------------------------------------------------------------------
# CUPED (pre-experiment covariate adjustment)
# --------------------------------------------------------------------------

def cuped_adjustment(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CUPED-adjusted outcome using pre-period metric x.

    y_adj = y - theta * (x - E[x]),  theta = Cov(y, x) / Var(x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return y.copy()
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    return y - theta * (x - x.mean())


def variance_reduction(original: np.ndarray, adjusted: np.ndarray) -> float:
    """Fractional variance reduction achieved by CUPED (0.35 => 35%)."""
    v0, v1 = np.var(original, ddof=1), np.var(adjusted, ddof=1)
    return float(1.0 - v1 / v0) if v0 > 0 else 0.0
